#!/usr/bin/env python3
"""Local execution broker — runnable prototype for #59293. Linux/POSIX, stdlib only.

    python scripts/local_exec_broker.py --socket /run/user/1000/hermes-broker.sock

The problem it exists to solve: the argv-only ``sudo -u`` carrier closed the same-UID policy
escape but broke ``execute_code``, because ``sudo`` is a *privilege* tool, not a *transport*.
Everything ``tools/code_kernel.py:_spawn`` hands its child today crosses the boundary
implicitly, and ``sudo`` drops all three:

  * the explicit ``env=`` dict         -> discarded under ``env_reset``
  * ``pass_fds=(death_r,)``            -> non-std descriptors are closed
  * the 0700 ``mkdtemp`` staging dir   -> the new uid cannot traverse it

So the broker owns child lifetime on the trusted side and transports each resource on a
channel that survives a uid switch:

  environment   the client sends the approved dict as JSON; the child env is built from it
                alone, never from the broker's own ``os.environ``.
  descriptors   the client passes open fds over ``SCM_RIGHTS``. The broker forwards them by
                number via ``pass_fds`` and publishes those numbers as ``HERMES_BROKER_FDS``,
                then closes its own copies so it never holds a peer's channel open.
  runner        the broker opens the staged runner itself (it owns the 0700 dir) and the
                child reads it back through ``/proc/self/fd/<n>``. Reopening a regular file
                through procfs re-checks the INODE bits but skips directory traversal, so the
                staging dir is never relaxed for anyone else.
  lifetime      the client connection IS the lease. Its EOF — close, crash, SIGKILL — is what
                kills the child process group, the same signal shape as the inherited
                parent-death pipe, but owned by the broker rather than inherited through sudo.

Future integration seam (deliberately NOT wired yet, so nothing dead lands in core):

  * ``tools/code_kernel.py:_spawn`` becomes a ``request_launch`` call — ``child_env`` is the
    ``env`` payload, ``death_r`` and the runner staging dir are what this already transports,
    and the returned connection replaces ``kernel.death_pipe_w`` as the liveness handle held
    by ``SessionKernel``.
  * ``tools/environments/local.py:_run_bash`` follows with argv + cwd added to the request.
  * ``tools/process_registry.py``'s systemd-scope isolation composes on the broker side,
    where the trusted uid still has a user bus.

Until those land this file is a prototype with its own behaviour test
(``tests/scripts/test_local_exec_broker.py``), not production surface.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading

# Child-visible fd numbers the broker forwarded on its behalf, comma separated.
FDS_ENV = "HERMES_BROKER_FDS"

MAX_FDS = 8
_HEADER_BYTES = 65536
_TERM_GRACE_SECONDS = 2.0


def request_launch(
    sock_path: str, *, runner: str, env: dict, fds: list, timeout: float = 30.0
):
    """Client side: ask the broker to launch *runner*; return ``(connection, reply)``.

    The caller MUST hold the returned connection open for as long as the child should
    live — the broker treats its EOF as the order to tear the child down.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    conn.connect(sock_path)
    body = (
        json.dumps({"op": "launch", "runner": runner, "env": env}).encode("utf-8")
        + b"\n"
    )
    conn.sendmsg(
        [body], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
    )
    return conn, json.loads(_read_line(conn))


def _read_line(conn) -> bytes:
    """Read one newline-terminated frame (no ancillary data expected)."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]


def _recv_request(conn):
    """Read one request frame plus every descriptor that rode along with it."""
    buf, fds = b"", []
    while b"\n" not in buf:
        msg, ancdata, _flags, _addr = conn.recvmsg(
            _HEADER_BYTES, socket.CMSG_SPACE(MAX_FDS * array.array("i").itemsize)
        )
        for level, kind, data in ancdata:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array.array("i")
                received.frombytes(data[: len(data) - (len(data) % received.itemsize)])
                fds.extend(received)
        if not msg:
            break
        buf += msg
    if b"\n" not in buf:
        return None, fds
    return json.loads(buf.split(b"\n", 1)[0]), fds


def _launch(request: dict, fds: list):
    """Spawn the child with everything handed over explicitly."""
    # The broker owns the 0700 staging dir, so it — not the child — resolves the path.
    runner_fd = os.open(request["runner"], os.O_RDONLY)
    try:
        child_env = dict(request.get("env") or {})
        child_env[FDS_ENV] = ",".join(str(fd) for fd in fds)
        # pass_fds keeps each descriptor at its own number in the child, which is what makes
        # both FDS_ENV and the /proc/self/fd runner path resolvable on the far side.
        return subprocess.Popen(
            [sys.executable, f"/proc/self/fd/{runner_fd}"],
            env=child_env,
            pass_fds=(runner_fd, *fds),
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        # Every forwarded descriptor is the peer's channel, not ours: the child holds its own
        # copies, and a retained copy here would keep a pipe from ever reaching EOF.
        os.close(runner_fd)
        for fd in fds:
            os.close(fd)


def _terminate(proc) -> None:
    """Tear the child's process group down and REAP it.

    ``start_new_session=True`` made the child its own group leader, so its pid is the pgid —
    no ``getpgid`` lookup to race against an already-exited child. Reaping is part of the
    contract, not cleanup: an unreaped child is still a ``/proc`` entry that answers
    ``kill(pid, 0)``, i.e. indistinguishable from one that outlived its lease.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)  # windows-footgun: ok — Linux-only broker
    try:
        proc.wait(timeout=_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(  # windows-footgun: ok — Linux-only broker
                proc.pid,
                signal.SIGKILL,  # windows-footgun: ok — Linux-only broker
            )
        proc.wait()


def _serve_connection(conn) -> None:
    proc = None
    try:
        request, fds = _recv_request(conn)
        if request is None:
            for fd in fds:
                os.close(fd)
            return
        proc = _launch(request, fds)
        conn.sendall(json.dumps({"ok": True, "pid": proc.pid}).encode("utf-8") + b"\n")
        # The connection IS the child's lease. Hold it open and read until EOF: a clean close,
        # a crashed client or a SIGKILLed one all surface the same way, which is exactly the
        # signal the inherited parent-death pipe gave us before sudo started closing it.
        with contextlib.suppress(OSError):
            conn.settimeout(None)
            while conn.recv(4096):
                pass
    finally:
        conn.close()
        if proc is not None:
            _terminate(proc)


def serve(sock_path: str) -> None:
    """Bind, announce readiness, then serve one connection per thread."""
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    os.chmod(sock_path, 0o600)
    listener.listen(16)
    print(json.dumps({"ready": True, "socket": sock_path}), flush=True)
    while True:
        conn, _addr = listener.accept()
        threading.Thread(target=_serve_connection, args=(conn,), daemon=True).start()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True, help="AF_UNIX path to bind")
    args = parser.parse_args(argv)
    serve(args.socket)
    return 0


if __name__ == "__main__":
    sys.exit(main())
