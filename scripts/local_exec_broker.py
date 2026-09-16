#!/usr/bin/env python3
"""Local execution broker — runnable prototype for #59293. Linux/POSIX, stdlib only.

    python scripts/local_exec_broker.py \
        --socket /run/user/1000/hermes-broker.sock --staging-root /run/user/1000/hermes-stage \
        --allow-uid 1001 --socket-mode 0660

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
  runner        the client passes its already-open regular runner as a distinct
                ``SCM_RIGHTS`` descriptor. The child executes ``/proc/self/fd/<n>``, so no
                pathname or staging-directory traversal occurs. The legacy ``runner`` path
                request remains available for compatibility and is resolved inside the
                broker-owned staging root before being opened.
  lifetime      the client connection IS the lease. Its EOF — close, crash, SIGKILL — is what
                kills the child process group, the same signal shape as the inherited
                parent-death pipe, but owned by the broker rather than inherited through sudo.

Everything the child does not receive explicitly, it does not get: stdin, stdout AND stderr
are all ``DEVNULL``, so no channel crosses the boundary by inheritance.

**The request frame is the only untrusted surface, so it is the one that is bounded.** A
request is one newline-terminated JSON object of at most ``MAX_REQUEST_BYTES`` carrying at
most ``MAX_FDS`` child descriptors plus one runner descriptor, read under a handshake
timeout. Every refusal is a structured ``{"ok": false, "error", "message"}`` reply, and —
the invariant that matters more — every refusal closes the descriptors the kernel already
installed on the broker's behalf. A retained copy is not merely a leaked fd: it is the
peer's channel, and it keeps their pipe from ever reaching EOF.

Every accepted connection is authenticated with Linux ``SO_PEERCRED`` before a request is
read. ``--allow-uid`` is repeatable and defaults to the broker's effective uid; socket
publication defaults to 0600, while ``--socket-mode`` can explicitly publish 0660 or 0666
for an authorized cross-uid client. Wider publication grants only reachability: the peer uid
allowlist remains mandatory authorization. The staging root must still be broker-owned 0700
for legacy pathname requests. The client's ``env`` payload is passed to the child verbatim
because at this boundary it is the approved child environment.

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
import errno
import fcntl
import json
import os
import selectors
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time

# Child-visible fd numbers the broker forwarded on its behalf, comma separated.
FDS_ENV = "HERMES_BROKER_FDS"

# Caps on the one untrusted surface. These bound the TOTAL for a request, not a single recv.
MAX_FDS = 8
MAX_RECEIVED_FDS = MAX_FDS + 1  # one runner plus MAX_FDS child descriptors
MAX_REQUEST_BYTES = 65536
DEFAULT_HANDSHAKE_TIMEOUT = 10.0

_INT_SIZE = array.array("i").itemsize
_UCRED = struct.Struct("=iII")
_RECV_CHUNK = 4096
_TERM_GRACE_SECONDS = 2.0
_KILL_GRACE_SECONDS = 2.0
_PUBLISH_SUFFIXES = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RUNNER_BOOTSTRAP = """\
import os
import sys

_runner_fd = int(sys.argv.pop())
_runner_path = sys.argv.pop()
sys.argv[:] = [_runner_path]
with os.fdopen(_runner_fd, "rb", closefd=False) as _runner:
    _runner_code = compile(_runner.read(), _runner_path, "exec")
exec(
    _runner_code,
    {
        "__name__": "__main__",
        "__file__": _runner_path,
        "__package__": None,
        "__cached__": None,
    },
)
"""


class BrokerError(RuntimeError):
    """A refusal, carrying the same ``code`` on both sides of the socket.

    The broker raises it, replies with ``code``/``message``, and the client re-raises it from
    that reply — so a caller sees a typed failure it can branch on instead of whatever
    exception happens to fall out of parsing an empty read.
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def request_launch(
    sock_path: str,
    *,
    runner: str | None = None,
    runner_fd: int | None = None,
    env: dict,
    fds: list,
    timeout: float = 30.0,
):
    """Client side: ask the broker to launch *runner*; return ``(connection, reply)``.

    The caller MUST hold the returned connection open for as long as the child should
    live — the broker treats its EOF as the order to tear the child down. On any failure the
    connection is closed here and a :class:`BrokerError` is raised; the caller never inherits
    a socket it did not get a child for.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(timeout)
        conn.connect(sock_path)
        if (runner is None) == (runner_fd is None):
            raise ValueError("exactly one of runner or runner_fd is required")
        request = {"op": "launch", "env": env}
        rights = list(fds)
        if runner_fd is not None:
            request["runner_fd"] = True
            rights.insert(0, runner_fd)
        else:
            request["runner"] = runner
        body = json.dumps(request).encode("utf-8") + b"\n"
        _sendmsg_all(conn, body, _ancillary(rights))
        reply = _read_reply(conn)
        if not reply.get("ok"):
            raise BrokerError(
                reply.get("error") or "unknown",
                reply.get("message") or "launch refused",
            )
    except BaseException:
        conn.close()
        raise
    return conn, reply


def _ancillary(fds):
    return (
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", list(fds)))]
        if fds
        else []
    )


def _sendmsg_all(conn, body: bytes, ancillary) -> None:
    """Send a complete frame, attaching descriptor rights to its first bytes only."""
    sent = 0
    first = True
    while sent < len(body):
        written = conn.sendmsg([body[sent:]], ancillary if first else [])
        if written <= 0:
            raise ConnectionError("sendmsg made no progress")
        sent += written
        first = False


def _read_reply(conn) -> dict:
    """Read one newline-terminated reply frame (no ancillary data expected)."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(_RECV_CHUNK)
        if not chunk:
            raise BrokerError(
                "no_reply", "broker closed the connection without replying"
            )
        buf += chunk
        if len(buf) > MAX_REQUEST_BYTES:
            raise BrokerError("bad_reply", "broker reply exceeded the frame cap")
    try:
        return json.loads(buf.split(b"\n", 1)[0])
    except json.JSONDecodeError as exc:
        raise BrokerError("bad_reply", f"broker reply was not JSON: {exc}") from exc


def _close_all(fds) -> None:
    while fds:
        with contextlib.suppress(OSError):
            os.close(fds.pop())


def _recv_request(conn, fds: list, handshake_timeout: float):
    """Read one bounded request frame, appending every received descriptor to *fds*.

    *fds* is the CALLER's list on purpose. The kernel installs descriptors into this process
    the moment they arrive, including on a request that turns out to be garbage; making the
    caller the owner from the first byte is what keeps "who closes this" answerable on every
    path out of here.
    """
    buf = b""
    deadline = time.monotonic() + handshake_timeout
    while b"\n" not in buf:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            conn.settimeout(remaining)
            msg, ancdata, flags, _addr = conn.recvmsg(
                _RECV_CHUNK, socket.CMSG_SPACE(MAX_RECEIVED_FDS * _INT_SIZE)
            )
        except TimeoutError as exc:
            raise BrokerError(
                "handshake_timeout", "no complete request within the handshake window"
            ) from exc
        for level, kind, data in ancdata:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                continue
            if len(data) % _INT_SIZE:
                raise BrokerError(
                    "truncated_ancillary", "partial descriptor in ancillary data"
                )
            received = array.array("i")
            received.frombytes(data)
            fds.extend(received)
        # The kernel installs as many descriptors as the control buffer holds, CLOSES the
        # rest and sets MSG_CTRUNC. Ignoring it means launching a child whose
        # HERMES_BROKER_FDS is silently shorter than what the client passed.
        if flags & socket.MSG_CTRUNC:
            raise BrokerError(
                "truncated_ancillary",
                "ancillary data was truncated; at most "
                f"{MAX_RECEIVED_FDS} descriptors per request",
            )
        if len(fds) > MAX_RECEIVED_FDS:
            raise BrokerError(
                "too_many_fds",
                f"at most {MAX_RECEIVED_FDS} descriptors may be passed per request",
            )
        if not msg:
            raise BrokerError(
                "incomplete_request", "peer closed before sending a complete request"
            )
        buf += msg
        if len(buf) > MAX_REQUEST_BYTES:
            raise BrokerError(
                "request_too_large",
                f"request exceeded {MAX_REQUEST_BYTES} bytes with no frame terminator",
            )
    try:
        request = json.loads(buf.split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BrokerError("bad_request", f"request was not JSON: {exc}") from exc
    return _validated(request)


def _validated(request):
    """Structural validation only — return ``(runner, env)``.

    This checks the SHAPE of the payload, not its content: the env dict is the approved child
    environment and is forwarded verbatim (see the trust-boundary note in the module
    docstring). NUL is rejected because it would truncate silently at ``execve``.
    """
    if not isinstance(request, dict):
        raise BrokerError("bad_request", "request must be a JSON object")
    if request.get("op") != "launch":
        raise BrokerError("bad_request", f"unsupported op {request.get('op')!r}")
    runner = request.get("runner")
    uses_runner_fd = request.get("runner_fd") is True
    if uses_runner_fd == (runner is not None):
        raise BrokerError(
            "bad_request", "request must carry exactly one of 'runner' or 'runner_fd'"
        )
    if not uses_runner_fd:
        if not isinstance(runner, str) or not runner:
            raise BrokerError("bad_request", "'runner' must be a non-empty string")
        if "\0" in runner:
            raise BrokerError("bad_request", "'runner' must not contain NUL")
    if "env" not in request:
        env = {}
    else:
        env = request["env"]
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise BrokerError(
            "bad_request", "'env' must be a JSON object of string to string"
        )
    if any("\0" in key or "\0" in value for key, value in env.items()):
        raise BrokerError("bad_request", "environment entries must not contain NUL")
    return runner, uses_runner_fd, env


def _resolve_runner(staging_root: str, runner: str) -> str:
    """Resolve *runner* to a real path strictly inside *staging_root*.

    The runner transport only means anything if the BROKER is the side that resolves the
    path. ``realpath`` collapses ``..`` and follows symlinks BEFORE the containment test, so
    a symlink staged inside the root but pointing out of it is refused rather than followed.
    """
    resolved = os.path.realpath(runner)
    if not resolved.startswith(staging_root + os.sep):
        raise BrokerError(
            "runner_outside_root",
            f"runner resolves outside the staging root {staging_root}",
        )
    return resolved


def _open_runner(path: str) -> int:
    """Open the staged runner. ``O_NOFOLLOW`` closes the realpath-then-open swap window."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise BrokerError("runner_not_regular", "runner is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _validate_runner_fd(fd: int) -> None:
    """Require an open, readable regular file suitable for ``/proc/self/fd`` execution."""
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise BrokerError("runner_not_regular", "runner is not a regular file")
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if flags & getattr(os, "O_PATH", 0) or flags & os.O_ACCMODE == os.O_WRONLY:
        raise BrokerError("runner_not_readable", "runner descriptor must be readable")


def _launch(runner_fd: int, env: dict, fds: list):
    """Spawn the child with everything handed over explicitly."""
    child_env = dict(env)
    child_env[FDS_ENV] = ",".join(str(fd) for fd in fds)
    # pass_fds keeps each descriptor at its own number in the child, which is what makes both
    # FDS_ENV and the /proc/self/fd runner path resolvable on the far side.
    # The bootstrap reads the already-open descriptor directly. Asking the interpreter to open
    # ``/proc/self/fd/N`` as a script would re-check the inode's mode bits and fail after a
    # legitimate cross-UID SCM_RIGHTS handoff, even though the descriptor itself is readable.
    runner_path = f"/proc/self/fd/{runner_fd}"
    return subprocess.Popen(
        [sys.executable, "-c", _RUNNER_BOOTSTRAP, runner_path, str(runner_fd)],
        env=child_env,
        pass_fds=(runner_fd, *fds),
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        # Explicit, like stdin and stdout. An inherited stderr is a channel the client never
        # asked for: a write handle into the trusted side's log stream, and an undrained pipe
        # the child can wedge itself on.
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _signal_group(pid: int, sig) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)  # windows-footgun: ok — Linux-only broker


def _wait_unreaped(pid: int, timeout: float) -> None:
    """Wait up to *timeout* for *pid* to exit, deliberately leaving it UNREAPED.

    ``Popen.wait``/``poll`` would reap it, and a reaped pid can be recycled — after which the
    pgid we are about to sweep may belong to somebody else entirely.
    """
    try:
        pidfd = os.pidfd_open(pid)
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(pidfd, selectors.EVENT_READ)
                sel.select(timeout)
        finally:
            os.close(pidfd)
    except OSError:
        deadline = time.monotonic() + timeout
        while not _exited_unreaped(pid):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.05, remaining))


def _exited_unreaped(pid: int) -> bool:
    """Observe child exit without releasing its pid for reuse."""
    try:
        return (
            os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        )
    except ChildProcessError:
        return True


def _terminate(proc) -> None:
    """Tear the child's process GROUP down and REAP the leader.

    ``start_new_session=True`` made the child its own group leader, so its pid is the pgid —
    no ``getpgid`` lookup to race against an already-exited child. Reaping is part of the
    contract, not cleanup: an unreaped child is still a ``/proc`` entry that answers
    ``kill(pid, 0)``, i.e. indistinguishable from one that outlived its lease.
    """
    _terminate_many([proc])


def _terminate_many(procs) -> None:
    """Broadcast teardown phases to *procs* under shared grace deadlines."""
    procs = [proc for proc in procs if proc.returncode is None]
    for proc in procs:
        _signal_group(proc.pid, signal.SIGTERM)
    term_deadline = time.monotonic() + _TERM_GRACE_SECONDS
    for proc in procs:
        _wait_unreaped(proc.pid, max(0.0, term_deadline - time.monotonic()))
    # The leader is still unreaped, so its pid — and with it the pgid — cannot have been
    # recycled. This is the only safe moment to sweep descendants that outlived the leader or
    # ignored the SIGTERM; they are not our children, so we signal them and let init reap.
    for proc in procs:
        _signal_group(
            proc.pid,
            signal.SIGKILL,  # windows-footgun: ok — Linux-only broker
        )
    kill_deadline = time.monotonic() + _KILL_GRACE_SECONDS
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            # Keep the Popen object (and therefore waitpid ownership) alive until the
            # uninterruptible syscall returns and SIGKILL can complete.
            threading.Thread(target=proc.wait, daemon=True).start()


class _Leases:
    """Every accepted connection, worker and child, so shutdown can drain all three.

    Registration precedes ``Thread.start`` so even a worker blocked inside ``Popen`` remains
    visible to shutdown. Closing every accepted connection breaks incomplete handshakes;
    waiting for the registry to empty covers children spawned after the initial snapshot.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._workers = {}
        self._shutting_down = False

    def register(self, conn, worker) -> bool:
        with self._condition:
            if self._shutting_down:
                return False
            self._workers[conn] = [worker, None]
            return True

    def add(self, conn, proc) -> bool:
        with self._condition:
            entry = self._workers.get(conn)
            if self._shutting_down or entry is None:
                return False
            entry[1] = proc
            return True

    def claim(self, conn, proc) -> bool:
        with self._condition:
            entry = self._workers.get(conn)
            if entry is None or entry[1] is not proc:
                return False
            entry[1] = None
            return True

    def finished(self, conn) -> None:
        with self._condition:
            self._workers.pop(conn, None)
            self._condition.notify_all()

    def drain(self) -> None:
        with self._condition:
            self._shutting_down = True
            entries = list(self._workers.items())
            procs = []
            for _conn, entry in entries:
                if entry[1] is not None:
                    procs.append(entry[1])
                    entry[1] = None
        for conn, _entry in entries:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
        _terminate_many(procs)
        with self._condition:
            while self._workers:
                self._condition.wait()


def _reply(conn, payload: dict) -> None:
    with contextlib.suppress(OSError):
        conn.sendall(json.dumps(payload).encode("utf-8") + b"\n")


def _await_lease_end(conn, proc) -> None:
    """Block until the lease ends — the client's EOF, or the child exiting on its own.

    Watching only the connection makes the lease one-directional: a child that finishes
    normally would leave this thread parked in ``recv`` until the client happened to
    disconnect, holding an unreaped child the whole time. The pidfd is the other half, and it
    reports the exit WITHOUT reaping, so ``_terminate`` still owns the group sweep.
    """
    conn.settimeout(None)
    try:
        pidfd = os.pidfd_open(proc.pid)
    except ProcessLookupError:
        return
    except OSError:
        with selectors.DefaultSelector() as sel:
            sel.register(conn, selectors.EVENT_READ)
            while not _exited_unreaped(proc.pid):
                if sel.select(0.05) and not conn.recv(_RECV_CHUNK):
                    return
        return
    try:
        with selectors.DefaultSelector() as sel:
            sel.register(pidfd, selectors.EVENT_READ)
            sel.register(conn, selectors.EVENT_READ)
            while True:
                for key, _mask in sel.select():
                    if key.fd == pidfd or not conn.recv(_RECV_CHUNK):
                        return
    finally:
        os.close(pidfd)


def _serve_connection(
    conn,
    staging_root: str,
    handshake_timeout: float,
    leases: _Leases,
    allowed_uids: frozenset[int] | None = None,
) -> None:
    proc, runner_fd = None, None
    # Descriptors the kernel installed on our behalf. One owner, one close, every path out.
    fds: list = []
    try:
        try:
            allowed_uids = allowed_uids or frozenset({os.geteuid()})
            _pid, peer_uid, _gid = _UCRED.unpack(
                conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _UCRED.size)
            )
            if peer_uid not in allowed_uids:
                raise BrokerError(
                    "peer_uid_not_allowed",
                    f"peer uid {peer_uid} is not allowed",
                )
            runner, uses_runner_fd, env = _recv_request(conn, fds, handshake_timeout)
            if uses_runner_fd:
                if not fds:
                    raise BrokerError(
                        "runner_fd_missing", "runner descriptor was not received"
                    )
                runner_fd = fds.pop(0)
                _validate_runner_fd(runner_fd)
            else:
                if len(fds) > MAX_FDS:
                    raise BrokerError(
                        "too_many_fds",
                        f"at most {MAX_FDS} descriptors may be passed per request",
                    )
                runner_fd = _open_runner(_resolve_runner(staging_root, runner))
            proc = _launch(runner_fd, env, fds)
        except BrokerError as exc:
            _reply(conn, {"ok": False, "error": exc.code, "message": exc.message})
            return
        except OSError as exc:
            _reply(conn, {"ok": False, "error": "launch_failed", "message": str(exc)})
            return
        finally:
            # Every forwarded descriptor is the peer's channel, not ours: the child holds its
            # own copies, and a retained copy here would keep a pipe from ever reaching EOF.
            _close_all(fds)
            if runner_fd is not None:
                os.close(runner_fd)
        # Registered BEFORE the reply: a SIGTERM racing the handshake must still find this
        # child, or it is orphaned in the one window where nobody is watching it.
        registered = leases.add(conn, proc)
        if not registered:
            _terminate(proc)
            return
        _reply(conn, {"ok": True, "pid": proc.pid})
        # The connection IS the child's lease. A clean close, a crashed client or a SIGKILLed
        # one all surface the same way, which is exactly the signal the inherited
        # parent-death pipe gave us before sudo started closing it.
        with contextlib.suppress(OSError):
            _await_lease_end(conn, proc)
    finally:
        with contextlib.suppress(OSError):
            conn.close()
        try:
            if proc is not None and leases.claim(conn, proc):
                _terminate(proc)
        finally:
            leases.finished(conn)


def _validated_staging_root(path: str) -> str:
    root = os.path.realpath(path)
    if not os.path.isdir(root):
        raise SystemExit(f"staging root is not a directory: {path}")
    info = os.stat(root)
    broker_uid = os.geteuid()  # windows-footgun: ok — Linux-only broker
    if info.st_uid != broker_uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise SystemExit(
            "staging root must be owned by the broker uid with permissions 0700: "
            f"{path}"
        )
    return root


def _validated_socket_mode(mode: int) -> int:
    if mode not in (0o600, 0o660, 0o666):
        raise ValueError("socket mode must be 0600, 0660, or 0666")
    return mode


def _clear_stale_socket(sock_path: str) -> None:
    """Remove a socket left behind by an unclean shutdown — and nothing else.

    ``lstat``, not ``stat``: the decision is about the path itself, so a symlink parked here
    pointing at something that matters is refused rather than followed and unlinked. Anything
    that is not a socket is somebody else's file; the broker declines to start instead.
    """
    try:
        mode = os.lstat(sock_path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise SystemExit(
            f"refusing to replace a path that is not a socket: {sock_path}"
        )

    # A socket pathname is not stale merely because it already exists. Probe it before
    # unlinking: stealing a live broker's path leaves that process running but unreachable.
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError as exc:
        raise SystemExit(
            f"refusing to replace an uncertain socket: {sock_path}: {exc}"
        ) from exc
    else:
        raise SystemExit(f"refusing to replace a live broker socket: {sock_path}")
    finally:
        probe.close()
    os.unlink(sock_path)


def _reclaim_stale_publish_dir(path: str) -> bool:
    """Reclaim only a private slot containing one unreachable broker socket."""
    try:
        info = os.lstat(path)
        entries = os.listdir(path)
    except OSError:
        return False
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()  # windows-footgun: ok — Linux-only broker
        or stat.S_IMODE(info.st_mode) != 0o700
        or entries != ["s"]
    ):
        return False
    socket_path = os.path.join(path, "s")
    try:
        if not stat.S_ISSOCK(os.lstat(socket_path).st_mode):
            return False
    except OSError:
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        probe.connect(socket_path)
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    except OSError:
        return False
    else:
        return False
    finally:
        probe.close()
    try:
        os.unlink(socket_path)
        os.rmdir(path)
    except OSError:
        return False
    return True


def _unlink_owned_socket(sock_path: str, owned_fd: int) -> None:
    """Unlink *sock_path* only while it still names the socket we bound."""
    try:
        socket_dir = os.path.dirname(sock_path) or "."
        lock_fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                current = os.lstat(sock_path)
            except FileNotFoundError:
                return
            owned = os.fstat(owned_fd)
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                os.unlink(sock_path)
        finally:
            os.close(lock_fd)
    finally:
        os.close(owned_fd)


def _install_shutdown(listener) -> None:
    """Turn SIGTERM/SIGINT into an ordinary exit from the accept loop.

    Closing the listener is what breaks ``accept``: under PEP 475 Python retries an
    EINTR-interrupted syscall itself, so a handler that merely set a flag would go unnoticed
    until the next connection happened to arrive.
    """

    def _shutdown(_signum, _frame):
        listener.close()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _shutdown)


def _start_worker(
    conn,
    root: str,
    handshake_timeout: float,
    leases: _Leases,
    allowed_uids: frozenset[int] | None = None,
) -> None:
    worker = threading.Thread(
        target=_serve_connection,
        args=(conn, root, handshake_timeout, leases, allowed_uids),
        daemon=True,
    )
    if not leases.register(conn, worker):
        conn.close()
        return
    try:
        worker.start()
    except RuntimeError:
        leases.finished(conn)
        conn.close()


def serve(
    sock_path: str,
    staging_root: str,
    *,
    handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
    allowed_uids: frozenset[int] | None = None,
    socket_mode: int = 0o600,
) -> None:
    """Bind, announce readiness, then serve one connection per thread."""
    root = _validated_staging_root(staging_root)
    socket_mode = _validated_socket_mode(socket_mode)
    allowed_uids = allowed_uids or frozenset({os.geteuid()})
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    leases = _Leases()
    owned_fd = None
    publish_dir = None
    temporary_socket = None
    try:
        socket_dir = os.path.dirname(sock_path) or "."
        lock_fd = os.open(socket_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            _clear_stale_socket(sock_path)
            for suffix in _PUBLISH_SUFFIXES:
                if suffix == os.path.basename(sock_path):
                    continue
                candidate = os.path.join(socket_dir, suffix)
                try:
                    os.mkdir(candidate, 0o700)
                except FileExistsError:
                    if not _reclaim_stale_publish_dir(candidate):
                        continue
                    os.mkdir(candidate, 0o700)
                publish_dir = candidate
                break
            else:
                raise SystemExit(
                    "no compact private socket publication directory was available",
                )
            os.chmod(publish_dir, 0o700)
            temporary_socket = os.path.join(publish_dir, "s")
            listener.bind(f"/proc/self/fd/{lock_fd}/{suffix}/s")
            listener.listen(16)
            os.chmod(temporary_socket, socket_mode)
            owned_fd = os.open(
                temporary_socket,
                os.O_PATH
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,  # windows-footgun: ok — Linux-only broker
            )
            os.link(temporary_socket, sock_path)
            os.unlink(temporary_socket)
            temporary_socket = None
            os.rmdir(publish_dir)
            publish_dir = None
        finally:
            os.close(lock_fd)
        _install_shutdown(listener)
        print(json.dumps({"ready": True, "socket": sock_path}), flush=True)
        while True:
            try:
                conn, _addr = listener.accept()
            except OSError as exc:
                if listener.fileno() == -1:
                    break  # the shutdown handler closed the listener
                if exc.errno == errno.ECONNABORTED:
                    continue
                if exc.errno in (errno.EMFILE, errno.ENFILE):
                    time.sleep(0.05)
                    continue
                raise
            _start_worker(conn, root, handshake_timeout, leases, allowed_uids)
    finally:
        with contextlib.suppress(OSError):
            listener.close()
        # Daemon threads will not unwind, so shutdown — not the workers — is what keeps the
        # "nothing outlives its lease" promise when the broker itself is the one going away.
        leases.drain()
        if owned_fd is not None:
            with contextlib.suppress(OSError):
                _unlink_owned_socket(sock_path, owned_fd)
        if temporary_socket is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_socket)
        if publish_dir is not None:
            with contextlib.suppress(OSError):
                os.rmdir(publish_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", required=True, help="AF_UNIX path to bind")
    parser.add_argument(
        "--staging-root",
        required=True,
        help="directory the broker owns; runners outside it are refused",
    )
    parser.add_argument(
        "--handshake-timeout",
        type=float,
        default=DEFAULT_HANDSHAKE_TIMEOUT,
        help="seconds a connection may take to send one complete request",
    )
    parser.add_argument(
        "--allow-uid",
        action="append",
        type=int,
        dest="allowed_uids",
        help="uid authorized to request launches; repeatable (default: broker euid)",
    )
    parser.add_argument(
        "--socket-mode",
        choices=("0600", "0660", "0666"),
        default="0600",
        help="published socket permissions; peer uid authorization still applies",
    )
    args = parser.parse_args(argv)
    allowed_uids = frozenset(
        args.allowed_uids if args.allowed_uids is not None else [os.geteuid()]
    )
    if any(uid < 0 for uid in allowed_uids):
        parser.error("--allow-uid must be non-negative")
    serve(
        args.socket,
        args.staging_root,
        handshake_timeout=args.handshake_timeout,
        allowed_uids=allowed_uids,
        socket_mode=int(args.socket_mode, 8),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
