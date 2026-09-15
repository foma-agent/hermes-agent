"""Behaviour tests for the local execution broker prototype (``scripts/local_exec_broker.py``).

Regression cover for #59293. The argv-only ``sudo -u`` carrier closed the same-UID policy
escape but broke ``execute_code`` three ways at once, all of them the same root shape —
*implicit* inheritance across a uid boundary:

  * ``sudo``'s ``env_reset`` discards the explicit ``env=`` dict ``code_kernel._spawn``
    passes to ``Popen``;
  * ``sudo`` closes inherited non-std descriptors, so the parent-death pipe
    (``HERMES_KERNEL_PARENT_DEATH_FD``) never reaches the kernel runner;
  * the kernel staging dir is ``tempfile.mkdtemp()`` (0700), which the new uid cannot
    traverse to reach ``hermes_kernel_runner.py``.

Two tests, one vertical contract each:

  1. **Transport + protocol.** What the broker accepts, what it refuses, and who owns a
     descriptor that arrived over ``SCM_RIGHTS``. Every refusal path is also an fd-ownership
     path: the broker must close what it received, or the peer's pipe never reaches EOF.
  2. **Lease lifetime.** Nothing outlives its lease (child, descendants, broker shutdown) and
     nothing pins a worker thread forever (a child that exits on its own, a silent peer).

Real processes throughout: a separately spawned broker, a real ``AF_UNIX`` connection, real
children and grandchildren. Nothing here is mocked — the point is the boundary.
"""

from __future__ import annotations

import array
import contextlib
import importlib.util
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BROKER_PATH = REPO_ROOT / "scripts" / "local_exec_broker.py"

# Wall-clock ceilings. Generous on purpose: these bound a FAILURE (the thing under test never
# happened) rather than pace a success, and the suite runner is not a quiet machine.
DEADLINE = 15.0
HANDSHAKE_TIMEOUT = 2.0


def _load_broker():
    """Import the broker script as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("local_exec_broker", BROKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_exec_broker"] = module
    spec.loader.exec_module(module)
    return module


def _socket_path():
    """A short-named socket path.

    Not under ``tmp_path``: pytest's per-test directory names push an ``AF_UNIX`` path past
    the 108-byte ``sun_path`` limit. ``code_kernel._bind_rpc_socket`` binds short names in
    ``gettempdir()`` for the same reason.
    """
    return os.path.join(tempfile.mkdtemp(prefix="hbrk_"), "b.sock")


def _start_broker(host_only_value, staging_root, *, sock_path=None, expect_ready=True):
    """Spawn the broker as its own process; return (proc, socket path).

    ``host_only_value`` lands in the BROKER's environment only. A child that can see it
    inherited the broker's env instead of receiving the approved one.
    """
    sock_path = sock_path or _socket_path()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(BROKER_PATH),
            "--socket",
            sock_path,
            "--staging-root",
            str(staging_root),
            "--handshake-timeout",
            str(HANDSHAKE_TIMEOUT),
        ],
        env={
            "PATH": os.environ.get("PATH", ""),
            "BROKER_PROBE_HOST_ONLY": host_only_value,
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if not expect_ready:
        return proc, sock_path
    ready = proc.stdout.readline()
    if not ready:
        proc.terminate()
        pytest.fail(f"broker never became ready; stderr:\n{proc.stderr.read()}")
    assert json.loads(ready)["ready"] is True, f"broker never became ready: {ready!r}"
    return proc, sock_path


def _stop_broker(proc):
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def _pid_running(pid: int) -> bool:
    """True while *pid* is a LIVE process. A killed-but-unreaped child is still a ``/proc``
    entry that answers ``kill(pid, 0)``, so excluding the zombie state makes the broker's own
    reaping part of the contract rather than an implementation detail."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat_line.rsplit(") ", 1)[1].split(" ", 1)[0] != "Z"


def _wait_until_gone(pid: int, timeout: float = DEADLINE) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return True
        time.sleep(0.05)
    return not _pid_running(pid)


def _fd_count(pid: int) -> int:
    return len(os.listdir(f"/proc/{pid}/fd"))


def _wait_for_fd_count(pid: int, ceiling: int, timeout: float = DEADLINE) -> int:
    """Poll the broker's open-descriptor count back down to at most *ceiling*.

    A ceiling rather than equality: worker threads close asynchronously, so an earlier lease
    may still be unwinding when the baseline is sampled. A leak GROWS the count, so the
    inequality still catches it, without turning teardown timing into a flake.
    """
    deadline = time.monotonic() + timeout
    count = _fd_count(pid)
    while time.monotonic() < deadline and count > ceiling:
        time.sleep(0.05)
        count = _fd_count(pid)
    return count


def _read_with_deadline(
    read_fd: int, *, until_eof: bool, timeout: float = DEADLINE
) -> bytes:
    """Read until EOF (or one newline), failing on a deadline instead of hanging.

    A bare blocking ``os.read`` here would turn the regression this file exists to catch —
    the broker retaining a descriptor it forwarded, leaving a live writer on the peer's pipe —
    into an indefinite hang instead of a failure. A hang is a much weaker CI signal.
    """
    os.set_blocking(read_fd, False)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as sel:
        sel.register(read_fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pytest.fail(
                    "pipe never reached EOF: the broker retained a descriptor it was "
                    f"handed (read so far: {b''.join(chunks)!r})"
                )
            if not sel.select(remaining):
                continue
            chunk = os.read(read_fd, 4096)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            if not until_eof and b"\n" in chunk:
                return b"".join(chunks).split(b"\n", 1)[0]


def _child_report(read_fd: int) -> dict:
    """Read the child's JSON report off the SCM_RIGHTS pipe.

    The empty-read guard matters: a child that never ran (or died before writing) closes its
    copy and yields EOF, and a bare ``json.loads`` would report that as an opaque decode
    error instead of naming what actually happened.
    """
    raw = _read_with_deadline(read_fd, until_eof=True)
    assert raw, "the child produced no report: it never ran, or died before writing"
    return json.loads(raw)


def _stage_runner(root, name, source):
    """Write *source* into the staging root, mirroring ``code_kernel._spawn``'s mkdtemp."""
    runner = Path(root) / name
    runner.write_text(textwrap.dedent(source), encoding="utf-8")
    return runner


def _staging_root(tmp_path):
    root = tmp_path / "staging"
    root.mkdir(mode=0o700)
    return root


def _launch_body(path):
    return json.dumps({"op": "launch", "runner": str(path), "env": {}}).encode() + b"\n"


def _raw_request(sock_path: str, body: bytes, fds: list):
    """Send a request the client helper would never construct; return the parsed reply.

    Returns ``None`` if the broker closed the connection without saying anything — which is
    itself a finding: every refusal must be a structured reply, not a silent hangup.
    """
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(DEADLINE)
    try:
        conn.connect(sock_path)
        ancillary = (
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
            if fds
            else []
        )
        _sendmsg_all(conn, body, ancillary)
        buf = b""
        while b"\n" not in buf:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
        return json.loads(buf.split(b"\n", 1)[0]) if b"\n" in buf else None
    finally:
        conn.close()


def _sendmsg_all(conn, body: bytes, ancillary) -> None:
    sent = 0
    first = True
    while sent < len(body):
        written = conn.sendmsg([body[sent:]], ancillary if first else [])
        if written <= 0:
            raise ConnectionError("sendmsg made no progress")
        sent += written
        first = False


@pytest.mark.linux_only
def test_broker_transports_approved_resources_and_refuses_invalid_requests_without_leaking_fds(
    tmp_path,
):
    """Transport + protocol: what crosses the boundary, what is refused, and who owns an fd.

    The accepted launch witnesses all three transports the ``sudo`` carrier severed, and
    witnesses them by MECHANISM, not by side effect: the child reports the inode behind its
    own ``/proc/self/fd`` argv, so a broker that handed over a plain filesystem path instead
    would fail. Every refusal then has to close the descriptors it was handed — the peer's
    pipe reaching EOF is the only proof the broker is not holding its channel open.
    """
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(
        root,
        "runner.py",
        """
        import json, os, sys
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        os.write(fd, json.dumps({
            "approved": os.environ.get("BROKER_PROBE_APPROVED", "<missing>"),
            "host_only_leaked": "BROKER_PROBE_HOST_ONLY" in os.environ,
            "argv0": sys.argv[0],
            "runner_inode": os.stat(sys.argv[0]).st_ino,
            "stderr_target": os.readlink("/proc/self/fd/2"),
        }).encode())
    """,
    )
    outside = tmp_path / "outside.py"
    outside.write_text("raise SystemExit(0)\n", encoding="utf-8")
    escape = root / "escape.py"
    escape.symlink_to(outside)

    proc, sock_path = _start_broker("host-only-secret", root)
    try:
        # Short writes resend only payload bytes; SCM_RIGHTS is attached exactly once.
        class ShortSender:
            def __init__(self, target):
                self.target = target
                self.ancillary_calls = 0

            def sendmsg(self, buffers, ancillary):
                self.ancillary_calls += bool(ancillary)
                return self.target.sendmsg([buffers[0][:3]], ancillary)

        send_sock, recv_sock = socket.socketpair()
        read_fd, write_fd = os.pipe()
        try:
            sender = ShortSender(send_sock)
            broker._sendmsg_all(
                sender, b"short-write-frame", broker._ancillary([write_fd])
            )
            received = b""
            while len(received) < len(b"short-write-frame"):
                received += recv_sock.recv(128)
            assert received == b"short-write-frame"
            assert sender.ancillary_calls == 1
        finally:
            send_sock.close()
            recv_sock.close()
            os.close(read_fd)
            os.close(write_fd)

        # --- accepted launch: env, descriptor and runner all arrive explicitly -----------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply = broker.request_launch(
                sock_path,
                runner=str(runner),
                env={"BROKER_PROBE_APPROVED": "approved-value-42"},
                fds=[write_fd],
            )
            try:
                assert reply["ok"] is True
                os.close(write_fd)
                write_fd = -1
                # The broker resolved and opened the runner before replying, so sealing the
                # staging dir now witnesses the /proc/self/fd hand-off: a child that had to
                # traverse the directory itself could no longer reach the file.
                os.chmod(root, 0o000)
                report = _child_report(read_fd)
            finally:
                os.chmod(root, 0o700)
                conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # (a) the approved value arrived; the broker's own environment did not.
        assert report["approved"] == "approved-value-42"
        assert report["host_only_leaked"] is False
        # (b) the runner crossed as a descriptor, not as a path: argv[0] is the broker's own
        #     open fd, and it resolves to the staged file's inode.
        assert report["argv0"].startswith("/proc/self/fd/")
        assert report["runner_inode"] == runner.stat().st_ino
        # (c) stderr is transported explicitly like stdin/stdout, not inherited from the
        #     broker. An inherited stderr is both a log-forgery channel for a future
        #     lower-privileged child and an undrained pipe the child can wedge itself on.
        assert report["stderr_target"] == "/dev/null"

        # --- refusals: each must be structured, and must close what it was handed --------
        baseline = _wait_for_fd_count(proc.pid, _fd_count(proc.pid))
        refusals = [
            (
                "runner outside the staging root",
                _launch_body(outside),
                1,
                "runner_outside_root",
            ),
            (
                "symlink escaping the staging root",
                _launch_body(escape),
                1,
                "runner_outside_root",
            ),
            (
                "runner that does not exist",
                _launch_body(root / "gone.py"),
                1,
                "launch_failed",
            ),
            ("body that is not JSON", b"{definitely not json\n", 1, "bad_request"),
            (
                "body that is not a launch request",
                b'{"op": "nope"}\n',
                1,
                "bad_request",
            ),
            (
                "environment that is not an object",
                json.dumps({"op": "launch", "runner": str(runner), "env": []}).encode()
                + b"\n",
                1,
                "bad_request",
            ),
            # More descriptors than one control buffer holds: the kernel truncates and sets
            # MSG_CTRUNC, so the broker must refuse rather than launch a child whose
            # HERMES_BROKER_FDS is silently short of what the client passed.
            (
                "more descriptors than MAX_FDS",
                _launch_body(runner),
                broker.MAX_FDS + 3,
                "truncated_ancillary",
            ),
            # A body with no newline in it at all: the cap has to be on the TOTAL received,
            # not on one recv, or a peer can drive the broker to arbitrary memory.
            (
                "body larger than the request cap",
                b"x" * (broker.MAX_REQUEST_BYTES + 4096),
                1,
                "request_too_large",
            ),
        ]
        for label, body, fd_copies, error in refusals:
            read_fd, write_fd = os.pipe()
            try:
                reply = _raw_request(sock_path, body, [write_fd] * fd_copies)
                os.close(write_fd)
                write_fd = -1
                assert reply is not None, f"{label}: broker closed without a reply"
                assert reply["ok"] is False, f"{label}: broker accepted the request"
                assert reply["error"] == error, label
                assert reply["message"], f"{label}: reply carries no message"
                assert _read_with_deadline(read_fd, until_eof=True) == b"", label
            finally:
                os.close(read_fd)
                if write_fd != -1:
                    os.close(write_fd)

        # Descriptor limits apply to the whole request, not one ancillary message.
        read_fd, write_fd = os.pipe()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.settimeout(DEADLINE)
            conn.connect(sock_path)
            first = broker.MAX_FDS // 2 + 1
            conn.sendmsg(
                [b'{"op":'],
                _ancillary := [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        array.array("i", [write_fd] * first),
                    )
                ],
            )
            conn.sendmsg([b'"launch"}\n'], _ancillary)
            reply = json.loads(conn.recv(4096).split(b"\n", 1)[0])
            os.close(write_fd)
            write_fd = -1
            assert reply["error"] == "too_many_fds"
            assert _read_with_deadline(read_fd, until_eof=True) == b""
        finally:
            conn.close()
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # Repeating the cheapest refusal exercises the accumulation the per-arm EOF check
        # cannot see: a per-request fd leak ends at RLIMIT_NOFILE, not at one bad request.
        for _ in range(40):
            _raw_request(sock_path, _launch_body(root / "gone.py"), [])
        assert _wait_for_fd_count(proc.pid, baseline) <= baseline

        # --- and the broker still works -------------------------------------------------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply = broker.request_launch(
                sock_path, runner=str(runner), env={}, fds=[write_fd]
            )
            os.close(write_fd)
            write_fd = -1
            assert reply["ok"] is True
            report = _child_report(read_fd)
            assert report["approved"] == "<missing>"
            conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # A refused request must also be a typed failure on the client, not a JSONDecodeError
        # escaping from inside the helper (which would also strand the client's own socket).
        with pytest.raises(broker.BrokerError) as excinfo:
            broker.request_launch(sock_path, runner=str(outside), env={}, fds=[])
        assert excinfo.value.code == "runner_outside_root"
    finally:
        _stop_broker(proc)


@pytest.mark.linux_only
def test_brokered_child_is_terminated_when_the_client_connection_disappears(
    tmp_path, monkeypatch
):
    """Lease lifetime: nothing outlives its lease, and nothing pins a worker forever.

    The client connection is the broker-owned replacement for the inherited parent-death pipe
    that ``sudo`` closes — the liveness signal crosses the boundary explicitly instead of by
    inheritance. That only holds if it holds for the whole process GROUP (a code kernel or a
    bash runner spawns descendants), for the broker's own shutdown (a lease whose owner dies
    still has to reap), and in the other direction too: a child that exits on its own, or a
    peer that never sends a request, must not park a worker thread forever.
    """
    broker = _load_broker()
    root = _staging_root(tmp_path)
    # Spawns a grandchild in the SAME process group, then reports both pids. A broker that
    # signalled only proc.pid instead of the group would leak the grandchild.
    group_runner = _stage_runner(
        root,
        "group_runner.py",
        """
        import json, os, subprocess, sys, time
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        kid = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        os.write(fd, (json.dumps({"child": os.getpid(), "grandchild": kid.pid}) + "\\n").encode())
        time.sleep(600)
    """,
    )
    quick_runner = _stage_runner(
        root,
        "quick_runner.py",
        """
        import os
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        os.write(fd, b"bye\\n")
    """,
    )
    term_runner = _stage_runner(
        root,
        "term_runner.py",
        """
        import os, signal, time
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        def stop(_signum, _frame):
            os.write(fd, f"term {time.monotonic()}\\n".encode())
            time.sleep(600)
        signal.signal(signal.SIGTERM, stop)
        os.write(fd, b"ready\\n")
        time.sleep(600)
    """,
    )
    proc, sock_path = _start_broker("host-only-secret", root)
    doomed: list = []
    try:
        # --- a live broker owns its socket; a second broker must not steal it -------------
        duplicate, _ = _start_broker(
            "host-only-secret",
            root,
            sock_path=sock_path,
            expect_ready=False,
        )
        try:
            assert duplicate.wait(timeout=HANDSHAKE_TIMEOUT) != 0, (
                "a second broker replaced the live broker's socket"
            )
        finally:
            if duplicate.poll() is None:
                _stop_broker(duplicate)

        # --- the lease governs the whole process group ----------------------------------
        read_fd, write_fd = os.pipe()
        try:
            conn, reply = broker.request_launch(
                sock_path, runner=str(group_runner), env={}, fds=[write_fd]
            )
            os.close(write_fd)
            write_fd = -1
            pids = json.loads(_read_with_deadline(read_fd, until_eof=False))
            doomed.extend(pids.values())
            assert reply["pid"] == pids["child"]
            assert _pid_running(pids["child"]) and _pid_running(pids["grandchild"])

            conn.close()  # the lease disappears

            assert _wait_until_gone(pids["child"]), (
                f"child {pids['child']} outlived the client connection"
            )
            assert _wait_until_gone(pids["grandchild"]), (
                f"grandchild {pids['grandchild']} survived the lease: the broker tore down "
                "the process it spawned, not the process GROUP it leads"
            )
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- a child that exits on its own releases the lease ---------------------------
        # Otherwise the worker thread sits in recv until the client happens to disconnect,
        # holding an unreaped child, and a caller polling the connection never learns the
        # child is gone.
        read_fd, write_fd = os.pipe()
        try:
            conn, reply = broker.request_launch(
                sock_path, runner=str(quick_runner), env={}, fds=[write_fd]
            )
            os.close(write_fd)
            write_fd = -1
            assert _read_with_deadline(read_fd, until_eof=False) == b"bye"
            conn.settimeout(DEADLINE)
            try:
                assert conn.recv(4096) == b""
            except TimeoutError:
                pytest.fail(
                    "the broker held the lease open after the child exited: the worker "
                    "thread is pinned until the client disconnects"
                )
            assert _wait_until_gone(reply["pid"]), "the exited child was never reaped"
            conn.close()
        finally:
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- a peer that never sends a request does not park a worker forever -----------
        silent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        silent.settimeout(DEADLINE)
        try:
            started = time.monotonic()
            silent.connect(sock_path)
            handshake = silent.recv(4096)
            elapsed = time.monotonic() - started
            assert handshake, "the broker closed a silent peer without saying why"
            assert (
                json.loads(handshake.split(b"\n", 1)[0])["error"] == "handshake_timeout"
            )
            assert elapsed >= HANDSHAKE_TIMEOUT, (
                f"the broker gave up after {elapsed:.2f}s, before the handshake window"
            )
        finally:
            silent.close()

        # The handshake window is cumulative: traffic cannot renew it one byte at a time.
        drip = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        drip.settimeout(DEADLINE)
        try:
            drip.connect(sock_path)
            started = time.monotonic()
            for byte in b'{"op":':
                time.sleep(HANDSHAKE_TIMEOUT / 5)
                try:
                    drip.sendall(bytes([byte]))
                except BrokenPipeError:
                    break
            reply = drip.recv(4096)
            elapsed = time.monotonic() - started
            assert json.loads(reply.split(b"\n", 1)[0])["error"] == "handshake_timeout"
            assert elapsed < HANDSHAKE_TIMEOUT * 1.5, (
                f"slow-drip bytes renewed the handshake deadline for {elapsed:.2f}s"
            )
        finally:
            drip.close()

        # --- the broker's own SIGTERM tears down every lease it is still holding --------
        read_fd, write_fd = os.pipe()
        held = None
        term_leases = []
        term_reads = []
        try:
            held, reply = broker.request_launch(
                sock_path, runner=str(group_runner), env={}, fds=[write_fd]
            )
            os.close(write_fd)
            write_fd = -1
            pids = json.loads(_read_with_deadline(read_fd, until_eof=False))
            doomed.extend(pids.values())

            for _ in range(2):
                term_read, term_write = os.pipe()
                term_conn, term_reply = broker.request_launch(
                    sock_path, runner=str(term_runner), env={}, fds=[term_write]
                )
                os.close(term_write)
                assert _read_with_deadline(term_read, until_eof=False) == b"ready"
                term_leases.append(term_conn)
                term_reads.append(term_read)
                doomed.append(term_reply["pid"])

            # The lease is still HELD: only the broker's death can end it.
            proc.send_signal(signal.SIGTERM)
            term_times = [
                float(_read_with_deadline(fd, until_eof=False).split()[1])
                for fd in term_reads
            ]
            assert max(term_times) - min(term_times) < 0.5, (
                "shutdown waited on one lease before notifying the next"
            )
            assert proc.wait(timeout=DEADLINE) is not None

            assert _wait_until_gone(pids["child"]), (
                f"child {pids['child']} was orphaned by the broker's shutdown; "
                "start_new_session detached it, so nothing else will ever reap it"
            )
            assert _wait_until_gone(pids["grandchild"]), (
                f"grandchild {pids['grandchild']} was orphaned by the broker's shutdown"
            )
            assert not os.path.exists(sock_path), (
                "the broker left its socket behind on a clean shutdown"
            )
        finally:
            if held is not None:
                held.close()
            for term_conn in term_leases:
                term_conn.close()
            for term_read in term_reads:
                os.close(term_read)
            os.close(read_fd)
            if write_fd != -1:
                os.close(write_fd)

        # --- restart over a STALE socket, but never over something that is not one ------
        stale_path = _socket_path()
        killed, _ = _start_broker("host-only-secret", root, sock_path=stale_path)
        killed.kill()
        killed.wait(timeout=DEADLINE)
        assert os.path.exists(stale_path), "expected an uncleaned socket after SIGKILL"
        contenders = [
            _start_broker(
                "host-only-secret", root, sock_path=stale_path, expect_ready=False
            )[0]
            for _ in range(2)
        ]
        deadline = time.monotonic() + DEADLINE
        while time.monotonic() < deadline and all(p.poll() is None for p in contenders):
            time.sleep(0.05)
        losers = [p for p in contenders if p.poll() is not None]
        winners = [p for p in contenders if p.poll() is None]
        assert len(losers) == len(winners) == 1
        winner = winners[0]
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(stale_path)
        finally:
            probe.close()
        assert os.path.exists(stale_path), "losing startup unlinked the winner's socket"
        _stop_broker(winner)

        # Cleanup is conditional on the bound pathname still naming this broker's socket.
        owned_path = _socket_path()
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        replacement = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            first.bind(owned_path)
            identity = broker._socket_identity(owned_path)
            os.unlink(owned_path)
            replacement.bind(owned_path)
            broker._unlink_owned_socket(owned_path, identity)
            assert os.path.exists(owned_path), "cleanup unlinked a replacement socket"
        finally:
            first.close()
            replacement.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(owned_path)

        occupied = Path(_socket_path())
        occupied.write_text("not a socket\n", encoding="utf-8")
        refused, _ = _start_broker(
            "host-only-secret", root, sock_path=str(occupied), expect_ready=False
        )
        assert refused.wait(timeout=DEADLINE) != 0
        assert occupied.read_text(encoding="utf-8") == "not a socket\n", (
            "the broker unlinked a path that was not a socket"
        )

        shared_root = tmp_path / "shared-staging"
        shared_root.mkdir(mode=0o750)
        refused_root, refused_path = _start_broker(
            "host-only-secret", shared_root, expect_ready=False
        )
        assert refused_root.wait(timeout=DEADLINE) != 0
        assert not os.path.exists(refused_path)

        # Teardown has exactly one owner, including a child spawned while shutdown is
        # taking its registry snapshot. A late registration is handed back to its worker;
        # a registered lease can be claimed only once.
        leases = broker._Leases()
        registry_client, registry_conn = socket.socketpair()
        registered = object()
        try:
            assert leases.register(registry_conn, threading.current_thread()) is True
            assert leases.add(registry_conn, registered) is True
            assert leases.claim(registry_conn, registered) is True
            assert leases.claim(registry_conn, registered) is False
            leases.finished(registry_conn)
            leases.drain()
            late_client, late_conn = socket.socketpair()
            try:
                assert leases.register(late_conn, threading.current_thread()) is False
            finally:
                late_client.close()
                late_conn.close()
        finally:
            registry_client.close()
            registry_conn.close()

        # A child still unreaped at the kill deadline retains a waiter that reaps it later.
        class DelayedExit:
            pid = 999_999_999
            returncode = None

            def __init__(self):
                self.waits = 0

            def wait(self, timeout=None):
                self.waits += 1
                if timeout is not None:
                    raise subprocess.TimeoutExpired("delayed", timeout)
                self.returncode = 0

        delayed = DelayedExit()
        waiter_daemon = None

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                nonlocal waiter_daemon
                waiter_daemon = daemon
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(broker, "_signal_group", lambda *_args: None)
        monkeypatch.setattr(broker, "_wait_unreaped", lambda *_args: None)
        monkeypatch.setattr(broker.threading, "Thread", ImmediateThread)
        broker._terminate_many([delayed])
        assert delayed.returncode == 0
        assert delayed.waits == 2
        assert waiter_daemon is True
    finally:
        # Best-effort sweep for a run that already failed. RuntimeError is suppressed
        # alongside OSError because ``tests/conftest.py``'s live-system guard refuses
        # ``os.kill`` outside the test subtree — which is precisely where an orphaned
        # grandchild ends up. Letting it raise here would bury the assertion that failed.
        for pid in doomed:
            if _pid_running(pid):
                with contextlib.suppress(OSError, RuntimeError):
                    os.kill(
                        pid,
                        signal.SIGKILL,  # windows-footgun: ok — linux_only test
                    )
        _stop_broker(proc)


@pytest.mark.linux_only
def test_shutdown_waits_for_a_worker_spawning_before_process_registration(
    tmp_path, monkeypatch
):
    broker = _load_broker()
    root = _staging_root(tmp_path)
    runner = _stage_runner(root, "blocked_launch.py", "import time; time.sleep(600)\n")
    client, accepted = socket.socketpair()
    leases = broker._Leases()
    launched = threading.Event()
    release_launch = threading.Event()
    child_pid = None
    real_launch = broker._launch

    def launch_then_block(*args, **kwargs):
        nonlocal child_pid
        proc = real_launch(*args, **kwargs)
        child_pid = proc.pid
        launched.set()
        assert release_launch.wait(DEADLINE)
        return proc

    monkeypatch.setattr(broker, "_launch", launch_then_block)
    worker = threading.Thread(
        target=broker._serve_connection,
        args=(accepted, str(root), HANDSHAKE_TIMEOUT, leases),
        daemon=True,
    )
    leases.register(accepted, worker)
    worker.start()
    drain = threading.Thread(target=leases.drain)
    try:
        client.sendall(_launch_body(runner))
        assert launched.wait(DEADLINE), "the real child was not spawned"
        drain.start()
        drain.join(0.2)
        assert drain.is_alive(), (
            "drain returned while a registered worker's spawned child was still "
            "between Popen and process registration"
        )
        release_launch.set()
        drain.join(DEADLINE)
        assert not drain.is_alive(), "drain did not wait for the registered worker"
        assert child_pid is not None and _wait_until_gone(child_pid)

        failed_client, failed_conn = socket.socketpair()

        class FailingThread:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start failed")

        monkeypatch.setattr(broker.threading, "Thread", FailingThread)
        failed_leases = broker._Leases()
        with pytest.raises(RuntimeError, match="thread start failed"):
            broker._start_worker(
                failed_conn, str(root), HANDSHAKE_TIMEOUT, failed_leases
            )
        failed_client.settimeout(DEADLINE)
        assert failed_client.recv(1) == b""
        failed_leases.drain()
        failed_client.close()
    finally:
        release_launch.set()
        client.close()
        accepted.close()
        worker.join(DEADLINE)
        drain.join(DEADLINE)
        if child_pid is not None and _pid_running(child_pid):
            os.kill(
                child_pid,
                signal.SIGKILL,  # windows-footgun: ok — linux_only test
            )
