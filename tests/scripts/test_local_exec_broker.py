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

The broker transports each of those explicitly instead, so these tests pin the transport
contract rather than the eventual uid switch: an approved env value arrives (and the
broker's own environment does not), a descriptor the broker only ever learns about through
``SCM_RIGHTS`` is the channel the child reports on, and the staged runner executes while its
directory stays 0700.

Real processes throughout: a separately spawned broker, a real ``AF_UNIX`` connection, a real
child. Nothing here is mocked — the point is the boundary.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BROKER_PATH = REPO_ROOT / "scripts" / "local_exec_broker.py"


def _load_broker():
    """Import the broker script as a module (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("local_exec_broker", BROKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_exec_broker"] = module
    spec.loader.exec_module(module)
    return module


def _start_broker(host_only_value):
    """Spawn the broker as its own process; return (proc, socket path).

    ``host_only_value`` lands in the BROKER's environment only. A child that can see it
    inherited the broker's env instead of receiving the approved one.

    The socket lives in its own short-named temp dir, not under ``tmp_path``: pytest's
    per-test directory names push an ``AF_UNIX`` path past the 108-byte sun_path limit.
    ``code_kernel._bind_rpc_socket`` binds short names in ``gettempdir()`` for the same reason.
    """
    sock_dir = tempfile.mkdtemp(prefix="hbrk_")
    sock_path = os.path.join(sock_dir, "b.sock")
    proc = subprocess.Popen(
        [sys.executable, str(BROKER_PATH), "--socket", sock_path],
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
    ready = proc.stdout.readline()
    if not ready:
        proc.terminate()
        pytest.fail(f"broker never became ready; stderr:\n{proc.stderr.read()}")
    assert json.loads(ready)["ready"] is True, f"broker never became ready: {ready!r}"
    return proc, sock_path


def _pid_running(pid: int) -> bool:
    """True while *pid* is a LIVE process. A killed-but-unreaped child is still a ``/proc``
    entry that answers ``kill(pid, 0)``, so excluding the zombie state makes the broker's own
    reaping part of the contract rather than an implementation detail."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat_line.rsplit(") ", 1)[1].split(" ", 1)[0] != "Z"


def _wait_until_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            return True
        time.sleep(0.05)
    return not _pid_running(pid)


def _stage_runner(tmp_path, source):
    """Write *source* into a 0700 staging dir, mirroring ``code_kernel._spawn``'s mkdtemp."""
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    runner = staging / "runner.py"
    runner.write_text(textwrap.dedent(source), encoding="utf-8")
    return staging, runner


@pytest.mark.linux_only
def test_brokered_child_gets_approved_env_and_scm_rights_fd_without_opening_staging_dir(
    tmp_path,
):
    """One launch proves all three transports the sudo carrier severed."""
    broker = _load_broker()
    staging, runner = _stage_runner(
        tmp_path,
        """
        import os
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        approved = os.environ.get("BROKER_PROBE_APPROVED", "<missing>")
        host_only = "LEAKED" if "BROKER_PROBE_HOST_ONLY" in os.environ else "clean"
        os.write(fd, ("%s|%s" % (approved, host_only)).encode())
    """,
    )
    proc, sock_path = _start_broker("host-only-secret")
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
            # EOF only arrives once the broker has closed its own copy of the descriptor,
            # so reading to EOF also proves the broker does not retain what it forwards.
            chunks = []
            while chunk := os.read(read_fd, 4096):
                chunks.append(chunk)
        finally:
            conn.close()
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)

    # (a) the approved value arrived; the broker's own environment did not.
    # (b) the only channel the child could have reported on is the SCM_RIGHTS descriptor:
    #     the pipe has no filesystem name, so the broker learned of it no other way.
    assert b"".join(chunks).decode() == "approved-value-42|clean"
    # (c) the runner ran, and its directory is still owner-only.
    assert stat.S_IMODE(staging.stat().st_mode) == 0o700


@pytest.mark.linux_only
def test_brokered_child_is_terminated_when_the_client_connection_disappears(tmp_path):
    """The client connection is the child's lease: its EOF tears the child down.

    This is the broker-owned replacement for the inherited parent-death pipe that ``sudo``
    closes — the liveness signal crosses the boundary explicitly instead of by inheritance.
    """
    broker = _load_broker()
    _staging, runner = _stage_runner(
        tmp_path,
        """
        import os, time
        fd = int(os.environ["HERMES_BROKER_FDS"].split(",")[0])
        os.write(fd, b"ready")
        time.sleep(600)
    """,
    )
    proc, sock_path = _start_broker("host-only-secret")
    read_fd, write_fd = os.pipe()
    child_pid = None
    try:
        conn, reply = broker.request_launch(
            sock_path, runner=str(runner), env={}, fds=[write_fd]
        )
        os.close(write_fd)
        write_fd = -1
        child_pid = reply["pid"]
        assert os.read(read_fd, 5) == b"ready"
        assert _pid_running(child_pid)

        conn.close()  # the lease disappears

        assert _wait_until_gone(child_pid, 10.0), (
            f"child {child_pid} outlived the client connection"
        )
    finally:
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
        if child_pid is not None and _pid_running(child_pid):
            with contextlib.suppress(OSError):
                os.kill(
                    child_pid,
                    signal.SIGKILL,  # windows-footgun: ok — linux_only test
                )
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
