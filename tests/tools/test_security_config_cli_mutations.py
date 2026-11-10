"""Policy CLI mutations require fresh human consent (PR #59337 / issue #59293)."""
import json
from pathlib import Path
import shlex
import sys
import uuid

import pytest

import hermes_yaml as yaml

from tools import approval, terminal_tool as terminal
from tools.approval_context import set_current_session_key, reset_current_session_key


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        "approvals:\n  mode: manual\n  deny: []\n"
        "display:\n  compact: true\ncommand_allowlist: []\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    yield config
    terminal.set_approval_callback(None)


def _command(arguments):
    launcher = Path(__file__).resolve().parents[2] / "hermes"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(launcher))} config {arguments}"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("arguments", [
    "set approvals.mode off",
    "unset approvals.deny",
    "unset 'approvals.deny'",
    'unset "approvals"',
    "unset approvals.deny; :",
    "unset approvals; :",
    "unset 'approvals.deny'; :",
    'unset "approvals"; :',
    "unset approvals.deny && :",
    "unset approvals # literal shell comment",
    "unset display.compact",
])
def test_cli_mutations_require_consent_before_terminal_execution(policy_home, arguments):
    before = policy_home.read_bytes()
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return "deny"

    command = _command(arguments)
    protected = "approvals" in arguments
    decision = approval.check_all_command_guards(command, "local", approval_callback=deny)
    assert decision["approved"] is (not protected)
    terminal.set_approval_callback(deny)
    result = json.loads(terminal.terminal_tool(command))
    if protected:
        assert result.get("status") == "blocked", result
        assert len(calls) == 2
        assert policy_home.read_bytes() == before
    else:
        assert result.get("exit_code") == 0, result
        assert "compact" not in yaml.safe_load(policy_home.read_text()).get("display", {})
        assert not calls


@pytest.mark.parametrize("grant", ["always", "session", "prefix"])
def test_policy_consent_cannot_be_reused(policy_home, monkeypatch, grant):
    session = uuid.uuid4().hex
    monkeypatch.setenv("HERMES_SESSION_KEY", session)
    token = set_current_session_key(session)
    command = _command("unset approvals.deny; :")
    before = policy_home.read_bytes()
    calls = []

    def choose(*args, **kwargs):
        calls.append((args, kwargs))
        return "always" if grant == "always" and len(calls) == 1 else "deny"

    try:
        if grant == "prefix":
            data = yaml.safe_load(policy_home.read_text())
            data["command_allowlist"] = [sys.executable + " *"]
            policy_home.write_text(yaml.safe_dump(data))
            before = policy_home.read_bytes()
        elif grant == "session":
            # Simulate a grant retained by a client predating one-shot policy consent.
            monkeypatch.setitem(approval._session_approved, session, {
                "modify Hermes security policy via config",
            })
        if grant == "always":
            assert approval.check_all_command_guards(command, "local", approval_callback=choose)["approved"]
        result = approval.check_all_command_guards(command, "local", approval_callback=choose)
        assert result["approved"] is False
        assert len(calls) == (2 if grant == "always" else 1)
        assert policy_home.read_bytes() == before
    finally:
        approval.clear_session(session)
        reset_current_session_key(token)
