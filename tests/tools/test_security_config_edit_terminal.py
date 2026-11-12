"""The config editor is a whole-policy writer (#59293 / #59337)."""
import json
from pathlib import Path
import shlex
import sys
import uuid

import pytest

from tools import approval, terminal_tool as terminal
from tools.approval_context import reset_current_session_key, set_current_session_key


@pytest.fixture
def config_editor(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text("approvals:\n  mode: manual\ndisplay:\n  compact: true\n")
    marker = tmp_path / "editor-called.json"
    editor = tmp_path / "fixture-editor"
    editor.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "config = pathlib.Path(sys.argv[1])\n"
        "config.write_text(config.read_text().replace('mode: manual', \"mode: 'off'\"))\n"
    )
    editor.chmod(0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    launcher = Path(__file__).resolve().parents[2] / "hermes"
    command = (
        f"EDITOR={shlex.quote(str(editor))} {shlex.quote(sys.executable)} "
        f"{shlex.quote(str(launcher))} config edit"
    )
    yield config, marker, command
    terminal.set_approval_callback(None)


@pytest.mark.platforms("posix")
def test_config_edit_denial_prevents_editor_start(config_editor):
    config, marker, command = config_editor
    before = config.read_bytes()
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return "deny"

    terminal.set_approval_callback(deny)
    result = json.loads(terminal.terminal_tool(command, timeout=30))
    assert result.get("status") == "blocked", {
        "result": result, "editor_started": marker.exists(), "config": config.read_text(),
    }
    assert not marker.exists()
    assert config.read_bytes() == before
    assert len(calls) == 1
    assert calls[0][1]["allow_permanent"] is False


@pytest.mark.platforms("posix")
def test_config_edit_once_launches_editor_on_exact_config(config_editor):
    config, marker, command = config_editor
    calls = []

    def once(*args, **kwargs):
        calls.append((args, kwargs))
        return "once"

    terminal.set_approval_callback(once)
    result = json.loads(terminal.terminal_tool(command, timeout=30))
    assert result.get("exit_code") == 0, result
    assert json.loads(marker.read_text()) == [str(config)]
    assert "mode: 'off'" in config.read_text()
    assert len(calls) == 1
    assert calls[0][1]["allow_permanent"] is False
    assert calls[0][1]["allow_session"] is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("grant", ["always", "session", "prefix"])
def test_config_edit_requires_fresh_consent(config_editor, monkeypatch, grant):
    config, marker, command = config_editor
    before = config.read_text()
    session = uuid.uuid4().hex
    monkeypatch.setenv("HERMES_SESSION_KEY", session)
    token = set_current_session_key(session)
    calls = []

    def choose(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return "always" if grant == "always" else "once"
        return "deny"

    terminal.set_approval_callback(choose)
    try:
        first = json.loads(terminal.terminal_tool(command, timeout=30))
        assert first.get("exit_code") == 0, first
        assert json.loads(marker.read_text()) == [str(config)]
        # Restore manual mode so the second call tests retained consent, not mode=off.
        config.write_text(before)
        marker.unlink()
        if grant == "session":
            monkeypatch.setitem(approval._session_approved, session, {
                "modify Hermes security policy via config",
            })
        elif grant == "prefix":
            config.write_text(before + f"command_allowlist:\n  - {sys.executable} *\n")
        second_before = config.read_bytes()
        second = json.loads(terminal.terminal_tool(command, timeout=30))
        assert second.get("status") == "blocked", second
        assert len(calls) == 2
        assert all(call[1]["allow_permanent"] is False for call in calls)
        assert all(call[1]["allow_session"] is False for call in calls)
        assert not marker.exists()
        assert config.read_bytes() == second_before
    finally:
        approval.clear_session(session)
        reset_current_session_key(token)


@pytest.mark.parametrize("command", [
    "hermes config edit",
    "'/opt/hermes' --profile disposable 'config' \"edit\"",
    "python -m hermes_cli.main config edit",
    "python hermes_cli/main.py config edit",
    "python /checkout/hermes config edit",
    "EDITOR=fixture-editor env hermes config edit",
    "env -S 'hermes config edit'",
    "uv run hermes config edit",
    "bash -c 'hermes config edit'",
])
def test_config_edit_spellings_require_nonpersistable_consent(config_editor, command):
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return "deny"

    result = approval.check_all_command_guards(command, "local", approval_callback=deny)
    assert result["approved"] is False
    assert len(calls) == 1
    assert calls[0][1]["allow_permanent"] is False
    assert calls[0][1]["allow_session"] is False


@pytest.mark.parametrize("command", [
    "hermes config get approvals.mode",
    "hermes config set display.compact false",
    "hermes config",
    "echo hermes config edit",
    "printf '%s' 'hermes config edit'",
    "hermes config set display.label 'hermes config edit'",
])
def test_ordinary_config_and_argument_data_need_no_policy_consent(config_editor, command):
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return "deny"

    result = approval.check_all_command_guards(command, "local", approval_callback=deny)
    assert result["approved"] is True
    assert not calls
