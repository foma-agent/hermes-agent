"""Plugin policy changes require fresh consent before terminal execution."""
import json
from pathlib import Path
import shlex
import sys
import uuid

import pytest

import hermes_yaml as yaml

from tools import approval, terminal_tool as terminal


@pytest.fixture
def policy_profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        "approvals:\n  mode: manual\n"
        "plugins:\n  enabled: [opaque]\n  disabled: []\n"
    )
    plugin = home / "plugins" / "opaque"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: opaque\nversion: '1.0'\n")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    return config


def _run_cli(arguments, workdir):
    launcher = Path(__file__).resolve().parents[2] / "hermes"
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(launcher))} {arguments}"
    return json.loads(terminal.terminal_tool(command, timeout=30, workdir=str(workdir)))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("arguments", [
    "plugins disable opaque",
    "plugins remove opaque",
    "config set plugins.enabled '[]'",
    "config set plugins.disabled '[opaque]'",
    "config set --force plugins '{\"enabled\": [], \"disabled\": [\"opaque\"]}'",
    "config unset plugins.enabled",
    "config unset plugins.disabled",
    "config unset plugins",
])
def test_plugin_policy_is_denied_before_cli_execution(tmp_path, policy_profile, arguments):
    before = policy_profile.read_bytes()
    plugin = policy_profile.parent / "plugins" / "opaque"
    plugin_files = {
        name: (plugin / name).read_bytes() for name in ("plugin.yaml", "__init__.py")
    }
    calls = []

    def assert_preserved():
        assert policy_profile.read_bytes() == before
        assert plugin.is_dir()
        assert {name: (plugin / name).read_bytes() for name in plugin_files} == plugin_files

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        assert_preserved()
        assert kwargs["allow_session"] is False
        assert kwargs["allow_permanent"] is False
        return "deny"

    terminal.set_approval_callback(deny)
    try:
        result = _run_cli(arguments, tmp_path)
        assert result.get("status") == "blocked", result
        assert_preserved()
        assert len(calls) == 1
        assert calls[0][1]["allow_session"] is False
        assert calls[0][1]["allow_permanent"] is False
    finally:
        terminal.set_approval_callback(None)


@pytest.mark.parametrize("command", [
    "hermes plugins list",
    "hermes plugins show opaque",
    "hermes config get plugins.enabled",
    "hermes config show",
    "hermes config set plugins_notes ordinary",
    "hermes config unset plugins_notes",
    "hermes config set display.skin default",
    "echo hermes plugins disable opaque",
    "printf '%s' 'hermes config unset plugins'",
])
def test_ordinary_commands_do_not_require_policy_consent(command):
    assert approval.detect_dangerous_command(command) == (False, None, None)


@pytest.mark.platforms("posix")
def test_explicit_once_executes_config_writer_without_retaining_consent(tmp_path, policy_profile):
    calls = []

    def decide(*args, **kwargs):
        calls.append((args, kwargs))
        return "once" if len(calls) == 1 else "deny"

    terminal.set_approval_callback(decide)
    try:
        result = _run_cli("config set plugins.disabled '[opaque]'", tmp_path)
        assert result.get("exit_code") == 0, result
        assert yaml.safe_load(policy_profile.read_text())["plugins"]["disabled"] == ["opaque"]
        before = policy_profile.read_bytes()
        result = _run_cli("config unset plugins", tmp_path)
        assert result.get("status") == "blocked", result
        assert policy_profile.read_bytes() == before
        assert len(calls) == 2
        for _, kwargs in calls:
            assert kwargs["allow_session"] is False
            assert kwargs["allow_permanent"] is False
    finally:
        terminal.set_approval_callback(None)
