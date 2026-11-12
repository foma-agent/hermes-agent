"""Smart-approval reviewer changes need fresh consent (#59293)."""
import json
from pathlib import Path
import shlex
import sys
import uuid

import pytest

from tools import approval, terminal_tool as terminal


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("arguments", [
    "set auxiliary.approval.model replacement",
    "set auxiliary.approval.provider custom",
    "set auxiliary.approval.base_url https://reviewer.invalid/v1",
    "set auxiliary.approval.fallback_chain '[]'",
    "set --force auxiliary '{\"approval\": {\"model\": \"replacement\"}}'",
    "unset auxiliary.approval",
    "unset auxiliary",
])
def test_reviewer_mutation_is_refused_before_terminal_effect(tmp_path, monkeypatch, arguments):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_text(
        "approvals:\n  mode: manual\n"
        "auxiliary:\n  approval:\n    model: original\n    provider: original-provider\n"
        "    base_url: https://original.invalid/v1\n    fallback_chain: []\n"
    )
    before = config.read_bytes()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_KEY", uuid.uuid4().hex)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return "deny"

    launcher = Path(__file__).resolve().parents[2] / "hermes"
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(launcher))} config {arguments}"
    terminal.set_approval_callback(deny)
    try:
        result = json.loads(terminal.terminal_tool(command, timeout=30, workdir=str(tmp_path)))
        assert result.get("status") == "blocked" and config.read_bytes() == before, result
        assert len(calls) == 1
    finally:
        terminal.set_approval_callback(None)


@pytest.mark.parametrize("command", [
    "hermes config get auxiliary.approval.model",
    "hermes config set auxiliary.title.model replacement",
    "hermes config unset auxiliary.title",
    "hermes config set auxiliary.approval_notes ordinary",
])
def test_unrelated_auxiliary_settings_remain_ordinary(command):
    assert approval.detect_dangerous_command(command) == (False, None, None)
