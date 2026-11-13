"""Literal child bypass front door: argv data is not launch authority."""
import json

import pytest

from tools import approval


@pytest.fixture
def parent(tmp_path, monkeypatch):
    (tmp_path / 'config.yaml').write_text(json.dumps({'approvals': {'mode': 'manual'}}))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_INTERACTIVE', '1')
    monkeypatch.delenv('HERMES_SINGLE_QUERY_SESSION', raising=False)


@pytest.mark.parametrize('command', [
    'hermes --yolo chat -q hello',
    'hermes chat --yolo -q hello',
    'python -Imhermes_cli.main chat --yolo -q hello',
    'python /checkout/hermes chat --yolo -q hello',
    'python hermes_cli/main.py chat --yolo',
    'HERMES_YOLO_MODE=1 hermes chat -q hello',
    'HERMES_YOLO_MODE=" YES " timeout 30 python -m hermes_cli.main chat',
    'env HERMES_YOLO_MODE=on hermes chat',
    'env -i HERMES_YOLO_MODE=TrUe python -m hermes_cli.main chat',
    "env -S 'HERMES_YOLO_MODE=1 hermes chat'",
    "env -S 'hermes chat' --yolo",
    'command hermes chat --yolo',
    'timeout 30 env HERMES_YOLO_MODE=1 hermes chat',
    'uv run --project /checkout hermes chat --yolo',
    "bash -c 'HERMES_YOLO_MODE=1 hermes chat'",
])
def test_bypass_spelling_requires_fresh_consent(parent, command):
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return 'deny'

    result = approval.check_all_command_guards(command, 'local', approval_callback=deny)
    assert result['approved'] is False
    assert len(calls) == 1
    assert calls[0][1]['allow_permanent'] is False
    assert calls[0][1]['allow_session'] is False


@pytest.mark.parametrize('command', [
    'hermes chat -q "explain --yolo"',
    "hermes chat --query='--yolo'",
    'hermes --model "--yolo" chat -q hello',
    'hermes -z "--yolo"',
    'hermes chat -- --yolo',
    'hermes config set display.label "--yolo"',
    'hermes chat -q hello --source "--yolo"',
    'echo hermes chat --yolo',
    "uv run echo 'hermes' '--yolo'",
    "uv run python other.py 'hermes' '--yolo'",
    "uv run --project hermes echo '--yolo'",
    "poetry run printf '%s' 'hermes' '--yolo'",
    "printf '%s' 'HERMES_YOLO_MODE=1 hermes chat --yolo'",
    'env HERMES_YOLO_MODE=0 hermes chat',
    'HERMES_YOLO_MODE=false hermes chat',
    'env HERMES_YOLO_MODE=not-true hermes chat',
    'env HERMES_YOLO_MODE=1 echo hermes chat',
    'HERMES_YOLO_MODE=1 python other.py --yolo',
    "env -S 'echo hermes chat --yolo'",
    'env HERMES_YOLO_MODE=1 HERMES_YOLO_MODE=off hermes chat',
    'HERMES_YOLO_MODE=1 env -u HERMES_YOLO_MODE hermes chat',
    'HERMES_YOLO_MODE=1 env -i hermes chat',
    'printf "%s" "HERMES_YOLO_MODE=1"; hermes chat',
])
def test_argument_data_and_false_environment_need_no_consent(parent, command):
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return 'deny'

    result = approval.check_all_command_guards(command, 'local', approval_callback=deny)
    assert result['approved'] is True
    assert calls == []
