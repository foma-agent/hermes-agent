"""Literal launch wrapper regressions for one-operation policy consent."""
import json

import pytest

from tools import approval


@pytest.fixture
def parent(tmp_path, monkeypatch):
    (tmp_path / 'config.yaml').write_text(json.dumps({'approvals': {'mode': 'manual'}}))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_INTERACTIVE', '1')
    monkeypatch.delenv('HERMES_SINGLE_QUERY_SESSION', raising=False)


def check_consent(command, expected):
    calls = []

    def deny(*args, **kwargs):
        calls.append(kwargs)
        return 'deny'

    result = approval.check_all_command_guards(command, 'local', approval_callback=deny)
    if expected:
        assert result['approved'] is False
        assert len(calls) == 1
        assert calls[0]['allow_session'] is False
        assert calls[0]['allow_permanent'] is False
    else:
        assert result['approved'] is True
        assert calls == []


@pytest.mark.parametrize('command, expected', [
    ('python -- /checkout/hermes chat --yolo', True),
    ('python -I -- hermes_cli/main.py chat --yolo', True),
    ('python -- /checkout/hermes config set approvals.mode automatic', True),
    ('python -- other.py hermes chat --yolo', False),
    ('python -- -m hermes_cli.main chat --yolo', False),
])
def test_python_script_terminator(parent, command, expected):
    check_consent(command, expected)


@pytest.mark.parametrize('command, expected', [
    ('uv tool run hermes chat --yolo', True),
    ('uv tool run --from hermes-agent hermes chat --yolo', True),
    ('uvx --from hermes-agent hermes chat --yolo', True),
    ('uv tool run python -m hermes_cli.main chat --yolo', True),
    ('uv tool run echo hermes chat --yolo', False),
    ('uv tool run python other.py hermes chat --yolo', False),
    ('uv tool run --from hermes echo chat --yolo', False),
    ('uvx --python hermes echo chat --yolo', False),
    ('uv --project hermes tool run echo chat --yolo', False),
    ('uv tool run hermes chat --query --yolo', False),
])
def test_runner_executable_position(parent, command, expected):
    check_consent(command, expected)


@pytest.mark.parametrize('command, expected', [
    ('''env -S "bash -c 'hermes chat --yolo'"''', True),
    ('''HERMES_YOLO_MODE=1 env -S "bash -c 'hermes chat'"''', True),
    ('''HERMES_YOLO_MODE=1 env -i -S "bash -c 'hermes chat'"''', False),
    ('''env -S "HERMES_YOLO_MODE=false bash -c 'hermes chat'"''', False),
    ('''HERMES_YOLO_MODE=1 env -S "bash -c 'env HERMES_YOLO_MODE=false hermes chat'"''', False),
    ('''env -S "echo bash -c 'hermes chat --yolo'"''', False),
    ('''env -S "bash script.sh -c 'hermes chat --yolo'"''', False),
    ('''env -S "bash -c 'echo hermes chat --yolo'"''', False),
])
def test_env_split_shell_payload(parent, command, expected):
    if not expected:
        # A shell -c may still require generic consent; only policy consent must be absent.
        calls = []

        def deny(*args, **kwargs):
            calls.append(kwargs)
            return 'deny'

        approval.check_all_command_guards(command, 'local', approval_callback=deny)
        assert all(call.get('allow_session', True) for call in calls)
        assert all(call.get('allow_permanent', True) for call in calls)
        return
    check_consent(command, expected)
