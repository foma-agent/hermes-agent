"""Policy scanning shares the bounded lexer budget; uncertainty stays human-only."""
import pytest

from tools import approval, approval_detection, approval_parser_budget


def test_policy_scan_records_uncertainty_when_hardline_scan_finished(monkeypatch):
    command = 'git push --force; ' + 'env HERMES_YOLO_MODE=0 ' * 50 + 'hermes config set approvals.mode off'
    # Bound the real lexer, not its result. The hardline pass finishes within
    # 100,000 units; resolving these literal wrappers consumes more than that.
    monkeypatch.setattr(approval_parser_budget, 'MAX_SCAN_UNITS', 100_000)
    monkeypatch.setattr(approval_parser_budget, 'SCAN_UNITS_PER_CHAR', 0)
    assert approval_detection.detect_hardline_command(command) == (False, None)
    assert approval_parser_budget.recorded_scan(command) is True
    assert approval_detection.detect_dangerous_command(command)[0]
    assert approval_parser_budget.recorded_scan(command) is False


@pytest.mark.parametrize('guard_name', ['check_all_command_guards', 'check_dangerous_command'])
def test_policy_scan_uncertainty_cannot_use_a_command_allowlist(tmp_path, monkeypatch, guard_name):
    command = 'env HERMES_YOLO_MODE=0 ' * 50 + 'hermes config set approvals.mode off'
    (tmp_path / 'config.yaml').write_text("approvals:\n  mode: manual\ncommand_allowlist:\n  - 'env *'\n")
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_INTERACTIVE', '1')
    monkeypatch.setenv('HERMES_SESSION_KEY', 'policy-budget-' + tmp_path.name)
    monkeypatch.setattr(approval, '_YOLO_MODE_FROZEN', False)
    monkeypatch.setattr(approval_parser_budget, 'MAX_SCAN_UNITS', 100_000)
    monkeypatch.setattr(approval_parser_budget, 'SCAN_UNITS_PER_CHAR', 0)
    assert approval_detection.detect_hardline_command(command) == (False, None)
    assert approval_parser_budget.recorded_scan(command) is True
    assert approval._command_matches_permanent_allowlist(command)
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return 'deny'

    result = getattr(approval, guard_name)(command, 'local', approval_callback=deny)
    assert result['approved'] is False
    assert len(calls) == 1
    assert calls[0][1]['allow_session'] is False
    assert calls[0][1]['allow_permanent'] is False
    assert approval_detection._SCAN_INCOMPLETE_DESCRIPTION in calls[0][0][1]


@pytest.mark.parametrize('guard_name', ['check_all_command_guards', 'check_dangerous_command'])
def test_allowlist_scan_uncertainty_cannot_use_cron_approve_mode(tmp_path, monkeypatch, guard_name):
    command = 'env HERMES_YOLO_MODE=0 ' * 50 + 'hermes config set approvals.mode off'
    (tmp_path / 'config.yaml').write_text(
        "approvals:\n  mode: manual\n  cron_mode: approve\ncommand_allowlist:\n  - 'env *'\n")
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_CRON_SESSION', '1')
    for name in ('HERMES_INTERACTIVE', 'HERMES_GATEWAY_SESSION', 'HERMES_EXEC_ASK'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(approval, '_YOLO_MODE_FROZEN', False)
    monkeypatch.setattr(approval_parser_budget, 'MAX_SCAN_UNITS', 100_000)
    monkeypatch.setattr(approval_parser_budget, 'SCAN_UNITS_PER_CHAR', 0)
    assert approval_detection.detect_hardline_command(command) == (False, None)
    assert approval_parser_budget.recorded_scan(command) is True
    assert approval._command_matches_permanent_allowlist(command)
    result = getattr(approval, guard_name)(command, 'local')
    assert result['approved'] is False
    assert 'no human is present' in result['message']


def test_remembered_applescript_finding_cannot_cover_a_later_policy_write(tmp_path, monkeypatch):
    (tmp_path / 'config.yaml').write_text('approvals:\n  mode: manual\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_INTERACTIVE', '1')
    session = 'policy-after-applescript'
    monkeypatch.setenv('HERMES_SESSION_KEY', session)
    command = "osascript -e 'close documents saving no'; hermes config set approvals.mode off"
    key = approval_detection.detect_dangerous_command("osascript -e 'close documents saving no'")[1]
    approval.approve_session(session, key)
    calls = []

    def deny(*args, **kwargs):
        calls.append((args, kwargs))
        return 'deny'

    try:
        result = approval.check_all_command_guards(command, 'local', approval_callback=deny)
        assert result['approved'] is False
        assert len(calls) == 1
        assert calls[0][0][1] == approval._SECURITY_CONFIG_APPROVAL_KEY
        assert calls[0][1]['allow_session'] is False
        assert calls[0][1]['allow_permanent'] is False
    finally:
        approval.clear_session(session)
