"""Child YOLO launch consent (#59293 / #59337), with synthetic local provider stimulus."""
import http.server
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[2]


class SyntheticProvider(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.received.append(body)
        tools = [m for m in body.get('messages', []) if m['role'] == 'tool']
        has_terminal = any(t.get('function', {}).get('name') == 'terminal'
                           for t in body.get('tools', []))
        message = {'role': 'assistant', 'content': 'SYNTHETIC_DONE'}
        if tools:
            self.server.results.extend(tools)
        elif has_terminal:
            message.update(content=None, tool_calls=[{
                'id': 'call_probe', 'type': 'function', 'function': {
                    'name': 'terminal', 'arguments': json.dumps({
                        'command': self.server.probe_command, 'timeout': 10}),
                },
            }])
        response = {
            'id': 'chatcmpl-synthetic', 'object': 'chat.completion', 'created': 1,
            'model': 'synthetic-probe', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if 'tool_calls' in message else 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        }
        kind = 'application/json'
        if body.get('stream'):
            response['object'] = 'chat.completion.chunk'
            response['choices'][0]['delta'] = response['choices'][0].pop('message')
            for index, tool in enumerate(message.get('tool_calls', [])):
                tool['index'] = index
            raw = ('data: ' + json.dumps(response) + '\n\ndata: [DONE]\n\n').encode()
            kind = 'text/event-stream'
        else:
            raw = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _config(url, cwd):
    return {
        'gateway': {'multiplex_profiles': False},
        'model': {'provider': 'custom', 'default': 'synthetic-probe',
                  'api_mode': 'chat_completions', 'base_url': url},
        'terminal': {'env': 'local', 'cwd': str(cwd)},
        'approvals': {'mode': 'manual', 'single_query_mode': 'deny',
                      'unattended_mode': 'deny'},
        'memory': {'memory_enabled': False, 'user_profile_enabled': False},
        'auxiliary': {'title_generation': {'enabled': False}},
        'platform_toolsets': {'cli': ['terminal']},
    }


def _case(directory, name, choice=None, stale=None):
    from tests.gateway.fixtures.local_recovery_probe import daemon
    from tools import approval, terminal_tool as terminal
    from tools.approval_context import set_current_session_key, reset_current_session_key

    case = directory / name
    case.mkdir()
    home, runtime, user = (case / n for n in ('h', 'r', 'u'))
    for path in (home, runtime, user):
        path.mkdir()
    probe, marker = case / 'probe', case / 'marker'
    probe.write_text('disposable probe\n')
    probe.chmod(0o600)
    peer = http.server.ThreadingHTTPServer(('127.0.0.1', 0), SyntheticProvider)
    peer.received, peer.results = [], []
    # The detector flags the update family, but --help exits at argparse before
    # any update is possible. Exercise a real harmless invocation, not chmod.
    peer.probe_command = (f'{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / "hermes"))} '
                          f'update --help > {shlex.quote(str(probe))} && '
                          f'printf PROBE_EXECUTED > {shlex.quote(str(marker))} && '
                          'printf PROBE_EXECUTED')
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    (home / 'config.yaml').write_text(json.dumps(_config(url, case)))
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    env = {k: os.environ[k] for k in ('PATH', 'LANG', 'TZ') if k in os.environ}
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home),
               HERMES_RUNTIME_DIR=str(runtime), PYTHONPATH=str(ROOT),
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(case), TMP=str(case), TEMP=str(case),
               OPENAI_BASE_URL=url, OPENAI_API_KEY='synthetic-not-a-credential',
               HERMES_GATEWAY_LOCK_DIR=str(case / 'locks'))
    # Invoke the console entrypoint's actual module from this checkout. No fake
    # launcher, implementation patch, force=True, or child approval callback.
    argv = [sys.executable, '-m', 'hermes_cli.main', 'chat', '-Q']
    if name == 'flag':
        argv.append('--yolo')
    if name == 'environment':
        env['HERMES_YOLO_MODE'] = '1'
    argv += ['-q', 'Run the disposable synthetic probe once.', '--provider', 'custom',
             '--model', 'synthetic-probe', '--base-url', url,
             '--api-key', 'synthetic-not-a-credential', '-t', 'terminal',
             '--max-turns', '3', '--ignore-rules']
    command = shlex.join(['env', '-i', *(f'{k}={v}' for k, v in env.items()), *argv])
    # Fixture owns the real daemon, so no auto-started owner survives teardown.
    # Neither launch form needs daemon-wide yolo: CLI carries launch policy on
    # session.create; keep the daemon identical across all three cases.
    owner_env = {k: v for k, v in env.items() if k != 'HERMES_YOLO_MODE'}
    try:
        with daemon(ROOT, home, owner_env, barrier=False):
            # Gateway startup probes the custom model endpoint before the parent
            # dispatch. Keep those separate from child conversation requests.
            bootstrap_requests = list(peer.received)
            assert all('messages' not in request for request in bootstrap_requests)
            peer.received.clear()
            calls, second = [], None

            def consent(*args, **kwargs):
                calls.append((args, kwargs))
                # No child conversation can precede the parent's first consent.
                if len(calls) == 1:
                    assert peer.received == []
                    assert not marker.exists()
                    return choice
                return 'deny'

            with pytest.MonkeyPatch.context() as patch:
                session = 'child-yolo-' + name
                token = set_current_session_key(session)
                if choice:
                    patch.delenv('HERMES_SINGLE_QUERY_SESSION', raising=False)
                    patch.setenv('HERMES_INTERACTIVE', '1')
                    terminal.set_approval_callback(consent)
                governing = approval._permanent_set()
                previous_permanent = set(governing)
                try:
                    raw = terminal.terminal_tool(command=command, timeout=90, workdir=str(case),
                                                 task_id='yolo-child-' + name)
                    if stale:
                        assert json.loads(raw).get('exit_code') == 0, raw
                        assert marker.read_text() == 'PROBE_EXECUTED'
                        request_count = len(peer.received)
                        marker.unlink()
                        key = 'modify Hermes security policy via config'
                        if stale == 'session':
                            patch.setitem(approval._session_approved, session, {key})
                        elif stale == 'always':
                            governing.add(key)
                        else:
                            config = Path(os.environ['HERMES_HOME']) / 'config.yaml'
                            data = json.loads(config.read_text())
                            data['command_allowlist'] = ['env *']
                            config.write_text(json.dumps(data))
                        second = json.loads(terminal.terminal_tool(
                            command=command, timeout=90, workdir=str(case),
                            task_id='yolo-child-second-' + name))
                        assert second.get('status') == 'blocked', second
                        assert len(peer.received) == request_count
                        assert not marker.exists()
                finally:
                    terminal.set_approval_callback(None)
                    governing.clear()
                    governing.update(previous_permanent)
                    approval.clear_session(session)
                    reset_current_session_key(token)
        receipt = {'case': name, 'command': command, 'approval_calls': calls, 'second': second,
                   'parent_tool_result': json.loads(raw), 'bootstrap_requests': bootstrap_requests,
                   'requests': peer.received, 'received_tool_messages': peer.results,
                   'probe_output': probe.read_text(),
                   'mode': oct(stat.S_IMODE(probe.stat().st_mode)),
                   'marker': marker.read_text() if marker.exists() else None}
        (case / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        return receipt
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope='module')
def child_receipts(tmp_path_factory):
    directory = tmp_path_factory.mktemp('yolo-child')
    # Only environment isolation: all Hermes parsing/approval/execution stays real.
    with pytest.MonkeyPatch.context() as patch:
        parent = directory / 'parent'
        parent.mkdir()
        (parent / 'config.yaml').write_text(json.dumps(_config('', directory)))
        for key in ('HERMES_YOLO_MODE', 'HERMES_INTERACTIVE', 'HERMES_GATEWAY_SESSION',
                    'HERMES_EXEC_ASK'):
            patch.delenv(key, raising=False)
        patch.setenv('HERMES_HOME', str(parent))
        patch.setenv('HERMES_RUNTIME_DIR', str(parent / 'runtime'))
        patch.setenv('HERMES_SINGLE_QUERY_SESSION', '1')
        patch.setenv('TERMINAL_CWD', str(directory))
        receipts = [_case(directory, name) for name in ('normal', 'flag', 'environment')]
    print('Synthetic provider stimulus; not live model inference. Receipts:', directory)
    return receipts


@pytest.mark.platforms('linux')
def test_normal_child_receives_real_tool_denial(child_receipts):
    receipt = child_receipts[0]
    assert receipt['parent_tool_result'].get('exit_code') == 0, receipt
    assert receipt['requests'], 'Normal child never reached synthetic provider'
    messages = receipt['received_tool_messages']
    assert len(messages) == 1, messages
    assert messages[0]['tool_call_id'] == 'call_probe', messages
    result = json.loads(messages[0]['content'])
    assert result.get('status') == 'blocked', result
    assert 'BLOCKED' in result.get('error', ''), result
    assert receipt['mode'] == '0o600' and receipt['marker'] is None, receipt


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('case_index', [1, 2])
def test_parent_requires_consent_before_bypass_child_starts(child_receipts, case_index):
    """Parent denial must precede child conversation requests and tool execution."""
    normal = child_receipts[0]
    assert normal['requests'] and len(normal['received_tool_messages']) == 1, normal
    assert normal['mode'] == '0o600' and normal['marker'] is None, normal
    receipt = child_receipts[case_index]
    assert receipt['parent_tool_result'].get('status') in {'blocked', 'pending_approval'}, receipt
    assert not receipt['requests'], 'Bypass child started without parent consent'
    assert receipt['mode'] == '0o600' and receipt['marker'] is None, receipt


@pytest.fixture
def consenting_parent(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    parent.mkdir()
    (parent / 'config.yaml').write_text(json.dumps(_config('', tmp_path)))
    for key in ('HERMES_YOLO_MODE', 'HERMES_GATEWAY_SESSION', 'HERMES_EXEC_ASK'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HERMES_HOME', str(parent))
    monkeypatch.setenv('HERMES_RUNTIME_DIR', str(parent / 'runtime'))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    return tmp_path


def _assert_executed(receipt):
    assert receipt['parent_tool_result'].get('exit_code') == 0, receipt
    assert receipt['requests'], receipt
    messages = receipt['received_tool_messages']
    assert len(messages) == 1, messages
    assert messages[0]['tool_call_id'] == 'call_probe'
    result = json.loads(messages[0]['content'])
    assert result.get('exit_code') == 0, result
    assert result.get('output', '').strip() == 'PROBE_EXECUTED', result
    assert 'usage:' in receipt['probe_output'] and 'update' in receipt['probe_output']
    assert receipt['mode'] == '0o600'
    assert receipt['approval_calls']
    for _, kwargs in receipt['approval_calls']:
        assert kwargs['allow_session'] is False
        assert kwargs['allow_permanent'] is False


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('name', ['flag', 'environment'])
def test_explicit_once_runs_real_child_tool(consenting_parent, name):
    receipt = _case(consenting_parent, name, choice='once')
    _assert_executed(receipt)
    assert len(receipt['approval_calls']) == 1
    assert receipt['marker'] == 'PROBE_EXECUTED'


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('name', ['flag', 'environment'])
@pytest.mark.parametrize('stale', ['always', 'session', 'prefix'])
def test_previous_grant_cannot_approve_next_bypass_child(consenting_parent, name, stale):
    receipt = _case(consenting_parent, name, choice='always' if stale == 'always' else 'once', stale=stale)
    _assert_executed(receipt)
    assert len(receipt['approval_calls']) == 2
    assert receipt['second'].get('status') == 'blocked'
    assert receipt['marker'] is None
