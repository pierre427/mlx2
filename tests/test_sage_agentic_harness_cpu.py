"""Exercise the full feature probe with a synthetic HTTP model, never a GPU."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlx2.agentic_chat import VERIFICATION_TRIGGER
from mlx2.server import validate_request
from scripts import check_sage_agentic as harness


@pytest.mark.parametrize("broken_report", [False, True, "wrong_answer"])
def test_feature_probe_checks_live_contracts_and_retains_failure(tmp_path, monkeypatch, broken_report):
    revision = "synthetic-source"
    monkeypatch.setattr(harness, "runtime_identity", lambda: {"source_sha256": "synthetic-sha"})
    original = harness.subprocess.check_output
    monkeypatch.setattr(harness.subprocess, "check_output", lambda command, **kwargs:
        revision if command[:3] == ['git', 'rev-parse', 'HEAD'] else original(command, **kwargs))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, code, value):
            data = json.dumps(value).encode(); self.send_response(code)
            self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self.reply(200, {'model': 'synthetic', 'artifact': 'fixture', 'max_context': 262144,
                'chat_template_options': {'verification': {'implemented': True}},
                'build_runtime': {'source_sha256': 'synthetic-sha'}})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            try: request = validate_request(body, structured_thinking=True, allow_buffered_tool_stream=True)
            except ValueError as error:
                self.reply(400, {'error': str(error)}); return
            if self.path == '/tokenize':
                ids = [1, 2, 3] if request.get('verification') else [1, 2]
                self.reply(200, {'count': len(ids), 'tokens': ids}); return
            if self.path == '/apply-template':
                self.reply(200, {'prompt': VERIFICATION_TRIGGER}); return
            if isinstance(body.get('tool_choice'), dict):
                tasks = [{'task': 'Return ALPHA-73 and 41', 'context': 'ALPHA-73: 17 + 24', 'expected_output': 'identifier and integer'},
                         {'task': 'Return BETA-29 and 58', 'context': 'BETA-29: 90 - 32', 'expected_output': 'identifier and integer'}]
                message = {'role': 'assistant', 'content': '', 'reasoning_content': 'Delegate independent tasks.', 'tool_calls': [
                    {'id': 'synthetic-spawn', 'type': 'function', 'function': {'name': 'spawn_subagent', 'arguments': json.dumps({'tasks': tasks})}}]}
            elif body['messages'][-1]['role'] == 'tool':
                message = {'role': 'assistant', 'content': 'ALPHA-73 = 41; BETA-29 = 58'}
            else:
                task = json.loads(body['messages'][-1]['content'])
                assert len(body['messages']) == 2  # no parent history reaches either child
                assert body['verification'] is True and body['thinking_budget'] == 32768
                report = {'answer': task['task'], 'status': 'completed', 'evidence': [task['context']], 'unresolved_issues': []}
                message = {'role': 'assistant', 'content': 'bad report' if broken_report is True else json.dumps({**report, 'answer': task['task'].replace('41', '141').replace('58', '158')} if broken_report == 'wrong_answer' else report)}
            self.reply(200, {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 32}})
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        kwargs = {'source_revision': revision, 'expected_checkpoint': 'synthetic@sha256:fixture', 'timeout': 5}
        if broken_report:
            with pytest.raises(AssertionError, match='ordered_child_answers' if broken_report == 'wrong_answer' else 'children_finished'):
                harness.probe(f'http://127.0.0.1:{server.server_port}', tmp_path / 'evidence', **kwargs)
            assert not json.loads((tmp_path / 'evidence/feature-smoke.json').read_text())['passed']
        else:
            report = harness.probe(f'http://127.0.0.1:{server.server_port}', tmp_path / 'evidence', **kwargs)
            assert report['passed'] and all(check['passed'] for check in report['checks'])
            assert (tmp_path / 'evidence/http.json').exists()
    finally:
        server.shutdown(); server.server_close(); thread.join()
