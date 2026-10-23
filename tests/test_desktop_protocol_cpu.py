"""Subprocess protocol isolation with synthetic workers; no model or services."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_background_python_and_native_diagnostics_never_enter_json_wire(tmp_path):
    script = textwrap.dedent('''
        import os, sys, threading, types
        cached_stdout = sys.stdout
        from mlx2 import desktop
        class TriggeredResult(dict):
            def items(self):
                ready, finished = threading.Event(), threading.Event()
                def diagnostic():
                    ready.wait(2)
                    print("background Python diagnostic", flush=True)
                    print("cached stdout diagnostic", file=cached_stdout, flush=True)
                    os.write(1, b"background native diagnostic\\n")
                    finished.set()
                worker = threading.Thread(target=diagnostic)
                worker.start()
                ready.set()
                assert finished.wait(2)
                worker.join(2)
                return super().items()
        class Controller:
            def __init__(self, *args):
                print("startup diagnostic", flush=True)
            def dispatch(self, request):
                return TriggeredResult(value="synthetic success")
            def close(self):
                os.write(1, b"shutdown native diagnostic\\n")
        sys.modules["mlx2.desktop_control"] = types.SimpleNamespace(DesktopController=Controller)
        desktop.main(["--home", sys.argv[1], "--trash-helper", "/never-executed"])
    ''')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    requests = ''.join(json.dumps({'id': identity, 'command': 'snapshot'}) + '\n' for identity in ['first', 'second'])
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], input=requests,
                            text=True, capture_output=True, env=environment, timeout=15, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [reply['id'] for reply in replies] == ['first', 'second']
    assert all(reply['ok'] and reply['result']['value'] == 'synthetic success' for reply in replies)
    assert result.stderr.count('background Python diagnostic') == 2
    assert result.stderr.count('cached stdout diagnostic') == 2
    assert result.stderr.count('background native diagnostic') == 2
    assert 'startup diagnostic' in result.stderr and 'shutdown native diagnostic' in result.stderr
