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


def test_network_waits_step_out_of_the_lane_while_ordinary_requests_stay_ordered(tmp_path):
    script = textwrap.dedent('''
        import sys, threading, time, types
        from mlx2 import desktop
        from mlx2.desktop_lane import LANE
        active = []
        class Controller:
            def __init__(self, *args):
                pass
            def dispatch(self, request):
                command = request["command"]
                active.append(command)
                try:
                    if len(active) > 1:
                        raise AssertionError("two requests ran inside the lane at once: %r" % active)
                    if command == "network":
                        active.remove(command)
                        try:
                            with LANE.released():  # e.g. a Wikimedia request
                                time.sleep(1.0)
                        finally:
                            active.append(command)
                    elif command == "busy":
                        time.sleep(0.3)  # ordinary work keeps the lane
                    return {"command": command}
                finally:
                    active.remove(command)
            def close(self):
                pass
        sys.modules["mlx2.desktop_control"] = types.SimpleNamespace(DesktopController=Controller)
        desktop.main(["--home", sys.argv[1], "--trash-helper", "/never-executed"])
    ''')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    commands = ['network', 'busy', 'first', 'second', 'third']
    requests = ''.join(json.dumps({'id': c, 'command': c}) + '\n' for c in commands)
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], input=requests,
                            text=True, capture_output=True, env=environment, timeout=15, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert all(reply['ok'] for reply in replies), replies
    # The network wait did not block later work, ordinary requests kept their
    # arrival order, and EOF still delivered the in-flight network reply.
    assert [reply['id'] for reply in replies] == ['busy', 'first', 'second', 'third', 'network']


def test_slow_wikimedia_search_does_not_block_the_real_controller(tmp_path):
    script = textwrap.dedent('''
        import io, json, sys, time, urllib.request
        from pathlib import Path
        from mlx2 import desktop, desktop_badges
        class Collector:
            def __init__(self, **_):
                pass
            def sample(self):
                return ()
            def close(self):
                pass
        class Badges:
            def __init__(self, _):
                pass
            def summarize(self, *_):
                return {"label": "Untested", "qualified": False}
        desktop.HostCollector = desktop.InferenceCollector = Collector
        desktop_badges.BadgeResolver = Badges
        home = Path(sys.argv[1]) / "app"; root = Path(sys.argv[1]) / "models"
        home.mkdir(); root.mkdir()
        desktop.atomic_json(home / "desktop-settings.json", {"root": str(root), "port": 18285, "context": 0, "lanes": 1, "route": "auto", "sticky_port": True})
        class Slow(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
        def urlopen(request, timeout):  # a slow Wikimedia response; no network is used
            time.sleep(1.5)
            return Slow(json.dumps({"query": {"pages": {}}}).encode())
        urllib.request.urlopen = urlopen
        desktop.main(["--home", str(home), "--trash-helper", "/never-executed"])
    ''')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=environment)
    import time
    def send(identity, command, **arguments):
        process.stdin.write(json.dumps({'id': identity, 'command': command, **arguments}) + '\n')
        process.stdin.flush()
    send('search', 'world_image_search', query='lighthouse', limit=3)
    time.sleep(0.2)
    started = time.monotonic()
    send('state', 'world_state')
    send('edit', 'world_object_upsert', object={'id': 'chat', 'label': 'Moved'})
    arrivals = []
    for _ in range(3):
        reply = json.loads(process.stdout.readline())
        arrivals.append((reply['id'], reply['ok'], time.monotonic() - started))
    process.stdin.close()
    assert process.wait(15) == 0
    assert [a[0] for a in arrivals] == ['state', 'edit', 'search'], arrivals
    assert all(ok for _, ok, _ in arrivals), arrivals
    assert arrivals[1][2] < 1.0, arrivals  # scene edits answered while the search waited


def test_termination_waits_for_the_running_command_and_unserializable_results_still_reply(tmp_path):
    script = textwrap.dedent('''
        import os, signal, sys, threading, time, types
        from mlx2 import desktop
        events = []
        class Controller:
            def __init__(self, *args):
                pass
            def dispatch(self, request):
                if request["command"] == "unserializable":
                    return {"value": object()}
                events.append("dispatch start")
                threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGTERM)).start()
                time.sleep(1.0)
                events.append("dispatch end")
                return {"value": "done"}
            def close(self):
                events.append("close")
                print("EVENTS " + ",".join(events), file=sys.stderr, flush=True)
        sys.modules["mlx2.desktop_control"] = types.SimpleNamespace(DesktopController=Controller)
        desktop.main(["--home", sys.argv[1], "--trash-helper", "/never-executed"])
    ''')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    process.stdin.write(json.dumps({'id': 'bad', 'command': 'unserializable'}) + '\n')
    process.stdin.write(json.dumps({'id': 'long', 'command': 'long'}) + '\n')
    process.stdin.flush()
    first = json.loads(process.stdout.readline())
    assert first['id'] == 'bad' and not first['ok']
    _, stderr = process.communicate(timeout=15)
    assert 'EVENTS dispatch start,dispatch end,close' in stderr, stderr
