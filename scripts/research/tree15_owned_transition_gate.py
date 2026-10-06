"""Fresh bounded normal-HTTP owned-cache transition gate; never resumes a series.

Parent GPU queue owns both leases. This script only verifies their identity and
runs one capped service child; its supervisor enforces aggregate process RSS and
wall limits. Results are functional evidence, not controlled performance.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / 'qualification/runs/tree15-b1-discriminator-20261003'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_binding(commit):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        raise RuntimeError('gate requires exact clean mlx2 source commit')
    paths = ['src/mlx2/server.py', 'src/mlx2/runtime/tensorfold_owned_router.py',
             'src/mlx2/runtime/tensorfold_owned_worker.py',
             'src/mlx2/adapters/qwen38_tensorfold_owned.py',
             'scripts/research/tree15_owned_transition_gate.py',
             'scripts/run_spomin_20x20.py',
             'qualification/runs/tree15-b1-discriminator-20261003/tensorfold_owned_spomin_cell.py']
    return {'revision': commit, 'files': {p: digest(ROOT / p) for p in paths}}


def lease_identity():
    session, lease = os.environ.get('GPUQ_SESSION'), os.environ.get('GPUQ_LEASE')
    if not session or not lease:
        raise RuntimeError('fresh parent GPUQ session/lease required')
    owners = []
    for name in ('/Users/Shared/mlxuag/gpu.lock', '/tmp/gpu.lock'):
        path = Path(name)
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError('both real GPU lock directories required')
        owner = json.loads((path / 'owner.json').read_text())
        if owner.get('session') != session or owner.get('lease_id') != lease:
            raise RuntimeError('GPU lock ownership differs from parent lease')
        pid = owner.get('pid')
        if type(pid) is not int or pid < 1:
            raise RuntimeError('GPU owner PID invalid')
        os.kill(pid, 0)
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError('dual GPU lock owners differ')
    return owners[0]


def get(base, path):
    with urlopen(base + path, timeout=3) as response:
        return json.load(response)


def post(base, body, route='auto'):
    started = time.monotonic()
    request = Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json',
                               'X-MLX2-TensorFold-Owned': '1',
                               'X-MLX2-TensorFold-Route': route})
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return {'response': payload, 'wall_seconds': time.monotonic() - started}


def exact(left, right):
    a, b = left['response'], right['response']
    keys = ('output_token_ids_sha256', 'identity', 'cache_layout')
    return (all(a['mlx2'].get(k) == b['mlx2'].get(k) and a['mlx2'].get(k)
                for k in keys) and a['choices'] == b['choices']
            and a['usage'] == b['usage'])


def valid_completion(row, cap):
    """A request cap is an upper bound; EOS may complete it earlier."""
    payload = row['response']
    count = payload['usage']['completion_tokens']
    choices = payload['choices']
    if type(count) is not int or not 1 <= count <= cap or len(choices) != 1:
        return False
    reason = choices[0].get('finish_reason')
    return reason == 'stop' or (reason == 'length' and count == cap)


def isolated(row):
    receipt = row['response']['mlx2']
    return (receipt['qualified'] is False and receipt['cache_policy'] == 'disabled'
            and receipt['apcv2_lookup'] is False and receipt['apcv2_store'] is False)


def check_transition(row):
    receipt = row['response']['mlx2']
    transitions = [(r['mode'], r['active_width']) for r in receipt['mode_transitions']]
    expected = [('b1_tree_eligible', 1), ('b2plus_shared_ordinary', 2),
                ('b1_tree_eligible', 1)]
    return transitions == expected and receipt['drafted'] > 0 and isolated(row)


def wait_round(base, mode, after=0, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get(base, '/v1/status')['tensorfold_owned']
        last = status.get('last_round') or {}
        if last.get('sequence', 0) > after and last.get('mode') == mode:
            return last['sequence']
        time.sleep(.01)
    raise TimeoutError('actual native round mode not observed: ' + mode)


def wait_idle(base, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get(base, '/v1/status')['tensorfold_owned']
        if all(status[k] == 0 for k in ('active_requests', 'opening_requests', 'pending_requests')):
            return status
        time.sleep(.02)
    raise TimeoutError('owned admission did not retire to idle')


def raw_peer(base, body):
    # Keep a real normal HTTP peer open through an observed B2 round, then FIN.
    port = int(base.rsplit(':', 1)[1])
    payload = json.dumps(body).encode()
    sock = socket.create_connection(('127.0.0.1', port), timeout=5)
    wire = (f'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n'
            'Content-Type: application/json\r\nX-MLX2-TensorFold-Owned: 1\r\n'
            f'Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n').encode()
    sock.sendall(wire + payload)
    return sock


def child(args):
    sys.path.insert(0, str(HELPERS))
    from tensorfold_owned_spomin_cell import source_cell, make_body, FROZEN_CORPUS_SHA256
    from mlx2.runtime.tensorfold_owned_worker import source_identity, mlx_lm_identity, artifact_identity, profile_identity
    # All identities and transcript reconciliation precede model load.
    identities = {'tensorfold': source_identity(args.source),
                  'mlx_lm': mlx_lm_identity(args.mlx_lm_source),
                  'target': artifact_identity(args.target), 'drafter': artifact_identity(args.drafter)}
    expected_identity = profile_identity(identities['tensorfold'], identities['target'],
                                         identities['drafter'], identities['mlx_lm'])
    domain, _, prepared, system = source_cell(0, args.target, args.prepared_report, args.prepared_sha)
    bodies = [make_body(system, prepared[0], 'full', 192),
              make_body(system, prepared[1], 'full', 64)]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = [sys.executable, '-m', 'mlx2.server', '--model', args.target,
               '--ordinary', '--host', '127.0.0.1', '--port', str(port),
               '--max-lanes', '8', '--max-inflight', '8', '--max-context', '8192',
               '--cache-bytes', str(512 << 20), '--tensorfold-owned-live',
               '--tensorfold-owned-source', args.source, '--tensorfold-owned-mlx-lm-source',
               args.mlx_lm_source, '--tensorfold-owned-drafter', args.drafter]
    output = {'schema': 'mlx2.tree15-owned-transition-gate.v1', 'status': 'running',
              'qualified': False, 'default_off': True, 'performance_controlled': False,
              'source': source_binding(args.source_commit), 'lease': lease_identity(),
              'identities': identities, 'expected_identity': expected_identity, 'corpus_sha256': FROZEN_CORPUS_SHA256,
              'prepared_report_sha256': args.prepared_sha, 'domain': domain,
              'cases': [p['receipt'] for p in prepared[:2]], 'command': command,
              'request_body_sha256': [hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() for b in bodies]}
    service = subprocess.Popen(command, cwd=ROOT, stdout=sys.stderr, stderr=sys.stderr)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    peer = None
    base = f'http://127.0.0.1:{port}'
    try:
        deadline = time.monotonic() + 60
        while True:
            if service.poll() is not None:
                raise RuntimeError('service exited during load')
            try:
                get(base, '/health'); break
            except Exception:
                if time.monotonic() > deadline: raise TimeoutError('service load deadline')
                time.sleep(.1)
        output['serial_controls'] = [post(base, b, 'serial') for b in bodies]
        if any(not isolated(row) or row['response']['mlx2']['drafted'] != 0
               or row['response']['mlx2']['active_width_modes'].get('b1_serial_ordinary', 0) < 1
               or row['response']['mlx2']['identity'] != expected_identity
               or not valid_completion(row, cap)
               for row, cap in zip(output['serial_controls'], (192, 64))):
            raise RuntimeError('serial control identity or completion semantics differ')
        before = get(base, '/v1/status')['tensorfold_owned'].get('last_round') or {}
        anchor = pool.submit(post, base, bodies[0])
        start = wait_round(base, 'b1_tree_eligible', before.get('sequence', 0))
        second = pool.submit(post, base, bodies[1])
        wait_round(base, 'b2plus_shared_ordinary', start)
        rows = [anchor.result(timeout=120), second.result(timeout=120)]
        output['natural_transition'] = rows
        if (not check_transition(rows[0]) or not isolated(rows[1])
                or rows[1]['response']['mlx2']['drafted'] != 0
                or rows[1]['response']['mlx2']['active_width_counts'].get('2', 0) < 1
                or not all(exact(a,b) for a,b in zip(rows, output['serial_controls']))):
            raise RuntimeError('normal HTTP natural transition/native serial parity failed')
        output['idle_after_natural'] = wait_idle(base)
        before = output['idle_after_natural'].get('last_round') or {}
        anchor = pool.submit(post, base, bodies[0])
        start = wait_round(base, 'b1_tree_eligible', before.get('sequence', 0))
        peer = raw_peer(base, bodies[1])
        wait_round(base, 'b2plus_shared_ordinary', start)
        peer.shutdown(socket.SHUT_RDWR); peer.close(); peer = None
        row = anchor.result(timeout=120)
        output['cancel_survivor'] = row
        if not check_transition(row) or not exact(row, output['serial_controls'][0]):
            raise RuntimeError('peer FIN survivor/native serial parity failed')
        output['idle_after_cancel'] = wait_idle(base)
        output['status'] = 'completed'
    except Exception as exc:
        output['status'] = 'failed'; output['error'] = repr(exc)
    finally:
        if peer: peer.close()
        pool.shutdown(wait=False, cancel_futures=True)
        service.terminate()
        try: service.wait(timeout=5)
        except subprocess.TimeoutExpired: service.kill(); service.wait(timeout=3)
        Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    return 0 if output['status'] == 'completed' else 1


def supervise(args, *, entrypoint=None, extra_source_paths=(), max_deadline_seconds=240):
    import psutil
    binding, lease = source_binding(args.source_commit), lease_identity()
    for path in extra_source_paths:
        binding['files'][path] = digest(ROOT / path)
    output_path = Path(args.output).resolve()
    if output_path.exists() or output_path.is_relative_to(ROOT):
        raise ValueError('fresh receipt path outside the source checkout required')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.deadline_seconds <= max_deadline_seconds or not 1 <= args.rss_gib <= 48:
        raise ValueError(f'gate limits must be <={max_deadline_seconds} seconds and <=48 GiB')
    command = [sys.executable, str(Path(entrypoint or __file__).resolve()), *sys.argv[1:], '--child']
    env = dict(os.environ); env['PYTHONPATH'] = str(ROOT / 'src')
    started = time.monotonic(); peak = 0; failure = None
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
            peak = max(peak, rss)
        except psutil.Error: rss = 0
        if time.monotonic() - started > args.deadline_seconds: failure = 'wall_deadline'
        if rss > args.rss_gib * (1 << 30): failure = 'process_tree_rss'
        if failure:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=3)
            break
        time.sleep(.1)
    # The child, HTTP server and native worker inherit this dedicated group.
    # Also clean descendants if service shutdown had to kill its parent.
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    time.sleep(.1)
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    path = Path(args.output)
    output = json.loads(path.read_text()) if path.exists() else {'status': 'failed'}
    output['supervisor'] = {'source': binding, 'lease': lease, 'returncode': process.returncode,
                            'wall_seconds': time.monotonic() - started, 'peak_tree_rss_bytes': peak,
                            'deadline_seconds': args.deadline_seconds, 'rss_gib': args.rss_gib,
                            'failure': failure}
    if failure or process.returncode: output['status'] = 'failed'
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    return 0 if output['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'drafter', 'mlx-lm-source', 'prepared-report',
                 'prepared-sha', 'source-commit', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline-seconds', type=float, default=240)
    parser.add_argument('--rss-gib', type=float, default=48)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child(args) if args.child else supervise(args))

if __name__ == '__main__': main()
