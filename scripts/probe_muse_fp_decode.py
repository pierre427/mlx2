#!/usr/bin/env python3
"""One owned Muse artifact window: reference, FP GQA16 probes, and APCv2.

Diagnostic timings only; no thermal admission, default or performance claim.
Refuse nonempty evidence directories. Always stop the server before yielding.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(path)


def environment():
    return {name: subprocess.run(command, capture_output=True, text=True).stdout.strip()
            for name, command in {
                'swap': ['sysctl', 'vm.swapusage'], 'thermals': ['pmset', '-g', 'therm'],
                'processes': ['ps', '-axo', 'pid,ppid,command'],
                'services': ['launchctl', 'list']}.items()}


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 20
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before a descendant that ignores SIGTERM; the
    # group, not the leader, owns the GPU work.
    while _group_alive(child.pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if _group_alive(child.pid):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        child.wait()


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership, http_json
    report = {'checkpoint': spec['checkpoint'], 'source_revision': spec['source_revision'],
              'qualified': False, 'performance_qualified': False,
              'scope': 'bounded functional and diagnostic A/B',
              'ownership': check_ownership(spec), 'before': environment(), 'passed': False}
    out = Path(spec['out'])
    server = None
    try:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != spec['source_revision']:
            raise RuntimeError('source revision changed after inventory')
        if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'tests'], cwd=ROOT, text=True).strip():
            raise RuntimeError('validation requires committed source/tests/probe')
        import mlx.core as mx
        from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter, inspect_artifact
        from mlx2.runtime.models import qsdpa_decode_metal as qdm
        from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

        assert inspect_artifact(spec['model'])['fingerprint'] == spec['fingerprint']
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(1 << 30)
        report['device'] = mx.device_info()
        report['mlx_version'] = mx.__version__
        report['kernel_sha256'] = hashlib.sha256(Path(qdm.__file__).read_bytes()).hexdigest()
        os.environ['MLX2_FP_DECODE_KERNEL'] = '0'
        adapter = MuseGlimmerAdapter(spec['model'])
        model = adapter.model
        request = {'messages': [{'role': 'user', 'content': 'Return only MUSE-7429.'}],
                   'reasoning_effort': 'none', 'enable_thinking': False}
        prompt = adapter.prompt_tokens(request)

        def generate(ids, cache, limit=24):
            rows, logits = [], model(mx.array([ids]), cache=cache)
            for _ in range(limit):
                token = int(mx.argmax(logits[0, -1]).item())
                if token in adapter.tokenizer.eos_token_ids:
                    break
                rows.append(token)
                logits = model(mx.array([[token]]), cache=cache)
            return rows

        reference = generate(prompt, model.make_cache())
        report['ordinary_reference'] = {'tokens': reference, 'text': adapter.tokenizer.decode(reference)}
        assert report['ordinary_reference']['text'].strip() == 'MUSE-7429'
        atomic(out / 'progress.json', report)
        # Metal diagnostics include malformed-shape refusals, real padded KV
        # views, non-unit dimension strides, both dtypes and old GQA ratios.
        env = {**os.environ, 'MLX2_RUN_GPU_TESTS': '1', 'PYTHONPATH': str(ROOT / 'src')}
        with (out / 'metal-tests.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pytest',
                'tests/test_qsdpa_decode_metal.py', '-k', 'fp_', '-q', '-o', 'addopts=',
                '--junitxml=' + str(out / 'metal-tests.xml')], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=300)
        report['metal_tests_returncode'] = result.returncode
        assert result.returncode == 0, 'Metal tests failed; see metal-tests.log'
        mx.set_default_device(mx.gpu)

        micro = []
        for n in (65536, 131072):
            mx.random.seed(123)
            q = mx.random.normal((1, 32, 1, 128)).astype(mx.bfloat16)
            k = mx.random.normal((1, 2, n, 128)).astype(mx.bfloat16)
            v = mx.random.normal((1, 2, n, 128)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            stock = lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -.5)
            for ns, hs, blocks in ((4, 4, 128), (4, 4, 256), (4, 2, 128), (4, 2, 256), (2, 1, 128), (4, 1, 64), (4, 1, 128), (8, 4, 128)):
                candidate = lambda: qdm.gqa_decode_attention_fp(q, k, v, scale=128 ** -.5,
                    simdgroups=ns, head_split=hs, blocks=blocks)
                expected, actual = stock(), candidate()
                error = float(mx.max(mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))).item())
                assert error < .005
                timings = {'stock': [], 'candidate': []}
                for rep in range(6):
                    for name, fn in ((('stock', stock), ('candidate', candidate)) if rep % 2 == 0
                                     else (('candidate', candidate), ('stock', stock))):
                        start = time.perf_counter_ns()
                        for _ in range(8):
                            mx.eval(fn())
                        timings[name].append((time.perf_counter_ns() - start) / 8)
                micro.append({'context': n, 'schedule': [ns, hs, blocks], 'max_abs_error': error,
                    'diagnostic_ns': {name: statistics.median(values) for name, values in timings.items()}})
        report['micro'] = micro
        atomic(out / 'progress.json', report)
        print('Metal tests and schedule sweep complete', flush=True)

        # A real long-context cache is produced only by ordinary prefill.
        # Two private copies then decode the same prompt under each law.
        repetitions = 22000
        for _ in range(5):
            request['messages'][0]['content'] = ('Background information. ' * repetitions
                + '\nReturn only MUSE-7429.')
            ids = adapter.prompt_tokens(request)
            if 65536 <= len(ids) < 69000:
                break
            repetitions = int(repetitions * 66560 / len(ids)) + 1
        else:
            raise RuntimeError('could not construct an engaging bounded context')
        base_cache = model.make_cache()
        for start in range(0, len(ids), 2048):
            part = ids[start:start + 2048]
            with prefill_checkpoint_rows([len(part)], width=len(part)):
                logits = model(mx.array([part]), cache=base_cache)
            mx.eval(logits, [c.state for c in base_cache])
        report['context_tokens'] = len(ids)
        print('Ordinary long-context prefill complete:', len(ids), flush=True)
        arms = {}
        first_divergence = None
        for enabled in (False, True):
            os.environ['MLX2_FP_DECODE_KERNEL'] = '1' if enabled else '0'
            cache = copy.deepcopy(base_cache)
            current = logits
            tokens, first_rows, elapsed = [], [], []
            before_calls = dict(qdm.STATS)
            for position in range(24):
                token = int(mx.argmax(current[0, -1]).item())
                # Only hold the ordinary rows in memory until comparison;
                # serialize one bounded window at the first selected mismatch.
                if not enabled:
                    first_rows.append(current[0, -1])
                elif first_divergence is None and position < len(arms['stock']['tokens']) and token != arms['stock']['tokens'][position]:
                    old = arms['stock'].pop('_logits')[position]
                    other = arms['stock']['tokens'][position]
                    logp = [mx.log_softmax(row.astype(mx.float32)) for row in (old, current[0, -1])]
                    first_divergence = {'position': position, 'stock_token': other, 'candidate_token': token,
                        'windows': [{'top_ids': mx.argsort(row)[-8:][::-1].tolist(),
                                     'selected_logp': float(row[chosen].item()),
                                     'alternate_logp': float(row[alternate].item())}
                                    for row, chosen, alternate in ((logp[0], other, token), (logp[1], token, other))]}
                if token in adapter.tokenizer.eos_token_ids:
                    break
                tokens.append(token)
                start = time.perf_counter_ns()
                current = model(mx.array([[token]]), cache=cache)
                mx.eval(current)
                elapsed.append(time.perf_counter_ns() - start)
            arms['candidate' if enabled else 'stock'] = {'tokens': tokens,
                'text': adapter.tokenizer.decode(tokens), 'decode_ns': elapsed,
                'calls': {name: qdm.STATS[name] - before_calls[name] for name in before_calls},
                **({'_logits': first_rows} if not enabled else {})}
        arms['stock'].pop('_logits', None)
        report['model_ab'] = arms
        report['first_divergence'] = first_divergence
        report['model_ab_exact_tokens'] = arms['stock']['tokens'] == arms['candidate']['tokens']
        assert all(arm['text'].strip() == 'MUSE-7429' for arm in arms.values())
        assert report['model_ab_exact_tokens'], 'classify first divergence before qualification'
        assert arms['candidate']['calls']['fp_decode_gqa16_calls'] > 0, 'candidate did not engage'
        model = adapter = base_cache = cache = logits = current = q = k = v = None  # release; closures reference these names
        gc.collect()
        mx.clear_cache()
        atomic(out / 'progress.json', report)

        # One loopback candidate server: cold/warm APCv2 under the same law.
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [sys.executable, '-m', 'mlx2.server', '--model', spec['model'],
            '--host', '127.0.0.1', '--port', str(port), '--fabric', 'off',
            '--max-lanes', '1', '--max-inflight', '1', '--max-context', '70000',
            '--prefill-step', '2048', '--cache-bytes', str(2 << 30),
            '--cache-dir', str(out / 'apcv2'), '--qualification-mode']
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('server exited; see server.log')
                try:
                    if http_json(base, '/v1/status').get('healthy'):
                        break
                except Exception:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError('server startup')
            model_id = http_json(base, '/v1/models')['data'][0]['id']
            body = {**request, 'model': model_id, 'max_tokens': 24, 'temperature': 0}
            report['cold'] = http_json(base, '/v1/chat/completions', body)
            report['warm'] = http_json(base, '/v1/chat/completions', body)
            report['status_after'] = http_json(base, '/v1/status')
        cold, warm = (report[name]['choices'][0]['message']['content'] for name in ('cold', 'warm'))
        assert cold.strip() == warm.strip() == 'MUSE-7429'
        assert report['status_after']['fp_decode_attention']['counts']['fp_decode_gqa16_calls'] > 0
        report['warm_cached_tokens'] = report['warm'].get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)
        assert report['warm_cached_tokens'] > 0
        report['passed'] = True
    except BaseException as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        stop(server)
        report['after'] = environment()
        atomic(out / 'result.json', report)
    print(json.dumps({'passed': report['passed'], 'error': report.get('error')}), flush=True)
    return 0 if report['passed'] else 1


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('owned', 'unit'))
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode == 'unit':
        return unit(spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    out = Path(spec['out'])
    if out.exists() and any(out.iterdir()):
        raise ValueError('refusing nonempty evidence directory')
    out.mkdir(parents=True, exist_ok=True)
    with DesktopCPGLease(out, spec['job_id'], checkpoint=spec['checkpoint'], source_revision=spec['source_revision']) as lease:
        command = [sys.executable, str(ROOT / 'scripts/run_with_gpu_locks.py'), '--session', lease.owner['cpg_session'],
            '--label', spec['job_id'], '--desktop-cpg-lease', str(lease.path), '--receipt', str(out / 'paired-locks.json'),
            '--', sys.executable, '-u', str(Path(__file__)), 'unit', str(args.spec.resolve())]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        try:
            deadline = time.monotonic() + 1200
            while child.poll() is None:
                lease.check()
                if time.monotonic() > deadline:
                    raise TimeoutError('bounded ownership window expired')
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            stop(child)


if __name__ == '__main__':
    raise SystemExit(main())
