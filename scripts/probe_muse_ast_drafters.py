#!/usr/bin/env python3
"""One owned target/drafter AST window; bounded correctness, diagnostic A/B.

The ordinary arm is a direct reference, never a drafter route. Each unit runs
one AST loopback server, preserves failed receipts, and cleans up before yield.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from probe_muse_fp_decode import ROOT, atomic, environment, stop


def oracle(name, text):
    text = text.strip()
    if name == 'code':
        if text.startswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        try:
            module = ast.parse(text)
            function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
            body = function.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]  # docstring
            # Only a lone return: earlier statements could rebind a or b.
            return (function.name == 'add' and [a.arg for a in function.args.args] == ['a', 'b']
                    and len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None
                    and ast.dump(body[0].value) == ast.dump(ast.parse('a + b', mode='eval').body))
        except (SyntaxError, StopIteration):
            return False
    if name == 'sequence':
        return text.split() == [str(n) for n in range(1, 31)]
    return text == 'MUSE-7429'


def counters(value, result=None):
    result = {} if result is None else result
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith('external_') and isinstance(child, (int, float)):
                result[key] = max(result.get(key, 0), child)
            else:
                counters(child, result)
    elif isinstance(value, list):
        for child in value:
            counters(child, result)
    return result


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership, http_json

    out = Path(spec['out'])
    report = {**{key: spec[key] for key in ('checkpoint', 'drafter_checkpoint', 'source_revision')},
              'scope': 'bounded AST correctness and diagnostic A/B; no default change',
              'ownership': check_ownership(spec), 'before': environment(),
              'passed': False, 'performance_qualified': False}
    server = None
    try:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == spec['source_revision']
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'tests'], cwd=ROOT, text=True).strip()
        import mlx.core as mx
        from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter, inspect_artifact
        from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

        assert inspect_artifact(spec['model'])['fingerprint'] == spec['fingerprint']
        mx.set_cache_limit(1 << 30)
        adapter = MuseGlimmerAdapter(spec['model'])
        model = adapter.model
        eos = set(adapter.tokenizer.eos_token_ids)
        options = {'reasoning_effort': 'none', 'enable_thinking': False}
        prompts = {
            'identifier': 'Return only MUSE-7429.',
            'sequence': 'Print integers 1 through 30 separated by single spaces. No other text.',
            'code': 'Write a Python function add(a, b) that returns a + b. Output code only without markdown.',
            'long_identifier': 'Background information. ' * 1400 + '\nReturn only MUSE-7429.',
        }
        limits = {'identifier': 24, 'sequence': 96, 'code': 96, 'long_identifier': 24}
        references = {}

        def ordinary(name, capture=None, sampled=False):
            import numpy as np
            from mlx2.runtime.sample_utils import LaneRNG, make_transformed_logprobs
            from mlx2.runtime.speculative_sampling import RequestRNG

            rng = RequestRNG(np.asarray(LaneRNG(173).key).tolist())
            transform = make_transformed_logprobs(.7, top_p=.95, top_k=64) if sampled else None
            ids = adapter.prompt_tokens({'messages': [{'role': 'user', 'content': prompts[name]}], **options})
            cache = model.make_cache()
            start = time.perf_counter()
            # Match serving's checkpoint law and final one-row prompt step.
            for offset in range(0, len(ids) - 1, 2048):
                part = ids[offset:min(offset + 2048, len(ids) - 1)]
                with prefill_checkpoint_rows([len(part)], width=len(part)):
                    logits = model(mx.array([part]), cache=cache)
                mx.eval(logits, [c.state for c in cache])
            logits = model(mx.array([[ids[-1]]]), cache=cache)
            tokens, window = [], None
            for position in range(limits[name]):
                token = (rng.sample(np.asarray(mx.exp(transform(logits[0, -1:])))[0]) if sampled
                         else int(mx.argmax(logits[0, -1]).item()))
                if capture == position:
                    logp = mx.log_softmax(logits[0, -1].astype(mx.float32))
                    top = mx.argsort(logp)[-16:][::-1].tolist()
                    window = {str(t): float(logp[t].item()) for t in top}
                if token in eos:
                    break
                tokens.append(token)
                logits = model(mx.array([[token]]), cache=cache)
            result = {'tokens': tokens, 'text': adapter.tokenizer.decode(tokens),
                      'prompt_tokens': len(ids), 'elapsed_seconds': time.perf_counter() - start}
            if window is not None:
                result['window'] = window
            return result

        for name in prompts:
            references[name] = ordinary(name)
            assert oracle(name, references[name]['text']), ('ordinary oracle', name, references[name]['text'])
            atomic(out / 'reference.json', references)
            print('ordinary reference:', name, references[name]['prompt_tokens'], flush=True)
        references['sampled_identifier'] = ordinary('identifier', sampled=True)
        references['sampled_identifier']['functional_oracle'] = oracle('identifier', references['sampled_identifier']['text'])
        atomic(out / 'reference.json', references)
        # A diagnostic mismatch can be inspected later by a fresh direct run.
        del model, adapter
        gc.collect()
        mx.clear_cache()
        module_spec = importlib.util.spec_from_file_location('ast_stream', ROOT / 'qualification/runs/ast-smokes-20261018/ast_smoke.py')
        helper = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(helper)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = [sys.executable, '-m', 'mlx2.server', '--model', spec['model'],
                   '--ast', '--ast-drafter', spec['drafter'], '--host', '127.0.0.1', '--port', str(port),
                   '--fabric', 'off', '--max-lanes', '2', '--max-inflight', '2', '--max-context', '20000',
                   '--prefill-step', '2048', '--cache-bytes', str(2 << 30),
                   '--cache-dir', str(out / 'apcv2'), '--qualification-mode', '--no-auto-route']
        report['server_command'] = command
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('AST server exited; see server.log')
                try:
                    status = http_json(base, '/v1/status')
                    if status.get('healthy'):
                        break
                except Exception:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError('AST startup')
            report['status_before'] = status
            # Bind the drafter the server loaded to the one this receipt names.
            loaded = (status.get('ast_route') or {}).get('drafter_fingerprint') or ''
            assert loaded and spec['drafter_checkpoint'].rsplit('@', 1)[-1].startswith(loaded), (
                'loaded drafter differs from drafter_checkpoint', loaded)
            model_id = http_json(base, '/v1/models')['data'][0]['id']
            def stream(name, **extra):
                result = helper.stream(port, model_id, prompts[name], limits[name],
                                       request_options={**options, **extra})
                assert result.get('done') and not result.get('error'), result
                assert oracle(name, result['text']), (name, result['text'])
                assert result['token_ids'] is not None, 'missing qualification token trace'
                speculation = result['speculation']
                assert speculation['ordinary_fallback'] is False
                assert speculation['batch_size_route']['current'] == 'tree15_provider'
                assert speculation['target_tree']['provider'] == 'attention_tree_v1'
                return result
            report['b1'] = {}
            for name in prompts:
                cold, warm = stream(name), stream(name)
                for arm in (cold, warm):
                    arm['tokens_without_eos'] = [t for t in arm['token_ids'] if t not in eos]
                report['b1'][name] = {'cold': cold, 'warm': warm,
                    'ordinary_exact': cold['tokens_without_eos'] == references[name]['tokens'],
                    'cold_warm_exact': cold['token_ids'] == warm['token_ids']}
                atomic(out / 'progress.json', report)
                assert cold['token_ids'] == warm['token_ids'], ('width-one cold/warm drift', name)
                assert warm['cached_tokens'] > 0, ('missing APCv2 hit', name)
                assert report['b1'][name]['ordinary_exact'], ('first divergence needs classification', name)
                print('AST B1 cold/warm:', name, flush=True)
            report['b2'] = []
            for replay in range(2):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(stream, ['sequence', 'code']))
                report['b2'].append(results)
                atomic(out / 'progress.json', report)
            for i, name in enumerate(['sequence', 'code']):
                assert report['b2'][0][i]['token_ids'] == report['b2'][1][i]['token_ids'], ('B2 replay drift', name)
                assert [t for t in report['b2'][0][i]['token_ids'] if t not in eos] == references[name]['tokens'], ('B2 ordinary drift', name)
            report['sampled'] = helper.stream(port, model_id, prompts['identifier'], 24,
                temperature=.7, seed=173, request_options={**options, 'top_p': .95, 'top_k': 64})
            report['sampled']['functional_oracle'] = oracle('identifier', report['sampled']['text'])
            report['sampled']['ordinary_control_exact'] = ([t for t in report['sampled']['token_ids'] if t not in eos]
                                                        == references['sampled_identifier']['tokens'])
            assert not report['sampled']['error'] and report['sampled']['done']
            assert report['sampled']['ordinary_control_exact'], 'sampled target law/control drift'
            report['status_after'] = http_json(base, '/v1/status')
            report['batching_after'] = http_json(base, '/v1/status/batching')
            report['counters'] = counters(report)
            assert report['counters'].get('external_auto_tree_rounds', 0) > 0, 'no actual AST rounds'
            assert report['counters'].get('external_auto_chain_rounds', 0) == 0, 'drafter escaped AST'
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
                   '--label', spec['job_id'], '--desktop-cpg-lease', str(lease.path),
                   '--receipt', str(out / 'paired-locks.json'), '--', sys.executable, '-u', str(Path(__file__)), 'unit', str(args.spec.resolve())]
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
