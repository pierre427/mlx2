#!/usr/bin/env python3
"""One pinned image target/drafter ownership window; ordinary then AST HTTP.

Correctness and diagnostic timings only. Never overwrite prior evidence.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gc
from io import BytesIO
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
from probe_muse_ast_drafters import counters


def fixtures(out):
    from PIL import Image, ImageDraw, ImageFont
    images = {'red': Image.new('RGB', (168, 140), 'red'),
              'blue': Image.new('RGB', (168, 140), 'blue')}
    card = Image.new('RGB', (560, 168), 'white')
    draw = ImageDraw.Draw(card)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 64)
    draw.text((35, 45), 'MUSE-7429', font=font, fill='black')
    images['ocr'] = card
    urls = {}
    for name, image in images.items():
        path = out / (name + '.png'); image.save(path)
        urls[name] = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()
    cases = {
        'red': (['red'], 'What color fills this image? Reply only RED or BLUE.', 'RED'),
        'blue': (['blue'], 'What color fills this image? Reply only RED or BLUE.', 'BLUE'),
        'ordered': (['red', 'blue'], 'Name the colors of the images in order. Reply only RED BLUE or BLUE RED.', 'RED BLUE'),
        'reversed': (['blue', 'red'], 'Name the colors of the images in order. Reply only RED BLUE or BLUE RED.', 'BLUE RED'),
        'ocr': (['ocr'], 'Read the code shown in the image. Reply with only that exact code.', 'MUSE-7429'),
        'red_repeat': (['red'], 'First write the word RED or BLUE for the color filling the image. Then print integers 1 through 30 separated by single spaces. No other text.', 'RED ' + ' '.join(map(str, range(1, 31)))),
        'blue_repeat': (['blue'], 'First write the word RED or BLUE for the color filling the image. Then print integers 1 through 30 separated by single spaces. No other text.', 'BLUE ' + ' '.join(map(str, range(1, 31)))),
    }
    return {name: {'content': [{'type': 'image_url', 'image_url': {'url': urls[key]}} for key in keys]
                  + [{'type': 'text', 'text': prompt}], 'expected': expected,
                  'max_tokens': 96 if name.endswith('repeat') else 32,
                  'identifier': name == 'ocr'}
            for name, (keys, prompt, expected) in cases.items()}


def oracle(case, text):
    if case.get('identifier'):
        return text.strip() == case['expected']
    return ' '.join(text.strip().upper().split()).rstrip('.') == case['expected']


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership, http_json
    report = {key: spec[key] for key in ('checkpoint', 'drafter_checkpoint', 'source_revision')}
    report.update(scope='bounded native-image AST correctness and diagnostic A/B',
                  qualified=False, performance_qualified=False, passed=False,
                  ownership=check_ownership(spec), before=environment())
    out, server = Path(spec['out']), None
    try:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == spec['source_revision']
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts', 'tests'], cwd=ROOT, text=True).strip()
        import mlx.core as mx
        from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter, inspect_artifact

        assert inspect_artifact(spec['model'])['fingerprint'] == spec['fingerprint']
        mx.set_cache_limit(1 << 30)
        adapter = MuseGlimmerAdapter(spec['model'])
        model, eos = adapter.model, set(adapter.tokenizer.eos_token_ids)
        report['device'] = mx.device_info()
        report['ordinary_artifact'] = adapter.identity
        cases = fixtures(out)
        options = {'reasoning_effort': 'none', 'enable_thinking': False}
        references = {}
        for name, case in cases.items():
            request = adapter.prepare_multimodal_request({'messages': [{'role': 'user', 'content': case['content']}], **options})
            prompt = adapter.prompt_tokens(request)
            cache, start = model.make_cache(), time.perf_counter()
            model.prefill_forward(mx.array([prompt[:-1]]), cache=cache, **request['_mlx2_prefill_inputs'])
            logits = model(mx.array([[prompt[-1]]]), cache=cache)
            tokens = []
            for _ in range(case['max_tokens']):
                token = int(mx.argmax(logits[0, -1]).item())
                if token in eos:
                    break
                tokens.append(token)
                logits = model(mx.array([[token]]), cache=cache)
            result = {'text': adapter.tokenizer.decode(tokens), 'tokens': tokens,
                      'prompt_tokens': len(prompt), 'prompt_ids': prompt,
                      'media_fingerprint': request['_mlx2_media_fingerprint'],
                      'elapsed_seconds': time.perf_counter() - start}
            result['functional_oracle'] = oracle(case, result['text'])
            references[name] = result
            atomic(out / 'reference.json', references)
            assert result['functional_oracle'], ('ordinary visual oracle', name, result['text'])
            print('ordinary image:', name, result['text'], flush=True)
        assert references['red']['prompt_ids'] == references['blue']['prompt_ids']
        assert references['red']['media_fingerprint'] != references['blue']['media_fingerprint']
        assert references['ordered']['prompt_ids'] == references['reversed']['prompt_ids']
        assert references['ordered']['media_fingerprint'] != references['reversed']['media_fingerprint']
        report['ordinary_vision_stats'] = dict(model.vision_stats)
        del model, adapter, cache, logits, request
        gc.collect(); mx.clear_cache()

        module_spec = importlib.util.spec_from_file_location('vision_stream', ROOT / 'qualification/runs/ast-smokes-20261018/ast_smoke.py')
        helper = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(helper)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = [sys.executable, '-m', 'mlx2.server', '--model', spec['model'],
            '--ast', '--ast-drafter', spec['drafter'], '--host', '127.0.0.1', '--port', str(port),
            '--fabric', 'off', '--max-lanes', '2', '--max-inflight', '2', '--max-context', '8192',
            '--prefill-step', '2048', '--cache-bytes', str(2 << 30), '--cache-dir', str(out / 'apcv2'),
            '--qualification-mode', '--no-auto-route']
        report['server_command'] = command
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            base, deadline = f'http://127.0.0.1:{port}', time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('vision AST server exited; see server.log')
                try:
                    status = http_json(base, '/v1/status')
                    if status.get('healthy'):
                        break
                except Exception:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError('vision AST startup')
            report['status_before'] = status
            # Bind the drafter the server loaded to the one this receipt names.
            loaded = (status.get('ast_route') or {}).get('drafter_fingerprint') or ''
            assert loaded and spec['drafter_checkpoint'].rsplit('@', 1)[-1].startswith(loaded), (
                'loaded drafter differs from drafter_checkpoint', loaded)
            model_id = http_json(base, '/v1/models')['data'][0]['id']
            def stream(name):
                case = cases[name]
                result = helper.stream(port, model_id, case['content'], case['max_tokens'], request_options=options)
                assert result.get('done') and not result.get('error'), result
                result['functional_oracle'] = oracle(case, result['text'])
                assert result['functional_oracle'], ('AST visual oracle', name, result['text'])
                assert result['token_ids'] is not None
                result['tokens_without_eos'] = [token for token in result['token_ids'] if token not in eos]
                return result
            report['b1'] = {}
            for name in cases:
                cold, warm = stream(name), stream(name)
                report['b1'][name] = {'cold': cold, 'warm': warm,
                    'ordinary_exact': cold['tokens_without_eos'] == references[name]['tokens'],
                    'cold_warm_exact': cold['token_ids'] == warm['token_ids']}
                atomic(out / 'progress.json', report)
                assert report['b1'][name]['ordinary_exact'], ('classify first divergence', name)
                assert report['b1'][name]['cold_warm_exact'], ('B1 replay drift', name)
                assert warm['cached_tokens'] > 0, ('no image APCv2 reuse', name)
                if not name.endswith('repeat'):
                    assert cold['cached_tokens'] == 0, ('changed-image cache isolation', name)
                for arm in (cold, warm):
                    s = arm['speculation']
                    assert s['ordinary_fallback'] is False
                    assert s['batch_size_route']['current'] == 'tree15_provider'
                print('AST image cold/warm:', name, flush=True)
            report['b2'] = []
            for _ in range(2):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    report['b2'].append(list(pool.map(stream, ['red_repeat', 'blue_repeat'])))
            for i, name in enumerate(['red_repeat', 'blue_repeat']):
                assert report['b2'][0][i]['token_ids'] == report['b2'][1][i]['token_ids']
                assert report['b2'][0][i]['tokens_without_eos'] == references[name]['tokens']
            report['status_after'] = http_json(base, '/v1/status')
            report['batching_after'] = http_json(base, '/v1/status/batching')
            report['counters'] = counters(report)
            assert report['counters'].get('external_media_prefill_rounds', 0) >= 5
            assert report['counters'].get('external_auto_tree_rounds', 0) > 0
            assert report['counters'].get('external_auto_chain_rounds', 0) == 0
            assert report['counters'].get('external_ordinary_fast_path_rounds', 0) == 0
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
                    raise TimeoutError('bounded image ownership window expired')
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            stop(child)


if __name__ == '__main__':
    raise SystemExit(main())
