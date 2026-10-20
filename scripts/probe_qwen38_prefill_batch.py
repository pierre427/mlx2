#!/usr/bin/env python3
"""Compare packed Qwen3.8 prefill with independent ordinary B1 chunks.

This bounded diagnostic requires a live CPG lease and paired GPU locks. It
does not mint a serving capability or change APCv2 publication authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from qualification_gpu_ownership import require_qualification_lease

ROOT = Path(__file__).resolve().parents[1]


def same(mx, left, right):
    if left is None or right is None:
        return left is None and right is None
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    unsigned = {2: mx.uint16, 4: mx.uint32}[left.dtype.size]
    return bool(mx.array_equal(left.view(unsigned), right.view(unsigned)).item())


def state_equal(mx, left, right):
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(state_equal(mx, a, b) for a, b in zip(left, right, strict=True))
        )
    return same(mx, left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--cpg-owner-lock', type=Path, required=True)
    parser.add_argument('--cpg-task', required=True)
    parser.add_argument('--cpg-generation', type=int, required=True)
    args = parser.parse_args()
    if not 2 <= args.batch <= 4 or not 64 <= args.chunk <= args.rows <= 4096:
        parser.error('require B2..4 and 64 <= chunk <= rows <= 4096')
    if os.environ.get('MLX_ENABLE_TF32') != '0':
        parser.error('MLX_ENABLE_TF32=0 is required')
    owner = require_qualification_lease(
        task_id=args.cpg_task, generation=args.cpg_generation,
        cpg_owner_lock=args.cpg_owner_lock,
    )
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True
    ).strip():
        parser.error('tracked source must be clean')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    report = {
        'schema': 'mlx2.qwen38-prefill-batch-probe.v1',
        'source_revision': revision, 'checkpoint': args.checkpoint,
        'model': str(args.model.resolve()), 'owner': owner,
        'policy': json.loads(args.policy.read_text()),
        'policy_sha256': hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        'batch': args.batch, 'rows': args.rows, 'chunk': args.chunk,
        'qualified': False, 'performance_claim': False, 'passed': False,
    }
    try:
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        from mlx2.adapters.registry import resolve_adapter

        adapter = resolve_adapter(args.model)(
            str(args.model), execution_policy=report['policy']
        )
        from mlx2.runtime.generate import _merge_caches
        from mlx2.runtime.models.qwen38_fused_gdn import stats

        model = adapter.model
        captures = tuple(adapter.draft_model.config.target_layer_ids)
        report['capture_layers'] = captures
        report['mtp_present'] = model.mtp is not None
        report['artifact'] = adapter.identity
        report['layout'] = adapter.layout
        fingerprint = adapter.identity.get('target_fingerprint', adapter.identity['fingerprint'])
        if args.checkpoint.split('@sha256:')[-1] != fingerprint:
            raise ValueError('target checkpoint fingerprint differs')
        text = (ROOT / 'docs/SERVING.md').read_text()
        corpus = list(adapter.tokenizer.encode(text))
        needed = args.rows + args.batch * 17
        corpus = (corpus * (-(-needed // len(corpus))))[:needed]
        tokens = [corpus[i * 17:i * 17 + args.rows] for i in range(args.batch)]
        stock = type(model).forward_with_taps

        def prefill(packed):
            caches = [model.make_cache() for _ in tokens]
            features = [[] for _ in tokens]
            logits = [[] for _ in tokens]
            before = stats(model)
            mx.synchronize()
            start = time.perf_counter()
            for lo in range(0, args.rows, args.chunk):
                blocks = [row[lo:lo + args.chunk] for row in tokens]
                width = len(blocks[0])
                if packed:
                    cache = _merge_caches(caches)
                    for plane in cache:
                        plane.prepare(lengths=[width] * args.batch, right_padding=[0] * args.batch)
                    with model.prefill_row_context([width] * args.batch, width=width):
                        out, taps = stock(model, mx.array(blocks), cache, captures)
                    mx.eval(out, taps, [c.state for c in cache])
                    for plane in cache:
                        plane.finalize()
                    caches = [[plane.extract(i) for plane in cache] for i in range(args.batch)]
                    for i in range(args.batch):
                        features[i].append(taps[i:i + 1])
                        logits[i].append(out[i:i + 1])
                else:
                    for i, block in enumerate(blocks):
                        out, taps = stock(model, mx.array([block]), caches[i], captures)
                        mx.eval(out, taps, [c.state for c in caches[i]])
                        features[i].append(taps)
                        logits[i].append(out)
            mx.eval([[c.state for c in cache] for cache in caches])
            mx.synchronize()
            elapsed = time.perf_counter() - start
            after = stats(model)
            counts = {k: after[k] - before[k] for k in (
                'prefill_calls', 'prefill_batch_calls', 'prefill_batch_rows', 'fallbacks'
            )}
            counts['reasons'] = {k: v - before['reasons'].get(k, 0)
                                for k, v in after['reasons'].items()
                                if v != before['reasons'].get(k, 0)}
            return (caches, [mx.concatenate(x, axis=1) for x in logits],
                    [mx.concatenate(x, axis=1) for x in features], elapsed, counts)

        serial_cache, serial_logits, serial_taps, serial_s, serial_counts = prefill(False)
        packed_cache, packed_logits, packed_taps, packed_s, packed_counts = prefill(True)
        report['observational_seconds'] = {'serial': serial_s, 'packed': packed_s}
        report['fused_gdn'] = {'serial': serial_counts, 'packed': packed_counts}
        lanes = []
        for i in range(args.batch):
            cache_equal = [state_equal(mx, a.state, b.state)
                           for a, b in zip(serial_cache[i], packed_cache[i], strict=True)]
            continuation = []
            nxt = int(mx.argmax(serial_logits[i][0, -1]).item())
            for _ in range(4):
                a, af = model.forward_with_taps(mx.array([[nxt]]), serial_cache[i], captures)
                b, bf = model.forward_with_taps(mx.array([[nxt]]), packed_cache[i], captures)
                mx.eval(a, b, af, bf)
                continuation.append(same(mx, a, b) and same(mx, af, bf))
                nxt = int(mx.argmax(a[0, -1]).item())
            lanes.append({
                'full_logits_exact': same(mx, serial_logits[i], packed_logits[i]),
                'all_trained_taps_exact': same(mx, serial_taps[i], packed_taps[i]),
                'prepared_state_exact': all(cache_equal),
                'different_state_layers': [j for j, equal in enumerate(cache_equal) if not equal],
                'continuation_exact': all(continuation),
            })
        report['lanes'] = lanes
        report['passed'] = all(all(v for k, v in lane.items() if k != 'different_state_layers')
                               for lane in lanes) and packed_counts['prefill_batch_calls'] > 0
        report['verdict'] = 'pass_exact_serial_equivalence' if report['passed'] else 'decline_serial_equivalence'
    except Exception as error:
        import traceback
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    finally:
        report['finished_unix'] = time.time()
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: report.get(k) for k in ('passed', 'verdict', 'lanes', 'error')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
