# SPDX-License-Identifier: MIT
"""GPU microbench + equality check: Source B1 tier rows, reference loop vs mixed_v1.

Synthetic experts at the real shapes (gate/up 2304x5120, down 5120x2304):
packed original MXFP4 objects built through the production HostExpert ->
materialize path, and Q2 objects (g64, g32h) from mx.quantize bits=2. Runs the
real ``SourceV41Weights._moe_routes`` with a minimal in-memory store (no model,
no disk reads), under routed_fused_fp32, for several row mixes and both
``source_q2_tier_math`` policies. Reports host graph-build time, synchronous
row latency (build + eval, as at a per-layer route sync) and pipelined
per-row time (36 rows, one eval), plus reference-vs-mixed output equality.

Run through the GPU queue only:
GPUQ_SESSION=... /Users/Shared/mlxuag/gpuq-reference.sh <label> \
    .venv/bin/python scripts/research/ds41_q2_mixed_row_bench.py --out <json>
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np


def host_expert(kind, rng):
    from mlx2.adapters.deepseek_v41_source_buffers import HostExpert, canonical_layout
    layout = canonical_layout(kind)
    data = np.empty(layout.nbytes, np.uint8)
    for t in layout.tensors:
        view = data[t.offset:t.offset + t.nbytes]
        n = int(np.prod(t.shape))
        if t.name.endswith('qweight'):
            view[:] = rng.integers(0, 256, t.nbytes, dtype=np.uint8)
        elif t.dtype == '|u1':                      # E8M0 scales near 2^-7
            view[:] = rng.integers(118, 124, n, dtype=np.uint8)
        else:                                       # Q1 F16 scales/biases
            view.view('<f2')[:] = (rng.standard_normal(n) * 0.01).astype('<f2')
    return HostExpert(data, layout)


def packed(kind, rng):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_buffers import materialize
    obj = materialize(host_expert(kind, rng))
    mx.eval(obj.backing, *obj.values())
    return obj


def q2(fmt, rng):
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
    group = 64 if fmt == 'g64' else 32
    arrays = {}
    for kind, shape in (('gate', (2304, 5120)), ('up', (2304, 5120)), ('down', (5120, 2304))):
        w = mx.array((rng.standard_normal(shape) * 0.02).astype(np.float16))
        q, s, b = mx.quantize(w, group_size=group, bits=2)
        arrays.update({kind + '.q2weight': q, kind + '.q2scales': s, kind + '.q2biases': b})
    mx.eval(arrays)
    return q1m.Q2Expert(arrays, q2_format=fmt)


def weights(experts, *, tier, policy):
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertStore
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    store = SimpleNamespace(_active=True, _error=None, _lock=threading.RLock(), _leased={}, _pending_uploads={},
                            _pending_host={}, packed_buffers=True, stats={'lookups': 0, 'windows': 0})
    store._ready = lambda: None
    store.window = MethodType(NativeExpertStore.window, store)
    store.observe_routes = lambda p, v: None
    store.q2_tier_active = lambda prefix: tier

    def acquire(prefix, e):
        store._leased[e] = experts[e]
        return experts[e]
    store.acquire = acquire
    w = object.__new__(SourceV41Weights); w._ready = lambda: None
    w.artifact = {'args': SimpleNamespace(hidden_size=5120, num_experts_per_tok=6)}
    w.expert_math_policy = 'routed_fused_fp32'; w.scheduled_expert_calls = {}; w.expert_stream = store
    w.source_q2_tier_math = policy
    return w


def measure(w, x, ids, scores, reps, pipeline):
    import mlx.core as mx
    for _ in range(5):
        mx.eval(w.moe(x, ids, scores, 'blk.4'))
    build, sync = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = w.moe(x, ids, scores, 'blk.4')
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        build.append((t1 - t0) * 1e3); sync.append((t2 - t0) * 1e3)
    piped = []
    for _ in range(max(3, reps // 10)):
        t0 = time.perf_counter()
        outs = [w.moe(x + i * 1e-3, ids, scores, 'blk.4') for i in range(pipeline)]
        mx.eval(outs)
        piped.append((time.perf_counter() - t0) * 1e3 / pipeline)
    return dict(build_ms=statistics.median(build), sync_ms=statistics.median(sync),
                sync_p90_ms=sorted(sync)[int(0.9 * len(sync))], pipelined_ms_per_row=statistics.median(piped))


def q2_components(reps):
    """Single Q2 expert, M=1: projection variants (why the per-expert law costs what it does)."""
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
    rng = np.random.default_rng(5)
    out = {}
    x = mx.array(rng.standard_normal((1, 5120)).astype(np.float32))
    for fmt in ('g64', 'g32h'):
        e = q2(fmt, rng); g = e.group_size
        gu_w = mx.concatenate([e['gate.q2weight'], e['up.q2weight']]); gu_s = mx.concatenate([e['gate.q2scales'], e['up.q2scales']])
        gu_b = mx.concatenate([e['gate.q2biases'], e['up.q2biases']])
        f32 = {k: v.astype(mx.float32) for k, v in e.items() if not k.endswith('weight')}
        mx.eval(gu_w, gu_s, gu_b, f32)
        qmm = lambda a, kind, src=e: mx.quantized_matmul(a, e[kind + '.q2weight'], src[kind + '.q2scales'], src[kind + '.q2biases'],
                                                          transpose=True, group_size=g, bits=2)
        variants = {
            'gate_up_separate_f16scales': lambda: [qmm(x, 'gate'), qmm(x, 'up')],
            'gate_up_separate_f32scales': lambda: [qmm(x, 'gate', f32), qmm(x, 'up', f32)],
            'gate_up_concatenated': lambda: [mx.quantized_matmul(x, gu_w, gu_s, gu_b, transpose=True, group_size=g, bits=2)],
            'down_only': lambda: [qmm(x[:, :2304], 'down')],
            'rotate_5120': lambda: [q1m.rotate_activation(x, 'block_hadamard_256')],
        }
        for name, fn in variants.items():
            for _ in range(5):
                mx.eval(fn())
            # 64 independent launches per eval: GPU time per call, launch-amortised.
            times = []
            for _ in range(reps):
                t0 = time.perf_counter(); mx.eval([fn() for _ in range(64)]); times.append((time.perf_counter() - t0) * 1e3 / 64)
            out[fmt + '.' + name + '_ms'] = statistics.median(times)
        if True:
            a = mx.quantized_matmul(x, gu_w, gu_s, gu_b, transpose=True, group_size=g, bits=2)
            b = mx.concatenate([qmm(x, 'gate'), qmm(x, 'up')], axis=1)
            out[fmt + '.concat_bitwise_equal'] = bool(mx.array_equal(a, b).item())
    return out


def main():
    import mlx.core as mx
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--reps', type=int, default=60)
    parser.add_argument('--pipeline', type=int, default=36)
    args = parser.parse_args()
    if Path(args.out).exists() and Path(args.out).stat().st_size:
        raise SystemExit('refusing to overwrite nonempty evidence ' + args.out)
    assert mx.default_device() == mx.gpu and mx.device_info().get('device_name') == 'Apple M5 Max'
    rng = np.random.default_rng(11)
    orig = {e: packed('original', rng) for e in (3, 40, 77, 120, 201, 333)}
    q1p = {e: packed('q1', rng) for e in (55,)}
    q2s = {fmt: {e: q2(fmt, rng) for e in (12, 90, 150, 260, 301, 370)} for fmt in ('g64', 'g32h')}
    x = mx.array(rng.standard_normal((1, 5120)).astype(np.float32))
    raw = np.sort(rng.random(6).astype(np.float32))[::-1]
    scores = mx.array((raw / raw.sum())[None, :].astype(np.float32))
    rows = {
        'all_q4_nontier': (orig, list(orig), False),
        'all_q4_tier_layer': (orig, list(orig), True),
    }
    for fmt in ('g64', 'g32h'):
        rows[f'q4x3_q2x3_{fmt}'] = ({**orig, **q2s[fmt]}, [3, 12, 40, 90, 77, 150], True)
        rows[f'q4x2_q1x1_q2x3_{fmt}'] = ({**orig, **q1p, **q2s[fmt]}, [3, 55, 12, 40, 90, 150], True)
        rows[f'q4x1_q2x5_{fmt}'] = ({**orig, **q2s[fmt]}, [3, 12, 90, 150, 260, 301], True)
        rows[f'q2x6_{fmt}'] = (q2s[fmt], list(q2s[fmt]), True)
        rows[f'q4x3_q2x2_dupzero_{fmt}'] = ({**orig, **q2s[fmt]}, [3, 12, 40, 3, 90, 77], True)
    result = dict(device=mx.device_info().get('device_name'), mlx=mx.__version__,
                  mlx2_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
                  checkpoint='synthetic (no model checkpoint): random MXFP4 originals, mx.quantize Q2 g64/g32h at real shapes',
                  reps=args.reps, pipeline=args.pipeline, rows={})
    for name, (experts, ids, tier) in rows.items():
        ids_np = np.array([ids], np.uint32)
        row_scores = scores
        if 'dupzero' in name:     # route drop: dropped slot 3 redirected to the top expert, weight 0
            s = np.array(scores); s[0, 3] = 0.0; row_scores = mx.array(s)
        entry = {}
        outputs = {}
        for policy in ('reference', 'mixed_v1'):
            w = weights(experts, tier=tier, policy=policy)
            outputs[policy] = np.array(w.moe(x, ids_np, row_scores, 'blk.4'))
            w.scheduled_expert_calls.clear()
            entry[policy] = measure(w, x, ids_np, row_scores, args.reps, args.pipeline)
            entry[policy]['counters_per_call'] = {k: v / (args.reps + 5 + args.pipeline * max(3, args.reps // 10))
                                                  for k, v in sorted(w.scheduled_expert_calls.items())}
        a, b = outputs['reference'], outputs['mixed_v1']
        entry['equal_bitwise'] = bool(np.array_equal(a, b))
        entry['max_abs_diff'] = float(np.abs(a - b).max())
        entry['max_abs_ref'] = float(np.abs(a).max())
        entry['finite'] = bool(np.isfinite(a).all() and np.isfinite(b).all())
        entry['sync_speedup'] = entry['reference']['sync_ms'] / entry['mixed_v1']['sync_ms']
        result['rows'][name] = entry
        print(name, {p: {k: round(v, 3) for k, v in entry[p].items() if k.endswith('ms') or k.endswith('row')}
                     for p in ('reference', 'mixed_v1')}, 'equal', entry['equal_bitwise'], 'maxdiff', entry['max_abs_diff'],
              flush=True)
    result['q2_components'] = q2_components(max(10, args.reps // 3))
    print(json.dumps(result['q2_components'], indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out + '.tmp'); tmp.write_text(json.dumps(result, indent=1)); tmp.rename(args.out)


if __name__ == '__main__':
    main()
