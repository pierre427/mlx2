#!/usr/bin/env python3
"""CPU-only WY KDA forward/VJP research probe; no checkpoint loads or policy edits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('raw_q', 'raw_k', 'v', 'forget_logits', 'beta_logits', 'A_log', 'dt_bias', 'initial_state')


def fixture(length, *, heads=2, key_dim=16, value_dim=12, gate='mild', seed=17):
    from mlx2.runtime.models.kimi_linear import kda_gates, l2_qk
    from mlx2.runtime.models.gated_delta import gated_delta_ops
    from mlx2.runtime.models.kda_chunk_wy import chunk_wy

    mx.random.seed(seed)
    b, h, d, dv = 1, heads, key_dim, value_dim
    shape = (b, length, h, d)
    raw_q, raw_k = mx.random.normal(shape), mx.random.normal(shape)
    v = mx.random.normal((b, length, h, dv))
    a = mx.random.normal(shape) + (-3. if gate == 'mild' else 0.)
    beta = mx.random.normal((b, length, h))
    a_log = mx.full((h,), np.log(30.) if gate == 'strong' else 0., dtype=mx.float32)
    dt_bias = mx.random.normal((h * d,)) * .1
    state = mx.random.normal((b, h, dv, d)) * .1
    primals = [raw_q, raw_k, v, a, beta, a_log, dt_bias, state]
    cotangents = [mx.random.normal((b, length, h, dv)), mx.random.normal(state.shape)]
    mx.eval(primals, cotangents)

    def forward(*values, method, segment=512):
        rq, rk, value, forget, update, alog, bias, s0 = values
        q, k = l2_qk(rq, rk)
        log_g, beta = kda_gates(forget, update, alog.reshape(h, 1), bias.reshape(h, d))
        if method == 'wy':
            return chunk_wy(q, k, value.astype(mx.float32), log_g, beta, s0, segment=segment)
        return gated_delta_ops(q, k, value.astype(mx.float32), mx.exp(log_g), beta, s0)

    return primals, cotangents, forward


def error(reference, value, *, atol, rtol):
    a, b = np.asarray(reference), np.asarray(value)
    difference = np.abs(a - b)
    norm = float(np.linalg.norm(a.reshape(-1)))
    return {'max_abs': float(difference.max()), 'relative_l2': float(np.linalg.norm(difference.reshape(-1))) / max(norm, 1e-30),
            'reference_l2': norm, 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'allclose': bool(np.allclose(a, b, atol=atol, rtol=rtol)), 'atol': atol, 'rtol': rtol}


def probe_case(length, *, segment=512, heads=2, key_dim=16, value_dim=12, gate='mild', repetitions=2):
    if mx.default_device() != mx.cpu:
        raise ValueError('This diagnostic must run on CPU')
    primals, cotangents, forward = fixture(length, heads=heads, key_dim=key_dim, value_dim=value_dim, gate=gate)
    results, timings = {}, {}
    for method in ('reference', 'wy'):
        samples = []
        for _ in range(repetitions):
            start = time.perf_counter()
            outputs, gradients = mx.vjp(lambda *x: forward(*x, method=method, segment=segment), primals, cotangents)
            mx.eval(outputs, gradients)
            samples.append(time.perf_counter() - start)
        results[method] = (outputs, gradients)
        timings[method] = samples
    reference, candidate = results['reference'], results['wy']
    outputs = {name: error(a, b, atol=2e-5, rtol=3e-4) for name, a, b in
               zip(('output', 'final_state'), reference[0], candidate[0], strict=True)}
    gradients = {name: error(a, b, atol=6e-5, rtol=2e-3) for name, a, b in
                 zip(NAMES, reference[1], candidate[1], strict=True)}
    return {'length': length, 'segment': segment, 'shape': {'batch': 1, 'heads': heads, 'key_dim': key_dim, 'value_dim': value_dim},
            'gate': gate, 'outputs': outputs, 'gradients': gradients,
            'passed': all(x['finite'] and x['allclose'] for x in [*outputs.values(), *gradients.values()]),
            'cpu_forward_and_vjp_seconds': timings,
            'timing_scope': 'synthetic CPU diagnostic only; includes graph construction; not GPU performance or checkpoint qualification'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[17, 65, 257, 513, 1025])
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--key-dim', type=int, default=32)
    parser.add_argument('--value-dim', type=int, default=32)
    parser.add_argument('--repetitions', type=int, default=2)
    args = parser.parse_args()
    if any(not 1 <= n <= 4096 for n in args.lengths) or not 1 <= args.heads <= 2 or not 1 <= args.key_dim <= 128 or not 1 <= args.value_dim <= 128:
        parser.error('bounded synthetic CPU shapes required')
    mx.set_default_device(mx.cpu)
    cases = []
    for gate in ('mild', 'strong'):
        for length in args.lengths:
            result = probe_case(length, heads=args.heads, key_dim=args.key_dim, value_dim=args.value_dim,
                                gate=gate, repetitions=args.repetitions)
            cases.append(result)
            print(json.dumps({'length': length, 'gate': gate, 'passed': result['passed'],
                              'timings': result['cpu_forward_and_vjp_seconds']}), flush=True)
    paths = [Path(__file__), ROOT / 'src/mlx2/runtime/models/kda_chunk_wy.py',
             ROOT / 'src/mlx2/runtime/models/gated_delta.py', ROOT / 'src/mlx2/runtime/models/kimi_linear.py']
    receipt = {'schema': 'mlx2.kimi-kda-wy-cpu-vjp.v1', 'checkpoint': 'synthetic-kda-wy-cpu@fixture-v1',
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               'device': 'cpu', 'cases': cases, 'all_passed': all(case['passed'] for case in cases),
               'peak_process_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'qualified': False, 'training_policy_changed': False,
               'limits': ['No GPU execution or full model load.', 'No training suffix was switched to WY.',
                          'Shapes reduce head count; CPU timings do not predict full Kimi GPU training throughput.',
                          'Forward and VJP tolerances admit FP32 arithmetic reordering, not exact bit parity.',
                          'Only negative softplus-derived log gates and unmasked sequences are exercised.']}
    write(args.output, receipt)
    print(json.dumps({'output': str(args.output), 'all_passed': receipt['all_passed']}))
