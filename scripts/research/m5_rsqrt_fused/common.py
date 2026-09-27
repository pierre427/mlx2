"""Research-only helpers. GPU callers must run through owned_exec.py/cpg_job.py."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def require_ownership():
    owner = json.loads(Path('/Users/Shared/mlxuag/gpu.lock/owner.json').read_text())
    parent = os.getppid()
    grandparent = int(subprocess.check_output(
        ['ps', '-o', 'ppid=', '-p', str(parent)], text=True).strip())
    if owner.get('pid') != grandparent or not owner.get('cpg_generation'):
        raise SystemExit('Refusing GPU work without owned_exec/cpg_job ancestry')
    command = subprocess.check_output(['ps', '-o', 'command=', '-p', str(parent)], text=True)
    if 'owned_exec.py' not in command:
        raise SystemExit('Refusing GPU work without tmp advisory-lock wrapper')
    return owner


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_arrays(ref, test):
    import mlx.core as mx
    import numpy as np
    mx.eval(ref, test)
    a32 = np.asarray(ref.astype(mx.float32))
    b32 = np.asarray(test.astype(mx.float32))
    assert a32.shape == b32.shape
    a, b = a32.astype(np.float64), b32.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    diff = np.where(finite, b - a, 0)
    norm = float(np.linalg.norm(a.reshape(-1)))
    error = float(np.linalg.norm(diff.reshape(-1)))
    mismatch = int(np.count_nonzero(a32.view(np.uint32) != b32.view(np.uint32)))
    result = dict(shape=list(a.shape), dtype=str(ref.dtype), count=a.size,
                  bit_mismatches=mismatch, mismatch_fraction=mismatch / a.size,
                  nonfinite_reference=int(np.count_nonzero(~np.isfinite(a))),
                  nonfinite_candidate=int(np.count_nonzero(~np.isfinite(b))),
                  max_abs=float(np.max(np.abs(diff))),
                  rms_error=float(np.sqrt(np.mean(diff * diff))),
                  relative_l2=error / norm if norm else (0.0 if not error else None))
    if ref.dtype == mx.bfloat16:
        # Ordered integer BF16 representations: actual ULP spacing, not unit-scale ULPs.
        ua = (a32.view(np.uint32) >> 16).astype(np.int64)
        ub = (b32.view(np.uint32) >> 16).astype(np.int64)
        oa = np.where(ua & 0x8000, 0x8000 - (ua & 0x7fff), 0x8000 + ua)
        ob = np.where(ub & 0x8000, 0x8000 - (ub & 0x7fff), 0x8000 + ub)
        result['max_bf16_ulps'] = int(np.max(np.abs(oa - ob)))
    return result


def time_variants(variants, rounds=21, inner=5, seed=123):
    """Interleaved synchronous wall time, includes Python/MLX dispatch and evaluation.

    Each callable must CREATE a fresh lazy kernel invocation on every call.
    Returning a pre-evaluated array invalidates this benchmark.
    """
    import mlx.core as mx
    import numpy as np
    rng = random.Random(seed)
    for fn in variants.values():
        for _ in range(3):
            mx.eval(fn())
    mx.synchronize()
    samples = {key: [] for key in variants}
    orders = []
    for _ in range(rounds):
        order = list(variants)
        rng.shuffle(order)
        orders.append(order)
        for key in order:
            start = time.perf_counter_ns()
            for _ in range(inner):
                mx.eval(variants[key]())
            samples[key].append((time.perf_counter_ns() - start) / inner / 1000)
    result = {'method': 'synchronous per-call wall time including Python/MLX dispatch',
              'rounds': rounds, 'inner': inner, 'orders': orders, 'arms': {}}
    bootstrap = np.random.default_rng(seed).integers(0, rounds, (4000, rounds))
    for key, values in samples.items():
        entry = dict(samples_us=values, median_us=float(np.median(values)),
                     p10_us=float(np.percentile(values, 10)), p90_us=float(np.percentile(values, 90)))
        if 'precise' in samples:
            ratios = np.array(samples['precise']) / np.array(values)
            entry['paired_median_speedup'] = float(np.median(ratios))
            entry['paired_bootstrap_95pct'] = np.percentile(
                np.median(ratios[bootstrap], axis=1), [2.5, 97.5]).tolist()
        result['arms'][key] = entry
    return result
