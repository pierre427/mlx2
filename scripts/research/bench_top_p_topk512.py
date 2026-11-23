#!/usr/bin/env python3
"""Micro-benchmark: dense ``apply_top_p`` vs the opt-in top-512 window path.

Research-only.  Times the compiled filter alone (no model) at the Inkling 2
vocabulary (V = 201024) for 1 and 8 rows, on peaked rows (nucleus inside the
window: fast path) and flat rows (fallback: window attempt + dense redo), and
checks on every timed input that the output is bitwise ``apply_top_p``.  The
"fast" arm includes the fit check's host read, as the opt-in sampler pays it.

``--decode-steps N`` (N > 0) adds a per-decode-step comparison of whole
samplers (``make_sampler(temp, top_p)``), each step being one draw plus the
host read of the token, as a decode loop does it:

* ``dense``: the default sampler, then ``token.tolist()`` (one sync).
* ``eager``: the opt-in sampler, whose fit flag is read with its own sync
  before the draw is queued, then the token read (two syncs).
* ``deferred``: ``sampler.deferred``, flag and token read in one ``mx.eval``,
  a miss adding one redo read.

Optional ``--step-work-mb`` queues a matmul of that size ahead of each draw,
standing in for the forward the eager read waits on.  Each step's tokens are
checked equal across the three arms (same seed per step).

Defaults to the CPU.  ``--device gpu`` runs on Metal; that is for the
coordinator's GPU window, not routine use.  The numbers are filter timings,
never a model-path or serving qualification.
"""

from __future__ import annotations

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
import time
from pathlib import Path


def _rows(mx, rows, vocab, scale, seed, dtype):
    mx.random.seed(seed)
    x = mx.random.normal((rows, vocab)) * scale
    lp = (x - mx.logsumexp(x, axis=-1, keepdims=True)).astype(dtype)
    mx.eval(lp)
    return lp


def _time(fn, arg, iters, warmup, mx):
    for _ in range(warmup):
        mx.eval(fn(arg))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        mx.eval(fn(arg))
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p25_ms": samples[len(samples) // 4],
        "p75_ms": samples[(3 * len(samples)) // 4],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--vocab", type=int, default=201024)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--top-p", type=float, nargs="+", default=[0.9, 0.95])
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode-steps", type=int, default=0,
                        help="per-step sampler comparison (dense / eager / deferred)")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--step-work-mb", type=float, default=0.0,
                        help="matmul queued before each draw (stand-in forward)")
    args = parser.parse_args(argv)

    import mlx.core as mx

    from mlx2.runtime import sample_utils as su

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    dtype = getattr(mx, args.dtype)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    results = []
    for top_p in args.top_p:
        dense = mx.compile(lambda x, p=top_p: su.apply_top_p(x, p))
        window = mx.compile(lambda x, p=top_p: su.apply_top_p_topk512(x, p))

        def fast(x, dense=dense, window=window):
            masked, fits = window(x)
            if bool(mx.all(fits).item()):
                return masked
            return dense(x)

        for rows in args.rows:
            for shape, scale in (("peaked", 8.0), ("flat", 0.3)):
                lp = _rows(mx, rows, args.vocab, scale, seed=rows, dtype=dtype)
                fits = su.apply_top_p_topk512(lp, top_p)[1].tolist()
                fits = fits if isinstance(fits, list) else [fits]
                equal = bool(mx.array_equal(fast(lp), su.apply_top_p(lp, top_p)).item())
                dense_t = _time(dense, lp, args.iters, args.warmup, mx)
                fast_t = _time(fast, lp, args.iters, args.warmup, mx)
                row = {
                    "top_p": top_p, "rows": rows, "input": shape,
                    "rows_fit": sum(bool(f) for f in fits),
                    "bitwise_equal": equal,
                    "dense": dense_t, "fast": fast_t,
                    "speedup": dense_t["median_ms"] / fast_t["median_ms"],
                }
                results.append(row)
                print(
                    f"top_p={top_p} rows={rows} {shape:6s} fit={row['rows_fit']}/{rows} "
                    f"equal={equal} dense={dense_t['median_ms']:.3f}ms "
                    f"fast={fast_t['median_ms']:.3f}ms x{row['speedup']:.2f}"
                )
                if not equal:
                    print("  MISMATCH: fast output differs from apply_top_p")
    steps = []
    if args.decode_steps > 0:
        steps = _decode_steps(mx, su, args, dtype)
    record = {
        "device": args.device, "vocab": args.vocab, "dtype": args.dtype,
        "mlx": mx.__version__, "mlx2_revision": revision,
        "iters": args.iters, "warmup": args.warmup, "results": results,
        "decode_steps": steps,
    }
    if args.output:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
    ok = all(r["bitwise_equal"] for r in results) and all(r["tokens_equal"] for r in steps)
    return 0 if ok else 1


def _decode_steps(mx, su, args, dtype):
    """Per-step wall time of dense / eager-check / deferred samplers."""
    import os

    env = su.SAMPLING_TOPP_TOPK512_ENV
    previous = os.environ.get(env)
    try:
        os.environ.pop(env, None)
        samplers = {}
        os.environ[env] = "0"
        for top_p in args.top_p:
            dense = su.make_sampler(temp=args.temp, top_p=top_p)
            os.environ[env] = "1"
            fast = su.make_sampler(temp=args.temp, top_p=top_p)
            os.environ[env] = "0"
            samplers[top_p] = (dense, fast)
    finally:
        if previous is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = previous
    side = int((args.step_work_mb * 2**20 / 4) ** 0.5) if args.step_work_mb > 0 else 0
    work = mx.random.normal((side, side)) if side else None

    def queue_work(lp):
        # A lazy dependency of the logprobs, as a forward would be.
        if work is None:
            return lp
        return lp + 0 * (work @ work).sum().astype(lp.dtype)

    def dense_step(sampler, lp):
        return sampler(queue_work(lp)).tolist()

    def eager_step(sampler, lp):
        return sampler(queue_work(lp)).tolist()

    def deferred_step(sampler, lp):
        pending = sampler.deferred(queue_work(lp))
        if pending.fits is None:
            return pending.value.tolist()
        mx.eval(pending.value, pending.fits)
        return su.resolve_top_p_deferred(pending).tolist()

    out = []
    for top_p, (dense, fast) in samplers.items():
        for rows in args.rows:
            for shape, scale in (("peaked", 8.0), ("flat", 0.3)):
                lp = _rows(mx, rows, args.vocab, scale, seed=rows, dtype=dtype)
                arms = {
                    "dense": (dense_step, dense),
                    "eager": (eager_step, fast),
                    "deferred": (deferred_step, fast),
                }
                times = {name: [] for name in arms}
                tokens = {name: [] for name in arms}
                for step in range(args.warmup + args.decode_steps):
                    for name, (fn, sampler) in arms.items():
                        mx.random.seed(1000 + step)
                        start = time.perf_counter()
                        got = fn(sampler, lp)
                        elapsed = (time.perf_counter() - start) * 1e3
                        if step >= args.warmup:
                            times[name].append(elapsed)
                            tokens[name].append(got)
                stats = dict(su.TOPP_TOPK512_STATS)
                row = {
                    "top_p": top_p, "rows": rows, "input": shape,
                    "temp": args.temp, "step_work_mb": args.step_work_mb,
                    "tokens_equal": tokens["eager"] == tokens["dense"] == tokens["deferred"],
                    "topp_stats_after": stats,
                }
                for name, samples in times.items():
                    samples.sort()
                    row[name] = {
                        "median_ms": statistics.median(samples),
                        "p25_ms": samples[len(samples) // 4],
                        "p75_ms": samples[(3 * len(samples)) // 4],
                    }
                out.append(row)
                print(
                    f"step top_p={top_p} rows={rows} {shape:6s} "
                    f"dense={row['dense']['median_ms']:.3f}ms "
                    f"eager={row['eager']['median_ms']:.3f}ms "
                    f"deferred={row['deferred']['median_ms']:.3f}ms "
                    f"tokens_equal={row['tokens_equal']}"
                )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
