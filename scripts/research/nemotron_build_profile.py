#!/usr/bin/env python3
"""Host profile of building one Nemotron-H exact verify graph (no eval timing).

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  On the Super VL NVFP4 export with the external
route's exact kernel policy, after a short prefill: build the lazy graph of
a 6-row chain verify ``--reps`` times under cProfile (each build evaluated
outside the profile), and write the top functions by own time and by
cumulative time, plus Python call counts per build.  Attribution only.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import signal
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron_tree_check as check  # noqa: E402
import nemotron_tree_profile as profile  # noqa: E402


def unit(out, args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    adapter = resolve_adapter(check.TARGET)(str(check.TARGET),
                                            execution_policy=dict(check.EXACT_POLICY))
    model = adapter.model
    ids = check._prompt(adapter.tokenizer, args.context)
    cache = model.make_cache()
    model._checkpoint_logits(mx.array([ids[:-1]]), cache)
    mx.eval([entry.state for entry in cache])
    tokens = profile._tree_shapes(model, cache, ids[-1])["chain8"][0][: args.rows]

    def build(private):
        transaction = HybridVerifyRows([private]).begin([args.rows])
        logits, taps = model.forward_with_taps(mx.array([tokens]), transaction.caches,
                                               check.CAPTURES)
        return [logits, taps] + [entry.state for entry in transaction.caches]

    for _ in range(3):
        mx.eval(build(profile._fresh(cache)))
    prof = cProfile.Profile()
    walls = []
    for _ in range(args.reps):
        private = profile._fresh(cache)
        mx.synchronize()
        t0 = time.perf_counter()
        prof.enable()
        outputs = build(private)
        prof.disable()
        walls.append(time.perf_counter() - t0)
        mx.eval(outputs)
    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    calls = sum(v[1] for v in stats.stats.values()) / args.reps
    stats.sort_stats("tottime").print_stats(45)
    stats.sort_stats("cumulative").print_stats(60)
    (out / "pstats.txt").write_text(stream.getvalue())
    report = {"schema": "mlx2.nemotron-build-profile.v1", "checkpoint": check.checkpoint(),
              "source": check.source(), "policy": check.EXACT_POLICY,
              "env": {k: v for k, v in os.environ.items() if k.startswith("MLX2_")},
              "context": args.context, "rows": args.rows, "reps": args.reps,
              "profiled_build_ms_median": round(statistics.median(walls) * 1e3, 2),
              "python_calls_per_build": round(calls), "qualified": False,
              "evidence_kind": "host attribution profile (cProfile inflates wall time)"}
    check.atomic(out / "report.json", report)
    print(json.dumps(report), flush=True)
    adapter.close()
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-build-profile-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--context", str(args.context), "--rows", str(args.rows),
                   "--reps", str(args.reps)]
        child = profile.subprocess_popen(command)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
