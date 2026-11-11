#!/usr/bin/env python3
"""Census of one Nemotron-H exact verify round: host build, GPU time, kernels.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  On the Super VL NVFP4 export with the external
route's exact kernel policy, per context and arm (one-token decode, chain
verify of 6 and 8 rows, a branched 6-row tree):

* ``build_ms``: wall time to construct the lazy graph (Python, plus any
  host synchronization hidden inside the forward);
* ``eval_ms``: wall time of ``mx.eval`` on everything the round produces
  (logits, taps and the private cache planes);
* ``primitives``: kernel-dispatching primitives in that graph (from
  ``mx.export_to_dot``; views and no-op reshapes excluded), with the top
  primitives and a per-stage share.

Attribution only, no thermal hold: never statistics or qualification.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
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

# Primitives that never dispatch a kernel on their own.
VIEWS = {"Reshape", "Squeeze", "ExpandDims", "Broadcast", "AsStrided",
         "Transpose", "Slice", "Depends", "StopGradient", "Split", "Unflatten",
         "Flatten", "NumberOfElements"}


def _census(out, name, outputs):
    import mlx.core as mx

    from mlx.utils import tree_flatten

    arrays = [value for _, value in tree_flatten(outputs) if isinstance(value, mx.array)]
    dot = out / f"{name}.dot"
    mx.export_to_dot(str(dot), *arrays)
    counts = collections.Counter(re.findall(r'label ="([^"]+)"', dot.read_text()))
    dispatch = {k: v for k, v in counts.items() if k not in VIEWS}
    dot.unlink()
    return {
        "primitives": sum(dispatch.values()),
        "views": sum(v for k, v in counts.items() if k in VIEWS),
        "top": dict(sorted(dispatch.items(), key=lambda kv: -kv[1])[:30]),
    }


def unit(out, args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    report = {
        "schema": "mlx2.nemotron-verify-census.v1", "checkpoint": check.checkpoint(),
        "source": check.source(), "mlx": mx.__version__, "policy": check.EXACT_POLICY,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MLX2_")},
        "qualified": False, "evidence_kind": "attribution census, no thermal hold",
        "contexts": {},
    }
    adapter = resolve_adapter(check.TARGET)(str(check.TARGET),
                                            execution_policy=dict(check.EXACT_POLICY))
    model = adapter.model
    executor = NemotronTreeExecutor()
    for context in args.contexts:
        ids = check._prompt(adapter.tokenizer, context)
        cache = model.make_cache()
        model._checkpoint_logits(mx.array([ids[:-1]]), cache)
        mx.eval([entry.state for entry in cache])
        shapes = profile._tree_shapes(model, cache, ids[-1])
        tokens8, _ = shapes["chain8"]
        tree6 = (shapes["tree8"][0][:6], shapes["tree8"][1][:6])

        def decode(private):
            logits, taps = model.forward_with_taps(mx.array([[ids[-1]]]), private, check.CAPTURES)
            return [logits, taps] + [entry.state for entry in private]

        def chain(rows):
            def run(private):
                transaction = HybridVerifyRows([private]).begin([rows])
                logits, taps = model.forward_with_taps(
                    mx.array([tokens8[:rows]]), transaction.caches, check.CAPTURES
                )
                planes = [entry.state for entry in transaction.caches]
                run.transaction = transaction
                return [logits, taps] + planes
            return run

        def tree_arm(shape):
            def run(private):
                logits, taps, transaction = executor.forward(
                    model, shape[0], shape[1], private, check.CAPTURES, cached=False
                )
                run.transaction = transaction
                return [logits, taps]
            return run

        arms = {"decode": decode, "chain6": chain(6), "chain8": chain(8),
                "tree6": tree_arm(tree6), "tree8": tree_arm(shapes["tree8"])}
        if args.arms:
            arms = {name: arms[name] for name in args.arms}
        row = {}
        for name, build in arms.items():
            for _ in range(2):  # warm: compiles, canaries, parity gates
                mx.eval(build(profile._fresh(cache)))
            builds, evals = [], []
            for _ in range(args.reps):
                private = profile._fresh(cache)
                mx.synchronize()
                t0 = time.perf_counter()
                outputs = build(private)
                t1 = time.perf_counter()
                mx.eval(outputs)
                mx.synchronize()
                t2 = time.perf_counter()
                builds.append(t1 - t0)
                evals.append(t2 - t1)
            census = _census(out, f"{context}-{name}", build(profile._fresh(cache)))
            row[name] = {
                "build_ms": round(statistics.median(builds) * 1e3, 2),
                "eval_ms": round(statistics.median(evals) * 1e3, 2),
                "total_ms": round(statistics.median([b + e for b, e in zip(builds, evals)]) * 1e3, 2),
                **census,
            }
            print(json.dumps({"context": context, "arm": name,
                              **{k: row[name][k] for k in ("build_ms", "eval_ms", "total_ms",
                                                            "primitives")}}), flush=True)
        report["contexts"][str(context)] = row
        check.atomic(out / "report.json", report)
        del cache
        mx.clear_cache()
    adapter.close()
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-verify-census-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--contexts", *map(str, args.contexts), "--reps", str(args.reps),
                   *(["--arms", *args.arms] if args.arms else [])]
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
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--arms", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
