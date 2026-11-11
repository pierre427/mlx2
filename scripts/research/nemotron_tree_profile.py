#!/usr/bin/env python3
"""Where a Nemotron-H exact tree verify round spends its time.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  The unit loads the Super VL NVFP4 export with the
external route's exact kernel policy and, per context, times:

* ``decode``: one ordinary one-token step;
* ``chainN``: the exact chain verify of N rows (``forward_with_taps`` on a
  speculating copy), N = 8 and 16;
* ``treeN_chain``: the tree executor on an N-row tree that is a plain chain
  (isolates tree-path overhead from topology);
* ``treeN``: the tree executor on a branched N-row tree (greedy trunk +
  runner-up branches, as nemotron_tree_check), N = 8 and 16.

Each arm reports the uninstrumented median wall time (``mx.eval``) and an
exclusive split by block (Mamba mixer, attention, MoE, head) measured with
``mx.eval`` at those boundaries (absolute numbers include the sync floor),
plus the distinct routed experts each MoE layer reads (the expert union).
Attribution only, no thermal hold: never statistics or qualification.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron_tree_check as check  # noqa: E402


def _tree_shapes(model, cache, anchor):
    import mlx.core as mx

    probe = copy.deepcopy(cache)
    trunk, runner_up, token = [], [], anchor
    for _ in range(15):
        logits = model(mx.array([[token]]), probe)[0, -1].astype(mx.float32)
        ranked = mx.argsort(-logits)[:2]
        mx.eval(ranked)
        token = int(ranked[0].item())
        trunk.append(token)
        runner_up.append(int(ranked[1].item()))
    shapes = {}
    for rows in (8, 16):
        shapes[f"chain{rows}"] = ([anchor] + trunk[: rows - 1], list(range(-1, rows - 1)))
    # Branched trees like the drafter's: trunk then runner-up siblings.
    tokens8 = [anchor] + trunk[:4]
    parents8 = [-1, 0, 1, 2, 3]
    for depth in range(3):
        tokens8.append(runner_up[depth])
        parents8.append(depth)
    shapes["tree8"] = (tokens8, parents8)
    shapes["tree16"] = check._tree(model, cache, anchor)
    return shapes


class Split:
    """Exclusive per-block wall time with ``mx.eval`` at block boundaries."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.experts = []
        self.active = False
        self._patched = []

    def wrap(self, owner, name, label):
        import mlx.core as mx

        original = getattr(owner, name)
        split = self

        def timed(*args, **kwargs):
            if not split.active:
                return original(*args, **kwargs)
            mx.eval([value for value in args if isinstance(value, mx.array)])
            started = time.perf_counter()
            output = original(*args, **kwargs)
            mx.eval(output)
            split.seconds[label] += time.perf_counter() - started
            split.calls[label] += 1
            return output

        setattr(owner, name, timed)
        self._patched.append((owner, name, original))

    def watch_experts(self):
        import mlx.core as mx
        import numpy as np

        from mlx2.runtime.row_exact import ops

        original = ops.routed_experts
        split = self

        def recorded(module, latent, inds, **kwargs):
            if split.active:
                mx.eval(inds)
                split.experts.append(
                    (int(inds.shape[1]), int(np.unique(np.asarray(inds)).size))
                )
            return original(module, latent, inds, **kwargs)

        ops.routed_experts = recorded
        self._patched.append((ops, "routed_experts", original))

    def restore(self):
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)


def _fresh(cache):
    import mlx.core as mx

    private = copy.deepcopy(cache)
    mx.eval([entry.state for entry in private])
    return private


def _median_wall(arm, reps):
    """Median of ``reps`` timed runs; each run's private cache is made first."""
    import mlx.core as mx

    samples = []
    for _ in range(reps + 1):
        state = arm["setup"]()
        mx.synchronize()
        started = time.perf_counter()
        arm["run"](state)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples[1:])


def unit(out, args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models import nemotron_h
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    report = {
        "schema": "mlx2.nemotron-tree-profile.v1", "checkpoint": check.checkpoint(),
        "source": check.source(), "mlx": mx.__version__, "policy": check.EXACT_POLICY,
        "qualified": False, "evidence_kind": "attribution profile, no thermal hold",
        "contexts": {},
    }
    adapter = resolve_adapter(check.TARGET)(str(check.TARGET),
                                            execution_policy=dict(check.EXACT_POLICY))
    model = adapter.model
    split = Split()
    split.wrap(nemotron_h.NemotronHMamba2Mixer, "__call__", "mamba")
    split.wrap(nemotron_h.NemotronHMamba2Mixer, "tree_forward", "mamba")
    split.wrap(nemotron_h.NemotronHAttention, "__call__", "attention")
    split.wrap(nemotron_h.Model, "_tree_attention_block", "attention_block")
    split.wrap(nemotron_h.NemotronHMoE, "__call__", "moe")
    split.wrap(nemotron_h.NemotronHMoE, "routed_batch_exact", "moe")
    split.wrap(nemotron_h.Model, "_row_exact_head", "head")
    split.wrap(nemotron_h.Model, "logits", "head")
    split.watch_experts()
    executor = NemotronTreeExecutor()
    try:
        for context in args.contexts:
            ids = check._prompt(adapter.tokenizer, context)
            cache = model.make_cache()
            model._checkpoint_logits(mx.array([ids[:-1]]), cache)
            mx.eval([entry.state for entry in cache])
            shapes = _tree_shapes(model, cache, ids[-1])
            row = {}

            def decode_run(private):
                logits, taps = model.forward_with_taps(
                    mx.array([[ids[-1]]]), private, check.CAPTURES
                )
                mx.eval(logits, taps)

            def chain(tokens):
                def setup():
                    return HybridVerifyRows([_fresh(cache)]).begin([len(tokens)])

                def run(transaction):
                    logits, taps = model.forward_with_taps(
                        mx.array([tokens]), transaction.caches, check.CAPTURES
                    )
                    mx.eval(logits, taps)
                    transaction.abort()
                return {"setup": setup, "run": run}

            def tree(tokens, parents):
                def run(private):
                    logits, taps, transaction = executor.forward(
                        model, tokens, parents, private, check.CAPTURES, cached=False,
                    )
                    mx.eval(logits, taps)
                    transaction.abort()
                return {"setup": lambda: _fresh(cache), "run": run}

            arms = {"decode": {"setup": lambda: _fresh(cache), "run": decode_run}}
            for rows in (8, 16):
                tokens, parents = shapes[f"chain{rows}"]
                arms[f"chain{rows}"] = chain(tokens)
                arms[f"tree{rows}_chain"] = tree(tokens, parents)
                arms[f"tree{rows}"] = tree(*shapes[f"tree{rows}"])
            for name, arm in arms.items():
                arm["run"](arm["setup"]())  # warm (compiles, parity probes, canaries)
                wall = _median_wall(arm, args.reps)
                split.seconds.clear()
                split.calls.clear()
                split.experts.clear()
                state = arm["setup"]()
                split.active = True
                arm["run"](state)
                split.active = False
                instrumented = dict(split.seconds)
                union = [unique for rows, unique in split.experts]
                row[name] = {
                    "wall_ms": wall * 1e3,
                    "split_ms": {key: value * 1e3 for key, value in instrumented.items()},
                    "split_calls": dict(split.calls),
                    "expert_union_mean": (sum(union) / len(union)) if union else None,
                    "expert_rows": split.experts[0][0] if split.experts else None,
                }
                print(json.dumps({"context": context, "arm": name, "wall_ms": round(wall * 1e3, 1),
                                  "split": {k: round(v * 1e3, 1) for k, v in instrumented.items()},
                                  "experts": row[name]["expert_union_mean"]}), flush=True)
            report["contexts"][str(context)] = {
                "trees": {key: {"tokens": t, "parents": p} for key, (t, p) in shapes.items()},
                "arms": row,
            }
            check.atomic(out / "report.json", report)
            del cache
            mx.clear_cache()
    finally:
        split.restore()
    report["receipt"] = {k: v for k, v in executor.receipt().items() if k != "source_identity"}
    check.atomic(out / "report.json", report)
    adapter.close()
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-tree-profile-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--contexts", *map(str, args.contexts), "--reps", str(args.reps)]
        child = subprocess_popen(command)
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


def subprocess_popen(command):
    import subprocess

    return subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192])
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
