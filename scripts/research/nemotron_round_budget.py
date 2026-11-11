#!/usr/bin/env python3
"""Fine-grained time budget of one Nemotron-H speculative round (and decode).

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  On the Super VL NVFP4 export with the external
route's exact kernel policy and the default LiLiCorr drafter:

* ``verify`` arms (chain of 6 and 8 rows, one-token decode): exclusive GPU
  time per component (Mamba projections / conv / SSM / gated norm; MoE
  router / latent in / expert gather / combine / latent out / shared
  expert; attention one-row FP8 projections / SDPA; block norms;
  embedding; head), each wrapped with ``mx.eval`` barriers and nested calls
  subtracted from their parent;
* ``draft``: one LiLiCorr ``draft_distributions`` call after a real context
  (backbone layers, owned lm_head, lattice scorer, host remainder);
* the uninstrumented wall time of each arm, the instrumented sum and the
  barrier count, so the synchronization cost of the split is visible.

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
import nemotron_tree_profile as profile  # noqa: E402

LILICORR = Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-LiLiCorr-mlx-bf16"


class Budget:
    """Exclusive wall time per label with ``mx.eval`` barriers."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.barriers = 0
        self.active = False
        self._stack = []
        self._patched = []

    def reset(self):
        self.seconds.clear()
        self.calls.clear()
        self.barriers = 0

    def _arrays(self, value):
        import mlx.core as mx

        if isinstance(value, mx.array):
            return [value]
        if isinstance(value, (list, tuple)):
            return [item for entry in value for item in self._arrays(entry)]
        if isinstance(value, dict):
            return [item for entry in value.values() for item in self._arrays(entry)]
        return []

    def wrap(self, owner, name, label):
        import mlx.core as mx

        original = getattr(owner, name)
        budget = self

        def timed(*args, **kwargs):
            if not budget.active:
                return original(*args, **kwargs)
            tag = label(*args, **kwargs) if callable(label) else label
            mx.eval(budget._arrays(args) + budget._arrays(kwargs))
            budget.barriers += 1
            started = time.perf_counter()
            budget._stack.append(0.0)
            try:
                output = original(*args, **kwargs)
                mx.eval(budget._arrays(output))
                budget.barriers += 1
            finally:
                child = budget._stack.pop()
                total = time.perf_counter() - started
                budget.seconds[tag] += total - child
                budget.calls[tag] += 1
                if budget._stack:
                    budget._stack[-1] += total
            return output

        setattr(owner, name, timed)
        self._patched.append((owner, name, original))

    def restore(self):
        for owner, name, original in reversed(self._patched):
            setattr(owner, name, original)


def _install(budget, model, draft):
    from mlx import nn

    from mlx2.runtime import nvfp4_linear
    from mlx2.runtime.models import nemotron_h
    from mlx2.runtime.row_exact import ops

    def projection_label(linear, x, *_a, **_k):
        out = int(linear["weight"].shape[0])
        return {18560: "mamba_in_proj", 4096: "mamba_out_proj"}.get(out, f"mamba_proj_{out}")

    budget.wrap(nemotron_h, "_short_projection", projection_label)
    budget.wrap(nemotron_h.NemotronHMamba2Mixer, "_conv", "mamba_conv")
    budget.wrap(nemotron_h.NemotronHMamba2Mixer, "_ssm", "mamba_ssm")
    budget.wrap(nemotron_h.MambaRMSNormGated, "__call__", "mamba_gated_norm")
    budget.wrap(nemotron_h.NemotronHMamba2Mixer, "__call__", "mamba_other")
    budget.wrap(ops, "router_select", "moe_router")
    budget.wrap(ops, "routed_experts", "moe_expert_gather")
    budget.wrap(ops, "combine_experts", "moe_combine")
    budget.wrap(ops, "linear", lambda module, x, *, stage, **_k: stage)
    budget.wrap(ops, "row_independent",
                lambda fn, x, *, stage, **_k: f"rowind_{stage}")
    budget.wrap(nemotron_h.NemotronHMoE, "__call__", "moe_decode")
    budget.wrap(nemotron_h.NemotronHMoE, "routed_batch_exact", "moe_other")
    budget.wrap(nemotron_h.NemotronHAttention, "__call__", "attention_other")
    budget.wrap(nemotron_h, "scaled_dot_product_attention", "attention_sdpa")
    budget.wrap(nvfp4_linear.ScaledFP8Linear, "__call__", "fp8_linear_one_row")
    budget.wrap(nvfp4_linear.ScaledNVFP4Linear, "__call__", "nvfp4_linear_one_row")
    budget.wrap(nn.RMSNorm, "__call__", "rms_norm")
    budget.wrap(nn.Embedding, "__call__", "embedding")
    budget.wrap(nemotron_h.Model, "_row_exact_head", "head_other")
    budget.wrap(nemotron_h.Model, "logits", "head_other")
    if draft is not None:
        layer_class = type(draft.layers[0])
        budget.wrap(layer_class, "__call__", "draft_layer")
        budget.wrap(type(draft), "_logits", "draft_lm_head")
        budget.wrap(type(draft.lilicorr), "__call__", "draft_lattice")
        budget.wrap(type(draft), "draft_distributions", "draft_host_and_other")


def _median_wall(setup, run, reps):
    import mlx.core as mx

    samples = []
    for _ in range(reps):
        state = setup()
        mx.synchronize()
        started = time.perf_counter()
        run(state)
        mx.synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def unit(out, args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    policy = {**check.EXACT_POLICY, "draft_model": str(LILICORR)}
    report = {
        "schema": "mlx2.nemotron-round-budget.v1", "checkpoint": check.checkpoint(),
        "source": check.source(), "mlx": mx.__version__, "policy": policy,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MLX2_")},
        "qualified": False,
        "evidence_kind": "attribution budget with mx.eval barriers, no thermal hold",
        "contexts": {},
    }
    adapter = resolve_adapter(check.TARGET)(str(check.TARGET), execution_policy=policy)
    model = adapter.model
    # Serving wraps LiLiCorr in proposal composition; time the backend itself.
    draft = getattr(adapter.draft_model, "backend", adapter.draft_model)
    report["draft_wrapper"] = type(adapter.draft_model).__name__
    layers = tuple(draft.config.target_layer_ids)
    budget = Budget()
    _install(budget, model, draft)
    try:
        for context in args.contexts:
            ids = check._prompt(adapter.tokenizer, context)
            cache = model.make_cache()
            taps = model.prefill_body(mx.array([ids[:-1]]), cache, layers)
            mx.eval(taps, [entry.state for entry in cache])
            shapes = profile._tree_shapes(model, cache, ids[-1])
            chain = shapes["chain8"][0]
            draft_cache = draft.make_cache()
            # Prompt context into the drafter (its pending-at-prefill law).
            draft.draft_distributions([ids[-1]], taps, draft_cache, 5, [None], [0.0],
                                      processor_histories=[ids[:-1]])
            mx.eval([entry.state for entry in draft_cache if hasattr(entry, "state")])
            tail = taps[:, -5:]
            row = {}

            def decode_setup():
                return profile._fresh(cache)

            def decode_run(private):
                logits, feats = model.forward_with_taps(mx.array([[ids[-1]]]), private, layers)
                mx.eval(logits, feats)

            def chain_setup(rows):
                return lambda: HybridVerifyRows([profile._fresh(cache)]).begin([rows])

            def chain_run(rows):
                def run(transaction):
                    logits, feats = model.forward_with_taps(
                        mx.array([chain[:rows]]), transaction.caches, layers
                    )
                    mx.eval(logits, feats)
                return run

            def draft_setup():
                private = copy.deepcopy(draft_cache)
                mx.eval([e.state for e in private if hasattr(e, "state")])
                return private

            def draft_run(private):
                draft.draft_distributions([chain[0]], tail, private, 5, [None], [0.0],
                                          processor_histories=[ids])

            # Does building the verify graph block on the GPU?  Build it once
            # with an idle GPU and once behind ~40 ms of queued dummy work,
            # counting Python-level eval/synchronize calls inside the build.
            def build_probe(busy):
                transaction = chain_setup(6)()
                mx.synchronize()
                if busy:
                    a = mx.random.normal((4096, 4096))
                    for _ in range(24):
                        a = (a @ a) * 1e-4
                    mx.async_eval(a)
                counts = {"eval": 0, "synchronize": 0, "async_eval": 0}
                saved = {name: getattr(mx, name) for name in counts}

                def counter(name):
                    def call(*a, **k):
                        counts[name] += 1
                        return saved[name](*a, **k)
                    return call

                for name in counts:
                    setattr(mx, name, counter(name))
                try:
                    started = time.perf_counter()
                    logits, feats = model.forward_with_taps(
                        mx.array([chain[:6]]), transaction.caches, layers
                    )
                    built = time.perf_counter() - started
                finally:
                    for name, value in saved.items():
                        setattr(mx, name, value)
                mx.eval(logits, feats)
                return {"build_ms": round(built * 1e3, 2), "python_syncs": counts}

            for _ in range(2):
                build_probe(False)
            row_probe = {"idle": build_probe(False), "busy": build_probe(True)}
            print(json.dumps({"context": context, "build_probe": row_probe}), flush=True)

            arms = {
                "decode": (decode_setup, decode_run),
                "chain6": (chain_setup(6), chain_run(6)),
                "chain8": (chain_setup(8), chain_run(8)),
                "draft5": (draft_setup, draft_run),
            }
            for name, (setup, run) in arms.items():
                for _ in range(2):
                    run(setup())
                wall = _median_wall(setup, run, args.reps)
                splits = []
                for _ in range(3):
                    state = setup()
                    budget.reset()
                    budget.active = True
                    mx.synchronize()
                    started = time.perf_counter()
                    run(state)
                    mx.synchronize()
                    total = time.perf_counter() - started
                    budget.active = False
                    splits.append((total, dict(budget.seconds), dict(budget.calls), budget.barriers))
                total, seconds, calls, barriers = sorted(splits, key=lambda item: item[0])[1]
                row[name] = {
                    "wall_ms": round(wall * 1e3, 2),
                    "instrumented_ms": round(total * 1e3, 2),
                    "barriers": barriers,
                    "split_ms": {k: round(v * 1e3, 2) for k, v in
                                 sorted(seconds.items(), key=lambda kv: -kv[1])},
                    "calls": calls,
                    "unattributed_ms": round((total - sum(seconds.values())) * 1e3, 2),
                }
                print(json.dumps({"context": context, "arm": name, "wall_ms": row[name]["wall_ms"],
                                  "instrumented_ms": row[name]["instrumented_ms"],
                                  "barriers": barriers}), flush=True)
            row["build_probe"] = row_probe
            report["contexts"][str(context)] = row
            check.atomic(out / "report.json", report)
            del cache, draft_cache
            mx.clear_cache()
    finally:
        budget.restore()
    adapter.close()
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-round-budget-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--contexts", *map(str, args.contexts), "--reps", str(args.reps)]
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
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024])
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
