#!/usr/bin/env python3
"""B1 decode kernel microbenchmark for upstream EXL3 on Metal (sanity timings).

Per module: FUSED_LINEAR at input splits KS = 1, 2, 4 (time per call from 64
back-to-back calls in one evaluation, accuracy against KS = 1, bitwise row
invariance of each KS at 1 vs 4 rows). Per MoE tail: shared expert as extra
slots vs separate launches. Whole model (``--model-decode``): B1 decode tok/s
for each (FUSED_KS, shared_slots) arm, toggled on one loaded nax_fused model.
Not thermally gated; not a published performance figure.

Run under the GPU lock chain:
  bench_exl3_decode_kernels.py --i-own-the-gpu --model DIR --out NEW_DIR [--model-decode]
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

from check_exl3_checkpoint import _artifact_name  # noqa: E402

CALLS = 64
# (FUSED_KS, shared_slots, FUSED_KS_BY_INPUT) per whole-model arm, interleaved
# so drift lands on every arm; --arms replaces it.
# (FUSED_KS, shared_slots, FUSED_KS_BY_INPUT, projection groups, residual
# folds, in-kernel shared gate logit)
ARMS = tuple(arm for _ in range(3) for arm in (
    (2, True, {}, True, False, False), (2, True, {}, True, True, False)))
SPLITS = (1, 2, 4)


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _err(got, ref):
    got, ref = np.asarray(got.astype(mx.float32), np.float64), np.asarray(ref.astype(mx.float32), np.float64)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def _per_call(fn, calls=CALLS, reps=5):
    mx.eval([fn() for _ in range(4)])
    samples = []
    for _ in range(reps):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval([fn() for _ in range(calls)])
        mx.synchronize()
        samples.append((time.perf_counter() - start) / calls)
    return float(np.median(samples))


def _modules(path, artifact, rng):
    from mlx2.runtime.models import exl3
    from mlx2.runtime.trellis.upstream import ShardReader, load_bank, load_linear

    plan = artifact["exl3_upstream"]
    read = ShardReader(path, artifact["weight_map"])
    keys = [k for k in sorted(plan.linears)
            if (".layers.10." in k or ".layers.11." in k) and ".mlp.shared_expert." not in k] + ["lm_head"]
    rows = []
    for key in keys:
        bank = exl3.Exl3DenseLinear(**load_linear(read, plan.linears[key]), law="fused").bank
        x1 = mx.array(rng.normal(0, 1, (1, bank.input_dims)).astype(np.float32)).astype(mx.bfloat16)
        x4 = mx.array(rng.normal(0, 1, (4, bank.input_dims)).astype(np.float32)).astype(mx.bfloat16)
        z1, z4 = mx.zeros((1,), mx.uint32), mx.zeros((4,), mx.uint32)
        base = bank.fused_rows(x1, z1, ks=1)
        for ks in SPLITS:
            admitted = exl3.fused_ksplit(bank.input_dims, ks)
            if admitted != ks:
                continue
            got = bank.fused_rows(x4, z4, ks=ks)
            alone = mx.concatenate([bank.fused_rows(x4[i : i + 1], z1, ks=ks) for i in range(4)])
            rows.append({
                "module": key, "in": bank.input_dims, "out": bank.output_dims, "ks": ks,
                "us_per_call": 1e6 * _per_call(lambda: bank.fused_rows(x1, z1, ks=ks)),
                "max_err_over_scale_vs_ks1": _err(bank.fused_rows(x1, z1, ks=ks), base),
                "bitwise_vs_one_row": bool(mx.array_equal(got.view(mx.uint16), alone.view(mx.uint16)).item()),
            })
            print("module", json.dumps(rows[-1]), flush=True)
        bank = None  # rebind, not del: the timing lambda above references it
        gc.collect()
    # MoE tail at B1: routed down per slot, and shared slots vs separate.
    root = next(k for k in sorted(plan.banks) if ".layers.10." in k).rsplit(".", 1)[0]
    projs = [exl3.Exl3Linear(**{k: v for k, v in load_bank(read, plan.banks[f"{root}.{p}"]).items()
                                if k in ("trellis", "suh", "svh")}) for p in ("gate_proj", "up_proj", "down_proj")]
    chain = exl3.Exl3SwitchGLU(*projs, execution="fused_chain_nax", trusted_router=True)
    chain.eval()
    shared = "model.language_model.layers.10.mlp.shared_expert"
    parts = {p: exl3.Exl3DenseLinear(**load_linear(read, plan.linears[f"{shared}.{p}"]), law="fused_nax")
             for p in ("gate_proj", "up_proj", "down_proj")}
    for part in parts.values():
        part.eval()
    mlp = exl3.Exl3FusedMLP(parts["gate_proj"], parts["up_proj"], parts["down_proj"])
    mlp.eval()
    import mlx.nn as nn

    gate_head = nn.Linear(projs[0].input_dims, 1, bias=False)
    gate_head.weight = gate_head.weight.astype(mx.bfloat16)
    block = SimpleNamespace(shared_expert=mlp, shared_expert_gate=gate_head)
    x = mx.array(rng.normal(0, 1, (1, projs[0].input_dims)).astype(np.float32)).astype(mx.bfloat16)
    ids = mx.array(rng.choice(projs[0].num_experts, 8, replace=False).astype(np.uint32))[None]
    scores = mx.softmax(mx.array(rng.normal(0, 1, (1, 8)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
    tails = []
    for ks in SPLITS:
        for slots in (False, True):
            exl3.FUSED_KS, exl3.FUSED_KS_BY_INPUT = ks, {}
            object.__setattr__(chain, "shared_slots", slots)
            tails.append({"ks": ks, "shared_slots": slots,
                          "us_per_call": 1e6 * _per_call(lambda: chain.decode_tail(block, x, ids, scores))})
            print("tail", json.dumps(tails[-1]), flush=True)
    exl3.FUSED_KS = 2
    return {"modules": rows, "tail": tails}


def _model_decode(path, rng, arms_spec=None):
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(path)(path, execution_policy={"exl3_execution": "nax_fused"})
    from mlx2.runtime.models import exl3
    from mlx2.runtime.models.cache import make_prompt_cache

    model = adapter.model
    switches = [m for _, m in model.named_modules() if isinstance(m, exl3.Exl3SwitchGLU)]
    prompt = rng.integers(1000, 100000, 512).tolist()
    arms = []
    arms_spec = ARMS if arms_spec is None else arms_spec
    grouped = {name: m.exl3_projection_group for name, m in model.named_modules()
               if getattr(m, "exl3_projection_group", None) is not None}
    modules = dict(model.named_modules())
    folded = {name: m.residual_slots for name, m in model.named_modules()
              if getattr(m, "residual_slots", None) is not None}
    for ks, slots, by_input, groups, residual, logit in arms_spec:
        exl3.FUSED_KS = ks
        exl3.FUSED_KS_BY_INPUT = dict(by_input)
        for name, group in grouped.items():
            object.__setattr__(modules[name], "exl3_projection_group", group if groups else None)
        for name, pair in folded.items():
            object.__setattr__(modules[name], "residual_slots", pair if residual else None)
        for sw in switches:
            object.__setattr__(sw, "shared_gate_logit", logit)
        for sw in switches:
            object.__setattr__(sw, "shared_slots", slots)
        cache = make_prompt_cache(model)
        logits = model(mx.array(prompt, mx.uint32)[None], cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        for _ in range(8):  # warm the decode kernels at this arm
            token = int(mx.argmax(model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]).item())
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(96):
            token = int(mx.argmax(model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]).item())
        arms.append({"ks": ks, "ks_by_input": {str(k): v for k, v in by_input.items()}, "shared_slots": slots,
                     "projection_groups": groups, "groups_installed": len(grouped),
                     "residual_folds": residual, "folds_installed": len(folded), "gate_logit": logit,
                     "decode_tok_s": 96 / (time.perf_counter() - start)})
        print("model", json.dumps(arms[-1]), flush=True)
        del cache, logits
        gc.collect()
        mx.clear_cache()
    exl3.FUSED_KS, exl3.FUSED_KS_BY_INPUT = 2, {}
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-decode", action="store_true")
    parser.add_argument("--model-only", action="store_true", help="skip the module timings")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("this runs Metal kernels; pass --i-own-the-gpu under the GPU lock")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {out}")
    from mlx2.adapters.registry import inspect_model

    path = Path(args.model).expanduser().resolve()
    artifact = inspect_model(path).artifact
    receipt = {
        "schema": "mlx2.exl3-decode-kernel-bench.v1",
        "checkpoint": f"{_artifact_name(path)}@{artifact['identity']['fingerprint']}",
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "note": "sanity timings, not thermally gated; not a published performance figure",
    }
    rng = np.random.default_rng(20261105)
    # Model modules import after the adapter pins its profile, so the whole
    # model arm runs first and the module arm reuses its imports.
    if args.model_decode:
        receipt["model"] = _model_decode(path, rng)
        gc.collect()
        mx.clear_cache()
    if not args.model_only:
        receipt.update(_modules(path, artifact, rng))
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print("wrote", out / "receipt.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
