#!/usr/bin/env python3
"""Metal checks for an upstream EXL3 checkpoint on the reference trellis path.

modules  The indexed EXL3 kernels against the host decoder on real tensors:
         dense linears and expert banks at every rate the checkpoint uses,
         1/4/37 rows, plus a corrupted-tile negative control.
model    The whole checkpoint through the Qwen3.6 adapter: greedy chat
         continuations (functional oracle), incremental steps against the
         full teacher-forced forward, every installed trellis module observed
         in the adapter receipt, then the affine reference model teacher-forced
         on the same tokens (top-1 agreement, KL). One model per process
         window: the EXL3 model is released before the reference loads.

Run under the GPU lock. Usage:
  check_exl3_metal.py --i-own-the-gpu --model EXL3 --reference AFFINE --out NEW_DIR
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from types import SimpleNamespace
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402

from check_exl3_checkpoint import _artifact_name  # noqa: E402
from mlx2.runtime.models.exl3 import Exl3DenseLinear, Exl3Linear  # noqa: E402
from mlx2.runtime.models.exl3_reference import effective_weight  # noqa: E402
from mlx2.runtime.trellis.upstream import ShardReader, load_bank, load_linear  # noqa: E402

MODULE_BAR = 1e-2
# Whole-model speed arms: one prefill forward per length (the expert-bank
# NAX path pays a fixed reconstruction cost per call), decode after the first.
SPEED_LAWS = ("reference", "nax", "nax_fused")
PREFILL_LENGTHS = (512, 2048, 4096)
# W8A8 is approximate: max error over scale against the exact fused law.
W8A8_BAR = 6e-2
CONTROL_BAR = 5e-2
ROWS = (1, 4, 37)
HEAD_WINDOW = (124160, 125184)
# Ordinary, distinct prompts (reasoning models degrade on repeated strings).
PROMPTS = (
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain in two sentences why the sky looks blue.",
    "List three prime numbers greater than 50.",
)
NEW_TOKENS = 48


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _err(got, ref):
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def _module_checks(path, artifact, rng):
    plan = artifact["exl3_upstream"]
    read = ShardReader(path, artifact["weight_map"])
    layer = 10
    picks = [k for k in sorted(plan.linears) if f".layers.{layer}." in k] + ["lm_head"]
    rows = []
    for key in picks:
        spec = plan.linears[key]
        t = load_linear(read, spec)
        module = Exl3DenseLinear(**t)
        module.eval()
        lo, hi = HEAD_WINDOW if key == "lm_head" else (0, spec.output_dims)
        w = effective_weight(
            np.asarray(t["trellis"][:, lo // 16 : hi // 16]), np.asarray(t["suh"]),
            np.asarray(t["svh"][lo:hi]), spec.codebook,
        )
        for m in ROWS:
            x = rng.normal(0, 1, (m, spec.input_dims)).astype(np.float32)
            got = module(mx.array(x).astype(mx.bfloat16)).astype(mx.float32)[:, lo:hi]
            ref = x.astype(np.float64) @ w
            rows.append({"module": key, "kind": "dense", "rows": m, "k": spec.bits_per_weight,
                         "max_err_over_scale": _err(got, ref)})
    for key in sorted(plan.banks):
        if f".layers.{layer}." not in key:
            continue
        spec = plan.banks[key]
        b = load_bank(read, spec)
        bank = Exl3Linear(b["trellis"], b["suh"], b["svh"])
        for m in ROWS:
            ids = rng.integers(0, spec.experts, m).astype(np.uint32)
            x = rng.normal(0, 1, (m, spec.input_dims)).astype(np.float32)
            got = bank(mx.array(x).astype(mx.bfloat16), mx.array(ids)).astype(mx.float32)
            ref = np.stack([
                x[r].astype(np.float64) @ effective_weight(
                    np.asarray(b["trellis"][int(e)]), np.asarray(b["suh"][int(e)]),
                    np.asarray(b["svh"][int(e)]), spec.codebook)
                for r, e in enumerate(ids)
            ])
            rows.append({"module": key, "kind": "bank", "rows": m, "k": spec.bits_per_weight,
                         "max_err_over_scale": _err(got, ref)})
    rows_law = _rows_law_checks(read, plan, layer, rng)
    fused_law = _fused_law_checks(read, plan, layer, rng)
    # Negative control: one corrupted tile must show.
    spec = plan.linears[picks[0]]
    t = load_linear(read, spec)
    # Corrupt output tile 0 across every input tile; the output Hadamard
    # spreads it over the first 128-column block, where it is measured.
    bad = np.array(t["trellis"])
    bad[:, 0] ^= 0x5A5A
    x = rng.normal(0, 1, (4, spec.input_dims)).astype(np.float32)
    w = effective_weight(np.asarray(t["trellis"]), np.asarray(t["suh"]), np.asarray(t["svh"]), spec.codebook)
    corrupted = Exl3DenseLinear(mx.array(bad), t["suh"], t["svh"])
    corrupted.eval()
    got = corrupted(mx.array(x)).astype(mx.float32)
    control = _err(np.asarray(got)[:, :128], (x.astype(np.float64) @ w)[:, :128])
    for row in rows:
        row["passed"] = row["max_err_over_scale"] <= MODULE_BAR
    return {"rows": rows, "rows_law": rows_law, "fused_law": fused_law,
            "corrupted_tile_err_over_scale": control,
            "control_detected": control >= CONTROL_BAR, "bar": MODULE_BAR,
            "passed": all(r["passed"] for r in rows) and control >= CONTROL_BAR
            and rows_law["passed"] and fused_law["passed"]}


def _merge_counts(dicts):
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def _router_checks(model, rng):
    """nax_fused keeps the one-launch routing for calls up to the block's
    launch rows and the eager routing above them; the combine sums slots in
    order, so both must agree bitwise, indices in slot order included.
    Runs after the adapter pinned its profile (model modules imported)."""
    from mlx2.runtime.models import qwen3_next as qn
    from mlx2.runtime.models import qwen4_moe_window as router_window

    block = next(m for name, m in model.named_modules() if name.endswith(".layers.10.mlp"))
    rows = []
    for tokens in (1, 2, 4, 16):
        x = mx.array(rng.normal(0, 1, (tokens, block.gate.weight.shape[1])).astype(np.float32)).astype(mx.bfloat16)
        logits = block.gate(x)
        li, ls = router_window.router_topk(logits, top_k=block.top_k, contract=block.moe_routing_contract)
        ei, es = qn._stock_routing(block, logits)
        li, ls = li.reshape(ei.shape), ls.reshape(es.shape)
        one = [qn._stock_routing(block, logits[t : t + 1]) for t in range(tokens)]
        rows.append({
            "tokens": tokens,
            "launch_max_rows": block.__dict__.get("moe_topk_launch_max_rows", 1),
            "launch_vs_eager_indices": bool(mx.array_equal(li, ei).item()),
            "launch_vs_eager_scores": bool(mx.array_equal(ls.view(mx.uint16), es.view(mx.uint16)).item()),
            "eager_vs_one_row": bool(mx.array_equal(ei, mx.concatenate([o[0] for o in one])).item()
                                     and mx.array_equal(es.view(mx.uint16),
                                                        mx.concatenate([o[1] for o in one]).view(mx.uint16)).item()),
        })
    return rows


def _fused_law_checks(read, plan, layer, rng):
    """One-launch dense linears and the fused expert pair against the
    reference law; bitwise row invariance of the fused dense kernel."""
    from mlx2.runtime.models.exl3 import Exl3SwitchGLU

    out = {"accuracy": [], "invariance": [], "switch": []}
    keys = [k for k in sorted(plan.linears) if f".layers.{layer}." in k] + ["lm_head"]
    for key in keys:
        t = load_linear(read, plan.linears[key])
        ref = Exl3DenseLinear(**t)
        fused = Exl3DenseLinear(**t, law="fused")
        ref.eval()
        fused.eval()
        for m in (1, 3, 8, 37):
            x = mx.array(rng.normal(0, 1, (m, ref.input_dims)).astype(np.float32)).astype(mx.bfloat16)
            got = fused(x)
            out["accuracy"].append({"module": key, "rows": m, "max_err_over_scale": _err(
                got.astype(mx.float32), ref(x).astype(mx.float32))})
            alone = mx.concatenate([fused(x[i : i + 1]) for i in range(m)])
            out["invariance"].append({"module": key, "rows": m, "bitwise": bool(
                mx.array_equal(got.view(mx.uint16), alone.view(mx.uint16)).item())})
    root = next(k for k in sorted(plan.banks) if f".layers.{layer}." in k).rsplit(".", 1)[0]
    specs = {p: plan.banks[f"{root}.{p}"] for p in ("gate_proj", "up_proj", "down_proj")}
    tensors = {p: load_bank(read, spec) for p, spec in specs.items()}
    projs = [Exl3Linear(t["trellis"], t["suh"], t["svh"]) for t in tensors.values()]
    # Expert banks on the tensor units at a prefill width (2048 routed slots).
    from mlx2.runtime.models.exl3 import NAX_BANK_MIN_ROWS

    row_projs = [Exl3Linear(t["trellis"], t["suh"], t["svh"], law="rows") for t in tensors.values()]
    ref_switch = Exl3SwitchGLU(*projs, execution="reference", trusted_router=True)
    nax_switch = Exl3SwitchGLU(*row_projs, execution="rows_nax", trusted_router=True)
    ref_switch.eval()
    nax_switch.eval()
    tokens = 256
    x = mx.array(rng.normal(0, 1, (tokens, specs["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
    ids = mx.array(np.stack([rng.choice(specs["gate_proj"].experts, 8, replace=False) for _ in range(tokens)]).astype(np.uint32))
    out["switch_nax"] = {"tokens": tokens, "slots": tokens * 8, "threshold": NAX_BANK_MIN_ROWS,
                         "max_err_over_scale": _err(nax_switch(x, ids).astype(mx.float32), ref_switch(x, ids).astype(mx.float32)),
                         "nax_calls": nax_switch.nax_calls}
    # The sorted-order switch (row-mapped prepare, scatter finish) is, row
    # for row, the arithmetic of three per-projection banks: bitwise.
    from mlx2.runtime.models.activations import swiglu as _swiglu
    from mlx2.runtime.models.exl3 import nax_bank, nax_layout, nax_switch as _nax_switch

    flat_ids = ids.reshape(-1)
    layout = nax_layout(flat_ids, specs["gate_proj"].experts)
    rows = mx.repeat(x, 8, axis=0)
    g = nax_bank(nax_switch.gate_proj, rows, flat_ids, layout)
    u = nax_bank(nax_switch.up_proj, rows, flat_ids, layout)
    banks = nax_bank(nax_switch.down_proj, _swiglu(g, u), flat_ids, layout).astype(mx.bfloat16)
    sorted_out = _nax_switch(nax_switch, x, flat_ids, 8, mx.bfloat16)
    out["switch_nax"]["sorted_bitwise_vs_banks"] = bool(
        mx.array_equal(sorted_out.view(mx.uint16), banks.view(mx.uint16)).item())
    # Fused decode chains: routed experts and a dense SwiGLU MLP.
    from mlx2.runtime.models.exl3 import Exl3FusedMLP

    chain = Exl3SwitchGLU(*projs, execution="fused_chain_nax", trusted_router=True)
    chain.eval()
    out["chain"] = []
    for tokens in (1, 4):
        x = mx.array(rng.normal(0, 1, (tokens, specs["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
        ids = mx.array(np.stack([rng.choice(specs["gate_proj"].experts, 8, replace=False) for _ in range(tokens)]).astype(np.uint32))
        got = chain(x, ids)
        row = {"module": root, "tokens": tokens, "max_err_over_scale": _err(
            got.astype(mx.float32), ref_switch(x, ids).astype(mx.float32))}
        alone = mx.concatenate([chain(x[t : t + 1], ids[t : t + 1]) for t in range(tokens)])
        row["bitwise_vs_one_token"] = bool(mx.array_equal(got.view(mx.uint16), alone.view(mx.uint16)).item())
        out["chain"].append(row)
    out["chain_calls"] = chain.chain_calls
    shared = f"model.language_model.layers.{layer}.mlp.shared_expert"
    mlp_parts = {p: Exl3DenseLinear(**load_linear(read, plan.linears[f"{shared}.{p}"]), law="fused_nax")
                 for p in ("gate_proj", "up_proj", "down_proj")}
    for m in mlp_parts.values():
        m.eval()
    mlp = Exl3FusedMLP(mlp_parts["gate_proj"], mlp_parts["up_proj"], mlp_parts["down_proj"])
    mlp.eval()
    from mlx2.runtime.models.activations import swiglu

    out["fused_mlp"] = []
    for m in (1, 4):
        x = mx.array(rng.normal(0, 1, (m, mlp_parts["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
        got = mlp(x)
        want = mlp_parts["down_proj"](swiglu(mlp_parts["gate_proj"](x), mlp_parts["up_proj"](x)))
        alone = mx.concatenate([mlp(x[i : i + 1]) for i in range(m)])
        out["fused_mlp"].append({"module": shared, "rows": m,
                                 "max_err_over_scale": _err(got.astype(mx.float32), want.astype(mx.float32)),
                                 "bitwise_vs_one_row": bool(mx.array_equal(got.view(mx.uint16), alone.view(mx.uint16)).item())})
    out["fused_mlp_calls"] = mlp.fused_calls
    # Whole decode tail (routed chain, shared expert, shared gate, one
    # combine launch) against the composed stock tail of the same modules.
    import mlx.nn as mlx_nn

    gate_head = mlx_nn.Linear(specs["gate_proj"].input_dims, 1, bias=False)
    gate_head.weight = (gate_head.weight * 4).astype(mx.bfloat16)
    block = SimpleNamespace(shared_expert=mlp, shared_expert_gate=gate_head)
    out["tail"] = []
    for tokens in (1, 4):
        x = mx.array(rng.normal(0, 1, (tokens, specs["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
        ids = mx.array(np.stack([rng.choice(specs["gate_proj"].experts, 8, replace=False) for _ in range(tokens)]).astype(np.uint32))
        scores = mx.softmax(mx.array(rng.normal(0, 1, (tokens, 8)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        got = chain.decode_tail(block, x, ids, scores)
        want = (chain(x, ids) * scores[..., None]).sum(axis=-2) + mx.sigmoid(gate_head(x)) * mlp(x)
        alone = mx.concatenate([chain.decode_tail(block, x[t : t + 1], ids[t : t + 1], scores[t : t + 1])
                                for t in range(tokens)])
        out["tail"].append({"module": root, "tokens": tokens,
                            "max_err_over_scale": _err(got.astype(mx.float32), want.astype(mx.float32)),
                            "bitwise_vs_one_token": bool(mx.array_equal(got.view(mx.uint16), alone.view(mx.uint16)).item())})
    out["tail_calls"] = chain.tail_calls
    out["tail_shared_slot_calls"] = chain.shared_slot_calls
    # Rev 4: the shared expert as extra slots of the routed launches, against
    # the separate shared launches (rev 3) of the same modules.
    out["shared_slots"] = []
    for tokens in (1, 4, 37):
        x = mx.array(rng.normal(0, 1, (tokens, specs["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
        ids = mx.array(np.stack([rng.choice(specs["gate_proj"].experts, 8, replace=False) for _ in range(tokens)]).astype(np.uint32))
        scores = mx.softmax(mx.array(rng.normal(0, 1, (tokens, 8)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        object.__setattr__(chain, "shared_slots", False)
        separate = chain.decode_tail(block, x, ids, scores)
        object.__setattr__(chain, "shared_slots", True)
        merged = chain.decode_tail(block, x, ids, scores)
        alone = mx.concatenate([chain.decode_tail(block, x[t : t + 1], ids[t : t + 1], scores[t : t + 1])
                                for t in range(tokens)])
        out["shared_slots"].append({
            "tokens": tokens,
            "max_err_over_scale_vs_separate": _err(merged.astype(mx.float32), separate.astype(mx.float32)),
            "bitwise_vs_separate": bool(mx.array_equal(merged.view(mx.uint16), separate.view(mx.uint16)).item()),
            "bitwise_vs_one_token": bool(mx.array_equal(merged.view(mx.uint16), alone.view(mx.uint16)).item()),
        })
    out["shared_slot_calls"] = chain.shared_slot_calls - out["tail_shared_slot_calls"]
    # Rev 4: same-input projections in one launch, bitwise each linear's own
    # fused call, and bitwise row-invariant.
    from mlx2.runtime.models.exl3 import fused_group

    out["groups"] = []
    for names in ((f"model.language_model.layers.{layer}.linear_attn.in_proj_qkv",
                   f"model.language_model.layers.{layer}.linear_attn.in_proj_z"),
                  tuple(f"model.language_model.layers.{layer + 1}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj"))):
        if not all(n in plan.linears for n in names):
            continue
        lins = [Exl3DenseLinear(**load_linear(read, plan.linears[n]), law="fused_nax") for n in names]
        for m in (1, 4):
            x = mx.array(rng.normal(0, 1, (m, lins[0].input_dims)).astype(np.float32)).astype(mx.bfloat16)
            got = fused_group(lins, x)
            own = [lin.bank.fused_rows(x, mx.zeros((m,), mx.uint32)) for lin in lins]
            alone = [mx.concatenate(parts) for parts in zip(*[fused_group(lins, x[i : i + 1]) for i in range(m)])]
            out["groups"].append({
                "modules": [n.rsplit(".", 1)[-1] for n in names], "rows": m,
                "bitwise_vs_own": all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()) for a, b in zip(got, own)),
                "bitwise_vs_one_row": all(bool(mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()) for a, b in zip(got, alone)),
            })
    # Sushi's fused pair must refuse rates it was not validated at.
    switch = Exl3SwitchGLU(*projs, execution="fused", trusted_router=True)
    switch.eval()
    x = mx.array(rng.normal(0, 1, (1, specs["gate_proj"].input_dims)).astype(np.float32)).astype(mx.bfloat16)
    ids = mx.array(rng.choice(specs["gate_proj"].experts, 8, replace=False).astype(np.uint32))[None]
    try:
        mx.eval(switch(x, ids))
        out["switch"].append({"module": root, "refused": False})
    except ValueError as exc:
        out["switch"].append({"module": root, "refused": True, "reason": str(exc)})
    # NAX: reconstruction is exact (codebook values are halves) and the
    # reconstruct+matmul law tracks the fused law at prefill widths.
    from mlx2.runtime.models.exl3 import NAX_MIN_ROWS
    from mlx2.runtime.models.exl3_reference import decode_inner

    out["nax"] = []
    for key in keys:
        t = load_linear(read, plan.linears[key])
        nax = Exl3DenseLinear(**t, law="fused_nax")
        nax.eval()
        fused = Exl3DenseLinear(**t, law="fused")
        fused.eval()
        columns = min(1024, nax.output_dims)
        rebuilt = np.asarray(nax.bank.reconstruct(mx.array([0], mx.uint32), 0, columns)[0])
        host = decode_inner(np.asarray(t["trellis"][:, : columns // 16]))
        row = {"module": key, "reconstruct_exact": bool(np.array_equal(rebuilt, host))}
        # The trunk cast rides in the finish launch: bitwise finish + astype.
        zeros = mx.zeros((NAX_MIN_ROWS,), mx.uint32)
        inner = mx.array(rng.normal(0, 1, (NAX_MIN_ROWS, nax.output_dims)).astype(np.float16))
        mapped = nax.bank.finish_map(inner, zeros, mx.arange(NAX_MIN_ROWS, dtype=mx.uint32), NAX_MIN_ROWS, mx.bfloat16)
        row["finish_map_bitwise"] = bool(mx.array_equal(
            mapped.view(mx.uint16), nax.bank.finish(inner, zeros).astype(mx.bfloat16).view(mx.uint16)).item())
        w8a8 = Exl3DenseLinear(**t, law="fused_w8a8")
        w8a8.eval()
        for m in (NAX_MIN_ROWS, 512):
            x = mx.array(rng.normal(0, 1, (m, nax.input_dims)).astype(np.float32)).astype(mx.bfloat16)
            reference = fused(x).astype(mx.float32)
            row[f"rows{m}_err_vs_fused"] = _err(nax(x).astype(mx.float32), reference)
            # Approximate law: recorded against its own looser bar.
            row[f"rows{m}_w8a8_err_vs_fused"] = _err(w8a8(x).astype(mx.float32), reference)
        out["nax"].append(row)
    out["passed"] = (all(r["max_err_over_scale"] <= MODULE_BAR for r in out["accuracy"])
                     and all(r["bitwise"] for r in out["invariance"])
                     and all(r["refused"] for r in out["switch"])
                     and out["switch_nax"]["max_err_over_scale"] <= MODULE_BAR and out["switch_nax"]["nax_calls"] == 1
                     and out["switch_nax"]["sorted_bitwise_vs_banks"]
                     and all(r["finish_map_bitwise"] for r in out["nax"])
                     and all(r["max_err_over_scale"] <= MODULE_BAR and r["bitwise_vs_one_token"] for r in out["chain"])
                     and out["chain_calls"] > 0
                     and all(r["max_err_over_scale"] <= MODULE_BAR and r["bitwise_vs_one_row"] for r in out["fused_mlp"])
                     and out["fused_mlp_calls"] > 0
                     and all(r["max_err_over_scale"] <= MODULE_BAR and r["bitwise_vs_one_token"] for r in out["tail"])
                     and out["tail_calls"] == 7 and out["tail_shared_slot_calls"] == 7
                     and len(out["groups"]) == 4
                     and all(r["bitwise_vs_own"] and r["bitwise_vs_one_row"] for r in out["groups"])
                     and all(r["max_err_over_scale_vs_separate"] <= MODULE_BAR and r["bitwise_vs_one_token"]
                             for r in out["shared_slots"])
                     and out["shared_slot_calls"] == sum(1 + t for t in (1, 4, 37))
                     and all(r["reconstruct_exact"] and r[f"rows{NAX_MIN_ROWS}_err_vs_fused"] <= MODULE_BAR
                             and r["rows512_err_vs_fused"] <= MODULE_BAR
                             and r[f"rows{NAX_MIN_ROWS}_w8a8_err_vs_fused"] <= W8A8_BAR
                             and r["rows512_w8a8_err_vs_fused"] <= W8A8_BAR for r in out["nax"]))
    return out


def _rows_law_checks(read, plan, layer, rng):
    """Rows law against the reference law, and bitwise row invariance.

    Invariance compares the kernel's float32 inner product (before the F16
    finish) of a batched call against one-row calls. Bank ids are drawn from
    two experts so blocks fill at grouping widths, and one row is moved
    through every block slot.
    """
    from mlx2.runtime.models.exl3 import GROUP_MIN_ROWS, ROWS_PER_BLOCK, row_plan

    out = {"accuracy": [], "invariance": []}

    def inner(linear, x, ids):
        prepared = linear.prepare(x, ids)
        return linear._rows(prepared, row_plan(ids, linear.num_experts))

    def single(linear, x, ids):
        return mx.concatenate([inner(linear, x[i : i + 1], ids[i : i + 1]) for i in range(x.shape[0])])

    def record(key, m, batched, alone, note=""):
        # uint32 views: array_equal would treat -0.0 and +0.0 as equal.
        same = mx.array_equal(batched.view(mx.uint32), alone.view(mx.uint32))
        out["invariance"].append({"module": key, "rows": m, "note": note, "bitwise": bool(same.item())})

    dense_key = next(k for k in sorted(plan.linears) if f".layers.{layer}." in k)
    t = load_linear(read, plan.linears[dense_key])
    ref_dense = Exl3DenseLinear(**t)
    rows_dense = Exl3DenseLinear(**t, law="rows")
    ref_dense.eval()
    rows_dense.eval()
    for m in (1, 4, 8, 9, 37, 128):
        x = mx.array(rng.normal(0, 1, (m, ref_dense.input_dims)).astype(np.float32)).astype(mx.bfloat16)
        out["accuracy"].append({"module": dense_key, "rows": m, "max_err_over_scale": _err(
            rows_dense(x).astype(mx.float32), ref_dense(x).astype(mx.float32))})
        ids = mx.zeros((m,), mx.uint32)
        record(dense_key, m, inner(rows_dense.bank, x, ids), single(rows_dense.bank, x, ids))
    bank_key = next(k for k in sorted(plan.banks) if f".layers.{layer}." in k)
    spec = plan.banks[bank_key]
    b = load_bank(read, spec)
    ref_bank = Exl3Linear(b["trellis"], b["suh"], b["svh"])
    rows_bank = Exl3Linear(b["trellis"], b["suh"], b["svh"], law="rows")
    pair = rng.choice(spec.experts, 2, replace=False)
    for m in (8, GROUP_MIN_ROWS - 1, GROUP_MIN_ROWS, 96):
        ids = mx.array(pair[rng.integers(0, 2, m)].astype(np.uint32))
        x = mx.array(rng.normal(0, 1, (m, spec.input_dims)).astype(np.float32)).astype(mx.bfloat16)
        out["accuracy"].append({"module": bank_key, "rows": m, "max_err_over_scale": _err(
            rows_bank(x, ids).astype(mx.float32), ref_bank(x, ids).astype(mx.float32))})
        record(bank_key, m, inner(rows_bank, x, ids), single(rows_bank, x, ids), "two experts")
    # One probe row through every slot of a full block: the probe's expert
    # sorts first, fillers use the other expert, so sorted position == slot.
    probe = mx.array(rng.normal(0, 1, (1, spec.input_dims)).astype(np.float32)).astype(mx.bfloat16)
    low, high = sorted(int(e) for e in pair)
    alone = inner(rows_bank, probe, mx.array([low], mx.uint32))
    for slot in range(ROWS_PER_BLOCK):
        filler = mx.array(rng.normal(0, 1, (GROUP_MIN_ROWS, spec.input_dims)).astype(np.float32)).astype(mx.bfloat16)
        x = mx.concatenate([filler[:slot], probe, filler[slot:]])
        ids = np.full(x.shape[0], high, np.uint32)
        ids[: slot + 1] = low  # the probe and the fillers before it share its expert
        ids = mx.array(ids)
        plan = row_plan(ids, rows_bank.num_experts)
        position = int(np.flatnonzero(np.asarray(plan.order) == slot)[0])
        got = rows_bank._rows(rows_bank.prepare(x, ids), plan)[slot : slot + 1]
        record(bank_key, x.shape[0], got, alone,
               f"probe at sorted position {position} (block slot {position % plan.rows})")
    out["passed"] = all(r["max_err_over_scale"] <= MODULE_BAR for r in out["accuracy"]) and all(
        r["bitwise"] for r in out["invariance"])
    return out


def _encode(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def _forward(model, ids):
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array(ids, mx.uint32)[None], cache=cache).astype(mx.float32)[0]
    mx.eval(logits)
    return logits, cache


def _greedy(model, ids, eos):
    logits, cache = _forward(model, ids)
    out, steps = [], []
    nxt = int(mx.argmax(logits[-1]).item())
    for _ in range(NEW_TOKENS):
        out.append(nxt)
        if nxt in eos:
            break
        row = model(mx.array([[nxt]], mx.uint32), cache=cache).astype(mx.float32)[0, -1]
        mx.eval(row)
        steps.append(np.asarray(row))
        nxt = int(np.argmax(steps[-1]))
    return out, np.stack(steps) if steps else np.zeros((0,))


def _top2(row):
    lp = _log_softmax(row)
    first, second = np.argsort(-lp)[:2]
    return [int(first), int(second)], float(lp[first] - lp[second])


def _flip(position, full_row, step_row):
    """Top-1 disagreement between the full forward and an incremental step.

    Both arms share the prefix (teacher forcing). Margins in nats; the
    near-tie contract's ceiling is 0.5. Only the step arm's continuation
    exists, so this records margins, not a full equivalence verdict.
    """
    full_ids, full_margin = _top2(full_row)
    step_ids, step_margin = _top2(step_row)
    same_pair = set(full_ids) == set(step_ids)
    met = same_pair and full_margin <= 0.5 and step_margin <= 0.5
    return {"position": position, "full_top2": full_ids, "full_margin": full_margin,
            "step_top2": step_ids, "step_margin": step_margin,
            "classification": "near_tie_margins_met" if met else "unclassified_high_margin"}


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _model_checks(path, reference, out, execution):
    from mlx2.adapters.registry import resolve_adapter

    started = time.time()
    adapter = resolve_adapter(path)(path, execution_policy={"exl3_execution": execution})
    load_s = time.time() - started
    model, tokenizer = adapter.model, adapter.tokenizer
    eos = set(tokenizer.eos_token_ids)
    installed = adapter.exl3
    cases = []
    for prompt in PROMPTS:
        ids = _encode(tokenizer, prompt)
        cont, steps = _greedy(model, ids, eos)
        full, _ = _forward(model, ids + cont)
        full = np.asarray(full)
        # Incremental step t predicts out[t+1]; the full forward row is ids+t.
        aligned = full[len(ids) : len(ids) + len(steps)]
        step_top1 = float(np.mean(aligned.argmax(-1) == steps.argmax(-1))) if len(steps) else 1.0
        flips = [_flip(t, aligned[t], steps[t]) for t in range(len(steps))
                 if aligned[t].argmax() != steps[t].argmax()]
        step_err = float(np.abs(aligned - steps).max() / np.abs(aligned).max()) if len(steps) else 0.0
        cases.append({
            "prompt": prompt, "prompt_tokens": len(ids), "continuation": cont,
            "text": tokenizer.decode(cont), "finite": bool(np.isfinite(full).all()),
            "step_vs_full_top1": step_top1, "step_vs_full_max_err_over_scale": step_err,
            "step_vs_full_flips": flips,
            "_logits": full[len(ids) - 1 : len(ids) + len(cont) - 1],
            "_ids": ids,
        })
    router_rows = _router_checks(model, np.random.default_rng(20261105)) if execution == "nax_fused" else []
    diagnostics = adapter.diagnostics().get("exl3", {})
    observed_linears = diagnostics.get("observed_linears", {})
    blocks = diagnostics.get("observed_blocks", [])
    observed = {
        "installed_linears": installed["installed_linears"],
        "installed_switches": installed["installed_switches"],
        "linears_called": observed_linears.get("called"),
        "switches_called": sum(1 for b in blocks if b.get(
            "reference_calls" if execution == "reference" else "rows_calls") or b.get("chain_calls")),
        "fused_calls": sum(b.get("fused_calls", 0) for b in blocks),
        "chain_calls": sum(b.get("chain_calls", 0) for b in blocks),
        "tail_calls": sum(b.get("tail_calls", 0) for b in blocks),
        "tail_declines": _merge_counts(b.get("tail_declines", {}) for b in blocks),
        "router_launch_calls": sum(
            getattr(m, "moe_topk_calls", {}).get("launch", 0) for _, m in model.named_modules()
            if isinstance(getattr(m, "moe_topk_calls", None), dict)),
        "laws_called": observed_linears.get("laws_called"),
        "skipped_affine_moe_defaults": getattr(adapter, "skipped_affine_moe_defaults", {}),
        "descriptor_weight_format": adapter.descriptor.metadata.get("weight_format"),
        "identity_fingerprint": adapter.identity["fingerprint"],
        "load_dtype_check": getattr(adapter, "load_dtype_check", None),
        "stored_half_cast": installed.get("stored_half_cast"),
        "router": router_rows,
        "projection_groups_installed": installed.get("projection_groups"),
        "projection_group_calls": sum(
            m.exl3_projection_group.calls for _, m in model.named_modules()
            if getattr(m, "exl3_projection_group", None) is not None),
    }
    peak = mx.get_peak_memory()
    del model, adapter
    gc.collect()
    mx.clear_cache()
    # The reference runs the same adapter class under another profile; a
    # second adapter in this process would see stale import-time flags.
    handoff = out / "handoff"
    handoff.mkdir()
    np.savez(handoff / "cases.npz", **{
        f"ids{i}": np.array(c["_ids"] + c["continuation"], np.int64) for i, c in enumerate(cases)
    }, **{f"start{i}": np.array(len(c["_ids"])) for i, c in enumerate(cases)},
       vocab=np.array(tokenizer.vocab_size))
    subprocess.run([sys.executable, __file__, "--i-own-the-gpu", "--parts", "reference",
                    "--model", str(path), "--reference", str(reference), "--out", str(handoff)],
                   check=True)
    refs = np.load(handoff / "reference.npz")
    for i, case in enumerate(cases):
        if not bool(refs[f"prompt_match{i}"]):
            raise SystemExit("reference tokenizer encodes the prompt differently")
        ref = refs[f"logits{i}"]
        case["reference_control_step_vs_full_max_err_over_scale"] = float(refs[f"control_err{i}"])
        case["reference_control_step_vs_full_flips"] = json.loads(str(refs[f"control_flips{i}"]))
        mine = case.pop("_logits")
        case.pop("_ids")
        lp, lq = _log_softmax(ref), _log_softmax(mine)
        case["reference_top1_agreement"] = float(np.mean(ref.argmax(-1) == mine.argmax(-1)))
        kl = (np.exp(lp) * (lp - lq)).sum(-1)
        case["kl_reference_to_exl3_mean"] = float(kl.mean())
        case["kl_reference_to_exl3_max"] = float(kl.max())
    for item in handoff.iterdir():
        item.unlink()
    handoff.rmdir()
    passed = (
        all(c["finite"] for c in cases)
        and observed["linears_called"] == observed["installed_linears"]
        and observed["switches_called"] == observed["installed_switches"]
        and observed["fused_calls"] == 0
        and observed["laws_called"] == [{"nax": "fused_nax", "nax_fused": "fused_nax"}.get(execution, execution)]
        and (execution != "nax_fused" or (observed["chain_calls"] > 0 and observed["tail_calls"] > 0
                                          and observed["router_launch_calls"] > 0
                                          and observed["projection_groups_installed"] == 40
                                          and observed["projection_group_calls"] > 0
                                          and all(r["launch_vs_eager_indices"] and r["launch_vs_eager_scores"]
                                                  and r["eager_vs_one_row"] for r in observed["router"])))
        and (observed["load_dtype_check"] or {}).get("ok") is True
        and all(f["classification"] == "near_tie_margins_met"
                for c in cases for f in c["step_vs_full_flips"])
    )
    return {"load_seconds": round(load_s, 1), "peak_memory_bytes": peak,
            "observed": observed, "cases": cases, "passed": passed}


def _timed(fn, reps=5):
    mx.eval(fn())
    samples = []
    for _ in range(reps):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _speed_checks(path, artifact, rng, laws=SPEED_LAWS, lengths=PREFILL_LENGTHS, chunk=0):
    """Module and whole-model timings for both laws (not a performance claim:
    no thermal gating, one host, sanity-level medians)."""
    from mlx2.adapters.registry import resolve_adapter

    plan = artifact["exl3_upstream"]
    read = ShardReader(path, artifact["weight_map"])
    out = {"modules": [], "model": []}
    picks = [next(k for k in sorted(plan.linears) if k.endswith("10.self_attn.q_proj")
                  or k.endswith("11.self_attn.q_proj")), "lm_head"]
    for key in picks:
        t = load_linear(read, plan.linears[key])
        from mlx2.runtime.models.exl3 import row_plan

        for law in ("reference", "rows", "fused", "fused_nax", "fused_w8a8"):
            module = Exl3DenseLinear(**t, law=law)
            module.eval()
            for m in (1, 8, 64, 512):
                x = mx.array(rng.normal(0, 1, (m, module.input_dims)).astype(np.float32)).astype(mx.bfloat16)
                out["modules"].append({"module": key, "law": law, "rows": m,
                                       "seconds": _timed(lambda: module(x))})
                if law == "rows" and m > 1:
                    # Block-size study: R does not change a row's arithmetic.
                    bank, ids = module.bank, mx.zeros((m,), mx.uint32)
                    for size in (4,):
                        study = row_plan(ids, 1, rows=size)
                        out["modules"].append({"module": key, "law": f"rows-r{size}", "rows": m,
                            "seconds": _timed(lambda: bank.finish(bank._rows(bank.prepare(x, ids), study), ids))})
    bank_spec = plan.banks[next(k for k in sorted(plan.banks) if ".layers.10." in k)]
    b = load_bank(read, bank_spec)
    for law in ("reference", "rows"):
        bank = Exl3Linear(b["trellis"], b["suh"], b["svh"], law=law)
        for m in (8, 64, 512, 4096):
            ids = mx.array(rng.integers(0, bank_spec.experts, m).astype(np.uint32))
            x = mx.array(rng.normal(0, 1, (m, bank.input_dims)).astype(np.float32)).astype(mx.bfloat16)
            out["modules"].append({"module": bank_spec.key, "law": law, "rows": m,
                                   "seconds": _timed(lambda: bank(x, ids))})
    gc.collect()
    mx.clear_cache()
    for law in laws:
        adapter = resolve_adapter(path)(path, execution_policy={"exl3_execution": law})
        # Model modules import only after the adapter pinned its profile.
        from mlx2.runtime.models.cache import make_prompt_cache

        model = adapter.model
        for length in lengths:
            prompt = rng.integers(1000, 100000, length).tolist()

            def prefill(model):
                # chunk > 0: the server's shape, one forward per chunk with
                # each chunk evaluated before the next (0: one forward).
                cache = make_prompt_cache(model)
                step = chunk or length
                for lo in range(0, length, step):
                    logits = model(mx.array(prompt[lo : lo + step], mx.uint32)[None], cache=cache)
                    if lo + step < length:
                        mx.eval(logits, [c.state for c in cache])
                return logits, cache

            mx.eval(prefill(model)[0])
            mx.synchronize()
            start = time.perf_counter()
            logits, cache = prefill(model)
            mx.eval(logits)
            mx.synchronize()
            prefill_s = time.perf_counter() - start
            row = {"law": law, "prefill_tokens": length, "prefill_seconds": prefill_s,
                   "prefill_tok_s": length / prefill_s}
            if chunk:
                row["prefill_chunk"] = chunk
            if length == lengths[0]:
                token = int(mx.argmax(logits[0, -1]).item())
                mx.synchronize()
                start = time.perf_counter()
                for _ in range(64):
                    step = model(mx.array([[token]], mx.uint32), cache=cache)
                    token = int(mx.argmax(step[0, -1]).item())
                row.update(decode_tokens=64, decode_tok_s=64 / (time.perf_counter() - start))
            out["model"].append(row)
            del logits, cache
            gc.collect()
            mx.clear_cache()
        del model, adapter
        gc.collect()
        mx.clear_cache()
    out["note"] = "sanity timings, not thermally gated; not a published performance figure"
    out["passed"] = True
    return out


def _reference_worker(reference, handoff):
    """Teacher-force the reference on the handed-off tokens (own process)."""
    from mlx2.adapters.registry import resolve_adapter

    cases = np.load(handoff / "cases.npz")
    adapter = resolve_adapter(reference)(reference)
    if adapter.tokenizer.vocab_size != int(cases["vocab"]):
        raise SystemExit("reference tokenizer has a different vocabulary")
    result = {}
    for i, prompt in enumerate(PROMPTS):
        ids, start = cases[f"ids{i}"].tolist(), int(cases[f"start{i}"])
        result[f"prompt_match{i}"] = np.array(_encode(adapter.tokenizer, prompt) == ids[:start])
        logits, _ = _forward(adapter.model, ids)
        result[f"logits{i}"] = np.asarray(logits)[start - 1 : len(ids) - 1]
        # Same-shape control: the reference's own step-versus-full drift on
        # its own greedy continuation (attribution, not a gate).
        prompt_ids = ids[:start]
        cont, steps = _greedy(adapter.model, prompt_ids, set(adapter.tokenizer.eos_token_ids))
        full = np.asarray(_forward(adapter.model, prompt_ids + cont)[0])
        aligned = full[start : start + len(steps)]
        result[f"control_err{i}"] = np.array(
            float(np.abs(aligned - steps).max() / np.abs(aligned).max()) if len(steps) else 0.0)
        flips = [_flip(t, aligned[t], steps[t]) for t in range(len(steps))
                 if aligned[t].argmax() != steps[t].argmax()]
        result[f"control_flips{i}"] = np.array(json.dumps(flips))
    np.savez(handoff / "reference.npz", **result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts", default="modules,model")
    parser.add_argument("--speed-laws", default=",".join(SPEED_LAWS))
    parser.add_argument("--prefill-lengths", default=",".join(map(str, PREFILL_LENGTHS)))
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--execution", default="reference",
                        choices=("reference", "rows", "fused", "fused_nax", "fused_w8a8", "nax", "nax_fused"))
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("this runs Metal kernels; pass --i-own-the-gpu under the GPU lock")
    out = Path(args.out)
    if args.parts == "reference":
        return _reference_worker(Path(args.reference).expanduser().resolve(), out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {out}")
    from mlx2.adapters.registry import inspect_model

    path = Path(args.model).expanduser().resolve()
    reference = Path(args.reference).expanduser().resolve()
    artifact = inspect_model(path).artifact
    if artifact.get("exl3_upstream") is None:
        raise SystemExit("not an upstream EXL3 checkpoint")
    receipt = {
        "schema": "mlx2.exl3-upstream-metal-check.v1",
        "checkpoint": f"{_artifact_name(path)}@{artifact['identity']['fingerprint']}",
        "reference_checkpoint": (
            f"{_artifact_name(reference)}@{inspect_model(reference).artifact['identity']['fingerprint']}"
        ),
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "device": str(mx.default_device()),
        "execution": args.execution,
    }
    parts = set(args.parts.split(","))
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    if "modules" in parts:
        receipt["modules"] = _module_checks(path, artifact, rng)
        print("modules passed", receipt["modules"]["passed"],
              "max err", max(r["max_err_over_scale"] for r in receipt["modules"]["rows"]),
              "control", receipt["modules"]["corrupted_tile_err_over_scale"], flush=True)
        gc.collect()
        mx.clear_cache()
    if "speed" in parts:
        receipt["speed"] = _speed_checks(
            path, artifact, rng, tuple(args.speed_laws.split(",")),
            tuple(int(n) for n in args.prefill_lengths.split(",")), args.prefill_chunk)
        for row in receipt["speed"]["model"]:
            print("speed", json.dumps(row, sort_keys=True), flush=True)
        gc.collect()
        mx.clear_cache()
    if "model" in parts:
        receipt["model"] = _model_checks(path, reference, out, args.execution)
        for case in receipt["model"]["cases"]:
            print(repr(case["text"][:120]), "| ref top1 %.3f KL %.4f | step top1 %.3f" % (
                case["reference_top1_agreement"], case["kl_reference_to_exl3_mean"],
                case["step_vs_full_top1"]), flush=True)
        print("observed", json.dumps(receipt["model"]["observed"], sort_keys=True), flush=True)
    receipt["passed"] = all(receipt[p]["passed"] for p in ("modules", "speed", "model") if p in receipt)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".receipt.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=float) + "\n")
    tmp.replace(out / "receipt.json")
    print("passed", receipt["passed"], "sha256",
          hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest())
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
