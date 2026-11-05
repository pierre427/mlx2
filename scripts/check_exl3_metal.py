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
    diagnostics = adapter.diagnostics().get("exl3", {})
    observed_linears = diagnostics.get("observed_linears", {})
    blocks = diagnostics.get("observed_blocks", [])
    observed = {
        "installed_linears": installed["installed_linears"],
        "installed_switches": installed["installed_switches"],
        "linears_called": observed_linears.get("called"),
        "switches_called": sum(1 for b in blocks if b.get(
            "reference_calls" if execution == "reference" else "rows_calls")),
        "fused_calls": sum(b.get("fused_calls", 0) for b in blocks),
        "laws_called": observed_linears.get("laws_called"),
        "skipped_affine_moe_defaults": getattr(adapter, "skipped_affine_moe_defaults", {}),
        "descriptor_weight_format": adapter.descriptor.metadata.get("weight_format"),
        "identity_fingerprint": adapter.identity["fingerprint"],
        "load_dtype_check": getattr(adapter, "load_dtype_check", None),
        "stored_half_cast": installed.get("stored_half_cast"),
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
        and observed["laws_called"] == [{"nax": "fused_nax"}.get(execution, execution)]
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


def _speed_checks(path, artifact, rng):
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
    prompt = rng.integers(1000, 100000, 512).tolist()
    for law in ("reference", "fused_nax", "nax"):
        adapter = resolve_adapter(path)(path, execution_policy={"exl3_execution": law})
        # Model modules import only after the adapter pinned its profile.
        from mlx2.runtime.models.cache import make_prompt_cache

        model = adapter.model

        def prefill(model):
            cache = make_prompt_cache(model)
            return model(mx.array(prompt, mx.uint32)[None], cache=cache), cache

        prefill(model)
        mx.synchronize()
        start = time.perf_counter()
        logits, cache = prefill(model)
        mx.eval(logits)
        mx.synchronize()
        prefill_s = time.perf_counter() - start
        token = int(mx.argmax(logits[0, -1]).item())
        mx.synchronize()
        start = time.perf_counter()
        for _ in range(64):
            row = model(mx.array([[token]], mx.uint32), cache=cache)
            token = int(mx.argmax(row[0, -1]).item())
        decode_s = time.perf_counter() - start
        out["model"].append({"law": law, "prefill_tokens": 512, "prefill_seconds": prefill_s,
                             "prefill_tok_s": 512 / prefill_s, "decode_tokens": 64,
                             "decode_tok_s": 64 / decode_s})
        del model, adapter, cache, logits
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
    parser.add_argument("--execution", default="reference",
                        choices=("reference", "rows", "fused", "fused_nax", "fused_w8a8", "nax"))
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
        receipt["speed"] = _speed_checks(path, artifact, rng)
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
