#!/usr/bin/env python3
"""Where the tensor units (NAX) could serve EXL3 decode and verify widths.

Per real Qwen3.6-35B-A3B module and width R (rows), the latency of one call,
each call synced on its own (independent back-to-back calls overlap on the
GPU), minus the measured sync floor:
- dense: ``fused`` (FUSED_LINEAR, the law below NAX_MIN_ROWS), ``rows`` (the
  four-row block law: weights decoded once per four rows), ``nax`` (prepare,
  fp16 reconstruction, mx.matmul, finish), ``nax_resident`` (the same with
  the fp16 weights reconstructed once and kept: the ceiling of a resident
  fp16 verify copy, at 4x the trellis bytes);
- MoE tail at T tokens (8 routed + shared): ``chain`` (decode_tail, the law
  below 1024 slots) and ``nax_switch`` (the expert-sorted NAX banks, forced).
Sanity timings, not thermally gated; not a published performance figure.

  survey_exl3_nax_widths.py --i-own-the-gpu --model DIR --out NEW_DIR
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

WIDTHS = (1, 2, 4, 8, 16, 32, 64)
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
REPS = 15


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _latency(fn, reps=REPS):
    mx.eval(fn())
    mx.synchronize()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _err(got, ref):
    got, ref = np.asarray(got.astype(mx.float32), np.float64), np.asarray(ref.astype(mx.float32), np.float64)
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("this runs Metal kernels; pass --i-own-the-gpu under the GPU lock")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence directory {out}")
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.models import exl3
    from mlx2.runtime.trellis.upstream import ShardReader, load_bank, load_linear

    path = Path(args.model).expanduser().resolve()
    artifact = inspect_model(path).artifact
    plan = artifact["exl3_upstream"]
    read = ShardReader(path, artifact["weight_map"])
    rng = np.random.default_rng(20261105)
    receipt = {
        "schema": "mlx2.exl3-nax-width-survey.v1",
        "checkpoint": f"{_artifact_name(path)}@{artifact['identity']['fingerprint']}",
        "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "mlx_version": mx.__version__,
        "note": "sanity timings, not thermally gated; not a published performance figure",
    }
    tiny = mx.zeros((1,))
    receipt["sync_floor_us"] = 1e6 * _latency(lambda: tiny + 1)
    floor = receipt["sync_floor_us"]
    print("sync floor us", round(floor, 1), flush=True)

    dense = []
    keys = ["model.language_model.layers.10.linear_attn.in_proj_qkv",
            "model.language_model.layers.10.linear_attn.out_proj",
            "model.language_model.layers.11.self_attn.q_proj",
            "model.language_model.layers.10.mlp.shared_expert.down_proj",
            "lm_head"]
    for key in keys:
        t = load_linear(read, plan.linears[key])
        fused = exl3.Exl3DenseLinear(**t, law="fused_nax")
        rows_law = exl3.Exl3DenseLinear(**t, law="rows")
        for m in (fused, rows_law):
            m.eval()
        bank = fused.bank
        resident = []
        for lo in range(0, bank.output_dims, exl3.NAX_SLICE_COLUMNS):
            columns = min(exl3.NAX_SLICE_COLUMNS, bank.output_dims - lo)
            resident.append(bank.reconstruct(mx.zeros((1,), mx.uint32), lo // 16, columns)[0])
        mx.eval(resident)

        def nax_resident(flat):
            ids = mx.zeros((flat.shape[0],), mx.uint32)
            prepared = bank.prepare(flat, ids)
            inner = [mx.matmul(prepared, w) for w in resident]
            inner = inner[0] if len(inner) == 1 else mx.concatenate(inner, axis=1)
            return bank.finish_map(inner, ids, mx.arange(flat.shape[0], dtype=mx.uint32), flat.shape[0], flat.dtype)

        for r in WIDTHS:
            x = mx.array(rng.normal(0, 1, (r, bank.input_dims)).astype(np.float32)).astype(mx.bfloat16)
            row = {"module": key, "in": bank.input_dims, "out": bank.output_dims, "rows": r}
            ref = fused._fused(x)
            for name, fn in (("fused", lambda: fused._fused(x)), ("rows", lambda: rows_law(x)),
                             ("nax", lambda: fused._nax(x)), ("nax_resident", lambda: nax_resident(x))):
                row[f"{name}_us"] = 1e6 * _latency(fn) - floor
                if name != "fused":
                    row[f"{name}_err"] = _err(fn(), ref)
            print("dense", json.dumps(row), flush=True)
            dense.append(row)
        # Rebind instead of del: the per-width closures above reference these names.
        fused = rows_law = resident = bank = None
        gc.collect()
        mx.clear_cache()
    receipt["dense"] = dense

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

    head = nn.Linear(projs[0].input_dims, 1, bias=False)
    head.weight = head.weight.astype(mx.bfloat16)
    block = SimpleNamespace(shared_expert=mlp, shared_expert_gate=head)
    moe = []
    for tokens in TOKENS:
        x = mx.array(rng.normal(0, 1, (tokens, projs[0].input_dims)).astype(np.float32)).astype(mx.bfloat16)
        ids = mx.array(np.stack([rng.choice(projs[0].num_experts, 8, replace=False)
                                 for _ in range(tokens)]).astype(np.uint32))
        scores = mx.softmax(mx.array(rng.normal(0, 1, (tokens, 8)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
        flat_ids = ids.reshape(-1)

        def nax_tail():
            y = exl3.nax_switch(chain, x, flat_ids, 8, x.dtype)
            y = (y.reshape(tokens, 8, -1) * scores[..., None]).sum(axis=-2)
            return y + mx.sigmoid(head(x)) * mlp(x)

        row = {"tokens": tokens, "slots": tokens * 8}
        ref = chain.decode_tail(block, x, ids, scores) if tokens * 8 < exl3.NAX_BANK_MIN_ROWS else None
        if ref is not None:
            row["chain_us"] = 1e6 * _latency(lambda: chain.decode_tail(block, x, ids, scores)) - floor
        row["nax_switch_us"] = 1e6 * _latency(nax_tail) - floor
        if ref is not None:
            row["nax_switch_err"] = _err(nax_tail(), ref)
        print("moe", json.dumps(row), flush=True)
        moe.append(row)
    receipt["moe"] = moe
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print("wrote", out / "receipt.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
