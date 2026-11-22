"""Inkling 2 decode/prefill and chunk drift: BF16 noise or a bug?

The GPU smoke saw max |logit| differences of 1.7 (decode vs prefill, 1400
tokens) and 3.6 (prefill chunk 512 vs 128), with argmax agreeing on every row.
This script repeats both comparisons twice: once with the served BF16 dense
weights, and once with every floating parameter cast to FP32 (experts stay
quantized). For each, it localizes the drift per layer visit on the last row.
BF16 rounding shrinks by orders of magnitude in FP32; a real path
difference does not.

Usage: python scripts/research/inkling2_drift_diag.py <model dir> <run dir>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text_tokens(tokenizer, n):
    parts = []
    for path in sorted((ROOT / "docs").glob("*.md")):
        parts.append(path.read_text())
        ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
        if len(ids) >= n:
            return ids[:n]
    raise SystemExit("not enough doc text")


def run(model, tokens, plan):
    """Logits for the plan's last chunk rows plus per-visit last-row hiddens."""
    import mlx.core as mx

    cache = model.make_cache()
    visits = len(model.model.layer_indices)
    for start, end in plan[:-1]:
        mx.eval(model.model(mx.array([tokens[start:end]]), cache))
    start, end = plan[-1]
    capture = {v: None for v in range(visits)}
    pre = model.model(mx.array([tokens[start:end]]), cache, capture=capture)
    logits = model._project_logits(model.model.norm(pre))
    hidden = mx.stack([capture[v][0, -1].astype(mx.float32) for v in range(visits)])
    mx.eval(logits, hidden)
    return logits[0].astype(mx.float32), hidden


def final_hidden(model, tokens, chunk):
    """Pre-norm final hidden for every position under one prefill chunk size."""
    import mlx.core as mx

    cache = model.make_cache()
    rows = []
    for a in range(0, len(tokens), chunk):
        pre = model.model(mx.array([tokens[a:a + chunk]]), cache)[0].astype(mx.float32)
        mx.eval(pre)
        rows.append(pre)
    return mx.concatenate(rows, axis=0)


def per_position(model, tokens, chunks=(128, 256, 512)):
    """Per-position relative drift of the final hidden versus one full prefill.

    A chunk-boundary semantic difference shows jumps just after each
    boundary that move with the chunk size; rounding drift does not."""
    import mlx.core as mx

    n = len(tokens)
    full = final_hidden(model, tokens, n)
    norm = mx.linalg.norm(full, axis=-1) + 1e-6
    out = {}
    for chunk in chunks:
        rel = (mx.linalg.norm(final_hidden(model, tokens, chunk) - full, axis=-1) / norm).tolist()
        near = [rel[i] for i in range(n) if i >= chunk and (i % chunk) < 8]
        far = [rel[i] for i in range(n) if i >= chunk and (i % chunk) >= 8]
        first = rel[:chunk]
        out[str(chunk)] = {
            "mean_rel_first_chunk": round(sum(first) / len(first), 6),
            "mean_rel_within_8_after_boundary": round(sum(near) / max(len(near), 1), 6),
            "mean_rel_elsewhere_after_first_chunk": round(sum(far) / max(len(far), 1), 6),
            "max_rel": round(max(rel), 6),
            "argmax_position": int(max(range(n), key=rel.__getitem__)),
            "per_position_rel": [round(x, 6) for x in rel],
        }
    return out


def compare(model, tokens):
    import mlx.core as mx

    n = len(tokens)
    full_logits, full_hidden = run(model, tokens, [(0, n)])
    out = {"position_drift_final_hidden": per_position(model, tokens)}
    # Decode: prefill n-8, then 8 one-token steps; compare each step's row.
    cache = model.make_cache()
    mx.eval(model.model(mx.array([tokens[: n - 8]]), cache))
    rows = []
    for i in range(n - 8, n):
        rows.append(model(mx.array([[tokens[i]]]), cache)[0, -1].astype(mx.float32))
    dec = mx.stack(rows)
    ref = full_logits[n - 8 :]
    out["decode_vs_prefill"] = {
        "max_abs_logit_diff": round(mx.abs(dec - ref).max().item(), 5),
        "argmax_agreement": int((dec.argmax(-1) == ref.argmax(-1)).sum().item()),
    }
    # Chunked prefill, last row, with per-visit hidden drift.
    chunk_logits, chunk_hidden = run(model, tokens, [(a, min(a + 128, n)) for a in range(0, n, 128)])
    rel = (mx.linalg.norm(chunk_hidden - full_hidden, axis=-1)
           / (mx.linalg.norm(full_hidden, axis=-1) + 1e-6)).tolist()
    out["chunk128_vs_full_last_row"] = {
        "max_abs_logit_diff": round(mx.abs(chunk_logits[-1] - full_logits[-1]).max().item(), 5),
        "argmax_same": bool(chunk_logits[-1].argmax().item() == full_logits[-1].argmax().item()),
        "per_visit_rel_hidden_diff": [round(x, 6) for x in rel],
    }
    return out


LOOP_PASSES = (1, 2, 3, 4, 6)  # trained count 3; every other count is out of distribution


def sweep(model, tokens):
    """Drift at each loop count (varloop), plus how far each count's
    predictions move from the trained 3-pass model on the same tokens."""
    import mlx.core as mx

    n = len(tokens)
    out, reference = {}, None
    for passes in (3, *[p for p in LOOP_PASSES if p != 3]):
        model.configure_kernels(**{**model.kernel_policy, "loop_passes": passes})
        result = compare(model, tokens)
        result["loop"] = model.loop_receipt()
        logits, _ = run(model, tokens, [(0, n)])
        if reference is None:
            reference = logits
        else:
            result["vs_trained_3_pass"] = {
                "argmax_agreement_fraction": round(
                    (logits.argmax(-1) == reference.argmax(-1)).astype(mx.float32).mean().item(), 4),
                "mean_abs_logit_diff": round(mx.abs(logits - reference).mean().item(), 4),
            }
        out[f"passes{passes}"] = result
    model.configure_kernels(**{**model.kernel_policy, "loop_passes": None})
    return out


def _brief(result):
    def strip(v):
        if isinstance(v, dict):
            return {k: strip(x) for k, x in v.items() if not k.startswith("per_")}
        return v
    return strip(result)


def main(argv=None):
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_drift_diag.py", "mlx2": head,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "diagnostic"}
    adapter = Inkling2Adapter(str(model_dir))
    try:
        receipt["checkpoint"] = adapter.checkpoint
        model = adapter.model
        tokens = text_tokens(adapter.tokenizer, 1400)
        for arm in ("bf16", "fp32_dense"):
            if arm == "fp32_dense":
                model._head_fp32 = None
                model.set_dtype(mx.float32)
            receipt[arm] = sweep(model, tokens)
            print(arm, json.dumps(_brief(receipt[arm])), flush=True)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = run_dir / "drift.json.partial"
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    os.replace(tmp, run_dir / "drift.json")


if __name__ == "__main__":
    main()
