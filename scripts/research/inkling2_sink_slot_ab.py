"""Inkling 2 ``inkling2_sink_slot``: exactness, decode rate and buffer pool at long context.

For each context, one model carries two caches. Cache A is prefilled with the
sink slot off and cache B with it on. Decode then alternates in blocks: A off,
B on, A off, B on, 32 greedy tokens per block. This records:

- prefill: last-row logits bitwise equal, prefill seconds per arm;
- decode: every step's logits bitwise equal between the arms, the tokens,
  and tok/s per block;
- memory after each block: active, MLX buffer pool (``get_cache_memory``)
  and peak.

The kernel receipt shows the slot engaged. This is a sanity A/B (no thermal
hold), not publishable statistics.

Usage: python scripts/research/inkling2_sink_slot_ab.py <model dir> <run dir> [contexts]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
BLOCK = 32
GIB = 2**30


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too short")


def memory():
    import mlx.core as mx

    return {"active_gib": round(mx.get_active_memory() / GIB, 2),
            "cache_gib": round(mx.get_cache_memory() / GIB, 2),
            "peak_gib": round(mx.get_peak_memory() / GIB, 2)}


def set_slot(model, on):
    model.configure_kernels(**{**model.kernel_policy, "sink_slot": on})


def prefill(model, tokens, on):
    import mlx.core as mx

    set_slot(model, on)
    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(tokens), CHUNK):
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        mx.clear_cache()  # ordinary serving prefill clears per chunk
    mx.synchronize()
    return cache, logits[0, -1], time.perf_counter() - t0


def decode_block(model, cache, last, on):
    import mlx.core as mx

    set_slot(model, on)
    token = int(mx.argmax(last).item())
    rows, tokens = [], []
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(BLOCK):
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        rows.append(logits)
        token = int(mx.argmax(logits).item())
    mx.synchronize()
    seconds = time.perf_counter() - t0
    return rows, tokens, rows[-1], round(BLOCK / seconds, 2)


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    contexts = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3 else "16384,65536").split(",")]
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import REFERENCE_POLICY, Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_sink_slot_ab.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "chunk": CHUNK, "block": BLOCK, "cells": []}
    out = run_dir / "sink_slot_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir), execution_policy=dict(REFERENCE_POLICY))  # reference path (MoE fast paths on)
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["base_kernel_policy"] = dict(model.kernel_policy)
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            tokens = corpus[:n]
            mx.reset_peak_memory()
            cache_a, last_a, sec_a = prefill(model, tokens, False)
            cache_b, last_b, sec_b = prefill(model, tokens, True)
            cell = {"context": n, "prefill_s": {"off": round(sec_a, 2), "on": round(sec_b, 2)},
                    "prefill_last_logits_bitwise": bool(mx.array_equal(last_a, last_b)),
                    "after_prefill": memory(), "blocks": []}
            equal, tok_equal = True, True
            for i in range(2):
                rows_a, toks_a, last_a, rate_a = decode_block(model, cache_a, last_a, False)
                mem_a = memory()
                rows_b, toks_b, last_b, rate_b = decode_block(model, cache_b, last_b, True)
                mem_b = memory()
                same = all(mx.array_equal(x, y) for x, y in zip(rows_a, rows_b))
                equal &= same
                tok_equal &= toks_a == toks_b
                cell["blocks"].append({"block": i, "off_tok_s": rate_a, "on_tok_s": rate_b,
                                       "logits_bitwise": same, "tokens_equal": toks_a == toks_b,
                                       "mem_after_off": mem_a, "mem_after_on": mem_b})
                print(n, cell["blocks"][-1], flush=True)
            cell["decode_logits_bitwise"] = equal
            cell["decode_tokens_equal"] = tok_equal
            receipt["cells"].append(cell)
            del cache_a, cache_b
            mx.clear_cache()
            save()
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
