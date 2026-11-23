"""Inkling 2 ``inkling2_kv_q8``: decode rate, logit gap and KV bytes on the GPU.

The adapter loads with the current defaults (``--base defaults``) or the
reference policy (``--base reference``); only ``kv_q8`` is toggled
(``Model.configure_kernels``, everything else kept). For each context one
model carries two caches: A prefilled with Q8 off (BF16 global K/V) and B
with it on (int8 global K/V). Decode then alternates in blocks (A off, B
on, ...), 32 greedy tokens per block. Arm B is fed arm A's tokens, so every
step compares the two arms on the same history. This records, per context:

- prefill seconds, tok/s and peak memory per arm, the last-row logit gap;
- decode tok/s per block and arm;
- per-step logit gap (max |diff|, median), KL(off || on) and argmax
  agreement on the shared history;
- the first step whose argmax differs, with both arms' top-5 log
  probabilities and their selected-versus-alternate margins (input for the
  near-tie classification in AGENTS.md; the functional oracle is not run
  here);
- KV bytes held by the global planes (BF16 vs int8 rows) and by the rings;
- memory after each block and the kernel receipt (``kv_q8`` route counters).

Q8 changes outputs: this measures the approximation, it does not qualify
it. A sanity A/B (no thermal hold), not publishable statistics.

Usage: python scripts/research/inkling2_kv_q8_ab.py <model dir> <run dir>
       [--contexts 16384,65536,131072] [--blocks 2] [--base defaults|reference]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))

from inkling2_fuse_ab import (
    BLOCK,
    GIB,
    Toggle,
    corpus_tokens,
    decode_block,
    memory,
    prefill,
)


def top5(row):
    import mlx.core as mx

    logp = row - mx.logsumexp(row)
    idx = mx.argsort(-logp)[:5]
    return [[int(i), round(float(logp[i].item()), 5)] for i in idx.tolist()]


def step_stats(row_a, row_b):
    import mlx.core as mx

    la = row_a - mx.logsumexp(row_a)
    lb = row_b - mx.logsumexp(row_b)
    return {"max_abs": float(mx.abs(row_a - row_b).max().item()),
            "kl": float((mx.exp(la) * (la - lb)).sum().item())}


def plane_bytes(cache):
    from mlx2.runtime.models.inkling2 import kv_q8

    planes = cache[0::2]
    held = {"global_int8": 0, "global_float": 0, "rings": 0}
    for plane in planes:
        name = type(plane).__name__
        part = kv_q8.kv_bytes([plane])
        if "Rotating" in name:
            held["rings"] += part["q8"] + part["other"]
        else:
            held["global_int8"] += part["q8"]
            held["global_float"] += part["other"]
    held = {k: round(v / 2**30, 4) for k, v in held.items()}
    held["unit"] = "GiB allocated (KVCache grows in 256-token steps)"
    return held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("run_dir")
    parser.add_argument("--contexts", default="16384,65536,131072")
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--base", choices=("defaults", "reference"), default="defaults")
    args = parser.parse_args()
    model_dir, run_dir = Path(args.model_dir), Path(args.run_dir)
    contexts = [int(c) for c in args.contexts.split(",")]
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import REFERENCE_POLICY, Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"],
                                capture_output=True, text=True).stdout.strip())
    receipt = {"script": "scripts/research/inkling2_kv_q8_ab.py", "mlx2": head,
               "mlx2_src_dirty": dirty, "mlx_version": mx.__version__,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "option": "kv_q8",
               "kind": "approximation + sanity A/B (no thermal hold; not publishable statistics)",
               "fidelity": "approximate", "qualified": False, "block": BLOCK, "cells": []}
    out = run_dir / "kv_q8_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    policy = dict(REFERENCE_POLICY) if args.base == "reference" else {}
    receipt["base"] = args.base
    adapter = Inkling2Adapter(str(model_dir), execution_policy=policy)
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["base_kernel_policy"] = dict(model.kernel_policy)
        toggle = Toggle(model, ["kv_q8"])
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            tokens = corpus[:n]
            mx.reset_peak_memory()
            mx.clear_cache()
            base_a = mx.get_active_memory()
            mx.reset_peak_memory()
            cache_a, last_a, sec_a = prefill(model, toggle, tokens, False)
            peak_a = mx.get_peak_memory() - base_a
            mx.clear_cache()
            base_b = mx.get_active_memory()
            mx.reset_peak_memory()
            cache_b, last_b, sec_b = prefill(model, toggle, tokens, True)
            peak_b = mx.get_peak_memory() - base_b
            cell = {"context": n, "prefill_s": {"off": round(sec_a, 2), "on": round(sec_b, 2)},
                    "prefill_tok_s": {"off": round(n / sec_a, 1), "on": round(n / sec_b, 1)},
                    "prefill_last": step_stats(last_a, last_b),
                    "prefill_last_argmax_equal": int(mx.argmax(last_a).item())
                    == int(mx.argmax(last_b).item()),
                    "kv_bytes": {"off": plane_bytes(cache_a), "on": plane_bytes(cache_b)},
                    "prefill_peak_delta_gib": {"off": round(peak_a / GIB, 2),
                                               "on": round(peak_b / GIB, 2),
                                               "note": "peak over the arm's prefill minus "
                                                       "active memory before it"},
                    "after_prefill": memory(), "blocks": []}
            agree, total, first, gaps, kls = 0, 0, None, [], []
            for i in range(args.blocks):
                rows_a, toks_a, own_a, last_a, rate_a = decode_block(
                    model, toggle, cache_a, last_a, False)
                mem_a = memory()
                rows_b, toks_b, own_b, last_b, rate_b = decode_block(
                    model, toggle, cache_b, last_b, True, forced=toks_a)
                mem_b = memory()
                stats = [step_stats(x, y) for x, y in zip(rows_a, rows_b)]
                gaps += [s["max_abs"] for s in stats]
                kls += [s["kl"] for s in stats]
                equal = [a == b for a, b in zip(own_a, own_b)]
                agree += sum(equal)
                total += len(equal)
                if first is None and not all(equal):
                    j = equal.index(False)
                    off5, on5 = top5(rows_a[j]), top5(rows_b[j])
                    first = {"block": i, "step": j, "token_in": toks_a[j],
                             "off_top5": off5, "on_top5": on5,
                             "off_margin": round(off5[0][1] - off5[1][1], 5),
                             "on_margin": round(on5[0][1] - on5[1][1], 5),
                             "same_top_two": {off5[0][0], off5[1][0]} == {on5[0][0], on5[1][0]},
                             **stats[j]}
                cell["blocks"].append({"block": i, "off_tok_s": rate_a, "on_tok_s": rate_b,
                                       "tokens": toks_a, "on_argmax_equal": sum(equal),
                                       "steps": len(equal),
                                       "max_abs_gap": max(s["max_abs"] for s in stats),
                                       "mem_after_off": mem_a, "mem_after_on": mem_b})
                print(n, {k: v for k, v in cell["blocks"][-1].items() if k != "tokens"},
                      flush=True)
            gaps.sort()
            cell["decode_argmax_agreement"] = f"{agree}/{total}"
            cell["decode_logit_gap"] = {"max_abs": gaps[-1], "median_abs": gaps[len(gaps) // 2],
                                        "kl_mean": sum(kls) / len(kls), "kl_max": max(kls)}
            cell["first_divergence"] = first
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
