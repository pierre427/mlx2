"""Inkling 2 sink-path decode sweep: concat vs sink slot vs sink SDPA, by context.

The sweep picks the balance point for ``inkling2_sink_sdpa`` and
``inkling2_sink_slot``. For each context it prefills once (512-token chunks;
neither option changes prefill). One cache then decodes greedily in
interleaved blocks: concat, slot, sdpa, concat, slot, sdpa, ... with
``BLOCK`` tokens per block. That gives every arm the same cache depth and
thermal state. It records tok/s per block per arm and the kernel counters.

Exactness is not measured here. The slot is bitwise-checked
(``inkling2_sink_slot_ab.py``), and sink SDPA is near-tie classified by the
ladders. This is a sanity A/B with no thermal hold, not publishable
statistics.

Usage: python scripts/research/inkling2_sink_threshold_sweep.py <model dir> <run dir> [contexts]
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
BLOCK = 16
ROUNDS = 3
ARMS = {"concat": {}, "slot": {"sink_slot": True}, "sdpa": {"sink_sdpa": True}}


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


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    contexts = [int(c) for c in (sys.argv[3] if len(sys.argv) > 3
                                 else "2048,4096,8192,16384,24576,32768,49152").split(",")]
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_sink_threshold_sweep.py", "mlx2": head,
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "interleaved decode-rate sweep (no thermal hold; not publishable statistics)",
               "block": BLOCK, "rounds": ROUNDS, "cells": []}
    out = run_dir / "sink_sweep.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    adapter = Inkling2Adapter(str(model_dir))
    try:
        model = adapter.model
        base = {**model.kernel_policy, "sink_slot": False, "sink_sdpa": False}
        receipt["checkpoint"] = adapter.checkpoint
        receipt["base_kernel_policy"] = base
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            model.configure_kernels(**base)
            cache = model.make_cache()
            t0 = time.perf_counter()
            for a in range(0, n, CHUNK):
                logits = model(mx.array([corpus[a:min(a + CHUNK, n)]]), cache=cache)
                mx.eval(logits)
                mx.clear_cache()
            prefill_s = time.perf_counter() - t0
            token = int(mx.argmax(logits[0, -1]).item())
            rates = {arm: [] for arm in ARMS}
            for _ in range(ROUNDS):
                for arm, extra in ARMS.items():
                    model.configure_kernels(**{**base, **extra})
                    logits = model(mx.array([[token]]), cache=cache)  # warm-up step
                    token = int(mx.argmax(logits[0, -1]).item())
                    mx.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(BLOCK):
                        logits = model(mx.array([[token]]), cache=cache)
                        token = int(mx.argmax(logits[0, -1]).item())
                    mx.synchronize()
                    rates[arm].append(round(BLOCK / (time.perf_counter() - t0), 2))
            med = {arm: statistics.median(v) for arm, v in rates.items()}
            cell = {"context": n, "prefill_s": round(prefill_s, 1), "tok_s": rates,
                    "median_tok_s": med,
                    "gain_vs_concat": {arm: round(med[arm] / med["concat"] - 1, 3)
                                       for arm in ("slot", "sdpa")}}
            receipt["cells"].append(cell)
            print(n, med, cell["gain_vs_concat"], flush=True)
            del cache
            mx.clear_cache()
            save()
        model.configure_kernels(**base)
        receipt["kernel_receipt"] = model.kernel_receipt()
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
