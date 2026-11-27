#!/usr/bin/env python3
"""Checkpoint A/B for ``inkling2_prefill_lazy_rescale`` (one bounded GPU unit).

Prefills the same ~64K-token repo-text prompt (2048-token chunks, the adapter's
default kernel policy, so global layers run ``inkling2_prefill_tiled_attention``)
four times, ABBA: eager, lazy, lazy, eager. Checks that every chunk's
last-row logits are bit-identical across all four passes, that the tiled
kernel engaged (``prefill_tiled_counts``), and times each chunk.

Usage: python scripts/research/inkling2_lazy_rescale_ckpt.py <model dir> <run dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
CHUNK = 2048
TOKENS = 65536


def main():
    model_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    import mlx.core as mx

    from inkling2_prefill_profile import corpus_tokens
    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.runtime.models.inkling2 import prefill_attention as pa

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"],
                                capture_output=True, text=True).stdout.strip())
    receipt = {"script": "scripts/research/inkling2_lazy_rescale_ckpt.py", "mlx2_commit": head,
               "src_dirty": dirty, "mlx": mx.__version__, "chunk": CHUNK,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thermal_hold": False,
               "kind": "sanity A/B, one run, not publishable statistics"}
    adapter = Inkling2Adapter(str(model_dir))
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["kernel_policy"] = dict(model.kernel_policy)
        tokens = corpus_tokens(adapter.tokenizer, TOKENS)
        passes = []
        for arm in ("eager", "lazy", "lazy", "eager"):
            pa.set_lazy_rescale(arm == "lazy")
            for _, module in model.named_modules():
                module.__dict__.pop("prefill_tiled_counts", None)
            cache = model.make_cache()
            chunks, digests = [], []
            t_all = time.perf_counter()
            for a in range(0, len(tokens), CHUNK):
                ids = mx.array([tokens[a:a + CHUNK]])
                mx.synchronize()
                t0 = time.perf_counter()
                logits = model(ids, cache=cache)[:, -1, :]
                mx.eval(logits)
                mx.synchronize()
                chunks.append(round((time.perf_counter() - t0) * 1e3, 1))
                digests.append(hashlib.sha256(bytes(logits.astype(mx.float32))).hexdigest()[:16])
            total = round(time.perf_counter() - t_all, 2)
            counts = {}
            for _, module in model.named_modules():
                for k, v in (module.__dict__.get("prefill_tiled_counts") or {}).items():
                    counts[k] = counts.get(k, 0) + v
            passes.append({"arm": arm, "total_s": total, "chunk_ms": chunks, "logit_digests": digests,
                           "prefill_tiled_counts": counts})
            print(json.dumps({"arm": arm, "total_s": total, "tiled": counts,
                              "last_chunk_ms": chunks[-1]}), flush=True)
            del cache
            mx.clear_cache()
        pa.set_lazy_rescale(False)
        receipt["passes"] = passes
        receipt["bitwise_all_passes"] = all(p["logit_digests"] == passes[0]["logit_digests"]
                                            for p in passes)
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = run_dir / "ckpt.json.partial"
    tmp.write_text(json.dumps(receipt, indent=1, default=str))
    os.replace(tmp, run_dir / "ckpt.json")
    print("bitwise_all_passes", receipt["bitwise_all_passes"])


if __name__ == "__main__":
    main()
