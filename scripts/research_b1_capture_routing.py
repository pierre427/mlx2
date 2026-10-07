#!/usr/bin/env python3
"""B1 step 1: real 8K served-chunk prefill on Qwen3.6-35B-A3B; capture routing.

Loads the model through its mlx2 adapter (default policy: NAX MoE gather
``fused``, served prefill chunk 2048), then

1. times the whole 8K prefill end to end, chunk by chunk exactly as the
   BatchGenerator does it (``model(chunk, cache)``, ``mx.eval`` of the cache
   state), with the NAX gather route ``fused`` (served default) and ``off``
   interleaved, >= 5 reps after a warm-up each;
2. runs one more ``fused`` prefill with the switch modules' ``__call__``
   wrapped to keep a reference to each layer's routing ``indices`` (no other
   behaviour change, no extra sync: the arrays are evaluated together with
   the cache after each chunk) and writes them to ``routing.npz``.

Research harness; writes into --out.  Run through scripts/run_with_gpu_locks.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def swapouts() -> int:
    out = subprocess.check_output(["vm_stat"], text=True)
    for line in out.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split(":")[1].strip().rstrip("."))
    return -1


def therm() -> str:
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def corpus_tokens(tokenizer, context):
    paths = sorted((ROOT / "docs").glob("*.md"))
    corpus = "\n\n".join(p.read_text(encoding="utf-8") for p in paths)
    try:
        ids = list(tokenizer.encode(corpus, add_special_tokens=False))
    except TypeError:
        ids = list(tokenizer.encode(corpus))
    stream = ids
    while len(stream) < context:
        stream = stream + ids
    return stream[:context], {
        "corpus": "docs/*.md joined by blank lines (first tokens)",
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
        "corpus_tokens": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import numpy as np

    mx.set_cache_limit(4 << 30)
    swap0 = swapouts()
    therm0 = therm()
    from mlx2.adapters.registry import resolve_adapter

    t0 = time.perf_counter()
    adapter = resolve_adapter(a.model)(a.model)
    load_s = time.perf_counter() - t0
    model = adapter.model
    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models.cache import make_prompt_cache

    served_mode = nax.MODE
    ids, token_meta = corpus_tokens(adapter.tokenizer, a.context)
    tokens = mx.array(ids, dtype=mx.int32)[None]

    layers = model.language_model.model.layers
    switch_of = {}
    bits_of = {}
    switch_types = set()
    for i, layer in enumerate(layers):
        sw = layer.mlp.switch_mlp
        switch_of[id(sw)] = i
        switch_types.add(type(sw).__name__)
        bits_of[i] = {
            name: int(getattr(sw, name).bits)
            for name in ("gate_proj", "up_proj", "down_proj", "gate_up_proj")
            if hasattr(sw, name) and hasattr(getattr(sw, name), "bits")
        }

    def prefill(capture=None):
        cache = make_prompt_cache(model)
        chunk_ms = []
        start_all = time.perf_counter()
        for c, start in enumerate(range(0, a.context, a.chunk)):
            toks = tokens[:, start:start + a.chunk]
            if capture is not None:
                capture["chunk"] = c
                capture["pending"] = []
            t = time.perf_counter()
            model(toks, cache=cache)
            extra = [arr for _, arr in capture["pending"]] if capture is not None else []
            mx.eval([e.state for e in cache] + extra)
            chunk_ms.append((time.perf_counter() - t) * 1e3)
            if capture is not None:
                for key, arr in capture["pending"]:
                    capture["store"][key] = np.array(arr).astype(np.int32)
        total = (time.perf_counter() - start_all) * 1e3
        del cache
        mx.clear_cache()
        return total, chunk_ms

    # Warm-up: one prefill per arm.
    for mode in ("fused", "off"):
        nax.set_mode(mode)
        prefill()

    reps = {"fused": [], "off": []}
    chunks = {"fused": [], "off": []}
    nax_status = None
    for r in range(a.reps):
        order = ["fused", "off"] if r % 2 == 0 else ["off", "fused"]
        for mode in order:
            nax.set_mode(mode)
            nax.status(reset=True)
            total, cms = prefill()
            reps[mode].append(total)
            chunks[mode].append(cms)
            if mode == "fused" and nax_status is None:
                nax_status = nax.status()
        print(json.dumps({"rep": r, "fused_ms": reps["fused"][-1], "off_ms": reps["off"][-1]}), flush=True)

    # Capture pass (served mode).
    nax.set_mode(served_mode)
    capture = {"store": {}, "chunk": 0, "pending": []}
    originals = {}
    for cls in (qwen3_next.FusedDownSwitchGLU, qwen3_next.FusedGateUpSwitchGLU):
        orig = cls.__call__
        originals[cls] = orig

        def wrapped(self, x, indices, *args, _orig=orig, **kw):
            layer = switch_of.get(id(self))
            if layer is not None and indices.ndim >= 2 and indices.size > 64:
                capture["pending"].append((f"L{layer:02d}_C{capture['chunk']}", indices))
            return _orig(self, x, indices, *args, **kw)

        cls.__call__ = wrapped
    try:
        cap_total, cap_chunks = prefill(capture)
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig
    np.savez_compressed(out / "routing.npz", **capture["store"])

    swap1 = swapouts()
    def summ(xs):
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs), "all": xs}
    report = {
        "schema": "mlx2.research.b1-capture.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model": a.model,
        "adapter": type(adapter).__name__,
        "fingerprint": getattr(adapter, "identity", {}).get("fingerprint"),
        "load_s": load_s,
        "context": a.context,
        "chunk": a.chunk,
        "served_nax_mode": served_mode,
        "switch_types": sorted(switch_types),
        "expert_bits_per_layer": bits_of,
        "token_meta": token_meta,
        "prefill_ms": {k: summ(v) for k, v in reps.items()},
        "prefill_chunk_ms": chunks,
        "nax_status_fused_rep": nax_status,
        "capture_prefill_ms": cap_total,
        "captured_keys": len(capture["store"]),
        "swapouts_before": swap0,
        "swapouts_after": swap1,
        "swap_rose": swap1 > swap0,
        "therm_before": therm0,
        "therm_after": therm(),
        "mlx_version": mx.__version__,
        "device": mx.device_info().get("device_name"),
    }
    (out / "capture.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("prefill_ms", "captured_keys", "swap_rose")}, default=str))


if __name__ == "__main__":
    main()
