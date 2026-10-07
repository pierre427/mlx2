#!/usr/bin/env python3
"""End-to-end A/B: 8K prefill on Qwen3.6-35B-A3B Heretic, NAX 6-bit on vs off.

Loads the model through its mlx2 adapter (default policy: NAX MoE gather
``fused``, served prefill chunk 2048) and times the 8K-token prefill chunk
by chunk exactly as research_b1_capture_routing.py does (``model(chunk,
cache)`` then ``mx.eval`` of the cache state) under two arms, interleaved
(ABBA order) after a warm-up of each:

* ``nax4``: NAX fused with the route's current widths (affine 4/8-bit; the
  6-bit layers keep the stock gathers);
* ``nax6``: NAX fused with 6-bit admitted too (``set_affine_bits``).

Then one more prefill per arm records each chunk's last-position logits and
a greedy decode continuation (tokens + per-step logits), compared bitwise
between the arms.  Run through run_with_gpu_locks.py.
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
sys.path.insert(0, str(ROOT / "scripts"))

from research_b1_capture_routing import corpus_tokens, swapouts, therm  # noqa: E402

ARMS = {"nax4": (4, 8), "nax6": (4, 6, 8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit"))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--decode", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import mlx.core as mx
    import numpy as np

    mx.set_cache_limit(4 << 30)
    swap0, therm0 = swapouts(), therm()
    from mlx2.adapters.registry import resolve_adapter

    t0 = time.perf_counter()
    adapter = resolve_adapter(a.model)(a.model)
    load_s = time.perf_counter() - t0
    model = adapter.model
    from mlx2.runtime.models import moe_nax_gather as nax
    from mlx2.runtime.models.cache import make_prompt_cache

    served_mode = nax.MODE
    if served_mode != "fused":
        raise SystemExit(f"adapter did not select NAX fused (got {served_mode!r})")
    ids, token_meta = corpus_tokens(adapter.tokenizer, a.context)
    tokens = mx.array(ids, dtype=mx.int32)[None]

    def prefill(record=False):
        cache = make_prompt_cache(model)
        chunk_ms, last = [], []
        t_all = time.perf_counter()
        for start in range(0, a.context, a.chunk):
            t = time.perf_counter()
            logits = model(tokens[:, start:start + a.chunk], cache=cache)
            extra = [logits[:, -1]] if record else []
            mx.eval([e.state for e in cache] + extra)
            chunk_ms.append((time.perf_counter() - t) * 1e3)
            if record:
                last.append(extra[0])
        total = (time.perf_counter() - t_all) * 1e3
        return total, chunk_ms, cache, last

    def run_arm(arm, record=False):
        nax.set_affine_bits(ARMS[arm])
        nax.status(reset=True)
        total, cms, cache, last = prefill(record)
        st = nax.status()
        dec = None
        if record:
            toks, step_logits = [], []
            nxt = mx.argmax(last[-1], axis=-1)
            for _ in range(a.decode):
                toks.append(int(nxt.item()))
                lg = model(nxt.reshape(1, 1), cache=cache)[:, -1]
                mx.eval(lg)
                step_logits.append(lg)
                nxt = mx.argmax(lg, axis=-1)
            dec = (toks, step_logits)
        del cache
        mx.clear_cache()
        return total, cms, st, last, dec

    for arm in ARMS:  # warm-up (also arms every kernel's canary)
        run_arm(arm)

    reps = {k: [] for k in ARMS}
    chunks = {k: [] for k in ARMS}
    statuses = {}
    for r in range(a.reps):
        order = list(ARMS) if r % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            total, cms, st, _, _ = run_arm(arm)
            reps[arm].append(total)
            chunks[arm].append(cms)
            statuses.setdefault(arm, st)
        print(json.dumps({"rep": r, **{k: round(v[-1], 1) for k, v in reps.items()}}), flush=True)
        if swapouts() > swap0:
            print(json.dumps({"abort": "swapouts rose"}), flush=True)
            break

    # Bit-identity: last-position logits per chunk + greedy decode.
    rec = {arm: run_arm(arm, record=True) for arm in ARMS}

    def u16(a_):
        return a_.view(mx.uint16) if a_.dtype in (mx.bfloat16, mx.float16) else a_.view(mx.uint32)

    def same(x, y):
        return x.shape == y.shape and x.dtype == y.dtype and bool(mx.array_equal(u16(x), u16(y)).item())

    def digest(arrs):
        h = hashlib.sha256()
        for x in arrs:
            h.update(np.array(u16(x)).tobytes())
        return h.hexdigest()[:16]

    a4, a6 = rec["nax4"], rec["nax6"]
    identity = {
        "chunk_last_logits_bitwise": [same(x, y) for x, y in zip(a4[3], a6[3])],
        "decode_tokens_equal": a4[4][0] == a6[4][0],
        "decode_tokens": a4[4][0],
        "decode_tokens_nax6": a6[4][0],
        "decode_logits_bitwise": all(same(x, y) for x, y in zip(a4[4][1], a6[4][1])),
        "digest_nax4": digest(a4[3] + a4[4][1]),
        "digest_nax6": digest(a6[3] + a6[4][1]),
        "status_nax4": a4[2],
        "status_nax6": a6[2],
        "decoded_text": adapter.tokenizer.decode(a4[4][0]),
    }

    swap1 = swapouts()

    def summ(xs):
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs), "all": xs}

    m4, m6 = statistics.median(reps["nax4"]), statistics.median(reps["nax6"])
    report = {
        "schema": "mlx2.research.nax6-e2e.v1",
        "source_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "dirty_src": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "src"], text=True).strip()),
        "model": a.model,
        "adapter": type(adapter).__name__,
        "fingerprint": getattr(adapter, "identity", {}).get("fingerprint"),
        "load_s": load_s,
        "context": a.context, "chunk": a.chunk, "served_nax_mode": served_mode,
        "arms": {k: list(v) for k, v in ARMS.items()},
        "token_meta": token_meta,
        "prefill_ms": {k: summ(v) for k, v in reps.items()},
        "prefill_chunk_ms": chunks,
        "speedup_median": m4 / m6,
        "saving_ms_median": m4 - m6,
        "status_first_timed_rep": statuses,
        "identity": identity,
        "swapouts_before": swap0, "swapouts_after": swap1, "swap_rose": swap1 > swap0,
        "therm_before": therm0, "therm_after": therm(),
        "mlx_version": mx.__version__, "device": mx.device_info().get("device_name"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"prefill_median_ms": {"nax4": m4, "nax6": m6}, "speedup": m4 / m6,
                      "identity": {k: identity[k] for k in ("chunk_last_logits_bitwise", "decode_tokens_equal",
                                                            "decode_logits_bitwise", "digest_nax4", "digest_nax6")},
                      "swap_rose": swap1 > swap0}, indent=1))


if __name__ == "__main__":
    main()
