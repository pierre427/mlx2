#!/usr/bin/env python3
"""End-to-end prefill tok/s A/B for the fused GroupRMSNorm lever.

One process, one loaded model (Qwen3.8-Flash-Next-MLX-4bit-MTP).  Arms differ
only in ``fgn.set_fused_group_norm_enabled``; the kernel engages at candidate
row counts (1024/2048/4096/8192/16384) and falls back to eager everywhere else
(including every decode step), so this measures exactly what a serving request
would see: the prefill pass with the lever on or off, plus greedy generation.

Thermal control: arms are interleaved per context length (A,B,A,B,...), each
round preceded by a cooldown, and every run is labelled with the round number.
Engagement is verified from the kernel's own counters; a null (zero calls in
the fused arm at a qualified width) raises instead of writing a receipt.

Usage:
  .venv/bin/python scripts/bench_fgn_end_to_end_tps.py \
      --out /tmp/fgn-ttft-ab.json --i-own-the-gpu
"""

import argparse
import json
import statistics
import time
from pathlib import Path

SCHEMA = "mlx2.fgn-end-to-end-ttft-ab.v1"


def encode(tok, text):
    ids = tok.encode(text)
    return [int(t) for t in ids]


def make_context_text(n_tokens, tok):
    """Build ~n_tokens of unique-ish text and trim to an exact token count."""
    base = (
        "The mlx-uag lab ports frontier-class models to Apple Silicon via MLX. "
        "Speculative decoding, prefix caching, and continuous batching interact "
        "through the APCv2 state machine, which owns frozen layer segments and "
        "revision-bound transactions. "
    )
    chunks = []
    i = 0
    total = 0
    while total < n_tokens + 512:
        # vary the sentence so tokens are not perfectly periodic
        chunks.append(f"[{i}] " + base * (1 + (i % 3)))
        total += len(encode(tok, f"[{i}] " + base * (1 + (i % 3))))
        i += 1
    text = " ".join(chunks)
    ids = encode(tok, text)[:n_tokens]
    return ids


def stop_token_ids(tok):
    try:
        eos = tok.eos_token_id
        return {eos} if eos is not None else set()
    except Exception:
        return set()


def run_arm(mx, model, prompt_ids, gen_tokens, lever, fgn, stop_ids):
    from mlx2.runtime.models.cache import make_prompt_cache

    fgn.set_fused_group_norm_enabled(lever)
    fgn.reset_fused_group_norm_stats()
    try:
        cache = list(make_prompt_cache(model))
        t0 = time.perf_counter()
        logits = model(mx.array([prompt_ids], dtype=mx.uint32), cache=cache)
        mx.eval(logits)
        ttft = time.perf_counter() - t0

        lps = []
        lg = logits[0, -1].astype(mx.float32)
        generated = 0
        for step in range(gen_tokens):
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            mx.eval(lp)
            lps.append(lp)
            generated += 1
            nxt = int(mx.argmax(lp, axis=-1).item())
            if nxt in stop_ids or step + 1 >= gen_tokens:
                break
            logits = model(mx.array([[nxt]], dtype=mx.uint32), cache=cache)
            lg = logits[0, -1].astype(mx.float32)
        t1 = time.perf_counter()
        stats = fgn.fused_group_norm_stats()
        try:
            del cache, logits, lg, lps
        except UnboundLocalError:
            pass
        mx.metal.clear_cache()
    finally:
        fgn.set_fused_group_norm_enabled(False)
    return {
        "ttft_seconds": ttft,
        "prefill_tok_per_s": len(prompt_ids) / ttft if ttft > 0 else None,
        "gen_seconds": t1 - t0 - ttft,
        "generated_tokens": generated,
        "decode_tok_per_s": generated / max(t1 - t0 - ttft, 1e-9),
        "counters": stats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default=str(Path.home() / "mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"))
    p.add_argument("--context-lengths", type=int, nargs="+",
                   default=[1024, 4096, 8192, 16384])
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3,
                   help="interleaved A/B rounds per context length")
    p.add_argument("--cooldown-s", type=float, default=20.0)
    p.add_argument("--warmup-runs", type=int, default=1,
                   help="unrecorded runs per arm per length before recording")
    p.add_argument("--out", required=True)
    p.add_argument("--i-own-the-gpu", action="store_true")
    a = p.parse_args()
    if not getattr(a, 'i_own_the_gpu'):
        raise SystemExit("refusing Metal without --i-own-the-gpu")

    import os
    import sys

    # Run the source tree this script lives in (the fgn branch), not whatever
    # editable install the venv points at.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_fused_group_norm as fgn

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("this harness needs the GPU queue")

    t0 = time.perf_counter()
    adapter_cls = resolve_adapter(a.model)
    adapter = adapter_cls(a.model)
    model = adapter.model
    tok = adapter.tokenizer
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s; peak Metal "
          f"{mx.get_peak_memory()/1e9:.2f} GB", flush=True)

    stops = stop_token_ids(tok)
    contexts = {}
    results = {"schema": SCHEMA, "model": a.model,
               "context_lengths": a.context_lengths, "gen_tokens": a.gen_tokens,
               "rounds": a.rounds, "arms": {
                   "eager": "MLX_QWEN4_FUSED_GROUP_NORM off (production)",
                   "fused": "MLX_QWEN4_FUSED_GROUP_NORM on, candidates "
                            "1024/2048/4096/8192/16384"},
               "cells": {}, "model_load_seconds": load_s,
               "host_started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    for length in a.context_lengths:
        if length not in contexts:
            ids = make_context_text(length, tok)
            assert len(ids) == length, f"built {len(ids)} tokens, want {length}"
            contexts[length] = ids
            print(f"context {length}: built ({len(ids)} tokens)", flush=True)

        cell = {"runs": []}
        for rnd in range(1, a.rounds + 1):
            for trial in range(a.warmup_runs + 1):
                for arm in ("eager", "fused"):
                    time.sleep(a.cooldown_s)
                    out = run_arm(mx, model, contexts[length], a.gen_tokens,
                                   arm == "fused", fgn, stops)
                    if arm == "fused" and out["counters"].get("calls", 0) == 0:
                        # Expected when T is below the smallest candidate row
                        # count (1024): the kernel declines every call and the
                        # fused arm is numerically identical to eager.  Flag it
                        # rather than recording a fake speedup.
                        out["kernel_engaged"] = False
                    elif arm == "fused":
                        out["kernel_engaged"] = True
                    if trial < a.warmup_runs:
                        continue
                    row = {"round": rnd, "arm": arm, **out}
                    cell["runs"].append(row)
                    print(f"T={length} r{rnd} {arm}: ttft={out['ttft_seconds']:.3f}s "
                          f"({out['prefill_tok_per_s']:.0f} tok/s) "
                          f"kernel_calls={out['counters'].get('calls', 0)}",
                          flush=True)
            # per-round summary
            ea = [r["ttft_seconds"] for r in cell["runs"]
                  if r["round"] == rnd and r["arm"] == "eager"]
            fu = [r["ttft_seconds"] for r in cell["runs"]
                  if r["round"] == rnd and r["arm"] == "fused"]
            if ea and fu:
                ratio = statistics.median(ea) / statistics.median(fu)
                print(f"  -> T={length} r{rnd}: eager {statistics.median(ea):.3f}s "
                      f"vs fused {statistics.median(fu):.3f}s = {ratio:.3f}x",
                      flush=True)
        results["cells"][str(length)] = cell

    # aggregate
    agg = {}
    for length, cell in results["cells"].items():
        for arm in ("eager", "fused"):
            vals = [r["ttft_seconds"] for r in cell["runs"] if r["arm"] == arm]
            tps = [r["prefill_tok_per_s"] for r in cell["runs"] if r["arm"] == arm]
            dec = [r["decode_tok_per_s"] for r in cell["runs"] if r["arm"] == arm]
            key = f"T{length}.{arm}"
            agg[key] = {
                "ttft_median_s": statistics.median(vals),
                "ttft_min_s": min(vals),
                "ttft_max_s": max(vals),
                "prefill_tps_median": statistics.median(tps),
                "decode_tps_median": statistics.median(dec),
                "n": len(vals),
            }
        e = agg[f"T{length}.eager"]["ttft_median_s"]
        f = agg[f"T{length}.fused"]["ttft_median_s"]
        agg[f"T{length}.speedup"] = round(e / f, 4) if f else None
    results["aggregate"] = agg
    results["host_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {a.out}")
    for k, v in agg.items():
        if "speedup" in k:
            print(f"{k}: {v}")
        elif isinstance(v, dict):
            print(f"{k}: ttft {v['ttft_median_s']:.3f}s "
                  f"prefill {v['prefill_tps_median']:.0f} tok/s "
                  f"decode {v['decode_tps_median']:.1f} tok/s (n={v['n']})")


if __name__ == "__main__":
    main()
