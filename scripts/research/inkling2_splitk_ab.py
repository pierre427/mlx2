"""Inkling 2 ``inkling2_verify_splitk``: attention microbench, verify and decode A/B.

Three parts, per context depth (default 16K, 64K, 128K):

1. **Attention microbenchmark** on random tensors at the Mini geometry
   (24 query heads, 8 KV heads, D 128, bias extent 1024, BF16), L = 8 and
   L = 1, global (all keys) and sliding (window 1024, keys cropped as the
   model crops them). Arms: ``concat`` (dense mask + concatenated zero sink
   K/V column + SDPA: the reference), ``slot`` (L > 1: the same with the
   sink row already in the buffer, as ``inkling2_sink_slot``) or
   ``sink_sdpa`` (L = 1: SDPA ``sinks``, as ``inkling2_sink_sdpa``), and
   ``splitk``. Median ms over interleaved repeats, and max |diff| of splitk
   vs the reference.
2. **DSpark-style verify**: two caches, A prefilled with the option off and
   B with it on (512-token chunks; prefill never engages the kernel, so the
   last-row logits must be bitwise equal). Then rounds of an 8-row verify
   forward through ``HybridVerifyRows`` (begin, forward, abort, as a
   rejected DSpark round), alternating off on A and on on B. Median forward
   seconds, and per-row logits diff, argmax agreement and top-two margins
   on the first round.
3. **One-token decode**: blocks of 32 greedy tokens, ABBA (off leads, then on
   leads). The leading arm picks the tokens greedily; the other arm is fed the
   same tokens, so every step compares the two arms on the same history:
   max |logit diff|, argmax agreement, and wherever argmax differs, both
   arms' top-two tokens and log-prob margins (``near_tie_candidate`` when the
   top-two sets match and both margins are at most 0.5 nat; the functional
   oracle is not run here). tok/s per block.

The off arm is the current best composed path: adapter defaults plus
``sink_sdpa`` and ``sink_slot`` (``--no-sink-sdpa`` / ``--no-sink-slot`` drop
them). The on arm adds ``verify_splitk``. Kernel-receipt deltas per arm show
the route engaged. A sanity A/B: no thermal hold, not publishable statistics.

Usage:
  python scripts/research/inkling2_splitk_ab.py <model dir> <run dir>
      [--contexts 16384,65536,131072] [--rounds 8] [--blocks 2]
      [--micro-only] [--no-sink-sdpa] [--no-sink-slot]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
BLOCK = 32
VERIFY_ROWS = 8
GIB = 2**30
H, HK, D, EXTENT, WINDOW = 24, 8, 128, 1024, 1024


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


# ----- 1. attention microbenchmark -----


def dense_mask(rows, offset, keys, window):
    """``banded_additive_mask`` from precomputed bias rows [B, L, H, E]."""
    import mlx.core as mx

    b, n, h, e = rows.shape
    dist = (mx.arange(n) + offset)[:, None] - mx.arange(keys)[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, e - 1)[None, None], (b, h, n, keys))
    pb = mx.take_along_axis(rows.transpose(0, 2, 1, 3), gidx, axis=-1)
    pb = mx.where((dist >= e)[None, None], mx.array(0.0, rows.dtype), pb)
    masked = dist < 0
    if window:
        masked = masked | (dist >= window)
    return mx.where(masked[None, None], mx.array(-1e30, rows.dtype), pb)


def micro_cell(context, length, kind, repeats):
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import splitk

    mx.random.seed(context + length)
    size = context + length if kind == "global" else WINDOW + length - 1
    window = 0 if kind == "global" else WINDOW
    q = (mx.random.normal((1, length, H, D)) * 0.3).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    # One buffer with a spare zero row past ``size``: the slot arm reads the
    # view [:size + 1], the others [:size].
    kbuf = mx.concatenate([(mx.random.normal((1, HK, size, D)) * 0.3).astype(mx.bfloat16),
                           mx.zeros((1, HK, 1, D), mx.bfloat16)], axis=2)
    vbuf = mx.concatenate([mx.random.normal((1, HK, size, D)).astype(mx.bfloat16),
                           mx.zeros((1, HK, 1, D), mx.bfloat16)], axis=2)
    k, v = kbuf[:, :, :size], vbuf[:, :, :size]
    rel = (mx.random.normal((1, length, H, EXTENT)) * 2).astype(mx.bfloat16)
    sinks = (mx.random.normal((1, length, H)) * 2).astype(mx.bfloat16)
    mx.eval(q, kbuf, vbuf, rel, sinks)
    kw = dict(offset=size - length, left_padding=None, window=window, scale=D ** -0.5)

    def mask_rows():
        return dense_mask(rel, kw["offset"], size, window)

    def concat():
        return splitk.reference_attention(q, k, v, rel, sinks, **kw)

    def slot():
        mask = mx.concatenate([mask_rows(), sinks.transpose(0, 2, 1)[..., None]], axis=-1)
        return mx.fast.scaled_dot_product_attention(
            q, kbuf, vbuf, scale=kw["scale"], mask=mask)

    def sink_sdpa():
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=kw["scale"], mask=mask_rows(), sinks=sinks[0, 0])

    def run_splitk():
        return splitk.gqa_splitk_attention(q, k, v, rel, sinks, **kw)

    arms = {"concat": concat, ("slot" if length > 1 else "sink_sdpa"):
            (slot if length > 1 else sink_sdpa), "splitk": run_splitk}
    ref = concat()
    mx.eval(ref)
    diffs = {}
    for name, fn in arms.items():
        out = fn()
        mx.eval(out)
        diffs[name] = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    times = {name: [] for name in arms}
    for _ in range(3):  # warm-up
        for fn in arms.values():
            mx.eval(fn())
    for _ in range(repeats):
        for name, fn in arms.items():
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fn())
            mx.synchronize()
            times[name].append((time.perf_counter() - t0) * 1e3)
    return {"context": context, "rows": length, "kind": kind, "keys": size,
            "median_ms": {n: round(statistics.median(t), 4) for n, t in times.items()},
            "max_abs_diff_vs_concat": diffs,
            "ref_max_abs": mx.abs(ref.astype(mx.float32)).max().item()}


# ----- 2/3. model arms -----


def set_arm(model, base, on):
    model.configure_kernels(**{**base, "verify_splitk": on})


def receipt_counts(model):
    r = model.kernel_receipt()
    return {"verify_splitk_calls": r["verify_splitk_calls"],
            "verify_splitk_declines": dict(r["verify_splitk_declines"]),
            "sink_sdpa_calls": r["sink_sdpa_calls"], "sink_slot_calls": r["sink_slot_calls"],
            "sink_slot_fallbacks": r["sink_slot_fallbacks"]}


def delta(after, before):
    out = {}
    for key, value in after.items():
        if isinstance(value, dict):
            out[key] = {k: v - before[key].get(k, 0) for k, v in value.items()
                        if v - before[key].get(k, 0)}
        else:
            out[key] = value - before[key]
    return out


def prefill(model, base, tokens, on):
    import mlx.core as mx

    set_arm(model, base, on)
    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(tokens), CHUNK):
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        mx.clear_cache()  # ordinary serving prefill clears per chunk
    mx.synchronize()
    return cache, logits[0, -1], time.perf_counter() - t0


def top2(row):
    import mlx.core as mx

    logp = row - mx.logsumexp(row)
    idx = mx.argsort(-logp)[:2].tolist()
    vals = [logp[i].item() for i in idx]
    return idx, vals


def compare_rows(rows_a, rows_b):
    """Per-row max |diff|, argmax agreement, top-two at disagreements."""
    import mlx.core as mx

    steps, splits = [], []
    for i, (a, b) in enumerate(zip(rows_a, rows_b)):
        same = int(mx.argmax(a).item()) == int(mx.argmax(b).item())
        steps.append({"max_abs_diff": round(mx.abs(a - b).max().item(), 6), "argmax_equal": same})
        if not same:
            (ta, va), (tb, vb) = top2(a), top2(b)
            ma, mb = va[0] - va[1], vb[0] - vb[1]
            splits.append({"step": i, "off_top2": ta, "off_margin_nat": round(ma, 4),
                           "on_top2": tb, "on_margin_nat": round(mb, 4),
                           "near_tie_candidate": set(ta) == set(tb) and ma <= 0.5 and mb <= 0.5})
    return {"steps": len(steps), "argmax_agree": sum(s["argmax_equal"] for s in steps),
            "max_abs_diff": max((s["max_abs_diff"] for s in steps), default=0.0),
            "per_step": steps, "splits": splits}


def verify_round(model, cache, tokens):
    import mlx.core as mx

    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    owner = HybridVerifyRows([cache])
    txn = owner.begin([len(tokens)])
    try:
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(mx.array([tokens]), cache=txn.caches)
        mx.eval(logits)
        mx.synchronize()
        seconds = time.perf_counter() - t0
    finally:
        txn.abort()
    return logits[0], seconds


def decode_block(model, cache, last, forced=None):
    """32 steps; greedy from ``last`` or fed ``forced``. (rows, tokens, tok/s)."""
    import mlx.core as mx

    token = int(mx.argmax(last).item()) if forced is None else forced[0]
    rows, tokens = [], []
    mx.synchronize()
    t0 = time.perf_counter()
    for step in range(BLOCK):
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        rows.append(logits)
        if forced is None:
            token = int(mx.argmax(logits).item())
        elif step + 1 < BLOCK:
            token = forced[step + 1]
    mx.synchronize()
    return rows, tokens, round(BLOCK / (time.perf_counter() - t0), 2)


def model_cell(model, base, tokens, verify_tokens, rounds, blocks):
    import mlx.core as mx

    n = len(tokens)
    mx.reset_peak_memory()
    cache_a, last_a, sec_a = prefill(model, base, tokens, False)
    r0 = receipt_counts(model)
    cache_b, last_b, sec_b = prefill(model, base, tokens, True)
    cell = {"context": n, "prefill_s": {"off": round(sec_a, 2), "on": round(sec_b, 2)},
            "prefill_last_logits_bitwise": bool(mx.array_equal(last_a, last_b)),
            "prefill_on_receipt_delta": delta(receipt_counts(model), r0),
            "after_prefill": memory()}

    # DSpark-style 8-row verify forward (rejected round: begin, forward, abort).
    times = {"off": [], "on": []}
    first = {}
    deltas = {"off": None, "on": None}
    for i in range(rounds + 1):  # round 0 is warm-up (times dropped)
        for arm, cache in (("off", cache_a), ("on", cache_b)):
            set_arm(model, base, arm == "on")
            before = receipt_counts(model)
            rows, seconds = verify_round(model, cache, verify_tokens)
            if deltas[arm] is None:
                deltas[arm] = delta(receipt_counts(model), before)
            if i == 0:
                first[arm] = [rows[j] for j in range(rows.shape[0])]
            else:
                times[arm].append(seconds)
    cell["verify"] = {"rows": len(verify_tokens), "rounds": rounds,
                      "median_forward_s": {a: round(statistics.median(t), 5) for a, t in times.items()},
                      "forward_s": {a: [round(s, 5) for s in t] for a, t in times.items()},
                      "receipt_delta_per_round": deltas,
                      "logits": compare_rows(first["off"], first["on"])}
    print(n, "verify", cell["verify"]["median_forward_s"], "argmax",
          cell["verify"]["logits"]["argmax_agree"], "/", len(verify_tokens), flush=True)

    # One-token decode, ABBA with teacher forcing.
    cell["decode"] = []
    for i in range(blocks):
        lead = "off" if i % 2 == 0 else "on"
        order = [("off", cache_a), ("on", cache_b)]
        if lead == "on":
            order.reverse()
        out = {}
        forced = None
        for arm, cache in order:
            set_arm(model, base, arm == "on")
            before = receipt_counts(model)
            last = last_a if arm == "off" else last_b
            rows, toks, rate = decode_block(model, cache, last, forced)
            forced = toks if forced is None else forced
            out[arm] = {"rows": rows, "tokens": toks, "tok_s": rate,
                        "receipt_delta": delta(receipt_counts(model), before), "mem": memory()}
        # Both arms continue from the leading arm's greedy choice.
        last_a = last_b = out[lead]["rows"][-1]
        cmp = compare_rows(out["off"]["rows"], out["on"]["rows"])
        block = {"block": i, "lead": lead, "off_tok_s": out["off"]["tok_s"],
                 "on_tok_s": out["on"]["tok_s"], "tokens": out[lead]["tokens"],
                 "argmax_agree": cmp["argmax_agree"], "steps": cmp["steps"],
                 "max_abs_diff": cmp["max_abs_diff"], "per_step": cmp["per_step"],
                 "splits": cmp["splits"],
                 "receipt_delta": {a: out[a]["receipt_delta"] for a in out},
                 "mem_after": {a: out[a]["mem"] for a in out}}
        cell["decode"].append(block)
        print(n, "decode", {k: block[k] for k in ("block", "lead", "off_tok_s", "on_tok_s",
                                                  "argmax_agree", "max_abs_diff")}, flush=True)
    del cache_a, cache_b
    mx.clear_cache()
    return cell


def device_name():
    import mlx.core as mx

    for info in (getattr(mx, "device_info", None), getattr(mx.metal, "device_info", None)):
        if info is not None:
            try:
                return info().get("device_name")
            except Exception:  # noqa: BLE001 - informational only
                pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--contexts", default="16384,65536,131072")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--micro-repeats", type=int, default=20)
    ap.add_argument("--micro-only", action="store_true")
    ap.add_argument("--no-sink-sdpa", action="store_true")
    ap.add_argument("--no-sink-slot", action="store_true")
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    for n in contexts:
        # A tail chunk of 1..31 rows would take the kernel during prefill,
        # and the two caches would no longer start from the same state.
        if 0 < n % CHUNK < 32:
            raise SystemExit(f"context {n}: tail chunk {n % CHUNK} < 32 rows would engage the kernel")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_splitk_ab.py", "mlx2": head,
               "src_dirty": bool(dirty), "mlx_version": mx.__version__,
               "device": device_name(),
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "chunk": CHUNK, "block": BLOCK, "verify_rows": VERIFY_ROWS,
               "contexts": contexts, "micro": [], "cells": []}
    out = run_dir / "splitk_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    with mx.stream(mx.gpu):
        for n in contexts:
            for length in (VERIFY_ROWS, 1):
                for kind in ("global", "sliding"):
                    cell = micro_cell(n, length, kind, args.micro_repeats)
                    receipt["micro"].append(cell)
                    print("micro", {k: cell[k] for k in ("context", "rows", "kind", "median_ms")},
                          flush=True)
            mx.clear_cache()
            save()
    if args.micro_only:
        receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        return

    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter(str(args.model_dir))  # current defaults (MoE fast paths on)
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["adapter_kernel_policy"] = dict(model.kernel_policy)
        base = {**model.kernel_policy, "sink_sdpa": not args.no_sink_sdpa,
                "sink_slot": not args.no_sink_slot}
        receipt["off_arm_policy"] = {**base, "verify_splitk": False}
        receipt["on_arm_policy"] = {**base, "verify_splitk": True}
        corpus = corpus_tokens(adapter.tokenizer, max(contexts) + VERIFY_ROWS)
        for n in contexts:
            cell = model_cell(model, base, corpus[:n], corpus[n:n + VERIFY_ROWS],
                              args.rounds, args.blocks)
            receipt["cells"].append(cell)
            save()
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / GIB, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
