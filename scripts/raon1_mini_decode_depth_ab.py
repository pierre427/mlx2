#!/usr/bin/env python3
"""In-process decode-at-depth A/B of Raon-1-mini's mla_decode lever (sanity, not publishable).

    PYTHONPATH=src:scripts python -s scripts/raon1_mini_decode_depth_ab.py --model ART --out DIR \
        [--cells 4096,32768,65536,131072] [--decode 96] [--needle-cells 131072] \
        [--lever mla_decode=nax_absorbed]

Same loaded weights, policy switched with ``Model.configure``.  Per cell and
arm (A = default policy, B = default + lever): a fresh cache, the same prompt
of exactly ``cell`` tokens of varied repository text (realistic_haystack,
never a repeated string; at needle cells an access code buried at 1/3 and a
question), prefilled in 2048-row chunks (prefill is identical in both arms:
the lever touches decode rows only), then ``--decode`` greedy tokens.  Decode
tok/s is timed over a fixed --decode tokens from the second (greedy past
EOS); answers are scored up to the first EOS.  Arm order alternates per cell.  Records both arms' tokens, the first
divergent position with both arms' top-2 margins, needle hits, and the MLA
counter delta per arm (engagement: B must show decode_nax_absorbed and no
latent_single for decode rows, A must show none).  No thermal hold
(AGENTS.md); the caller owns the GPU lease.  Writes ab.json atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _arrays(tree, mx):
    """Every mx.array in a nested cache state (None slots skipped)."""
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, (list, tuple)):
        return [a for item in tree for a in _arrays(item, mx)]
    return []


CEILING = 0.5


def classify(a, b, first):
    """AGENTS.md near-tie contract at the first divergence (functional oracle separate).

    Shared top-two with both margins <= 0.5; exact-tie clause (one arm's margin
    0, the other picked one of the tied tokens with margin <= 0.5); multi-way
    clause (each arm's pick within 0.5 nat of the other arm's top log-prob).
    A token missing from a top-5 window leaves the row unclassified unless the
    window proves it further than 0.5 below the top (high margin).
    """
    wa, wb = dict(map(tuple, a["top5"][first])), dict(map(tuple, b["top5"][first]))
    pa, pb = a["tokens"][first], b["tokens"][first]
    ma, mb = a["margins"][first], b["margins"][first]
    top2a = {t for t, _ in sorted(wa.items(), key=lambda r: -r[1])[:2]}
    top2b = {t for t, _ in sorted(wb.items(), key=lambda r: -r[1])[:2]}
    out = {"pick_a": pa, "pick_b": pb, "margin_a": ma, "margin_b": mb, "top2_a": sorted(top2a), "top2_b": sorted(top2b),
           "shared_prefix": a["tokens"][:first] == b["tokens"][:first]}
    if not out["shared_prefix"]:
        return {**out, "verdict": "unclassified", "why": "prefix differs"}
    if top2a == top2b and ma <= CEILING and mb <= CEILING:
        return {**out, "verdict": "near_tie_candidate", "clause": "shared_top_two"}
    for tied, other_pick, other_margin, tied_top2 in ((ma, pb, mb, top2a), (mb, pa, ma, top2b)):
        if tied == 0.0 and other_pick in tied_top2 and other_margin <= CEILING:
            return {**out, "verdict": "near_tie_candidate", "clause": "exact_tie"}
    top_a, top_b = max(wa.values()), max(wb.values())
    if pb not in wa or pa not in wb:
        low = (pb not in wa and min(wa.values()) < top_a - CEILING) or (pa not in wb and min(wb.values()) < top_b - CEILING)
        return {**out, "verdict": "high_margin" if low else "unclassified", "why": "pick outside the other top-5"}
    gaps = (top_a - wa[pb], top_b - wb[pa])
    out["multiway_gaps"] = [round(g, 4) for g in gaps]
    if all(g <= CEILING for g in gaps):
        return {**out, "verdict": "near_tie_candidate", "clause": "multi_way"}
    return {**out, "verdict": "high_margin"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", default="4096,32768,65536,131072")
    parser.add_argument("--needle-cells", default="131072")
    parser.add_argument("--decode", type=int, default=96)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--lever", default="mla_decode=nax_absorbed")
    # MLX's allocator pool is unbounded by default. A chunked prefill frees
    # buffers of ever-growing sizes (the KV/latent cache grows each chunk, and
    # the score tiles widen with it); none is reused, so the pool keeps them
    # until macOS pages them out (the 128K cells' one-way swap write-out,
    # 2026-11-26/27). Bound the pool and return it after every chunk. Values only
    # change allocation, never results.
    parser.add_argument("--cache-limit-gib", type=float, default=8.0)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx

    mx.set_cache_limit(int(args.cache_limit_gib * (1 << 30)))
    from transformers import AutoTokenizer

    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R
    from mlx2.runtime.ubc_evict import load_shards_evicting
    from realistic_haystack import bury, haystack

    key, value = args.lever.split("=", 1)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    arms = {"A": A.parse_policy({}), "B": A.parse_policy({key: value})}
    artifact = A.inspect_artifact(args.model)
    model = R.Model(R.ModelArgs.from_dict(artifact["config"]))
    R.install_for_layout(model, artifact["layout"], artifact["config"], artifact["weight_map"].keys())
    root = Path(artifact["identity"]["path"])
    weights = model.sanitize(load_shards_evicting([root / n for n in sorted(set(artifact["weight_map"].values()))]))
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    mx.eval(model.parameters())
    model.set_reasoning_effort("none")
    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    report = {"schema": "mlx2.raon1-mini-decode-depth-ab.v1", "checkpoint": artifact["checkpoint"],
              "lever": args.lever, "arms": arms, "thermal_hold": False,
              "mlx2_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip(),
              "mlx": mx.__version__, "cache_limit_gib": args.cache_limit_gib,
              "decode_tokens": args.decode, "cells": []}
    needle_cells = {int(c) for c in args.needle_cells.split(",") if c}
    offset = 0

    def mla_counts():
        return Counter(model.mechanism_status()["mla"])

    def write():
        tmp = args.out / "ab.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, args.out / "ab.json")

    for cell in [int(c) for c in args.cells.split(",")]:
        needle = cell in needle_cells
        question = ("\n\nWhat is the access code stated in the text above? Reply with the code only."
                    if needle else "\n\nIn one short sentence, what is the main topic of the text above?")
        overhead = len(tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                               enable_thinking=False, return_dict=True)["input_ids"])
        text = haystack(cell * 6, offset=offset)
        offset += len(text)
        budget = cell - overhead
        code = None
        if needle:
            code = f"RX-{(cell * 7) % 9000 + 1000}-Q{'LPWHJN'[cell % 6]}"
            fact = f"The access code for this archive is {code}."
            budget -= len(tok.encode(fact, add_special_tokens=False)) + 2
        body = tok.decode(tok.encode(text, add_special_tokens=False)[:budget])
        if needle:
            body = bury(body, fact)
        ids = list(tok.apply_chat_template([{"role": "user", "content": body + question}], add_generation_prompt=True,
                                           enable_thinking=False, return_dict=True)["input_ids"])
        row = {"cell": cell, "prompt_tokens": len(ids), "needle_code": code, "arms": {}}
        # Chunk bounds over ids[:-1]; a last chunk of <= 16 rows would reach
        # the decode gate, so it is merged into the chunk before it.
        bounds = list(range(0, len(ids) - 1, args.chunk)) + [len(ids) - 1]
        if len(bounds) > 2 and bounds[-1] - bounds[-2] <= 16:
            bounds.pop(-2)
        row["prefill_chunks"] = len(bounds) - 1
        # Alternate arm order per cell so B is not always second.
        order = ("A", "B") if len(report["cells"]) % 2 == 0 else ("B", "A")
        row["arm_order"] = list(order)
        for name in order:
            model.configure(arms[name])
            cache = model.make_cache()
            mx.clear_cache()
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            for start, stop in zip(bounds, bounds[1:]):
                logits = model(mx.array([ids[start:stop]]), cache)[:, -1]
                mx.eval(logits, _arrays([getattr(c, "state", None) for c in cache], mx))
                mx.clear_cache()
            prefill_peak_gib = mx.get_peak_memory() / (1 << 30)
            prefill_s = time.perf_counter() - t0
            before = mla_counts()
            out, margins, tops = [], [], []
            x = mx.array([[ids[-1]]])
            t1 = None
            # A fixed --decode tokens are timed (greedy continues past EOS so
            # both arms time the same count); text, needle and the token
            # comparison use the tokens up to the first EOS.
            for i in range(args.decode):
                lg = model(x, cache)[0, -1].astype(mx.float32)
                top = mx.sort(mx.topk(lg, 2))
                nxt = int(mx.argmax(lg))
                margins.append(float(top[1] - top[0]))
                # Top-5 window (token id, log-prob) at every step, so a first
                # divergence classifies from its own position (AGENTS.md).
                lp = lg - mx.logsumexp(lg)
                ids5 = mx.argpartition(-lp, 4)[:5]
                tops.append(sorted(([int(t), round(float(lp[t]), 5)] for t in ids5.tolist()), key=lambda r: -r[1]))
                out.append(nxt)
                if i == 0:
                    t1 = time.perf_counter()
                x = mx.array([[nxt]])
            elapsed = time.perf_counter() - t1
            delta = {k: v - before.get(k, 0) for k, v in mla_counts().items() if v - before.get(k, 0)}
            eos = next((j for j, t in enumerate(out) if t in A.EOS_TOKEN_IDS), len(out) - 1)
            out, margins, tops = out[:eos + 1], margins[:eos + 1], tops[:eos + 1]
            answer = tok.decode(out)
            row["arms"][name] = {
                "prefill_s": round(prefill_s, 2), "prefill_peak_gib": round(prefill_peak_gib, 2),
                "peak_gib": round(mx.get_peak_memory() / (1 << 30), 2), "timed_tokens": args.decode - 1, "answer_tokens": len(out),
                "decode_tok_s": round((args.decode - 1) / elapsed, 2) if args.decode > 1 else None,
                "tokens": out, "margins": [round(m, 4) for m in margins], "top5": tops, "text": answer[:300],
                "mla_counter_delta": delta,
                **({"needle_found": code in answer.replace("‑", "-").replace("–", "-")} if needle else {})}
            del cache
            mx.clear_cache()
        ta, tb = row["arms"]["A"]["tokens"], row["arms"]["B"]["tokens"]
        first = next((j for j, (p, q) in enumerate(zip(ta, tb)) if p != q), None)
        row["identical"] = ta == tb
        row["first_divergence"] = first
        if first is not None:
            row["margin_a"] = row["arms"]["A"]["margins"][first]
            row["margin_b"] = row["arms"]["B"]["margins"][first]
            row["near_tie"] = classify(row["arms"]["A"], row["arms"]["B"], first)
        sa, sb = row["arms"]["A"]["decode_tok_s"], row["arms"]["B"]["decode_tok_s"]
        row["decode_ratio_b_over_a"] = round(sb / sa, 4) if sa and sb else None
        db, da = row["arms"]["B"]["mla_counter_delta"], row["arms"]["A"]["mla_counter_delta"]
        # Full engagement: every decode row in every MLA layer took the kernel
        # in B, none in A.
        row["engaged"] = (db.get("decode_nax_absorbed", 0) > 0 and db.get("decode_nax_declined", 0) == 0
                          and not any(k.startswith("latent_single") for k in db)
                          and not any(k.startswith("decode_nax") for k in da))
        report["cells"].append(row)
        write()
        print(json.dumps({k: v for k, v in row.items() if k != "arms"}
                         | {n: {k: v for k, v in a.items() if k not in ("tokens", "margins", "text", "top5")}
                            for n, a in row["arms"].items()}), flush=True)
    write()


if __name__ == "__main__":
    main()
