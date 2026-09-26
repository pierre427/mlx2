#!/usr/bin/env python3
"""Classify greedy mismatches against ordinary by a teacher-forced forward.

For each greedy output that differs from the ordinary arm's (same rep, cell
and prompt), re-tokenize both outputs, find the first differing token, run
the target once over prompt + the shared prefix (ordinary decode path, one
forward), and report the two tokens' logits there.  A mismatch is a near-tie
when the arm's token is the runner-up and the logit gap to the reference
token is below ``--tie-logits``; anything else is a correctness failure.

GPU job (loads the 27B target once).  ``--i-own-the-gpu`` is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="~/mlx-models/Qwen3.8-27B-oQ4e-mtp")
    parser.add_argument("--tie-logits", type=float, default=0.5)
    parser.add_argument("--nonce-salt", default="sp-dflash2")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        raise SystemExit("refusing: pass --i-own-the-gpu inside the GPU lock")

    import ab_qwen38_dflash2 as ab
    from summarize_qwen38_dflash2 import greedy_gate, load

    cells, _ = load(args.results)
    rows = {}
    for cell in cells:
        for row in cell["rows"]:
            rows[(cell["arm"], cell["rep"], cell["workload"], cell["width"],
                  cell["temperature"], row["prompt_index"])] = row
    mismatches = [e for e in greedy_gate(cells) if not e["equal"]]

    import mlx.core as mx

    mx.set_cache_limit(4 << 30)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    adapter = Qwen3827BAdapter(args.model)
    tokenizer = adapter.tokenizer
    results = []
    for entry in mismatches:
        key = (entry["rep"], entry["workload"], entry["width"], 0.0, entry["prompt_index"])
        ref = rows[("ord",) + key]
        arm = rows[(entry["arm"],) + key]
        nonce = hashlib.sha256(
            f"{args.nonce_salt}:{entry['rep']}:{entry['workload']}:{entry['width']}:0.0".encode()
        ).hexdigest()[:16]
        item = ab.WORKLOADS[entry["workload"]][entry["prompt_index"]]
        request = {"messages": ab._messages(item, nonce), "enable_thinking": False}
        prompt = list(adapter.prompt_tokens(request))
        ref_tokens = list(tokenizer.encode(ref["output"], add_special_tokens=False))
        arm_tokens = list(tokenizer.encode(arm["output"], add_special_tokens=False))
        j = next((i for i, (a, b) in enumerate(zip(ref_tokens, arm_tokens)) if a != b),
                 min(len(ref_tokens), len(arm_tokens)))
        if j >= min(len(ref_tokens), len(arm_tokens)):
            results.append({**entry, "classification": "length_only", "token_index": j})
            continue
        cache = adapter.model.make_cache()
        logits = adapter.model(mx.array([prompt + ref_tokens[:j]]), cache=cache)[0, -1]
        logits = logits.astype(mx.float32)
        order = mx.argsort(-logits)[:5].tolist()
        ref_token, arm_token = ref_tokens[j], arm_tokens[j]
        l_ref, l_arm = float(logits[ref_token].item()), float(logits[arm_token].item())
        top = [(int(t), float(logits[t].item())) for t in order]
        rank_arm = order.index(arm_token) if arm_token in order else None
        gap = abs(l_ref - l_arm)
        tie = rank_arm is not None and rank_arm <= 1 and gap < args.tie_logits
        results.append({
            **{k: v for k, v in entry.items() if not k.endswith("context")},
            "token_index": j, "ref_token": tokenizer.decode([ref_token]),
            "arm_token": tokenizer.decode([arm_token]), "logit_ref": l_ref,
            "logit_arm": l_arm, "gap": gap, "arm_rank": rank_arm,
            "teacher_top1_is_ref": order[0] == ref_token, "top5": top,
            "classification": "near_tie" if tie else "FAIL",
        })
        del cache, logits
        mx.clear_cache()
        print(json.dumps(results[-1]), flush=True)
    summary = {
        "mismatches": len(mismatches),
        "near_ties": sum(r["classification"] == "near_tie" for r in results),
        "length_only": sum(r["classification"] == "length_only" for r in results),
        "failures": sum(r["classification"] == "FAIL" for r in results),
        "tie_logits": args.tie_logits,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
