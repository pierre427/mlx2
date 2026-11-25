#!/usr/bin/env python3
"""In-process A/B of one Raon-1-mini numeric lever (same weights, policy switched).

    PYTHONPATH=src python -I scripts/raon1_mini_lever_ab.py --model ART --lever mhc=fused --out DIR

Per arm: teacher-forced logits on the oracle prompt, greedy continuations of
three prompts (first divergent token with both arms' top-2 margins), and
ABBA-alternated decode tok/s.  GPU lease only.  Writes ab.json atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PROMPTS = ["Explain how a heat pump works in four sentences.",
           "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
           "How many clips did Natalia sell altogether in April and May?",
           "Write a Python function that returns the n-th Fibonacci number iteratively."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lever", required=True, help="key=value on top of the default policy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)
    import mlx.core as mx
    from transformers import AutoTokenizer

    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R
    from mlx2.runtime.ubc_evict import load_shards_evicting
    import raon1_mini_reference_oracle as O

    key, value = args.lever.split("=", 1)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    base = A.parse_policy({})
    arms = {"A": base, "B": A.parse_policy({key: value})}
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
    tree = Path(__file__).resolve().parents[1]
    report = {"checkpoint": artifact["checkpoint"], "lever": args.lever, "arms": arms,
              "mlx2_revision": subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip(),
              "mlx": mx.__version__}
    oracle_ids = O.tokens(root)
    logits, greedy = {}, {}

    def greedy_run(ids, n):
        cache = model.make_cache()
        out, rows = [], []
        x = mx.array([ids])
        for _ in range(n):
            lg = model(x, cache)[0, -1].astype(mx.float32)
            nxt = int(mx.argmax(lg))
            top = mx.sort(lg)[-2:]
            rows.append((nxt, float(top[1] - top[0])))
            out.append(nxt)
            if nxt in A.EOS_TOKEN_IDS:
                break
            x = mx.array([[nxt]])
        return out, rows

    for name, policy in arms.items():
        model.configure(policy)
        logits[name] = model(mx.array([oracle_ids]))[0].astype(mx.float32)
        mx.eval(logits[name])
        greedy[name] = []
        for text in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          enable_thinking=False, return_dict=True)["input_ids"]
            greedy[name].append(greedy_run(list(ids), args.tokens))
    la, lb = logits["A"], logits["B"]
    pa, pb = la - mx.logsumexp(la, -1, keepdims=True), lb - mx.logsumexp(lb, -1, keepdims=True)
    report["teacher_forced"] = {
        "positions": int(la.shape[0]), "max_abs": float(mx.abs(la - lb).max()),
        "top1_agree": float((mx.argmax(la, -1) == mx.argmax(lb, -1)).astype(mx.float32).mean()),
        "kl_mean": float((mx.exp(pa) * (pa - pb)).sum(-1).mean())}
    report["greedy"] = []
    for i, text in enumerate(PROMPTS):
        (ta, ra), (tb, rb) = greedy["A"][i], greedy["B"][i]
        first = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        row = {"prompt": text[:60], "tokens_a": len(ta), "tokens_b": len(tb), "identical": ta == tb,
               "first_divergence": first, "text_a": tok.decode(ta)[:300], "text_b": tok.decode(tb)[:300]}
        if first is not None:
            row["margin_a"], row["margin_b"] = ra[first][1], rb[first][1]
        report["greedy"].append(row)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True,
                                  enable_thinking=False, return_dict=True)["input_ids"]
    speed = {"A": [], "B": []}
    for name in ("A", "B", "B", "A", "B", "A", "A", "B"):
        model.configure(arms[name])
        cache = model.make_cache()
        lg = model(mx.array([ids]), cache)
        nxt = mx.argmax(lg[:, -1], -1)
        mx.eval(nxt)
        t0 = time.time()
        for _ in range(96):
            nxt = mx.argmax(model(nxt[:, None], cache)[:, -1], -1)
            mx.async_eval(nxt)
        mx.eval(nxt)
        speed[name].append(96 / (time.time() - t0))
    report["decode_tok_s"] = {k: [round(v, 1) for v in vals] for k, vals in speed.items()}
    report["decode_ratio_b_over_a"] = round(sum(speed["B"]) / sum(speed["A"]), 4)
    tmp = args.out / "ab.json.tmp"
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, args.out / "ab.json")
    print(json.dumps({k: v for k, v in report.items() if k not in ("greedy",)}, indent=1))
    for row in report["greedy"]:
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("text")}))


if __name__ == "__main__":
    main()
