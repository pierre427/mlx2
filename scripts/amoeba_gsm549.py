#!/usr/bin/env python3
"""GSM8K test idx 549 as an amoeba spot-validation: the problem, ~30 structural variants, and a one-model screen.

Problem (benchmarks/gsm8k/test.jsonl line 550, idx 549, gold 398): "needed 2 more toys than he already had to
have enough toys to make a sorted play set five times larger than James's play set, which had 80 toys".
Answer = m * n - k. The q8 27B answers 199 (coordinator, 6/6).

items(seed) -> 31 items: test = the original + 10 variants, train = 20 variants (train items are only for
fitting maps / choosing hitch settings in amoeba_hub_transfer --task gsm549; never scored). Variants change
names, pronouns, objects, the multiplier m, the base count n and the shortfall k, over four phrasings:
  t1  the original phrasing with new numbers/names (the hard one)
  t2  "k short of having enough ... m times as large as"
  t3  "to make ... m times the size of ..., would need k more than she has"
  t4  base first, then "needs k more than he already owns to have enough"
Every item carries its template id, so a result can be read per phrasing. Numeric answers compare after
stripping commas, $, a trailing ".0" and surrounding text ("FINAL: 398 toys" -> 398).

Formats (--format, both here and in amoeba_hub_transfer --prompt-format):
  chat        plain chat, thinking off, "End your reply with a line of the form FINAL: <answer>."
  gsm8k5shot  the GSM8K harness default (spec_acceptance_gsm8k): raw completion, 5 worked train examples as
              "Question: ...\nAnswer: ..." blocks, then "Question: <q>\nAnswer:"; stop at the next "Question:";
              answer = number after "####", else the last number. This is the shape the 27B's 199 most likely
              came from, so a transfer test should first reproduce the host failure in it.

screen: one model per process (inside an owned window, or --cpu for a smoke). Plain chat, thinking off by
default (the same prompt shape amoeba_hub_transfer uses), one greedy sample plus --samples samples at
--temperature (seeded); --split picks test (default), train or all; --only limits to item ids. Writes
rows.jsonl and run.json (checkpoint fingerprint, mlx2 revision, per-item pass counts, tok/s). A donor
"solves 549 reliably" when the original passes greedy and every sample (>= 3/3 with --samples 2).
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ORIGINAL = ("Jonathan was sad to learn he needed 2 more toys than he already had to have enough toys to make a sorted "
            "play set five times larger than James's play set, which had 80 toys. How many toys does Jonathan "
            "currently have?")
SUFFIX = "\nEnd your reply with a line of the form FINAL: <answer>."
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
NAMES = [("Maria", "she"), ("Owen", "he"), ("Priya", "she"), ("Lucas", "he"), ("Hana", "she"), ("Diego", "he"),
         ("Amara", "she"), ("Felix", "he"), ("Noor", "she"), ("Tomas", "he"), ("Elena", "she"), ("Kofi", "he")]
OTHERS = ["Sam", "Ruth", "Alex", "Mina", "Leo", "Iris", "Ben", "Zoe"]
THINGS = [("stickers", "sticker album"), ("marbles", "marble collection"), ("stamps", "stamp album"),
          ("blocks", "block tower"), ("cards", "card deck"), ("beads", "bead necklace"), ("shells", "shell display"),
          ("coins", "coin collection")]
TEMPLATES = {
    "t1": "{A} was sad to learn {pa} needed {k} more {obj} than {pa} already had to have enough {obj} to make a sorted "
          "{set} {mw} times larger than {B}'s {set}, which had {n} {obj}. How many {obj} does {A} currently have?",
    "t2": "{A} found out that {pa} was {k} {obj} short of having enough {obj} to build a {set} {mw} times as large as "
          "{B}'s {set}, which has {n} {obj}. How many {obj} does {A} have right now?",
    "t3": "To make a {set} that is {mw} times the size of {B}'s {set} of {n} {obj}, {A} would need {k} more {obj} "
          "than {pa} has. How many {obj} does {A} have?",
    "t4": "{B}'s {set} has {n} {obj}. {A} wants a {set} {mw} times bigger, but learns {pa} needs {k} more {obj} than "
          "{pa} already owns to have enough. How many {obj} does {A} own now?",
}


def items(seed=20261115):
    rng = np.random.default_rng(seed)
    out = [{"id": "gsm549", "template": "original", "split": "test", "numeric": True,
            "question": ORIGINAL, "prompt": ORIGINAL + SUFFIX, "answer": "398"}]
    # test: 6 x t1 (the failing phrasing), 2 x t4, 1 x t2, 1 x t3; train: 5 of each
    plan = ["t1"] * 6 + ["t4"] * 2 + ["t2", "t3"] + ["t1", "t2", "t3", "t4"] * 5
    seen = {(5, 80, 2)}
    for j, tid in enumerate(plan):
        while True:
            m = int(rng.integers(2, 10))
            n = int(rng.integers(4, 25)) * 5
            k = int(rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 15]))
            if (m, n, k) not in seen and m * n - k > 0:
                seen.add((m, n, k))
                break
        A, pa = NAMES[int(rng.integers(len(NAMES)))]
        B = OTHERS[int(rng.integers(len(OTHERS)))]
        obj, set_ = THINGS[int(rng.integers(len(THINGS)))]
        q = TEMPLATES[tid].format(A=A, pa=pa, B=B, obj=obj, set=set_, mw=WORDS[m], n=n, k=k)
        out.append({"id": f"v{j:02d}_{tid}", "template": tid, "split": "test" if j < 10 else "train", "numeric": True,
                    "question": q, "prompt": q + SUFFIX, "answer": str(m * n - k), "m": m, "n": n, "k": k})
    return out


STOP = "Question:"


def few_shot_prefix(shots=5):
    import spec_acceptance_gsm8k as sg
    train = [json.loads(l) for l in (ROOT / "benchmarks/gsm8k/train_few_shot.jsonl").read_text().split("\n") if l.strip()]
    return sg.few_shot_prefix(train, shots)


def prompt_ids(tok, render, it, fmt, thinking=False):
    """Token ids for an item; render(content, thinking) is the chat renderer for fmt == chat."""
    if fmt == "chat":
        return render(it["prompt"], thinking)
    text = few_shot_prefix() + f"Question: {it['question']}\nAnswer:"
    return tok.encode(text, add_special_tokens=True)


def pred_of(text, fmt):
    if fmt == "chat":
        import amoeba_transplant_screen as sc
        return sc.extract_final(text)
    import spec_acceptance_gsm8k as sg
    return sg.extract_answer(text.split(STOP, 1)[0])


def stopped(text, fmt):
    return fmt != "chat" and STOP in text


def num_of(pred):
    if pred is None:
        return None
    s = pred.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = m[0]
    return v[:-2] if v.endswith(".0") else v


def correct(pred, it):
    if it.get("numeric"):
        return num_of(pred) == it["answer"]
    return pred == it["answer"]


def screen(args):
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if args.mem_cap_gb:
        mx.set_memory_limit(int(args.mem_cap_gb * 1e9))
    import amoeba_hub_align as ha
    from mlx2.adapters.registry import inspect_model
    from mlx2.serving import render_prompt_tokens
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ad = inspect_model(args.model).adapter_type(str(args.model))
    if args.cpu:
        ha.dequantize_model(mx, ad.model, mx.float32)
    model, tok = ad.model, ad.tokenizer
    ck = f"{Path(args.model).name}@{ad.identity['fingerprint']}"
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    its = [i for i in items(args.seed) if args.split == "all" or i["split"] == args.split]
    if args.only:
        its = [i for i in its if i["id"] in set(args.only.split(","))]
    eos = set(tok.eos_token_ids)
    print(f"screen {ck} rev {rev} dirty {dirty} items {len(its)} thinking {args.thinking} load {time.time() - t0:.0f}s",
          flush=True)
    rows, per = open(out / "rows.jsonl", "w"), {}
    render = lambda c, th: [int(t) for t in render_prompt_tokens(  # noqa: E731
        ad, {"messages": [{"role": "user", "content": c}], "enable_thinking": th})]
    if args.format != "chat":
        args.max_tokens = min(args.max_tokens, 256)  # the harness default
    for it in its:
        p = prompt_ids(tok, render, it, args.format, args.thinking)
        for s in range(1 + args.samples):
            if time.time() - t0 > args.budget_s:
                break
            mx.random.seed(args.seed + 1000 * s + len(per))
            cache, gen = model.make_cache(), []
            lg = model(mx.array([p]), cache=cache)[0, -1]
            tg = time.time()
            for _ in range(args.max_tokens):
                if s == 0:
                    t = int(mx.argmax(lg).item())
                else:
                    t = int(mx.random.categorical(lg.astype(mx.float32) / args.temperature).item())
                if t in eos:
                    break
                gen.append(t)
                if args.format != "chat" and len(gen) % 4 == 0 and stopped(tok.decode(gen), args.format):
                    break
                lg = model(mx.array([[t]]), cache=cache)[0, -1]
            del cache
            mx.clear_cache()
            text = tok.decode(gen)
            pred = pred_of(text, args.format)
            ok = correct(pred, it)
            dt = time.time() - tg
            r = {"item": it["id"], "template": it["template"], "sample": s, "greedy": s == 0, "pred": pred,
                 "answer": it["answer"], "correct": ok, "tokens": len(gen), "gen_s": round(dt, 1),
                 "tok_s": round(len(gen) / max(dt, 1e-9), 2), "hit_cap": len(gen) >= args.max_tokens, "text": text}
            rows.write(json.dumps(r) + "\n")
            rows.flush()
            per.setdefault(it["id"], []).append(ok)
            print(f"{time.strftime('%H:%M:%S')} {it['id']} s{s}: pred {pred!r} gold {it['answer']} ok {ok} "
                  f"tokens {len(gen)} ({dt:.0f}s)", flush=True)
    rows.close()
    orig = per.get("gsm549", [])
    summ = {"per_item": {k: f"{sum(v)}/{len(v)}" for k, v in per.items()},
            "greedy_acc": round(float(np.mean([v[0] for v in per.values()])), 4) if per else None,
            "all_samples_acc": round(float(np.mean([x for v in per.values() for x in v])), 4) if per else None,
            "solves_549_reliably": bool(orig) and all(orig) and len(orig) >= 3}
    (out / "run.json").write_text(json.dumps({
        "schema": "amoeba-gsm549-screen/v1", "checkpoint": ck, "mlx2_source_revision": rev, "tree_dirty": dirty,
        "args": vars(args), "summary": summ, "wall_s": round(time.time() - t0),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2)}, indent=1))
    print(json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("items", help="print the item set as JSON lines")
    v.add_argument("--seed", type=int, default=20261115)
    s = sub.add_parser("screen")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--split", choices=("test", "train", "all"), default="test")
    s.add_argument("--only", default="")
    s.add_argument("--samples", type=int, default=2)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--thinking", action="store_true")
    s.add_argument("--format", choices=("chat", "gsm8k5shot"), default="chat")
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--budget-s", type=float, default=3300)
    s.add_argument("--seed", type=int, default=20261115)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--mem-cap-gb", type=float, default=0)
    a = ap.parse_args()
    if a.cmd == "items":
        for it in items(a.seed):
            print(json.dumps(it))
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        screen(a)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda n, f: (_ for _ in ()).throw(SystemExit(f"terminated by signal {n}")))
    main()
