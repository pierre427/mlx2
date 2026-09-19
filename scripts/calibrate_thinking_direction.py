"""Calibrate and evaluate a thinking "commit direction" (the TAS alpha actuator).

Thin CLI over `mlx2.thinking_calibration` (the same code the server runs when it
calibrates an artifact by itself at startup):

  trace    greedy natural traces with thinking on; keep the ones that close
  extract  positional commit-minus-reflect direction per layer, bound to the
           artifact's identity (`<out>.json` names it; a direction is only ever
           loaded for the artifact it was measured on)
  grid     held-out prompts x arms: off, always-on alpha, two-mode, and
           same-norm RANDOM directions on the same schedule

Nothing here touches a served route.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "src")
from mlx2.thinking_calibration import (  # noqa: E402
    CALIBRATION_PROMPTS as CALIBRATION, SCHEMA, artifact_identity, choose_layer, extract_directions,
    generate, is_correct,
)

HELD_OUT = [  # (name, prompt, expected substring or None, kind)
    ("explain_9", "Explain in five sentences what compiler optimization number 9 of a typical -O2 pipeline might do.", None, "run_on"),
    ("explain_13", "Explain in five sentences what compiler optimization number 13 of a typical -O2 pipeline might do.", None, "run_on"),
    ("explain_15", "Explain in five sentences what compiler optimization number 15 of a typical -O2 pipeline might do.", None, "run_on"),
    ("explain_18", "Explain in five sentences what compiler optimization number 18 of a typical -O2 pipeline might do.", None, "run_on"),
    ("translate_house", "Translate the English word 'house' to French, then use the French word in two short French sentences.", "maison", "run_on"),
    ("capital_hungary", "What is the capital of Hungary? One word.", "budapest", "run_on"),
    ("add_62_130", "What is 62 + 130? Reply with the number only.", "192", "control"),
    ("add_44_88", "What is 44 + 88? Reply with the number only.", "132", "control"),
    ("paris_coastal", "Is Paris a coastal city? Answer yes or no.", "no", "control"),
    ("primes_below_60", "How many prime numbers are there below 60? Reply with the number only.", "17", "hard"),
    ("digit_sum", "What is the sum of the digits of 2 to the power of 20? Reply with the number only.", "31", "hard"),
    ("train_meet", "Two trains start 300 km apart and head toward each other at 70 km/h and 80 km/h. After how many hours do they meet? Reply with the number only.", "2", "hard"),
    ("sort_words", "Sort these words alphabetically and reply with them comma-separated: pear, apple, mango, fig, banana, cherry.", "apple, banana, cherry, fig, mango, pear", "hard"),
    ("mod_pow", "What is 7 to the power of 5, modulo 13? Reply with the number only.", "11", "hard"),
    ("count_r", "How many times does the letter r appear in the word 'strawberry'? Reply with the number only.", "3", "hard"),
    ("bat_ball", "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How many cents does the ball cost? Reply with the number only.", "5", "hard"),
]


def load(path):
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

    return NorthMiniCodeAdapter(path)


def correct(row, expected):
    return is_correct(row, expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["trace", "extract", "grid"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces")
    ap.add_argument("--vectors")
    ap.add_argument("--layers", default="12,16,20,24,28,32,36,40,44")
    ap.add_argument("--max-think", type=int, default=1500)
    ap.add_argument("--commit-tail", type=int, default=24)
    ap.add_argument("--reflect-frac", type=float, default=0.45)
    ap.add_argument("--arms", help="JSON list of arms for grid mode")
    args = ap.parse_args()
    adapter = load(args.model)
    close_id = adapter.thinking_close_token_ids()[0]
    eos = adapter.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    layers = [int(x) for x in args.layers.split(",")]

    if args.mode == "trace":
        with open(args.out, "w") as out:
            for index, (prompt, expected) in enumerate(CALIBRATION):
                started = time.time()
                row = generate(adapter, prompt, close_id=close_id, eos_ids=eos_ids, max_think=args.max_think)
                row.update(index=index, prompt=prompt, correct=correct(row, expected))
                out.write(json.dumps(row) + "\n"); out.flush()
                print(f"[trace] {index:2d} close@{row['close_step']:5d} n={len(row['token_ids']):5d} correct={row['correct']} rep4={row['rep4']} {time.time()-started:5.1f}s | {row['answer'][:50]!r}", flush=True)
        return

    if args.mode == "extract":
        traces = [json.loads(line) for line in open(args.traces)]
        arrays, report = extract_directions(adapter, traces, layers, commit_tail=args.commit_tail, reflect_frac=args.reflect_frac,
                                            progress=lambda tr: print(f"[extract] trace {tr['index']:2d} close@{tr['close_step']}", flush=True))
        for L, row in report["layers"].items():
            print(f"  L{int(L):2d} rms={row['rms']:8.2f} rel={row['relative_norm']:.3f} consistency={row['consistency']:+.3f}", flush=True)
        report.update(schema=SCHEMA, artifact_identity=artifact_identity(args.model), origin="manual",
                      layer=choose_layer(report, int(adapter.model.args.num_hidden_layers)))
        np.savez(args.out, **arrays)
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
        return

    z = np.load(args.vectors)
    arms = json.loads(args.arms)
    rng = np.random.default_rng(20260918)
    results = {}
    for arm in arms:
        name, L = arm["name"], arm.get("layer")
        vec_low = vec_high = None
        if L is not None:
            vhat = z[f"v_{L}"]
            if arm.get("random"):
                r = rng.standard_normal(vhat.shape); vhat = (r / np.linalg.norm(r)).astype(np.float32)
            scale = float(z[f"rms_{L}"])
            vec_low = mx.array(arm["alpha"] * scale * vhat)[None, None, :]
            if arm.get("hammer"):
                vec_high = mx.array(arm["hammer"] * scale * vhat)[None, None, :]

        def steer(step, L=L, lo=vec_low, hi=vec_high, arm=arm):
            if L is None or step < arm.get("from", 0):
                return None
            return (L, hi if (hi is not None and step >= arm.get("hammer_from", 10**9)) else lo)

        rows = {}
        for case, prompt, expected, kind in HELD_OUT:
            started = time.time()
            row = generate(adapter, prompt, close_id=close_id, eos_ids=eos_ids, max_think=args.max_think, steer=steer if L is not None else None)
            row.update(kind=kind, correct=correct(row, expected), seconds=round(time.time() - started, 1))
            row["think_tokens"] = row["close_step"] if row["close_step"] >= 0 else len(row["token_ids"])
            del row["token_ids"]
            rows[case] = row
            print(f"[grid] {name:22s} {case:16s} {kind:8s} {'OK ' if row['correct'] else ('ans' if row['answer'] else '---')} think={row['think_tokens']:5d} close={row['close_step']>=0} rep4={row['rep4']} | {row['answer'][:48]!r}", flush=True)
        results[name] = {"arm": arm, "rows": rows,
                         "closed": sum(r["close_step"] >= 0 for r in rows.values()), "correct": sum(r["correct"] for r in rows.values()),
                         "think_tokens": sum(r["think_tokens"] for r in rows.values())}
        print(f"== {name}: closed {results[name]['closed']}/{len(rows)} correct {results[name]['correct']}/{len(rows)} think_tokens {results[name]['think_tokens']}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
