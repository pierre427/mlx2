#!/usr/bin/env python3
"""Evaluate one (quantized or reference) artifact against a layer_kl reference: held-out
perplexity, KL(reference || model), top-1 agreement, plus a short functional check.

Original mlx2 script.  The caller owns the GPU (CPG lease + paired locks; ``--i-own-gpu``).
The model loads through its mlx2 adapter (the serving load path), so a planned artifact
that evaluates here also loads in serving.  Sanity numbers, single pass, no thermal hold.

    knapsack_eval.py --model DIR --reference REFDIR --out result.json --i-own-gpu [--splits held]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

# Raw-completion prompts (no chat template: the target is a thinking model) with substring
# oracles.  Greedy, ``--max-new`` tokens.  Factual, arithmetic and code, as in the external
# result's split (factual perplexity moved most there).
PROMPTS = [
    ("fact-capital", "The capital of France is the city of", ["Paris"]),
    ("fact-gold", "The chemical symbol for the element gold is \"", ["Au\""]),
    ("fact-planet", "The largest planet in our solar system is", ["Jupiter"]),
    ("fact-austen", "The novel Pride and Prejudice was written by", ["Austen"]),
    ("fact-everest", "The highest mountain on Earth above sea level is Mount", ["Everest"]),
    ("fact-water", "At sea level, pure water boils at a temperature of", ["100 degrees", "100°", "100 °",
                                                                          "212 degrees", "212°"]),
    ("math-mul", "Question: What is 12 multiplied by 13?\nAnswer: 12 * 13 =", ["156"]),
    ("math-add", "Question: What is 245 plus 378?\nAnswer: 245 + 378 =", ["623"]),
    ("math-sub", "Question: What is 1000 minus 357?\nAnswer: 1000 - 357 =", ["643"]),
    ("code-add", "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return", ["a + b", "a+b"]),
    ("code-rev", "def reverse(xs):\n    \"\"\"Return the list xs reversed.\"\"\"\n    return",
     ["xs[::-1]", "reversed(xs)", "list(reversed"]),
    ("code-def", "In Python, a function is defined with the keyword \"", ["def\""]),
]


def evaluate(model, reference: Path, splits, K):
    import mlx.core as mx

    meta, texts = K.load_reference(reference)
    skip = int(meta["skip"])
    total = {"kl": 0.0, "top1": 0, "pos": 0, "nll": 0.0, "nll_pos": 0}
    per_text = []
    for row, text in zip(meta["texts"], texts):
        if row["split"] not in splits:
            continue
        ids = list(map(int, text["ids"]))
        ref = np.load(Path(reference) / row["file"], mmap_mode="r")
        logits = model(mx.array([ids]))[0, skip:]
        if tuple(logits.shape) != tuple(ref.shape):
            raise SystemExit(f"model logits {tuple(logits.shape)} do not match the reference "
                             f"{tuple(ref.shape)} for {text['id']!r}")
        logp = K._log_softmax(mx, logits)
        del logits
        targets = mx.array(ids[skip + 1:])
        nll = -mx.take_along_axis(logp[:-1], targets[:, None], axis=-1).sum()
        kl_sum, agree = 0.0, 0
        for start in range(0, ref.shape[0], 256):
            r = mx.array(np.asarray(ref[start:start + 256], dtype=np.float32))
            v = logp[start:start + 256]
            kl = (mx.exp(r) * (r - v)).sum(axis=-1)   # per row, as layer_kl.score
            same = (mx.argmax(r, axis=-1) == mx.argmax(v, axis=-1))
            mx.eval(kl, same)
            kl_sum += float(np.array(kl, dtype=np.float64).sum())
            agree += int(np.array(same).sum())
        mx.eval(nll)
        n = int(ref.shape[0])
        per_text.append({"id": text["id"], "split": row["split"], "kl_mean": kl_sum / n,
                         "top1": agree / n, "ppl": math.exp(float(nll.item()) / (n - 1))})
        total["kl"] += kl_sum
        total["top1"] += agree
        total["pos"] += n
        total["nll"] += float(nll.item())
        total["nll_pos"] += n - 1
        del logp
        mx.clear_cache()
    if not total["pos"]:
        raise SystemExit(f"no reference texts in splits {splits}")
    return {"splits": list(splits), "kl_mean": total["kl"] / total["pos"], "top1": total["top1"] / total["pos"],
            "ppl": math.exp(total["nll"] / total["nll_pos"]), "positions": total["pos"], "per_text": per_text}


def functional(model, tokenizer, max_new: int):
    import mlx.core as mx

    rows = []
    for name, prompt, oracles in PROMPTS:
        ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        out = []
        for _ in range(max_new):  # greedy, no cache: short prompts, sanity only
            nxt = int(mx.argmax(model(mx.array([ids + out]))[0, -1]).item())
            out.append(nxt)
        text = tokenizer.decode(out)
        rows.append({"name": name, "pass": any(o in text for o in oracles), "completion": text,
                     "tokens": out})
    return {"passed": sum(r["pass"] for r in rows), "total": len(rows), "rows": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", default="held")
    p.add_argument("--max-new", type=int, default=12)
    p.add_argument("--i-own-gpu", action="store_true")
    a = p.parse_args(argv)
    if not a.i_own_gpu:
        raise SystemExit("--i-own-gpu: hold the CPG lease and paired locks")
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    import mlx.core as mx
    from layer_kl import commit, load_adapter

    from mlx2 import layer_kl as K

    from mlx2.conversion.knapsack import checkpoint_name

    # Refuse a reference from another checkpoint before spending the load.
    plan_info = json.loads((a.model / "config.json").read_text()).get("knapsack_plan")
    ref_ckpt = K.load_reference(a.reference)[0]["checkpoint"]
    if plan_info and ref_ckpt.rpartition("@")[2] != plan_info["checkpoint"].rpartition("@")[2]:
        raise SystemExit(f"reference is {ref_ckpt}, but this artifact comes from {plan_info['checkpoint']}")
    t0 = time.monotonic()
    adapter, checkpoint, _ = load_adapter(a.model, None, wire=True)
    checkpoint = f"{checkpoint_name(a.model)}@{checkpoint.rpartition('@')[2]}"
    load_s = time.monotonic() - t0
    try:
        if not plan_info and ref_ckpt.rpartition("@")[2] != checkpoint.rpartition("@")[2]:
            raise SystemExit(f"reference is {ref_ckpt}, but this model is {checkpoint}")
        result = {"checkpoint": checkpoint, "mlx2": commit(), "knapsack_plan": plan_info,
                  "load_seconds": round(load_s, 2),
                  "mlx_active_gb_after_load": round(mx.get_active_memory() / 2**30, 3),
                  "eval": evaluate(adapter.model, a.reference, tuple(a.splits.split(",")), K),
                  "functional": functional(adapter.model, adapter.tokenizer, a.max_new),
                  "seconds": None}
    finally:
        adapter.close()
    result["seconds"] = round(time.monotonic() - t0, 2)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    e, f = result["eval"], result["functional"]
    print(json.dumps({"checkpoint": checkpoint, "ppl": e["ppl"], "kl_mean": e["kl_mean"], "top1": e["top1"],
                      "functional": f"{f['passed']}/{f['total']}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
