#!/usr/bin/env python3
"""Pairwise output identity of spec_acceptance_gsm8k arms (one or more receipts).

Each arm is named ``<receipt label>/<arm name>``; for every requested pair it
reports how many questions have byte-identical completion text, and for the
others the first differing character offset.  Text identity is the cheap
greedy-parity check (identical text from identical prompts means the same
tokens were emitted up to tokenizer round-trip); a differing row still needs
the near-tie localization of AGENTS.md before it is called equivalent.

Usage: compare_gsm8k_arm_texts.py label=receipt.json [label=receipt.json ...]
         [--pairs a/x:b/y ...] [--out compare.json]
Without --pairs every arm is compared with the first arm of the first receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(specs):
    arms = {}
    meta = {}
    for spec in specs:
        label, path = spec.split("=", 1)
        receipt = json.loads(Path(path).read_text())
        meta[label] = {"checkpoint": receipt.get("checkpoint"), "path": path}
        for arm in receipt.get("arms", []):
            name = f"{label}/{arm['name']}"
            rows = {r.get("test_index", r.get("index")): r for r in arm.get("rows", [])}
            arms[name] = {"rows": rows, "passed": arm.get("passed"), "error": arm.get("error"),
                          "revision": arm.get("revision"),
                          "routes": sorted({str(r.get("route")) for r in rows.values()}),
                          "accuracy": (sum(bool(r.get("correct")) for r in rows.values()) / len(rows))
                          if rows else None}
    return arms, meta


def first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def compare(arms, left, right):
    a, b = arms[left]["rows"], arms[right]["rows"]
    common = sorted(set(a) & set(b), key=lambda k: (str(type(k)), k))
    diffs = []
    for key in common:
        ta, tb = a[key].get("text") or "", b[key].get("text") or ""
        if ta != tb:
            diffs.append({"test_index": key, "first_diff_char": first_diff(ta, tb),
                          "len": [len(ta), len(tb)],
                          "correct": [a[key].get("correct"), b[key].get("correct")]})
    return {"left": left, "right": right, "questions": len(common),
            "identical": len(common) - len(diffs), "differing": diffs,
            "accuracy": [arms[left]["accuracy"], arms[right]["accuracy"]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("receipts", nargs="+")
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--out")
    args = ap.parse_args()
    arms, meta = load(args.receipts)
    names = list(arms)
    pairs = [p.split(":", 1) for p in args.pairs] if args.pairs else [(names[0], n) for n in names[1:]]
    result = {"receipts": meta,
              "arms": {n: {k: v for k, v in a.items() if k != "rows"} for n, a in arms.items()},
              "pairs": [compare(arms, l, r) for l, r in pairs if l in arms and r in arms],
              "missing_pairs": [f"{l}:{r}" for l, r in pairs if l not in arms or r not in arms]}
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    for pair in result["pairs"]:
        print(f"{pair['left']} vs {pair['right']}: {pair['identical']}/{pair['questions']} identical, "
              f"accuracy {pair['accuracy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
