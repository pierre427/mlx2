#!/usr/bin/env python3
"""Summarize ``ab_qwen38_dflash2.py`` JSONL: medians, spread, tau, greedy gate.

Per arm x workload x width x temperature: median and min-max of per-request
decode tok/s over every rep and prompt, median TTFT, median aggregate cell
tok/s, and acceptance length per verify round (tau: tokens emitted per target
verify forward, bonus included).  The greedy gate compares every greedy
output with the ordinary arm's output for the same rep/cell/prompt (the
nonce is arm-invariant, so prompts are identical).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def tau(row):
    spec = row.get("speculation")
    if spec and spec.get("external_rounds"):
        return 1.0 + spec["accepted"] / spec["external_rounds"]
    mtp = row.get("mtp") or {}
    for rounds_key, accepted_key in (
        ("cycles", "accepted"), ("verify_rounds", "accepted_tokens"),
        ("rounds", "accepted"),
    ):
        if mtp.get(rounds_key) and accepted_key in mtp:
            return 1.0 + mtp[accepted_key] / mtp[rounds_key]
    hist = (spec or {}).get("verify_accept_hist") or mtp.get("verify_accept_hist")
    if hist:
        rounds = sum(hist.values())
        return 1.0 + sum(int(k) * v for k, v in hist.items()) / rounds
    return None


def load(paths):
    cells, summaries = [], []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            if "summary" in record:
                summaries.append(record)
            elif "arm" in record and "rows" in record:
                cells.append(record)
    return cells, summaries


def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def table(cells):
    groups = defaultdict(list)
    for cell in cells:
        groups[(cell["arm"], cell["workload"], cell["width"], cell["temperature"])].append(cell)
    out = {}
    for key, group in sorted(groups.items()):
        rows = [row for cell in group for row in cell["rows"]]
        out["/".join(map(str, key))] = {
            "decode_tok_s": spread([r["decode_tok_s"] for r in rows]),
            "aggregate_tok_s": spread([c["aggregate_tok_s"] for c in group]),
            "ttft_s": spread([r["ttft_s"] for r in rows]),
            "tau": spread([tau(r) for r in rows]),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
        }
    return out


def greedy_gate(cells):
    reference = {}
    for cell in cells:
        if cell["arm"] == "ord" and cell["temperature"] == 0:
            for row in cell["rows"]:
                key = (
                    cell["rep"],
                    cell["workload"],
                    cell["width"],
                    row["prompt_index"],
                )
                if key in reference:
                    raise ValueError(
                        f"duplicate ordinary greedy reference for {key}"
                    )
                reference[key] = row
    results = []
    for cell in cells:
        if cell["arm"] == "ord" or cell["temperature"] != 0:
            continue
        for row in cell["rows"]:
            key = (cell["rep"], cell["workload"], cell["width"], row["prompt_index"])
            ref = reference.get(key)
            entry = {"arm": cell["arm"], "rep": cell["rep"], "workload": cell["workload"],
                     "width": cell["width"], "prompt_index": row["prompt_index"]}
            if ref is None:
                entry.update({
                    "equal": False,
                    "reason": "missing_ordinary_reference",
                })
                results.append(entry)
                continue
            same = ref["output_sha256"] == row["output_sha256"]
            entry["equal"] = same
            if not same:
                a, b = ref["output"], row["output"]
                common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                entry.update({"first_diff_char": common, "ref_len": len(a), "arm_len": len(b),
                              "ref_context": a[max(0, common - 40):common + 20],
                              "arm_context": b[max(0, common - 40):common + 20]})
            results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    cells, summaries = load(args.paths)
    result = {"table": table(cells), "greedy_gate": greedy_gate(cells), "arm_summaries": summaries}
    gate = result["greedy_gate"]
    result["greedy_gate_counts"] = {
        "compared": len(gate), "equal": sum(e["equal"] for e in gate),
        "by_arm": {arm: [sum(e["equal"] for e in gate if e["arm"] == arm),
                         sum(1 for e in gate if e["arm"] == arm)]
                   for arm in sorted({e["arm"] for e in gate})},
    }
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    for key, value in result["table"].items():
        d, t, a = value["decode_tok_s"], value["ttft_s"], value["tau"]
        print(f"{key:28s} decode {d['median']:6.1f} [{d['min']:.1f}-{d['max']:.1f}] n={d['n']:2d} "
              f"agg {value['aggregate_tok_s']['median']:6.1f} ttft {t['median']:.2f}s "
              f"tau {a['median'] if a else float('nan'):.2f}")
    print(json.dumps(result["greedy_gate_counts"]))


if __name__ == "__main__":
    main()
