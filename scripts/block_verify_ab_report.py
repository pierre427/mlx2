#!/usr/bin/env python3
"""Summarize a block-verification A/B receipt (scripts/block_verify_ab_unit_20261126.sh).

Per arm: GSM8K accuracy, mean accepted length, decode tok/s median, wall
throughput (completion tokens over the arm's request span, the B>1 number),
and whether the block arm's route receipts name exact block verification.
Per verification mode: B1 vs B4 row parity (same seed per test index; the
fast-route contract asks B1 == B4).
"""
import json
import sys


def wall_throughput(rows):
    spans = [(r["started_at"], r["started_at"] + r["elapsed_seconds"]) for r in rows
             if r.get("started_at") is not None and r.get("elapsed_seconds") is not None]
    if not spans:
        return None
    span = max(end for _, end in spans) - min(start for start, _ in spans)
    return sum(r.get("completion_tokens") or 0 for r in rows) / span if span > 0 else None


def main(path):
    receipt = json.load(open(path))
    arms = {arm["name"]: arm for arm in receipt["arms"]}
    out = {"checkpoint": receipt.get("checkpoint"), "driver_revision": receipt.get("driver_revision"),
           "arms": {}, "parity": {}}
    for name, arm in arms.items():
        summary = arm.get("summary") or {}
        rows = arm.get("rows") or []
        acceptance = summary.get("acceptance") or {}
        out["arms"][name] = {
            "passed": arm.get("passed"), "error": arm.get("error"),
            "revision": (arm.get("revision") or {}).get("head"),
            "concurrency": arm.get("concurrency", 1),
            "questions": summary.get("questions"), "accuracy": summary.get("accuracy"),
            "mean_accepted_length": acceptance.get("mean_accepted_length"),
            "per_position": acceptance.get("per_position"),
            "decode_tps_median": summary.get("decode_tps_median"),
            "wall_tok_s": wall_throughput(rows),
            "routes": summary.get("routes"),
            # mlx2.speculation.verification names the verifier per request.
            "block_receipt_requests": sum(r.get("verification") == "exact_block" for r in rows),
            "block_verified_rounds": sum(r.get("block_verified_rounds") or 0 for r in rows),
        }
    for mode in ("token", "block"):
        one, four = arms.get(f"{mode}-b1"), arms.get(f"{mode}-b4")
        if not (one and four and one.get("rows") and four.get("rows")):
            continue
        a = {r["test_index"]: r["text"] for r in one["rows"]}
        b = {r["test_index"]: r["text"] for r in four["rows"]}
        common = sorted(set(a) & set(b))
        differ = [i for i in common if a[i] != b[i]]
        out["parity"][mode] = {"rows": len(common), "identical": len(common) - len(differ),
                               "differing_test_indices": differ}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
