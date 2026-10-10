#!/usr/bin/env python3
"""Read-only reassessment of a controlled cold/warm prompt-boundary gate.

Never launches inference, rewrites the input receipt, or changes qualification.
The source run and producer remain immutable even when their gate was wrong.
"""
import argparse
import hashlib
import json
from pathlib import Path

from mlx2.cache_receipts import prompt_boundary_replay_evidence


def assess(report):
    arms = {arm["arm"]: arm for arm in report["arms"]}
    cold = {row["lane"]: row for row in arms["cold"]["rows"]}
    warm = {row["lane"]: row for row in arms["warm"]["rows"]}
    recipes = {row["lane"]: row for row in report["prompt_recipes"]}
    errors = []
    if (len(cold) != len(arms["cold"]["rows"])
            or len(warm) != len(arms["warm"]["rows"])
            or len(recipes) != len(report["prompt_recipes"])):
        errors.append("duplicate lane identities cannot establish paired replay")
    if not cold or cold.keys() != warm.keys() or cold.keys() != recipes.keys():
        errors.append("cold/warm/recipe lane sets are incomplete or differ")
    initial = report.get("server_initial_status", {}).get("apcv2", {})
    if any(initial.get(field) != 0 for field in ("lookups", "hits", "stores")):
        errors.append("initial APCv2 activity is not proven empty")
    if initial.get("persistence", {}).get("enabled") is not False:
        errors.append("persistent cache import was not excluded")
    evidence = {}
    for lane in sorted(cold.keys() & warm.keys() & recipes.keys()):
        c, w = cold[lane], warm[lane]
        result = prompt_boundary_replay_evidence(
            c["route_receipt"], w["route_receipt"],
            expected_prompt_tokens=recipes[lane]["prompt_tokens"],
            cold_prompt_sha256=c.get("prompt_sha256"),
            warm_prompt_sha256=w.get("prompt_sha256"),
        )
        evidence[str(lane)] = result
        errors.extend(f"lane {lane}: {error}" for error in result["errors"])
    return {
        "schema": "mlx2.prompt-boundary-gate-review.v1",
        "scope": "prompt_boundary_replay_only",
        "source_revision": report.get("source_revision"),
        "original_state": report.get("state"),
        "original_failures": report.get("failures"),
        "prompt_boundary_gate": "failed" if errors else "passed",
        "lanes": evidence,
        "errors": errors,
        "qualification_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    review = assess(json.loads(raw))
    review["source_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(review, indent=2, sort_keys=True))
    return int(review["prompt_boundary_gate"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
