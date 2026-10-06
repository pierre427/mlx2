#!/usr/bin/env python3
"""Summarize the first deterministic checkpoint/continuation divergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def first_failed_step(probe: dict | None) -> dict | None:
    if not isinstance(probe, dict):
        return None
    thresholds = probe.get("thresholds") or {}
    for row in probe.get("steps") or []:
        if row.get("passed") is not False:
            continue
        reasons = []
        if row.get("argmax_equal") is False:
            reasons.append("argmax")
        if row.get("total_variation", 0) > thresholds.get(
            "total_variation_max", float("inf")
        ):
            reasons.append("total_variation")
        if row.get("serial_to_actual_kl", 0) > thresholds.get(
            "serial_to_actual_kl_max", float("inf")
        ):
            reasons.append("serial_to_actual_kl")
        logits = row.get("logits") or {}
        if logits.get("relative_l2", 0) > thresholds.get(
            "logit_relative_l2_max", float("inf")
        ):
            reasons.append("logit_relative_l2")
        if logits.get("normalized_max", 0) > thresholds.get(
            "logit_normalized_max", float("inf")
        ):
            reasons.append("logit_normalized_max")
        return {
            "step": int(row["step"]),
            "reasons": reasons,
            "argmax_equal": row.get("argmax_equal"),
            "hidden_first_exact_divergent_layer": (
                (row.get("hidden") or {}).get("first_exact_divergent_layer")
            ),
            "hidden_first_tolerance_failed_layer": (
                (row.get("hidden") or {}).get("first_tolerance_failed_layer")
            ),
            "cache_first_exact_divergent_layer": (
                (row.get("cache") or {}).get("first_exact_divergent_layer")
            ),
            "cache_first_tolerance_failed_layer": (
                (row.get("cache") or {}).get("first_tolerance_failed_layer")
            ),
        }
    return None


def summarize(result: dict) -> dict:
    failed_round = next(
        (row for row in result.get("rounds") or [] if row.get("equal") is False),
        None,
    )
    logical = (failed_round or {}).get("logical_state") or {}
    localized = (failed_round or {}).get("logical_state_divergence") or {}
    failed_layers = logical.get("failed_layers") or []
    categories = {
        name: first_failed_step(probe)
        for name, probe in sorted((result.get("continuation_categories") or {}).items())
    }
    planted = result.get("planted_deepest_branch") or {}
    return {
        "schema": "mlx2.tensorfold-checkpoint-divergence-summary.v1",
        "input_schema": result.get("schema"),
        "source_commit": result.get("source_commit"),
        "prompt_tokens": result.get("prompt_tokens"),
        "passed": result.get("passed"),
        "first_failed_round": (
            None
            if failed_round is None
            else {
                "round": failed_round.get("round"),
                "span": failed_round.get("span"),
                "accepted_drafts": failed_round.get("accepted_drafts"),
                "continuation_category": failed_round.get("continuation_category"),
                "first_exact_divergent_layer": localized.get(
                    "first_exact_divergent_layer"
                ),
                "first_tolerance_failed_layer": localized.get(
                    "first_tolerance_failed_layer",
                    failed_layers[0] if failed_layers else None,
                ),
                "transaction_oracle_failed_layers": (
                    (failed_round.get("transaction_commit_oracle") or {}).get(
                        "failed_layers"
                    )
                    or []
                ),
            }
        ),
        "continuation_categories": categories,
        "planted_deepest_branch": {
            "structure_passed": planted.get(
                "structure_passed",
                bool(
                    planted
                    and planted.get("draft_depth")
                    == planted.get("maximum_draft_depth")
                    and (planted.get("transaction_commit_oracle") or {}).get("passed")
                ),
            ),
            "first_failed_step": first_failed_step(planted.get("continuation")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summaries = [
        {"input": str(path), **summarize(json.loads(path.read_text()))}
        for path in args.inputs
    ]
    payload = {"results": summaries}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists():
            raise RuntimeError(f"refusing existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
