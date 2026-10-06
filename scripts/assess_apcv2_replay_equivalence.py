#!/usr/bin/env python3
"""Fail-closed APCv2 cold/warm replay-equivalence verdict.

Width one remains token-exact.  At physical width greater than one, a first
divergence may be classified as ``near_tie_equivalent`` only from bounded
first-divergence evidence.  A matched ordinary-versus-ordinary control is
optional attribution and rate evidence; it is not an equivalence gate.
This script evaluates already-captured evidence; it never loads a model or
constructs Metal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "mlx2.apcv2-replay-equivalence-input.v1"
OUTPUT_SCHEMA = "mlx2.apcv2-replay-equivalence-verdict.v1"
DEFAULT_MARGIN_NATS = 0.5
DEFAULT_ALPHA = 0.05
REQUIRED_STATE_CHECKS = (
    "revision_bound",
    "cache_layout_bound",
    "prompt_boundary_bound",
    "warm_hit",
    "stream_complete",
)


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _fisher_one_sided(candidate_hits: int, candidate_n: int,
                      control_hits: int, control_n: int) -> float:
    """P(candidate excess this large or larger | one shared rate)."""
    total_hits = candidate_hits + control_hits
    total = candidate_n + control_n
    if total == 0 or total_hits == 0 or total_hits == total:
        return 1.0

    def probability(hits: int) -> float:
        return (
            math.comb(candidate_n, hits)
            * math.comb(control_n, total_hits - hits)
            / math.comb(total, total_hits)
        )

    lower = max(0, total_hits - control_n)
    upper = min(candidate_n, total_hits)
    return sum(
        probability(hits)
        for hits in range(candidate_hits, upper + 1)
        if lower <= hits <= upper
    )


def _minimum_detectable_excess(candidate_n: int, control_n: int,
                               alpha: float) -> int | None:
    for hits in range(candidate_n + 1):
        if _fisher_one_sided(hits, candidate_n, 0, control_n) < alpha:
            return hits
    return None


def _top_two(arm: Any, label: str, problems: list[str]) -> tuple[int, set[int], float] | None:
    if not isinstance(arm, dict):
        problems.append(f"{label} arm is missing")
        return None
    selected = arm.get("selected_token_id")
    rows = arm.get("top_two")
    if type(selected) is not int or not isinstance(rows, list) or len(rows) != 2:
        problems.append(f"{label} must record selected_token_id and exactly two logits")
        return None
    parsed: list[tuple[int, float]] = []
    for row in rows:
        token = row.get("token_id") if isinstance(row, dict) else None
        logprob = row.get("logprob") if isinstance(row, dict) else None
        if type(token) is not int or not _finite(logprob):
            problems.append(f"{label} top-two row is malformed")
            return None
        parsed.append((token, float(logprob)))
    if parsed[0][0] == parsed[1][0] or selected not in {row[0] for row in parsed}:
        problems.append(f"{label} top-two token ids are not two distinct tokens including selection")
        return None
    selected_lp = next(lp for token, lp in parsed if token == selected)
    alternate_lp = next(lp for token, lp in parsed if token != selected)
    margin = selected_lp - alternate_lp
    if margin < -1e-6:
        problems.append(f"{label} selected token is below its recorded alternative")
    return selected, {row[0] for row in parsed}, margin


def _classify_row(row: Any, margin_ceiling: float) -> dict[str, Any]:
    problems: list[str] = []
    if not isinstance(row, dict):
        return {"classification": "failed", "problems": ["row is not an object"]}
    width = row.get("physical_width")
    cold = row.get("cold_token_ids")
    warm = row.get("warm_token_ids")
    if type(width) is not int or width < 1:
        problems.append("physical_width must be a positive integer")
    if not isinstance(cold, list) or not all(type(token) is int for token in cold):
        problems.append("cold_token_ids must be an integer list")
        cold = []
    if not isinstance(warm, list) or not all(type(token) is int for token in warm):
        problems.append("warm_token_ids must be an integer list")
        warm = []
    if not cold or not warm:
        problems.append("cold and warm token traces must be nonempty")
    exact = cold == warm
    if exact:
        return {
            "prompt_sha256": row.get("prompt_sha256"),
            "physical_width": width,
            "classification": "exact_token_parity" if not problems else "failed",
            "problems": problems,
        }
    if width == 1:
        problems.append("width-one cold/warm token identity is strict")

    divergence = row.get("first_divergence")
    if not isinstance(divergence, dict):
        problems.append("mismatch lacks first_divergence evidence")
        divergence = {}
    index = divergence.get("index")
    expected_index = next(
        (i for i, (left, right) in enumerate(zip(cold, warm)) if left != right),
        min(len(cold), len(warm)),
    )
    if type(index) is not int or index != expected_index:
        problems.append(
            f"first_divergence index {index!r} does not match token traces ({expected_index})"
        )
    if divergence.get("shared_prefix_exact") is not True:
        problems.append("shared prefix was not recorded exact")
    if type(index) is int and index >= min(len(cold), len(warm)):
        problems.append("length-only divergence is not near-tie evidence")

    cold_top = _top_two(divergence.get("cold"), "cold", problems)
    warm_top = _top_two(divergence.get("warm"), "warm", problems)
    margins: dict[str, float] = {}
    if cold_top and warm_top:
        cold_selected, cold_set, cold_margin = cold_top
        warm_selected, warm_set, warm_margin = warm_top
        margins = {"cold": cold_margin, "warm": warm_margin}
        if cold_set != warm_set:
            problems.append("cold and warm unordered top-two token sets differ")
        if cold_selected == warm_selected:
            problems.append("recorded selections do not describe a token flip")
        if type(index) is int and 0 <= index < min(len(cold), len(warm)):
            if cold_selected != cold[index] or warm_selected != warm[index]:
                problems.append("selected token ids do not match first-divergence traces")
        for label, margin in margins.items():
            if margin > margin_ceiling:
                problems.append(
                    f"{label} selected-versus-alternate margin {margin:.6g} exceeds "
                    f"{margin_ceiling:.6g} nats"
                )

    oracles = row.get("functional_oracles")
    if not isinstance(oracles, dict) or not oracles:
        problems.append("functional_oracles must be a nonempty object")
    elif any(value is not True for value in oracles.values()):
        problems.append("both continuations did not pass every functional oracle")

    return {
        "prompt_sha256": row.get("prompt_sha256"),
        "physical_width": width,
        "classification": "near_tie_equivalent" if not problems else "failed",
        "first_divergence_index": index,
        "margins_nats": margins,
        "problems": problems,
    }


def verdict(evidence: dict[str, Any], *, margin_ceiling: float = DEFAULT_MARGIN_NATS,
            alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    failures: list[str] = []
    if evidence.get("schema") != INPUT_SCHEMA:
        failures.append(f"input schema must be {INPUT_SCHEMA}")
    identity = evidence.get("identity")
    if not isinstance(identity, dict):
        failures.append("identity is missing")
        identity = {}
    for key in ("model", "profile", "cache_layout"):
        if not isinstance(identity.get(key), str) or not identity[key]:
            failures.append(f"identity.{key} is missing")
    for key in ("artifact_sha256", "runtime_source_sha256", "serving_shape_sha256"):
        if not _sha256(identity.get(key)):
            failures.append(f"identity.{key} must be a sha256")

    checks = evidence.get("state_checks")
    if not isinstance(checks, dict):
        checks = {}
    for name in REQUIRED_STATE_CHECKS:
        if checks.get(name) is not True:
            failures.append(f"state check failed or missing: {name}")

    rows = evidence.get("rows")
    if not isinstance(rows, list) or not rows:
        failures.append("rows must be a nonempty list")
        rows = []
    classified = [_classify_row(row, margin_ceiling) for row in rows]
    for index, row in enumerate(classified):
        failures.extend(f"row {index}: {problem}" for problem in row["problems"])

    near_ties = sum(row["classification"] == "near_tie_equivalent" for row in classified)
    candidate_n = len(classified)
    control = evidence.get("ordinary_control")
    differential: dict[str, Any]
    if near_ties:
        differential = {
            "candidate_mismatches": near_ties,
            "candidate_comparisons": candidate_n,
            "control_required_for_equivalence": False,
            "control_present": isinstance(control, dict),
            "alpha": alpha,
            "comparable": False,
            "passed": None,
            "interpretation": "diagnostic only; does not gate numerical equivalence",
        }
        if isinstance(control, dict):
            control_hits = control.get("mismatches")
            control_n = control.get("comparisons")
            shape_matches = (
                control.get("serving_shape_sha256")
                == identity.get("serving_shape_sha256")
            )
            counts_valid = (
                type(control_hits) is int
                and type(control_n) is int
                and control_n >= 1
                and 0 <= control_hits <= control_n
            )
            differential["control_shape_matches"] = shape_matches
            differential["control_counts_valid"] = counts_valid
            if shape_matches and counts_valid:
                p_value = _fisher_one_sided(
                    near_ties, candidate_n, control_hits, control_n
                )
                differential.update({
                    "control_mismatches": control_hits,
                    "control_comparisons": control_n,
                    "comparable": True,
                    "p_value": p_value,
                    "minimum_detectable_excess_against_clean_control": (
                        _minimum_detectable_excess(candidate_n, control_n, alpha)
                    ),
                    "passed": p_value >= alpha,
                    "material_excess_observed": p_value < alpha,
                })
    else:
        differential = {
            "candidate_mismatches": 0,
            "candidate_comparisons": candidate_n,
            "control_required_for_equivalence": False,
            "passed": True,
        }

    return {
        "schema": OUTPUT_SCHEMA,
        "identity": identity,
        "margin_ceiling_nats": margin_ceiling,
        "differential": differential,
        "rows": classified,
        "passed": bool(classified) and not failures,
        "failures": failures,
        "qualification_claim": False,
        "semantics": (
            "host-only replay-equivalence verdict; a pass is one companion gate, "
            "not model or route qualification"
        ),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f"{path.name}.part-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--margin-nats", type=float, default=DEFAULT_MARGIN_NATS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()
    if not _finite(args.margin_nats) or args.margin_nats < 0:
        parser.error("--margin-nats must be finite and nonnegative")
    if not _finite(args.alpha) or not 0 < args.alpha < 1:
        parser.error("--alpha must be in (0, 1)")
    raw = args.evidence.read_bytes()
    evidence = json.loads(raw)
    result = verdict(evidence, margin_ceiling=args.margin_nats, alpha=args.alpha)
    result["input_file"] = {
        "path": str(args.evidence.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
