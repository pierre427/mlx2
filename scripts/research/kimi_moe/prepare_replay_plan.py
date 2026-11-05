"""Freeze selected first-divergence sites after all five corpus arms finish.

This is CPU-only preparation. It never acquires GPU ownership or launches models.
Missing candidate windows remain explicit recovery work in separate evidence.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib

import summarize

ARMS = ("adaptive10", "adaptive20", "adaptive30", "adaptive50", "adaptive70")


def tokens(row):
    response = row["attempts"][-1]["response"]
    ids = response["mlx2"]["probe_output_tokens"]
    if not ids or len(ids) != response["usage"]["completion_tokens"]:
        raise ValueError("Incomplete output token receipt")
    return ids


def inspect():
    base = summarize.rows([summarize.BASE], "default8")
    if len(base) != 400:
        raise ValueError("Expected exactly400 unique baseline cases")
    positions = collections.defaultdict(set)
    recoveries = []
    selected = []
    source_specs = {}
    counts = {}
    terminal_only = []
    for arm in ARMS:
        rows = summarize.rows(summarize.CAND, arm)
        counts[arm] = len(rows)
        if set(rows) - set(base):
            raise ValueError("Unknown candidate case identity")
        for cid, (path, row) in rows.items():
            base_path, ordinary = base[cid]
            if any(
                row["case"][key] != ordinary["case"][key]
                for key in ("case_id", "domain", "question")
            ):
                raise ValueError("Prompt identity drift: " + cid)
            for parent in (path.parent, base_path.parent):
                if str(parent) not in source_specs:
                    spec_path = parent / "spec.json"
                    spec = summarize.load(spec_path)
                    source_specs[str(parent)] = {
                        "source_revision": spec["source_revision"],
                        "source_snapshot_sha256": spec["source_snapshot_sha256"],
                        "checkpoint": spec["checkpoint"],
                        "spec_sha256": hashlib.sha256(
                            spec_path.read_bytes()
                        ).hexdigest(),
                    }
            if (
                source_specs[str(path.parent)]["checkpoint"]
                != source_specs[str(base_path.parent)]["checkpoint"]
            ):
                raise ValueError("Checkpoint identity drift")
            expected, candidate = tokens(ordinary), tokens(row)
            pos = next(
                (i for i, (x, y) in enumerate(zip(expected, candidate)) if x != y), None
            )
            comparison = row.get("baseline_comparison", {})
            if comparison and comparison.get("first_divergence") != pos:
                raise ValueError("Stored divergence differs from actual selected site")
            if pos is None:
                if expected != candidate:
                    terminal_only.append(
                        {"arm": arm, "case_id": cid, "receipt": str(path)}
                    )
                continue
            positions[cid].add(pos)
            window = comparison.get("candidate_window", {})
            actual = (
                window.get("response", {})
                .get("mlx2", {})
                .get("probe_output_tokens", [])
            )
            choices = window.get("response", {}).get("choices", [])
            logrows = (
                ((choices[0].get("logprobs") or {}).get("content") or [])
                if choices
                else []
            )
            bounded_logit = (
                len(logrows) == 1
                and logrows[0].get("id") == candidate[pos]
                and isinstance(logrows[0].get("logprob"), (int, float))
                and math.isfinite(logrows[0]["logprob"])
                and 0 < len(logrows[0].get("top_logprobs", [])) <= 20
            )
            verified = (
                window.get("position_zero_based") == pos
                and actual == candidate[: pos + 1]
                and window.get("replayed_prefix_and_selection") is True
                and bounded_logit
            )
            site = {
                "arm": arm,
                "case_id": cid,
                "position_zero_based": pos,
                "baseline_receipt": str(base_path),
                "candidate_receipt": str(path),
                "baseline_output_tokens_sha256": hashlib.sha256(
                    json.dumps(expected).encode()
                ).hexdigest(),
                "candidate_output_tokens_sha256": hashlib.sha256(
                    json.dumps(candidate).encode()
                ).hexdigest(),
                "candidate_window_verified": verified,
            }
            selected.append(site)
            if not verified:
                recoveries.append(
                    {
                        **site,
                        "reason": "Missing or unverified candidate first-divergence window; recover in fresh evidence, never overwrite original.",
                    }
                )
    cases = [
        {**base[cid][1]["case"], "replay_positions": sorted(positions[cid])}
        for cid in sorted(base)
        if cid in positions
    ]
    return {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "ready_for_final_plan": all(counts.get(a) == 400 for a in ARMS),
        "counts": counts,
        "selected_case_count": len(cases),
        "selected_policy_sites": selected,
        "source_specs": source_specs,
        "unique_position_count": sum(len(c["replay_positions"]) for c in cases),
        "candidate_window_recovery": recoveries,
        "terminal_only_changes": terminal_only,
        "cases": cases,
        "scope": "Intentional cross-policy first-divergence comparisons, not APCv2 equivalence. Baseline replay must verify the entire prefix and selected token before interpreting the bounded window. Earlier source/profile drift stays unclassified pending its own first-divergence diagnosis.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--maximum-replay-tokens", type=int, default=15000)
    p.add_argument("--maximum-cases", type=int, default=40)
    args = p.parse_args()
    report = inspect()
    compact = {
        k: report[k]
        for k in (
            "ready_for_final_plan",
            "counts",
            "selected_case_count",
            "unique_position_count",
        )
    }
    compact["candidate_windows_to_recover"] = len(report["candidate_window_recovery"])
    print(json.dumps(compact))
    if args.inspect_only:
        return
    if not report["ready_for_final_plan"]:
        p.error("Final plan refused until all five arms have400 unique saved responses")
    if args.output is None or args.maximum_replay_tokens < 1 or args.maximum_cases < 1:
        p.error("Fresh output directory and positive unit bounds are required")
    batches, batch, cost = [], [], 0
    for case in report["cases"]:
        amount = sum(pos + 1 for pos in case["replay_positions"])
        if amount > args.maximum_replay_tokens:
            p.error(
                "One case exceeds the replay-token budget; split its preselected sites explicitly"
            )
        if batch and (
            len(batch) >= args.maximum_cases
            or cost + amount > args.maximum_replay_tokens
        ):
            batches.append((batch, cost))
            batch, cost = [], 0
        batch.append(case)
        cost += amount
    if batch:
        batches.append((batch, cost))
    args.output.mkdir(parents=True, exist_ok=False)
    units = []
    for i, (cases, cost) in enumerate(batches):
        path = args.output / f"baseline-replay-{i:03}-cases.json"
        path.write_text(json.dumps(cases, indent=2) + "\n")
        units.append(
            {
                "name": f"baseline-replay-{i:03}",
                "cases_file": str(path.resolve()),
                "case_count": len(cases),
                "preselected_decode_tokens": cost,
            }
        )
    report["units"] = units
    report["maximum_replay_tokens"] = args.maximum_replay_tokens
    report["maximum_cases"] = args.maximum_cases
    report["script_sha256"] = hashlib.sha256(
        pathlib.Path(__file__).read_bytes()
    ).hexdigest()
    (args.output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(units)} bounded stock8 replay units; no GPU work launched")


if __name__ == "__main__":
    main()
