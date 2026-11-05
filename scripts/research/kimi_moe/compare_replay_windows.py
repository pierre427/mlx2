"""CPU-only descriptive choice distances; never APCv2 numerical equivalence.

Provenance: replay-comparison-provenance-20261104T0728Z.json.
"""

import argparse
import collections
import datetime
import json
import math
from pathlib import Path

import summarize

ARMS = ("adaptive10", "adaptive20", "adaptive30", "adaptive50", "adaptive70")


def ids(row):
    r = row["attempts"][-1]["response"]
    values = r["mlx2"]["probe_output_tokens"]
    if not values or len(values) != r["usage"]["completion_tokens"]:
        raise ValueError("Incomplete token receipt")
    return values


def site(window, expected, position):
    if window["position_zero_based"] != position:
        raise ValueError("wrong_selected_site")
    r = window["response"]
    if not window["replayed_prefix_and_selection"] or r["mlx2"]["probe_output_tokens"] != expected[: position + 1]:
        raise ValueError("earlier_replay_drift")
    rows = r["choices"][0]["logprobs"]["content"]
    if len(rows) != 1 or rows[0]["id"] != expected[position]:
        raise ValueError("nonbounded_or_wrong_selected_logit")
    row = rows[0]
    top = row["top_logprobs"]
    if not 1 <= len(top) <= 20 or len({p["id"] for p in top}) != len(top):
        raise ValueError("invalid_top_window")
    values = {p["id"]: p["logprob"] for p in top}
    if not all(type(v) in (int, float) and math.isfinite(v) for v in [row["logprob"], *values.values()]):
        raise ValueError("nonfinite_logprob")
    selected = values.get(expected[position])
    if selected is None or abs(selected - row["logprob"]) > 1e-6:
        raise ValueError("inconsistent_selected_logprob")
    if max(values.values()) - selected > 1e-6:
        raise ValueError("selected_not_top_in_replay_window")
    return values


def cross_gap(values, alternate):
    top = max(values.values())
    if alternate in values:
        return {"value": top - values[alternate], "kind": "exact"}
    return {"value": top - min(values.values()), "kind": "lower_bound_missing_token"}


def compare(base_ids, cand_ids, position, base_window, cand_window):
    if base_ids[:position] != cand_ids[:position] or base_ids[position] == cand_ids[position]:
        return {"classification": "unclassified_not_shared_first_divergence"}
    try:
        a = site(base_window, base_ids, position)
        b = site(cand_window, cand_ids, position)
    except (KeyError, TypeError, ValueError) as error:
        return {"classification": "unclassified_invalid_window", "reason": str(error)}
    ga, gb = cross_gap(a, cand_ids[position]), cross_gap(b, base_ids[position])
    if ga["value"] > .5 or gb["value"] > .5:
        classification = "local_choices_separated"
    elif ga["kind"] == gb["kind"] == "exact":
        classification = "local_choices_close"
    else:
        classification = "unclassified_missing_cross_token"
    return {"classification": classification, "baseline_gap_for_candidate_choice": ga, "candidate_gap_for_baseline_choice": gb}


def baselines():
    sites = {}
    units = []
    excluded = []
    root = summarize.RUNS / "kimi-baseline-replay-20261104-r1"
    for unit in sorted(root.iterdir()):
        if not unit.is_dir():
            continue
        v, r = unit / "replay-validation.json", unit / "receipt.json"
        if not v.exists() or not r.exists():
            excluded.append({"directory": str(unit), "reason": "validation_or_raw_receipt_pending"})
            continue
        if not summarize.load(v)["passed"] or not summarize.load(r).get("server_stopped"):
            excluded.append({"directory": str(unit), "reason": "validation_failed_or_raw_not_closed", "validation": summarize.load(v)})
            continue
        spec = summarize.load(unit / "spec.json")
        if spec["candidate"] != {"mode": "fixed", "k": 8} or not spec["replay_only"]:
            raise ValueError("Non-stock replay control")
        units.append({"directory": str(unit), "source_revision": spec["source_revision"], "checkpoint": spec["checkpoint"], "original_restored_ready": (unit / "service-restored-ready.json").exists()})
        for path in unit.glob("*.*.json"):
            row = summarize.load(path)
            if "case" not in row or "windows" not in row:
                continue
            for window in row["windows"]:
                key = row["case"]["case_id"], window["position_zero_based"]
                if key in sites:
                    raise ValueError("Ambiguous duplicate baseline site " + str(key))
                sites[key] = (path, row, window, spec)
    return sites, units, excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    out = p.parse_args().output
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Nonempty analysis evidence")
    stock = summarize.rows([summarize.BASE], "default8")
    stock_audit = summarize.audits([summarize.BASE], "default8", stock)
    controls, units, excluded = baselines()
    summary = {"created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "baseline_sites_collected": len(controls), "planned_baseline_sites": 1272, "baseline_units": units, "excluded_or_pending_baseline_units": excluded, "arms": {}, "scope": "Descriptive0.5nat cross-choice distance at firstdivergence only. NeverAPCv2equivalence, accuracy, qualitygain orperformance. Firstdivergence may beformatting anddoesnotcausallyexplain fullresponsegrade. Partialcasecoverageexplicit; sourcebridge isselected-sitebound, notfullanswerparity. Lifecyclecompletionseparatefrom scientificwindowvalidation."}
    result = []
    for arm in ARMS:
        candidates = summarize.rows(summarize.CAND, arm)
        audits = summarize.audits(summarize.CAND, arm, candidates)
        counts = collections.Counter()
        for cid, (path, row) in sorted(candidates.items()):
            base_path, reference = stock[cid]
            if any(row["case"][k] != reference["case"][k] for k in ("case_id", "domain", "question")):
                raise ValueError("Prompt identity drift")
            a, b = ids(reference), ids(row)
            pos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if row.get("baseline_comparison", {}).get("first_divergence") != pos:
                raise ValueError("Stored firstdivergence changed")
            detail = {"arm": arm, "case_id": cid, "position_zero_based": pos, "reference": str(base_path), "candidate": str(path)}
            if cid in stock_audit and cid in audits:
                detail["assistant_core_grades"] = {"baseline": stock_audit[cid]["score_0_to_4"], "candidate": audits[cid]["score_0_to_4"]}
            if pos is None:
                detail["classification"] = "observed_reference_output_identical" if a == b else "unclassified_terminal_or_length_change"
            elif (cid, pos) not in controls:
                detail["classification"] = "baseline_site_not_collected"
            else:
                control_path, control, bw, bs = controls[cid, pos]
                if any(control["case"][k] != reference["case"][k] for k in ("case_id", "domain", "question")):
                    raise ValueError("Control prompt identity drift")
                cs = summarize.load(path.parent / "spec.json")
                if cs["checkpoint"] != bs["checkpoint"]:
                    raise ValueError("Checkpoint mismatch")
                cw = row.get("baseline_comparison", {}).get("candidate_window")
                detail.update(compare(a, b, pos, bw, cw or {}))
                detail["baseline_site_receipt"] = str(control_path)
                detail["source_scope"] = {"reference": summarize.load(base_path.parent / "spec.json")["source_revision"], "baseline_window": bs["source_revision"], "candidate_window": cs["source_revision"]}
            counts[detail["classification"]] += 1
            result.append(detail)
        summary["arms"][arm] = {"saved_responses": len(candidates), "classifications": dict(counts)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparisons.jsonl").write_text("".join(json.dumps(r)+"\n" for r in result))
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"baseline_sites": len(controls), "arms": summary["arms"]}))


if __name__ == "__main__":
    main()
