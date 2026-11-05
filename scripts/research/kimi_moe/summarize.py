"""Aggregate immutable per-case receipts; semantic audits stay separate from screens."""

import collections
import datetime
import hashlib
import json
import os
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNS = ROOT / "qualification/runs"
WORK = pathlib.Path(__file__).parent
# The r2/r3 domain runs were recorded in another checkout; point
# MLX2_KIMI_MOE_OLD_RUNS at its qualification/runs (default: this checkout's).
OLD_RUNS = pathlib.Path(os.environ.get("MLX2_KIMI_MOE_OLD_RUNS", str(RUNS)))
BASE = OLD_RUNS / "kimi-domain-adaptive-20261103-r2"
CAND = [
    OLD_RUNS / "kimi-domain-adaptive-20261103-r3",
    OLD_RUNS / "kimi-domain-adaptive-20261103-r4",
    RUNS / "kimi-domain-adaptive-20261104-r5",
]


def load(p):
    return json.loads(p.read_text())


def rows(roots, arm):
    out = {}
    for root in roots:
        for p in root.glob(arm + "-*/*.*.json"):
            if p.name.startswith("paired-"):
                continue
            d = load(p)
            if "case" not in d or "attempts" not in d:
                continue
            cid = d["case"]["case_id"]
            if cid in out:
                raise ValueError("duplicate target identity " + arm + "/" + cid)
            out[cid] = (p, d)
    return out


def audits(roots, arm, receipts):
    result = {}
    for root in roots:
        for path in root.glob(
            "semantic-" + ("baseline" if arm == "default8" else arm) + "-audit.jsonl"
        ):
            for line in path.read_text().splitlines():
                d = json.loads(line)
                cid = d["case_id"]
                if cid in result:
                    raise ValueError("duplicate semantic audit " + arm + "/" + cid)
                if cid not in receipts:
                    raise ValueError("audit without output " + cid)
                text = receipts[cid][1]["attempts"][-1]["response"]["choices"][0][
                    "message"
                ]["content"]
                if hashlib.sha256(text.encode()).hexdigest() != d["output_sha256"]:
                    raise ValueError("audit hash mismatch " + cid)
                result[cid] = d
    return result


def summarize_arm(receipts):
    hist = collections.Counter()
    layers = collections.defaultdict(collections.Counter)
    finish = collections.Counter()
    replay = collections.Counter()
    source_counts = collections.Counter()
    checkpoints = collections.Counter()
    unit_counts = collections.Counter(p.parent for p, _ in receipts.values())
    specs = {unit: load(unit / "spec.json") for unit in unit_counts}
    for p, d in receipts.values():
        spec = specs[p.parent]
        source_counts[spec["source_revision"]] += 1
        checkpoints[spec["checkpoint"]] += 1
        response = d["attempts"][-1]["response"]
        finish[response["choices"][0]["finish_reason"]] += 1
        for k, v in d["routes"].items():
            if k.startswith("rows_k"):
                hist[int(k[6:])] += v
            elif k.startswith("layer_"):
                _, l, width = k.split("_")
                layers[int(l)][int(width[1:])] += v
        c = d.get("baseline_comparison", {})
        if c:
            if c.get("exact_tokens"):
                replay["exact_output_tokens"] += 1
            elif c.get("first_divergence") is None:
                replay["length_or_terminal_change_no_window"] += 1
            else:
                replay["differing_outputs"] += 1
                w = c.get("candidate_window")
                replay[
                    "window_verified"
                    if w and w.get("replayed_prefix_and_selection")
                    else "window_unverified_or_pending"
                ] += 1
    words = [d["screen"]["words"] for _, d in receipts.values()]
    return {
        "completed_outputs": len(receipts),
        "saved_responses": len(receipts),
        "naturally_stopped_outputs": finish.get("stop", 0),
        "token_capped_cases": [
            {"case_id": d["case"]["case_id"], "receipt": str(p),
             "attempt_caps": [a["maximum"] for a in d["attempts"]]}
            for p, d in receipts.values()
            if d["attempts"][-1]["response"]["choices"][0]["finish_reason"] == "length"
        ],
        "observed_source_output_counts": dict(source_counts),
        "observed_checkpoint_output_counts": dict(checkpoints),
        "source_bound_units": [
            {
                "evidence_directory": str(unit),
                "outputs": count,
                "source_revision": specs[unit]["source_revision"],
                "source_snapshot_sha256": specs[unit]["source_snapshot_sha256"],
                "benchmark_default_routes": specs[unit].get("benchmark_default_routes"),
            }
            for unit, count in sorted(unit_counts.items())
        ],
        "finish_reason_counts": dict(finish),
        "response_words": {
            "median": statistics.median(words),
            "min": min(words),
            "max": max(words),
        }
        if words
        else None,
        "routing_measurement_valid": False,
        "routing_method_correction": str(
            WORK / "routing-counter-method-correction-20261104T0522Z.json"
        ),
        "routed_expert_histogram": {},
        "mean_routed_experts": None,
        "layer_histograms": {},
        "layer_mean_routed_experts": {},
        "unreliable_status_delta_diagnostic": {
            "histogram": dict(sorted(hist.items())),
            "layer_histograms": {
                l: dict(sorted(h.items())) for l, h in sorted(layers.items())
            },
            "scope": "Historical stale status deltas only; not request-bound expert work or performance evidence.",
        },
        "candidate_first_divergence_replay": dict(replay),
        "scope": "Per-request routing measurements withdrawn: execution diagnostics are published on a one-second cadence, so immediate status deltas can be stale, shift work between cases and include logit/warm replay. Raw responses, tokens and semantic grades are separate evidence. Saved responses may be token-capped; inspect finish_reason_counts. No expert-count or performance inference from these deltas.",
    }


def main():
    base = rows([BASE], "default8")
    ba = audits([BASE], "default8", base)
    output = {
        "updated_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "checkpoint": load(BASE / "study.json")["checkpoint"],
        "source_revision": load(BASE / "study.json")["source_revision"],
        "source_revision_scope": "Top-level revision names the baseline. Each arm reports exact observed source counts; the 23 default and 19 adaptive0.30 source-bridge matches are limited anchors, not observed new-source parity on all400 baseline outputs.",
        "default8": summarize_arm(base),
        "policies": {},
        "method_limits": [
            "Routing status deltas are not request-bound; prior expert means/histograms and replay exclusions are superseded by the05:22UTC method correction.",
            "150core templates;400questions include correlated framing variants.",
            "Semantic scores are assistant unblinded0..4rubric points, not accuracy percentages or independent human evaluation.",
            "Framing robustness slice is exploratory and separately reported.",
            "No default, production-speed, general-quality or qualification claim.",
            "Cross-policy first-choice closeness is not APCv2 numerical equivalence; B1cold/warm gates remain exact.",
        ],
    }
    for arm in ["adaptive10", "adaptive20", "adaptive30", "adaptive50", "adaptive70"]:
        rs = rows(CAND, arm)
        aa = audits(CAND, arm, rs)
        s = summarize_arm(rs)
        ids = sorted(set(aa) & set(ba))
        verdict = collections.Counter(
            "better"
            if aa[c]["score_0_to_4"] > ba[c]["score_0_to_4"]
            else "worse"
            if aa[c]["score_0_to_4"] < ba[c]["score_0_to_4"]
            else "tied"
            for c in ids
        )
        s["semantic_core_pairs"] = {
            "reviewed": len(ids),
            "expected": 150,
            "baseline_score": sum(ba[c]["score_0_to_4"] for c in ids),
            "candidate_score": sum(aa[c]["score_0_to_4"] for c in ids),
            "maximum": 4 * len(ids),
            "verdict_counts": dict(verdict),
            "baseline_score_le2": sum(ba[c]["score_0_to_4"] <= 2 for c in ids),
            "candidate_score_le2": sum(aa[c]["score_0_to_4"] <= 2 for c in ids),
        }
        bydomain = collections.defaultdict(list)
        for c in ids:
            bydomain[c.split(".")[0]].append(c)
        s["semantic_by_domain"] = {
            d: {
                "pairs": len(cs),
                "baseline_score": sum(ba[c]["score_0_to_4"] for c in cs),
                "candidate_score": sum(aa[c]["score_0_to_4"] for c in cs),
            }
            for d, cs in bydomain.items()
        }
        s["semantic_domain_macro"] = {
            "domains_reviewed": len(bydomain),
            "expected_domains": 20,
            "baseline_mean_0_to_4": statistics.mean(
                sum(ba[c]["score_0_to_4"] for c in cs) / len(cs)
                for cs in bydomain.values()
            )
            if bydomain
            else None,
            "candidate_mean_0_to_4": statistics.mean(
                sum(aa[c]["score_0_to_4"] for c in cs) / len(cs)
                for cs in bydomain.values()
            )
            if bydomain
            else None,
            "scope": "Equal weight per reviewed domain. Core-template counts differ:10each in first10domains,5each in next10. Keep separate from raw question-weighted rubric sums.",
        }
        output["policies"][arm] = s
    (WORK / "aggregate.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: (
                    {
                        "outputs": v["completed_outputs"],
                        "mean_experts": v["mean_routed_experts"],
                        "semantic": v.get("semantic_core_pairs"),
                    }
                    if k == "default8"
                    else v
                )
                for k, v in {"default8": output["default8"]}.items()
            }
        )
    )
    for k, s in output["policies"].items():
        print(
            k,
            s["completed_outputs"],
            s["mean_routed_experts"],
            s["semantic_core_pairs"],
        )


if __name__ == "__main__":
    main()
