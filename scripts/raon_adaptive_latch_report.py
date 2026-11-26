#!/usr/bin/env python3
"""Raon-1-mini adaptive-latch unit: cost table builder and arm report.

Companion of ``scripts/raon_adaptive_latch_unit_20261126.sh`` (inventory and
plan: docs/research/2026-11-26-raon-adaptive-latch-inventory.md).

``costs``  builds the ``adaptive_verification`` cost model from the unit's own
           fixed-depth DFlash2 arms (nd1..nd5) on the quiet 1K prose cell: the
           cost of a depth-k round is that arm's mean round wall time
           (draft + verify + host, ``MLX2_ROUND_PROBE``), so ``draft_cost`` is
           0.  Depth 0 is the ordinary arm's mean step.  Writes one policy per
           variant (plain, reject cooldown) and validates each with the
           runtime's own parser.
``report`` collects every arm directory: prose cells (decode tok/s, tokens per
           probe-booked round), GSM8K (accuracy, mean acceptance length,
           decode tok/s, text agreement with the ordinary and default arms),
           and latch engagement (scheduler counters and the per-request
           cost-park / adaptive receipts in /v1/status).

Sanity/attribution only: no thermal hold, unqualified, not a performance
claim.  Original mlx2 code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "mlx2.raon-adaptive-latch-unit.v1"
LATCH_PREFIXES = ("external_cost_", "external_adaptive_")


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def cell_rounds(cell):
    """Sum probe-booked rounds, tokens and wall time over a cell's requests
    and every route label (a parked round may carry its own label)."""
    rounds = tokens = wall = 0
    labels = set()
    for request in cell.get("requests") or []:
        for label, block in (request.get("probe") or {}).items():
            labels.add(label)
            rounds += int(block.get("rounds") or 0)
            tokens += int(block.get("tokens") or 0)
            wall += int(block.get("wall_ns") or 0)
    return {"rounds": rounds, "tokens": tokens, "wall_ns": wall, "labels": sorted(labels)}


def prose_cells(phase0):
    out = {}
    for cell in (phase0 or {}).get("cells") or []:
        if cell.get("condition") != "quiet":
            continue
        totals = cell_rounds(cell)
        rows = cell.get("requests") or []
        out[str(cell["size"])] = {
            "decode_tok_s_median": cell.get("decode_tok_s_median"),
            "requests": len(rows),
            "prompt_tokens": rows[0].get("prompt_tokens") if rows else None,
            "completion_tokens": [r.get("completion_tokens") for r in rows],
            "tokens_per_round": (totals["tokens"] / totals["rounds"]) if totals["rounds"] else None,
            "round_ms": (totals["wall_ns"] / totals["rounds"] / 1e6) if totals["rounds"] else None,
            "rounds": totals["rounds"],
            "probe_labels": totals["labels"],
        }
    return out


def round_ms(phase0_path, size):
    cells = prose_cells(_load(phase0_path))
    cell = cells.get(str(size))
    if not cell or not cell["round_ms"]:
        raise SystemExit(f"{phase0_path}: no quiet {size} cell with probe-booked rounds")
    return cell["round_ms"]


def cmd_costs(args):
    sys.path.insert(0, str(Path(args.src)))
    from mlx2.runtime.acceptance_estimator import AdaptiveVerificationPolicy

    depths = {}
    for item in args.depth:
        key, _, path = item.partition("=")
        depths[int(key)] = path
    maximum = max(depths)
    if sorted(depths) != list(range(1, maximum + 1)):
        raise SystemExit("--depth must cover 1..num_draft without gaps")
    costs = [round_ms(args.ordinary, args.size)] + [
        round_ms(depths[k], args.size) for k in range(1, maximum + 1)]
    base = {
        "verification_costs": [round(c, 4) for c in costs],
        "draft_cost": 0.0,
        "cohort_size": 1,
        "min_observations": args.min_observations,
        "full_depth_interval": args.full_depth_interval,
        "min_gain": args.min_gain,
        "cost_source": args.cost_source,
    }
    variants = {"adaptive": base,
                "adaptive-cooldown": {**base, "reject_cooldown": args.reject_cooldown}}
    out = Path(args.output_dir)
    record = {"schema": SCHEMA + ".costs", "size": args.size, "unit": "ms per round",
              "semantics": ("depth k = mean round wall (draft+verify+host) of the fixed nd=k arm; "
                            "depth 0 = mean ordinary step; draft_cost 0 (inside each round)"),
              "sources": {"0": args.ordinary, **{str(k): v for k, v in sorted(depths.items())}},
              "variants": {}}
    for name, policy in variants.items():
        AdaptiveVerificationPolicy.from_value(policy, maximum)
        (out / f"{name}.json").write_text(json.dumps({"adaptive_verification": policy}, indent=1) + "\n")
        record["variants"][name] = policy
    (out / "adaptive-costs.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))
    return 0


def _walk(node, key):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v, key)


def latch_counters(status):
    if not status:
        return None
    scheduler = status.get("scheduler") or {}
    counters = {k: v for k, v in sorted(scheduler.items())
                if k.startswith(LATCH_PREFIXES) and v not in (0, {}, None, [])}
    parks = []
    for receipt in status.get("recent_receipts") or []:
        spec = receipt.get("speculation") or {}
        handoff = next(_walk(receipt, "cost_per_commit_handoff"), None)
        adaptive = next(_walk(receipt, "adaptive_verification"), None)
        entry = {"current_execution": spec.get("current_execution"),
                 "external_rounds": spec.get("external_rounds"),
                 "accepted": spec.get("accepted"), "proposed": spec.get("proposed")}
        if isinstance(handoff, dict):
            rev = handoff.get("reversible") or {}
            entry["cost_park"] = {k: rev.get(k) for k in (
                "state", "parks", "reprobes", "failed_reprobes", "reengagements")}
            entry["cost_park"]["selection_authority"] = handoff.get("selection_authority")
        if isinstance(adaptive, dict):
            entry["adaptive"] = {k: adaptive.get(k) for k in (
                "trimmed_rounds", "request_rounds", "request_trimmed_rounds",
                "depth_histogram", "reject_cooldown", "observed_used")}
        parks.append(entry)
    settings = status.get("settings") or {}
    return {"scheduler": counters, "recent_requests": parks,
            "route": settings.get("route"),
            "route_selection_source": settings.get("route_selection_source"),
            "ast_route": settings.get("ast_route") or status.get("ast_route"),
            "route_receipt": status.get("route_receipt")}


def counter_delta(final, before):
    """Numeric scheduler counters accrued after ``before`` (dict values skipped)."""
    final, before = final or {}, before or {}
    out = {}
    for key, value in final.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delta = value - (before.get(key) or 0)
            if delta:
                out[key] = delta
    return out


def gsm8k(receipt):
    if not receipt or not receipt.get("arms"):
        return None
    arm = receipt["arms"][0]
    summary = arm.get("summary") or {}
    acc = summary.get("acceptance") or {}
    rounds = acc.get("verify_rounds") or 0
    depth0 = int((acc.get("depth_hist") or {}).get("0") or 0)
    drafting = rounds - depth0
    return {
        "questions": summary.get("questions"), "accuracy": summary.get("accuracy"),
        "invalid_rate": summary.get("invalid_rate"),
        "decode_tps_median": summary.get("decode_tps_median"),
        "completion_tokens": summary.get("completion_tokens"),
        "mean_acceptance_length": acc.get("mean_acceptance_length"),
        "depth0_rounds": depth0,
        "mean_acceptance_length_drafting_rounds": (
            1 + acc["accepted_tokens"] / drafting if drafting and "accepted_tokens" in acc else None),
        "per_position": acc.get("per_position"),
        "depth_hist": acc.get("depth_hist"),
        "texts": {row["test_index"]: row.get("text") for row in arm.get("rows") or []},
    }


def agreement(texts, reference):
    if not texts or not reference:
        return None
    shared = [k for k in texts if k in reference]
    same = sum(texts[k] == reference[k] for k in shared)
    return {"identical": same, "compared": len(shared),
            "differ": sorted(k for k in shared if texts[k] != reference[k])}


def cmd_report(args):
    runs = Path(args.runs)
    arms = []
    for line in (runs / "arms.txt").read_text().splitlines():
        name = line.split("|", 1)[0]
        path = runs / name
        if not path.is_dir():
            continue
        g = gsm8k(_load(path / "gsm8k" / "receipt.json"))
        after_prose = latch_counters(_load(path / "status-prose.json"))
        final = latch_counters(_load(path / "status-final.json"))
        arms.append({
            "arm": name, "route": line.rsplit("|", 1)[-1],
            "prose": prose_cells(_load(path / "phase0" / "phase0.json")),
            "gsm8k": g,
            "latch_after_prose": after_prose,
            "latch_final": final,
            # Per workload: the prose snapshot is taken before GSM8K starts.
            "counters_prose": (after_prose or {}).get("scheduler") or {},
            "counters_gsm8k": counter_delta((final or {}).get("scheduler"),
                                            (after_prose or {}).get("scheduler")),
        })
    by_name = {a["arm"]: a for a in arms}
    default = by_name.get("dflash2-nd5")
    def texts(arm):
        return ((arm or {}).get("gsm8k") or {}).get("texts")

    for arm in arms:
        # A presence_penalty 0 arm compares with the ordinary arm of the same request law.
        if arm["arm"].endswith("-pp0") and "ordinary-pp0" in by_name:
            reference = by_name["ordinary-pp0"]
        else:
            reference = by_name.get("ordinary")
        arm["reference_arm"] = (reference or {}).get("arm")
        g = arm["gsm8k"]
        if g:
            g["text_vs_ordinary"] = agreement(g["texts"], texts(reference))
            g["text_vs_default"] = agreement(g["texts"], texts(default))
        for size, cell in arm["prose"].items():
            ref = ((reference or {}).get("prose") or {}).get(size) or {}
            if cell["decode_tok_s_median"] and ref.get("decode_tok_s_median"):
                cell["vs_ordinary"] = cell["decode_tok_s_median"] / ref["decode_tok_s_median"]
        ref_g = (reference or {}).get("gsm8k") or {}
        if g and g["decode_tps_median"] and ref_g.get("decode_tps_median"):
            g["vs_ordinary"] = g["decode_tps_median"] / ref_g["decode_tps_median"]
    report = {"schema": SCHEMA, "checkpoint": args.checkpoint, "source_revision": args.source_revision,
              "qualified": False, "thermal_hold": False,
              "performance_claim": "none: sanity/attribution unit, single window, no thermal hold",
              "costs": _load(runs / "adaptive-costs.json"), "arms": arms}
    for arm in report["arms"]:
        if arm["gsm8k"]:
            arm["gsm8k"].pop("texts", None)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    Path(args.output).with_suffix(".md").write_text(markdown(report) + "\n")
    print(markdown(report))
    return 0


def _f(value, spec=".1f"):
    return "-" if value is None else format(value, spec)


def solo_rounds(latch):
    """Reject-cooldown solo rounds: a generator-wide count that every receipt
    repeats, so take the largest (newest) value, never a sum."""
    values = [((r.get("adaptive") or {}).get("reject_cooldown") or {}).get("solo_rounds") or 0
              for r in (latch or {}).get("recent_requests") or []]
    return max(values, default=0)


def markdown(report):
    lines = [(f"Checkpoint `{report['checkpoint']}`, mlx2 `{report['source_revision']}`; "
              "one lane, quiet host, no thermal hold, unqualified."), "",
             ("| arm | prose 1K tok/s (x ord) | tok/round | prose 16K tok/s (x ord) | tok/round "
              "| GSM8K acc | MAL (drafting) | GSM8K tok/s (x ord) | texts = ordinary | parks p/g | trimmed p/g | solo |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in report["arms"]:
        p1, p16 = arm["prose"].get("1024") or {}, arm["prose"].get("16384") or {}
        g = arm["gsm8k"] or {}
        cp, cg = arm["counters_prose"], arm["counters_gsm8k"]
        agree = g.get("text_vs_ordinary") or {}
        lines.append(
            f"| {arm['arm']} | {_f(p1.get('decode_tok_s_median'))} ({_f(p1.get('vs_ordinary'), '.2f')}) "
            f"| {_f(p1.get('tokens_per_round'), '.2f')} "
            f"| {_f(p16.get('decode_tok_s_median'))} ({_f(p16.get('vs_ordinary'), '.2f')}) "
            f"| {_f(p16.get('tokens_per_round'), '.2f')} "
            f"| {_f(g.get('accuracy'), '.3f')} | {_f(g.get('mean_acceptance_length'), '.2f')} "
            f"({_f(g.get('mean_acceptance_length_drafting_rounds'), '.2f')}) "
            f"| {_f(g.get('decode_tps_median'))} ({_f(g.get('vs_ordinary'), '.2f')}) "
            f"| {agree.get('identical', '-')}/{agree.get('compared', '-')} "
            f"| {cp.get('external_cost_park_events', 0)}/{cg.get('external_cost_park_events', 0)} "
            f"| {cp.get('external_adaptive_trimmed_rounds', 0)}/{cg.get('external_adaptive_trimmed_rounds', 0)} "
            f"| {solo_rounds(arm['latch_final'])} |")
    lines += ["", ("tok/round counts probe-booked rounds of every label (a parked or solo round books "
                   "one token). MAL (drafting) leaves depth-0 rounds out. x ord and texts compare with ordinary, "
                   "or with ordinary-pp0 for a -pp0 arm. 'parks' and 'trimmed' are "
                   "scheduler counters, prose/GSM8K. 'trimmed' also counts rounds whose prompt-lookup copy was "
                   "shorter than the depth: read it with the depth histograms in report.json. 'solo' is the "
                   "generator-wide reject-cooldown count read from the newest receipt. The adaptive arms' "
                   "estimator persists across requests, so their GSM8K starts from a prose-trained estimator; "
                   "their cost table comes from the 1K prose cell (all probe labels). Differing texts are "
                   "exact in law, not bitwise: classify near ties before any equivalence claim.")]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("costs")
    c.add_argument("--ordinary", required=True, help="ordinary arm phase0.json")
    c.add_argument("--depth", action="append", required=True, help="K=path/to/phase0.json, K = 1..num_draft")
    c.add_argument("--size", type=int, default=1024)
    c.add_argument("--src", required=True, help="mlx2 src/ of the snapshot the arms run")
    c.add_argument("--min-observations", type=int, default=32)
    c.add_argument("--full-depth-interval", type=int, default=32)
    c.add_argument("--min-gain", type=float, default=0.05)
    c.add_argument("--reject-cooldown", type=int, default=8)
    c.add_argument("--cost-source", required=True)
    c.add_argument("--output-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--runs", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--source-revision", required=True)
    r.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "costs":
        return cmd_costs(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
