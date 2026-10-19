#!/usr/bin/env python3
"""Emit token-free performance/geometry audit data from completed local units."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

PERFORMANCE = ("qwen27-r4", "muse-r1", "qwen36-r1", "flash-native-r1", "flash-dflash-r1")
GEOMETRY = ("geometry-qwen27-r2", "geometry-muse-r1", "geometry-qwen36-r1",
            "geometry-flash-native-r1", "geometry-flash-dflash-r2")


def read(path):
    return json.loads(path.read_text())


def performance(path):
    meta, summary = read(path / "meta.json"), read(path / "summary.json")
    assert meta["status"] == "completed"
    arms = {}
    for arm in ("ordinary", "before", "fixed"):
        rows = [value[arm] for value in summary.values()]
        report_path = path / f"{arm}.json"
        report = read(report_path)
        measured = [row for name, row in report["phases"].items() if name.startswith("b1_r")]
        assert len(measured) == sum(len(row["exact_token_parity"]) for row in rows)
        assert all(not row.get("error") and row.get("done") for row in measured)
        selected = bool((meta.get("arms", {}).get(arm, {}).get("policy") or {})
                        .get("continuation_pool", {}).get("ast_planner"))
        specs = [row.get("speculation") or {} for row in measured]
        plans = [spec.get("ast_execution_plan") or {} for spec in specs]
        observed = [plan.get("schema") == "mlx2.ast-execution-plan.v1"
                    and plan.get("cost_samples", 0) > 0
                    and spec.get("external_rounds", 0) > 0
                    and spec.get("ordinary_fallback") is False
                    and spec.get("progressive_verification", {}).get("observed_used") is True
                    and row.get("route") == "external_draft"
                    for row, spec, plan in zip(measured, specs, plans)]
        if selected:
            assert all(observed), "selected AST planner lacks actual execution evidence"
        arms[arm] = {
            "decode_tps_medians": {key: value[arm]["decode_tps_median"] for key, value in summary.items()},
            "wall_tps_medians": {key: value[arm]["wall_tps_median"] for key, value in summary.items()},
            "decode_tps_samples": {key: value[arm]["decode_tps"] for key, value in summary.items()},
            "wall_tps_samples": {key: [report["phases"][f"b1_r{r + 1}_{key}"]["wall_tokens_per_second"]
                                       for r in range(len(value[arm]["decode_tps"]))]
                                 for key, value in summary.items()},
            "exact_parity_by_prompt": {key: value[arm]["exact_token_parity"] for key, value in summary.items()},
            "exact_parity": sum(sum(row["exact_token_parity"]) for row in rows),
            "comparisons": sum(len(row["exact_token_parity"]) for row in rows),
            "swapouts_delta": sum(sum(row["swapouts_delta"]) for row in rows),
            "maximum_start_c": max(t for row in rows for t in row["thermal_start_c"]),
            "maximum_end_c": max(t for row in rows for t in row["thermal_end_c"]),
            "thermal_start_c_by_prompt": {key: value[arm]["thermal_start_c"] for key, value in summary.items()},
            "thermal_end_c_by_prompt": {key: value[arm]["thermal_end_c"] for key, value in summary.items()},
            "planner": {"selected": selected, "observed_requests": sum(observed),
                        "measured_requests": len(measured),
                        "backend_hist": dict(Counter(plan["backend"] for plan, used in zip(plans, observed) if used)),
                        "minimum_cost_samples": min((plan["cost_samples"] for plan, used in zip(plans, observed) if used), default=0)},
            "raw_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    draft_path = meta["policy"]["draft_model"]
    draft = meta["inventory"]["auxiliaries"].get(draft_path)
    return {"unit": path.name, "checkpoint": meta["checkpoint"], "config": meta["config"],
            "drafter_checkpoint": meta["checkpoint"] if draft is None else draft["checkpoint"],
            "hardware": meta["hardware"], "source_heads": meta["source_heads"],
            "shape": meta["shape"], "policy": meta["policy"], "arms": arms,
            "arm_policies": {arm: value.get("policy") for arm, value in meta.get("arms", {}).items()},
            "order": meta.get("order"),
            "fixed_vs_before_pct": {key: value["fixed_vs_before_pct"] for key, value in summary.items()},
            "fixed_vs_ordinary_ratio": {key: value["fixed_vs_ordinary_ratio"] for key, value in summary.items()},
            "raw_summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest()}


def geometry(path):
    meta = read(path / "meta.json")
    assert meta["status"] == "complete"
    trace_bytes = (path / "rounds.jsonl").read_bytes()
    rounds = [json.loads(line) for line in trace_bytes.splitlines()]
    assert rounds and all(row["completed"] for row in rounds)
    events = [event for row in rounds for event in row["events"]]
    tiles = [e for e in events if e["kind"] == "target_tile"]
    admitted = [e for e in events if e["kind"] == "admitted_topology"]
    sources = [e for e in events if e["kind"] == "source_slate"]
    selected = [e for e in events if e["kind"] == "verified_selection"]
    stages = defaultdict(Counter)
    for row in rounds:
        uids = {session["uid"] for session in row["sessions"]}
        for tile in (e for e in row["events"] if e["kind"] == "target_tile"):
            layout = tile["layout"]
            assert len(tile["mapping"]) == tile["shape"][0]
            assert all(m["uid"] in uids for m in tile["mapping"])
            assert sum(m["input_rows"] for m in tile["mapping"]) == layout["logical_rows"]
            assert tile["shape"][0] * tile["shape"][1] == layout["physical_rows"]
            assert layout["physical_rows"] - layout["logical_rows"] == layout["padding_rows"]
            assert tile["executor"]["row_exact"] is True
            plan = layout.get("ast_plan")
            if plan:
                assert plan["input_lengths"] == [m["input_rows"] for m in tile["mapping"]]
                assert all(plan[key] == layout[key] for key in ("logical_rows", "physical_rows", "padding_rows"))
            minimum = layout.get("packed_min_query_len")
            if minimum is not None and tile["shape"][0] > 1:
                assert tile["shape"][1] >= minimum
            for window in tile["windows"]:
                assert not window["failures"]
                for stage, routes in window["stages"].items():
                    stages[stage].update(routes)
    verdicts = read(path / "verdicts.json")
    parity = {kind: {"exact": sum(row[0] is True for row in values.values()), "comparisons": len(values),
                     "unclassified": [label for label, row in values.items() if row[0] is not True]}
              for kind, values in verdicts.items() if kind != "b4_warm_cached_tokens"}
    ragged = next((e for e in tiles if e["layout"]["padding_rows"]), None)
    sample = None if ragged is None else {k: ragged[k] for k in ("shape", "layout", "executor", "windows")}
    if sample:
        sample["layout"] = {k: v for k, v in sample["layout"].items()
                            if k not in ("request_lane_uids", "physical_lane_uids", "lane_uids")}
    draft = meta["inventory"]["auxiliaries"].get(meta["policy"]["draft_model"])
    return {
        "unit": path.name, "checkpoint": meta["checkpoint"], "config": meta["config"],
        "drafter_checkpoint": meta["checkpoint"] if draft is None else draft["checkpoint"],
        "source_revision": meta["source_revision"], "shape": meta["shape"],
        "policy": meta["policy"],
        "performance_evidence": False, "rounds": len(rounds),
        "cohort_width_hist": Counter(len(row["sessions"]) for row in rounds),
        "mixed_topology_rounds": sum(
            len(row["sessions"]) > 1 and len({len(e["topology"]["paths"]) for e in row["events"]
                                             if e["kind"] == "admitted_topology"}) > 1 for row in rounds),
        "admitted_branch_hist": Counter(len(e["topology"]["paths"]) for e in admitted),
        "zero_admitted_meaning": "no multi-branch slate; primary path still available",
        "latch_reason_hist": Counter(e["latch"]["reason"] for e in admitted if e["latch"]),
        "topology_kind_hist": Counter(e["topology_kind"] for e in admitted),
        "source_path_depth_hist": Counter(depth for e in sources for depth in e["topology"]["path_depths"]),
        "source_ranking_authority_hist": Counter(r["authority"] for e in sources for r in e["ranking"]),
        "source_tree_calls": sum(len(e["source_geometry"]) for e in sources),
        "source_full_depth_leaves": sum(g.get("full_depth_leaves", 0) for e in sources for g in e["source_geometry"].values()),
        "codebook_cache_hits": sum(g.get("codebook_cache_hit", False) for e in sources for g in e["source_geometry"].values()),
        "target_execution_hist": Counter(row["target_execution"] for row in rounds),
        "tile_policy_hist": Counter(str(row["tile_policy"]) for row in rounds),
        "configured_tiles": sorted({row["configured_tile"] for row in rounds}),
        "progressive_tiles": len(tiles),
        "mixed_session_tiles": sum(len({m["uid"] for m in e["mapping"]}) > 1 for e in tiles),
        "tile_session_width_hist": Counter(len({m["uid"] for m in e["mapping"]}) for e in tiles),
        "physical_width_hist": Counter(e["shape"][0] for e in tiles),
        "token_width_hist": Counter(e["shape"][1] for e in tiles),
        "packed_minimum_attested_tiles": sum("packed_min_query_len" in e["layout"] for e in tiles),
        "packed_minimum_split_launches": sum(e["layout"].get("split_for_packed_min_query_len", False) for e in tiles),
        "multi_lane_one_token_launches": sum(e["shape"][0] > 1 and e["shape"][1] == 1 for e in tiles),
        "multi_lane_wider_launches": sum(e["shape"][0] > 1 and e["shape"][1] > 1 for e in tiles),
        "logical_valid_token_rows": sum(e["layout"]["logical_rows"] for e in tiles),
        "physical_slab_token_positions": sum(e["layout"]["physical_rows"] for e in tiles),
        "padding_token_positions": sum(e["layout"]["padding_rows"] for e in tiles),
        "backend_mode_hist": Counter(e["layout"]["backend_mode"] for e in tiles),
        "executor_algorithm_hist": Counter(e["executor"]["adapter_detail"].get("algorithm") for e in tiles),
        "executor_route_hist": Counter(e["executor"]["adapter_detail"].get("route") for e in tiles),
        "executor_gdn_route_hist": Counter(e["executor"]["adapter_detail"].get("gdn_route") for e in tiles),
        "progressive_window_stage_counts": dict(stages),
        "selected_branch_hist": Counter(e["winner"] + 1 for e in selected),
        "selected_token_uplift": sum(max(0, e["committed"][e["winner"]] - e["committed"][0]) for e in selected),
        "unvisited_branches": sum(a < 0 for e in selected for a in e["accepts"]),
        "parity": parity, "geometry_invariants_passed": True, "ragged_tile_example": sample,
        "raw_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "planner": planner_summary(admitted, sources, tiles, stages),
    }


def planner_summary(admitted, sources, tiles, stages):
    profiles = [e["execution_planner"] for e in admitted if e.get("execution_planner")]
    budgets = [e["source_budget"] for e in admitted if e.get("source_budget")]
    diagnostics = [g for e in sources for g in e["source_geometry"].values()]
    plans = [e["layout"]["ast_plan"] for e in tiles if e["layout"].get("ast_plan")]
    return {
        "observed_used": bool(profiles),
        "initial_profile": profiles[0] if profiles else None,
        "maximum_cost_samples": max((p["cost_samples"] for p in profiles), default=0),
        "source_depth_branch_hist": dict(Counter(f"{b['depth']}x{b['branches']}" for b in budgets)),
        "source_reason_hist": dict(Counter(b["reason"] for b in budgets)),
        "planned_physical_launches": len(plans),
        "planned_tile_hist": dict(Counter(str(p["tile"]) for p in plans)),
        "planned_batch_partition_hist": dict(Counter(str(sorted(p["batch_row_counts"])) for p in plans)),
        "source_reused_backbone": sum(g.get("backbone_reused") is True for g in diagnostics),
        "source_private_head_call_count": sum(g.get("private_head_calls", 0) for g in diagnostics),
        "source_head_call_attestations": sum("private_head_calls" in g for g in diagnostics),
        "source_diagnostic_count": len(diagnostics),
        "padding_stage_rows_skipped": sum(v for k, v in stages.get("valid_row_compaction", {}).items()
                                         if k.endswith("_padding_skipped")),
        "padding_counter_unit": "stage-row operations, not unique token positions or elapsed time",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--available-only", action="store_true")
    ap.add_argument("--performance-units", nargs="*", default=PERFORMANCE)
    ap.add_argument("--geometry-units", nargs="*", default=GEOMETRY)
    args = ap.parse_args()
    def units(names, summarize):
        return [summarize(args.root / name) for name in names
                if not args.available_only or (args.root / name / "meta.json").exists()
                and read(args.root / name / "meta.json")["status"] in ("complete", "completed")]
    print(json.dumps({"schema": "mlx2.ast-audit-measurements.v1",
                      "performance": units(args.performance_units, performance),
                      "geometry": units(args.geometry_units, geometry)}, indent=2))


if __name__ == "__main__":
    main()
