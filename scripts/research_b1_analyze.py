#!/usr/bin/env python3
"""B1 + C2 analysis (CPU only): planner vs best-plan headroom, gather share, A8 Amdahl."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main(run_dir: str, tiles_dir: str = "tiles"):
    d = Path(run_dir)
    cap = json.loads((d / "capture" / "capture.json").read_text())
    tiles = json.loads((d / tiles_dir / "tiles.json").read_text())
    e2e_fused = cap["prefill_ms"]["fused"]["median"]
    e2e_off = cap["prefill_ms"]["off"]["median"]
    base = tiles["baseline_eval_ms"]
    calls = tiles["calls"]
    four = [c for c in calls if c["bits"] == 4]
    six = [c for c in calls if c["bits"] == 6]

    out = {"e2e_prefill_ms": {"nax_fused_served": e2e_fused, "nax_off": e2e_off},
           "baseline_eval_ms": base, "n_calls": {"4bit": len(four), "6bit": len(six)}}

    # Headroom on the NAX (4-bit) calls.
    per_proj = {}
    for proj in ("gate_up", "down"):
        planner = sum(c[proj]["planner_ms"] for c in four)
        best5 = sum(c[proj]["best_planner_set_ms"] for c in four)
        bestgrid = sum(c[proj]["best_grid_ms"] for c in four)
        stock = sum(c[proj]["stock_ms"] for c in four)
        # Best single global plan over all calls (what a better static table gets).
        plans = [k.split("|")[1] for k in four[0]["median_ms"] if k.startswith(proj + "|") and not k.endswith("stock")]
        totals = {p: sum(c["median_ms"].get(f"{proj}|{p}", float("inf")) for c in four) for p in plans}
        best_static = min(totals, key=totals.get)
        # Split-half de-biased per-call oracle: choose on even reps, score on odd reps.
        split = None
        if "times_ms" in four[0]:
            s = 0.0
            for c in four:
                tm = {k.split("|")[1]: v for k, v in c["times_ms"].items() if k.startswith(proj + "|") and not k.endswith("stock")}
                pick = min(tm, key=lambda p: statistics.median(tm[p][0::2]))
                s += statistics.median(tm[pick][1::2])
            plan_odd = sum(statistics.median(
                c["times_ms"][f"{proj}|{c['planner_choice'][proj]}"][1::2]) for c in four)
            split = {"planner_odd_ms": plan_odd, "oracle_split_ms": s, "headroom_ms": plan_odd - s}
        wins = defaultdict(int)
        for c in four:
            wins[c[proj]["best_planner_set"]] += 1
        per_proj[proj] = {
            "planner_choice": four[0]["planner_choice"][proj],
            "sum_planner_ms": planner, "sum_best_of_5_ms": best5, "sum_best_grid_ms": bestgrid,
            "sum_stock_ms": stock,
            "headroom_best5_ms": planner - best5, "headroom_grid_ms": planner - bestgrid,
            "best_static_plan": best_static, "best_static_sum_ms": totals[best_static],
            "headroom_static_ms": planner - totals[best_static],
            "static_totals_ms": dict(sorted(totals.items(), key=lambda kv: kv[1])),
            "best_of_5_win_counts": dict(wins),
            "split_half": split,
            "nax_speedup_vs_stock": stock / planner,
        }
    out["nax_calls"] = per_proj
    nax_total = sum(per_proj[p]["sum_planner_ms"] for p in per_proj)
    six_gu = sum(c["median_ms"]["gate_up|stock"] for c in six)
    six_dn = sum(c["median_ms"]["down|stock"] for c in six)
    out["stock_6bit_calls"] = {"sum_gate_up_ms": six_gu, "sum_down_ms": six_dn}
    gather_total = nax_total + six_gu + six_dn
    out["gather_share"] = {
        "nax_4bit_ms": nax_total, "stock_6bit_ms": six_gu + six_dn, "all_routed_gathers_ms": gather_total,
        "all_routed_gathers_share_of_e2e": gather_total / e2e_fused,
        "nax_share_of_e2e": nax_total / e2e_fused,
        "sync_overhead_note_ms": base * 2 * len(calls),
    }
    h = {}
    for kind in ("best5", "grid", "static"):
        ms = sum(per_proj[p][f"headroom_{kind}_ms"] for p in per_proj)
        h[kind] = {"ms": ms, "pct_of_nax_gather": 100 * ms / nax_total, "pct_of_e2e": 100 * ms / e2e_fused}
    if all(per_proj[p]["split_half"] for p in per_proj):
        ms = sum(per_proj[p]["split_half"]["headroom_ms"] for p in per_proj)
        h["split_half_oracle"] = {"ms": ms, "pct_of_e2e": 100 * ms / e2e_fused}
    out["headroom"] = h
    out["histogram_planner_worth_building"] = h["best5"]["pct_of_e2e"] >= 3.0

    # C2: routed gate+up share and A8 Amdahl.
    gu = per_proj["gate_up"]["sum_planner_ms"] + six_gu
    frac = gu / e2e_fused
    frac_4 = per_proj["gate_up"]["sum_planner_ms"] / e2e_fused
    amdahl = {}
    for s in (1.3, 1.4, 1.5):
        amdahl[str(s)] = {
            "all_layers_gain_pct": 100 * (1 / (1 - frac + frac / s) - 1),
            "nax_4bit_layers_only_gain_pct": 100 * (1 / (1 - frac_4 + frac_4 / s) - 1),
        }
    out["c2"] = {"routed_gate_up_ms": gu, "routed_gate_up_fraction": frac,
                 "nax_4bit_gate_up_fraction": frac_4, "stock_6bit_gate_up_fraction": six_gu / e2e_fused,
                 "amdahl": amdahl}
    (d / f"analysis-{tiles_dir}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
