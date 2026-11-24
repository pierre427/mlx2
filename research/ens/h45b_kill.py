"""H45b kill measures (ledger; revised after the 2026-11-24 review):

  gain       gain6 minus fixed6 on unseen depths 4-6 (bus), per seed and pooled; also gain6 minus decay6 (what kappa
             adds over pure decay). Kill: gain6 - fixed6 < 0.05.
  adaptive   per-ROW Spearman between true depth and effective depth (1 + sum of round gains), bus mode, on GRAPH
             families (length-matched: every depth has the same prompt length, so prompt length cannot drive it) and
             on all families; plus the effect size: mean effective depth at depth 6 minus depth 1 (GRAPH). Kill: GRAPH
             rho < 0.3 or effect size < 0.25 (a rank test alone passes on 1e-4 differences). decay6 is the control:
             its effective depth is constant by construction, so its rho is undefined (None).
  reference  the H45 loop arm told its depth (passes = steps), from --ref <H45 evidence dir>. NOT step-matched: it
             trained 48k steps, the H45b arms 24k. Seeds without a reference receipt are listed (never silent).
  note       GRAPH prompts state the hop count (HOP V(hops)); a pass means effective depth follows that token, not
             that the loop discovered the depth on its own.

  python -m research.ens.h45b_kill <evidence dir> <seed>... [--ref <H45 evidence dir>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx_, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx_) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx_) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def adaptive(depth, cls=None):
    xs, ys, by = [], [], {}
    for v in depth.values():
        if "bus_eff_depth_rows" not in v or (cls and v["class"] != cls):
            continue
        xs += [v["steps"]] * len(v["bus_eff_depth_rows"])
        ys += v["bus_eff_depth_rows"]
        by.setdefault(v["steps"], []).extend(v["bus_eff_depth_rows"])
    if not xs:
        return None
    means = {k: sum(r) / len(r) for k, r in sorted(by.items())}
    rho = spearman(xs, ys)
    return {"rho_rows": None if rho is None else round(rho, 4),
            "effect_d6_minus_d1": round(means[max(means)] - means[min(means)], 4),
            "mean_by_depth": {str(k): round(m, 4) for k, m in means.items()}}


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    ref = None
    if "--ref" in argv:
        i = argv.index("--ref")
        ref = Path(argv[i + 1])
        del argv[i:i + 2]
    ev, seeds = Path(argv[0]), argv[1:]
    out = {"per_seed": {}}
    for s in seeds:
        row = {}
        for arm, base in (("fixed6", ev), ("decay6", ev), ("gain6", ev), ("loop", ref)):
            f = None if base is None else base / f"{arm}-s{s}" / "receipt.json"
            if f is None or not f.exists():
                continue
            r = json.loads(f.read_text())
            ks = [x["kappa"] for x in r.get("log", []) if "kappa" in x]
            tail = ks[-max(1, len(ks) // 10):] if ks else []
            row[arm] = {"kappa_last10pct": ({"mean": round(sum(tail) / len(tail), 4),
                                             "mean_abs": round(sum(abs(x) for x in tail) / len(tail), 4)}
                                            if tail else None),
                        **{"unseen_bus": r["summary"]["ALL.unseen"]["bus"], "trained_bus": r["summary"]["ALL.trained"]["bus"],
                        "loop_gain": r.get("loop_gain")}}
            if arm in ("decay6", "gain6"):
                row[arm]["adaptive_graph"] = adaptive(r["depth_eval"], "GRAPH")
                row[arm]["adaptive_all"] = adaptive(r["depth_eval"])
        out["per_seed"][s] = row
        if ref is not None and "loop" not in row:
            out.setdefault("reference_missing_seeds", []).append(s)
    done = [r for r in out["per_seed"].values() if all(a in r for a in ("gain6", "fixed6", "decay6"))]
    if done:
        mean = lambda xs: round(sum(xs) / len(xs), 4)
        out["gain6_minus_fixed6_unseen"] = mean([r["gain6"]["unseen_bus"] - r["fixed6"]["unseen_bus"] for r in done])
        out["gain6_minus_decay6_unseen"] = mean([r["gain6"]["unseen_bus"] - r["decay6"]["unseen_bus"] for r in done])
        rhos = [r["gain6"]["adaptive_graph"]["rho_rows"] for r in done]
        effs = [r["gain6"]["adaptive_graph"]["effect_d6_minus_d1"] for r in done]
        out["gain6_graph_rho_rows"] = None if None in rhos else mean(rhos)
        out["gain6_graph_effect"] = mean(effs)
        out["kill"] = {"gain_too_small": out["gain6_minus_fixed6_unseen"] < 0.05,
                       "depth_not_adaptive": out["gain6_graph_rho_rows"] is None or out["gain6_graph_rho_rows"] < 0.3
                       or out["gain6_graph_effect"] < 0.25}
        out["verdict"] = "killed" if any(out["kill"].values()) else "supported"
    if ref is not None and len(out.get("reference_missing_seeds", [])) == len(seeds):
        print(json.dumps(out, indent=1))
        raise SystemExit(f"--ref {ref}: no H45 loop receipt for any seed")   # review: never drop it silently
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
