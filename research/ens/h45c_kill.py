"""H45c kill measures (LoopRes read vs the H45 plain loop) over core receipts in one directory (Pierre, 2026-11-25).
Means over the seeds given; the directory holds loop-s<k>/ loopres-s<k>/ [loopres_content-s<k>/ uniform3-s<k>/]
(symlinks to the H45 receipts are fine).

  python -m research.ens.h45c_kill <dir> --seeds 0 1 2 3 4

  K1 depth   loopres - loop bus accuracy at unseen depths 4-6 < 0.05                (the read does not buy depth)
  K2 cost    loop - loopres bus accuracy on trained depths 1-3 > 0.02                (the read costs the core)
  Reported, not kills: the trained-depth ordinary cost; the one-pass attribution (loopres - loop on bus_one_pass
  and ordinary_one_pass: loop's one pass is the plain graph, loopres's is its read at one pass, so a gain there
  comes from the cross-layer read alone, not from more passes); loopres_content (the role-key ablation); uniform3.

Receipts are compared only when their training settings agree (steps, batch, accum, lr, p_ordinary, warmup, and the
recorded machine and device class); the settings are printed, and a mismatch or a missing value is an error, never
a silent drop. Receipts written before --accum existed ran with accum 1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARMS = ("loop", "loopres", "loopres_content", "uniform3")
NEEDED = ("loop", "loopres")
SETTINGS = ("steps", "batch", "accum", "lr", "p_ordinary", "warmup")


def _receipt(root, arm, seed):
    p = Path(root) / f"{arm}-s{seed}" / "receipt.json"
    return json.loads(p.read_text()) if p.exists() else None


def settings(r) -> dict:
    a = {"accum": 1} | r.get("args", {})
    out = {k: a.get(k) for k in SETTINGS}
    out["machine"] = r.get("machine")
    out["device"] = r.get("device")
    out["mlx"] = r.get("mlx")            # review 2026-11-25: an MLX change mid-campaign shifts numerics
    out["host"] = r.get("host")
    return out


def kills(root, seeds, *, log=sys.stderr):
    R = {arm: {k: _receipt(root, arm, k) for k in seeds} for arm in ARMS}
    for arm in NEEDED:
        missing = [k for k, r in R[arm].items() if r is None]
        if missing:
            raise SystemExit(f"missing {arm} receipts for seeds {missing}")
    # optional arms are used only when present for every seed; a partial set is an error, not a silent subset
    for arm in ARMS:
        have = [k for k, r in R[arm].items() if r is not None]
        if have and len(have) != len(seeds):
            raise SystemExit(f"{arm} receipts exist for seeds {have} but not all of {seeds}")
    used = {arm: R[arm] for arm in ARMS if all(r is not None for r in R[arm].values())}
    table = {f"{arm}-s{k}": settings(r) for arm, rs in used.items() for k, r in rs.items()}
    print("training settings:", file=log)
    for name, st in table.items():
        print(f"  {name:22s} {json.dumps(st)}", file=log)
    for name, st in table.items():
        unset = [k for k in SETTINGS if st[k] is None]
        if unset:
            raise SystemExit(f"{name}: training setting(s) {unset} not recorded")
    distinct = {json.dumps(st, sort_keys=True) for st in table.values()}
    if len(distinct) > 1:
        raise SystemExit("refusing to compare receipts with different training settings:\n  " +
                         "\n  ".join(sorted(distinct)))

    def m(arm, block, key):
        xs = []
        for k, r in used[arm].items():
            v = r.get("summary", {}).get(block, {}).get(key)
            if v is None:
                raise SystemExit(f"{arm}-s{k}: summary[{block!r}][{key!r}] missing")
            xs.append(v)
        return round(sum(xs) / len(xs), 4)

    out = {"seeds": seeds, "settings": json.loads(distinct.pop())}
    lu, ru = m("loop", "ALL.unseen", "bus"), m("loopres", "ALL.unseen", "bus")
    out["K1_depth"] = {"loopres_unseen_bus": ru, "loop_unseen_bus": lu, "loopres_minus_loop": round(ru - lu, 4),
                       "op_guess": m("loop", "ALL.unseen", "op_guess"),
                       "per_class": {c: {"loopres": m("loopres", f"{c}.unseen", "bus"),
                                         "loop": m("loop", f"{c}.unseen", "bus")} for c in ("GRAPH", "FST", "PROGRAM")},
                       "killed": round(ru - lu, 4) < 0.05}
    cost = round(m("loop", "ALL.trained", "bus") - m("loopres", "ALL.trained", "bus"), 4)
    out["K2_cost"] = {"loop_minus_loopres_trained_bus": cost, "killed": cost > 0.02}
    out["ordinary_cost_reported"] = {
        "loop_minus_loopres_trained_ordinary": round(m("loop", "ALL.trained", "ordinary")
                                                     - m("loopres", "ALL.trained", "ordinary"), 4)}
    out["one_pass_attribution"] = {
        f"{blk}.{key}": round(m("loopres", f"ALL.{blk}", key) - m("loop", f"ALL.{blk}", key), 4)
        for blk in ("trained", "unseen") for key in ("bus_one_pass", "ordinary_one_pass")}
    if "loopres_content" in used:
        cu = m("loopres_content", "ALL.unseen", "bus")
        out["content_ablation"] = {"loopres_content_unseen_bus": cu,
                                   "loopres_content_minus_loop_unseen": round(cu - lu, 4),
                                   "loopres_minus_loopres_content_unseen": round(ru - cu, 4),
                                   "loopres_content_trained_bus": m("loopres_content", "ALL.trained", "bus")}
    if "uniform3" in used:
        out["compute_control"] = {"uniform3_unseen_bus": m("uniform3", "ALL.unseen", "bus"),
                                  "uniform3_trained_bus": m("uniform3", "ALL.trained", "bus")}
    out["any_killed"] = any(v["killed"] for k, v in out.items() if k.startswith("K"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    print(json.dumps(kills(a.root, a.seeds), indent=1))


if __name__ == "__main__":
    main()
