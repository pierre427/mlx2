"""H45 kill measures over paired loop / fixed / uniform3 core receipts (computed here so every reader applies the
same rule; review 2026-11-23). Means over the seeds given.

  python -m research.ens.h45_kill <dir with loop-s<k>/ fixed-s<k>/ [uniform3-s<k>/]> --seeds 0 1 2

  K1 depth      loop bus accuracy at unseen depths 4-6 <= fixed + 0.10      (more passes do not buy depth)
  K2 cost       fixed - loop > 0.02 on trained depths, bus or ordinary         (the port costs the core)
  K3 work       at unseen depths, loop with s - 1 passes matches the s answer within 0.05 of loop with s passes
                (the pass count is not doing the work; the oracle agreement floor is reported beside it)
  Compute control (reported, not a kill): loop vs uniform3 at unseen depths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _s(root, arm, seed):
    p = Path(root) / f"{arm}-s{seed}" / "receipt.json"
    return json.loads(p.read_text())["summary"] if p.exists() else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def kills(root, seeds):
    S = {arm: [_s(root, arm, k) for k in seeds] for arm in ("loop", "fixed", "uniform3")}
    if any(x is None for x in S["loop"] + S["fixed"]):
        raise SystemExit("missing loop or fixed receipts")

    def m(arm, block, key):
        return _mean([s.get(block, {}).get(key) for s in S[arm] if s])
    out = {"seeds": seeds}
    lu, fu = m("loop", "ALL.unseen", "bus"), m("fixed", "ALL.unseen", "bus")
    out["K1_depth"] = {"loop_unseen_bus": lu, "fixed_unseen_bus": fu, "op_guess": m("loop", "ALL.unseen", "op_guess"),
                       "per_class": {c: {"loop": m("loop", f"{c}.unseen", "bus"), "fixed": m("fixed", f"{c}.unseen", "bus")}
                                     for c in ("GRAPH", "FST")},
                       "killed": lu <= fu + 0.10}
    cost = {mode: round(m("fixed", "ALL.trained", mode) - m("loop", "ALL.trained", mode), 4)
            for mode in ("bus", "ordinary")}
    out["K2_cost"] = {"fixed_minus_loop_trained": cost,
                      "loop_one_pass_ordinary_trained": m("loop", "ALL.trained", "ordinary_one_pass"),
                      "killed": max(cost.values()) > 0.02}
    ms = m("loop", "ALL.unseen", "bus_minus1_vs_s")
    out["K3_work"] = {"loop_s_passes": lu, "loop_s_minus1_vs_s_answer": ms,
                      "loop_s_minus1_vs_its_own_answer": m("loop", "ALL.unseen", "bus_minus1_vs_true"),
                      "oracle_agreement_floor": m("loop", "ALL.unseen", "minus1_agreement"),
                      "killed": ms >= lu - 0.05}
    if all(S["uniform3"]):
        out["compute_control"] = {"loop_unseen_bus": lu, "uniform3_unseen_bus": m("uniform3", "ALL.unseen", "bus"),
                                  "uniform3_trained_bus": m("uniform3", "ALL.trained", "bus"),
                                  "loop_trained_bus": m("loop", "ALL.trained", "bus")}
    out["any_killed"] = any(v["killed"] for k, v in out.items() if k.startswith("K"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    print(json.dumps(kills(p.parse_args(argv).root, p.parse_args(argv).seeds), indent=1))


if __name__ == "__main__":
    main()
