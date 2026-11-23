"""A0 decision from confirmation receipts (wiki/docs/ens/a0-decision-record.md, amended 2026-11-22).

Primary: held accuracy of the MLP contributor, C6 - C7, per (confirm family, paired seed). Seeds are averaged
within a family and families are weighted equally. The uncertainty is a bootstrap over families and over paired
seed IDs on separate axes, keeping arms paired, with leave-one-family-out alongside. Also reported: C6 - C6R, and
on - max(bypass, z-blind) per arm.

  advance        lower 95% bound of C6 - C7 > delta, regression <= epsilon, BUS-001 passes on every family
                 (otherwise blocked_by_regression / blocked_by_bus001)
  harness_failure  the positive control (min per family and seed) or CORE-001 fails
  stop           upper 95% bound < delta
  inconclusive   otherwise
A failed positive control is a harness failure, not a negative result.

  python -m research.ens.decide <confirm evidence dir>
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from research.ens import families as F

DELTA = 0.10          # approved 2026-11-21
EPSILON = 0.02
BOTH_MIN = 0.5
WRONG_MARGIN = 0.05
POSITIVE_CONTROL = 0.9
MIN_PAIRED_SEEDS = 5
B = 4000


CONFIRM = ("program.MUL.h1", "program.XOR.h1", "graph.6-1.h1", "graph.8-1.h1")
ARMS = ("c6", "c6r", "c7")
GATING_ARMS = ("c6", "c7")     # option A (Pierre 2026-11-22): C6R is reported only and never gates the verdict
# the protocol the C6 / C7 cores must have been trained with (dev choices: H29 schedule, H30 depth); set by --protocol
PROTOCOL = None
KINDS = ("mlp", "oracle-table")


def load(root: Path):
    """(data, meta). data = {(arm, kind, family, seed): results} from completed seeds only (seed<k>/DONE).
    Every receipt is checked against its directory name and the confirmation protocol (review 2026-11-22)."""
    out, meta = {}, {"excluded_seeds": [], "core001": {}, "rejected": [], "completed_seeds": []}
    for sd in sorted(root.glob("seed*")):
        if not sd.is_dir() or not sd.name[4:].isdecimal():
            continue
        seed = int(sd.name[4:])
        if sd.name != f"seed{seed}":
            raise SystemExit(f"ambiguous seed directory {sd.name!r}")
        if not (sd / "DONE").exists():
            meta["excluded_seeds"].append(seed)
            continue
        meta["completed_seeds"].append(seed)
        checkpoints = {}
        for arm in ARMS:
            core = sd / f"core-{arm}"
            receipt = core / "receipt.json"
            trained = json.loads(receipt.read_text()) if receipt.is_file() else {}
            ckpt = trained.get("checkpoint")
            ok_core = bool(ckpt and trained.get("arm") == arm and trained.get("seed") == seed)
            if ok_core and arm in GATING_ARMS and PROTOCOL is not None:
                c, a_ = trained.get("config", {}), trained.get("args", {})
                got = {"site": trained.get("site"), "tau": c.get("tau"), "bus_width": c.get("bus_width"),
                       "layers": c.get("layers"), "d": c.get("d"), "steps": a_.get("steps"),
                       "p_ordinary": a_.get("p_ordinary"), "matryoshka": a_.get("matryoshka", 0.0)}
                if got != PROTOCOL:
                    meta.setdefault("protocol_mismatch", []).append({"core": f"{arm}/seed{seed}", "got": got})
                    ok_core = False
            if ok_core:
                checkpoints[arm] = ckpt
            oracle = core / "core001.json"
            checked = json.loads(oracle.read_text()) if oracle.is_file() else {}
            meta["core001"][f"{arm}/seed{seed}"] = bool(arm in checkpoints
                and checked.get("core_checkpoint") == checkpoints[arm] and checked.get("pass") is True)
        for p in sorted(sd.glob("eval-*/receipt.json")):
            r = json.loads(p.read_text())
            arm = p.parent.name[5:].split("-")[0]
            family, kind = r.get("family"), r.get("module_kind")
            results = r.get("results", {})
            metrics = ("on", "both", "wrong", "bypass", "zblind", "ordinary_suite")
            ok = (r.get("schema") == "ens.a0.eval_receipt.v1" and arm in checkpoints and arm == r.get("arm")
                  and r.get("core_checkpoint") == checkpoints[arm] and r.get("split") == "held"
                  and r.get("quant", "none") == "none" and results.get("half", "full") == "full"
                  and family in CONFIRM and r.get("family_digest") == F.BY_ID[family].digest and kind in KINDS
                  and p.parent.name == f"eval-{arm}-{kind}-{family}"
                  and all(type(results.get(k)) in (float, int) and math.isfinite(results[k])
                          and 0 <= results[k] <= 1 for k in metrics))
            if not ok:
                meta["rejected"].append(str(p.relative_to(root)))
                continue
            out[(arm, r["module_kind"], r["family"], seed)] = r["results"]
    return out, meta


def complete_grid(data, seeds, *, missing_as_loss=False):
    """Every (arm, kind, family, seed) cell must exist. A missing cell fails closed, or with missing_as_loss
    (the record's second-failure rule) counts as no extension: on = bypass, both = 0."""
    missing = [(a, k, f, s) for a in GATING_ARMS for k in KINDS for f in CONFIRM for s in seeds if (a, k, f, s) not in data]
    if missing and not missing_as_loss:
        raise SystemExit(f"incomplete grid, {len(missing)} cells missing, e.g. {missing[:4]}")
    for a, k, f, s in missing:
        ref = next((data[(a, kk, f, s)] for kk in KINDS if (a, kk, f, s) in data), None)
        if ref is None:
            raise SystemExit(f"cannot score missing cell as bypass without a measured baseline for {(a, f, s)}")
        byp = ref["bypass"]
        data[(a, k, f, s)] = {"on": byp, "bypass": byp, "zblind": ref["zblind"],
                              "both": 0.0, "wrong": byp, "ordinary_suite": ref["ordinary_suite"],
                              "missing": True}
    return missing


def _mean(xs):
    return sum(xs) / len(xs)


def estimate(diff):
    """diff: {family: {seed: value}} -> seed-mean per family, then an equal-weight family mean."""
    return _mean([_mean(list(s.values())) for s in diff.values()])


def bootstrap(diff, b=B, seed=0):
    rng = random.Random(seed)
    fams = sorted(diff)
    seeds = sorted({s for d in diff.values() for s in d})
    vals = []
    for _ in range(b):
        fs = [fams[rng.randrange(len(fams))] for _ in fams]
        ss = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        per = []
        for f in fs:
            xs = [diff[f][s] for s in ss if s in diff[f]]
            if xs:
                per.append(_mean(xs))
        vals.append(_mean(per))
    vals.sort()
    return vals[int(0.025 * b)], vals[int(0.975 * b) - 1]


def decide(data, meta=None, *, missing_as_loss=False) -> dict:
    data = dict(data)  # missing-cell scores are local to this verdict
    meta = meta or {"excluded_seeds": [], "core001": {}, "rejected": []}
    fams = list(CONFIRM)
    seeds = sorted(meta.get("completed_seeds", {k[3] for k in data}))
    if not seeds:
        raise SystemExit("no completed seeds")
    missing = complete_grid(data, seeds, missing_as_loss=missing_as_loss)

    def on(arm, f, s):
        r = data.get((arm, "mlp", f, s))
        return None if r is None else r["on"]
    diffs = {}
    for other in ("c6r", "c7"):
        d = defaultdict(dict)
        for f in fams:
            for s in seeds:
                a, b_ = on("c6", f, s), on(other, f, s)
                if a is not None and b_ is not None:
                    d[f][s] = a - b_
        diffs[other] = dict(d)
    prim = diffs["c7"]
    lo, hi = bootstrap(prim)
    res = {"families": fams, "seeds": seeds, "delta": DELTA, "epsilon": EPSILON,
           "excluded_seeds": meta["excluded_seeds"], "rejected_receipts": meta["rejected"],
           "missing_cells_counted_as_loss": [list(m) for m in missing],
           "primary": "c6_minus_c7", "c6_minus_c7": estimate(prim), "ci95": [lo, hi],
           "leave_one_family_out": {f: estimate({g: v for g, v in prim.items() if g != f}) for f in prim},
           "c6_minus_c6r": estimate(diffs["c6r"]) if any(diffs["c6r"].values()) else None,   # reported only
           "per_family": {f: {"c6_minus_c7": _mean(list(prim[f].values()))} for f in prim}}
    # margins over the bars each arm must beat
    res["margin_over_bar"] = {arm: {f: _mean([data[(arm, "mlp", f, s)]["on"] - max(data[(arm, "mlp", f, s)]["bypass"],
                                                                                  data[(arm, "mlp", f, s)].get("zblind") or 0)
                                             for s in seeds if (arm, "mlp", f, s) in data])
                                    for f in fams if any((arm, "mlp", f, s) in data for s in seeds)}
                              for arm in ("c6", "c6r", "c7")}
    # BUS-001 on C6, per family (seed means)
    bus = {}
    for f in fams:
        rs = [data[("c6", "mlp", f, s)] for s in seeds if ("c6", "mlp", f, s) in data]
        both, wrong, byp = _mean([r["both"] for r in rs]), _mean([r["wrong"] for r in rs]), _mean([r["bypass"] for r in rs])
        bus[f] = {"both": both, "wrong": wrong, "bypass": byp, "pass": both >= BOTH_MIN and wrong <= byp + WRONG_MARGIN}
    res["bus001"] = bus
    # regression: ordinary suite, bus empty, C7 - C6 (positive = C6 worse)
    ords = {arm: _mean([r["ordinary_suite"] for k, r in data.items() if k[0] == arm]) for arm in ("c6", "c7")}
    res["ordinary_suite"] = ords
    res["regression"] = ords["c7"] - ords["c6"]
    # positive control: the oracle-table probe on C6 reaches the ceiling on every family
    # the minimum per (family, seed): one weak seed must not hide behind four good ones (review)
    pc = {f: min(data[("c6", "oracle-table", f, s)]["on"] for s in seeds) for f in fams}
    res["positive_control"] = {"min_per_family": pc, "pass": all(v >= POSITIVE_CONTROL for v in pc.values())}
    c001 = meta["core001"]
    gating = {k: v for k, v in c001.items() if k.split("/")[0] in GATING_ARMS}
    res["core001"] = {"per_core": c001, "pass": bool(gating) and all(v is True for v in gating.values()),
                      "note": "C6R is reported only; its CORE-001 does not gate"}
    res["protocol"] = PROTOCOL
    res["protocol_mismatch"] = meta.get("protocol_mismatch", [])
    bus_ok = all(b_["pass"] for b_ in bus.values())
    reg_ok = res["regression"] <= EPSILON
    if not res["positive_control"]["pass"] or not res["core001"]["pass"]:
        res["verdict"] = "harness_failure"
    elif len(seeds) < MIN_PAIRED_SEEDS:
        res["verdict"] = "incomplete"
    elif hi < DELTA:
        res["verdict"] = "stop"
    elif lo > DELTA:
        res["verdict"] = "advance" if (bus_ok and reg_ok) else \
            ("blocked_by_regression" if not reg_ok else "blocked_by_bus001")
    else:
        res["verdict"] = "inconclusive"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--protocol", required=True,
                    help="JSON of the training protocol C6/C7 cores must match: site, tau, bus_width, layers, d, "
                         "steps, p_ordinary, matryoshka")
    ap.add_argument("--missing-as-loss", action="store_true",
                    help="apply the record's second-failure rule to missing cells instead of failing closed")
    args = ap.parse_args(argv)
    global PROTOCOL
    PROTOCOL = json.loads(args.protocol)
    data, meta = load(Path(args.root))
    print(json.dumps(decide(data, meta, missing_as_loss=args.missing_as_loss), indent=1))


if __name__ == "__main__":
    main()
