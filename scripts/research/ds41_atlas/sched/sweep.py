#!/usr/bin/env python3
"""Simulation sweeps for the DS4.1 Source expert-prefetch scheduler (CPU only, 4 worker processes).

Stages (each writes sweep_<stage>.json atomically; tuning uses TRAIN-request metrics only, every table
reports HELDOUT-request metrics separately):
  tune      prefetch k / priority / threshold, cycle alpha, nextuse horizon -- on train
  grid      RAM 40/60/75/90 GB x compute 6/10/12/15 ms x parse low/high x policy list
  t7        NVMe holds the hottest 60%/80% of experts (train frequency), the rest on the T7 (45 MB/s
            measured-prior or 400 MB/s); optional M3-RAM byte tier for the hottest T7-only experts
  pipeline  whole-layer ranges on M3/M2 (idle/busy RTT, compute factor 1.0/2.0/3.5/5.0)
  sens      NVMe bandwidth x0.5/x2, router_frac
usage: sweep.py STAGE [STAGE...]
"""
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

from common import OUT, atomic_json, provenance
from eventsim import run_one


def go(stage, specs):
    path = f"{OUT}/sweep_{stage}.json"
    t = time.time()
    out = provenance(stage=stage, n_runs=len(specs))
    with Pool(4) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(run_one, specs)):
            res.append(r)
            if i % 20 == 0:
                print(stage, i, len(specs), round(time.time() - t), flush=True)
    out["runs"] = res
    out["elapsed_s"] = round(time.time() - t, 1)
    atomic_json(path, out)
    print(stage, "done", out["elapsed_s"], flush=True)
    return res


def tuned():
    p = f"{OUT}/tuned.json"
    return json.load(open(p)) if os.path.exists(p) else {}


def S(cfg_over=None, **sim):
    return {"cfg_over": cfg_over or {}, "sim": sim}


def stage_tune():
    specs = []
    for parse in ("low", "high"):
        co = {"parse": parse}
        for pf in ("d1", "d2", "xtokAB", "xtokAB+d1", "xtokABp+d1", "xtokA"):
            for k in (4, 6, 8, 12, 16, 24):
                for prio in ("value", "edf"):
                    for th in ((0.0,) if parse == "low" else (0.5, 1.0, 2.0, 4.0)):
                        specs.append(S(co, policy="lru", prefetch=pf, k=k, prio=prio, theta_mult=th))
        for a in (0.1, 0.3, 0.5):
            specs.append(S(co, policy="cycle", cycle_alpha=a))
        for H in ("H16", "H64", "H256"):
            specs.append(S(co, policy="nextuse", nextuse_model=H))
            specs.append(S(co, policy="nextuse", nextuse_model=H + "_pn", prefetch="xtokAB+d1", k=8))
        specs.append(S(co, policy="lru"))
    return specs


def pick_tuned(res):
    """Best train s/token per (parse, prefetcher) and best eviction knobs."""
    best = {}
    for r in res:
        sim = r["sim"]; parse = r["cfg_over"].get("parse", "low")
        key = (parse, sim.get("policy"), sim.get("prefetch", "none"))
        s = r["result"]["train"]["s_per_token"]
        if key not in best or s < best[key][0]:
            best[key] = (s, sim)
    out = {}
    for (parse, pol, pf), (s, sim) in best.items():
        out[f"{parse}|{pol}|{pf}"] = {"train_s_per_token": s, "sim": sim}
    return out


def policies(t, parse):
    """Policy list for the grid, using tuned knobs (fall back to defaults)."""
    def knobs(pol, pf):
        e = t.get(f"{parse}|{pol}|{pf}")
        if not e:
            return {}
        return {k: v for k, v in e["sim"].items() if k in ("k", "prio", "theta_mult", "cycle_alpha", "nextuse_model")}
    cyc = knobs("cycle", "none").get("cycle_alpha", 0.3)
    nu = knobs("nextuse", "none").get("nextuse_model", "H64")
    nupn = knobs("nextuse", "xtokAB+d1").get("nextuse_model", "H64_pn")
    pf_k = {pf: knobs("lru", pf) for pf in ("d1", "xtokAB", "xtokAB+d1", "xtokA")}
    base = pf_k["xtokAB+d1"] or {"k": 8}
    P = [("lru", "none", {}), ("cycle", "none", {"cycle_alpha": cyc}), ("nextuse", "none", {"nextuse_model": nu}),
         ("lru", "d1", {"k": 8, **{k: v for k, v in pf_k["d1"].items() if k != "k"}}),   # K3.1 recipe k=8
         ("lru", "d1", pf_k["d1"] or {"k": 8}),
         ("lru", "xtokA", pf_k["xtokA"] or {"k": 8}),
         ("lru", "xtokAB", pf_k["xtokAB"] or {"k": 8}),
         ("lru", "xtokAB+d1", base),
         ("lru", "xtokABp+d1", base),
         ("cycle", "d1", {**(pf_k["d1"] or {"k": 8}), "cycle_alpha": cyc}),
         ("nextuse", "d1", {**(pf_k["d1"] or {"k": 8}), "nextuse_model": nu}),
         ("cycle", "xtokAB+d1", {**base, "cycle_alpha": cyc}),
         ("nextuse", "xtokAB+d1", {**base, "nextuse_model": nupn}),
         ("lru", "h2+d1", base), ("lru", "h3+d1", base),
         ("lru", "oracle_d1", {"k": 6}), ("lru", "oracle_xtok", {"k": 6}),
         ("belady", "none", {}), ("belady", "oracle_xtok", {"k": 6})]
    seen, out = set(), []
    for pol, pf, kw in P:
        key = json.dumps([pol, pf, kw], sort_keys=True)
        if key not in seen:
            seen.add(key); out.append((pol, pf, kw))
    return out


def stage_grid():
    t = tuned(); specs = []
    for parse in ("low", "high"):
        for ram, comp in itertools.product((40, 60, 75, 90), (6, 10, 12, 15)):
            for pol, pf, kw in policies(t, parse):
                specs.append(S({"parse": parse, "compute": comp}, policy=pol, prefetch=pf, ram_gb=ram, **kw))
    return specs


def stage_t7():
    t = tuned(); specs = []
    best = policies(t, "low")
    pfk = [kw for pol, pf, kw in best if pol == "lru" and pf == "xtokAB+d1"][0]
    d1k = [kw for pol, pf, kw in best if pol == "lru" and pf == "d1"][-1]
    nu = [kw for pol, pf, kw in best if pol == "nextuse" and pf == "none"][0]
    nupn = [kw for pol, pf, kw in best if pol == "nextuse" and pf == "xtokAB+d1"][0]
    for ram, bw, frac, m3 in itertools.product((40, 75), (0.045, 0.4), (0.6, 0.8), (False, True)):
        co = {"compute": 12, "t7_bw": bw}
        common = dict(ram_gb=ram, t7_subset=True, nvme_frac=frac, m3ram_tier=m3)
        for pol, pf, kw in (("lru", "none", {}), ("gds", "none", {}), ("gds", "none", {"gds_cap": 4.0}),
                            ("gds", "none", {"gds_cap": 16.0}), ("gds", "none", {"gds_cap": 64.0}), ("nextuse", "none", nu),
                            ("nextuse_cost", "none", nu), ("cycle_cost", "none", {}),
                            ("lru", "d1", d1k), ("gds", "d1", d1k), ("gds", "d1", {**d1k, "gds_cap": 16.0}),
                            ("lru", "xtokAB+d1", pfk), ("gds", "xtokAB+d1", pfk), ("gds", "h3+d1", pfk),
                            ("gds", "xtokAB+d1", {**pfk, "prio": "value"}),
                            ("nextuse_cost", "xtokAB+d1", {**pfk, **nupn}), ("belady", "none", {})):
            specs.append(S(co, policy=pol, prefetch=pf, **common, **kw))
    return specs


def contiguous(start, n):
    return list(range(start, start + n))


def stage_pipeline():
    t = tuned(); specs = []
    best = policies(t, "low")
    pfk = [kw for pol, pf, kw in best if pol == "lru" and pf == "d1"][-1]
    places = {"none": {}}
    # M3 ranges of 1..3 layers at a few positions; M2 ranges adjacent (one fewer hop) or absent
    for n3 in (1, 2, 3):
        for s3 in (0, 18, 37 - n3 + 1):
            places[f"m3[{s3}:{s3 + n3}]"] = {"m3": contiguous(s3, n3)}
    places["m3[34:37]+m2[37:39]"] = {"m3": contiguous(34, 3), "m2": contiguous(37, 2)}
    places["m3[37:40]+m2[35:37]"] = {"m3": contiguous(37, 3), "m2": contiguous(35, 2)}
    places["m2[38:40]"] = {"m2": contiguous(38, 2)}
    for (name, rem), ram, state, fac in itertools.product(places.items(), (40, 75), ("idle", "busy"), (1.0, 2.0, 3.5)):
        if name == "none" and (state != "idle" or fac != 1.0):
            continue
        co = {"compute": 12, "m3_factor": fac, "m2_factor": max(fac, fac * 5 / 3.5)}
        for pol, pf, kw in (("lru", "none", {}), ("lru", "d1", pfk)):
            specs.append({**S(co, policy=pol, prefetch=pf, ram_gb=ram, remote=rem, remote_state=state, **kw),
                          "placement": name})
    # T7-subset case where a remote stage removes expensive misses
    for (name, rem), state in itertools.product([("none", {}), ("m3[37:40]", {"m3": [37, 38, 39]}),
                                                 ("m3[37:40]+m2[35:37]", places["m3[37:40]+m2[35:37]"])], ("idle", "busy")):
        co = {"compute": 12, "t7_bw": 0.045, "m3_factor": 3.5, "m2_factor": 5.0}
        specs.append({**S(co, policy="gds", prefetch="d1", ram_gb=75, remote=rem, remote_state=state,
                          t7_subset=True, nvme_frac=0.6, **pfk), "placement": name, "t7": True})
    return specs


def stage_sens():
    t = tuned(); specs = []
    best = policies(t, "low")
    pick = [(p, f, k) for p, f, k in best if (p, f) in {("lru", "none"), ("cycle", "none"), ("nextuse", "none"),
                                                         ("lru", "d1"), ("lru", "xtokAB+d1"), ("nextuse", "xtokAB+d1"),
                                                         ("belady", "none")}]
    for nb, rf in itertools.product((5.25, 10.5, 21.0), (0.25, 0.5, 0.75)):
        for pol, pf, kw in pick:
            specs.append(S({"compute": 12, "nvme_bw": nb, "router_frac": rf}, policy=pol, prefetch=pf, ram_gb=75, **kw))
    return specs


def stage_extra():
    """Pin the unpredictable early layers (d1 has no input for layer 0) in M5 RAM, with/without d1."""
    t = tuned(); specs = []
    d1k = [kw for pol, pf, kw in policies(t, "low") if pol == "lru" and pf == "d1"][-1]
    for ram, pins in itertools.product((40, 75, 90), ((), (0,), (0, 1), (0, 1, 2))):
        for pol, pf, kw in (("lru", "none", {}), ("lru", "d1", d1k)):
            specs.append(S({"compute": 12}, policy=pol, prefetch=pf, ram_gb=ram, pin_layers=list(pins), **kw))
    return specs


def stage_t7b():
    """T7 follow-up: restrict speculative prefetch to tiers whose service time fits inside the d1 lead."""
    t = tuned(); specs = []
    best = policies(t, "low")
    d1k = [kw for pol, pf, kw in best if pol == "lru" and pf == "d1"][-1]
    pfk = [kw for pol, pf, kw in best if pol == "lru" and pf == "xtokAB+d1"][0]
    nu = [kw for pol, pf, kw in best if pol == "nextuse" and pf == "none"][0]
    nupn = [kw for pol, pf, kw in best if pol == "nextuse" and pf == "xtokAB+d1"][0]
    for frac, m3 in itertools.product((0.6, 0.8), (False, True)):
        co = {"compute": 12, "t7_bw": 0.045}
        common = dict(ram_gb=75, t7_subset=True, nvme_frac=frac, m3ram_tier=m3)
        for pol, pf, kw in (("gds", "d1", {**d1k, "prefetch_max_ms": 5.0}),
                            ("nextuse_cost", "d1", {**d1k, **nu, "prefetch_max_ms": 5.0}),
                            ("nextuse_cost", "d1", {**d1k, **nu}),
                            ("nextuse_cost", "xtokAB+d1", {**pfk, **nupn, "prefetch_max_ms": 5.0}),
                            ("cycle_cost", "d1", {**d1k, "cycle_alpha": 0.1, "prefetch_max_ms": 5.0})):
            specs.append(S(co, policy=pol, prefetch=pf, **common, **kw))
    return specs


def main():
    for stage in sys.argv[1:]:
        if stage == "tune":
            res = go("tune", stage_tune())
            atomic_json(f"{OUT}/tuned.json", pick_tuned(res))
        elif stage == "grid":
            go("grid", stage_grid())
        elif stage == "t7":
            go("t7", stage_t7())
        elif stage == "pipeline":
            go("pipeline", stage_pipeline())
        elif stage == "sens":
            go("sens", stage_sens())
        elif stage == "extra":
            go("extra", stage_extra())
        elif stage == "t7b":
            go("t7b", stage_t7b())


if __name__ == "__main__":
    main()
