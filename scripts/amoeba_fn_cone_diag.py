#!/usr/bin/env python3
"""Cone diagnosis for FN boxes (Pierre / coordinator 2026-11-16: "is lang_cmn's 95%-of-pool box a cone artifact?").
Inside a GPU window; one model load for every run.

For each --run (a carve run dir with ranking_zero.npz and carve.json: a pilot box or a lambda arm), rebuilds that run's
own val and held sets with the run's loader arguments (same rows, windows, crops and loss masks; --val-windows may
enlarge val, recorded), then scores, at each free-unit keep fraction of --fracs (share of the units outside T2):
  learned   T2 + the top fraction of the run's learned ranking (the driver's global cut)
  random    T2 + a random core with the learned core's per-layer counts (the driver's _random_like, seeded)
on val and on held (pooled dCE, held rows 1 greedy : 3 T=0.7 as the boxes). separation = random - learned.
Optionally (--held-bisect, a DIAGNOSTIC, never an export: held stays the test set for every box verdict) the learned
ranking's free fraction at held +0.05 (8 bisection steps).

Reading (coordinator): learned no better than random at <= 30% free -> the cone did not narrow (lambda too weak or too
few steps); learned clearly better at low fractions but the export still climbed -> the thin-val bisect.

Usage: amoeba_fn_cone_diag.py --model M --t2 T2.npz --out OUT.json --run NAME=DIR [--run ...] [--fracs ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    ap.add_argument("--fracs", default="0.05,0.1,0.2,0.3,0.5,0.7,0.9")
    ap.add_argument("--val-windows", type=int, default=0, help="val windows per document (0 = the run's own)")
    ap.add_argument("--held-bisect", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import corpus as C
    from research.amoeba.carve import evals as E
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists (evidence is never overwritten)")
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    O.footprint_watchdog(92, log)
    O.setup_mlx(88, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    import os
    tables = getattr(getattr(host, "adapter", None), "_tables", None)
    if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
            Path(a.model).expanduser().resolve() / "ple_rows.bin":
        raise SystemExit("PLE offload is not active (FN rule: assert it)")
    fixed = D.load_mask(f"{a.t2}:core", host, expert_units=True)
    fx = D.as_np(fixed, host) > 0
    valid = np.zeros_like(fx)
    for l, n in enumerate(host.widths):
        valid[l, :n] = True
    nfree = int((valid & ~fx).sum())
    fracs = [float(x) for x in a.fracs.split(",") if x]
    res = {"schema": "mlx2.amoeba-fn-cone-diag.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "t2": a.t2, "free_units": nfree, "fracs_of_free": fracs, "val_windows_override": a.val_windows or None,
           "random": "per-layer counts of the learned core, seeded (the driver's _random_like)",
           "held_weighting": "pooled over rows", "mlx2_source_revision": __import__("subprocess").run(
               ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None,
           "runs": {}}

    def cut(Z, ff):
        s = Z.astype(np.float64).copy()
        s[~valid | fx] = -np.inf
        k = max(1, int(round(ff * nfree)))
        thr = np.sort(s[np.isfinite(s)])[-k]
        return (fx | ((s >= thr) & np.isfinite(s))).astype(np.float32)

    for spec in a.run:
        name, _, d = spec.partition("=")
        d = Path(d)
        cj = json.loads((d / "carve.json").read_text())
        g = cj["args"]
        if cj.get("identity", {}).get("moe_units") != "neuron":
            raise SystemExit(f"{name}: not a neuron-unit run")
        # the rebuild must be this run's model, base tier and data (review 2026-11-16)
        if host.fingerprint != cj.get("fingerprint"):
            raise SystemExit(f"{name}: fingerprint {host.fingerprint} != the run's {cj.get('fingerprint')}")
        if g.get("fixed") != f"{a.t2}:core" or not g.get("fixed_expert_units"):
            raise SystemExit(f"{name}: --t2 {a.t2} is not the run's fixed tier {g.get('fixed')} (expert units)")
        Z = np.load(d / "ranking_zero.npz")["Z"]
        split, crec = C.load(g["corpus"], host, seq_len=g["seq_len"], n_mean=g["n_mean"], n_fit=g["n_fit"],
                             n_held=g["n_held"], seed=g["seed"], log=log, windows=g["windows"],
                             window_seed=g["window_seed"], strip_prefix=g["strip_prefix"],
                             held_file=g["held_file"] or None, held_len=g["held_len"], keep_prompt=g["keep_prompt"],
                             val_file=g["val_file"] or None, code_cut=g["code_cut"],
                             val_windows=a.val_windows or g.get("val_windows", 0))
        for k in ("sha256_12", "held_sha256_12", "val_sha256_12"):
            if crec.get(k) != (cj.get("corpus") or {}).get(k):
                raise SystemExit(f"{name}: rebuilt {k} {crec.get(k)} != the run's {(cj.get('corpus') or {}).get(k)}")
        if not (a.val_windows or 0) and crec.get("val_counts") != (cj.get("corpus") or {}).get("val_counts"):
            raise SystemExit(f"{name}: rebuilt val counts {crec.get('val_counts')} != the run's")
        held = {dn: {"held": v["held"]} for dn, v in split.items() if v.get("held")}
        val = {dn: {"spare": v} for dn, v in C.VAL.items() if v}
        ch, cv = {}, {}
        sc = lambda M, s, part, cc: E.english_cost(host, s, cc, E.configure(host, D.from_np(M, host), None, host.L),  # noqa: E731
                                                     part=part, pooled=True)["mean_d_ce"]
        rr = {"dir": str(d), "lam": g["lam"], "steps": g["steps"], "lr": g["lr"],
              "val_rows": sum(len(v["spare"]) for v in val.values()), "held_rows": sum(len(v["held"]) for v in held.values()),
              "trained_free_frac": float(((Z > 0) & valid & ~fx).sum() / nfree),
              "export_free_frac": None, "rows": []}
        if (d / "export_core.npz").exists():
            ex = np.load(d / "export_core.npz")["core"] > 0
            rr["export_free_frac"] = float((ex & ~fx & valid).sum() / nfree)
        for i, ff in enumerate(fracs):
            M = cut(Z, ff)
            R = D.as_np(D._random_like(D.from_np(M, host), fixed, None, host, a.seed + i), host)
            row = {"free_frac": ff, "learned_val": sc(M, val, "spare", cv), "random_val": sc(R, val, "spare", cv),
                   "learned_held": sc(M, held, "held", ch), "random_held": sc(R, held, "held", ch)}
            row["sep_val"] = row["random_val"] - row["learned_val"]
            row["sep_held"] = row["random_held"] - row["learned_held"]
            rr["rows"].append(row)
            log(name, json.dumps({k: round(v, 4) for k, v in row.items()}))
        if a.held_bisect:          # diagnostic size only; never an export or a verdict
            lo, hi, ev = 0.0, 1.0, []
            for _ in range(8):
                mid = (lo + hi) / 2
                dd = sc(cut(Z, mid), held, "held", ch)
                ev.append((round(mid, 4), round(dd, 4)))
                lo, hi = (mid, hi) if dd > 0.05 else (lo, mid)
            rr["held_0.05_free_frac_diagnostic"] = hi
            rr["held_bisect_evals"] = ev
        low = [r for r in rr["rows"] if r["free_frac"] <= 0.3]
        rr["reading"] = ("cone did not narrow (learned ~ random at <= 30% free)"
                         if low and max(r["sep_held"] for r in low) < 0.1 else
                         "learned clearly better than random at low fractions")
        res["runs"][name] = rr
        Path(str(out) + ".partial").write_text(json.dumps(res, indent=1))     # a later run's crash keeps this one
    res["mlx_peak_gb"] = O.peak_gb()
    out.write_text(json.dumps(res, indent=1))
    for n, rr in res["runs"].items():
        print(f"\n{n}: lam {rr['lam']} steps {rr['steps']} val rows {rr['val_rows']} held rows {rr['held_rows']} "
              f"trained {rr['trained_free_frac']:.3f} export {rr['export_free_frac']} -> {rr['reading']}")
        print("free | learned val | random val | learned held | random held | sep held")
        for r in rr["rows"]:
            print(f"{r['free_frac']:.2f} | {r['learned_val']:.3f} | {r['random_val']:.3f} | {r['learned_held']:.3f} | "
                  f"{r['random_held']:.3f} | {r['sep_held']:.3f}")
        if "held_0.05_free_frac_diagnostic" in rr:
            print("held +0.05 at free frac (diagnostic):", rr["held_0.05_free_frac_diagnostic"])


if __name__ == "__main__":
    main()
