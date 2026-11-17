#!/usr/bin/env python3
"""Cut a host model's stub for a topic, so a donor's stronger box for that topic can be remapped onto it.

Contract (DESIGN: docs/research/amoeba-topics/DESIGN.md, "Stub extraction"):
  1. candidates: the host's neurons for topic X, from its registry: units labelled X or a descendant
     whose functional gate passed; otherwise X's core-resident neuron map (core tag).
  2. rank candidates by per-neuron selectivity d for X on the FIT split; the stub is the shortest
     prefix whose removal costs X (fit texts) >= --frac of the whole candidate set's removal cost
     (bisection; monotonicity of the evaluated points is recorded as bisection_monotone).
  3. report on HELD texts: removal ΔNLL for X against a compartment-matched twin (same per-layer
     counts, core neurons replaced from the core, pool neurons from the pool), off-topic control,
     per-layer-band strength and the normalised layer profile.
  4. split: the stub's neurons leave their current units (core or boxes) and become a stub unit
     (split_units: exact, no neuron in two units, coverage unchanged).
The pure functions (select_prefix, split_units, compartment_twin, layer_profile, stub_record) are
unit-tested in tests/test_amoeba_topics_stub.py; the CLI runs them on a real model (CPU for small
models, or the 27B inside an owned GPU window).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T

SCHEMA = "mlx2.amoeba-topic-stub.v1"


# ----------------------------------------------------------------------------- pure functions

def select_prefix(n: int, cost, frac: float = 0.9, min_k: int = 1):
    """Smallest k in [min_k, n] with cost(k) >= frac * cost(n), by bisection. cost(k) = removal cost
    of the top-k ranked candidates. Returns k, cost(n), and the evaluated points (for the record)."""
    if n <= 0:
        return 0, 0.0, []
    evals = {}

    def c(k):
        if k not in evals:
            evals[k] = float(cost(k))
        return evals[k]
    total = c(n)
    if total <= 0:
        return n, total, sorted(evals.items())
    target = frac * total
    lo, hi = min_k, n
    if c(lo) >= target:
        return lo, total, sorted(evals.items())
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if c(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi, total, sorted(evals.items())


def is_monotone(evals, tol=1e-3):
    """True if the evaluated prefix costs never fall by more than tol as k grows (select_prefix's assumption)."""
    v = [c for _, c in sorted(evals)]
    return all(b >= a - tol for a, b in zip(v, v[1:]))


def masks_match_lists(mask: np.ndarray, pairs: np.ndarray) -> bool:
    """A bool [L,N] mask is bit-exact against its (layer, neuron) index list."""
    m = np.zeros_like(mask, dtype=bool)
    if len(pairs):
        m[pairs[:, 0], pairs[:, 1]] = True
    return bool((m == mask).all()) and len(np.unique(pairs, axis=0)) == len(pairs)


def split_units(units: dict, stub: np.ndarray, stub_name: str = "stub", shared: np.ndarray | None = None):
    """units: name -> bool [L, N] masks (disjoint). Returns new units with stub's neurons removed from
    each unit and a new unit stub_name; plus a per-unit count of neurons moved. Neurons in `shared`
    (several topics use them, weighted) are COPIED into the stub and stay in their unit, never moved.
    Raises if the inputs overlap or the stub covers neurons that no unit holds."""
    names = list(units)
    acc = np.zeros_like(stub, dtype=np.int32)
    for k in names:
        acc += units[k].astype(np.int32)
    if (acc > 1).any():
        raise ValueError("input units overlap")
    if (stub & (acc == 0)).any():
        raise ValueError("stub covers neurons outside every unit")
    out, moved = {}, {}
    take = stub if shared is None else stub & ~shared
    for k in names:
        out[k] = units[k] & ~take
        moved[k] = int((units[k] & take).sum())
    if stub_name in out:
        raise ValueError(f"unit name {stub_name} exists")
    out[stub_name] = stub.copy()
    return out, moved


def check_split(before: dict, after: dict, shared: np.ndarray | None = None):
    """Invariants: coverage identical; units disjoint except on shared neurons (in exactly two: their
    unit and the stub)."""
    cov_b = np.zeros_like(next(iter(before.values())), dtype=bool)
    for m in before.values():
        cov_b |= m
    acc = np.zeros(cov_b.shape, np.int32)
    for m in after.values():
        acc += m.astype(np.int32)
    ok_overlap = (acc <= 1) if shared is None else ((acc <= 1) | (shared & (acc == 2)))
    return bool(ok_overlap.all() and ((acc > 0) == cov_b).all())


def compartment_twin(pairs: np.ndarray, core: np.ndarray, exclude: np.ndarray, rng):
    """A random set with the same per-layer counts AND the same core/pool split as `pairs` [n,2],
    drawn outside `exclude` (bool [L,N], normally the stub itself)."""
    L, N = core.shape
    tw = []
    for l in np.unique(pairs[:, 0]):
        cols = pairs[pairs[:, 0] == l, 1]
        for in_core in (True, False):
            k = int((core[l, cols] == in_core).sum())
            if not k:
                continue
            avail = np.where((core[l] == in_core) & ~exclude[l])[0]
            pick = rng.choice(avail, size=min(k, len(avail)), replace=False)
            tw += [(int(l), int(c)) for c in pick]
    return np.array(sorted(tw), dtype=np.int32).reshape(-1, 2)


def layer_profile(band_costs, bands, L):
    """Per-layer fraction of the stub's removal strength (each band's positive cost spread evenly over
    its layers, normalised to sum 1; all zeros if no band costs anything)."""
    prof = np.zeros(L)
    for c, b in zip(band_costs, bands):
        if c > 0:
            prof[np.asarray(b)] += c / len(b)
    s = prof.sum()
    return (prof / s).tolist() if s > 0 else prof.tolist()


def stub_record(topic, host_ckpt, pairs, d_fit, d_held, source_units, measured, bands, L, extra=None):
    by_l = T.by_layer(pairs) if len(pairs) else {}
    rec = {"schema": SCHEMA, "topic": topic, "host": host_ckpt, "neurons": int(len(pairs)),
           "per_layer_counts": {str(l): int(len(c)) for l, c in sorted(by_l.items())},
           "selectivity": {"d_fit_mean": round(float(np.mean(d_fit)), 3) if len(d_fit) else None,
                           "d_held_mean": round(float(np.mean(d_held)), 3) if len(d_held) else None},
           "source_units": source_units, "functional": measured,
           "layer_profile": layer_profile([b["remove_over_twin"] for b in measured.get("bands", [])],
                                          [list(range(b["layers"][0], b["layers"][1] + 1)) for b in measured.get("bands", [])],
                                          L) if measured.get("bands") else None}
    if extra:
        rec.update(extra)
    return rec


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx27b"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--registry", required=True, help="<model>.topics.json (its .npz beside it)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--texts", type=int, default=24)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mem-policy", default="")
    ap.add_argument("--need-gb", type=float, default=6)
    ap.add_argument("--watchdog-gb", type=float, default=10)
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    log = T.make_log(out)
    if args.backend == "mlx27b":
        import amoeba27b_lib as A
        A.admit(args.need_gb, args.mem_policy or "strict", log)
        A.footprint_watchdog(args.watchdog_gb, log)
        A.setup_mlx(56, 4)
    else:
        import amoeba_memcheck
        amoeba_memcheck.require(args.need_gb, args.mem_policy or "small", log=log)
        amoeba_memcheck.rss_watchdog(args.watchdog_gb, log=log)
    tax = T.Taxonomy()
    X = args.topic
    plan = json.loads((Path(args.plan) / "plan.json").read_text())
    U = np.load(Path(args.plan) / "units.npz")
    core = np.load(plan["core"])["core"] > 0
    L, N = core.shape
    node_ids = json.loads((Path(args.arrays) / "neuron_d_nodes.json").read_text())
    if X not in node_ids:
        raise SystemExit(f"{X} is not a tested node of this labelling run")
    k = node_ids.index(X)
    dfit = np.asarray(np.load(Path(args.arrays) / "neuron_d_fit.npy", mmap_mode="r")[k], np.float32)
    dheld = np.asarray(np.load(Path(args.arrays) / "neuron_d_held.npy", mmap_mode="r")[k], np.float32)
    # 1. candidates: units whose functional gate passed for X or a descendant (registry rows), else the
    #    registry's core-resident map for X
    reg = json.loads(Path(args.registry).read_text())
    if reg["model"]["checkpoint"] != plan["checkpoint"]:
        raise SystemExit("registry and plan are for different checkpoints")
    maps = np.load(Path(args.registry).with_suffix("").with_suffix(".topics.npz"))
    src, cand = [], []
    for t, row in reg["topics"].items():
        if X not in tax.ancestors(t):
            continue
        for u in row.get("units", []):
            if u["functional"]["pass"]:
                cand.append(maps[u["neuron_map"]["npz_key"]])
                src.append(f"{t}|{u['unit']}")
    if not cand:
        row = reg["topics"].get(X, {})
        ct = row.get("core_tag")
        if not ct:
            raise SystemExit(f"{X}: no functionally passing unit and no core tag in the registry (status {row.get('status')})")
        cand.append(maps[ct["neuron_map"]["npz_key"]])
        src.append(f"{X}|core")
    pairs = np.unique(np.concatenate(cand), axis=0)
    order = np.argsort(-dfit[pairs[:, 0], pairs[:, 1]], kind="stable")
    ranked = pairs[order]
    log(f"{X}: {len(ranked)} candidates from {src}")
    # data
    be = T.make_backend(args.backend, args.model, log=log, threads=args.threads or None)
    if be.checkpoint != plan["checkpoint"]:
        raise SystemExit(f"plan is for {plan['checkpoint']}, model is {be.checkpoint}")
    items = T.load_items(args.corpus)
    toks = T.tokenize_items(be.encode, items, 64, 48)
    mem = T.topic_membership(tax, [it for it, _ in toks])

    def texts(split, pred, key):
        idx = [i for i, (it, _) in enumerate(toks) if it["split"] == split and pred(mem[i])]
        r = np.random.default_rng(T.stable_seed(key))
        return [toks[i][1] for i in sorted(r.permutation(idx)[:args.texts])]
    fit_x = texts("fit", lambda m: X in m, X + "/fit")
    held_x = texts("held", lambda m: X in m, X + "/held")
    root = X.split(".")[0]
    held_ctl = texts("held", lambda m: root not in m, "!" + root)

    def nll(seqs, mask):
        be.set_mask(mask)
        v = []
        for g in T.batches_by_len(seqs, args.bs):
            v += list(be.nll([seqs[j] for j in g]))
        be.set_mask(None)
        return float(np.mean(v))
    full = np.ones((L, N), np.float32)
    base_fit = nll(fit_x, None)

    def cost(kk):
        m = full.copy()
        m[ranked[:kk, 0], ranked[:kk, 1]] = 0
        return nll(fit_x, m) - base_fit
    t0 = time.time()
    kk, total, evals = select_prefix(len(ranked), cost, args.frac)
    stub = ranked[:kk]
    log(f"stub: {kk}/{len(ranked)} neurons carry >= {args.frac} of the fit removal cost {total:.4f} ({time.time() - t0:.0f}s)")
    smask = np.zeros((L, N), bool)
    smask[stub[:, 0], stub[:, 1]] = True
    assert masks_match_lists(smask, stub)
    rng = np.random.default_rng(args.seed)
    twin = compartment_twin(stub, core, smask, rng)
    tmask = np.zeros((L, N), bool)
    tmask[twin[:, 0], twin[:, 1]] = True
    meas = {}
    for name, seqs in (("topic_held", held_x), ("control_held", held_ctl)):
        b = nll(seqs, None)
        a = nll(seqs, full * ~smask)
        t = nll(seqs, full * ~tmask)
        meas[name] = {"n_texts": len(seqs), "nll_full": round(b, 4), "nll_minus_stub": round(a, 4),
                      "nll_minus_twin": round(t, 4), "remove_over_twin": round(a - t, 4)}
    bands = np.array_split(np.arange(L), args.bands)
    bb = []
    for bl in bands:
        sm = smask.copy()
        sm[[l for l in range(L) if l not in set(bl.tolist())]] = False
        tm = tmask.copy()
        tm[[l for l in range(L) if l not in set(bl.tolist())]] = False
        if not sm.any():
            bb.append({"layers": [int(bl[0]), int(bl[-1])], "neurons": 0, "remove_over_twin": 0.0})
            continue
        bb.append({"layers": [int(bl[0]), int(bl[-1])], "neurons": int(sm.sum()),
                   "remove_over_twin": round(nll(held_x, full * ~sm) - nll(held_x, full * ~tm), 4)})
    meas["bands"] = bb
    # 4. split against the host's current units (core + labelled units), and check invariants
    units = {"core": core.copy()}
    for u in plan["units"]:
        if u["kind"] == "cluster" and u.get("key"):
            m = np.zeros((L, N), bool)
            p = U[u["key"]]
            m[p[:, 0], p[:, 1]] = True
            units[u["unit"]] = m & ~core
    pool_rest = ~core
    for k2, m in units.items():
        if k2 != "core":
            pool_rest &= ~m
    units["pool_unassigned"] = pool_rest
    after, moved = split_units(units, smask, f"stub:{X}")
    ok = check_split(units, after)
    rec = stub_record(X, be.checkpoint, stub, dfit[stub[:, 0], stub[:, 1]], dheld[stub[:, 0], stub[:, 1]], src, meas,
                      bands, L, extra={"candidates": int(len(ranked)), "frac": args.frac, "fit_removal_cost_all": round(total, 4),
                                       "bisection": [[int(a), round(b, 4)] for a, b in evals], "split_moved": moved,
                                       "split_invariants_ok": ok, "bisection_monotone": is_monotone(evals),
                                       "masks_bit_exact": all(masks_match_lists(m, np.argwhere(m)) for m in after.values()),
                                       "registry": str(args.registry), "mlx2_source_revision": T.git_rev(),
                                       "corpus_sha256": T.sha_file(args.corpus), "twin": "compartment-matched, same per-layer counts"})
    np.savez_compressed(out / "stub.npz", stub=stub.astype(np.int32), twin=twin.astype(np.int32))
    rec["stub_npz_sha256"] = T.sha_file(out / "stub.npz")
    T.write_json(out / "stub.json", rec)
    log(f"done: topic rm/twin {meas['topic_held']['remove_over_twin']}, control {meas['control_held']['remove_over_twin']}, "
        f"invariants {ok}")


if __name__ == "__main__":
    main()
