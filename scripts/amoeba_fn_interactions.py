#!/usr/bin/env python3
"""Tile interaction arm for an FN neuron partition (opt-in; Pierre 2026-11-19 "do it", after the "neural ecosystem
atlas" reading: docs/research/2026-11-19-ecosystem-interactions.md). Standalone: the box battery
(scripts/amoeba_box_battery.py) is imported read-only for its served composition and held-set reader, and nothing
here changes what the battery does. One model load, forward only, MoE neuron units, inside a GPU window.

Closing a tile = its units contribute nothing (the battery's drop rule). Every quantity is a held-text CE change in
the FULL model unless a context says otherwise, measured per document (so each value carries a paired standard
error), always against the all-open masked path (the battery's path offset cancels).

Arms (--arms, default pairs):
  pairs   pairwise closure non-additivity on a chosen tile set: the --boxes-top largest boxes, the --joint-top largest
          joint tiles of --joint-partition, and member-sharing partners (each chosen box's largest pair tile, each
          chosen joint tile's largest member box). For a pair (A, B) and each eval set s (A's topics, B's topics,
          English): I = dL(A u B) - dL(A) - dL(B). Noise floor: --random-pairs random-unit pairs, each matched to a
          real pair's per-layer counts (disjoint, outside the real pair), evaluated on that pair's sets; floor = the
          --floor-q quantile of |I_random| pooled. A (pair, set) is significant when |I| > max(floor, z x SE).
          Labels per direction (thresholds = max(floor, z x SE) of each term): redundancy (I > 0: each covers the
          other), overlap (I < 0: sub-additive), hidden_backup (B alone ~ 0 but dL(B | A closed) > 0), rival (dL(B | A
          closed) < 0: closing B helps once A is gone), dependent (B mattered alone, dL(B | A closed) < dL(B) / 2:
          B loses its job when A goes; the atlas "unemployed" count per keystone). Role reversal: a pair whose
          significant I has opposite signs on two of its sets.
  backup  damage-then-close: X closed with a neighbour Y already closed vs X alone, excess = dL(X | Y) - dL(X)
          (= I(X, Y)). Sources: --moves CORE_ASSIGN_OUT (scripts/amoeba_fn_core_assign.py: X = a target tile's moved
          core units; Y = the target tile's original units, on its members' sets + English, and, for up to
          --backup-controls non-member boxes Z, Y = everything outside Z's served composition in --partition, on Z's
          set: the new served composition closes X for Z) and --detached FILE (amoeba_fn_partition.py
          --emit-detached: X = the units a box's weak claims were detached from and that its served composition now
          closes; Y = everything else outside that composition, on the box's set). Floor: --backup-random random X'
          per spec (X's per-layer counts, outside X and Y), floor = their largest |excess|. For a served context the
          random X' can only come from the open composition (core, box, its tiles: the units the box uses most), so
          that floor leans high (misses rather than invents backups). hidden_backup = excess significant and X
          alone not.
  repair  for up to --repair-boxes boxes that fail the battery's served +0.05 (read from --battery at run time;
          missing -> refused), in the served composition S (the damaged context): give back one closed tile
          (candidates: the --repair-cands closed tiles with the most map drive on the box, plus --repair-random random
          closed tiles of similar size), or close one open non-core tile other than the box (--repair-competitors,
          the atlas "remove a competitor" move), each measured on the box's held set; recovery = gain / (L(S) -
          L(full)); the best two give-backs together; the best give-back's English cost. Spearman(drive, gain) says
          whether the map screen ranks what the forward pass measures.
Keystone screen (CPU, no forward pass): a tile's map drive share on a topic, D_T(s) = sum of the topic map's
mean_target over the tile's units / over all units (English: the loaded maps' mean_baseline); a pair's screen score =
sum over its sets of min(D_A(s), D_B(s)) (co-drive). The --max-pairs best-screened pairs are measured, plus
--screen-controls of the lowest-screened, so Spearman(screen, |I|) checks the screen.
Budget: every (closure, set) is one forward pass over --docs held documents. The plan (OUT/plan.json) counts them
before the model loads and drops the lowest-screened pairs until the count x --sec-per-eval fits --budget-min; at run
time no new work starts after --budget-min of evaluation (rows written so far stay; truncation recorded).
Memory (coordinator 2026-11-19, after a kernel panic): tiles are kept as packed bit masks (2 MB each), a dense mask is
built per closure and freed, the joint partition's 13,557 tiles are never all decompressed (only the chosen ones),
and the repo's footprint watchdog runs (--watchdog-gb 80, MLX limit --memory-gb 76).

Held sets, notes and skips come from --battery DIR/battery.meta.json (the battery run this arm sits beside: same
specs, same seeded document order); refused when it is missing. OUT must be new or empty.

Usage: amoeba_fn_interactions.py --model M --partition DIR [--joint-partition DIR] --battery DIR --out DIR
       [--arms pairs,backup,repair] [--moves DIR] [--detached FILE] [--dry-run] [--assert-ple]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_box_battery as BAT

ARMS = ("pairs", "backup", "repair")
# fn-boxes2-dd1ef1d7d battery: ~140 s per box for 8 held-set evaluations at 80 docs x 256 tokens
SEC_PER_EVAL = 17.5
EPS = 1e-6          # nats: no threshold is below this (exact float cancellation is not an interaction)


class Deadline(Exception):
    pass


# ------------------------------------------------------------------ packed masks

def pack(mask):
    return np.packbits(np.ascontiguousarray(mask, dtype=bool).ravel())


def unpack(bits, shape):
    return np.unpackbits(bits, count=int(shape[0]) * int(shape[1])).view(bool).reshape(shape)


def bor(*bs):
    r = bs[0].copy()
    for b in bs[1:]:
        np.bitwise_or(r, b, out=r)
    return r


def bandnot(a, b):
    return np.bitwise_and(a, np.bitwise_not(b))


def nbits(bits):
    return int(np.unpackbits(bits).sum())


SHORTFALL = [0]      # units a random draw could not match (a layer with too few candidates); reported in meta


def random_disjoint(mask, exclude, g):
    """The per-layer counts of `mask` drawn at random among the units outside `exclude` (rng g). A layer with fewer
    candidates than needed takes them all; the shortfall is counted in SHORTFALL."""
    out = np.zeros_like(mask)
    for l in range(mask.shape[0]):
        k = int(mask[l].sum())
        if k:
            cand = np.flatnonzero(~exclude[l])
            SHORTFALL[0] += max(0, k - cand.size)
            out[l, g.choice(cand, min(k, cand.size), replace=False)] = True
    return out


# ------------------------------------------------------------------ statistics

def mean_se(v):
    v = np.asarray(v, np.float64)
    if v.size < 2:
        return float(v.mean()) if v.size else 0.0, float("inf")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return None

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(a.size)
        r[o] = np.arange(a.size)
        for v in np.unique(a):                       # ties: mean rank
            w = a == v
            r[w] = r[w].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def floor_of(values, q):
    v = np.abs(np.asarray(values, float))
    return float(np.quantile(v, q)) if v.size else 0.0


# ------------------------------------------------------------------ evaluator

class Evaluator:
    """Per-document CE of a closure on a held set, cached by (closure key, set). score_fn(closed [L, W] bool, set) ->
    per-document mean CE. Counts evaluations and refuses new ones past the deadline."""

    def __init__(self, score_fn, shape, budget_s=None, log=print):
        self.score_fn, self.shape, self.log = score_fn, tuple(shape), log
        self.cache, self.n, self.t_eval = {}, 0, 0.0
        self.budget_s, self.t0 = budget_s, None
        self.end = None if budget_s is None else time.time() + budget_s    # the whole run
        self.stop_at = self.end                                              # the current arm's share
        self.empty = pack(np.zeros(self.shape, bool))

    def window(self, frac_of_rest):
        """Give the next arm this fraction of the time left (an arm cannot eat the later arms' share)."""
        if self.end is not None:
            now = time.time()
            self.stop_at = now + max(0.0, self.end - now) * frac_of_rest

    def ce(self, key, bits, s):
        k = (key, s)
        if k not in self.cache:
            if self.t0 is None:
                self.t0 = time.time()
            if self.stop_at is not None and self.n and time.time() > self.stop_at:
                raise Deadline(f"time share reached (whole budget {self.budget_s:.0f} s)")
            t = time.time()
            closed = unpack(bits, self.shape)
            self.cache[k] = np.asarray(self.score_fn(closed, s), np.float64)
            del closed
            self.n += 1
            self.t_eval += time.time() - t
        return self.cache[k]

    def base(self, s):
        return self.ce("", self.empty, s)

    def d(self, key, bits, s):
        """dL per document of closing `bits` (vs the all-open path)."""
        return self.ce(key, bits, s) - self.base(s)


def ukey(*keys):
    return "|".join(sorted(set(x for k in keys for x in k.split("|") if x)))


def pair_terms(ev, ka, a, kb, b, s):
    """(dA, dB, I) per-document vectors on set s."""
    dA, dB = ev.d(ka, a, s), ev.d(kb, b, s)
    dAB = ev.d(ukey(ka, kb), bor(a, b), s)
    return dA, dB, dAB - dA - dB


def labels(dA, seA, dB, seB, I, seI, floor, z):
    """Atlas-style roles of one (pair, set). Each term's threshold = max(floor, z x its SE)."""
    tA, tB, tI = max(floor, z * seA, EPS), max(floor, z * seB, EPS), max(floor, z * seI, EPS)
    out = []
    if abs(I) <= tI:
        return out
    out.append("redundancy" if I > 0 else "overlap")
    for (x, tx, nx), (y, ty, ny) in (((dA, tA, "A"), (dB, tB, "B")), ((dB, tB, "B"), (dA, tA, "A"))):
        cond = x + I                                   # dL(x | y closed)
        if abs(x) <= tx and cond > tI:
            out.append(f"hidden_backup:{nx}_for_{ny}")
        if x >= -tx and cond < -tI:
            out.append(f"rival:{nx}_after_{ny}")
        if x > tx and cond < 0.5 * x:
            out.append(f"dependent:{nx}_on_{ny}")
    return out


# ------------------------------------------------------------------ arms (pure: no model)

def run_random_floor(ev, specs, shape, seed, q, log=print):
    """specs: [(name, a_bits, b_bits, sets)]; each gets one random disjoint pair of matched per-layer counts."""
    vals, rows = [], []
    g = np.random.default_rng(seed)
    for i, (name, a, b, sets) in enumerate(specs):
        A, B = unpack(a, shape), unpack(b, shape)
        ex = A | B
        rA = random_disjoint(A, ex, g)
        rB = random_disjoint(B, ex | rA, g)
        del A, B, ex
        ra, rb = pack(rA), pack(rB)
        del rA, rB
        row = {"matched_to": name, "units": [nbits(ra), nbits(rb)], "I": {}}
        try:
            for s in sets:
                _, _, I = pair_terms(ev, f"rand{i}a", ra, f"rand{i}b", rb, s)
                m, se = mean_se(I)
                row["I"][s] = {"mean": m, "se": se}
                vals.append(m)
        except Deadline as e:
            row["truncated"] = str(e)
            rows.append(row)
            break
        rows.append(row)
        log(f"random pair {i} (matched to {name}): " + " ".join(f"{s} {v['mean']:+.4f}" for s, v in row["I"].items()))
    return {"q": q, "floor": floor_of(vals, q), "values": len(vals), "rows": rows}


def run_pairs(ev, tiles, pairs, floor, z, put=None, log=print):
    """tiles: {name: bits}; pairs: [(a, b, sets, screen)]. Rows per pair."""
    rows = []
    for a, b, sets, scr in pairs:
        row = {"a": a, "b": b, "screen": scr, "sets": {}, "flags": []}
        try:
            for s in sets:
                dA, dB, I = pair_terms(ev, a, tiles[a], b, tiles[b], s)
                (mA, sA), (mB, sB), (mI, sI) = mean_se(dA), mean_se(dB), mean_se(I)
                lab = labels(mA, sA, mB, sB, mI, sI, floor, z)
                row["sets"][s] = {"dA": mA, "dA_se": sA, "dB": mB, "dB_se": sB, "I": mI, "I_se": sI,
                                  "I_norm": mI / max(abs(mA) + abs(mB), 1e-9), "labels": lab}
                row["flags"] += [f"{x}@{s}" for x in lab]
        except Deadline as e:
            row["truncated"] = str(e)
            rows.append(row)
            if put:
                put(row)
            break
        sig = {s: v["I"] for s, v in row["sets"].items() if v["labels"]}
        row["role_reversal"] = bool(any(v > 0 for v in sig.values()) and any(v < 0 for v in sig.values()))
        rows.append(row)
        if put:
            put(row)
        log(f"pair {a} x {b}: " + " ".join(f"{s} I {v['I']:+.4f}+-{v['I_se']:.4f} {','.join(v['labels'])}"
                                            for s, v in row["sets"].items()))
    return rows


def keystones(rows):
    """Per tile A: the partners that depend on it (dependent:<partner>_on_<A> on any set)."""
    out = {}
    for r in rows:
        for s, v in r["sets"].items():
            for lab in v["labels"]:
                if lab.startswith("dependent:"):
                    x, _, y = lab[10:].partition("_on_")
                    ks, dep = (r["a"], r["b"]) if y == "A" else (r["b"], r["a"])
                    out.setdefault(ks, set()).add(dep)
    return {k: sorted(v) for k, v in sorted(out.items(), key=lambda x: -len(x[1]))}


def run_backup(ev, specs, shape, z, n_random, seed, put=None, log=print):
    """specs: [{"name", "x": bits, "contexts": [(ctx_name, y_bits, sets)]}]. Per context and set: alone = dL(X),
    after = dL(X | Y closed), excess = after - alone; floor from random X' (X's per-layer counts, outside X u Y)."""
    g = np.random.default_rng(seed)
    rows = []
    for si, sp in enumerate(specs):
        x = sp["x"]
        row = {"name": sp["name"], "units": nbits(x), "contexts": {}, "flags": []}
        try:
            for ci, (cn, y, sets) in enumerate(sp["contexts"]):
                X = unpack(x, shape)
                ex = X | unpack(y, shape)
                rnd = [pack(random_disjoint(X, ex, g)) for _ in range(n_random)]
                del X, ex
                ky = f"ctx{si}.{ci}"
                ctx = {"units_closed": nbits(y), "sets": {}}
                for s in sets:
                    alone = ev.d(f"bx{si}", x, s)
                    after = ev.d(ukey(ky, f"bx{si}"), bor(x, y), s) - ev.d(ky, y, s)
                    exc = after - alone
                    rv = []
                    for ri, r in enumerate(rnd):
                        ra = ev.d(f"bx{si}r{ri}.{ci}", r, s)
                        rf = ev.d(ukey(ky, f"bx{si}r{ri}.{ci}"), bor(r, y), s) - ev.d(ky, y, s)
                        rv.append(float((rf - ra).mean()))
                    (ma, sa), (mf, sf), (me, se_) = mean_se(alone), mean_se(after), mean_se(exc)
                    fl = floor_of(rv, 1.0)
                    sig = me > max(fl, z * se_, EPS)
                    neg = abs(ma) <= max(fl, z * sa, EPS)
                    v = {"alone": ma, "alone_se": sa, "after": mf, "after_se": sf, "excess": me, "excess_se": se_,
                         "random_excess": rv, "floor": fl, "excess_significant": bool(sig),
                         "alone_negligible": bool(neg), "hidden_backup": bool(sig and neg)}
                    ctx["sets"][s] = v
                    if sig:
                        row["flags"].append(f"{'hidden_backup' if neg else 'excess'}:{cn}@{s}")
                row["contexts"][cn] = ctx
        except Deadline as e:
            row["truncated"] = str(e)
            rows.append(row)
            if put:
                put(row)
            break
        rows.append(row)
        if put:
            put(row)
        log(f"backup {sp['name']}: flags {row['flags']}")
    return rows


def run_repair(ev, specs, en, put=None, log=print):
    """specs: [{"box", "closed": bits (outside the served composition), "give": [(name, bits, drive)],
    "random": [(name, bits, drive)], "compete": [(name, bits)]}]."""
    rows = []
    for sp in specs:
        b, cl = sp["box"], sp["closed"]
        kc = f"served:{b}"
        row = {"box": b, "give_back": {}, "random": {}, "compete": {}}
        try:
            served = ev.ce(kc, cl, b)
            gap = mean_se(served - ev.base(b))
            row["served_gap"] = {"mean": gap[0], "se": gap[1]}

            def gain(k, bits):
                return mean_se(served - ev.ce(k, bits, b))
            for grp, lst in (("give_back", sp["give"]), ("random", sp["random"])):
                for name, bits, drive in lst:
                    gm, gs = gain(f"{kc}-{name}", bandnot(cl, bits))
                    row[grp][name] = {"gain": gm, "gain_se": gs, "recovery": gm / gap[0] if gap[0] > 0 else None,
                                      "drive": drive, "units": nbits(bits)}
            for name, bits in sp["compete"]:
                gm, gs = gain(f"{kc}+{name}", bor(cl, bits))
                row["compete"][name] = {"gain": gm, "gain_se": gs, "recovery": gm / gap[0] if gap[0] > 0 else None,
                                        "units": nbits(bits)}
            gb = sorted(row["give_back"].items(), key=lambda kv: -kv[1]["gain"])
            if gb:
                best_n = gb[0][0]
                best_bits = dict((n, bb) for n, bb, _ in sp["give"])[best_n]
                row["best_give_back"] = best_n
                e0 = ev.ce(kc, cl, en)
                e1 = ev.ce(f"{kc}-{best_n}", bandnot(cl, best_bits), en)
                row["best_english_change"] = mean_se(e1 - e0)[0]
                if len(gb) >= 2:
                    n2 = gb[1][0]
                    b2 = dict((n, bb) for n, bb, _ in sp["give"])[n2]
                    gm, gs = gain(f"{kc}-{best_n}-{n2}", bandnot(bandnot(cl, best_bits), b2))
                    row["best_two"] = {"tiles": [best_n, n2], "gain": gm, "gain_se": gs,
                                       "recovery": gm / gap[0] if gap[0] > 0 else None}
                row["spearman_drive_gain"] = spearman([v["drive"] for _, v in gb], [v["gain"] for _, v in gb])
            comp = [(n, v) for n, v in row["compete"].items() if v["gain"] > 0]
            if comp:
                row["best_competitor"] = max(comp, key=lambda kv: kv[1]["gain"])[0]
        except Deadline as e:
            row["truncated"] = str(e)
            rows.append(row)
            if put:
                put(row)
            break
        rows.append(row)
        if put:
            put(row)
        log(f"repair {b}: gap {row['served_gap']['mean']:+.4f}; best give-back {row.get('best_give_back')} "
            f"{(row['give_back'].get(row.get('best_give_back')) or {}).get('recovery')}; competitor "
            f"{row.get('best_competitor')}")
    return rows


# ------------------------------------------------------------------ selection and screen (CPU)

def tile_members(key, prep):
    if key.startswith("box:"):
        return [key[4:]]
    if key.startswith("pair:"):
        return key[5:].split("__")
    if key.startswith("joint:"):
        return key[6:].split("+")
    if key.startswith("group:"):
        return list((((prep.get("groups") or {}).get("kept") or {}).get(key[6:]) or {}).get("members") or [])
    return []


def topics_of(key, prep, have, en_box, en, cap):
    """The held sets that stand for a tile: its members with held sets (largest boxes first, at most cap); the English
    box stands for English."""
    mem = tile_members(key, prep)
    if en_box and mem == [en_box]:
        return [en]
    units = lambda m: (prep["boxes"].get(m) or {}).get("units", 0)
    return sorted([m for m in mem if m in have], key=lambda m: (-units(m), m))[:cap]


def choose_tiles(prep, jprep, have, en_box, boxes_top, joint_top, share, min_units=1000):
    """-> [(key, source "main" | "joint")] in a fixed order. Partner pair tiles below min_units are skipped (with
    nesting a box's pairs can be a few units)."""
    bx = sorted((b for b, v in prep["boxes"].items() if b in have or b == en_box),
                key=lambda b: (-prep["boxes"][b].get("units", 0), b))[:boxes_top]
    out = [(f"box:{b}", "main") for b in bx]
    jt = (jprep or {}).get("joint", {}).get("tiles") or {}
    js = sorted((k for k in jt if any(m in have for m in jt[k]["members"])),
                key=lambda k: (-jt[k]["units"], k))[:joint_top]
    out += [(f"joint:{k}", "joint") for k in js]
    if share:
        seen = {k for k, _ in out}
        for b in bx:                                    # the box's largest pair tile (shares member b)
            ps = sorted(((k, n) for k, n in prep["pairs"].items() if b in k.split("__") and n >= min_units
                         and any(m in have for m in k.split("__"))), key=lambda x: (-x[1], x[0]))
            if ps and f"pair:{ps[0][0]}" not in seen:
                out.append((f"pair:{ps[0][0]}", "main"))
                seen.add(f"pair:{ps[0][0]}")
        for k in js:                                    # the joint tile's largest member box (shares that member)
            mem = sorted((m for m in jt[k]["members"] if m in have and prep["boxes"][m]["units"] >= min_units),
                         key=lambda m: (-prep["boxes"][m]["units"], m))
            if mem and f"box:{mem[0]}" not in seen:
                out.append((f"box:{mem[0]}", "main"))
                seen.add(f"box:{mem[0]}")
    return out


def related(a, b, prep):
    ma, mb = tile_members(a, prep), tile_members(b, prep)
    return any(BAT.lineage(x, y) for x in ma for y in mb)


def plan_pairs(tiles, topics, drive, en, max_pairs, controls):
    """All pairs of the chosen tiles, scored by co-drive; the best max_pairs + `controls` of the lowest."""
    names = [k for k, _ in tiles]
    allp = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sets = list(dict.fromkeys(topics[a] + topics[b] + [en]))
            scr = float(sum(min(drive[a].get(s, 0.0), drive[b].get(s, 0.0)) for s in sets))
            allp.append((a, b, sets, scr))
    allp.sort(key=lambda p: (-p[3], p[0], p[1]))
    top = allp[:max_pairs]
    low = [p for p in allp[max_pairs:]][-controls:] if controls else []
    return top, low


def count_pair_evals(pairs, floor_specs, base_sets):
    """Forward passes the pairs arm needs: base per set, each tile once per set, each union per set, the random
    floor's three masks per set."""
    singles, unions = set(), set()
    for a, b, sets, _ in pairs:
        for s in sets:
            singles |= {(a, s), (b, s)}
            unions.add((a, b, s))
    fl = sum(3 * len(sets) for _, _, _, sets in floor_specs)
    return len(set(base_sets)) + len(singles) + len(unions) + fl


# ------------------------------------------------------------------ model side

def doc_ce(host, seqs, cfg_fn, bs=8):
    """Per-document mean CE: research/amoeba/carve/evals.text_ce's positions, batching and loss mask, kept per row."""
    import mlx.core as mx

    from research.amoeba.carve.carving import ce_rows
    from research.amoeba.carve.corpus import loss_from
    tot, cnt = np.zeros(len(seqs)), np.zeros(len(seqs))
    by_len = {}
    for i, q in enumerate(seqs):
        by_len.setdefault(len(q), []).append(i)
    for _, idx in sorted(by_len.items()):
        for j in range(0, len(idx), bs):
            ii = idx[j:j + bs]
            rows = [seqs[k] for k in ii]
            ids = mx.array(rows)
            T = ids.shape[1]
            w = np.array([[1.0 if t >= max(2, loss_from(q)) else 0.0 for t in range(2, T)] for q in rows], np.float32)
            if cfg_fn:
                cfg_fn()
            lg = host.logits_at(ids, list(range(1, T - 1)))
            host.ctl.reset() if cfg_fn else None
            ce = (ce_rows(lg, ids[:, 2:]) * mx.array(w)).sum(axis=1)
            mx.eval(ce)
            tot[ii] = np.array(ce, dtype=np.float64)
            cnt[ii] = w.sum(1)
            del lg, ce
            mx.clear_cache()
    return tot / np.maximum(cnt, 1)


def load_held(host, specs, docs, seed, log):
    """The battery's held loading (amoeba_box_battery.py main): same reader, seeded order and length cut."""
    held = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        rest, _, tok = rest.partition("@")
        T = int(tok or 256)
        kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
        path, _, filt = rest.removeprefix("gsm8k:").partition("?")
        rows = BAT.read_held_rows(path, kind, filt)
        np.random.default_rng(seed).shuffle(rows)
        seqs = []
        for r in rows:
            t = host.encode(r["text"])
            if len(t) >= T:
                seqs.append(t[:T])
            if len(seqs) >= docs:
                break
        if not seqs:
            raise SystemExit(f"held {name}: no document of {T} tokens")
        held[name] = seqs
        log(f"held {name}: {len(seqs)} x {T} tokens")
    return held


def map_vec(path, L):
    z = np.load(Path(path) / "map.npz")
    return (np.concatenate([z[f"mean_target_{s}"] for s in range(L)]).astype(np.float32),
            np.concatenate([z[f"mean_baseline_{s}"] for s in range(L)]).astype(np.float32))


def _wired_arg(spec):
    """--wired-gb: a bad value fails at parse time, before the model loads (ops.wired_arg, imported lazily)."""
    from research.amoeba.carve.ops import wired_arg
    return wired_arg(spec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--joint-partition", default="")
    ap.add_argument("--battery", required=True, help="the battery's out dir (battery.meta.json: held sets, notes, "
                    "skips; battery.jsonl: the repair arm's failing boxes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="pairs")
    ap.add_argument("--english", default="english")
    ap.add_argument("--docs", type=int, default=0, help="0: the battery's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boxes-top", type=int, default=6)
    ap.add_argument("--joint-top", type=int, default=4)
    ap.add_argument("--no-share", action="store_true", help="no member-sharing partner tiles")
    ap.add_argument("--max-topics", type=int, default=2)
    ap.add_argument("--min-tile-units", type=int, default=1000, help="smallest member-sharing partner tile")
    ap.add_argument("--max-pairs", type=int, default=24)
    ap.add_argument("--screen-controls", type=int, default=3)
    ap.add_argument("--random-pairs", type=int, default=8)
    ap.add_argument("--floor-q", type=float, default=0.95)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--moves", default="", help="core-assign out dir (backup arm)")
    ap.add_argument("--detached", default="", help="amoeba_fn_partition.py --emit-detached file (backup arm)")
    ap.add_argument("--backup-max-tiles", type=int, default=6)
    ap.add_argument("--backup-controls", type=int, default=3)
    ap.add_argument("--backup-random", type=int, default=2)
    ap.add_argument("--repair-boxes", type=int, default=3)
    ap.add_argument("--repair-cands", type=int, default=6)
    ap.add_argument("--repair-random", type=int, default=2)
    ap.add_argument("--repair-competitors", type=int, default=3)
    ap.add_argument("--budget-min", type=float, default=100.0, help="evaluation time cap (minutes)")
    ap.add_argument("--sec-per-eval", type=float, default=0.0, help="planning rate (0: 17.5 s x docs/80)")
    ap.add_argument("--dry-run", action="store_true", help="CPU: selection, screen and plan only")
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=76)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto", type=_wired_arg,
                    help="wire the model weights after load (as scripts/amoeba_box_battery.py --wired-gb): auto = MLX "
                         "active after load + 10 GiB, GiB, or 0 = off; capped at --memory-gb; env AMOEBA_WIRED_GB")
    a = ap.parse_args(argv)
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"--arms: one or more of {ARMS}")
    if "backup" in arms and not (a.moves or a.detached):
        raise SystemExit("--arms backup needs --moves and/or --detached")
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty (fresh evidence directory required)")
    bdir = Path(a.battery)
    if not (bdir / "battery.meta.json").exists():
        raise SystemExit(f"no {bdir}/battery.meta.json: the battery this arm reads its held sets from has not "
                         "finished (refusing)")
    bmeta = json.loads((bdir / "battery.meta.json").read_text())
    pd = Path(a.partition)
    prep = json.loads((pd / "partition.json").read_text())
    tz = np.load(pd / "tiles.npz")
    layers = [int(x) for x in tz["layers"]]
    keys = [k for k in tz.files if k != "layers"]
    core = tz["core"]
    shape = core.shape
    if bmeta.get("fingerprint") != prep.get("fingerprint"):
        raise SystemExit("--battery ran on another model than --partition")
    same_part = lambda x: bool(x) and Path(x).resolve() == pd.resolve()
    if not same_part(bmeta.get("partition")):
        log(f"note: the battery ran on {bmeta.get('partition')}, this arm on {pd}")
    jprep = jz = None
    if a.joint_partition:
        jd = Path(a.joint_partition)
        jprep = json.loads((jd / "partition.json").read_text())
        jz = np.load(jd / "tiles.npz")
        if jprep.get("fingerprint") != prep.get("fingerprint") or [int(x) for x in jz["layers"]] != layers:
            raise SystemExit("--joint-partition: another model or layer list")
    specs = {k: v["spec"] for k, v in bmeta["held"].items()}
    skipped = bmeta.get("skipped") or {}
    EN = a.english
    if EN not in specs:
        raise SystemExit(f"the battery has no held set {EN}")
    en_box, _ = BAT.english_box(prep)
    have = {k for k in specs if k not in skipped and k in prep["boxes"]}
    docs = a.docs or int(bmeta.get("docs") or 80)
    spe = a.sec_per_eval or SEC_PER_EVAL * docs / 80
    plan = {"docs": docs, "sec_per_eval_planned": spe, "budget_min": a.budget_min, "arms": arms}
    need_sets = {EN}

    def mask(key, src="main"):
        return (jz if src == "joint" else tz)[key]

    # ---- pairs: tiles, screen, plan
    tiles, pairs, controls, floor_specs = {}, [], [], []
    if "pairs" in arms:
        chosen = choose_tiles(prep, jprep, have, en_box, a.boxes_top, a.joint_top if jz is not None else 0,
                              not a.no_share, a.min_tile_units)
        topics = {k: topics_of(k, prep, have, en_box, EN, a.max_topics) for k, _ in chosen}
        chosen = [(k, s) for k, s in chosen if topics[k]]
        pos = {}
        for k, src in chosen:
            m = mask(k, src)
            tiles[k] = pack(m)
            pos[k] = np.flatnonzero(m).astype(np.int64)
            del m
        need_topics = sorted({s for k, _ in chosen for s in topics[k]} - {EN})
        drive = {k: {} for k, _ in chosen}
        en_acc, en_n = None, 0
        for s in need_topics:                          # one map resident at a time
            mt, mb = map_vec(prep["boxes"][s]["map"], len(layers))
            tot = float(mt.sum()) or 1.0
            for k in pos:
                drive[k][s] = float(mt[pos[k]].sum()) / tot
            en_acc = mb if en_acc is None else en_acc + mb
            en_n += 1
            del mt, mb
        if en_acc is not None:
            tot = float(en_acc.sum()) or 1.0
            for k in pos:
                drive[k][EN] = float(en_acc[pos[k]].sum()) / tot
        del en_acc, pos
        top, controls = plan_pairs(chosen, topics, drive, EN, a.max_pairs, a.screen_controls)

        def fspecs(ps):
            sp = ps[::max(1, len(ps) // max(1, a.random_pairs))][:a.random_pairs] if ps else []
            return [(f"{p[0]} x {p[1]}", tiles[p[0]], tiles[p[1]], p[2]) for p in sp]
        base_sets = {s for p in top + controls for s in p[2]}
        share = 0.7 if {"repair", "backup"} & set(arms) else 1.0       # the other arms get the rest
        while True:                                     # fit the budget: drop the lowest-screened measured pairs,
            floor_specs = fspecs(top)                   # then the screen controls
            n = count_pair_evals(top + controls, floor_specs, base_sets)
            if n * spe <= a.budget_min * 60 * share or not (top or controls):
                break
            if top:
                top = top[:-1]
            else:
                controls = controls[:-1]
        pairs = top + controls
        need_sets |= {s for p in pairs for s in p[2]}
        plan["pairs"] = {"tiles": {k: {"source": src, "units": int(np.unpackbits(tiles[k]).sum()),
                                       "topics": topics[k], "drive": drive[k]} for k, src in chosen},
                         "measured": [{"a": p[0], "b": p[1], "sets": p[2], "screen": p[3],
                                       "shares_members": related(p[0], p[1], prep)} for p in top],
                         "screen_controls": [{"a": p[0], "b": p[1], "sets": p[2], "screen": p[3]} for p in controls],
                         "random_floor_matched_to": [x[0] for x in floor_specs],
                         "evals": count_pair_evals(pairs, floor_specs, [])}
        log(f"pairs plan: {len(chosen)} tiles, {len(top)} measured + {len(controls)} controls, "
            f"{len(floor_specs)} random pairs, {plan['pairs']['evals']} evals")
    # ---- repair: failing boxes from the battery rows
    rep_specs = []
    if "repair" in arms:
        bj = bdir / "battery.jsonl"
        if not bj.exists():
            raise SystemExit(f"--arms repair: no {bj} (refusing)")
        last, matched = {}, 0
        for x in bj.read_text().split("\n"):
            try:
                r = json.loads(x)
            except ValueError:
                continue
            st = r.get("settings") or {}
            if not same_part(st.get("partition")) or st.get("docs") != bmeta.get("docs"):
                continue
            matched += 1
            if r["box"] in have:
                last[r["box"]] = r                      # the last row per box
        if not matched:
            raise SystemExit(f"--arms repair: no row of {bj} was made on {pd} (refusing)")
        fails = sorted(((r["core_plus"][b] - r["path_offset"].get(b, 0.0), b) for b, r in last.items()
                        if not r["verdict"]["served_+0.05"]), reverse=True)
        fails = [b for _, b in fails][:a.repair_boxes]
        plan["repair"] = {"failing_boxes_chosen": fails}
        if fails:
            lab = np.full(core.size, -1, np.int16)      # tile id per unit: one tile decompressed at a time
            tkeys = [k for k in keys if k != "core"]
            if len(tkeys) >= 32767:
                raise SystemExit("too many tiles for an int16 label array")
            for i, k in enumerate(tkeys):
                lab[tz[k].ravel()] = i
            sizes = np.bincount(lab[lab >= 0], minlength=len(tkeys))
            g = np.random.default_rng(a.seed + 77)
            for b in fails:
                keep = BAT.served_keep(lambda k: tz[k], keys, prep, b, core)
                opened = np.bincount(lab[keep.ravel() & (lab >= 0)], minlength=len(tkeys)) > 0
                closed_bits = pack(~keep)
                del keep
                mt, _ = map_vec(prep["boxes"][b]["map"], len(layers))
                dr = np.bincount(lab[lab >= 0], weights=mt[lab >= 0], minlength=len(tkeys)) / (float(mt.sum()) or 1.0)
                del mt
                cand = [i for i in np.argsort(-dr) if not opened[i] and sizes[i] > 0][:a.repair_cands]
                med = float(np.median(sizes[cand])) if cand else 0.0
                pool = [i for i in range(len(tkeys)) if not opened[i] and i not in cand and med / 2 <= sizes[i] <= med * 2]
                rnd = list(g.choice(pool, min(a.repair_random, len(pool)), replace=False)) if pool else []
                comp = [i for i in np.argsort(-sizes) if opened[i] and tkeys[i] != f"box:{b}"][:a.repair_competitors]
                tb = lambda i: pack(lab.reshape(shape) == i)
                rep_specs.append({"box": b, "closed": closed_bits,
                                  "give": [(tkeys[i], tb(i), float(dr[i])) for i in cand],
                                  "random": [(tkeys[i], tb(i), float(dr[i])) for i in rnd],
                                  "compete": [(tkeys[i], tb(i)) for i in comp]})
                need_sets |= {b}
            del lab
        plan["repair"]["evals"] = sum(2 + len(s["give"]) + len(s["random"]) + len(s["compete"]) + 3 for s in rep_specs)
    # ---- backup: core-assign moves and / or detached claims
    bk_specs = []
    if "backup" in arms:
        if a.moves:
            md = Path(a.moves)
            mz = np.load(md / "moves.npz")
            if [int(v) for v in mz["layers"]] != layers:
                raise SystemExit("--moves: another layer list than --partition")
            mrows = sorted(((k, mz[k]) for k in mz.files if k != "layers"), key=lambda kv: -kv[1].size)
            mrows = mrows[:a.backup_max_tiles]
            scored = set()
            mb = md / "battery" / "battery.jsonl"
            for x in (mb.read_text().split("\n") if mb.exists() else []):
                try:
                    scored.add(json.loads(x)["box"])
                except (ValueError, KeyError):
                    pass
            g = np.random.default_rng(a.seed + 5)
            for fk, posn in mrows:
                k = fk.replace("__", ":", 1)
                if posn.size and (posn.max() >= core.size or not core.ravel()[posn].all()):
                    raise SystemExit(f"--moves: {k}'s moved units are not in --partition's core (another partition?)")
                X = np.zeros(core.size, bool)
                X[posn] = True
                x = pack(X)
                del X
                mem = tile_members(k, prep)
                ctxs = []
                if k in keys:
                    sets = list(dict.fromkeys(topics_of(k, prep, have, en_box, EN, a.max_topics) + [EN]))
                    ctxs.append((f"tile:{k}", pack(tz[k]), sets))
                    need_sets |= set(sets)
                pool = sorted(b for b in (scored or have) if b in have
                              and not any(BAT.lineage(b, m) for m in mem))
                for z_ in (list(g.choice(pool, min(a.backup_controls, len(pool)), replace=False)) if pool else []):
                    keep = BAT.served_keep(lambda kk: tz[kk], keys, prep, z_, core)
                    y = pack(~keep)
                    del keep
                    ctxs.append((f"served:{z_}", bandnot(y, x), [z_]))
                    need_sets.add(z_)
                bk_specs.append({"name": f"moves:{k}", "x": x, "contexts": ctxs})
        if a.detached:
            dz = np.load(a.detached)
            if [int(v) for v in dz["layers"]] != layers:
                raise SystemExit("--detached: another layer list")
            if not (prep.get("detach") or {}).get("share"):
                raise SystemExit("--detached: --partition was not made with a detach share")
            if any(dz[k].size and dz[k].max() >= core.size for k in dz.files if k != "layers"):
                raise SystemExit("--detached: positions outside the partition's layout")
            drows = sorted(((k, dz[k]) for k in dz.files if k != "layers" and k in have),
                           key=lambda kv: -kv[1].size)[:a.backup_max_tiles]
            for b, posn in drows:
                keep = BAT.served_keep(lambda kk: tz[kk], keys, prep, b, core)
                X = np.zeros(core.size, bool)
                X[posn] = True
                X &= ~keep.ravel()                          # the detached units its served composition closes
                if not X.any():
                    continue
                x = pack(X)
                y = pack(~keep.ravel() & ~X)
                del keep, X
                bk_specs.append({"name": f"detached:{b}", "x": x, "contexts": [(f"served:{b}", y, [b])]})
                need_sets.add(b)
        cost = lambda sp: sum(len(c[2]) * (3 + 2 * a.backup_random) for c in sp["contexts"])
        cap = a.budget_min * 60 * (0.3 if "pairs" in arms else 1.0) * (0.6 if "repair" in arms else 1.0)
        dropped = []
        while bk_specs and sum(map(cost, bk_specs)) * spe > cap:     # drop the smallest moves / detaches first
            i = min(range(len(bk_specs)), key=lambda j: nbits(bk_specs[j]["x"]))
            dropped.append(bk_specs.pop(i)["name"])
        plan["backup"] = {"dropped_for_budget": dropped, "specs": [{"name": s["name"], "units": nbits(s["x"]),
                                     "contexts": [(c[0], c[2]) for c in s["contexts"]]} for s in bk_specs],
                          "evals": sum(len(c[2]) * (3 + 2 * a.backup_random) for s in bk_specs for c in s["contexts"])}
    plan["sets"] = sorted(need_sets)
    plan["evals_total"] = len(need_sets) + sum((plan.get(k) or {}).get("evals", 0) for k in ARMS)
    plan["expected_eval_min"] = round(plan["evals_total"] * spe / 60, 1)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    log(f"plan: {plan['evals_total']} forward passes over {docs} docs, ~{plan['expected_eval_min']} min at "
        f"{spe:.1f} s each (cap {a.budget_min} min)")
    if a.dry_run:
        return plan
    # ---- GPU
    from research.amoeba.carve import ops as O
    O.prepare_wiring(a.wired_gb)                       # before any Metal call (MLX reads the refresh env then)
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
    from research.amoeba.carve.host import Host
    O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb,
                         watcher=True)   # SIGKILL watcher: the same FN model as the battery (2026-11-19 panic)
    O.setup_mlx(a.memory_gb, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    if host.fingerprint != prep["fingerprint"]:
        raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    wired = O.wire_weights(a.wired_gb, log, memory_gb=a.memory_gb)       # after load: Host evaluated the weights
    W = max(host.widths)
    if shape != (len(layers), W):
        raise SystemExit(f"tile shape {shape} != [moe layers, {W}]")
    held = load_held(host, [specs[s] for s in sorted(need_sets)], docs, a.seed, log)

    def score(closed, s):
        m = np.ones((host.L, W), np.float32)
        m[layers] = (~closed).astype(np.float32)
        cfg = E.configure(host, D.from_np(m, host), None, host.L)
        del m
        return doc_ce(host, held[s], cfg)
    ev = Evaluator(score, shape, a.budget_min * 60, log)
    res = run_all(ev, a, arms, tiles, pairs, floor_specs, bk_specs, rep_specs, EN, out, shape, log)
    meta = {"checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "partition": str(pd), "joint_partition": a.joint_partition or None, "battery": str(bdir),
            "held": {s: specs[s] for s in sorted(need_sets)}, "held_notes": {k: v for k, v in
                                                                            (bmeta.get("held_notes") or {}).items()
                                                                            if k in need_sets},
            "docs": docs, "seed": a.seed, "z": a.z, "floor_q": a.floor_q, "arms": arms,
            "evals": ev.n, "eval_s": round(ev.t_eval, 1), "sec_per_eval_measured": round(ev.t_eval / max(ev.n, 1), 2),
            "watchdog": {"limit_gb": a.watchdog_gb, "memory_gb": a.memory_gb, "swap_growth_gb": a.swap_growth_gb,
                         "waiver_note": a.waiver_note or None, **dict(getattr(O, "WATCH", {}))},
            "wired": wired,
            "random_shortfall_units": SHORTFALL[0], "mlx_peak_gb": O.peak_gb(), **res}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_all(ev, a, arms, tiles, pairs, floor_specs, bk_specs, rep_specs, EN, out, shape, log):
    """The arms in order (floor, pairs, backup, repair), rows appended as they finish; OUT/interactions.md."""
    def putter(name):
        def put(row):
            with (out / name).open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        return put
    res, md = {}, ["# Tile interactions\n"]
    later = [x for x in ("backup", "repair") if x in arms]
    if "pairs" in arms and pairs:
        ev.window(0.7 if later else 1.0)
        fl = run_random_floor(ev, floor_specs, shape, a.seed + 11, a.floor_q, log)
        (out / "floor.json").write_text(json.dumps(fl, indent=1))
        rows = run_pairs(ev, tiles, pairs, fl["floor"], a.z, putter("pairs.jsonl"), log)
        ks = keystones(rows)
        meas = [r for r in rows if "truncated" not in r]
        scr = spearman([r["screen"] for r in meas], [max(abs(v["I"]) for v in r["sets"].values()) for r in meas])
        res.update(floor=fl["floor"], pairs_measured=len(meas), pairs_truncated=len(meas) < len(pairs), keystones=ks, spearman_screen_absI=scr,
                   pairs_flagged=sum(1 for r in meas if r["flags"]),
                   role_reversals=[f"{r['a']} x {r['b']}" for r in meas if r.get("role_reversal")])
        md += [f"Noise floor (q{a.floor_q} of |I| over {fl['values']} random-pair values): {fl['floor']:.4f}; "
               f"z = {a.z}. Spearman(screen, max |I|) = {scr}.\n",
               "| A | B | set | dA | dB | I +- SE | labels |", "|---|---|---|---|---|---|---|"]
        for r in meas:
            for s, v in r["sets"].items():
                md.append(f"| {r['a']} | {r['b']} | {s} | {v['dA']:+.4f} | {v['dB']:+.4f} | {v['I']:+.4f} +- "
                          f"{v['I_se']:.4f} | {', '.join(v['labels'])} |")
        md += ["\nKeystones (partners that lose their job when the tile closes): "
               + (", ".join(f"{k}: {len(v)}" for k, v in ks.items()) or "none")]
    if "backup" in arms and bk_specs:
        ev.window(0.6 if "repair" in arms else 1.0)
        rows = run_backup(ev, bk_specs, shape, a.z, a.backup_random, a.seed + 13, putter("backup.jsonl"), log)
        res["backup_flags"] = {r["name"]: r["flags"] for r in rows}
        res["backup_truncated"] = any("truncated" in r for r in rows) or len(rows) < len(bk_specs)
        md += ["\n## Damage-then-close\n", "| X | context | set | alone | after | excess +- SE | floor | verdict |",
               "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for cn, c in r["contexts"].items():
                for s, v in c["sets"].items():
                    md.append(f"| {r['name']} | {cn} | {s} | {v['alone']:+.4f} | {v['after']:+.4f} | "
                              f"{v['excess']:+.4f} +- {v['excess_se']:.4f} | {v['floor']:.4f} | "
                              f"{'hidden backup' if v['hidden_backup'] else 'excess' if v['excess_significant'] else '-'} |")
    if "repair" in arms and rep_specs:
        ev.window(1.0)
        rows = run_repair(ev, rep_specs, EN, putter("repair.jsonl"), log)
        res["repair_truncated"] = any("truncated" in r for r in rows) or len(rows) < len(rep_specs)
        res["repair"] = {r["box"]: {k: r.get(k) for k in ("best_give_back", "best_competitor", "spearman_drive_gain")}
                         for r in rows}
        md += ["\n## Repair (served composition = the damaged context)\n",
               "| box | served gap | best give-back (recovery) | best two | best competitor | rho(drive, gain) |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            if "served_gap" not in r:
                continue
            bg = r.get("best_give_back")
            rc = (r["give_back"].get(bg) or {}).get("recovery") if bg else None
            md.append(f"| {r['box']} | {r['served_gap']['mean']:+.4f} | {bg} ({rc}) | "
                      f"{(r.get('best_two') or {}).get('recovery')} | {r.get('best_competitor')} | "
                      f"{r.get('spearman_drive_gain')} |")
    (out / "interactions.md").write_text("\n".join(md) + "\n")
    return res


if __name__ == "__main__":
    main()
