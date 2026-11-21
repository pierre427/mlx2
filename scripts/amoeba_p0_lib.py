"""Shared pieces of the two P0 aMoEba experiments (scripts/amoeba_fn_suff.py, scripts/amoeba_fn_pareto.py;
docs/research/2026-11-20-amoeba-next-experiments.md 1 and 4): held loading with per-document groups, a scorer that
writes per-document rows (7.1), the random controls (7.5) and the bitwise determinism checks (7.2).

Held sets are read with the box battery's reader and seeded order (scripts/amoeba_box_battery.py read_held_rows;
np.random.default_rng(seed).shuffle, documents of >= T tokens, their first T tokens), so rows [0, docs) are the
battery's documents and rows [docs, 2 x docs) are the next, disjoint ones (1.4 / 4.5: "rows 81-160").
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import amoeba_battery_stats as ST
import amoeba_box_battery as BAT


def parse_spec(spec):
    """NAME=[gsm8k:]PATH[?filter][@TOKENS] -> (name, kind, path, filter, T)."""
    name, _, rest = spec.partition("=")
    rest, _, tok = rest.partition("@")
    kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
    path, _, filt = rest.removeprefix("gsm8k:").partition("?")
    return name, kind, path, filt, int(tok or 256)


def with_split(spec, split):
    """The same held spec reading another split (?split=val); a spec with another filter is refused."""
    name, kind, path, filt, T = parse_spec(spec)
    if kind != "corpus" or (filt and not filt.startswith("split=")):
        raise SystemExit(f"{spec}: cannot switch its split")
    return f"{name}={path}?split={split}@{T}"


def load_docs(host, spec, start, stop, seed):
    """Documents [start, stop) of the battery's seeded, length-filtered order: (token lists, row dicts)."""
    name, kind, path, filt, T = parse_spec(spec)
    rows = BAT.read_held_rows(path, kind, filt)
    np.random.default_rng(seed).shuffle(rows)
    seqs, kept = [], []
    n = 0
    for r in rows:
        t = host.encode(r["text"])
        if len(t) < T:
            continue
        if n >= start:
            seqs.append(t[:T])
            kept.append({k: v for k, v in r.items() if k != "text"})
        n += 1
        if n >= stop:
            break
    if len(seqs) < stop - start:
        raise SystemExit(f"held {spec}: {len(seqs)} documents of {T} tokens in rows [{start}, {stop}), "
                         f"{stop - start} wanted")
    return seqs, kept


class Budget(Exception):
    """No new work after the time budget (rows written so far stay)."""


class Scorer:
    """Masked held scoring with per-document rows. keep: [len(layers), W] bool over the partition's MoE layers
    (True = open); every other layer stays whole. mean: per-layer {layer: [E*F]} fill for closed neurons (None =
    zero ablation, the drop rule); thresh: {layer: t} per-token |h| threshold (TEAL-style reference)."""

    def __init__(self, host, E, D, layers, W, out, log=print, budget_s=None):
        self.host, self.E, self.D, self.layers, self.W, self.out, self.log = host, E, D, list(layers), W, Path(out), log
        self.held, self.meta, self.clean = {}, {}, {}
        self.n = 0
        self.t = 0.0
        self.t0 = time.time()
        self.budget_s = budget_s
        self.perdoc = self.out / "perdoc.jsonl"
        self.done = ST.load_perdoc(self.perdoc)

    def add_set(self, name, seqs, rows, spec):
        self.held[name] = seqs
        self.meta[name] = ST.doc_meta(rows, spec)
        p = self.out / "perdoc.docs.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=0))
        tmp.replace(p)

    def check_budget(self):
        if self.budget_s is not None and time.time() - self.t0 > self.budget_s:
            raise Budget()

    def _cfg(self, keep, mean, thresh, open_acc):
        import mlx.core as mx
        host = self.host
        m = np.ones((host.L, self.W), np.float32)
        m[self.layers] = keep.astype(np.float32)
        base = self.E.configure(host, self.D.from_np(m, host), None, host.L)
        mm = None
        if mean is not None:
            mm = [None] * host.L
            for l, v in mean.items():
                mm[l] = mx.array(np.asarray(v, np.float32))

        def f():
            base()
            c = host.ctl
            c.moe_mean, c.moe_thresh, c.moe_open = mm, thresh, open_acc
        return f

    def score(self, key, arm, keep, sets, mean=None, thresh=None, open_acc=None, reuse=True):
        """Per-set mean dCE (pooled, as the battery) for one closure; per-document rows appended. A triple already
        in the sidecar (a resumed run) is read back instead of rescored when reuse."""
        out, todo = {}, []
        for s in sets:
            r = self.done.get((key, arm, s)) if reuse else None
            if r is not None:
                out[s] = float(np.sum(np.asarray(r["ce"]) - np.asarray(r["clean_ce"])) / max(1, np.sum(r["n"])))
            else:
                todo.append(s)
        if todo:
            self.check_budget()
            t0 = time.time()
            pd = {}
            cfg = self._cfg(keep, mean, thresh, open_acc)
            r = self.E.english_cost(self.host, {s: {"held": self.held[s]} for s in todo}, self.clean, cfg,
                                    pooled=False, perdoc=pd)
            for s in todo:
                out[s] = float(r[s])
                self.done[(key, arm, s)] = ST.append_perdoc(self.perdoc, key, arm, s, pd[s])
            self.n += len(todo)
            self.t += time.time() - t0
        return out

    def dce(self, key, arm, s):
        """Per-document dCE of a scored triple."""
        return ST.doc_dce(self.done[(key, arm, s)])

    def same(self, a, b, s):
        """Bitwise equality of two scored triples' per-document CE sums (7.2)."""
        ra, rb = self.done[(*a, s)], self.done[(*b, s)]
        return ra["ce"] == rb["ce"] and ra["n"] == rb["n"]


def random_like(mask, g, cand=None):
    """The battery's layer-matched random control (BAT.random_like) with an optional candidate set: the same per-layer
    counts among `cand` (default: every unit outside mask). Returns (mask, shortfall units)."""
    if cand is None:
        return BAT.random_like(mask, g), 0
    rnd = np.zeros_like(mask)
    short = 0
    for l in range(mask.shape[0]):
        k = int(mask[l].sum())
        if k:
            c = np.flatnonzero(cand[l])
            take = min(k, c.size)
            short += k - take
            if take:
                rnd[l, g.choice(c, take, replace=False)] = True
    return rnd, short


def random_in_experts(mask, cand, F, g):
    """7.5 random-in-same-experts: the same per-(layer, expert) counts as mask, drawn among cand's units of that
    expert (unit id = expert * F + neuron). Returns (mask, shortfall units: where an expert has too few)."""
    L, N = mask.shape
    E = N // F
    rnd = np.zeros_like(mask)
    short = 0
    for l in range(L):
        per = mask[l].reshape(E, F).sum(1)
        cl = cand[l].reshape(E, F)
        for e in np.flatnonzero(per):
            c = np.flatnonzero(cl[e])
            take = min(int(per[e]), c.size)
            short += int(per[e]) - take
            if take:
                rnd[l, e * F + g.choice(c, take, replace=False)] = True
    return rnd, short


def atomic_json(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def append_jsonl(path, row):
    return ST.append_line(path, row)


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for x in p.read_text().split("\n"):
        try:
            out.append(json.loads(x))
        except ValueError:
            continue
    return out
