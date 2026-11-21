#!/usr/bin/env python3
"""Closable fraction vs English cost for FN neuron units (P0, Pierre 2026-11-20 "Yes" to queueing both P0
experiments; docs/research/2026-11-20-amoeba-next-experiments.md 4, with the 7.1-7.3 fixes). One model load, inside a
GPU window. Each cell closes a unit set chosen by an ORDER at a FRACTION of the 48 MoE layers' neuron units (zero
ablation, the drop rule, unless noted) and scores it per document (OUT/perdoc.jsonl, 7.1).

Orders (first-to-close to last; ties broken by a seeded permutation):
  O-rand     layer-matched random (seeds 0, 1, 2): every layer closes the same fraction
  O-fire     ascending English mean_target of the English map (the partition's core_map); units the map selects
             close last
  O-part     the joint partition's own closure sequence: joint, group and box tiles first, then pair tiles, each
             kind by ascending tile size (the English box is left out: its gate is the text being English); units
             of a partly closed tile in seeded order; the core never closes (cells past the closable share: n/a)
  O-causal0  ascending English N0_u = -E_seq[dCE/dm_u] (AtP, zero baseline; research/amoeba/carve/fisher.accumulate
             labels=ce; needs the gradient gate, else recorded as unavailable)
  O-fisher   ascending English F_u = E_seq[(d log p(y)/dm_u)^2], y sampled (empirical Fisher; labels=sampled)
  O-reap-n   ascending E[w |h|] x ||W_down[:, f]|| on English fit (neuron REAP saliency; moestats)
  O-reap-e   REAP at expert grain, E_routed[w ||f_e(x)||] (whole experts; the fraction rounds to whole experts)
  O-freq-e   ascending routing frequency on English fit (SEER-style; whole experts)
Grid (4.7, trimmed): every order at 10/20/30/40% (O-rand all 3 seeds); mean compensation at 20% for every order (a
closed neuron takes its conditional mean mu, research/amoeba/carve/units.py Control.moe_mean); the depth schedules
D-tenp (TENP's trapezoid: 1.5f flat over the first quarter of layers, linear to 0.5f over the middle half, 0.5f over
the last quarter, rescaled to mean f) and D-ends (at most f/2 in layers 0-5 and 42-47, the rest spread over the
middle) at the 4 fractions for the best order; TEAL-style per-token |h| thresholds at the same 4 fractions (a
measurement, not a servable closure); the do-nothing cell (fraction 0 through the same path, must be bitwise the
unmasked model); repeat cells O-rand and O-causal0 at 20% on English val rows 81-160.
Sets per cell (val, used to choose): English val (lang-prose-eng val split), GSM8K train rows (dCE), deu val, soc.law
held rows 81-160 (the battery spent rows 1-80; that build has no val split) and the capitals knowledge probe
(scripts/amoeba27b_lib.py COUNTRIES, "The capital of {} is", teacher-forced exact match of the whole answer, all 143
countries; per-item correct kept for paired comparisons). Held verdict (4.5): the chosen order, schedule and
compensation at its CF@0.05 fraction, and O-fire and O-rand at the same fraction, once on English held rows 81-160.
The held verdict runs after the forward-only orders and again if a causal order then wins.

Decision rules (4.5, pre-registered; scripts/amoeba_battery_stats.py PARETO_RULES) -> OUT/summary.json and
OUT/summary.md (rerunnable: amoeba_battery_stats.py pareto --dir OUT). Rows: OUT/cells.jsonl. Resumable from the
sidecar; --budget-min stops new scoring (rows so far stay).

Usage: amoeba_fn_pareto.py --model M --partition DIR --joint-partition DIR --battery DIR --out DIR [--assert-ple]
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
import amoeba_battery_stats as ST  # noqa: E402
import amoeba_box_battery as BAT  # noqa: E402
import amoeba_p0_lib as P0  # noqa: E402

ORDERS = ("O-rand", "O-fire", "O-part", "O-causal0", "O-fisher", "O-reap-n", "O-reap-e", "O-freq-e")
FORWARD_ORDERS = ("O-rand", "O-fire", "O-part", "O-reap-n", "O-reap-e", "O-freq-e")
CAUSAL = ("O-causal0", "O-fisher")
EXPERT_GRAIN = ("O-reap-e", "O-freq-e")
NEURON_GRAIN = ("O-fire", "O-causal0", "O-fisher", "O-reap-n")           # O-part is tile grain, O-rand the control
SETS = ("english", "gsm8k", "deu", "soc.law")
REP = "english#rep"
HELD = "english@held"


# ----------------------------------------------------------------------------- orders (CPU)

class Order:
    """A closing order over [L, N] units. glob: int32 [L*N] closing position (0 first; a never-closing unit gets
    L*N); grain "neuron" or "expert" (F units per expert, closed whole); layer_matched: close per layer."""

    def __init__(self, name, L, N, F, glob=None, lrank=None, grain="neuron", limit=None):
        self.name, self.L, self.N, self.F, self.grain = name, L, N, F, grain
        self.glob, self._lrank = glob, lrank
        self.limit = limit if limit is not None else L * N          # closable units (O-part: the non-core share)

    def lrank(self):
        """Within-layer closing position [L, N] (layer-matched orders have it; global orders derive it)."""
        if self._lrank is None:
            g = self.glob.reshape(self.L, self.N)
            self._lrank = np.argsort(np.argsort(g, axis=1, kind="stable"), axis=1, kind="stable").astype(np.int32)
        return self._lrank

    def closed(self, frac, per_layer=None):
        """bool [L, N] closed units at a unit fraction (uniform; a layer-matched order closes it in every layer), or
        at per-layer fractions (depth schedules: each layer's first units in this order's within-layer sequence);
        None when the order cannot close that much (O-part past its closable share)."""
        L, N, F = self.L, self.N, self.F
        if per_layer is None and self.glob is not None:
            k = int(round(frac * L * N))
            if self.grain == "expert":
                k = int(round(k / F)) * F
            if k > self.limit:
                return None
            return (self.glob < k).reshape(L, N)
        fl = np.full(L, frac) if per_layer is None else np.asarray(per_layer, np.float64)
        lr = self.lrank()
        ks = np.rint(fl * N).astype(np.int64)
        if self.grain == "expert":
            ks = np.rint(ks / F).astype(np.int64) * F
        if self.glob is not None:
            cap = (self.glob.reshape(L, N) < self.limit).sum(1)
            if (ks > cap).any():
                return None
        return lr < ks[:, None]


def global_order(name, key, F, seed, grain="neuron", never=None):
    """Ascending key closes first; ties by a seeded permutation; never (bool [L, N]) units never close."""
    L, N = key.shape
    g = np.random.default_rng(seed)
    tie = g.permutation(L * N)
    k = key.reshape(-1).astype(np.float64)
    o = np.lexsort((tie, k)) if never is None else np.lexsort((tie, k, never.reshape(-1)))
    glob = np.empty(L * N, np.int32)                                  # 15.7M units < 2^31 (63 MB per order)
    glob[o] = np.arange(L * N, dtype=np.int32)
    limit = L * N if never is None else int((~never).sum())
    return Order(name, L, N, F, glob=glob, grain=grain, limit=limit)


def expert_order(name, ekey, F, seed):
    """Whole experts by ascending ekey [L, E] (ties seeded): unit position = expert position * F + neuron."""
    L, E = ekey.shape
    g = np.random.default_rng(seed)
    o = np.lexsort((g.permutation(L * E), ekey.reshape(-1).astype(np.float64)))
    epos = np.empty(L * E, np.int64)
    epos[o] = np.arange(L * E)
    glob = (epos.reshape(L, E, 1) * F + np.arange(F)).reshape(-1).astype(np.int32)
    return Order(name, L, E * F, F, glob=glob, grain="expert")


def rand_order(L, N, F, seed):
    g = np.random.default_rng(seed)
    lr = np.stack([np.argsort(g.permutation(N)).astype(np.int32) for _ in range(L)])
    return Order(f"O-rand.s{seed}", L, N, F, lrank=lr)


def part_sequence(jprep, keys, sizes):
    """O-part's tile sequence: joint, group and box tiles (one kind), then pair tiles, each by ascending size (name
    breaks ties); the English box is left out."""
    en, _ = BAT.english_box(jprep)
    first = [k for k in keys if k.startswith(("joint:", "group:", "box:")) and k != f"box:{en}"]
    pairs = [k for k in keys if k.startswith("pair:")]
    return sorted(first, key=lambda k: (sizes[k], k)) + sorted(pairs, key=lambda k: (sizes[k], k))


def part_order(jz, jprep, F, seed):
    L, N = jz.shape
    keys = [k for k in jz.keys if k != "core"]
    sizes = {k: jz.units(k) for k in keys}
    seq = part_sequence(jprep, keys, sizes)
    g = np.random.default_rng(seed)
    glob = np.full(L * N, L * N, np.int32)
    pos = 0
    for k in seq:
        ix = jz.idx[k].astype(np.int64)
        ix = ix[glob[ix] == L * N]                                     # a unit is in one tile
        ix = ix[g.permutation(ix.size)]
        glob[ix] = pos + np.arange(ix.size)
        pos += ix.size
    return Order("O-part", L, N, F, glob=glob, limit=pos), {"tiles": len(seq), "closable_units": pos,
                                                           "closable_share": pos / (L * N)}


def depth_fractions(kind, f, L):
    """Per-layer fractions with mean f (clipped to [0, 1], the clip's excess spread over unclipped layers)."""
    x = np.arange(L, dtype=np.float64)
    if kind == "D-tenp":
        q1, q3 = L / 4, 3 * L / 4
        prof = np.where(x < q1, 1.5, np.where(x >= q3, 0.5, 1.5 - (x - q1) / (q3 - q1)))
        fl = prof * f / prof.mean()
    elif kind == "D-ends":
        ends = (x < 6) | (x >= L - 6)
        fl = np.where(ends, f / 2, 0.0)
        mid = ~ends
        fl[mid] = (f * L - fl[ends].sum()) / max(1, mid.sum())
    else:
        raise ValueError(kind)
    for _ in range(8):
        over = fl > 1.0
        if not over.any():
            break
        ex = (fl[over] - 1.0).sum()
        fl[over] = 1.0
        free = fl < 1.0
        fl[free] += ex / max(1, free.sum())
    return fl


# ----------------------------------------------------------------------------- capitals probe

def capital_items(host):
    import amoeba27b_lib as A
    tpl = A.CAP_TEMPLATES[0]
    items = []
    for c, k in A.COUNTRIES:
        p, ans = host.encode(tpl.format(c)), host.encode(" " + k)
        items.append({"ids": p + ans, "pos": list(range(len(p) - 1, len(p) + len(ans) - 1)), "tgt": ans, "label": c})
    return items


def capitals(host, items, cfg_fn=None, cap=16):
    """Teacher-forced exact match of the whole answer per item (amoeba27b_lib.task_eval's rule): 0/1 list."""
    import mlx.core as mx
    by = {}
    for i, it in enumerate(items):
        by.setdefault(len(it["tgt"]), []).append(i)
    out = [0] * len(items)
    for _, idx in sorted(by.items()):
        for j in range(0, len(idx), cap):
            grp = idx[j:j + cap]
            T = max(len(items[i]["ids"]) for i in grp)
            ids = mx.array([items[i]["ids"] + [0] * (T - len(items[i]["ids"])) for i in grp])
            pos = mx.array([items[i]["pos"] for i in grp])
            if cfg_fn:
                cfg_fn()
            lg = host.logits_at(ids, pos)
            host.ctl.reset()
            tgt = mx.array([items[i]["tgt"] for i in grp])
            ok = (lg.argmax(-1) == tgt).all(-1)
            mx.eval(ok)
            for i, v in zip(grp, np.asarray(ok).tolist()):
                out[i] = int(bool(v))
            mx.clear_cache()
    return out


# ----------------------------------------------------------------------------- driver

def _wired_arg(spec):
    from research.amoeba.carve.ops import wired_arg
    return wired_arg(spec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--joint-partition", required=True, help="O-part's partition (amoeba_fn_partition --joint-floor)")
    ap.add_argument("--battery", required=True, help="the finished battery run (held specs: english, deu, soc.law)")
    ap.add_argument("--english-map", default="", help="O-fire's map dir (default: partition.json inputs.core_map)")
    ap.add_argument("--gsm8k", default=str(ROOT / "benchmarks/gsm8k/train.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--english", default="english")
    ap.add_argument("--fractions", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--comp-fraction", type=float, default=0.2)
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rand-seeds", default="0,1,2")
    ap.add_argument("--stats-docs", type=int, default=80, help="English fit documents for REAP / frequency / TEAL")
    ap.add_argument("--mu-docs", type=int, default=24)
    ap.add_argument("--mu-sets", default="english," + ",".join(sorted(BAT.LANG10)))
    ap.add_argument("--attrib-seqs", type=int, default=400)
    ap.add_argument("--attrib-tokens", type=int, default=256)
    ap.add_argument("--attrib-budget-min", type=float, default=40.0, help="split equally between the two labels")
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--attrib-extra-gb", type=float, default=5.0,
                    help="skip the attribution when one full-length backward's MLX peak exceeds the forward phase's "
                         "MLX peak by more than this (the smoke itself is one such backward: a kill there resumes "
                         "with RESUME=1 EXTRA=--no-causal)")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--watchdog-hard-gb", type=float, default=None)
    ap.add_argument("--watchdog-every-s", type=float, default=1.0)
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--compressor-growth-gb", type=float, default=12.0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto", type=_wired_arg)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd, jd = Path(a.partition), Path(a.joint_partition)
    prep, jprep = (json.loads((d / "partition.json").read_text()) for d in (pd, jd))
    if prep.get("fingerprint") != jprep.get("fingerprint"):
        raise SystemExit("--joint-partition is of another model")
    tz, jz = BAT.Tiles(pd / "tiles.npz"), BAT.Tiles(jd / "tiles.npz")
    layers = tz.layers
    if jz.layers != layers or jz.shape != tz.shape:
        raise SystemExit("--joint-partition: another layer list or shape")
    bmeta = json.loads((Path(a.battery) / "battery.meta.json").read_text())
    if bmeta.get("fingerprint") != prep.get("fingerprint"):
        raise SystemExit("--battery is of another model")
    specs = {k: v["spec"] for k, v in bmeta["held"].items()}
    EN = a.english
    for s in (EN, "deu", "soc.law"):
        if s not in specs:
            raise SystemExit(f"--battery has no held spec for {s}")
    emap = Path(a.english_map or prep["inputs"]["core_map"])
    fr = [float(x) for x in a.fractions.split(",") if x]
    seeds = [int(x) for x in a.rand_seeds.split(",") if x]
    settings = {"partition": str(pd), "joint_partition": str(jd), "battery": a.battery, "english_map": str(emap),
                "gsm8k": a.gsm8k, "fractions": fr, "comp_fraction": a.comp_fraction, "docs": a.docs, "seed": a.seed,
                "rand_seeds": seeds, "stats_docs": a.stats_docs, "mu_docs": a.mu_docs, "mu_sets": a.mu_sets,
                "attrib_seqs": a.attrib_seqs, "attrib_tokens": a.attrib_tokens, "english": EN}
    # --no-causal is not a setting: a run killed in the attribution capture resumes with it (recorded in meta)
    sp = out / "settings.json"
    if sp.exists() and json.loads(sp.read_text()) != settings:
        raise SystemExit(f"{out} holds a run with other settings (settings.json): use a fresh --out")
    P0.atomic_json(sp, settings)
    from research.amoeba.carve import ops as O
    O.prepare_wiring(a.wired_gb)
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
    from research.amoeba.carve import fisher as FI
    from research.amoeba.carve import moestats as MS
    from research.amoeba.carve.host import Host
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    P0.atomic_json(out / "status.json", {"status": "started", "settings": settings})
    O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb, every_s=a.watchdog_every_s,
                         min_avail_gb=a.min_avail_gb, hard_gb=a.watchdog_hard_gb,
                         compressor_growth_gb=a.compressor_growth_gb, receipt=out / "watchdog-stop.json",
                         watcher=True)
    O.setup_mlx(a.memory_gb, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    if host.fingerprint != prep["fingerprint"]:
        raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    wired = O.wire_weights(a.wired_gb, log, memory_gb=a.memory_gb)
    checks, attrib = {}, {}
    prev = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    attempts = list(prev.get("attempts") or []) + [{"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                    "mlx2_source_revision": O.git_rev(), "causal": not a.no_causal}]

    def write_meta(status, **extra):
        """meta.json from the model load on (a killed run still names its checkpoint and revision)."""
        P0.atomic_json(out / "meta.json", {
            "checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "settings": settings, "checks": checks, "attribution": attrib, "status": status, "attempts": attempts,
            "wired": wired, **extra})
    write_meta("running: loading held sets")
    W = max(host.widths)
    nsite = {s.layer: s for s in MS.neuron_sites(host)}
    if sorted(nsite) != sorted(layers) or tz.shape != (len(layers), W):
        raise SystemExit("partition layers / shape are not the host's neuron MoE sites")
    L, N = tz.shape
    F = nsite[layers[0]].F
    Ex = N // F
    sc = P0.Scorer(host, E, D, layers, W, out, log, None if a.budget_min is None else a.budget_min * 60)
    sp_en_val = P0.with_split(specs[EN], "val")
    for name, spec, lo, hi in ((EN, sp_en_val, 0, a.docs), (REP, sp_en_val, a.docs, 2 * a.docs),
                               ("deu", P0.with_split(specs["deu"], "val"), 0, a.docs),
                               ("soc.law", specs["soc.law"], 2 * a.docs, 3 * a.docs),     # [80, 160): suff's repeat
                               ("gsm8k", f"gsm8k=gsm8k:{a.gsm8k}@160", 0, a.docs),
                               (HELD, specs[EN], a.docs, 2 * a.docs)):
        seqs, rows = P0.load_docs(host, spec, lo, hi, a.seed)
        sc.add_set(name, seqs, rows, spec + f" rows [{lo}, {hi})")
    items = capital_items(host)
    cells_path = out / "cells.jsonl"
    cells = {r["cell"]: r for r in P0.read_jsonl(cells_path)}
    cap_clean = cells.get("_clean", {}).get("capitals_correct")
    if cap_clean is None:
        cap_clean = capitals(host, items)
        cells["_clean"] = P0.append_jsonl(cells_path, {"cell": "_clean", "capitals_correct": cap_clean,
                                                        "capitals_acc": float(np.mean(cap_clean))})
    # ---- statistics for the orders, mean compensation and TEAL (forward only)
    stp = out / "stats.npz"
    if stp.exists():
        z = np.load(stp)
        wh, cnt, reap = z["wh"], z["count"], z["reap_e"]
        mu = {l: z["mu"][i] for i, l in enumerate(layers)}
        teal = {float(q): z["teal"][j] for j, q in enumerate(z["teal_q"])}
        cn = z["colnorm"]
    else:
        seqs, _ = P0.load_docs(host, P0.with_split(specs[EN], "fit"), 0, a.stats_docs, a.seed)
        st = MS.capture(host, seqs, quantiles=tuple(fr), log=log)
        wh = np.stack([st[l]["wh"] for l in layers])
        cnt = np.stack([st[l]["count"] for l in layers])
        reap = np.stack([st[l]["reap_e"] for l in layers])
        teal = {q: np.array([st[l]["habs_q"][q] for l in layers]) for q in fr}
        mseqs = []
        for s in [x for x in a.mu_sets.split(",") if x]:
            q, _ = P0.load_docs(host, P0.with_split(specs[s], "fit"), 0, a.mu_docs, a.seed)
            mseqs += q
        stm = MS.capture(host, mseqs, log=log)
        mu = {l: stm[l]["mu"] for l in layers}
        cnd = MS.down_col_norms(host)
        cn = np.stack([cnd[l] for l in layers])
        np.savez(out / "stats.tmp.npz", wh=wh, count=cnt, reap_e=reap, mu=np.stack([mu[l] for l in layers]),
                 teal_q=np.array(fr), teal=np.stack([teal[q] for q in fr]), colnorm=cn)
        (out / "stats.tmp.npz").replace(stp)
    zm = np.load(emap / "map.npz")
    mt = np.stack([zm[f"mean_target_{s}"] for s in range(L)]).astype(np.float64)
    sel = np.stack([zm[f"target_{s}"] for s in range(L)]).astype(bool)
    orders = {f"O-rand.s{s}": rand_order(L, N, F, 100 + s) for s in seeds}
    orders["O-fire"] = Order("O-fire", L, N, F, glob=_fire_glob(mt, sel, a.seed + 1))
    del mt, sel, zm
    orders["O-part"], part_info = part_order(jz, jprep, F, a.seed + 2)
    orders["O-reap-n"] = global_order("O-reap-n", wh * cn, F, a.seed + 3)
    orders["O-reap-e"] = expert_order("O-reap-e", reap, F, a.seed + 4)
    orders["O-freq-e"] = expert_order("O-freq-e", cnt, F, a.seed + 5)
    truncated = None
    fwd_peak = [0.0]
    write_meta("running: cells", part=part_info)
    t_start = time.time()

    def run_cell(cell, order, frac, closed, comp="zero", schedule="base", sets=SETS, thresh=None, cap=True):
        if cell in cells and all((cell, "cell", s) in sc.done for s in sets):
            return cells[cell]
        if closed is None and thresh is None:
            row = {"cell": cell, "order": order, "fraction": frac, "comp": comp, "schedule": schedule,
                   "na": "the order cannot close this fraction"}
            cells[cell] = P0.append_jsonl(cells_path, row)
            return row
        keep = np.ones((L, N), bool) if closed is None else ~closed
        acc = {} if thresh is not None else None
        vals = sc.score(cell, "cell", keep, list(sets), mean=mu if comp == "mean" else None, thresh=thresh,
                        open_acc=acc)
        row = {"cell": cell, "order": order, "fraction": frac, "comp": comp, "schedule": schedule,
               "closed_units": int(L * N - keep.sum()), "closed_share": float(1 - keep.mean()),
               "closed_share_layers": [round(float(x), 5) for x in 1 - keep.mean(1)], "dce": vals}
        if acc:
            o = sum(float(v[0]) for v in acc.values())
            t = sum(float(v[1]) for v in acc.values())
            row["teal_open_share"] = o / t if t else None
        if cap:
            sc.check_budget()
            cfg = sc._cfg(keep, mu if comp == "mean" else None, thresh, None)
            cc = capitals(host, items, cfg)
            row.update(capitals_correct=cc, capitals_acc=float(np.mean(cc)))
        cells[cell] = P0.append_jsonl(cells_path, row)
        log(f"{cell}: closed {row['closed_share']:.3f}; " + " ".join(f"{k} {v:+.4f}" for k, v in vals.items())
            + (f"; capitals {row['capitals_acc']:.3f}" if cap else ""))
        return row

    def grid(names):
        for o in names:
            for f in fr:
                run_cell(f"{o}@{f:g}", o, f, orders[o].closed(f))
        for o in names:                                # mean compensation: every order once (O-rand: seed 0)
            if o.startswith("O-rand.") and o != f"O-rand.s{seeds[0]}":
                continue
            f = a.comp_fraction
            run_cell(f"{o}@{f:g}|mean", o, f, orders[o].closed(f), comp="mean")

    def best_and_depth(names):
        summ = summarize(out, write=False)
        cf = {o: (summ["orders"].get(o) or {}).get("cf") for o in names if not o.startswith("O-rand.")}
        cf = {o: v for o, v in cf.items() if v is not None}
        if not cf:
            return None
        best = max(cf, key=lambda o: (cf[o], o))
        for kind in ("D-tenp", "D-ends"):
            for f in fr:
                run_cell(f"{best}@{f:g}|{kind}", best, f, orders[best].closed(f, depth_fractions(kind, f, L)),
                         schedule=kind)
        return best

    def held_verdict():
        """4.5: once, after every val cell: the choice at its CF fraction, O-fire and O-rand at the same fraction,
        on English held rows 81-160. Cells are keyed by content, so a resumed run never relabels a stale row."""
        summ = summarize(out, write=False)
        ch = summ.get("choice")
        if not ch or not ch.get("fraction"):
            return {"skipped": "no choice with CF > 0 on val"}
        f = ch["fraction"]
        o, sched, comp = ch["order"], ch["schedule"], ch["comp"]
        if o not in orders and o in CAUSAL and (out / "attrib.npz").exists():
            # resumed with --no-causal after an attempt that saved the attribution and scored causal cells: the val
            # choice (from cells.jsonl) is a causal order this attempt did not build. Its mask is CPU-only.
            z = np.load(out / "attrib.npz")
            orders[o] = global_order(o, z["n0" if o == "O-causal0" else "fisher"], F,
                                     a.seed + (8 if o == "O-causal0" else 9))
        if o not in orders:
            return {"skipped": f"the val choice {o} has no order in this attempt"}
        pl = None if sched == "base" else depth_fractions(sched, f, L)
        res = {"choice": ch}
        for role, oo, sch, cm in (("choice", o, sched, comp), ("O-fire", "O-fire", "base", "zero"),
                                  ("O-rand", f"O-rand.s{seeds[0]}", "base", "zero")):
            name = f"held:{oo}|{sch}|{cm}@{f:.4f}"
            row = run_cell(name, oo, f, orders[oo].closed(f, pl if role == "choice" else None), comp=cm,
                           schedule=sch, sets=(HELD,), cap=False)
            row = dict(row, role=role)
            res[role] = {"cell": name, "dce": (row.get("dce") or {}).get(HELD), "na": row.get("na")}
        P0.atomic_json(out / "held_verdict.json", res)
        return res

    try:
        run_cell("do-nothing@0", "none", 0.0, np.zeros((L, N), bool), sets=SETS + (REP, HELD))
        r0 = {s: sc.done[("do-nothing@0", "cell", s)] for s in SETS + (REP, HELD)}
        checks["do_nothing_bitwise"] = {s: bool(r["ce"] == r["clean_ce"]) for s, r in r0.items()}
        checks["do_nothing_capitals_equal"] = cells["do-nothing@0"].get("capitals_correct") == cap_clean
        grid(list(orders))
        run_cell(f"O-rand.s{seeds[0]}@{a.comp_fraction:g}|repeat", f"O-rand.s{seeds[0]}", a.comp_fraction,
                 orders[f"O-rand.s{seeds[0]}"].closed(a.comp_fraction), sets=(REP,), cap=False)
        for f in fr:
            run_cell(f"TEAL@{f:g}", "TEAL", f, None, thresh={l: float(teal[f][i]) for i, l in enumerate(layers)})
        best_and_depth([o for o in orders])
        write_meta("running: forward cells done", part=part_info)
        if not a.no_causal:
            ap_ = out / "attrib.npz"
            if ap_.exists():
                z = np.load(ap_)
                n0, fi = z["n0"], z["fisher"]
                attrib = json.loads((out / "attrib.json").read_text())
            else:
                import mlx.core as mx
                seqs, _ = P0.load_docs(host, P0.with_split(specs[EN], "fit"), 0, a.attrib_seqs, a.seed + 7)
                seqs = [q[:a.attrib_tokens] for q in seqs]
                ok = None
                write_meta("running: attribution gate (a kill here: resume with EXTRA=--no-causal)", part=part_info)
                try:                                         # the carve driver's gates (research/amoeba/carve/__main__)
                    host.check_paths(mx.array(seqs[:2]))
                    gate_ids = mx.array(seqs[:1])[:, :16]
                    if host.grad_mode == "layer_ref":
                        lr = host.check_layer_reference(mx.array(seqs[:2]))
                        attrib["layer_reference_ok"] = bool(lr.get("ok"))
                        ok = bool(lr.get("ok")) and host.check_directional_grad(gate_ids)
                    else:
                        ok = host.check_layerwise_grad(gate_ids)
                    if ok:                                   # memory smoke: one full-length backward, measured
                        import mlx.core as mx_
                        fwd_peak[0] = max(fwd_peak[0], O.peak_gb())      # the forward phase's peak, kept for meta
                        mx_.reset_peak_memory()
                        t0 = time.time()
                        FI.accumulate(host, seqs[:1], "ce", log=lambda *_: None, layers=layers)
                        bk = O.peak_gb()
                        attrib["smoke"] = {"tokens": len(seqs[0]), "s": round(time.time() - t0, 1),
                                           "mlx_peak_gb": bk, "forward_mlx_peak_gb": fwd_peak[0]}
                        # every later sequence needs the same memory (the sums are numpy): a backward that needs
                        # more than --attrib-extra-gb over the forward phase's MLX peak is skipped (the battery's
                        # footprint sat ~10 GB over its MLX peak, 84 vs 74, under a 92 GB watchdog)
                        if bk > fwd_peak[0] + a.attrib_extra_gb:
                            ok = False
                            attrib["skipped"] = (f"one {len(seqs[0])}-token backward peaked at {bk} GB MLX, > the "
                                                 f"forward peak {fwd_peak[0]} + --attrib-extra-gb {a.attrib_extra_gb}")
                except Exception as e:                       # noqa: BLE001  (recorded; forward orders stand)
                    attrib["gate_error"] = repr(e)[:400]
                    ok = False
                attrib["gate_ok"] = bool(ok)
                attrib["grad_mode"] = getattr(host, "grad_mode", None)
                n0 = fi = None
                if ok:
                    half = a.attrib_budget_min * 30
                    r1 = FI.accumulate(host, seqs, "ce", seed=a.seed, log=log, budget_s=half, layers=layers)
                    r2 = FI.accumulate(host, seqs, "sampled", seed=a.seed, log=log, budget_s=half, layers=layers)
                    n0 = -np.stack([r1["sum"][l] for l in layers]) / max(1, r1["seqs"])
                    fi = np.stack([r2["sum"][l] for l in layers]) / max(1, r2["seqs"])
                    attrib.update(ce_seqs=r1["seqs"], ce_s=r1["s"], sampled_seqs=r2["seqs"], sampled_s=r2["s"],
                                  tokens=a.attrib_tokens, source="English fit split (seed + 7 order)")
                    np.savez(out / "attrib.tmp.npz", n0=n0, fisher=fi)
                    (out / "attrib.tmp.npz").replace(ap_)
                P0.atomic_json(out / "attrib.json", attrib)
            if n0 is not None:
                orders["O-causal0"] = global_order("O-causal0", n0, F, a.seed + 8)
                orders["O-fisher"] = global_order("O-fisher", fi, F, a.seed + 9)
                grid(list(CAUSAL))
                run_cell(f"O-causal0@{a.comp_fraction:g}|repeat", "O-causal0", a.comp_fraction,
                         orders["O-causal0"].closed(a.comp_fraction), sets=(REP,), cap=False)
                best_and_depth(list(orders))
        checks["held_verdict"] = held_verdict()
        rr = run_cell("do-nothing@0|repeat", "none", 0.0, np.zeros((L, N), bool), sets=(EN,), cap=False)
        checks["repeat_cell_bitwise"] = sc.same(("do-nothing@0", "cell"), ("do-nothing@0|repeat", "cell"), EN) \
            if rr else None
        first = f"O-rand.s{seeds[0]}@{fr[0]:g}"
        sc.score(first + "|repeat-end", "cell", ~orders[f"O-rand.s{seeds[0]}"].closed(fr[0]), [EN], reuse=True)
        checks["first_cell_repeat_bitwise"] = sc.same((first, "cell"), (first + "|repeat-end", "cell"), EN)
    except P0.Budget:
        truncated = f"--budget-min {a.budget_min}: no new scoring after {round((time.time() - t_start) / 60, 1)} min"
        log(truncated)
        if "held_verdict" not in checks:             # the verdict on the cells there are (3 scorings past the budget)
            sc.budget_s = None
            checks["held_verdict"] = held_verdict()
            checks["held_verdict"]["note"] = "scored after the budget stop, on the val cells finished by then"
    status = "finished" if not truncated else "truncated"
    write_meta(status, truncated=truncated, part=part_info, capitals_clean_acc=float(np.mean(cap_clean)),
               units=[L, N], F=F, experts=Ex, scorings=sc.n, scoring_s=round(sc.t, 1),
               sec_per_scoring=round(sc.t / max(1, sc.n), 2),
               watchdog={"limit_gb": a.watchdog_gb, "swap_growth_gb": a.swap_growth_gb,
                         "waiver_note": a.waiver_note or None,
                         **{k: v for k, v in getattr(O, "WATCH", {}).items() if k.startswith(("peak_", "observed_"))}},
               mlx_peak_gb=max(O.peak_gb(), fwd_peak[0]))
    meta = {"status": status}
    ST.to_npz(out / "perdoc.jsonl", out / "battery.perdoc.npz")
    summ = summarize(out)
    P0.atomic_json(out / "status.json", {"status": meta["status"]})
    log("summary:", json.dumps(summ.get("decisions")))
    return 0


def _fire_glob(mt, sel, seed):
    """O-fire: ascending mean_target, the English map's selected units last."""
    L, N = mt.shape
    tie = np.random.default_rng(seed).permutation(L * N)
    o = np.lexsort((tie, mt.reshape(-1), sel.reshape(-1)))
    glob = np.empty(L * N, np.int32)
    glob[o] = np.arange(L * N, dtype=np.int32)
    return glob


# ----------------------------------------------------------------------------- summary (CPU)

def summarize(out, rules=ST.PARETO_RULES, n_boot=ST.N_BOOT, write=True):
    """Apply the 4.5 rules to OUT's cells; writes OUT/summary.json and OUT/summary.md when write."""
    out = Path(out)
    rows = ST.load_perdoc(out / "perdoc.jsonl")
    docs = json.loads((out / "perdoc.docs.json").read_text())
    settings = json.loads((out / "settings.json").read_text())
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    cells = {r["cell"]: r for r in P0.read_jsonl(out / "cells.jsonl")}
    EN = settings["english"]
    g, st = docs[EN]["groups"], docs[EN]["strata"]
    clean = cells.get("_clean", {})
    cap0 = clean.get("capitals_acc")

    def dce(cell, s):
        r = rows.get((cell, "cell", s))
        return None if r is None else ST.doc_dce(r)

    def flagged(c):
        acc = c.get("capitals_acc")
        e = (c.get("dce") or {}).get(EN)
        return bool(cap0 is not None and acc is not None and cap0 - acc > rules["cliff_points"]
                    and e is not None and e <= rules["bar"])
    curves = {}
    by_order = {}
    for c in cells.values():
        if c["cell"].startswith(("_", "held:", "do-nothing")) or "|" in c["cell"] and \
                not c["cell"].split("|")[1].startswith("D-") or c.get("na"):
            continue
        key = c["order"] if c.get("schedule", "base") == "base" else f"{c['order']}|{c['schedule']}"
        if c.get("comp") != "zero":
            continue
        v = dce(c["cell"], EN)
        if v is None:
            continue
        by_order.setdefault(key, []).append({"fraction": c["fraction"], "english": v, "flagged": flagged(c),
                                             "cell": c["cell"]})
    for k, cs in sorted(by_order.items()):
        curves[k] = ST.pareto_cf(cs, g, st, rules, n_boot)
        for x, c in zip(curves[k]["cells"], sorted(cs, key=lambda c: c["fraction"])):
            x["cell"] = c["cell"]
    cf = {k: v["cf"] for k, v in curves.items()}
    rand = [k for k in curves if k.startswith("O-rand.") and "|" not in k]
    rband = [min(curves[k]["band"][0] for k in rand), max(curves[k]["band"][1] for k in rand)] if rand else None
    base = {k: v for k, v in cf.items() if "|" not in k}
    causal = {k: v for k, v in base.items() if k in CAUSAL}
    ck = meta.get("checks") or {}
    failed = [k for k, x in ck.items() if k != "held_verdict" and
              (x is False or (isinstance(x, dict) and not all(bool(v) for v in x.values())))]
    missing = [k for k in ("do_nothing_bitwise", "repeat_cell_bitwise") if k not in ck]
    dec = {"checks_ok": not failed, "checks_failed": failed, "checks_missing": missing,
           "checks_note": "a failed bitwise control means every number here carries that difference as a floor: "
                          "read the decisions only after resolving it"}
    # floor (7.4 / 7.6): each repeat cell's English mean on disjoint val rows vs its cell's
    fl = {}
    for c in cells.values():
        if c["cell"].endswith("|repeat") and not c["cell"].startswith("do-nothing"):
            base_cell = c["cell"][: -len("|repeat")]
            r1, r2 = dce(base_cell, EN), dce(c["cell"], REP)
            if r1 is not None and r2 is not None:
                fl[base_cell] = abs(float(r2.mean()) - float(r1.mean()))
    floor = max(fl.values()) if fl else None
    dec["floor"] = {"english_doc_resample": fl, "max": floor, "rule": rules["floor"]}
    if "O-fire" in base and rband:
        fire_m = curves["O-fire"]["cf_mean"]
        inside = rband[0] <= fire_m <= rband[1]
        bc = max(causal.values()) if causal else None
        dec["h4"] = {"cf_fire": base["O-fire"], "cf_fire_mean_curve": fire_m, "rand_band": rband,
                     "fire_inside_rand_band": inside, "cf_best_causal": bc,
                     "best_causal": max(causal, key=causal.get) if causal else None,
                     "holds": (bool(inside and bc > 0 and bc >= rules["h4_x"] * base["O-fire"]) if bc is not None
                               else None),
                     "note": None if causal else "no causal order scored (gradient gate or --no-causal)"}
    neur = {k: v for k, v in base.items() if k in NEURON_GRAIN}
    expg = {k: v for k, v in base.items() if k in EXPERT_GRAIN}
    if "O-reap-n" in base and "O-reap-e" in base and expg:
        bn = max(neur, key=lambda k: neur[k], default=None)
        be = max(expg, key=expg.get)
        d1 = base["O-reap-n"] - base["O-reap-e"]
        d2 = (neur[bn] - expg[be]) if bn else None
        dec["grain"] = {"reap_n_minus_reap_e": d1, "best_neuron": bn, "best_expert": be, "best_neuron_minus_best_expert": d2,
                        "neuron_beats_expert": bool(d1 >= rules["grain_points"] and d2 is not None
                                                    and d2 >= rules["grain_points"])}
    # the choice on val: best (order, schedule) by CF; compensation by the 4.5 candidate rule at comp_fraction
    choice = None
    cand = {k: v for k, v in cf.items() if not k.startswith(("O-rand.", "TEAL"))}
    if cand:
        k = max(cand, key=lambda x: (cand[x], x))
        o, _, sched = k.partition("|")
        comp = _comp_rule(cells, rows, docs, settings, o, rules, n_boot)
        choice = {"order": o, "schedule": sched or "base", "cf": cand[k],
                  "comp": "mean" if comp and comp.get("candidate") else "zero", "fraction": cand[k]}
        if comp and floor is not None:
            comp["above_floor"] = bool(comp["english_gain"]["mean"] >= rules["floor_x"] * floor)
            comp["candidate"] = bool(comp["candidate"] and comp["above_floor"])
            choice["comp"] = "mean" if comp["candidate"] else "zero"
        dec["mean_compensation"] = comp
    comp_all = {o: _comp_rule(cells, rows, docs, settings, o, rules, n_boot) for o in base}
    held = {}
    hv = json.loads((out / "held_verdict.json").read_text()) if (out / "held_verdict.json").exists() else {}
    for role in ("choice", "O-fire", "O-rand"):
        name = (hv.get(role) or {}).get("cell")
        c = cells.get(name or "")
        v = dce(name, HELD) if c else None
        if v is not None:
            hg, hs = docs[HELD]["groups"], docs[HELD]["strata"]
            held[role] = {"cell": name, "order": c["order"], "fraction": c["fraction"], "comp": c["comp"],
                          "schedule": c["schedule"], **ST.summary_of(v, hg, hs, n_boot)}
    if held:
        dec["held_verdict"] = {"rows": held, "val_choice": hv.get("choice"),
                               "rule": "a core rebuild (4.6) is justified only by these held numbers"}
        if "choice" in held:
            h = held["choice"]
            dec["after_D2"] = {"cf_held_ub_ok": bool(h["ci95"][1] <= rules["bar"]), "fraction": h["fraction"],
                               "causal_core": bool(h["order"] in CAUSAL and h["fraction"] >= 0.30
                                                   and h["ci95"][1] <= rules["bar"]),
                               "note": "a causal order with CF >= 30% confirmed on held -> re-partition from the "
                                       "causal core before 2 stage 2 and 5's second base"}
    teal = [{"cell": c["cell"], "fraction": c["fraction"], "open_share": c.get("teal_open_share"),
             "english": (c.get("dce") or {}).get(EN), "capitals_acc": c.get("capitals_acc")}
            for c in cells.values() if c.get("order") == "TEAL"]
    summ = {"experiment": "closable fraction vs English cost (docs/research/2026-11-20-amoeba-next-experiments.md 4)",
            "checkpoint": meta.get("checkpoint"), "mlx2_source_revision": meta.get("mlx2_source_revision"),
            "status": meta.get("status"), "truncated": meta.get("truncated"), "rules": rules,
            "checks": meta.get("checks"), "capitals_clean_acc": cap0, "orders": curves, "choice": choice,
            "decisions": dec, "mean_compensation_by_order": comp_all, "teal_reference": teal,
            "existing_partition_points": "4.1 table, plotted as given (not rerun)",
            "labels": "unqualified research (AGENTS.md); val text chooses, held rows 81-160 give the verdict"}
    if write:
        P0.atomic_json(out / "summary.json", summ)
        md = [f"# Closable fraction vs English cost: {meta.get('checkpoint')} at mlx2 "
              f"{meta.get('mlx2_source_revision')}\n", f"Status {meta.get('status')}. Unqualified research. Rules: 4.5.\n",
              f"Decisions: {json.dumps(dec)[:3000]}\n", "| order | CF@0.05 | band | cells (fraction: mean / ub95) |",
              "|---|---|---|---|"]
        for k, v in curves.items():
            md.append(f"| {k} | {v['cf']:.3f}{'+' if v['cf_censored'] else ''} | [{v['band'][0]:.3f}, "
                      f"{v['band'][1]:.3f}] | " + ", ".join(f"{x['fraction']:g}: {x['english_mean']:+.4f} / "
                                                          f"{x['english_ub95']:+.4f}{' (cliff)' if x['flagged'] else ''}"
                                                          for x in v["cells"]) + " |")
        (out / "summary.md").write_text("\n".join(md) + "\n")
    return summ


def _comp_rule(cells, rows, docs, settings, order, rules, n_boot):
    """4.5 mean compensation at comp_fraction for one order: zero vs mean cell, paired per document."""
    f = settings["comp_fraction"]
    z, m = cells.get(f"{order}@{f:g}"), cells.get(f"{order}@{f:g}|mean")
    EN = settings["english"]
    if not z or not m or z.get("na") or m.get("na"):
        return None

    def d(cell, s):
        r = rows.get((cell, "cell", s))
        return None if r is None else ST.doc_dce(r)
    ez, em = d(z["cell"], EN), d(m["cell"], EN)
    if ez is None or em is None:
        return None
    gain = ST.paired(ez, em, docs[EN]["groups"], docs[EN]["strata"], n_boot)
    out = {"english_gain": gain}
    gz, gm = d(z["cell"], "gsm8k"), d(m["cell"], "gsm8k")
    gw = ST.paired(gm, gz, docs["gsm8k"]["groups"], docs["gsm8k"]["strata"], n_boot) if gz is not None and gm is not None \
        else None
    cz, cm = z.get("capitals_correct"), m.get("capitals_correct")
    cw = None
    if cz and cm:
        ids = [str(i) for i in range(len(cz))]
        cw = ST.paired(np.asarray(cz, float), np.asarray(cm, float), ids, None, n_boot)    # zero - mean (harm > 0)
    out.update(gsm8k_mean_minus_zero=gw, capitals_zero_minus_mean=cw)
    gsm_ok = gw is None or gw["ci95"][0] <= 0
    cap_ok = cw is None or cw["ci95"][0] <= 0
    out["candidate"] = bool(gain["mean"] >= rules["comp_min"] and gain["ci95"][0] > 0 and gsm_ok and cap_ok)
    out["note"] = "never a default without Pierre's approval"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
