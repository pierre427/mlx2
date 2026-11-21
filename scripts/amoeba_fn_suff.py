#!/usr/bin/env python3
"""Core-without-box control for an FN neuron partition: what each box adds when served (P0, Pierre 2026-11-20 "Yes"
to queueing both P0 experiments; docs/research/2026-11-20-amoeba-next-experiments.md 1, with the 7.1-7.5 fixes).
One model load, forward only, MoE neuron units, inside a GPU window. Standalone beside the box battery: the battery
(scripts/amoeba_box_battery.py) is imported read-only for its tiles, served compositions and held reader, and its
finished run (--battery DIR: battery.meta.json + battery.jsonl) supplies the held specs, notes, skipped boxes, box
list and the necessity numbers. Nothing here changes what the battery does.

Per box B, scored on B's held set (own) and on --english, per document (OUT/perdoc.jsonl, 7.1):
  core_plus             core + B's served composition (the battery's core_plus: family tiles, ancestors, B's box,
                        pairs and subtree; the English box for an English-text box), the served bar
  core_alone            core + B's family tiles and ancestors, without B's own box, pair and subtree tiles
                        (+ the English box when core_plus has it): sufficiency of the core
  core_plus_rand        core_alone + a random set of the same per-layer size as B's own tiles (core_plus minus
                        core_alone), drawn from the units core_plus closes (seed 1; seed 2 on the subset)
  core_plus_rand_inexp  core_alone + a random set of the same per-(layer, expert) counts as B's own tiles, drawn
                        inside those experts among the units core_plus closes (7.5). Those experts' other units are
                        mostly core (already open), so this control is often far from size-matched: its fill is
                        recorded and below 90% it is reported but kept out of the 2x ratio (SUFF_RULES)
Subset only (--subset, default auto = the 10 languages, the 9 topics other than sci.physics with the largest
necessity_net in --battery, and sci.physics; 1.4):
  core_plus_rand2, core_only (core + the English box when B's composition has it; no family or ancestor tiles),
  without / random (the battery's removal of B's box and its layer-matched random removal, rescored per document on
  own), random_inexp (7.5 on the necessity side: B's per-(layer, expert) counts removed inside its experts), and under mean ablation (7.3: a closed neuron takes its conditional mean mu instead of 0; research/amoeba/
  carve/units.py Control.moe_mean): core_alone_mean, core_plus_mean (+ English), without_mean, random_mean.
Repeat (--repeat-boxes, default auto = the 5 languages and 5 topics of the subset with the largest necessity_net):
  core_alone and core_plus on held rows 81-160 of the same seeded order (own#rep) and on English held rows 161-240
  (english#rep; rows 81-160 are the Pareto window's final held verdict, 7.7), the document-sampling floor (7.6).
Controls (7.2): the all-open masked path on English and the first box (must equal the unmasked model bit for bit);
the do-nothing arm (core_plus rebuilt as core_alone + own tiles + an empty extra set; bitwise = core_plus); the
empty-set mean arm (all open, mean fill on; bitwise = unmasked); the repeat cell (the first scored arm scored again
at the end; bitwise). mu: research/amoeba/carve/moestats.capture over the fit split of --english and the 10
languages (--mu-docs each, the 1.3 default reference distribution; recorded).

Decision rules (1.5, pre-registered; scripts/amoeba_battery_stats.py SUFF_RULES) are applied at the end:
OUT/summary.json and OUT/summary.md (also rerunnable on the CPU: amoeba_battery_stats.py suff --dir OUT). Rows per
box: OUT/suff.jsonl; mask receipts (fills), written before a box is scored: OUT/suff_info.jsonl. Work order (each
phase resumable from the sidecar): controls, subset boxes, mu + mean arms,
repeat, the other boxes, the repeat cell. --budget-min stops new scoring (rows so far stay; the summary records it).

Usage: amoeba_fn_suff.py --model M --partition DIR --battery DIR --out DIR [--english english] [--assert-ple]
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

BASE_ARMS = ("core_plus", "core_alone", "core_plus_rand", "core_plus_rand_inexp")
SUBSET_ARMS = ("core_plus_rand2", "core_only", "without", "random", "random_inexp")
MEAN_ARMS = ("core_alone_mean", "core_plus_mean", "without_mean", "random_mean")
REP = "#rep"


def reference_rows(bdir, held_specs=None):
    """The finished battery's rows in run order, with its settings (docs, seed). A battery dir scored more than
    once holds rows of several settings: the reference is the latest run's (the last row's settings, like the
    battery's own resume and coreassign), restricted to rows on held_specs (battery.meta.json's held specs, the
    documents this run scores) when given."""
    rows = P0.read_jsonl(Path(bdir) / "battery.jsonl")
    if held_specs is not None:
        want = sorted(held_specs)
        rows = [r for r in rows if (r.get("settings") or {}).get("held") == want]
    if not rows:
        raise SystemExit(f"{bdir}: no battery.jsonl rows" + ("" if held_specs is None else
                         " on the held specs of battery.meta.json"))
    s0 = rows[-1].get("settings")
    rows = [r for r in rows if r.get("settings") == s0]
    seen, out = set(), []
    for r in rows:
        if r["box"] not in seen:
            seen.add(r["box"])
            out.append(r)
    return out, s0


def necessity(r):
    """without_own - random_own of a battery row (the path offset cancels)."""
    b = r["box"]
    return float(r["without"][b] - r["random"][b])


def auto_subset(rows, langs=BAT.LANG10):
    nec = {r["box"]: necessity(r) for r in rows}
    have = [b for b in sorted(langs) if b in nec]
    topics = sorted((b for b in nec if b not in langs and b != "sci.physics"), key=lambda b: (-nec[b], b))[:9]
    return have + topics + (["sci.physics"] if "sci.physics" in nec else [])


def auto_repeat(subset, rows, langs=BAT.LANG10):
    nec = {r["box"]: necessity(r) for r in rows}
    ls = sorted((b for b in subset if b in langs), key=lambda b: (-nec.get(b, 0), b))[:5]
    ts = sorted((b for b in subset if b not in langs), key=lambda b: (-nec.get(b, 0), b))[:5]
    return ls + ts


class Comp:
    """The compositions of one partition (masks over its MoE layers, True = open)."""

    def __init__(self, tz, prep):
        self.tz, self.prep, self.keys = tz, prep, list(tz.keys)
        self.get = lambda k: tz[k]                                                  # noqa: E731
        self.core = tz["core"]
        self.ebox, self.eclosed = BAT.english_box(prep)

    def english_open(self, b):
        return bool(self.ebox) and b not in self.eclosed and b != self.ebox

    def core_plus(self, b):
        return BAT.served_keep(self.get, self.keys, self.prep, b, self.core)

    def core_alone(self, b):
        keep = self.core.copy()
        if self.english_open(b):
            keep |= BAT.english_tiles(self.get, self.keys, self.prep)
        for x in BAT.ancestors_of(self.prep, b):
            keep |= self.get(f"box:{x}")
        for k in BAT.group_keys_of(self.prep, b):
            keep |= self.get(k)
        return keep

    def core_only(self, b):
        keep = self.core.copy()
        if self.english_open(b):
            keep |= BAT.english_tiles(self.get, self.keys, self.prep)
        return keep


def build_arms(comp, b, i_battery, seed, F, subset):
    """{arm: (keep mask, info)} of box b (zero ablation; the mean arms reuse core_alone / core_plus / without /
    random). i_battery: b's index in the battery run (its random removal's seed, BAT: seed + 1000 + i)."""
    cp, ca = comp.core_plus(b), comp.core_alone(b)
    if (ca & ~cp).any():
        raise SystemExit(f"{b}: core_alone is not inside core_plus")
    own = cp & ~ca
    closed = ~cp
    r1, s1 = P0.random_like(own, np.random.default_rng(seed + 31000 + i_battery), cand=closed)
    ri, si = P0.random_in_experts(own, closed, F, np.random.default_rng(seed + 32000 + i_battery))
    arms = {"core_plus": cp, "core_alone": ca, "core_plus_rand": ca | r1, "core_plus_rand_inexp": ca | ri}
    info = {"own_tile_units": int(own.sum()), "closed_by_core_plus": int(closed.sum()),
            "rand_shortfall": s1, "rand_inexp_shortfall": si,
            "kept_units": {k: int(v.sum()) for k, v in arms.items()}}
    if subset:
        r2, s2 = P0.random_like(own, np.random.default_rng(seed + 33000 + i_battery), cand=closed)
        box = comp.get(f"box:{b}")
        rnd = BAT.random_like(box, np.random.default_rng(seed + 1000 + i_battery))     # the battery's own draw
        # 7.5 on the necessity side: the box's per-(layer, expert) counts removed among those experts' other units
        rie, sie = P0.random_in_experts(box, ~box, F, np.random.default_rng(seed + 34000 + i_battery))
        arms.update(core_plus_rand2=ca | r2, core_only=comp.core_only(b), without=~box, random=~rnd,
                    random_inexp=~rie)
        info.update(rand2_shortfall=s2, random_inexp_shortfall=sie, box_units=int(box.sum()))
        info["kept_units"].update({k: int(arms[k].sum()) for k in SUBSET_ARMS})
    return arms, info


def _wired_arg(spec):
    from research.amoeba.carve.ops import wired_arg
    return wired_arg(spec)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--battery", required=True, help="the finished battery run on this partition (meta + rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--english", default="english")
    ap.add_argument("--boxes", default="", help="comma list (default: every box the battery scored)")
    ap.add_argument("--subset", default="auto")
    ap.add_argument("--repeat-boxes", default="auto")
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-docs", type=int, default=24, help="fit documents per set for mu (English + 10 languages)")
    ap.add_argument("--mu-sets", default="english," + ",".join(sorted(BAT.LANG10)))
    ap.add_argument("--no-mean", action="store_true", help="skip mu and the mean-ablation arms")
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
    pd = Path(a.partition)
    prep = json.loads((pd / "partition.json").read_text())
    tz = BAT.Tiles(pd / "tiles.npz")
    layers = tz.layers
    bdir = Path(a.battery)
    bmeta = json.loads((bdir / "battery.meta.json").read_text())
    if bmeta.get("partition") and Path(bmeta["partition"]).resolve() != pd.resolve():
        raise SystemExit(f"--battery ran on {bmeta['partition']}, not {pd}")
    if bmeta.get("fingerprint") != prep.get("fingerprint"):
        raise SystemExit("--battery is of another model")
    ref, rset = reference_rows(bdir, [v["spec"] for v in bmeta["held"].values()])
    if rset.get("docs") != a.docs or rset.get("seed") != a.seed:
        raise SystemExit(f"--battery rows were made with docs {rset.get('docs')} seed {rset.get('seed')}")
    order = [r["box"] for r in ref]
    refrow = {r["box"]: r for r in ref}
    boxes = [b for b in (a.boxes.split(",") if a.boxes else order) if b]
    missing = [b for b in boxes if b not in refrow]
    if missing:
        raise SystemExit(f"boxes without a battery row: {missing}")
    subset = auto_subset(ref) if a.subset == "auto" else [b for b in a.subset.split(",") if b]
    subset = [b for b in subset if b in boxes]
    rep = auto_repeat(subset, ref) if a.repeat_boxes == "auto" else [b for b in a.repeat_boxes.split(",") if b]
    rep = [b for b in rep if b in boxes]
    specs = {k: v["spec"] for k, v in bmeta["held"].items()}
    EN = a.english
    need = [EN] + boxes
    nospec = [s for s in need if s not in specs]
    if nospec:
        raise SystemExit(f"--battery has no held spec for {nospec}")
    mu_sets = [s for s in a.mu_sets.split(",") if s]
    settings = {"partition": str(pd), "battery": str(bdir), "docs": a.docs, "seed": a.seed, "english": EN,
                "boxes": boxes, "subset": subset, "repeat_boxes": rep, "mu_docs": a.mu_docs, "mu_sets": mu_sets,
                "mean": not a.no_mean}
    sp = out / "settings.json"
    if sp.exists() and json.loads(sp.read_text()) != settings:
        raise SystemExit(f"{out} holds a run with other settings (settings.json): use a fresh --out")
    P0.atomic_json(sp, settings)
    from research.amoeba.carve import ops as O
    O.prepare_wiring(a.wired_gb)
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
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
    checks = {}
    prev = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    attempts = list(prev.get("attempts") or []) + [{"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                    "mlx2_source_revision": O.git_rev()}]

    def write_meta(status, **extra):
        """meta.json from the model load on (a killed run still names its checkpoint and revision)."""
        P0.atomic_json(out / "meta.json", {
            "checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "settings": settings, "checks": checks, "status": status, "attempts": attempts, "wired": wired,
            "held_notes": {k: v for k, v in (bmeta.get("held_notes") or {}).items() if k in need}, **extra})
    write_meta("running: loading held sets")
    W = max(host.widths)
    if tz.shape != (len(layers), W):
        raise SystemExit(f"tile shape {tz.shape} != [moe layers, {W}]")
    nsite = {s.layer: s for s in MS.neuron_sites(host)}
    if sorted(nsite) != sorted(layers):
        raise SystemExit(f"partition layers {layers} are not the host's neuron MoE layers {sorted(nsite)}")
    F = nsite[layers[0]].F
    sc = P0.Scorer(host, E, D, layers, W, out, log, None if a.budget_min is None else a.budget_min * 60)
    for s in need:
        seqs, rows = P0.load_docs(host, specs[s], 0, a.docs, a.seed)
        sc.add_set(s, seqs, rows, specs[s])
    rep_dropped = {}
    for s in [EN] + rep:      # English: rows [160, 240), so the Pareto window's final held rows [80, 160) stay unspent
        lo = 2 * a.docs if s == EN else a.docs
        try:
            seqs, rows = P0.load_docs(host, specs[s], lo, lo + a.docs, a.seed)
        except SystemExit as e:                     # a held set without 2 x docs long documents: no repeat for it
            if s == EN:
                raise
            rep_dropped[s] = str(e)
            log(f"repeat {s} dropped: {e}")
            continue
        sc.add_set(s + REP, seqs, rows, specs[s] + f" rows [{lo}, {lo + a.docs})")
    rep = [b for b in rep if b not in rep_dropped]
    log(f"held: {len(sc.held)} sets x {a.docs} docs; boxes {len(boxes)}, subset {len(subset)}, repeat {len(rep)}")
    comp = Comp(tz, prep)
    allopen = np.ones(tz.shape, bool)
    rows_path = out / "suff.jsonl"
    finished = {r["box"]: r for r in P0.read_jsonl(rows_path)}
    truncated = None
    t_start = time.time()

    info_path = out / "suff_info.jsonl"
    info_done = {r["box"] for r in P0.read_jsonl(info_path)}

    def box_job(b, mean=None, mean_only=False):
        i = order.index(b)
        arms, info = build_arms(comp, b, i, a.seed, F, b in subset)
        if b not in info_done:                     # mask receipts (fills) before any of the box's scoring
            info_done.add(b)
            P0.append_jsonl(info_path, {"box": b, **info})
        vals = {}
        if not mean_only:
            for arm in BASE_ARMS + (SUBSET_ARMS if b in subset else ()):
                sets = [b] if arm in ("without", "random", "random_inexp") else [b, EN]
                vals[arm] = sc.score(b, arm, arms[arm], sets)
        if mean is not None:
            for arm in MEAN_ARMS:
                base = arm[:-5]
                sets = [b, EN] if arm == "core_plus_mean" else [b]
                vals[arm] = sc.score(b, arm, arms[base], sets, mean=mean)
        return arms, info, vals

    first = None
    mu_info = None
    try:
        # controls: the all-open masked path (= unmasked, bitwise) on English and the first box
        b0 = (subset or boxes)[0]
        sc.score("_control", "all_open", allopen, [EN, b0])
        for s in (EN, b0):
            r = sc.done[("_control", "all_open", s)]
            checks.setdefault("path_offset_exact", {})[s] = bool(r["ce"] == r["clean_ce"])
        first = (b0, "core_plus", b0)
        for b in subset:
            arms, info, vals = box_job(b)
            if b == subset[0]:                     # do-nothing: core_alone + own tiles + an empty extra set
                own = arms["core_plus"] & ~arms["core_alone"]
                sc.score(b, "core_plus_empty", arms["core_alone"] | own | np.zeros_like(own), [b, EN])
                checks["do_nothing_bitwise"] = {s: sc.same((b, "core_plus"), (b, "core_plus_empty"), s)
                                                for s in (b, EN)}
            if b not in finished:
                finished[b] = P0.append_jsonl(rows_path, {"box": b, "subset": True, **info, "dce": vals})
            log(f"{b}: " + " ".join(f"{k} {v[b]:+.4f}" for k, v in vals.items()))
        write_meta("running: subset done", repeat_boxes_run=rep, repeat_dropped=rep_dropped)
        if not a.no_mean and subset:
            mp = out / "mu.npz"
            if mp.exists():
                z = np.load(mp)
                mu = {l: z[f"mu_{l}"] for l in layers}
                mu_info = json.loads((out / "mu.json").read_text())
            else:
                t0 = time.time()
                seqs = []
                src = {}
                for s in mu_sets:
                    if s not in specs:
                        raise SystemExit(f"--mu-sets {s}: no held spec in --battery")
                    q, _ = P0.load_docs(host, P0.with_split(specs[s], "fit"), 0, a.mu_docs, a.seed)
                    seqs += q
                    src[s] = len(q)
                st = MS.capture(host, seqs, log=log)
                mu = {l: st[l]["mu"] for l in layers}
                np.savez(out / "mu.tmp.npz", **{f"mu_{l}": mu[l] for l in layers})
                (out / "mu.tmp.npz").replace(mp)
                mu_info = {"source": "fit split", "sets": src, "seed": a.seed, "tokens": st[layers[0]]["tokens"],
                           "s": round(time.time() - t0, 1),
                           "definition": "conditional mean of h[e, f] over the slots routed to e (moestats.capture)"}
                P0.atomic_json(out / "mu.json", mu_info)
            sc.score("_control", "all_open_mean", allopen, [EN], mean=mu)
            r = sc.done[("_control", "all_open_mean", EN)]
            checks["empty_mean_bitwise"] = bool(r["ce"] == r["clean_ce"])
            for b in subset:
                _, _, vals = box_job(b, mean=mu, mean_only=True)
                log(f"{b} mean: " + " ".join(f"{k} {v[b]:+.4f}" for k, v in vals.items()))
        write_meta("running: mean arms done", repeat_boxes_run=rep, repeat_dropped=rep_dropped, mu=mu_info)
        for b in rep:
            arms = build_arms(comp, b, order.index(b), a.seed, F, b in subset)[0]
            for arm in ("core_alone", "core_plus"):
                sc.score(b, arm, arms[arm], [b + REP, EN + REP])
        for b in boxes:
            if b in subset:
                continue
            arms, info, vals = box_job(b)
            if b not in finished:
                finished[b] = P0.append_jsonl(rows_path, {"box": b, "subset": False, **info, "dce": vals})
            log(f"{b}: " + " ".join(f"{k} {v[b]:+.4f}" for k, v in vals.items()))
        b, arm, s = first
        arms, _ = build_arms(comp, b, order.index(b), a.seed, F, b in subset)
        sc.score(b, arm + "@repeat", arms[arm], [s], reuse=False)
        checks["repeat_cell_bitwise"] = sc.same((b, arm), (b, arm + "@repeat"), s)
    except P0.Budget:
        truncated = f"--budget-min {a.budget_min}: no new scoring after {round((time.time() - t_start) / 60, 1)} min"
        log(truncated)
    status = "finished" if not truncated else "truncated"
    write_meta(status, truncated=truncated, mu=mu_info, repeat_boxes_run=rep, repeat_dropped=rep_dropped,
               scorings=sc.n, scoring_s=round(sc.t, 1), sec_per_scoring=round(sc.t / max(1, sc.n), 2),
               watchdog={"limit_gb": a.watchdog_gb, "swap_growth_gb": a.swap_growth_gb,
                         "waiver_note": a.waiver_note or None,
                         **{k: v for k, v in getattr(O, "WATCH", {}).items() if k.startswith(("peak_", "observed_"))}},
               mlx_peak_gb=O.peak_gb())
    meta = {"status": status}
    ST.to_npz(out / "perdoc.jsonl", out / "battery.perdoc.npz")
    summ = summarize(out, str(bdir / "battery.jsonl"))
    P0.atomic_json(out / "status.json", {"status": meta["status"]})
    log("summary:", json.dumps(summ["counts"]))
    return 0


# ----------------------------------------------------------------------------- summary (CPU)

def summarize(out, reference_battery=None, rules=ST.SUFF_RULES, n_boot=ST.N_BOOT):
    """Apply the 1.5 rules to OUT's per-document rows; writes OUT/summary.json and OUT/summary.md."""
    out = Path(out)
    rows = ST.load_perdoc(out / "perdoc.jsonl")
    docs = json.loads((out / "perdoc.docs.json").read_text())
    settings = json.loads((out / "settings.json").read_text())
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    EN = settings["english"]
    ref = {}
    if reference_battery and Path(reference_battery).exists():
        # the same rows as reference_rows: on the held specs of battery.meta.json, the latest row of each box
        try:
            bm = json.loads((Path(reference_battery).parent / "battery.meta.json").read_text())
            want = sorted(x["spec"] for x in bm["held"].values())
        except (OSError, ValueError, KeyError, TypeError):
            want = None
        for r in P0.read_jsonl(reference_battery):
            rs = r.get("settings") or {}
            if rs.get("docs") == settings["docs"] and rs.get("seed") == settings["seed"] and \
                    (want is None or rs.get("held") == want):
                ref[r["box"]] = r

    def v(b, arm, s):
        r = rows.get((b, arm, s))
        return None if r is None else ST.doc_dce(r)
    info = {r["box"]: r for r in P0.read_jsonl(out / "suff_info.jsonl")}
    boxes = {}
    for b in settings["boxes"]:
        own = {arm: v(b, arm, b) for arm in BASE_ARMS}
        if any(x is None for x in own.values()):
            continue
        g, st = docs[b]["groups"], docs[b]["strata"]
        nec, nec_src = None, None
        wo, rn = v(b, "without", b), v(b, "random", b)
        if wo is not None and rn is not None:
            nec_pd = ST.paired(wo, rn, g, st, n_boot)
            nec, nec_src = nec_pd["mean"], "per-document rescore (this run)"
        elif b in ref:
            nec, nec_src = necessity(ref[b]), "reference battery means"
        mean_own = None
        mnec = None
        if all(v(b, arm, b) is not None for arm in MEAN_ARMS):
            mean_own = {"core_plus": v(b, "core_plus_mean", b), "core_alone": v(b, "core_alone_mean", b)}
            mnec = float(np.mean(v(b, "without_mean", b) - v(b, "random_mean", b)))
        inf = info.get(b) or {}
        ou = inf.get("own_tile_units") or 0
        fill = {k: (1 - inf[sk] / ou) if ou and sk in inf else (None if ou else 0.0)
                for k, sk in (("core_plus_rand", "rand_shortfall"), ("core_plus_rand_inexp", "rand_inexp_shortfall"))}
        if not inf:                                # no mask receipt: the in-expert control is never taken as matched
            fill = {"core_plus_rand": None, "core_plus_rand_inexp": 0.0}
        row = ST.suff_box(own, rules, nec, g, st, n_boot, mean_own=mean_own, mean_necessity_net=mnec, fill=fill)
        row["necessity_source"] = nec_src
        ri = v(b, "random_inexp", b)
        if wo is not None and ri is not None:
            bu = inf.get("box_units") or 0
            row["necessity_net_inexp"] = {**ST.paired(wo, ri, g, st, n_boot),
                                          "fill": (1 - inf.get("random_inexp_shortfall", 0) / bu) if bu else None,
                                          "note": "without - a random removal of the box's per-expert counts "
                                                  "inside its experts (7.5): expert choice vs neuron choice"}
        row["kind"] = "language" if b in BAT.LANG10 else "topic"
        row["subset"] = b in settings["subset"]
        en = {arm: v(b, arm, EN) for arm in BASE_ARMS}
        if all(x is not None for x in en.values()):
            row["english"] = {"core_plus": float(en["core_plus"].mean()), "core_alone": float(en["core_alone"].mean()),
                              "suff_gain": ST.paired(en["core_alone"], en["core_plus"], docs[EN]["groups"],
                                                     docs[EN]["strata"], n_boot)}
        if v(b, "core_plus_rand2", b) is not None:
            row["rand_gain_seed2"] = ST.paired(own["core_alone"], v(b, "core_plus_rand2", b), g, st, n_boot)
        if v(b, "core_only", b) is not None:
            row["core_only_own"] = float(v(b, "core_only", b).mean())
        if b in ref:
            rb = ref[b]
            off = (rb.get("path_offset") or {}).get(b, 0.0)
            row["reference_core_plus_own"] = float(rb["core_plus"][b] - off)
            row["reproduces_battery_core_plus"] = abs(row["core_plus_own"] - row["reference_core_plus_own"])
        if (b, "core_plus", b + REP) in rows and (b, "core_alone", b + REP) in rows:
            gr, sr = docs[b + REP]["groups"], docs[b + REP]["strata"]
            row["repeat"] = {"suff_gain": ST.paired(v(b, "core_alone", b + REP), v(b, "core_plus", b + REP), gr, sr,
                                                    n_boot),
                             "core_plus_own": float(v(b, "core_plus", b + REP).mean()),
                             "core_alone_own": float(v(b, "core_alone", b + REP).mean())}
        boxes[b] = row
    # floors (7.6): document resampling (repeat rows) and the random-reopening seed
    rep_d = [r["repeat"]["suff_gain"]["mean"] - r["suff_gain"]["mean"] for r in boxes.values() if "repeat" in r]
    seed_d = [r["rand_gain_seed2"]["mean"] - r["rand_gain"]["mean"] for r in boxes.values()
              if "rand_gain_seed2" in r and "rand_gain" in r]
    floor = {"doc_resample_sd": float(np.std(rep_d, ddof=1) / np.sqrt(2)) if len(rep_d) > 1 else None,
             "doc_resample_n": len(rep_d),
             "random_seed_sd": float(np.std(seed_d, ddof=1) / np.sqrt(2)) if len(seed_d) > 1 else None,
             "random_seed_n": len(seed_d),
             "note": "sd of (repeat - original) / sqrt 2 = one draw's sampling sd; the seed floor likewise"}
    fsd = floor["doc_resample_sd"]
    for r in boxes.values():                                 # 7.6: no finding under floor_x x the measured floor
        r["below_floor"] = (bool(r["useful"] and r["suff_gain"]["mean"] < rules["floor_x"] * fsd)
                            if fsd is not None else None)
    topics = {b: r for b, r in boxes.items() if r["kind"] == "topic"}
    counts = {"boxes": len(boxes), "topics": len(topics), "languages": len(boxes) - len(topics)}
    for kind, grp in (("topic", topics), ("language", {b: r for b, r in boxes.items() if r["kind"] == "language"})):
        counts[kind] = {"served": {x: sum(1 for r in grp.values() if r["served"] == x) for x in ("yes", "vacuous", "no")},
                        "useful": sum(1 for r in grp.values() if r["useful"]),
                        "four_way": {x: sum(1 for r in grp.values() if r["four_way"] == x)
                                     for x in ("real_box", "redundant_or", "needed_core_plus_fails", "cosmetic")},
                        "ablation_sensitive": sorted(b for b, r in grp.items() if r.get("ablation_sensitive"))}
    passes = [r for r in topics.values() if r["served"] in ("yes", "vacuous")]
    vac = sum(1 for r in passes if r["served"] == "vacuous")
    share = vac / len(passes) if passes else None
    ck = meta.get("checks") or {}
    failed = [k for k, x in ck.items() if x is False or (isinstance(x, dict) and not all(x.values()))]
    missing = [k for k in ("path_offset_exact", "do_nothing_bitwise", "repeat_cell_bitwise") if k not in ck]
    decisions = {
        "checks_ok": not failed, "checks_failed": failed, "checks_missing": missing,
        "checks_note": "a failed bitwise control means every number here carries that difference as a floor: read "
                       "the decisions only after resolving it",
        "topic_served_passes": len(passes), "topic_vacuous": vac, "topic_vacuous_share": share,
        "after_D1": (None if share is None else
                     "move 4 (Pareto) ahead of 2 for the night; treat 5 as the only topic-gating route "
                     "(>= 80% of topic served passes are vacuous)" if share >= 0.8 else
                     "keep the plan order (< 80% of topic served passes are vacuous)"),
        "expected_false_positive_useful_topics": ST.expected_false_positives(len(topics), fsd),
        "useful_topics": sorted(b for b, r in topics.items() if r["useful"]),
        "useful_topics_above_floor": sorted(b for b, r in topics.items() if r["useful"] and not r["below_floor"]),
        "counting_rule": "a rule triggers for a box only on its own CI; across boxes, the expected false positives "
                         "at the measured document-resampling floor are reported instead of claiming marginal boxes",
    }
    summ = {"experiment": "core-without-box control (docs/research/2026-11-20-amoeba-next-experiments.md 1)",
            "checkpoint": meta.get("checkpoint"), "mlx2_source_revision": meta.get("mlx2_source_revision"),
            "status": meta.get("status"), "truncated": meta.get("truncated"), "rules": rules,
            "checks": meta.get("checks"), "floor": floor, "counts": counts, "decisions": decisions,
            "mu": meta.get("mu"), "boxes": boxes,
            "labels": "unqualified research (AGENTS.md); per-document dCE, group-bootstrap 95% CI, 2,000 resamples"}
    P0.atomic_json(out / "summary.json", summ)
    md = [f"# Core-without-box control: {meta.get('checkpoint')} at mlx2 {meta.get('mlx2_source_revision')}\n",
          f"Status {meta.get('status')}; {len(boxes)} boxes scored. Unqualified research. Rules: 1.5 (pre-registered).\n",
          f"Topic served passes vacuous: {vac} of {len(passes)}" + (f" ({100 * share:.0f}%)" if share is not None else "")
          + f". {decisions['after_D1']}\n", f"Checks: {json.dumps(meta.get('checks'))}\n",
          f"Floor: {json.dumps(floor)}\n",
          "| box | kind | core_plus own | core_alone own | suff_gain [CI] | rand_gain / inexp | necessity_net | served "
          "| useful | four-way | mean four-way |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in boxes.items():
        sg = r["suff_gain"]
        md.append(f"| {b} | {r['kind']} | {r['core_plus_own']:+.4f} | {r['core_alone_own']:+.4f} | "
                  f"{sg['mean']:+.4f} [{sg['ci95'][0]:+.4f}, {sg['ci95'][1]:+.4f}] | "
                  f"{r.get('rand_gain', {}).get('mean', float('nan')):+.4f} / "
                  f"{r.get('rand_gain_inexp', {}).get('mean', float('nan')):+.4f} | "
                  + (f"{r['necessity_net']:+.4f}" if r["necessity_net"] is not None else "-")
                  + f" | {r['served']} | {'yes' if r['useful'] else 'no'} | {r['four_way'] or '-'} | "
                  f"{(r.get('mean_ablation') or {}).get('four_way') or '-'} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return summ


if __name__ == "__main__":
    raise SystemExit(main())
