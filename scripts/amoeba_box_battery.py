#!/usr/bin/env python3
"""Box battery: test every box of a neuron partition (scripts/amoeba_fn_partition.py) in one model load (Pierre
2026-11-16/17: removing a box must not damage any other topic; boxes must reach +0.05 on held as served). Inside a GPU
window, neuron units, forward only. Per box B (its tile only; pair tiles stay in):
  without      the full model minus B: held dCE on B's own topic and on --unrelated topics (English plus two others
               of another kind), to show B is its topic's and nobody else's;
  random       the same per-layer counts removed at random among all units outside B (seeded): own topic and English;
  core_plus    only the core, B and the pair tiles that contain B kept (every other box and pair removed): held dCE on
               B's own topic (the +0.05 target as served) and on English.
Verdict per box (thresholds recorded): exclusive = own_without >= --min-own and >= --own-x x random_own, and every
unrelated_without <= --max-unrelated; served = core_plus_own <= --target. A path-offset arm (nothing removed) is
scored once. Results go to OUT/battery.jsonl (one row per box, written as it finishes: a rerun skips finished boxes)
and OUT/battery.md.

Held sets: --held NAME=CORPUS[?field=value][@TOKENS] (split "held", the first TOKENS tokens of up to --docs documents of
at least that length, seeded order). Each box needs a held set of the same name; --unrelated-map lists, per box, the
unrelated held set names (default: eng plus two chosen from the other kind: a language for a subject or inference
box, a subject for a language box). Unrelated sets per box: English, one far set of the other kind and one far set
of the same kind (FAR below: e.g. Mandarin -> French, a subject -> cooking / astronomy), never the box's own.

Opt-in arms (--arms, default "boxes" = the battery above, unchanged; Pierre 2026-11-18 "have we been too strict?"):
  unclaimed    the unclaimed units (the partition's --emit-unassigned file, a subset of the core) removed: the full
               model minus them on every held set; the core (group tiles folded back in, i.e. the core without
               family tiles) with and without them on English and every --global-held set (GSM8K); and per box the
               core minus them plus the box and its pair tiles on its held set and English (OUT/unclaimed.jsonl);
  groups       per family tile group:<name> of a v2 partition (scripts/amoeba_fn_partition.py --groups): removed vs
               the same per-layer counts removed at random outside it, on every member's held set and English; and
               per member box, served = core + every group tile containing it + the box + its pair tiles, on its held
               set and English (OUT/groups.jsonl). Verdicts: removal_beats_random = every member's net hurt minus the
               random hurt >= --group-min-hurt; served_+0.05 = every member and English <= --target.
A held spec NAME=gsm8k:PATH[@TOKENS] reads GSM8K rows (question + answer, every row: the test split is never used to
carve). --held-note NAME=TEXT is printed with every row and table that uses NAME (e.g. a held split that is not a
separate held build). --reference-battery OLD/battery.jsonl adds last run's core+box values beside the unclaimed arm.

Usage: amoeba_box_battery.py --model M --partition DIR --held cmn=C.jsonl --held eng=E.jsonl ... --out DIR
       [--arms boxes,unclaimed,groups] [--unclaimed U.npz] [--global-held gsm8k] [--held-note NAME=TEXT]
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
LANGS = {"cmn", "fra", "deu", "spa", "jpn", "rus", "arb", "hin", "swh", "vie", "eng"}
FAR_LANG = {"cmn": "fra", "jpn": "fra"}                  # default far language: cmn (Han-script boxes: French)
FAR_SUBJ = ("life.cooking", "sci.astronomy")            # far subjects (the first that is not the box itself)


ARMS = ("boxes", "unclaimed", "groups")


def random_like(mask, g):
    """The same per-layer counts drawn at random among the units outside `mask` (rng g)."""
    rnd = np.zeros_like(mask)
    for l in range(mask.shape[0]):
        k = int(mask[l].sum())
        if k:
            cand = np.flatnonzero(~mask[l])
            rnd[l, g.choice(cand, min(k, cand.size), replace=False)] = True
    return rnd


def pair_keys(keys, b):
    return [k for k in keys if k.startswith("pair:") and b in k[5:].split("__")]


def group_keys_of(prep, b):
    """The kept family tiles (v2 partition) whose members include box b."""
    kept = (prep.get("groups") or {}).get("kept") or {}
    return [f"group:{g}" for g in sorted(kept) if b in kept[g]["members"]]


def served_keep(get, keys, prep, b, core):
    """Core + every group tile containing b + box b + b's pair tiles."""
    keep = core | get(f"box:{b}")
    for k in group_keys_of(prep, b) + pair_keys(keys, b):
        keep = keep | get(k)
    return keep


def folded_core(get, keys):
    """The core with every group tile folded back in (= the partition's core without family tiles)."""
    core = get("core").copy()
    for k in keys:
        if k.startswith("group:"):
            core |= get(k)
    return core


def read_held_rows(path, kind, filt):
    if kind == "gsm8k":
        if filt:
            raise SystemExit(f"a gsm8k held spec takes no ?field=value filter ({path}?{filt})")
        rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        return [{"text": f"Question: {r['question']}\nAnswer: {r['answer']}"} for r in rows]
    rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    rows = [r for r in rows if r.get("split") == "held"]
    if filt:
        fk, _, fv = filt.partition("=")
        rows = [r for r in rows if str(r.get(fk)) == fv]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held", action="append", required=True)
    ap.add_argument("--boxes", default="", help="comma list (default: every box of the partition)")
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-own", type=float, default=0.10)
    ap.add_argument("--own-x", type=float, default=2.0)
    ap.add_argument("--max-unrelated", type=float, default=0.02)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--arms", default="boxes", help="comma list of " + ", ".join(ARMS))
    ap.add_argument("--unclaimed", default="", help="unclaimed units npz (default PARTITION/unassigned.npz)")
    ap.add_argument("--global-held", default="", help="comma list of held sets for the global unclaimed rows")
    ap.add_argument("--group-min-hurt", type=float, default=0.01)
    ap.add_argument("--held-note", action="append", default=[], help="NAME=TEXT")
    ap.add_argument("--reference-battery", default="")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"--arms: one or more of {ARMS}")
    notes = {}
    for x in a.held_note:
        k, _, v = x.partition("=")
        notes[k] = v
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd = Path(a.partition)
    prep = json.loads((pd / "partition.json").read_text())
    tz = np.load(pd / "tiles.npz")
    layers = [int(x) for x in tz["layers"]]
    keys = [k for k in tz.files if k != "layers"]              # tiles are read lazily (review: ~435 pair masks of
    core_t = tz["core"]                                         # 15.7 MB each would sit beside the model)
    boxes = [b for b in (a.boxes.split(",") if a.boxes else sorted(k[4:] for k in keys if k.startswith("box:"))) if b]
    if "groups" in arms and not (prep.get("groups") or {}).get("kept"):
        raise SystemExit("--arms groups needs a partition with kept family tiles (amoeba_fn_partition.py --groups)")
    upath = Path(a.unclaimed) if a.unclaimed else pd / "unassigned.npz"
    if "unclaimed" in arms:
        if not upath.exists():
            raise SystemExit(f"--arms unclaimed: {upath} missing (amoeba_fn_partition.py --emit-unassigned)")
        uz = np.load(upath)
        if [int(x) for x in uz["layers"]] != layers or uz["unassigned"].shape != core_t.shape:
            raise SystemExit("unclaimed mask of another layout")
        U = uz["unassigned"].astype(bool)
        if (U & ~folded_core(lambda k: tz[k], keys)).any():
            raise SystemExit("unclaimed units outside the core: not this partition's")
    O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb)
    O.setup_mlx(a.memory_gb, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    if host.fingerprint != prep["fingerprint"]:
        raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    W = max(host.widths)
    if core_t.shape != (len(layers), W):
        raise SystemExit(f"tile shape {core_t.shape} != [moe layers, {W}]")
    held, htok = {}, {}
    for spec in a.held:
        name, _, rest = spec.partition("=")
        rest, _, tok = rest.partition("@")
        T = int(tok or 256)
        kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
        path, _, filt = rest.removeprefix("gsm8k:").partition("?")
        rows = read_held_rows(path, kind, filt)
        if kind == "gsm8k":
            notes.setdefault(name, "GSM8K test rows (question + answer) as held text; dCE, not accuracy")
        np.random.default_rng(a.seed).shuffle(rows)
        seqs = []
        for r in rows:
            t = host.encode(r["text"])
            if len(t) >= T:
                seqs.append(t[:T])
            if len(seqs) >= a.docs:
                break
        if not seqs:
            raise SystemExit(f"held {name}: no document of {T} tokens")
        held[name], htok[name] = {"held": seqs}, T
        log(f"held {name}: {len(seqs)} x {T} tokens")
    missing = [b for b in boxes if b not in held] + [x for x in a.global_held.split(",") if x and x not in held]
    if missing or "eng" not in held:
        raise SystemExit(f"held sets missing for {missing + ([] if 'eng' in held else ['eng'])}")
    clean = {}

    def score(keep_rows, names):
        m = np.ones((host.L, W), np.float32)
        m[layers] = keep_rows.astype(np.float32)
        cfg = E.configure(host, D.from_np(m, host), None, host.L)
        r = E.english_cost(host, {k: held[k] for k in names}, clean, cfg, pooled=False)
        return {k: v for k, v in r.items() if k != "mean_d_ce"}

    allones = np.ones_like(core_t)
    settings = {"docs": a.docs, "seed": a.seed, "partition": str(pd), "held": sorted(a.held)}
    done = set()
    bj = out / "battery.jsonl"
    if bj.exists():                         # reuse only complete rows made with these settings
        for x in bj.read_text().split("\n"):
            try:
                r = json.loads(x)
            except ValueError:
                continue
            if r.get("settings") == settings:
                done.add(r["box"])
    po = out / "path_offset.json"
    if po.exists() and set(held) - set(json.loads(po.read_text())):     # a held set added on resume: rescore all
        po.unlink()
    if not po.exists():
        po.write_text(json.dumps(score(allones, list(held)), indent=1))
    off = json.loads(po.read_text())
    log("path offset:", off)
    langs = [h for h in held if h in LANGS and h != "eng"]
    others = [h for h in held if h not in LANGS]
    for i, b in enumerate(boxes if "boxes" in arms else []):
        if b in done:
            continue
        t0 = time.time()
        box = tz[f"box:{b}"]
        if b in LANGS:
            far_other = next((s for s in FAR_SUBJ if s in held), None)
            far_same = FAR_LANG.get(b, "cmn")
        else:
            far_other = "cmn"
            far_same = next((s for s in FAR_SUBJ if s in held and s != b), None)
        unrelated = ["eng"] + [x for x in (far_other, far_same) if x and x in held and x != b]
        without = score(~box, [b] + unrelated)
        rnd = random_like(box, np.random.default_rng(a.seed + 1000 + i))
        random_ = score(~rnd, [b, "eng"])
        keep = core_t | box
        pairs = [k for k in keys if k.startswith("pair:") and b in k[5:].split("__")]
        for k in pairs:
            keep = keep | tz[k]
        core_plus = score(keep, [b, "eng"])
        net = lambda d: {k: v - off.get(k, 0.0) for k, v in d.items()}      # noqa: E731  (path offset removed)
        nw, nr, nc = net(without), net(random_), net(core_plus)
        own, ro = nw[b], nr[b]
        verdict = {"exclusive": bool(own >= a.min_own and own >= a.own_x * max(ro, 1e-9)
                                     and all(nw[u] <= a.max_unrelated for u in unrelated)),
                   "served_+0.05": bool(nc[b] <= a.target), "on": "dCE net of the path offset"}
        row = {"box": b, "units": int(box.sum()), "pairs": pairs, "unrelated": unrelated, "without": without,
               "random": random_, "core_plus": core_plus, "path_offset": off, "verdict": verdict,
               "settings": settings, "wall_s": round(time.time() - t0, 1)}
        with bj.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{b}: units {row['units']:,}; without own {own:+.3f} " +
            " ".join(f"{u} {without[u]:+.3f}" for u in unrelated) +
            f"; random own {ro:+.3f} eng {random_['eng']:+.3f}; core+box own {core_plus[b]:+.3f} eng "
            f"{core_plus['eng']:+.3f}; {verdict}")
    net = lambda d: {k: v - off.get(k, 0.0) for k, v in d.items()}          # noqa: E731
    get = lambda k: tz[k]                                                     # noqa: E731
    asettings = {**settings, "target": a.target, "group_min_hurt": a.group_min_hurt, "global_held": a.global_held,
                 "unclaimed": str(upath)}
    mark = lambda k: f"{k}\u2020" if k in notes else k                       # noqa: E731  (a held-set note below)

    def arm_rows(path):
        got = {}
        if path.exists():
            for x in path.read_text().split("\n"):
                try:
                    r = json.loads(x)
                except ValueError:
                    continue
                if r.get("settings") == asettings:
                    got[r["key"]] = r
        return got

    def put(path, row):
        row["settings"] = asettings
        row["held_notes"] = {k: notes[k] for k in row.get("sets", []) if k in notes}
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    extra_md = []
    if "unclaimed" in arms:
        uj = out / "unclaimed.jsonl"
        got = arm_rows(uj)
        C0 = folded_core(get, keys)
        gl = [x for x in a.global_held.split(",") if x]
        jobs = [("global:full_minus_unclaimed", lambda: ~U, list(held)),
                ("global:core", lambda: C0, ["eng"] + gl),
                ("global:core_minus_unclaimed", lambda: C0 & ~U, ["eng"] + gl)]
        for b in boxes:
            jobs.append((f"box:{b}", lambda b=b: (C0 & ~U) | get(f"box:{b}") |
                         np.logical_or.reduce([get(k) for k in pair_keys(keys, b)] or [np.zeros_like(U)]),
                         [b, "eng"]))
        for key, mk, sets in jobs:
            if key in got:
                continue
            t0 = time.time()
            keep = mk()
            d = score(keep, sets)
            nd = net(d)
            row = {"arm": "unclaimed", "key": key, "sets": sets, "kept_units": int(keep.sum()),
                   "unclaimed_units": int(U.sum()), "dce": d, "net": nd,
                   "served_+0.05": bool(all(v <= a.target for v in nd.values())), "wall_s": round(time.time() - t0, 1)}
            got[key] = put(uj, row)
            log(f"unclaimed {key}: kept {row['kept_units']:,}; " + " ".join(f"{k} {v:+.3f}" for k, v in nd.items()))
        ref = {}
        if a.reference_battery and Path(a.reference_battery).exists():    # last run's core+box, net of its offset
            for x in Path(a.reference_battery).read_text().split("\n"):
                try:
                    r = json.loads(x)
                except ValueError:
                    continue
                rs = r.get("settings") or {}
                if rs.get("docs") == a.docs and rs.get("seed") == a.seed:
                    po_ = r.get("path_offset") or {}
                    ref[r["box"]] = {k: v - po_.get(k, 0.0) for k, v in r["core_plus"].items()}
        extra_md += [f"\nUnclaimed arm: {int(U.sum()):,} unclaimed units removed from the core "
                     f"(core without family tiles {int(C0.sum()):,}); held dCE net of the path offset.\n",
                     "| row | kept units | held dCE | reference core+box (last run) | +0.05 |", "|---|---|---|---|---|"]
        for key, _, _ in jobs:
            r = got[key]
            b = key[4:] if key.startswith("box:") else None
            rf = ref.get(b) if b else None
            extra_md.append(f"| {key} | {r['kept_units']:,} | " + ", ".join(f"{mark(k)} {v:+.3f}" for k, v in r["net"].items())
                            + " | " + (f"{b} {rf[b]:+.3f}, eng {rf['eng']:+.3f}" if rf else "-")
                            + f" | {'yes' if r['served_+0.05'] else 'NO'} |")
    if "groups" in arms:
        gj = out / "groups.jsonl"
        got = arm_rows(gj)
        kept = prep["groups"]["kept"]
        core = tz["core"]
        served = {}
        for k, r in got.items():
            if k.startswith("served:"):
                served[k[7:]] = r
        for gi, gname in enumerate(sorted(kept)):
            mem = kept[gname]["members"]
            have = [m for m in mem if m in held]
            for m in have:
                if m in served:
                    continue
                t0 = time.time()
                keep = served_keep(get, keys, prep, m, core)
                d = score(keep, [m, "eng"])
                nd = net(d)
                served[m] = put(gj, {"arm": "groups", "key": f"served:{m}", "sets": [m, "eng"], "dce": d, "net": nd,
                                     "kept_units": int(keep.sum()), "groups": group_keys_of(prep, m),
                                     "pairs": pair_keys(keys, m), "wall_s": round(time.time() - t0, 1)})
                log(f"served {m}: {nd[m]:+.3f} eng {nd['eng']:+.3f} (groups {served[m]['groups']})")
            key = f"group:{gname}"
            if key in got:
                continue
            t0 = time.time()
            G = tz[key]
            sets = have + ["eng"]
            without = score(~G, sets)
            rnd = random_like(G, np.random.default_rng(a.seed + 5000 + gi))
            random_ = score(~rnd, sets)
            nw, nr = net(without), net(random_)
            hurt = {m: nw[m] - nr[m] for m in have}
            verdict = {"removal_beats_random": bool(have and all(h >= a.group_min_hurt for h in hurt.values())),
                       "served_+0.05": bool(have and all(served[m]["net"][m] <= a.target and
                                                         served[m]["net"]["eng"] <= a.target for m in have)),
                       "members_without_held": [m for m in mem if m not in held], "on": "dCE net of the path offset"}
            row = put(gj, {"arm": "groups", "key": key, "sets": sets, "members": mem, "units": int(G.sum()),
                           "parent": kept[gname].get("parent"), "without": without, "random": random_,
                           "hurt_minus_random": hurt, "served": {m: served[m]["net"] for m in have},
                           "verdict": verdict, "wall_s": round(time.time() - t0, 1)})
            got[key] = row
            log(f"{key}: units {row['units']:,}; " + " ".join(f"{m} {nw[m]:+.3f}/{nr[m]:+.3f}" for m in have)
                + f" eng {nw['eng']:+.3f}/{nr['eng']:+.3f}; {verdict}")
        extra_md += ["\nFamily tiles: removal (group tile removed) vs a random same-count removal, and served = core "
                     "+ every group tile containing the member + its box + its pair tiles; held dCE net of the path "
                     "offset.\n",
                     "| group | units | members: without / random | eng without / random | served member / eng | "
                     "beats random | +0.05 served |", "|---|---|---|---|---|---|---|"]
        for gname in sorted(kept):
            r = got[f"group:{gname}"]
            have = [m for m in r["members"] if m in held]
            nw, nr = net(r["without"]), net(r["random"])
            extra_md.append(f"| {gname} | {r['units']:,} | " + ", ".join(f"{mark(m)} {nw[m]:+.3f} / {nr[m]:+.3f}" for m in have)
                            + f" | {nw['eng']:+.3f} / {nr['eng']:+.3f} | "
                            + ", ".join(f"{mark(m)} {r['served'][m][m]:+.3f} / {r['served'][m]['eng']:+.3f}" for m in have)
                            + f" | {'yes' if r['verdict']['removal_beats_random'] else 'NO'} | "
                              f"{'yes' if r['verdict']['served_+0.05'] else 'NO'} |")
    if notes:
        extra_md += ["\nHeld-set notes (\u2020 in the tables):"] + [f"- {k}: {v}" for k, v in sorted(notes.items())]
    rows = []
    for x in bj.read_text().split("\n") if bj.exists() else []:
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if r.get("settings") == settings:
            rows.append(r)
    meta = {"checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "partition": str(pd), "rule": prep.get("rule"), "docs": a.docs, "held_tokens": htok,
            "held": {k: {"spec": s, "docs_loaded": len(held[k]["held"])} for k, s in
                     ((x.partition("=")[0], x) for x in a.held)}, "path_offset": off,
            "thresholds": {"min_own": a.min_own, "own_x": a.own_x, "max_unrelated": a.max_unrelated, "target": a.target},
            "watchdog": {"swap_growth_gb": a.swap_growth_gb, "waiver_note": a.waiver_note or None},
            "mlx_peak_gb": O.peak_gb()}
    if arms != ["boxes"] or notes:
        meta.update(arms=arms, held_notes=notes, arm_thresholds={"target": a.target, "group_min_hurt":
                                                                  a.group_min_hurt},
                    unclaimed=str(upath) if "unclaimed" in arms else None)
    (out / "battery.meta.json").write_text(json.dumps(meta, indent=1))
    lines = [f"Battery on {host.checkpoint}, partition {pd.name} ({prep.get('rule')}); held dCE, {a.docs} docs per set.\n",
             "| box | units | own without | unrelated without | random own / eng | core+box own / eng | exclusive | +0.05 served |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['box']} | {r['units']:,} | {r['without'][r['box']]:+.3f} | "
                     + ", ".join(f"{u} {r['without'][u]:+.3f}" for u in r["unrelated"])
                     + f" | {r['random'][r['box']]:+.3f} / {r['random']['eng']:+.3f} | {r['core_plus'][r['box']]:+.3f} / "
                       f"{r['core_plus']['eng']:+.3f} | {'yes' if r['verdict']['exclusive'] else 'NO'} | "
                       f"{'yes' if r['verdict']['served_+0.05'] else 'NO'} |")
    if "boxes" not in arms:
        lines = lines[:1]
    lines += extra_md
    (out / "battery.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
