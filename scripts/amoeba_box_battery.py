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

Usage: amoeba_box_battery.py --model M --partition DIR --held cmn=C.jsonl --held eng=E.jsonl ... --out DIR
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
    a = ap.parse_args()
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
        path, _, filt = rest.partition("?")
        rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        rows = [r for r in rows if r.get("split") == "held"]
        if filt:
            fk, _, fv = filt.partition("=")
            rows = [r for r in rows if str(r.get(fk)) == fv]
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
    missing = [b for b in boxes if b not in held]
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
    if not po.exists():
        po.write_text(json.dumps(score(allones, list(held)), indent=1))
    off = json.loads(po.read_text())
    log("path offset:", off)
    langs = [h for h in held if h in LANGS and h != "eng"]
    others = [h for h in held if h not in LANGS]
    for i, b in enumerate(boxes):
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
        rnd = np.zeros_like(box)
        g = np.random.default_rng(a.seed + 1000 + i)
        for l in range(box.shape[0]):
            k = int(box[l].sum())
            if k:
                cand = np.flatnonzero(~box[l])
                rnd[l, g.choice(cand, min(k, cand.size), replace=False)] = True
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
    rows = []
    for x in bj.read_text().split("\n"):
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
    (out / "battery.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
