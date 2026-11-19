#!/usr/bin/env python3
"""Exclusive-neuron box test (Pierre 2026-11-16): a box takes only the neurons its topic uses exclusively; removing the
box from the model (core = every other neuron) must not damage any other topic, while its own topic degrades. Core +
box = the full model, so "box present" is not the test. Inside a GPU window, neuron units, forward only.

From a firing map (scripts/amoeba_firing_map.py: map.npz with target, mean_target, mean_baseline per neuron), the
candidate box at exclusivity r is: target neurons whose baseline mean contribution is at most r x their target mean
(r = inf: every target neuron). For each --ratios r:
  without box   held dCE (model with the box's neurons zeroed vs the full model) on every --held language;
  random        the same per-layer counts zeroed at random among the non-target neurons (seeded), on --random-langs.
Held: each language's lang-prose corpus split "held" (never carved on or mapped), the first --tokens tokens of
--docs documents. Writes OUT/box_r<r>.npz (the box mask) and OUT/report.json, prints a table.

Usage: amoeba_exclusive_box_test.py --model M --map DIR --out DIR --held cmn=CORPUS --held eng=CORPUS [...]
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
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held", action="append", required=True, help="LANG=corpus.jsonl (split held is used)")
    ap.add_argument("--target-lang", default="cmn")
    ap.add_argument("--ratios", default="0.05,0.1,0.25,inf")
    ap.add_argument("--random-langs", default="cmn,eng")
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--swap-waiver-note", default="")
    a = ap.parse_args()
    import mlx.core as mx
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    z = np.load(Path(a.map) / "map.npz")
    rep_map = json.loads((Path(a.map) / "report.json").read_text())
    tgt, mt, mb = z["target"], z["mean_target"].astype(np.float64), z["mean_baseline"].astype(np.float64)
    O.footprint_watchdog(92, log, swap_growth_gb=a.swap_growth_gb)
    O.setup_mlx(88, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    import os
    if "Flash-Next" in str(a.model):
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    if host.fingerprint != rep_map.get("fingerprint"):
        raise SystemExit(f"map was made on {rep_map.get('fingerprint')}, this model is {host.fingerprint}")
    layers = [int(x) for x in z["moe_layers"]]
    W = max(host.widths)
    if tgt.shape != (len(layers), W):
        raise SystemExit(f"map shape {tgt.shape} does not match [moe layers, {W}]")
    held = {}
    for spec in a.held:
        lang, _, p = spec.partition("=")
        rows = [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]
        rows = [r for r in rows if r.get("split") == "held"]
        rng = np.random.default_rng(a.seed)
        rng.shuffle(rows)
        seqs = []
        for r in rows:
            t = host.encode(r["text"])
            if len(t) >= a.tokens:
                seqs.append(t[:a.tokens])
            if len(seqs) >= a.docs:
                break
        if not seqs:
            raise SystemExit(f"held {lang}: no document of {a.tokens} tokens in split held")
        held[lang] = {"held": seqs}
        log(f"held {lang}: {len(seqs)} documents x {a.tokens} tokens ({p})")
    if a.target_lang not in held:
        raise SystemExit(f"--target-lang {a.target_lang} is not among --held")
    clean = {}

    def full_mask():
        return np.ones((host.L, W), np.float32)

    def score(remove, langs):
        m = full_mask()
        m[layers] = np.where(remove, 0.0, 1.0)
        cfg = E.configure(host, D.from_np(m, host), None, host.L)
        return E.english_cost(host, {k: held[k] for k in langs}, clean, cfg, pooled=False)

    res = {"schema": "mlx2.amoeba-exclusive-box-test.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mlx2_source_revision": O.git_rev(), "map": a.map, "map_revision": rep_map.get("mlx2_source_revision"),
           "target_lang": a.target_lang, "watchdog_waiver": ({"swap_growth_gb": a.swap_growth_gb,
                                                               "note": a.swap_waiver_note} if a.swap_growth_gb != 2.0 else None), "held_docs": {k: len(v["held"]) for k, v in held.items()}, "tokens": a.tokens,
           "baseline_corpus": (rep_map.get("args") or {}).get("baseline"),
           "rule": "box = target neurons with the map's baseline-corpus mean <= r x target mean (exclusive against that "
                   "baseline only; the other held languages test the rest); test = held dCE of the model without "
                   "the box (core = all other neurons), every language; random = same per-layer counts among "
                   "non-target neurons", "tiers": []}
    rlangs = [x for x in a.random_langs.split(",") if x in held]
    non_target = ~tgt
    t0 = time.time()
    # the masked path vs the stock fused call: an all-ones mask removes nothing, so its dCE is the path offset every
    # "without box" number carries (review 2026-11-16)
    res["path_offset"] = {k: v for k, v in score(np.zeros_like(tgt), list(held)).items() if k != "mean_d_ce"}
    log("path offset (nothing removed): " + ", ".join(f"{k} {v:+.4f}" for k, v in res["path_offset"].items()))
    for i, rs in enumerate(a.ratios.split(",")):
        r = float(rs)
        box = tgt & (mb <= r * mt) if np.isfinite(r) else tgt.copy()
        np.savez_compressed(out / f"box_r{rs}.npz", box=box, moe_layers=np.array(layers), ratio=np.array(r))
        without = score(box, list(held))
        rnd = np.zeros_like(box)
        g = np.random.default_rng(a.seed + 100 + i)
        for l in range(box.shape[0]):
            k = int(box[l].sum())
            pool = np.flatnonzero(non_target[l])
            if k:
                rnd[l, g.choice(pool, min(k, pool.size), replace=False)] = True
        random_ = score(rnd, rlangs)
        tier = {"ratio": rs, "random_neurons": int(rnd.sum()), "box_neurons": int(box.sum()), "box_share": float(box.mean()),
                "box_full_experts": int((box.reshape(len(layers), -1, int(rep_map["F"])).sum(2) == int(rep_map["F"])).sum()),
                "without_box": {k: v for k, v in without.items() if k != "mean_d_ce"},
                "random_removal": {k: v for k, v in random_.items() if k != "mean_d_ce"},
                "wall_s": round(time.time() - t0, 1)}
        res["tiers"].append(tier)
        log(f"r={rs}: box {tier['box_neurons']} neurons; without box " +
            ", ".join(f"{k} {v:+.4f}" for k, v in tier["without_box"].items()) + "; random " +
            ", ".join(f"{k} {v:+.4f}" for k, v in tier["random_removal"].items()))
        (out / "report.json").write_text(json.dumps(res, indent=1))
    res["mlx_peak_gb"] = O.peak_gb()
    (out / "report.json").write_text(json.dumps(res, indent=1))
    langs = list(held)
    print("\n| r | box neurons | " + " | ".join(f"{k} without box" for k in langs) + " | " +
          " | ".join(f"{k} random" for k in rlangs) + " |")
    for t in res["tiers"]:
        print(f"| {t['ratio']} | {t['box_neurons']:,} | " + " | ".join(f"{t['without_box'][k]:+.3f}" for k in langs) +
              " | " + " | ".join(f"{t['random_removal'][k]:+.3f}" for k in rlangs) + " |")


if __name__ == "__main__":
    main()
