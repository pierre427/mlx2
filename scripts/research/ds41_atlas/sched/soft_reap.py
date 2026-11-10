#!/usr/bin/env python3
"""Offline sizing of runtime soft-reaping on DS4.1 Source (Pierre 2026-11-08).

Per decode token and layer the router picks top-6; combine weights are the selected
scores renormalised (sum 1). Policies, all on a global LRU cache, held-out decode only:
  exact        : load every selected expert
  drop<tau     : a selected expert that is NOT resident and has weight < tau is skipped
                 (no load, no insert); remaining weights renormalised
  q1<tau       : same experts are read as a Q1 copy (6.64 MB, 0.353 of an MXFP4 read), not cached
Reports misses/token (MXFP4-equivalent reads), dropped/Q1 routing mass. Routing mass is a proxy,
NOT a quality measurement; quality needs GPU GSM8K + functional oracles.

usage: soft_reap.py CAPTURE_DIR TRACE.npz [--caps 2127,3989] [--out json]
"""
import argparse, json, os, sys
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cache_sim import read_capture  # noqa: E402

L = 40
Q1_COST = 6_635_520 / 18_800_640


def scores(capture, T):
    out = np.zeros((T, L, 6), np.float32); pos = [0] * L
    for layer, rows, _t, _ids, s, _x in read_capture(f"{capture}/capture.bin"):
        out[pos[layer]:pos[layer] + rows, layer] = np.asarray(s, np.float32).reshape(rows, 6); pos[layer] += rows
    return out


def run(ids, w, step, split, dec, cap, mode, tau):
    cache = OrderedDict(); miss = 0.0; n = 0; mass = 0.0; touched = 0
    order = np.argsort(step, kind="stable"); i = 0; T = len(order)
    while i < T:
        j = i
        while j < T and step[order[j]] == step[order[i]]: j += 1
        toks = order[i:j]; i = j
        ev = len(toks) == 1 and dec[toks[0]] == 1 and split[toks[0]] == 1
        single = len(toks) == 1 and dec[toks[0]] == 1
        for layer in range(L):
            if single:
                t = toks[0]; sel = [(int(e), float(w[t, layer, r])) for r, e in enumerate(ids[t, layer])]
            else:
                sel = [(int(e), 1.0) for e in np.unique(ids[toks, layer].reshape(-1))]
            m = 0.0; dropped = 0.0
            for e, wt in sel:
                k = (layer, e)
                if k in cache:
                    cache.move_to_end(k); continue
                if single and mode != "exact" and wt < tau:
                    dropped += wt
                    m += Q1_COST if mode == "q1" else 0.0
                    continue
                m += 1.0
                if len(cache) >= cap:
                    for v in cache:
                        if v[0] != layer: break
                    del cache[v]
                cache[k] = True
            if ev:
                miss += m; mass += dropped; touched += dropped > 0
        if ev: n += 1
    return {"misses_per_token": round(miss / n, 2), "mass_per_token_layer": round(mass / (n * L), 4),
            "layers_touched_per_token": round(touched / n, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture"); ap.add_argument("trace"); ap.add_argument("--caps", default="2127,3989"); ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(a.trace); ids = z["ids"]; T = len(ids)
    s = scores(a.capture, T); w = s / np.maximum(s.sum(-1, keepdims=True), 1e-12)
    dec = z["tok_decode"]; split = z["tok_split"]; step = z["tok_step"]
    ho = (dec == 1) & (split == 1)
    srt = -np.sort(-w[ho], axis=-1)
    stats = {"mean_weight_by_rank": [round(float(x), 4) for x in srt.mean((0, 1))],
             "p10_weight_by_rank": [round(float(x), 4) for x in np.percentile(srt, 10, axis=(0, 1))],
             "share_rank6_below": {t: round(float((srt[..., 5] < t).mean()), 3) for t in (0.05, 0.08, 0.1, 0.12)},
             "share_any_below": {t: round(float((srt < t).any(-1).mean()), 3) for t in (0.05, 0.08, 0.1, 0.12)},
             "mean_slots_below": {t: round(float((srt < t).sum(-1).mean()), 3) for t in (0.05, 0.08, 0.1, 0.12)}}
    print(json.dumps(stats), flush=True)
    res = {"heldout_decode_tokens": int(ho.sum()), "weight_stats": stats, "rows": []}
    for cap in map(int, a.caps.split(",")):
        for mode, tau in [("exact", 0)] + [(m, t) for m in ("drop", "q1") for t in (0.05, 0.08, 0.1, 0.12)]:
            r = {"cap": cap, "gb": round(cap * 18_800_640 / 1e9, 1), "mode": mode, "tau": tau,
                 **run(ids, w, step, split, dec, cap, mode, tau)}
            res["rows"].append(r); print(r, flush=True)
    if a.out:
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
