#!/usr/bin/env python3
"""Static tier coverage on held-out decode tokens (DS4.1 Source, CPU only).

Per layer, experts are ranked by train-split usage; the top band is Q4 (18.8 MB), the next Q2
(11.06 MB), the next Q1 (6.64 MB), the rest not resident. Reports, for held-out decode tokens, the
share of expert uses and of routing weight that each tier serves, and the tier of each token's
rank-1 expert. Static map only (no recency), so a lower bound on what dynamic pools would hit.
usage: tier_coverage.py CAPTURE TRACE.npz
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soft_reap import scores  # noqa: E402
L, E = 40, 384
SZ = {"q4": 18_800_640, "q2": 11_059_200, "q1": 6_635_520}

cap, trace = sys.argv[1], sys.argv[2]
z = np.load(trace); ids = z["ids"]; split = z["tok_split"]; dec = z["tok_decode"]
s = scores(cap, len(ids)); w = s / np.maximum(s.sum(-1, keepdims=True), 1e-12)
train = np.zeros((L, E)); ho = np.where((dec == 1) & (split == 1))[0]
for t in np.where(split == 0)[0]:
    for l in range(L): train[l, ids[t, l]] += 1
rank = np.argsort(np.argsort(-train, axis=1, kind="stable"), axis=1)   # 0 = most used per layer
mixes = {          # ~2.0 GB per layer = ~80 GB of experts
    "A 50 q4 / 40 q2 / 93 q1": (50, 40, 93),
    "B 25 q4 / 60 q2 / 130 q1": (25, 60, 130),
    "C 10 q4 / 50 q2 / 190 q1": (10, 50, 190),
    "D 0 q4 / 0 q2 / 300 q1": (0, 0, 300),
    "E 0 q4 / 180 q2": (0, 180, 0),
    "today 75GB: 99 q4 (static map)": (99, 0, 0),
}
out = {}
for name, (a, b, c) in mixes.items():
    gb = L * (a * SZ["q4"] + b * SZ["q2"] + c * SZ["q1"]) / 1e9
    tier = np.full((L, E), 3)
    tier[rank < a + b + c] = 2; tier[rank < a + b] = 1; tier[rank < a] = 0
    tt = tier[np.arange(L)[None, :, None], ids[ho]]          # [T,40,6]
    ww = w[ho]
    use = [float((tt == k).mean()) for k in range(4)]
    wt = [float((ww * (tt == k)).sum() / ww.sum()) for k in range(4)]
    top = np.argmax(ww, -1); t1 = np.take_along_axis(tt, top[..., None], -1)[..., 0]
    r1 = [float((t1 == k).mean()) for k in range(4)]
    out[name] = {"gb": round(gb, 1), "uses_q4_q2_q1_miss": [round(x, 3) for x in use],
                 "weight_q4_q2_q1_miss": [round(x, 3) for x in wt], "rank1_q4_q2_q1_miss": [round(x, 3) for x in r1],
                 "misses_per_token": round(use[3] * 240, 1)}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(trace), "..", "tier_coverage.json"), "w"), indent=1)

# ---- soft reaping on top (static map): drop the k lowest-weight of each token's top-6 on a
# layer range, and/or skip a non-resident expert whose weight is below tau. Misses per token and
# the routing weight removed are reported for the decode tokens above.
print("\nwith soft reaping (misses per token / weight removed per token-layer):")
order = np.argsort(np.argsort(ww := w[ho], axis=-1, kind="stable"), axis=-1)   # 0 = lowest weight
rows = {}
for name, (a, b, c) in mixes.items():
    tier = np.full((L, E), 3)
    tier[rank < a + b + c] = 2; tier[rank < a + b] = 1; tier[rank < a] = 0
    tt = tier[np.arange(L)[None, :, None], ids[ho]]
    res = {}
    for label, k, lo, hi, tau in [("none", 0, 0, -1, 0), ("drop2 L13-26", 2, 13, 26, 0), ("drop1 all", 1, 0, 39, 0),
                                  ("drop2 all", 2, 0, 39, 0), ("skip-miss<0.10", 0, 0, -1, 0.10),
                                  ("drop2 L13-26 + skip-miss<0.10", 2, 13, 26, 0.10)]:
        lay = np.zeros(L, bool); lay[lo:hi + 1] = True
        dropped = (order < k) & lay[None, :, None]
        skipped = (~dropped) & (tt == 3) & (ww < tau)
        miss = (~dropped) & (~skipped) & (tt == 3)
        removed = (ww * (dropped | skipped)).sum() / (len(ho) * L)
        res[label] = (round(float(miss.sum() / len(ho)), 1), round(float(removed), 3))
    rows[name] = res
    print(name, json.dumps(res), flush=True)
