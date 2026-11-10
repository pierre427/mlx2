#!/usr/bin/env python3
"""Cache effect of per-layer keep counts (top-k by routing weight, rest skipped) on DS4.1 Source.

Held-out decode tokens, global LRU at several RAM sizes. Reports misses/token, MB read per token,
share of expert uses served from RAM, and the working set (distinct experts the kept slots touch,
and how many per layer cover 95% of kept uses). CPU only.
usage: keep_plans.py CAPTURE TRACE.npz
"""
import json, os, sys
from collections import OrderedDict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soft_reap import scores, L  # noqa: E402
EB = 18_800_640


def plan(spec):
    k = np.full(L, 6)
    for (a, b), v in spec:
        k[a:b + 1] = v
    return k


PLANS = {
    "exact (6 everywhere)": plan([]),
    "keep 3 everywhere": plan([((0, 39), 3)]),
    "keep 2 everywhere": plan([((0, 39), 2)]),
    "keep 1 everywhere (known to break)": plan([((0, 39), 1)]),
    "keep 1 tolerant, keep 3 on 10-14 & 20-24": plan([((0, 39), 1), ((10, 14), 3), ((20, 24), 3)]),
    "keep 2 tolerant, keep 3 on 10-14 & 20-24": plan([((0, 39), 2), ((10, 14), 3), ((20, 24), 3)]),
}


def main():
    cap, trace = sys.argv[1], sys.argv[2]
    z = np.load(trace); ids = z["ids"]; step = z["tok_step"]; split = z["tok_split"]; dec = z["tok_decode"]
    s = scores(cap, len(ids)); w = s / np.maximum(s.sum(-1, keepdims=True), 1e-12)
    rank = np.argsort(np.argsort(-w, axis=-1, kind="stable"), axis=-1)          # 0 = top weight
    order = np.argsort(step, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(step[order])) + 1)
    res = {}
    for name, keep in PLANS.items():
        kept_mask = rank < keep[None, :, None]
        ho = (dec == 1) & (split == 1)
        # working set over held-out decode
        ws = [np.unique(ids[ho, l][kept_mask[ho, l]]) for l in range(L)]
        cover95 = []
        for l in range(L):
            vals, cnt = np.unique(ids[ho, l][kept_mask[ho, l]], return_counts=True)
            c = np.sort(cnt)[::-1].cumsum() / cnt.sum()
            cover95.append(int(np.searchsorted(c, 0.95) + 1))
        row = {"uses_per_token": int(keep.sum()), "distinct_experts_heldout": int(sum(len(x) for x in ws)),
               "experts_for_95pct_of_uses": int(sum(cover95)),
               "gb_for_95pct_at_q4": round(sum(cover95) * EB / 1e9, 1)}
        for gb in (75, 90):
            cap_n = int(gb * 1e9 // EB); cache = OrderedDict(); miss = n = uses = 0
            for g in groups:
                single = len(g) == 1 and dec[g[0]] == 1
                ev = single and split[g[0]] == 1
                for l in range(L):
                    if single:
                        t = g[0]; sel = ids[t, l][kept_mask[t, l]]
                    else:
                        sel = np.unique(ids[g, l].reshape(-1))
                    for e in sel:
                        key = (l, int(e))
                        if ev: uses += 1
                        if key in cache:
                            cache.move_to_end(key); continue
                        if ev: miss += 1
                        if len(cache) >= cap_n:
                            for v in cache:
                                if v[0] != l: break
                            del cache[v]
                        cache[key] = True
                if ev: n += 1
            row[f"{gb}GB"] = {"misses_per_token": round(miss / n, 2), "mb_read_per_token": round(miss / n * EB / 1e6, 1),
                              "served_from_ram": round(1 - miss / max(uses, 1), 4)}
        res[name] = row
        print(name, json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(trace), "..", "keep_plans.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
