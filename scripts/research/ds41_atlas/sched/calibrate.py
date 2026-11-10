#!/usr/bin/env python3
"""Calibrate the cost model against the capture run itself (no new M5 benchmark).

The capture server ran original_lfu (lifetime LFU by token-expert selections, LRU ties) with a 75 GB
budget; its counters (status-after.json) give total misses and demand wait. We replay that policy on
the trace, check the replayed miss total against the counter, then regress each decode layer-call's
wall delta (capture timestamps) on the replayed miss count of that layer call:
    delta_ms(layer call) ~ c0 + m * misses
c0 = per-layer time with no miss (compute + capture instrumentation), m = effective stall per miss
(read + parse/copy + upload, with the runtime's 16 read workers). Capture instrumentation (per-layer
sync + x copy) inflates c0; it is reported as an upper bound for compute.
"""
import heapq
import json

import numpy as np

from common import CAPTURE, EXPERT_BYTES, OUT, atomic_json, load_trace, provenance

CAP = int(75e9 // EXPERT_BYTES)


def replay_lfu(z):
    L, R, P = z["rec_layer"], z["rec_rows"], z["rec_pos"]
    ids = z["ids"]
    counts, last, cache, heap, clock = {}, {}, set(), [], 0
    miss = np.zeros(len(L), np.int32)
    for r in range(len(L)):
        layer, rows, p = int(L[r]), int(R[r]), int(P[r])
        sel = ids[p:p + rows, layer].reshape(-1)
        u, c = np.unique(sel, return_counts=True)
        clock += 1
        for e, n in zip(u.tolist(), c.tolist()):
            k = (layer, e)
            counts[k] = counts.get(k, 0) + n
            last[k] = clock
            if k not in cache:
                miss[r] += 1
                while len(cache) >= CAP:
                    cnt, lst, v = heapq.heappop(heap)
                    if v in cache and cnt == counts[v] and lst == last[v]:
                        cache.discard(v)
                cache.add(k)
            heapq.heappush(heap, (counts[k], clock, k))
    return miss


def main():
    z = load_trace()
    st = json.load(open(f"{CAPTURE}/status-after.json"))["execution"]["expert_stream"]["counters"]
    miss = replay_lfu(z)
    L, R, S, t = z["rec_layer"], z["rec_rows"], z["rec_step"], z["rec_t"]
    dec = R == 1
    # wall delta of a decode layer call = t[next record] - t[this] when both in the same step
    nxt_same = np.zeros(len(L), bool); nxt_same[:-1] = (S[1:] == S[:-1]) & dec[:-1]
    idx = np.where(nxt_same)[0]
    # the record at layer l is emitted after layer l's router; the delta to l+1 covers layer l's experts
    # (miss stall at layer l) + attention/dense of layer l+1.
    delta = (t[idx + 1] - t[idx]) / 1e6
    m = miss[idx].astype(np.float64)
    keep = delta < 500
    A = np.stack([np.ones(keep.sum()), m[keep]], 1)
    coef, *_ = np.linalg.lstsq(A, delta[keep], rcond=None)
    by = {int(k): {"n": int((m[keep] == k).sum()), "median_ms": round(float(np.median(delta[keep][m[keep] == k])), 2),
                   "mean_ms": round(float(delta[keep][m[keep] == k].mean()), 2)}
          for k in range(7) if (m[keep] == k).sum() > 50}
    i0 = np.where(z["rec_layer"] == 0)[0]
    step_s = np.diff(t[i0]) / 1e9
    ok = dec[i0][1:] & dec[i0][:-1] & (step_s < 5)
    res = provenance(policy_replayed="lifetime LFU by token selections, LRU ties, 75e9 B (3989 experts)")
    res.update({
        "counter_misses_total": st["v41_expert_cache_misses"],
        "replayed_misses_total": int(miss.sum()),
        "replayed_decode_misses_per_token": round(float(miss[dec].sum() / dec.sum() * 40), 2),
        "counter_demand_wait_s": st["v41_expert_demand_wait_ns"] / 1e9,
        "counter_wait_per_miss_ms": st["v41_expert_demand_wait_ns"] / 1e6 / st["v41_expert_cache_misses"],
        "decode_step_s_mean": round(float(step_s[ok].mean()), 4),
        "decode_step_s_median": round(float(np.median(step_s[ok])), 4),
        "regression_ms": {"c0_per_layer": round(float(coef[0]), 3), "per_miss": round(float(coef[1]), 3)},
        "delta_by_replayed_misses": by,
        "note": "c0 includes capture instrumentation (per-layer host sync + x copy); measured from the "
                "capture run timestamps, not a new benchmark. Per-miss slope is effective (16 read workers, "
                "misses of one layer overlap).",
    })
    atomic_json(f"{OUT}/calibration.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
