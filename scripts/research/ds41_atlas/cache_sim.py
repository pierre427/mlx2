#!/usr/bin/env python3
"""Expert-cache simulator over a DS4.1 Source routing capture (routing_capture_server.py format).

Replays the captured (layer, token) expert selections in serving order and counts misses per
token for RAM tiers of different sizes and policies, then for three-Mac layer placements
(Pierre 2026-10-17: resident -> pipeline -> streaming). Pure numpy; CPU only.

usage: cache_sim.py CAPTURE_DIR [--sizes 2000,3000,...] [--out sim.json]
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

LAYERS, EXPERTS, TOPK = 40, 384, 6
EXPERT_BYTES = 18_800_640


def read_capture(path, keep_x=False):
    """Yields (layer, rows, t_ns, ids[rows,6], scores, x|None) records in serving order."""
    with open(path, "rb") as f:
        while True:
            try:
                head = np.load(f, allow_pickle=False)
            except (EOFError, ValueError):
                return
            ids = np.load(f, allow_pickle=False)
            scores = np.load(f, allow_pickle=False)
            if keep_x:
                x = np.load(f, allow_pickle=False)
            else:
                np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                f.seek(int(np.prod(shape)) * dtype.itemsize, os.SEEK_CUR)
                x = None
            yield int(head[0]), int(head[1]), int(head[2]), ids.astype(np.int32), scores, x


def access_stream(capture_dir):
    """Flat access list [(layer, expert, token_step)] plus per-access request id from replies windows.

    Prefill rows of one call are one step (they are all needed for that call); each decode call is one step.
    """
    replies = [json.loads(line) for line in open(f"{capture_dir}/replies.jsonl")]
    windows = [(r["t0_ns"], r["t1_ns"], r["id"], r["cat"], r["split"]) for r in replies]
    acc, step, last_layer = [], 0, None
    meta = []
    for layer, rows, t, ids, _s, _x in read_capture(f"{capture_dir}/capture.bin"):
        if last_layer is not None and layer <= last_layer:
            step += 1                       # a new forward pass begins at a lower layer
        last_layer = layer
        req = next((w for w in windows if w[0] <= t <= w[1]), None)
        uniq = np.unique(ids.reshape(-1))
        for e in uniq:
            acc.append((layer, int(e), step))
        meta.append((step, layer, rows, req[2] if req else None, req[3] if req else None, req[4] if req else None,
                     ids.size))
    return acc, meta


def simulate(acc, capacity, policy, pinned=frozenset(), alpha=0.1):
    """Returns misses per access list position (bool array). pinned keys never miss and use no capacity."""
    cache = OrderedDict()
    miss = np.zeros(len(acc), bool)
    if policy == "belady":
        nxt = [0] * len(acc)
        seen = {}
        for i in range(len(acc) - 1, -1, -1):
            k = acc[i][:2]
            nxt[i] = seen.get(k, 1 << 60)
            seen[k] = i
        import heapq
        heap, where = [], {}
        for i, (layer, e, _s) in enumerate(acc):
            k = (layer, e)
            if k in pinned:
                continue
            if k in where:
                where[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k)); continue
            miss[i] = True
            if len(where) >= capacity:
                while True:
                    negn, victim = heapq.heappop(heap)
                    if where.get(victim) == -negn:
                        del where[victim]; break
            where[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
        return miss
    freq = {}
    ema = {}
    ema_t = {}

    # Reuse EMA decayed by elapsed steps, the law of sched/eventsim.py (access_stats and the cycle
    # score): value * (1 - alpha) ** (step - last_update) + alpha on access, decayed again when scored.
    def bump(k, step):
        ema[k] = ema.get(k, 0.0) * (1 - alpha) ** (step - ema_t.get(k, step)) + alpha
        ema_t[k] = step

    for i, (layer, e, step) in enumerate(acc):
        k = (layer, e)
        if k in pinned:
            continue
        freq[k] = freq.get(k, 0) + 1
        if k in cache:
            cache.move_to_end(k)
            if policy == "cycle":
                bump(k, step)
            continue
        miss[i] = True
        if len(cache) >= capacity:
            if policy == "lru":
                cache.popitem(last=False)
            elif policy == "lfu":
                victim = min(cache, key=lambda q: (freq[q], 0))
                del cache[victim]
            elif policy == "cycle":
                # cycle-aware: among the oldest 64 entries, evict the lowest reuse-EMA weighted by layer
                # distance to the next layer the decode will reach (layers just passed are needed last).
                cand = list(cache)[:64]
                def score(q):
                    dist = (q[0] - layer) % LAYERS or LAYERS
                    return ema.get(q, 0.0) * (1 - alpha) ** (step - ema_t.get(q, step)) / dist
                victim = min(cand, key=score)
                del cache[victim]
        cache[k] = True
        bump(k, step)
    return miss


def per_token(miss, acc, meta):
    decode_steps = {m[0] for m in meta if m[2] == 1}
    m = sum(1 for i, a in enumerate(acc) if miss[i] and a[2] in decode_steps)
    return m / max(1, len(decode_steps))


def placements():
    """Candidate layer placements: remote RAM capacity in experts (approx, after dense + OS headroom)."""
    m3 = int(26e9 // EXPERT_BYTES)       # M3 Pro 36 GB: ~26 GB for experts
    m2 = int(16e9 // EXPERT_BYTES)       # M2 24 GB: ~16 GB for experts
    return {"m3_experts": m3, "m2_experts": m2,
            "m3_full_layers": m3 // EXPERTS, "m2_full_layers": m2 // EXPERTS}


def contiguous_ranges(traffic, lengths, total):
    """Greedy: place each stage's contiguous window where it covers the most remaining traffic."""
    chosen, budget = set(), total
    for n in lengths:
        n = min(n, budget)
        if n <= 0:
            break
        best = max((sum(traffic[L] for L in range(s0, s0 + n) if L not in chosen), s0)
                   for s0 in range(LAYERS - n + 1) if not any(L in chosen for L in range(s0, s0 + n)))
        chosen |= set(range(best[1], best[1] + n)); budget -= n
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--sizes", default="1000,2000,3000,4000,4800,6000,7200,9000")
    ap.add_argument("--policies", default="lru,cycle,belady")
    ap.add_argument("--out")
    a = ap.parse_args()
    acc, meta = access_stream(a.capture_dir)
    decode = [m for m in meta if m[2] == 1]
    res = {"accesses": len(acc), "steps": len({m[0] for m in meta}), "decode_layer_calls": len(decode),
           "placement_capacity": placements(), "ram": {}}
    use = {}
    for layer, e, _s in acc:
        use[(layer, e)] = use.get((layer, e), 0) + 1
    res["distinct_experts_touched"] = len(use)
    res["touched_by_layer"] = [sum(1 for k in use if k[0] == L) for L in range(LAYERS)]
    for size in map(int, a.sizes.split(",")):
        for pol in a.policies.split(","):
            miss = simulate(acc, size, pol)
            res["ram"][f"{pol}@{size}"] = {"misses_per_decode_token": round(per_token(miss, acc, meta), 2),
                                           "gb": round(size * EXPERT_BYTES / 1e9, 1)}
            print(pol, size, res["ram"][f"{pol}@{size}"], flush=True)
    # Pipeline placements: whole layer ranges resident on the M3/M2 (pinned = no M5 miss, no M5 capacity).
    cap = placements()
    for m5 in (3989, 4787):
        for remote_layers in (0, cap["m3_full_layers"], cap["m3_full_layers"] + cap["m2_full_layers"]):
            # contiguous stage ranges (a pipeline hop per range) with the most distinct expert traffic
            ranked = contiguous_ranges(res["touched_by_layer"], [cap["m3_full_layers"], cap["m2_full_layers"]], remote_layers)
            pinned = frozenset((L, e) for L in ranked for e in range(EXPERTS))
            miss = simulate(acc, m5, "lru", pinned=pinned)
            key = f"pipeline m5={m5} remote_layers={remote_layers}"
            res.setdefault("pipeline", {})[key] = {"layers": sorted(ranked),
                                                   "misses_per_decode_token": round(per_token(miss, acc, meta), 2)}
            print(key, res["pipeline"][key]["misses_per_decode_token"], flush=True)
    if a.out:
        tmp = a.out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
