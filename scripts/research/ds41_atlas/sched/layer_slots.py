#!/usr/bin/env python3
"""Layer-wise slot reservation vs global policies over the DS4.1 Source routing trace.

Pierre 2026-11-08: reserve a number of slots per layer and evict the least-reused expert,
fed by a usage table that keeps growing during runs. Compares against global LRU, lifetime
LFU (the runtime default), per-layer partitions and hybrids. Table warm start = train-split
counts; evaluation = held-out decode tokens only. CPU only, pure Python.

usage: layer_slots.py TRACE.npz [--caps 2127,3989,4787] [--out json]
"""
import argparse, heapq, json, os
from collections import OrderedDict
import numpy as np

L, E = 40, 384


def stream(trace):
    z = np.load(trace)
    ids, step, split, dec = z["ids"], z["tok_step"], z["tok_split"], z["tok_decode"]
    order = np.argsort(step, kind="stable")
    steps, cur, toks = [], None, []
    for t in order:
        s = int(step[t])
        if s != cur and toks:
            steps.append(toks); toks = []
        cur = s; toks.append(t)
    if toks: steps.append(toks)
    acc, ev = [], []                     # acc[i] = list of (layer, expert) in serving order; ev[i] = held-out decode step
    for toks in steps:
        a = []
        for layer in range(L):
            for e in np.unique(ids[toks, layer].reshape(-1)):
                a.append((layer, int(e)))
        acc.append(a)
        ev.append(len(toks) == 1 and dec[toks[0]] == 1 and split[toks[0]] == 1)
    train = np.zeros((L, E), np.int64)
    for t in np.where(split == 0)[0]:
        for layer in range(L):
            train[layer, ids[t, layer]] += 1
    return acc, ev, train


class Global:
    """kind: lru | lfu (lifetime, lazy heap) | decay (sampled oldest-64, decayed count)."""
    def __init__(self, cap, kind, table=None, half=64, floor=0, pin=None):
        self.cap, self.kind, self.floor, self.pin = cap, kind, floor, pin or set()
        self.c = OrderedDict(); self.per = [0] * L
        self.cnt = {} if table is None else {(l, e): float(table[l, e]) for l in range(L) for e in range(E)}
        self.heap = []; self.g = 0.5 ** (1 / half); self.now = 0; self.last = {}

    def score(self, k):
        if self.kind == "decay":
            return self.cnt.get(k, 0.0) * self.g ** (self.now - self.last.get(k, self.now))
        return self.cnt.get(k, 0.0)

    def touch(self, k):
        if self.kind == "decay":
            self.cnt[k] = self.score(k) + 1.0; self.last[k] = self.now
        else:
            self.cnt[k] = self.cnt.get(k, 0.0) + 1.0
        if self.kind == "lfu": heapq.heappush(self.heap, (self.cnt[k], k))

    def victim(self, layer_now):
        if self.kind == "lfu":
            skipped = []
            try:
                while True:
                    c, k = heapq.heappop(self.heap)
                    if k not in self.c or self.cnt[k] != c:
                        continue                      # stale entry
                    if k[0] != layer_now and self.per[k[0]] > self.floor:
                        return k
                    skipped.append((c, k))
            finally:
                for item in skipped: heapq.heappush(self.heap, item)
        cand = []
        for k in self.c:
            if k[0] != layer_now and self.per[k[0]] > self.floor:
                cand.append(k)
                if self.kind == "lru" or len(cand) >= 64: break
        return cand[0] if self.kind == "lru" else min(cand, key=self.score)

    def access(self, k, layer_now):
        if k in self.pin:
            self.touch(k); return False
        if k in self.c:
            self.touch(k); self.c.move_to_end(k); return False
        if len(self.c) >= self.cap:                 # evict before touching: the new key's heap entry
            v = self.victim(layer_now)              # must not be discarded as non-resident
            del self.c[v]; self.per[v[0]] -= 1
        self.c[k] = True; self.per[k[0]] += 1
        self.touch(k)
        return True


class Partition:
    """Fixed slots per layer; within a layer evict LRU or least-count (table-fed, online)."""
    def __init__(self, cap, kind, table=None):
        self.slots = cap // L; self.kind = kind
        self.c = [OrderedDict() for _ in range(L)]
        self.cnt = np.zeros((L, E)) if table is None else table.astype(float).copy()

    def access(self, k, layer_now):
        l, e = k; self.cnt[l, e] += 1; c = self.c[l]
        if e in c:
            c.move_to_end(e); return False
        if len(c) >= self.slots:
            if self.kind == "lru": c.popitem(last=False)
            else:
                v = min(c, key=lambda q: self.cnt[l, q]); del c[v]
        c[e] = True
        return True


def run(acc, ev, make):
    cache = make(); miss = 0; n = 0
    for a, is_eval in zip(acc, ev):
        if hasattr(cache, "now"): cache.now += 1
        m = 0
        for k in a:
            m += cache.access(k, k[0])
        if is_eval: miss += m; n += 1
    return round(miss / max(n, 1), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace"); ap.add_argument("--caps", default="2127,3989,4787"); ap.add_argument("--out")
    a = ap.parse_args()
    acc, ev, train = stream(a.trace)
    top16 = {(l, int(e)) for l in range(L) for e in np.argsort(-train[l])[:16]}
    res = {"heldout_decode_tokens": int(sum(ev)), "rows": []}
    for cap in map(int, a.caps.split(",")):
        pols = {
            "global LRU": lambda: Global(cap, "lru"),
            "global lifetime LFU, cold (runtime default)": lambda: Global(cap, "lfu"),
            "global lifetime LFU, train-table warm": lambda: Global(cap, "lfu", table=train),
            "global decayed LFU h=64 (oldest-64 sample)": lambda: Global(cap, "decay"),
            "global decayed LFU h=256, train-table warm": lambda: Global(cap, "decay", table=train, half=256),
            "per-layer partition LRU": lambda: Partition(cap, "lru"),
            "per-layer partition least-used, train-table warm": lambda: Partition(cap, "lfu", table=train),
            "global LRU + 16/layer floor": lambda: Global(cap, "lru", floor=16),
            "global decayed h=64 + 16/layer floor": lambda: Global(cap, "decay", floor=16),
            "16/layer pinned table-top + global LRU rest": lambda: Global(cap - 640, "lru", pin=top16),
        }
        for name, mk in pols.items():
            r = run(acc, ev, mk)
            row = {"cap_experts": cap, "gb": round(cap * 18_800_640 / 1e9, 1), "policy": name, "misses_per_token": r}
            res["rows"].append(row); print(row, flush=True)
    if a.out:
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
