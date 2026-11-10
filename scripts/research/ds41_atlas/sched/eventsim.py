#!/usr/bin/env python3
"""Timed event simulator for DS4.1 Source routed-expert caching + prefetch (CPU only).

Replays the routing capture in serving order (persistent cache across the 72 requests, as a server).
Prefill steps warm the cache (union of each prefill call's experts per layer, untimed). Decode tokens are
timed layer by layer:

  layer L on the M5:  [router_frac*c compute] -> router decision -> demand-fetch every routed expert that is
                      not resident (waits for in-flight ones) -> [(1-router_frac)*c compute + GPU steal]
  layer L on a remote pipeline stage: all its experts resident there; compute c*factor; a hop (RTT/2 +
                      hidden-state bytes/link) whenever consecutive layers change node.

Channels: one serial pipe per byte tier (nvme, t7, m3ram) with service ms/expert from costs.py; strict
priority -- demand first (FIFO), then prefetches by priority (expected stall saved = p * miss cost, or
earliest-deadline-first); an in-progress transfer is never aborted; a queued prefetch is dropped when its
target (token, layer) has passed or the expert is already resident/in flight.

Eviction policies: lru, cycle (K3.1 sweep: reuse-EMA / layer distance, 64-key sample), gds (GreedyDual,
cost-aware LRU), nextuse (learned P(reuse within H steps) from nextuse.json, 64-key sample), nextuse_cost,
belady (offline bound, farthest next use).
Prefetchers: none, d1, d2, xtokA (x_39(t) head, issued at token t layer 39), xtokAB (issued at token t+1
layer 0), xtokAB+d1, horizon h2/h3 (draft-token stand-ins: UPPER BOUND), oracle_d1, oracle_xtok.

Outputs per run: seconds/token, stall/token, misses/token by tier, prefetch loads/useful, channel
utilisation, separately for train and heldout requests.
"""
import heapq
import json
import math
import random
from collections import OrderedDict, deque

import numpy as np

import costs as C
from common import EXPERT_BYTES, EXPERTS, LAYERS, OUT, TOPK, load_trace

NK = LAYERS * EXPERTS

_TR = None
_PR = None


def trace():
    global _TR
    if _TR is None:
        z = load_trace()
        # step table: for each forward step, (kind, row0, rows); prefill unions per layer
        L, R, P, S = z["rec_layer"], z["rec_rows"], z["rec_pos"], z["rec_step"]
        first = np.where(L == 0)[0]
        steps = []
        ids = z["ids"]
        for i in first:
            rows, p = int(R[i]), int(P[i])
            if rows == 1:
                steps.append((1, p))
            else:
                uni = [np.unique(ids[p:p + rows, l]).tolist() for l in range(LAYERS)]
                steps.append((0, p, uni))
        z["steps"] = steps
        # static per-key usage frequency on TRAIN rows (prior feature for nextuse, placement ranking)
        tr = z["tok_split"] == 0
        f = np.zeros(NK, np.int64)
        for l in range(LAYERS):
            np.add.at(f, l * EXPERTS + ids[tr, l].reshape(-1).astype(np.int64), 1)
        z["freq_train"] = f
        _TR = z
    return _TR


def preds():
    global _PR
    if _PR is None:
        _PR = dict(np.load(f"{OUT}/preds.npz"))
    return _PR


def next_use_steps(tr, remote_layers=frozenset()):
    """For belady/nextuse training: per access (in sim order) the step index of the key's next access."""
    acc_keys, acc_step = [], []
    for si, st in enumerate(tr["steps"]):
        if st[0] == 1:
            row = tr["ids"][st[1]]
            for l in range(LAYERS):
                if l in remote_layers:
                    continue
                for e in row[l].tolist():
                    acc_keys.append(l * EXPERTS + e); acc_step.append(si)
        else:
            for l in range(LAYERS):
                if l in remote_layers:
                    continue
                for e in st[2][l]:
                    acc_keys.append(l * EXPERTS + e); acc_step.append(si)
    k = np.array(acc_keys, np.int64); s = np.array(acc_step, np.int64)
    nxt = np.full(len(k), 1 << 40, np.int64)
    first = np.full(NK, 1 << 40, np.int64)
    last = {}
    order = np.argsort(k, kind="stable")
    ks, ss = k[order], s[order]
    same = ks[1:] == ks[:-1]
    nn = np.full(len(k), 1 << 40, np.int64)
    nn[:-1][same] = ss[1:][same]
    nxt[order] = nn
    # first occurrence per key
    starts = np.r_[0, np.where(~same)[0] + 1]
    first[ks[starts]] = ss[starts]
    # also next ACCESS INDEX for exact belady
    nidx = np.full(len(k), 1 << 40, np.int64)
    oi = order
    nni = np.full(len(k), 1 << 40, np.int64)
    nni[:-1][same] = oi[1:][same]
    nidx[oi] = nni
    firsti = np.full(NK, 1 << 40, np.int64)
    firsti[ks[starts]] = oi[starts]
    return nxt.tolist(), first.tolist(), nidx.tolist(), firsti.tolist()


# ---------------------------------------------------------------- caches
class LRU:
    def __init__(self, cap, **_):
        self.cap = cap; self.d = OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def __len__(self):
        return len(self.d)

    def touch(self, k, sim):
        self.d.move_to_end(k)

    def insert(self, k, sim, protect):
        self.d[k] = None
        while len(self.d) > self.cap:
            for v in self.d:
                if v not in protect and v != k:
                    break
            del self.d[v]; sim.evicted(v)


class GDS:
    """GreedyDual: H = clock + cost(k); evict min H (cost-aware LRU; equals LRU when costs are equal)."""
    def __init__(self, cap, **_):
        self.cap = cap; self.h = {}; self.heap = []; self.clock = 0.0; self.seq = 0

    def __contains__(self, k):
        return k in self.h

    def __len__(self):
        return len(self.h)

    def _set(self, k, sim):
        self.seq += 1
        v = self.clock + min(sim.cost_rel[k], sim.gds_cap) + self.seq * 1e-9
        self.h[k] = v; heapq.heappush(self.heap, (v, k))

    def touch(self, k, sim):
        self._set(k, sim)

    def insert(self, k, sim, protect):
        self._set(k, sim)
        held = []
        while len(self.h) > self.cap:
            v, q = heapq.heappop(self.heap)
            if self.h.get(q) != v:
                continue
            if q in protect or q == k:
                held.append((v, q)); continue
            del self.h[q]; self.clock = v; sim.evicted(q)
        for x in held:
            heapq.heappush(self.heap, x)


class Belady:
    def __init__(self, cap, **_):
        self.cap = cap; self.s = set(); self.heap = []

    def __contains__(self, k):
        return k in self.s

    def __len__(self):
        return len(self.s)

    def touch(self, k, sim):
        heapq.heappush(self.heap, (-sim.nextpos[k], k))

    def insert(self, k, sim, protect):
        self.s.add(k); heapq.heappush(self.heap, (-sim.nextpos[k], k))
        held = []
        while len(self.s) > self.cap:
            nn, q = heapq.heappop(self.heap)
            if q not in self.s or -nn != sim.nextpos[q]:
                continue
            if q in protect or q == k:
                held.append((nn, q)); continue
            self.s.discard(q); sim.evicted(q)
        for x in held:
            heapq.heappush(self.heap, x)


class Sampled:
    """Evict min score over a random 64-key sample of residents (score from the simulator's key stats)."""
    def __init__(self, cap, score, seed=0, **_):
        self.cap = cap; self.keys = []; self.pos = {}; self.rng = random.Random(seed); self.score = score

    def __contains__(self, k):
        return k in self.pos

    def __len__(self):
        return len(self.keys)

    def touch(self, k, sim):
        pass

    def insert(self, k, sim, protect):
        if k not in self.pos:
            self.pos[k] = len(self.keys); self.keys.append(k)
        while len(self.keys) > self.cap:
            n = len(self.keys)
            cand = [self.keys[self.rng.randrange(n)] for _ in range(64)]
            cand = [c for c in cand if c != k and c not in protect] or [c for c in cand if c != k]
            v = min(cand, key=lambda q: self.score(q, sim))
            i = self.pos.pop(v); last = self.keys.pop()
            if i < len(self.keys):
                self.keys[i] = last; self.pos[last] = i
            sim.evicted(v)


# ---------------------------------------------------------------- simulator
class Sim:
    def __init__(self, cfg, ram_gb=75, policy="lru", prefetch="none", k=8, theta_mult=1.0, theta_abs=0.0, prio="value",
                 t7_subset=False, nvme_frac=0.6, m3ram_tier=False, remote=None, remote_state="idle",
                 nextuse=None, cycle_alpha=0.3, seed=0, ram_experts=None, gds_cap=1e9, pin_layers=(), prefetch_max_ms=1e9):
        self.cfg = cfg; self.tr = trace(); self.gds_cap = gds_cap; self.pf_max = prefetch_max_ms
        self.c = cfg["compute_ms_per_layer"]["value"]; self.rf = cfg["router_frac"]["value"]
        self.steal = cfg["parse"][cfg["parse"]["mode"]]["gpu_steal_ms"]
        self.cap = ram_experts if ram_experts is not None else int(ram_gb * 1e9 // EXPERT_BYTES)
        self.policy, self.prefetch, self.k, self.prio = policy, prefetch, k, prio
        self.theta = theta_mult * self.steal + theta_abs
        self.theta_mult = theta_mult
        # ---- tier placement of every key ----
        tiers = ["nvme"] * NK
        freq = self.tr["freq_train"]
        if t7_subset:
            order = np.argsort(-freq, kind="stable")
            n_nvme = int(nvme_frac * NK)
            for q in order[n_nvme:].tolist():
                tiers[q] = "t7"
            if m3ram_tier:
                m3cap = cfg["remote"]["m3"]["expert_capacity"]
                for q in order[n_nvme:n_nvme + m3cap].tolist():
                    tiers[q] = "m3ram"
        self.tier = tiers
        self.svc = {t: C.expert_ms(t, cfg) for t in ("nvme", "t7", "m3ram")}
        if remote_state == "busy":
            self.svc["m3ram"] += cfg["remote"]["m3"]["rtt_ms_busy"] - cfg["remote"]["m3"]["rtt_ms_idle"]
        self.cost = [self.svc[t] for t in tiers]
        self.cost_rel = [x / self.svc["nvme"] for x in self.cost]
        # ---- remote pipeline stages: {layer: node} ----
        self.node = [None] * LAYERS
        self.remote_layers = set()
        for nd, layers in (remote or {}).items():
            for l in layers:
                self.node[l] = nd; self.remote_layers.add(l)
        self.remote_state = remote_state
        # layers pinned resident in M5 RAM (all 384 experts; costs 384 experts of capacity each)
        self.pinned = set(pin_layers)
        self.remote_layers |= self.pinned
        self.cap -= EXPERTS * len(self.pinned)
        # ---- per-key stats for scores ----
        self.last = [-(1 << 30)] * NK
        self.ema = [0.0] * NK; self.ema_t = [0] * NK
        self.emas = [0.0] * NK; self.emas_t = [0] * NK
        self.pnext = [0.0] * NK; self.pnext_row = [-1] * NK
        self.cycle_alpha = cycle_alpha
        self.logf = np.log1p(freq).tolist()
        self.nu = nextuse
        if policy == "belady" or policy == "nextuse_train":
            self.nxt_step, first, self.nidx, firsti = next_use_steps(self.tr, frozenset(self.remote_layers))
            self.nextpos = firsti
        self.cache = self._make_cache(policy, seed)
        self.rng = random.Random(seed)

    # ---- scores for sampled eviction ----
    def _make_cache(self, policy, seed):
        if policy == "lru":
            return LRU(self.cap)
        if policy == "gds":
            return GDS(self.cap)
        if policy == "belady":
            return Belady(self.cap)
        if policy in ("cycle", "cycle_cost"):
            a = self.cycle_alpha
            costw = policy == "cycle_cost"

            def score(q, sim):
                u = sim.ema[q] * (1 - a) ** (sim.tok - sim.ema_t[q])
                d = ((q // EXPERTS) - sim.cur_layer - 1) % LAYERS + 1
                s = (u + sim.boost(q)) / d
                return s * sim.cost_rel[q] if costw else s
            return Sampled(self.cap, score, seed)
        if policy in ("nextuse", "nextuse_cost"):
            w = self.nu["w"]; costw = policy == "nextuse_cost"
            af, aslow = self.nu["alpha_fast"], self.nu["alpha_slow"]

            def score(q, sim):
                age = sim.tok - sim.last[q]
                ef = sim.ema[q] * (1 - af) ** (sim.tok - sim.ema_t[q])
                es = sim.emas[q] * (1 - aslow) ** (sim.tok - sim.emas_t[q])
                d = (((q // EXPERTS) - sim.cur_layer - 1) % LAYERS + 1) / LAYERS
                pn = sim.pnext[q] if sim.pnext_row[q] >= sim.row else 0.0
                z = (w[0] + w[1] * math.log1p(age) + w[2] * ef + w[3] * es + w[4] * sim.logf[q] + w[5] * d
                     + w[6] * pn)
                p = 1.0 / (1.0 + math.exp(-z))
                return p * sim.cost_rel[q] if costw else p
            return Sampled(self.cap, score, seed)
        raise ValueError(policy)

    def boost(self, q):
        return self.pnext[q] if self.pnext_row[q] >= self.row else 0.0

    def evicted(self, q):
        if q in self.pref_live:
            self.pref_live.discard(q); self.st["pref_wasted"] += 1

    # ---- key stats update on access ----
    def access_stats(self, q):
        t = self.tok
        self.last[q] = t
        if self.policy in ("cycle", "cycle_cost"):
            a = self.cycle_alpha
            self.ema[q] = self.ema[q] * (1 - a) ** (t - self.ema_t[q]) + a; self.ema_t[q] = t
        elif self.nu is not None:
            af, aslow = self.nu["alpha_fast"], self.nu["alpha_slow"]
            if self.ema_t[q] != t:
                self.ema[q] = self.ema[q] * (1 - af) ** (t - self.ema_t[q]) + af; self.ema_t[q] = t
                self.emas[q] = self.emas[q] * (1 - aslow) ** (t - self.emas_t[q]) + aslow; self.emas_t[q] = t

    # ---- channels ----
    def _land(self, ch, key, t_end):
        self.inflight.discard(key)
        if key in self.remote_layers_keys:
            return
        if key not in self.cache:
            self.cache.insert(key, self, self.protect)
            if self.cur_kind[ch] == "p":
                self.pref_live.add(key)
        self.steal_acc += self.steal
        self.busy[ch] += t_end - self.cur_start[ch]

    def advance(self, to):
        for ch in self.channels:
            while True:
                cur = self.cur[ch]
                if cur is not None:
                    if cur[1] <= to:
                        self.cur[ch] = None; self.free[ch] = cur[1]
                        self._land(ch, cur[0], cur[1])
                        continue
                    break
                dq = self.dq[ch]
                if dq:
                    key, issued = dq[0]
                    if key in self.cache or key in self.inflight:
                        dq.popleft(); continue
                    start = max(self.free[ch], issued)
                    if start > to:
                        break
                    dq.popleft()
                    self.cur[ch] = (key, start + self.svc[ch]); self.cur_kind[ch] = "d"; self.cur_start[ch] = start
                    self.inflight.add(key); self.st["loads_demand_" + ch] += 1
                    continue
                pq = self.pq[ch]
                started = False
                while pq:
                    p0, p1, dl, issued, key = pq[0]
                    if (key in self.cache or key in self.inflight or dl < self.pos
                            or self.queued.get(key) != (p0, p1)):
                        heapq.heappop(pq)
                        if self.queued.get(key) == (p0, p1):
                            del self.queued[key]
                        continue
                    start = max(self.free[ch], issued)
                    if start > to:
                        break
                    heapq.heappop(pq); del self.queued[key]
                    self.cur[ch] = (key, start + self.svc[ch]); self.cur_kind[ch] = "p"; self.cur_start[ch] = start
                    self.inflight.add(key); self.st["loads_prefetch"] += 1
                    started = True
                    break
                if not started:
                    break

    def issue(self, row_target, layer_target, ranked, cal, now, h_scale=1.0):
        """Queue prefetches for (row_target, layer_target) from a ranked id list with per-rank hit probs."""
        if layer_target in self.remote_layers:
            return
        dl = row_target * LAYERS + layer_target
        base = layer_target * EXPERTS
        for r in range(min(self.k, len(ranked))):
            e = ranked[r]
            if e < 0:
                break
            q = base + e
            p = cal[r] * h_scale
            if row_target == self.row + 1 or row_target == self.row:
                if p > self.pnext[q] or self.pnext_row[q] < row_target:
                    self.pnext[q] = p; self.pnext_row[q] = row_target
            if q in self.cache or q in self.inflight or self.cost[q] > self.pf_max:
                continue
            val = p * self.cost[q]
            if val < self.theta:
                continue
            pr = (-val, dl) if self.prio == "value" else (dl, -val)
            old = self.queued.get(q)
            if old is not None and old <= pr:
                continue
            ch = self.tier[q]
            self.queued[q] = pr
            heapq.heappush(self.pq[ch], (pr[0], pr[1], dl, now, q))

    def run(self):
        tr = self.tr; P = preds() if self.prefetch != "none" else {}
        ids = tr["ids"]; split = tr["tok_split"]
        self.channels = ["nvme", "t7", "m3ram"]
        self.free = {c: 0.0 for c in self.channels}; self.cur = {c: None for c in self.channels}
        self.cur_kind = {c: "" for c in self.channels}; self.cur_start = {c: 0.0 for c in self.channels}
        self.busy = {c: 0.0 for c in self.channels}
        self.dq = {c: deque() for c in self.channels}; self.pq = {c: [] for c in self.channels}
        self.queued = {}; self.inflight = set(); self.pref_live = set(); self.protect = set()
        self.remote_layers_keys = set()
        self.steal_acc = 0.0
        st = self.st = {k: 0 for k in ("loads_demand_nvme", "loads_demand_t7", "loads_demand_m3ram", "loads_prefetch",
                                       "pref_used", "pref_wasted", "prefill_loads")}
        acc = {s: {"tok": 0, "time": 0.0, "stall": 0.0, "steal": 0.0, "hop": 0.0, "remote_extra": 0.0,
                   "miss_nvme": 0, "miss_t7": 0, "miss_m3ram": 0, "miss_inflight": 0} for s in (0, 1, -1)}
        cfgr = self.cfg["remote"]
        hidden = cfgr["hidden_bytes"]

        def hop_ms(a, b):
            nd = a if a is not None else b
            r = cfgr[nd]
            rtt = r["rtt_ms_busy"] if self.remote_state == "busy" else r["rtt_ms_idle"]
            bw = r.get("link_GBps", 1.01)
            return rtt / 2 + hidden / (bw * 1e9) * 1e3

        pf = self.prefetch
        use_d = {"d1": 1, "d2": 2, "xtokAB+d1": 1, "xtokABp+d1": 1, "h2+d1": 1, "h3+d1": 1, "oracle_d1": 1}.get(pf)
        use_ab = pf in ("xtokAB", "xtokAB+d1", "xtokABp", "xtokABp+d1", "h2+d1", "h3+d1", "oracle_xtok")
        ab_key = "xtok_ABp" if "ABp" in pf else "xtok_AB"
        use_a = pf in ("xtokA",)
        horizon = {"h2+d1": 2, "h3+d1": 3}.get(pf, 1)
        oracle = pf.startswith("oracle")
        one6 = [1.0] * TOPK
        if use_d and not oracle:
            Pd, Cd = P[f"pred_d{use_d}"], P[f"cal_d{use_d}"].tolist()
        if use_ab and not oracle:
            Pab, Cab = P["pred_" + ab_key], P["cal_" + ab_key].tolist()
            Ph = {h: (P[f"pred_xtok_AB_h{h}"], P[f"cal_xtok_AB_h{h}"].tolist()) for h in range(2, horizon + 1)}
        if use_a:
            Pa, Ca = P["pred_xtok_A"], P["cal_xtok_A"].tolist()
        nidx_i = 0
        now = 0.0
        self.tok = 0; self.row = -1; self.pos = 0; self.cur_layer = 0
        T = ids.shape[0]
        for si, stp in enumerate(tr["steps"]):
            self.tok = si
            if stp[0] == 0:                      # prefill: warm (untimed), reset channels
                for ch in self.channels:
                    self.dq[ch].clear(); self.pq[ch].clear(); self.cur[ch] = None; self.free[ch] = now
                self.queued.clear(); self.inflight.clear()
                p0 = stp[1]
                self.row = p0; self.pos = p0 * LAYERS
                for l in range(LAYERS):
                    if l in self.remote_layers:
                        continue
                    self.cur_layer = l
                    for e in stp[2][l]:
                        q = l * EXPERTS + e
                        if self.policy == "belady":
                            self.nextpos[q] = self.nidx[nidx_i]
                        nidx_i += 1
                        self.access_stats(q)
                        if q in self.cache:
                            self.cache.touch(q, self)
                        else:
                            st["prefill_loads"] += 1
                            self.cache.insert(q, self, ())
                continue
            r = stp[1]; self.row = r
            sp = int(split[r]); a = acc[sp]
            t0 = now
            row_ids = ids[r].tolist()
            if use_d and not oracle:
                pd_row = Pd[r].tolist()
            prev_node = self.node[LAYERS - 1]
            for l in range(LAYERS):
                self.cur_layer = l; self.pos = r * LAYERS + l
                nd = self.node[l]
                if nd != prev_node:
                    hm = hop_ms(prev_node, nd); now += hm; a["hop"] += hm
                prev_node = nd
                if nd is not None:
                    fac = cfgr[nd]["compute_factor"]
                    now += self.c * fac; a["remote_extra"] += self.c * (fac - 1)
                    self.advance(now)
                    continue
                if l in self.pinned:
                    now += self.c
                    self.advance(now)
                    continue
                t_router = now + self.c * self.rf
                # ---- prefetch issue at this router time ----
                if pf != "none":
                    if l == 0 and use_ab:
                        if oracle:
                            for lt in range(1, LAYERS):
                                self.issue(r, lt, row_ids[lt], one6, t_router)
                        else:
                            prow = Pab[r].tolist()
                            for lt in range(1, LAYERS):
                                self.issue(r, lt, prow[lt], Cab[lt], t_router)
                            for h in range(2, horizon + 1):
                                rr = r + h - 1
                                if rr < T and tr["tok_decode"][rr] and tr["tok_req"][rr] == tr["tok_req"][r]:
                                    ph, ch_ = Ph[h]
                                    prow = ph[rr].tolist()
                                    for lt in range(LAYERS):
                                        self.issue(rr, lt, prow[lt], ch_[lt], t_router)
                    if use_d and l + use_d < LAYERS:
                        lt = l + use_d
                        if oracle:
                            self.issue(r, lt, row_ids[lt], one6, t_router)
                        else:
                            self.issue(r, lt, pd_row[lt], Cd[lt], t_router)
                    if use_a and l == LAYERS - 1 and r + 1 < T and tr["tok_decode"][r + 1] \
                            and tr["tok_req"][r + 1] == tr["tok_req"][r]:
                        prow = Pa[r + 1].tolist()
                        for lt in range(LAYERS):
                            self.issue(r + 1, lt, prow[lt], Ca[lt], t_router)
                self.advance(t_router)
                need = [l * EXPERTS + e for e in row_ids[l]]
                self.protect = set(need)
                missing = []
                for q in need:
                    if self.policy == "belady":
                        self.nextpos[q] = self.nidx[nidx_i]
                    nidx_i += 1
                    if q in self.cache:
                        continue
                    missing.append(q)
                    if q in self.inflight:
                        a["miss_inflight"] += 1
                    else:
                        a["miss_" + self.tier[q]] += 1
                        self.dq[self.tier[q]].append((q, t_router))
                ready = t_router
                if missing:
                    self.advance(t_router)
                    while any(q not in self.cache for q in missing):
                        tn = min(self.cur[ch][1] for ch in self.channels if self.cur[ch] is not None)
                        self.advance(tn); ready = tn
                a["stall"] += ready - t_router
                for q in need:
                    if q in self.pref_live:
                        self.pref_live.discard(q); st["pref_used"] += 1
                    self.access_stats(q)
                    self.cache.touch(q, self)
                self.protect = set()
                now = ready + self.c * (1 - self.rf)
                if self.steal_acc:
                    now += self.steal_acc; a["steal"] += self.steal_acc; self.steal_acc = 0.0
            if prev_node is not None and self.node[0] is None:
                pass
            a["tok"] += 1; a["time"] += now - t0
        out = {}
        for s, name in ((0, "train"), (1, "heldout"), (-1, "none")):
            a = acc[s]
            n = max(1, a["tok"])
            if a["tok"] == 0:
                continue
            out[name] = {"tokens": a["tok"], "s_per_token": a["time"] / n / 1e3,
                         "overhead_s_per_token": (a["time"] / n - LAYERS * self.c) / 1e3,
                         "stall_s_per_token": a["stall"] / n / 1e3, "steal_s_per_token": a["steal"] / n / 1e3,
                         "hop_s_per_token": a["hop"] / n / 1e3, "remote_extra_s_per_token": a["remote_extra"] / n / 1e3,
                         "misses_per_token": (a["miss_nvme"] + a["miss_t7"] + a["miss_m3ram"] + a["miss_inflight"]) / n,
                         "miss_nvme": a["miss_nvme"] / n, "miss_t7": a["miss_t7"] / n, "miss_m3ram": a["miss_m3ram"] / n,
                         "miss_inflight": a["miss_inflight"] / n}
        dtok = sum(acc[s]["tok"] for s in acc)
        ttot = sum(acc[s]["time"] for s in acc)
        out["all"] = {"loads_prefetch_per_token": st["loads_prefetch"] / dtok,
                      "pref_used_per_token": st["pref_used"] / dtok, "pref_wasted_per_token": st["pref_wasted"] / dtok,
                      "util": {c: self.busy[c] / max(ttot, 1e-9) for c in self.channels}}
        return out


def run_one(spec):
    """spec: dict(cfg_over={...}, sim={...}); returns spec + result (picklable for multiprocessing)."""
    cfg = C.make(**spec.get("cfg_over", {}))
    kw = dict(spec.get("sim", {}))
    name = kw.pop("nextuse_model", None)
    if kw.get("policy", "lru").startswith("nextuse"):
        nu = json.load(open(f"{OUT}/nextuse.json"))
        kw["nextuse"] = nu["models"][name] if name else nu["model"]
    import time
    t = time.time()
    res = Sim(cfg, **kw).run()
    return {**spec, "result": res, "sim_seconds": round(time.time() - t, 1)}


if __name__ == "__main__":
    import sys
    spec = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"sim": {"policy": "lru"}}
    r = run_one(spec)
    print(json.dumps(r, indent=1))
