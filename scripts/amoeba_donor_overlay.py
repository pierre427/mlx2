#!/usr/bin/env python3
"""aMoEba donor-wins box overlay with late-layer sinks (Pierre 2026-11-19 ~21:20-21:50). Unqualified research.

Pair (the bridge lab's physics pair, shared tokenizer): host Studio-Operator-Qwen3.5-9B-4bit, donor
Qwen3.8-27B-BF16-q8-mlx2, topic sci.physics. Same selection as the bridge lab (scripts/amoeba_bridge_lab.py
topic_selection, --split-manifest, --split-check): 160 train / 40 val probe texts, the 200 spent held texts ("dev",
development data) and, read here for the first time, the 297 untouched reserved final-test texts ("held") that the
split manifest names. Every number names <artifact>@<fingerprint> and the mlx2 commit (run.json).

Mechanism (static weights at serve time, no donor):
  overlay neurons  new SiLU-GLU units added to the host MLP at a layer, beside its own units: unit u has a gate row,
                   an up row and a write (down) row, out += g_t * sum_u silu(x . wg_u) (x . wu_u) wd_u, x = the MLP's
                   own normalised input. Base weights stay frozen; only overlay parameters train.
  box overlay      at the host's box layers: the host box chain found from HOST activations on the train texts at
                   the donor-better positions (topic mass there minus English mass on the retain texts, the lab's
                   select rule, layers at least 2 apart, below the sink layers). Unit u copies one host box unit h:
                   wg_u = wg_h, wu_u = --read-scale x wu_h (activation --read-scale x the host unit's), wd_u =
                   --write-scale x the host unit's down column (near zero: the overlay starts as a near no-op; 0 =
                   exact no-op). A layer asking for more units than the ranking has distinct units repeats the ranking
                   with --dup-jitter relative noise on the read rows (identical copies would get identical updates).
  sinks            the same units in the last --n-sink-layers host layers (a reader for what the box overlay
                   writes: frozen late layers may not read new writes), initialised from host late units ranked by
                   topic contrast x next-token promotion (the lab's H5 sink rule: the unit's down column through the
                   final-norm gain and unembedding, mean z of the train continuations' next tokens at the donor-better
                   positions); trained jointly with the box overlay.
  gate             per overlay layer g_t = sigmoid(w . z_t + b), z_t = the standardised activations of that layer's
                   first --gate-units distinct HOST units the overlay copies (frozen host rows: the topic box itself),
                   w, b initialised by a logistic fit topic (donor-better positions) vs English and then trained;
                   --gate-l1 on the English retain step. Closed gate (CLOSED) = no overlay write at all: the host.
  core delta       (arm ov_sink_core only; Pierre ~21:50 "lightly fine tune english and the core") per overlay layer a
                   rank --core-rank delta on the down columns of the --core-units host units with the largest English
                   mass that are not topic-contrast units (topic/English < --min-ratio) and not copied by the overlay:
                   out += (h_core @ U) @ V, V = 0 at init, --core-lr-mult, L2 --core-l2. ALWAYS OPEN (not gated): its
                   drift is paid on every request; the gate-closed cost of that arm is reported separately.
  tile_sft         the bridge lab's tile (gated SiLU tile at every host layer's output, w_out = 0, blend-in cap
                   --norm-budget x the layer increment), host-only CE: the 0c bar, rerun at the same budget.

Selection (stage arms): per train text and scored position, donor-better = donor log p(gold) - host log p(gold) >=
--win-margin nats. Losses: kl_win = KL(donor || arm) at donor-better positions only; kl_all = at every position;
ce_gold = cross-entropy on the text (host-only); sft_donor = cross-entropy on the donor's own greedy continuation of
the first --resp-prefix tokens of each donor-better text (text-level: the donor's nats/byte on the gold suffix lower
than the host's by --text-margin). Donor targets are the donor's top --topk log-probs plus one tail bucket (KL of the
coarsened distributions, a lower bound on the full KL; exact on the top-k support), stored once, then the donor is
unloaded. Every arm selects its checkpoint by validation cross-entropy on the gold text (val_ce, never a donor output:
review v4), among checkpoints whose validation English drift KL(host || arm) is within --drift-cap (the lab's
pick_epoch; adaptive drift weight).

Arms (stage arms; equal trainable budget --budget-vecs D-vectors: an overlay unit = 3, a tile unit = 2)
  ov_sink_win      box overlay + sink, donor-wins KL                      (Pierre's idea)
  ov_win           box overlay only, donor-wins KL                        (does the sink matter)
  sink_win         sinks only, donor-wins KL                              (does the sink carry it alone)
  ov_sink_all      box overlay + sink, KL to the donor at every position  (does selecting wins matter)
  ov_sink_gold     box overlay + sink, host-only CE on the text           (donor vs plain capacity)
  ov_sink_rand     box overlay at random layers / units + sink, donor-wins KL (does host alignment matter)
  ov_sink_sft      box overlay + sink, CE on the donor's responses         (the SFT form of donor-wins)
  tile_sft         the lab tile, host-only CE, same budget                (the bar to beat; 0c +0.0388 on dev)
  ov_sink_core     ov_sink_win + the light core delta (conditional; staged, not in the main window)
Stage scale (Pierre ~21:35-21:45): the ov_sink configuration on the whole fit-side physics corpus
(scripts/amoeba_overlay_corpus.py chunks), host-only CE (and --scale-kl-mults donor-wins KL on the same chunks), at
per-layer multiples --scale-mults of the base per-layer counts (every overlay layer gets m x its base count); equal
optimizer steps per size (a timing probe sets them so the sweep fits --train-budget-s); gentle insertion (write init
--write-scale, linear warmup --warmup-frac, write LR x --write-lr-mult); train-chunk loss, val CE, dev CE and drift
at each checkpoint (train vs held curve; the 297 are read once, at the end).
Report per arm / size: held and dev nats/byte, paired SE, source-group bootstrap CI and the flag against
--min-useful-npb (scripts/amoeba_paired_stats.py), gap closed, English drift (val and test), the closed-gate cost
(nats/byte and forward ms vs the host), trainable parameters, selected checkpoint. Overlays are saved
(OUT/overlays/<arm>/overlay.safetensors + overlay.json, a route receipt template); load_overlay() puts one on a host.
Serving status: loadable and runnable in research harnesses through load_overlay / install (opt-in, gate receipts);
NOT wired into mlx2 serving routes. Path: an adapter-level MLP hook keyed by the overlay manifest's host fingerprint,
route receipt {route: amoeba_overlay, qualification: unqualified, gate_open_fraction}.
Gradients through the host run inside mlx2.runtime.models.differentiable.reference_path() (no VJP for Metal kernels).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))
import amoeba_bridge_lab as LAB  # noqa: E402  (module-level selection / split / epoch-pick helpers)
import amoeba_paired_stats as PS  # noqa: E402

CORE = LAB.CORE
VEC_PER_UNIT = 3   # gate row, up row, write row
VEC_PER_TILE = 2   # w_in column, w_out row
_STATE = {}


# ---------------------------------------------------------------------------------------------- selection masks
def win_positions(host_gold, donor_gold, margin):
    """bool [n]: positions where the donor's log p(gold) beats the host's by >= margin nats"""
    return (np.asarray(donor_gold, np.float64) - np.asarray(host_gold, np.float64)) >= margin


def text_win(host_nats, donor_nats, nbytes, margin):
    """the donor's nats/byte on the text is lower than the host's by >= margin"""
    return (host_nats - donor_nats) / max(nbytes, 1) >= margin


# ---------------------------------------------------------------------------------------------- budget accounting
def overlay_counts(budget_vecs, box_layers, sink_layers, sink_share):
    """{layer: units} spending floor(budget_vecs / 3) units: sink_share of them over the sink layers, the rest over
    the box layers (either list may be empty), each split evenly (remainder to the first layers)"""
    n = budget_vecs // VEC_PER_UNIT
    ns = int(round(n * sink_share)) if box_layers and sink_layers else (n if sink_layers else 0)
    nb = n - ns
    out = {}
    for ls, k in ((list(box_layers), nb), (list(sink_layers), ns)):
        for i, l in enumerate(ls):
            out[int(l)] = k // len(ls) + (1 if i < k % len(ls) else 0)
    return out


def tile_units(budget_vecs, n_layers):
    return max(1, budget_vecs // (VEC_PER_TILE * n_layers))


def scale_counts(base, mult):
    return {l: int(round(n * mult)) for l, n in base.items()}


def overlay_param_count(counts, D, gate_units=None, core=None):
    """trainable parameters: 3 D per unit, gate w (gate units) + b per layer, core U (k x r) + V (r x D)"""
    p = sum(VEC_PER_UNIT * D * n for n in counts.values())
    for l in counts:
        p += (gate_units or {}).get(l, 0) + 1
        if core and l in core:
            k, r = core[l]
            p += k * r + r * D
    return int(p)


def overlay_train_gb(n_params, bytes_per=4, adam=True, snapshots=2):
    """parameters + gradients + Adam (two moments) + kept checkpoints (the current pick and the lowest-drift one;
    init is rebuilt, not kept), GB"""
    return n_params * bytes_per * (1 + 1 + (2 if adam else 0) + snapshots) / 1e9


def activation_gb(counts, tokens, bytes_per=4, factor=6):
    """rough backward working set of the overlay hidden activations: factor tensors of [tokens, units] per layer"""
    return sum(n for n in counts.values()) * tokens * bytes_per * factor / 1e9


# ---------------------------------------------------------------------------------------------- unit ranking
def rank_units(mt, me, min_ratio):
    """one layer: host units ordered by topic-minus-English mass, eligible (mt >= min_ratio me) first"""
    score = mt - me
    elig = mt >= min_ratio * me
    key = np.where(elig, 0, 1)
    return np.lexsort((-score, key)), int(elig.sum())


def select_layers(mt, me, k, nl, min_ratio, allowed):
    """the lab's layer rule over allowed layers: score = sum of the top-k eligible units' (mt - me), the nl best
    layers at least 2 apart"""
    sc = []
    for l in range(len(mt)):
        elig = mt[l] >= min_ratio * me[l]
        s = np.sort(np.where(elig, mt[l] - me[l], -np.inf))[::-1][:k]
        s = s[np.isfinite(s)]
        sc.append(float(s.sum()) if len(s) else -np.inf)
    sc = np.array(sc)
    layers = []
    for l in np.argsort(-sc):
        if int(l) in allowed and np.isfinite(sc[l]) and all(abs(int(l) - m) >= 2 for m in layers):
            layers.append(int(l))
        if len(layers) == nl:
            break
    return sorted(layers), sc


def sink_order(order, cand, score):
    """sink ranking: candidates with a positive score by score, then every other unit in contrast order"""
    order, cand, score = np.asarray(order), np.asarray(cand), np.asarray(score)
    pos = np.argsort(-score, kind="stable")
    good = cand[pos[score[pos] > 0]]
    gs = set(good.tolist())
    return np.concatenate([good, np.array([u for u in order if u not in gs], dtype=order.dtype)]).astype(np.int64)


def expand_units(order, n):
    """n unit ids from a ranking: the ranking repeated when n exceeds it; copy index 0 for the first pass"""
    order = np.asarray(order)
    reps = -(-n // len(order))
    ids = np.tile(order, reps)[:n]
    copy = np.repeat(np.arange(reps), len(order))[:n]
    return ids, copy


def fit_gate(z1, z0, l2=1e-2, iters=400, lr=0.5):
    """balanced logistic regression (numpy) topic (z1) vs English (z0) on standardised features; (w, b, acc)"""
    X = np.concatenate([z1, z0]).astype(np.float64)
    y = np.concatenate([np.ones(len(z1)), np.zeros(len(z0))])
    sw = np.where(y == 1, 0.5 / max(len(z1), 1), 0.5 / max(len(z0), 1))
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = p - y
        w -= lr * (X.T @ (sw * g) + l2 * w)
        b -= lr * float((sw * g).sum())
    p = 1 / (1 + np.exp(-(X @ w + b)))
    acc = float((sw * ((p >= 0.5) == (y == 1))).sum())
    return w.astype(np.float32), np.float32(b), acc


# ---------------------------------------------------------------------------------------------- overlay
def build_overlay(spec, rows, D, read_scale=1.1, write_scale=0.01, dup_jitter=0.1, seed=0, dtype="float32",
                  core=None):
    """(params, consts) for spec {layer: {"units": host ids [n], "copy": copy index [n], "gate_units": host ids,
    "gate_mu", "gate_sd", "gate_w", "gate_b"}}. rows(l, ids) -> (wg [k, D], wu [k, D], wd [k, D]) host rows (numpy
    float32; wd = the down columns, transposed). core {layer: {"units": ids, "rank": r}} adds the core delta."""
    import mlx.core as mx
    dt = {"float32": mx.float32, "bfloat16": mx.bfloat16}[dtype]
    g = np.random.default_rng(seed)
    P, C = {}, {}
    for l, s in spec.items():
        k = f"L{l}"
        if len(s["units"]):
            wg, wu, wd = rows(l, s["units"])
            jit = (np.asarray(s.get("copy", np.zeros(len(s["units"])))) > 0)[:, None]
            if dup_jitter > 0 and jit.any():
                for w_ in (wg, wu):
                    nz = g.standard_normal(w_.shape).astype(np.float32)
                    nz *= (np.linalg.norm(w_, axis=1, keepdims=True) / np.sqrt(D)) * dup_jitter
                    w_ += np.where(jit, nz, 0)
            P[k + ".wg"] = mx.array(wg.astype(np.float32)).astype(dt)
            P[k + ".wu"] = mx.array((read_scale * wu).astype(np.float32)).astype(dt)
            P[k + ".wd"] = mx.array((write_scale * wd).astype(np.float32)).astype(dt)
            gg, gu, _ = rows(l, s["gate_units"])
            C[k + ".gg"], C[k + ".gu"] = mx.array(gg), mx.array(gu)
            C[k + ".mu"], C[k + ".sd"] = mx.array(s["gate_mu"]), mx.array(s["gate_sd"])
            P[k + ".gw"] = mx.array(s["gate_w"])
            P[k + ".gb"] = mx.array(np.array([s["gate_b"]], np.float32))
        if core and l in core:
            cg, cu_, _ = rows(l, core[l]["units"])
            r = core[l]["rank"]
            C[k + ".cg"], C[k + ".cu"] = mx.array(cg), mx.array(cu_)
            P[k + ".U"] = mx.array((g.standard_normal((len(core[l]["units"]), r)) / np.sqrt(len(core[l]["units"])))
                                   .astype(np.float32))
            P[k + ".V"] = mx.zeros((r, D))
    return P, C


def overlay_layers(P, C):
    return sorted({int(k.split(".")[0][1:]) for k in list(P) + list(C)})


GATE_REC = {"on": False, "g": []}
CLOSED = [False]


def overlay_delta(P, C, l, x):
    """[B, T, D] float32 overlay write at layer l for MLP input x, or None (nothing at this layer / gate closed and
    no core delta)"""
    import mlx.core as mx
    import mlx.nn as nn
    k = f"L{l}"
    out = None
    xf = x.astype(mx.float32)
    if (k + ".wg") in P and not CLOSED[0]:
        xg = xf.astype(P[k + ".wg"].dtype)
        h = nn.silu(xg @ P[k + ".wg"].T) * (xg @ P[k + ".wu"].T)
        o = (h @ P[k + ".wd"]).astype(mx.float32)
        a = mx.stop_gradient(nn.silu(xf @ C[k + ".gg"].T) * (xf @ C[k + ".gu"].T))
        z = (a - C[k + ".mu"]) / C[k + ".sd"]
        gt = mx.sigmoid(z @ P[k + ".gw"] + P[k + ".gb"])[..., None]
        if GATE_REC["on"]:
            GATE_REC["g"].append(gt)
        out = gt * o
    if (k + ".U") in P:
        hc = mx.stop_gradient(nn.silu(xf @ C[k + ".cg"].T) * (xf @ C[k + ".cu"].T))
        d = (hc @ P[k + ".U"]) @ P[k + ".V"]
        out = d if out is None else out + d
    return out


def core_l2(P):
    import mlx.core as mx
    t = [((P[k] @ P[k.replace(".U", ".V")]) ** 2).sum() for k in P if k.endswith(".U")]
    return mx.stack(t).sum() if t else mx.array(0.0)


# ---------------------------------------------------------------------------------------------- tiles (lab)
def build_tiles(layers, n, D, seed):
    import mlx.core as mx
    g = np.random.default_rng(seed)
    P = {}
    for l in layers:
        P[f"T{l}.w_in"] = mx.array((g.standard_normal((D, n)) / np.sqrt(D)).astype(np.float32))
        P[f"T{l}.w_out"] = mx.zeros((n, D))
        P[f"T{l}.gw"] = mx.zeros((D,))
        P[f"T{l}.gb"] = mx.zeros((1,))
    return P


def tile_delta(P, l, x):
    import mlx.core as mx
    import mlx.nn as nn
    k = f"T{l}"
    xf = x.astype(mx.float32)
    xn = xf * mx.rsqrt((xf * xf).mean(-1, keepdims=True) + 1e-6)
    gt = mx.sigmoid(mx.stop_gradient(xn) @ P[k + ".gw"] + P[k + ".gb"])[..., None]
    if GATE_REC["on"]:
        GATE_REC["g"].append(gt)
    return gt * (nn.silu(xn @ P[k + ".w_in"]) @ P[k + ".w_out"])


def tile_layers_of(P):
    return sorted({int(k.split(".")[0][1:]) for k in P if k.startswith("T")})


def n_params(P):
    return int(sum(v.size for v in P.values()))


# ---------------------------------------------------------------------------------------------- hooks and runs
MLP_FN = {}      # id(mlp) -> fn(mlp, x, out) -> delta or None
_MLP_CLS = {}


def patch_mlps(layers):
    for L in layers:
        base = type(L.mlp)
        if getattr(base, "_amoeba_overlay", False):
            continue
        if base not in _MLP_CLS:
            def call(self, x, *a, _base=base, **kw):
                out = _base.__call__(self, x, *a, **kw)
                f = MLP_FN.get(id(self))
                if f is None:
                    return out
                d = f(self, x, out)
                return out if d is None else out + d.astype(out.dtype)
            _MLP_CLS[base] = type(f"Overlay{base.__name__}", (base,), {"__call__": call, "_amoeba_overlay": True})
        L.mlp.__class__ = _MLP_CLS[base]


def text_layers(model):
    lm = getattr(model, "language_model", model)
    return list(lm.model.layers)


def forward(model, ids2d, P=None, C=None, tiles=None, res32=True, norm_budget=0.25, capture=None, st=None):
    """logits [B, T, V] float32. P/C: overlay; tiles: tile params (layer outputs, capped at norm_budget x the layer
    increment as the lab does); capture {layer: None | idx}: MLP hidden activations (float32) into st[layer]."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx2.experimental.weight_space import residual_ablation as ra
    layers = text_layers(model)
    ol = set(overlay_layers(P, C)) if P is not None else set()
    tl = set(tile_layers_of(tiles)) if tiles is not None else set()
    hooked = tl | {l - 1 for l in tl if l > 0} | ({0} if res32 else set())
    prev = {}

    def hook(l):
        def h(o):
            x = o[0] if isinstance(o, tuple) else o
            if res32 and l == 0:
                x = x.astype(mx.float32)
            if l in tl:
                v = tile_delta(tiles, l, x)
                ref = prev.get(l - 1)
                d0 = (x - ref) if ref is not None else x
                loc = mx.stop_gradient(mx.sqrt((d0.astype(mx.float32) ** 2).sum(-1, keepdims=True)))
                r_ = mx.sqrt((v ** 2).sum(-1, keepdims=True) + 1e-12) / (loc + 1e-12)
                if norm_budget > 0:
                    v = v * mx.minimum(1.0, norm_budget / (r_ + 1e-12))
                x = x + v.astype(x.dtype)
            prev[l] = x
            return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
        return h

    for l in ol | set(capture or {}):
        mlp = layers[l].mlp

        def fn(m_, x, out, l=l):
            if capture and l in capture:
                hid = nn.silu(m_.gate_proj(x)) * m_.up_proj(x)
                idx = capture[l]
                st[l] = (hid if idx is None else hid[..., idx]).astype(mx.float32)
            return overlay_delta(P, C, l, x) if l in ol else None
        MLP_FN[id(mlp)] = fn
    try:
        with ra._hooked([(layers[l], hook(l)) for l in sorted(hooked)]):
            lg = model(mx.array(ids2d), cache=model.make_cache())
    finally:
        for l in ol | set(capture or {}):
            MLP_FN.pop(id(layers[l].mlp), None)
    lg = lg.astype(mx.float32)
    if lg.shape[1] != len(ids2d[0]):
        raise RuntimeError(f"{lg.shape[1]} logit rows for {len(ids2d[0])} tokens: every row is needed")
    return lg


# ---------------------------------------------------------------------------------------------- losses
def gold_nll(lg, tgt):
    """lg [B, T, V] logits, tgt [B, T] int: -log p(tgt) [B, T]"""
    import mlx.core as mx
    return mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, tgt[..., None], axis=-1)[..., 0]


def topk_pack(lp, k):
    """numpy log-probs [N, V] -> (idx [N, k] int32, lp_k [N, k] float32, log tail mass [N] float32)"""
    lp = np.asarray(lp, np.float32)
    idx = np.argpartition(-lp, k - 1, axis=-1)[:, :k].astype(np.int32)
    lk = np.take_along_axis(lp, idx, -1)
    tail = 1.0 - np.exp(lk.astype(np.float64)).sum(-1)
    return idx, lk, np.log(np.maximum(tail, 1e-12)).astype(np.float32)


def kl_topk(lg, idx, lpk, ltail):
    """KL(p || q) of the top-k + tail coarsening: lg [N, V] arm logits, idx/lpk [N, k], ltail [N]; returns [N]"""
    import mlx.core as mx
    lse = mx.logsumexp(lg, axis=-1, keepdims=True)
    lq = mx.take_along_axis(lg, idx, axis=-1) - lse
    s_ = mx.minimum(mx.logsumexp(lq, axis=-1), -1e-7)  # log of the top-k mass under q
    lqt = mx.log(-mx.expm1(s_))                         # log(1 - q_k mass) without the float32 cancellation
    pk = mx.exp(lpk)
    pt = mx.exp(ltail)
    return (pk * (lpk - lq)).sum(-1) + pt * (ltail - lqt)


def kl_full(lp_ref, lg):
    """KL(ref || arm) per row: lp_ref [N, V] log-probs, lg [N, V] arm logits"""
    import mlx.core as mx
    lq = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    return (mx.exp(lp_ref) * (lp_ref - lq)).sum(-1)


# ---------------------------------------------------------------------------------------------- optimizer
class Adam:
    """Adam over a flat dict, per-key learning-rate multiplier (prefix match on the key's suffix after the layer)
    and linear warmup over `warmup` steps (gentle insertion)."""

    def __init__(self, lr, mults=None, warmup=0, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.mults, self.warmup = lr, dict(mults or {}), int(warmup)
        self.b1, self.b2, self.eps, self.t, self.m, self.v = b1, b2, eps, 0, {}, {}

    def mult(self, k):
        suf = k.split(".", 1)[1] if "." in k else k
        return self.mults.get(suf, 1.0)

    def lr_now(self):
        return self.lr * (min(1.0, self.t / self.warmup) if self.warmup > 0 else 1.0)

    def step(self, P, G):
        import mlx.core as mx
        self.t += 1
        lr = self.lr_now()
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        out = {}
        for k, p in P.items():
            g = G.get(k)
            if g is None:
                out[k] = p
                continue
            g = g.astype(mx.float32)
            m = self.b1 * self.m.get(k, mx.zeros_like(g)) + (1 - self.b1) * g
            v = self.b2 * self.v.get(k, mx.zeros_like(g)) + (1 - self.b2) * g * g
            self.m[k], self.v[k] = m.astype(p.dtype), v.astype(p.dtype)
            out[k] = (p.astype(mx.float32) - lr * self.mult(k) * (m / c1) / (mx.sqrt(v / c2) + self.eps)).astype(p.dtype)
        return out


# ---------------------------------------------------------------------------------------------- save / load
def save_overlay(path, P, C, meta):
    import mlx.core as mx
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrs = {f"p.{k}": v for k, v in P.items()} | {f"c.{k}": v for k, v in C.items()}
    mx.save_safetensors(str(path / "overlay.safetensors"), arrs)
    meta = dict(meta, schema="mlx2.amoeba-overlay.v1", files={"weights": "overlay.safetensors"},
                weights_sha256=hashlib.sha256((path / "overlay.safetensors").read_bytes()).hexdigest())
    (path / "overlay.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def load_overlay(path, host_checkpoint=None):
    """(P, C, meta) from save_overlay's directory; refuses another host checkpoint"""
    import mlx.core as mx
    path = Path(path)
    meta = json.loads((path / "overlay.json").read_text())
    if host_checkpoint and meta.get("host_checkpoint") != host_checkpoint:
        raise SystemExit(f"overlay is for {meta.get('host_checkpoint')}, host is {host_checkpoint}")
    arrs = mx.load(str(path / "overlay.safetensors"))
    P = {k[2:]: v for k, v in arrs.items() if k.startswith("p.")}
    C = {k[2:]: v for k, v in arrs.items() if k.startswith("c.")}
    return P, C, meta


@contextlib.contextmanager
def install(model, P, C, receipt=None):
    """serve-side opt-in: the overlay on every forward of `model` while the context lives (gate receipts into
    receipt['gate_open'] per layer when given)"""
    layers = text_layers(model)
    patch_mlps(layers)
    ol = overlay_layers(P, C)
    for l in ol:
        def fn(m_, x, out, l=l):
            d = overlay_delta(P, C, l, x)
            if receipt is not None and f"L{l}.gw" in P and not CLOSED[0]:
                import mlx.core as mx
                import mlx.nn as nn
                xf = x.astype(mx.float32)
                k = f"L{l}"
                a = nn.silu(xf @ C[k + ".gg"].T) * (xf @ C[k + ".gu"].T)
                g = mx.sigmoid(((a - C[k + ".mu"]) / C[k + ".sd"]) @ P[k + ".gw"] + P[k + ".gb"])
                receipt.setdefault("gate_open", {}).setdefault(str(l), []).append(float(g.mean().item()))
            return d
        MLP_FN[id(layers[l].mlp)] = fn
    try:
        yield
    finally:
        for l in ol:
            MLP_FN.pop(id(layers[l].mlp), None)


# ---------------------------------------------------------------------------------------------- main
def main(argv=None, loader=None):
    """loader (tests): path -> {"ad", "model", "layers", "tok", "ck"} in place of the mlx2 adapter load"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["arms", "scale"])
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default=str(
        Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/topic-probes-9a1f35475/corpus.jsonl"))
    ap.add_argument("--topic", default="sci.physics")
    ap.add_argument("--topic-tokens", type=int, default=64)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--split-check", action="store_true")
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-held", type=int, default=0, help="0 = every reserved final-test text")
    ap.add_argument("--eng-retain", type=int, default=24)
    ap.add_argument("--eng-val", type=int, default=8)
    ap.add_argument("--eng-test", type=int, default=16)
    ap.add_argument("--eng-tokens", type=int, default=96)
    ap.add_argument("--arms", default="ov_sink_win,ov_win,sink_win,ov_sink_all,ov_sink_gold,ov_sink_rand,ov_sink_sft,"
                                      "tile_sft")
    ap.add_argument("--budget-vecs", type=int, default=2048,
                    help="trainable D-vectors per arm (0c's tile: 32 units x 32 layers x 2)")
    ap.add_argument("--sink-share", type=float, default=0.25)
    ap.add_argument("--n-box-layers", type=int, default=4)
    ap.add_argument("--n-sink-layers", type=int, default=2)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--gate-units", type=int, default=64)
    ap.add_argument("--read-scale", type=float, default=1.1)
    ap.add_argument("--write-scale", type=float, default=0.01)
    ap.add_argument("--dup-jitter", type=float, default=0.1)
    ap.add_argument("--win-margin", type=float, default=0.5)
    ap.add_argument("--text-margin", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--resp-prefix", type=int, default=32)
    ap.add_argument("--resp-tokens", type=int, default=32)
    ap.add_argument("--core-units", type=int, default=256)
    ap.add_argument("--core-rank", type=int, default=8)
    ap.add_argument("--core-lr-mult", type=float, default=0.1)
    ap.add_argument("--core-l2", type=float, default=1e-2)
    ap.add_argument("--core-loss", default="kl_win", choices=["kl_win", "ce_gold"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--read-lr-mult", type=float, default=0.1)
    ap.add_argument("--write-lr-mult", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--tile-lr", type=float, default=1e-3)
    ap.add_argument("--norm-budget", type=float, default=0.25, help="tile_sft blend-in cap (the lab's)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--evals-per-epoch", type=int, default=4)
    ap.add_argument("--drift-weight", type=float, default=1.0)
    ap.add_argument("--drift-cap", type=float, default=0.05)
    ap.add_argument("--drift-adapt", type=float, default=4.0)
    ap.add_argument("--drift-weight-max", type=float, default=256.0)
    ap.add_argument("--gate-l1", type=float, default=0.1)
    ap.add_argument("--overlay-dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--corpus", default="", help="stage scale: amoeba_overlay_corpus.py output directory")
    ap.add_argument("--scale-mults", default="1,4,16,100")
    ap.add_argument("--scale-kl-mults", default="", help="stage scale: sizes also trained with donor-wins KL")
    ap.add_argument("--scale-batch", type=int, default=8)
    ap.add_argument("--scale-evals", type=int, default=8, help="checkpoints per size")
    ap.add_argument("--train-budget-s", type=float, default=6000, help="stage scale: optimizer wall for the sweep")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-epochs", type=float, default=3.0)
    ap.add_argument("--max-overlay-gb", type=float, default=40.0)
    ap.add_argument("--train-probe-chunks", type=int, default=64)
    ap.add_argument("--track-dev", type=int, default=64,
                    help="stage scale: dev texts scored at every checkpoint (the held-side curve; final scoring "
                         "reads every dev and held text)")
    ap.add_argument("--save-overlays", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--disk-snapshot-params", type=float, default=1e8,
                    help="overlays with more trainable parameters keep their checkpoints on disk")
    ap.add_argument("--scale-write-lr", default="inv_sqrt", choices=["inv_sqrt", "inv", "flat"],
                    help="stage scale: write-row LR multiplier per size m (x 1/sqrt(m), x 1/m, or none), so a wider "
                         "overlay does not move its output m times faster per Adam step")
    ap.add_argument("--largest-fitting", action="store_true",
                    help="stage scale: run only the largest --scale-mults size whose measured probe peak fits")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--min-useful-npb", type=float, default=0.001)
    ap.add_argument("--mem-cap-gb", type=float, default=0)
    ap.add_argument("--watchdog-gb", type=float, default=0)
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--swap-base-gb", type=float, default=None)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto")
    ap.add_argument("--budget-s", type=float, default=10800)
    ap.add_argument("--grad-ref-max-kl", type=float, default=0.01)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.watchdog_gb and a.swap_growth_gb > 2.0 and not a.waiver_note.strip():
        sys.exit("--swap-growth-gb above 2 needs --waiver-note naming the approval")
    known = {"ov_sink_win", "ov_win", "sink_win", "ov_sink_all", "ov_sink_gold", "ov_sink_rand", "ov_sink_sft",
             "tile_sft", "ov_sink_core"}
    arms_req = [x.strip() for x in a.arms.split(",") if x.strip()]
    if a.stage == "arms" and (bad := [x for x in arms_req if x not in known]):
        sys.exit(f"unknown --arms {bad}")
    if a.stage == "scale" and not a.corpus:
        sys.exit("--stage scale needs --corpus")
    out = Path(a.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLX2_PREFILL_TAIL_SKIP", "0")  # every logit row is scored
    t0 = time.time()
    receipt = {"schema": "mlx2.amoeba-donor-overlay.v1", "status": "running", "stage": "loading", "args": vars(a),
               "qualification": "unqualified research"}
    (out / "run.json").write_text(json.dumps(receipt, indent=1))
    _STATE.clear()
    _STATE.update(receipt=receipt, out=out)  # for run_guarded: a refusal before save() exists
    import threading
    LOCK = threading.Lock()
    WATCH = None
    from research.amoeba.carve import ops as CO
    if not a.cpu:
        CO.prepare_wiring(a.wired_gb)
    if a.watchdog_gb:
        import amoeba27b_lib as wd_
        sg = a.swap_growth_gb
        if a.swap_base_gb is not None:
            sg = a.swap_growth_gb - max(0.0, wd_.swap_used_gb() - a.swap_base_gb)
            if sg <= 0:
                sys.exit("swap already grew past the window's allowance")
        from research.amoeba.carve.memwatch import STOP_MARKER

        def wd_log(msg):
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
            with open(out / "watchdog.log", "a") as f_:
                f_.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            if "aborting" in msg or STOP_MARKER in msg:
                try:
                    if LOCK.acquire(timeout=2):
                        rec = dict(receipt, status="failed (watchdog)", reason=msg, wall_s=round(time.time() - t0, 1))
                        tmp = out / "run.json.wd.tmp"
                        tmp.write_text(json.dumps(rec, indent=1, default=str))
                        tmp.replace(out / "run.json")
                except BaseException:
                    pass
        CO.footprint_watchdog(a.watchdog_gb, wd_log, min_avail_gb=a.min_avail_gb, swap_growth_gb=sg,
                              receipt=out / "watchdog-stop.json", watcher=True)
        WATCH = CO.WATCH
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    if a.mem_cap_gb:
        mx.set_memory_limit(int(a.mem_cap_gb * 1e9))
    mx.set_cache_limit(2 << 30)
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.models import differentiable as dref
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()) or bool(subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/amoeba_donor_overlay.py", "scripts/amoeba_bridge_lab.py"],
        cwd=ROOT, capture_output=True, text=True).stdout.strip())

    def save(**kw):
        with LOCK:
            receipt.update(kw, wall_s=round(time.time() - t0, 1))
            if WATCH is not None:
                receipt["watchdog"] = dict(WATCH, waiver_note=a.waiver_note or None, swap_base_gb=a.swap_base_gb)
            tmp = out / "run.json.tmp"
            tmp.write_text(json.dumps(receipt, indent=1, default=str))
            tmp.replace(out / "run.json")
    _STATE["save"] = save

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def budget_left():
        return a.budget_s - (time.time() - t0)

    def load(path):
        if loader is not None:
            return loader(path)
        ad = inspect_model(path).adapter_type(str(path))
        m = ad.model
        m.eval()
        return {"ad": ad, "model": m, "layers": text_layers(m), "tok": ad.tokenizer,
                "ck": f"{Path(path).name}@{ad.identity['fingerprint']}"}
    H = load(a.host)
    patch_mlps(H["layers"])
    tok = H["tok"]
    LH = len(H["layers"])
    D = int(H["layers"][0].mlp.down_proj.weight.shape[0])
    I_ = int(H["layers"][0].mlp.gate_proj.weight.shape[0]) if not hasattr(H["layers"][0].mlp.gate_proj, "scales") \
        else int(H["layers"][0].mlp.gate_proj.scales.shape[0])
    Dn = load(a.donor)
    from mlx.utils import tree_flatten
    donor_gb = sum(v.nbytes for _, v in tree_flatten(Dn["model"].parameters())) / 1e9
    man = json.loads(Path(a.split_manifest).read_text())
    for k_, ck_ in (("host_checkpoint", H["ck"]), ("donor_checkpoint", Dn["ck"])):
        if man.get(k_) and man[k_] != ck_:
            sys.exit(f"--split-manifest names {k_} {man[k_]}, loaded {ck_}")
    if not a.cpu:
        receipt["wired"] = CO.wire_weights(a.wired_gb, log, memory_gb=a.mem_cap_gb or None)
    receipt.update(host_checkpoint=H["ck"], donor_checkpoint=Dn["ck"], mlx2_source_revision=rev, tree_dirty=dirty,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   probes_sha256=hashlib.sha256(Path(a.probes).read_bytes()).hexdigest(),
                   core_corpus_sha256=hashlib.sha256(Path(CORE).read_bytes()).hexdigest(),
                   split_manifest={"path": str(Path(a.split_manifest).resolve()),
                                   "sha256": hashlib.sha256(Path(a.split_manifest).read_bytes()).hexdigest()},
                   label=a.label, host_layers=LH, hidden=D, mlp_width=I_,
                   box_status="not carved: the host box chain is this script's topic-contrast unit ranking at the "
                              "donor-better positions; it has NOT passed the carving bar")
    save(stage="selection")

    # ----------------------------------------------------------------------------------- texts
    rows_ = [json.loads(x) for x in open(a.probes)]
    excl_ids, excl_groups = LAB.split_exclusions(man)

    def dcheck(r, ids_):
        if Dn["tok"].encode(r["text"], add_special_tokens=False)[:a.topic_tokens] != ids_:
            sys.exit(f"donor tokenizer differs on probe text {r['id']}")
    enc = lambda t: tok.encode(t, add_special_tokens=False)  # noqa: E731
    items = LAB.topic_selection(rows_, a.topic, a.seed, a.n_train, a.n_val, a.n_test, a.topic_tokens, enc,
                                check=dcheck, exclude_ids=excl_ids, exclude_groups=excl_groups)
    if a.split_check and (err := LAB.split_check(man, items)):
        save(status="failed", reason=f"split check: {err}")
        sys.exit(f"--split-check: {err}")
    if any(it["probe_id"] in excl_ids or str(it["group"]) in excl_groups for it in items):
        sys.exit("a reserved final-test text was selected")
    by_id = {r["id"]: r for r in rows_}
    held_ids = list((man.get("reserved_final_test") or {}).get("probe_ids", []))
    if a.n_held:
        held_ids = held_ids[:a.n_held]
    held_items, held_short = [], []
    for pid in held_ids:
        r = by_id[pid]
        ids_ = enc(r["text"])[:a.topic_tokens]
        if len(ids_) < a.topic_tokens:
            held_short.append(pid)
            continue
        dcheck(r, ids_)
        held_items.append({"probe_id": pid, "full": ids_, "ids": ids_[:2], "group": r.get("group"), "split": "held"})
    used = {it["probe_id"] for it in items}
    if used & {h["probe_id"] for h in held_items}:
        sys.exit("held texts overlap the selection")

    def seq(it):
        ids = list(it["full"])
        P_ = len(it["ids"])
        return {"probe_id": it["probe_id"], "group": it.get("group"), "split": it["split"], "ids": ids, "P": P_,
                "bytes": len(tok.decode(ids[P_:]).encode("utf-8"))}
    tr = [seq(it) for it in items if it["split"] == "train"]
    va = [seq(it) for it in items if it["split"] == "val"]
    dev = [seq(it) for it in items if it["split"] == "test"]
    held = [seq(it) for it in held_items]
    eng_all = [json.loads(x) for x in open(CORE)]
    fit_e = [r for r in eng_all if r["split"] == "fit"]
    held_e = [r for r in eng_all if r["split"] == "held"]
    g_ = np.random.default_rng(a.seed + 7)  # the lab's English draw (same seed, same texts)
    pick_f = g_.choice(len(fit_e), a.eng_retain + a.eng_val, replace=False)
    pick_h = g_.choice(len(held_e), a.eng_test, replace=False)
    eng = ([{"id": fit_e[i]["id"], "role": "retain", "text": fit_e[i]["text"]} for i in pick_f[:a.eng_retain]]
           + [{"id": fit_e[i]["id"], "role": "eng_val", "text": fit_e[i]["text"]} for i in pick_f[a.eng_retain:]]
           + [{"id": held_e[i]["id"], "role": "eng_test", "text": held_e[i]["text"]} for i in pick_h])
    for e in eng:
        e["ids"] = enc(e["text"])[:a.eng_tokens]
    retain = [e for e in eng if e["role"] == "retain"]
    eng_val = [e for e in eng if e["role"] == "eng_val"]
    eng_test = [e for e in eng if e["role"] == "eng_test"]
    receipt["texts"] = {"train": len(tr), "val": len(va), "dev_spent": len(dev), "held_untouched": len(held),
                        "held_short_dropped": held_short, "english": {k: sum(e["role"] == k for e in eng)
                                                                       for k in ("retain", "eng_val", "eng_test")},
                        "held_note": "the reserved final-test texts (split manifest reserved_final_test); scored here "
                                     "once, at the end of each arm"}
    log(f"host {H['ck']} donor {Dn['ck']} rev {rev} dirty {dirty}; train {len(tr)} val {len(va)} dev {len(dev)} "
        f"held {len(held)}")

    def bs(ids):
        return [ids]

    # ----------------------------------------------------------------------------------- donor + host precompute
    def gold(M, s):
        lg = forward(M["model"], bs(s["ids"]), res32=M is H)[0]  # host: the arms' float32-residual path
        nll = gold_nll(lg[None, s["P"] - 1:-1], mx.array([s["ids"][s["P"]:]]))[0]
        return lg, np.array(nll)
    for s in tr + va + dev + held:
        lg, nll = gold(Dn, s)
        s["d_nll"] = nll
        if s["split"] in ("train", "val"):
            lp = lg[s["P"] - 1:-1]
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            s["dK"] = topk_pack(np.array(lp), a.topk)
        del lg
    log("donor gold and top-k done")
    for s in tr + va + dev + held:
        _, s["h_nll"] = gold(H, s)
        s["win"] = win_positions(-s["h_nll"], -s["d_nll"], a.win_margin)
    for e in eng:
        lg = forward(H["model"], bs(e["ids"]), res32=True)[0]
        e["h_lp"] = (lg - mx.logsumexp(lg, axis=-1, keepdims=True)).astype(mx.float16)
        mx.eval(e["h_lp"])
    npb = lambda ss, k: np.array([s[k].sum() / s["bytes"] for s in ss])  # noqa: E731
    host_held, donor_held = npb(held, "h_nll"), npb(held, "d_nll")
    host_dev, donor_dev = npb(dev, "h_nll"), npb(dev, "d_nll")
    wins = {"margin_nats": a.win_margin, "train_positions": int(sum(len(s["win"]) for s in tr)),
            "train_win_positions": int(sum(s["win"].sum() for s in tr)),
            "train_texts_with_win": int(sum(s["win"].any() for s in tr)),
            "train_text_wins": int(sum(text_win(s["h_nll"].sum(), s["d_nll"].sum(), s["bytes"], a.text_margin)
                                       for s in tr)),
            "held_host_npb": float(host_held.mean()), "held_donor_npb": float(donor_held.mean()),
            "dev_host_npb": float(host_dev.mean()), "dev_donor_npb": float(donor_dev.mean()),
            "held_gap_paired": PS.paired(host_held - donor_held, [s["group"] for s in held], mue=a.min_useful_npb)}
    receipt["selection"] = wins
    save(stage="donor-wins selection")
    log(f"donor-better positions {wins['train_win_positions']}/{wins['train_positions']}; held host "
        f"{host_held.mean():.4f} donor {donor_held.mean():.4f} npb")

    # sft_donor targets: the donor's greedy continuation of each text-level donor-win text's prefix
    def donor_gen(prefix, n):
        c = Dn["model"].make_cache()
        lg = Dn["model"](mx.array([prefix]), cache=c)[0, -1]
        toks = []
        eos = set(tok.eos_token_ids)
        for _ in range(n):
            t = int(mx.argmax(lg).item())
            toks.append(t)
            if t in eos:
                break
            lg = Dn["model"](mx.array([[t]]), cache=c)[0, -1]
        return toks
    resp = []
    if a.stage == "arms" and "ov_sink_sft" in arms_req:
        for s in tr:
            pre = s["ids"][:a.resp_prefix]
            pn = np.arange(a.resp_prefix - s["P"], len(s["h_nll"]))  # scored positions of the gold suffix
            nb_ = len(tok.decode(s["ids"][a.resp_prefix:]).encode("utf-8"))
            if text_win(s["h_nll"][pn].sum(), s["d_nll"][pn].sum(), nb_, a.text_margin):
                resp.append({"probe_id": s["probe_id"], "ids": pre + donor_gen(pre, a.resp_tokens),
                             "P": a.resp_prefix})
        receipt["sft_donor"] = {"texts": len(resp), "prefix": a.resp_prefix, "tokens": a.resp_tokens}
        log(f"donor responses: {len(resp)}")

    # ----------------------------------------------------------------------------------- host unit statistics
    def dense(lin):
        if "scales" in lin:
            kw = {"mode": lin.mode} if hasattr(lin, "mode") else {}
            return mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits, **kw)
        return lin.weight
    wn = {}
    for l in range(LH):
        w = dense(H["layers"][l].mlp.down_proj).astype(mx.float32)
        wn[l] = np.array(mx.sqrt((w * w).sum(0)))
        del w
    mt, me = np.zeros((LH, I_)), np.zeros((LH, I_))
    nt = ne = 0
    rng = np.random.default_rng(a.seed + 11)
    ZT = {l: [] for l in range(LH)}
    ZE = {l: [] for l in range(LH)}
    cap_t = cap_e = 1024
    n_win = sum(int(s["win"].sum()) for s in tr)
    n_eng = sum(max(len(e["ids"]) - 1, 0) for e in retain)
    for s in tr:
        if not s["win"].any():
            continue
        st = {}
        forward(H["model"], bs(s["ids"]), res32=True, capture={l: None for l in range(LH)}, st=st)
        pos = (s["P"] - 1 + np.nonzero(s["win"])[0])
        keep = rng.random(len(pos)) < cap_t / max(n_win, 1)  # gate-fit sample: uniform over every text
        for l in range(LH):
            u = np.array(st[l][0])[pos]
            mt[l] += np.abs(u).sum(0) * wn[l]
            ZT[l].extend(u[keep].astype(np.float16))
        nt += len(pos)
    for e in retain:
        st = {}
        forward(H["model"], bs(e["ids"]), res32=True, capture={l: None for l in range(LH)}, st=st)
        keep = rng.random(len(e["ids"]) - 1) < cap_e / max(n_eng, 1)
        for l in range(LH):
            u = np.array(st[l][0])[1:]  # position 0 (the attention-sink token) excluded, as the lab does
            me[l] += np.abs(u).sum(0) * wn[l]
            ZE[l].extend(u[keep].astype(np.float16))
        ne += len(e["ids"]) - 1
    mt /= max(nt, 1)
    me /= max(ne, 1)
    ZT = {l: np.array(v[:cap_t], np.float32).reshape(-1, I_) for l, v in ZT.items()}
    ZE = {l: np.array(v[:cap_e], np.float32).reshape(-1, I_) for l, v in ZE.items()}
    sink_layers = list(range(LH - a.n_sink_layers, LH))
    allowed = set(range(1, LH - a.n_sink_layers))
    box_layers, lscore = select_layers(mt, me, 64, a.n_box_layers, a.min_ratio, allowed)
    rand_layers = sorted(rng.choice(sorted(allowed), a.n_box_layers, replace=False).tolist())
    orders = {l: rank_units(mt[l], me[l], a.min_ratio) for l in range(LH)}
    # sink ranking: contrast x next-token promotion through the final-norm gain and the unembedding
    lm = getattr(H["model"], "language_model", H["model"])
    gfin = lm.model.norm.weight.astype(mx.float32)
    nxt = np.concatenate([np.array(s["ids"][s["P"]:])[s["win"]] for s in tr]) if tr else np.zeros(0, int)

    def unembed(h):
        return (lm.logits(h) if hasattr(lm, "logits") else lm.lm_head(h)).astype(mx.float32)
    sink_rank = {}
    for l in sink_layers:
        order, ne_ = orders[l]
        cand = order[:min(len(order), 2048)]
        wd = dense(H["layers"][l].mlp.down_proj).astype(mx.float32)
        zs = []
        for c0 in range(0, len(cand), 256):
            cc = mx.array(cand[c0:c0 + 256].astype(np.int32))
            v = unembed((wd[:, cc].T * gfin)[None])[0]
            v = (v - mx.mean(v, axis=-1, keepdims=True)) / (mx.sqrt(mx.var(v, axis=-1, keepdims=True)) + 1e-6)
            zs.append(np.array(v[:, mx.array(nxt.astype(np.int32))].mean(-1)) if len(nxt) else np.zeros(len(cc)))
        z = np.concatenate(zs)
        contrast = (mt[l] - me[l])[cand]
        sc = np.maximum(contrast, 0) * np.maximum(z, 0)
        sink_rank[l] = sink_order(order, cand, sc)
        del wd
    box_info = {"box_layers": box_layers, "sink_layers": sink_layers, "rand_layers": rand_layers,
                "layer_score": lscore.tolist(),
                "eligible_units": {str(l): orders[l][1] for l in set(box_layers) | set(sink_layers)},
                "rule": f"units ranked by topic mass at the donor-better train positions minus English mass on the "
                        f"retain texts (|a| x |W_down col|), eligible = topic >= {a.min_ratio} x English first; box "
                        f"layers: the lab's select (top-64 eligible score, >= 2 apart, below the sink layers); sinks: "
                        f"the last {a.n_sink_layers} layers, ranked by max(contrast,0) x max(next-token z,0) over the "
                        "2048 best-contrast units, then contrast"}
    (out / "box.json").write_text(json.dumps(box_info, indent=1))
    receipt["box"] = box_info
    save(stage="host box chain")
    log(f"host box layers {box_layers}, sinks {sink_layers}, random layers {rand_layers}")

    row_cache = {}

    def rows(l, ids):
        if l not in row_cache:
            m_ = H["layers"][l].mlp
            row_cache.clear()  # one layer at a time (a full layer is 3 x I x D floats)
            row_cache[l] = tuple(np.array(dense(p).astype(mx.float32)) for p in (m_.gate_proj, m_.up_proj))
            row_cache[l] += (np.array(dense(m_.down_proj).astype(mx.float32)).T,)
        wg, wu, wd = row_cache[l]
        ids = np.asarray(ids)
        return wg[ids].copy(), wu[ids].copy(), wd[ids].copy()

    def make_spec(counts, rank_of, rand=False):
        spec = {}
        g2 = np.random.default_rng(a.seed + 21)
        for l, n in counts.items():
            order = g2.permutation(I_) if rand else rank_of(l)
            ids, copy = expand_units(order, n)
            gu = np.asarray(order[:min(a.gate_units, len(order))])
            zt, ze = ZT[l][:, gu], ZE[l][:, gu]
            allz = np.concatenate([zt, ze])
            mu, sd = (allz.mean(0), allz.std(0)) if len(allz) else (np.zeros(len(gu)), np.ones(len(gu)))
            sd = np.maximum(sd, 1e-3 + 0.05 * float(np.median(sd)))  # near-dead units cannot blow up off-distribution
            if len(zt) and len(ze):
                w, b, acc = fit_gate((zt - mu) / sd, (ze - mu) / sd)
            else:  # no topic or English positions to fit on: a half-open gate
                w, b, acc = np.zeros(len(gu), np.float32), np.float32(0), None
            spec[l] = {"units": ids, "copy": copy, "gate_units": gu, "gate_mu": mu.astype(np.float32),
                       "gate_sd": sd.astype(np.float32), "gate_w": w, "gate_b": b, "gate_fit_acc": acc}
        return spec

    def rank_of(l):
        return sink_rank[l] if l in sink_layers else orders[l][0]

    def core_spec(spec):
        out_ = {}
        for l in spec:
            used_ = set(np.asarray(spec[l]["units"]).tolist())
            ok = [u for u in np.argsort(-me[l]) if mt[l][u] < a.min_ratio * me[l][u] and u not in used_]
            out_[l] = {"units": np.asarray(ok[:a.core_units]), "rank": a.core_rank}
        return out_

    # ----------------------------------------------------------------------------------- arm definitions
    base_counts = overlay_counts(a.budget_vecs, box_layers, sink_layers, a.sink_share)
    ARMS = {"ov_sink_win": {"counts": base_counts, "loss": "kl_win"},
            "ov_win": {"counts": overlay_counts(a.budget_vecs, box_layers, [], 0), "loss": "kl_win"},
            "sink_win": {"counts": overlay_counts(a.budget_vecs, [], sink_layers, 1), "loss": "kl_win"},
            "ov_sink_all": {"counts": base_counts, "loss": "kl_all"},
            "ov_sink_gold": {"counts": base_counts, "loss": "ce_gold"},
            "ov_sink_rand": {"counts": overlay_counts(a.budget_vecs, rand_layers, sink_layers, a.sink_share),
                             "loss": "kl_win", "rand": True},
            "ov_sink_sft": {"counts": base_counts, "loss": "sft_donor"},
            "tile_sft": {"tile": True, "loss": "ce_gold"},
            "ov_sink_core": {"counts": base_counts, "loss": a.core_loss, "core": True}}
    receipt["budget"] = {"budget_vecs": a.budget_vecs, "base_counts": {str(k): v for k, v in base_counts.items()},
                         "tile_units_per_layer": tile_units(a.budget_vecs, LH)}

    def build_arm(name, arm):
        if arm.get("tile"):
            return build_tiles(range(LH), tile_units(a.budget_vecs, LH), D, a.seed + 1), {}, None
        cnt = arm["counts"]
        if arm.get("rand"):
            box_part = {l: n for l, n in cnt.items() if l not in sink_layers}
            spec = make_spec(box_part, rank_of, rand=True) | make_spec(
                {l: n for l, n in cnt.items() if l in sink_layers}, rank_of)
        else:
            spec = make_spec(cnt, rank_of)
        core = core_spec(spec) if arm.get("core") else None
        P, C = build_overlay(spec, rows, D, a.read_scale, a.write_scale, a.dup_jitter, a.seed + 3, a.overlay_dtype,
                             core=core)
        info = {"counts": {str(l): int(n) for l, n in cnt.items()},
                "gate_fit_acc": {str(l): spec[l]["gate_fit_acc"] for l in spec},
                "core_units": {str(l): len(c["units"]) for l, c in (core or {}).items()} or None}
        return P, C, info

    # ----------------------------------------------------------------------------------- training / scoring
    def arm_logits(P, C, ids2d, tile=False):
        if tile:
            return forward(H["model"], ids2d, tiles=P, res32=True, norm_budget=a.norm_budget)
        return forward(H["model"], ids2d, P=P, C=C, res32=True)

    def seq_nll(P, C, s, tile=False):
        lg = arm_logits(P, C, bs(s["ids"]), tile)[0]
        return np.array(gold_nll(lg[None, s["P"] - 1:-1], mx.array([s["ids"][s["P"]:]]))[0])

    def npb_of(P, C, ss, tile=False):
        return np.array([seq_nll(P, C, s, tile).sum() / s["bytes"] for s in ss])

    def drift_of(P, C, es, tile=False):
        """mean per-token KL(host || arm) on English; the host reference is recomputed in float32 (exact: the
        stored float16 log-probs are for the retain step's loss only)"""
        v = []
        for e in es:
            lh = forward(H["model"], bs(e["ids"]), res32=True)[0]
            v.append(float(kl_full(lh - mx.logsumexp(lh, axis=-1, keepdims=True),
                                   arm_logits(P, C, bs(e["ids"]), tile)[0]).mean().item()))
        return float(np.mean(v))

    def loss_topic(P, C, batch, kind, tile):
        if kind == "kl_win_live":  # stage scale: donor log-probs and the donor-better mask came with the batch
            ids2d = [s["ids"] for s in batch["seqs"]]
            lg = arm_logits(P, C, ids2d, tile)[:, :-1]
            kl = kl_full(batch["lpd"].reshape(-1, lg.shape[-1]), lg.reshape(-1, lg.shape[-1]))
            m = batch["mask"].reshape(-1)
            return (kl * m).sum() / mx.maximum(m.sum(), 1.0)
        L_ = max(len(s["ids"]) for s in batch)
        ids2d = [s["ids"] + [s["ids"][-1]] * (L_ - len(s["ids"])) for s in batch]
        lg = arm_logits(P, C, ids2d, tile)
        p0 = batch[0]["P"]
        if kind in ("ce_gold", "sft_donor"):
            if kind == "sft_donor":  # donor responses: score the response tokens (eos padding masked)
                tg = [r[1:] for r in ids2d]
                msk = np.zeros((len(batch), L_ - 1), np.float32)
                for i, s in enumerate(batch):
                    msk[i, s["P"] - 1:len(s["ids"]) - 1] = 1
                nll = gold_nll(lg[:, :L_ - 1], mx.array(tg))
                return (nll * mx.array(msk)).sum() / float(msk.sum())
            nll = gold_nll(lg[:, p0 - 1:-1], mx.array([s["ids"][p0:] for s in batch]))
            return nll.mean()
        rows_lg = lg[:, p0 - 1:-1].reshape(-1, lg.shape[-1])
        idx = mx.array(np.concatenate([s["dK"][0] for s in batch]))
        lpk = mx.array(np.concatenate([s["dK"][1] for s in batch]))
        lt = mx.array(np.concatenate([s["dK"][2] for s in batch]))
        kl = kl_topk(rows_lg, idx, lpk, lt)
        if kind == "kl_all":
            return kl.mean()
        m = np.concatenate([s["win"] for s in batch]).astype(np.float32)
        return (kl * mx.array(m)).sum() / max(float(m.sum()), 1.0)

    def loss_eng(P, C, e, lam, tile):
        GATE_REC["on"], GATE_REC["g"] = a.gate_l1 > 0, []
        try:
            lg = arm_logits(P, C, bs(e["ids"]), tile)[0]
        finally:
            GATE_REC["on"] = False
        loss = lam * kl_full(e["h_lp"].astype(mx.float32), lg).mean()
        if GATE_REC["g"]:
            loss = loss + a.gate_l1 * mx.stack([g_.mean() for g_ in GATE_REC["g"]]).mean()
        GATE_REC["g"] = []
        if any(k.endswith(".U") for k in P):
            loss = loss + a.core_l2 * core_l2(P)
        return loss

    def mults(tile, wmult=1.0):
        if tile:
            return {}
        return {"wg": a.read_lr_mult, "wu": a.read_lr_mult, "wd": a.write_lr_mult * wmult, "U": a.core_lr_mult,
                "V": a.core_lr_mult}

    def on_ref(f):
        def g(*x, **kw):
            with dref.reference_path():
                return f(*x, **kw)
        return g

    def gate_means(P, C, ss):
        """mean sigmoid gate per overlay layer over the texts' positions (the gate is soft: it never shuts)"""
        GATE_REC["on"], GATE_REC["g"] = True, []
        try:
            acc = []
            for s in ss:
                GATE_REC["g"] = []
                mx.eval(arm_logits(P, C, bs(s["ids"])))
                acc.append([float(g_.mean().item()) for g_ in GATE_REC["g"]])
        finally:
            GATE_REC["on"], GATE_REC["g"] = False, []
        gl = [l for l in overlay_layers(P, C) if f"L{l}.gw" in P]
        return {str(l): round(float(np.mean([r[i] for r in acc])), 4) for i, l in enumerate(gl)} if acc and gl else {}

    def served_path(P, C):
        """the selected overlay through install() on the serving path (no float32 residual): held nats/byte minus
        the plain host's on the same path"""
        rec = {}
        h_, o_ = [], []
        for s in held[:96]:
            tg = mx.array([s["ids"][s["P"]:]])
            lh = H["model"](mx.array([s["ids"]]), cache=H["model"].make_cache()).astype(mx.float32)
            if lh.shape[1] != len(s["ids"]):
                raise RuntimeError("served path returned fewer logit rows than tokens (prefill tail skip on?)")
            with install(H["model"], P, C, rec):
                lo = H["model"](mx.array([s["ids"]]), cache=H["model"].make_cache()).astype(mx.float32)
            if lo.shape[1] != len(s["ids"]):
                raise RuntimeError("served path returned fewer logit rows than tokens")
            h_.append(float(gold_nll(lh[:, s["P"] - 1:-1], tg).sum().item()) / s["bytes"])
            o_.append(float(gold_nll(lo[:, s["P"] - 1:-1], tg).sum().item()) / s["bytes"])
        d = np.array(h_) - np.array(o_)
        return {"path": "install() on the serving forward (bf16 residual)", "delta_vs_host_held": float(d.mean()),
                "texts": len(d), "paired": PS.paired(d, [s["group"] for s in held[:96]], mue=a.min_useful_npb),
                "gate_open_mean": {k: round(float(np.mean(v)), 4) for k, v in rec.get("gate_open", {}).items()}}

    def closed_cost(P, C, tile):
        """overlay removed (CLOSED: every gated write skipped): held nats/byte minus the host (zero unless a core
        delta is on), forward ms of host / overlay removed / overlay live; and the live sigmoid gate's mean on
        English and held text (the gate is soft, so a served overlay always computes and writes a little)"""
        if tile:
            return None
        CLOSED[0] = True
        try:
            d = npb_of(P, C, held) - host_held
            ts = []
            for fn in (lambda s: forward(H["model"], bs(s["ids"]), res32=True),
                       lambda s: arm_logits(P, C, bs(s["ids"]))):
                for s in held[:2]:
                    mx.eval(fn(s))
                t_ = time.time()
                for s in held[:16]:
                    mx.eval(fn(s))
                ts.append((time.time() - t_) / 16 * 1000)
        finally:
            CLOSED[0] = False
        t_ = time.time()
        for s in held[:16]:
            mx.eval(arm_logits(P, C, bs(s["ids"])))
        t_open = (time.time() - t_) / 16 * 1000
        return {"npb_closed_minus_host_mean": float(d.mean()), "npb_closed_minus_host_maxabs": float(np.abs(d).max()),
                "forward_ms": {"host": round(ts[0], 2), "overlay_removed": round(ts[1], 2),
                               "overlay_live": round(t_open, 2)},
                "gate_mean_english_test": gate_means(P, C, eng_test), "gate_mean_held": gate_means(P, C, held[:32]),
                "note": "'closed' = the overlay removed (the opt-in off); the sigmoid gate itself never reaches 0, see "
                        "gate_mean_english_test. A core delta (ov_sink_core) is always on, so its closed cost is not 0"}

    def score(name, P, C, tile, tinfo, extra):
        held_npb = npb_of(P, C, held, tile)
        dev_npb = npb_of(P, C, dev, tile)
        d = host_held - held_npb
        dd = host_dev - dev_npb
        gap = host_held - donor_held
        e = {"held_npb": float(held_npb.mean()), "host_held_npb": float(host_held.mean()),
             "donor_held_npb": float(donor_held.mean()), "delta_vs_host_held": float(d.mean()),
             "gap_closed_held": float(d.mean() / gap.mean()) if abs(gap.mean()) > 1e-9 else None,
             "paired_vs_host_held": PS.paired(d, [s["group"] for s in held], mue=a.min_useful_npb),
             "dev_npb": float(dev_npb.mean()), "delta_vs_host_dev": float(dd.mean()),
             "paired_vs_host_dev": PS.paired(dd, [s["group"] for s in dev], mue=a.min_useful_npb),
             "eng_drift_test": drift_of(P, C, eng_test, tile), "eng_drift_val": drift_of(P, C, eng_val, tile),
             "closed_gate": closed_cost(P, C, tile), "trainable_params": n_params(P),
             "served_path": None if tile else served_path(P, C)} | (extra or {})
        if tinfo:
            e |= {k: v for k, v in tinfo.items() if k != "curve"}
        with open(out / "rows.jsonl", "a") as f_:
            for s, v, w in zip(held, held_npb, host_held):
                f_.write(json.dumps({"arm": name, "set": "held", "probe_id": s["probe_id"], "group": s["group"],
                                     "npb": float(v), "host_npb": float(w)}) + "\n")
            for s, v, w in zip(dev, dev_npb, host_dev):
                f_.write(json.dumps({"arm": name, "set": "dev", "probe_id": s["probe_id"], "group": s["group"],
                                     "npb": float(v), "host_npb": float(w)}) + "\n")
        return e, held_npb

    def train_loop(name, Pbox, C, tile, lr, batches, n_steps, eval_at, loss_kind, train_probe=None, rebuild=None,
                   wmult=1.0):
        """generic: batches(step) -> list of seqs; eval at the steps in eval_at; retain step every step. Keeps only
        the checkpoints the pick can still return (the current pick, the lowest-drift one); init (checkpoint 0) is
        rebuilt by rebuild() when picked (its parameters are not held through training)"""
        opt = Adam(lr, mults(tile, wmult), warmup=int(round(a.warmup_frac * n_steps)) if not tile else 0)
        lam = [a.drift_weight]
        gt = on_ref(mx.value_and_grad(lambda p, b: loss_topic(p, C, b, loss_kind, tile)))
        ge = on_ref(mx.value_and_grad(lambda p, e, l_: loss_eng(p, C, e, l_, tile)))

        def evals(step, ls):
            r = {"arm": name, "step": step, "train_loss": float(np.mean(ls)) if ls else None,
                 "val_ce": float(npb_of(P_[0], C, va, tile).mean()), "val_drift": drift_of(P_[0], C, eng_val, tile),
                 "drift_weight": lam[0], "lr": opt.lr_now()}
            if train_probe is not None:
                tp_ = [seq_nll(P_[0], C, s, tile) for s in train_probe]
                r["train_probe_ce_npt"] = float(np.mean([x.mean() for x in tp_]))
                r["train_probe_ce"] = float(np.mean([x.sum() / s["bytes"] for x, s in zip(tp_, train_probe)]))
                r["dev_ce"] = float(npb_of(P_[0], C, dev[:a.track_dev], tile).mean())
                r["dev_ce_texts"] = len(dev[:a.track_dev])
            return r
        P = Pbox.pop()  # the caller handed over its only reference: init is not held through training
        P_ = [P]
        hist = [dict(evals(0, []), epoch=0)]
        snaps = {0: dict(P)} if rebuild is None else {}
        del P
        disk = n_params(P_[0]) > a.disk_snapshot_params  # large overlays: kept checkpoints live on disk, not in memory
        sdir = out / "ckpt" / name

        def keep_snap(ep_):
            if disk:
                sdir.mkdir(parents=True, exist_ok=True)
                mx.save_safetensors(str(sdir / f"{ep_}.safetensors"), P_[0])
                snaps[ep_] = sdir / f"{ep_}.safetensors"
            else:
                snaps[ep_] = dict(P_[0])

        def drop_snap(ep_):
            v_ = snaps.pop(ep_)
            if disk and isinstance(v_, Path):
                v_.unlink(missing_ok=True)
        log(f"  {name} step 0: val CE {hist[0]['val_ce']:.4f} drift {hist[0]['val_drift']:.4f}")
        ls = []
        for step in range(1, n_steps + 1):
            b = batches(step)
            l_, g = gt(P_[0], b)
            mx.eval(l_, g)  # one backward graph at a time: the topic and retain activations never coexist
            ls.append(float(l_.item()))
            if a.drift_weight > 0 and retain:
                _, g2 = ge(P_[0], retain[step % len(retain)], lam[0])
                for k in g:
                    g[k] = g[k] + g2[k]
                    mx.eval(g[k])
                del g2
            P_[0] = opt.step(P_[0], g)
            del g
            mx.eval(P_[0], *opt.m.values(), *opt.v.values())
            stop = budget_left() < 600
            if step in eval_at or stop or step == n_steps:
                r = dict(evals(step, ls), epoch=len(hist))
                hist.append(r)
                keep = {LAB.pick_epoch(hist, "val_ce", a.drift_cap)[0],
                        min(hist, key=lambda h_: h_["val_drift"])["epoch"]} | ({0} if rebuild is None else set())
                if r["epoch"] in keep:
                    keep_snap(r["epoch"])
                for k_ in [k_ for k_ in snaps if k_ not in keep]:
                    drop_snap(k_)
                if r["val_drift"] > a.drift_cap and lam[0] > 0:
                    lam[0] = min(lam[0] * a.drift_adapt, a.drift_weight_max)
                with open(out / "curves.jsonl", "a") as f_:
                    f_.write(json.dumps(r) + "\n")
                log(f"  {name} step {step}/{n_steps}: train {r['train_loss']:.4f} val CE {r['val_ce']:.4f} "
                    f"drift {r['val_drift']:.4f}" + (f" dev {r['dev_ce']:.4f}" if "dev_ce" in r else ""))
                ls = []
                if stop:
                    break
        ck, within = LAB.pick_epoch(hist, "val_ce", a.drift_cap)
        P_[0] = None
        opt.m.clear()
        opt.v.clear()
        mx.clear_cache()
        if ck in snaps:
            best = snaps[ck] if not isinstance(snaps[ck], Path) else dict(mx.load(str(snaps[ck])))
        else:
            best = rebuild()
        for k_ in list(snaps):
            drop_snap(k_)
        return best, {"selected_checkpoint": ck, "selected_step": hist[ck]["step"], "checkpoints": len(hist) - 1,
                           "steps_run": hist[-1]["step"], "steps_planned": n_steps,
                           "truncated": hist[-1]["step"] < n_steps, "snapshots_on_disk": disk,
                           "reverted_to_init": ck == 0, "within_drift_cap": bool(within),
                           "selection_rule": f"lowest val_ce (gold text nats/byte, never a donor output) among "
                                             f"checkpoints with val drift <= {a.drift_cap}",
                           "val_ce_selected": hist[ck]["val_ce"], "drift_weight_final": lam[0], "curve": hist}

    def preflight(P, C, tile, name):
        """write scale 0 (overlay) / w_out 0 (tile) = the host exactly; gate closed = the host exactly"""
        s = va[0]
        ref = forward(H["model"], bs(s["ids"]), res32=True)
        if tile:
            dz = float(mx.abs(arm_logits(P, C, bs(s["ids"]), True) - ref).max().item())
            return {"init_equals_host_max_abs_logit": dz}
        Z = {k: (mx.zeros_like(v) if k.endswith((".wd", ".V")) else v) for k, v in P.items()}
        dz = float(mx.abs(arm_logits(Z, C, bs(s["ids"])) - ref).max().item())
        CLOSED[0] = True
        try:
            Pc = {k: (mx.zeros_like(v) if k.endswith(".V") else v) for k, v in P.items()}
            dc = float(mx.abs(arm_logits(Pc, C, bs(s["ids"])) - ref).max().item())
        finally:
            CLOSED[0] = False
        d0 = float(mx.abs(arm_logits(P, C, bs(s["ids"])) - ref).max().item())
        return {"write0_equals_host_max_abs_logit": dz, "gate_closed_equals_host_max_abs_logit": dc,
                "init_vs_host_max_abs_logit": d0}

    # host gradient path parity (as the lab): reference vs kernel forward, one real gradient
    def grad_check():
        s = tr[0]
        lk = forward(H["model"], bs(s["ids"]), res32=True)[0]
        with dref.reference_path():
            lr_ = forward(H["model"], bs(s["ids"]), res32=True)[0]
        pk = lk - mx.logsumexp(lk, -1, keepdims=True)
        pr = lr_ - mx.logsumexp(lr_, -1, keepdims=True)
        kl = float((mx.exp(pk) * (pk - pr)).sum(-1).mean().item())
        return {"kl_kernel_ref_mean": kl, "max_kl_allowed": a.grad_ref_max_kl}
    hg = grad_check()
    receipt["host_grad_path"] = hg
    if not hg["kl_kernel_ref_mean"] <= a.grad_ref_max_kl:
        save(status="failed", reason=f"reference path differs from kernel path: {hg}")
        sys.exit("reference path parity failed")
    kl_mults = [float(x) for x in a.scale_kl_mults.split(",") if x.strip()] if a.stage == "scale" else []
    Dn_keep = {"ck": Dn["ck"]}

    def unload_donor():
        """drop every reference to the donor (the adapter holds the model too) and say how much memory left"""
        import gc
        before = mx.get_active_memory()
        for k_ in ("model", "layers", "ad"):
            Dn.pop(k_, None)
        gc.collect()
        mx.clear_cache()
        freed = (before - mx.get_active_memory()) / 1e9
        if donor_gb > 1 and freed < 0.9 * donor_gb:
            save(status="failed", reason=f"donor unload freed {freed:.1f} GB of {donor_gb:.1f}: a reference survives")
            sys.exit("donor not unloaded")
        receipt["donor_unload"] = {"active_gb_before": round(before / 1e9, 2), "donor_gb": round(donor_gb, 2),
                                   "active_gb_after": round(mx.get_active_memory() / 1e9, 2)}
        log(f"donor unloaded: active {before / 1e9:.1f} -> {mx.get_active_memory() / 1e9:.1f} GB")
    if not kl_mults:
        unload_donor()
    save(stage="arms" if a.stage == "arms" else "scale")
    summary = {"arms": {}, "selection": wins, "held_texts": len(held), "dev_texts": len(dev),
               "min_useful_effect_npb": a.min_useful_npb, "host_checkpoint": H["ck"],
               "donor_checkpoint": Dn_keep["ck"], "mlx2_source_revision": rev,
               "placement_note": "every overlay arm, the host-only CE controls included, is placed (layers, units, gate "
                                 "fit, sink ranking) from host activations at the donor-better train positions: the "
                                 "ce_gold arms are donor-placed, host-trained"}
    NPB = {}

    def save_ov(name, P, C, meta):
        if a.save_overlays == "off" or (a.save_overlays == "auto" and n_params(P) > 4e9):
            return None
        return save_overlay(out / "overlays" / name, P, C, dict(meta, host_checkpoint=H["ck"], mlx2_source_revision=rev,
                                                                 route="amoeba_overlay", opt_in=True,
                                                                 qualification="unqualified research"))

    if a.stage == "arms":
        arms_req = sorted(arms_req, key=lambda n: n != "tile_sft")  # the bar first: a budget cut never drops it
        for name in arms_req:
            if budget_left() < 900:
                summary["stopped_before"] = name
                log(f"budget: stopping before {name}")
                break
            arm = ARMS[name]
            ta = time.time()
            tile = bool(arm.get("tile"))
            P, C, info = build_arm(name, arm)
            pf = preflight(P, C, tile, name)
            if max(v for k, v in pf.items() if k != "init_vs_host_max_abs_logit") > 1e-4:
                save(status="failed", reason=f"{name} preflight: {pf}")
                sys.exit(f"{name}: preflight failed {pf}")
            data = resp if arm["loss"] == "sft_donor" else tr
            leak_ = {d_["probe_id"] for d_ in data} & ({s_["probe_id"] for s_ in held + dev + va})
            if leak_:
                save(status="failed", reason=f"{name}: training texts overlap held/dev/val: {sorted(leak_)[:5]}")
                sys.exit(f"{name}: training leak {sorted(leak_)[:5]}")
            if not data:
                summary["arms"][name] = {"skipped": "no training sequences"}
                continue
            nb_ = (len(data) + a.batch - 1) // a.batch
            n_steps = nb_ * a.epochs
            marks = {int(round(nb_ * (k + 1) / a.evals_per_epoch)) + ep * nb_ for ep in range(a.epochs)
                     for k in range(a.evals_per_epoch)}
            order_rng = np.random.default_rng(a.seed + 501)  # every arm sees the same text order
            orders_ = [order_rng.permutation(len(data)) for _ in range(a.epochs)]

            def batches(step, data=data, nb_=nb_, orders_=orders_):
                ep, j = divmod(step - 1, nb_)
                return [data[i] for i in orders_[ep][j * a.batch:(j + 1) * a.batch]]
            box_, P = [P], None
            P, tinfo = train_loop(name, box_, C, tile, a.tile_lr if tile else a.lr, batches, n_steps, marks, arm["loss"])
            e, hn = score(name, P, C, tile, tinfo, {"loss": arm["loss"], "preflight": pf, "build": info,
                                                     "train_sequences": len(data)})
            e["wall_s"] = round(time.time() - ta, 1)
            e["overlay_saved"] = bool(save_ov(name, P, C, {"arm": name, "loss": arm["loss"], "build": info}))
            NPB[name] = hn
            summary["arms"][name] = e
            (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            log(f"{name}: held {e['held_npb']:.4f} (host {e['host_held_npb']:.4f}) delta {e['delta_vs_host_held']:+.4f} "
                f"dev delta {e['delta_vs_host_dev']:+.4f} drift {e['eng_drift_test']:.4f}")
            save(stage=f"done {name}")
            mx.clear_cache()
        grp = [s["group"] for s in held]
        con = {}
        pairs = [("ov_sink_win", "tile_sft"), ("ov_sink_win", "ov_win"), ("ov_sink_win", "sink_win"),
                 ("ov_sink_win", "ov_sink_all"), ("ov_sink_win", "ov_sink_gold"), ("ov_sink_win", "ov_sink_rand"),
                 ("ov_sink_sft", "ov_sink_win"), ("ov_sink_gold", "tile_sft"), ("ov_sink_all", "tile_sft"),
                 ("ov_sink_core", "ov_sink_win"), ("ov_sink_core", "tile_sft")]
        for x, y in pairs:
            if x in NPB and y in NPB:
                c = PS.paired(NPB[y] - NPB[x], grp, mue=a.min_useful_npb)
                c["reading"] = f"positive = {x} has lower held loss than {y}"
                con[f"{x} vs {y}"] = c
        summary["contrasts_held"] = con
        # the best overlay arm is chosen on DEV (development data), then tested once on held: no winner's curse
        best = max((n for n in NPB if n not in ("tile_sft", "ov_sink_core")),
                   key=lambda n: summary["arms"][n]["delta_vs_host_dev"], default=None)
        if best:
            b_ = summary["arms"][best]
            beat = PS.paired(NPB["tile_sft"] - NPB[best], grp, mue=a.min_useful_npb) if "tile_sft" in NPB else None
            lo_h = (b_["paired_vs_host_held"].get("ci95_group") or [None])[0]
            lo_t = (beat or {}).get("ci95_group", [None])[0]
            summary["poor_rule"] = {
                "rule": "best overlay arm = largest DEV gain vs the host (tile_sft and the core arm excluded); poor = "
                        "its HELD gain vs the host has a source-group 95% lower bound < 0.001 nats/byte, OR it does "
                        "not beat tile_sft on held (group 95% lower bound of tile_sft minus best <= 0); undetermined "
                        "when tile_sft or the best arm did not finish; poor triggers the staged core window",
                "best_overlay_arm": best, "best_vs_host_held_lower": lo_h, "best_vs_tile_sft_held_lower": lo_t,
                "poor": (None if beat is None or lo_h is None or b_.get("truncated") or
                         summary["arms"]["tile_sft"].get("truncated") else bool(lo_h < a.min_useful_npb or lo_t <= 0))}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        trunc = [n for n, e in summary["arms"].items() if e.get("truncated")]
        summary["truncated_arms"] = trunc
        save(status="complete" if "stopped_before" not in summary and not trunc else "partial (budget)", stage="done",
             summary=summary)
        return 0

    # ----------------------------------------------------------------------------------- stage scale
    cdir = Path(a.corpus).expanduser()
    cman = json.loads((cdir / "manifest.json").read_text())
    if hashlib.sha256((cdir / "chunks.jsonl").read_bytes()).hexdigest() != cman["chunks_sha256"]:
        sys.exit("corpus chunks sha256 differs from its manifest")
    if cman["probes_sha256"] != receipt["probes_sha256"] or cman["split_manifest_sha256"] != \
            receipt["split_manifest"]["sha256"]:
        sys.exit("corpus was built against other probes or another split manifest")
    chunks = [json.loads(x) for x in open(cdir / "chunks.jsonl")]
    eval_groups = {LAB_norm(r["group"]) for r in rows_ if r["split"] == "held"} | {LAB_norm(s["group"]) for s in va}
    leak = [c["id"] for c in chunks if LAB_norm(c["group"]) in eval_groups]
    if leak:
        save(status="failed", reason=f"corpus chunks share groups with held / val texts: {leak[:5]}")
        sys.exit("held/fit disjointness failed")
    receipt["corpus"] = {"path": str(cdir), "chunks": len(chunks), "chunks_sha256": cman["chunks_sha256"],
                         "builder_revision": cman.get("mlx2_source_revision"), "recheck_here": "group-disjoint from "
                         "every held-split probe group and the val texts' groups (0 shared)", "build_recheck":
                         cman.get("recheck")}
    for c in chunks:
        c["P"] = 1
        c["ids"] = list(c["ids"])
    probe_g = np.random.default_rng(a.seed + 77)
    train_probe = [chunks[i] for i in probe_g.choice(len(chunks), min(a.train_probe_chunks, len(chunks)), replace=False)]
    for c in train_probe:
        c["bytes"] = len(tok.decode(c["ids"][1:]).encode("utf-8"))
    tk_ = Path(a.host).expanduser().resolve() / "tokenizer.json"
    if loader is None and (not cman.get("tokenizer_sha256") or not tk_.exists()) or tk_.exists() and \
            hashlib.sha256(tk_.read_bytes()).hexdigest() != cman.get("tokenizer_sha256"):
        sys.exit("corpus was tokenized with another tokenizer than the host's")
    mults_ = [float(x) for x in a.scale_mults.split(",") if x.strip()]
    plan = [(m, "kl_win_live") for m in kl_mults] + [(m, "ce_gold") for m in mults_]  # donor-resident sizes first
    feas = {}
    for m, _ in plan + [(1000.0, "ce_gold")]:
        cnt = scale_counts(base_counts, m)
        np_ = overlay_param_count(cnt, D, {l: min(a.gate_units, I_) for l in cnt})
        bpp = 4 if a.overlay_dtype == "float32" else 2
        gb = overlay_train_gb(np_, bpp) + activation_gb(cnt, a.scale_batch * len(chunks[0]["ids"]), bpp)
        feas[f"x{m:g}"] = {"per_layer": {str(l): n for l, n in cnt.items()}, "params": np_, "train_gb_est": round(gb, 1),
                           "fits": gb <= a.max_overlay_gb}
    receipt["scale_feasibility"] = feas
    save(stage="scale feasibility")
    plan = [(m, k) for m, k in plan if feas[f"x{m:g}"]["fits"]]
    B = a.scale_batch
    perm_rng = np.random.default_rng(a.seed + 601)
    perm = [perm_rng.permutation(len(chunks)) for _ in range(int(np.ceil(a.max_epochs)) + 1)]
    nb_ = len(chunks) // B

    def batches(step):
        ep, j = divmod(step - 1, nb_)
        return [chunks[i] for i in perm[ep][j * B:(j + 1) * B]]

    def live(b):
        """donor-wins KL on a corpus batch: donor log-probs and the donor-better mask (donor log p(gold) - clean
        host log p(gold) >= --win-margin), no gradient"""
        ids2d = [s["ids"] for s in b]
        tg = mx.array([r[1:] for r in ids2d])
        ld = forward(Dn["model"], ids2d, res32=False)[:, :-1]
        lh = forward(H["model"], ids2d, res32=True)[:, :-1]
        m = ((gold_nll(lh, tg) - gold_nll(ld, tg)) >= a.win_margin).astype(mx.float32)
        lpd = ld - mx.logsumexp(ld, axis=-1, keepdims=True)
        mx.eval(lpd, m)
        return {"seqs": b, "lpd": lpd, "mask": m}

    def batches_for(kind):
        return (lambda st_: live(batches(st_))) if kind == "kl_win_live" else batches
    # timing and memory probe: 3 steps per size from a throwaway init, so every size gets the same optimizer steps,
    # and a size whose measured peak (+ its two kept checkpoints when they stay in memory) passes --mem-cap-gb is
    # dropped before any training
    sps, peak = {}, {}
    cap_gb = a.mem_cap_gb or 1e9
    for m, kind in plan:
        mx.reset_peak_memory()
        P, C, _ = build_arm("probe", {"counts": scale_counts(base_counts, m)})
        gt = on_ref(mx.value_and_grad(lambda p, b: loss_topic(p, C, b, kind, False)))
        bf = batches_for(kind)
        l_, g = gt(P, bf(1))
        mx.eval(l_, g)
        t_ = time.time()
        for st_ in range(2, 5):
            l_, g = gt(P, bf(st_))
            mx.eval(l_, g)
        sps[f"x{m:g}_{kind}"] = (time.time() - t_) / 3 * 1.25  # + retain step, Adam, evals
        npar = n_params(P)
        bpp_ = 4 if a.overlay_dtype == "float32" else 2
        # beyond the probe (P + grad): old and new Adam m and v, the new P (+ two in-memory snapshots)
        extra = npar * bpp_ * (5 + (2 if npar <= a.disk_snapshot_params else 0)) / 1e9
        resident_donor = donor_gb if Dn.get("model") is not None and kind != "kl_win_live" else 0.0
        peak[f"x{m:g}_{kind}"] = {"probe_peak_gb": round(mx.get_peak_memory() / 1e9, 2), "train_peak_est_gb":
                                  round(mx.get_peak_memory() / 1e9 + extra - resident_donor, 2),
                                  "donor_gb_subtracted": round(resident_donor, 2)}
        P = C = g = None
        mx.clear_cache()
    receipt["scale_probe_memory"] = peak
    fit_ = [(m, k) for m, k in plan if peak[f"x{m:g}_{k}"]["train_peak_est_gb"] <= cap_gb]
    dropped = [f"x{m:g}_{k}" for m, k in plan if (m, k) not in fit_]
    if a.largest_fitting and fit_:
        fit_ = [max(fit_, key=lambda mk: mk[0])]
    for k_ in list(sps):
        if k_ not in {f"x{m:g}_{k}" for m, k in fit_}:
            sps.pop(k_)
    plan = fit_
    receipt["scale_dropped_by_memory"] = dropped
    if not plan:
        save(status="failed", reason=f"no size fits --mem-cap-gb {a.mem_cap_gb}: {peak}")
        sys.exit("no scale size fits")
    steps = int(a.train_budget_s / max(sum(sps.values()), 1e-6))
    steps = min(steps, int(a.max_epochs * nb_)) if a.max_steps == 0 else min(a.max_steps, int(a.max_epochs * nb_))
    receipt["scale_plan"] = {"sizes": [f"x{m:g}_{k}" for m, k in plan], "s_per_step_est": sps, "steps_per_size": steps,
                             "batch": B, "chunk_tokens": len(chunks[0]["ids"]), "epochs": round(steps * B / len(chunks), 3),
                             "corpus_fraction_seen": round(min(1.0, steps * B / len(chunks)), 3),
                             "x1000_note": feas.get("x1000")}
    save(stage="scale training")
    log(f"scale plan {receipt['scale_plan']}")
    eval_at = {int(round(steps * (k + 1) / a.scale_evals)) for k in range(a.scale_evals)}
    for m, kind in plan:
        name = f"scale_x{m:g}_{kind}"
        if kind != "kl_win_live" and Dn.get("model") is not None:
            unload_donor()  # the donor-wins sizes ran first
        if budget_left() < 900:
            summary["stopped_before"] = name
            break
        ta = time.time()
        cnt = scale_counts(base_counts, m)
        P, C, info = build_arm(name, {"counts": cnt})
        pf = preflight(P, C, False, name)
        if max(v for k, v in pf.items() if k != "init_vs_host_max_abs_logit") > 1e-4:
            save(status="failed", reason=f"{name} preflight: {pf}")
            sys.exit(f"{name}: preflight failed {pf}")
        box_, P = [P], None
        P, tinfo = train_loop(name, box_, C, False, a.lr, batches_for(kind), steps, eval_at, kind,
                              train_probe=train_probe,
                              rebuild=lambda cnt=cnt, name=name: build_arm(name, {"counts": cnt})[0],
                              wmult={"inv_sqrt": 1 / np.sqrt(m), "inv": 1 / m, "flat": 1.0}[a.scale_write_lr])
        distinct = {str(l): int(len(set(expand_units(rank_of(l), n)[0].tolist()))) for l, n in cnt.items()}
        e, hn = score(name, P, C, False, tinfo, {"loss": kind, "mult_per_layer": m, "preflight": pf, "build": info,
                                                 "distinct_host_units_copied": distinct,
                                                 "write_lr_mult": a.write_lr_mult * {"inv_sqrt": 1 / np.sqrt(m),
                                                                                     "inv": 1 / m, "flat": 1.0}[
                                                     a.scale_write_lr],
                                                 "steps": steps, "tokens_seen": tinfo["steps_run"] * B *
                                                 len(chunks[0]["ids"])})
        e["wall_s"] = round(time.time() - ta, 1)
        e["overlay_saved"] = bool(save_ov(name, P, C, {"arm": name, "loss": kind, "build": info}))
        NPB[name] = hn
        summary["arms"][name] = e
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"{name}: held delta {e['delta_vs_host_held']:+.4f} dev delta {e['delta_vs_host_dev']:+.4f} "
            f"drift {e['eng_drift_test']:.4f} params {e['trainable_params']}")
        save(stage=f"done {name}")
        P = C = None
        mx.clear_cache()
    names = [n for n in NPB]
    grp = [s["group"] for s in held]
    summary["scaling"] = {
        "sizes": [{"arm": n, "mult_per_layer": summary["arms"][n]["mult_per_layer"],
                   "params": summary["arms"][n]["trainable_params"],
                   "held_gain": summary["arms"][n]["delta_vs_host_held"],
                   "dev_gain": summary["arms"][n]["delta_vs_host_dev"],
                   "eng_drift_test": summary["arms"][n]["eng_drift_test"]} for n in names],
        "steps": [PS.paired(NPB[y] - NPB[x], grp, mue=a.min_useful_npb) | {"pair": f"{x} vs {y}"}
                  for kind_ in ("ce_gold", "kl_win_live")
                  for y, x in zip(*(lambda ns: (ns, ns[1:]))(sorted(
                      [n for n in names if summary["arms"][n]["loss"] == kind_],
                      key=lambda n: summary["arms"][n]["mult_per_layer"])))],
        "corpus_fraction_seen": receipt["scale_plan"]["corpus_fraction_seen"],
        "reading": "steps: positive = the larger size has lower held loss; rising = every step's group lower bound "
                   "> 0; the 0c bar is tile_sft +0.0388 nats/byte on the DEV texts with drift 0.055 (compare dev_gain)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    trunc = [n for n, e in summary["arms"].items() if e.get("truncated")]
    summary["truncated_arms"] = trunc
    save(status="complete" if "stopped_before" not in summary and not trunc else "partial (budget)", stage="done",
         summary=summary)
    return 0


def LAB_norm(g):
    return str(g).split("#", 1)[0]


def run_guarded(argv=None, loader=None):
    """main() with the exit handler: any abnormal exit, a sys.exit(<message>) refusal included, leaves run.json
    at "failed" with the reason instead of "running" (as amoeba_bridge_lab does). A receipt some other path
    already closed (failed, failed (watchdog), complete) is left as written."""
    _STATE.clear()
    try:
        return main(argv, loader)
    except BaseException as exc:
        out = _STATE.get("out")
        try:
            running = out is not None and json.loads((out / "run.json").read_text()).get("status") == "running"
        except Exception:
            running = False
        if running and not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            why = (str(exc.code) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}")[:500]
            try:
                if "save" in _STATE:
                    _STATE["save"](status="failed", reason=why)
                else:  # refused before the full receipt and its saver exist
                    (out / "run.json").write_text(json.dumps(dict(_STATE["receipt"], status="failed", reason=why),
                                                             indent=1, default=str))
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    sys.exit(run_guarded())
