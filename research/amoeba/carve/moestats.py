"""MoE neuron statistics in one forward pass (docs/research/2026-11-20-amoeba-next-experiments.md 1.3 and 4.3).

capture(host, seqs) runs the masked-path forward with a stats hook (forward only, the model unchanged) and returns,
per neuron MoE site (unit = expert * F + neuron, positions >= 1):

  mu        [E*F] conditional mean of the activated intermediate h[e, f] over the slots routed to e (the mean-
            ablation value; evals.mean_acts computes the same, this adds the other sums in the same pass)
  wh        [E*F] mean over tokens of w_e * |h[e, f]| (0 when e is not routed): the neuron-level REAP saliency
            before the down-column norm
  count     [E]   routed slots per expert (routing frequency; SEER-style "heavy hitters")
  reap_e    [E]   mean over the slots routed to e of w_e * ||f_e(x)|| (REAP, Lasby et al. 2025)
  tokens    int   positions counted
  habs_q    {q: threshold} per-layer quantiles of |h| over routed slots, from a strided subsample (TEAL-style)

down_col_norms(host) returns, per neuron MoE site, ||W_down[e][:, f]|| (dequantized in chunks of experts, so no
whole-layer float copy sits beside the model).

Status: implemented and CPU-tested on toy hosts; not yet run on FN.
"""
from __future__ import annotations

import numpy as np


def neuron_sites(host):
    return [s for s in host.sites if s.is_moe and s.units == "neuron"]


def capture(host, seqs, bs=8, quantiles=(), sample_per_batch=65536, log=None):
    import mlx.core as mx
    sites = {s.layer: s for s in neuron_sites(host)}
    acc = {l: {"h": mx.zeros((s.E * s.F,), mx.float32), "wh": mx.zeros((s.E * s.F,), mx.float32),
               "cnt": mx.zeros((s.E,), mx.float32), "reap": mx.zeros((s.E,), mx.float32)} for l, s in sites.items()}
    samples = {l: [] for l in sites}
    tok = [0]

    def stats(l, kind, p):
        if l not in sites:
            return
        s = sites[l]
        k = p["inds"].shape[-1]
        inds = p["inds"][:, 1:].astype(mx.int32)                               # B, T-1, k
        h = p["h"][:, 1:, :k].astype(mx.float32)                               # B, T-1, k, F
        w = p["w"][:, 1:, :k].astype(mx.float32)                               # B, T-1, k
        rows = p["rows"][:, 1:, :k].astype(mx.float32)                         # B, T-1, k, D
        flat = (inds[..., None] * s.F + mx.arange(s.F, dtype=mx.int32)).reshape(-1)
        a = acc[l]
        a["h"] = a["h"].at[flat].add(h.reshape(-1))
        a["wh"] = a["wh"].at[flat].add((mx.abs(h) * w[..., None]).reshape(-1))
        ei = inds.reshape(-1)
        a["cnt"] = a["cnt"].at[ei].add(mx.ones((ei.size,), mx.float32))
        a["reap"] = a["reap"].at[ei].add((w * mx.sqrt((rows * rows).sum(-1))).reshape(-1))
        if quantiles:
            ha = mx.abs(h).reshape(-1)
            step = max(1, int(ha.size) // sample_per_batch)
            samples[l].append(np.asarray(ha[::step], np.float32))      # to numpy now: never hold h alive
        mx.eval(*a.values())

    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        by_len = {}
        for q in batch:
            by_len.setdefault(len(q), []).append(q)
        for _, rows_ in sorted(by_len.items()):
            host.ctl.reset()
            host.ctl.stats = stats
            host.ctl.stats_fast = True                  # the captured rows are the block output (no second MoE call)
            mx.eval(host.hidden(mx.array(rows_)))
            host.ctl.reset()
            tok[0] += sum(len(q) - 1 for q in rows_)
            for l in samples:
                samples[l] = [np.concatenate(samples[l])] if samples[l] else []
            mx.clear_cache()
        if log and (i // bs) % 8 == 0:
            log(f"  moe stats {min(i + bs, len(seqs))}/{len(seqs)}")
    out = {}
    T = max(1, tok[0])
    for l, s in sites.items():
        a = {k: np.asarray(v, np.float64) for k, v in acc[l].items()}
        cnt = a["cnt"]
        out[l] = {"mu": (a["h"].reshape(s.E, s.F) / np.maximum(cnt, 1.0)[:, None]).reshape(-1).astype(np.float32),
                  "wh": (a["wh"] / T).astype(np.float32), "count": cnt, "reap_e": a["reap"] / np.maximum(cnt, 1.0),
                  "tokens": int(T)}
        if quantiles:
            v = samples[l][0] if samples[l] else np.zeros(1, np.float32)
            out[l]["habs_q"] = {float(q): float(np.quantile(v, q)) for q in quantiles}
    return out


def down_col_norms(host, chunk=32):
    """Per neuron MoE site: [E*F] float32 ||W_down[e][:, f]|| (D-norm of each down_proj input column)."""
    import mlx.core as mx
    from .units import _switch
    out = {}
    for s in neuron_sites(host):
        dp = _switch(s.module, s.kind).down_proj
        E = int(dp.weight.shape[0])
        rows = []
        for e0 in range(0, E, chunk):
            sl = slice(e0, min(E, e0 + chunk))
            if getattr(dp, "scales", None) is not None:
                kw = {"mode": dp.mode} if getattr(dp, "mode", None) not in (None, "affine") else {}
                b = getattr(dp, "biases", None)
                W = mx.dequantize(dp.weight[sl], dp.scales[sl], None if b is None else b[sl], dp.group_size, dp.bits,
                                  **kw)
            else:
                W = dp.weight[sl]
            W = W.astype(mx.float32)                                           # [e, D, F]
            n = mx.sqrt((W * W).sum(axis=1))
            mx.eval(n)
            rows.append(np.asarray(n, np.float32))
            del W, n
            mx.clear_cache()
        out[s.layer] = np.concatenate(rows, 0)[: s.E].reshape(-1)
    return out
