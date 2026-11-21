"""English cost and unit statistics over any donor (domain lists come from the corpus split).

text_ce / english_cost / mean_acts: scripts/amoeba27b_lib.py (claude/amoeba-27b-80bc6a@37d75f144)
generalised to corpus-defined domains and to ragged unit sites. Same positions (1..T-2), same
batching, so a 27B run here gives the 27B windows' numbers.
"""
from __future__ import annotations

from .carving import ce_rows


def text_ce(host, seqs, cfg_fn=None, bs=8, rows_out=None):
    """Mean CE over scored targets. Sequences are batched by length (a fixed held file mixes lengths); a sequence
    with a loss mask (corpus.LOSS_FROM: on-policy rows) scores targets from its first scored index only.

    rows_out (opt-in, docs/research/2026-11-20-amoeba-next-experiments.md 7.1): a dict that receives "ce" and "n",
    float64 / int arrays in the order of seqs: each sequence's CE sum over its scored targets and their count. The
    pooled return value is computed exactly as without it (the same batch sums), so it is unchanged bit for bit."""
    import mlx.core as mx
    import numpy as np
    from .corpus import loss_from
    tot, n = 0.0, 0
    by_len = {}
    for j, q in enumerate(seqs):
        by_len.setdefault(len(q), []).append(j)
    if rows_out is not None:
        rows_out["ce"] = np.zeros(len(seqs), np.float64)
        rows_out["n"] = np.zeros(len(seqs), np.int64)
    for _, group in sorted(by_len.items()):
        for i in range(0, len(group), bs):
            ii = group[i:i + bs]
            rows = [seqs[j] for j in ii]
            ids = mx.array(rows)
            T = ids.shape[1]
            w = np.array([[1.0 if t >= max(2, loss_from(q)) else 0.0 for t in range(2, T)] for q in rows], np.float32)
            if cfg_fn:
                cfg_fn()
            lg = host.logits_at(ids, list(range(1, T - 1)))
            host.ctl.reset() if cfg_fn else None
            cw = ce_rows(lg, ids[:, 2:]) * mx.array(w)
            ce = cw.sum()
            if rows_out is not None:
                per = cw.sum(axis=1)
                mx.eval(ce, per)
                rows_out["ce"][ii] = np.array(per, dtype=np.float64)
                rows_out["n"][ii] = w.sum(1).astype(np.int64)
            else:
                mx.eval(ce)
            tot += float(ce)
            n += int(w.sum())
            mx.clear_cache()
    return tot / max(1, n), n


def english_cost(host, split, clean_cache, cfg_fn, part="held", domains=None, pooled=False, perdoc=None):
    """dCE per domain and their mean. pooled=True (FN boxes, coordinator 2026-11-16): the mean is weighted by each
    domain's sequence count (every held row counts once: probe windows, 1 greedy and 3 T=0.7 continuations per
    prompt), and the counts are returned under "domain_rows".

    perdoc (opt-in, a dict): also receives, per domain, {"ce", "clean_ce", "n"}: per-document CE sums of this arm
    and of the clean model, and the scored-target counts (text_ce rows_out); the returned values are unchanged. The
    clean per-document sums are cached in clean_cache under ("rows", domain)."""
    domains = list(domains or split)
    r = {}
    for dn in domains:
        if not split[dn].get(part):
            continue
        if perdoc is not None and ("rows", dn) not in clean_cache:
            cr = {}
            c0 = text_ce(host, split[dn][part], rows_out=cr)[0]
            clean_cache.setdefault(dn, c0)
            clean_cache[("rows", dn)] = cr
        if dn not in clean_cache:
            clean_cache[dn] = text_ce(host, split[dn][part])[0]
        if perdoc is not None:
            rr = {}
            r[dn] = text_ce(host, split[dn][part], cfg_fn, rows_out=rr)[0] - clean_cache[dn]
            perdoc[dn] = {"ce": rr["ce"], "clean_ce": clean_cache[("rows", dn)]["ce"], "n": rr["n"]}
        else:
            r[dn] = text_ce(host, split[dn][part], cfg_fn)[0] - clean_cache[dn]
    used = [d for d in domains if d in r]
    # nothing scored (every domain empty): None, never 0.0 (review 2026-11-16: an empty held set read as a pass)
    if pooled:
        n = {d: len(split[d][part]) for d in used}
        tot = sum(n.values())
        r["mean_d_ce"] = sum(r[d] * n[d] for d in used) / tot if tot else None
        r["domain_rows"] = n
    else:
        r["mean_d_ce"] = sum(r[d] for d in used) / len(used) if used else None
    return r


def configure(host, core=None, mean=None, sink=0, dense=(), boxes=()):
    def f():
        c = host.ctl
        c.reset()
        c.core, c.mean, c.sink_layers, c.dense_layers = core, mean, sink, tuple(dense)
        c.boxes = list(boxes)
    return f


def mean_acts(host, seqs, bs=8):
    """Mean unit signal over positions >= 1, per layer: the down_proj input for dense sites (the
    mean-ablation value), the mean routed weight per expert for MoE sites (selection rate x weight;
    reported only, MoE masks use the drop rule). Ragged list of float32 arrays.

    Neuron MoE units (--moe-units neuron): the CONDITIONAL mean of the activated intermediate h[e, f]
    over the tokens routed to expert e (sum over routed slots / slot count of e; 0 for an expert never
    routed). Reported only: MoE neuron masks use the drop rule (zero ablation of h) in every ablation.
    Accumulated by scatter-add over the routed slots, never as a tokens x E x F one-hot."""
    import mlx.core as mx
    acc = [mx.zeros((n,), dtype=mx.float32) for n in host.widths]
    cnt = [0]
    neuron = {s.layer: s for s in host.sites if s.is_moe and s.units == "neuron"}
    ecnt = {l: mx.zeros((s.E,), dtype=mx.float32) for l, s in neuron.items()}

    def stats(l, kind, p):
        if l in neuron:
            s = neuron[l]
            inds = p["inds"][:, 1:].astype(mx.int32)
            h = p["h"][:, 1:].astype(mx.float32)                                  # B, T-1, k, F
            flat = (inds[..., None] * s.F + mx.arange(s.F, dtype=mx.int32)).reshape(-1)
            acc[l] = acc[l].at[flat].add(h.reshape(-1))
            ecnt[l] = ecnt[l].at[inds.reshape(-1)].add(mx.ones((inds.size,), mx.float32))
        elif kind == "dense":
            a = p["act"]
            acc[l] = acc[l] + a[:, 1:].astype(mx.float32).reshape(-1, a.shape[-1]).sum(0)
        elif kind is not None:
            inds, w = p["inds"][:, 1:], p["w"][:, 1:]
            n = host.widths[l]
            oh = (inds[..., None] == mx.arange(n)).astype(mx.float32) * w[..., None]
            acc[l] = acc[l] + oh.reshape(-1, oh.shape[-2], n).sum((0, 1))
        if l == 0:
            cnt[0] += p["x"].shape[0] * (p["x"].shape[1] - 1) if p.get("x") is not None else 0
    for i in range(0, len(seqs), bs):
        host.ctl.reset()
        host.ctl.stats = stats
        mx.eval(host.hidden(mx.array(seqs[i:i + bs])), *acc, *ecnt.values())
        host.ctl.reset()
        mx.clear_cache()
    if cnt[0] == 0:
        cnt[0] = sum(len(s) - 1 for s in seqs)
    out = [a / cnt[0] if l not in neuron else
           a / mx.repeat(mx.maximum(ecnt[l], 1.0), neuron[l].F) for l, a in enumerate(acc)]
    return mx.stack(out) if host.N is not None else out


def dense_mean(host, mean):
    """Mean-ablation values for dense sites only (None at MoE sites: they use the drop rule)."""
    if mean is None:
        return None
    kinds = [s.kind for s in host.sites]
    if all(k == "dense" for k in kinds):
        return mean
    import mlx.core as mx
    return [mean[l] if k == "dense" else None for l, k in enumerate(kinds)] if host.N is None else \
        mx.stack([mean[l] if k == "dense" else mx.zeros_like(mean[l]) for l, k in enumerate(kinds)])
