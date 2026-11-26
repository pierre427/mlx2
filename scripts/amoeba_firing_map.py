#!/usr/bin/env python3
"""Neuron firing map: which MoE expert neurons a language fires, with document-level statistics (Pierre 2026-11-16:
"statistically significant neural firing maps, not one offs"; "optimize the performance of this part"). Inside a GPU
window. Generic over the carve driver's MoE sites (neuron units: unit id = expert * F + neuron).

Forward only (no gradients, no head): every --target and --baseline document (split "fit" only; val / held / held2
untouched) is cut to its first --tokens tokens (shorter documents are skipped and counted) and run in batches of
--batch documents through the model's own trunk. At every MoE layer the capture hook gives the routed slots' activated
hidden h [B, T, k, F] (the down_proj input) and routing weights w; a neuron's contribution at a token is |h| * w (0 when
its expert is not routed). Per document the mean contribution over its tokens is one sample; per neuron and group the
script keeps the sum and sum of squares of those samples (float32 on the GPU, float64 at the end), and per expert the
routed share. Groups: target / baseline, each split in two halves by document (seeded) for a reproducibility check.

Statistics per neuron (target vs baseline): mean difference, Cohen's d (pooled sd), Welch t, two-sided normal p
(n >= a few hundred documents per side), Benjamini-Hochberg over all neurons. A neuron is a target neuron at
q <= --q and d >= --min-d (target fires it more). Split-half: the same test on each half; reported as the overlap
(Jaccard) of the two halves' target sets and the correlation of their d. Also: per layer counts, overlap with a base
tier (--base, e.g. T2 expanded to neurons), concentration (how many experts hold 90% of the target neurons).

Writes OUT/map.npz (per-group n, mean, sd; d, t, q, target mask) and OUT/report.json (+ printed summary, tokens/s).

Usage: amoeba_firing_map.py --model M --target CORPUS.jsonl --baseline CORPUS.jsonl --out DIR [--base T2.npz]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_docs(p, split):
    rows = [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]
    return [r for r in rows if r.get("split") == split]


def bh(p):
    """Benjamini-Hochberg q-values (flat array)."""
    n = p.size
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="", help="a base tier NPZ (key core; expert units are expanded to neurons)")
    ap.add_argument("--split", default="fit")
    ap.add_argument("--max-docs", type=int, default=3000, help="per side")
    ap.add_argument("--tokens", type=int, default=256, help="tokens per document (shorter documents are skipped; "
                    "256 keeps nearly every document of both corpora, review 2026-11-16)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--min-d", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--no-fast", action="store_true", help="run the stock MoE call after the capture (2x MoE cost)")
    ap.add_argument("--bench", type=int, default=0, help="stop after this many batches (throughput probe)")
    a = ap.parse_args()
    import mlx.core as mx
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    O.footprint_watchdog(a.watchdog_gb, log)
    O.setup_mlx(a.memory_gb, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    import os
    if "Flash-Next" in str(a.model) or "qwen4" in str(getattr(host, "adapter", "")).lower():
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    sites = [s for s in host.sites if getattr(s, "E", None)]
    if not sites:
        raise SystemExit("no MoE sites (neuron units) in this model")
    E, F = sites[0].E, sites[0].F
    if any((s.E, s.F) != (E, F) for s in sites):
        raise SystemExit("mixed expert shapes across layers: not supported here")
    moe_layers = [s.layer for s in sites]
    Lm = len(moe_layers)
    lidx = {l: i for i, l in enumerate(moe_layers)}
    T = a.tokens
    base = None                                    # checked before the capture (review: a late crash loses the window)
    if a.base:
        bm = np.load(a.base)["core"] > 0
        if bm.shape == (len(host.widths), E):          # expert units -> neurons (unit = expert * F + neuron)
            bm = np.repeat(bm, F, axis=1)
        if bm.shape != (len(host.widths), E * F):
            raise SystemExit(f"--base {a.base}: shape {bm.shape} is neither [L, E] nor [L, E*F]")
        base = bm[moe_layers]

    def prep(path, side):
        docs = read_docs(path, a.split)
        rng = np.random.default_rng(a.seed + (0 if side == "target" else 1))
        rng.shuffle(docs)
        ids, skipped = [], 0
        for r in docs:
            if len(ids) >= a.max_docs:
                break
            t = host.encode(r["text"])
            if len(t) < T:
                skipped += 1
                continue
            ids.append((t[:T], r.get("doc_id"), r.get("source") or "all"))
        return ids, skipped, len(docs)
    data = {}
    for side, p in (("target", a.target), ("baseline", a.baseline)):
        ids, sk, nd = prep(p, side)
        data[side] = ids
        log(f"{side}: {len(ids)} documents of {T} tokens ({sk} shorter skipped, {nd} in split {a.split})")
    tsrc = sorted({s for _, _, s in data["target"]})
    # target groups by (source, half): a language neuron must hold against the baseline in every target source (the
    # sources differ in register; review 2026-11-16), and in both halves
    groups = [f"target:{s}:{h}" for s in tsrc for h in (0, 1)] + ["baseline:all:0", "baseline:all:1"]
    S1 = {g: [mx.zeros((E * F,), mx.float32) for _ in range(Lm)] for g in groups}
    S2 = {g: [mx.zeros((E * F,), mx.float32) for _ in range(Lm)] for g in groups}
    R = {g: [mx.zeros((E,), mx.float32) for _ in range(Lm)] for g in groups}
    n = {g: 0 for g in groups}
    cur = {}

    def stats(l, kind, p):
        h = p.get("h")
        if h is None or l not in lidx:
            return
        inds, w = p["inds"], p["w"]
        B = inds.shape[0]
        k = inds.shape[-1]
        flat = (mx.arange(B, dtype=mx.int32)[:, None, None] * E + inds.astype(mx.int32)).reshape(-1)
        val = (mx.abs(h.astype(mx.float32)) * w.astype(mx.float32)[..., None]).reshape(-1, F)
        acc = mx.zeros((B * E, F), mx.float32).at[flat].add(val)
        m = acc.reshape(B, E * F) / T                      # per-document mean contribution
        g, i = cur["g"], lidx[l]
        S1[g][i] = S1[g][i] + m.sum(0)
        S2[g][i] = S2[g][i] + (m * m).sum(0)
        R[g][i] = R[g][i] + mx.zeros((B * E,), mx.float32).at[flat].add(1.0).reshape(B, E).sum(0) / (T * k)
        mx.eval(S1[g][i], S2[g][i], R[g][i])      # per layer: the capture buffers are freed (review: one lazy graph
                                                  # over 48 layers kept 5-10 GB of h / val alive at batch 16)

    host.ctl.reset()
    host.ctl.stats = stats
    host.ctl.stats_fast = not a.no_fast     # one MoE call per layer (the capture's rows), not capture + stock call
    t0, ntok, nb = time.time(), 0, 0
    rng = np.random.default_rng(a.seed + 7)
    for side in ("target", "baseline"):
        docs = data[side]
        half = rng.permutation(len(docs)) % 2
        for g in [x for x in groups if x.startswith(side + ":")]:
            _, src, hv = g.split(":")
            sel = [docs[j][0] for j in range(len(docs)) if half[j] == int(hv) and (side == "baseline" or docs[j][2] == src)]
            cur["g"] = g
            for b in range(0, len(sel), a.batch):
                chunk = sel[b:b + a.batch]
                x = mx.array(chunk, dtype=mx.int32)
                mx.eval(host.hidden(x), *S1[cur["g"]], *S2[cur["g"]], *R[cur["g"]])
                mx.clear_cache()
                n[cur["g"]] += len(chunk)
                ntok += len(chunk) * T
                nb += 1
                if nb % 20 == 1:
                    el = time.time() - t0
                    log(f"{cur['g']} batch {nb}: {ntok} tokens, {ntok / el:.0f} tok/s, peak {O.peak_gb()} GB")
                if a.bench and nb >= a.bench:
                    break
            if a.bench and nb >= a.bench:
                break
        if a.bench and nb >= a.bench:
            break
    host.ctl.reset()
    host.ctl.stats_fast = False
    wall = time.time() - t0
    tps = ntok / wall
    log(f"capture: {ntok} tokens in {wall:.0f} s = {tps:.0f} tok/s")
    rep = {"schema": "mlx2.amoeba-firing-map.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mlx2_source_revision": O.git_rev() if hasattr(O, "git_rev") else None, "args": vars(a),
           "unit": "expert * F + neuron", "E": E, "F": F, "moe_layers": moe_layers,
           "contribution": "|h| * routing weight (0 when not routed), mean over the document's first T tokens",
           "docs": n, "tokens": ntok, "corpus_sha256_12": {"target": O.sha12(a.target), "baseline": O.sha12(a.baseline)}
           if hasattr(O, "sha12") else None, "wall_s": round(wall, 1), "tokens_per_s": round(tps, 1),
           "mlx_peak_gb": O.peak_gb()}
    if a.bench:
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return
    s1 = {g: np.stack([np.array(v, dtype=np.float64) for v in S1[g]]) for g in groups}
    s2 = {g: np.stack([np.array(v, dtype=np.float64) for v in S2[g]]) for g in groups}
    rr = {g: np.stack([np.array(v, dtype=np.float64) for v in R[g]]) for g in groups}
    # the raw per-group sums first (review: any later analysis can then be redone on CPU without the GPU)
    np.savez(out / "sums.npz", groups=np.array(groups), n=np.array([n[g] for g in groups]),
             **{f"S1_{j}": s1[g].astype(np.float32) for j, g in enumerate(groups)},
             **{f"S2_{j}": s2[g].astype(np.float32) for j, g in enumerate(groups)},
             **{f"R_{j}": rr[g].astype(np.float32) for j, g in enumerate(groups)}, moe_layers=np.array(moe_layers))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    T_ALL = [g for g in groups if g.startswith("target:")]
    B_ALL = [g for g in groups if g.startswith("baseline:")]

    def moments(gs):
        N = sum(n[g] for g in gs)
        m = sum(s1[g] for g in gs) / N
        v = np.maximum(sum(s2[g] for g in gs) / N - m * m, 0.0) * N / max(1, N - 1)
        return N, m, v

    def test(tg, bg):
        n1, m1, v1 = moments(tg)
        n0, m0, v0 = moments(bg)
        se = np.sqrt(v1 / n1 + v0 / n0)
        sd = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / max(1, n1 + n0 - 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, (m1 - m0) / se, 0.0)
            d = np.where(sd > 0, (m1 - m0) / sd, 0.0)
        p = _erfc_p(t.ravel())
        q = bh(p).reshape(t.shape)
        return {"m1": m1, "m0": m0, "t": t, "d": d, "q": q, "target": (q <= a.q) & (d >= a.min_d),
                "baseline_up": (q <= a.q) & (d <= -a.min_d)}
    full = test(T_ALL, B_ALL)
    h0 = test([g for g in T_ALL if g.endswith(":0")], ["baseline:all:0"])
    h1 = test([g for g in T_ALL if g.endswith(":1")], ["baseline:all:1"])
    by_src = {s: test([g for g in T_ALL if g.split(":")[1] == s], B_ALL) for s in tsrc}
    tm = full["target"]
    inter = (h0["target"] & h1["target"]).sum()
    union = (h0["target"] | h1["target"]).sum()
    corr = float(np.corrcoef(h0["d"].ravel(), h1["d"].ravel())[0, 1])
    per_layer = tm.sum(1)
    ex = tm.reshape(Lm, E, F).sum(2)                   # target neurons per expert
    flat = np.sort(ex.ravel())[::-1]
    cum = np.cumsum(flat) / max(1, flat.sum())
    rep.update({
        "test": {"q": a.q, "min_d": a.min_d, "p": "two-sided normal on Welch t", "fdr": "Benjamini-Hochberg"},
        "target_neurons": int(tm.sum()), "target_share": float(tm.mean()),
        "baseline_up_neurons": int(full["baseline_up"].sum()),
        "per_layer_target": per_layer.tolist(),
        "split_half": {"target_h0": int(h0["target"].sum()), "target_h1": int(h1["target"].sum()),
                       "jaccard": float(inter / max(1, union)), "d_corr": corr,
                       "full_in_both_halves": float((tm & h0["target"] & h1["target"]).sum() / max(1, tm.sum()))},
        "by_target_source": {s: {"docs": sum(n[g] for g in T_ALL if g.split(":")[1] == s),
                                 "target_neurons": int(v["target"].sum())} for s, v in by_src.items()},
        "target_in_every_source": int(np.logical_and.reduce([v["target"] for v in by_src.values()]).sum()),
        "full_target_in_every_source": float((tm & np.logical_and.reduce([v["target"] for v in by_src.values()])).sum()
                                             / max(1, tm.sum())),
        "experts_holding_90pct": int(np.searchsorted(cum, 0.9) + 1) if flat.sum() else 0,
        "experts_with_any": int((ex > 0).sum()), "experts_total": int(Lm * E),
        "d_quantiles_target": np.quantile(full["d"][tm], [0.1, 0.5, 0.9]).tolist() if tm.any() else None,
        "routed_share_target_vs_baseline_corr": float(np.corrcoef(
            sum(rr[g] for g in T_ALL).ravel(), sum(rr[g] for g in B_ALL).ravel())[0, 1]),
    })
    if base is not None:
        rep["base"] = {"path": a.base, "target_in_base": int((tm & base).sum()),
                       "target_outside_base": int((tm & ~base).sum()), "base_neurons": int(base.sum())}
    np.savez_compressed(out / "map.npz", d=full["d"].astype(np.float32), t=full["t"].astype(np.float32),
                        q=full["q"].astype(np.float32), target=tm, baseline_up=full["baseline_up"],
                        mean_target=full["m1"].astype(np.float32), mean_baseline=full["m0"].astype(np.float32),
                        every_source=np.logical_and.reduce([v["target"] for v in by_src.values()]),
                        routed_target=(sum(rr[g] for g in T_ALL) / max(1, sum(n[g] for g in T_ALL))).astype(np.float32),
                        routed_baseline=(sum(rr[g] for g in B_ALL) / max(1, sum(n[g] for g in B_ALL))).astype(np.float32),
                        moe_layers=np.array(moe_layers))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("args", "per_layer_target")}, indent=1))


def _erfc_p(t):
    """Vectorised two-sided normal p (float64), accurate in the tail via scipy-free erfc."""
    x = np.abs(t) / math.sqrt(2)
    # Numerical Recipes erfcc (Chebyshev fit, relative error < 1.2e-7)
    z = x
    tt = 1.0 / (1.0 + 0.5 * z)
    y = tt * np.exp(-z * z - 1.26551223 + tt * (1.00002368 + tt * (0.37409196 + tt * (0.09678418 + tt * (
        -0.18628806 + tt * (0.27886807 + tt * (-1.13520398 + tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277)))))))))
    return np.minimum(1.0, y)


if __name__ == "__main__":
    main()
