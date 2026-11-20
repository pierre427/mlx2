#!/usr/bin/env python3
"""Next-token gating on Flash-Next: quality and kernel A/B in one owned GPU window (research, unqualified).

Pierre (2026-11-19): can step t+1's gates be decided at the end of step t?  CPU proxy study:
scripts/amoeba_nexttok_gate_cpu.py.  This window measures it on FN with the fn-boxes2 partition.

Gate probes (none existed for FN: the tile-gate kernel had only random probes and no box router was ever
trained; these are the minimal ones, fit on fit rows only, never held):
  A  same-step probe: per box, a logistic probe on the residual input of layer --probe-layer (v, default 3)
     at token t, target = the document's topic (box b's fit corpus = 1, English and the other boxes' fit
     docs = 0).  Same-step gating = A(h_v(t)); layers < v run with every tile open (they run before the
     gate exists: the "dense early layers").
  B  next-token probe: the same target at token t, read from step t-1's final (normed) hidden state.
  C  token prior: P(topic b | input token x_t) from fit counts (x_t is the token sampled at step t-1).
Next-token arms: lag = A on h_v(t-1); hidden = B; token = C; with hysteresis N and fail-open (p > 0.2).
Every next-token arm gates every layer (layers < v included).  Prefill rule: the first --prompt tokens use
the same-step set and start the hysteresis state as their union (age 0; the first decode step then ages it
like any other step).  Verify keeps layers < v dense, so its gated layers run the same-step set exactly
(same-step bits); the prediction only feeds prefetch, and the recompute rate is what a paged tile set would
pay (rows whose predicted set differs from same-step).

Part Q (quality, full FN, teacher forced, neuron-unit masks through research/amoeba/carve):
  held CE on decode positions per arm: full, core, oracle (the doc's own box + its pairs: the battery's
  served composition), same-step, next-token arms; dCE vs full (the +0.05 bar) and vs same-step; exact
  gate-set match rate vs same-step; missed opens; recompute rate; skipped box bytes.  Open loop: every
  arm's gates come from the full model's states (same-step's are exact: layers < v are ungated).
Part K (kernels, real FN layers --kernel-layers, packed with the real partition): per decode step, stock
  SwitchGLU vs same-step (gate kernel + block kernels) vs next-token (block kernels on the predicted set +
  the step-end probe and hysteresis update) vs next-token+verify with resident tiles (same-step gates at the
  gated layers, the prediction only feeds prefetch: its cost is the step-end op) vs next-token+recompute
  (the paging law: predicted pass + verify gate + recompute of disagreeing rows + merge), on the gate
  streams Part Q produced; mpp (NAX) and simd; bitwise recompute == same-step per step.
  ms per step for the layer set and per token extrapolated to 48 layers (MLP only, not full decode).

Usage (inside scripts/amoeba27b_window_q.sh): amoeba_nexttok_fn_window.py --out DIR [--swap-growth-gb 8
  --waiver-note "..."]; --dry-run prints the plan on the CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = Path.home() / "mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
ART = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/fn-boxes2-dd1ef1d7d"
BOXES = "fra,deu,cmn,jpn,sci.astronomy,life.cooking,ml.inference,hum.philosophy"
ENGLISH = str(Path.home() / "Desktop/amoeba-corpus/builds/lang-prose-eng-5cc2e1584/corpus.jsonl")
HYST = (0, 4, 16, 64)


def git_rev():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                          text=True).stdout.strip() or "unknown"


def read_rows(path, split):
    rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    if split == "fit" and not any(r.get("split") for r in rows):
        return rows                                    # a fit-only build carries no split field
    return [r for r in rows if r.get("split") == split]


def docs(host, path, split, n, T, seed):
    rows = read_rows(path, split)
    np.random.default_rng(seed).shuffle(rows)
    out = []
    for r in rows:
        t = host.encode(r["text"])
        if len(t) >= T:
            out.append(t[:T])
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Probes (pure numpy / mlx; fit rows only)
# ---------------------------------------------------------------------------


def fit_logistic(X, Y, steps=400, lr=0.05, l2=1e-3):
    """Multi-label logistic regression on standardized X (mlx, Adam); returns (W [B, d], b [B]) on raw X."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    mu = X.mean(0, dtype=np.float64).astype(np.float32)
    sd = (X.std(0, dtype=np.float64) + 1e-4).astype(np.float32)
    Xs = mx.array((X - mu) / sd)
    Yt = mx.array(Y.astype(np.float32))
    pos = np.clip(Y.mean(0), 1e-4, 1 - 1e-4)
    lin = nn.Linear(X.shape[1], Y.shape[1])
    lin.weight = mx.zeros_like(lin.weight)
    lin.bias = mx.array(np.log(pos / (1 - pos)).astype(np.float32))
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m):
        z = m(Xs)
        return nn.losses.binary_cross_entropy(z, Yt, with_logits=True, reduction="mean") + l2 * (m.weight ** 2).sum()
    step = nn.value_and_grad(lin, loss_fn)
    for _ in range(steps):
        loss, g = step(lin)
        opt.update(lin, g)
        mx.eval(lin.parameters(), loss)
    W = np.array(lin.weight) / sd[None]
    b = np.array(lin.bias) - (W * mu[None]).sum(1)
    return W.astype(np.float32), b.astype(np.float32), float(loss)


def token_prior(tokens, Y, alpha=1.0):
    marg = Y.mean(0)
    seen, inv = np.unique(tokens, return_inverse=True)
    sums = np.zeros((len(seen), Y.shape[1]))
    np.add.at(sums, inv, Y)
    cnt = np.bincount(inv, minlength=len(seen)).astype(np.float64)
    prob = (sums + alpha * marg) / (cnt[:, None] + alpha)
    table = dict(zip(seen.tolist(), range(len(seen))))
    return lambda toks: np.stack([prob[table[t]] if t in table else marg for t in toks])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def hysteresis(P, S, N, prompt):
    """P, S bool [T, B]: positions < prompt use S; the decode state starts at the prompt union (age 0)."""
    O = S.copy()
    age = np.where(S[:prompt].any(0), 0, 1 << 20)
    for t in range(prompt, len(P)):
        age = np.where(P[t], 0, np.minimum(age + 1, 1 << 20))
        O[t] = age <= N
    return O


def gate_arms(Hv, Hf, toks, A, B, C, prompt, fail_open_p, hyst=HYST):
    """Per doc: dict arm -> bool [T, boxes].  Hv/Hf [T, d] (full-model states), toks [T]."""
    S = (Hv @ A[0].T + A[1]) > 0
    lag = np.concatenate([S[:1], S[:-1]])
    pb = sigmoid(Hf @ B[0].T + B[1])
    hid = np.concatenate([pb[:1], pb[:-1]])            # B reads step t-1's final state for token t
    tp = C(list(toks))                                  # C reads token t, sampled at step t-1
    preds = {"lag": lag, "hidden": hid > 0.5, "hidden_failopen": hid > fail_open_p, "token": tp > 0.5,
             "token_failopen": tp > fail_open_p, "lag|token": lag | (tp > 0.5)}
    out = {"same": S}
    for k, P in preds.items():
        for n in hyst:
            out[f"{k}/N{n}"] = hysteresis(P, S, n, prompt)
    return out


def stream_metrics(S, O, dom, nb, total, names, prompt, switch=8):
    """Box-level (S, O lists of [T, B]); nb [B] box bytes (units) per box; total = all MLP units."""
    from amoeba_nexttok_metrics import metrics
    return metrics([s[:, None] for s in S], [o[:, None] for o in O], dom, nb[None].astype(np.float64),
                   float(total), names, prompt, switch)


# ---------------------------------------------------------------------------
# Part Q
# ---------------------------------------------------------------------------


def capture(host, seqs, v, bs=8):
    """Full-model states: layer v's residual input and the final (normed) hidden, float16 numpy [n, T, d]."""
    import mlx.core as mx
    Hv, Hf = [], []
    for i in range(0, len(seqs), bs):
        host.ctl.reset()
        recs, out = host.record(mx.array(seqs[i:i + bs]))
        hv = recs[v][0].astype(mx.float16)
        hf = (out[0] if isinstance(out, tuple) else out).astype(mx.float16)
        mx.eval(hv, hf)
        Hv.append(np.array(hv))
        Hf.append(np.array(hf))
        del recs
        mx.clear_cache()
    return np.concatenate(Hv), np.concatenate(Hf)


def device_masks(core, groups):
    """mx masks, built once: core [L, W] and one per gate group (float16 halves the resident bytes)."""
    import mlx.core as mx
    return mx.array(core.astype(np.float16)), {k: mx.array(m.astype(np.float16)) for k, m in groups.items()}


def masked_ce(host, seqs, masks, gates, prompt, dense_layers=(), bs=8):
    """Mean CE per doc over decode targets t >= prompt under core + per-token gated groups.

    masks = device_masks(...) or None (the full model); gates {group: [n, T] bool} (open bits per input
    position); ``dense_layers`` run whole (same-step gating's layers before the gate layer)."""
    import mlx.core as mx
    from research.amoeba.carve.carving import ce_rows
    out = []
    for i in range(0, len(seqs), bs):
        ids = mx.array(seqs[i:i + bs])
        c = host.ctl
        c.reset()
        if masks is not None:
            c.core = masks[0]
            c.dense_layers = tuple(dense_layers)
            c.boxes = [{"name": k, "mask": m, "token_gate": mx.array(gates[k][i:i + bs].astype(np.float16))[..., None]}
                       for k, m in masks[1].items()]
        T = ids.shape[1]
        lg = host.logits_at(ids, list(range(prompt - 1, T - 1)))
        ce = ce_rows(lg, ids[:, prompt:])
        mx.eval(ce)
        out.extend(np.array(ce).mean(1).tolist())
        c.reset()
        mx.clear_cache()
    return out


def group_gates(arm, groups, names):
    """Box gates [T, boxes] -> gate-group gates: a group (the chosen member boxes of a box or pair tile)
    opens when any of its boxes is open (pair tiles: the OR rule)."""
    return {k: arm[..., [names.index(b) for b in k]].any(-1) for k in groups}


def part_q(a, host, log, receipt):
    import mlx.core as mx
    pd = Path(a.partition)
    prep = json.loads((pd / "partition.json").read_text())
    tz = np.load(pd / "tiles.npz")
    layers = [int(x) for x in tz["layers"]]
    if layers != list(range(host.L)):
        raise SystemExit("partition does not cover every layer")
    names = [b for b in a.boxes.split(",") if b]
    meta = json.loads(Path(a.battery_meta).read_text())
    held_spec = {k: v["spec"].split("=", 1)[1].rsplit("@", 1)[0] for k, v in meta["held"].items()}
    T, P = a.tokens, a.prompt
    fit, held = {}, {}
    for b in names:
        fit[b] = docs(host, prep["boxes"][b]["target_corpus"], "fit", a.fit_docs, T, a.seed)
        held[b] = docs(host, held_spec[b].split("?")[0], "held", a.held_docs, T, a.seed + 1)
    fit["english"] = docs(host, a.english, "fit", a.fit_docs, T, a.seed)
    held["english"] = docs(host, a.english, "held", a.held_docs, T, a.seed + 1)
    for k in fit:
        if len(fit[k]) < 4 or len(held[k]) < 4:
            raise SystemExit(f"{k}: too few docs of {T} tokens (fit {len(fit[k])}, held {len(held[k])})")
    # switch streams: half a held doc of one topic, half of another (topic switch at T/2)
    rng = np.random.default_rng(a.seed + 2)
    topics = list(held)
    sw, swdom = [], []
    for i in range(a.switch_docs):
        x, y = rng.choice(len(topics), 2, replace=False)
        dx, dy = held[topics[x]][i % len(held[topics[x]])], held[topics[y]][(i + 1) % len(held[topics[y]])]
        sw.append(dx[:T // 2] + dy[:T - T // 2])
        swdom.append([topics[x]] * (T // 2) + [topics[y]] * (T - T // 2))
    log(f"docs: fit {sum(map(len, fit.values()))}, held {sum(map(len, held.values()))}, switch {len(sw)}")
    v = a.probe_layer
    t0 = time.time()
    fseqs = [s for k in fit for s in fit[k]]
    flab = np.array([[k == b for b in names] for k in fit for _ in fit[k]], bool)
    Hv, Hf = capture(host, fseqs, v)
    log(f"fit capture {Hv.shape} in {time.time() - t0:.0f}s")
    Xv = Hv[:, 1:].reshape(-1, Hv.shape[-1]).astype(np.float32)          # position 0 is a sink: skipped
    Yt = np.repeat(flab, T - 1, axis=0)
    Wa, ba, la = fit_logistic(Xv, Yt)
    Xf = Hf[:, :-1].reshape(-1, Hf.shape[-1]).astype(np.float32)          # step t-1 state -> token t label
    Wb, bb, lb_ = fit_logistic(Xf, np.repeat(flab, T - 1, axis=0))
    C = token_prior(np.array([s[1:] for s in fseqs]).reshape(-1), Yt.astype(np.float64))
    del Hv, Hf, Xv, Xf
    probes = {"A_same_step": {"input": f"layer {v} residual input, token t", "fit_loss": la},
              "B_next_token": {"input": "final normed hidden, token t-1", "fit_loss": lb_},
              "C_token_prior": {"input": "input token t (sampled at step t-1)", "smoothing": "add-1 to marginal"},
              "target": "document topic (fit corpus of the box = 1; English and other boxes' fit docs = 0)",
              "fit_rows": int(len(Yt)), "fit_docs": {k: len(s) for k, s in fit.items()}}
    np.savez(Path(a.out) / "probes.npz", A_w=Wa, A_b=ba, B_w=Wb, B_b=bb, boxes=np.array(names))
    # held states and gate arms
    hseqs = [s for k in held for s in held[k]] + sw
    hdom = [[k] * T for k in held for _ in held[k]] + swdom
    Hv, Hf = capture(host, hseqs, v)
    arms = [gate_arms(Hv[i].astype(np.float32), Hf[i].astype(np.float32), hseqs[i], (Wa, ba), (Wb, bb), C, P,
                      a.fail_open_p) for i in range(len(hseqs))]
    del Hv, Hf
    oracle = [np.array([[d == b for b in names] for d in dm], bool) for dm in hdom]
    # tiles touching a chosen box, merged by their chosen members (a pair with one chosen member opens with
    # that box; one with two opens with either): <= boxes + boxes*(boxes-1)/2 gate groups
    pairs = {k: tuple(k[5:].split("__")) for k in tz.files if k.startswith("pair:")}
    tiles = {f"box:{b}": (b,) for b in names}
    tiles.update({k: tuple(sorted(m for m in mem if m in names)) for k, mem in pairs.items()
                  if any(m in names for m in mem)})
    groups = {}
    for t, key in tiles.items():
        groups[key] = groups.get(key, np.zeros(tz["core"].shape, bool)) | tz[t].astype(bool)
    core = tz["core"].astype(bool)
    total = int(core.size)
    nb = np.array([int(tz[f"box:{b}"].sum()) for b in names])
    log(f"gate groups {len(groups)} from {len(tiles)} tiles; box units {dict(zip(names, nb.tolist()))}")
    res = {"probes": probes, "change_rate_same_step": {}, "arms": {}}
    from amoeba_nexttok_metrics import change_rate
    res["change_rate_same_step"] = change_rate([x["same"][:, None] for x in arms], P)
    for k in arms[0]:
        m = stream_metrics([x["same"] for x in arms], [x[k] for x in arms], hdom, nb, total, names, P)
        m["exact_set_match_rate"] = float(np.mean(np.concatenate(
            [(x[k][P:] == x["same"][P:]).all(1) for x in arms])))
        res["arms"][k] = m
    res["oracle_vs_same"] = stream_metrics([x["same"] for x in arms], oracle, hdom, nb, total, names, P)
    log("gate metrics: " + ", ".join(f"{k} missed {v_['missed_per_needed']:.3f} skip {v_['box_bytes_skipped']:.3f}"
                                     for k, v_ in res["arms"].items() if k.endswith(("N0", "N16")) or k == "same"))
    # quality
    qarms = {"full": None, "core": "core", "oracle": "oracle", "same": "same"}
    for k in a.quality_arms.split(","):
        qarms[k] = k
    ce = {}
    dm = device_masks(core, groups)
    for arm, key in qarms.items():
        if key is None:
            gates, masks = None, None
        else:
            masks = dm
            if key == "core":
                gates = {k: np.zeros((len(hseqs), T), bool) for k in groups}
            else:
                src = oracle if key == "oracle" else [x[key] for x in arms]
                gg = [group_gates(s_, groups, names) for s_ in src]
                gates = {k: np.stack([g[k] for g in gg]) for k in groups}
        ce[arm] = masked_ce(host, hseqs, masks, gates, P, dense_layers=range(v) if key == "same" else ())
        log(f"CE {arm}: {np.mean(ce[arm]):.4f} ({time.time() - t0:.0f}s)")
    del dm
    sets = {k: list(range(sum(len(held[j]) for j in topics[:i]), sum(len(held[j]) for j in topics[:i + 1])))
            for i, k in enumerate(topics)}
    sets["switch"] = list(range(len(hseqs) - len(sw), len(hseqs)))
    res["held_ce"] = {arm: {g: float(np.mean([v_[i] for i in idx])) for g, idx in sets.items()}
                      for arm, v_ in ce.items()}
    for arm in ce:
        res["held_ce"][arm]["dce_vs_full"] = {g: res["held_ce"][arm][g] - res["held_ce"]["full"][g] for g in sets}
        res["held_ce"][arm]["dce_vs_same"] = {g: res["held_ce"][arm][g] - res["held_ce"]["same"][g] for g in sets}
        res["held_ce"][arm]["passes_0.05_vs_full_all_groups"] = all(
            x <= a.target for x in res["held_ce"][arm]["dce_vs_full"].values())
    res["verify_recompute"] = {k: {"rows_recomputed_exact": m["rows_any_diff"],
                                   "rows_recomputed_missed_only": m["rows_any_miss"]}
                               for k, m in res["arms"].items()}
    res["held_docs_per_set"] = {g: len(i) for g, i in sets.items()}
    res["gate_groups"] = [list(k) for k in groups]
    receipt["part_q"] = res
    streams = {"names": names,
               "same": [x["same"] for x in arms[:a.kernel_streams]],
               **{k: [x[k] for x in arms[:a.kernel_streams]] for k in a.kernel_arms.split(",")}}
    return streams, (Wa, ba), (Wb, bb)


# ---------------------------------------------------------------------------
# Part K
# ---------------------------------------------------------------------------


def part_k(a, log, receipt, streams, A, B):
    import mlx.core as mx
    import amoeba_blocks_bench as BB
    from mlx2.runtime.amoeba_blocks import layout as L
    from mlx2.runtime.amoeba_blocks import nexttok as NT
    from mlx2.runtime.amoeba_blocks import ops as O
    from mlx2.runtime.lane import matmul as lane
    layers = [int(x) for x in a.kernel_layers.split(",")]
    tnames, owners, _, _ = BB.fn_real_partition(layers)
    graph = L.graph_from_partition_names(tnames)
    weights = {l: BB.load_layer("fn", l) for l in layers}
    g0 = weights[layers[0]]["gate_proj"]
    E, d = g0.experts, g0.k
    gs = weights[layers[0]]["down_proj"].group_size
    lbs = {l: L.plan_layer(owners[l], tiles=len(tnames), group_size=gs, chunk=32) for l in layers}
    blocks = {l: O.DeviceBlocks.from_layout(lbs[l]) for l in layers}
    packed = {l: (O.pack_qweight(weights[l]["gate_proj"], lbs[l]), O.pack_qweight(weights[l]["up_proj"], lbs[l]))
              for l in layers}
    stock = {l: BB.stock_switch(weights[l]) for l in layers}
    names = streams["names"]
    gi = [graph.boxes.index(b) for b in names]

    def probe_tables(W, b):
        nb = len(graph.boxes)
        pw = np.zeros((nb, d), np.float32)
        pb = np.full(nb, -1e4, np.float32)             # boxes outside the study: always closed
        pw[gi], pb[gi] = W, b
        return O.GateTables.from_graph(graph, pw, pb)
    tab_a, tab_b = probe_tables(*A), probe_tables(*B)

    def topen_rows(box_bits):                          # [T, chosen boxes] -> uint8 [T, tiles]
        full = np.zeros((len(box_bits), len(graph.boxes)), bool)
        full[:, gi] = box_bits
        return graph.tile_open(full)
    steps = a.kernel_steps
    seqs = streams["same"]
    rows_s = np.concatenate([topen_rows(s[a.prompt:a.prompt + steps]) for s in seqs])
    backends = [x for x in a.backends.split(",") if x != "mpp" or lane._tensor_units()]
    rng = np.random.default_rng(a.seed)
    res = {"layers": layers, "tiles": len(tnames), "backends_dropped": sorted(set(a.backends.split(",")) - set(backends)),
           "steps": int(len(rows_s)), "inputs": "random bf16 x and random top-10 routes per step (timing); gate sets "
                                                "from Part Q's held streams", "cells": []}
    for Bsz in [int(x) for x in a.kernel_batch.split(",")]:
        n = len(rows_s) // Bsz * Bsz
        xs = mx.array(rng.standard_normal((n, d)).astype(np.float32)).astype(mx.bfloat16)
        hs = xs
        inds = mx.array(np.stack([rng.choice(E, 10, replace=False) for _ in range(n)]).astype(np.int32))
        mx.eval(xs, inds)

        def step_rows(i):
            return slice(i * Bsz, (i + 1) * Bsz)
        nsteps = n // Bsz

        def run_base():
            outs = []
            for i in range(nsteps):
                sl = step_rows(i)
                outs.append([stock[l](xs[sl], inds[sl]) for l in layers])
            return outs

        def run_blocks(be, top, sl):
            return [O.block_switch_glu(xs[sl], inds[sl], *packed[l], weights[l]["down_proj"], blocks[l], top,
                                       backend_name=be)[0] for l in layers]
        for be in backends:
            S_dev = mx.array(rows_s[:n])
            cells = {}

            def same():
                """same-step: the gate kernel at the gate layer (its decisions come from Part Q), then the layers"""
                outs = []
                for i in range(nsteps):
                    sl = step_rows(i)
                    outs.append([O.tile_gates(hs[sl], tab_a)[0]] + run_blocks(be, S_dev[sl], sl))
                return outs
            cells["same_step"] = same
            for arm in a.kernel_arms.split(","):
                rows_p = np.concatenate([topen_rows(s[a.prompt:a.prompt + steps]) for s in streams[arm]])[:n]
                P_dev = mx.array(rows_p)

                def nxt(P_dev=P_dev):
                    """next-token: the predicted set gates every layer; the step-end probe + hysteresis update"""
                    gate = NT.NextTokenGate(tab_b, Bsz, NT.NextTokenConfig(hysteresis=16))
                    outs = []
                    for i in range(nsteps):
                        sl = step_rows(i)
                        ys = run_blocks(be, P_dev[sl], sl)
                        outs.append(ys + [gate.advance(h_last=hs[sl]), gate.age])
                    return outs

                def ver(P_dev=P_dev):
                    """next-token + verify, resident tiles: same-step gates at the gated layers (exact), the
                    predicted set only for prefetch (the step-end op)"""
                    gate = NT.NextTokenGate(tab_b, Bsz, NT.NextTokenConfig(hysteresis=16))
                    outs = []
                    for i in range(nsteps):
                        sl = step_rows(i)
                        outs.append([O.tile_gates(hs[sl], tab_a)[0]] + run_blocks(be, S_dev[sl], sl)
                                    + [gate.advance(h_last=hs[sl]), gate.age])
                    return outs

                def rec(P_dev=P_dev):
                    """next-token + recompute (the paging law): run the predicted (resident) set, recompute the rows
                    whose same-step set differs, merge (NT.verified)"""
                    outs = []
                    for i in range(nsteps):
                        sl = step_rows(i)
                        ys = [O.tile_gates(hs[sl], tab_a)[0]]
                        for l in layers:
                            def mlp(x, t, l=l, sl=sl):
                                return O.block_switch_glu(x, inds[sl], *packed[l], weights[l]["down_proj"], blocks[l],
                                                          t, backend_name=be)[0]
                            ys.append(NT.verified(mlp, xs[sl], P_dev[sl], S_dev[sl])[0])
                        outs.append(ys)
                    return outs
                cells[f"next:{arm}"] = nxt
                cells[f"verify:{arm}"] = ver
                cells[f"recompute:{arm}"] = rec
                ref = same()
                got = rec()
                nx = nxt()
                mx.eval(ref, got, nx)
                exact = all(mx.array_equal(p, q).item() for r_, g_ in zip(ref, got) for p, q in zip(r_[1:], g_[1:]))
                row_eq = float(np.mean([mx.array_equal(p, q).item() for r_, g_ in zip(ref, nx)
                                        for p, q in zip(r_[1:], g_[:len(layers)])]))
                res["cells"].append({"arm": f"recompute:{arm}", "backend": be, "batch": Bsz,
                                     "recompute_equals_same_step_bitwise": bool(exact),
                                     "next_equals_same_step_layer_outputs": row_eq,
                                     "recompute_rows": float((rows_p != rows_s[:n]).any(1).mean()),
                                     "missed_open_rows": float((rows_s[:n].astype(bool) & ~rows_p.astype(bool)).any(1).mean())})
                log(f"K {be} B={Bsz} {arm}: recompute==same {exact}, next==same {row_eq:.3f}")
            for name, fn in [("stock_base", run_base)] + list(cells.items()):
                r = BB.timeit(fn, a.reps, a.warmup)
                ms = r["median_ms"] / nsteps
                res["cells"].append({"arm": name, "backend": "stock" if name == "stock_base" else be, "batch": Bsz,
                                     "ms_per_step_layer_set": ms, "ms_per_token_48_layers_mlp": ms * 48 / len(layers) / Bsz,
                                     "tokens_per_s_mlp_48_layers": Bsz / (ms * 48 / len(layers) / 1e3), **r})
                log(f"K {name} {be} B={Bsz}: {ms:.3f} ms/step ({len(layers)} layers)")
        gt = BB.timeit(lambda: [O.tile_gates(hs[step_rows(i)], tab_b)[0] for i in range(nsteps)], a.reps, a.warmup)
        res["cells"].append({"arm": "gate_kernel_alone", "batch": Bsz, "ms_per_step": gt["median_ms"] / nsteps})
    receipt["part_k"] = res


# ---------------------------------------------------------------------------


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--partition", default=str(ART / "partition"))
    ap.add_argument("--boxes", default=BOXES)
    ap.add_argument("--battery-meta", default=str(ART / "battery" / "battery.meta.json"),
                    help="the box battery's meta (its held set specs are the held docs here)")
    ap.add_argument("--english", default=ENGLISH)
    ap.add_argument("--fit-docs", type=int, default=24)
    ap.add_argument("--held-docs", type=int, default=16)
    ap.add_argument("--switch-docs", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--probe-layer", type=int, default=3)
    ap.add_argument("--fail-open-p", type=float, default=0.2)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--quality-arms", default="lag/N0,lag/N4,lag/N16,lag/N64,hidden/N0,hidden/N16,"
                                              "hidden_failopen/N16,token/N16,lag|token/N16")
    ap.add_argument("--kernel-arms", default="lag/N16,hidden/N16")
    ap.add_argument("--kernel-layers", default="3,23,43")
    ap.add_argument("--kernel-batch", default="1,4")
    ap.add_argument("--kernel-steps", type=int, default=64)
    ap.add_argument("--kernel-streams", type=int, default=4)
    ap.add_argument("--backends", default="mpp,simd")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="q,k")
    ap.add_argument("--memory-gb", type=float, default=80)
    ap.add_argument("--watchdog-gb", type=float, default=88)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto", type=_wired_arg,
                    help="wire the FN weights after load (research/amoeba/carve/ops.py wire_weights, as the box "
                         "battery): auto = MLX active after load + 10 GiB, GiB, or 0 = off; capped at --memory-gb")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def _wired_arg(spec):
    """--wired-gb: a bad value fails at parse time, before the model loads (ops.wired_arg, imported lazily)."""
    from research.amoeba.carve.ops import wired_arg
    return wired_arg(spec)


def main():
    a = parser().parse_args()
    if a.dry_run:
        print(json.dumps({"plan": vars(a), "estimate_min": {"load": 3, "part_q": 10, "part_k": 10, "cap": 45}},
                         indent=1))
        return 0
    if a.swap_growth_gb > 2 and not a.waiver_note:
        sys.exit("--swap-growth-gb above 2 needs --waiver-note (who approved it)")
    if not a.out:
        sys.exit("--out is required")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    from research.amoeba.carve import ops as CO
    CO.prepare_wiring(a.wired_gb)                      # before any Metal call (MLX reads the refresh env then)
    from research.amoeba.carve.host import Host
    CO.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb, receipt=out / "watchdog-stop.json",
                          watcher=True)                # memwatch + its SIGKILL watcher (2026-11-19 FN memory stops)
    CO.setup_mlx(a.memory_gb, 2)
    import mlx.core as mx
    receipt = {"schema": "mlx2.amoeba-nexttok-fn-window.v1", "mlx2_revision": git_rev(), "mlx": mx.__version__,
               "device": mx.device_info().get("device_name"), "args": vars(a),
               "status": "unqualified research A/B; no route selects next-token gating",
               "watchdog": {"swap_growth_gb": a.swap_growth_gb, "waiver_note": a.waiver_note or None}}
    t0 = time.time()
    try:
        streams = A = B = None
        if "q" in a.parts:
            host = Host(a.model, log=log, moe_units="neuron")
            prep = json.loads((Path(a.partition) / "partition.json").read_text())
            if host.fingerprint != prep["fingerprint"]:
                raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
            tables = getattr(getattr(host, "adapter", None), "_tables", None)
            if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                    Path(a.model).expanduser().resolve() / "ple_rows.bin":
                raise SystemExit("PLE offload is not active (FN rule: assert it)")
            receipt["wired"] = CO.wire_weights(a.wired_gb, log, memory_gb=a.memory_gb)   # Host evaluated the weights
            receipt["checkpoint"] = host.checkpoint
            receipt["partition"] = {"path": a.partition, "checkpoint": prep["checkpoint"], "rule": prep["rule"]}
            streams, A, B = part_q(a, host, log, receipt)
            del host
            mx.clear_cache()
        if "k" in a.parts and streams is not None:
            try:
                part_k(a, log, receipt, streams, A, B)
            except Exception as e:                    # keep Part Q's results; record the failure
                receipt["part_k_error"] = f"{type(e).__name__}: {e}"
                log(f"part K failed: {receipt['part_k_error']}")
                raise
    finally:
        receipt["seconds"] = round(time.time() - t0, 1)
        receipt["watch"] = {k: v for k, v in CO.WATCH.items() if k != "wired"}
        tmp = out / "receipt.json.tmp"
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out / "receipt.json")
        log(f"done in {receipt['seconds']}s -> {out / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
