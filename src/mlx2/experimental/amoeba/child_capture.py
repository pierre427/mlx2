"""Map step 1 capture (child process): per-probe unit responses in the topic-capture format.

Writes what scripts/amoeba_map_cluster.py and scripts/amoeba_topics_label.py read (the format of
scripts/amoeba_topics_capture.py): OUT/resp_L<l>.npy fp16 [S, width_l], items.json, mu.npy,
nll_full.npy, n_bytes.npy, meta.json. Unit signal (mapping-workflow s5.1.1): per text, the PEAK over
positions >= 1 of |act - mu| (dense neurons, mu = mean activation on the fit texts) or of routed weight x
||expert row|| (expert units), read through the carve host's unit-site hooks, so any donor the carve
host loads works. MoE layers are mapped in neuron units (e*F+f) by default, as the carve's units (--moe-units). The guard before any response is read is the carve host's own ones-mask parity.

    python -m mlx2.experimental.amoeba.child_capture --model M --corpus C --out DIR [--cpu] ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .progress import child_event


def site_signal(h, mu):
    """The stats hook and its per-layer outputs: per text, each unit's PEAK over positions >= 1
    (mapping-workflow s5.1.1). Returns (hook, out, extra) where after a forward out[l] is [B, width_l] and, for
    dense layers with mu None, extra[l] = (max act, min act, sum act) over positions >= 1, so |act - mu| peaks
    and mu itself can be finished after the layer (max(max - mu, mu - min) is the exact peak).
      dense neuron            |act - mu| (act = the down_proj input)
      MoE neuron unit e*F+f   |w * h| over the routes that chose expert e (h = the expert's intermediate activation,
                              w its routed weight): mapping-workflow s3, MoE units are neurons inside experts
      MoE expert unit e       w * ||expert output row|| over the routes that chose e"""
    import mlx.core as mx
    out, extra = [None] * h.L, [None] * h.L

    def st(l, kind, p):
        if kind == "dense":
            x = p["act"][:, 1:].astype(mx.float32)
            if mu is None:                   # rolling capture: finish |act - mu| once mu is known
                extra[l] = (x.max(1), x.min(1), x.sum(1))
                return
            x = x - mx.array(np.asarray(mu[l, :x.shape[-1]], dtype=np.float32))
            out[l] = mx.abs(x).max(1)
            return
        site = h.sites[l]
        inds, w = p["inds"][:, 1:], p["w"][:, 1:].astype(mx.float32)          # B, T-1, k
        B = inds.shape[0]
        if site.units == "neuron":
            if p.get("h") is None:
                raise SystemExit(f"layer {l}: the neuron site gave no intermediate activations")
            F = int(p["F"])
            n = site.n
            act = mx.abs(p["h"][:, 1:].astype(mx.float32) * w[..., None])      # B, T-1, k, F
            idx = inds[..., None] * F + mx.arange(F)                           # B, T-1, k, F
            idx = idx + (mx.arange(B) * n)[:, None, None, None]
            flat = mx.zeros((B * n,), mx.float32).at[idx.reshape(-1)].maximum(act.reshape(-1))
            out[l] = flat.reshape(B, n)
            return
        ye = mx.linalg.norm(p["rows"][:, 1:].astype(mx.float32), axis=-1) * w   # B, T-1, k
        n = h.widths[l]
        idx = inds + (mx.arange(B) * n)[:, None, None]
        flat = mx.zeros((B * n,), mx.float32).at[idx.reshape(-1)].maximum(ye.reshape(-1))
        out[l] = flat.reshape(B, n)
    return st, out, extra


def peaks(h, ids, mu):
    """One ordinary forward with the hook: [per layer: [B, width] float32 peaks or None]."""
    import mlx.core as mx
    st, out, _ = site_signal(h, mu)
    h.ctl.reset()
    h.ctl.stats = st
    mx.eval(h.hidden(mx.array(ids)), *[o for o in out if o is not None])
    h.ctl.reset()
    return [np.array(o) if o is not None else None for o in out]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--moe-units", default="neuron", choices=["neuron", "expert"],
                    help="MoE map units: neurons inside routed experts (mapping-workflow s3), or whole experts")
    a = ap.parse_args(argv)
    import amoeba_topics_lib as T
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import TopicsBackend
    out = T.fresh_dir(a.out)
    log = T.make_log(out)
    O.footprint_watchdog(a.watchdog_gb, log)
    O.setup_mlx(a.memory_gb, 4)
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    rec = {"schema": "mlx2.amoeba-topic-capture.v1", "args": vars(a), "mlx2_source_revision": T.git_rev(),
           "corpus_sha256": T.sha_file(a.corpus), "writer": "mlx2.experimental.amoeba.child_capture"}
    from research.amoeba.carve.host import Host
    be = TopicsBackend(a.model, log=log, host=Host(a.model, log=log, moe_units=a.moe_units))
    h = be.h
    items = T.load_items(a.corpus)
    toks = T.tokenize_items(be.encode, items, a.seq_len, a.min_len)
    log(f"{len(toks)}/{len(items)} items with >= {a.min_len} tokens")
    seqs = [ids for _, ids in toks]
    par = h.check_ones_parity(mx.array([s for s in seqs if len(s) == a.seq_len][:2] or [seqs[0]]))
    par["ok"] = par["top1_agree"] == 1.0 and par["kl_max"] <= 1e-3     # s7: top-1 100%, KL ~0
    rec["ones_parity"] = par
    if not par["ok"]:
        T.write_json(out / "meta.failed.json", rec)         # never meta.json: a retry must redo the capture
        raise SystemExit(f"ones-mask parity failed: {par}")
    widths = list(h.widths)
    L, N = h.L, max(widths)
    # mu: mean activation per dense neuron over positions >= 1 of the fit texts (expert units take no mu)
    fit = [s for (it, s) in toks if it["split"] == "fit"]
    mu = np.zeros((L, N), np.float32)
    cnt = np.zeros(L, np.float64)
    acc = [None] * L

    def st(l, kind, p):
        if kind == "dense":
            v = p["act"][:, 1:].astype(mx.float32).sum((0, 1))
            acc[l] = v if acc[l] is None else acc[l] + v
            cnt[l] += p["act"].shape[0] * (p["act"].shape[1] - 1)
    t0 = time.time()
    for g in T.batches_by_len(fit, a.bs):
        h.ctl.reset()
        h.ctl.stats = st
        mx.eval(h.hidden(mx.array([fit[j] for j in g])), *[x for x in acc if x is not None])
    h.ctl.reset()
    for l in range(L):
        if acc[l] is not None and cnt[l]:
            mu[l, :widths[l]] = np.array(acc[l]) / cnt[l]
    np.save(out / "mu.npy", mu)
    rec["mu"] = {"source": "computed on the corpus fit split (positions >= 1)", "n_texts": len(fit),
                 "seconds": round(time.time() - t0, 1)}
    S = len(seqs)
    mm = [np.lib.format.open_memmap(out / f"resp_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, int(widths[l])))
          for l in range(L)]
    nll = np.zeros(S, np.float32)
    n_bytes = np.array([len(h.decode(s[2:]).encode("utf-8")) for s in seqs], np.int32)
    np.save(out / "n_bytes.npy", n_bytes)
    child_event("steps", steps=S)
    t0, done = time.time(), 0
    for g in T.batches_by_len(seqs, a.bs):
        ids = [seqs[j] for j in g]
        r = peaks(h, ids, mu)
        for l in range(L):
            if r[l] is not None:
                mm[l][g, :r[l].shape[-1]] = r[l].astype(np.float16)
        ce = be.nll(ids)
        nll[g] = ce.mean(-1) if ce.ndim == 2 else ce
        done += len(g)
        if done % 128 < len(g) or done == S:
            el = time.time() - t0
            child_event("step", step=done - 1, s_per_step=round(el / done, 3), peak_gb=O.peak_gb())
            log(f"  {done}/{S} ({el:.0f}s)")
    for m in mm:
        m.flush()
    np.save(out / "nll_full.npy", nll)
    T.write_json(out / "items.json", [{"id": it["id"], "topics": it["topics"], "split": it["split"],
                                       "n_tokens": len(s)} for it, s in toks])
    rec.update({"checkpoint": h.checkpoint, "fingerprint": h.fingerprint, "sequences": S, "layers": L, "neurons": N,
                "widths": [int(w) for w in widths], "unit_kinds": [s.kind for s in h.sites],
                "moe_units": h.moe_units, "capture_s": round(time.time() - t0, 1), "mlx_peak_gb": O.peak_gb(),
                "watch": dict(O.WATCH),
                "unit": "per-text peak over positions >= 1 of |act - mu| (dense), of routed weight x ||row|| (experts), fp16",
                "window": "the first seq_len tokens of each probe text; probe texts are windows cut at random "
                          "positions when the probe corpus was built (mapping-workflow s4 rule 1)"})
    T.write_json(out / "meta.json", rec)
    log(f"done: {S} texts")


if __name__ == "__main__":
    main()
