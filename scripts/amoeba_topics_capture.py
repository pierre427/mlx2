#!/usr/bin/env python3
"""Capture per-text MLP neuron responses on the topic probe corpus (DESIGN: docs/research/amoeba-topics).

For each probe text (first --seq-len host tokens) and every MLP neuron: mean over positions >= 1 of
|act - mu|, where act is the pre-down_proj activation and mu the host's per-neuron mean activation
(--mean-npz, e.g. the 27B core run's mean.npz; else computed on the corpus's fit texts). Also the
full-model mean NLL of each text.

Writes OUT/: resp_L<l>.npy fp16 [S, N] per layer, nll_full.npy [S], items.json (id, topics, split,
n_tokens), mu.npy, meta.json. OUT is research-artifacts (GBs, third-party text derived): never git.

  27B (GPU, inside an owned window):
    amoeba_topics_capture.py --backend mlx27b --model ~/mlx-models/Qwen3.8-27B-BF16-q8-mlx2 \
       --mean-npz docs/research/amoeba-27b/core-696591553/mean.npz --corpus .../corpus.jsonl --out ...
  small models (CPU): --backend torch --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import amoeba_topics_lib as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "mlx27b", "carve"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mean-npz", default="")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--min-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N items")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="mlx27b on the CPU (slice smoke tests only)")
    ap.add_argument("--mem-policy", default="", help="amoeba27b_lib.admit policy (27B only)")
    ap.add_argument("--need-gb", type=float, default=40, help="27B: 40; torch small models: e.g. 6")
    ap.add_argument("--watchdog-gb", type=float, default=60)
    ap.add_argument("--mlx-memory-gb", type=float, default=56, help="mlx27b / carve: mx.set_memory_limit")
    ap.add_argument("--watchdog-min-avail-gb", type=float, default=15.0,
                    help="mlx27b / carve: abort when machine available memory falls below this (M3: ~3)")
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    log = T.make_log(out)
    rec = {"schema": "mlx2.amoeba-topic-capture.v1", "args": vars(args), "mlx2_source_revision": T.git_rev(),
           "corpus_sha256": T.sha_file(args.corpus)}
    if args.backend in ("mlx27b", "carve"):
        import amoeba27b_lib as A
        rec["memcheck"] = A.admit(args.need_gb, args.mem_policy or "strict", log)
        A.footprint_watchdog(args.watchdog_gb, log, min_avail_gb=args.watchdog_min_avail_gb)
        A.setup_mlx(args.mlx_memory_gb, 4)
        if args.cpu:
            import mlx.core as mx
            mx.set_default_device(mx.cpu)
    else:
        import amoeba_memcheck
        rec["memcheck"] = amoeba_memcheck.require(args.need_gb, args.mem_policy or "small", log=log)
        amoeba_memcheck.rss_watchdog(args.watchdog_gb, log=log)
    if args.backend == "carve":
        # any donor the carve host loads (dense neurons and/or routed experts, research/amoeba/carve)
        from research.amoeba.carve.host import TopicsBackend
        be = TopicsBackend(args.model, log=log)
        # every unit site must engage and every re-implemented router must reproduce its block on the
        # real weights before any response is read (research/amoeba/carve/units.py)
        import mlx.core as mx
        probe = [ids for _, ids in T.tokenize_items(be.encode, T.load_items(args.corpus)[:64], args.seq_len,
                                                    args.seq_len)][:4]        # full-length windows batch evenly
        par = be.h.check_site_parity(mx.array(probe))
        rec["site_parity"] = par
        log(f"site parity: ok={par['ok']} max_rel_l2={par['max_rel_l2']:.2e} not_engaged={par['not_engaged']}")
        if not par["ok"]:
            T.write_json(out / "meta.json", rec)
            raise SystemExit(f"site parity failed: {par['over_tol'] or par['not_engaged']}")
    else:
        be = T.make_backend(args.backend, args.model, log=log, threads=args.threads or None)
    rec["checkpoint"] = be.checkpoint
    rec["fingerprint"] = be.fingerprint          # full; joins across sessions use this, not the fp8
    items = T.load_items(args.corpus)
    if args.limit:
        items = items[:args.limit]
    toks = T.tokenize_items(be.encode, items, args.seq_len, args.min_len)
    log(f"{len(toks)}/{len(items)} items with >= {args.min_len} tokens")
    seqs = [ids for _, ids in toks]
    if args.mean_npz:
        mu = np.load(args.mean_npz)["mean"].astype(np.float32)
        rec["mu"] = {"source": args.mean_npz, "sha256_12": T.sha_file(args.mean_npz, 12)}
    else:
        fit = [s for (it, s) in toks if it["split"] == "fit"]
        mu = be.mean_act(fit, bs=args.bs)
        rec["mu"] = {"source": "computed on the corpus fit split", "n_texts": len(fit)}
    np.save(out / "mu.npy", mu)
    # ragged sites (dense first layers in an MoE model, layers without an MLP): each resp_L file is
    # written at its own layer's width (0 for a layer without a site); meta "neurons" is the widest
    # width and readers pad on load (amoeba_topics_lib.load_resp)
    widths = list(getattr(be, "widths", None) or [be.N] * be.L)
    S, L, N = len(seqs), be.L, (be.N if be.N is not None else max(widths))
    mm = [np.lib.format.open_memmap(out / f"resp_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, int(widths[l])))
          for l in range(L)]
    nll = np.zeros(S, dtype=np.float32)
    # bytes of the scored targets (tokens 2..T-1), so NLL can be compared per byte across tokenizers
    dec = getattr(be, "tok", None) or getattr(getattr(be, "host", None), "tok", None)
    n_bytes = np.array([len(dec.decode(s[2:]).encode("utf-8")) for s in seqs], np.int32)
    np.save(out / "n_bytes.npy", n_bytes)
    t0, done = time.time(), 0
    for g in T.batches_by_len(seqs, args.bs):
        ids = [seqs[j] for j in g]
        st = be.stats(ids, mu)
        for l in range(L):
            if st[l] is None:
                continue                      # no unit site in this layer: stays zero (padding)
            mm[l][g, :st[l].shape[-1]] = st[l].astype(np.float16)
        nll[g] = be.nll(ids)
        done += len(g)
        if done % 256 < len(g):
            log(f"  {done}/{S} ({time.time() - t0:.0f}s)")
    for m in mm:
        m.flush()
    np.save(out / "nll_full.npy", nll)
    T.write_json(out / "items.json", [{"id": it["id"], "topics": it["topics"], "split": it["split"], "n_tokens": len(s)}
                                      for it, s in toks])
    rec.update({"sequences": S, "layers": L, "neurons": N, "widths": [int(w) for w in widths],
                "unit_kinds": [getattr(s, "kind", None) for s in getattr(getattr(be, "h", None), "sites", [])] or None,
                "capture_s": round(time.time() - t0, 1),
                "unit": "per-text mean over positions >= 1 of |act - mu|, fp16, pre-down_proj activation"})
    if args.backend == "mlx27b":
        rec["mlx_peak_gb"] = be.A.peak_gb()
        rec["watch"] = dict(be.A.WATCH)
    elif args.backend == "carve":
        rec["mlx_peak_gb"] = A.peak_gb()
        rec["watch"] = dict(A.WATCH)
    T.write_json(out / "meta.json", rec)
    log(f"done: {S} texts in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
