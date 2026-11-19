#!/usr/bin/env python3
"""Belady-with-finite-lookahead rows for the DS4.1 Source expert-cache sim (CPU only, no GPU).

Recon 2026-11-13 probe 3. Prior art: arXiv 2611.15842 (expert eviction for MoE offloading, timed copies
on OLMoE-1B-7B): Belady copies 2.1x fewer experts than LRU, a 16-token demand lookahead recovers 96% of
that gain, counting past use 27%.

Replays the 2026-11-08 routing capture through ``cache_sim.simulate`` for LRU, lifetime LFU, full Belady
and ``belady_w<W>`` (``cache_sim.simulate_lookahead``) and reports expert copies (count and bytes) and
the share of the LRU -> Belady gain each lookahead window recovers.

usage: belady_lookahead_rows.py CAPTURE_DIR --out rows.json [--sizes ...] [--windows ...]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cache_sim as S  # noqa: E402

CHECKPOINT = ("DeepSeek-V4.1-Flash-1017-Source@"
              "c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb")


def load_stream(capture_dir, cache_path):
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        acc = list(zip(z["layer"].tolist(), z["expert"].tolist(), z["step"].tolist()))
        decode_steps = set(z["decode_steps"].tolist())
        return acc, decode_steps
    acc, meta = S.access_stream(capture_dir)
    decode_steps = {m[0] for m in meta if m[2] == 1}
    if cache_path:
        a = np.asarray(acc, np.int64)
        tmp = cache_path + ".tmp.npz"
        np.savez(tmp, layer=a[:, 0], expert=a[:, 1], step=a[:, 2],
                 decode_steps=np.asarray(sorted(decode_steps), np.int64))
        os.replace(tmp, cache_path)
    return acc, decode_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream-cache", help="npz cache of the flat access stream (scratch)")
    ap.add_argument("--sizes", default="2000,3989,4787,7200")
    ap.add_argument("--windows", default="0,1,4,8,16")
    ap.add_argument("--layer-windows", default="",
                    help="belady_l<D>: lookahead of D layer calls (1 = perfect next-layer predictor)")
    ap.add_argument("--baselines", default="lru,belady",
                    help="lfu is O(capacity) per miss (hours here); its rows come from cache-design/sim.json")
    ap.add_argument("--reference-sim", help="earlier cache_sim.py output to copy LFU rows from")
    a = ap.parse_args()

    t0 = time.time()
    acc, decode_steps = load_stream(a.capture_dir, a.stream_cache)
    is_decode = np.fromiter((x[2] in decode_steps for x in acc), bool, len(acc))
    n_decode = len(decode_steps)
    print(f"stream: {len(acc)} accesses, {n_decode} decode steps ({time.time()-t0:.0f}s)", flush=True)

    def ints(text):
        return [int(v) for v in text.split(",") if v.strip()]

    policies = (a.baselines.split(",") + [f"belady_w{w}" for w in ints(a.windows)]
                + [f"belady_l{d}" for d in ints(a.layer_windows)])
    reference = json.load(open(a.reference_sim)) if a.reference_sim else None
    rows = {}
    for size in map(int, a.sizes.split(",")):
        cell = {}
        for pol in policies:
            t1 = time.time()
            miss = S.simulate(acc, size, pol)
            total = int(miss.sum())
            dec = int((miss & is_decode).sum())
            cell[pol] = {
                "copies_total": total,
                "copies_decode": dec,
                "misses_per_decode_token": round(dec / n_decode, 2),
                "bytes_total": total * S.EXPERT_BYTES,
                "bytes_decode_per_token": round(dec * S.EXPERT_BYTES / n_decode),
                "sim_s": round(time.time() - t1, 1),
            }
            if pol.startswith(("belady_w", "belady_l")):
                cell[pol]["lru_scan_steps"] = int(S.simulate_lookahead.last_scans)
            print(size, pol, cell[pol]["misses_per_decode_token"], total, flush=True)
        if reference and "lfu" not in cell and f"lfu@{size}" in reference.get("ram", {}):
            ref = reference["ram"]
            cell["lfu_reference"] = {
                "misses_per_decode_token": ref[f"lfu@{size}"]["misses_per_decode_token"],
                "copies_decode": round(ref[f"lfu@{size}"]["misses_per_decode_token"] * n_decode),
                "source": os.path.abspath(a.reference_sim),
                "check_lru_matches": ref[f"lru@{size}"]["misses_per_decode_token"],
                "check_belady_matches": ref[f"belady@{size}"]["misses_per_decode_token"],
            }
        lru, bel = cell["lru"], cell["belady"]
        for pol, r in cell.items():
            if pol == "lfu_reference":
                gain = lru["copies_decode"] - bel["copies_decode"]
                r["recovered_copies_decode"] = round((lru["copies_decode"] - r["copies_decode"]) / gain, 3)
                continue
            for kind in ("copies_decode", "copies_total"):
                gain = lru[kind] - bel[kind]
                r[f"recovered_{kind}"] = (round((lru[kind] - r[kind]) / gain, 3) if gain else None)
            r["copies_vs_belady_decode"] = round(r["copies_decode"] / max(1, bel["copies_decode"]), 3)
        cell["_meta"] = {"capacity_experts": size, "gb": round(size * S.EXPERT_BYTES / 1e9, 1),
                         "lru_over_belady_decode": round(lru["copies_decode"] / max(1, bel["copies_decode"]), 2)}
        rows[str(size)] = cell

    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", HERE], capture_output=True,
                                text=True, cwd=HERE).stdout.strip())
    out = {
        "probe": "ds41_belady_lookahead_rows",
        "checkpoint": CHECKPOINT,
        "capture": os.path.abspath(a.capture_dir),
        "capture_mlx2_commit": open(os.path.join(a.capture_dir, "mlx2_commit.txt")).read().strip(),
        "source_revision": rev + (" (dirty research scripts)" if dirty else ""),
        "host": platform.node(), "cpu_only": True,
        "accesses": len(acc), "decode_steps": n_decode, "expert_bytes": S.EXPERT_BYTES,
        "layer_window_definition": ("belady_lD knows the accesses of the next D layer calls (one call = "
                                    "one (step, layer) capture record) in serving order; same victim rule."),
        "window_definition": ("belady_wW knows every access to the end of step s+W (s = current step; "
                              "decode step = one token, prefill call = one step); W=0 is the rest of "
                              "the current forward pass. Victim: LRU among keys with no known use, else "
                              "farthest known next use."),
        "recovered_definition": "(LRU - policy) / (LRU - Belady) copies",
        "rows": rows,
    }
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, a.out)


if __name__ == "__main__":
    main()
