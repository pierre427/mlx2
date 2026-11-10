"""Does a decode-step KV append copy the whole buffer? (GLM-5.3 MLA cache shapes)

Synthetic, no model: 11 layers x (latent (1,1,T,512) fp32 + raw indexer
(1,1,T,256) fp32) KVCaches prefilled to T, then ``--steps`` one-row appends,
each step reading the appended buffer the way gathered decode does (a gather of
2051 rows), serial (eval per step) and pipelined (async_eval of step N+1 before
waiting on N). Per-step time and per-step peak-above-active bytes at each T: an
in-place append is flat in T and adds ~0 peak; a full-buffer copy grows with T
and adds ~one buffer of peak. ``--kv-append rotate_v1 --ring K`` runs the same
steps on glm53_kv_append ring planes (K full buffers; K=1 is the reference
law); pipelined time should be flat in T for K >= 2. Research only.
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--kv-append", choices=("reference", "rotate_v1"), default="reference")
    parser.add_argument("--ring", type=int, default=3, help="rotate_v1 buffers per plane")
    parser.add_argument("--payload", type=int, default=0,
                        help="4096^2 fp32 gemvs before each layer's appends, independent of the cache (a model-sized step)")
    parser.add_argument("--payload-ops", type=int, default=0,
                        help="tiny elementwise kernels before each layer's appends (a model-like kernel count)")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    if args.kv_append == "rotate_v1":
        import mlx2.runtime.models.glm53_kv_append as kv_module

        kv_module.RING_BUFFERS = args.ring

    from mlx2.runtime.models import glm53_kv_append
    from mlx2.runtime.models.cache import KVCache

    layers, rows = 11, 2051
    record = {"schema": "mlx2.glm53-kv-update-bench.v1", "source_revision": args.source_revision,
              "mlx": mx.__version__, "kv_append": args.kv_append,
              "ring": args.ring if args.kv_append == "rotate_v1" else None, "payload": args.payload,
              "payload_ops": args.payload_ops,
              "cells": []}
    for tokens in args.tokens:
        for mode in ("serial", "pipelined"):
            caches = []
            for _ in range(layers):
                pair = []
                for width in (512, 256):
                    c = KVCache()
                    c.update_and_fetch(mx.random.normal((1, 1, tokens, width)), mx.zeros((1, 1, tokens, 0)))
                    pair.append(c)
                caches.append(pair)
                if args.kv_append != "reference":
                    glm53_kv_append.engage(pair, args.kv_append)
            mx.eval([c.state for pair in caches for c in pair])
            ids = mx.random.randint(0, tokens, (rows,))
            row_latent, row_raw = mx.random.normal((1, 1, 1, 512)), mx.random.normal((1, 1, 1, 256))
            weight = mx.random.normal((4096, 4096)) * 4096**-0.5
            mx.eval(ids, row_latent, row_raw, weight)

            def step(caches=caches, ids=ids, row_latent=row_latent, row_raw=row_raw, weight=weight):
                outs = []
                h = row_latent[0, 0, :, :1] * mx.ones((1, 4096))
                for latent, raw in caches:
                    for _ in range(args.payload):  # model-like work between MLA layers
                        h = h @ weight
                    for _ in range(args.payload_ops):
                        h = h * 0.999 + 1e-3
                    k, _ = latent.update_and_fetch(row_latent, mx.zeros((1, 1, 1, 0)))
                    r, _ = raw.update_and_fetch(row_raw, mx.zeros((1, 1, 1, 0)))
                    outs.append(mx.take(k[0, 0], ids, axis=0).sum() + r[0, 0, -4:].sum())
                return mx.stack(outs).sum() + h.sum() * 0

            mx.eval(step())
            gaps, peaks = [], []
            if mode == "serial":
                for _ in range(args.steps):
                    mx.reset_peak_memory()
                    active = mx.get_active_memory()
                    t0 = time.perf_counter()
                    mx.eval(step())
                    gaps.append(time.perf_counter() - t0)
                    peaks.append(mx.get_peak_memory() - active)
            else:
                pending = step()
                mx.async_eval(pending)
                last = time.perf_counter()
                for _ in range(args.steps):
                    mx.reset_peak_memory()
                    active = mx.get_active_memory()
                    nxt = step()
                    mx.async_eval(nxt)
                    pending.item()
                    now = time.perf_counter()
                    gaps.append(now - last)
                    peaks.append(mx.get_peak_memory() - active)
                    last = now
                    pending = nxt
            gaps.sort()
            cell = {"tokens": tokens, "mode": mode, "kv_append": args.kv_append,
                    "p50_ms": gaps[len(gaps) // 2] * 1e3, "p90_ms": gaps[int(len(gaps) * 0.9)] * 1e3,
                    "max_ms": gaps[-1] * 1e3,
                    "peak_above_active_mb_p50": (sorted(peaks)[len(peaks) // 2] / 2**20) if peaks else None,
                    "buffer_mb_per_layer": tokens * (512 + 256) * 4 / 2**20}
            record["cells"].append(cell)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()}), flush=True)
            del caches
            mx.clear_cache()
    record["kv_append_status"] = glm53_kv_append.route_status()
    (args.out / "kv-update.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
