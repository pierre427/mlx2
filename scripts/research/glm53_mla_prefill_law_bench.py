"""GLM-5.3 MLA prefill attention: absorbed (512-wide latent) vs expanded (64 heads x 256).

Synthetic shapes only (no checkpoint): one DSA layer's attention for a chunk
of L prompt queries over T committed keys, as the model computes it today
(``_mla_masked``: q (1, 64, L, 512) against the shared latent (1, 1, T, 512)
as both K and V through ``mx.fast.scaled_dot_product_attention``) and as the
un-absorbed form every other GLM-5.3 stack uses for prompts (K = latent @
k_b^T and V = latent @ v_b^T per head, D = 256, the fused D256 kernel), with
a causal mask and with a boolean selection mask of the sparse law's shape.
Times the K/V expansion matmuls separately.  Research only; prints a table.

usage: glm53_mla_prefill_law_bench.py --out DIR [--rows 2048] [--contexts 2048 4096 8192 16384]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def timed(fn, reps=5):
    fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    L = args.rows
    for T in args.contexts:
        first = T - L
        mx.random.seed(T)
        latent = mx.random.normal((1, 1, T, 512)).astype(mx.float32)
        q_heads = mx.random.normal((1, 64, L, 256)).astype(mx.float32)
        k_b = mx.random.normal((64, 512, 256)).astype(mx.float32) * 0.05  # (heads, latent, qdim)
        v_b = mx.random.normal((64, 256, 512)).astype(mx.float32) * 0.05  # (heads, vdim, latent)
        pos = first + mx.arange(L)
        causal = (mx.arange(T)[None, :] <= pos[:, None])[None, None]
        # A sparse-law-shaped boolean mask: ~2051 visible keys per query row.
        keep = mx.random.uniform(shape=(1, 1, L, T)) < min(1.0, 2051 / T)
        sparse_mask = mx.logical_and(causal, keep)
        mx.eval(latent, q_heads, k_b, v_b, causal, sparse_mask)

        def absorbed(mask):
            # The whole absorbed consumer: q -> k_b, attention in latent space,
            # then the per-head v_b projection (the model's _mla_finish), so
            # both laws are timed to the same 256-wide per-head output.
            q_lat = mx.einsum("bhlq,hkq->bhlk", q_heads, k_b)  # (1, 64, L, 512)
            out = mx.fast.scaled_dot_product_attention(q_lat, latent, latent, scale=256**-0.5, mask=mask)
            return mx.einsum("bhlk,hvk->bhlv", out, v_b)

        def expanded(mask, dtype):
            lat = latent[0, 0]  # (T, 512)
            K = mx.einsum("tk,hkq->htq", lat, k_b).astype(dtype)[None]  # (1, 64, T, 256)
            V = mx.einsum("tk,hvk->htv", lat, v_b).astype(dtype)[None]  # (1, 64, T, 256)
            return mx.fast.scaled_dot_product_attention(
                q_heads.astype(dtype), K, V, scale=256**-0.5, mask=mask)

        def expansion_only(dtype):
            lat = latent[0, 0]
            return (mx.einsum("tk,hkq->htq", lat, k_b).astype(dtype),
                    mx.einsum("tk,hvk->htv", lat, v_b).astype(dtype))

        row = {"L": L, "T": T}
        for name, fn in (
            ("absorbed_causal", lambda: mx.eval(absorbed(causal))),
            ("absorbed_sparse", lambda: mx.eval(absorbed(sparse_mask))),
            ("expanded_f32_causal", lambda: mx.eval(expanded(causal, mx.float32))),
            ("expanded_f32_sparse", lambda: mx.eval(expanded(sparse_mask, mx.float32))),
            ("expanded_f16_causal", lambda: mx.eval(expanded(causal, mx.float16))),
            ("expanded_f16_sparse", lambda: mx.eval(expanded(sparse_mask, mx.float16))),
            ("expansion_only_f32", lambda: mx.eval(*expansion_only(mx.float32))),
            ("expansion_only_f16", lambda: mx.eval(*expansion_only(mx.float16))),
        ):
            try:
                row[name] = round(timed(fn, args.reps), 2)
            except Exception as error:  # noqa: BLE001 - research table
                row[name] = f"error: {error}"[:120]
        # Numerical agreement of the laws on the causal mask (fp32 expanded vs absorbed).
        ev = absorbed(causal)
        e = expanded(causal, mx.float32)
        mx.eval(e, ev)
        row["max_abs_diff_f32"] = float(mx.max(mx.abs(ev - e)).item())
        row["max_abs_ref"] = float(mx.max(mx.abs(ev)).item())
        rows.append(row)
        print(json.dumps(row), flush=True)
        mx.clear_cache()
    (args.out / "bench.json").write_text(json.dumps(
        {"schema": "mlx2.glm53-mla-prefill-law-bench.v1", "mlx": mx.__version__,
         "device": mx.device_info(), "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
