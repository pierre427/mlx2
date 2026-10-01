"""Metal bit-exactness and timing of the fused QSA block scores.

Compares ``qwen4_qsa_scores.block_scores`` with the stock MLX chain it
replaces (``stock_scores``) as raw bytes, and the ``argpartition`` block ids
taken from each, over decode (1 row), MTP verify (2, 3 rows) and wider rows,
at 16K..128K-token contexts, for random keys and for pooled keys produced by
the indexer's own pooling of random raw keys.  Then times both (32 chained
launches per eval, median of 7).

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/check_qwen4_qsa_scores.py --out r.json
"""

import argparse
import json
import math
import statistics
import time

import mlx.core as mx
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2051 * 4 + 3, 16384, 32768, 65536, 98304, 131072])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    a = ap.parse_args()

    from mlx2.runtime.models import qwen4_qsa_scores as S

    top = 512
    report = {"mlx": mx.__version__, "cases": [], "timing": []}
    mismatches = 0
    for context in a.contexts:
        for rows in a.rows:
            for seed in range(a.seeds):
                rng = np.random.default_rng(context * 31 + rows * 7 + seed)
                blocks = context // 4
                offset = context - rows  # rows end at the context's last position
                q = mx.array(rng.normal(size=(1, rows, 4, 128)).astype(np.float32)).astype(mx.bfloat16)
                if seed % 2:
                    # magnitudes like the served indexer: normed, mostly small, a few large
                    pooled = mx.array((rng.normal(size=(1, blocks, 128)) *
                                       rng.choice([0.05, 1.0, 4.0], size=(1, blocks, 1))).astype(np.float32)
                                      ).astype(mx.bfloat16)
                else:
                    pooled = mx.array(rng.normal(size=(1, blocks, 128)).astype(np.float32)).astype(mx.bfloat16)
                if seed == 2:
                    # ties: duplicated blocks make equal scores across ids
                    pooled = mx.concatenate([pooled[:, : blocks // 2], pooled[:, : blocks - blocks // 2]], axis=1)
                q_pos = mx.arange(offset, offset + rows)[None, :]
                starts = mx.arange(blocks) * 4
                valid = (starts + 3)[None, None, :] <= q_pos[..., None]
                reason = S.supported(q, pooled)
                if reason is not None:
                    report["cases"].append({"context": context, "rows": rows, "seed": seed, "refused": reason})
                    continue
                want = S.stock_scores(q, pooled, valid, 128)
                got = S.block_scores(q, pooled, offset, 4)
                k = min(top, blocks)
                ids_want = mx.argpartition(want, kth=blocks - k, axis=-1)[..., -k:]
                ids_got = mx.argpartition(got, kth=blocks - k, axis=-1)[..., -k:]
                mx.eval(want, got, ids_want, ids_got)
                w, g = np.array(want), np.array(got)
                same = bool(w.tobytes() == g.tobytes())
                same_ids = bool(np.array_equal(np.array(ids_want), np.array(ids_got)))
                diff = int(np.sum(w.view(np.uint32) != g.view(np.uint32)))
                if not (same and same_ids):
                    mismatches += 1
                report["cases"].append({"context": context, "rows": rows, "seed": seed,
                                        "bytes_identical": same, "ids_identical": same_ids,
                                        "differing_scores": diff})
                print(context, rows, seed, "bytes", same, "ids", same_ids, "diff", diff, flush=True)
    # batched rows (the 4-lane cache): one logical offset per row, as an array
    for context in (16384, 32768, 65536):
        for rows in (1, 3):
            rng = np.random.default_rng(context + rows)
            batch, blocks = 4, context // 4
            offsets = mx.array([context - rows, context - rows - 5, context - rows - 99, context - rows - 1001],
                               dtype=mx.int32)
            q = mx.array(rng.normal(size=(batch, rows, 4, 128)).astype(np.float32)).astype(mx.bfloat16)
            pooled = mx.array(rng.normal(size=(batch, blocks, 128)).astype(np.float32)).astype(mx.bfloat16)
            q_pos = offsets[:, None] + mx.arange(rows)[None, :]
            valid = (mx.arange(blocks) * 4 + 3)[None, None, :] <= q_pos[..., None]
            reason = S.supported(q, pooled) or S.offset_supported(offsets, batch)
            if reason is not None:
                report["cases"].append({"context": context, "rows": rows, "batch": batch, "refused": reason})
                mismatches += 1
                continue
            want = S.stock_scores(q, pooled, valid, 128)
            got = S.block_scores(q, pooled, offsets, 4)
            ids_want = mx.argpartition(want, kth=blocks - top, axis=-1)[..., -top:]
            ids_got = mx.argpartition(got, kth=blocks - top, axis=-1)[..., -top:]
            mx.eval(want, got, ids_want, ids_got)
            same = bool(np.array(want).tobytes() == np.array(got).tobytes())
            same_ids = bool(np.array_equal(np.array(ids_want), np.array(ids_got)))
            if not (same and same_ids):
                mismatches += 1
            report["cases"].append({"context": context, "rows": rows, "batch": batch,
                                    "bytes_identical": same, "ids_identical": same_ids})
            print(context, rows, "batch", batch, "bytes", same, "ids", same_ids, flush=True)
    report["mismatching_cases"] = mismatches
    report["checked_cases"] = sum(1 for c in report["cases"] if "refused" not in c)

    if not a.no_timing:
        def bench(fn, iters=32, reps=7):
            mx.eval(fn())
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                mx.eval([fn() for _ in range(iters)])
                ts.append((time.perf_counter() - t) / iters)
            return round(1e6 * statistics.median(ts), 2)

        for context in (16384, 32768, 65536, 131072):
            for rows in (1, 3):
                blocks = context // 4
                q = mx.random.normal((1, rows, 4, 128)).astype(mx.bfloat16)
                pooled = mx.random.normal((1, blocks, 128)).astype(mx.bfloat16)
                offset = context - rows
                q_pos = mx.arange(offset, offset + rows)[None, :]
                valid = ((mx.arange(blocks) * 4 + 3)[None, None, :] <= q_pos[..., None])
                mx.eval(q, pooled, valid)
                k = top
                t_stock = bench(lambda: S.stock_scores(q, pooled, valid, 128))
                t_fused = bench(lambda: S.block_scores(q, pooled, offset, 4))
                t_stock_sel = bench(lambda: mx.argpartition(S.stock_scores(q, pooled, valid, 128),
                                                            kth=blocks - k, axis=-1)[..., -k:])
                t_fused_sel = bench(lambda: mx.argpartition(S.block_scores(q, pooled, offset, 4),
                                                            kth=blocks - k, axis=-1)[..., -k:])
                row = {"context": context, "rows": rows, "stock_us": t_stock, "fused_us": t_fused,
                       "stock_with_argpartition_us": t_stock_sel, "fused_with_argpartition_us": t_fused_sel}
                report["timing"].append(row)
                print("TIMING", json.dumps(row), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print("MISMATCHING CASES", mismatches, "of", report["checked_cases"], flush=True)
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
