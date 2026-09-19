# Batched MTP verify matmuls: fold strided x into M, qmv_wide tiles to 8 (2026-09-19, M5 Max)

mlx fork branch `claude/qmv-wide-tile8` @ `ad71bfaa7` (on top of `c9ff96be0`,
the col_reduce fix). Now installed in the shared mlx2 venv as
`0.32.2.dev20260919+ad71bfaa7`. Rollback: `c9ff96be0`.

## What was wrong

Batched self-MTP verify calls `quantized_matmul` with a strided
`(lanes, L, K)` view (for example `(2, 3, 5120)` on the 27B at 2 lanes). With 2-D
weights, MLX only folds leading dims into M when x is fully row-contiguous. The
view is not, so it took the batched path and streamed the whole weight matrix
once per lane. Separately, affine `qmv_wide` capped its tile at 5 vectors, so
M = 6..8 re-read the weights in two tiles.

The fork now copies such x to contiguous, but only when there is more than one
batch element. It also instantiates affine `qmv_wide` at 6/7/8 vectors and
tiles up to 8.

## Evidence

All kernel timings are **cache-cold**: they rotate through about 1 GiB of
distinct weight copies so every op streams from DRAM. Warm-cache numbers
(`qmm-multicolumn-m5max-20260919/`) overstate small-M headroom, because a
single 20-45 MB matrix sits in the system-level cache.

- `cold-default.jsonl` / `cold-forceqmm.jsonl`: M=1..16 on the old build.
  `MLX_QMV_LIMIT=0` forces split-K qmm, which costs a flat ~4.1-4.5x M=1 on
  most shapes (1.6x on the 27B up projection). qmv_wide is 1.0-1.4x up to M=5
  and 1.8-4.9x at M=6..12.
- `strided-*.jsonl`, `s2-*.jsonl` (`strided.py`; the `s2` runs used lanes
  1x3, 1x4, 2x3): the real strided verify shapes, old (`base`) vs new.
  - 2 lanes: 4-bit **+20-37%**, 8-bit **+61-79%**.
  - 4 lanes x 3: 4-bit +12-36%, 8-bit +64-105%.
  - 1 lane: within ±4% (noise).
  - 4 lanes x 4 (M=16): 27B up +158%, but qkv -10% and Flash-Next attn
    -14%, because flattened M=16 crosses into split-K qmm.
- `e2e-summaries.jsonl`: `benchmark_serving.py`, 3 rounds, widths 1/2/4,
  greedy, 160 tokens. Main vs new, run as three interleaved pairs on the 27B
  (`ab_interleaved.sh`) plus one pair on the 35B.

| Model | 1 lane | 2 lanes | 4 lanes |
|---|---|---|---|
| Qwen3.8-27B-oQ4e-mtp (mean of 3 pairs) | 45.0 → 42.9 (-4.6%, within sd 2.4) | 54.0 → **58.9 (+9.1%)** | 62.4 → **65.7 (+5.3%)** |
| Qwen3.6-35B-A3B oQ4e-mtp (1 pair) | 135.5 → 131.0 | 184.1 → 187.9 | 223.2 → 227.0 |

Greedy outputs are byte-identical between main and new in every pair. mlx
`test_quantized` and `test_reduce` pass on Metal (70 tests), including the new
`test_qmv_strided_batch_and_wide_tiles`. Without `MLX_ENABLE_TF32=0`, 1364
subtests fail on both the old and the new build, because the float32 reference
matmul runs in TF32.

## Why the end-to-end gain is small

These matmuls are only part of a verify cycle; GDN, attention, sampling and
scheduling make up the rest. The 35B routes its experts through `gather_qmm`,
which this change does not touch. Bigger small-M wins would need an M5
neural-accelerator (NAX) small-M kernel. A scalar `simdgroup_matrix` prototype
was slower than qmv_wide.
