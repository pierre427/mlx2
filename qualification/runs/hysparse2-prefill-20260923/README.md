# HySparse2 prefill paths on the GPU (2026-09-23)

- **Machine:** M5 Max with 128 GiB, mlx `0.32.2.dev20260919+39400a0d4`,
  `MLX_ENABLE_TF32=0`.
- **Lock:** the GPU lock was taken through the waiters protocol, 16:41–16:48
  for the benchmark and 16:48–16:53 for the exactness check.
- **Noise source:** a long-running co-resident `mlx2.server` (Qwen3.8-27B)
  stayed up throughout. The arms are interleaved, two rounds each, and each arm's best
  time is reported.
- **Model:** the paper layout (12 SWA + FA + 12 SWA | 4 × (FA + 5 SA)), hidden
  size 2048, MQA with 64 query heads × 256, a dense SwiGLU MLP (4096), bf16,
  random weights, 6.16B parameters. The numbers are time and memory only;
  quality is not measured.

Produced by `scripts/bench_hysparse2_prefill.py` → `bench.json`/`bench.log`
and `scripts/check_hysparse2_gpu_exactness.py` → `exactness.json`.

## Time to first-token logits (best of 2)

| Prompt | serving loop (lazy exit) | paper exit | suffix bound | suffix vs paper | cost model predicted |
|---:|---:|---:|---:|---:|---:|
| 4K | 0.875 s | 0.883 s | 0.531 s | 1.66× | 1.67× |
| 16K | 3.95 s | 4.06 s | 2.03 s | 2.00× | 2.09× |
| 64K | 21.3 s | 22.2 s | 9.06 s | 2.46× | 2.83× |
| 128K | 53.0 s | 55.7 s | 18.2 s | 3.05× | 3.72× |

- **The serving loop already gets the paper's exit.** mlx2's
  `BatchGenerator` loop evaluates only cache state, so MLX never runs the
  cross-decoder on the prompt rows. Its times match `paper_exit` to within 5%.
- **The suffix bound scales linearly in the prompt:** 9.06 s at 64K and 18.2 s
  at 128K. The paper exit grows superlinearly.
- **Below the FLOP prediction at 64K and 128K.** The cost model counts FLOPs.
  Long-context full attention on the GPU reaches higher FLOP efficiency than
  the projections do, so the measured gap is smaller than the predicted one.
- **No-exit cost is only indicative.** `no_exit` forces the cross-decoder on
  every row at chunk 256 through the unfused explicit-score path: 4.8× and 5.5×
  slower than `paper_exit` at 4K and 16K. This overstates what the paper's exit
  saves, because the cross FA path here is a reference path, not a kernel.
- **Peak memory:** 12.4–14.7 GiB in every arm. Weights take 11.5 GiB; the
  suffix bound peaks lowest at 64K and 128K.

## Exactness on the GPU

| Run | top-2 margin | paper exit max abs diff | suffix bound max abs diff | argmax |
|---|---:|---:|---:|---|
| fp32, 32K | 0.115 | 1.4e-5 | 1.5e-5 | agrees |
| bf16, 128K | 0.078 | 0.074 | 0.094 | agrees |

The benchmark's first 128K run reported one argmax disagreement for
`suffix_bound`. There, the path-to-path bf16 difference (0.078) was as large as
the top-2 margin. The exactness run shows the same near-tie with a different
prompt. In fp32 both paths agree with the reference to within 1.5e-5. The CPU
tests pin exactness at 2e-5 in fp32.
