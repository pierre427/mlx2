# NAX small-M quantized matmul for batched verify (2026-09-19, M5 Max)

mlx fork branch `claude/qmv-wide-tile8` @ `39400a0d4` (on top of `ad71bfaa7`). It
is now installed in the shared mlx2 venv as `0.32.2.dev20260919+39400a0d4`.
Rollback: `ad71bfaa7`. Disable at runtime with `MLX_QMV_NAX=0`.

## What it is

`affine_qmv_nax` (`kernels/quantized_nax.h`) handles affine 4/8-bit weights with
fp16/bf16 activations, 2-D weights and K % 128 == 0. It is dispatched for
12 <= M <= 16, and for M = 8..11 when N*K >= 16M. Each threadgroup owns TN
output rows (64 for 4-bit, 32 for 8-bit) and a split-K range. Split-K is sized
for about 256 threadgroups, minimum 2. For each 128-wide K chunk, the
threadgroup dequantizes the weights into threadgroup memory with coalesced
16-byte loads, then runs `matmul2d(16, TN, 128)` on the M5 matrix units into a
cooperative fp32 accumulator. Split partials are summed by the strided
reduce. `qmv_wide` keeps M < 8 and small matrices at M = 8..11.

## How it was chosen (`prototypes/`, all JIT via `mx.fast.metal_kernel`)

| Stage | Question | Result |
|---|---|---|
| s1 / s1b / s1c | NAX throughput at small tiles (fp16 operands) | Cold, the time is flat in M from 1 to 16. The accelerator consumes about 0.4e12 weights/s from device memory and about 1.0-1.4e12 from threadgroup memory. |
| s2 | NAX rate with the weight tile already in threadgroup memory | 1.0-1.4e12 weights/s (up to 44 TFLOPS useful at M=16). This is just enough for 4-bit weights at DRAM bandwidth. |
| s3 | First fused dequant -> tile -> NAX kernel | Flat ~1.5-2.0x the one-row cost at every M. |
| s4 | Double buffering (warp-specialized producer simdgroups), 8-bit, group sizes, bf16 | No overlap gain: single-buffer tn64/nt128 is best for 4-bit (~1.6x), tn32/nt64 for 8-bit (1.1-1.3x). |
| s5 | NAX reading raw `uint4b_format` / `uint8_t` weights directly (per-group scale epilogue) | Exact, but slower: 4-bit 2.05-2.4x, 8-bit 1.4-1.6x. |

## Integrated results (cache-cold; time relative to M=1 of the shipped build)

`cold-shipped*.jsonl` is the shipped `ad71bfaa7` build. `cold-naxoff.jsonl` is
the new build with `MLX_QMV_NAX=0` (matches shipped). `cold-naxon.jsonl` is the
first integration, with a fixed split of 2. `cold-naxon2.jsonl` adds adaptive
split-K. The final M = 8..11 size gate was derived from `naxon2`.

- 4-bit, 27B and Flash-Next shapes: M=8 goes from 1.81-2.43x to 1.53-1.58x;
  M=12-16 from 3.46-4.77x to 1.55-1.65x. The exception is 27B up at M=16,
  which stays at 1.6x in both builds.
- 8-bit: up to 3.0x becomes 1.08-1.46x.
- 1024x2048 shared-expert projection: +18-39% at M >= 12. At M=8 it would
  lose 19-20%, so that case is gated to `qmv_wide`.

## End to end (`ab_nax.sh`, `e2e-summaries.jsonl`)

Interleaved pairs, main = `ad71bfaa7` against new via a PYTHONPATH overlay:
three pairs on the 27B and two on the 35B, `benchmark_serving.py` 3 rounds,
greedy, 160 tokens. Aggregate tok/s:

| Model | 1 lane | 2 lanes | 4 lanes |
|---|---|---|---|
| Qwen3.8-27B-oQ4e-mtp | 46.6 -> 45.4 (noise) | 64.6 -> 61.9 (noise; main sd 4.1) | **71.1 -> 83.8 (+17.9%)** |
| Qwen3.6-35B-A3B oQ4e-mtp | 137.1 -> 133.0 (noise) | 189.2 -> 190.1 | 227.3 -> 234.6 (+3.2%) |

Verify M is 3 at 1 lane and 6 at 2 lanes, so NAX only engages at 4 lanes
(M=12). The 35B gains less because its experts use `gather_qmm`. Greedy
outputs are identical at 1 and 2 lanes. At 4 lanes, 2 of 4 prompts (27B) and 3
of 4 (35B) diverge but are deterministic per arm: NAX dequantizes into the
activation dtype in threadgroup memory, like the existing prefill `qmm_t_nax`,
whereas qmv_wide uses fp32.

On the shared venv after install (`bench-shared-smoke.json` row): 27B 49.4
tok/s at 1 lane and 87.6 at 4 lanes. Outputs are identical to the overlay run.
mlx `test_quantized` + `test_reduce` pass on Metal (71 tests, `MLX_ENABLE_TF32=0`),
including `test_qmv_nax_small_m`.

## Not done

- 2-lane verify (M=6) is still on `qmv_wide`: NAX is not faster there.
- `gather_qmm` for MoE experts has no NAX small-M path.
- The ANE composition measurement spike (see chat, 2026-09-19) has not been
  run.
