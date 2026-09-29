# Lane matmul: row-invariant small-M projections for every weight format (2026-09-28)

`mlx2.runtime.lane` is one matmul family for affine 2/3/4/5/6/8-bit weights
(group sizes 32/64/128) and unquantized bf16/fp16 on the M5 tensor units.
Every row of a 1–128-row call is bitwise equal to that row computed alone,
and the weights are read once for all rows, so 16 rows cost about as much as
one. The detected `auto` policy is selected for most dense models, with
per-format crossovers. Muse-Glimmer q4 resolves to `off` after the 2026-09-29
context qualification; mixture-of-experts models also resolve to `off`. See
the default-on sweep below and the `--lane-matmul` section in docs/SERVING.md.
It is also the verify building block that the
[DFlash2 verdict](QWEN38-DFLASH-TREE-VERDICT-2026-09-28.md) calls for.

## Design

- **Arithmetic:** `y[m, n] = Σ_g s[n,g]·P[m,n,g] + b[n,g]·xs[m,g]`.
  `P` is the MPP `matmul2d` product of one weight group. `xs` is the group
  sum of `x`, computed in the same kernel: each of the four lanes sharing a
  row sums a quarter of the group, and the quarters combine as
  `(q0+q1)+(q2+q3)`. The K slices are fixed by weight shape and format and
  reduced in slice order. Unquantized weights accumulate `P` directly.
- **Weight formats:** q4 (`uint4b`), q8 (`uint8`) and bf16/fp16 feed the
  tensor units in MLX's layout. q2, q3, q5 and q6 are decoded per group from
  MLX's LSB-first bitstream into threadgroup `uint8` by generated, fully
  unrolled code.
- **Installation** (`lane.install(model)`) swaps each supported
  `QuantizedLinear`/`Linear` instance to a lane subclass. It stacks
  same-format siblings that read one input (q/k/v, gate/up, the GDN
  `in_proj_*`; the names are generic, and adapters can pass their own) into
  one launch, using zero-copy row views so there is no second weight copy.
  Unsupported projections, calls above `max_rows` (prefill), non-M5 devices,
  and calls below `min_rows` stay on stock kernels.
- **Numerical law:** `min_rows=4` (the default, "crossover") keeps stock
  arithmetic for 1–3 rows. `min_rows=1` ("exact") makes every verify row
  equal the same row decoded alone. Both are distinct from stock MLX; the law
  ID is reported by `install()` and must be bound into a route's cache
  identity.

## Evidence

**Correctness** (`scripts/lane_matmul_gate.py`, M5 Max): 76/76 cases pass
across every bit width, group size, and bf16/fp16 activation type, and 28/28
pass at model shapes, including a 2.5 GB bf16 vocabulary head. In every case
each row of a 1–128-row call is bitwise equal to the row alone, and the error
against an fp32 dequantized reference is at or below MLX's own. 36 CPU tests
cover the rest. One of them runs the *generated* decode against MLX's
dequantization for 2/3/5/6-bit at every group size; others cover geometry,
grouping with view-backed weights, and fallback.

**Cost inside mlx2's own models.** Each cell is the median of eight in-process
alternating runs, in ms per forward, stock / lane grouped (exact mode)
(`lane-ab5-*.json`):

| Model | Context | 1 row | 3 | 4 | 8 | 16 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B oQ4e (338 q4 + 166 q5) | 32 | 34.8 / 44.5 | 45.0 / 49.7 | 50.6 / 49.9 | 82.8 / 50.3 | 108.8 / 55.6 |
| | 8K | 35.9 / 45.8 | 49.4 / 53.3 | 56.3 / 56.1 | 92.0 / 58.7 | 105.1 / 57.4 |
| | 24K | 38.6 / 49.5 | 56.5 / 60.3 | 62.0 / 61.9 | 106.9 / 77.0 | 115.4 / 68.3 |
| Qwen3.8-27B uniform 4-bit | 32 | 32.5 / 40.9 | 36.3 / 41.6 | 39.0 / 42.1 | 70.1 / 46.8 | 69.0 / 45.1 |
| | 24K | 36.5 / 45.2 | 49.5 / 53.5 | 55.5 / 55.7 | 98.3 / 69.7 | 90.0 / 62.7 |
| Muse-Glimmer-30B 4-bit | 32 | 33.2 / 43.8 | 38.4 / 47.4 | 40.8 / 47.3 | 69.4 / 49.4 | 68.3 / 50.9 |

- From 8 rows the lane matmul cuts the forward by 29–49% on both
  architectures. On oQ4e it also beats TensorFold's q4-only kernel from 4
  rows up, because it covers the q5 layers.
- Stacking siblings saves 1–3 ms per forward. Folding the group sum into the
  kernel removed a launch per projection. Neither closes the one-row gap.
- **The remaining one-row cost is the kernel itself.** Each forward is about
  +10 ms (+28%) versus stock at one row. Per projection, a one-row tensor tile
  costs about 1.35× MLX's one-row kernel (`proj_cost.py`). Running a one-row
  call as two rows did not help at model level.
- **The crossover is about 4 rows on Qwen3.8 and 5–6 on Muse.** With the
  default `min_rows=4`, one-token decode and native MTP's 3-row verify are
  unchanged, and 8–16-row verifies and batched decode get the flat curve.

## Default-on sweep: 34 artifacts at 8K and 32K

`scripts/lane_model_sweep.py` loaded every supported artifact through its
registry adapter, the same way the server does (receipts in
`qualification/runs/lane-default-sweep-20260928/`). It installed the lane
matmul and, at 8K and 32K tokens of real text, compared one 1/4/8/16-row
forward under stock and lane. It also compared a 16-row verify under stock
and under lane against 16 one-token stock steps (what decode produces).

- **Quality.** On every model and context, lane's 16-row output agreed with
  one-token decode as well as stock's own multi-row kernel did: 14–16 of 16
  top-1 for both, with similar log-prob spreads.
- **Cost, lane versus stock:**

| Group | Artifacts | 4 rows | 8 rows | 16 rows |
|---|---|---:|---:|---:|
| Dense q4/q5/q6 (incl. mixed oQ4e) | Qwen3.8 ×5, Qwen3.6-27B, Muse ×2 | +5 to +23% | **−8 to −34%** | **−12 to −39%** |
| Dense q8 | Qwen3.8/3.6-27B, Muse, Gemma 4 31B | +13 to +21% | −2 to −15% | −4 to −12% |
| Dense bf16 | ThinkingCap, CRACK, Muse ×3, Gemma 4 31B, Gemma 3n, MiniCPM-o | +18 to +26% | +8 to +24% | **−9 to −28%** |
| MoE | Qwen3.6-A3B ×3, North ×2, Laguna, Xing ×2, Nemotron, Gemma 4 26B ×2, Flash-Next ×2 | −2 to +27% | −7 to +12% | −11 to +1% |

  These results set the built-in thresholds. q2–q8 cross over at 8 rows and
  bf16/fp16 at 16. MoE models resolve to off: their expert layers use
  `gather_qmm`, which lane does not cover. MoE detection uses the artifact
  config's expert count or an expert module; it matched all 34 artifacts
  (13 MoE, 21 dense).
- **Swap.** Both Flash-Next runs swapped (about 1.1M and 2.0M pages). The
  harness loads Flash-Next outside the server's expert-streaming memory
  policy and copies the 32K cache three times. Their timings are not
  representative; the stock/lane comparison is same-process. The driver
  then aborted any job whose swap-outs rose by more than 100k pages; no
  later job triggered it.

**Serving, default `auto`** (`mlx2.server`, 8 concurrent greedy requests, 96
tokens each, `serving/`):

| Model | `--lane-matmul off` | default `auto` |
|---|---:|---:|
| Qwen3.8-27B oQ4e (dense) | 104.0 / 102.9 tok/s | **143.2 / 140.6** (repeat 142.1 / 140.0) |

That is +37% aggregate batched-decode throughput. All 8 outputs were
identical in every run. `/v1/status` reported the resolved policy (505
projections, 139 groups) and live counters, and `/metrics` exported
`mlx2_lane_matmul_*`. On Qwen3.6-35B-A3B the same default resolved to off:
nothing was installed, there was no settings change, and the metrics read
zero.

The subsequent Muse-Glimmer-30B 4-bit qualification compared three thermally
controlled width-one repetitions at 1K, 4K and 16K tokens. With lane matmul
`off`, median decode was 26.8, 26.3 and 24.8 tokens/s. With `auto`, it was
28.5, 21.0 and 20.8 tokens/s. The 20×20 domain workload passed 400/400
for both modes with no swap; `auto` improved its median mixed-round aggregate
rate from 55.9 to 57.5 tokens/s. This small batching gain does not compensate
for the 4K and 16K single-prompt losses. The grouped installer also raised
the ready process footprint from about 17.1 to 25.1 GiB. The family default
therefore resolves to `off` for Muse q4; other Muse formats retain the
detected crossover pending their own qualification. An explicit policy can
still select the kernel for a qualified batched workload. On M3 Pro, installation is skipped because
the lane kernel requires M5 hardware. These are end-to-end measurements, not
the isolated projection microbenchmarks above.

## Next

1. Wire the Qwen3.8 DFlash2 route. Use chain or tree verify of 7–16 rows
   through the model's normal forward, with GDN speculation/trim and the lane
   matmul at `min_rows=4`. Qualify it end to end against native MTP.
2. Re-run qualification: the default changes the route identity of every
   dense model.
3. Extend coverage to MoE expert layers (a lane `gather_qmm`) if MoE
   batching matters; today they measure neutral and resolve to off.
4. Reduce the one-row tile cost only if exact mode becomes the default. The
   candidates are TensorFold's tiled weight layout (8–15% per its notes) and
   fusing norms into the projection input.
