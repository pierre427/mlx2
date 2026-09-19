# Multi-column quantized matmul on M5 Max (llama.cpp#29110 probe, 2026-09-19)

`scripts/bench_qmm_multicolumn.py --i-own-the-gpu`: fp16 activations,
group size 64. The median of 30 samples of 20 chained ops. Each value is
time(M) / time(1).

| Shape | Bits | M=1 µs | M=2 | M=3 | M=4 | M=8 |
|---|---|---|---|---|---|---|
| 27B qkv 8192×5120 | 4 | 38.2 | 1.08 | 1.27 | 1.50 | 3.24 |
| 27B up 17408×5120 | 4 | 67.1 | 1.33 | 1.71 | **2.10** | 5.90 |
| 27B down 5120×17408 | 4 | 67.4 | 1.50 | 1.95 | **2.35** | 4.34 |
| 27B up | 8 | 169.6 | 1.00 | 1.00 | 1.00 | 2.17 |
| 27B down | 8 | 167.3 | 1.01 | 1.06 | 1.06 | 2.26 |
| A3B attn 4096×2048 | 4 | 16.8 | 1.21 | 1.49 | 1.52 | 2.35 |
| A3B expert 512×2048 | 4 | 11.0 | 1.02 | 1.10 | 1.19 | 1.27 |
| lm_head 151936×2048 | 4 | 305.8 | 0.95 | 1.53 | 2.60 | 4.30 |

**Superseded.** These are warm-cache numbers: one matrix repeated, which sits
in the system-level cache. With DRAM-resident weights, M=2..5 costs only
1.0-1.4x M=1, so the "~2x headroom" read below is wrong. The real gap is at
M >= 6, plus strided batched inputs re-reading the weights per lane. See
`../qmm-verify-flatten-tile8-20260919/`.

**Original reading (superseded).** 8-bit weights stay at the weight-read floor up to M=4. 4-bit dense
projections do not: at M=4 they cost 2.1-2.35× M=1, which is close to the
8-bit cost. The 4-bit path is dequant or compute bound at 2-4 columns, which is
exactly the MTP verify window (k+1). A multi-column 4-bit kernel that shares
dequant across columns has up to ~2× headroom on 27B dense verify matmuls.
MoE experts are already cheap (1.19× at M=4), which matches llama.cpp's +4%
on 35B-A3B. This reopens the 08-21 kill, which was measured on Muse only.
