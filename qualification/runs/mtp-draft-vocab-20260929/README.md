# MTP draft-vocabulary qualification — 2026-09-29

## Outcome

The artifact-bound 65,536-token native-MTP proposal head is implemented and was observed used on an Apple M5 Max. It remains default-off and is **not selected** for production: it produced a small, repeatable single-stream gain, but no aggregate benefit at batch widths 4 or 8, and the B4 exact-token parity gate failed on a repeated near-tie.

This is a qualification-harness result, not a serving-route claim. No production route was changed.

## Configuration

- Model artifact: `Qwen3.8-Flash-Next-MLX-4bit-MTP`
- MLX: `0.32.2.dev20260919+39400a0d4`
- Hardware: Apple M5 Max, 128 GiB unified memory
- Draft vocabulary: 65,536 sorted unique ids from the upstream English/code list
- Target verification: unchanged full vocabulary
- Workload: greedy decode, 128 emitted tokens per lane, 3 counterbalanced repetitions per arm, one model load
- Widths: B1, B4, B8
- Peak active MLX memory: 72.46 GiB
- Timed-round swapout growth: 0 MB

The local model's proposal `lm_head` is affine 4-bit (`group_size=64`), unlike the upstream recipe's INT8 head. That leaves less head bandwidth for this reduction to save and is a material difference from the DGX result.

## Results

| Width | Full-head median | Reduced-head median | Median paired ratio | Median tokens/cycle | Exact greedy parity |
| --- | ---: | ---: | ---: | ---: | --- |
| B1 | 100.05 tok/s | 103.67 tok/s | 1.0343 (+3.43%) | 2.909 / 2.909 | 3/3 |
| B4 | 128.53 tok/s | 129.79 tok/s | 0.9908 (-0.92%) | 2.612 / 2.560 | 0/3 |
| B8 | 145.81 tok/s | 143.56 tok/s | 0.9916 (-0.84%) | 2.606 / 2.579 | 3/3 |

The paired B1 gains ranged from +3.37% to +5.43%. B4 ranged from -2.80% to +2.85%, and B8 from -1.54% to +2.30%.

Mechanism receipts showed reduced-head calls in every candidate round and full-head calls in every control round. A constrained-request probe observed the capability-based full-vocabulary bypass. Target verification was full-vocabulary in both arms.

## B4 divergence

All three B4 repetitions first diverged at lane 1, position 118. The control emitted token 2222; the reduced arm emitted token 15773. A separate teacher-forced full-target evaluation at the shared prefix produced:

- token 15773: logit 20.625, target argmax
- token 2222: logit 20.500, runner-up
- margin: 0.125

The repeated near-tie is consistent with block-shape/BF16 rounding in the full target-head evaluation rather than a verification bypass: the reduced arm selected the independently recomputed target argmax. It nevertheless fails the exact-token parity gate.

## Decision

- Implemented: yes
- CPU verified: yes
- GPU observed used: yes, in the isolated qualification harness
- Constrained full-head bypass observed used: yes
- Qualified for selection: no
- Selected by default or by a production route: no

Further work should focus on proposal acceptance and batched economics, not promotion of this implementation as-is.

## Evidence

- `smoke.json` — SHA-256 `fa00c87e41210dc875fc63fcfd58ca87ca4adae34c0b36562750ff9b4e6f4e67`
- `ab-b1-b4-b8.json` — SHA-256 `50299f47fcbce9c8c5f3dbd3c53529b7bd26234b7910466a9cff89ec7decbcf8`
- `divergence-analysis.json` — SHA-256 `71e6899235038ac9c8c9e4851ebc1e368a965e8470ea7e3784260c4e707818bd`

The A/B receipt records the tested source identity, local diff and untracked-file hashes, artifact fingerprint, per-round timings, arm order, mechanism counters, token outputs, memory, and swap readings.
