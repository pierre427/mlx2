# Qwen4 exact-set frequency and B4 fold gate — 2026-09-17

## Decision

Do **not** integrate or select the experimental B4 shared-prefix fold.

The isolated B4 Metal kernel has a real kernel-only win on a model-produced exact-hit case, but the complete candidate loses after exact-set proof and private-suffix preparation. More importantly, exact B4 hits were rare in the short-divergence workload and absent after a 32-token divergence. The existing B2 fold also lost to the row-local path on both the hit and miss samples at this geometry.

This is an experiment result, not route qualification. The B4 code remains isolated from serving; `qualification=false` and `route_selected=false` in the receipt.

## Bound identities

- Source HEAD: `d9260acec352c602ded321df6490d67bf92cd30e`
- Checkpoint: `~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP`
- Checkpoint configuration SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`
- Model family: `qwen4_exp`, 48 layers, 12 QSA layers, 24 query heads, 2 KV heads, head dimension 256, QSA block size 4, top-k 512 blocks
- Runtime: MLX `0.32.2.dev20260915+2a817ad94`
- Device: Apple M5 Max, 128 GiB unified memory
- Precision: checkpoint-native BF16 activations; A/B rerun with `MLX_ENABLE_TF32=0`

## Workload and exact-set oracle

The harness prefills one real 2,048-token code/documentation prefix, merges that cache into a physical B4 cache, appends four equal-width but semantically different suffixes, and performs 16 real greedy decode steps. Every one of the 12 QSA layers is observed, yielding 192 layer-step opportunities per workload.

The oracle is the same contract used by the existing B2 fold: compact each row's selected blocks, retain only block IDs below the immutable 2,048-token base boundary, and require ordered equality. High Jaccard overlap is not counted as a hit. Full-set equality is recorded separately.

| Divergence before decode | B2 exact base hit | B4 exact base hit | Mean anchor Jaccard | Minimum Jaccard |
|---|---:|---:|---:|---:|
| 4 tokens | 60 / 192 (31.25%) | 4 / 192 (2.08%) | 99.58% | 98.05% |
| 32 tokens | 0 / 192 (0%) | 0 / 192 (0%) | 97.58% | 95.31% |

For these samples, full selected-set hits were identical to immutable-base hits. The 32-token result is the important negative control: visually high overlap did not satisfy exact reuse once the branches had diverged.

Coverage is one prompt family and 384 total layer-step opportunities. It is enough to reject promotion of this mechanism, not enough to claim a population-wide production hit rate.

## Model-bound A/B

The benchmark uses the actual layer-31, decode-step-2 tensors from one observed B4 exact hit in the four-token workload. Queries, BF16 attention K/V, indexer queries, pooled keys, and production selected IDs all came from the bound checkpoint. The production selection replay reproduced the captured selected sets.

Preferred receipt: 10 warmups and 100 measured repetitions.

| Component | B2 existing fold | B4 candidate | Row-local comparator |
|---|---:|---:|---:|
| Selection replay | 0.3218 ms | 0.2589 ms | shared with its same-width comparator |
| Selection compaction | 0.3143 ms | 0.2576 ms | shared with its same-width comparator |
| Exact-set proof, diagnostic only | 0.2673 ms | 0.2944 ms | none |
| Private suffix gather | none | 0.1658 ms | none |
| Attention kernel, exact hit | 0.4249 ms | 0.2415 ms | B2 0.3899 ms; B4 0.4864 ms |
| Composed median | 1.0610 ms | 1.2183 ms | B2 1.0261 ms; B4 1.0030 ms |
| End-to-end speedup versus row-local | **0.967x** | **0.823x** | 1.000x |

The existing B2 entry point performs its device proof internally, so its composed total does not add the diagnostic proof measurement a second time. The B4 prototype does not contain a production proof; its composed total therefore adds the measured B4 proof and the required private-suffix gather.

The B4 kernel by itself was 2.01x faster than the B4 row-local kernel, but proof plus suffix preparation more than consumed the gain. On the model-bound miss sample, the existing B2 fold was also slower: 1.0960 ms composed versus 1.0364 ms row-local, or 0.946x.

Correctness on the hit sample:

- Existing B2 fold versus row-local: bitwise equal on hit and miss samples.
- B4 candidate versus row-local: BF16 maximum absolute error `0.0078125`, mean absolute error `0.0001373`; the large maximum relative error is confined to near-zero reference values.
- Candidate engagement receipt: `1`.

The B4 result is numerical agreement, not bitwise equivalence, because the prototype changes reduction order. That is another reason it remains unqualified.

## State classification

| State | B2 exact fold | B4 prototype |
|---|---|---|
| Implemented | Yes, existing private-delta entry point | Yes, isolated experiment only |
| Qualified | No new qualification from this experiment | No |
| Selected | Existing policy may request B2 under its own gates; this experiment does not change policy | No |
| Observed used | Yes, direct model-bound harness only | Yes, direct model-bound exact-hit harness only |

No serving route, policy default, scheduler behavior, or qualification declaration was changed.

## Artifacts

- `qualification/experiments/qwen4-exact-set-b4-20260917/frequency-suffix4.json` — `c74f81c5c0989fac4e707011f093a938d77c1f85d6bb2f4f61060090ba02de1f`
- `qualification/experiments/qwen4-exact-set-b4-20260917/frequency.json` — `ffb15b1b657d22b81305416ca255507724bdbe098c48e30ebd57ba27cdb4cfa7`
- `qualification/experiments/qwen4-exact-set-b4-20260917/model-bound-ab-repeat.json` — `1f151add06274a8826873f7989f13d3f5dc227deea7c7517c9170c4cfe273021`
- `qualification/experiments/qwen4-exact-set-b4-20260917/model-bound-ab.json` — first 30-repeat corroboration
- `model-sample-suffix4.npz` and `model-sample.npz` — captured model tensors for the hit and miss cases
- `scripts/capture_qwen4_exact_set_frequency.py` — repeatable real-model frequency capture
- `scripts/benchmark_qwen4_exact_set_b4.py` — repeatable model-bound A/B

Focused validation: `tests/test_segmented_qsa_metal.py` plus the private-delta/exact-set cases in `tests/test_qwen4_qsa_indexed.py` completed with five passes and four environment-gated skips. Both new scripts compile, both real-model captures completed, and both A/B receipts report `passed=true`.

## Revisit condition

Do not revisit B4 exact folding unless a design removes the separate proof and suffix-gather costs (for example, a single kernel that proves, gathers, and folds without host specialization) **and** a broader trace shows materially higher exact B4 hit frequency. Near-equal sets are a different mechanism and would require a new exact union/intersection design; they must not be treated as exact-set hits.
