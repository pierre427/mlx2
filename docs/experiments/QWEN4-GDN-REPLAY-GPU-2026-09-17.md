# Qwen4 compact GDN replay GPU qualification — 2026-09-17

## Verdict

The reconstruct-at-commit approach is qualified and selected by default in the
Flash-Next adapter's B=1 fused Qwen4 speculative-verify profile. Direct
low-level imports remain default-off, and snapshot plus generic replay remain
available as exact reference and fallback paths.

## Boundaries

- Device: Apple M5 Max, MLX `0.32.2.dev20260915+2a817ad94`.
- Model: `Qwen3.8-Flash-Next-MLX-4bit-MTP`, artifact fingerprint
  `77a35a9692b49e736bd0daa708d8b14c71c05abb03f81f8c707a1fc1e372bb25`.
- Serialization: exclusive CPG GPU lease plus matching `/tmp/gpu.lock` and
  `/Users/Shared/mlxuag/gpu.lock` receipts.
- Qualification did not change a live service. The later promotion changes the
  default environment of newly loaded Flash-Next adapters; no running process
  was restarted during this work.

## Correctness and engagement

The isolated Metal probe compiled snapshot verify, compact verify, and every
partial reconstruction specialization at widths 3, 4, and 8. For randomized
production-geometry inputs, output, final convolution state, final recurrent
state, and every partial reconstructed state were bit-exact.

The model-bound probe used 36 recurrent layers and 12 counterbalanced
repetitions per width. Snapshot and compact arms began from separate identical
prefills, ran a speculative verify, rejected a suffix, materialized rollback,
and ran a continuation token. Every verify logit, restored convolution state,
restored recurrent state, continuation logit, and continuation argmax matched
exactly. Every compact repetition recorded 36 replay verify calls and 36 replay
rollback calls with zero replay fallbacks.

## Performance

Medians below cover the full model. `round` is verify plus a tested partial
rollback. Peak deltas are MLX allocator observations relative to the resident
model and two prefilled caches, not process RSS.

| Verify width | Accepted | Snapshot verify | Compact verify | Verify speedup | Snapshot round | Compact round | Round speedup | Snapshot peak delta | Compact peak delta | Peak reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 31.438 ms | 30.598 ms | 1.027x | 32.317 ms | 32.585 ms | 0.992x | 345.1 MB | 118.5 MB | 2.91x |
| 4 | 2 | 34.889 ms | 33.651 ms | 1.037x | 35.666 ms | 35.532 ms | 1.004x | 463.7 MB | 122.2 MB | 3.80x |
| 8 | 4 | 47.646 ms | 45.138 ms | 1.056x | 48.646 ms | 47.299 ms | 1.028x | 931.5 MB | 136.8 MB | 6.81x |

Compact verification won at all three widths. Reconstruction adds roughly
0.9–1.1 ms over snapshot restoration. At width 3, compact replay breaks even
when partial rollback occurs in less than about 93.6% of rounds; widths 4 and 8
remain net-positive even at the tested rollback on every round. Actual serving
benefit still depends on the accepted-length distribution and scheduler mix.

## Receipts

- `qualification/experiments/qwen4-gdn-replay-20260917/kernel-microbench.json`
- `qualification/experiments/qwen4-gdn-replay-20260917/model-bound-r12.json`
- `scripts/bench_qwen4_gdn_replay_gpu.py`
- `scripts/qualify_qwen4_gdn_replay_model.py`

## Selection gate

Before selecting the route, run a serving-level A/B using the same prompt mix
and speculative policy on both arms. Require nonzero replay counters, zero
unexplained fallbacks, exact token parity for deterministic requests, and a
non-regressing distribution of end-to-end latency and peak memory. Width 3
should be evaluated against its observed partial-rollback rate rather than
selected from verify-only timing.

## Quick independent A/B confirmation

A fresh process used a different 14-token prompt and six counterbalanced
repetitions per width. Exact restored state, continuation logits, and argmax
tokens passed again; each compact arm recorded 36 replay verify calls and 36
rollback calls with zero fallbacks.

| Verify width | Verify speedup | Verify + rollback speedup | Peak reduction |
| ---: | ---: | ---: | ---: |
| 3 | 1.046x | 1.018x | 2.95x |
| 4 | 1.027x | 1.009x | 3.84x |
| 8 | 1.052x | 1.030x | 6.90x |

Receipt:
`qualification/experiments/qwen4-gdn-replay-20260917/quick-ab-r6.json`.

## Promotion

Following the independent A/B confirmation, the Flash-Next adapter profile now
sets `MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK=1`. This is a profile-scoped default:
the runtime constant remains opt-in when `qwen4_exp` is imported without the
qualified adapter environment. Existing bounded counters continue to prove
actual verify and rollback engagement and expose any fallback reason. The
selection receipt is
`qualification/experiments/qwen4-gdn-replay-20260917/selection.json`.

## Post-promotion default-profile smoke

The committed Flash-Next profile was then loaded in a fresh process under the
GPU lease. All 36 recurrent modules initialized in `compact` mode from the
adapter environment. Widths 3, 4, and 8 again produced bit-exact verify logits,
restored recurrent and convolution state, continuation logits, and argmax
tokens. Each width recorded 36 compact verify calls and 36 compact rollback
calls with zero replay fallbacks.

This smoke also exposed and fixed a qualification-harness ordering defect: the
oracle had imported `qwen4_exp` before constructing the adapter, so its
module-level default did not reflect the production load order. The runtime
path was unaffected. Receipt:
`qualification/experiments/qwen4-gdn-replay-20260917/post-promotion-smoke.json`
(SHA-256
`40c017aa61951ebad68dbb0ab52df26c2d9a946c31daa20470aa2862d44bf426`).
