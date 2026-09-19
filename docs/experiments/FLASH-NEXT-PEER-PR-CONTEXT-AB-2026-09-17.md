# Flash-Next peer-PR context A/B — 2026-09-17

## Scope

This is an exploratory, source-frozen, one-run-per-cell comparison. It is not a
statistical performance qualification.

- **A:** current advanced Flash-Next stack with the three peer-PR-derived paths
  restored from revision `4036baa816b7c95ee454c64a26653d7f5ec31eae`
- **B:** the same stack with the peer-PR hardening retained
- Differing paths:
  `src/mlx2/runtime/generate.py`,
  `src/mlx2/adapters/qwen38_memory.py`, and
  `src/mlx2/runtime/cache_planes.py`
- A source SHA-256:
  `b8def8bd9cef20c5d518c6dd010ebd469503a52859df42d73a6890b8c0093e3c`
- B source SHA-256:
  `3d57ce6ee8f0a40261cb62f42ef4bbf60293823f4fb91e64d5fea06cf749b32c`
- Both source-bound route qualification receipts passed before the ladder.
- The context order alternated A/B and each accepted cell used a fresh server
  and fresh cell cache.

## Results

All 12 cells passed their applicable route, cache, counter, and source gates.
`Request tok/s` is `64 / server elapsed seconds`; it includes warm TTFT and is
not a steady-state decode-only rate.

| Context | A route | B route | Cold TTFT A / B (s) | Warm TTFT A / B (s) | Request tok/s A / B | B wall delta | Peak A / B (GB) |
|---:|---|---|---:|---:|---:|---:|---:|
| 2,048 | continuous self-MTP | continuous self-MTP | 1.470 / 1.324 | 0.429 / 0.493 | 48.70 / 46.12 | +5.6% | 76.335 / 76.335 |
| 8,192 | continuous self-MTP | continuous self-MTP | 4.877 / 4.850 | 0.486 / 0.438 | 39.85 / 41.14 | -3.2% | 77.408 / 77.408 |
| 32,768 | continuous self-MTP | continuous self-MTP | 20.413 / 20.883 | 0.546 / 0.564 | 39.28 / 37.46 | +4.8% | 79.974 / 79.975 |
| 65,536 | continuous self-MTP | continuous self-MTP | 68.327 / 61.975 | 0.602 / 0.498 | 33.88 / 35.88 | -5.6% | 84.119 / 84.115 |
| 131,072 | continuous self-MTP | continuous self-MTP | 127.106 / 125.559 | 0.691 / 0.846 | 20.32 / 20.49 | -0.8% | 89.107 / 90.722 |
| 262,016 | ordinary | segmented self-MTP | 326.501 / 240.089 | 3.160 / 2.145 | 9.88 / 14.02 | **-29.6%** | 96.431 / 96.432 |

At 262,016 tokens, A's memory admission fell back to ordinary decode. B kept
segmented self-MTP selected, proposed 53 draft tokens, accepted 36
(`67.9%`), and completed the 64-token measured request `29.6%` faster. Its cold
TTFT was `26.5%` lower and its request-level completion throughput was `41.9%`
higher, with effectively unchanged peak Metal allocation (`+0.0005 GB`).

Below 262K the sign and size of the single-run deltas vary. The supported
finding is therefore no obvious broad regression plus a concrete 262K route
admission improvement; the ladder does not support a general speedup claim.

## Evidence

- Final ladder:
  `qualification/runs/flash-next-peer-pr-ab-20260917/context-ladder-final.json`
- A route qualification:
  `qualification/runs/flash-next-peer-pr-ab-20260917/route-A.json`
- B route qualification:
  `qualification/runs/flash-next-peer-pr-ab-20260917/route-B.json`
- Frozen manifest and source audit:
  `qualification/runs/flash-next-peer-pr-ab-20260917/context-ab.generated.json`
- Rejected attempts are retained beside the final report. The final rejected
  attempt was caused by a conservative gate expecting ordinary fallback for B;
  its receipt instead showed engaged segmented self-MTP. The corrected gate
  required that observed B-only mechanism and reran the cell from a fresh
  server/cache.
