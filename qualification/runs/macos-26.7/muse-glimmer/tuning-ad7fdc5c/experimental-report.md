# Muse Glimmer DFlash tuning (experimental)

Runtime source: `ad7fdc5caa4fda337258b2cec93a9c4dedc65de8cf3afa0f971a07259ce6b760`

This evidence is experimental and does not change route qualification or canonical selection.

## B1 serialized comparison (three valid unseeded runs per arm)

| Arm | Run tok/s | Mean | Median | Range | TTFT mean | Acceptance | External rounds | Rollbacks | Width |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 32.293 / 32.250 / 25.392 | 29.978 | 32.250 | 25.392-32.293 | 0.0871s | n/a | n/a | n/a | 1 |
| num_draft=2 | 41.877 / 41.855 / 30.450 | 38.061 | 41.855 | 30.450-41.877 | 0.0667s | 0.6996 | 270/270/275 | 105/105/106 | 1/1 |
| num_draft=3 | 38.686 / 45.936 / 45.840 | 43.487 | 45.840 | 38.686-45.936 | 0.0668s | 0.6060 | 238/221/221 | 140/132/133 | 1/1 |

All valid B1 brackets recorded thermal state 0 before and after, swap 0, four APCv2 paired-cache resumes for every DFlash run, and zero draft fallbacks.

## B4 single-run exploratory comparison (seed 424242)

| Arm | Aggregate tok/s | Acceptance | Receipt width | External rounds | Rollbacks |
|---|---:|---:|---:|---:|---:|
| ordinary | 104.433 | n/a | 4 | 0 | 0 |
| num_draft=2 | 48.518 | 0.6965 | 4/3 | 168 | 108 |
| num_draft=3 | 34.525 | 0.5838 | 4/4 | 146 | 141 |

These B4 cells have one run each and remain exploratory. Earlier unseeded DFlash B4 runs measured 48.614 tok/s for num_draft=2 and 49.095 tok/s for num_draft=3.

## Semantics and exclusions

- `max_lanes=1` forces both target and draft B1; it does not isolate draft width while preserving target B4.
- There is no current serving control for draft B1 with target B4.
- Long-context reuse was skipped because the prior qualification cache had no safely reusable persisted state.
- `b1-r2-ordinary` was excluded after the thermal watchdog entered state 1 and deliberately stopped the server.
- `b1-r2-valid-nd3` was excluded because thermal preflight failed before server start.

Exact paths and hashes are recorded in `experimental-report.json`.
