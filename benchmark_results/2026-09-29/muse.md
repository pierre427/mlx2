# muse

Artifact: `Muse-Glimmer-30B-mlx-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `c7f48468db2ef9c3de4cb912be24ecc9fbed36d83f3b8386a0b224ee7ba876ca`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **47.2 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `054d5e8fe712271a12fae367f1ddabad47edd98b`; artifact config SHA-256: `c7f48468db2ef9c3de4cb912be24ecc9fbed36d83f3b8386a0b224ee7ba876ca`.

### Context performance

Three-repetition ladder: **partial or interrupted**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 1.20 | 29.9 |
| 1,024 | 4 | 3 | True | 5.83 | 14.5 |
| 4,096 | 1 | 3 | True | 6.44 | 25.2 |
| 4,096 | 4 | 3 | True | 22.46 | 14.1 |
| 16,384 | 1 | 3 | True | 26.06 | 24.8 |
| 16,384 | 4 | 3 | True | 80.18 | 13.3 |
| 32,768 | 1 | 3 | True | 47.29 | 25.3 |
| 32,768 | 4 | 3 | True | 147.69 | 12.7 |
| 65,536 | 1 | 3 | True | 100.96 | 22.4 |
Source commit: `42a6095dacb800bdf5dc91406c6f050f07ecd253`; owned-run swap-out delta: 0 pages.
The user paused the queue before the 131K run; the harness return code `-15` records interruption, not a model failure.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `c7f48468db2ef9c3de4cb912be24ecc9fbed36d83f3b8386a0b224ee7ba876ca`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **12.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `c7f48468db2ef9c3de4cb912be24ecc9fbed36d83f3b8386a0b224ee7ba876ca`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 9.98 | 8.4 |
| 4,096 | 1 | 3 | True | 40.83 | 8.3 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature-run status: **contaminated**; applicable operations: 9; engaged and passing in this run: 7.
Passing observations in a partial/contaminated run; not promoted by this report: block_persistence, cache_capsules, host_memory_signals, memory_preemption, pld_rotating_replay, spomin_live_compaction, srpt_prefill_scheduling.
Open or inconclusive operations: apc_rolling_checkpoints, fly_verification.
Feature engagement and qualification do not select a production route or establish production use.

## Interpretation

The M5 ladder was interrupted by the user's pause after nine passing cells. M3 external DFlash2 load swap was fixed, but Fly repeated-prefix checks still returned 429.
