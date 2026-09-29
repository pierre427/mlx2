# north

Artifact: `North-Mini-Code-1.0-mlx-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `c3d6524961be73684789c25558e9d1a8f0bb47e14c6a4adcc8c2dac48b2688d8`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **20**.
Median aggregate generated rate across rounds: **221.9 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `054d5e8fe712271a12fae367f1ddabad47edd98b`; artifact config SHA-256: `c3d6524961be73684789c25558e9d1a8f0bb47e14c6a4adcc8c2dac48b2688d8`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `c3d6524961be73684789c25558e9d1a8f0bb47e14c6a4adcc8c2dac48b2688d8`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **8**.
Median aggregate generated rate across rounds: **82.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `c3d6524961be73684789c25558e9d1a8f0bb47e14c6a4adcc8c2dac48b2688d8`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 1.63 | 48.0 |
| 4,096 | 1 | 3 | True | 6.93 | 42.1 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature-run status: **pass**; applicable operations: 8; engaged and passing in this run: 8.
Qualified exercised operations: apc_rolling_checkpoints, block_persistence, cache_capsules, host_memory_signals, memory_preemption, pld_rotating_replay, spomin_live_compaction, srpt_prefill_scheduling.
Feature engagement and qualification do not select a production route or establish production use.
