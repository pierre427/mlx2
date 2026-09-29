# qwen36-27b-heretic-4bit

Artifact: `Qwen3.6-27B-Abliterated-Heretic-Uncensored-MLX-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `02895cd1ac8eaff1ba9727548f61db21581298b2c68333b12574b01bef49a8b0`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **39.0 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `4f8abd3d125069717b3820ef2931a95e2989ae87`; artifact config SHA-256: `02895cd1ac8eaff1ba9727548f61db21581298b2c68333b12574b01bef49a8b0`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `02895cd1ac8eaff1ba9727548f61db21581298b2c68333b12574b01bef49a8b0`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **11.4 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `02895cd1ac8eaff1ba9727548f61db21581298b2c68333b12574b01bef49a8b0`.

### Context performance

Three-repetition ladder: **partial or interrupted**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 10.40 | 7.5 |
| 4,096 | 1 | 1 | False | 40.22 | 7.4 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.
Separate capped 4K retry: **passed** 3 measured runs at 7.43 median decode tokens/s/stream; 4 GiB APCv2 cache and 8K serving context.

### Feature qualification

Feature-run status: **partial**; applicable operations: 8; engaged and passing in this run: 4.
Passing observations in a partial/contaminated run; not promoted by this report: block_persistence, cache_capsules, host_memory_signals, memory_preemption.
Open or inconclusive operations: apc_junction_snapshots, apc_rolling_checkpoints, srpt_prefill_scheduling.
Feature engagement and qualification do not select a production route or establish production use.

## Interpretation

The standard-cap M3 width-one ladder was partial. A separate 4 GiB cache / 8K context retry passed three measured 4K single-stream runs; it does not turn the standard-cap ladder into a pass.
