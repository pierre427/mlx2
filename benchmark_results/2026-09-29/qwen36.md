# qwen36

Artifact: `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `cb39a2ee10670cd7536cb2c629d60d41bf4857c5aa880213c17bbd2a06925fad`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **109.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `cb39a2ee10670cd7536cb2c629d60d41bf4857c5aa880213c17bbd2a06925fad`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 0.42 | 145.6 |
| 1,024 | 4 | 3 | True | 1.53 | 54.9 |
| 4,096 | 1 | 3 | True | 1.03 | 143.9 |
| 4,096 | 4 | 3 | True | 4.36 | 47.7 |
| 16,384 | 1 | 3 | True | 3.70 | 128.9 |
| 16,384 | 4 | 3 | True | 19.94 | 28.1 |
| 32,768 | 1 | 3 | True | 14.52 | 103.7 |
| 32,768 | 4 | 3 | True | 37.78 | 27.0 |
| 65,536 | 1 | 3 | True | 33.55 | 79.7 |
| 131,072 | 1 | 3 | True | 91.05 | 49.0 |
| 262,144 | 1 | 3 | True | 266.58 | 32.3 |
Source commit: `0ecbf804441d4d9da22c4f3837f2bc314890e581`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `cb39a2ee10670cd7536cb2c629d60d41bf4857c5aa880213c17bbd2a06925fad`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **41.9 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `cb39a2ee10670cd7536cb2c629d60d41bf4857c5aa880213c17bbd2a06925fad`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 1.74 | 61.7 |
| 4,096 | 1 | 3 | True | 5.81 | 57.7 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature-run status: **partial**; applicable operations: 14; engaged and passing in this run: 6.
Passing observations in a partial/contaminated run; not promoted by this report: apc_junction_snapshots, block_persistence, cache_capsules, host_memory_signals, memory_preemption, self_mtp_copy_draft.
Open or inconclusive operations: apc_rolling_checkpoints, fly_verification, srpt_prefill_scheduling.
Feature engagement and qualification do not select a production route or establish production use.
