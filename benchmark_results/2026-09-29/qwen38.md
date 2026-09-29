# qwen38

Artifact: `Qwen3.8-27B-oQ4e-mtp`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `7a063b70e128d238784dce2c594cb7e0f714f44c4387975c99acc8fcf6511eb5`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **36.0 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `7a063b70e128d238784dce2c594cb7e0f714f44c4387975c99acc8fcf6511eb5`.

### Context performance

Three-repetition ladder: **partial or interrupted**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 1.28 | 51.6 |
| 1,024 | 4 | 3 | True | 6.60 | 20.2 |
| 4,096 | 1 | 3 | True | 4.38 | 49.9 |
| 4,096 | 4 | 3 | True | 22.50 | 15.9 |
| 16,384 | 1 | 3 | True | 23.76 | 52.3 |
| 16,384 | 4 | 3 | True | 65.32 | 13.8 |
| 32,768 | 1 | 3 | True | 47.44 | 45.8 |
| 32,768 | 4 | 3 | False | 125.52 | 12.0 |
| 65,536 | 1 | 3 | True | 109.93 | 32.5 |
| 131,072 | 1 | 3 | True | 272.63 | 26.7 |
Source commit: `b84d2096b10b266dcaa1f8566b0b4c2f2ef2984e`; owned-run swap-out delta: 0 pages.
The unmeasured 262K warmup is excluded from the table; the first measured request received HTTP 429.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

The M5 32K four-stream cell missed warm APCv2 reuse; measured 262K was refused by memory admission after a successful unmeasured warmup.
