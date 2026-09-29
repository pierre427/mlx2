# flash-next

Artifact: `Qwen3.8-Flash-Next-MLX-4bit-MTP`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`.

### 20×20 domain and batching

20×20 gate: **contaminated**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **1**.
Median aggregate generated rate across rounds: **54.6 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **277732 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **False**.
Source commit: `b083fa2bfe841ebfb100388a1fb840832a82e6af`; artifact config SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

File-backed PLE was observed in clean smoke, but the full 20×20 run swapped during load; the stress gate remains contaminated. The harness sampled width one while scheduler counters reported multi-request cycles; that batching discrepancy needs review.
