# xing-bf16

Artifact: `Xing4.0-29B-A4B-mlx-bf16`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp1`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `82f99bc65dc682e51574da5ea7f0c6c52f0816050ae09a954eb448fc76bc22e8`.

### 20×20 domain and batching

20×20 gate: **contaminated**.
Graded correct: **392/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **61.5 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Issue counts: empty=11.
Owned-run swap-out delta: **17680 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **—**.
Source commit: `225f7a3050870becf3c1e5e48008fe60bd3ab8df`; artifact config SHA-256: `82f99bc65dc682e51574da5ea7f0c6c52f0816050ae09a954eb448fc76bc22e8`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

Eleven empty answers and swap growth kept the quality and memory gates open.
