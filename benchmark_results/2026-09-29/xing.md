# xing

Artifact: `Xing4.0-29B-A4B-mlx-6bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp1`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `c5efd39a33ad34f63684baffaceacb6d93c3416e60a5d6e83c196855ff5e23a7`.

### 20×20 domain and batching

20×20 gate: **error**.
Graded correct: **386/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **87.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Issue counts: empty=15.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **—**.
Source commit: `054d5e8fe712271a12fae367f1ddabad47edd98b`; artifact config SHA-256: `c5efd39a33ad34f63684baffaceacb6d93c3416e60a5d6e83c196855ff5e23a7`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

Fifteen empty final answers kept the default-route quality gate open.
