# laguna

Artifact: `ba635f7386219675b9b71acdc2812d12fa69aab2`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `1437abc6957a0d5e4a729f31ca5dcaa28d795cae0aad241893691e55834c3d17`.

### 20×20 domain and batching

20×20 gate: **error**.
Graded correct: **377/400**; HTTP errors: **0**; observed peak batch width: **8**.
Median aggregate generated rate across rounds: **195.1 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Issue counts: empty=24.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **—**.
Source commit: `b0262be59d470ce01e57ce2c8596b79d5754ac80`; artifact config SHA-256: `1437abc6957a0d5e4a729f31ca5dcaa28d795cae0aad241893691e55834c3d17`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

Default reasoning exhausted the answer allowance in many requests. A guarded candidate scored 400/400 but leaked `</think>` and was not promoted.
