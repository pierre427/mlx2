# qwen38-mlx-6bit

Artifact: `Qwen3.8-27B-MLX-6bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `f2490d237e42fc8418ee62b7416e360785b7de308f43d8f794ccf4f185bdeec9`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **38.5 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `6fe82d590cd0a488238e02325383fbba210695fc`; artifact config SHA-256: `f2490d237e42fc8418ee62b7416e360785b7de308f43d8f794ccf4f185bdeec9`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.
