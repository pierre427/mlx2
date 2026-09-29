# qwen36-heretic-4bit

Artifact: `Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `cfb9d34208e818c1a2d66ec54132f624d6a2f13563cbee4059933339d07af080`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **129.9 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e4ed6c4bfe32b6a6e417d8e5d6ff865435789be9`; artifact config SHA-256: `cfb9d34208e818c1a2d66ec54132f624d6a2f13563cbee4059933339d07af080`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.
