# qwen38-uncensored-oq4e

Artifact: `Qwen3.8-27B-Uncensored-oQ4e-fp16-mtp`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `d2c97522c56bf7bd64b7cd859f90679e0381499e203c4312b4564ad58cc32ed2`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **35.1 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6c6e3887e608d1afbf16cb61ca79f9a20c38df1`; artifact config SHA-256: `d2c97522c56bf7bd64b7cd859f90679e0381499e203c4312b4564ad58cc32ed2`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.
