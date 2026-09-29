# muse-original

Artifact: `Muse-Glimmer-30B`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **398/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **23.4 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `68d9c8c050efa7daa433d444f00a2c50da0711d6`; artifact config SHA-256: `5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.
