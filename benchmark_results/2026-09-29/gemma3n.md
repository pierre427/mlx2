# gemma3n

Artifact: `5e092ebca197cdcd8d8b195040accf22693501bc`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `11c9aa9516fdfc9d16bf0d5f8a0ff4c53d3789aad479760d489714b0c48db3cb`.

### 20×20 domain and batching

20×20 gate: **error**.
Graded correct: **293/400**; HTTP errors: **60**; observed peak batch width: **8**.
Median aggregate generated rate across rounds: **169.9 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Issue counts: char_run=2, http_400=60.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **—**.
Source commit: `028d70972034fc849a79e92f17767547a09819b1`; artifact config SHA-256: `11c9aa9516fdfc9d16bf0d5f8a0ff4c53d3789aad479760d489714b0c48db3cb`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

Sixty grammar/tool requests were correctly refused as unsupported; supported text tasks also missed quality checks. Media needs its own workload.
