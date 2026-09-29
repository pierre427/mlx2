# thinkingcap-27b

Artifact: `ThinkingCap-Qwen3.8-27B`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **16.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6c6e3887e608d1afbf16cb61ca79f9a20c38df1`; artifact config SHA-256: `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.
