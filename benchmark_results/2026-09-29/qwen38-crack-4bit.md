# qwen38-crack-4bit

Artifact: `Qwen3.8-27B-CRACK-MLX-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `2f6574591e1fc469c170b4e862420fdb0936df50ed761566703fc16d19edd718`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **46.4 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `5de35615e7c63eba107edb72109b021111b95caa`; artifact config SHA-256: `2f6574591e1fc469c170b4e862420fdb0936df50ed761566703fc16d19edd718`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `2f6574591e1fc469c170b4e862420fdb0936df50ed761566703fc16d19edd718`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **14.1 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `2f6574591e1fc469c170b4e862420fdb0936df50ed761566703fc16d19edd718`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 10.13 | 8.6 |
| 4,096 | 1 | 3 | True | 40.04 | 8.4 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature-run status: **partial**; applicable operations: 8; engaged and passing in this run: 2.
Passing observations in a partial/contaminated run; not promoted by this report: block_persistence, host_memory_signals.
Open or inconclusive operations: apc_junction_snapshots, apc_rolling_checkpoints, cache_capsules, memory_preemption, srpt_prefill_scheduling.
Feature engagement and qualification do not select a production route or establish production use.

## Interpretation

The M3 gdn_core candidate differed from its base on one open-ended prompt; it is not selected.
