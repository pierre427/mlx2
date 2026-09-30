# qwen38-mlx-4bit

Artifact: `Qwen3.8-27B-MLX-4bit`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **46.8 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `39e5b4597137a643fb47d43fe6f9a3e5d37fbcd0`; artifact config SHA-256: `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Feature qualification was not run on this model and host.

## M3 Pro, 36 GB

Served smoke: **passed**; default route: `ordinary`; source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **4**.
Median aggregate generated rate across rounds: **13.7 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85`.

### Context performance

Three-repetition ladder: **passed**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 10.06 | 8.6 |
| 4,096 | 1 | 3 | True | 39.84 | 8.4 |
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; owned-run swap-out delta: 0 pages.

### Feature qualification

Feature-run status: **partial**; applicable operations: 8; engaged and passing in this run: 5.
Passing observations in a partial/contaminated run; not promoted by this report: apc_junction_snapshots, block_persistence, cache_capsules, host_memory_signals, memory_preemption.
Open or inconclusive operations: apc_rolling_checkpoints, srpt_prefill_scheduling.
Feature engagement and qualification do not select a production route or establish production use.

## Current-main control

This checkpoint was also used for the M5 single-prompt [mlx2, TensorFold, and omlx comparison](upstream-main-comparison.md). That later control has its own source revisions, settings, and thermal receipts; it is separate from this campaign's M3 ladder and 20×20 workload.

## M3 current-source rerun, 2026-09-30

On source `37f986a4`, the M3 served smoke passed with the ordinary default route and the same artifact config SHA-256 listed above. A four-lane, 4 GiB-cache 20×20 passed **400/400**, with zero HTTP errors or graded issues, one recovered 429 retry, observed peak batch width four, APCv2 repeated-prefix reuse, and zero swapouts. Median aggregate generated rate was **13.8 tokens/s** over the 20 mixed-workload rounds. This is source-bound domain throughput, not single-stream decode speed. The smoke and stress receipt SHA-256 values are `83e33de304f9a4bb4bbe2d6cc95ae54d2e49014aa66cfeb18a58efab2df309ed` and `9a5c6a6b213763bdefbf98bb81cbd9b37c25d024284ba080bf0c2ba1d6933a41`.

The basic M3 context ladder on this source remains pending. The earlier 1K and 4K numbers above remain tied to their older source.
