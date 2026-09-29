# flash-next

Artifact: `Qwen3.8-Flash-Next-MLX-4bit-MTP`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; ordinary-reply peak-width field: **1**.
Median aggregate generated rate across rounds: **55.3 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Native-MTP batched target forwards during 20×20: **903**; true-batched requests: **903**. Ordinary reply width is not the native-MTP batching metric.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `8a9f9d7aed119bc4a388010e4ad9f3771b34065d`; artifact config SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`.

### Context performance

No three-repetition performance ladder in this campaign.

### Feature qualification

Core feature sweep: **partial**, 7/8 applicable checks in one combined run, APCv2 budget 16 GiB; source `d3993dc6ff4c50ee03bcdbb9b10688fe2f1a38f5`; swap-out delta 0 pages.
Earlier 8 GiB combined sweep: **partial**, 7/8 checks; rolling recovery missed as APCv2 recorded 19 pressure spills. Its isolated retry passed separately below.
The 16 GiB rerun still recorded 19 APCv2 pressure spills and only 5 cached retry tokens; the combined interaction remains open.
Combined-run open checks: apc_rolling_checkpoints.
Isolated rolling recovery: **pass**, 1 APCv2 rolling hit(s), 2048 retry cached tokens, swap-out delta 0 pages; source `e2182e5ebd37704728cfded071fe26e34c57fd67`.
FLy greedy route selected with sampled exact fallback; relaxed accepts observed: 0. Approximate relaxation has no observed-use claim from this probe and remains default-off.
Cache capsules are inapplicable: this hybrid artifact has no plain KVCache plane eligible for capsule fanout.
Current-default PLE and smoke probe: **passed**, 4016 sidecar rows read, 0 swap-out pages from before load through shutdown; source `f620742976dfad99bfee84b956aa9f3caf84c10f`.
Safe kernel sweep: **pass**, 5/5 output-equal engaged arms, swap-out delta 0 pages; source `e2182e5ebd37704728cfded071fe26e34c57fd67`. Passing arms: fused_gdn_decode, fused_gdn_verify_replay, fused_gdn_dynamic_accept, moe_router_kernel, eager_dispatch. Gate/up fusion was excluded after its separate swap failure.
Kernel rates in this sweep use one quick repetition; they do not establish a thermal performance gain or change route selection.

### TensorFold row kernel and gated speculation candidate

Opt-in profile: adapted TensorFold q4/group-64 row matvec for eligible dense projections; native MTP with a single-stream acceptance/goodput latch; prompt-copy proposals inside MTP with match-strength and yield latches. This prompt-copy path is PLD-style, not the separate `prompt_lookup` route. The full TensorFold fused Flash executor is not integrated or qualified.

The first latched ladder was interrupted after cold/warm output mismatches. An empty scheduler batch had retained the prior request's adaptive depth and goodput state. The same-source fixed-depth control matched 3/3 at 1K; after the request-boundary reset, the latched 1K gate matched 3/3. The interrupted cells are excluded from this ladder.

Thermal ladder: **failed**; source `5c3837a18040bc4ac996e679c978a033bdd0fa76`; execution policy SHA-256 `a48294b5f23fd7ea52d2fb91590e9d88e0cc0eadf46a3ceef7dc2bee81c1e372`; owned-run swap-out delta **0 pages**.

| Prompt tokens | Runs | Cell passed | Cold/warm equal | Median cold TTFT (s) | Median prefill (tokens/s) | Median decode (tokens/s) | Decode spread |
|---:|---:|:---:|:---:|---:|---:|---:|---:|
| 1,024 | 3 | yes | 3/3 | 1.48 | 692 | 74.9 | 10% |
| 4,096 | 3 | yes | 3/3 | 3.17 | 1292 | 71.9 | 4% |
| 16,384 | 3 | yes | 3/3 | 16.10 | 1018 | 55.8 | 24% |
| 32,768 | 3 | yes | 3/3 | 28.30 | 1158 | 58.3 | 30% |
| 65,536 | 3 | yes | 3/3 | 56.21 | 1166 | 56.2 | 15% |
| 131,072 | 3 | yes | 3/3 | 115.09 | 1139 | 52.6 | 13% |

Temperature-zero cold/warm equality: **18/18** measured repeats. Any mismatch keeps this candidate unqualified.
The 262,144-token cell did not pass: HTTPError: HTTP Error 429: Too Many Requests; 0 measured runs. This cell is excluded from performance medians.

Observed latch counters across the owned run: MTP depth changes 28, MTP cost probes 140, prompt-copy rounds 352, copy gate declines 66. These show exercised candidate behavior, not a production default.

#### Candidate 20×20, APCv2, and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; ordinary-reply peak-width field: **15**.
Median aggregate generated rate across rounds: **62.1 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Native-MTP batched target forwards during 20×20: **296**; true-batched requests: **296**. Ordinary reply width is not the native-MTP batching metric.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `083eea7875fdc102b0c202b2f03eb1b175c1e14d`; artifact config SHA-256: `2fe9ba742da993ffe27c68f56ddc30deff43ed5aeb07d25a82cc6381d9208d9b`.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

The earlier contaminated 20×20 remains historical. A source-bound MoE isolation run measured zero swap-out pages for the split control and expert-only dispatch, versus 389,404 pages during fused gate/up load plus 33,784 while active. This identifies the triggering option, not the underlying allocator mechanism. The current Flash default retains split gate/up projections and file-backed PLE; the current-source stress and focused feature receipts above are separate gates.
