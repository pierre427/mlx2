# qwen38

Artifact: `Qwen3.8-27B-oQ4e-mtp`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.

Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature.

## M5 Max, 128 GB

Served smoke: **passed**; default route: `mtp2`; source commit: `b81aaf03846cf9e22decb195d96032d0b9eddd5a`; artifact config SHA-256: `7a063b70e128d238784dce2c594cb7e0f714f44c4387975c99acc8fcf6511eb5`.

### 20×20 domain and batching

20×20 gate: **passed**.
Graded correct: **400/400**; HTTP errors: **0**; observed peak batch width: **3**.
Median aggregate generated rate across rounds: **36.0 tokens/s**. This is mixed-workload throughput, not single-stream decode speed.
Owned-run swap-out delta: **0 pages**; APCv2 repeated-prefix probe: **True**; batching engaged: **True**.
Source commit: `e6f5799c9817475bb0e268a7d5d807692eeb1ee3`; artifact config SHA-256: `7a063b70e128d238784dce2c594cb7e0f714f44c4387975c99acc8fcf6511eb5`.

### Context performance

Three-repetition ladder: **partial or interrupted**. Only completed measured cells appear below.

| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |
|---:|---:|---:|:---:|---:|---:|
| 1,024 | 1 | 3 | True | 1.28 | 51.6 |
| 1,024 | 4 | 3 | True | 6.60 | 20.2 |
| 4,096 | 1 | 3 | True | 4.38 | 49.9 |
| 4,096 | 4 | 3 | True | 22.50 | 15.9 |
| 16,384 | 1 | 3 | True | 23.76 | 52.3 |
| 16,384 | 4 | 3 | True | 65.32 | 13.8 |
| 32,768 | 1 | 3 | True | 47.44 | 45.8 |
| 32,768 | 4 | 3 | False | 125.52 | 12.0 |
| 65,536 | 1 | 3 | True | 109.93 | 32.5 |
| 131,072 | 1 | 3 | True | 272.63 | 26.7 |
Source commit: `b84d2096b10b266dcaa1f8566b0b4c2f2ef2984e`; owned-run swap-out delta: 0 pages.
The unmeasured 262K warmup is excluded from the table; the first measured request received HTTP 429.

### Feature qualification

On current private source `688ad376`, the first combined M5 run with a 4 GiB APC cache engaged seven of nine applicable feature checks. Junction snapshots published four times but had no junction hit; rolling checkpoints had no publication or retry hit. Host swapouts rose by 47,960 pages during ordinary feature work, so the entire attempt is **contaminated** and is not a qualification pass. The receipt SHA-256 is `500b944dc7b6eabd3c08c8c6182dc9c726a895e7cce8e531e3aac4a7afd160a1`.

Three fresh-server runs on the same source used an 8 GiB APC cache and a 32K context cap, with zero swapouts throughout. The seven-feature combined run passed cache capsules, block persistence, SRPT prefill scheduling, memory preemption, host memory signals, Fly verification, and self-MTP copy draft. An isolated junction run published two snapshots and recorded three junction hits. An isolated rolling run used a 300-unit prefill and 64-token slices; its cancelled request's retry reused 1,024 tokens and recorded one rolling hit. All applicable feature checks therefore have passing engagement evidence **across these three clean runs**; there is no single 9/9 combined receipt. The original 4 GiB result remains a capacity and host-swap warning for that profile, not proof of a runtime defect or a clean pass.

| Clean M5 feature receipt | Result | SHA-256 |
|---|---|---|
| Seven-feature combined, 8 GiB cache | 7/7 passed | `d9930573d6ac94b20cfc1b96a8bfe0588c132eb3fd6a52b826771cb3d6fbf653` |
| APCv2 junction snapshots, isolated | passed | `34669be6d7c598a00af5ca9c247fc0a2beffc454f60e4bf3e754b2d99a58b850` |
| APCv2 rolling recovery, isolated | passed | `060f754c470785c2808bc9cadaf51156750d0b0367e0698f1a3f6ed913170fda` |

These checks qualify feature engagement under the recorded policies. They do not select those policies as defaults, show performance benefit, or prove production use. The existing MTP2 default route and its 20×20 result remain separate source-bound evidence.

## M3 Pro, 36 GB

Model artifact not staged on this host; no load or performance verdict.

## Interpretation

The M5 32K four-stream cell missed warm APCv2 reuse; measured 262K was refused by memory admission after a successful unmeasured warmup. Those long-context gates remain open. The new feature checks do not resolve them.

A later isolated M5 32K thermal rerun on source `91c74b7f` used an 8 GiB APCv2 cap and a 32K context cap. Width one passed all three repetitions. Width four returned all 24 needle checks with no swapouts or thermal contamination, but only **3 of 4 warm lanes** reused cache on each repetition, so the APCv2 gate failed again. The missing lane varied across runs. End status reported 156 APCv2 evictions and 103 publication rejections; capacity pressure is a lead, not a proven cause. The run and ladder receipt SHA-256 values are `3ca3b5edcc7b5597ef4985c9582afea6b21e772fe4c761ac7b0ecf38bff455e7` and `0e80075c213c2c6e05d5309cdcc7edfb459fa44f78cc19505a8b6541eb4c7269`. A 16 GiB candidate has not run because another owned job held the M5 GPU lock.
