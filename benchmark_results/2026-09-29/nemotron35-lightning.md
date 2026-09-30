# Nemotron 3.5 Lightning 30B-A3B, MLX 8-bit

Artifact: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit`, config SHA-256 `a1b0135c0973322d188a836c86746e69ea07c02b241e1f26681bf5123345359b`. The target is pinned to revision `a9db86e1fe5baf448346efd33541ce117b5b8403`; the separately verified original BF16 MTP head is pinned to revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` and SHA-256 `64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0`.

**Current-source boundary:** private main gained `46139b74`, changing the Lightning Mamba time-step clamp and therefore model outputs, after the earlier runs below. Their passes and speed measurements remain source-bound historical evidence. Ordinary smoke and the capped 20×20 workload passed on newer source `9f5e525d`; its context ladder and MTP serving qualification remain pending.

## Host coverage

The M5 Max 128 GB is the qualification host for this checkpoint. The M3 Pro 36 GB received and hash-verified all 15 target files and the MTP sidecar; CPU preflight resolved the adapter and both declared routes. Its isolated checkout reached integrated private source `9699e669` with 32/32 adapter and bug-sweep CPU tests, then was advanced to post-clamp source `8e90684b` and passed 33/33. M5 ready-state measurements before and after the expert-conversion fix both allocated 33.75 GiB of Metal memory and peaked at 36.12 GiB during load. The M3 has 36 GiB total physical memory, leaving no safe room for the OS and service reserves. Its GPU load was therefore not attempted; no M3 throughput or serving qualification is claimed.

## Ordinary served smoke

On private source `bbfa0d3c`, the M5 served the ordinary route and passed the arithmetic and knowledge prompts. It reported the artifact's temperature **1.0** and top-p **0.95** defaults with no observed default mismatch. The response receipts selected ordinary decode with APCv2. This is a served candidate smoke, not a production observed-use claim.

## Status of further gates

The first M5 ordinary 20×20 attempt completed concurrency and cancellation probes, then passed 11 domain rounds without HTTP errors. Host swapouts rose by 34,296 pages during round 10, so the owned runner was stopped. Its result is **contaminated and incomplete**; concurrent artifact transfer had finished earlier, and the cause of the swap rise has not been established.

The first capped retry completed 400/400 HTTP requests, used eight-way ordinary batches for 380 replies, had no swapouts, and remained healthy. The graded gate failed at 397/400 because three answers had empty visible content. All three spent the 2,048-token thinking allowance on reasoning and ended at the token limit. This is a complete **failed quality gate**, not a qualified 20×20 pass.

The second capped retry raised the recorded thinking allowance to 4,096 tokens. It **passed 400/400 graded answers**, with zero HTTP errors, zero issues, zero swapouts, an APCv2 reuse pass, cancellation recovery, and eight-way ordinary batches on 380 replies. The median aggregate generated rate was **181.15 tokens/s** across 20 rounds (range 161.6–204.4). This qualifies the **eight-lane, 4 GiB cache** profile for this workload; it does not qualify the earlier 20-lane profile.

The first thermally controlled ladder attempt passed all three repetitions of its 1K and 4K cells, but its M5 runner ignored the requested 4 GiB cache cap and started a server with a 48 GiB cap. It was stopped at 16K and is **not** the capped-profile qualification result. The runner now applies explicit caps on either host and records the effective cap.

The corrected **1K–32K** ladder on source `7dd504ba` passed all eight cells with three thermally admitted repetitions each, all 120 needle checks correct, and zero swapouts. The table gives medians; width-four decode is per stream, not aggregate throughput.

| Context | Width | Cold TTFT | Prefill tok/s | Decode tok/s per stream |
|---:|---:|---:|---:|---:|
| 1K | 1 | 0.524 s | 1,950 | 99.1 |
| 1K | 4 | 1.105 s | 1,024 | 52.1 |
| 4K | 1 | 1.150 s | 3,560 | 98.6 |
| 4K | 4 | 3.196 s | 1,542 | 48.5 |
| 16K | 1 | 3.670 s | 4,464 | 91.5 |
| 16K | 4 | 11.464 s | 1,683 | 43.3 |
| 32K | 1 | 7.590 s | 4,317 | 84.6 |
| 32K | 4 | 26.299 s | 1,425 | 37.9 |

The integrated-source ladder on `ab212902` used the same server command, 4 GiB cache cap, and three-repetition protocol. All eight cells and 120 needle checks passed, with zero swapouts. Its medians were:

| Context | Width | Cold TTFT | Prefill tok/s | Decode tok/s per stream |
|---:|---:|---:|---:|---:|
| 1K | 1 | 0.547 s | 1,869 | 101.6 |
| 1K | 4 | 1.116 s | 1,028 | 50.5 |
| 4K | 1 | 1.168 s | 3,506 | 98.7 |
| 4K | 4 | 3.219 s | 1,545 | 45.5 |
| 16K | 1 | 7.202 s | 2,275 | 66.9 |
| 16K | 4 | 20.663 s | 925 | 30.3 |
| 32K | 1 | 13.252 s | 2,473 | 71.2 |
| 32K | 4 | 40.494 s | 929 | 28.3 |

The prior ladder began about 10 minutes after its preceding 20×20 run; the integrated-source ladder began 39 seconds after the newer 20×20 run. The thermal probe admitted every measured repetition and recorded no warning, but that unequal cooldown prevents attributing the long-context speed difference to source changes. An isolated, cooled 16K repeat is pending.

After the Mamba clamp correction, source `9f5e525d` passed ordinary smoke and the same capped 20×20 profile: **400/400** graded answers, zero HTTP errors, zero issues, zero swapouts, APCv2 reuse, cancellation recovery, and eight-way batches on 380 replies. Its median aggregate generated rate was **197.8 tokens/s** (20 rounds), versus **201.5** on `d17430de`, a 1.8% decrease. It generated 234,479 completion tokens in 1,199.69 seconds, a weighted **195.45 tokens/s**, versus 202.77 on `d17430de`, a 3.6% decrease. The profiles match, but the model's Mamba computation and outputs changed; this is an observed workload comparison, not a controlled attribution. A cooled context ladder on the corrected source is pending.

The 64K–262K width-one extension was **interrupted** after newer main commits changed Nemotron runtime math and APC/lane ownership. At its older source, the 64K cell missed one exact-code needle repetition (4/6 checks); one completed 131K repetition missed both needles. Those are real failures for that source, but the incomplete ladder does not qualify long context on the newer main. Its 8 GiB cache profile had zero swapouts. On integrated private source `d17430de`, a repeat ordinary smoke passed both prompts and all default checks with no sampling drift. The matching capped 20×20 run passed **400/400** graded answers, zero HTTP errors, zero issues, and zero swapouts. Its median aggregate generated rate was **201.5 tokens/s** across 20 rounds (range 183.0–224.3), 11.2% above the prior capped run's 181.15 tokens/s median. The prior and newer runs generated 236,786 and 237,876 completion tokens in 1,303.91 and 1,173.16 seconds, respectively. This comparison is not thermally controlled and does not isolate the cause of the change. Applicable feature gates remain pending. The MTP route remains implemented and opt-in pending served qualification. No selected speculative route is claimed.

## Receipt integrity

| Host | Receipt | SHA-256 |
|---|---|---|
| M5 | ordinary served smoke | `f69dc87cc392b8accfd21de2abc01e9f5ca54d9585eaa976484a69c522812221` |
| M5 | interrupted 20×20 attempt, contaminated | `d15d39d247587671b36e84eab87c92017e5133de3bc10ca5ec09c4e24e87b793` |
| M5 | complete 20×20 retry, 397/400 quality gate | `e7d2f56f9a85851b767d59f701585f54e39410c28b19c212b1463efa193137eb` |
| M5 | qualified capped 20×20, 400/400 | `fe5c3bdd4ca20e5e4836e4f62832997a8c096b833480a04e25c43b5a92bcad95` |
| M5 | aborted 48 GiB-cap ladder attempt | `d82bb0592d16ada8fb3a2ee7459ab97a527d818006019eae78bb0fc8937b7aed` |
| M5 | passed 1K–32K ladder, 3 reps per cell | `819b294cd992bb7de6927ae4080af68c037033805aeebb2ef02799cdef46c676` |
| M5 | interrupted 64K–262K ladder, long-context needle failures | `ebd28267c7878d1481ec88da455048edffa3dd113fb7769c041000b7494d6cff` |
| M5 | ordinary smoke on integrated main `d17430de` | `3f2489141708bf0facabf407f45659dca5391b10c00af87b1f07d9161d448914` |
| M5 | integrated-source capped 20×20, 400/400 | `07ca24f3bd89487b62b6e634b4c0de35de97439e5410ca66499f7e837280e333` |
| M5 | integrated-source passed 1K–32K ladder | `7ce1c48a3a1abf35c0b728582263ca7a14f50230d986b2f7a237c390dd3d5563` |
| M5 | ordinary smoke after Mamba clamp on `9f5e525d` | `aeb42bda9c5b9b78acfeeeadf50e605fec65c54c8fc28346623fe1a44c91ce17` |
| M5 | ordinary 20×20 after Mamba clamp, 400/400 | `6ea4b814ce2835c05427adc23bd52ada600a61f939bd94b8cf32be90fd2ed6c3` |
| M3 | hash-verified load feasibility | `119c027c459044036e7d36016576feced932a5e8e40d516876feda83db9fc1d4` |
| M3 | integrated-source CPU and memory feasibility | `0b9861b9d31ddf88516c5702f5e41d2cec74898e9810095344e349e884eee3cc` |
| M3 | post-clamp source CPU and memory feasibility | `8810035cf000a2a748bcdf68738d4fc1eb039edc1b2a8980df8a0ae58b9b05f1` |
