# Nemotron 3.5 Lightning 30B-A3B, MLX 8-bit

Artifact: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit`, config SHA-256 `a1b0135c0973322d188a836c86746e69ea07c02b241e1f26681bf5123345359b`. The target is pinned to revision `a9db86e1fe5baf448346efd33541ce117b5b8403`; the separately verified original BF16 MTP head is pinned to revision `a9904d24bcc1d289a1950fa9d2b978c47cf903b9` and SHA-256 `64577b275ca4e7e5266eae0903674f7f46ec2a8cbf4f4f1a3207f80d503cd1d0`.

## Host coverage

The M5 Max 128 GB is the qualification host for this checkpoint. The M3 Pro 36 GB received and hash-verified all 15 target files and the MTP sidecar; CPU preflight resolved the adapter and both declared routes. An M5 ready-state measurement for the same artifact allocated 33.75 GiB of Metal memory and peaked at 36.12 GiB during load. The M3 has 36 GiB total physical memory, leaving no safe room for the OS and service reserves. Its GPU load was therefore not attempted; no M3 throughput or serving qualification is claimed.

## Ordinary served smoke

On private source `bbfa0d3c`, the M5 served the ordinary route and passed the arithmetic and knowledge prompts. It reported the artifact's temperature **1.0** and top-p **0.95** defaults with no observed default mismatch. The response receipts selected ordinary decode with APCv2. This is a served candidate smoke, not a production observed-use claim.

## Status of further gates

The first M5 ordinary 20×20 attempt completed concurrency and cancellation probes, then passed 11 domain rounds without HTTP errors. Host swapouts rose by 34,296 pages during round 10, so the owned runner was stopped. Its result is **contaminated and incomplete**; concurrent artifact transfer had finished earlier, and the cause of the swap rise has not been established.

The first capped retry completed 400/400 HTTP requests, used eight-way ordinary batches for 380 replies, had no swapouts, and remained healthy. The graded gate failed at 397/400 because three answers had empty visible content. All three spent the 2,048-token thinking allowance on reasoning and ended at the token limit. This is a complete **failed quality gate**, not a qualified 20×20 pass.

The second capped retry raised the recorded thinking allowance to 4,096 tokens. It **passed 400/400 graded answers**, with zero HTTP errors, zero issues, zero swapouts, an APCv2 reuse pass, cancellation recovery, and eight-way ordinary batches on 380 replies. The median aggregate generated rate was **181.15 tokens/s** across 20 rounds (range 161.6–204.4). This qualifies the **eight-lane, 4 GiB cache** profile for this workload; it does not qualify the earlier 20-lane profile.

The first thermally controlled ladder attempt passed all three repetitions of its 1K and 4K cells, but its M5 runner ignored the requested 4 GiB cache cap and started a server with a 48 GiB cap. It was stopped at 16K and is **not** the capped-profile qualification result. The runner now applies explicit caps on either host and records the effective cap; the 1K–32K retry is running. Applicable feature gates remain pending. The MTP route remains implemented and opt-in pending served qualification. No selected speculative route is claimed.

## Receipt integrity

| Host | Receipt | SHA-256 |
|---|---|---|
| M5 | ordinary served smoke | `f69dc87cc392b8accfd21de2abc01e9f5ca54d9585eaa976484a69c522812221` |
| M5 | interrupted 20×20 attempt, contaminated | `d15d39d247587671b36e84eab87c92017e5133de3bc10ca5ec09c4e24e87b793` |
| M5 | complete 20×20 retry, 397/400 quality gate | `e7d2f56f9a85851b767d59f701585f54e39410c28b19c212b1463efa193137eb` |
| M5 | qualified capped 20×20, 400/400 | `fe5c3bdd4ca20e5e4836e4f62832997a8c096b833480a04e25c43b5a92bcad95` |
| M5 | aborted 48 GiB-cap ladder attempt | `d82bb0592d16ada8fb3a2ee7459ab97a527d818006019eae78bb0fc8937b7aed` |
| M3 | hash-verified load feasibility | `119c027c459044036e7d36016576feced932a5e8e40d516876feda83db9fc1d4` |
