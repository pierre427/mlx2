# OpenRouter coding pilot: quality and total cost

2026-09-15. **API spend: $1.21204162. No generated candidate adopted.**

## Decision

This pilot does not demonstrate end-to-end savings for broad model ports. The
API calls were inexpensive; architectural review, repair, and integration still
required substantial frontier work, and none of six candidates completed the
requested modern serving contract after one bounded repair round. DeepSeek
produced the strongest partial Qwen ordinary-decode port; GLM produced the
strongest partial Muse port. These are assessments of these patches, not general
model rankings.

Use cheap agents next for narrow, mechanically checkable work with complete
inputs and an executable CPU acceptance suite. A single selected candidate and
one bounded repair round would be a more useful production experiment than
paying to review three incomplete implementations of each task. Do not delegate
new cache-state semantics, capability qualification, or native tool protocols
without stronger acceptance fixtures.

## Actual provider charges

| Model | Qwen3.8 27B | Muse Glimmer | Combined |
|---|---:|---:|---:|
| GLM-5.3-Flash | $0.24448 | $0.16805 | $0.41253 |
| MiMo-V2.5 | $0.12725 | $0.20186 | $0.32912 |
| DeepSeek V4.1 Flash | $0.28569 | $0.18116 | $0.46685 |

The six main runs cost $1.20849110, including repair. Setup failures and the
authentication smoke add $0.00355052. The [ledger](openrouter-cost-ledger.json)
deduplicates response IDs and sums provider `usage.cost`, including cached-input
billing. Unbilled HTTP failures are not assigned invented charges.

Live provider prices were checked before dispatch. GLM routes used $0.10/M input;
MiMo used providers between $0.119 and $0.14/M input. DeepSeek's cheapest route was
$0.15/M input, exactly the original ceiling rather than below it; the separately
named request was treated as an exception and disclosed before dispatch. Price
caps prevented more expensive fallback routes. Output prices and caching also
matter; input price alone does not predict a completed task's cost.

Official provider pages: [GLM](https://openrouter.ai/z-ai/glm-5.3-flash),
[MiMo](https://openrouter.ai/xiaomi/mimo-v2.5/providers),
[DeepSeek](https://openrouter.ai/deepseek/deepseek-v4.1-flash), and
[price-cap routing](https://openrouter.ai/docs/guides/routing/provider-selection#max-price).
These are the experiment's checked prices, not a promise about future prices.

## Quality after repair

| Model | Qwen result | Muse result |
|---|---|---|
| GLM | Partial repairs; descriptor/adapter contract and artifact claims still inconsistent | Substantial improvement; tool support, qualification contract and chunk-safe parsing incomplete |
| MiMo | No source changes during the bounded repair pass; original blockers remain | Adapter imports a nonexistent parser; reasoning/header leakage and invented tool grammar |
| DeepSeek | Useful ordinary port, corrected artifact/MTP metadata and quantization; native MTP and segmented integration still missing | Repair adds a fatal shape constraint that rejects the supplied actual model config |

Detailed independent reviews: [Qwen](openrouter-qwen-review.md) and
[Muse](openrouter-muse-review.md). Passing generated-test counts did not establish
correctness: independent probes found startup failures and protocol defects
those tests missed. No external candidate was GPU-tested, qualified, or deployed.

## Review-inclusive economics

Compare equivalent finished outcomes:

```
direct = frontier implementation + normal review and verification
hybrid = external API + orchestration + frontier review, repair,
         integration and verification + amortized harness setup
```

Offloading saves money only when `hybrid < direct`. Normal review exists in both
workflows; the relevant penalty is the additional review/correction caused by
offloading. Frontier token costs for these tasks were unavailable, so no dollar
saving or hourly rate is inferred. A subscription may have little marginal cash
cost while still consuming scarce usage and operator time.

Review timestamps show approximately 13 minutes of Qwen review intervals and
10m40s of Muse review slices across the three candidates each. These were
parallel agent wall-time intervals, not human billable hours, active compute,
or a sum to be presented as end-to-end latency. They exclude some orchestration,
reporting, and provider waits; substantial implementation remained afterward.
The independent local ports took roughly 19 and 20m40s respectively, but had
broader scope and better inputs. These are not matched timing controls.

One-time broker construction/debugging should be amortized separately. Provider
failures, repeated inspection, incomplete repairs and integration review recur.
No production implementation effort was saved in this run because the local
ports were independently completed as the comparison baseline.

## Experimental limits and next acceptance gate

The external agents used the same isolated mlx2 baseline and selected unified
source/metadata. Their tools allowed bounded file access, edits, AST checks and
radio posts, but no test execution. They lacked the real Muse tokenizer/chat
template and complete ATEM fixture. Local agents had those inputs, execution
access and knowledge from their completed ports when reviewing candidates.
Provider/quantization choices and retry histories also differed. These limitations
prevent treating the result as a controlled comparison of intrinsic model ability.
Inventing unsupported protocols or claiming unexecuted tests passed remains a
candidate defect even under those limitations.

A fair follow-up should provide identical complete fixtures, a guarded CPU test
runner, frozen acceptance tests, one clearly bounded task, and a frontier-only
control. Record external billing, frontier input/output usage where available,
review/repair rounds, final acceptance and elapsed time separately. Require an
accepted patch with less total frontier work before expanding offload scope.

## Durable evidence

Broker scripts, isolated candidate trees, immutable first-pass and repaired
snapshots, prompts, feedback, provider receipts, and reviewer probes are archived
at `~/Library/Application Support/mlx2/experiments/openrouter-2026-09-15/`.
References to `/tmp/mlx2-openrouter/` in the detailed reports map to this archive
with the same relative paths. Credentials were read only by the broker and were
not made available to model tools or copied into the archive.
