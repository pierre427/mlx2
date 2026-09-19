# OpenRouter Qwen3.8 27B port review

## Scope and evidence

Independent CPU/static review of three isolated candidate ports from the same
initial mlx2 baseline. The candidates received model source and artifact
metadata, a bounded file-reading/writing broker, and no execution tool. They did
not receive the concurrently implemented production Qwen adapter. No candidate
runtime has been adopted into production and no candidate was loaded on a GPU.

First-pass source snapshots are preserved under
`/tmp/mlx2-openrouter/reviews/{glm,mimo,deepseek41}-qwen38-firstpass`.
Broker prompt/response logs remain in the corresponding `*-qwen38-logs`
directories. First-pass billing summaries are copied to
`/tmp/mlx2-openrouter/reviews/qwen-firstpass-costs.json`. These are local experiment
artifacts, not model qualification evidence.

Before executing generated tests, the reviewer read them. The reviewer harness
blocks every `mlx` import and disables pytest plugin autoload. Tensor tests were
skipped or excluded. Consequently the results below establish Python/metadata
contracts only; they say nothing about Metal execution, model numerics, actual
batched cache mutation, or serving throughput.

## Frozen first pass

| Candidate | Provider-reported cost | Responses | Guarded CPU test result | Disposition |
|---|---:|---:|---|---|
| GLM 5.3 Flash | $0.13106922 | 58 | 7 passed, 3 skipped | Useful partial tensor closure; ordinary route contract fails |
| MiMo V2.5 | $0.07533423 | 40, turn limit | 22 passed, 2 failed | Model import and ordinary route contract fail |
| DeepSeek V4.1 Flash | $0.18060852 | 41 | 4 passed, 1 failed, 2 tensor tests excluded | Best ordinary cache/adapter alignment; artifact and mixed-quantization defects remain |

The total first-pass provider charge was **$0.38701197**. Response counts are not
uniform budgets: GLM received an inspection continuation; DeepSeek encountered
provider failures and continuation. Summary elapsed seconds may cover only a
continuation and must not be compared as total task latency. MiMo reached its
turn limit without final prose.

### GLM

Implemented a substantial tensor model closure and adapter with no legacy
runtime import. However, its descriptor omits APCv2, layered cache, streaming,
tools and reasoning capabilities required by the ordinary qualification profile.
A reviewer-created all-pass *synthetic* receipt therefore still fails capability
validation. This is a test of the gate, not a real qualification receipt.

MTP discovery recognizes `mtp.*` but misses actual `language_model.mtp.*`
artifact keys. Both available embedded-head artifacts are incorrectly reported
headless. The code also retains a MoE compiled-decode claim on the dense model,
adds an unqualified rotating-cache option, and labels the MIT source Apache-2.0.
Three tests allocate tensors after an optional MLX import; the reviewer guard
skips them. One of those tests contradicts its own head-present fixture.

### DeepSeek

Implemented the most coherent ordinary lifecycle of the three, including the
model-derived layout and required APCv2/layered capabilities. It removed the
irrelevant compiled-decode claim and accurately said its own tests were unrun.

Its universal headless-artifact assumption fails against the supplied metadata:
two complete artifacts contain 29 native MTP tensors each. Its quantization
predicate ignores per-module overrides: the oQ4e artifacts contain 166 5-bit
overrides under a 4-bit global default. That changes the graph's packed-weight
geometry. Provenance mislabels the MIT source Apache-2.0. Two tensor tests do not
force CPU execution and were excluded from this review run.

### MiMo

Produced substantial metadata tests and appended model classes, but imports
`Qwen3NextAttention` from a baseline module that does not define that symbol.
The adapter also lacks `layout`, and the descriptor omits mandatory APCv2 and
layered-cache capabilities. The latter defect is reproduced by two of its own
CPU tests.

It silently answers an MTP profile request with ordinary decode, misses both
embedded-head artifacts, ignores mixed quantization overrides, retains a MoE
compiled-decode claim, and mislabels the MIT source Apache-2.0. Documentation
claims passing tests despite having no execution tool; those claims are
unsupported and conflict with the review results. Its constructor description
also says no GPU execution despite calling `mx.eval` on model parameters.

### Shared remaining work

All candidates need broader artifact validation before model allocation and a
truthful split between implemented capability and qualified route. None delivers
the generic plain-KV segmented batching mechanism required for these dense
hybrid models' modern batched MTP path. The production port developed that
mechanism separately and verified it with CPU state/attention oracles.

All real model routes remain unqualified in this experiment. No candidate
created a fake qualification receipt. Bounded repair instructions are preserved
in `/tmp/mlx2-openrouter/feedback-{glm,mimo,deepseek41}-qwen38.txt`.

## Bounded repair pass

First-pass results above remain unchanged. Root supplied concrete reviewer
feedback and allowed one additional bounded repair pass; no further repair was
requested by the reviewer.

| Candidate | Cumulative provider charge | Added charge | Frozen repair outcome |
|---|---:|---:|---|
| GLM | $0.24447841 | $0.11340919 | Two files changed; still blocked, 5 passed/2 failed/3 skipped |
| MiMo | $0.12725239 | $0.05191816 | Zero source/doc/test changes; all first-pass defects remain |
| DeepSeek | $0.28569186 | $0.10508334 | Most feedback addressed; 11 passed/1 error/2 skipped; useful ordinary candidate |

GLM changed its tensor model and shared descriptor. It removed the irrelevant
compiled-decode claim and approximate rotating-cache option, and corrected the
model file's MIT header. It did not update the adapter, tests or provenance/docs.
The descriptor now declares ordinary required capabilities, but its new layout
`qwen38-hybrid-layer-segments-v1` disagrees with the adapter's
`qwen3_8-hybrid-gdn-kv-v1`. MTP is declared without implementing the promised
per-artifact reconciliation. Embedded-head discovery remains wrong. Two old
descriptor tests now fail, so the patch is internally unfinished.

MiMo used its additional 20 responses to inspect files and metadata. A recursive
comparison against the preserved first pass finds no source, documentation or
test change. Re-running identical tests would add no evidence; the first-pass
22-pass/2-fail result remains applicable.

DeepSeek addressed the central review findings: native MTP artifact classes,
mixed quantization overrides, MIT provenance, strict topology/shard checks and
CPU-forced tensor fixtures. The validator accepts all eight supplied full-model
metadata records. It accurately documents the missing generic plain-KV segmented
builder and common registry integration, and explicitly rejects MTP serving.
It is the strongest repaired Qwen candidate: a useful ordinary-port slice,
although it does not pass the full task scope. Its own test suite has a collection
error: imported `tested_shard_files` matches pytest's test naming rule and is run
without the required `path` fixture. The module-local collection hook does not
deselect tensor tests as claimed; the reviewer import guard skipped them, and
the new fixture would force CPU if enabled. MTP capability is still a
family-level declaration rather than bound to the individual artifact.

Frozen repaired snapshots and billing summaries are in
`/tmp/mlx2-openrouter/reviews/{glm,mimo,deepseek41}-qwen38-repaired`. Total
first-pass plus repair charge is **$0.65742266**, of which **$0.27041069** was
the repair increment. No candidate code was adopted. Native runtime/GPU
qualification was outside scope; DeepSeek's remaining unimplemented segmented
MTP and registry work are additional to that expected qualification requirement.

## Cost effectiveness

First-pass review ran approximately **21:35–21:46 UTC** on 2026-09-15, an
11-minute wall interval for all three Qwen candidates. This includes reading,
guarded tests, feedback, coordination and one unrelated production adapter fix;
it is not a measurement of active model compute. The independently completed
production Qwen port took roughly **21:14–21:33 UTC**, about 19 minutes, with
broader scope: actual artifact inspection, generic segmented KV implementation,
tokenizer validation and peer review. It had execution access unavailable to the
external candidates. These elapsed intervals include parallel work and are not
billable token or compute measurements. The reviewer already knew the
independently completed production port, which made review easier; the external
candidates had restricted snapshot access and no execution feedback, which made
their task harder. These are not controlled equal-work timing measurements.

Bounded repair review occupied approximately **21:50:33–21:51:44** for GLM/MiMo
and **21:55:47–21:56:36** for DeepSeek, about two further minutes of elapsed
review intervals. Provider waits and report drafting are separate. The combined
first-pass and repair review intervals are approximately 13 minutes; they are
still not a billable compute measurement or an equal-scope comparison.

Repair review, integration and remaining fixes add effort beyond that first
review interval. These durations cannot establish a dollar saving.
No external candidate code has yet saved production implementation work in this
run because the production port was completed independently. The experiment
shows potential value in mechanical tensor-class mining and test scaffolding;
artifact selection, capability contracts, mixed quantization, cache semantics
and qualification still required detailed frontier review.

For a future single-candidate workflow, let `D` be the cost of doing the complete
task directly here, `E` the external model charge, and `R` frontier review,
repair and integration cost. Offloading saves money only if `E + R < D`.
Equivalently the allowed review fraction is `R / D < 1 - E / D`.
Local frontier token usage and dollar cost were not available, so assigning a
dollar saving or treating elapsed minutes as billable compute would be invented.
Using all three candidates adds selection and review cost; the observed $0.387
is the experiment's first-pass charge ($0.657 after repair), not the cost of one finished production
port.

One-time broker construction/debugging should be recorded separately from
recurring work. Provider failures, retries, supervision, review and repair are
recurring workflow costs. The billing table includes provider-reported charges;
it does not monetize either category of frontier effort.
