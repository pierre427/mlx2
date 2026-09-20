# Flash-Next optimized serving parity

## What constitutes parity

The comparison is the deployed optimized unified launcher at
`~/Library/Application Support/mlxuag-fn-unified-opt/serve.sh` (source pin
`ecf14eeb`), plus proven modern mechanisms mined from unified
`1e2bc604f71d070bee970c3e7db8b60f7855599b`. The launcher selected APCv2,
segmented batched MTP2, adaptive prefill, file-backed PLE with a 2 GiB LRU,
compiled PLE, pooled QSA, fused GDN decode/verification and eager dispatch.
It explicitly disabled whole-decode compilation, megakernels, shared suffixes
and speculative rate gating. Experimental source switches are not evidence
that a mechanism improved that serving workload.

A mechanism has four separate states:

- **Implemented**: executable code and its dependencies exist here.
- **Qualified**: evidence matches the model artifact, source, native MLX and
  execution policy. CPU/oracle evidence alone does not qualify serving.
- **Selected**: the source-bound policy makes the mechanism eligible.
- **Observed used**: a production mechanism counter or request receipt proves
  execution on the measured workload.

`/v1/status` exposes policy, environment, batch/cache counters, indexed-QSA
admission/decline reasons, PLE prefetch, fused GDN, MoE dispatch and promotion.
A process-local counter only describes that process. Re-read after restart.

## Mechanism matrix

This is an implementation/qualification handoff, **not a claim that all
candidate profiles have passed**. The deployment receipt and final reports in
`qualification/` decide which complete profile is qualified.

| Mechanism | Implemented | Candidate selection | Qualification and observed evidence |
|---|---|---|---|
| APCv2, target/MTP paired snapshots | Yes; sole cache engine | On | Lifecycle tests; HTTP cold/warm and leases required |
| Layered COW state and revision-bound transactions | Yes | On | CPU branch/rollback/oracle tests; warm cohort serving required |
| Bounded APC resident + idle disk tier | Yes | 12 GiB / 64 GiB, 180 s idle | Lifecycle spill/restore tests; status reports actual bounds |
| Segmented true-batched MTP | Yes | MTP2 default; k1/k3 selectable | Forced-acceptance tests; real batched forwards required |
| Adapter-default native MTP | Yes | No route flag on a complete MTP artifact | Pre-load metadata resolution; explicit `--native-mtp` is equivalent |
| Ordinary decode reference | Yes | `--ordinary` separate qualified profile | Full same HTTP lifecycle; independent source-bound receipt |
| Continuous batching, memory admission, adaptive prefill | Yes | On | Batch/mixed-warm/churn/cancellation; actual widths in receipt |
| File-backed PLE + 2 GiB LRU + compiled PLE | Yes | On | Table lookup/hit counters and compile diagnostics |
| Known cached-tail PLE prefetch | Yes | On | Required `ple_tail_prefetch_tables > 0` |
| Pooled QSA and scatter-chosen | Yes | On | Shared-cache oracles and model execution |
| Fused GDN decode/verify, eager dispatch | Yes | On | Actual fused counters and fallbacks in diagnostics |
| Fused MoE gate/up and expert dispatch | Yes | Gate/up on; expert auto | Actual scalar/tile4 dispatch counters; shape fallback preserved |
| Fast RMSNorm | Yes | Source width gate | Numerical/model qualification; no whole-decode compilation claim |
| Shared-prefix QSA suffix state | Yes; HTTP checkpoint attestation added | Auto, source context/output-budget crossover | Must observe `shared_qsa_batched_selections`; equal text alone never attests state |
| Asynchronous QSA physical promotion | Yes | On, output-budget gated | Required `async_qsa_promotion_engaged > 0`; cohort transitions tested |
| Indexed QSA | Yes | Auto, measured shape/context gates | 62 Metal oracle tests passed exact current MLX build; serving still requires device-attested use |
| Private-delta indexed attention | Yes | Speculative widths at >=64K; M1 remains disabled | M1–M9 exactness oracle passes; >64K serving requires actual private-delta calls |
| Two-row exact-set folding | Yes | Source proof and geometry gates | Random/stale-proof Metal tests; source enforces exact set identity |
| Native indexed output-gate epilogue | Yes | Optional, independently selectable | 8 optional Metal tests passed (native gate M1/M3, 32 seeds); full-model default decision awaits measured benefit |
| Sequential fused indexed merge | Yes | Off | Correct component, but later native pass-2 is 11–13% faster in unified evidence |
| Whole-decode compiled glue | Source experiment excluded | Off | Unified exact brackets were neutral/slower; compiled PLE remains on |
| Megakernel | Excluded from served batch profile | Off | Unified ordinary-only win involved accepted +0.234% PPL shift; self-MTP gain weak and not integrated with modern batched contract |
| NAX / split-stage QSA | Yes | Source hardware/shape auto gates; NAX decode off | Long-prefill candidate qualification must inspect native route; do not infer decode use from availability |
| Fused GDN catch-up / extra projections | Present where mined | Off | Narrow catch-up was exact but neutral/slightly slower end-to-end in unified |
| PLE vector-shift/gather-concat variants | Present | Off | No serving promotion without comparative evidence |
| Adaptive speculative rate/depth router | Not a selectable batch feature | Off | Unified batch API also explicitly rejects rate gating/windowed/per-lane adaptive depth |
| FLy relaxed verification | External exact-rejection and self-MTP greedy seams implemented | Off; execution-policy only | Approximate and unqualified; sampled/block self-MTP and all structured-output lanes remain exact |
| Quantized KV / approximate cache compaction | Components exist | Not selected | Fidelity-changing routes need independent quality, rollback and cache-publication qualification |
| Budgeted APCv2 interior checkpoints | Exact self-MTP and ordinary hybrid capture implemented; turn/tail/`auto` placement (rm04) | Off; execution-policy only (`"auto"` preset pending GPU go) | CPU tiny-Qwen4 edited-prefix reuse and warm/cold equality; incompatible routes fail startup and selection requires captured/published qualification evidence |
| APCv2 session park/resume + restart persistence | Yes; exact committed state only | Disk-tier opt-in; persistent directory separately opt-in | CPU tiny-Qwen4 park/prefetch warm decode equals cold; restart/rescan/digest/identity/lock tests |
| Additional QSA summaries serialized through APC | Present | Off | Separate derived-state persistence path needs a selected serving qualification |

## New serving integration repairs

- Hermes `options.num_ctx`, `reasoning_effort`, `think`, and
  `chat_template_kwargs.enable_thinking` normalize into explicit controls.
  Qwen effort is a thinking toggle; Muse's native effort remains a strength.
- `min_p`, logit bias, and repetition/presence/frequency penalties are passed
  into lane-specific sampling and MTP target verification. Request receipts
  record controls; model vocabulary checks prevent out-of-range token indexing.
- External DFlash2 applies lane processors to each sequential selector law with
  the same prefix used by target verification, and verifies against that masked
  proposal q. Permanent target-only fallback lanes skip draft/tap/rollback work
  and publish no stale draft sidecar; transient final-budget K=0 remains paired.
- Shared state uses a common immutable APC COW lineage **and generation**, an
  identical seed token, and a paired MTP sidecar. Ordinary equal prompt text
  cannot authorize reuse of numerically different cache state.
- Shared representation conversion creates a validated successor transaction.
  A short-output cohort cannot permanently disable later asynchronous promotion.
- An empty scheduler poll does not erase all cached prefixes. Memory-queued
  retries retire at most one oldest entry; sidecar hits update recency.
- Default-off APCv2 interior checkpoints stop hybrid prefill at deterministic
  absolute `S * 2^i` positions, snapshot the exact target/MTP pair, and publish
  with lower retention than the committed `P - 1` boundary. Snapshot bytes are
  admitted up front. A nonzero policy fails startup on external draft, PLD,
  approximate-KV, or an adapter cache without exact checkpoint capture; route
  selection also requires captured and published qualification evidence.
  Approximate state is never published. Design references: omlx#3456,
  Rapid-MLX#3435/#3463/#3351, and
  vllm#52244.
- Context is a qualified profile setting. CLI accepts the artifact's 262K
  domain; requested context and admission still must fit the tested profile and
  available memory. This does not claim 262K was successfully served.
- Metadata-only adapter resolution rejects a missing native MTP head before
  allocation. New models still require their own complete qualification.

## Memory admission geometry

The initial generic full-attention envelope charged 0.44 GiB per 1K tokens,
which denied a 131K Flash-Next request despite about 47 GiB of measured
headroom. The adapter now describes its actual 12 QSA target layers, one QSA
MTP layer, 36 recurrent layers and PLE state. It conservatively charges all
cache arrays at fp32 (normally bf16), doubles recurrent state for rollback,
includes raw/pooled QSA keys, allocation slack and token bookkeeping. The
result is about 8 GiB at 131K and 15 GiB at 262K per unprepared lane.

This is a state estimate, not a total process memory claim. The common
controller additionally retains the **20 GiB service/driver reserve** and
calibrated kernel transients. Actual retained cache bytes remain a floor;
resident caches are not double charged as new allocations. Settings bind the
complete estimator schema/dimensions to the qualification record. CPU tests
verify the bound against real tiny-model target and draft cache bytes at
aligned and unaligned lengths; root's real-model qualification must establish
the usable context/concurrency domain.

## Reproduction

Routine tests force MLX CPU before module imports. Actual Metal tests explicitly
opt in and belong in an exclusively owned GPU window.

```sh
.venv/bin/python -m pytest
MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 .venv/bin/python -m pytest tests/test_qwen4_qsa_indexed.py
.venv/bin/python -m mlx2.server --model /path/to/model --port 8285 \
  --qualification-mode --max-context 131072 --cache-dir /tmp/mlx2-parity-apc
.venv/bin/python scripts/qualify_serving.py --output qualification/candidate.json
```

`--execution-policy policy.json` makes choices reproducible and qualification
bound. Defaults live in `FlashNextPolicy`; inherited lab environment switches
are cleared. Non-default policy keys fail validation if unknown.
`--require-feature` can demand extra mechanism engagement; applicable selected
advanced mechanisms are required automatically by qualification.

The deployable qualification gate also requires observed file-backed PLE
lookups, healthy compiled-PLE hits, pooled-QSA reuse, scatter-chosen QSA,
fused GDN decode (and verification for MTP), eager dispatch, and fused MoE
dispatch whenever the source-bound environment selects them. These are
counter checks, not configuration assertions. The experiment matrix binds
each arm to that complete route receipt before accepting any ladder or batch
cell; shared-suffix QSA and asynchronous physical promotion are proved by
distinct eligible cohorts because they are alternative ownership routes.

The policy now spells out source defaults that affect tensor execution:
`MLX_GDN_PACKED=1`, `MLX_GDN_CORE=0`, scatter-chosen QSA on, fused MoE gate/up
on, and fused expert dispatch in `auto`. These match the mined parity source
and its served profile; making them explicit prevents an inherited lab
environment from changing a source-bound qualification. Startup clears every
`MLX_GDN_*` experiment before installing those two selected GDN values.

The initial kernel build gate used an explicit unverified-build override only
while testing. Deployment never accepts that override. Its exact build/header
receipt is `qualification/indexed-kernel-build.json`.

## Supporting unified evidence

- `wiki/docs/experiments/qwen4-compiled-glue-2026-09-02.md`: exact compiled glue,
  roughly -0.8% ordinary and +0.26%/-1.20% self-MTP.
- `wiki/docs/experiments/qwen4-megakernel-dual-width-2026-09-03.md`: ordinary
  1.72–1.77x with an accepted quality cost; self-MTP not compelling to integrate.
- `wiki/docs/plans/qwen4-qsa-indexed-fused-merge.md`: sequential merge exactness,
  later native pass-2 comparison and native output-gate reverse brackets.

Those are source-history measurements, not mlx2 performance results. The mlx2
artifact and execution policy must earn their own measured result.

## Qualification progress (2026-09-15)

The first 131,072-context run after the geometry fix passed its functional
HTTP, Hermes, sampling, batching, warm cache and near-limit context checks.
Root measured about 92.85 seconds for the cold near-limit request and a
84.80 GiB peak. Mechanism counters observed asynchronous promotion (13),
indexed QSA (112), and known-tail PLE prefetch (26). This run did **not**
qualify the complete policy: shared QSA and private-delta counters were zero.
The three-token long-context sentinel completed as B1 before establishing a
shared cohort. The qualification workload now uses sustained 64-token output
and leaves 128 prompt tokens of slack, with repeated cold/warm and concurrent
warm requests. Required mechanism counters remain mandatory. These results
are progress evidence, not a deployable qualification record.

## Throughput comparison

After each matching ordinary/MTP profile has passed qualification, run the
same warmed HTTP workload against that service in the exclusive GPU window:

```sh
.venv/bin/python scripts/benchmark_serving.py --url http://127.0.0.1:8285 \
  --rounds 5 --widths 1 2 4 --max-tokens 160 \
  --output qualification/benchmark-mtp2.json
# Restart the matching qualified --ordinary profile, then repeat with:
# --output qualification/benchmark-ordinary.json
```

The harness alternates batch-width order, requires warm cache hits, verifies
that B1/B2/B4 were actually observed, and rejects runtime/settings/artifact
changes. It retains every request receipt, output hash and warmup comparison.
Cross-width output hashes are reported for review: full-string identity is
not assumed across numerically different ordinary/MTP batch shapes. Report
median aggregate tokens/s for each width and the matched MTP/ordinary ratio;
this is end-to-end serving throughput, not isolated kernel attribution.

The subsequent warm-cohort pressure investigation found and fixed two separate
issues. Admission now reclaims allocator scratch and remeasures headroom before
routing a lane to ordinary decoding. More importantly, the long-lived serving
worker no longer retains completed lookup, boundary and response cache objects
in loop-local variables after APC eviction. A worker-loop weak-reference test
verifies all five cache transfers are released while the thread is still idle;
restoring the old local ownership in memory reproduces the test failure. These
repairs retain the conservative fp32 geometry and 20 GiB reserve.

Shared-suffix and physical-promotion storage are alternative routes within one
cohort. A short-output cohort admitted to shared QSA keeps the immutable base
and private tails; promoting it to a dense physical cache would undo the copy
avoidance and its storage ABI is different. The dispatcher explicitly retains
segmented execution for that cohort, with per-request reason
`shared_suffix_retains_segmented_owner` and a dedicated decline counter.
Other eligible cohorts still use asynchronous promotion. CPU integration tests
cover both selected routes and verify no promotion failure is swallowed.

Warm admission now leases an existing resident checkpoint before measuring its
cost or evicting other entries. A complete near-tail checkpoint is charged for
one **full measured target-and-draft copy**, conservatively scaled to the
requested extent, plus fp32 growth slack, verification work and the unchanged
reserve. It is not treated as free merely because it uses COW. Cold or long
uncached-tail requests retain the configuration-derived fp32 bound. A
resident-only lookup cannot restore disk arrays before the cold allocation
gate. Pressure eviction skips leased checkpoints, and reclamation completes
queued device work before allocator pages and headroom are measured again.
CPU regressions cover each ownership and admission boundary.

The model-specific growth estimate also separates fixed recurrent state from
per-token state. Dividing the entire measured cache by a short prompt length
incorrectly projected hundreds of megabytes of fixed GDN state as repeated
growth, serializing warm B2/B4 requests. For a declared geometry, the controller
uses its finite-difference growth and conservatively extrapolates only measured
bytes above the complete geometry bound. A real tiny-model APC/BatchGenerator
CPU regression now joins four different warm prefix lengths into observed B4
and matches each cold-reference prefix; the old formula fails that test.

Cycle admission now also reclaims unused APC checkpoints before splitting a
warm cohort or migrating it to ordinary decode. It evicts only unleased entries,
then synchronizes, clears allocator scratch and measures actual headroom again.
The pass stops when the requested cohort fits, no unleased entry remains, or
its bounded eviction count is reached. A capacity-only lane/verification limit
does not flush cache entries. CPU tests reproduce scratch-only recovery failing
and one unused-checkpoint eviction enabling the full pair, while retaining the
20 GiB reserve; full-model shared/private engagement remains a separate gate.

Hard admission can wait for delayed host-memory recovery. A failed headroom
check retains its original resident lookup and lease in a bounded worker queue,
retries every 250 ms, and expires after 60 seconds. Active batches continue
between retries; pending requests do not occupy execution lanes. Cancellation,
timeout and shutdown release the lease. Disk restoration still waits behind
the unchanged allocation gate. Status exposes `memory_waiting` and deferral,
retry and timeout counters. Worker-level CPU tests verify recovery while an
active request progresses, one lookup/tokenization per request, and cleanup on
all three terminal paths.

Before any lack-of-headroom deferral or final rejection, the shared reclaim
gate rate-limits `mx.clear_cache()`, then admission remeasures rather than
trusting stale `active + cached` allocator accounting. The memory-waiting
deadline and parallel-sample footprint guard use that same last-chance path;
`memory_cache_reclaims_before_reject` records engagement (design reference:
omlx#3732).

Optional async physical promotion now receives a non-mutating admission
preview before any APC eviction or lane migration. On pressure it synchronizes
the copy stream once: completed destination allocations are resident and no
longer counted again as pending. If measured headroom still cannot support the
cohort, promotion is cancelled and drained before normal memory policy runs.
The batch yields for at most two seconds if host accounting lags this release,
remeasuring without crediting hypothetical free bytes. The outer serving loop
honors this explicit wait reason and retains APC entries during the grace.
Requests record `memory_pressure_retains_segmented_owner`; counters distinguish
settled allocations from memory declines. Fitting async workloads retain copy
overlap and still require observed promotion in qualification. CPU tests cover
both pressure outcomes and show the old admission order evicting a checkpoint.
The preview compares with a capacity-capped ceiling, so lane or verification
row limits do not wait for memory that cannot change their outcome. A pressure
cancellation drain failure propagates before any further reclamation.
