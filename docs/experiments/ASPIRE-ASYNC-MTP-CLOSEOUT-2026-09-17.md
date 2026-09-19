# ASPIRE-style asynchronous MTP closeout — 2026-09-17

## Decision

Close this investigation without integrating asynchronous or heterogeneous
per-request MTP scheduling. On the measured native Qwen3.6 artifact, every
supported speculative depth was slower than the ordinary same-artifact route.
The observed oracle therefore selects `K=0` for every measured batch width.

This is a negative result for the current mlx2 native-MTP mechanism and
workloads. It is not a reproduction or refutation of ASPIRE's CUDA sparse-
attention system: mlx2 does not implement ASPIRE's unified mixed forward or
intra-draft attention refresh.

## Evidence state

| Item | State | Evidence |
|---|---|---|
| Request-local policy and offline replay | Implemented prototype | Pure-Python CPU module, fixed/feedback/ASPIRE-style policies and deterministic replay |
| Post-verification proposal tracing | Implemented probe | Default-off server wrapper records host-visible depth, accepted prefix, context, width and cycle time |
| Fixed K1/K2/K3 native-MTP routes | Observed used | `segmented_self_mtp` receipts recorded actual compute widths through B4 |
| ASPIRE mixed forward and refresh layer | Not implemented | No production runtime or route changes |
| Heterogeneous per-lane draft depth | Explicitly excluded | Current batching rejects mixed active depths; the probe did not bypass this boundary |
| ASPIRE scheduler | Not qualified, selected or deployed | No fixed speculative depth beat ordinary decode |

## Frozen identity

- Runtime source SHA-256:
  `7756c1a0b9a543c0ebbdb73a7d9bd501b68d8eae778baa146824347b6659e978`
- Native MLX SHA-256:
  `30335939714520c3306cfa58597100aae08bbeabdc60a4bc8c35c32c607450a8`
- MLX: `0.32.2.dev20260915+2a817ad94`
- Python: `3.12.13`; macOS: `26.7`
- Model artifact:
  `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp`
- Artifact fingerprint:
  `414a2fc7b4b3cb6451401032b9517ab91a73c1ffb95fe3c67323c685acf134fa`

The ordinary, K1, K2 and K3 arms used the same runtime and artifact. Candidate
servers were serialized under an exclusive CPG lease and matching
`/tmp/gpu.lock`. Swap did not grow, thermal/performance warnings remained
absent, APCv2/COW leases drained to zero, and all candidate listeners were
stopped before the lease and lock were released.

## Results

### Short-context warm HTTP throughput

Aggregate tokens per second, deterministic prompts, 128 generated tokens per
request:

| Compute width | Ordinary | K1 | K2 | K3 |
|---:|---:|---:|---:|---:|
| B1 | 123.20 | 85.12 | 76.59 | 80.60 |
| B2 | 199.84 | 119.08 | 117.25 | 108.96 |
| B4 | 301.61 | 161.86 | 148.44 | 132.42 |

K4 was attempted and failed closed at adapter construction because the artifact
supports only `num_draft` values 1, 2 and 3.

### Long-context warm HTTP throughput

The four prompts contained 15,224–15,227 rendered prompt tokens. Each request
generated 64 tokens after a separate warmup.

| Compute width | Ordinary | K1 | K1 / ordinary | K2 | K2 / ordinary |
|---:|---:|---:|---:|---:|---:|
| B1 | 101.17 | 78.64 | 0.777x | 69.36 | 0.686x |
| B4 | 203.77 | 127.44 | 0.625x | 107.69 | 0.528x |

Observed long-context draft acceptance was approximately 50.9% for K1 and
28.8% for K2. The increasing draft/verification cost was not repaid by the
accepted tokens.

### Correctness boundary

All responses were finite, the MTP mechanism counters advanced, and no runtime
failure was recorded. Temperature-zero output hashes did not match between the
ordinary and MTP arms. Ordinary output also changed in some cross-width cells,
so the mismatch is not uniquely attributable to speculation. Exact route
equivalence is nevertheless not established, and the performance result must
not be promoted to qualification.

## CPU prototype versus hardware

The earlier synthetic CPU replay predicted an ASPIRE-style gain of
`1.155x–1.196x` over fixed K2 across five seeds. That result modeled policy
economics with synthetic acceptance and cost coefficients. The real GPU matrix
invalidated its transfer assumption: ordinary decode beat every supported
speculative depth. The CPU prototype remains useful as an offline trace tool,
not as evidence for runtime selection.

### Depth-zero runtime cost

The adaptive controller's K=0 round now takes a direct target-only advance:
it skips draft-head execution and generic propose/commit rollback orchestration,
while retaining target hidden/token pairs in `pending_hs`/`pending_ts` for an
exact later K>0 teacher-forced re-entry. Segmented rows still publish their one
exact target delta through the live revision-bound lineage; removing that
ownership operation would make later state validation unsound.

On CPU with the repository's tiny Qwen4 fixture, five 80-round samples before
the change measured K=0 at 1.185x-1.268x ordinary decode (median about 1.21x),
with zero MTP-head calls. After the direct advance they measured
1.108x-1.199x (median about 1.15x). This is a control-path measurement, not a
Metal performance claim. `self_mtp_zero_fast_rounds`, skipped-draft-forward and
skipped-proposal-roundtrip counters prove selection; a CPU regression switches
from K=0 back to K=1 and matches ordinary greedy output across the boundary.

## Revisit gate

Do not resume this line for scheduler integration alone. Revisit only after a
different drafting mechanism or attention path materially improves acceptance
or reduces draft cost. Before implementing heterogeneous scheduling, require:

1. one fixed-depth arm to beat ordinary same-artifact decode at an observed
   physical batch width;
2. exact route-equivalence evidence under the intended correctness contract;
3. real timing-derived cost coefficients and per-round acceptance traces; and
4. a separately reviewed mixed-forward design that preserves APCv2 ownership,
   revision-bound state and ordinary fail-closed fallback.

## Artifacts and reproduction

- Summary: [`artifacts/aspire-gpu-2026-09-17/viability-summary.json`](../../artifacts/aspire-gpu-2026-09-17/viability-summary.json)
- Raw HTTP reports and proposal traces:
  [`artifacts/aspire-gpu-2026-09-17/`](../../artifacts/aspire-gpu-2026-09-17/)
- CPU policy/replay:
  [`src/mlx2/runtime/async_mtp_scheduler.py`](../../src/mlx2/runtime/async_mtp_scheduler.py)
- Synthetic replay runner:
  [`scripts/bench_async_mtp_scheduler.py`](../../scripts/bench_async_mtp_scheduler.py)
- Default-off GPU trace wrapper:
  [`scripts/serve_async_mtp_trace.py`](../../scripts/serve_async_mtp_trace.py)
- Long-context HTTP runner:
  [`scripts/benchmark_aspire_long_context.py`](../../scripts/benchmark_aspire_long_context.py)
- Provenance:
  [`provenance/aspire-scheduler-prototype.json`](../../provenance/aspire-scheduler-prototype.json)

Focused closeout verification passed ten tests and Ruff. These artifacts are
retained to prevent the same unprofitable scheduler-only experiment from being
repeated without a changed drafting premise.
