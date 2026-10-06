# Varlen, TensorFold, and speculative decoding closeout

Date: 2026-10-05 (America/Toronto)

## Executive result

### Same-day packed-target follow-up

The principal decode gap named in the original closeout is now implemented:
bounded B1–B4 DFlash trees use one physical TensorFold target forward per
packed cohort round rather than one lane forward per request behind a shared
fence.  The first live run then isolated and corrected a row-geometry parity
fault by moving the selected Qwen3.8 lane law from 32 to 128 rows and chunking
wider calls on that same numerical law.

Default-policy B2 and B4 spot checks now match the B1 greedy output hashes for
both short prompts.  A deliberately skewed B4 cohort observed both varlen MLP
row elimination and the above-128 chunk path.  DFlash2 now prefers at least
three proposals, Qwen policy rejects a floor below two, and only terminal
headroom may shorten a final round.  These changes are implemented and
observed-used but remain unqualified; the longer controlled performance and
partial-accept exactness campaign is still outstanding.

The Qwen3.8-27B serving slice now has working, CPU-regressed implementations
for atomic skewed-cohort prefill, adapter-owned varlen MLP compaction,
TensorFold-backed bounded tree verification, exact-law PLD-to-DFlash proposal
composition, idle allocator reclamation, and an explicit lane-pressure tree
budget policy.

The final corrected configuration is selected by default within Qwen3.8's
explicit external-DFlash route; the external route itself remains explicit and
unqualified. It selects varlen prefill, TensorFold tree verification, a
15/7/4/3 proposal-node budget at widths 1 through 4, and dense-MLP compaction
only at or above 25% padding. Ordinary decode remains the reference path.

The full corrected-geometry A/B-B/A ran 1,680 timed requests with zero
control/control, candidate/candidate, or cross-arm output mismatches. Combined
decode was 51.441 tokens/s candidate versus 48.667 control, or 1.05699x. Each
candidate arm selected 21 of 105 varlen scopes and skipped 1,570,496 padded MLP
rows. This supports the selected default but is not thermal qualification or a
1.2x-1.3x general serving claim.

## Source state and validation boundary

- Performance source: tracked revision `9e2596a50245ab8c5959706abf5b3d7b33c478ef`
  plus the dirty nine-file patch used by the run harnesses.
- Current integration base: `61a6b83a026ba8aec719d2e00df333ba8906ef7b`.
  This includes the later Qwen route-identity and serving-regression fixes.
- Reconciled patch digest: `57db21b7b3e88e4f0f1d253d1043b0bc6279de7322d0d24fb965798d42ef98ab`.
- The patch was reapplied to the current integration base with one manual
  semantic merge: both main's normalized adaptive-verification receipt helper
  and this work's lane-budget policy helper are retained.
- `git diff --check` and Python compilation pass.
- Post-reconciliation CPU validation: 399 DFlash/tree/composition tests plus
  196 route-identity, qualification, geometry, retention, and quiesce tests;
  595 passed in total. The real-artifact header-schema test was deliberately
  deselected because its paired local artifact is not available to that test.
- The GPU performance evidence remains source-bound to the measured dirty
  source. Current-main integration has CPU regression evidence but has not been
  re-benchmarked on GPU.

## Implemented architecture

### Scheduler and varlen prefill

- A declared `batch_cohort` is held atomically at the prompt/decode boundary.
  A shorter row may finish prefill early, but it cannot begin decode while a
  declared peer is still prefilling.
- Scheduler-phase waits are reported separately from genuine memory-admission
  waits, preventing the serving stall watchdog from killing a healthy skewed
  cohort.
- A completed final lane clears unreferenced allocator scratch before the next
  cohort is admitted.
- Adapter-owned row geometry remains the authority for dense MLP compaction.
  The scheduler contains no model-name branch.
- Ordinary decode and serial external prefill remain available as reference
  paths. The candidate remains explicit-policy only.

### TensorFold and lane depth

- The existing pinned TensorFold target bridge is used for bounded tree target
  execution. It is not copied into mlx2.
- `tree_node_budget_by_lanes` is validated by the adapter, applied to admission
  pricing and proposal generation, and emitted in route and scheduler receipts.
- The original fixed-15-per-lane policy expanded B4 verification to 64 target
  rows. The selected mapping is `{1: 15, 2: 7, 3: 4, 4: 3}`, yielding
  16/16/15/16 total target rows while retaining at least three proposals.
- State: lane-budget mechanism implemented and CPU tested; the 15/7/4/3
  schedule is selected and observed-used in the exact A/B-B/A.

### Speculative proposal composition

- DFlash2 now declares `stochastic_exact_law` because it publishes the exact
  dense selector distribution for every sampled token.
- Proposal composition may replace selected rows with deterministic PLD or MTP
  point masses while preserving DFlash's exact stochastic law for untouched
  rows.
- Receipts no longer mislabel a mixed PLD/DFlash block as globally
  deterministic.
- Tiny-model CPU oracles verify mixed-law sampling, cache alignment, serving
  output against greedy ordinary decode, and state-sidecar validity.
- Live composed PLD-to-DFlash execution was observed. Native MTP plus external
  DFlash was not artifact-composable: the tested DFlash target has no embedded
  MTP heads, while the MTP artifact has a different revision identity.

## 20-by-20 mixed-domain workload

The complete order-1 pair graded 400/400 requests correctly in each arm, with
no HTTP errors. The reverse-order candidate was stopped after four rounds, so
the experiment is an incomplete A/B and cannot select a winner.

| Arm | End-to-end tok/s | Median aggregate decode tok/s | Correct |
| --- | ---: | ---: | ---: |
| Control | 43.65 | 56.04 | 400/400 |
| Varlen candidate | 43.11 | 56.61 | 400/400 |

The candidate was 1.0% faster on the inclusive decode metric and 1.2% slower
end to end. This is neutral evidence, not the expected 1.2x-1.3x gain. The
campaign is recorded as `interrupted_by_user_after_order1`.

## Context ladder

The uncontrolled single pass completed every feasible cell from 1K through
128K. The subsequent campaign requested three strict thermally controlled
repetitions per cell. A cell was accepted only while the macOS thermal probe
remained at state zero; after three post-cell thermal breaches the shape was
marked thermally limited and later repetitions were skipped.

Controlled medians use accepted cells only:

| Context | Batch | Accepted reps | Decode tok/s median | Prefill tok/s median | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1K | 1 | 3 | 164.39 | 423.32 | complete |
| 1K | 2 | 3 | 191.18 | 768.52 | complete |
| 1K | 4 | 3 | 141.58 | 735.86 | complete |
| 4K | 1 | 3 | 132.97 | 800.13 | complete |
| 4K | 2 | 3 | 100.88 | 964.10 | complete |
| 4K | 4 | 3 | 81.31 | 843.60 | complete |
| 16K | 1 | 3 | 61.39 | 851.39 | complete |
| 16K | 2 | 3 | 123.51 | 893.30 | complete |
| 16K | 4 | 2 | 75.10 | 693.77 | thermally limited in rep 3 |
| 32K | 1 | 3 | 59.32 | 794.05 | complete |
| 32K | 2 | 1 | 81.30 | 539.56 | later reps thermally limited |
| 32K | 4 | 1 | 54.70 | 532.70 | later reps thermally limited |
| 64K | 1 | 1 | 48.00 | 440.32 | later reps thermally limited |
| 64K | 2 | 1 | 63.15 | 477.33 | later reps thermally limited |
| 64K | 4 | 1 | 56.73 | 511.84 | later reps thermally limited |
| 128K | 1 | 1 | 69.73 | 441.98 | later reps thermally limited |
| 128K | 2 | 0 | - | - | three thermal rejections |
| 128K | 4 | 0 | - | - | memory limited |

The report status is `completed_with_thermal_limits`; the harness itself
completed with return code zero. Rejected attempts are retained as observations
but are excluded from controlled medians.

## Common benchmark from Downloads

The supplied `mlx2-benchmark` `bench_stream.py` and fixtures were run for one
warmup and three measured repetitions. Outputs were stable within each fixture. The coding
output parses as Python, the structured output parses as JSON with six steps,
and all batch lanes completed with stable hashes.

| Runtime | Coding | Creative | JSON | B1 aggregate | B4 aggregate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current mlx2, M5 Max | 112.15 | 45.83 | 129.59 | 52.85 | 47.66 |
| Rapid packet, M3 Ultra | 108.41 | 57.85 | 111.18 | 57.33 | unsupported |
| Direct TensorFold packet, M3 Ultra | 106.84 | 51.42 | 97.59 | 52.51 | 90.87 |
| Older mlx2 packet, M3 Ultra | 46.93 | 21.25 | 47.76 | 27.06 | 32.46 |

Relative to the older mlx2 packet, current mlx2 measured 2.39x coding, 2.16x
creative, 2.71x JSON, 1.95x B1, and 1.47x B4. Relative to direct TensorFold it
was 1.05x coding, 0.89x creative, 1.33x JSON, 1.01x B1, and 0.52x B4.

These are cross-host comparisons: current mlx2 ran on an M5 Max with 128 GiB,
while the supplied packet used an M3 Ultra with 256 GiB. They are not a pure
software delta. TensorFold target execution and tree verification were
observed-used. Varlen was selected but not observed in this benchmark because
the batched fixture rows had equal prompt lengths; no speedup in this table may
be attributed to varlen.

## Rapid-MLX community benchmark

The official Rapid-MLX 0.15.3 source benchmark completed on the same M5 Max,
without speculative decoding:

| Case | Prefill tok/s median | Decode tok/s median |
| --- | ---: | ---: |
| PP512 / TG128 | 862.02 | 33.52 |
| PP2048 / TG512 | 739.16 | 32.58 |

The run remained nominal thermally and peaked at 17,288 MiB active memory.

## Proposal-arm results

All results below are candidate performance observations, not qualification:

- PLD alone: about 28 tok/s at B1 and 28 aggregate tok/s at B4. It was not
  competitive on these prompts.
- Native MTP: 77.80 code / 49.32 chat tok/s at B1; approximately 182.50 code /
  118.09 chat aggregate tok/s at B4.
- PLD plus native MTP did not improve the MTP arm on this fixture.
- PLD-to-DFlash exact-law composition: 91.85 code / 39.76 chat tok/s at B1;
  approximately 172.34 code / 75.33 chat aggregate tok/s at B4. PLD was
  observed-used on the repeating code prompt and absent on the chat prompt.

Composition therefore helps some repetitive workloads but is not a universal
winner over native MTP. Policy selection must remain workload- and
artifact-aware.

## Commit-readiness decision

Ready to commit together:

- the nine source/test changes;
- the new provenance record and this report;
- durable final JSON results, policies, harnesses, and GPU-lock receipts for
  the final context ladder, Downloads benchmark, Rapid benchmark, proposal
  arms, and the completed order-1 20-by-20 pair.

Do not stage as product evidence:

- abandoned startup attempts and pre-fix/interrupted subdirectories;
- the incomplete first 20-by-20 directory;
- the incomplete reverse-order candidate logs as if they were a completed A/B;
- raw server logs unless needed for a specific diagnostic.

The source patch is CPU-ready on current main. It is not yet performance-
qualified on current main because the GPU measurements preceded the
route-identity merge. The adaptive lane-budget policy is also not selected.

## Remaining gaps and recommended next run

1. Run three thermally controlled repetitions of the selected 15/7/4/3 plus
   25%-crossover policy and retain the default only if the exact gain persists.
2. Repair or explicitly redefine TensorFold's ordinary-serial numerical
   contract; accepted-prefix transaction bookkeeping is exact against its own
   arithmetic, but an immediate ordinary-serial token comparison still drifts.
3. Focus optimization on B4 decode: current mlx2 is close to direct TensorFold
   at B1 but only about half its B4 aggregate rate in the supplied packet.
   Instrument target MLP, stacking/scattering, proposal construction, and
   verification fences separately before adding another kernel.
4. Sweep adjacent B2/B3/B4 budgets around 7/4/3 while retaining at least three
   proposals and about 16 total physical target rows.
5. Qualify exact-law PLD-to-DFlash composition with a larger deterministic
   corpus and non-greedy seeds before making it an automatic scheduler choice.

## Evidence identities

- Context single-pass JSON: `d6ee74551b13e8d50b259a0bce225b149b337c354ad27b00bc67f234e50ed4d9`
- Context thermal JSON: `a6a7af93756d178551f163cc984a9dcc6dfeb6c75ce04ff5d88edc8d107b3ae2`
- Downloads benchmark JSON: `198549ae27c9deb7f9b7645ef89d168837e04b1df0e31b91c363514e819dae9e`
- Rapid benchmark JSON: `c6c5b2979642e3c8db5ec94baebf9a3d5e12166913eb02ddb2641eb18ba00f9b`
- Composed B1 JSON: `e906036d496e1e04d0f77db3e49f755d4dcf44c5cadeee6e69146973d6cd557b`
- Composed B4 JSON: `ff49be290beaa1a561eb2101eadb568c838300ec028ac93a7d7659dd788db8fe`
- 20-by-20 campaign JSON: `e3a29f97b26e79b317ecd7152da3f90a310b44d2ea125280eaf960ba7647f2b5`
