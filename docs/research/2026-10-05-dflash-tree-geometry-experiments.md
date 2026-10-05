# DFlash tree verification geometry experiments — 2026-10-05

## Follow-up: packed target correction and row-geometry fix

This follow-up supersedes the original conclusion below that cohorting only
placed independent lane forwards behind one fence.  The historical matrix is
retained because it identifies the bottleneck the correction removed.

`qwen38_tensorfold.forward_many` now calls the pinned TensorFold
`multi_tree_forward` once for the whole cohort and commits accepted prefixes
with `commit_streams`.  Per-request cache and recurrent state remain lane-owned;
only real tree rows share the physical model trunk.  New receipts distinguish
logical lane rounds from packed rounds and physical target forwards.

The first live correction exposed a second issue: the default lane-matmul
window ended at 32 rows.  B1 prefill and B4 K15 verification used the
row-invariant lane law, while the roughly 52–62-row B2 prefill crossed to stock
MLX arithmetic and changed the greedy chat output.  The Qwen3.8 adapter now
declares `max_rows=128` and exact chunking above 128 whenever packed varlen/tree
execution is selected.  Operator policy may still override this for explicit
experiments, and the source of each setting is emitted in status.

One-repetition live observations with the adapter default, not a CLI override:

| Width | Code decode | Chat decode | B1 output parity |
|---:|---:|---:|---|
| B2 | 108.37 t/s/lane | 39.23 t/s/lane | both prompt hashes match |
| B4 | 61.52 t/s/lane | 22.45 t/s/lane | both prompt hashes match |

These are diagnostic observations, not controlled performance claims.  In the
skewed B4 proof the prompts tokenized to 89, 166, 253, and 369 tokens.  Varlen
prefill observed 599 rectangular padding-token rows and skipped 38,336
layer-level MLP padding rows.  The above-128 lane path recorded 496 chunked
projection calls, 4,992 launches, and 615,104 rows.  Route receipts reported a
minimum DFlash proposal length of three, except that terminal token headroom is
allowed to shorten a final round.

State after this follow-up: packed execution, adapter geometry, varlen row
skipping, and the proposal floor are implemented and observed-used on the
explicit Qwen3.8 candidate.  They remain unqualified and are not a universal
default.

### 64-row kernel follow-up

The packed B4 tile is 64 physical target positions: four independent lane
trees, each with one anchor plus up to fifteen candidate nodes.  Splitting the
row axis into two 32-row launches or four 16-row launches preserved the lane
law but was 2--7% slower at 64 rows, so scheduler-side row splitting was
rejected.

The useful change was internal: retain the 32-column output tile and run one
16-row MPP fragment per threadgroup instead of two.  Interleaved q4/group-64
tests were bitwise equal and reduced the 64-row gate/up and down projection
medians by 10.1% and 14.1% respectively.  The broader native gate passed
38/38 all-format small cases and 4/4 Qwen model shapes across the full tested
1--128-row invariance ladder.

The interleaved A/B also covered the adapter's 128-row packed-prefill chunk.
It retained bitwise equality and reduced gate/up from 0.62608 to 0.55795 ms
and down from 0.68612 to 0.58757 ms, so the decode geometry does not regress
the selected long-prefill chunk.

The same one-repetition live harness retained output hashes and was neutral at
B1/B2.  At B4 it moved code from 61.52 to 64.49 t/s/lane and chat from 22.45
to 23.70 t/s/lane.  This is an observed diagnostic improvement, not a
thermally controlled performance claim.  It narrows the next work to the
route-level long-context correctness gate recorded below and controlled
timing, rather than further scheduler row splitting.

### Long-context partial acceptance

The external route now receipts the joint verification span and accepted
draft prefix.  On a 16,384-token B1 prompt, exact-law ordinary decode and the
DFlash/TensorFold tree route produced identical 64-token output.  The tree
route observed three `16:2` strict partial commits, one `16:0` rollback, three
`16:15` full accepts, and one terminal `6:5` round.  This closes the
serving-level planted partial-commit gate.  It does not prove equality of every
intermediate recurrent checkpoint tensor; retain that as a narrower internal
diagnostic rather than inflating final-output parity into full state proof.

### Upstream geometry candidates

- **Probe — MTPLX #595.** Its default-off 4-bit kernels target exactly five,
  7–16, and 17–32 rows while reading each weight tile once.  The 16/32-row
  cells map directly to B1/B2 K15 verification, but its new 17–32 route is not
  stock-bit-exact.  Compare it against mlx2's row-invariant law at 16/32/64
  rows before considering any implementation intake.
- **Probe — oMLX #4105.** Its useful concepts are row-exact 16-row verify
  kernels and deferred recurrent-state materialization on partial acceptance.
  The current PR is open and has a reported GLM-5.3 rollback regression, so
  neither its wider copy proposal nor its rollback assumptions transfer as a
  generic scheduler policy.
- **Watch for the Qwen4 adapter — oMLX #4245.** Group-size-32 exact paired
  hyper-connection projections are a strong model-specific B1 optimization,
  but the current Qwen3.8 dense-hybrid artifact has no hyper-connection
  geometry.  Any future Qwen4 intake must derive group size and projection
  pairing from loaded weights and fail closed for unsupported geometry.

## Result

The experiments confirm that proposal blocks are verified together, but they
also confirm that the current TensorFold multi-lane route is not a true packed
cross-request target operation.

- One DFlash tree is verified in one target call. At cohort limit 1 the run
  recorded 291 lane-tree rounds and 299 target calls; the eight additional
  calls were ordinary tail rounds.
- Cohorting is mechanically active. Raising the limit from 1 to 2 to 4 changed
  the average lane trees per target call from 0.97 to 1.86 to 3.90 and target
  positions per call from 14.45 to 27.24 to 56.73.
- That reduction in calls did not reduce total target time. Target launch plus
  wait was 13.38 s at limit 1, 12.76 s at limit 2, and 13.25 s at limit 4.
  `qwen38_tensorfold.forward_many` still invokes the target once per lane and
  uses one final fence; it does not execute one packed target trunk.
- K15 was the best tested tree budget. Reducing the per-lane budget to K7 or K3
  increased target-call count and reduced throughput. K3 also changed the
  greedy chat output hash and is not a valid selectable candidate.
- Width scaling is consistent with serial lane work. Code median per-lane
  decode was 193.93 t/s at B1, 100.21 t/s at B2, and 49.84 t/s at B4. The
  corresponding aggregate decode-window rates were 192.72, 182.72, and
  176.25 t/s rather than increasing with lane count.
- Greedy chat output was shape-dependent: B1 and B4 K15 agreed, B2 K15 differed,
  and B4 K3 differed again. The route remains unqualified.

No configuration tested here should become a qualified default. The useful
outcome is a narrowed implementation target: preserve the full proposal block,
but replace fence-only cohorting with a real packed target forward and first
resolve shape-dependent greedy parity.

## Controls

- Target: `Qwen3.8-27B-MLX-4bit`
- Drafter: `Qwen3.8-27B-DFlash2`
- Selected route: bounded tree15, TensorFold target, varlen dense MLP and
  external varlen prefill
- Prompts: one short code prompt and one short chat prompt
- Output: 64 tokens per request
- Sampling: greedy
- Repetitions: one warmup plus two measured repetitions per cell
- APCv2: reuse failed closed as
  `external_varlen_prefill_not_batch_invariant`; writes were also suppressed
- GPU ownership: shared waiter ledger plus both GPU locks for every native run

The campaign was deliberately brief and was not thermally counterbalanced.
Sequential arm order can contribute drift, so small differences are diagnostic
rather than publication-quality performance claims.

## Cohort-limit matrix: B4, K15

| Cohort limit | Target calls | Lane trees / call | Target positions / call | Median lane decode, code | Median lane decode, chat | Aggregate wall, code | Aggregate wall, chat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 299 | 0.97 | 14.45 | 49.84 t/s | 17.48 t/s | 153.34 t/s | 63.29 t/s |
| 2 | 159 | 1.86 | 27.24 | 47.66 t/s | 16.14 t/s | 146.84 t/s | 58.45 t/s |
| 4 | 77 | 3.90 | 56.73 | 46.28 t/s | 15.27 t/s | 144.58 t/s | 56.16 t/s |

The aggregate decode-window estimate (from first observed token to final
completion) was 176.25/66.30 t/s at limit 1, 173.99/61.94 at limit 2, and
184.28/60.98 at limit 4 for code/chat. The code-only limit-4 improvement does
not survive the total request wall metric, does not generalize to chat, and is
too small for selection under the non-counterbalanced run order.

### Phase attribution

| Cohort limit | Tree round total | Target launch total | Target wait total | Draft prelaunch total | Draft wait total |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.03 s | 10.45 s | 2.93 s | 0.64 s | 0.28 s |
| 2 | 16.31 s | 10.53 s | 2.23 s | 1.59 s | 1.17 s |
| 4 | 17.11 s | 11.39 s | 1.85 s | 1.72 s | 1.35 s |

The wider cohort saves target waits but adds target launch work and draft
coordination. This is the expected signature of independent lane forwards
collected behind one evaluation fence.

## Tree-node budget matrix: B4, cohort limit 1

| Budget | Target calls | Accepted / proposed nodes | Speculative emissions / target call | Median lane decode, code | Median lane decode, chat | Aggregate wall, code | Aggregate wall, chat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| K3 | 495 | 69.3% | 2.96 | 21.26 t/s | 15.72 t/s | 76.41 t/s | 57.99 t/s |
| K7 | 342 | 51.3% | 4.33 | 32.25 t/s | 17.10 t/s | 107.28 t/s | 61.43 t/s |
| K15 | 299 | 30.0% | 5.02 | 49.84 t/s | 17.48 t/s | 153.34 t/s | 63.29 t/s |

Acceptance percentage alone is misleading: K3 accepts a larger fraction of a
much smaller candidate set but emits fewer useful tokens per expensive target
call. K15 amortizes the target call best on both prompts.

## Width probe: K15, cohort limit 1

| Width | Target calls | Target positions / call | Median lane decode, code | Median lane decode, chat | Aggregate decode-window, code | Aggregate decode-window, chat | Chat hash |
|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | 60 | 14.22 | 193.93 t/s | 68.12 t/s | 192.72 t/s | 67.97 t/s | `aad807ad…` |
| B2 | 118 | 15.10 | 100.21 t/s | 43.95 t/s | 182.72 t/s | 84.58 t/s | `7f9c97f0…` |
| B4 | 299 | 14.45 | 49.84 t/s | 17.48 t/s | 176.25 t/s | 66.30 t/s | `aad807ad…` |

The code output hash was stable at every width. The chat hash was not. Before
any performance selection, capture the first divergent token and compare the
target logits, argmax margin, row position, parent map, cache offsets, and
TensorFold/reference logits at B1/B2/B4 and K3/K7/K15.

## One proposal block, one target verification

The user-requested invariant is already implemented for both routes:

- Linear external DFlash constructs each verify row as
  `[anchor] + all proposed tokens`, then invokes one target
  `forward_with_taps` (or one TensorFold tree call) for the whole row. A
  five-token proposal is therefore verified by one six-position target
  forward, not five target forwards.
- Native self-MTP builds `verify_ids` with shape
  `B x max(K + 1)` and calls the target verifier and target head once per
  committed cycle. Existing B4 MTP evidence records 513 committed cycles,
  513 batched target forwards, and 1,515 draft-head forwards (three draft
  steps per cycle). MTP3 is therefore verifying all three proposals together
  in one four-position target forward.

The remaining problem is cross-lane execution, not serial verification inside
one proposal block.

## Implementation completed for the experiment

The bounded Qwen policy now accepts an explicit
`tensorfold_cohort_limit` integer. It is revision-bound policy state, validated
from 1 through the selected bounded route width, conflicts with the legacy
environment override, and appears in execution settings and route receipts.
It remains an explicit experimental control and is not a default selection.

Validation: 290 targeted Qwen policy and DFlash tree-bridge tests passed using
the worktree source under paired GPU locks. Python compilation and
`git diff --check` passed. Every performance run completed with both GPU locks
restored to empty idle sentinels.

## Next implementation sequence

1. **Fix parity first.** Add a deterministic first-divergence probe that stores
   per-token target argmax, top-two margin, parent row, position id, cache
   offset, and reference/TensorFold logits for the mismatching chat prompt.
   Do not qualify or default-select the route until B1/B2/B4 and budget changes
   preserve the ordinary greedy reference.
2. **Implement true packed target execution.** Replace the per-lane
   `forward_many` list-comprehension with an adapter-owned packed tree forward:
   concatenate real target positions, carry lane offsets/parent maps/position
   ids, use block-diagonal or varlen attention metadata, execute each model
   layer once over the packed rows, and scatter logits/features back to the
   lane transactions. Varlen dense MLP can then operate on the same packed row
   axis.
3. **Prove physical execution.** Receipt counters must distinguish logical lane
   trees, packed target calls, model-layer calls, evaluation fences, logical
   rows, and physical padded rows. A packed B4 call must show one target trunk
   invocation per layer, not four invocations and one fence.
4. **Retest K15 first.** K15 is the measured proposal-width winner. Do not spend
   another tuning cycle on K3/K7 until the real packed trunk exists.
5. **Then qualify scheduler crossover.** Run thermally counterbalanced B1/B2/B4
   homogeneous and heterogeneous prompts, followed by the deterministic 20x20
   traffic campaign. Select a cohort width only when it improves total wall
   throughput without hurting latency or parity.

Machine-readable evidence is in
`qualification/runs/dflash-tree-geometry-20261005/campaign.json`; raw arm JSON,
server logs, policies, test receipts, and paired-lock receipts are in the same
directory.
