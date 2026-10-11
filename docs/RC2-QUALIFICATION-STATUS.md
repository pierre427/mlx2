# RC2 qualification status — 2026-10-10

This is a checkpoint for the RC2 qualification campaign. It reports
functional qualification evidence, not performance, route selection, or routine
production use. Qualification applies only to the exact tested runtime,
artifact, settings, and host identity; this public summary is not a portable
qualification receipt and does not certify another checkout or host.

The current reviewed implementation snapshot is `b5a6e5a42`, with runtime
source SHA-256
`8792b04a1089306c92d98aef1fc606e10f63fbd4725b9715bcd356bf99b60109` and
qualification-harness SHA-256
`cddc61b8cea37e6277ce3b9ef8b41f78748ada4f037c560767525655da296ba8`. Its full
CPU preflights passed on M3 and M5, including the default test suite and 64
import guards on each host. This establishes source-stable CPU validation only.
Model qualification has not been rerun against this source identity, so the
model and profile outcomes below describe earlier, receipt-bound runs and do
not qualify the current snapshot.

## Decision routes

Four typed decision routes—Clef, Decision2, JEV, and the 27B decision route—each
passed all 13 required functional gates on an earlier internal validation
snapshot.
The checks covered two connections, a 4 MiB request-body limit, the normal
context-truncation default, and text, `noul`, choice, and score request paths. This establishes functional
qualification only for those earlier source identities, exact tested routes,
and settings. Requalification against the current source is pending. These
results do not select the routes as defaults or establish that production
traffic has used them.

## Vision and media routes

| Route/profile | Evidence and current boundary |
| --- | --- |
| SmolVLM, LFM2.5-VL and Qwen2.5-VL ordinary decode | Three complete passes on an earlier source identity at the tested 4K context, one lane, 256-token prefill, and 1 GiB cache profile. The evidence is limited to that identity and profile; current-source requalification is pending. |
| Gemma 4 media | Media qualification passed on an earlier source identity. The generic Hermes literal-prompt probe remains unqualified, and current-source requalification is pending. |
| Gemma 3n media | An earlier source-bound run was not qualified: the cold continuous-batch reference mismatched, and video-logit parity failed. Requalification against the current source is pending. |
| MiniCPM-O media | Earlier runs did not qualify the route: image feature/token parity failed. Audio parity was exact, but the aggregate gate remained coupled and therefore did not pass. A separate diagnostic-only matrix on an earlier snapshot first diverged at batched vision-tower output while sequential and count-one controls were exact. On the current snapshot, a sequential-default media run passed its five media checks, but end-to-end route validation failed because the generic reader did not receive the descriptor needed to bind policy evidence. The route remains unqualified; descriptor propagation and a fresh source-bound rerun are pending. The implementation and policy tests pass CPU validation. |

These results describe the tested cases only. A pass for one modality or probe
does not override a failed or still-unqualified gate for the route as a whole.

## DLoop and performance work

The public DLoop harness now rejects rows with no decode tokens instead of
dividing by zero. This is a CPU-tested harness correctness fix; it is not model
qualification or a performance result. The bounded first-divergence probe and
its CPU tests also exist on the current snapshot. The GPU run below used an
earlier frozen identity and does not qualify the current snapshot. A matched-Q4 width-one behavior run
completed with loop8 extension/span engagement, but the independent audit
rejected exact-token equivalence in 48 rows against the depth-one self-MTP
control across fixed5–fixed8 and loop8. This is a self-MTP arm comparison, not
an ordinary-decode regression result; ordinary-reference qualification remains
pending. The run's process exit status does not override the arm-equivalence
failure.

A bounded follow-up on the current frozen snapshot traced a divergence to a
near-tied ordinary next-token choice: depth one matched ordinary decode, while
deeper fixed and loop arms differed. Saved ordinary/self-MTP continuations
matched; one cold continuation selected the tied alternative. This diagnostic
does not qualify a width, establish a cache defect, or support a wider default.

The separate state-oracle findings were invalidated by a collector-boundary
bug: the final `response.token` was omitted, so saved and cold continuations
used different token tails. Those findings do not establish a runtime cache
defect. Width two and wider, along with the second artifact's DLoop ladder,
remain held pending collector correction, review, and fresh admission. This is
correctness/state evidence only, not a performance result.

The separate Flash Next 32K performance investigation is documented in
`FLASH_NEXT_32K_PERF_FIX_20261010.md`. Release-wide thermally controlled
ladders remain pending. The release remains `0.1.0rc1`; `0.1.0rc2` is pending
completion and review of the remaining release gates.

## Flash Next feature inventory

The complete Flash Next feature inventory was exercised on an internal frozen
source at `874030a82e37f859e2e45b22c93b31725fe4184f`. Its source-stable CPU
preflight passed 13,823 tests, 813 skips, 118 subtests, and all 64 isolated
import guards. This evidence does not transfer to the independently curated
public checkout.

Fifteen profiles qualified on that exact source, artifact, host, and serving
shape: ordinary decode, fp32 head logits, APC junction checkpoints, persistent
blocks, QSA sparse batching in gather and indexed modes, QSA direct selectors
at widths four and eight, PLE serial and adaptive reads, GDN prefill chunks at
eight and sixteen, invariant prefill, varlen sparse-MoE prefill, and fp16 GDN
state. The fp16 state result is explicitly an approximate-operation
qualification; it does not establish fp32 parity. Every profile retained its
existing selection and default state, and observed use is limited to the
qualification campaign.

Sixteen profiles remained unqualified: native MTP, GDN dynamic accept,
row-exact windows, indexed fused merge/output gating, batched QSA/NAX, PLE
early dispatch, TensorFold QMV, native and Metal TensorFold prefill, adaptive
MTP depth, Fly verification, prompt lookup, APC rolling, APC in-flight wait,
cache capsules, and Spomin surgery. The recurring failure classes were missing
run-local engagement, a missing MTP-to-ordinary handoff, unsupported projection
topology, required baseline smoke failures, cold/warm equality failures, and
bounded width-four admission refusals. No functional failure received an
automatic retry.

A controlled width-four investigation also corrected a receipt interpretation:
a cold APCv2 miss correctly has no checkpoint role, while the matched warm
lookup reports a committed prompt boundary. The preserved comparison passed
strict token identity, streams, needles, producing width, and warm prefix reuse.
That shared-gate correction is correctness evidence and does not convert any
failed feature profile into a pass.

These ladders were qualification runs. Thermal or swap contamination was
retained as a performance exclusion rather than a correctness failure, and no
ladder number is eligible for a performance claim.

## State terminology

- **Implemented** means the code path exists.
- **Qualified** means the required evidence passed for the exact bound test
  identity and settings described above.
- **Selected** means a route has been explicitly chosen for a deployment; the
  results here do not make that choice.
- **Observed-used** means execution evidence shows a mechanism actually ran.
  Qualification traces can demonstrate the exercised mechanisms; they do not
  establish routine production use.
