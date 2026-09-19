# Advanced inference imports campaign — 2026-09-18

## Executive result

Branch `codex/peer-advanced-imports-20260917` integrates the selected advanced
mechanisms from OMLX, Rapid-MLX, and the lab-owned `mlx-lm-unified` into mlx2's
APCv2-only runtime. The work is rebased semantically by merge onto main
`d41f581`; no legacy APC or replaced serving backend was introduced.

The final implementation state is:

| Capability | State | Default/selection boundary | Evidence |
| --- | --- | --- | --- |
| Decode-time fairness | Implemented | Native ordinary/MTP routes; bounded host-only counters | CPU policy and scheduler-turn tests |
| Adaptive native-MTP depth | Implemented, observed-used | Default-off; qualification-only | Two live Qwen3.6 requests; 81/98 boundaries and one depth change each |
| PLD rolling admission and verify-cliff planning | Implemented | Default-off; requires current-source qualification receipt | CPU policy/runtime tests |
| Cache capsules | Implemented, observed-used | Default-off; qualification-only ordinary decode | Exact final-source Muse GPU receipt on `bb88231` |
| APCv2 persistent blocks | Implemented, mechanism-smoked | Optional qualification path | GPU-backed 511-token restore from nine verified blocks |
| Rotating-cache undo/replay | Implemented, mechanism-smoked | Default-off compatibility transaction | Apple-GPU synthetic accepted-prefix replay |
| Spomin live KV surgery | Implemented, mechanism-smoked | Default-off; qualification-only B=1 ordinary route | Apple-GPU synthetic Qwen4 attention parity; no live model selection claimed |
| Approximate KV quantization | Implemented, CPU-qualified tiny-model path | Default-off; adapter and live-route qualification required | Revision-bound Q8 and K8V4 cache conversion, quantized attention/batch merge, warm-prefix private requantization; no GPU/model selection claim |

No feature in this campaign is claimed as a performance win or promoted to a
production-selected route. `implemented`, `qualified`, `selected`, and
`observed-used` remain separate states.

## Imported ideas and mlx2 adaptations

### Scheduling and concurrency

- OMLX supplied the strongest concrete scheduler mechanism: wall-time decode
  debt plus contention-aware prefill capping. mlx2 reimplemented this as a
  model-neutral, host-only policy and added route identity and receipts.
- Rapid-MLX supplied the adaptive MTP depth-controller idea. mlx2 made depth
  changes cohort-wide and boundary-atomic, added an exact depth-zero ordinary
  park, and bounded depth-one re-entry probes.
- `mlx-lm-unified` supplied PLD rolling-admission and verify-cliff ideas. mlx2
  retained per-lane target verification, exact rewind/replay, and current-source
  qualification gating rather than claiming cross-request target batching.

### APCv2, cache capsules, and copy-on-write fanout

- The capsule lifecycle comes from `mlx-lm-unified`: generation-stamped
  capture, staged external construction, late-result disposal, and leases.
- mlx2 binds capsules to model, source revision, adapter, tokenizer, cache
  layout, semantic identity, and runtime layout. Fanout attaches atomically or
  falls back.
- Mixed caches are supported without pretending every plane is a capsule:
  exact plain-KV planes use capsule construction while rotating planes use
  their ordinary merge. One transaction reserves the entire Bn result before
  either path materializes, every plane claims its byte share, and the
  reservation survives until the last scheduler consumer.
- Latest-main APCv2 hardening is preserved: each sibling first leases the exact
  committed boundary. A capsule is an optional replacement after that proof,
  not a way around the boundary contract.
- Persistent APCv2 storage can use checksummed blocks with contained paths,
  save identity, corruption detection, and orphan cleanup. Whole-file snapshots
  remain available when block persistence is disabled.

### KV surgery and approximate state

- `mlx-lm-unified` supplied the rotating-cache rollback/replay and Spomin
  mechanisms. mlx2 made rotating rollback multi-cache atomic and kept Spomin
  behind drained-work, revision, cache-privacy, B=1, unquantized-QSA,
  inactive-MTP, default-RoPE, and recurrent-state refusal checks.
- Rapid-MLX K8V4/TurboQuant work was used as a design input. mlx2 now owns
  concrete `kv_q8` and `kv_k8v4` operations for eligible Qwen3.6/Qwen3.8
  adapters, implemented through the cache classes' quantized representation
  and attention path. Publication remains revision-bound and request-private;
  warm exact APCv2 hits are requantized into private branches, exact and
  approximate lanes cannot merge, and normal serving still requires matching
  qualification evidence. CPU tiny-model execution is not GPU/model-route
  qualification, and no broader TurboQuant mechanism is claimed.

## Technical assessment of the source projects

For scheduling fairness, OMLX had the most modern directly reusable design;
mlx2 now has the stronger qualification and receipt boundary around it. For
adaptive speculative depth and approximate-KV experimentation, Rapid-MLX was
the most aggressive and modern, while mlx2 is more conservative about
selection, state identity, and failure behavior. For cache lifecycle, cache
surgery, and branching fanout, `mlx-lm-unified` was the most technically
complete source. The resulting mlx2 implementation is now stricter in
transactional capacity accounting, exact APCv2 boundary proof, route receipts,
and fail-closed publication, but it does not claim the breadth of experimental
tensor kernels present in Rapid-MLX.

The ideas worth carrying forward are the same ones implemented here: wall-time
fairness instead of request-count fairness; cohort-wide adaptive speculation;
generation-bound cache objects with explicit ownership; full-footprint
reservation before fanout; exact rollback/replay; and adapter-owned approximate
operations. A future K8V4 or TurboQuant import should remain a separate tensor
qualification project rather than being implied by the control seam.

## Verification

### CPU

- Final post-main-merge suite: 924 collected, 20 configured skips, zero
  failures.
- Focused cache-capsule suite: 29 passed.
- Expanded cache/APCv2/serving/adaptive-MTP suite: 183 passed before the final
  main merge; the final full suite supersedes it.
- Changed-file Ruff `F,E9`, Python compilation, JSON parsing, and
  `git diff --check` passed.

An initial full-suite command used the system Python and failed collection
because `mlx2` was not installed there. It was rerun with the documented
project virtual environment and `PYTHONPATH=src`; this setup error is not
counted as a product failure.

### Apple GPU

- 18 explicit segmented-QSA/optional-epilogue Metal tests passed.
- Adaptive MTP was observed on two concurrent live Qwen3.6-35B requests with
  finite output, acceptance 0.455/0.521, and one depth change per request.
- On exact final implementation source `bb88231`, live Muse Glimmer B1-to-B2 fanout
  leased a 108-token committed APCv2 boundary for each sibling and engaged a
  52-plane mixed prepared cache: 13 capsule planes and 39 ordinary planes.
  Both siblings reported 108 reused tokens.
- APCv2 persistent blocks restored 511 logical tokens from nine 4096-byte
  blocks with matching byte counts and no restore failure.
- Synthetic Apple-GPU smokes passed for rotating replay, Spomin attention
  surgery, and the approximate-state control seam.

The exact Muse receipt is
`qualification/receipts/advanced-imports-gpu-20260918/muse-mixed-capsule-bb88231.json`.
All GPU work held the CPG exclusive lease plus `/tmp/gpu.lock` and
`/Users/Shared/mlxuag/gpu.lock`. The test server was stopped, the production
Qwen service was restored ready on port 8282, both locks were removed, and the
lease was released.

## Independent review

The in-repo integration auditor found three meaningful capsule issues during
development: an all-planes compatibility gate that prevented mixed caches,
misleading 52-plane wording, and missing reservation of ordinary mixed planes.
All three were fixed. A follow-up audit found no remaining blocker; its two
hardening suggestions (atomic claim locking and a real mixed BatchGenerator
receipt assertion) were also implemented.

The requested Claude Opus 5 medium review was explicitly waived by the user on
2026-09-18 after the external CLI remained unauthenticated. No Claude review
was performed, and this report does not imply an external-model verdict. The
completed in-repo review was performed by an independent Codex subagent.

## Operational state and residual risk

- Production service: restored and ready on port 8282.
- GPU locks: absent.
- CPG GPU lease: released.
- Worktree: isolated at `/private/tmp/mlx2-peer-advanced-imports-20260917`.
- Branch: `codex/peer-advanced-imports-20260917`.
- Exact tested implementation commit: `bb88231`.

Residual risks are deliberately narrow: the advanced routes remain mostly
default-off and qualification-only; the GPU runs establish mechanism
engagement and finite output, not throughput or quality parity; Spomin has no
live supported-model serving qualification; and approximate KV has no imported
tensor implementation.
