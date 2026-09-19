# Qwen3.6 35B-A3B overnight qualification tracker

Window: **2026-09-16 evening through 2026-09-17 morning**
Status: **CPU/source preflight closed; ordinary B20 qualified; native MTP capacity qualified through B13 and fails closed above it**
Scope: the frozen candidate source and the two local Qwen3.6 artifacts.

This tracker is a run ledger, not a qualification claim. Keep implemented,
qualified, selected and observed-used separate. Do not install a receipt or
change a service default during an unattended run.

## Final evidence closeout — 2026-09-17

The final current-source CPU suite collected **731 tests and exited 0**. The
runtime remains APCv2-only: no legacy APC engine, compatibility fallback or
legacy serving product was added.

### Implemented

- APCv2 retains at least one committed prompt boundary per configured lane.
- Explicit batch cohorts stage atomically, are isolated by `(tenant_id,
  cohort_id)`, and release every reserved job/slot on timeout or failure.
- Cohort lane attachment is transactional. Decode begins only after the whole
  cohort attaches; any preparation or headroom failure rolls the group back.
- Scheduler admission is all-or-nothing for a declared cohort. It tries every
  lane at the requested draft depth, then every lane at a uniform lower depth,
  and otherwise fails the group with HTTP 429.
- The selected lower depth remains bound across generation-batch merges.
  Atomic cohorts cannot enter the generic starved-lane demotion to ordinary
  decode. Receipts expose requested depth, actual depth and admission stage.

These mechanisms are implemented and CPU-covered. GPU evidence below states
which widths are actually qualified.

### Exact-source qualified evidence

The final batch artifact is
[`qualification/runs/qwen36-35b-a3b/targeted-pass4/batch-stress-final-source-capacity.json`](../../qualification/runs/qwen36-35b-a3b/targeted-pass4/batch-stress-final-source-capacity.json),
SHA256
`d0e719282c21580cca7ce2afaffd5e9736ee7864cb9c6246fc628820be8c6af3`.
Its runtime source SHA256 is
`7e2cfadce9da37b635fbdee058a78459bcdb327483b89a02f7a435382c6d1a42`.

| Arm | Result | Median aggregate throughput | Qualification meaning |
|---|---:|---:|---|
| Product-artifact ordinary B20 | 20/20 pass | 356.584 tok/s | Exact-source B20 ordinary/APCv2 qualified. |
| M-artifact ordinary B20 | 20/20 pass | 390.184 tok/s | Exact-source same-artifact ordinary/APCv2 qualified. |
| Native MTP requested B20 | 0/20; every cell HTTP 429 | n/a | Correct atomic capacity rejection; B20 speculative throughput is not qualified. |

The clean speculative capacity proof is
[`qualification/runs/qwen36-35b-a3b/targeted-pass4/mtp2-b13-live-capacity-proof-adaptive.json`](../../qualification/runs/qwen36-35b-a3b/targeted-pass4/mtp2-b13-live-capacity-proof-adaptive.json),
SHA256
`69ffd988724a803df781439e56b7376aefa7cb992b152a461168f36eb5074dc4`.
It passed at actual compute width 13 with **169.585 tok/s** aggregate.
Every request remained `segmented_self_mtp`; adaptive uniform k1/k2 execution
was observed with nonzero draft activity and no ordinary fallback. Targeted
B14, B16 and B20 attempts all failed closed with HTTP 429. The exact-source
native-MTP capacity boundary is therefore **B13 pass; B14 and above not
admitted** for this artifact, policy and host state.

### Adjacent-source evidence and remaining gap

The thermal context ladder artifact is
[`qualification/runs/qwen36-35b-a3b/targeted-pass4/context-ladder-r4.json`](../../qualification/runs/qwen36-35b-a3b/targeted-pass4/context-ladder-r4.json),
SHA256
`eaa7843ffb104fe5e9ab6fe5f47bd5f056596d5a1a18b56feb0b964f5d285849`.
It passed all **36/36** cells at runtime source SHA256
`1f6ebf54af5ac05b20770bb0f4ab22e32cb04cecd943e34c4e7745d5fc73c766`.
That source immediately preceded the batch-cohort and atomic-admission changes.
Those later paths are inactive for its ungrouped maximum-width-four cells, so
the ladder remains strong adjacent-source evidence, but it is not an
exact-current-source hash qualification claim.

The remaining serving gap is native speculative B20 capacity. B13 is the
largest clean exact-source speculative width observed; B14/B16/B20 correctly
reject rather than splitting, silently lowering route semantics or falling
back to ordinary decode. Raising that boundary requires a separately qualified
memory/capacity improvement, not a weaker admission oracle.

## Source closeout addendum — 2026-09-16

The repository/source closeout is complete and published. This closes the CPU,
contract, provenance, packaging and handoff work that was ready in the worktree.
It does not mark any GPU qualification phase below as complete.

Frozen identities:

- Forgejo repository: private `user/mlx2`;
- source-closeout implementation head before this ledger addendum:
  `e79369574955b2fc079bb00b01789278116f30be`;
- runtime source SHA256:
  `ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`;
- local `HEAD`, `origin/main`, Forgejo `refs/heads/main` and the Forgejo UI
  were verified at that implementation commit; the documentation-only ledger
  update is published and verified separately.

Landed milestones:

| Commit | Closed work |
|---|---|
| `f41c481` | Enabled shared QSA on the parent MTP route. |
| `e2b8017` | Fixed heterogeneous shared-QSA aggregation. Equal-shape selections with different block grids remain row-local; a common grid can aggregate. Added both focused cases and retained the warm heterogeneous-prefix B4 regression. |
| `e990f85` | Added deterministic ordinary-decode demotion when all MTP lanes remain paused across the bounded starvation window. |
| `c70df74` | Added the Qwen3.6 35B-A3B adapter/model slice, strict artifact inspection, native-MTP detection, fused-GDN candidate geometry, prompt-lookup oracle/policies, probes, bounded receipts and mined-source provenance. |
| `6ff074d` | Added batch telemetry, qualification-only fault controls, bounded nonstreaming parallel sampling, structured output and compaction contracts. |
| `46fedb0` | Reconciled the README, serving/results/provenance documents and resume handoff with the final evidence boundary. |
| `e793695` | Removed tracker whitespace defects and completed publication formatting. |

Audit blockers closed before publication:

- structured output now requires the selected route's qualified `grammar`
  capability outside candidate mode;
- unknown JSON Schema keywords, invalid continuations and regex match-budget
  overruns fail closed;
- structured-output token pieces are reused per tokenizer and each request's
  prefix-result cache is bounded rather than retained by a process-wide method
  cache;
- the serving extra declares its direct `regex` dependency;
- `n=1..8` nonstreaming sampling reserves the complete cohort before publishing
  jobs and retains lane, inflight and physical-headroom gates;
- a compaction transaction cannot publish fidelity stronger than its qualified
  profile;
- Qwen3.6 replay/fused-GDN probes require the exact zero-difference predicates
  claimed by their documentation;
- fused-GDN admission binds Qwen4/Agnes and Qwen3.6 to their own architecture
  geometries;
- wheel and source archives include `qwen36-35b.NOTICE` and
  `LICENSE.unified-MIT` alongside the mined runtime code;
- the Qwen3.6 provenance record includes the candidate HTTP/APCv2 smoke, and
  the fused-GDN plan says that five matched rounds remain outstanding.

Closeout verification:

- 621 CPU tests passed, 16 Metal-only tests skipped and 59 parameterized
  subtests ran with no failures;
- focused structured-output, compaction-fidelity, parallel-admission,
  qualification-routing, Qwen3.6 geometry and shared-QSA suites passed;
- `compileall`, `git diff --check`, Qwen3.6 JSON validation, qualification
  harness trust-anchor comparison and the original-source Apple-header scan
  passed;
- `uv build` passed, and both built archives were opened and checked for the
  retained MIT notice/license files;
- ports 8296-8298 were free and no Qwen3.6 candidate server was running at
  source closeout.

No new full-model GPU job ran during this closeout. Existing Qwen3.6 receipts
remain bounded historical candidate evidence tied to their recorded earlier
source hashes. Current-source GPU qualification, route selection, deployment,
five-round B1/B2/B4 comparison, near-limit testing, prompt-lookup serving
integration and live compaction remain open below.

## Tonight's closure target

The overnight is complete only when every row below is either closed with a
receipt or recorded as a bounded blocker with an owner and exact next command.
Passing a short smoke does not close a qualification row.

| Workstream | State at start | Tonight's close condition |
|---|---|---|
| Ordinary/APCv2 serving | bounded candidate smoke passed | complete serving qualification plus the matched context and batch evidence below |
| Native MTP2 | bounded warm B2 smoke passed | observe real B1/B2/B4 target and draft execution, APCv2 paired-state restore, transactions, rollback and clean quiescence |
| Plain B1 versus speculative batching | not reported as one comparison | publish plain B1 and aggregate speculative B2/B4; add B20 only after the separate stress run passes |
| PLD | indexed oracle, adaptive lookback and exact verifier implemented; no serving route | finish the CPU serving slice and receipts, or leave a precise integration blocker; GPU measurements are allowed only if the route fails closed and CPU gates pass |
| Fused GDN decode | direct bounded correctness/timing pass | finish the five-round 512-step oracle; serving selection remains blocked unless capability policy and observed-use receipts are implemented and tested tonight |
| Context ladder | 32K and 64K bounded checks only | three runs per cell at 32K, 64K, 128K and near-limit, alternating qualified arms under thermal state 0 |
| Batch stress | short B2/B4 smokes only | separate 20 rounds x actual B20 run per qualified arm, without thermal gating |
| Shared-QSA regression | fixed and covered for heterogeneous and uniform block grids; full suite passed | rerun during a future frozen GPU campaign; any recurrence blocks serving qualification |

Plan-audit snapshot at 20:40 EDT: Python 3.12.13 and MLX
`0.32.2.dev20260915+2a817ad94` match; the full suite and the named shared-QSA
B4 regression passed; current JSON validated; ports 8296/8297 were free; both
artifacts existed; no candidate server or `/tmp/gpu.lock` was observed. This is
readiness evidence only. Recheck it after freezing the overnight source hash.

### Fifteen-minute supervisor loop

Every 15 minutes while work is active:

1. heartbeat the CPG worker and GPU lease, then poll and acknowledge agent-radio
   messages;
2. record the active phase/cell, last completed receipt and next command;
3. sample server readiness, queue/inflight counts, APCv2/COW leases, process
   footprint, swap and thermal state;
4. preserve any new partial receipt and confirm its source/artifact/settings
   identity;
5. stop or reassign work that has made no observable progress for two loops;
6. update the morning handoff ledger. The loop must never restart a failed arm
   with changed settings under the same receipt name.

## Run identity and safety gate

- [x] Freeze the source-closeout worktree at commit
  `e79369574955b2fc079bb00b01789278116f30be` and runtime hash
  `ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`.
  A future GPU campaign must record `git rev-parse HEAD`, `git status --short`,
  `git diff --stat` and the then-current runtime hash in every top-level receipt.
- [x] Confirm `.venv/bin/python` reports Python 3.12 and MLX
  `0.32.2.dev20260915+2a817ad94`. Read the MLX version through
  `importlib.metadata.version("mlx")`; the top-level `mlx` module does not
  expose `__version__`.
- [ ] Acquire the exclusive CPG GPU lease and create the matching
  `/tmp/gpu.lock`. Record campaign, lease id, generation, owner and purpose.
- [ ] Inventory active listeners and GPU-using services. Quiesce only services
  explicitly placed in this window; record their exact prior state so they can
  be restored.
- [ ] Record initial swap, process footprint, MLX active/peak memory, free disk,
  and thermal state. Use a dedicated APCv2 cache directory for each server arm.
- [x] Verify ports `8296` and `8297` are unused. If PLD becomes eligible, also
  reserve and verify `8298`.
- [ ] Set a hard morning deadline. Do not let a failed arm block restoration.

Global stop conditions:

- non-finite logits or output;
- any target-token, convolution-state or recurrent-state correctness mismatch;
- artifact, source, native-runtime or settings identity changes during an arm;
- thermal pressure above nominal for two consecutive samples, except during
  the Phase 5 batch-stress run where thermal data is diagnostic only;
- swap growth above 2 GiB from the campaign baseline;
- an admission invariant, cache revision, rollback, lease or worker failure;
- a server that loses readiness or cannot drain within 60 seconds;
- any leaked APC/COW lease after quiescence.

On a stop condition, preserve the partial receipt and logs, mark the arm
`FAILED`, stop that server, and proceed only to cleanup. Do not silently rerun
with different settings under the same receipt name.

## Artifact inventory

| Arm | Artifact | Intended route | Prior evidence |
|---|---|---|---|
| O | `~/mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit` | APCv2 ordinary reference | Strict load, split replay, HTTP cold/warm, structured JSON, physical `n=2`, 32K and 64K passed |
| M-O | MTP artifact below forced through `--ordinary` | Same-artifact plain reference for measuring MTP uplift | Not yet run; required because O and M are different products and quantizations |
| M | `~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp` | Native segmented MTP2 | Strict load, ordinary replay and bounded warm two-lane MTP smoke passed |
| G-O | ordinary artifact above | Opt-in single-token fused GDN trial | 32 matched steps, 960 calls, bit-exact state/logits, 1.077x warm speedup |
| G-M | MTP artifact above, ordinary trunk only | Opt-in single-token fused GDN trial | 16 matched steps, 480 calls, bit-exact state/logits, 1.107x warm speedup |

## Phase 0 — preflight without a model

- [x] Rerun the full CPU/Metal-aware suite against the frozen identity. The
  previously failing shared-QSA regression
  `test_actual_warm_heterogeneous_prefixes_form_true_b4_under_hybrid_budget`
  passed during the plan audit. If it regresses, preserve the failure and mark
  serving qualification blocked; do not treat it as an allowed failure.
- [x] Run the focused Qwen3.6, PLD, MTP, APCv2, admission, serving-contract and
  structured-output tests separately so their result is visible even if the
  full suite stops elsewhere.
- [x] Run `git diff --check` and `uv build`.
- [x] Validate all JSON plans and existing receipts with `python3 -m json.tool`.
- [x] Confirm no candidate server is already running.

Suggested commands:

```bash
cd ~/Desktop/mlx2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/pytest -q -p no:cacheprovider \
  tests/test_qwen36_35b_port.py \
  tests/test_prompt_lookup_core.py \
  tests/test_batched_mtp.py \
  tests/test_segmented_mtp.py \
  tests/test_apcv2_lifecycle.py \
  tests/test_admission_progress.py \
  tests/test_serving_contract.py \
  tests/test_serving_batch_features.py \
  tests/test_structured_output.py
git diff --check
uv build
python3 -m json.tool qualification/qwen36-gpu-plan.json >/dev/null
```

Phase result: `PASS`
Notes: 621 CPU tests passed, 16 Metal-only tests skipped, 59 subtests ran;
archives retained notices; Qwen3.6 JSON parsed; ports 8296-8298 were free.

## Phase 0.5 — CPU implementation closure during cooldown windows

This phase may run in parallel with thermal cooldown because it must not load
production tensors. Recompute the runtime source hash after any accepted edit;
all later GPU receipts must bind the new hash.

### Required tonight

- [ ] Add immutable product-ordinary, MTP-artifact-ordinary and native-MTP2
  execution-policy JSON files for Qwen3.6. Pin adapter choices such as proposal
  depth there. Add companion launch manifests that pin cache geometry,
  context, lane limits and every optimized toggle instead of relying on
  ambient environment variables.
- [ ] Add a Qwen3.6 experiment manifest for the alternating context ladder and
  the independent 20x20 batch suite. Validate it with
  `scripts/run_qualification_matrix.py --validate-only` before acquiring the
  GPU lease.
- [ ] Preserve the now-passing shared-QSA B4 regression and add a focused
  assertion that a requested physical B4 is observed as compute width 4.
- [ ] Extend `qualify_serving.py` or add a Qwen3.6 HTTP probe for JSON object,
  strict JSON schema, grammar, physical `n=2`, overload 429 with `Retry-After`,
  and populated TTFT/ITL percentiles, Jain fairness and per-tenant rates. The
  current generic qualifier does not close those rows.
- [ ] Fix the context-matrix runner before using it as evidence: alternate arms
  after each valid measurement, prove that each prime is cold, capture both
  cold and warm TTFT, sample swap, and implement the stated two-consecutive-
  sample thermal rule. Its current order runs all contexts for one arm first.
- [ ] Extend the batch-stress runner or add companion probes for distinct
  tenants/fairness, progress events, overload 429, latency percentiles and
  full-length nonempty comparable completions. Current B20 coverage proves
  warm APCv2, width 20, throughput and counter deltas only.
- [ ] Add a source/artifact/settings manifest writer so direct probes, serving
  qualification, context cells and batch cells share the same identity tuple.
- [ ] Separate the full unit suite from the live-model qualifier. Bind an exact
  passing preflight-test receipt into `qualify_serving.py` instead of launching
  the full suite after a 23 GiB model is resident; do not let test-side MLX work
  contaminate serving thermal or memory evidence.
- [ ] Add a closeout checker that fails on a live listener, inflight request,
  queue entry, APCv2/COW lease, unreleased GPU lease or stale `/tmp/gpu.lock`.

### PLD serving slice

Timebox this CPU integration lane to two hours before the first GPU serving
arm. Target-forward PLD is not currently wired and may be too large to finish
without displacing qualification. At the timebox, either continue from a
passing end-to-end CPU slice or mark it `BLOCKED` with the first missing call
boundary and exact tests. Oracle-only code is not a reason to delay ordinary
or MTP qualification.

- [ ] Connect `IndexedPromptLookup`, `AdaptiveLookback` and
  `verify_prompt_lookup` to target-forward generation behind an explicit
  Qwen3.6 execution-policy capability. Do not add a legacy APC or serving
  fallback.
- [ ] Bind PLD proposal state to the lane and cache revision. Exact target
  verification must own commit/rollback; stale markers, row migration and
  membership changes must fail closed.
- [ ] Compose PLD with APCv2 warm restore, layered attention/recurrent state,
  cancellation, segmented batching and scheduler admission. Preserve ordinary
  decode as the reference route.
- [ ] Emit route receipts and counters for retrieval cycles, proposed/accepted
  tokens, source kind, lookback changes, target verification width, rollback,
  ordinary fallback and actual compute width. A qualification arm must fail if
  the requested PLD mechanism counter is zero.
- [ ] Add CPU tests for full, partial and zero acceptance; bonus-token
  ownership; stale rollback markers; APCv2 restore; B2/B4 lane isolation;
  cancellation; deterministic sampling; and MTP/PLD policy incompatibility or
  explicit hybrid composition. Do not silently run MTP when PLD was requested.

PLD is eligible for a GPU arm only when these tests pass and the server exposes
an exact `ordinary`, `mtp2` or `pld` route receipt. If the slice cannot be
finished tonight, record the first missing call boundary and the tests needed;
do not report PLD performance from oracle-only code.

### Fused-GDN serving prerequisite

- [ ] Replace the environment-only trial switch with an execution-policy
  choice whose capability declaration, admission reason and counters appear in
  `/v1/status` and every terminal route receipt.
- [ ] Test B1 admission, multi-token/speculative refusal, reference fallback
  and zero-counter failure gates. Keep HTTP fused GDN out of the overnight GPU
  matrix unless these CPU tests pass before Phase 2 starts.

Phase result: `NOT RUN / PASS / PARTIAL / FAIL`
New source hash: __________________  Blocker ledger: ________________________

## Phase 1 — extended fused-GDN correctness and timing

Purpose: strengthen the promising single-token result before any serving
selection. This remains a direct model probe; it does not qualify HTTP serving
or speculative verification.

- [ ] Run five matched 512-step rounds on the ordinary artifact.
- [ ] Run five matched 512-step rounds on the MTP artifact's ordinary trunk.
- [ ] Require zero logit, convolution-state and recurrent-state difference,
  identical argmaxes, nonzero fused calls and zero fallbacks in every round.
- [ ] Report each round and median warm speedup; do not average away a slow or
  incorrect round.
- [ ] Record peak memory and thermal samples around every round.

```bash
cd ~/Desktop/mlx2
QWEN36_ORDINARY=~/mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit
QWEN36_MTP=~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp

for round in 1 2 3 4 5; do
  .venv/bin/python scripts/qwen36_fused_gdn_probe.py \
    --model "$QWEN36_ORDINARY" --steps 512 \
    --output "qualification/runs/qwen36-35b-a3b/overnight-fused-ordinary-r${round}.json"
done

for round in 1 2 3 4 5; do
  .venv/bin/python scripts/qwen36_fused_gdn_probe.py \
    --model "$QWEN36_MTP" --steps 512 \
    --output "qualification/runs/qwen36-35b-a3b/overnight-fused-mtp-artifact-r${round}.json"
done
```

Phase result: `NOT RUN / PASS / FAIL`
Ordinary median speedup: ______  MTP-artifact median speedup: ______
Notes/receipts: _____________________________________________________________

## Phase 2 — ordinary serving qualification

Start the server in a dedicated terminal. The baseline intentionally keeps
fused GDN, compiled decode and optimized MoE disabled.

```bash
cd ~/Desktop/mlx2
QWEN36_ORDINARY=~/mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit
.venv/bin/python -m mlx2.server \
  --model "$QWEN36_ORDINARY" --host 127.0.0.1 --port 8296 \
  --execution-policy qualification/policies/qwen36-ordinary.json \
  --max-context 262144 --max-lanes 4 --max-inflight 8 \
  --cache-bytes 12884901888 \
  --cache-dir ~/Library/Caches/mlx2/qwen36-ordinary-overnight \
  --ordinary --qualification-mode
```

Run from another terminal:

```bash
cd ~/Desktop/mlx2
.venv/bin/python scripts/qualify_serving.py \
  --url http://127.0.0.1:8296 --timeout 14400 --quiescence-timeout 180 \
  --output qualification/runs/qwen36-35b-a3b/overnight-ordinary-qualification.json

.venv/bin/python scripts/benchmark_serving.py \
  --url http://127.0.0.1:8296 --rounds 5 --widths 1 2 4 \
  --max-tokens 256 --timeout 7200 \
  --output qualification/runs/qwen36-35b-a3b/overnight-ordinary-benchmark.json
```

Required observations:

- [ ] cold and APCv2-warm responses with correct cached-token counts;
- [ ] streaming, cancellation and recovery;
- [ ] tools, reasoning, logprobs, grammar, JSON object and strict JSON schema;
- [ ] physical `n=2` plus actual compute widths B1/B2/B4 where requested;
- [ ] overload returns 429 with `Retry-After`;
- [ ] TTFT/ITL p50/p95/p99, Jain fairness and per-tenant rates populate;
- [ ] 32K, 64K, 128K and near-limit cold/warm cells, followed by over-limit
  rejection without losing readiness;
- [ ] cache/COW leases return to zero after quiescence;
- [ ] no optimized-mechanism usage is attributed to this stock arm.

Phase result: `NOT RUN / PASS / FAIL`
Qualification receipt: __________________  Benchmark receipt: _______________

## Phase 2.5 — same-artifact plain reference

The Phase 2 artifact is a different product and quantization from the native
MTP artifact. It cannot measure MTP uplift. After Phase 2 drains, serve the MTP
artifact with `--ordinary` and collect the same B1/B2/B4 workload as Phase 3.
This M-O arm is the reference for every MTP speedup claim.

```bash
cd ~/Desktop/mlx2
QWEN36_MTP=~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp
.venv/bin/python -m mlx2.server \
  --model "$QWEN36_MTP" --host 127.0.0.1 --port 8297 \
  --execution-policy qualification/policies/qwen36-mtp-artifact-ordinary.json \
  --max-context 262144 --max-lanes 4 --max-inflight 8 \
  --cache-bytes 12884901888 \
  --cache-dir ~/Library/Caches/mlx2/qwen36-mtp-artifact-ordinary-overnight \
  --ordinary --qualification-mode

.venv/bin/python scripts/qualify_serving.py \
  --url http://127.0.0.1:8297 --timeout 14400 --quiescence-timeout 180 \
  --output qualification/runs/qwen36-35b-a3b/overnight-mtp-artifact-ordinary-qualification.json

.venv/bin/python scripts/benchmark_serving.py \
  --url http://127.0.0.1:8297 --rounds 5 --widths 1 2 4 \
  --max-tokens 256 --timeout 7200 \
  --output qualification/runs/qwen36-35b-a3b/overnight-mtp-artifact-ordinary-benchmark.json
```

- [ ] Every counted request reports ordinary execution and no MTP/PLD use.
- [ ] Artifact, tokenizer, prompts, sampling, output budget and server settings
  match Phase 3 except for the requested route.
- [ ] B1/B2/B4 actual compute widths and clean quiescence pass.

Phase result: `NOT RUN / PASS / FAIL`
Qualification receipt: __________________  Benchmark receipt: _______________

## Phase 3 — native MTP2 serving qualification

Stop and fully drain Phase 2.5 before starting this arm. Reuse the port only
after the listener is gone, and use a different cache directory. The MTP
artifact should select its native MTP candidate in qualification mode; verify
`/v1/models` and the first terminal route receipt before continuing.

```bash
cd ~/Desktop/mlx2
QWEN36_MTP=~/mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp
.venv/bin/python -m mlx2.server \
  --model "$QWEN36_MTP" --host 127.0.0.1 --port 8297 \
  --execution-policy qualification/policies/qwen36-mtp2.json \
  --max-context 262144 --max-lanes 4 --max-inflight 8 \
  --cache-bytes 12884901888 \
  --cache-dir ~/Library/Caches/mlx2/qwen36-mtp-overnight \
  --qualification-mode
```

```bash
cd ~/Desktop/mlx2
.venv/bin/python scripts/qualify_serving.py \
  --url http://127.0.0.1:8297 --timeout 14400 --quiescence-timeout 180 \
  --require-feature segmented_transaction \
  --require-feature segmented_rollback \
  --output qualification/runs/qwen36-35b-a3b/overnight-mtp2-qualification.json

.venv/bin/python scripts/benchmark_serving.py \
  --url http://127.0.0.1:8297 --rounds 5 --widths 1 2 4 \
  --max-tokens 256 --timeout 7200 \
  --output qualification/runs/qwen36-35b-a3b/overnight-mtp2-benchmark.json
```

Required observations:

- [ ] target and draft forward counters are nonzero;
- [ ] committed segmented transactions and actual rollback evidence are nonzero;
- [ ] true batched widths B1/B2/B4 are observed, not inferred from HTTP concurrency;
- [ ] initial-token ordering, stop/EOS delivery boundaries and ordinary fallback
  remain correct;
- [ ] warm APCv2 restores both target and draft state without full-prefix
  materialization;
- [ ] starved MTP lanes demote to ordinary decode after eight idle boundaries;
- [ ] no request hits the 60-second progress failure under admissible load;
- [ ] all target/draft/COW leases return to zero after quiescence.
- [ ] proposal depth 2 is present in the route receipt and every request that
  contributes to the performance result; record acceptance and effective
  tokens per target forward.
- [ ] run a bounded proposal-depth-3 comparison only after MTP2 passes. Keep it
  a separate candidate identity and retain it only if exactness holds and the
  matched B1/B2/B4 result improves.

Phase result: `NOT RUN / PASS / FAIL`
Qualification receipt: __________________  Benchmark receipt: _______________

## Phase 3.5 — PLD serving qualification, conditional

Run this phase only if the Phase 0.5 PLD serving slice and all focused tests
pass. PLD gets its own execution-policy file, port/cache directory and receipt;
it must never reuse an MTP qualification identity.

- [ ] Ordinary and PLD outputs match under deterministic decoding for repeated,
  partially repeated and no-match prompts.
- [ ] B1/B2/B4 receipts show nonzero PLD proposals and exact target
  verification, with actual compute width greater than one for B2/B4.
- [ ] APCv2 cold/warm and restored-process requests preserve proposal history
  and cache revision without publishing approximate state.
- [ ] Cancellation and lane detach leave no rollback marker, cache lease or
  indexed-source ownership behind.
- [ ] Report proposed, accepted and bonus tokens, acceptance, target forwards,
  retrieval source, lookback changes and plain fallback separately.
- [ ] Fail the arm if every request falls back to ordinary decode or if a PLD
  receipt is absent.

Phase result: `NOT RUN / PASS / BLOCKED / FAIL`
Qualification receipt: __________________  Benchmark receipt: _______________

## Phase 4 — thermally controlled context ladder

Run only arms with a passing serving qualification bound to the exact current
source/artifact/settings identity. Required cells are 32K, 64K, 128K and the
route's near-limit prompt. Run three valid measurements per cell and alternate
arms after every valid measurement (`O`, `M-O`, `MTP2`, and qualified PLD).
Do not run all repetitions of one arm consecutively. Treat comparisons across
different artifacts as product comparisons; calculate MTP uplift only from
the matched M-O/MTP2 pair.

- [ ] Create and validate
  `qualification/qwen36-overnight-experiments.json` before starting.
- [ ] Require thermal state 0 and zero swap growth at both sides of every cell.
  A rejected preflight does not count as one of the three runs.
- [ ] Use identical tokenized prompts, output budgets and sampling controls
  across arms. Record the prompt-file hash and actual token count.
- [ ] Record cold and exact APCv2-warm TTFT, prefill tok/s, decode tok/s,
  end-to-end latency, cached tokens, peak memory and mechanism counters.
- [ ] Preserve the run order and all rejected/partial cells. Resume only passed
  identity-matching cells; never overwrite an attempt.
- [ ] Stop the ladder on two consecutive non-nominal thermal samples, more than
  2 GiB swap growth, a correctness mismatch or any lost readiness.

```bash
cd ~/Desktop/mlx2
.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest qualification/qwen36-overnight-experiments.json \
  --suite context \
  --output qualification/runs/qwen36-35b-a3b/overnight-context-ladder.json \
  --resume
```

Phase result: `PASS — 36/36 adjacent-source cells; exact-current-source rerun not performed`
Context ladder receipt:
`qualification/runs/qwen36-35b-a3b/targeted-pass4/context-ladder-r4.json`
(SHA256 `eaa7843ffb104fe5e9ab6fe5f47bd5f056596d5a1a18b56feb0b964f5d285849`)

## Phase 5 — independent batch qualification and 20x20

This is separate from the context ladder. Restart each passing arm with
`--max-lanes 20 --max-inflight 40` and a clean arm-specific cache directory.
Do not reuse the B4 server identity.

The 20x20 run intentionally has **no thermal admission, cooldown or alternating
arm controls**. Continue recording thermal samples as diagnostics, but do not
use them to select or reject individual measurements. Hard safety stops for
swap, correctness, lost readiness and leaked state still apply.

- [ ] Run O, M-O, MTP2, and qualified PLD as separate server processes.
- [ ] Require exactly 20 rounds of 20 simultaneous requests per arm.
- [ ] Require actual compute width 20 in every counted round. HTTP concurrency
  alone is insufficient.
- [ ] Record aggregate and per-request throughput, TTFT/ITL p50/p95/p99, Jain
  fairness, tenant rates, queue time, progress events and 429 behavior.
- [ ] For MTP/PLD, require nonzero mechanism counters and report acceptance,
  target forwards, rollback and ordinary fallback. Reject silent ordinary
  execution.
- [ ] Drain fully between arms and require zero inflight, queue and cache/COW
  leases before starting the next server.

```bash
cd ~/Desktop/mlx2
.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest qualification/qwen36-overnight-experiments.json \
  --suite batch_stress \
  --output qualification/runs/qwen36-35b-a3b/overnight-20x20.json \
  --resume
```

Phase result: `PARTIAL — O and M-O B20 passed 20/20; MTP B20 failed closed 0/20; targeted MTP capacity passes B13 and rejects B14+`
20x20 receipt:
`qualification/runs/qwen36-35b-a3b/targeted-pass4/batch-stress-final-source-capacity.json`
(SHA256 `d0e719282c21580cca7ce2afaffd5e9736ee7864cb9c6246fc628820be8c6af3`)

## Phase 6 — matched performance report

Only compare arms whose qualification receipt passed and whose identity matches
the current source, native MLX, artifact and settings.

Record separately:

| Metric | Product O plain | M-artifact plain | MTP2 | PLD, if qualified | Fused GDN direct probe |
|---|---:|---:|---:|---:|---:|
| Cold TTFT 32K | | | | | n/a |
| Warm TTFT 32K | | | | | n/a |
| Cold TTFT 64K | | | | | n/a |
| Warm TTFT 64K | | | | | n/a |
| Cold/warm TTFT 128K | | | | | n/a |
| Cold/warm TTFT near-limit | | | | | n/a |
| Plain B1 decode tok/s | | | n/a | n/a | |
| Speculative B1 decode tok/s | n/a | n/a | | | n/a |
| B2 aggregate tok/s | reference only | matched reference | | | n/a |
| B4 aggregate tok/s | reference only | matched reference | | | n/a |
| B20 aggregate tok/s | reference only | matched reference | | | n/a |
| Acceptance / target forwards | n/a | n/a | | | n/a |
| ITL p50/p95/p99 | | | | | n/a |
| Peak process footprint | | | | | |
| Peak MLX memory | | | | | |
| Swap delta | | | | | |
| Observed mechanism counters | stock only | stock only | MTP target/draft/transactions | retrieval/verify/rollback | fused calls/fallbacks |

Do not combine prefill, decode, cache reuse, TTFT or end-to-end throughput into
one speed number. Do not describe MTP or PLD-assisted throughput as native plain
decode. State which artifact backs PLD and compare it only with plain decode on
that same artifact.

Phase result: `NOT RUN / PASS / FAIL`
Report path: ________________________________________________________________

## Deferred unless Phase 0.5 closes the prerequisite

- **Fused GDN HTTP selection:** the kernel starts as a live-toggle/direct probe.
  It may enter the overnight matrix only if Phase 0.5 adds a capability-bound
  execution policy, status counters, receipt requirements and focused tests.
- **Fused GDN speculative verify/catchup:** Qwen3.6 currently falls back to the
  reference path for speculative or multi-token geometry. The direct MTP
  artifact probe exercised its ordinary trunk, not fused speculative verify.
- **Compiled decode replay:** not integrated for Qwen3.6 in `mlx2`.
- **Optimized MoE router/expert kernels:** not selected or qualified for this
  geometry.
- **PLD GPU performance:** oracle-only evidence is prohibited. It becomes an
  overnight GPU arm only after the target-forward serving, revision ownership,
  batching, APCv2 composition and receipt gates in Phase 0.5 pass.
- **Live Spomin KV surgery:** only the revision-bound compaction control plane
  exists; there is no live surgery or six-hour soak to run yet.

These are implementation prerequisites, not failed qualification cells.

## Morning closeout

- [ ] Stop the candidate server and verify the listener is gone.
- [ ] Wait for quiescence and confirm every lease/inflight/queue counter is zero.
- [ ] Release the exclusive GPU lease and remove the matching `/tmp/gpu.lock`.
- [ ] Restore every quiesced service to its recorded prior state and verify its
  actual listener, `/health`, and model identity where applicable.
- [ ] Record final swap, footprint, MLX memory, disk and thermal state.
- [ ] Validate every JSON receipt and preserve partial failures.
- [ ] Write a short evidence summary separating pass, fail, not-run and blocked.
- [ ] For every unresolved item, record its first missing call boundary, owner,
  exact next command, required environment and whether it needs a GPU lease.
- [ ] Reconcile CPG task state and agent-radio threads: acknowledge deliveries,
  release claimed tasks/resources, close threads and complete every worker.
- [ ] Do **not** install qualification or change the default route without an
  explicit reviewed promotion decision.

Overall outcome: `CPU PASS; ORDINARY B20 PASS; NATIVE MTP B13 PASS / B14+ CAPACITY BLOCKED; CONTEXT 36/36 ADJACENT-SOURCE`
Started: 2026-09-16  Finished: 2026-09-17
Operator: Codex closeout workflow
Source hash: `7e2cfadce9da37b635fbdee058a78459bcdb327483b89a02f7a435382c6d1a42`
(exact batch/B13); context-ladder source
`1f6ebf54af5ac05b20770bb0f4ab22e32cb04cecd943e34c4e7745d5fc73c766`
Restoration receipt: _______________________________________________________
