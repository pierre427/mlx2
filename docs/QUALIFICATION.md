# Qualifying a model

This is the procedure for qualifying a model route for serving. The current
correctness verdict and context-ladder producers are in
qualification/runs/qualify-1010-correctness/. Older qualify-* run directories
preserve their campaign-specific scripts and evidence; use them as historical
records, not as the current policy or a mutable source template.

## What qualification means

Qualifying a route answers three questions, in order:

1. **Does it load and decode properly?** Preflight and the ordinary route's
   smoke.
2. **Do the default options work?** Smoke on every route the model ships,
   plus the MTP handoff gates for MTP routes, where handoff is on by default.
3. **Does the route preserve expected behavior across context and cache
   conditions?** The context ladder completes all three repetitions in every
   cell, every needle is found, warm requests hit APCv2, and cold/warm answers
   agree. Throughput is assessed separately and cannot qualify or disqualify a
   route.

A route is **qualified** when all three hold on a pinned snapshot, on the
host that will serve it. A passing unit test, a smoke run on its own or a
fast benchmark does not qualify anything.

Qualification is a statement of confidence, **not permission to run**. An
unqualified model can still be loaded and served; we just cannot say for
sure that it runs properly, and its status must say so (see
[AGENTS.md](../AGENTS.md)). The server has three modes:

| Start with | `/v1/status` `qualification` | Route |
|---|---|---|
| `--qualification <receipt>` | `qualified` | the receipt's profile and qualified capabilities |
| nothing | `unqualified` (and a startup warning) | the adapter's declared capabilities |
| `--qualification-mode` | `candidate` | declared capabilities, plus test-only extras (fault injection, failure capture, candidate policies) |

Every response's route receipt repeats the state. Unqualified runs include
the exact default mechanisms (MTP ordinary handoff, adaptive MTP depth) and
every registered model, including multimodal routes such as Gemma 3n, Gemma 4,
MiniCPM-O, LFM2.5-VL, SmolVLM2, and Qwen2.5-VL, which remain labelled
unqualified until each exact route has a source-bound model-path qualification.
Native source-bound candidate producers have been reviewed for all six media
families: Gemma 3n, Gemma 4, MiniCPM-O, SmolVLM2, LFM2.5-VL, and
Qwen2.5-VL. None has completed fresh GPU model-path qualification for this
campaign. Candidate receipts do not select or qualify a serving route, and
model-grid smoke evidence is not model-path qualification. Deliberate exceptions: opt-in experiments (cache capsules, block
persistence, live Spomin surgery, unverified kernels, MTP acceptance logging,
and the opt-in prefill candidates: `tensorfold_prefill`, `gdn_prefill_chunk`,
`invariant_prefill`, `varlen_dense_mlp`/`varlen_sparse_moe` and
`external_varlen_prefill`) stay with `--qualification-mode` or a matching
receipt, and approximate operations (approximate KV, a neural semantic
bridge) still need `--qualification-mode` or a receipt that carries their
evidence. A refusal for a missing dependency (an unpinned mlx-vlm, an absent
drafter) is not a qualification gate.

Qualification is **not performance testing**. It runs on a working machine:
swap and thermal noise are recorded, not disqualifying, and its throughput
numbers are not benchmark-grade. See [Performance testing](#performance-testing).

## What a qualification binds

Every receipt records `runtime_identity()`: a SHA-256 of the mlx2 source, of
the native mlx build, and the Python and macOS versions, plus the artifact
fingerprint and serving settings. Consequences:

- A receipt qualifies **this host's runtime only**. Another host, a
  different mlx wheel, or a macOS update produces a different identity and
  needs its own qualification. Performance numbers never transfer.
- **A receipt applies only when its declared source scope and every recorded
  identity field match.** The full source digest changes when src/ changes, so
  the snapshot-wide preflight must be rerun after a source edit. A route-scoped
  source hash can remain unchanged when an unrelated src file changes; reuse
  route evidence only after checking its declared scope and all other bound
  fields against the new snapshot. Freeze the source, then run a fresh
  interpreter-bound preflight and the qualification gates required for that
  identity. Reconcile changes to tests, scripts, native build, interpreter,
  host, artifact, route, or serving settings against their recorded bindings
  before reusing evidence.
- `scripts/qualify_serving.py` is pinned by hash in
  `src/mlx2/qualification.py` (`APPROVED_QUALIFICATION_HARNESS`), and the MTP
  benchmark is pinned in `qualify_serving.py`
  (`APPROVED_ADAPTIVE_BENCHMARK_SHA256`). A receipt may not authorize its own
  producer; see [Changing the harness](#changing-the-harness).

## Before you start

1. **Own the GPU.** Take the shared lock per job through `gpuq.sh` (waiters
   protocol: `/Users/Shared/mlxuag/gpu.lock` plus
   `gpu.lock.waiters/`). Never hold it around a whole batch.
2. **One model resident.** Stop every other model server first, including
   launchd services: `KeepAlive` services restart on kill and must be
   `launchctl bootout`. Check `ps` for foreign `mlx2.server`, `mlx_lm`,
   `mlx_vlm`, `rapid-mlx`, `llama-server` processes.
3. **Check memory headroom.** Record vm_stat Swapouts before and after each
   measured run. Swap, foreign GPU activity, and post-run thermal throttling
   are environmental evidence: they do not fail functional qualification
   unless they cause an admission failure or a correctness gate fails. They
   make the run ineligible for performance claims. A model plus its ladder
   cache must fit; if it does not, lower the cache or context and record that
   choice.
4. **Tell other sessions** if the run holds the GPU longer than about 55
   minutes in one job.

## Step 1 — Pin a snapshot

Qualify from a frozen checkout, never from a working tree, so that `main`
can keep moving.

```bash
SHA=<commit>                       # full SHA on the qualify branch
D="${MLX2_QUALIFIED_ROOT:-$HOME/.local/share/mlx2/qualified}/${SHA:0:8}-snapshot"
git clone -q --no-checkout "$MLX2_REPO" "$D"
git -C "$D" fetch -q <worktree-or-remote> "$SHA"
git -C "$D" checkout -q --detach "$SHA"
echo "$SHA" > "$D/SNAPSHOT_REVISION"
```

Do the work on a branch off `main` (`qualify/<sha>`), push it to the configured project remote, and
point the campaign at the snapshot with `MLX2_CAMPAIGN_ROOT="$D"`. When you
re-pin, `diff -rq` the new snapshot against the old one (excluding `.git`,
`__pycache__`) and confirm that only the files you meant to change differ.

## Step 2 — Preflight

```bash
python scripts/qualify_serving.py --preflight-only --output <run>/preflight.json
```

CPU-only: runs the full unit suite and validates artifacts, tokenizer and
configuration. Later steps reuse it (`--reuse-preflight`,
`--preflight-receipt`).

The receipt is bound to the runtime identity (the `src/` digest and the mlx
build), the harness and the pytest configuration, and records a per-file
digest of the inputs the unit suite reads besides `src/`: `tests/**`,
`scripts/**` (every file, not only `.py`: tests assert on
`scripts/fixtures/*.json` through the scripts that load them, and glob
`scripts/*.py`), `provenance/**` (records and the NOTICE files a qualifier
hashes), `native/**`, `qualification/**` except `qualification/runs/`
(plans, policies, top-level records, committed artifacts a test round-trips,
corpora, experiments, history, receipts), `docs/experiments/**` and
`docs/PROVENANCE.md` (`PREFLIGHT_TREE_PATTERNS` and
`PREFLIGHT_TREE_EXCLUDED_PREFIXES` in `qualify_serving.py`).
The identity's `test_source_sha256` is the aggregate digest of that tree; the
cross-host receipt binds it.

The receipt also binds the interpreter that ran the suite. Its identity
records the executable as invoked, the implementation and the version
(`interpreter_identity()`: `sys.executable`, `platform.python_implementation()`,
`sys.version_info[:3]`), and the validator compares them with the Python it
runs under, never with a value read from the receipt: implementation and
version exactly, executables with symlinks resolved on both sides (a venv's
`python`, `python3` and the binary they point at are one interpreter; a path
that no longer exists is refused). The head of every command the receipt ran
(the full receipt's two lanes, a delta's lanes and its base) must be that
executable. The arguments after the head only say what pytest was asked: a
command headed by `true` with pytest's arguments exits 0 and runs no test, so
such a receipt is refused. A receipt written under another interpreter needs
a full preflight under this one. The cross-host (v3) receipt is validated on
the target host: the target proof's interpreter identity and command head
must be the validator's own, and the source proof's interpreter identity and
command heads are compared exactly with a source interpreter identity the
caller supplies independently (`source_interpreter`, what
`interpreter_identity()` prints on the source host under the venv that ran
the suite), since the target cannot resolve paths on the source host.
Campaign evidence under `qualification/runs/` is
left out by design: a campaign writes there and must not invalidate the
receipt, so the few tests that assert on frozen run evidence are outside the
binding and an edit to frozen evidence needs a full preflight by hand. The git
revision is recorded but not bound, so a commit that changes none of the bound
files (docs, campaign evidence) keeps the receipt valid. When only bound
non-`src/` files changed, do not rerun the full suite: write a delta that
reruns only the test modules the change can reach, and pass the delta as the
preflight receipt:

```bash
python scripts/qualify_serving.py --preflight-delta \
  --base-preflight <run>/preflight.json --output <run>/preflight-delta.json
```

The delta reruns every changed or added test module, plus every test module
the change can reach: a file is named by its module stem for a `.py` file
(tests load scripts by path and helpers by import) and by its file name
otherwise, and every `.py` file in the tree whose text names it (a test
module, a test helper or a script) is a consumer. Consumers are followed to a
fixed point, so the selection is the transitive reverse closure: a changed
test module reruns the modules that import from it, and a helper or script
reached only through another helper or script still selects the tests at the
end of the chain. A module that walks a bound directory (`glob`, `rglob`,
`iterdir`, `walk`, `listdir` or `scandir`, with the directory's name on the
call's line or the two before it, as `test_no_hardcoded_home_paths.py` does
over `scripts/`) reads files under it without naming one, so it is selected
for each changed file there whose name matches the literal glob pattern it
uses, or for every file there when the walk has no literal pattern. The
selection is lexical, so a generic file name over-selects, and
the closure amplifies that (a script the harness names reaches the tests
that load the harness); that is safe, only slower. The validator recomputes
the changes and the impacted set from the base and the current tree, and
refuses a delta that skipped one; it also requires an integer return code 0
for every lane the delta ran (JSON `false` or a missing code is not one).

The delta fails closed instead of running nothing: a changed, added or
removed file that is not a test module and from which no chain reaches a
test module cannot be placed by the scan (a load it cannot see must not pass
with no test run), so the delta is refused and the full preflight is needed;
the validator applies the same rule, so such a receipt is refused however it
was written. A change under `src/`, to the mlx build, to `qualify_serving.py`
or to the pytest configuration moves the binding, and a `conftest.py` change
reaches every test; both the writer and the validator refuse those. Each
needs the full preflight.

## Step 3 — Smoke, per route

```bash
python <run>/run_campaign.py --phase smoke --reuse-preflight smoke-<model>-<route>
```

Each stage starts the route's server and runs: the feature smoke
(`feature_smoke.py`: text, streaming, tools, reasoning, grammar, APC), the
official OpenAI/Anthropic SDK checks, `qualify_serving.py`, an APC
persist-seed / persist-rescan cycle and an APC prefetch check. Run every
route a model ships (ordinary, MTP, prompt-lookup, DFlash2, ...).

Reading results:

- **MTP routes with handoff on by default fail `feature_mtp_ordinary_handoff`
  in smoke by design.** Only the adaptive benchmark (Step 4) can produce that
  evidence, and the qualifier deliberately fails the run there rather than
  skip the check. Every other check must pass.
- **Routes that select SRPT prefill scheduling run a forcing load**
  (`prefill_scheduling_forced_reorder`). SRPT is on by default for native
  MTP routes with `--max-lanes` of 4 or more, and `feature_prefill_scheduling`
  then requires the scheduler to have reordered a prefill. Ordinary traffic
  never queues a short prompt behind an older multi-slice one, so the
  qualifier builds that queue itself. One streaming *hold* request keeps a
  lane busy. An atomic `batch_cohort` of two tiny requests is queued next;
  the server never attaches a cohort beside live work, so it holds
  everything queued behind it. Then a *long* prompt (one `prefill_step`
  plus 512 tokens, with a nonce so APCv2 cannot serve it) and one *short*
  prompt are queued, each only after `/v1/status` `queue_depth` shows the
  previous request waiting. Closing the hold stream frees its lane, and on
  the next worker pass the cohort, the long prompt and the short prompt are
  admitted together. While the cohort decodes, other prefill waits; when it
  finishes, the scheduler picks between a long and a short prompt that are
  both fully queued. SRPT serves the short one first and the long one counts
  as overtaken. The check passes only if `prefill_scheduling_bypasses` plus
  `prefill_scheduling_bypass_forced` in `/v1/status` `scheduler` rose during
  the load. The result does not depend on timing: every step waits on the
  server's own counters, and the only retry is the cohort's 1 s staging
  deadline, before anything is released. An ordinary route that selects
  SRPT prefills two prompts per round, so the load queues a second short
  prompt there and needs `--max-lanes` of 5 or more; a route with too few
  lanes fails this check with that reason. The evidence is recorded under
  `prefill_scheduling_forcing` in `qualification.json`. Before 2026-10-01
  no receipt in the repository had ever passed `feature_prefill_scheduling`
  (Flash-Next Uncensored on `709bedf8`, for example, failed it).
- **Feature checks count this run's engagement.** Most `feature_*` checks
  read the growth of their counters from the qualifier's initial
  `/v1/status` to the final one, so a mechanism engaged at load or by
  earlier traffic on the same server does not pass. `apc_persistence`
  (restart-bound by design), `host_memory_signals` (a gauge) and
  `fp32_head_logits` (a load-time transform) are deliberate exceptions.
  `indexed_fused_merge` and `indexed_output_gate` have only a latched flag
  and no run counter: if the initial `/v1/status` already shows the flag
  set, the check fails, and its evidence gives the reason. Qualify them on
  a freshly started server, as the campaign drivers do.
- Before calling a failure a regression, read the raw model text: render
  with `/apply-template` and replay through `/v1/completions`. Known
  model-behaviour failures (not regressions): Gemma3n completions/audio/
  hermes, MiniCPM-o hermes, Flash-Next MTP2 / Laguna
  `tools_required_named_parallel` (near-tie parameter names), and the
  Muse-Glimmer 30B 8-bit qualifier `tools` check (declared, below).
- **Declared known model behaviour in the qualifier.** The qualifier stops
  at its first failed check, so a model-behaviour failure early in the run
  hides every later check. A failure Pierre has explicitly classified as
  model behaviour can be declared in `src/mlx2/known_model_behaviour.py`,
  bound to one check name, the exact artifact fingerprints the receipts
  record (`/v1/status` `artifact`; a speculative route's composite identity
  is its own entry), a failure shape, a reason and an evidence path. When the
  check fails in exactly that shape on that artifact, the qualifier records
  it as `"status": "known_model_behaviour"` with `"passed": false`, the
  declaration and the evidence, adds it to the receipt's top-level
  `known_model_behaviour`, and keeps running. The receipt passes only if
  every other check passes. The route loader re-verifies the entry against
  the declaration and the shape, and appends
  `;known_model_behaviour=<check>:<id>` to the route receipt, so it is never
  silently a pass. Any other failure of that check, or the same failure on
  another artifact, still fails and stops. Declared:
  - `muse-glimmer-30b-8bit-tools-auto-asks-for-city` (2026-10-08): check
    `tools` on `Muse-Glimmer-30B-mlx-8bit` (`e4f23559…`, ordinary and
    prompt-lookup) and its DFlash2 route (`116136e4…`). Shape
    `tools_auto_text_reply`: the auto response ends `stop` with text and no
    tool call, and the same request re-asked with `tool_choice: required`
    returns exactly one `weather(city=Toronto)` call (the round trip
    continues from that call). Evidence:
    [qualify-1007-extra](../qualification/runs/qualify-1007-extra/QUALIFIED.md).
    A drafter or composition-policy change moves the DFlash2 fingerprint and
    the exception stops applying there (fails closed) until re-declared.
- Smoke is functional: swap or foreign GPU activity during it does not
  invalidate a pass. Rerun only when a failure could plausibly come from the
  noise (a timeout, a 429 admission refusal).

## Step 4 — MTP routes: handoff benchmark, then handoff qualification

```bash
python scripts/benchmark_adaptive_mtp.py --model <model> \
  --mtp-ordinary-handoff-max-width 4 --max-context 131072 --cache-gib 16 \
  --output <run>/handoff/mtp-handoff-<model>.json
python <run>/server_jobs.py handoff-qualify <model>
```

The benchmark runs ordinary, fixed-depth MTP, and handoff arms at one, eight,
and sixteen streams. Token/state agreement, a valid handoff boundary, observed
handoff execution, and feature smoke are correctness evidence. Throughput is
reported separately and cannot fail the qualification gate. Keep the speed
comparison in the performance assessment; record the behavior evidence and
route receipt with the qualified source identity.

## Step 5 — Thermal context ladder

```bash
python <run>/server_jobs.py ladder <model> short   # 1K, 4K, 16K, 32K
python <run>/server_jobs.py ladder <model> long    # 64K, 128K, 256K
```

Run each part through `gpuq.sh` and `job_guard.py` (the orchestrator does
both). Models with a 32K ceiling run `short` only.

What a cell is:

- Context lengths 1K–256K up to the model's maximum. Up to 32K each length
  runs at 1 stream and at 4 concurrent streams.
- One warm-up, then **3 measured runs**. Each run sends a cold request
  (must not reuse cache beyond the chat template's fixed preamble) and a
  warm request (must hit APCv2), with needle-in-a-haystack prompts.
- **Admission** before each run: thermal state nominal or fair, battery
  ≤ 40 °C, virtual ≤ 45 °C, three settled samples 15 s apart, no pmset
  warnings (policy: `qualification/four-model-experiments.json` `thermal`).
- **After each run**: invalid only if two consecutive samples show throttling
  (state serious or worse, or a pmset warning).
- **Contaminated attempt** (retained in the report and retried up to twice):
  swap-outs rose, a foreign GPU-capable process was active, the cold request
  reused cache, or post-run throttling was seen. An admission timeout leaves
  the cell pending; it is not a contaminated attempt or qualification failure.

The ladder records performance and environmental observations, but these do
not determine qualification. Use the correctness-only verdict:

    python qualification/runs/qualify-1010-correctness/qualification_verdict.py ladder.json

A ladder qualifies when it finishes, every cell completes exactly three
safely admitted runs (nominal or fair under the campaign temperature and
settling policy), every stream and retrieval needle completes, cold and
warm answers agree, every warm request reuses at least 90% of its prompt
through APCv2, and no cell errors. A thermal admission timeout leaves the run
pending without a route verdict; a failed functional gate fails qualification.
Swap, foreign activity, and post-run throttle samples are recorded under
measurement_noise; they do not fail functional qualification. Do not use
throughput values to qualify a route.

### APCv2 replay and multi-lane near ties

APCv2 qualification keeps cache-state correctness separate from numerical
output equivalence. Revision, cache layout, prompt-boundary publication,
nonzero warm hits, stream completion, and width-one cold/warm token identity
remain strict. A state mismatch, a missing hit, or a width-one token mismatch
is a qualification failure.

At physical decode width greater than one, quantized or bf16 reductions may
legitimately flip a greedy choice at a near tie even when APCv2 restored the
correct state. Do not fail a route solely for such a row, and do not waive it
from text similarity. Localize the first divergent output token and capture
only that already-selected position; full per-token logprob serialization can
change scheduling, and later logits are downstream consequences.

A multi-lane cold/warm mismatch may be recorded as
`near_tie_equivalent` only when all of these hold:

- cold and warm share the complete token prefix before the divergence;
- their unordered top-two token sets at the divergence are identical;
- in both arms, the selected token and alternative differ by no more than
  0.5 nats (equality passes), matching the established ordinary-width/MTP
  numerical ceiling;
- both resulting continuations pass the model's functional oracle;
- every mismatched row is classified and no unexplained high-margin
  divergence remains.

Receipts must report `near_tie_equivalent`, never `exact_token_parity`, and
retain the raw token ids, logprobs, margins, physical width, cache receipt, and
control comparison. A row satisfying the complete classification contract is
a passing numerical-equivalence result, not a correctness failure, regression,
APCv2 failure, or qualification failure. It must not make its cell fail or be
counted in failed-cell or unexplained-mismatch totals. Exact token parity
remains a separate, stronger observation. An unclassified row or failed
contract gate remains a failure. A single classified row cannot qualify a
route retroactively; the complete replicated ladder and all other qualification
gates must still pass.

**Decision (2026-10-07, Pierre): near-tie flips are acceptable.** The 2026-10-06
sweep (finding SPEC-02) noted that this contract has no rate gate and no bound
on the cross-arm log-odds shift, so a small, systematic state perturbation
would first surface as a near tie and pass. That was reviewed and accepted:
the contract stays as written, and the control's divergence excess stays
diagnostic. State bugs are caught by the strict gates (width-one identity,
state and revision checks, warm hits, functional oracles), not by this row
classification. Do not re-raise SPEC-02 as a defect.

A same-artifact, same-serving-shape ordinary-versus-ordinary control should be
retained when available to describe attribution and mismatch rates. It is not
required for the numerical-equivalence classification and cannot turn an
otherwise valid near tie into a failure. A material excess remains a diagnostic
signal to investigate and report, not a correctness or qualification failure
without an independent state, functional, or high-margin gate failure.

Use `scripts/assess_apcv2_replay_equivalence.py` to evaluate the companion
evidence. It fails closed on missing identity or state checks, malformed or
post-divergence evidence, a width-one mismatch, or a margin above the ceiling.
When a matched control is supplied it reports the one-sided divergence excess
as diagnostic evidence without changing the equivalence result. Its `passed`
result is only a replay-equivalence gate and always carries
`qualification_claim: false`; the ladder verdict remains responsible for the
complete route qualification.

`job_guard.py` also watches the whole job for swap and foreign GPU activity.
`swap_policy.py` separates swap during measured runs (proven by each run's
own before/after counter) from swap during warm-ups and gaps; both are
recorded for anyone reading the numbers, neither blocks qualification.

**Chunk-phase worst case (approximate grid mechanisms only).** A mechanism
that compresses, evicts or selects KV on a fixed token grid of stride S can
keep or lose a needle depending on its offset modulo S. arXiv 2609.36322
reports swings of up to 40 points. The ladder cannot show this: its needle
sits in the prompt's last tokens, after all the filler. For such a
mechanism:

- place the needle inside the region it compresses;
- sweep the needle's start over every phase 0..S-1 (for S > 64: every phase
  that straddles a boundary, plus steps of S/8);
- gate on the worst phase at each context length, reported beside the mean.

Today this applies to live Spomin compaction (1024-token segments). Exact
mechanisms are unaffected: APCv2 checkpoint lattices, prefill chunks and
QSA's bit-exact paths change reuse or speed, never the answer. Per-token KV
quantization (`kv_q8`, `kv_k8v4`) has no token grid.

**Shared-KV and sliding-window codecs (future approximate adapters).** Bind
the codec, parameters, skip decision and state revision to the KV owner and
all its consumers, not independently to each consumer layer. Count owned
caches, not logical layers. Gate first prefill, continuation prefill and
decode at `window - 1`, `window` and `window + 1`, then eviction, snapshot
restore, accepted-prefix rollback and lane removal. Preserve absolute token
positions and the causal window through every transition. Verify at matched
resident bytes with teacher-forced agreement and the retrieval ladder; any
approximate operation must remain explicitly qualified and revision-bound,
outside exact APCv2 publication. No alternate prefix-cache engine is implied.
This is validation guidance, not codec implementation or qualification; see
the vLLM #40108 design-input entry in [PROVENANCE.md](PROVENANCE.md).

## Step 6 — Record the result

A route is qualified when all of these hold for the same exact source, interpreter, host, artifact, route, and serving settings:

| Gate | Evidence |
|---|---|
| Preflight | `preflight.json` |
| Smoke, every route | `results/smoke-<model>-<route>/qualification.json` (MTP: all but the handoff check) |
| MTP handoff (MTP routes) | `handoff/mtp-handoff-<model>.json` and `receipts/<model>-qualification.json` |
| Context ladder, short and long | correctness-only qualification_verdict.py -> qualified: true for each part, saved next to the evidence |

Add the model to the run's `QUALIFIED.md` with the snapshot SHA, the route,
and the paths above. List non-qualifying models with the reason. Commit the
evidence on the qualify branch and push it to the configured project remote.

To serve a qualified route at startup, point its launchd plist
(`~/Library/LaunchAgents/com.example.mlx2-*.plist`: `PYTHONPATH`,
`--execution-policy`) at the new snapshot, then `launchctl bootout` and
`bootstrap` it. Confirm `/v1/status` reports the expected `runtime` and
profile.

## Performance testing

Performance claims (throughput we quote, A/B comparisons, or regressions)
need a separate assessment using
qualification/runs/qualify-1010-correctness/performance_assessment.py.
It requires matching host, model, route, artifact, runtime, settings, and
context cells; exactly three repetitions per cell in nominal thermal state;
and no swap, foreign GPU activity, or post-run throttling in either ladder.
Functional ladders use safe nominal-or-fair job admission. For a controlled
performance ladder, pass `--performance-mode`; it requires nominal admission
before each measured cell and returns pending without spending repetitions if
nominal conditions are unavailable. A separate
performance policy may compare throughput with a reference. An environmentally
noisy qualification ladder remains usable for functional qualification when
its correctness and admission gates pass, but its throughput is not quoteable.
A qualification result never depends on the performance assessment.

## Changing the harness

If a harness script is wrong, fix it; do not work around it in the run:

1. Write a test that reproduces the bug and fails, then fix it.
2. Re-pin: update `APPROVED_ADAPTIVE_BENCHMARK_SHA256` in
   `qualify_serving.py` if the benchmark changed, then
   `APPROVED_QUALIFICATION_HARNESS` in `src/mlx2/qualification.py` if
   `qualify_serving.py` changed. Run the full test suite.
3. Commit on the qualify branch, land the same fix on `main`, cut a new
   snapshot, and rerun only what the fix can reach:
   - **Under `src/`, or `qualify_serving.py` / its pins:** the source
     identity or harness changed, so rerun preflight and every gate for the
     models you are qualifying.
   - **Other scripts or tests only** (for example `scripts/sdk_smoke.py`):
     the runtime identity is unchanged. Write a preflight delta (Step 2) and
     rerun only the gate that runs the fixed script. Receipts and ladders from
     the earlier snapshot stand.

Example: from `c48e4836` streams end with a usage-only chunk that carries no
receipt, and the MTP benchmark read its receipt from that chunk. Every
handoff benchmark failed "missing feature evidence" regardless of the
server. Fixed in `876b7719` (main `ee3ce701`).

Example: `feature_prefill_scheduling` was required on every SRPT route,
but the qualifier sent no load that could make SRPT reorder anything, so
the check could not pass. The prefill-scheduling forcing load
(2026-10-01) added a load that does, and changed the
`APPROVED_QUALIFICATION_HARNESS` pin. Every receipt produced by the earlier
harness has to be produced again.

Example: the declared known-model-behaviour path for the `tools` check
(2026-10-08, see Step 3) changed `qualify_serving.py` and the
`APPROVED_QUALIFICATION_HARNESS` pin (`a12f213a…` → `99a9ea18…`).

Example: `feature_verify_bitexact`, `feature_apc_inflight_prefix_wait`,
`feature_memory_preemption` and several older gates read lifetime counters,
so a qualifier run against a server that had engaged the mechanism earlier
passed without engaging it. They became run deltas (2026-10-08), and the pin
moved `99a9ea18…` → `8860803e…`. The same day `qwen38_fused_gdn` stopped
counting fused prefill chunks (a separate switch) as decode-switch
engagement: `8860803e…` → `9b76c3a8…`. Then the remaining adapter-diagnostic
gates (segmented, indexed, pooled and scatter QSA, PLE, Flash-Next fused GDN
decode, fused MoE, APC sessions) became run deltas too, and the two indexed
merge latches fail closed when already set: `9b76c3a8…` → `815f6e0e…`.

## Pitfalls seen in practice

- **Stopping a job by pattern kills the watcher too.** `pkill -f` on a
  string that also appears in your monitor's own command line kills the
  monitor. Match on a PID instead.
- **"done" in the orchestrator state is sticky.** A watcher that waits for a
  label to appear in `done` fires at once if a previous run already
  recorded it. Compare against the log offset from when you started.
- **Exit code 241 is a SIGTERM** (-15) passed up through `job_guard.py`,
  not a crash. Look for who sent it before triaging the model.
- **Don't launch jobs from a short-lived watcher.** A monitor that expires
  can kill the job it started. Launch detached in a new session with default
  signal dispositions (e.g. `subprocess.Popen(..., start_new_session=True)`)
  and watch separately. Not `nohup`: it sets SIGHUP to ignored in every
  child, which fails the preflight's
  `test_exit_trace.py::...[SIGHUP]` (2026-10-01).
- **Flash-Next swaps** at this ladder's cache size on this host. That does
  not block qualification, but its numbers are not performance-grade, and its
  262K cell was refused (429) — reduce the cache or the context for that
  cell.
- **Needle failures at long context are real quality failures**, not
  harness noise (Xing4.0 retrieved nothing at 64K and 128K).

### Route source identity (2026-10-09)

Serving may report a reviewed `runtime.source_scope`. Its source digest binds
the route's adapter/model closure, shared lifecycle/cache code, lazy imports,
reviewed dispatchers and owned data. `build_runtime` remains available for whole
build diagnosis. The qualifier independently recomputes the active scope for
its stability gate. Whole-suite preflight receipts still bind the complete
current build; a route digest cannot stand in for full-suite source coverage.

This changes the compatibility namespace and the approved qualifier digest.
Existing receipts are not migrated or relabelled as qualified. Unknown dynamic
dependencies keep whole-source binding. When source scope is uncertain, bind the complete source tree and rerun
preflight before reusing qualification evidence.
