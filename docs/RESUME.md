# Resume mlx2 qualification

Updated **2026-09-17** after the advanced-runtime integration and CPU bug
sweep. The published source workset is closed at the head below; GPU
qualification and route selection remain separate work.

## Frozen source state

- Published source head: `5b1b341` (`main` and `origin/main` at this update).
- The `5b1b341` CPU-only bug-sweep commit records a full CPU-suite pass. Its
  fixes include fail-closed MTP warm-hit handling, non-atomic APCv2 fanout
  publication, concurrent parallel-sample draining, resilient APCv2 stores
  and restores, and stricter route/qualification gates.
- The shared-QSA heterogeneous warm-prefix B4 regression is fixed. Equal top-k
  shapes with different block grids remain row-local; uniform grids aggregate.
- Structured output now rejects unsupported schema keywords, bounds regex
  matching and request-local prefix caching, and requires a qualified grammar
  capability outside candidate mode.
- Parallel `n` sampling reserves the whole cohort before publishing jobs.
- Distribution archives include retained provenance notices and MIT licenses.

These checks prove source consistency and CPU contracts. They do not qualify a
full-model route.

## Preserved candidate evidence

**Qwen3.6 decision 2026-09-18.** The same-artifact ordinary reference is now
recorded ([experiment](experiments/QWEN36-MTP-VS-ORDINARY-2026-09-18.md)):
native MTP is 0.35–0.63x the ordinary route at every width from B1 to B20 at
k=1 and k=2, with 1–3 GiB per-lane speculative transient. The ordinary APCv2
route is the Qwen3.6 candidate to select; no MTP uplift claim is possible. The
earlier B14+ capacity block was host state (a resident production service
left ~24 GiB headroom); with the service quiesced B14/B16/B20 admit and the
controller's 1.76 GiB/lane transient constant matches the measured 1.73 GiB.
Current-source qualification receipts exist for both the MTP2 and
prompt-lookup routes at 32K (`qualification/runs/qwen36-35b-a3b/sanity-b-20260918/`).

Qwen3.6 35B-A3B has bounded earlier-source evidence for strict artifact load,
full-versus-split replay, ordinary/APCv2 HTTP, 32K and 64K context, warm B2
native MTP, and direct fused-GDN equality trials. The receipts under
`qualification/runs/qwen36-35b-a3b/` retain their original source identities.
No receipt matches the frozen source above, and no Qwen3.6 route is selected.

The Flash-Next repaired 262K rerun was interrupted before completion. The
historical 16K profiles and benchmark remain archived evidence for their exact
older identities only.

## Next qualification sequence

1. Read the Qwen3.6 port report and inert overnight tracker, or the Flash-Next
   parity report, depending on the target.
2. Acquire the exclusive GPU lease and filesystem lock. Record service state,
   runtime/native identity, artifact fingerprint, settings, swap and thermal
   baseline before loading a model.
3. Start one candidate server with explicit policy and dedicated APCv2 storage.
4. Run the strict qualifier. Grammar-capable routes must pass the observed
   strict JSON-schema check. Preserve partial and failed receipts.
5. Run alternating five-round B1/B2/B4 benchmarks only for matching qualified
   routes. Require observed compute widths rather than HTTP concurrency.
6. For Qwen3.6, add the same-artifact ordinary reference before attributing
   uplift to native MTP. Complete near-limit, rollback and long-duration gates.
7. Install or deploy only an exact matching successful receipt, then verify the
   persistent service identity and client smoke. Restore every quiesced service
   and release both lease mechanisms.

Prompt lookup is now wired into the shared serving lifecycle as an explicit,
target-verified route. It is fail-closed behind adapter capability,
qualification and route-selection gates, emits proposal/rollback receipts,
and remains **unqualified and unselected** for production models. APCv2
one-prefill fanout and the bounded host prompt cache are likewise implemented
and CPU-tested, but neither implementation status nor an older receipt proves
current-source GPU qualification or deployment. Static review found that the
fanout path could release siblings and emit `one_prefill=true` without proving
every sibling acquired the committed APCv2 boundary. It now pre-acquires and
retains every sibling branch before publication, fails the group closed on a
store or lookup miss, and derives `one_prefill` from that attestation. Prompt
lookup admission also charges the `num_draft + 1` target-forward width. These
repairs are CPU-tested but still need current-source GPU qualification. The
fanout attestation now also requires exact coverage of every token in the
published committed boundary; an `N-1` prefix hit cannot claim one-prefill
reuse. Prompt-lookup policy now rejects boolean, string, fractional, zero, and
negative `num_draft` values before readiness, and the validated policy object
is shared by qualification settings and runtime construction so receipts
cannot describe a different draft depth from the one executed. The same strict
contract now covers n-gram geometry, hot-segment and rejection bounds,
lookback thresholds/ladders, retrieval segments, and adaptive fallback values;
NaN, strings, booleans, fractional integers, invalid ranges, and malformed
segments fail before readiness instead of being silently coerced at admission.
The host prompt cache now bounds insertion work as well as retained storage:
disabled caches do not consume token iterables, and enabled caches inspect at
most `max_tokens + 1` items before rejecting an oversized value. Separate
`disabled_skips` and `oversize_skips` counters preserve the reason in status.

**Superseded 2026-09-18 (later the same day) — Spomin is connected to the
request lifecycle.** The decision below was correct for the Qwen4-only backend;
it is reversed by giving the manager an adapter-owned backend and a second
backend for standard attention (`runtime/spomin_standard_surgery.py`: stock
`KVCache` full layers, `RotatingKVCache` sliding layers, per-layer `nn.RoPE`
read from the module, NoPE layers untouched). Muse-Glimmer and North-Mini-Code
declare it; Flash-Next/Qwen3.x keep the Qwen4 backend, which still refuses
hybrid state. Lifecycle changes: `--spomin-live-surgery JSON` (qualification
mode only), manager status in `/v1/status`, the one-lane restriction replaced
by an isolated B=1 prefill boundary with normal multi-lane decode, a protected
leading segment (attention sink), and two fixes found on the way: compacted
state was being published into the exact APCv2 prefix cache at both store
sites (now skipped, `apcv2_store_skipped_approximate`), and the generator
swapped batch caches for extracted single-row caches after a transform, so a
compacted lane could not decode. State: implemented, default-off and
**CPU-qualified on a tiny North fixture; production-model/GPU qualification is
still absent** — selectable only with `feature_spomin_surgery` observed by the
qualifier's `spomin_surgery` check. The isolated prefill now snapshots and
publishes the exact pre-surgery boundary under the original full prompt before
the compacted lane enters decode. The approximate compacted boundary and
completion remain excluded from exact APCv2, while repeated identical long
prompts can reuse the exact shadow and undergo a fresh private surgery.
Snapshot/store failures are bounded counters. A warm APCv2 prefix no longer
declines surgery: the earlier GPU run showed the shared chat-template prefix
made every request after the first ineligible, and the boundary cache is
request-private by construction.

**Decision 2026-09-18 (earlier) — Spomin stays a lab mechanism.** Its preconditions
(B=1, unquantized QSA, native MTP inactive, request-private cache, no
recurrent/PLE state) exclude every model this project serves: Flash-Next is a
GDN+PLE hybrid and Qwen3.6 is a GDN hybrid, so the manager refuses before an
epoch exists. Connecting it to the request lifecycle would add plumbing with no
reachable route. It remains default-off with its focused Metal evidence;
revisit only with a dense unquantized-QSA Qwen4 target and a compacted-history
repair for hybrid planes.

Revision-bound Spomin/Qwen4 live surgery is implemented as a default-off
manager and cache backend with exact transcript, request-private cache, epoch,
barrier and MTP-inactive guards. Its focused tiny-model Metal evidence does not
make it a normal serving route: the manager is not yet connected to the shared
request lifecycle, so it is neither selectable nor observed-used. QSA MTP
amendment counters are exposed through adapter runtime facts and tested as
bounded/resettable observability; `calls` is partitioned into no-ops,
amendments, and failures so missing captured state or block-grid rewinds cannot
silently disappear from diagnostics. These counters remain evidence, not a
route-selection claim. The manager now refuses hybrid recurrent/PLE state
before allocating an epoch and turns expected stale-revision, invalid-plan,
capability, and backend failures into declined receipts at the transaction
boundary. Closing an
unapplied epoch now records one terminal `closed_without_apply` receipt, while
the transaction publishes that same terminal receipt to its caller; stale or
repeated closes remain no-ops. Replaying `apply` after a terminal outcome now
returns the original same-epoch receipt without rewriting an applied or
refused result as `stale_epoch` or incrementing a second terminal counter. QSA
amendment snapshots, resets, and
multi-counter updates are serialized so runtime facts cannot expose torn
counts or lose updates under concurrent status requests.

On Qwen3.6 the fused GDN kernels remain opt-in and unselected until their
extended matched rounds and serving capability/receipt work are complete. The
Flash-Next candidate profile selects fused GDN decode/verify and compact replay
rollback, and its qualifier requires each to be observed.

Keep implemented, qualified, selected and observed-used states separate. See
[serving](SERVING.md), [results](RESULTS.md),
[Qwen3.6](ports/QWEN36-35B-A3B.md), and
[the qualification tracker](ports/QWEN36-35B-A3B-OVERNIGHT-2026-09-16.md).
