# Independent review: optional QSA promotion under memory pressure

Reviewed 2026-09-15, CPU/source only. Reviewed source freeze:
`873a986a504872b3558871e59996341e5540f077459bc1e48b00211f413a5ed4`.

**Outcome:** no remaining blocker in the bounded accounting/ordering review.
The reviewer independently ran 48 focused CPU tests and checked the actual
serving callback. This does not establish successful 262K cache retention,
GPU performance, or deployment readiness. The root-owned live rerun was
interrupted when the user requested a pause. No complete new qualification
profile or benchmark was produced; consult matching qualification receipts and
[results](../RESULTS.md) for subsequent evidence.

## Failure that motivated the change

The preceding combined source
`2ab2c2f8019c388cb81472997f6464f10a6c3a237719334e2b65c041c4840e7d`
passed its shorter API/batch checks but failed the near-262K warm-prefix check.
The cold request used 262,051 prompt tokens and generated 64 tokens in
200.136588 seconds. Repeating it took 206.273932 seconds. Both reported zero
cached tokens. Those are failure observations, not throughput qualification.

The saved failure and debug logs are outside the repository at:

`~/Library/Application Support/mlx2/experiments/serving-parity-2026-09-15/`

The relevant file is
`flash-next-combined-262k-promotion-pressure-failure.json`; the directory's
manifest records the archived evidence. Root preserved this failure before
rerunning the repaired source.

Source inspection explained a harmful ordering: an asynchronous physical-QSA
destination could already be allocated while still being charged in full as
future pending memory. The mutating admission callback could evict the saved
prompt checkpoint before optional promotion was canceled during lane migration.
The terminal recurrent state cannot simply rewind to reconstruct that lost
prompt boundary. This explains the observed miss, but only a successful live
rerun can establish that the repair resolves the complete workload.

## Reviewed mechanism

- `SegmentedPhysicalPromotionTicket.pending_bytes` reports the reserved future
  allocation until `settle_for_admission()` synchronizes its stream. Settlement
  preserves the ticket and authoritative segmented state; it does not publish
  the physical route. The total allocation remains recorded in its receipt.
- Pressure preflight uses a side-effect-free admission preview. It first
  settles the ticket and remeasures. If memory still constrains the route, it
  cancels/drains the optional copy and drops references before reclamation.
- The actual serving callback reads `execution_headroom()`: the minimum of
  available host memory and the recommended working set minus the larger of
  MLX active-plus-cached memory or process footprint. Settled destination bytes
  therefore remain part of the measured resident cost when pending bytes become
  zero. No hypothetical reclaimed bytes are credited.
- After cancellation, a nonblocking grace period lasts at most two seconds,
  with reclamation retries every 250 ms. Both the generator and outer serving
  loop preserve useful APC entries during this accounting-recovery wait. If
  measured pressure persists, normal admission resumes after the deadline.
- A ceiling-memory plan retains the same lane and verification-row caps.
  Comparing its modes/depths with the live plan distinguishes memory pressure
  from a capacity-only split. Capacity limits do not trigger the recovery wait.
- Pressure-path drain failures propagate while retaining the ticket, before
  reclamation or APC eviction. Normal membership changes retain their existing
  handling.
- The normal unconstrained path does not force settlement or synchronization.
  Async promotion remains enabled and eligible; memory decline is recorded per
  cohort. The real CPU ticket test settles, commits, and successfully finishes
  promotion, preserving the recurrent arrays and a valid promotion receipt.

## Independent counterexamples and fixes

### Capacity-only false memory pressure

Using the production callback/controller with the test batch fixture, set
`free_memory=1000 GiB`, `verification_row_cap=1`, and a constant 1 MiB cache
estimator. The first repair treated every non-`full` decision as memory pressure.
It settled and canceled the ticket, then returned no progress with a two-second
grace period despite a cap that no amount of memory could remove.

After the ceiling-plan fix, the same independent probe returns immediate normal
admission with no pressure settlement or grace. The preexisting callback may
clear allocator scratch; a normal membership change cancels its incompatible
ticket. The regression separately covers a two-lane saturation cap of one.

### Drain error before reclamation

The existing cancellation helper logged and swallowed a drain failure. For the
new pressure path that could allow accounting/reclamation to continue after
unsuccessful cleanup. The owner added strict cancellation for that path. An
injected drain exception now propagates; the ticket remains owned and no
reclamation or eviction callback runs.

### Old-order checkpoint eviction

The pressure tests deliberately call the old mutating admission order with a
21 GiB headroom bank and an 8 GiB optional allocation. That counterfactual evicts
a checkpoint. The repaired path settles/cancels/reclaims first and admits the
lane without that eviction. This is a controlled callback oracle, not a claim
that the same byte values describe a real GPU allocation.

## Verification and limits

Independent command, setting CPU before importing the test modules:

```bash
.venv/bin/python -c 'import mlx.core as mx; mx.set_default_device(mx.cpu); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_promotion_pressure.py", "tests/test_segmented_physical_promotion.py", "tests/test_admission_progress.py", "tests/test_execution_policy.py", "tests/test_flash_next_memory.py", "tests/test_serving_retention.py"]))'
```

Result: **48 passed**. The focused set checks normal overlap, settlement without
cancellation, cancellation before eviction, delayed host-accounting recovery,
bounded timeout, capacity-only limits, drain failure, real tiny CPU ticket
promotion, admission progress and idle cache retention. `git diff --check` also
passed for the changed implementation/tests. The independent reviewer edited
no source or tests and loaded no production weights.

The implementation owner's full-suite log records **392 CPU tests passed, 16
Metal tests skipped, and 59 subtests**. This is owner execution evidence,
separate from the independently rerun subset. The final log is archived as
`cpu-promotion-pressure-full.log`. The preceding capacity-test expectation
failure (391 passed, one failed) is preserved separately as
`cpu-promotion-pressure-full-intermediate.log`; both hashes are recorded in the
refreshed archive manifest.

The synthetic accounting tests establish ordering and ownership under controlled
measurements. Tiny real CPU cache tests establish the ticket's state transition.
Neither proves device allocation timing, long-context host recovery latency,
actual warm-cache reuse, observed GPU promotion engagement, or performance.
Those require the matching live qualification and benchmark receipts.
