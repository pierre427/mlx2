# Sanity/qualification of the CPU-only ("B") items — 2026-09-18

Model: Qwen3.6-35B-A3B native-MTP artifact (`...-oQ4e-mtp`), candidate mode,
`--max-context 32768 --max-lanes 4 --max-inflight 8 --cache-bytes 2 GiB
--persistent-block-bytes 4096`, dedicated cache dir. The production
`com.example.fn-uncensored-mlx-serve` service was quiesced for the window
and restored afterwards.

## Receipts (source `e33e8087f4df…`, the tree after the two qualifier fixes)

| Receipt | Route | Result |
|---|---|---|
| `mtp2-32k-2gib-qualification-c.json` | `qwen36-35b-a3b-apcv2-mtp2` | **35/35 PASS** |
| `pld-32k-2gib-qualification-c.json` | `qwen36-35b-a3b-apcv2-ordinary-pld` (`{"prompt_lookup": {"num_draft": 8, "ngram_min": 3, "ngram_max": 6}}`) | **30/30 PASS**; observed `prompt_lookup=56`, proposals `448`, rollbacks `56` |
| `b-items-probe-mtp2.json` | targeted probes on the MTP2 server | see below |

Earlier attempts in this directory are kept as evidence of two qualifier
defects found and fixed during the window (they are harness fixes, not server
fixes): `mtp2-32k-1gib-*` failed `shared_warm_requests` because the qualifier
skipped priming the shared-cohort prompt whenever `max_context <= 131072`
(both cohort members were legitimately cold); `pld-32k-2gib-qualification.json`
failed `batch` because the check demanded an observed compute width >= 2,
which the prompt-lookup route never reports (it verifies per lane at width 1
by design) — it now accepts wall-time overlap of the four lanes for that route.
`pld-32k-2gib-qualification-b.json` failed only `runtime_stable` because the
harness hash was re-pinned while that server was running.

## Targeted probes (MTP2 server)

- **Scheduler-waiting watchdog:** a late arrival waited **78.5 s** behind a
  single-lane 5,554-token decode and completed 200 (`memory_admission_timeouts`
  unchanged). Before `b386750` this was a 429 "memory admission did not permit
  progress".
- **Decode-time fairness:** two ~19K-token prefills arriving during two active
  decodes: `decode_fairness_prefill_chunks +63`, `debt_deferrals +113`,
  `debt_repayments +176`; decodes finished in 22–23 s, prefills in 31.7 s, all 200.
- **APCv2 pressure under a 2 GiB cap with 4 KiB persistent blocks:** 13–14
  pressure spills, 7 idle spills, 4–5 restores, 0 restore failures, 0 budget
  deferrals, 0 store failures; warm hits after restore (`cached_tokens` 2059/2299).
- **Promotion-join short arrival:** two 400-token requests plus a `max_tokens=8`
  arrival mid-decode all completed; the specific reconciliation branch
  (`physical_join_flag_reconciled`) needs QSA async promotion, a Qwen4 path,
  so it was not reachable on Qwen3.6.
- **Host prompt cache:** 3 identical requests → 1 store, 2 hits (the
  `oversize_skips`/`disabled_skips` counters appear only when nonzero).
- **PLD copy task:** 60-line document reproduced verbatim, 600 tokens in 8.5 s
  (70 tok/s), 568 proposed / 119 accepted, 71 retrieval cycles.

## Defect found by this window (fixed after the receipts)

Three candidate servers survived SIGTERM as zombies holding 8–10 GiB each:
`Py_FinalizeEx → atexit → multiprocessing SemLock.acquire`. Idle
`multiprocessing.Pool` scanner workers hold the task-queue read lock while
blocked in `recv`; killed from outside, they left the parent's pool finalizer
deadlocked. The scanner pool now uses `ProcessPoolExecutor` and is shut down
explicitly from `ServingEngine.close()` and at exit; a subprocess test kills a
worker and asserts the interpreter exits. This fix changes the runtime source
hash, so the receipts above bind the pre-fix source; the fix touches only pool
lifecycle, not decoding.

## Not exercised live (no runtime consumer or not reachable on this model)

Rotating-cache undo/replay (module + tests only), Spomin live surgery (manager
not connected to the request lifecycle), approximate-KV seam (control plane
only), QSA amendment counters and the promotion-join reconciliation branch
(Flash-Next/Qwen4 paths).
