# 32K four-request performance investigation, 2026-10-10

The one-repetition Flash Next ladder on M5 at 8a710be83 recorded cold median
decode of 4.83 tokens/s/request (October 7 median: 17.98), and warm median
TTFT of 47.97 seconds (prior: 3.13). Only two of four warm requests reused
APCv2. These are performance observations, not route qualification.

Two general admission defects were found:

- A target-only prompt-boundary checkpoint is correctly sent to ordinary
  decode on a self-MTP route, but admission still requested a draft transient
  and discarded the measured warm cache size in favour of a cold projection.
  Admission now charges its actual target-state copy and depth zero. Shorter
  draft-less prefixes remain misses; no synthetic draft state is introduced.
- Bounded self-MTP prefill allocates cache before emitting its first token.
  The serving admission ledger continued subtracting the entire initial
  reservation, even though measured headroom already excluded allocated cache.
  The executor now reports evaluated net cache growth since the initial state;
  the ledger subtracts that growth from its outstanding grant. Initial warm
  aliases receive no credit, remaining growth and transients remain reserved,
  and cancellation, preparation and close retire the credit.

The second defect can keep a co-arriving request outside the executor until
siblings start decoding. The late prefill then stretches their decode interval
and competes with already-published checkpoints. The single-cell rerun will
check the observed consequence; static analysis is not proof of recovery.

Validation: 58 focused CPU tests passed, including warm ordinary accounting,
partial allocation, initial alias exclusion, ownership cleanup, grant safety,
depth fallback and prefill projection bounds. No qualification campaign or
background worker was restarted. The requested GPU validation is one cold/warm
32K width-four performance cell under a fresh lease and paired locks.

## First GPU rerun and subsequent CPU follow-up

The first rerun used ea8113b7e under CPG generation 397 and paired locks.
Cold prefill rose from 547.8 to 580.4 tokens/s/request, cold decode from 4.83
to 6.10, and cold four-request wall-clock fell from 88.94 to 79.54 seconds.
Warm hits rose from 2/4 to 3/4 and warm wall-clock fell from 53.34 to 29.81
seconds. Thermal state stayed nominal; no contamination was flagged. All
streams completed and returned the needle, but the cache-hit gate remained
unsatisfied. Historical decode (17.98) and 4/4 warm hits were not recovered.

The rerun trace makes the scheduling failure concrete. Three cold responses
reported compute widths `[3, 4]`, while the late fourth response reported
`[1, 4]`. The first cohort therefore decoded while the fourth request was
still prefilling, then all four eventually merged into ordinary decode. In the
warm phase, three requests had near-complete APCv2 hits and one was a full
miss, but all four TTFTs clustered at 24.16-24.90 seconds: the warm hits waited
for the miss instead of entering decode. APCv2 recorded 19 value evictions,
all with zero prior hits, and only three of four warm requests survived as
full-prefix hits.

The causal sequence was:

1. The width policy had already selected ordinary execution above width three,
   but joining admission still priced and allocated self-MTP depth two.
2. That speculative admission admitted three rows first, splitting one logical
   four-request arrival into a three-row cohort and a late singleton.
3. Admission evicted reusable APCv2 entries when ordinary decode already fit,
   solely to seek an optional speculative depth that the static width policy
   would immediately retire.
4. Warm coarrival timing compared remaining cold work with total context,
   including cached history. A request with only a short uncached tail could
   therefore wait behind a full 32K miss.

Further investigation after the rerun found that the static ordinary-width
policy still paid speculative admission costs before retiring MTP. An idle
non-atomic cohort whose selected width exceeds the static MTP threshold now
reserves the greater of a serial preparation peak and ordinary batch
transients. At the prepared ownership seam, an ordinary admission check prices
all target merge copies before handing off the cohort. Adaptive park policies,
active cohorts and strict atomic cohorts retain their existing contracts.

Admission also now tries the ordinary floor before evicting APC to buy an
optional draft depth; cycle eviction stops once all rows can progress. Cohort
hold and exemption comparisons use initial uncached work, excluding cached
history, so a nearly warm anchor is not held behind a full cold prefill merely
because both prompts have 32K total context. CPU regression tests preserve equal
cold cohort formation, outlier latency, cache ownership and memory bounds.

These fixes are route-generic. They operate on the executor's declared
ordinary-handoff policy, admission depths, target-cache bytes, and initial
uncached work; they contain no Flash Next or model-name branch. The same
contracts apply to any self-MTP adapter that exposes a static ordinary handoff.

Validation before the post-fix retest: 225 focused CPU tests and nine subtests
passed; an additional merge-copy refusal regression passed with the other four
new pure CPU tests. These follow-up changes were made after the first GPU
rerun and must not be attributed to that first measurement.

## Post-fix GPU retest

One further cold/warm repetition ran at ffc2c384b under CPG generation 401
with paired locks. Cold decode recovered to 26.92 tokens/s/request and cold
wall-clock was 78.02 seconds. All four warm requests reused at least 99.996%
of their prompts; warm TTFT was 1.99 seconds and warm wall-clock was 6.56
seconds. This exceeds the October 7 median for decode and improves its cold
and warm wall-clock, while cold prefill (472.77 tokens/s/request) and TTFT
(69.31 seconds) regressed relative to the first rerun. The result therefore
recovers the decode and warm-cache failures without claiming a uniform
improvement in every phase.

Cold receipts still show arrival skew: three rows entered self-MTP at width
three before handing off at width four, while the fourth reports ordinary
width four. Their final decode rates nevertheless clustered at 24.48-27.03
tokens/s, and the warm pass had complete cache coverage. Cold cohort formation
and prefill efficiency remain opportunities for further work.

All eight streams returned the correct needle. The strict harness remains
failed because only 2/4 warm texts were byte-identical to their cold outputs,
the already documented batch-composition nondeterminism. Thermal state stayed
nominal, swapouts were unchanged, and no foreign activity or refusal was
recorded. This is one observation, not a replicated benchmark or qualification
claim. Background qualification remains stopped.

The post-fix comparison above includes prefill, cold/warm TTFT, per-request
decode, and cold/warm concurrent HTTP wall-clock. Machine-local receipts and
raw server logs remain in the private evidence repository.
