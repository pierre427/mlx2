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
