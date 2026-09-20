# rm06b North normalization (2026-09-20)

The two `PREREGISTERED-*-plan.json` files are the execution plans and go/no-go
criteria, written **before** either job ran. They are committed so the bars
cannot be moved after seeing the numbers.

Results land beside them when the shared GPU serves the queued jobs:

- `quality-ab.json` -- ordinary-route quality A/B, legacy LayerNorm vs fixed
  RMSNorm. **GO if `comparison.relative_ppl_change < 0.01`.** A low
  `mean_greedy_prefix_ratio` is expected: the fix changes greedy outputs, and
  that is a documented North serving-behaviour change. Each arm's
  `norm_counters` must show 50 (49 decoder norms + the final one) or the arm
  was refused.
- `accept-greedy-r0-r2.json` -- rm06's North EAGLE acceptance bench, rerun with
  the corrected norms. rm06's bars, unchanged: mean greedy common-prefix ratio
  >= 0.9 AND B1 >= 1.25x AND B4 >= 1.0x AND no refused arm. rm06 measured 0.67x
  at B1 with acceptance length 1.60-2.0. All four clauses must pass to reopen
  rm06's no-go.

The normalization finding itself does not depend on either job: it rests on the
reference implementation, the checkpoint's own config, and the companion
drafter's declared `norm_type`. See `tests/test_north_norm_choice.py`.
