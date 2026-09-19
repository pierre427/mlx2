# Flash-Next plain-route late join on real weights (2026-09-19, Metal)

`flash_next_late_join.py` runs Qwen3.8-Flash-Next-MLX-4bit-MTP (ordinary
route) with `prefill_step_size=256`. Request A has 3,000 prompt tokens. B
(1,500) joins at step 3 and C (700) at step 5. Each row is compared with its
solo greedy run, plus an all-start-together control. It ran on main
(`latejoin2-base.json`) and on the fix branch (`latejoin2-fix.json`).

Every row matches its solo run in both trees, and so does the control.
Instrumentation counted 2 cold/warm `Qwen4ArraysCache.extend` calls per tree
and 0 in `merge`. These appear to be extends of an empty batch cache, which
have no rows to fill. In this configuration the real route never co-batches a
cold row with a warm one mid-prefill, so the zero-fill bug fixed in
`ArraysCache._empty_slot` was not reachable here. The CPU repro in
`tests/test_qwen4_cold_join_ple_history.py` still reproduces it, and the fix
stays as a guard.
