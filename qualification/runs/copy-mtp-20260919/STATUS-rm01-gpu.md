STATUS rm01-gpu | phase=done | 100% | ETA 0m | done: 7 queued GPU jobs (2 width curves, 5 interleaved A/Bs); 27B t0 meets the pre-registered criterion in full; code B=1 gain 1.20-1.33x on both models; B>1 gain absent, default now single-lane | next: coordinator fold (lever stays default-off); one quiet-GPU rerun for 35B prose dispersion | blockers: none

state: GPU_PARTIAL

verdict: **go at B=1** (Qwen3.8-27B, temperature 0, full pre-registered criterion met).
**not proven at 35B** (prose mean parity holds, the worst-rep dispersion rule does
not clear under 12-worker GPU contention). **no gain at B>1** on either model, so
copy drafts are now single-lane by default. Lever stays `implemented`, default-off;
no default changes recommended beyond the ones already on the branch.

branch: claude/rm01-copy-mtp-20260919 (worktree /private/tmp/mlx2-rm01-copy-mtp, off main 92bdfb9)
commits (GPU session on top of the CPU five):
- cee4a97 Record and gate peak Metal memory per arm in the copy-draft A/B harness
- 44857b2 A/B harness: optional --ordinary baseline arm and --native-mtp passthrough (coordinator request re main 16059c7)
- 1247212 Default copy drafts to single-lane: batched_max_span 0
- 53ee500 Warm copy verify widths before measuring the A/B; record the 35B evidence

## Width curves (verify cost model, B=1, context 2048)

| model | t(1 row) | row cost to 9 rows | draft step | shape |
|---|---|---|---|---|
| Qwen3.8-27B | 40.0 ms | 0.137 | 0.122 (4.89 ms) | step at 7->8 rows (63.7->79.7 ms), plateau 24-32, jump at 33 |
| Qwen3.6-35B-A3B | 11.4 ms | 0.116 | 0.153 (1.75 ms) | smooth, no cliff below 33 |

Both sit close to the shipped defaults (row 0.1, draft 0.15), so the A/Bs ran
with the default cost model. Evidence: `width-27b.json`, `width-35b.json`.

## Interleaved A/B (decode tok/s, on = copy drafts, off = plain self-MTP)

| run | prose B1 | prose B4 | code B1 | code B4 | peak mem |
|---|---|---|---|---|---|
| 27B t0 (pre-fix, cohort copies allowed) | 1.005 | 0.967 | 1.202 | 0.948 | +0.017 GiB |
| **27B t0 (single-lane default) -> GO** | 0.999 (worst 0.962) | 1.003 (0.974) | **1.204** (1.158) | 1.009 (0.985) | -0.000 GiB |
| 27B t0.7 | 0.981 (0.948) | 0.998 (0.956) | **1.261** (1.234) | 0.994 (0.970) | -0.018 GiB |
| 35B t0 (warm-up fix) | 0.983 (0.954) | 1.003 (0.960) | **1.312** (1.285) | 1.013 (0.962) | -0.001 GiB |
| 35B t0, 6 reps | 0.980 (0.909) | 0.980 (0.884) | **1.334** (1.138) | 1.033 (0.948) | -0.005 GiB |
| 35B t0, tuned gate | 0.995 (0.952) | 0.982 (0.954) | **1.303** (1.200) | 0.973 (0.929) | -0.006 GiB |

(worst = worst on-rep / best off-rep.) Against main's new 35B default, the
ordinary route, copy drafts run **1.85x** on code B1 and 1.05x on prose B1;
plain self-MTP is 1.41x on code B1. At B=4 ordinary beats both on prose
(0.77x), which is what main's 16059c7 already says.

- `copy_rounds` > 0 in every on-arm code cell at B=1 (124-149 per cell);
  the harness refuses an on-arm whose counter does not move there.
- Greedy cross-arm output mismatches (9-11 of 16 keys) are **not** attributable
  to the lever: each arm disagrees with itself just as often (within-arm
  nondeterminism 8-12), because a B=4 cohort's composition varies per rep.
- Peak Metal memory: parity within +/-0.02 GiB in every run (budget 0.5 GiB).

## What the GPU changed on the branch

1. **Cohort copies lose.** With copies capped at the cohort head depth, 27B B=4
   went 0.948 on code and 0.967 on prose while B=1 gained 20%: the batched win
   is bounded by the head depth, its cost is not. `batched_max_span` now
   defaults to **0** (no copies while more than one lane is verified); `null`
   restores the old head-depth cap. A cohort that drains to one lane copies
   again, which is why B=4 cells still show a few copy rounds.
2. **First-use shape compilation is a measurement artifact.** A copy row
   verifies at a width the head never proposes, so its kernels compiled inside
   the first measured cell and charged the on-arm ~3% of a 35B prose run
   (35B t0: 0.970 -> 0.983 once the warm-up issues the same code requests in
   every arm). Host-side copy bookkeeping is only ~4 us/round (CPU microbench),
   so it was never the index.
3. **The dispersion half of the prose rule is not measurable on a shared GPU.**
   Off-arm reps alone span 6.6% (35B prose B1: 122.5-130.6 tok/s) with 11 other
   workers on the device, so "worst on-rep >= 0.96 x best off-rep" fails on
   noise. The mean half (>= 0.98) holds in every run: 0.980-1.005.

## Tests

- full suite at 53ee500 (1692 collected): 0 failed, 54 skipped, exit 0 -- log
  `full-suite-gpu.log` in this directory (this repo's pytest config prints no
  summary line; the run is marks-only).
- tests/test_copy_draft_self_mtp.py: 28 passed, including the new
  `test_default_policy_refuses_cohort_copies_but_copies_solo` (mechanism
  assertion for the single-lane default) and the harness verdict tests
  (peak-memory gate, ordinary arm, --native-mtp route args).

## Recommendation to the coordinator

- Fold the branch. The lever stays **default-off** and **implemented**, not
  qualified: 27B B=1 meets the criterion, 35B prose dispersion does not.
- Do **not** change any serving default to enable copy drafts.
- The changed defaults inside the (off) policy are: `batched_max_span=0`.
- Worth one quiet-GPU rerun (no other workers): 35B t0, reps 5, prose+code,
  and the tuned gate `{"min_samples": 1, "min_yield_ratio": 1.15,
  "reprobe_interval": 64}`, which took 35B prose B1 from 0.983 to 0.995 with
  code still 1.303. If that clears, the 35B verdict flips to go.

## Evidence

- `qualification/runs/copy-mtp-20260919/` on the branch: `width-27b.json`,
  `width-35b.json`, `ab-27b-t0.json`, `ab-27b-t0-v2.json`, `ab-27b-t07.json`,
  `ab-35b-t0.json`, `ab-35b-t0-v2.json`, `ab-35b-t0-reps6.json`,
  `ab-35b-t0-gate.json`, `SUMMARY.json`, per-arm server logs and job logs.
- wiki: `wiki/docs/experiments/mlx2-rm01-copy-mtp-2026-09-19.md` ("GPU results").
