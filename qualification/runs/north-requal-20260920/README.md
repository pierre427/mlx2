# North re-qualification on the corrected normalization (2026-09-20)

rm06b corrected North's normalization (`nn.LayerNorm(eps=1e-5)` →
`RMSNorm(eps=1e-6)`, 49 decoder norms plus the final one). The corrected body
generates different greedy text, so **everything North was ever measured on was
measured on a different model**. `INVENTORY.md` is the explicit list of what
that voids. This directory is the re-measurement.

Branch `claude/rm06b-north-norm-20260920`. Every job ran through the shared-GPU
lock wrapper.

## Results

### Ordinary serving route: QUALIFIED

`north-ordinary-qualification.json` — **`passed: true`, 30 of 30 checks**, long
context inline, harness `3ec83a5d…` (the hash `src/mlx2/qualification.py` pins)
against `preflight.json`, a receipt bound to the exact revision under test.

`shared_cohort_priming` passes. It failed on 2026-09-18 in
`integration-gpu-20260918/north-ordinary-qualification.json`, again in that
run's `first-pass/` twin, and again in the Spomin variant. It asks the model to
begin its output with a literal marker at near-limit context with thinking off —
a behaviour check — and the mis-normalized body was failing it. This is the
first passing North serving qualification the lab holds from the pinned harness.

### Commit direction: recalibrated, gated, re-shipped at L32

`calib-direction.{npz,json}` (standalone `scripts/calibrate_thinking_direction.py`)
and `cache/north-ordinary/commit-directions/1d2ebefc….{npz,json}` (the server's
own startup `auto_calibrate`) are **bit-identical**: max absolute difference 0.0
on every `v_L`, identical `rms_L`. 24 prompts, 21 with a usable reasoning span.

Cross-trace cosine consistency, against the void 2026-09-18 table:

| Layer | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-20 (RMSNorm) | +0.409 | +0.447 | +0.470 | +0.489 | +0.511 | **+0.518** | +0.513 | +0.500 | +0.500 |
| 2026-09-18 (LayerNorm, void) | +0.366 | +0.408 | +0.429 | +0.448 | +0.460 | +0.466 | +0.455 | +0.433 | +0.429 |

Higher at every probed depth, and the peak moved L28 → L32, so
`COMMIT_DIRECTION_LAYER` is now 32.

### Held-out grid (`calib-grid-a.json`, `calib-grid-b.json`)

16 prompts not used for calibration, `max_think` 2400, greedy, B=1 direct model
— the same protocol as the campaign it replaces.

| Arm | Closed | Correct | Reasoning tokens | run-on / control / hard |
|---|---|---|---|---|
| off | 14/16 | 14/16 | 7,683 | 6,666 / 155 / 862 |
| **L32 α0.2 (shipped)** | **16/16** | **16/16** | **2,404 (−69%)** | 1,536 / 145 / 723 |
| L28 α0.2 (the old layer) | 16/16 | 16/16 | 2,694 | 1,914 / 147 / 633 |
| L32 **random** α0.2 | 15/16 | 15/16 | 7,169 | 6,110 / 171 / 888 |

The saving is concentrated in the run-on prompts (6,666 → 1,536) while the hard
problems are barely shortened (862 → 723) and stay correct, so α0.2 is not
truncating thought. Two prompts never closed unsteered (2,400 tokens, empty
answers); steered they close at 91 and 77 tokens with correct answers.

**One honest difference from the void campaign.** In 2026-09-18 the random
control was *worse than doing nothing* (11/16 at 13,189 tokens against 13/16 at
9,187). Here it is marginally better than nothing (15/16 at 7,169 against 14/16
at 7,683) — it just does not reproduce the effect. So the claim this run
supports is the weaker and more defensible one: **a same-norm random direction
on the same schedule does not shorten reasoning; the calibrated direction does.**
It is no longer evidence that a wrong direction actively harms.

## Provenance guards

`qualification/four-model-experiments.json`'s North arm and the 2026-09-18
integration campaign both start their server with `PYTHONPATH=src` and
`cwd=~/Desktop/mlx2`, which imports mlx2 from the **main
checkout** no matter which worktree drives it. `run_requal.py` therefore
resolves `mlx2.__file__` and `runtime/models/cohere2_moe.__file__` under exactly
the env and cwd the server receives, refuses anything not under this worktree,
and scans the server log for the main checkout's path. See
`north-ordinary-provenance.json`. It also validates the preflight receipt's
identity **before** taking the GPU lock: the first attempt spent a shared slot
discovering a stale receipt 0.4 s after a 70 s model load.

## What this run does NOT qualify

- the thermally controlled context ladder (2K … 499,936 tokens);
- the width-20 batch stress suite and the frozen Spomin 20×20 suite;
- any quality or grading number — 20×20 sanity, the thinking-guard campaign and
  the alpha campaign are all still void and were **not** re-run here;
- the prompt-lookup route on North;
- anything on the M3 host (skipped; see `INVENTORY.md` and the wiki page).
