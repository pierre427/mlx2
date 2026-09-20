# What North claimed before 2026-09-20, and what the normalization fix voids

rm06b established that until 2026-09-20 mlx2 served North-Mini-Code-1.0 with
`nn.LayerNorm(hidden, eps=layer_norm_eps=1e-5)` where the reference
(`transformers/models/cohere2_moe/modeling_cohere2_moe.py`, and this
checkpoint's own `rms_norm_eps: 1e-06`) is `RMSNorm(eps=1e-6)` — for all 49
`input_layernorm`s and the final `model.norm`. The same mistake sat in the
companion `CohereEagleDraftModel`'s final norm. Branch
`claude/rm06b-north-norm-20260920`, commits 8a58073 / 15761b1.

**Every number below was measured on that mis-normalized body.** The fix's own
A/B (`qualification/runs/rm06b-north-norm-20260920/quality-ab.json`) measured
perplexity 8.6716 → 8.5880 (−0.96%) and a mean greedy common-prefix ratio well
under 1: the corrected body *does not generate the same text*. So these are not
"probably still right, to within noise" — greedy North output changed, and
every quality, acceptance and grading number is a measurement of a different
model than the one mlx2 now serves.

This file is the list. A row that nobody re-runs is a claim the lab should stop
making.

## A. Serving qualification records

| Record | What it claimed | Status now |
|---|---|---|
| `runs/integration-gpu-20260918/north-ordinary-qualification.json` | `scripts/qualify_serving.py` on `--ordinary` | **VOID**, and it never passed anyway: `passed=false`, `shared_cohort_priming: FAIL` (14/15). Same in `first-pass/`. |
| `runs/integration-gpu-20260918/north-spomin-qualification.json` | qualifier with Spomin live surgery, 28K capacity | **VOID**; also `passed=false`, same single failure |
| `runs/sanity-20x20-20260918/north-ordinary-qualification.json` | qualifier on `--ordinary` | **VOID**; this one `passed=true` |
| `runs/thinking-budget-20260918/north-qualification.json` | qualifier with the thinking budget in force | **VOID**; `passed=true` |
| `runs/north-alpha-calibration-20260918/default-on-qualification.json` | qualifier with North's default-on budget+alpha | **VOID**; `passed=true`, and it also rests on the stale direction (§C) |
| `runs/macos-26.7/north-mini-code/ordinary/attempt-*` (three attempts, 2026-09-16) | candidate route-qualification attempts under the thermal runbook; all three terminated on an infrastructure failure (`stop-check-failure`, `http-body-cap`, `thermal-stop`) | already non-claims; **VOID** as measurements regardless |

Net: **North has no valid serving qualification record.** Three of the five
machine records that exist are `passed=false` or aborted; the two green ones
were produced on the wrong model body.

## B. Behavioural / quality campaigns

| Record | Headline number | Status |
|---|---|---|
| `runs/sanity-20x20-20260918/north-ordinary-sanity.json` | 398/400 graded correct, 20,393 completion tokens, peak width 15 | **VOID** |
| `runs/sanity-20x20-20260918/north-pld-sanity.json` | 398/400, PLD accepted 1,035 of 5,045 proposals, target width 1 | **VOID** |
| `runs/sanity-20x20-20260918/second-pass/north-*` | repeat pass | **VOID** |
| `runs/thinking-budget-20260918/north-sanity.json` | 394/400, 6 empty answers, 81,743 completion tokens (the *unguarded* baseline) | **VOID** |
| `runs/north-thinking-guard-20260918/north-sanity-guarded.json` | 399/400, 0 empty, 66,703 tokens (−18%) — the number quoted in `docs/SERVING.md` and the brag book | **VOID** |
| `runs/north-alpha-calibration-20260918/sanity-ordinary-alpha.json`, `sanity-pld-alpha.json` | 400/400 at 45,631 tokens (−44% vs unguarded) — the headline alpha-steering result | **VOID** (wrong body *and* stale direction) |
| `runs/known-limits-20260918/north-ordinary-thinking-sanity.json` | 386/400, 16 empty | **VOID** |
| `runs/known-limits-20260918/north-pld-batched-sanity.json` | 398/400, batched PLD verify at observed width 20 | **VOID** as a quality number; the *mechanism* claim (batched verify ran, width 20) is a counter, not a model measurement, and survives |
| `runs/known-limits-20260918/probe_north_structured.py` results | structured-output envelope behaviour | mechanism survives, generated content **VOID** |
| `runs/quality-campaign-20260919/` North rows | pass-3/4/5 triage | **VOID** |

## C. Alpha steering — the default-on lever

`NorthMiniCodeAdapter.THINKING_GUARD_DEFAULTS` ships `thinking_budget 512` and
`thinking_steer_alpha 0.2` **on by default for North**, and `docs/SERVING.md`
says so. The direction it steers with,
`src/mlx2/adapters/assets/north_mini_code_commit_direction.{npz,json}`, was
calibrated 2026-09-18 **in the old residual geometry** (layer 28, consistency
+0.46, 22 traces) — a mean-centred LayerNorm residual stream, not the RMSNorm
one mlx2 now runs.

rm06b bumped `thinking_calibration.SCHEMA` to `mlx2.commit-direction.v2`. The
shipped sidecar still says `v1`, and `artifact_identity()` hashes `SCHEMA` into
the digest, so it fails closed twice over: `load_bound_direction` rejects the
schema, and the identity no longer matches either. Consequences:

- a default North server no longer steers with the stale vector — it
  auto-calibrates at startup and, if the gates fail, serves budget-guard-only;
- `docs/QUALIFICATION-EXPERIMENTS.md` requires North's matrix arm to report
  `settings.thinking_steer.calibration.state == "calibrated"`, so the
  **four-model matrix arm for North cannot pass until a v2 direction exists**;
- `runs/quality-campaign-20260919/campaign_config.py` preflight raises
  `RuntimeError("North calibrated commit direction is not bound to this
  artifact")`, so **that campaign's preflight fails on this branch** until a v2
  asset ships;
- the whole 2026-09-18 calibration campaign — the per-layer consistency table,
  the held-out grid (L28 α0.2: 16/16 at −83% reasoning tokens), and the random
  controls — is **VOID**.
- `runs/steering-binding-20260918/` is a *binding* experiment, not a quality
  one: it shows the server refuses a direction from another artifact and
  rejects its own 8-bit calibration. Its mechanism findings survive; its
  4-bit "ready in 6 s, origin shipped, layer 28" row no longer describes the
  code.

## D. Speculation

| Record | Claim | Status |
|---|---|---|
| `runs/rm06-north-20260919/greedy-r0-r2.json`, `sampled-t07.json` | external Cohere-EAGLE route: best 0.67x at B=1, acceptance length 1.60–2.0 → rm06 NO-GO | superseded by `runs/rm06b-north-norm-20260920/accept-greedy-r0-r2.json`, which **re-ran it on the corrected body**: best B1 0.630x, B4 0.494x, acceptance 1.633/1.920/2.064/2.070. The no-go stands and the "wrong norm was capping acceptance" hypothesis is refuted. **Already re-qualified; no action.** |
| `runs/rm06-diagnostics-20260919/round-profile-north.json`, `smoke-north.json`, `serving-smoke-north.json` | per-round cost decomposition (~7.4 ms/round for the EAGLE chain) | timing, not model output; survives as orientation |
| `docs/SERVING.md` §"External drafters", North row (0.67x at B=1) | | **stale**: superseded by rm06b's 0.630x |

## E. Docs and defaults that carry a North claim

- `docs/ports/NORTH-MINI-CODE.md` — corrected by rm06b (1a33938); its "CPU
  evidence" section is CPU-only and unaffected; its serving numbers are void.
- `docs/SERVING.md` — carries the guarded-sanity 399/400 and −18% tokens, the
  alpha-steering 400/400 / −44%, and the North external-draft row. All void or
  stale; **no explicit "measured before the norm fix" warning yet**.
- `docs/RELEASE-BRAG-BOOK.html` — two feature cards quote North numbers
  ("394 → 399 of 400 with 18% fewer tokens"; "reasoning tokens fell 83% at
  16/16 correct"). Both **VOID**.
- `docs/QUALIFICATION-EXPERIMENTS.md` — the North matrix arm's
  `status_requirements` (profile, cache layout, 13 global / 36 sliding layers,
  adapter thinking defaults, `calibration.state == "calibrated"`). The
  structural requirements survive the fix; the calibration one now fails
  closed (§C).
- `qualification/four-model-experiments.json` — North's `ordinary` arm, its
  `activate_command`, and its context ladder to 499,936 tokens. Never run to
  completion on the corrected body. Note its `activate_command` uses
  `PYTHONPATH=src` with `--cwd ~/Desktop/mlx2`: it can only
  ever measure the main checkout.
- `NorthMiniCodeAdapter.profile_name()` → `north-mini-code-apcv2-ordinary`,
  and `COMMIT_DIRECTION_LAYER = 28`. The profile name is a label, not a
  measurement. **Layer 28 is a measurement** and is void until re-chosen.
- `src/mlx2/qualification.py APPROVED_QUALIFICATION_HARNESS` — pins
  `scripts/qualify_serving.py` at `3ec83a5d…`, which is the current file on
  this branch. No re-pin needed; the 2026-09-18 North records were produced by
  the older `7d9e1a3b…` harness, which is a second reason they cannot be
  carried forward.

## F. What survives

Mechanism counters and structural facts are properties of the serving stack,
not of the model's arithmetic, and are unaffected:

- the cache geometry (13 global + 36 rotating, `north-mini-code-layer-segments-v1`,
  4,096 window), and the admission bound derived from it;
- APCv2 behaviour on North (fan-out groups, COW branches, cached-token counts);
- that batched PLD verify reaches observed width 20 on North;
- that Spomin live surgery runs on North (probe 7/7);
- steering *binding* behaviour (refuses a foreign direction; rejects a
  calibration that fails its gates);
- every CPU test in `tests/test_north_mini_code_port.py`,
  `tests/test_north_norm_choice.py` and the adapter-registry tests;
- rm06's cost decomposition of the EAGLE round.
