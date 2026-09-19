# North-Mini-Code alpha calibration campaign (2026-09-18)

Goal: port the lab's termination-attractor *alpha* actuator to North — a
per-model "commit direction" in the residual stream that pulls a reasoning
trace toward its own close — and check it against same-norm random controls.
Script: `scripts/calibrate_thinking_direction.py` (trace / extract / grid).
Artifact: 4-bit `North-Mini-Code-1.0-mlx-4bit`, greedy, thinking on, B=1 direct
model for calibration and the grid; server-level checks below.

## 1. Traces and direction

24 calibration prompts (arithmetic, facts, short explanations, code), all closed
and correct; 22 had a reasoning span long enough to use (681 reflection
positions, 528 commit positions). Direction per layer =
mean(residual | last 24 in-think tokens before `<|END_THINKING|>`) −
mean(residual | positions in the 8%–45% window of the think span).

| Layer | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 |
|---|---|---|---|---|---|---|---|---|---|
| cross-trace cosine consistency | +0.37 | +0.41 | +0.43 | +0.45 | **+0.46** | **+0.47** | +0.46 | +0.43 | +0.43 |

Consistency is clearly positive at every depth and peaks at ~60% depth
(L28–L36 of 49), the same band the lab found on Qwen3.6 (L23/40) and Qwen3.8
(L36/64).

## 2. Held-out grid (16 prompts not used for calibration, `max_think` 2400)

Six known run-on prompts, three easy controls, seven problems needing
reasoning. Steering `h_L += alpha * rms_L * v_hat` on decode steps while the
reasoning channel is open.

| Arm | Closed | Correct | Reasoning tokens | run-on / control / hard tokens |
|---|---|---|---|---|
| off | 13/16 | 13/16 | 9,187 | 8,288 / 168 / 731 |
| **L28 alpha 0.2** | **16/16** | **16/16** | **1,531 (−83%)** | 764 / 121 / 646 |
| L32 alpha 0.2 | 16/16 | 16/16 | 2,048 | 1,345 / 142 / 561 |
| L32 alpha 0.4 | 16/16 | 16/16 | 1,362 (−85%) | 416 / 103 / 843 |
| L32 two-mode (0.2, hammer 0.8 from 600) | 16/16 | 16/16 | 2,048 | hammer never reached |
| L32 **random** alpha 0.2 | 11/16 | 11/16 | 13,189 | 12,111 / 167 / 911 |
| L32 **random** alpha 0.4 | 12/16 | 12/16 | 11,203 | 10,148 / 162 / 893 |
| L32 **random** two-mode | 12/16 | 12/16 | 11,467 | 10,404 / 173 / 890 |

- The effect is the **direction**, not the perturbation: a same-norm random
  vector on the same schedule is worse than doing nothing.
- The savings come almost entirely from the run-on prompts (8,288 → 764);
  reasoning on the hard problems is barely shortened (731 → 646) and stays
  correct, so alpha 0.2 is not simply truncating thought.
- L28 alpha 0.2 is the shipped operating point
  (`NorthMiniCodeAdapter.COMMIT_DIRECTION_LAYER = 28`).

## 3. Through the server (batched, 20 lanes)

`--thinking-budget 512 --thinking-steer-alpha 0.2`; 20×20 sanity driver with
thinking left on (auto-detected).

| Route | Correct | Empty answers | Completion tokens | Notes |
|---|---|---|---|---|
| ordinary, unguarded (earlier run) | 394/400 | 6 | 81,743 | baseline |
| ordinary, guard only (earlier run) | 399/400 | 0 | 66,703 | "Wien" grader miss |
| prompt lookup + alpha 0.2 | **400/400** | 0 | **45,631** | batched verify width 20, steering applied to verify forwards |
| ordinary + alpha 0.2 | **400/400** | 0 | **45,870 (−44% vs unguarded)** | width 20; median round 10.3 s vs 19.5 s unguarded and 14.4 s guard-only |

### Harder multi-step set (16 problems, batched through the server)

| Arm | Correct | Completion tokens |
|---|---|---|
| off | 16/16 | 2,257 |
| guard only | 16/16 | 2,353 |
| alpha 0.2 | 16/16 | 1,881 (−17%) |
| alpha 0.4 | 16/16 | 1,580 (−30%) |

No accuracy cost at either strength. The honest limit of this evidence: North
solved every one of these with at most ~240 reasoning tokens, so the set never
forced long deliberate reasoning — it shows alpha does not break ordinary
multi-step work, not that it is safe on problems that need thousands of
thinking tokens.

## Composition

- **Batching:** yes. The guard is per-lane; steering is one `[B,1,D]` tensor
  with zero rows for lanes that do not steer, so steered and unsteered lanes
  share a forward (unit-tested: the unsteered lane is bit-identical).
- **Speculative decoding:** the logit-level guard (budget, run-on alarm, ramped
  release) is a pure function of the token ids, so it composes with every
  route's verify rows and rollbacks. Alpha steering is wired into the ordinary
  decode step and prompt-lookup verify forwards. It is *not* applied on native
  MTP or external-draft routes: the drafter would not see the steered target
  and acceptance would fall; those routes keep the logit-level guard.
- **Prefix cache:** steered lanes skip the end-of-request exact store; the
  prompt boundary (stored before any steering) is unaffected.

## Caveats

Greedy decoding, one quantized artifact, small prompt sets, lenient graders.
The lab's Qwen3.8 result warns that an over-strong hammer can drop material
parts of an answer; keep alpha ≤ 0.4 and re-qualify on real agentic tasks
before trusting it on long agentic work.

**Default (2026-09-18, Pierre's decision):** the guard (budget 512) and alpha
0.2 are now ON by default for North-Mini-Code via the adapter's
`thinking_guard_defaults()`; `--thinking-steer-alpha 0` / `--thinking-budget 0`
(or the same fields with 0 on a request) turn them off.

Procedure note: the first server pass collided with another session's server
on port 8297 (my script's pattern-based cleanup then killed it). The scripts now
use a private port, kill only their own PID, and refuse to start while
`/tmp/gpu.lock` is held.

## Default-on smoke (GPU, after the defaults landed)

North served with **no** thinking flags (`default-on-smoke.json`):

- `settings`: `thinking_budget` 512, `thinking_steer.alpha` 0.2,
  `thinking_defaults_source` `adapter`.
- "What is the capital of Hungary? One word." — the prompt that used to run on:
  answers `Budapest` in 75 tokens, 70 steered steps at layer 28, closes on its
  own at token 69 (no alarm, no forced close).
- The same request with `thinking_budget: 0` and `thinking_steer_alpha: 0`: no
  guard, 2,400 tokens, empty answer — the original failure, confirming both that
  the opt-out works and that the defaults are what fix it.
- Full qualifier with the defaults in force: 28/28 checks
  (`default-on-qualification.json`).
