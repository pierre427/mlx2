# North-Mini-Code run-on reasoning: tau/soft-release + JUICE budget (2026-09-18)

Source ideas (lab wiki): *reasoning-termination-margin* and
*termination-attractor-steering* (run-on is a termination-margin failure; use a
state-aware release, not a static stop bias; *tau* = CUSUM run-on alarm,
*alpha* = actuator) and *puzzle-juice-budget* (effort-scaled reasoning budget
with a soft landing before a hard close). Implemented here as
`src/mlx2/thinking_guard.py`: budget + content-blind run-on alarm -> ramped bias
on `<|END_THINKING|>` -> hard close only at the budget.

## 13-prompt experiment (greedy, `max_tokens` 2400, all arms concurrent)

Six prompts that ran on in the 20x20 sanity run, three easy controls, four
problems that genuinely need reasoning.

| Arm | Answered | Correct | Completion tokens | Hard closes |
|---|---|---|---|---|
| guard off | 10/13 | 10/13 | 9,968 | — |
| budget 1024 | 13/13 | 13/13 | 3,671 | 0 |
| budget 512 | 13/13 | 13/13 | 3,079 | 0 |
| budget 256 | 13/13 | 13/13 | 2,610 | 0 |

- Off: `explain_9`, `explain_18` and `capital_hungary` burn all 2,400 tokens and
  answer nothing ("Budapest? Actually ... Budapest? No, it's Budapest?").
- The run-on alarm fires at token 106 on the Budapest loop and ~240–290 on the
  explain loops, long before any budget; the soft budget catches the slow
  ditherers. The model then closes by itself within ~11 tokens — the hard close
  was never used, which puts North in the "loaded spring" regime (close token
  high-ranked but suppressed).
- Controls and the hard problems (primes below 60, digit sum of 2^20, trains,
  sorting) never trip the guard and are unchanged, even at budget 256.

## 20x20 sanity, thinking on by default, `--thinking-budget 512` (high = 2048)

| | Correct | Empty answers | Completion tokens | Median round |
|---|---|---|---|---|
| unguarded (+2048 allowance) | 394/400 | 6 | 81,743 | 19.5 s |
| guarded | **399/400** | **0** | 66,703 (−18%) | 14.4 s (−26%) |

The one miss is "Wien" for the capital of Austria — correct, in German; the
grader wants "Vienna".

## JUICE and alpha

JUICE applies directly: it is the effort-scaled budget half of the guard
(`--thinking-budget` anchor x `reasoning_effort`). The *alpha* activation
steering does not port cheaply: it needs a per-model commit direction from a
hidden-state calibration campaign (the wiki's Qwen3.8 result used a
model-specific layer-36 direction and warned that an over-strong hammer drops
material guards). Given that the logit-level release already fixes every North
case without a forced close, it was not attempted.

Caveats: greedy decoding, one model, lenient graders, small prompt set. The
guard is default-off; nothing about North's served defaults changed.
