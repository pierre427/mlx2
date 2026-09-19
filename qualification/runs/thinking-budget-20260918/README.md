# Harness budgets for thinking-default models (2026-09-18)

`/v1/status.thinking_default` reports whether a request that says nothing about
reasoning opens a think channel (today: North-Mini-Code only; Qwen3.x,
Flash-Next and Muse default off). Harnesses size their budgets from it.

- **Qualifier** (`scripts/qualify_serving.py`): every request that will actually
  think on such a server gets +2048 completion tokens, clamped to the API's
  8192 ceiling; requests that turn thinking off keep their exact budgets. The
  fixed-budget checks (`batch`, near-limit context) turn thinking off
  explicitly because they measure batching/context, not reasoning. The report
  records `thinking_budget`. The `batch` check also reads prompt-lookup's verify
  width from the speculation receipt now that it can be shared.
- **20×20 sanity driver**: detects the same flag, leaves reasoning on and
  applies the same allowance (`SANITY_THINK=0/1` overrides).

North-Mini-Code on GPU, thinking on by default:

| Run | Before (+600–800 tokens, forced by env) | After (+2048, auto-detected) |
|---|---|---|
| 20×20 sanity | 386/400, 16 empty answers | **394/400**, 6 empty answers, arithmetic 40/40 |
| Full qualifier | — | **28/28 checks pass** |

The six remaining empties are reasoning that ran past ~2,150 tokens without
converging (four on the deliberately vague "optimization number N" prompt, two
second-guessing a spelling) — model behaviour, not a budget that more tokens
would reliably fix.
