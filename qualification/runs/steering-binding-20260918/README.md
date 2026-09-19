# Steering is bound to the calibrated artifact (GPU, 2026-09-18)

`run.sh` starts five servers in turn (private port, own-PID cleanup, waits for
the shared GPU lock) and records `/v1/status.settings.thinking_steer` plus one
known run-on prompt. `results.jsonl` has the rows.

| Stage | Expected | Result |
|---|---|---|
| North **4-bit**, no flags | steer with the shipped direction | ready in 6 s; `calibration.state: calibrated`, `origin: shipped`, layer 28, identity `729563c7…`; "Budapest" in 75 tokens, 70 steered steps |
| North **8-bit**, no flags | calibrate itself, then decide | calibrated in 47 s (19 usable traces, layer 32) and **rejected its own result**: held-out reasoning was not shortened (556 → 597 tokens) and a same-norm random direction did as well (568). Steering off, budget guard on; "Budapest" in 47 tokens |
| North 8-bit again | reuse the verdict | same outcome (this pass predates the remembered-verdict change; a rejection is now stored as `<identity>.rejected.json` so the minute is spent once) |
| North 8-bit, `--thinking-steer-alpha 0.2 --no-thinking-auto-calibration` | refuse to start | `ValueError: steering was requested but no commit direction is calibrated for this artifact: automatic calibration is disabled` |
| Qwen3.6, `--thinking-steer-alpha 0.2` | refuse to start | `ValueError: --thinking-steer-alpha needs a model with residual taps and a single thinking-close token` |

The 8-bit result is the interesting one. Its unsteered validation reasoning is
already short and every trace closes: the 8-bit build does not run on where the
4-bit build does, which is the lab's quantization corollary (a 4-bit perturbation
near the stop-decision margin flips borderline prompts into run-on). With nothing
to fix there is no directional benefit to measure, the random control ties, and
the gates correctly keep steering off. `north8-auto-calibration.json` is absent
because nothing is written for a failed calibration except the verdict.
