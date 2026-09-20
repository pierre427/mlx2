# Inverse-findings fixes, GPU A/B check (2026-09-19)

This run compares main before the fixes (`ffc6ceb`, labelled "old") with the fixed tree (`89817ef`, labelled "new") on the GPU.
The fixes are the five "Inverse findings" from the PR survey:

1. the signal coalescing fix,
2. required parameters in non-strict tool grammars,
3. tool markup inside markdown code,
4. the unclosed trailing Qwen `<parameter>`,
5. the incremental ThinkingGuard.

The run used one server at a time on the private port 8391, with `--qualification-mode`, and held `/tmp/gpu.lock` while it ran.
All requests used greedy decoding at temperature 0.
The drivers are `run_ab.py`, `guard_parity.py` and `guard_bench.py`.

## Results

| # | Probe | Old (`ffc6ceb`) | New (`89817ef`) |
|---|---|---|---|
| 1 | North PLD with `--drain-on-sigterm 60`. Two live streams, then SIGTERM followed 20 ms later by SIGINT to the process group. | Both streams were cut after 7 chunks with no finish reason. The server exited after 0.4 s. | Both streams completed: 120 chunks each, finish `stop`. The server exited cleanly after 3.0 s. The log shows `signal 2 arrived 0.025s after signal 15; treating both as one shutdown request`. |
| 2 | Named non-strict `get_weather` with required `city`, 5 prompts. | North: 1 of 5 returned 502 "missing a required parameter". Muse: 2 of 5 returned 502 "Malformed or missing ATEM parameters". Flash-Next: 5 of 5 included `city`, so the model did not show the bug. | North, Muse and Flash-Next all returned 5 of 5 calls with `city`. Flash-Next's output was identical to the old tree. |
| 3 | Flash-Next with tools declared and `tool_choice: auto`. The prompts ask for a code-block example of the tool markup. | The fenced Oslo example became a phantom `get_weather` call, and the text kept an empty code block. The fenced JSON-style example returned 502 "No function provided." | Both fenced examples were returned as text with no call. An ordinary auto tool call still parsed. |
| 4 | Unclosed trailing `<parameter>` | Not reproduced on the GPU, because it depends on the model producing malformed output. The parser runs on the CPU and is covered by `tests/test_output.py`. | — |
| 5 | Guard cost per step on the GPU, using the generator's `TokenBuffer` with a no-guard baseline subtracted. Median over 256 steps. | 49 µs at 1k tokens, 213 µs at 4k, 825 µs at 16k. | About 0 µs at 1k, 9 µs at 4k, 21 µs at 16k. |

On Flash-Next, one fence prompt produced a tool call on both trees.
That prompt asked for the markup "inside inline backticks", but the model emitted a `<tool_call>` with no text before it.
That is a real call made against the prompt's "do NOT call" instruction, not a phantom.

## ThinkingGuard parity

**In-process check** (`guard_parity.json`).
This drove the old and new guard modules with the same streams of GPU arrays:

- 24 streams and 21,600 decode steps;
- 3,247 speculative verify rows followed by rollbacks;
- 7 run-on trips, 13 soft-budget trips, 14 forced closes, and close markers.

The logits matched bit for bit and the receipts matched, with **0 mismatches**.

**Served check.**
North ran on PLD with the default guard: budget 512 and steering alpha 0.2.
The prompt set was 8 default prompts, 4 prompts at `thinking_budget` 48, and 4 hard prompts each at budgets 24 and 64.

- Every hard-prompt request tripped `budget_soft` and forced the close.
- `north-new-a`, `north-new-b` and `north-old-a` matched on **20 of 20** prompts. That covers the content, reasoning, token count and finish reason, and the full guard receipt: `think_tokens`, `tripped`, `tripped_at`, `released_at` and `forced_close`.

The first pair of runs, `north-new` against `north-old`, differed on 4 of 12 prompts.
Each difference started mid-reasoning at a near-tie token, for example "Let me recall" against "Let's recall".
The difference did not recur in the three-server control run, which includes an old-tree server.
The guard's logits are also bitwise identical, so these differences are not caused by the guard.
Those two stages ran with `--drain-on-sigterm`; the control stages did not.

The guard receipt in `north-new.json` and `north-old.json` is `null` because of a bug in the driver's key.
The receipt is at `mlx2.request_controls.thinking_guard`, and the key was fixed before the control stages ran.

## Files

- `{stage}.json`: per-stage probe results.
- `{stage}-server.log`: the server log for each stage.
- `policies/`: the execution policies. All of them set `constrained_tool_grammar: true`.
