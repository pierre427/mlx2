# Smoke pass 5 triage

Pass 5 completed 17 stages; Laguna was excluded from this review because its
diagnostic rerun was live when the work began.  Every non-Laguna failure below
was checked against the retained request/reply, the route receipt or qualifier
evidence, and the corresponding `server-base.log`.  A fail-closed 502 for a
missing required tool is correct API behavior: `openai_compat.enforce_tool_contract`
requires a call for `required`, only the named function for named choice, and at
most one call when parallel calls are disabled.

Abbreviations in the table: Muse = all three Muse stages; North = both North
stages; Xing = all three Xing stages.  Globs in evidence cells identify the
same retained artifact in every listed stage.

| Failing check | Stages | Classification | Evidence path and finding | Action |
| --- | --- | --- | --- | --- |
| SDK `openai.responses.nonstream_reasoning_schema_logprobs` | `smoke-qwen36-ordinary` | HARNESS | `results/smoke-qwen36-ordinary/base-official-sdks.log` reports `JSONDecodeError`; `raw/core/018-structured-after-thinking.json` proves the same model returns enforced JSON with `thinking_budget=128`, while `scripts/sdk_smoke.py` supplied that budget only to Chat. Server log records HTTP 200, so this was an empty final channel, not transport failure. | Pass the shared 128-token budget through Responses and retain `response.model_dump()` on parse failure. Rerun stage. |
| Qualifier `feature_shared_qsa` | `smoke-flash-next-mtp2` | HARNESS | `results/smoke-flash-next-mtp2/qualification.json` records auto policy, base 32,547, remaining 63, cutoff 32, one policy check and one correct decline; `server-base.log` is healthy and all 40 core checks pass. The qualifier's fixed 64-token cohort was outside the selected policy domain. | Size only the shared-QSA cohort probe to the runtime crossover (32 at 32K; up to 64 at 64K+) and re-pin the qualifier hash. Rerun stage. |
| Core `tools_required` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/011-tools-required.json` retains the required choice and fail-closed 502 `model did not emit a required tool call`; matching `server-base.log` entries are 502. | Quality note; no code change. |
| Core `tools_named` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/012-tools-named.json` retains named `weather` and fail-closed 502 `did not exclusively call`; matching server logs are 502. | Quality note; no code change. |
| Core `tools_parallel_false` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/013-tools-parallel-false.json` requests `required` with `parallel_tool_calls=false`; replies fail closed because no call was emitted, rather than falsely claiming single-call compliance. | Quality note; no code change. |
| Core `responses_function_roundtrip` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/021-responses-function-roundtrip.json` retains named `weather` and a 502 for no exclusive named call; server logs confirm the Responses 502. | Quality note; no code change. |
| Core `messages_tools_any` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/028-messages-tools-any.json` requests Anthropic `any` and gets the correct API-shaped 502 because no required call was emitted. | Quality note; no code change. |
| Core `messages_tools_named` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/029-messages-tools-named.json` requests named `weather` and gets the correct API-shaped 502 for no exclusive named call. | Quality note; no code change. |
| Core `messages_tool_result_roundtrip` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/030-messages-tool-result-roundtrip.json` cannot start the round trip because the required first `weather` call is absent; the 502 is contract enforcement. | Quality note; no code change. |
| Core `messages_stop_sequences` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/raw/core/034-messages-stop-sequences.json` returns only `A`/`A\n`, `end_turn`, and `stop_sequence=null`; the model reached EOS before emitting `END`, and server logs correctly report 200. | Quality note; no code change. |
| SDK `openai.chat.tools_required_named_parallel` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/base-official-sdks.log` records 502 for no required/exclusive `weather` call; matching `server-base.log` records the rejection. | Quality note; no code change. |
| SDK `openai.responses.function_roundtrip` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/base-official-sdks.log` records 502 `did not exclusively call required function 'weather'`; matching server logs record Responses 502. | Quality note; no code change. |
| SDK `anthropic.messages.system_tool_choices` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/base-official-sdks.log` records API-shaped 502 `model did not emit a required tool call`; server logs agree. | Quality note; no code change. |
| SDK `anthropic.messages.tool_result_roundtrip` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/base-official-sdks.log` records API-shaped 502 for no exclusive `weather` call, matching the retained core round-trip failure and server logs. | Quality note; no code change. |
| SDK `anthropic.messages.stop_and_length` | Muse | MODEL BEHAVIOUR | `results/smoke-muse-*/base-official-sdks.log` records the assertion; the same prompt in `raw/core/034-messages-stop-sequences.json` shows EOS before `END`, with truthful `end_turn`/null stop metadata. | Quality note; no code change. |
| Core `messages_stop_sequences` | North | MODEL BEHAVIOUR | `results/smoke-north-*/raw/core/034-messages-stop-sequences.json` returns `A` plus whitespace, `end_turn`, and no stop sequence; server logs correctly report 200. | Quality note; no code change. |
| SDK `anthropic.messages.system_tool_choices` | North | MODEL BEHAVIOUR | `results/smoke-north-*/base-official-sdks.log` records either no required call or `Model produced an incomplete North action block`; `raw/core/029-messages-tools-named.json` proves complete named calls parse on both routes. | Quality note; preserve fail-closed malformed-action handling. |
| SDK `anthropic.messages.tool_result_roundtrip` | North | MODEL BEHAVIOUR | `results/smoke-north-*/base-official-sdks.log` records incomplete North action blocks; `raw/core/030-messages-tool-result-roundtrip.json` proves the longer core round trip succeeds on both routes, isolating prompt/output behavior rather than the API bridge. | Quality note; no code change. |
| SDK `anthropic.messages.stop_and_length` | North | MODEL BEHAVIOUR | `results/smoke-north-*/base-official-sdks.log` records the assertion; route-local core raw `034` shows the model ends before `END` and the server reports that accurately. | Quality note; no code change. |
| Core `stop_strings` | `smoke-xing-mtp1` | MODEL BEHAVIOUR | `results/smoke-xing-mtp1/raw/core/003-stop-strings.json` emits an unrelated campaign essay to 96 tokens, `finish_reason=length`, and no requested marker; server log records 200. | Quality note; no code change. |
| Core `messages_tools_named` | `smoke-xing-mtp1` | MODEL BEHAVIOUR | `results/smoke-xing-mtp1/raw/core/029-messages-tools-named.json` is an API-shaped 502 for no exclusive `weather` call; server log confirms the rejection. | Quality note; no code change. |
| Core `messages_tool_result_roundtrip` | `smoke-xing-ordinary`, `smoke-xing-prompt-lookup` | MODEL BEHAVIOUR | `results/smoke-xing-{ordinary,prompt-lookup}/raw/core/030-messages-tool-result-roundtrip.json` fails closed because the required first named call is absent; other Xing named-tool checks pass, so parsing is available. | Quality note; no code change. |
| Core `messages_stop_sequences` | Xing | MODEL BEHAVIOUR | `results/smoke-xing-*/raw/core/034-messages-stop-sequences.json` emits `I wrote A`, `AI`, or an unrelated assistant introduction, then EOS; every reply truthfully reports `end_turn` and null stop sequence. | Quality note; no code change. |
| SDK `anthropic.messages.stop_and_length` | Xing | MODEL BEHAVIOUR | `results/smoke-xing-*/base-official-sdks.log` records the assertion; route-local raw core `034` supplies the generated text and confirms that `END` was never emitted. | Quality note; no code change. |
| Core `completions` | `smoke-gemma3n-ordinary` | MODEL BEHAVIOUR | `results/smoke-gemma3n-ordinary/raw/core/001-completions.json` is HTTP 200, EOS after one token, and empty text; server log records a successful request, not a runtime error. | Quality note; no code change. |
| Core `stop_strings` | `smoke-gemma3n-ordinary` | MODEL BEHAVIOUR | `results/smoke-gemma3n-ordinary/raw/core/003-stop-strings.json` loops on `campaign` to 96 tokens without exact `CAMPAIGN_STOP`; `finish_reason=length` is correct. | Quality note; no code change. |
| Core `multimodal_audio` | `smoke-gemma3n-ordinary` | MODEL BEHAVIOUR | `results/smoke-gemma3n-ordinary/raw/core/040-multimodal-audio.json` accepts the WAV and returns HTTP 200, EOS after one token, and empty text; image/text media checks pass in the same feature file. | Quality note; no code change. |
| Qualifier `hermes_client` | `smoke-gemma3n-ordinary` | MODEL BEHAVIOUR | `results/smoke-gemma3n-ordinary/qualification.json` retains an HTTP 200 refusal instead of exact `HERMES_READY`; receipt shows ordinary execution and thinking disabled. | Quality note; keep exact oracle. |
| Core `stop_strings` | `smoke-minicpmo-ordinary` | MODEL BEHAVIOUR | `results/smoke-minicpmo-ordinary/raw/core/003-stop-strings.json` emits `CAMPAIGN\\_STOP`, not the requested byte sequence, then reaches the 96-token limit; server stop metadata is correct. | Quality note; no code change. |
| Qualifier `hermes_client` | `smoke-minicpmo-ordinary` | MODEL BEHAVIOUR | `results/smoke-minicpmo-ordinary/qualification.json` retains HTTP 200 text `HERMES\\_READY`; the inserted backslash violates the exact reply instruction and the ordinary-route receipt is healthy. | Quality note; keep exact oracle. |

## Orchestrator verdict

Accepted MODEL BEHAVIOUR quality notes, non-blocking for 20x20: every Muse
failure listed above; North `messages_stop_sequences` plus the three Anthropic
SDK failures; Xing MTP1 `stop_strings`, the three route-specific Anthropic tool
failures, all three `messages_stop_sequences`, and all three SDK
`stop_and_length` failures; Gemma completions, stop string, audio, and
`hermes_client`; MiniCPM-o stop string and `hermes_client`.

Stages that must rerun because a harness defect was fixed:

- `smoke-qwen36-ordinary`
- `smoke-flash-next-mtp2`

No mlx2 product defect was found in the 17-stage pass-5 result set.  Laguna is
outside this verdict and remains governed by its separate diagnostic rerun.
