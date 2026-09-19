# Smoke pass 3 triage

Pass 3 served the immutable source copy at `da00a92`, but its preflight receipt
came from pass 1 and its campaign harness was repaired while later stages were
still running. The results are therefore diagnostic evidence, not a coherent
qualification receipt. All smoke stages must be rerun from one pinned source
and harness identity.

The pass reached `phase=done` at 03:56:01 with all 18 stage summaries marked
failed. All 18 base qualifiers contain the stale-preflight assertion. All 18
standalone APC Metal-prefetch receipts passed exact token-ID equality; all
persistence cycles passed except the Muse dFlash2 cascade caused by the
external-draft insertion exception classified below. MiniCPM's final base
cycle added no new product exception: its feature failure was the escaped stop
marker below, and its stale SDK matrix repeated the capability/lifecycle
harness failures already classified here.

## Findings

| Check or failure family | Affected stages | Classification | Evidence | Resolution |
| --- | --- | --- | --- | --- |
| Qualifier identity mismatch | Every completed stage | Harness | `results/smoke-qwen36-ordinary/base-qualifier.log` and peers report that the preflight receipt does not match the served source/harness identity. | `run_campaign.py` now regenerates preflight unconditionally against `MLX2_CAMPAIGN_ROOT` before starting stages. |
| `tools_parallel_false` with a parsed second call | Qwen3.6, Qwen3.8, Flash-Next, Xing routes | Harness expectation; product enforcement correct | `results/smoke-qwen36-ordinary/raw/core/013-tools-parallel-false.json` is HTTP 502, `parallel_tool_calls:false permits at most one tool call`; Xing's post-check variant says `model emitted parallel calls while parallel_tool_calls was false`. | Treat either exact fail-closed contract error, or an HTTP 200 reply with one call, as enforcement success. |
| Required/named tool produces no, wrong, or incomplete call | Muse, North, Xing, and several official-SDK cases | Model behaviour | `results/smoke-muse-ordinary/raw/core/011-tools-required.json` says no required call; named says it did not exclusively call `weather`; North/Xing logs report incomplete action/tool blocks. `src/mlx2/openai_compat.py::enforce_tool_contract` rejects these cases by contract. | Keep as model-quality failures. Prompts are now explicit; `tool_choice:auto` is no longer graded as though it required a call. The 502 artifact does not retain discarded free text, so text alongside a rejected call cannot be reconstructed from pass 3. |
| Admin quiesce/suspend/resume | Early text-model routes | Harness | `results/smoke-qwen36-ordinary/raw/core/038-apc-admin-quiesce-suspend-resume.json`: cold and warm are both `ADMIN_OK`; quiesce/resume are HTTP 202; resume reports `drain_cancelled` because the harness never waited beyond `draining`. | Wait for `suspended`, resume with explicit session prefetch, wait for `resident`, and compare warm against warm. The CPU tiny hybrid check additionally proves exact pre-suspend/post-resume token IDs and unchanged APC entry count through the same engine lifecycle. |
| SDK JSON schema parse failures | All reasoning text adapters; Muse returned an explicit 400 | Harness | `*-official-sdks.log` shows `JSONDecodeError` after a fixture-only `__json__` prompt and an eight-token reasoning budget. Muse correctly says structured output requires thinking disabled. | Use a real JSON instruction and 256-token budget; capability-gate reasoning-to-grammar deferral. |
| Responses typed stream event assertion | Text adapters | Harness | Official-SDK logs show `AssertionError`; feature raw streams contain multiple valid `response.output_text.delta` events. | Validate ordered lifecycle events while allowing one or more delta events. |
| Responses/Anthropic function-result round trips | Qwen3.8 and Flash-Next 500s; other models sometimes 502 before a call | Harness for the 500s; model behaviour for no call | Server logs show Jinja `No user query found in messages`; the second stateless request omitted its original user turn. A 502 before the second turn is a genuine missing/wrong tool call. | Preserve the original user item/turn in both round-trip histories. Keep genuine tool-obedience failures as quality notes. |
| SDK stop/stream and feature text checks consume only reasoning | North, Laguna, Xing; several SDK runs | Harness | Raw replies have HTTP 200 with non-empty reasoning but empty visible text at the output cap, for example `results/smoke-xing-ordinary/raw/core/006-seeded-determinism.json`. | Disable thinking for checks unrelated to reasoning; bound stored Responses generations; retain dedicated default-thinking and budget checks. |
| SDK requests undeclared tools/reasoning and leaves generations unbounded | Gemma and other non-tool/non-reasoning adapters | Harness | Gemma's SDK log runs a reasoning+schema case that returns the correct 400, then requests required tools despite no declared tool capability; the former tool request also ran at the server default token ceiling. | Read `/v1/status` capabilities, skip undeclared feature cases, and put an explicit output-token bound on every generative SDK request. |
| `apc_skip_writing_prefix_cache` | Muse, North, Laguna, Xing | Harness | The two requests report the same pre-existing shared-prefix hit. The flag prevents new writes; it does not disable reads. | Assert that the second request does not increase cached tokens over the first. |
| Thinking signature / structured-after-thinking on Muse | All Muse routes | Harness applicability | Raw replies are HTTP 400: no adapter-declared thinking-close token. | Muse no longer declares `thinking-deferral`; deferral/signature checks and SDK cases skip. |
| FLy verification | Qwen3.6 MTP2, Qwen3.8 MTP2, Flash-Next MTP2 | Harness | `raw/opt-in/003-fly-verification.json` shows `verification=exact`, `fly_disabled_reason=logits_processors`, with inherited `presence_penalty=1.5`; the grader also read the wrong counter level. | Send neutral penalties and require the route receipt itself to report `verification=fly` and not disabled. |
| Interior APC opt-in startup refusal | Qwen3.6 PLD; Muse, North, Laguna, and Xing routes | Harness configuration; product refusal correct | `server-opt-in.log` says interior checkpoints cannot capture on the selected prompt-lookup, external-draft, or adapter cache route. | Declare interior APC per route, remove unsupported policy keys, and skip its check elsewhere. Xing PLD subsequently started and passed its opt-in cycle with the repaired policy. |
| Flash-Next constrained tool grammar | Flash-Next ordinary | Model behaviour | `results/smoke-flash-next-ordinary/raw/opt-in/000-constrained-tool-grammar.json` is HTTP 502 `model produced an incomplete tool call`; the opt-in policy engaged and failed closed. | No product change; keep as a quality note. |
| Muse dFlash2 HTTP 500 cascade | Muse dFlash2 base/persistence/SDK checks | Product | `results/smoke-muse-dflash2/server-base.log` repeatedly reports `ExternalDraftBatchGenerator.insert() got an unexpected keyword argument 'apc_interior_positions'`. | Align the external-draft insertion seam with serving kwargs; accept empty ordinary metadata and fail closed if unsupported interior/multimodal metadata is actually supplied. |
| Gemma output contains `▁` boundaries | Gemma ordinary | Product | `results/smoke-gemma3n-ordinary/raw/core/000-chat.json` contains `I'm▁sorry...`; the shared mlx-vlm adapter selected the byte-BPE detokenizer. | Adapter-owned selection now gives Gemma the SentencePiece streaming detokenizer while MiniCPM keeps byte-BPE. |
| Gemma image input rejected | Gemma ordinary | Harness | `raw/core/039-multimodal-image.json` is HTTP 400 `image minimum dimension is 3 pixels`; the synthetic PNG was 8x1. | Generate a valid 8x8 PNG; a CPU test checks its IHDR dimensions. |
| Gemma silent-audio response is empty | Gemma ordinary | Model behaviour | `raw/core/040-multimodal-audio.json` is HTTP 200 with `finish_reason=stop` and empty text for 100 ms of silence. | Quality note only; mlx2 accepted and executed the input contract. |
| Gemma raw completion empty and requested stop marker absent | Gemma ordinary | Model behaviour | `raw/core/001-completions.json` ends immediately with empty text; `003-stop-strings.json` loops on `campaign` and never emits `CAMPAIGN_STOP`. mlx2 returns a valid HTTP 200 response in both cases. | Quality note only; no runtime change. |
| MiniCPM exact stop marker absent | MiniCPM ordinary | Model behaviour | `results/smoke-minicpmo-ordinary/raw/core/003-stop-strings.json` emits the Markdown-escaped text `CAMPAIGN\\_STOP`, not the requested byte sequence `CAMPAIGN_STOP`; HTTP 200 therefore correctly ends at the token limit. | Quality note only; retain the exact API stop-sequence assertion. |
| Laguna structured grammar has no continuation | Laguna ordinary | Product candidate, deferred | `raw/core/{007-json-object,008-strict-json-schema-ref,009-regex-grammar,018-structured-after-thinking}.json` all fail closed with `structured-output grammar has no valid token continuation`. CPU tokenizer checks admit and complete representative `YES` and JSON token paths, but pass 3 retained no generated token trace. | Do not guess at a tensor/runtime fix. Rerun Laguna with the corrected non-thinking harness and capture the failing token prefix if it recurs. |

The standalone APC Metal-prefetch checks passed in pass 3 because they park the
session, wait for the worker idle boundary, prefetch to resident state, and
compare token IDs. The former feature check instead resumed immediately from
`draining`; its failure label incorrectly implied a warm/cold output mismatch.

## Exact smoke rerun set

```text
smoke-qwen36-ordinary
smoke-qwen36-mtp2
smoke-qwen36-prompt-lookup
smoke-qwen38-ordinary
smoke-qwen38-mtp2
smoke-flash-next-ordinary
smoke-flash-next-mtp2
smoke-muse-ordinary
smoke-muse-dflash2
smoke-muse-prompt-lookup
smoke-north-ordinary
smoke-north-prompt-lookup
smoke-laguna-ordinary
smoke-xing-ordinary
smoke-xing-mtp1
smoke-xing-prompt-lookup
smoke-gemma3n-ordinary
smoke-minicpmo-ordinary
```
