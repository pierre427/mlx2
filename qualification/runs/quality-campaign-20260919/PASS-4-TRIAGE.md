# Smoke pass 4 triage

Pass 4 served the immutable source copy at `458f792` and reached `phase=done`
at 05:14:32 with all 18 stage summaries available.  This report classifies
every failure in that final `status.json`.  The fixes below were validated on
CPU only; the affected GPU stages remain unqualified until rerun from the new
commit.  The feature harness is a worktree script, so later stages also began
retaining request bodies after that repair; the served runtime, qualifier, and
official-SDK script stayed pinned to the detached copy.

## Findings

| Check or failure family | Affected stages | Classification | Evidence | Resolution |
| --- | --- | --- | --- | --- |
| Qualifier `reasoning` | Qwen3.6 ordinary, MTP2, prompt lookup | Harness (qualifier regression) | `results/smoke-qwen36-ordinary/qualification.json` records three attempts capped at 512, 1024, and 2048 tokens, each with non-empty reasoning, empty final content, and `finish_reason=length`; the MTP2 and prompt-lookup records show the same progression.  `with_thinking_budget` added no allowance because `/v1/status.thinking_default` is false, even though these adapters declare structured-output thinking deferral.  The campaign server had no anchor budget. | Send a 128-token request thinking budget when the adapter declares thinking deferral, for both plain and structured reasoning probes.  The check still requires private reasoning plus the final answer `221`; adapters without a declared close token retain their natural-closing path.  Re-pin the approved qualifier hash in `src/mlx2/qualification.py`. |
| Earlier reasoning qualification discrepancy | Historical Qwen3.6, Xing, and North records | Historical harness/profile explanation | `qualification/runs/qwen36-35b-a3b/targeted-pass4/ordinary-qualification-r7.json` used the old literal `</think>` prompt and false-passed after a length-capped response was parsed as visible content containing `221`.  `qualification/runs/xing4-0-gpu-20260918/x6-ordinary-qualification.json` declares a 4096-token model allowance and naturally closes.  `qualification/runs/north-alpha-calibration-20260918/default-on-qualification.json` was served by `run_gpu.sh` with a 512-token thinking anchor and steering; North's high guard resolves to 2048. | Do not restore the control-token prompt or weaken the oracle.  The bounded current probe exercises the adapter-owned `ThinkingGuard`; no campaign-wide server anchor is needed. |
| Qualifier `feature_apc_persistence` | Every base cycle that reached the qualifier's final feature checks | Harness configuration | The base server commands in `status.json` select `--apc-persist-dir --apc-persist-on-shutdown`, but the qualifier observes only that one process and therefore cannot witness a shutdown/rescan/restore lifecycle.  Qwen3.6 aborted earlier at `reasoning`; completed qualifiers from Qwen3.8 onward consistently expose this failure.  The separate `feature-persist-seed.json` and `feature-persist-rescan.json` cycles pass the actual lifecycle. | Do not advertise persistence on base or opt-in servers.  Select it only for the seed and rescan cycles, where it is both exercised and observed. |
| SDK `openai.chat.nonstream` JSON decode | Qwen3.6 MTP2 and prompt lookup | Harness | The case requests strict JSON schema with high reasoning and `max_tokens=256` but supplied no thinking budget.  Its capped reasoning reply has no final JSON, producing the SDK `JSONDecodeError`.  Route-local `raw/core/018-structured-after-thinking.json` is HTTP 200 with valid JSON and `structured_output.enforced=true` on ordinary, MTP2, and prompt lookup when a 128-token budget is supplied. | Add the same 128-token thinking budget to this structured SDK case and include the raw choice in any future parse exception.  No speculative-grammar product change is warranted. |
| Flash-Next constrained tool grammar | Flash-Next ordinary | Harness oracle | `results/smoke-flash-next-ordinary/raw/opt-in/000-constrained-tool-grammar.json` is HTTP 502 because the model exhausted 128 tokens inside an unconstrained city string.  The grammar correctly permitted that unbounded schema continuation and failed closed on the incomplete call. | Use a finite exact-language schema (`city` must equal `Toronto`), disable parallel calls, and validate the exact call.  Future raw evidence stores both request and response. |
| Flash-Next server health failure, `OrderedDict mutated during iteration` | Flash-Next MTP2 | Product | `results/smoke-flash-next-mtp2/server-base.log` traces the 503 through `FlashNextAdapter.diagnostics()` into `self.model.named_modules()` while MLX was lazily mutating the module tree during speculative execution. | Snapshot module references after load/quantization and use that immutable tuple for MoE and fused-GDN diagnostic counters.  A CPU regression makes `named_modules()` raise after the snapshot and proves diagnostics no longer rewalk it. |
| Muse budget-state and budget-history checks | Muse ordinary | Harness applicability | `results/smoke-muse-ordinary/raw/core/015-thinking-budget-state-aware.json` and `016-thinking-budget-history.json` are the correct HTTP 400: `thinking_budget needs an adapter-declared thinking-close token`.  Muse declares reasoning but not thinking deferral.  DFlash2 and prompt lookup correctly skipped these checks because their runs began after the harness repair. | Require both reasoning and thinking-deferral capabilities before sending explicit budget checks. |
| Required/named tool checks and round trips | All Muse routes; both North routes | Model behaviour | Muse feature raws `011`-`013`, `021`, and `028`-`030` return fail-closed 502 errors for no required call or a call not exclusively named `weather`; its auto-tool raw `010` demonstrates that parsing works when a call is actually emitted.  North's official SDK logs report incomplete North action blocks.  `openai_compat.enforce_tool_contract` requires at least one call for `required`, exactly the selected name for named choice, and no more than one for `parallel_tool_calls=false`. | No code change.  Keep the 502 contract enforcement and record these as model-quality failures.  The rejected free text is not retained by the pass-4 response artifact, so text alongside the missing/incomplete call cannot be reconstructed after the fact. |
| Anthropic `messages_stop_sequences` / SDK `stop_and_length` | All Muse routes; both North routes | Model behaviour | `results/smoke-muse-ordinary/raw/core/034-messages-stop-sequences.json` and `results/smoke-north-ordinary/raw/core/034-messages-stop-sequences.json` return HTTP 200 with text `A`, `stop_reason=end_turn`, and `stop_sequence=null`: each model ended before emitting `END`.  The SDK case checks the same exact marker before its separate one-token length assertion. | No code change.  The server accurately reports EOS rather than claiming that an absent client stop sequence fired. |
| Laguna structured output | Laguna ordinary | Product; root cause deferred | Feature raws `007`, `008`, `009`, and `018` all return HTTP 502 `structured-output grammar has no valid token continuation`; the qualifier and both structured official-SDK cases fail at the same seam.  CPU inspection loaded only the real tokenizer: representative JSON/regex paths completed, every token transition in the strict schema's 145 reachable automaton states remained viable, and no dead state was reachable.  Pass 4 retained neither the rejected generated token IDs nor decoded prefix, so it does not distinguish an integration-history error from a bad sampled token after masking. | Do not guess at a tokenizer or tensor fix.  This remains a product defect and unqualified route.  A GPU rerun must retain the failing constrained prefix before a targeted CPU regression and correction can be made. |
| SDK `anthropic.messages.thinking_signature_roundtrip` | Laguna ordinary | Harness | `raw/core/031-messages-thinking-signature-roundtrip.json` proves the first response contains signed thinking and the signed history is accepted by a 200 follow-up.  The follow-up answers directly with non-empty text; the SDK case incorrectly required it to open a second thinking block. | Accept either non-empty thinking or non-empty text on the successful follow-up while still requiring the first signature. |
| Anthropic tool choices | Xing ordinary (`any`, named), MTP1 (tool-result named follow-up), prompt lookup (named) | Model behaviour | Xing feature raws `028-messages-tools-any.json`, `029-messages-tools-named.json`, and MTP1 `030-messages-tool-result-roundtrip.json` are fail-closed 502 replies for no required call or no exclusively named `weather` call.  OpenAI tool cases and the corresponding official-SDK cases pass, so the runtime parser/enforcement path is functioning. | No code change; retain as interface-specific model quality evidence. |
| OpenAI/Anthropic exact stop markers | Xing ordinary, MTP1, and prompt lookup | Model behaviour | Ordinary `raw/core/034-messages-stop-sequences.json` returns `My result: B`; MTP1 `003-stop-strings.json` writes an unrelated campaign essay to the token cap and `034` starts explaining the prompt; prompt lookup `034` emits only `A`.  None emits the requested exact marker, and the responses correctly report length or EOS rather than a stop-sequence match. | No code change; the server's stop metadata is correct. |
| Qualifier `hermes_client` | Gemma 3n ordinary; MiniCPM-o ordinary | Model behaviour | Gemma's `qualification.json` contains an HTTP 200 refusal instead of `HERMES_READY`.  MiniCPM returns `HERMES\\_READY`, a different byte sequence.  Both receipts show thinking disabled as requested and ordinary execution. | No code change; retain the exact-text qualification oracle. |
| Empty completion and missed stop markers | Gemma 3n ordinary | Model behaviour | `raw/core/001-completions.json` returns HTTP 200, EOS after one token, and empty text.  `003-stop-strings.json` loops on `campaign` to the 96-token cap, and `034-messages-stop-sequences.json` emits unrelated text then EOS; neither contains the requested marker. | No code change; the serving metadata matches the generated tokens. |
| Silent audio produces no answer | Gemma 3n ordinary | Model behaviour | `raw/core/040-multimodal-audio.json` shows the valid WAV request was accepted, then the model returned HTTP 200, EOS after one token, and empty text.  Image and plain-text multimodal checks pass in the same run. | No code change; retain as multimodal model-quality evidence. |
| Escaped exact stop marker | MiniCPM-o ordinary | Model behaviour | `raw/core/003-stop-strings.json` emits `CAMPAIGN\\_STOP`, not the requested byte sequence `CAMPAIGN_STOP`, so the server correctly reaches the token cap without claiming a stop match. | No code change; retain exact API stop semantics. |

## CPU verification

- Required full pytest command: 1615 tests collected, 100% completed, exit 0.
- All-phase dry run: `DRY-RUN PASS cpu_only=true models=9 policies=25 smoke=18 sanity=18 ladder=9`.
- Changed-Python undefined-name/import lint: `ruff check --no-cache --select F,E9`, all checks passed.

## Exact smoke rerun set

The persistence-selection correction changes every base-cycle receipt, so all
18 smoke stages must be rerun even when their only observed failure was that
global harness mismatch:

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
