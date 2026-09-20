# Optimized unified API parity

Source comparison: local unified `1e2bc604f71d070bee970c3e7db8b60f7855599b`.

Wire-shape baseline: official OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create),
[Chat Completions API](https://developers.openai.com/api/reference/resources/chat/methods/create),
[Embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create),
[Batch guide](https://developers.openai.com/api/docs/guides/batch),
and [function-calling guide](https://developers.openai.com/api/docs/guides/function-calling),
plus Anthropic's [Messages API](https://platform.claude.com/docs/en/api/messages/create),
reviewed 2026-09-19. Compatibility below is an explicitly bounded subset, not
an assertion that unsupported OpenAI platform state or hosted services exist
locally.

Dynamic adapter control follows vLLM's documented
[`/v1/load_lora_adapter` and `/v1/unload_lora_adapter`](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
shape. The engine implements a drained, revision-bound single-adapter LoRA
lifecycle when an operator configures an allowlisted `--lora-dir`.

The loopback-only `/v1/admin/quiesce`, `/v1/admin/resume`, and
`/v1/admin/state` routes are mlx2 operator extensions, not OpenAI or Anthropic
compatibility claims. While admission is closed, model-executing endpoints
retain their native OpenAI- or Anthropic-shaped 503 errors and include
`Retry-After`; read-only resource routes remain available. See
[Quiesce, suspend and resume](SERVING.md#quiesce-suspend-and-resume).

## Token probabilities

`logprobs: true` and `top_logprobs: 0..11` now return per-token
`choices[0].logprobs.content`. A positive top count also requests probabilities.
Each entry identifies the **actual emitted token**, with optional sorted
alternatives. Negative infinity is represented as -9999 for valid JSON.
Stop-token entries are included, matching execution token accounting and source
behavior. Token strings use the tokenizer's vocabulary pieces, as unified does.
Every decode route that normalizes target logits does so in float32 before
sampling and reporting, including ordinary, prompt-lookup, native-MTP and
sampled external-draft verification; temperature-zero verification uses the
equivalent raw-logit argmax. Low-precision normalization therefore cannot
create artificial ties.

**Returned logprob values are not reproducible under concurrency.** Two
identical requests decoded in the same batch can receive different logprob
values for the same emitted token, and the same request repeated across server
runs can differ again. Measured on this stack (Flash-Next, greedy, identical
prompts and batch composition): at width one, eight of eight prompts reproduced
token-for-token; at width 16, one of eight did. Two lanes running the *same*
prompt in one batch reported the same top-two tokens with margins of 0.25 and
0.625 nats at the same position. The emitted tokens are unaffected except where
two candidates are within a bf16 rounding step, in which case the tie may break
either way. Clients that need reproducible logprobs or token-identical output
must serve at width one, and clients must not treat a logprob margin measured
under concurrency as a stable quantity. This is a property of batched GPU
reduction order, not a defect.

Nonstreaming responses collect entries in generation order. Streaming sends each
probability entry once in its own choice chunk; text parsing can delay or group
text independently. Unified currently only serializes these entries in its
nonstreaming response. Probability requests perform additional device-to-host
reads; requests without them do no additional probability serialization.

These are the execution route's target distributions. Ordinary decoding reports
normalized post-processor logits before the sampler's temperature/top filters.
Native MTP reports target verifier probabilities **after** its temperature/top
filters. Accepted draft tokens use the corresponding target row; rejected-draft
replacement/bonus tokens use the target row at the accepted-prefix boundary,
never the draft or residual sampling distribution. These are existing unified
runtime semantics, not a promise of equal numerical values across differently
configured routes. CPU tests execute the actual production output-construction
expression for zero, one and two accepted draft tokens.

## Other controls

- `max_tokens`, its Chat alias `max_completion_tokens`, Responses
  `max_output_tokens`, and Hermes `options.num_predict` accept 1 through
  2,097,152 tokens; `min_tokens` accepts 0 through the same ceiling. When an
  OpenAI-shaped request omits its output cap, serving admission uses
  `min(--default-max-tokens, effective_context - prompt_tokens)`, with at least
  one output token required; `--default-max-tokens` defaults to 65,536. At that
  default, prompt-aware clamping cannot turn a request that fit the prior
  512-token default into a context rejection. An explicit cap is never clamped:
  prompt plus requested output must fit or the request fails. The receipt
  records the effective `max_tokens` and `max_tokens_defaulted`.
- Explicit `thinking_budget` on OpenAI-shaped and Hermes/Ollama requests is
  independently bounded from 0 through 2,097,152; it is not constrained by the
  output cap. Only Anthropic `thinking.budget_tokens` must be less than its
  required `/v1/messages` `max_tokens`. Effort-scaled default thinking budgets
  retain their independent 8,192-token ceiling.
- `response_format: {"type":"text"}` is accepted and normalized away.
- Regex grammar, JSON object and a bounded canonical strict-JSON-schema subset
  are implemented. Unknown schema keywords, match-budget overruns and empty
  continuations fail closed. Production use requires the selected profile to
  declare `grammar` and a matching qualification receipt with observed strict
  schema enforcement.
- Nonstreaming `n=1..8` uses independent seeds and request state. It reserves
  the whole cohort before queue publication and rejects requests that exceed
  lane, inflight or measured physical-headroom bounds.
- Tool `description` is optional (OpenAI and Anthropic both require only the
  name and the schema), so a definition without one is accepted and rendered
  with an empty description instead of failing the chat template.
- `tool_choice` accepts `auto`, `none`, `required`, and one named function.
  `parallel_tool_calls:false` is enforced by the output parser: a model that
  emits a second call under it has the surplus dropped (counted as
  `tool_calls{reason="parallel_bound_truncated"}`) rather than failing the
  request, so over-calling is never reported as a server fault on any
  surface or mid-stream. Strict function schemas use the same
  bounded canonical JSON-schema subset as structured output; every successful
  call is parsed and checked after generation, while missing required/named
  calls or invalid arguments fail closed. Responses streaming buffers function
  calls through terminal validation before emitting typed argument events;
  strict Chat streaming uses the same terminal buffer: mlx2 validates the
  complete call before sending HTTP headers, then emits standard Chat chunks.
  Main's post-generation validation is the default. Required/named choices
  install the adapter-owned decode grammar only when the serving qualification
  selects `constrained_tool_grammar:true`; unsupported adapters and conflicting
  request constraints skip it with a receipt and counter. EOS/stop before a
  required call remains a post-generation contract failure; exhausting
  `max_tokens` reports `length` / Anthropic `max_tokens` and preserves already
  parsed content instead of reclassifying truncation as malformed output.
  Muse's thinking-off recipient header is enforced separately from that opt-in
  body grammar: named choice opens the named recipient, required permits only
  tool recipients, and auto permits user or tool recipients while excluding
  the `self` reasoning channel.
  North required/named requests similarly install a history-pure action-branch
  processor. It forces the template's `<|START_ACTION|>` path immediately when
  thinking is off, or only after `<|END_THINKING|>` when thinking is on; `auto`
  and `none` remain unconstrained.
- `POST /v1/responses` implements a bounded text/function subset: string or
  text-message input, instructions, sampling/output limits, metadata, text
  formats, function tools, tool choice, usage, output message items,
  function-call items, typed text/function SSE events, and mlx2 route receipts.
  A tenant-scoped store supports retrieve, delete, `previous_response_id`, and
  the cursor-shaped `GET /v1/responses/{response_id}/input_items` resource.
  Input-item pages are reconstructed from the continuation context main already
  stores; the raw Responses input is never retained a second time and therefore
  does not change the store's byte accounting or eviction boundary. File data
  is exposed only as its resolved text, reasoning signatures are regenerated
  only when explicitly included, and unavailable legacy context returns 409.
  Unknown `include` values return 400.
  It is process-local by default or atomically restart-durable with
  `--api-state-dir`.
  Signed reasoning items use `include:["reasoning.encrypted_content"]`: the
  payload is HMAC-authenticated for model and tenant (not encrypted), invalid
  or cross-scope values are dropped and counted. Reasoning output items are
  emitted only for that include value; only those explicitly requested items
  are retained in stored continuation context. Streaming and non-streaming
  terminal payloads are identical for requested reasoning and logprobs.
  `message.output_text.logprobs` serializes token bytes only when requested,
  using typed Responses text-delta events rather than Chat chunks.
  Common client fields are validated and ignored when they do not alter local
  execution. Store failures retain main's request-failure semantics.
- `POST /v1/messages` and `/v1/messages/count_tokens` implement the bounded
  Anthropic text/tool subset over the same engine. Errors and SSE are
  Anthropic-shaped, including unknown-model and streamed contract failures;
  thinking blocks carry signed `signature_delta`, cache reads
  are excluded from `input_tokens`, tool choice maps onto shared controls,
  assistant prefill is rejected, and invalid model tool JSON is 502 (or an
  in-stream error). `thinking.budget_tokens` selects history-pure close mode
  and must remain one token below the required `max_tokens` value. Extended
  thinking is off when `thinking` is omitted or explicitly disabled; only
  `thinking: {"type":"enabled", ...}` enables an adapter's thinking mode.
  A matched client stop remains OpenAI `finish_reason:"stop"`, while Messages
  reports `stop_reason:"stop_sequence"` and the exact matched string for every
  adapter-owned output parser.
  Assistant `text.citations:null` and `tool_use.caller:null` emitted by the
  official SDK round-trip safely; a direct caller is accepted, while actual
  citations and server-tool callers remain outside this client-tool subset and
  fail closed.
- Streaming Chat, Completions, Responses and Messages accept
  `return_progress: true` (Splash/llama.cpp-style prefill progress; not an
  OpenAI or Anthropic field). Updates `{processed, total, cached, replay,
  time_ms}` ride an empty-delta Chat chunk, `response.in_progress`, or an
  Anthropic `ping`, all standard event types that official SDKs accept; see
  [Prompt progress](SERVING.md#prompt-progress-tokenize-and-apply-template).
  Non-streaming requests that set it get 400. `POST /tokenize` (`{tokens,
  count, max_model_len}`) and `POST /apply-template` (`{prompt}`) are mlx2
  extensions in the vLLM/llama.cpp/Splash style over Chat or Completions
  bodies; they run no inference, reject media parts, and `/apply-template`
  returns 501 when the adapter has no text renderer.
- The Files API supports bounded multipart storage, retrieval, deletion and
  cursor listing. UTF-8 text-file inputs are rendered into text prompts.
  The JSONL Batch API supports chat, completions, Responses and embeddings with
  standard request/result rows, cursor listing and a `24h` completion-window
  contract. Restored active batches fail explicitly with `server_restarted`.
- Embeddings execute a mean-pooled, L2-normalized decoder input representation;
  reranking uses cosine similarity over it. This is implemented for decoder
  adapters but is not represented as retrieval-model qualification.
- Dynamic LoRA supports explicit Linear/QuantizedLinear keys, exact tensor
  coverage, zero serving dropout, one active adapter, inflight drain, reversible
  unload, cache invalidation and revision receipts. It is disabled until
  `--lora-dir` is configured and remains GPU-unqualified.
- Responses image, video and audio parts cross an explicit multimodal adapter
  contract. Bounded data/File decoding, image checks, WAV metadata and capped
  video sampling are available. Optional mlx-vlm adapters now implement Gemma
  3n image/audio plus ordered native-video requests (uniform frames, explicit
  timestamps and media-bound APCv2 identity) and MiniCPM-o image/audio. The
  MiniCPM-o route honors artifact slicing, bounded shape-coherent vision
  batches and exact-rate audio chunks. The real downloaded processors are
  host-qualified in
  `qualification/runs/multimodal-processors-20260918/receipt.json`, without
  importing MLX or weights. Production receipts must additionally prove image,
  video/input-audio, encoder batching and media-APCv2 checks as declared by the
  adapter; processor evidence alone cannot select those capabilities. Text-only
  adapters return 501 rather than simulating a media tower.
- `POST /v1/audio/speech` implements the OpenAI binary output-audio boundary
  (`input`, named/custom `voice`, optional `instructions`, `response_format`,
  `speed`, and `stream_format:"audio"`). SSE is recognized but returns 501
  until an adapter owns a qualified audio-event stream. Binary output dispatches only when the
  selected route declares qualified `output_audio` and returns a typed
  `AudioOutput`; MIME mismatches fail closed. Current upstream mlx-vlm can load
  MiniCPM-o TTS tensors, but its generation contract is a spoken chat response
  conditioned on a reference WAV. That is not the named-voice, exact-input
  speech contract above, so mlx2 deliberately does not advertise
  `output_audio` until a voice registry and qualified renderer are present.
- Responses MCP tools execute through an optional operator-allowlisted
  Streamable HTTP backend (`--tool-backend-config`). OpenAI-shaped server and
  tool filters are honored, approval must be `"never"`, and model/tool loops
  are capped at eight. Other hosted tools remain unavailable.

## Coding-agent clients (`--agent-compat`)

State: **implemented** and CPU-tested against the real clients;
**GPU-unverified**. It is selected per request:

- the `X-MLX2-Agent-Compat` header;
- the tenant policy;
- in the default `--agent-compat auto` mode, detection of the real clients'
  identity headers.

Plain SDK requests behave exactly as on main. See
[SERVING: coding-agent client compatibility](SERVING.md#coding-agent-client-compatibility-per-request-auto-detected)
for precedence, receipts, spoofing and conversation-consistency rules.

Against main, both clients fail on their first request.

- **codex-cli 0.145.0** (`wire_api = "responses"`, `store=false`) sends
  `client_metadata`, a `custom` `apply_patch` tool with a Lark grammar, a
  `namespace` tool group, and `web_search`/`tool_search`.
- **Claude Code 2.1.269** sends `thinking:{type:"adaptive",display:"omitted"}`,
  `output_config`, `context_management` and mid-conversation `system`
  messages.

`client_metadata` is now validated and ignored unconditionally, like `user`.
Everything else applies only to requests that resolve agent compatibility on.
History that only agent-compat can produce returns 400 when replayed without
it, rather than being misparsed. This covers custom tool calls, namespaced
calls, `phase`, omitted-thinking signatures and `previous_response_id` chains
across modes.

**Responses**

- **`custom` tools** (format `text`, `grammar` with `lark` or `regex`).
  - Each is rendered to the chat template as a function with one required
    string property, `input`, and the grammar is appended to the description.
  - Calls come back as `custom_tool_call` items (`ctc_` ids).
  - Streaming emits `output_item.added`, then
    `response.custom_tool_call_input.delta`, then `.done`, then
    `output_item.done`.
  - `custom_tool_call`/`custom_tool_call_output` replay into the same shim.
  - `tool_choice:{type:"custom",name}` is supported.
- **Grammar enforcement.** `--custom-tool-grammar validate` lowers the grammar
  onto the `response_format` regex automaton, which has a 4,096-character cap:
  - `regex` syntax is used directly;
  - `lark` goes through a bounded non-recursive subset (`lark_regex.py`),
    which covers Codex's `apply_patch`.

  Each emitted `input` must full-match, otherwise the request fails with 502
  or `response.failed`. A recursive or unsupported grammar returns 400 at
  request time. `off` (the default) only describes the grammar, which matches
  SGLang and vLLM.
- **`namespace` tools** flatten to `ns.inner`. Emitted calls split back into
  `name` + `namespace`, and replayed calls re-qualify.
- **Hosted tools.** `web_search(_preview)`, `tool_search`, `image_generation`,
  `local_shell`, `file_search` and `code_interpreter` are dropped, never
  simulated, and listed in `mlx2.agent_compat.dropped_tools`. Any other
  unknown tool type still fails closed.
- **Message `phase`.** Accepted on input. On output, message items carry
  `commentary` when the response also calls tools and `final_answer`
  otherwise. They carry it on `output_item.done` and in the terminal
  response; `output_item.added` omits it.
- **Replay merging.** Replayed reasoning, assistant text and calls from one
  model turn merge into a single assistant message.
- **System placement.** Leading `developer`/`system` messages are joined, and
  later ones fold into the adjacent user turn, because common templates
  reject non-leading system messages.
- **Other replay fields.** `function_call_output`/`custom_tool_call_output`
  accept arrays of text parts. Reasoning items accept Codex's `content` field
  but replay only through the HMAC `encrypted_content`.

**Messages**

- **Adaptive thinking.** `thinking:{type:"adaptive"}` enables thinking with
  the effort-scaled budget, not history mode.
  - `display:"omitted"` streams no `thinking_delta`. Its `signature_delta` is
    a *carrying* signature: HMAC-authenticated for model and tenant, not
    encrypted.
  - Replaying an empty thinking block with that signature restores the
    reasoning. Forged or cross-scope signatures are dropped and counted.
- **`output_config.effort`** maps to `reasoning_effort`.
- **`context_management`:**
  - `clear_thinking_20251015` with `keep:"all"` is a no-op;
  - `keep:{type:"thinking_turns",value:n}` drops older turns' reasoning;
  - other edits, for example `clear_tool_uses_20250919`, return 400.
- Mid-conversation `system` messages fold into user turns.

**Evidence and counters**

- `scripts/agent_client_conformance.py --scripted` runs the installed
  `codex exec` and `claude -p` against the scripted CPU server:
  - Codex applies a real `apply_patch` custom call and replays reasoning,
    commentary and the call on turn two;
  - Claude Code completes a `Read` round trip with omitted-thinking restore;
  - all requests returned 200.
- `tests/test_agent_conformance.py` replays scrubbed captures of those
  sessions.
- The counters are the `agent_compat_*` keys in `/v1/status` `counts` and
  `mlx2_runtime_events_total{component="agent_compat"}`.
  `/v1/status.agent_compat` reports the settings.

**Known gaps**

These fail closed or are documented:

- `agent_message`, `compaction`, `tool_search_call/_output` input items and
  `POST /v1/responses/compact`;
- image parts in tool output on text routes;
- stored `input_items` views, which render custom calls in their function-shim
  form;
- decode-time grammar enforcement for custom tools. The same regex automaton
  supports prefix checks, but routing it through the adapter-owned constrained
  tool grammar needs GPU qualification;
- a `phase` on `output_item.added`;
- `output_config.format`;
- `context_management` edits other than `clear_thinking`.

## Verification and next live gate

`scripts/sdk_smoke.py` runs a CPU-only scripted server and re-executes its
client cases under a separately supplied interpreter. With OpenAI Python
2.43.0 and Anthropic Python 0.111.0, all 14 official-SDK cases pass: Chat
Completions and Responses non-stream/stream parsing, typed Responses stream
events and resources, Anthropic Messages content/stream helpers, tool and
thinking round trips, token counting, and typed 400/404 exceptions. The
script loads no weights and does not establish model or GPU qualification.

The original 50-test probability/API slice passed in
`test_logprobs_cpu.py`, `test_serving_contract.py` and
`test_qualification.py`, with an import guard proving no MLX import. The
Responses and strict-tool extensions have their own current CPU qualification
receipt; this historical count is not silently rewritten. The current work
ran no GPU tests.
They cover actual emitted-token identity, target-row selection across MTP
acceptance boundaries, bounded top alternatives, finite JSON, request validation,
text format normalization and HTTP/SSE ordering. The earlier API slice loaded
no model; current advanced-state qualification additionally uses tiny random
CPU models. No GPU test is claimed.

`scripts/qualify_serving.py` now includes a required `logprobs` check. It sends
the reference, probability and SSE requests below, verifies token counts,
actual emitted token fields and ranked alternatives, and records the semantics
receipt. Old qualification receipts without this check fail closed.

Root's new-identity live qualification sends ordinary and MTP requests
with `temperature:0`, `max_tokens:8`, `logprobs:true`, `top_logprobs:3`, and
`response_format:{"type":"text"}`. Check entry count equals completion tokens,
IDs are in vocabulary, probabilities are finite/nonpositive, each top list has
three descending entries, and text matches the equivalent request without
probabilities. Repeat once with SSE and collect each entry exactly once. For
sampled MTP, exercise accepted drafts and replacement/bonus rows and preserve the
route receipt; CPU row-selection tests do not substitute for native numerics.
