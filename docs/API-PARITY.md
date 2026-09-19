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
- `tool_choice` accepts `auto`, `none`, `required`, and one named function.
  `parallel_tool_calls:false` is enforced. Strict function schemas use the same
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
  required call remains a post-generation contract failure.
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
  and must remain one token below the required `max_tokens` value.
  Assistant `text.citations:null` and `tool_use.caller:null` emitted by the
  official SDK round-trip safely; a direct caller is accepted, while actual
  citations and server-tool callers remain outside this client-tool subset and
  fail closed.
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
