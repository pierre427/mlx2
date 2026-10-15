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

`logprobs: true` and `top_logprobs: 0..20` now return per-token
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
Sampled native MTP and external-draft routes report target verifier
probabilities **after** their temperature/top filters. At temperature zero,
they report processed log-softmax rather than the one-hot greedy verification
law. The machine-readable `request_controls.logprob_distribution_contract`
in the terminal route receipt declares these laws; it does not imply all
tokens used one producing route, since MTP may hand off to ordinary decode.
Accepted draft tokens use the corresponding target row; rejected-draft
replacement/bonus tokens use the target row at the accepted-prefix boundary,
never the draft or residual sampling distribution. These are existing unified
runtime semantics, not a promise of equal numerical values across differently
configured routes. CPU tests execute the actual production output-construction
expression for zero, one and two accepted draft tokens.

## Other controls

APCv2 persistent session mutations have a bounded filesystem wait. A manifest
or deletion deadline returns HTTP 503 with `apc_persistence_indeterminate`:
the filesystem operation may still finish, so this is not cancellation or a
successful park/delete. A completed deletion that leaves files behind returns
`apc_persistence_incomplete`. Both disable further operations on that APC
instance and require reload. `persistence.metadata_transaction` reports the
completion state, operation, worker ownership and reload requirement. Close
retains the directory lock until all filesystem workers actually drain; a
replacement owner then inventories the durable result. Session byte/file
accounting uses immutable payload receipts and performs no filesystem reads.
Startup inventory/security checks and ordinary-close lock release remain
outside this serving deadline contract.

- A recognised top-level Chat, Completions or Responses field (or Hermes
  `options` member) sent as JSON `null` is treated exactly as an absent field,
  as the OpenAI schema's nullable fields and openai-python's `None` keywords
  mean: `{"stop": null, "max_tokens": null}` gets the defaults, not a 400. An
  unsupported field is still a 400 whatever its value, `null` included.
  Required fields still fail (`messages`, `prompt`, `input`), and Anthropic
  `/v1/messages` still requires a non-null `max_tokens`. Chat and Completions
  validate `user` as text and ignore it, as Responses does.
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
  `/v1/status.reasoning_replay` exposes the v1 authentication, plaintext
  payload encoding, lack of confidentiality, and key-persistence contract.
  Treat these tokens as readable sensitive content despite the compatibility
  field name; the v1 format and existing replay behavior remain unchanged.
  A stateless `input += response.output` replay accepts the empty assistant
  message item of a turn that produced no answer text (it only reasoned, or
  stopped at once); empty user, system and developer text is still rejected.
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
- The pinned `sentence-transformers/all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` has an embedding-only serving
  adapter. It executes all six bidirectional BERT layers in float32, uses the
  artifact's uncased WordPiece tokenizer with right truncation at 256 tokens
  including `[CLS]`/`[SEP]`, averages attention-mask-selected contextual rows
  including special tokens, and L2-normalizes the resulting 384-dimensional
  vector. All execution-relevant artifact files are content-bound; the exact tensor schema
  and full weight hash are verified before loading. No download or remote code
  is used. `dimensions` may be omitted or exactly 384; dimension truncation is
  refused. Multi-input requests execute independent width-one forwards, with
  at most 2048 inputs and 1 MiB total UTF-8 text. Usage counts retained tokens,
  including special tokens. Status and operation receipts identify the trained
  recipe, artifact and observed execution. The route remains unqualified:
  CPU reference parity and the bounded public retrieval evaluation below are not
  broad retrieval-quality or GPU qualification.
  Generation, reranking, APCv2/KV caching, streaming and generation policies are
  unsupported on this route. Start it with `mlx2-serve --model /local/snapshot`
  and no generation route flag; use `/v1/embeddings`.
- The pinned `cross-encoder/ms-marco-MiniLM-L6-v2` revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a` serves `/v1/rerank` through
  a distinct relevance-trained BERT cross-encoder. It jointly encodes each
  query/document pair using uncased WordPiece, longest-first right truncation
  to 512 tokens including special tokens, the trained CLS dense/tanh pooler,
  and the trained scalar classifier. `relevance_score` is a raw logit with
  higher values indicating greater relevance; it is neither a calibrated
  probability nor constrained to 0–1. The artifact specifies identity activation.
  The exact metadata, tokenizer and full checkpoint bytes are verified; no
  download or remote code runs at serving startup. Up to 1000 documents and
  1 MiB total UTF-8 input are admitted, with independent width-one forwards.
  Usage sums retained tokens for every pair, including the repeated query.
  Embedding and generation routes return 501. Both trained representation
  routes return an `mlx2` recipe/artifact/use receipt and remain unqualified.
- A deterministic CPU diagnostic on 24 public SciFact test queries and 360
  candidate abstracts measured embedding nDCG@10 0.7674 versus TF-IDF 0.6901;
  recall@10 was 0.8333 versus 0.7500. MS MARCO reranking of the embedding top
  ten reduced nDCG@10 to 0.7450. This is a reduced-corpus diagnostic, with gold
  documents explicitly retained alongside lexical hard negatives and fixed
  background documents, not a full BEIR result. It supports neither broad
  quality qualification nor default reranking for scientific retrieval.
  See `provenance/minilm-retrieval-evaluation.json` and
  `scripts/evaluate_minilm_retrieval_cpu.py`; raw public dataset and evaluation
  rows stay outside the repository.
- Eligible decoder adapters retain their mean-pooled, L2-normalized input
  representation and cosine reranking. This is not represented as trained
  retrieval-model qualification. Running
  a decoder backbone could expose contextual hidden representations as a
  separate explicit adapter operation; it would still use generation-trained
  weights. Additional trained embedding models need their own checkpoint,
  tokenizer, pooling/projection and normalization contract. Task-trained
  reranking requires the appropriate query/document input and
  scoring head or documented scoring procedure. Neither is supplied by
  relabelling decoder mean pooling or cosine similarity.
- Dynamic LoRA supports explicit Linear/QuantizedLinear keys, exact tensor
  coverage, zero serving dropout, one active adapter, inflight drain, reversible
  unload, cache invalidation and revision receipts. It is disabled until
  `--lora-dir` is configured and remains GPU-unqualified.
- Responses image, video and audio parts cross an explicit multimodal adapter
  contract. Bounded data/File decoding, image checks, WAV metadata and capped
  video sampling are available. Optional mlx-vlm adapters now implement Gemma
  3n image/audio plus ordered native-video requests (uniform frames, explicit
  timestamps and media-bound APCv2 identity) and MiniCPM-o image/audio. The
  MiniCPM-o route honors artifact slicing and exact-rate audio chunks, with
  sequential vision as its default. Explicit operator vision batching and
  `minicpm_o_media.per_item_feature_reuse:true` are separate, incompatible
  opt-ins. Per-item reuse retains the source's width-one encoder calls and
  hashes exact processor pixel bytes, dtype, shape and target geometry under
  model/source/policy, authenticated tenant and LoRA identities. It is bounded
  in-memory reuse (256 items, 2 GiB); unsupported representations or inputs
  exceeding the 512 MiB hashing budget retain whole-request reuse. It is
  CPU-tested, default-off and unqualified; no disk persistence or performance
  claim is implied. Gemma 3n keeps its original grouping. All adapter vision
  feature cache keys include the trusted tenant scope independently of APCv2.
  The real downloaded processors are
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
  are capped at eight. `max_output_tokens` bounds every round together, not
  each round; a loop that spends it ends `incomplete`. Other hosted tools
  remain unavailable.
  Each configured server can set `timeout_seconds` (default 30),
  `max_response_bytes` (8 MiB), `max_sse_events` (1024), and `max_json_depth`
  (128). Time, bytes and event budgets span one complete operation, including
  session initialization/retry and discovery pagination; time also covers
  waiting for the session lock. The transport refuses redirects and interrupts
  active sockets on timeout or caller cancellation. HTTP disconnects cancel
  hosted execution for streaming and nonstreaming responses. Cancellation
  cannot undo an already-issued remote tool side effect. OS DNS lookups use
  at most four bounded background helpers; a cancelled or timed-out lookup
  cannot subsequently send a request, even if the OS resolver remains blocked.
  `/v1/status.hosted_tools` reports configured
  limits without credentials or URLs.

The generic output parser retains unfinished tool bodies in chunks and searches
only newly received text plus the marker overlap. Grammar-identified quoted
closers are not retried on every later token. Distinct ambiguous closers share
a 64 Mi-character cumulative grammar-input budget; exhaustion fails closed,
including under tolerant marker handling. This bounds pathological parsing
work without changing well-formed calls or terminal truncation semantics.

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
