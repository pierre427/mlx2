# Serving mlx2

This guide describes the modernized working source. Implemented behavior is not
qualified or deployed merely because it is present in the tree.

Aggregate operational telemetry is available as Prometheus text at
`GET /metrics`; detailed receipts remain on the JSON status endpoints. See
[Metrics and telemetry](METRICS.md) for names, units, label policy and the
CPU-only qualification boundary.

> **North-Mini-Code numbers measured before 2026-09-20 are void.** Until that
> date this port built North's `input_layernorm`s and its final `model.norm` as
> `nn.LayerNorm(eps=1e-5)` where the reference is `RMSNorm(eps=1e-6)`. The
> corrected body has a different perplexity (8.6716 → 8.5880) and different
> greedy output, so every North quality, grading, acceptance and qualification
> record taken before 2026-09-20 measured a model mlx2 no longer serves —
> including this document's guarded-sanity and alpha-steering figures. Mechanism
> counters and cache geometry are unaffected. The list of void records is
> `qualification/runs/north-requal-20260920/INVENTORY.md`; the correction is
> `docs/ports/NORTH-MINI-CODE.md`.

## Runtime and artifact

The Flash-Next artifact is the local unified-layout
`Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint, model type `qwen4_exp`. Its release
name remains in the directory and model ID; this is the Qwen4 Flash-Next family
used by this project. The similarly named uncensored mlx-serve artifact has a
different layout.

The loader supports mixed quantization and the embedded MTP head, checks the
PLE manifest and loads expected weights strictly. PLE embedding rows use the
artifact's `ple_rows.bin`. It reads local files without remote model code;
weights are not part of this repository.

Qualification binds tokenizer/configuration/PLE-manifest contents plus weight
file names, sizes and modification timestamps. This is a **local artifact
identity**, not a digest of every weight byte. Editing weights while preserving
size and timestamps is outside that identity guarantee.

The tested MLX build comes from local revision
`2a817ad94fd574810f0e69b66d698a35ed2d93e9`, installed as
`0.32.2.dev20260915+2a817ad94`. mlx2 does not import that source checkout.
`requirements-qualified.txt` pins the other Python packages. To reproduce:

1. Create a Python 3.12 environment.
2. Build that exact MLX revision or use its retained wheel. The development
   build is not assumed to be available on PyPI.
3. Install `requirements-qualified.txt`, then `pip install --no-deps -e .`.
4. Acquire the same artifact independently and qualify the new environment.

Runtime receipts include source/native binary hashes and Python, macOS and
dependency versions. Qualification on another host produces a new local receipt.

## Default route selection

When no execution-route flag is present, metadata-only adapter resolution picks
the adapter's default before any weights are loaded. Complete embedded-head
Qwen4 Flash-Next, Qwen3.8 and Xing artifacts default to `native_mtp`;
the same adapter families fall back to `ordinary` when the resolved artifact
has no MTP capability. Muse Glimmer, North Mini Code, Laguna XS 2.1, Gemma 3n
and MiniCPM-o default to `ordinary`. Qwen3.6-35B-A3B also defaults to `ordinary`
even with an MTP head: measured on GPU, native MTP matched ordinary single-stream
(102.0 vs 102.7 tok/s) and lost 25-34% batched; pass `--native-mtp` to opt in.

The mutually exclusive `--native-mtp`, `--ordinary`, `--external-draft` and
`--prompt-lookup` flags always override an adapter default. `--native-mtp`
fails before listener binding or weight loading when the resolved artifact has
no implemented MTP route; external draft still requires an explicit flag and a
matching draft policy. Startup logs the resolved route and whether its source
was `adapter_default` or `explicit_flag`.

`/v1/status.settings.route` and `route_selection_source` expose the decision,
and terminal response receipts repeat both fields. Qualification identity binds
the resolved `route`; selection provenance is observational, so a record made
with explicit `--ordinary` also matches the same adapter-default ordinary route.

## Request API and Hermes controls

`POST /v1/chat/completions` accepts text system/user/assistant/tool messages.
Function calls use OpenAI-shaped tool definitions and JSON-string arguments;
`tool_choice` supports `auto`, `none`, `required`, and a named function.
`parallel_tool_calls:false` and the bounded strict JSON-schema subset are
checked before a successful response is returned. Clients execute tools and
return their results. A valid request whose generated calls violate that
contract fails with HTTP 502; malformed client contracts fail with 400. The
one contract the parser can satisfy on the model's behalf is
`parallel_tool_calls:false`: a surplus call is dropped at the parser and
counted (`tool_calls{reason="parallel_bound_truncated"}`), so the terminal
check never sees a set it would have to fail, and over-calling -- a model
behaviour -- never reaches the client as a 5xx or a mid-stream error. Strict,
required, named and single-call Chat streams are buffered through terminal
validation before the first response byte. `description` is optional in both
accepted tool schemas, so a definition that omits it is carried as the empty
string rather than reaching the chat template as a Jinja `Undefined`; a chat
template that fails to render anyway answers 400 naming the omitted field when
the failure identifies one, and otherwise 500 saying template rendering
failed -- never an unhandled `TypeError`. On an open stream (a hosted-tool
continuation renders a second prompt) the same failure arrives as the surface's
SSE error event. Reasoning appears in
`reasoning_content` separately from answer text.
Muse additionally enforces its trained recipient header when thinking is off:
`none`/no-tools keeps the direct-user header, named choice opens that function,
`required` permits only declared function recipients, and `auto` permits user
or declared functions but not the `self` reasoning recipient. This adapter-owned
history-pure constraint applies on ordinary, prompt-lookup and external-draft
routes independently of the optional full ATEM decode grammar; terminal tool
validation remains authoritative.
North required/named choices likewise apply a request-scoped, history-pure
constraint to the trained `<|START_ACTION|>` branch. With thinking enabled it
passes reasoning through unchanged and activates only after
`<|END_THINKING|>`; `auto` and `none` are unchanged. The processor constrains
the branch marker, while main's terminal contract remains authoritative for a
named function and its arguments. A partial action cut off by `max_tokens` is
dropped as an incomplete call and returns `finish_reason:"length"` (Anthropic
`stop_reason:"max_tokens"`), retaining any reasoning or content parsed before
the action. EOS/stop with the same malformed block still fails closed.
`POST /v1/completions` accepts a text prompt.

Main's post-generation validation is the default tool contract. Two
output-changing parser mechanisms are explicit execution-policy opt-ins:
`constrained_tool_grammar:true` applies an adapter-owned decode grammar where
the adapter and request combination can express it, and
`tolerant_tool_markers:true` returns malformed unconstrained markers as text.
Both default false and are qualification-bound. Decode grammar falls back to
main validation, with a counter and receipt, when the adapter has no hook or a
request combines required/named tools with another constraint or `min_tokens`.

`POST /v1/responses` provides a bounded text/function subset over the same
request lifecycle. It accepts text input messages,
`instructions`, sampling/output controls, metadata, strict text formats and
function tools, and returns OpenAI-shaped message/function-call output items,
usage and the mlx2 route receipt. Text streaming uses typed Responses events
with monotonic `sequence_number` values; requested logprobs ride
`response.output_text.delta`, and requested signed reasoning plus accumulated
logprobs are present in the terminal response exactly as in non-streaming mode.
Function calls are buffered until
terminal contract validation, then emitted as typed argument delta/done and
output-item events. Responses default to `store:true` in a bounded,
tenant-scoped store and support retrieve, delete and `previous_response_id`
continuation. Stored requests are also available through the cursor-shaped
`GET /v1/responses/{response_id}/input_items` resource. This is a reduced view
derived from the existing stored continuation context, not another stored copy
of the raw request: inline file base64 and input reasoning signatures are never
added to the store. Resolved file text, messages, client function calls/results
and reasoning summaries are reconstructed; an explicitly included reasoning
signature is regenerated at read time. Non-derivable records return 409 and
unknown `include` values return 400. Configure
`--api-state-dir` for atomic persistence across restart;
without it state remains process-local. The prior request's top-level
`instructions` are deliberately not carried. Stored reasoning remains absent
for ordinary requests and is retained only when
`include:["reasoning.encrypted_content"]` explicitly requests it.

`POST /v1/messages` and `/v1/messages/count_tokens` expose the bounded
Anthropic Messages text/tool contract over the same lifecycle. System text
blocks and prompt-cache markers are validated; client tools support
`auto`/`any`/named choices, tool-use/result continuation, and parallel-call
control. Signed thinking blocks can be returned unchanged on a later turn.
Following Anthropic's opt-in contract, omitting `thinking` is translated to an
explicit `enable_thinking:false`, as is `thinking:{"type":"disabled"}`;
`thinking:{"type":"enabled","budget_tokens":N}` enables thinking and the
history-pure budget mode. The omitted and disabled forms therefore share the
same rendered-prompt cache identity, while enabled thinking remains distinct.
The stream is Anthropic-typed and compatible with the official SDK's
`messages.stream(...)` final-message helper. This remains a client-tool subset:
real citation payloads and server-tool callers fail closed.

`POST /v1/files` accepts bounded multipart uploads for `batch` or `user_data`;
GET listing (`limit`, `after`, `purpose`), metadata/content and DELETE are
tenant-scoped. Responses `input_file`
accepts uploaded or inline base64 UTF-8 text, JSON, JSONL and Markdown and
renders a named file boundary into the prompt. Image, video and audio parts are
dispatched only to an adapter with a multimodal hook. The optional Gemma 3n and
MiniCPM-o mlx-vlm adapters perform an isolated encoder prefill, then rejoin
ordinary continuous decode. APCv2 namespaces include the ordered media and
processor policy fingerprint; text-only adapters return 501.

Processor-only evidence does not authorize a production media route. Gemma 3n
requires receipt checks for image, video, input audio, bounded encoder batching
and media APCv2 reuse. MiniCPM-o requires image, input audio, bounded encoder
batching and media APCv2 reuse. `load_qualified_route` rejects a generic text
receipt that lacks any adapter-declared check.

The retained candidate GPU smoke receipt at
`qualification/runs/multimodal-gpu-smoke-20260918/receipt.json` proves finite
Gemma video/audio and MiniCPM-o image/audio execution, plus an exact repeated
Gemma video request that reused 574 of 575 prompt tokens through APCv2. It does
not satisfy the broader production qualification checks above.

`POST /v1/audio/speech` accepts the OpenAI `input`, named/custom `voice`,
optional `instructions`, `response_format`, `speed` and binary `stream_format`
contract. SSE is recognized but remains capability-gated. Binary output remains
unavailable unless the selected, qualified adapter declares `output_audio` and
returns an `AudioOutput` with the exact requested MIME type. MiniCPM-o is
currently input-audio capable. Although mlx-vlm 0.7.1 has a MiniCPM-o TTS path,
it requires a reference WAV and generates a spoken chat response; it is not an
OpenAI-compatible named-voice renderer for the exact `input`. The route stays
fail-closed until that semantic and qualification gap is closed.

With `--tool-backend-config PATH`, Responses `mcp` tools may use an
operator-allowlisted Streamable HTTP server. Request URLs must exactly match the
configured label, discovery respects `allowed_tools`, only
`require_approval:"never"` is accepted, and at most eight model/tool rounds run.
Without this configuration MCP and hosted tools fail closed.

`POST /v1/batches`, GET `/v1/batches/{id}`, GET `/v1/batches`, and POST
`/v1/batches/{id}/cancel` implement a bounded asynchronous JSONL lifecycle for
chat, completions, Responses and embeddings. Rows require the standard
`custom_id`, `method`, `url`, and `body` fields; result and error JSONL are
published through the Files API. The local completion window is the standard
`24h`. With `--api-state-dir`, terminal state survives restart and interrupted
active work is marked failed. Remote pricing semantics are not claimed.

`POST /v1/embeddings` and `POST /v1/rerank` execute mean-pooled normalized
decoder input embeddings and cosine scores. This is an explicit generative-model
representation, not retrieval-model qualification. `/v1/load_lora_adapter` and
`/v1/unload_lora_adapter` follow vLLM's control shape. With `--lora-dir`, mlx2
drains inflight work, loads an explicit-key zero-dropout LoRA with exact
shape/coverage checks, invalidates APCv2 and host prompt state, advances the
model revision, and restores original modules on unload. One adapter may be
active at a time.

**Concurrent multi-LoRA** (default off; qualification-only until
GPU-qualified). `--max-loras N --max-lora-rank R` (requires `--lora-dir`)
serves up to N adapters plus the base model in one mixed physical batch; a
request selects an adapter with `"model": "<lora_name>"` (vLLM shape), and
`/v1/models` lists registered adapters with `parent` set to the base. In this
mode `/v1/load_lora_adapter` registers without draining (it drains once only
when the adapter adds module keys that are not yet wrapped) and
`/v1/unload_lora_adapter` refuses an adapter pinned by live requests.
Each wrapped Linear holds stacked slot tensors; the per-row delta is
`gather_mm(gather_mm(x, A, ids), B, ids)` with slot 0 = base, and an all-base
batch skips the delta (base rows stay bit-identical). Admission pins a
resident slot (LRU eviction of unpinned adapters); when every slot is pinned
the request is deferred like memory admission, and a declared atomic cohort
fails whole. The adapter content fingerprint joins the APCv2 namespace, so a
prefix is never reused across adapters (or between an adapter and the base).
Receipts carry `lora` (`mlx2.multi-lora.v1`: name, fingerprint, slot,
residency hit/loaded/evicted, apc_namespace); `status.multi_lora` and
`mlx2_multi_lora_events_total` count forwards (base-only/mixed/single),
rows, slot hits/loads/evictions/deferrals and delta applications. v1 is
ordinary-route only: MTP, prompt lookup, external draft, int8 prefill,
approximate KV, live Spomin surgery and cache capsules are refused at startup.

| Control | Behavior |
|---|---|
| `temperature`, `top_p`, `top_k`, `min_p`, `seed` | Per-request sampling; unset fields take the model's vendor defaults (see *Vendor sampling defaults*); seed is an unsigned 32-bit integer; a positive temperature must keep `1/temperature` finite in float32; `top_k` must be below the tokenizer vocabulary; `min_tokens` cannot be combined with structured output; the `top_p` nucleus is computed in float32 on every device (bfloat16 cumulative sums over a 248K vocabulary collapsed it on the CPU backend) |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `logit_bias` | Applied to the request's target distribution on every route, including MTP, prompt-lookup and external-draft verification (acceptance uses the same penalized distribution, so speculation stays exact). `repetition_penalty` (vLLM extension field) is multiplicative and sign-aware over prompt and generated tokens (HF/vLLM semantics); `presence_penalty` and `frequency_penalty` are additive over generated tokens only (OpenAI/vLLM semantics) |
| `sampling_profile` | mlx2 extension: select one of the model's vendor sampling profiles by name (e.g. `"coding"`); an unknown name is rejected (400). Also accepted by `/v1/responses` |
| `max_tokens` / `max_completion_tokens` | Explicit output cap, 1–2,097,152; prompt plus output must fit both request and served context limits, and an explicit overflow fails rather than clamping. When omitted, admission uses `min(--default-max-tokens, effective_context - prompt_tokens)` with at least one output token; `--default-max-tokens` defaults to 65,536. Receipts expose effective `max_tokens` and `max_tokens_defaulted` |
| `min_tokens` | Minimum output, 0–2,097,152, bounded again by the effective `max_tokens`; unavailable with structured output |
| `thinking_budget` | Explicit reasoning budget, independently bounded from 0 through 2,097,152 on OpenAI-shaped and Hermes/Ollama requests. This does not change the effort-scaled default budget, which remains capped at 8,192. Anthropic's separate `thinking.budget_tokens` rule is described below |
| `stop` | Up to four stop strings |
| `stream` | Chat/completions SSE or typed Responses text/function events, with cancellation when the consumer disconnects or stops draining; strict/required/named/single-tool Chat calls are terminal-buffered and validated before headers |
| `enable_thinking`, `think`, `chat_template_kwargs.enable_thinking` | Boolean aliases; conflicting toggles are rejected |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra`; adapter-specific meaning recorded in the receipt |
| `options` | Supports `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `num_predict` and `num_ctx` |
| `options.num_predict` / `options.num_ctx` | Normalize to output cap / request context ceiling; conflicting duplicate controls are rejected |
| `logprobs`, `top_logprobs` | Float32-normalized emitted-token probability and up to 11 sorted alternatives, nonstreaming or SSE; normalization precision is route-independent |
| `response_format: {"type":"text"}` | Accepted as ordinary text generation |
| `response_format: {"type":"json_object"}` | Recursive JSON constraint on a grammar-qualified route |
| strict `json_schema` / `grammar` | Bounded canonical schema subset or regex; unsupported keywords, invalid continuations and match-budget overruns fail closed. Two engines enforce the constraint (see *Structured output engines* below); the receipt's `structured_output.engine` names the one used. A grammar with no admissible continuation fails that request closed (502 nonstreaming; SSE `error` then `[DONE]` on a stream) without affecting other lanes. With thinking enabled the grammar is deferred past the adapter's thinking-close marker (see *Structured output with thinking*) |
| `n` | 1 to 8 nonstreaming samples; the cohort must fit lane, inflight and physical-headroom gates before any job is published |

Responses `max_output_tokens` and Anthropic Messages `max_tokens` enter the
same admission path. Responses omission uses the prompt-aware
`--default-max-tokens` value (65,536 by default) above. Anthropic Messages
continues to require `max_tokens`; its
`thinking.budget_tokens` maps to explicit `thinking_budget` and must be less
than that required cap. The server `--max-context` flag and its 262,144-token
default are unchanged. Operators can set the omission default with
`--default-max-tokens N` (1–2,097,152; default 65,536). The configured value is
qualification-bound and appears in `/v1/status.settings.default_max_tokens` and
each request receipt. Lowering it restores admission width for clients that do
not send an explicit output cap; explicit caps remain unaffected.

All output parsers retain the exact client stop string they match in the route
receipt. OpenAI Chat keeps `finish_reason:"stop"`; Anthropic Messages maps the
same receipt to `stop_reason:"stop_sequence"` plus `stop_sequence`.

### Output-default admission projection

The following CPU-only calculation uses the production
`FlashNextCacheBudget.project` geometry recorded for the Flash-Next artifact
(13 QSA planes, 36 recurrent planes) and
`SelfMTPLaneAdmissionController.lane_gib` at native MTP depth 2. On the
controller's 128 GiB operating point, the 72.5 GiB resident model leaves
55.5 GiB free; the 20 GiB service/driver reserve that the host-scaled rule
yields on a 128 GiB host leaves 35.5 GiB
usable. Figures are cold-lane projections and include the 1.76 GiB depth-2
verify transient; the controller's 16-lane saturation cap remains active.

| Prompt tokens | `--default-max-tokens 512` reserved context | Old lane GiB | Old admitted lanes | Default 65,536 effective output | New reserved context | New lane GiB | New admitted lanes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,560 | 2.141 | 16 | 65,536 | 67,584 | 5.870 | 6 |
| 16,384 | 16,896 | 2.963 | 11 | 65,536 | 81,920 | 6.693 | 5 |
| 65,536 | 66,048 | 5.782 | 6 | 65,536 | 131,072 | 9.512 | 3 |

Admission therefore reserves the full 64K default up front. That materially
reduces projected width for clients that omit an output cap: 16 to 6 lanes for
a 2K prompt, 11 to 5 at 16K, and 6 to 3 at 64K in this Flash-Next projection.
This lane does not change the admission algorithm; clients that need width
should send a realistic explicit cap.

### Vendor sampling defaults

Each model adapter declares the sampling settings its vendor recommends
(`sampling_defaults`, built from `mlx2.sampling_defaults.SamplingDefaults` /
`VendorSampling`), cited from the artifact's `generation_config.json` and the
vendor model card. A default fills a field only when the request leaves it
unset: an explicit value always wins, including `temperature: 0` and an
explicit neutral penalty. Fields neither the request nor the profile sets take
the neutral value (top_p 1, top_k and min_p off, penalties off), as vLLM does.
An adapter with no declaration keeps the historical engine fallback
(temperature 0.7, top_p 0.8, top_k 20).

Profile selection: `sampling_profile` from the request if given; otherwise,
for chat requests, the profile for the thinking mode the adapter resolves
(thinking on or off) when the vendor gives mode-specific values; otherwise the
vendor's general profile. Raw completions have no known mode and use the
general profile. Every receipt records `request_controls.sampling` (the
explicit request fields), `effective_sampling` (what the samplers used) and
`sampling_defaults` (selected profile and reason, applied defaults with their
source per field, and the neutral/legacy fallbacks). `/v1/status` publishes
the declared profiles under `sampling_defaults`, plus any drift between them
and the served artifact's `generation_config.json` (reported, never applied).

| Adapter (model type) | Profile | Values | Source |
|---|---|---|---|
| Flash-Next (`qwen4_exp`) | `thinking` (general; thinking on) | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0 | `generation_config.json` (temperature/top_p/top_k) + Qwen/Qwen3.8-Flash-Next model card |
| | `instruct` (thinking off) | temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5, repetition 1.0 | Qwen/Qwen3.8-Flash-Next model card |
| Qwen3.8-27B (`qwen3_5`) | `thinking` / `instruct` | as Flash-Next | `generation_config.json` + Qwen/Qwen3.8-27B model card |
| Qwen3.6-35B-A3B (`qwen3_5_moe`) | `thinking` (general; thinking on) | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 1.5, repetition 1.0 | `generation_config.json` + Qwen/Qwen3.6-35B-A3B model card |
| | `coding` (request only) | temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0 | model card: thinking mode, precise coding |
| | `instruct` (thinking off) | temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence 1.5, repetition 1.0 | model card |
| Muse Glimmer (`muse_glimmer`) | `general` | temperature 1.0, top_p 0.95, top_k 64 | Muse-Glimmer-30B model card (its `generation_config.json` has only `do_sample: false`) |
| North Mini Code (`cohere2_moe`) | `general` | temperature 1.0, top_p 0.95 | CohereLabs/North-Mini-Code-1.0 model card (no sampled fields in `generation_config.json`) |
| Laguna XS 2.1 (`laguna`) | `general` | temperature 1.0, top_p 1.0, top_k 20, min_p 0 | `generation_config.json` (do_sample true); top_k from the poolside model card best practices |
| Xing4.0 (`xing4_0`) | `general` (both thinking modes) | temperature 1.0, top_p 0.95, repetition 1.05 | `generation_config.json` (do_sample true); model card: complex reasoning / general |
| | `coding` / `agent` (request only) | temperature 0.8, top_p 0.95, repetition 1.05 | model card: coding / agent tasks |

The `qwen3_5` adapter also loads Qwen3.6-27B artifacts; they receive the
Qwen3.8-27B declaration (the Qwen3.6 card differs only in thinking-mode
presence_penalty 1.5). Penalty defaults turn on the per-lane logits-processor
path, which costs some decode throughput relative to penalty-free requests;
send explicit neutral penalties to opt out. The serving qualifier pins
`temperature: 0` and neutral penalties on every greedy check and verifies
vendor-default recording in its `sampling_defaults` check.

For Qwen, a positive effort label enables thinking; it does not select a
calibrated token or compute budget. Explicit thinking toggles take precedence.
Muse has model-specific effort handling, documented in its port report. Unknown
`options` and unsupported template keys are rejected rather than silently ignored.

### Structured output engines

State: **implemented**, CPU-tested against the scanner token-for-token
(including the real Qwen3.6 248K vocabulary); **GPU-unverified** and not yet
**qualified** on any served route. The qualifier records the engine but does
not make it a pass condition.

- `automaton` (default when the pattern compiles): `structured_automaton.py`
  compiles the constraint once into a deterministic automaton over code point
  classes (Thompson NFA, subset construction, minimization; hard caps of 20,000
  DFA states and 200,000 NFA states, past which it refuses). The recursive
  `json_object` grammar compiles to a deterministic pushdown automaton, so
  nesting depth is unbounded. The admissible-token mask of an automaton state
  is one vectorized walk of the automaton down a trie of the vocabulary,
  memoized per (automaton, state, inspected stack frames) in a byte-bounded LRU
  (64 MiB of packed bits per vocabulary). The processor advances the state by
  the text each new token adds; it never re-walks the output. The mask is exact
  and complete for every sampler, so the receipt reports
  `tail_mass_bound: 0.0`, `parallel_scans: 0`, and the scanner pool is not
  used. EOS is admissible iff the state is accepting. Measured on the Qwen3.6
  vocabulary (CPU, M-series): compile 3-7 ms, first visit of a state 1-4 ms,
  repeat visits ~20 us. The server freezes the adapter's generation stop-token
  ids once at load and gives that same tuple to batch stop matching,
  `min_tokens`, and every structured processor. Once a constraint is active, a
  stop token is admitted only from an accepting state; before a deferred
  constraint activates, normal generation-stop behavior is unchanged. The
  processor recognizes the decode loop's one unused post-terminal logit
  evaluation as control framing rather than feeding the terminal token back
  through the grammar.
- `scanner` (fallback): the `regex` partial-match scanner. Pieces are examined
  in descending logit order against a canonical (state-equivalent,
  depth-bounded) prefix; greedy decoding stops at the first admissible token
  (exact argmax), and sampling finishes the vocabulary tail exactly on a pool
  of scanner processes (`MLX2_STRUCTURED_WORKERS`, default 8; 0 disables) when
  that fits the 2 s per-token budget. If it does not, the request continues
  with a recorded bound: `structured_output.tail_mass_bound` is the largest
  unexamined-to-admitted mass ratio (0.0 means every mask was exact) and
  `parallel_scans` counts exact pool finishes.

Engine selection is fail-safe, never fail-open: a pattern outside the
automaton's exactly supported subset (lookaround, backreferences, anchors,
atomic groups and possessive quantifiers, inline flags, conditionals,
numbered recursion, fuzzy matching, nested/POSIX sets, non-LL(1) recursion, or
anything past the state caps) stays on the scanner. Everything
`response_format` itself generates is inside the subset.
`MLX2_STRUCTURED_AUTOMATON=0` forces the scanner for every request.

Byte-fallback and partial-UTF-8 vocabulary pieces (any piece whose own decode
contains U+FFFD), empty pieces and pieces beyond the tokenizer vocabulary are
never admissible under either engine; special tokens are matched as their
literal decoded text. A character that the tokenizer can only spell across
several byte tokens therefore cannot be generated under a constraint.

Byte-fallback pieces: on a byte-level BPE vocabulary (Qwen: 944 ids) some
tokens are only part of a UTF-8 character, so their isolated decode is U+FFFD
and a text trie cannot admit them. The automaton engine recovers each token's
exact bytes, carries the incomplete UTF-8 tail beside its configuration, and
admits such a piece when its bytes complete or extend a character the grammar
accepts from that point (overlongs and surrogates excluded; EOS is never
admissible inside a character). Characters reachable only through these pieces
(much of the astral planes, e.g. U+10FFFF or U+1F732) are therefore valid in
JSON strings and raw grammars. A schema `const`/`enum` is ASCII-escaped by the
compiler and never needed them. The byte view is verified against the decoded
pieces when it is built; a vocabulary without one (and the scanner fallback
engine) keeps the old rule that these pieces are inadmissible.

The automaton engine fails a lane closed when recursive nesting passes 256
levels, and keeps rollback configurations for the trailing 256 positions only
(older positions are re-derived on demand).

### Thinking guard (run-on reasoning release)

A reasoning model can circle a formed answer without ever emitting its
thinking-close token. The guard is the lab's state-aware release
(termination-attractor work: *tau* alarm + ramped actuator) combined with a
JUICE-style effort budget:

- **budget** — `thinking_budget` on a request, or `--thinking-budget N` as the
  server's medium-effort anchor scaled by `reasoning_effort` (minimal 0.125x,
  low 0.375x, medium 1x, high 4x, capped at 8192; unspecified effort is
  `high`). Release starts at 80% of the budget; the channel is closed outright
  only at the budget itself.
- **run-on alarm** — a CUSUM over recurring 6-grams of the reasoning tokens
  (content-blind), which releases a loop long before any budget.
- **release** — a bias on the adapter-declared thinking-close token, ramped
  +2 nats per step, so the model winds down over a few tokens instead of being
  cut mid-thought.

Defaults are adapter-owned: a model adapter may ship
`thinking_guard_defaults()`, applied when the operator sets no flag. **North-
Mini-Code ships budget 512 and alpha 0.2, so both are on by default for that
model**; every other model defaults to off. An operator flag always wins and an
explicit `0` disables a lever (`--thinking-steer-alpha 0`, `--thinking-budget
0`); a request can do the same with `thinking_steer_alpha: 0` /
`thinking_budget: 0`. `settings.thinking_defaults_source` reports `adapter`,
`operator` or `none`.

It is a pure function of the generated ids (safe under speculative verify rows
and rollback), acts only while the reasoning channel is open, composes with
grammar deferral, and needs a single-token close marker (North, Qwen). Receipt:
`mlx2.request_controls.thinking_guard`. State: implemented, on by default for
North-Mini-Code only. Its GPU verification
(`qualification/runs/north-thinking-guard-20260918/`, 399/400 at −18% tokens)
**was measured before the 2026-09-20 normalization fix and is void**; the
mechanism is unaffected, the numbers are not.

`thinking_budget_mode:"history"` is the explicit opt-in force-close mode used
by Anthropic thinking translation. It derives every mask from committed token
history, scans only the generated suffix and completes multi-token close
markers exactly across MTP draft/verify/rollback. The default remains
`"state_aware"`; history mode replaces only the budget mask. The run-on guard,
adapter defaults and alpha steering remain active and are listed under
`mlx2.request_controls.thinking_mechanisms`.

**Alpha steering** is the activation-space actuator from the same research.
While a lane is reasoning, `alpha * rms_L * v_hat` is added to the residual
stream after decoder layer `L`, where `v_hat` is a per-model *commit direction*
calibrated from naturally closed traces (mean residual over the last 24
in-think tokens minus the mean over the early reflection window; positions
only, content-blind). It needs a model that exposes `residual_taps` and an
adapter that owns a calibration asset — today North-Mini-Code
(`adapters/assets/north_mini_code_commit_direction.npz`, layer 32 since the
2026-09-20 recalibration; see below). Enable with
`--thinking-steer-alpha A` (request `thinking_steer_alpha`, 0–1) and optionally
`--thinking-steer-hammer H` for a stronger pull after the alarm trips. It
batches (one forward, a zero row for unsteered lanes) and also steers
prompt-lookup and external-draft verify forwards (one vector per lane over its
verify block, skipped for a block that already carries the close token;
counters `external_verify_steer_rounds` / `external_verify_steered_lanes`).
The verified target law is then the steered law, so speculation stays exact.
Native MTP keeps the logit-level guard only. Steered lanes are not stored as exact APCv2 prefixes
(their K/V was written under steering); the prompt boundary still is.
**A direction is bound to the artifact it was measured on.** Each stored
direction names an `artifact_identity` (config, tokenizer, chat template and
sampled weight bytes; host-independent) and is only ever loaded for that
identity, because a direction from another build is a same-norm wrong vector,
and the 2026-09-20 grid measured such a vector as leaving reasoning essentially
as long as no steering at all (7,169 tokens against 7,683 unsteered, versus
2,404 for the calibrated one). The void 2026-09-18 campaign measured a random
direction as *worse* than no steering; that stronger claim is not reproduced on
the corrected body and should not be repeated. When steering is
wanted and nothing is bound — North's 8-bit build, a re-quant, a fine-tune —
the server calibrates at startup (about two minutes: 24 natural traces, a
positional direction per layer, then held-out validation against no steering
and a random control) and stores the result under
`<cache-dir>/commit-directions/<identity>.npz`. The gates are: at least 10
usable closed traces, cross-trace consistency ≥ 0.30 at the chosen layer, no
lost accuracy, no fewer closes, shorter reasoning, and a random direction that
does not do as well. If any gate fails nothing is written and steering stays
off; if the operator passed `--thinking-steer-alpha` explicitly the server
refuses to start instead (fail closed), as it does for a model with no residual
taps. `--no-thinking-auto-calibration` skips the attempt;
`settings.thinking_steer.calibration` reports the identity, the state
(`calibrated` / `uncalibrated` / `unsupported`), the source and any gate
failures. Recalibrate with `scripts/calibrate_thinking_direction.py` whenever
the weights, quantization or chat template change. The adapter file carries the
operating guidance.

The 2026-09-18 calibration campaign
(`qualification/runs/north-alpha-calibration-20260918/`) measured the direction
in the **pre-fix LayerNorm residual geometry**; both it and the direction it
produced are void. `thinking_calibration.SCHEMA` was bumped to
`mlx2.commit-direction.v2` so the stale asset failed closed twice — on the
schema and on the artifact identity, which hashes the schema.

The shipped asset was **recalibrated on the corrected body on 2026-09-20**
(`qualification/runs/north-requal-20260920/`). Cross-trace consistency is higher
at every depth than before the fix and the peak moved from L28 to **L32**
(+0.518 against +0.511 at L28), so `COMMIT_DIRECTION_LAYER` is now 32. Its
gates: 21 usable closed traces of 24, and on the held-out set 8/8 correct at 685
reasoning tokens against 7/8 at 2,221 unsteered and 7/8 at 2,246 for a same-norm
random direction — the random control does not reproduce the effect. The
server's own startup `auto_calibrate` and the standalone
`scripts/calibrate_thinking_direction.py` produced bit-identical arrays.

### Structured output with thinking

State: **implemented**, CPU-tested; **GPU-unverified**, not yet **qualified**
(`structured_output_thinking` in `scripts/qualify_serving.py` is the live
check).

A chat request may combine `response_format`/`grammar` with thinking when the
adapter declares its thinking-close marker (`thinking_close_token_ids()`;
`/v1/status` reports it as `structured_output.thinking_deferral`). The generic
`<think>`/`</think>` adapters declare the marker only when it is a single
atomic token. Adapters that declare none keep the 400 `structured output
requires thinking to be disabled`.

- While the generated ids do not contain the marker, logits pass through
  untouched: reasoning is unconstrained, and so are generation stop tokens.
- Constraining starts at the token after the marker's first occurrence, with an
  empty constrained prefix. For `json_object` and `json_schema` the deferred
  grammar also admits JSON whitespace before the root value (Qwen emits a blank
  line after `</think>`), so `content` is the JSON document, possibly preceded
  by whitespace. A raw `grammar` regex is the client's exact language and is not
  widened.
- If generation hits `max_tokens` before the marker, the response is an ordinary
  `finish_reason: "length"` with `reasoning_content` only and empty `content`,
  not a 502. If the model emits a generation stop token while still deferred,
  that is allowed (no grammar is active) and `content` is empty.
- The receipt carries `structured_output.deferred` and
  `structured_output.deferred_tokens` (tokens generated before constraining
  began, marker included; every completion token when the marker never came).
- Under `--qualification-mode`, a no-continuation 502 also carries the bounded
  `mlx2.structured_output_failure` diagnostic and writes the same diagnostic to
  the server log. It contains the constraint/engine, recent token pieces and
  bytes, automaton state, and top raw-logit tokens. Normal serving omits it.
- `min_tokens` with structured output stays rejected, and raw
  `/v1/completions` never opens a reasoning channel, so its behaviour is
  unchanged.

Example Hermes-compatible request:

```json
{
  "messages": [{"role": "user", "content": "Explain prefix caching."}],
  "options": {"num_ctx": 16384, "num_predict": 128, "temperature": 0.7},
  "think": true,
  "reasoning_effort": "high"
}
```

Requests are limited to at most eight nonstreaming choices, a context-derived
bounded input envelope (2 to 32 MiB), and 2,097,152 requested generated tokens
per sample. The served/request context limit and memory admission remain
independent, tighter bounds.
Structured output is installed only when the selected route declares and has
qualified `grammar`; current-source qualification includes an actual strict
JSON-schema request. Strict and forced named function tools are enforced for
nonstreaming requests. Multimodal payloads remain unsupported. See
[API parity](API-PARITY.md).

Token probabilities describe the actual emitted token. Ordinary decoding
reports normalized post-processor logits before sampler transforms. Native MTP
and external draft verification report the transformed **target** law, including
accepted proposals and correction/bonus tokens; they do not report the draft
or residual law. Probability serialization adds device-to-host work. Request
receipts identify the route's semantics.

The default listener is loopback and has no authentication layer. Remote use
requires an authenticated front end. HTTP connections, inflight requests and
output queues are bounded. A slow consumer is cancelled; a non-streaming client that closes its connection while waiting is detected and its job cancelled (`client_disconnects` counter). A dead generation
worker fails readiness and causes the HTTP process to exit for its supervisor.

North-Mini-Code declares `<|END_THINKING|>` as its deferral marker and an
answer *envelope*: its `<|START_TEXT|>` / `<|END_TEXT|>` channel markers are
ordinary vocabulary entries, so a grammar would otherwise mask the framing the
model was trained on. The processor keeps them out of the grammar's view and
admits one opener before any answer text, a closer once the answer is a
complete match, and only end-of-turn after it. Structured requests on North
therefore think by default like everything else.

### Coding-agent client compatibility (per request, auto-detected)

State: **implemented** and CPU-tested with the real Codex CLI and Claude Code
binaries; **GPU-unverified**.

Agent compatibility translates Codex (Responses) and Claude Code (Messages)
request shapes onto the chat contract. See
[API parity: coding-agent clients](API-PARITY.md#coding-agent-clients-agent-compat).
The server resolves one decision per request. `--agent-compat` selects the
mode:

| Mode | Resolution |
|---|---|
| `auto` (default) | header > tenant policy > client detection > off |
| `opt-in` | header > tenant policy > off |
| `on` (a bare `--agent-compat` also means `on`) | on, unless the header says `off` |
| `off` | never; the header is ignored |

**The three sources:**

- **Header.** `X-MLX2-Agent-Compat: on|off` takes the highest precedence;
  `off` always wins. Any other value returns 400.
- **Tenant policy.** `--agent-compat-tenants <json>` maps a tenant id (the
  `X-Tenant-ID` the server resolves today) to
  `{"agent_compat": "on"|"off", "custom_tool_grammar": "validate"|"off"}`,
  where the grammar key is optional. Once rm11 (tenant auth, branch
  `claude/rm11-tenant-auth-20260919`) merges, this should become key-bound
  through an `agent_compat` field in its keys file, so that a tenant's mode
  follows its authenticated key rather than a self-asserted header. This
  branch doesn't depend on rm11's code.
- **Detection** (`auto` only). The rules come only from identity headers
  observed on the wire (`tests/fixtures/agent_clients/headers.json`):
  - **Codex** on `/v1/responses`: an `originator` of the form `codex_<name>`,
    plus a `user-agent` that begins `<that originator>/<version>`. codex-cli
    0.145.0 sends `codex_exec` / `codex_exec/0.145.0 (...)`.
  - **Claude Code** on `/v1/messages`: a `user-agent` of
    `claude-cli/<version>`, plus `x-app` or a `claude-code-*` entry in
    `anthropic-beta`. Claude Code 2.1.269 sends both.

  Both signals must agree, versions are matched loosely, and each client is
  recognized only on its own API family. The official OpenAI and Anthropic
  Python SDKs (`OpenAI/Python`, `Anthropic/Python`) are never detected, so
  their traffic is byte-identical to translation off.

**Custom-tool grammar.** The mode resolves in the same order:
`X-MLX2-Custom-Tool-Grammar: validate|off` > tenant `custom_tool_grammar` >
`--custom-tool-grammar` (default `off`).

**Receipts and counters.** Whenever the decision differs from main's, the
receipt carries `agent_compat: {enabled, source: header|tenant|detected|server,
client: codex|claude_code|null, grammar}`. Plain SDK traffic gets no receipt
field. The per-source counters are `agent_compat_source_*`, and the
per-client counters are `agent_compat_detected_*`.

**Spoofing.** Detection can only *enable translation*. Reasoning and thinking
signatures are still HMAC-verified against model and tenant, and the resolved
custom-tool grammar is still enforced, so a spoofed identity gains nothing it
couldn't request with the header.

**Conversation consistency.** A stored response records its mode in the
`mlx2.agent_compat` receipt. The following fail with a clear 400 naming the
resolved source:

- a `previous_response_id` chain continued under the other mode;
- `custom_tool_call(_output)`, namespaced `function_call` or `phase` message
  items replayed without agent compatibility;
- an omitted-thinking (`mlx2.thinkingc`) signature replayed without agent
  compatibility.

These never misparse. Mode conflicts are counted in
`agent_compat_mode_conflicts`.

**APCv2.** The two modes render prompts differently: developer/system
folding, the tool shims and replay merging all differ. So there is no prefix
reuse across modes, and a conversation must keep one mode for its turns to
share a prefix. Within one mode the rendering is deterministic, and turn N is
an exact prefix of turn N+1.

**Harness.** `scripts/agent_client_conformance.py`. `--scripted` runs on CPU
through auto-detection. `--model ... --i-own-the-gpu` runs real sessions
through auto-detection with no opt-in flag; `--compat-mode on` is available
for comparison.

### Prompt progress, tokenize and apply-template

Streaming Chat, Completions, Responses and Messages requests accept
`"return_progress": true` (default false; a non-streaming request that sets it
is rejected with 400 because there is no pre-output event to carry it).
While the prompt is prefilled the stream reports

```json
{"prompt_progress": {"processed": 3072, "total": 8192, "cached": 2048,
                     "replay": false, "time_ms": 412}}
```

`total` is the admitted prompt length, `cached` the APCv2 hit at admission,
`processed` the prompt tokens now covered including the cache, and `time_ms`
the time since admission started. Chat/Completions carry it on an empty-delta
chunk (`delta: {}` / `text: ""`), Responses on `response.in_progress`, and
Messages on `ping`; every update precedes the first output token. Updates
follow completed prefill slices on the ordinary, native-MTP, prompt-lookup
and external-draft generators; `processed` strictly increases (a future
recovery replay restarts it and says `replay: true`). Updates are coalesced:
each job holds at most one queued progress event, a newer update rewrites a
still-queued one, and a full event queue drops the update instead of
triggering the slow-consumer 429. Strict/required/named Chat tool streams and
hosted-tool Responses streams are buffered until terminal validation and
therefore carry no progress. The terminal receipt of a `return_progress`
request adds `prompt_progress: {updates, dropped}`; other receipts are
unchanged. Queueing and memory-admission waits do not advance the counter.

`POST /tokenize` and `POST /apply-template` take a Chat (`messages`) or
Completions (`prompt`) body, validated exactly as a generation request, and
run no inference. `/tokenize` returns
`{"tokens": [...], "count": N, "max_model_len": <effective context>}` from the
same adapter `prompt_tokens` call admission uses (so `count` equals
`/v1/messages/count_tokens` and the served `prompt_tokens`). It can inspect
prompts longer than the context limit. `/apply-template` returns
`{"prompt": "..."}` from the adapter's optional `render_prompt(request) -> str`;
Flash-Next/Qwen3.8/Qwen3.6, North, Muse, Xing and mlx-vlm implement it, with
the invariant `encode(prompt, add_special_tokens=False) == prompt_tokens`
(checked against the real local tokenizers in
`tests/test_progress_tokenize.py`). Adapters without it return 501. Media
parts are rejected with 400: their prompt ids depend on encoder preparation
these read-only endpoints do not run. Both are ordinary POST routes (any
request authentication gate applies to them, unlike `/health`) and are
labelled `tokenize` / `apply_template` in HTTP metrics.

## Cache, scheduling and memory

**APCv2 is the sole prefix cache**, including the ordinary reference route.
Its radix index is an internal data structure. Layered frozen segments and
copy-on-write branches preserve ownership. Target, recurrent, sparse/QSA and
draft state remain bound to their model/runtime/layout revision. Prompt-boundary
and completed-generation snapshots publish only committed state.

The prefix cache is one namespace for every client by default, so a client's
`cached_tokens` reflects any earlier client's prompt. `--tenant-scoped-cache`
binds the APCv2 namespace to the request tenant and is part of the
qualification settings. Without tenant authentication (below), that tenant is
the unauthenticated `X-Tenant-ID` header. Persistent-block snapshots carry a
per-process keyed MAC in their manifest in addition to per-block digests, so a
swapped payload with recomputed checksums is refused as corrupt.

A new API request keeps its new sampling seed when it hits APC. Explicit
continuation state is separate. Idle disk snapshots pair target and speculative
state; corrupt or missing files become misses. Disk caching is process-local
scratch, not a persistent conversation store. Use a dedicated directory per
process. The disk tier is bounded at 64 GiB and normally spills after 180 idle
seconds; resident capacity belongs to the qualified profile.

Default-off `execution_policy.apc_interior_checkpoints` publishes bounded exact
hybrid-prefill boundaries only on supported routes and after captured/published
qualifier observations. `skip_writing_prefix_cache:true` suppresses publication
for one request. Placement is `pow2` by default: the legacy `min_stride * 2^k` lattice, with
settings identical to earlier releases. Two other placements exist:

- `placement: "turns"` checkpoints before each chat-template turn-start token.
  The token is detected from the tokenizer's generation prompt, or an adapter
  declares it with `apc_turn_marker_ids()`.
- `placement: "auto"` takes, in order: the end of the system/tool preamble,
  the start of the final user turn, then a nested tail lattice
  (`floor((P-2)/s)*s` for `s = min_stride * 4^k`), spaced at least
  `min_stride` apart.

`headroom_fraction` caps checkpoint bytes at a fraction of the headroom left
after admission. `min_uncached_fraction` (default 0) skips capture entirely on
a request that already resumed from a hit covering more than
`1 - min_uncached_fraction` of its prompt: such a prompt is a linear
continuation whose own `P-1` boundary already serves the next turn, so the
checkpoints it would capture are pure cost. The string `"auto"` expands to
`{count:4, min_stride:256, placement:"auto", headroom_fraction:0.5,
min_uncached_fraction:0.5}` -- the configuration that carried the GPU
qualification. The preset is what `"auto"` means, not what any route serves:
`apc_interior_checkpoints` remains opt-in per route profile until the serving
profiles are themselves re-qualified with it on. The
2026-09-19/20 GPU qualification (Flash-Next MTP and Qwen3.8 27B MTP,
`qualification/runs/interior-ckpt-20260919/`) ran exactly that
`headroom_fraction:0.5` / `min_uncached_fraction:0.5` arm: a 12K shared preamble
cut TTFT 8.30 s to 0.52 s and a 30K shared document 23.75 s to 0.90 s, with
zero warm-vs-cold token diffs and the linear control within noise (-0.7%). At
`headroom_fraction:0.25` the 30K document's checkpoints (~1.1 GiB each) did not
fit the measured headroom and RAG gained nothing. Chat turns already resume from each turn's own
P-1 and finished entries. Interior checkpoints pay off when one prompt diverges
at a point no earlier request ended on, such as a shared system prompt plus tool
schemas across sessions, or a shared RAG document. A reused interior entry is
evicted like an ordinary entry rather than first, and interior entries hold
their own resident-entry count allowance (`max_interior_entries`, default the
APCv2 `max_size`) so a cache full of prompt boundaries cannot evict a fresh
checkpoint at its own publication. Checkpoints inside a media
span are never captured. `scripts/qualify_interior_checkpoints.py` is the
GPU-gated qualification: it interleaves the `off`, `pow2` and `auto` arms and
refuses any arm whose interior-hit counter is zero. Admission performs one final allocator reclaim before
eviction/rejection and charges each admitted allocation once.

With `--apc-persist-dir`, exact APCv2 entries use immutable digests and native
runtime identity, bounded corruption quarantine and startup rescan. A validated
`session_id` (or matching `X-mlx2-Session-ID`) tags tenant-owned entries;
`/v1/apc/sessions` exposes list/state plus park, resume and delete. Pin TTL,
resident/disk budgets and shutdown spill are independently bounded by the
`--apc-session-*` and `--apc-persist-*` flags. This is unrelated to the MCP
`session_id` transport header in the tool backend.

A qualified profile that selects APC persistence must observe a write in an
earlier process, startup rescan registration, and a restore in the candidate
process. Profiles with disk-backed sessions must also observe park, resume,
prefetch restore, and the resulting hit; configured capability without these
lifecycle observations is rejected.

### Tenant authentication (opt-in)

By default the tenant is the client-supplied `X-Tenant-ID` header, or
`"default"` if the header is absent. The following consumers scope by it:

- the APCv2 namespace under `--tenant-scoped-cache`;
- APC sessions and their per-tenant pin caps;
- the Responses, Files and Batches stores;
- `previous_response_id`;
- media `file_id`s;
- batch cohorts;
- the tenant binding in reasoning signatures.

Nothing verifies the header.

Two options turn on authenticated mode:

- `--tenant-auth-keys-file` takes a JSON file of SHA-256 API-key digests
  mapped to tenants:
  `{"version":1,"keys":[{"key_id","tenant","sha256","scopes"?,"disabled"?,"adapters"?}]}`.
  The file must not be group- or world-writable.
- `--tenant-auth-token-secret-file` (mode 0600) or
  `--tenant-auth-token-secret-env` takes a secret of at least 32 bytes. It
  signs `mlx2t1.` HMAC tokens that carry `sub`, `iat`, `exp` and `scp`.

To mint credentials:

- `python -m mlx2.tenant_auth new-key --tenant T --key-id K` prints a new key
  once, together with its file entry.
- `python -m mlx2.tenant_auth mint-token --tenant T --secret-file F` prints a
  token.

In authenticated mode:

- **Credentials.** Clients send the credential as `Authorization: Bearer` or
  `x-api-key`:
  - Codex uses `env_key` and sends it as Bearer.
  - Claude Code sends `ANTHROPIC_API_KEY` as `x-api-key` and
    `ANTHROPIC_AUTH_TOKEN` as Bearer.
  - If both headers are sent and they differ, the request gets 401.
- **Guarded paths.** Every path is guarded except `/health` and loopback
  `/metrics`, including unknown paths.
  - Admin routes keep their own loopback check and admin token.
  - Failures return 401 (with `WWW-Authenticate: Bearer`) or 403. Each
    failure reason has a fixed message.
  - The connection closes after a refusal.
- **Tenant header.** The tenant always comes from the credential.
  `--tenant-header-policy must-match` (the default) refuses a request with
  403 if its `X-Tenant-ID` differs from the credential's tenant. `ignore`
  drops the header.
- **LoRA scope.** `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`
  require the `adapters` scope.
- **Shared cache.** Startup refuses authenticated mode without
  `--tenant-scoped-cache`, unless you pass `--tenant-auth-allow-shared-cache`.
  An authenticated tenant on a shared prefix cache can still learn other
  tenants' prompts from TTFT and `cached_tokens`.
- **Secret separation.** Startup also refuses a token secret equal to the
  reasoning signing key.
- **Receipts and counters.**
  - Responses carry `X-MLX2-Tenant-Auth: api_key|token`. This header reports
    the method only.
  - `/v1/status.tenant_auth` reports the mode, the verified counts by method
    and the failure counts by reason.
  - Prometheus exports `mlx2_tenant_auth_total`,
    `mlx2_tenant_auth_failures_total` and `mlx2_tenant_auth_header_total`.
  - Tenant names, key ids and credentials never appear in metrics, logs or
    error bodies.
- **Existing state.** Authenticated tenants reuse the plain tenant strings
  already recorded in `--apc-persist-dir` and `--api-state-dir`. Start
  authenticated mode on fresh state directories, or pick tenant ids that never
  appeared unauthenticated.

Tenant authentication does not add TLS or rate limiting; a fronting proxy
provides those. Tokens are revoked by rotating the secret; keys are revoked
with `disabled`.

### Quiesce, suspend and resume

The opt-in local administration surface moves the live process through
`serving -> draining -> quiesced` or `serving -> draining -> suspended`, then
back to `serving`. `GET /v1/admin/state` returns the current state, its
since-time, per-state timestamps and the last transition result. The same
object is present at `/v1/status.quiesce`, and `mlx2_server_state` exports the
state as a one-hot Prometheus gauge.

Administration is fail closed. Every `/v1/admin/*` request must originate from
`127.0.0.0/8` or `::1`; forwarded-address headers are not trusted. If
`--admin-token-file PATH` is configured, the loopback request must additionally
send `Authorization: Bearer TOKEN`. The token is compared in constant time and
the server refuses to start unless the file is regular, owned by the process
uid and mode `0600`. Without a token file the endpoints are loopback-only;
non-loopback requests always receive 403.

- `POST /v1/admin/quiesce` accepts exactly
  `{"drain_timeout_seconds": 600, "suspend": true}`; both fields are optional,
  the timeout is bounded to 0.1 through 3600 seconds, and the response is 202.
  Repeating the request while draining, quiesced or suspended returns the
  current state without starting another drain. `suspend:true` returns 409
  before draining unless `--apc-persist-dir` or the process-local scratch
  `--cache-dir` tier is configured.
- `POST /v1/admin/resume` accepts an optional bounded
  `prefetch_sessions` list of `{tenant, session_id}` objects. Admission reopens
  immediately; disk APC entries otherwise restore lazily. A resume during
  draining cancels the drain before the worker owns its idle boundary, so no
  suspend occurs. Requested session prefetches then run through the existing
  worker-owned resume path.
- `GET /v1/admin/state` is read-only. Empty POST bodies select defaults; unknown
  fields and malformed objects are rejected.

Closing admission does not cancel accepted work. Existing streams keep
streaming, and queued, memory-deferred, fan-out, multi-choice/cohort, hosted
continuation and already-created Batch API work can finish. New generation,
embedding, rerank, audio, batch-submission and session-resume-prefetch work gets
503 with `Retry-After` and its normal OpenAI or Anthropic error envelope. Models,
status, metrics, stored-response reads, token counting, and session get/list/
delete remain available. `/health` deliberately returns 503 with
`{"status":"draining"}`, `quiesced`, or `suspended` so a load balancer routes
away. At the drain deadline the model worker terminates remaining work with the
explicit 503 reason `drain timeout`, records it, and completes the transition.

Suspend runs only on the model worker at its idle boundary. It releases
request/fan-out cache capsules, spills every exact resident APCv2 checkpoint to
the persistent tier when configured or otherwise the scratch tier, and clears
MLX allocator cache. It preserves session tags, pins, retention roles,
speculative sidecars and interior checkpoints; individual unwritable entries
are dropped and counted without crashing the service. Approximate state is
never persisted. Model weights and the process stay resident. The transition
result reports entries, bytes, failures, duration, and MLX active/cached memory
before and after.

`--drain-on-sigterm SECONDS` opts SIGTERM and SIGINT into a drain without
suspending, followed by the normal shutdown path. Thus
`--apc-persist-on-shutdown` still applies. A second signal uses the immediate
shutdown path, and omitting the flag preserves immediate signal handling.

The qualification runner may use this surface to take a service out of
rotation and release caches instead of stopping unrelated services. That is an
operator workflow note, not qualification evidence: route selection still
requires artifact-, settings- and observation-matched receipts.

### APCv2 Metal resume/prefetch check

Run the dedicated check only in an exclusively owned GPU window. It refuses
model or Metal execution unless the ownership acknowledgement is present and
prints the complete plan before loading anything:

```bash
PYTHONPATH=src .venv/bin/python scripts/gpu_check_apc_prefetch.py \
  --i-own-the-gpu \
  --output qualification/apc-prefetch-metal.json
```

Use `--dry-run` without the acknowledgement to inspect the plan on CPU, or add
`--model-path /absolute/path/to/artifact` to replace the built-in tiny Qwen4
hybrid fixture. A pass means resume was queued while ordinary decode was
active on the same model worker, the restore did not run until that worker's
idle boundary, the prefetched checkpoint produced an APCv2 hit, cold and warm
generated token IDs matched exactly, and the worker reported no MLX
stream/thread failure. The check is verification evidence only; it does not by
itself qualify or select APC persistence for a production route.

External drafts apply logits processors through disposable `probe()` state
before grammar-masked proposal scoring. Permanent ordinary external cohorts
and depth-zero self-MTP use direct target paths; self-MTP still honors the
true-batched knob. Default-off FLy relaxed verification requires execution
policy selection and qualifier observation; structured and unsupported sampled
lanes remain exact.

Default-off **self-MTP copy drafts** (`execution_policy.self_mtp_copy_draft`,
e.g. `{"num_draft": 2, "self_mtp_copy_draft": {"enabled": true}}`) let a
self-MTP lane propose a verbatim span from its own context, meaning the prompt
(including APC-restored history) plus generated tokens, in place of the MTP
head's drafts for one round.
- **Verification.** Copied rows join the same ragged batched verify forward
  and are verified exactly. Greedy takes the matching prefix; sampled lanes
  sample every row from the transformed target law and accept while the
  sample equals the copy, which is the speculative-sampling law for a
  point-mass draft. FLy never relaxes a copied row.
- **Sizing and gating.** A congestion window sizes the span (probe 2,
  double on a full accept, 1.5x accepted on a short one), capped by
  `max_span` (default 8). In a cohort wider than one lane, the span is
  capped by `batched_max_span`, which defaults to **0**: batched cohorts do
  not copy at all. On GPU (2026-09-19, Qwen3.8-27B) cohort copies capped at
  the head depth lost 5% on code and 3% on prose at B=4 while B=1 gained
  20%, so the lever is single-lane by default; `null` restores the
  head-depth cap and a positive value sets it explicitly. A windowed yield-per-cost gate declines copies
  that lose to the head and re-probes periodically.
- **Composition.** Adaptive depth sees head rounds only. A K=0 cohort still
  verifies a pending copy.
- **Constraints.** The policy is refused off the native self-MTP route; the
  prompt-lookup route stays mutually exclusive. Selection requires the
  `self_mtp_copy_draft` qualification feature, which is observed from the
  scheduler counter `self_mtp_copy_rounds`.
- **Observability.** Receipts carry `mtp.copy_draft` with per-source
  copy/head counters. Prometheus exports `self_mtp_copy_*` under mechanism
  `self_mtp_copy_draft`.

Admission uses host availability, Darwin process footprint and MLX allocator
accounting. It preserves a **host-scaled service/driver reserve**, derived
from two readings: installed RAM and Metal's
`max_recommended_working_set_size`. The host's non-lane quota is 25% of
physical RAM, and `host - advisory` is the part macOS has already withheld,
so the service reserve supplies only the shortfall,
`clamp(3 GiB, 16 GiB, advisory - 0.75 * host)`. The driver allowance is
charged inside the process budget and is not withheld by the advisory, so it
scales with the advisory instead: `clamp(0.75 GiB, 4 GiB, advisory * 4/112)`.
A 128 GiB host (112.0 GiB advisory) therefore still reserves exactly
16 GiB + 4 GiB, bit for bit; a 36 GiB M3 Pro (28.08 GiB advisory) reserves
3.0 + 1.003 = 4.003 GiB.

The withheld share is *not* a constant fraction of the host -- 12.5% at
128 GiB, 22.0% at 36 GiB, both measured -- which is why the earlier rule of
12.5% of installed RAM double-charged the OS margin on small machines. It
charged a second OS margin sized as though macOS always withheld 12.5%, and
it charged it against the advisory residue, the term that binds once a model
is resident. The physical rule is exactly this rule evaluated at a 0.875
advisory ratio, so it remains the fallback when the advisory cannot be read.
The server probes `mx.device_info()` once at start for both figures and
injects them; with neither reading the reserve falls back to the 128 GiB
calibration. An explicit override may raise the reserve but not lower it
below the host floor.
Flash-Next supplies
model-specific cache geometry; a usable resident checkpoint is leased before
pressure eviction and charged for the required target/draft copy and growth.
A resident-only lookup cannot restore disk arrays before allocation admission.
Reclamation drains pending work, clears scratch and remeasures headroom; leased
checkpoints are protected.

Optional asynchronous QSA promotion yields before useful checkpoint eviction.
Under memory pressure, admission first settles its queued allocation so resident
bytes are not charged again as pending. If pressure remains, it cancels and
drains that optional copy, then allows up to two seconds for measured accounting
to recover, retrying reclamation every 250 ms. It credits no hypothetical freed
memory. Capacity-only limits skip this wait; unconstrained promotion retains
its asynchronous overlap.

A request awaiting delayed memory recovery retains its lookup/lease in a bounded
worker queue, retries every 250 ms and expires after 60 seconds. It occupies no
execution lane while waiting. Active requests continue between retries;
cancellation, timeout and shutdown release waiting leases. Executing requests
that cannot progress for 60 seconds also fail explicitly. Admission never
forces an over-budget single request to run.

A request that a live segmented width lock keeps deferring continues as ordinary decode beside the cohort after four deferrals (`width_lock_plain_fallbacks`) instead of waiting for the cohort to drain. Segmented cohorts retain independent row state. Flash-Next can reclaim unused
checkpoints before reducing a memory-constrained cohort. The external draft
candidate instead progresses a fitting subset first and reclaims only when no
subset fits; its rotating selection prevents starvation. These policies retain
lane limits and do not evict checkpoints solely because a lane cap is reached.

`observed_compute_widths` reports native MTP widths that actually ran. Ordinary
and external routes have their own receipts. HTTP concurrency is not a compute
width guarantee. Request `ttft_seconds` and `elapsed_seconds` begin when the
worker starts preparing the request and include any subsequent memory wait;
benchmark wall time also includes initial ingress queueing.

Parallel `n` sampling uses APCv2 one-prefill fanout only after the leader's
committed prompt boundary is stored and every sibling has acquired a leased
branch covering that boundary (apart from the one-token replay anchor required
by exact cache continuation). Store failure, eviction, a short lookup, or a
partial multi-sibling acquisition fails the sibling group closed before it is
published. The per-request `parallel_prefill.one_prefill` field is derived from
that attestation rather than group membership.

Prompt-lookup decoding remains a distinct speculative route. Admission charges
the target verification forward at width `num_draft + 1`; its throughput and
proposal/rollback engagement must not be reported as ordinary decode.

### Batched prompt-lookup verification

Prompt lookup used to verify one lane per target forward. Lanes whose planes
are all stock `KVCache` / `RotatingKVCache` (Muse-Glimmer, North-Mini-Code) now
share one forward per round through the same segmented KV transaction the
external-draft route uses: every lane appends its own ragged verify block, and
the commit republishes exactly the consumed prefix per lane from the K/V it
already computed, so a rejected tail costs no replay forward. Lanes stay
request-private B1 caches; nothing is merged. Hybrid models with recurrent
planes keep the per-lane driver. Policy key `batched_verify` (default `true`;
`rotating_replay: true` selects that per-lane transaction instead). Counters:
`pld_batched_rounds`, `pld_batched_lanes`, `pld_batched_max_width`; receipts
and `execution_width` report the shared width. State: implemented, exact
against per-lane and plain greedy on CPU; GPU sanity recorded under
`qualification/runs/known-limits-20260918/`.

### Bit-exact verify (`--verify-bitexact`, default off)

The mlx fork picks a quantized-matmul kernel by row count M, and the kernels
reduce in different orders:

- `qmv_fast` at M = 1;
- `qmv_wide` for small M;
- NAX `affine_qmv_nax` at M = 8..16 on M5;
- split-K `qmm` above that.

So a request's greedy output can depend on how many lanes share its verify
call. At 4 lanes the verify M is 12, which routes to NAX, and 2 of 4 prompts
on the 27B diverge from 1 lane
(`qualification/runs/qmm-nax-small-m-20260919/`).

`--verify-bitexact` turns on the fork's batch-invariant mode,
`mx.metal.set_qmv_bitexact` (fork branch
`claude/rm09-bitexact-verify-20260919`). Every transposed quantized matmul
with M <= `qmv_bitexact_max_m` (64) computes each row with exactly the
single-row `qmv_fast` arithmetic, through a multi-row kernel that only shares
weight loads between rows. Larger M uses the non-split `qmm`. MoE gathers use
`gather_qmv`, bounded to 8 x 64 gathered rows.

- **Scope.** The mode is process-global and bound once in the worker before
  the first request. Startup fails if the installed mlx lacks the API.
- **Requests.** A request can set `verify_bitexact: true`. On a server
  without the mode that is a 400, not a silently width-dependent result.
- **Receipts.** They carry `verify_bitexact`, which is `false` unless proven,
  plus `verify_bitexact_detail`. The claim needs all of:
  - the mode stayed on for the whole request;
  - mlx reports the mode as on;
  - mlx's host-side route counter (`qmv_bitexact_dispatches`, never syncs)
    advanced while the request ran.
- **Residual sources.** The detail also lists the width-dependent sources the
  mode does not cover:
  - `unquantized_matmul`: on the 27B this is the BF16 `mtp.fc` draft head, so
    it affects acceptance only;
  - `moe_gather_above_bound`;
  - `joined_slab_sdpa`: segmented batch caches attend each row over its own KV
    and are unaffected;
  - `qsa_shared_suffix_merge`;
  - `qmm_above_max_m_unverified`.
- **Qualification.** Settings record the mode, the profile gains a
  `-verify-bitexact` suffix, and qualification requires
  `feature_verify_bitexact` (a route counter above zero).
- **Metrics.** Prometheus exports `mlx2_verify_bitexact_*`.
- **Cost.** Measured, not estimated. At M = 1 `qmv_fast` already runs at the
  memory roofline (510-540 GB/s on an M5 Max), so the per-row arithmetic the
  mode insists on is added ALU on a saturated pipe and the cost grows roughly
  linearly with M. Cache-cold over the 27B projections
  (`qualification/runs/rm09-bitexact-20260919/`, row tile 4): M = 4 is 1.9x
  the M = 1 time, M = 8 is 3.7x, M = 12 is 5.5x (median; 7.1x worst shape)
  and M = 16 is 7.5x. The default route is about 2x at M = 12. End to end on
  the 27B the mode costs about 45% of 4-lane throughput (47.0 vs 85.6 tok/s)
  and about 23% at 1 lane (35.6 vs 46.1 tok/s). It is opt-in: for determinism
  and qualification runs, never the default.

State: implemented and GPU-qualified for bit-equality, 2026-09-19
(`qualification/runs/rm09-bitexact-20260919/`).

- Fork kernel tests, TF32 off: green, including bitwise row equality across
  M = 1..64 and the gather route.
- 27B end to end (`scripts/gpu_check_verify_bitexact.py`): the bit-exact arm
  is 4/4 identical between 1 lane and 4 lanes in both repetitions. The
  control arm is 2/4 in both, diverging at the same offsets.
- The cache-cold cost criterion (M = 12 at most 3x M = 1) was **not** met at
  5.5x; see the wiki no-go note. Nothing about the mechanism failed - no row
  ever mismatched - so the mode stands as an opt-in determinism tool and not
  as something the serving default can afford.

### Rotating replay for prompt lookup (default-off)

Stock `RotatingKVCache` layers (sliding-window attention) cannot be trimmed
once the ring has wrapped, so the prompt-lookup route copies every ring's K/V
before each verify forward and restores the copy on rollback. The
`prompt_lookup` policy knob `rotating_replay: true` replaces that copy with a
per-round `RotatingReplayTransaction` (`src/mlx2/runtime/rotating_replay.py`):

- The transaction arms each stock ring for the verify inputs (anchor plus
  proposal). On partial acceptance it restores the pre-verify ring and its
  replay callable is the round's single replay forward over the committed
  inputs, which advances every lane cache together; non-rotating caches keep
  the existing snapshot/trim rewind. On full acceptance the verify forward is
  published as-is with no replay. Proposal-free rounds verify only the anchor,
  cannot roll back, and take neither a transaction nor a copy.
- `max_proposal_tokens` bounds the proposal the transaction will accept
  (default `max(num_draft, verify_cliff_end)`). A refused transaction (bound
  exceeded, a non-stock rotating cache, an already armed ring) falls back to
  the copy snapshot for that round; the lane continues and output is unchanged.
- The knob is generator-scoped policy; per-request configuration cannot select
  it.

State: **implemented, default-off, unqualified**. CPU exactness against the
copy snapshot is covered by `tests/test_pld_rotating_replay.py` (wrapped ring,
partial acceptance, refusal fallback); no serving route has been qualified with
it and no profile selects it. When a served policy enables it, qualification
requires the observed-use check `feature_prompt_lookup_rotating_replay`.

Scheduler counters: `pld_rotating_replay_rounds` (verify rounds owned by a
transaction), `pld_rotating_replay_replayed_tokens` (tokens re-run by the shared
replay forward after partial acceptance) and `pld_rotating_replay_refusals`
(rounds that fell back to the copy snapshot).

### Per-round verification histograms in the receipt (always on)

Every speculative route reports two per-round distributions, as plain JSON
objects mapping a **value** to the **number of rounds** that saw it. They are
host integer counters bumped once per verify round from values already on the
host, so they cost nothing on the device and are populated unconditionally.

| Field | Key | Invariant |
|---|---|---|
| `verify_span_hist` | positions in that round's verification forward | `sum(span * rounds)` = verified positions |
| `verify_accept_hist` | speculative tokens accepted in that round | `sum(accepted * rounds)` = the route's `accepted` aggregate |

Where they appear:

- **Self-MTP** (`mtp.stats.verify_span_hist`, `mtp.stats.verify_accept_hist`)
  for both the continuous-batched and segmented routes. One entry per
  committed cycle, copy-draft rounds included; the span key is the round's
  draft depth `+ 1`. `sum(rounds)` equals `mtp.stats.cycles`.
- **External draft** (`speculative_receipt.verify_span_hist`,
  `...verify_accept_hist`). Recorded only for rounds that actually proposed,
  so `sum(rounds)` equals `external_rounds` and the ordinary fast path does
  not dilute the distribution.
- **Prompt lookup** (`verify_span_hist`, `verify_accept_hist` in the
  `mlx2.prompt-lookup-live.v1` receipt). `verify_span_hist` predates this and
  keeps its meaning; `verify_accept_hist` is the new one. `sum(rounds)` equals
  `cycles`, i.e. plain rounds are included as span 1 / accept 0.

**Why the accept histogram and not just the aggregate.** A draft accepted to
depth `k` would also have been accepted under any shallower cap, so the commit
rate at every depth below the one that was run follows from a single run's
histogram by truncation:

```
tau(k) = sum((min(a, k) + 1) * rounds for a, rounds in hist.items()) / sum(rounds)
```

`tau(k)` is tokens committed per target verification forward at cap `k`; the
`+ 1` is the bonus token the target always authorises. A depth sweep that used
to need one arm per depth now needs one arm at the deepest depth, and because
numerator and denominator come from the same run their noise cancels instead
of compounding -- cross-run repeatability of the same quantity on this machine
measures around 5%, larger than most effects such a sweep is trying to
resolve.

The `draft_cycles` / `draft_accepted` aggregates are exactly the
zeroth and first moments of `verify_accept_hist`, and are unchanged.

### Self-MTP acceptance logging (default-off, qualification-only)

`--mtp-acceptance-log PATH` writes bounded JSONL rows of per-draft-position
features, token ids and conditional acceptance labels for the native self-MTP
route. It is restricted to `--qualification-mode` and refuses any other route.

- **Features.** Each lane carries a `DraftConfidenceProbe`
  (`src/mlx2/runtime/mtp_confidence.py`). For every drafted position, the
  batched self-MTP round computes the draft `top1_prob`, `entropy` and
  `margin` on device, plus an optional fixed sketch of the MTP hidden state.
  They are fetched in the **same `mx.eval` as the greedy accept payload**, so
  there is no extra host sync -- a CPU test asserts identical sync sites.
- **Labels.** The first rejected position is labelled 0; later positions and
  lookahead positions are censored (`null`), so a consumer can compute
  conditional per-position acceptance without counting a position the round
  never reached.
- **Lookahead.** `--mtp-acceptance-log-lookahead N` drafts N extra unverified
  greedy positions per cycle, so the log observes positions past the served
  depth instead of only ever seeing the depths the route already runs. The
  drafts are dropped before verification and the draft cache is trimmed as
  usual, so greedy output is bit-identical; sampled lanes never look ahead, so
  their RNG streams are unchanged.

Scheduler counters: `mtp_confidence_feature_cycles` (cycles whose features
were observed) and `mtp_acceptance_log_records` (rows written). Both are
bounded integer counters and map to the `self_mtp` mechanism, not to
`adaptive_mtp`: the log is independent of the depth controller.

Offline consumers, both CPU:

- `scripts/train_mtp_confidence.py` fits a logistic acceptance head with
  hashed previous-token and draft-token biases, calibrates it by sequential
  temperature scaling, and reports holdout ECE/AUC.
- `scripts/best_constant_depth.py` combines a **measured** cycle-cost table
  with the log's **measured** conditional acceptance and prints the constant
  depth with the highest expected committed tokens per second at each width.

`scripts/mtp_confidence_gpu.py {profile,collect,ab,diag}` is the GPU package
that produces both inputs.

**Why there is no confidence-scheduled depth controller.** A DSpark-style
per-cycle scheduler built on exactly these confidences
([arXiv 2607.05147](https://arxiv.org/abs/2607.05147)) was implemented and
measured on GPU (rm10, `qualification/runs/mtp-confidence-20260919/`). It is a
**no-go and is not on this tree**: its arms ran 14.0/5.7/3.2% below fixed
depth on 27B at 1/2/4 lanes, against a pre-registered "never more than 2%
below fixed" clause. Separability was never the problem -- holdout top-1 AUC
is 0.93-0.94 at the first drafted position on all three models and the trained
head adds 1-2 points -- but `scripts/best_constant_depth.py` finds the adapter
cap (`num_draft` 3) optimal at every width on 27B and 35B-A3B, 7-15% ahead of
depth 2, while one arm's run-to-run spread is about 3%. There is nothing for a
per-cycle policy to win, so what is kept here is the measurement apparatus and
not the actuator.

### Live Spomin prompt compaction (default-off, approximate)

`--spomin-live-surgery '{"enabled": true, "capacity_tokens": N}'` compacts a
cold prompt at its isolated B=1 post-prefill boundary once it exceeds
`0.70 * N` tokens, removing the oldest unprotected fixed-size segments
(`segment_tokens`, default 1024; the first `protect_prefix_segments` and the
most recent segment are kept) down to `0.65 * N`. The surgical backend is
adapter-owned: standard attention (`KVCache` full layers, `RotatingKVCache`
sliding layers, RoPE parameters read from each layer, NoPE layers untouched)
for Muse-Glimmer and North-Mini-Code, QSA for dense Qwen4. Hybrid recurrent
state, armed MTP, non-default RoPE, a ring that still holds removed tokens, and
a warm shared APCv2 prefix are declined with a receipt; nothing is edited.

State: **implemented, CPU-qualified on a tiny North fixture; production-model
and GPU qualification remain open**. Restricted to `--qualification-mode` on
the ordinary route (no MTP, prompt lookup, external draft, or `n>1`).
Decode lanes batch normally; only prefill is isolated. Compacted state is
approximate and is never published to APCv2
(`counts.apcv2_store_skipped_approximate`). Evidence surfaces:
`mlx2.spomin_live_surgery` in each response receipt,
`/v1/status.spomin_live_surgery` (counts by status and reason), and the
qualifier's `spomin_surgery` check plus the `feature_spomin_surgery` gate.
Before surgery, mlx2 snapshots the exact full-prompt boundary. A successful
surgery may publish that revision-bound exact boundary to APCv2, so an
identical later request restores the original prompt and performs a fresh
private surgery; compacted and completion state remain excluded from exact
stores. Snapshot/store failures preserve safe serving and increment the
bounded `spomin` telemetry reasons.

### Approximate KV quantization (`kv_q8`, `kv_k8v4`)

State: **implemented, default-off, unqualified.** No model has qualification
evidence for it; it is not selected by any shipped profile and has not been
observed-used outside CPU tests. The fidelity harness and selection gate below
exist; the GPU measurements they need have not been run.

KV quantization is the first operation published through the approximate-state
seam (`runtime/approximate_state.py`, `runtime/approximate_kv.py`). It exists on
the **ordinary route**, and on the **native self-MTP route** only with
`compose_mtp` (see below).

- **Adapter-owned.** An adapter opts in with `approximate_kv_operations()`
  returning `{name: KVQuantizationDescriptor}` (key/value bits, group size,
  rotate, start). The default is empty. Only the Qwen3.8-27B adapter and its
  Qwen3.6-35B-A3B subclass declare `kv_q8` (K8/V8) and `kv_k8v4` (K8/V4), group
  size 64: their full-attention layers allocate plain `KVCache` and attend
  through the quantized-SDPA-capable helper, and their gated-delta
  `ArraysCache` state has no `to_quantized` and stays exact. Flash-Next (QSA
  index ledger, shared suffix, segmented promotion), Muse Glimmer and North Mini
  Code (sliding-window rotating planes) declare nothing. The serving lifecycle
  never branches on a model name.
- **Selection.** `--approximate-kv '{"operation": "kv_k8v4", "enabled": true}'`
  (engine: `ServingEngine(approximate_kv=...)`). Optional keys: `start_tokens`,
  `evidence`, `compose_mtp`; unknown keys fail. It requires
  `--qualification-mode`, or a `--qualification` record whose settings match
  and whose `feature_approximate_kv` and `feature_approximate_kv_fidelity`
  checks passed. It is refused with native MTP (unless `compose_mtp`), prompt
  lookup, external draft, live Spomin surgery and cache capsules. An adapter
  that does not declare the operation, or whose freshly allocated cache cannot
  take it (the server probes this before becoming ready), fails startup.
- **Mechanics.** The lane cache is quantized *before* the lane is inserted, from
  token 0, so prefill and decode both run on quantized planes and lanes batch
  through `BatchQuantizedKVCache`. `start_tokens` is an admission threshold, not
  a mid-decode switch: a prompt shorter than it stays an exact lane for its
  whole life (receipt `declined` / `below_start_tokens`). Exact and quantized
  lanes cannot merge into one batch, so `start_tokens > 0` requires
  `--max-lanes 1`.
- **Revision-bound.** The operation revision is the SHA-256 of the adapter
  artifact fingerprint, the operation name and its descriptor.
  `ApproximateKVController.apply` checks it, stages the lane state privately and
  requires the operation to advance the revision. In candidate mode the receipt
  carries `qualified: false`.
- **APCv2 isolation.** The exact prefix cache never receives approximate state.
  `ServingEngine._publish_checkpoint` is the only path into APCv2 (idle/disk
  spills, persistent blocks and cache capsules all derive from stored entries);
  it skips both the prompt-boundary and end-of-request stores for an
  approximate lane, and independently skips any cache that structurally
  contains quantized planes. Each skip increments
  `apcv2_store_skipped_approximate`. The policy is therefore not part of the
  cache semantic fingerprint. A warm exact prefix may be *read*: the leased COW
  branch is copied into a private plane list and quantized there, leaving the
  shared entry untouched (`requantized_prefix_tokens` in the receipt). Parallel
  `n` samples skip APCv2 fanout and prefill independently
  (`approximate_kv_fanout_bypassed`).
- **Evidence surface.** The policy and descriptor are part of `settings`, so a
  qualification record binds to them, and a qualified approximate route is
  registered with `approximate` fidelity rather than `numerically_bounded`.
  Every request receipt has `approximate_kv` (null while the policy is off;
  otherwise `status` `applied`/`declined`,
  `operation`, `fidelity`, `quantized_layers`, `start_tokens`, source and target
  revisions). `/v1/status` has an `approximate_kv` block (`state`, descriptor,
  revision, `applied`, `declined`, `requantized_prefix_hits`,
  `apcv2_store_skipped_approximate`). `qualify_serving.py --require-feature
  approximate_kv` fails a candidate whose applied count is zero. Memory
  admission still charges the exact-KV estimate.
- **Self-MTP composition (`compose_mtp`).** Default off; `"compose_mtp": true`
  is valid only with native MTP (`compose_mtp` on the ordinary route is
  refused, and approximate KV on MTP without it is refused). The engine
  quantizes each lane's **target** attention planes before insertion; the MTP
  head's draft cache is created by `make_mtp_cache()` and stays exact. The
  draft only proposes and the verify pass plus the exact sampling law decide,
  so the route is lossless *relative to the quantized target*: on CPU, greedy
  MTP + quantized output equals the ordinary quantized route token for token.
  Segmented true-batched self-MTP previously declined quantized rows
  (`true_batched_declined`, falling back to per-lane B1 forwards);
  `SegmentedBatchQuantizedKVCache` (`runtime/segmented_plain_kv.py`) is the
  quantized row view: per-row quantized SDPA over each lane's authoritative
  `QuantizedKVCache`, ragged trim for rollback, private `extract`. Mixed,
  mismatched-layout or normalized rows still decline to B1. Counters:
  `segmented_mtp.quantized_kv_segmented_layers` and
  `quantized_kv_segmented_attention_calls`; `/v1/status.approximate_kv`
  gains `mtp_lanes`, `compose_mtp`, `draft_cache`; each receipt's
  `approximate_kv` gains `route` (`ordinary`/`mtp`) and `draft_cache`. A
  qualified MTP + quantized route additionally requires
  `feature_approximate_kv_mtp` (observed MTP lanes). The key appears in
  `settings.approximate_kv` only when set, so existing ordinary-route records
  keep their settings hash.
- **Fidelity gate.** `scripts/measure_kv_quant_fidelity.py` measures, per
  operation and context (GPU default 4K/16K/32K/64K), teacher-forced per-token
  KL(exact‖quant), top-1 agreement, top-5 overlap and log-prob delta against
  the exact cache of the same model on the same stream; planted-needle
  retrieval on both arms (`quant_losses`); measured attention-plane bytes; and
  interleaved decode tok/s. The quantized arm comes from the adapter-declared
  operation, and an arm with a wrong quantized-plane count is refused.
  `runtime/kv_quant_fidelity.evaluate_fidelity_report` judges a report
  fail-closed (GPU device, the adapter fingerprint, contexts 4K/16K/32K
  present, ≥128 scored tokens, nonzero mechanism count):

  | Per context ≥ 4K | `kv_q8` | `kv_k8v4` |
  |---|---|---|
  | mean KL (nats/token) | ≤ 0.005 | ≤ 0.02 |
  | p99 KL | ≤ 0.05 | ≤ 0.2 |
  | top-1 agreement | ≥ 0.99 | ≥ 0.97 |
  | attention bytes vs exact | ≤ 0.55 | ≤ 0.42 |
  | needle losses (exact hit, quant miss) | 0 | ≤ 1 |

  These are priors to be confirmed by the first GPU runs. `qualify_serving.py
  --kv-fidelity-report report.json` turns a passing verdict into
  `feature_approximate_kv_fidelity`. `--cpu-tiny` validates the harness on a
  tiny model; its report can never pass. `scripts/bench_kv_quant_mtp.py` is
  the GPU serving A/B (ordinary/MTP × exact/quantized, interleaved) and
  refuses a quantized arm whose counters did not move.
- **Not composed (deferred).** APCv2 still never stores approximate state, so
  there is no fidelity-namespaced prefix cache and no smaller quantized
  spill/persistence. That needs a second semantic namespace keyed by the
  operation revision, quantized COW branches and persisted
  `QuantizedKVCache` metadata; it is worth building only after a route
  passes the fidelity gate.

### Prefill chunk size is part of a request's answer (`prefill_chunk` receipt)

The exact path is deterministic for a **fixed** prefill chunk size and only
for a fixed prefill chunk size. Changing the chunk size changes the number of
rows `M` in every linear layer's call, MLX picks its reduction strategy from
`M`, and the resulting rounding difference propagates through the KV and GDN
state into different logits. Measured on Qwen3.8-27B at 16K: a bit-for-bit
repeat of one chunk size agrees 1.000 with KL exactly 0.0, while 4096 against
2048 agrees on only 0.891 of top-1 tokens with max |delta logit| 6.4. Neither
chunk size is "wrong"; they merely differ, at rounding scale, amplified by
near-ties into top-1 flips. See
`qualification/runs/rm15-chunk-variance-20260920/`.

`BatchGenerator` is built with `adaptive_prefill=True`, so the chunk size is
chosen per round from `adaptive_prefill_slices` (default 64/128/256/512, up to
`prefill_step`) using a **server-wide** prefill-time EWMA, whether any lane is
currently decoding, the decode-time fairness cap, and the next APC interior
checkpoint. All four depend on what other requests are doing, so the same
prompt can get a different chunk schedule — and therefore a different
answer — purely because of load.

Every request receipt therefore carries `prefill_chunk`
(`mlx2.prefill-chunk-trace.v1`): `configured_step`, `adaptive`,
`adaptive_slices`, a `widths` histogram of the chunk sizes that request's own
prefill actually ran at, `rounds`, `first`, `last`, and `varied`. `varied` is
true when the schedule changed mid-prefill for a reason other than the
prompt's own short tail chunk; those requests are reproducible only by
replaying the same chunk schedule, not by replaying the prompt. Counters:
`prefill_chunk_rounds_recorded` and `prefill_chunk_varied_requests`.

**For qualification and any A/B whose verdict rests on exact outputs, pin the
chunk size** (fixed `prefill_step`, no concurrent decode) and record it.
An unpinned comparison spends part of its margin on chunk-schedule noise.

### Host memory signals (default-off)

`execution_policy.host_memory_signals: {"enabled": true,
"fall_after_seconds": 5.0}` is server-owned (stripped before the adapter sees
the policy and recorded in qualification settings only when enabled). When
on:

- Admission headroom (`execution_headroom`, lane admission, parallel-sample
  fanout, self-MTP depth admission, interior-checkpoint budgeting) takes its
  host term from `host_statistics64(HOST_VM_INFO64)` using Splash's estimate
  `hw.memsize - (active + inactive + speculative + wired + compressor -
  file_backed - purgeable) * page`, instead of psutil (which on macOS is
  free + inactive). The two differ in both directions: the estimate counts
  file-backed and purgeable pages as available but not inactive anonymous or
  speculative pages, so it reads higher on page-cache-heavy hosts and lower
  when inactive anonymous memory dominates (measured 2.5 GiB lower on a
  128 GiB host with a model resident). The 20 GiB hard reserve, the Metal
  recommended working set and the physical-footprint floor are unchanged and
  still bound it. psutil remains the fallback when the Mach probe is
  unavailable.
- `kern.memorystatus_vm_pressure_level` (1/2/4) is polled and mapped to
  `PressureLevel` NORMAL/WARN/CRITICAL. A rise is reported immediately; a fall
  only after every reading for `fall_after_seconds` stayed lower, landing on
  the highest level seen in that window. `ServingEngine.memory_pressure_level()`
  exposes it to runtime consumers (constant NORMAL when disabled), and the
  status snapshot exports `host_memory_pressure_level` and
  `host_memory_available_bytes`.

Independently of the policy, the self-MTP free-memory probe now reads the
same free + inactive + speculative pages through `host_statistics64` rather
than spawning `/usr/bin/vm_stat` per admission decision.

### MoE expert disk streaming (default-off)

`execution_policy.moe_expert_streaming: {"enabled": true, "cache_gib": 8.0,
"read_workers": 16, "atlas": false, "atlas_path": null, "trace_path": null}`
is server-owned (stripped before the adapter sees the policy, recorded in
qualification settings only when enabled, and requiring
`feature_moe_expert_streaming` — a run that never faulted an expert was
resident and proves nothing). Default behaviour is byte-identical: with the
key absent nothing in the load or decode path changes.

When on, every quantized routed-expert projection the addressing layer can
resolve to checkpoint byte ranges leaves the parameter tree. Resolution is
direct when the live module's name matches a checkpoint tensor, and by
reassembly when the adapter *fuses* projections at load: the Qwen3-Next
family concatenates `gate_proj` and `up_proj` into `gate_up_proj`
(`MLX_QWEN4_MOE_FUSED_GATE_UP`, **default on**), and one fused expert row is
the concatenation of the two checkpoint rows for that expert, which is still
pure byte-range addressing. A transform that changes the *number* of experts
— folding the shared expert in as routed index E
(`MLX_QWEN4_MOE_SHARED_IN_GATHER`, default off) — is refused rather than
addressed, because the byte ranges would still resolve and would silently
return the wrong expert. One expert is six
contiguous byte ranges of the stacked `[E, ...]` tensors (measured: 800 KiB
per `.weight` row and 2.637 MiB per expert for Qwen3.8-Flash-Next, 768 KiB for
Qwen3.6-35B-A3B), read with `pread` out of the model's own safetensors shards
— no repack, no new on-disk format. The resident rows are remapped through
`rhs_indices` into the untouched `mx.gather_qmm`.

- **Fetch is reactive.** A miss is resolved when the router names the expert.
  There is deliberately no cross-layer speculative prefetch: adjacent-layer
  expert overlap measures at chance on three architectures and an end-to-end
  A/B gave −2.9 % decode.
- **Eviction is a plain per-layer LRU**, dict-keyed on the expert id. The
  whole of a forward pass's working set is held for the duration of that pass,
  because every index resolves before the gather runs.
- **`cache_gib` is an enforced ceiling, not an estimate.** It is subtracted
  once, before any lane is costed (`SelfMTPLaneAdmissionController`'s
  `stream_reserve_gib`), so a streamed model's admission charge is
  `R_fixed + B_stream` rather than its file size. A configuration that cannot
  hold one step's experts refuses to start with a printed budget instead of
  thrashing. Capacity is sized from the measured resident-fraction curve
  against Metal's `max_recommended_working_set_size`, not "as big as fits":
  decode throughput peaks and then collapses as the cache squeezes the kernel
  page cache.
- **Single lane only.** `max_lanes` must be 1; output would be identical with
  more, but page-in counts — the only diagnostic this feature has — would not
  be reproducible.
- **`atlas` collects, and never pins.** It accumulates decay-weighted
  per-`(layer, expert)` counts, persists them atomically
  (`weight_atlas.json` + `.bin`, index-digest bound, CRC and sentinel
  checked, ignored rather than fatal when stale or torn), and with
  `trace_path` records an access trace. Residency is unaffected in this
  iteration. `scripts/analyze_expert_atlas.py` replays the trace and reports
  the hit rate an atlas-pinned set *would* have achieved versus the LRU that
  actually ran; that counterfactual is the gate for ever building pinning.
  **Run it held out.** On the M3, replaying the trace an atlas was built from
  said pinning would win 10.23 % hit rate at `pin_fraction` 0.5. Replaying a
  *different* prompt's trace against that same atlas said it wins 0.16 % at
  0.1 and **loses** from 0.2 upward (+741 page-ins at 0.5). The first number
  was overfitting to its own trace. On this evidence pinning is not worth
  building, and that is the answer the collect-only rule existed to get.

Streamed and fully-resident execution are bit-identical, token for token, for
the same prompt: paging changes where a byte sits, never its value, and the
index remap is order preserving so a sorted gather stays sorted.
`tests/test_moe_expert_streaming.py` proves this at capacities from "holds
everything" down to "evicts on every access". **Any throughput number from a
streamed run is not a benchmark** and must not be recorded in a perf receipt
or a qualification matrix perf field.

Measured on an M3 Pro (36 GiB, 28.08 GiB advisory) with
Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-q4 (40 layers x 256 experts,
576 KiB per expert per projection, 18.12 GiB expert table): installing
streaming drops active memory from 19.68 GiB to **1.565 GiB** of
non-streamable remainder, after which the cache fills to its ceiling.
Resident (A), streamed at a ceiling holding the whole table (B) and streamed
at a 1 GiB ceiling (C, capacity 15 of 256 per layer, 13,878 evictions) all
produced a **max absolute logit delta of 0.0** over 16 greedy steps of a
248,320-entry vocabulary. Artifacts in
`qualification/runs/moe-expert-streaming-20260920/`.

Streaming is installed after the adapter loads, so it bounds steady-state
resident cost, not the load-time peak (that peak is addressed separately by
per-shard UBC eviction). It earns its place only when the expert table cannot
fit at all; on a model that already fits it costs page-ins and buys nothing.

### External-round snapshots without deepcopy (opt-in)

Every external draft/verify round (DFlash2 on Muse) first captures each lane's
committed boundary so a mid-round failure restores the lane exactly. The round
checkpoint, the prompt-boundary `target_cache`, finish `prompt_cache` and
`remove(..., return_prompt_caches=True)` copies use `copy.deepcopy`. That
never duplicated KV bytes: `mx.array.__deepcopy__` shares the immutable
buffer, so the checkpoint and the live round are already a free current/next
double buffer (a 1 GiB array deep-copies in ~20 us with no allocation; a
48-layer KV/rotating cache list in ~0.2 ms of host time).

`MLX_LM_EXTERNAL_ROUND_COW=1` switches these copies (and prompt lookup's
prompt-boundary and finish caches) to descriptor COW, the mechanism of the
native MTP committed boundary. It additionally rejects cache graphs carrying
live transaction state (`COWCacheUnsupported` falls back to the deep copy,
counted as `external_cow_fallbacks` / `pld_cow_fallbacks`) but costs roughly
1.5x the host time of the deep copy, so it stays off by default. Snapshots are
taken outside `SegmentedKVRows.begin/commit`. Tokens, RNG draws, receipts and
published caches are identical in both modes.

The round is split into phases on `ExternalDraftBatchGenerator`:
`_snapshot_round` / `_restore_round`, `_propose` (one proposal block per
cohort row, `None` for no draft), `_verify` (one `RoundDecision` per row) and
`_commit`. Alternative drafters and verifiers plug into `_propose` / `_verify`
only.

### Rolling prefill checkpoints (default-off)

`execution_policy.apc_rolling_checkpoints: {"interval_tokens": N}` (absent or
`0` = off; otherwise `N >= 16`, Splash uses 4096) takes disposable exact state checkpoints every `N` prompt tokens
(absolute multiples, so same-prefix requests share keys) and publishes each one
to APCv2 immediately under retention role `prefill_rolling`. A concurrent
same-prefix request, or a retry of a cancelled/failed request, resumes from the
newest one. A lane retires its previous rolling checkpoint when it publishes the
next, and retires the last one once its committed prompt boundary is stored;
cancel, failure and preemption keep it. A peer lease defers retirement until
the lease is released, and a peer that resumed from the same checkpoint keeps it
alive until it too moves on. Rolling entries rank below interior checkpoints and
are evicted first; a rolling publication never downgrades an existing entry.
Interior checkpoints hold their own resident-entry pool
(`max_interior_entries`); rolling entries share the ordinary `max_size` pool,
where rank -1 makes them the first entries that pool gives up.

Route behaviour: on checkpointed hybrid targets (ordinary and self-MTP) the
checkpoints are captured during prefill; chunked prefill stops exactly at each
boundary. On trimmable KV-only targets (ordinary route) nothing is captured
while prefill runs; instead a cancelled prefill that advanced at least `N`
tokens publishes its exact partial cache. External-draft, prompt-lookup and
approximate-KV routes fail closed at startup, as do other cache topologies.

All boundaries (rolling, interior, junction) come from one plan
(`runtime/state_boundaries.py`): positions are deduplicated keeping the
strongest purpose, and admission budgets them by priority junction > deepest
interior > rolling (one live rolling slot), inside the same
`apc_interior_checkpoints.headroom_fraction` cap that bounds an interior-only
plan. Interior candidates themselves come from `runtime/interior_placement`
(turn/tail/lattice placement, media floor, continuation skip) -- the P1 plan
consumes them, it does not re-derive them. Rolling captures are skipped while
the host memory pressure level is WARN or above. `skip_writing_prefix_cache`
suppresses planning. When enabled, `settings.apc_rolling_checkpoints` and a
per-request `state_boundaries` receipt (planned counts by purpose, publications)
appear; qualification requires the `feature_apc_rolling_checkpoints`
observation. Design reference: Splash rolling checkpoints (rev f58d36dd,
Apache-2.0).

### APCv2 junction snapshots (default-off)

A checkpointed hybrid (recurrent + attention) cache cannot branch inside a
stored longer path: recurrent state does not trim, so a request that shares a
prefix with a stored conversation but diverges from it can only snap back to
one of that entry's recorded prefill-chunk checkpoints (bounded by
`MLX_LM_STATE_CHECKPOINT_MAX`/`_STRIDE`), or miss with `untrimmable_branch`.
Agent traffic does this constantly: many requests share a long system/tool
prefix and diverge at the same point.

`execution_policy.apc_junction_checkpoints: true` makes admission plan one
extra exact state snapshot at the divergence. APCv2 lookups report
`branch_tokens`, the length a stored longer path shares with the prompt when
that exceeds what was restored (hits and misses). Admission passes it as the
junction to `plan_state_boundaries`; it takes budget priority over interior
lattice checkpoints. It is dropped outside `cached < junction < P - 1`,
inside a media span, and when the request sets `skip_writing_prefix_cache`. Prefill
stops at the junction and snapshots through the interior-checkpoint capture
path; the prompt-boundary publish stores it with retention role `junction`
(ranked like a default entry). The next request diverging there resumes
exactly at the junction on the ordinary and self-MTP routes (the MTP draft
state is captured with it).

Like interior checkpoints it requires a checkpointed-hybrid adapter cache and
fails closed at startup on external-draft, prompt-lookup and approximate-KV
routes; KV-only topologies trim and never need junctions. When enabled,
`settings.apc_junction_checkpoints` is recorded and the qualification feature
check `feature_apc_junction_checkpoints` is selected. Counters:
`apc_junction_checkpoints_{planned,degraded,captured,published,skipped_*}`
(Prometheus `mlx2_runtime_events_total{component="apcv2_junction"}`) and
`mlx2_prefix_cache_junction_hits_total`.

### Prefill scheduling: SRPT with a bypass cap (default-off)

Server-owned `execution_policy.prefill_scheduling` selects one prefill order
(`PrefillOrder` in `runtime/adaptive_policy.py`) for both the ordinary and
native self-MTP `BatchGenerator` routes; prompt-lookup and external-draft
routes reject it at startup.

```json
{"prefill_scheduling": {"order": "srpt", "max_bypass": 3, "one_slice_contention": true}}
```

Every key is optional inside the object; its presence enables the policy.
Absent, scheduling is exactly main's: the ordinary path admits FIFO and the
self-MTP path serves the shortest residual among multi-slice prompts, with the
omlx#3726 alternate-turn interleave for one-call prompts and the adaptive
prefill age deadline.

- **SRPT.** The request with the fewest remaining prompt tokens is served
  next (ties: more APC-cached tokens, then queue position). On the ordinary
  path SRPT picks move to the queue head before the state-budget check, so the
  budget measures exactly the admitted rows; a media row is still admitted
  alone. On the self-MTP path SRPT replaces the alternate-turn interleave: a
  one-call prompt is promoted ahead of a long prefill whenever it is shorter.
- **Bypass cap.** Each service of a later arrival counts one bypass against
  every older waiting request. A request bypassed `max_bypass` times in a row
  is served first (oldest first); its count resets whenever it is served. The
  adaptive-prefill age deadline still applies on top.
- **Ordering window (self-MTP route only).** `_next_mtp` admits a FIFO prefix
  of the prefill queue and applies the order *inside* that prefix, so a
  request accrues bypasses only while it is still a candidate. The cap
  therefore needs a window of at least `max_bypass + 1`. That window is
  `--max-lanes`, for the reasons `_segment_aware_cohort_size`'s docstring
  gives: adapters bind the cohort size to `max_lanes`, and it is also
  `completion_batch_size` directly. This is the bypass cap's specific
  consequence of that general fact.

  With the default `max_bypass` of 3 the cap needs `--max-lanes >= 4`, which
  is also the `--max-lanes` default -- so the shipped configuration sits
  exactly on the boundary, and `--max-lanes 1..3` (37 qualification receipts
  record a window of 1) is refused at startup rather than silently losing the guarantee. Measured
  on CPU with the default cap: `bypass_forced` is 0 at window 2 and 3, 2 at
  window 4, 4 at window 8, and byte-identical to the ordinary route at 16. The
  ordinary (non-MTP) route orders the whole queue via `_order_prefill_queue`
  and is unaffected. A GPU A/B of this policy must therefore run both arms at
  `--max-lanes >= max_bypass + 1`; at exactly 4 the cap fires only weakly.
- **One-slice contention.** When a prefill that needs several slices runs
  while another prompt could finish in one slice (its remaining tokens,
  clamped by its next APC interior checkpoint, fit the bounded slice), the
  long slice is bounded by `DecodeTimeFairness.stall_bound` - the same
  ~500 ms throughput grid decode fairness uses - even on an idle server or
  with decode fairness off. It never accrues or blocks on decode debt.
- **Cohorts.** While a declared `batch_cohort` member is among the candidates,
  one-call prompts are not promoted and the cohort attaches as one batch, as
  on main.

Counters: `prefill_scheduling_bypasses`, `prefill_scheduling_bypass_forced`
and `prefill_scheduling_one_slice_clamps` in scheduler stats (present only when
enabled). The qualification settings gain `prefill_scheduling` only when
selected, which requires the `feature_prefill_scheduling` observation.
### Preempt-and-replay under memory pressure (default-off)

Without it, a lane the memory controller cannot grow for 60 s fails with
HTTP 429 ("memory admission did not permit progress"). The server-owned
execution policy `"memory_preemption": {"enabled": true, "stall_seconds": 60,
"on_pressure": true}` (`src/mlx2/memory_preemption.py`) parks a lane instead
and replays it later:

- Trigger (only with at least two active lanes): a lane has made no progress
  for `stall_seconds`, or, with `on_pressure`, the host pressure level is
  CRITICAL.
  Pressure reads `ServingEngine.memory_pressure_level`, which stays NORMAL
  until host memory signals (item 9) are wired in.
- Victim: the youngest *preemptible* lane by first admission time; for a
  stall, never one older than the oldest stalled lane. The worker
  removes it, leases the APCv2 prefix of its replay prompt (the committed
  prompt boundary today; a rolling prefill checkpoint once those are
  published, via the same lookup) before releasing its old branch, and puts
  it at the head of the deferred queue. The other stalled lanes get a fresh
  stall window.
- Recovery: while a preempted job waits, or a replay is still re-prefilling,
  ordinary admission pauses and other deferred jobs' deadlines restart.
  A replay attaches when the pressure level is below CRITICAL and every lane
  it was preempted for has progressed; only one replay runs at a time and no
  further preemption happens during recovery.
- Replay: the normal admission path with prompt = original prompt + delivered
  tokens and `max_tokens` reduced by the delivered count. The detokenizer,
  output parser and stop state stay on the job, so the client stream simply
  continues; processors are rebuilt with the original prompt length.
- Preemptible: prefill-phase lanes (restart as a fresh admission with the same
  seed), and decode-phase lanes that are greedy or ordinary-route sampled (the
  lane RNG is rebuilt from the seed plus one draw per delivered token) whose
  every logits processor declares `history_pure`. Never: declared
  `batch_cohort` members, parallel samples (`n>1`, fanout), cache-capsule
  lanes, multimodal requests, approximate or Spomin-applied lanes, steered
  decode, decode under int8 prefill, and sampled speculative decode. A job is
  replayed at most twice; after that, and whenever no lane qualifies, a
  replay is already pending, or the stalled lane is alone, the stalled lanes
  get today's 429.
- Drain: a quiesce cancels a pending replay with 503 rather than waiting for
  it, and no lane is preempted while draining.
- Refused at startup together with approximate KV or live Spomin surgery.
- In `--qualification-mode` a request may inject the mechanism with
  `"mlx_fault": {"kind": "memory_preempt", "after_tokens": N}`: that lane is
  parked once it has delivered `N` tokens (`0` = during prefill) and replayed
  exactly as a stall or CRITICAL pressure level would, which is how a harness
  observes it without real memory pressure (`scripts/bench_suspend_replay.py`).

A replay is an exact continuation up to prefill-versus-decode kernel numerics
(the same caveat as any exact prefix restore); CPU tests assert greedy,
sampled and history-pure-processor replays token-identical to an
uninterrupted run (`tests/test_suspend_replay.py`).

Receipts gain `preemption` (`null`, or `{"schema":
"mlx2.memory-preemption.v1", "replays", "events"}` with trigger, phase,
committed tokens and cached replay prefix) and `settings.memory_preemption`
appears only when enabled. Counters: `memory_preemptions` (split
`memory_preemptions_stall`/`_pressure`/`_fault`), `preempted_replays`,
`memory_preemption_drain_cancellations`, plus
`memory_preemption_fault_declined` and `memory_preemption_fault_unfired`.
State: **implemented, default-off, unqualified**; an enabled policy requires
the observed-use check `feature_memory_preemption`.

An injected `memory_preempt` fault is never silently dropped. The eligibility
rule correctly declines a decode-phase lane that could not replay exactly
(`decode_replay_block`: sampled speculative route, steering, int8 prefill, or
a processor that is not `history_pure`), and a lane in prefill has
`completion_tokens == 0` so that rule does not apply -- which is why
`after_tokens: 0` fired while `after_tokens: N > 0` did nothing. The decline
is now counted, logged with its reason, and reported in the receipt as
`preemption.fault_unfired`; a fault whose threshold was never evaluated
reports `never_reached`, distinguishing an eligibility decline from a trigger
defect. The bench fails the run when any injected fault did not fire, because
an exactness comparison on a request that was never preempted passes
vacuously rather than passing.

**Scope: prefill-phase lanes only.** A lane that has delivered any token is
not preemptible, on any route. Replay rebuilds a preempted lane by looking
prompt-plus-delivered-tokens up in APCv2 and *prefilling* whatever the lookup
does not cover, but the original run produced those tokens by *decode*, and
the two paths are not bit-equal: measured on CPU, next-token logits after
eight tokens produced by decode differ from the same eight replayed as one
prefill by up to 1.6e-06 on the dense tiny model and 7.7e-07 on the hybrid
GDN/MTP one, and fused Metal prefill kernels are further from decode than
that. Greedy hides the difference until it exceeds the top-2 margin at the
resume position, and then the argmax flips and every later token diverges.
The GPU requeue saw exactly that: the ordinary arm diverged from its first
replayed token while the self-MTP arm matched, same mechanism and a
different margin -- so the self-MTP pass was luck, and blocking only the
ordinary route would have left a route that is exact by coincidence.

Prefill-phase preemption is unaffected and exact by construction: nothing
has been decoded, so there is no decode-produced state to rebuild. That is
also the case the stall watchdog exists for -- a lane starved before it
produces anything. Preemptible today: any prefill-phase lane that is not a
declared cohort member, fanout sibling, `n>1` sample, cache capsule,
multimodal prefill, or an approximate/spomin lane, and that is under the
per-job replay cap.

`decode_replay_block` now returns `decode_state_not_reconstructible` for
every decode-phase lane; its earlier per-cause reasons (sampled speculative
route, steering, int8 prefill, a processor that is not `history_pure`) were
narrower statements of the same problem and are subsumed. Re-enabling decode
replay needs a prefill path measured bit-equal to decode, not an argument
that the difference is small. P5 (`history_pure`, item 12) consequently no
longer gates preemption.

Replay exactness is asserted where it is achievable: a faulted request is
compared against a reference taken at the same batch width in the same
process (`scripts/bench_suspend_replay.py`, the `decode_replay` and
`prefill_replay` arms), and both reproduce it byte for byte. It is **not**
asserted across batch positions. Served output is not reproducible across
batch composition -- measured on this stack with the same binary and prompts,
8 of 8 prompts reproduced token-for-token at width 1 and 1 of 8 at width 16,
and two lanes running the same prompt in the same batch reported different
logprob margins for the same emitted token. The bench's two concurrent arms
therefore assert what is true at width: the victim was preempted, the peer
(which is never faulted) was not, both completed, and replays equal
preemptions with no store failures. This is a restatement, not a relaxed
tolerance: a cross-position token-identity check cannot pass honestly and was
failing for reasons unrelated to preemption.

## Flash-Next execution choices

The candidate policy selects persistent MTP2, segmented target/draft batching,
fused GDN, fused MoE paths, file-backed PLE with a 2 GiB LRU, compiled PLE,
pooled QSA and known-tail PLE prefetch. Indexed/shared QSA and asynchronous
promotion use explicit automatic gates. Shared segmented suffix state and
physical promotion are different cache representations; one cohort does not
use both simultaneously. Applicable mechanisms need observed-use checks in
qualification, not merely enabled flags.

`--execution-policy policy.json` binds overrides to the qualification identity.
Unknown policy keys fail. Optional indexed fused epilogues and source experiments
remain separate choices. Whole-decode compilation, megakernels, adaptive batch
rate/depth routing and approximate cache transformations are not default serving
features. [Flash-Next parity](FLASHNEXT-PARITY.md) records the reasons, source
comparison and mechanism-specific evidence.

The server-owned boolean keys `constrained_tool_grammar` and
`tolerant_tool_markers` are validated exactly, removed before adapter policy
parsing, and recorded in serving qualification settings.
`thinking_budget_mode` and `thinking_steer_alpha` are generation-only request
controls: they do not affect rendered prompt tokens and therefore do not split
host prompt-cache entries. The selected budget mode is present in the request
control receipt.

### GPU-resident accepted count for GDN rollback (default-off)

A partial MTP acceptance rewinds each GatedDeltaNet layer to the accepted
prefix. By default that prefix is a host integer: compact replay dispatches a
reconstruct kernel specialized on it (one compile per partial width), and the
generic path replays `q[:, :m]` once per distinct row length in a ragged batch.
Two opt-in switches take the prefix as a device count instead:

- Flash-Next: execution-policy key `"fused_gdn_dynamic_accept": true` (sets
  `MLX_QWEN4_FUSED_GDN_DYNAMIC_ACCEPT=1`; the adapter strips inherited
  `MLX_QWEN*` variables, so the policy is the supported switch). Compact replay
  then uses `qwen4_fused_gdn_reconstruct_dynamic`, which reads an `int32`
  count per row, clamps it to the tape, and rebuilds every row in one dispatch.
  The probe compiles that one kernel instead of every partial width. Selecting
  it adds the qualification check `fused_gdn_dynamic_accept`, observed through
  `fused_gdn.replay_dynamic_rollback_calls`.
- Generic GDN (Qwen3.5/3.6/3.8, and Qwen4 when the fused path declines):
  `MLX_LM_GDN_ARRAY_ACCEPT=1` stages a per-row rollback that replays the full
  verify width with `a = b = -inf` past each row's count. That makes the decay
  exactly 1 and beta exactly 0, so rejected steps leave the state unchanged and
  the result equals the sliced replay bit for bit. Ragged trims become one graph
  per layer. Every serving adapter strips inherited `MLX_LM_*` variables and
  none maps a policy key to this one yet, so today it applies only to direct
  runtime use (lab scripts, tests).

Both keep `fn(m: int)` for `ExactRollbackBoundary` and fan-out. Neither changes
when the host reads the acceptance: a greedy self-MTP round still has exactly
one `hybrid.*` verify sync. The larger gain, enqueuing the next verify before
that read, is not implemented yet. `scripts/bench_gpu_accept_count.py --device
gpu` checks the dynamic kernel against the template kernel bit for bit and runs
the A/B timings.

## Candidate startup and qualification

The 2026-09-16 source closeout is frozen at runtime hash
`ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`.
Its CPU suite passes, but the latest repaired 262K Flash-Next run was interrupted
and Qwen3.6 receipts bind earlier candidate sources. No new complete serving
profile or five-round benchmark matches this source. The commands below describe
a future exclusively owned qualification window, not a running or qualified
deployment.

Run only one full-model service at a time in the owned GPU window. The following
is the candidate configuration under evaluation; specifying its context ceiling
does not establish successful qualification at that limit:

```bash
MLX2_MODEL_PATH="$HOME/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
.venv/bin/python -m mlx2.server \
  --model "$MLX2_MODEL_PATH" --port 8285 \
  --max-context 262144 --max-lanes 4 --max-inflight 8 \
  --cache-bytes 17179869184 \
  --cache-dir "$HOME/Library/Caches/mlx2/candidate-apcv2" \
  --qualification-mode
```

Normal serving requires `--qualification /path/to/matching.json` in place of
candidate mode, with exactly the qualified settings. The gate checks source,
native runtime, artifact, policy and required successful checks before selecting
a route. With this Flash-Next artifact no flag selects its adapter-default native
MTP route; `--ordinary` selects a separate reference profile with its own receipt.
`--external-draft` requires a draft-model policy and is mutually exclusive with
`--ordinary`; ordinary evidence cannot qualify the external route.

```bash
.venv/bin/python scripts/qualify_serving.py --url http://127.0.0.1:8285 \
  --output qualification/flash-next-mtp2.json
.venv/bin/python scripts/benchmark_serving.py --url http://127.0.0.1:8285 \
  --rounds 5 --widths 1 2 4 --max-tokens 160 \
  --output qualification/benchmark-mtp2.json
```

Restart the matching ordinary candidate and write distinct ordinary reports for
a comparison. The qualifier exercises HTTP cold/warm output, streaming, tools,
reasoning, Hermes options, transformed sampling, logprobs, batching, mixed warm
prefixes, near-limit context/reuse, over-limit rejection, cancellation, recovery,
strict JSON-schema enforcement on grammar-capable models, state tests and leaked
leases. Applicable advanced mechanisms must be observed.

The five-round benchmark checks warm hits and that each requested B1/B2/B4 width
appeared in execution receipts. It retains all observed widths, request data and
output hashes and rejects identity changes. Sequential profile runs measure
this warmed HTTP workload, not universal quality, thermally controlled kernel
speed or per-feature attribution.

For a tensor-free official-client compatibility check, use a separate Python
environment that already contains both SDKs:

```bash
PYTHONPATH=src .venv/bin/python scripts/sdk_smoke.py \
  --sdk-python /path/to/sdk-venv/bin/python
```

The parent pins MLX to CPU before importing mlx2, starts a scripted server on a
random loopback port, and runs the client in the supplied interpreter. With no
`--sdk-python`, it prints a clear skip. Set `MLX2_SDK_PYTHON` to opt the pytest
wrapper into the same check; normal CI remains skipped.

Keep these states separate:

- **Implemented:** the mechanism and dependencies exist.
- **Qualified:** evidence matches the artifact/runtime/settings and required checks.
- **Selected:** a capability route was chosen from that evidence.
- **Observed used:** receipts/counters demonstrate execution in the workload.

### External drafters for North Mini Code and Laguna XS 2.1 (candidate)

Two more drafter families run on the same external draft/verify executor
(exact speculative sampling law, segmented target transactions, recovery
checkpoints, paired APCv2 draft sidecars). Both are implemented and
CPU-verified only; neither route is qualified or selected.

| Target | Drafter | Execution policy | Default depth | Receipt `kind` |
|---|---|---|---|---|
| North Mini Code 1.0 | `CohereLabs/North-Mini-Code-1.0-eagle` (EAGLE-1 Cohere head, chain) | `{"draft_model": <snapshot>, "num_draft": 1-7}` | 3 | `external_cohere_eagle` |
| Laguna XS 2.1 | `poolside/Laguna-XS-2.1-DFlash` (causal DFlash block) | `{"draft_model": <snapshot>, "num_draft": 1-15}` | 7 | `external_laguna_dflash` |

The North head fuses the embedding of the token *after* each target feature
with the target's final-norm hidden state, so the executor passes those
`context_tokens` to drafters that declare `requires_context_tokens`
(counter `external_context_token_pairings`); DFlash-family drafters see the
original calls. Chain positions never enter the draft KV cache. The Laguna
drafter is a separate class from DFlash2: its block is causal, has no
candidate selector, and normalizes each target tap.

**GPU measurement (2026-09-19): both families are a no-go and stay
default-off.** `scripts/bench_external_draft_acceptance.py` (Metal-only,
refuses without `--i-own-the-gpu`; `--merge` pools split runs) over three
interleaved repeats: North best 0.67x at B=1 (K=1, acceptance length 1.60),
Laguna best 1.08x at B=1 (K=3, acceptance 2.66), and every B=4 arm 0.29-0.60x
because the ordinary route batches four lanes into one forward while the
external route verifies `4 x (K+1)` rows. No arm was refused and no draft
fallback fired, so this is an economics result, not a defect: on these sparse
MoE targets the verify block at `M=K+1` costs more than the acceptance length
returns (North M=4 is 12.1 ms against an 8.1 ms M=1, and the EAGLE-1 chain
adds 7.4 ms per round for three sequential 262k-vocab head forwards).
Evidence: `qualification/runs/rm06-north-20260919/`,
`qualification/runs/rm06-laguna-20260919/`,
`qualification/runs/rm06-diagnostics-20260919/`. The North arm was re-run on the
corrected normalization (2026-09-20): best B1 0.630x, best B4 0.494x, acceptance
length 1.633/1.920/2.064/2.070 at K=1..4 — the no-go is unchanged and the wrong
norm was **not** what capped acceptance
(`qualification/runs/rm06b-north-norm-20260920/accept-greedy-r0-r2.json`). Read
the North numbers above as the superseded pre-fix measurement. Serving-level
behaviour is
healthy: `scripts/smoke_external_route_serving.py --family north` drives the
real artifact and drafter through the server (streaming, warm APCv2 reuse,
thinking, steering, B=4) with correct receipts and zero fallbacks.

`MLX2_EXTERNAL_ROUND_TIMING=1` reports per-phase host milliseconds for one
external round (draft, verify forward, laws, transaction, emit). It is off by
default; the bounded mechanism counters are not.

The segmented KV transaction that backs every verify block (self-MTP, prompt
lookup, external draft) verifies lane membership and cache stamps when a
round opens and when it commits or aborts, not on every per-layer view call:
the per-layer scan cost about 8 ms of Python per round on a 49-layer target.
The per-layer path still refuses a closed, stale or re-owned transaction, an
outside mutation is still caught before anything is published, and the
`integrity_scans` counter on the row owner makes a round that never verified
visible.

Qwen3.8 27B, Qwen3.6 35B-A3B and Muse Glimmer/DFlash2 remain candidates without
production GPU qualification or deployment. Their pre-GPU reports are required
reading before that work: [Qwen3.8 27B](ports/QWEN38-27B.md),
[Qwen3.6 35B-A3B](ports/QWEN36-35B-A3B.md),
[Muse](ports/MUSE-GLIMMER.md), [DFlash2](ports/MUSE-DFLASH2.md).

## Host-local service configuration

Deployment definitions and service-restoration receipts are host-specific and
intentionally kept outside this repository. Discover the configured model ID
through `/v1/models`, verify `/health` and `/v1/status`, and qualify every
source, artifact or settings change before selecting it for service.

## HTTP security

### Host allowlist, Origin check and optional API key

Every request passes one gate before routing (`src/mlx2/http_security.py`,
adapted from Splash):

1. **Host allowlist.** When the server binds a loopback address (the default
   `--host 127.0.0.1`, or `localhost`/`::1`), the `Host` header must name
   `localhost`, `127.0.0.1`, `[::1]`, the `--host` value, or an
   `--allowed-host NAME` (repeatable), with any port. Otherwise the response
   is `421`. This blocks DNS rebinding: a browser page on an attacker domain
   that resolves to 127.0.0.1 still sends `Host: attacker.example`. A
   non-loopback bind (`0.0.0.0`, a LAN address) checks `Host` only when at
   least one `--allowed-host` is given; without one the server logs a warning
   and skips the check. With the allowlist active, the literal address of the
   socket the client reached is also accepted, so dialling the LAN IP works.
2. **Origin == Host.** A request carrying an `Origin` header is refused with
   `403` unless its scheme is http(s) and its host and port equal the `Host`
   header's (default ports by scheme). SDKs, curl and CLI agents send no
   `Origin` and are unaffected. This applies on every bind.
3. **API key (off by default).** `--api-key-file PATH` (owner-only 0600,
   one line, same rules as `--admin-token-file`) or `--api-key-env NAME`
   requires the key on every route except `GET /health`, as either
   `Authorization: Bearer <key>` or `x-api-key: <key>`. Keys are visible
   ASCII; comparison is constant-time. Failures are `401` with
   `WWW-Authenticate: Bearer`; Anthropic-route (`/v1/messages*`) errors use
   the Anthropic error shape. `/metrics` requires the key when one is set.
   The server does not read `MLX2_API_KEY` implicitly, so an ambient export
   cannot turn on authentication for clients that send no key; pass
   `--api-key-env MLX2_API_KEY` to opt in. Client launchers use the same
   loader, `mlx2.http_security.load_api_key(..., required=False)`.

The admin surface keeps its loopback-client rule and, when configured, its
`--admin-token-file` bearer token; with an admin token the API key is not
also required there (both use `Authorization: Bearer`). Without an admin
token, admin routes require the API key like any other route. Host and
Origin checks apply to admin and `/health` too.

A refused request is answered before its body is read and the connection is
closed, so a body is never parsed as a pipelined request. Refusals appear in
the existing per-route HTTP status metrics. The gate is transport policy and
is not part of qualification identity. Embedders calling
`handler_for(engine)` directly get no gate unless they pass
`http_security=policy_for_bind(...)`.
## Persistent signatures and agent launcher

### Reasoning signing key under `--api-state-dir`

Returned thinking signatures and Responses reasoning tokens are HMACs under the
server's reasoning signing key. Without a configured key that key is random per
process, so with `--api-state-dir` the stored Responses survived a restart
but every signature issued before it failed verification, and the verified
reasoning was dropped from the next turn.

When `--api-state-dir` is set and neither `--reasoning-signing-key-file` nor
`--reasoning-signing-key-env` is given, the server now uses
`<state>/reasoning-signing.key`, creating it on first start (32 random bytes,
hex; mode 0600; state directory 0700 if created). The file is published with
an exclusive link, so two servers starting on one state directory agree on one
key. An existing file must be a regular file owned by the server's uid with
mode 0400 or 0600, not a symlink; otherwise startup fails. `/v1/status`
`reasoning_signing.ephemeral` reports `false` and `key_id` is stable
across restarts. `--reasoning-signing-ephemeral` keeps the old per-process key.
Without `--api-state-dir` nothing changes.

Tokens remain signed, not encrypted: the Responses token still carries the
reasoning text as base64. Encryption would need `cryptography`, which is not a
dependency.

### `mlx2 claude|codex|opencode`

The `mlx2` console script starts an installed coding agent against a running
server. It never writes the agent's global configuration: settings go to that
one process through environment variables and command-line overrides, then the
launcher `exec`s the agent.

```bash
mlx2-serve --model ... &          # default http://127.0.0.1:8285
mlx2 claude                        # or: mlx2 codex, mlx2 opencode
mlx2 --url http://host:8285 --model <id> codex exec "fix the tests"
```

Launcher options go before the agent name; everything after it is passed to
the agent unchanged. `--url` defaults to `MLX2_URL`, then
`http://127.0.0.1:8285`. `--model` defaults to the model `/v1/models`
reports; the context limit comes from `/v1/status` `max_context`. The API key
is read from `--api-key-file` (owner-only 0600 file) or `MLX2_API_KEY`, and
defaults to the placeholder `local`, because each agent requires a key even
when the server checks none. The key is sent as a bearer token on the discovery
requests.

| Agent | Endpoint | What is set |
|---|---|---|
| Claude Code | `/v1/messages` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL`, `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW` (= `max_context`); removes `ANTHROPIC_API_KEY`; sets the cloud-provider switches (`CLAUDE_CODE_USE_BEDROCK`, `_VERTEX`, `_FOUNDRY`, `_ANTHROPIC_AWS`, `_ANTHROPIC_GOOGLE_CLOUD`, `_GATEWAY`, `_MANTLE`) to `0`; argv `--disallowedTools WebSearch --model <id> --permission-mode default` |
| Codex | `/v1/responses` | `-c` overrides: `model`, `model_provider="mlx2"`, `model_providers.mlx2={base_url, env_key="MLX2_API_KEY", wire_api="responses"}`, `web_search="disabled"`, `model_context_window`, `model_auto_compact_token_limit` (90%); user `-c` overrides follow and win |
| OpenCode | `/v1/chat/completions` | `OPENCODE_CONFIG_CONTENT`: merged into any existing value; an `mlx2` provider (`@ai-sdk/openai-compatible`), `model`/`small_model` and the built-in agents' models, `none`…`xhigh` reasoning variants, image input only when the model reports `vision`, context/input/output limits (output = min(32768, context/4)) |

The variable names were checked against the installed Claude Code 2.1.269,
Codex 0.145.0 and OpenCode 1.18.5. Hosted web search is disabled for Claude
Code and Codex because mlx2 does not provide it; local tools and MCP are
unchanged. Adapted from Splash `install/clients.py` (Apache-2.0).

## Tool-call grammar during generation

### Auto alternation, quantifiers, JSON answers and streaming (default-off)

Two server-owned execution-policy booleans extend `constrained_tool_grammar`
(both require it, both default false, both are stripped before adapter policy
parsing and appear in qualification settings only when true):

- `constrained_tool_grammar_auto:true` composes the adapter's `tool_constraint`
  block into a whole-output grammar (`src/mlx2/tool_grammar.py`):

  | Request | Language | Calls |
  |---|---|---|
  | `required` / named | the adapter block | `+` (parallel) or exactly one |
  | `auto` with a `strict` tool | free text, optionally one block and a text tail | `*` (parallel) or `?` |
  | any choice with a JSON `response_format` | the block **or** the JSON answer | as above; forced choices admit calls only |

  Free text is any text that never contains the adapter's
  `tool_call_open_marker` (Qwen/Flash-Next `<tool_call>`, North
  `<|START_ACTION|>`, Muse `<atem:function_calls>`), written without lookaround
  so the exact token automaton compiles it. Non-strict `auto` without a JSON
  answer is untouched (`decode_grammar:"disabled"`). Thinking is not folded
  into the grammar: the processor still defers past the thinking-close marker,
  so `thinking_budget` and the thinking guard compose unchanged. Covered:
  Flash-Next and its Qwen3.8 27B / Qwen3.6 35B-A3B subclasses (Qwen XML),
  North, Muse. Adapters without `tool_constraint` (Xing, mlx-vlm Gemma 3n /
  MiniCPM-o, Laguna) or, for
  `auto` text, without an opener skip with `skipped_adapter_unsupported`;
  strict schemas a builder cannot represent skip with
  `skipped_grammar_unrepresentable`; routes without the grammar capability
  (or thinking without a marker) skip with `skipped_route_unsupported`. A
  `grammar` or `min_tokens` still skips with `skipped_request_combination`.
  Every skip keeps main's terminal `enforce_tool_contract`. Receipts add
  `request_controls.tool_choice.grammar = {shape, calls}` when engaged.
- `tool_grammar_streaming:true`: when admission engaged the tool grammar, a
  streamed Chat request with strict/required/named/single-call tools is no
  longer buffered — each tool call is sent as its block completes, and the
  terminal tool contract runs before the final chunk (a violation, or a
  mid-generation grammar failure, becomes an SSE `error` event as for
  structured output). Responses streams open each `function_call` item
  (`output_item.added`, `function_call_arguments.delta`/`.done`) on arrival
  and send its `output_item.done` after the contract; the final payload keeps
  the streamed order. Anthropic streams already emit `input_json_delta` per
  completed call. Granularity is one delta per completed call, not per
  argument token; buffered Chat streams (grammar not engaged) are unchanged.

Counters: `decode_grammar_auto_engaged` and `decode_grammar_streamed` under
`mlx2_runtime_events_total{component="tool_calls"}`. Selected policies demand
`feature_tool_grammar_auto` / `feature_tool_grammar_streaming` qualification
checks.

Processor properties (P5): logits processors may declare `history_pure` (the
mask is rebuildable from token history) and `dormant(tokens)` (the next row
passes logits through, e.g. a grammar deferred before the thinking-close
marker). Structured output, thinking budget/guard, min-tokens, penalties/bias
and the North/Muse tool processors declare them; consumers read them with
`getattr(p, ..., False)`.
### DFlash2 batched pairwise selection

Default off. The Muse external-draft execution policy key
`pairwise_selection` accepts `"host"` (default, the sequential selector) or
`"batched"`. Batched selection computes the DFlash2 candidates, unary scores
and the full `[B, K, C, C]` predecessor/successor edge table in one pass, then
samples the whole block with a vectorized walk over pre-drawn uniforms and
reads it back once (`verify_sync` site `external.draft.block_eval`, once per
draft group). Uniforms are drawn from each lane's `RequestRNG` in position
order, so tokens, proposal laws (within float32 rounding) and RNG state match
the host path; verification is unchanged and remains exact. Rows with logits
processors (grammar, tool recipients) keep the sequential host path; a draft
group that contains any processor row stays on it entirely. When selected,
`settings.execution_policy.pairwise_selection` is `"batched"`, qualification
requires `feature_external_pairwise_selection`, and the scheduler reports
`external_pairwise_selection_groups`/`_lanes`.
## Longer DFlash2 blocks

### DFlash2 block width (`num_draft`)

The external DFlash2 route proposes `num_draft` tokens per round and verifies
them with the anchor in one target forward of width `num_draft + 1`. The Muse
execution policy (`qualification/policies/muse-dflash2.json`) accepts any
integer `1 <= num_draft < block_size`, where `block_size` is read from the
drafter artifact's `dflash_config`; the executor re-checks the same bound.
The shipped `Muse-Glimmer-30B-DFlash2` drafter was trained with
`block_size: 16`, so counts up to 15 are legal (its model card evaluates 15).
The policy default stays `num_draft: 4` until the GPU sweep selects another.
Splash's own Qwen DFlash2 packages are trained at block 8 (7 proposals); that
geometry does not transfer to this drafter.

Width is charged where it is spent: every decode round reserves
`num_draft + 1` rows per lane before any drafting or RNG draw
(`ExternalDraftBatchGenerator._admit`), and a round that does not fit defers
or runs a narrower cohort without mutating lane state. Request admission
itself charges ordinary rows only, so a long block degrades batch width under
pressure rather than failing a round. Draft context stays exact at any width:
each round appends up to `num_draft + 1` committed target rows, and sliding
draft layers keep exactly the last `sliding_window - 1` of them.

`scripts/bench_external_block_size.py` sweeps `num_draft` at batch 1 and 4
(greedy and sampled) against an ordinary-target reference and reports decode
tok/s, acceptance length (tokens per verify round), accept rate, the
per-round accepted histogram, and greedy token exactness.
