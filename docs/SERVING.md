# Serving mlx2

This guide describes the modernized working source. Implemented behavior is not
qualified or deployed merely because it is present in the tree.

Aggregate operational telemetry is available as Prometheus text at
`GET /metrics`; detailed receipts remain on the JSON status endpoints. See
[Metrics and telemetry](METRICS.md) for names, units, label policy and the
CPU-only qualification boundary.

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

## Request API and Hermes controls

`POST /v1/chat/completions` accepts text system/user/assistant/tool messages.
Function calls use OpenAI-shaped tool definitions and JSON-string arguments;
`tool_choice` supports `auto`, `none`, `required`, and a named function.
`parallel_tool_calls:false` and the bounded strict JSON-schema subset are
checked before a successful response is returned. Clients execute tools and
return their results. A valid request whose generated calls violate that
contract fails with HTTP 502; malformed client contracts fail with 400. Strict,
required, named and single-call Chat streams are buffered through terminal
validation before the first response byte. Reasoning appears in
`reasoning_content` separately from answer text.
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
| `logprobs`, `top_logprobs` | Emitted-token probability and up to 11 sorted alternatives, nonstreaming or SSE |
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

### Output-default admission projection

The following CPU-only calculation uses the production
`FlashNextCacheBudget.project` geometry recorded for the Flash-Next artifact
(13 QSA planes, 36 recurrent planes) and
`SelfMTPLaneAdmissionController.lane_gib` at native MTP depth 2. On the
controller's 128 GiB operating point, the 72.5 GiB resident model leaves
55.5 GiB free; the unchanged 20 GiB service/driver reserve leaves 35.5 GiB
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
North-Mini-Code only, GPU-verified on North-Mini-Code
(`qualification/runs/north-thinking-guard-20260918/`).

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
(`adapters/assets/north_mini_code_commit_direction.npz`, layer 28). Enable with
`--thinking-steer-alpha A` (request `thinking_steer_alpha`, 0–1) and optionally
`--thinking-steer-hammer H` for a stronger pull after the alarm trips. It
batches (one forward, a zero row for unsteered lanes) and also steers
prompt-lookup verify forwards; native MTP and external-draft routes keep the
logit-level guard only. Steered lanes are not stored as exact APCv2 prefixes
(their K/V was written under steering); the prompt boundary still is.
**A direction is bound to the artifact it was measured on.** Each stored
direction names an `artifact_identity` (config, tokenizer, chat template and
sampled weight bytes; host-independent) and is only ever loaded for that
identity, because a direction from another build is a same-norm wrong vector
and the campaign measured those as worse than no steering. When steering is
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
failures. Calibration, held-out grid and random controls:
`qualification/runs/north-alpha-calibration-20260918/`; recalibrate with
`scripts/calibrate_thinking_direction.py` whenever weights, quantization or the
chat template change. The adapter file carries the operating guidance.

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

## Cache, scheduling and memory

**APCv2 is the sole prefix cache**, including the ordinary reference route.
Its radix index is an internal data structure. Layered frozen segments and
copy-on-write branches preserve ownership. Target, recurrent, sparse/QSA and
draft state remain bound to their model/runtime/layout revision. Prompt-boundary
and completed-generation snapshots publish only committed state.

The prefix cache is one namespace for every client by default, so a client's
`cached_tokens` reflects any earlier client's prompt; `--tenant-scoped-cache`
binds the APCv2 namespace to `X-Tenant-ID` (which is not authenticated) and is
part of the qualification settings. Persistent-block snapshots carry a
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
for one request. Admission performs one final allocator reclaim before
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

Admission uses host availability, Darwin process footprint and MLX allocator
accounting. It preserves a **20 GiB service/driver reserve**. Flash-Next supplies
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
observed-used outside CPU tests.

KV quantization is the first operation published through the approximate-state
seam (`runtime/approximate_state.py`, `runtime/approximate_kv.py`). It exists on
the **ordinary route only**.

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
  `evidence`; unknown keys fail. It requires `--qualification-mode`, or a
  `--qualification` record whose settings match and whose
  `feature_approximate_kv` check passed. It is refused with native MTP, prompt
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
a route. `--ordinary` selects a separate reference profile with its own receipt.
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
