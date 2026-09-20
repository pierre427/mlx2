# North Mini Code 1.0 port

## 2026-09-20 correction: the port ran the wrong normalization

Until 2026-09-20 this port built North's per-layer `input_layernorm` and its
final `model.norm` as `nn.LayerNorm(hidden, eps=layer_norm_eps=1e-5)` -- the
mean-centred Cohere norm. The reference is `RMSNorm` with `eps=1e-6`:
HF transformers `models/cohere2_moe/modeling_cohere2_moe.py` selects the norm
*class* from `rms_norm_eps` (decoder layer and final norm), and this
checkpoint's `config.json` sets `rms_norm_eps: 1e-06`. The `layer_norm_eps:
1e-05` beside it is only `Cohere2MoeConfig`'s own default being serialized and
is inert here. mlx-vlm's independent `cohere2_moe` port agrees.

The selector never reached the model: `BaseModelArgs.from_dict` drops keys the
dataclass does not declare, and `rms_norm_eps` was not declared. The weights
could not catch it either -- both norms have identical parameter shapes (one
`[hidden]` gamma, no bias), and the checkpoint carries no `*.bias` tensor at
all, so the wrong norm loaded perfectly clean.

**Consequently every North measurement taken before 2026-09-20 was taken on a
mis-normalized model**, including this document's serving numbers, the
integration and 20x20 sanity campaigns, the alpha-steering calibration, and
rm06's speculation study. Greedy outputs changed with the fix; that is a
deliberate serving-behaviour change for North.

The shipped alpha-steering commit direction was calibrated in the old residual
geometry and stopped binding (`thinking_calibration.SCHEMA` is now
`mlx2.commit-direction.v2`). It was **recalibrated on the corrected body on
2026-09-20** and re-shipped at layer 32 (was 28), passing the held-out and
random-control gates; see `qualification/runs/north-requal-20260920/`.

The list of every pre-2026-09-20 North record this voids, with per-record
status, is `qualification/runs/north-requal-20260920/INVENTORY.md`. Two things
it surfaced: North had **no passing serving qualification record even before the
fix** (`integration-gpu-20260918`'s north-ordinary run and its first-pass twin
both failed `shared_cohort_priming`), and the three records that did pass were
produced by a qualifier hash that `src/mlx2/qualification.py` no longer pins.

See `wiki/docs/experiments/mlx2-rm06b-north-norm-2026-09-20.md`,
`wiki/docs/experiments/mlx2-north-requalification-2026-09-20.md` and
`tests/test_north_norm_choice.py`.

## Current state

North Mini Code has a CPU-validated ordinary serving slice in mlx2. It is
implemented and registry-selectable, but remains **unqualified and unselected**
until the real q4 artifact passes strict loading and the serving qualification
suites on the current runtime.

The initial target is
`~/mlx-models/North-Mini-Code-1.0-mlx-4bit`, fingerprint
`6c88a4d3a5387abe2d97ac3d5eb70e484fb33615bb0c0f2ff8e7ef22b442f2e4`.
Bounded header inspection reconciled 1,226 indexed tensor records across four
shards, including exact names, shard mappings, shapes, dtypes, byte ranges,
and q8 router overrides without reading tensor payloads. The
q8 reference is also present, with fingerprint
`1a533fdcdc118be2426fa90f725e82b031c94f95aae97f8e25c511b31580c3c5`.
No production tensor payload was loaded during this port.

## Architecture and current mechanisms

- `cohere2_moe`, 49 parallel decoder blocks, hidden size 2,048.
- Layer 0 is dense. Layers 1 through 48 use 128 sigmoid-routed experts with
  top-8 selection and no shared expert.
- Thirteen layers use global NoPE attention. Thirty-six layers use traditional
  RoPE and a 4,096-token sliding window.
- The ordinary cache contains 13 unbounded `KVCache` layers and 36 bounded
  `RotatingKVCache` layers. The declared layout is
  `north-mini-code-layer-segments-v1`.
- Admission uses an architecture-derived fp32 upper bound. At 500,000 tokens
  it charges about 27.6 GiB per lane: 13 growing global caches, 36 capped 4K
  caches, and four exact restore snapshots for each sliding cache. The shared
  20 GiB reserve remains intact. A 3.1 GiB per-lane workspace assumption is
  deliberately conservative and remains unqualified until live measurement.
- The shared worker supplies continuous batching, scheduling, cancellation,
  memory admission, streaming, APCv2 prefix reuse, persistence, route receipts,
  and request-scoped sampling. No legacy APC or model-specific scheduler branch
  was added.
- North reasoning, text, and JSON action blocks are parsed incrementally across
  chunk boundaries. Tool history is normalized without mutating the request.
  Image content fails closed.
- Required and named tool requests constrain North's trained action branch at
  `<|START_ACTION|>`. The request-scoped processor is a pure function of token
  history, activates after `<|END_THINKING|>` when reasoning is enabled, and is
  safe for prompt-lookup/speculative probes. Function-name identity remains an
  authoritative post-generation check because the action JSON permits flexible
  key order and an optional `tool_call_id`.
- An action block truncated by the output-token budget now terminates as
  `length` / Anthropic `max_tokens` while preserving prior parsed output;
  malformed action blocks ending by EOS or stop continue to fail closed.
- The candidate profile is `north-mini-code-apcv2-ordinary`. Native MTP and
  external-draft capabilities are absent. With no route flag the adapter
  selects ordinary; explicit `--ordinary` is equivalent, while `--native-mtp`
  fails before listener binding or model allocation.

## Speculation and optimization gaps

The target artifact contains no MTP, EAGLE, `nextn`, or draft tensors. The
older locally trained North MTP head is intentionally excluded: historical
live acceptance was below break-even and the route regressed throughput.

An official separate `CohereLabs/North-Mini-Code-1.0-eagle` snapshot is present
locally at revision `8c7fcb575f107e9968b61cc93a756e6fc2c86713`. Metadata describes
39 tensors, 79,194,624 parameters, and three dense sliding-attention layers.
It is a serious follow-up candidate, but needs a first-class revision-bound
EAGLE contract:

1. EAGLE consumes target hidden features and cannot be labeled as native
   embedded MTP or a token-only external draft.
2. Its three rotating caches must compose with per-lane segmented rollback,
   APCv2 sidecars, cancellation, batching, and disk restore.
3. The adapter must bind target, tokenizer, draft revision, and settings into
   one route identity and emit nonzero mechanism counters.
4. Greedy equivalence, sampled residual correctness, acceptance, memory, and
   net throughput need live qualification. Historical teacher-forced results
   are orientation, not mlx2 qualification.

Prompt lookup decoding was historically strong on copy-heavy North code edits.
The current server does not expose a qualified ordinary-plus-PLD route, so this
port does not claim it. Adding one requires a shared batch lifecycle and receipt
counters rather than an adapter-local generation loop.

Known exclusions are explicit: runtime Q/K/V concatenation was slower on North
q4, global-layer low-bit KV was slower, and dynamic expert-k was rejected for
this sigmoid router. Router weights remain q8 in the local q4 artifact.

## Qualification contract

The first GPU window should use the q4 target, `--max-context 500000`, and a
distinct APCv2 directory. Omitting a route flag selects North's ordinary
default; retaining `--ordinary` documents the same resolved route explicitly.

The HTTP request envelope is bounded independently from token admission. By
default, `max_request_bytes` is derived from `max_context` at 32 bytes per
context token plus 64 KiB of JSON overhead, with a 2 MiB floor and 32 MiB hard
ceiling. The resolved value is exposed in the `/v1/status` `http` block and is
kept separate from the model route settings. `--max-request-bytes` may select a
stricter positive limit but cannot exceed the global ceiling. Oversized bodies
are rejected with HTTP 413 before the server reads them.

Qualification begins with strict payload loading and short coherent
generation, then verifies:

- exact profile and candidate receipt on every response;
- APCv2 cold store and warm hit with positive `cached_tokens`;
- 13 global plus 36 rotating cache layers with no lease leak;
- ordinary compute width, B1/B2/B4 and width-20 scheduling;
- cancellation and disconnected-client cleanup;
- reasoning-off and reasoning-on channel separation;
- declared tool action round trip and stop behavior;
- stable runtime, artifact, settings, profile, and source identity.

Proposed thermally controlled context cells are 2,048, 8,192, 32,768, 65,536,
131,072, 262,144, and 499,936 prompt tokens with 64 output tokens, three runs
per cell. An ordinary-only run establishes capacity but cannot substitute for
a two-arm comparison.

The separate no-thermal batch work includes the width-20 stress suite, requiring
actual observed compute width 20, and the frozen Spomin suite: 20 domains by 20
cases, paired full and compacted exact rebuilds, HTTP batch size 20. A capacity
failure is a failed cell; the harness must not shorten prompts silently.

## CPU evidence

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_north_mini_code_port.py
tests/test_adapter_registry.py -q` passed 44 tests. A tiny explicit-CPU model
verified full versus split replay, cache offsets, and mixed global/rotating cache
construction. With identical weights, the port matched the mined model bit for
bit for ordinary full and split replay. Header-only inspection reconciled all
1,226 q4 and q8 index entries and confirmed 48 q8 router overrides in q4 while
expert and attention projections remain q4. Static checks cover artifact
identity, strict shard headers, topology and absent-MTP refusal, request
normalization, and streaming
channel/tool parsing.

This evidence validates the implementation boundary only. It does not qualify
the production artifact, GPU math, throughput, long context, APCv2 restore, or
batch behavior.
