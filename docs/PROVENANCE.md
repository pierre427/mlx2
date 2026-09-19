# Provenance

## Laguna XS 2.1

The Laguna sparse-MoE tensor model and Poolside XML-like tool parser were
mined from the local read-only `mlx-lm-unified` revision recorded in
[`laguna-xs21.json`](../provenance/laguna-xs21.json). The retained MIT notice
is [`laguna-xs21.NOTICE`](../provenance/laguna-xs21.NOTICE). mlx2 adds strict
artifact topology/shard inspection, APCv2 full/sliding cache binding, ordinary
decode, continuous batching, prompt lookup, grammar, reasoning-channel and
tool-call integration. The locally cached causal Laguna DFlash artifact uses a
different draft contract from mlx2's qualified DFlash2 closure, so speculation
remains fail-closed and unselected rather than being represented as delivered.
The dedicated Laguna q8 expert-down/top-8 reduction and sigmoid/correction-bias
router Metal kernels are original mlx2 work. They preserve the ordinary MLX
graph as the reference route and remain candidate-gated until exact-device
numerical and performance qualification records engagement for each arm.

This file is append-only. Every implementation mined from another tree must
record its origin and the validation performed after adaptation.

## 2026-09-15 — clean-generation seed

- Source reference: local `mlx-lm-unified`, revision
  `1e2bc604f71d070bee970c3e7db8b60f7855599b`.
- Newly written here: capability contracts, routing, state transactions, cache
  ownership, bounded telemetry, execution protocols, and model descriptors.
- Design inputs inspected, without copying their implementations:
  `cache_planes.py`, `batch_runtime.py`, `server.py`, `generate.py`,
  `tool_protocol.py`, `grammar_decode.py`, and the Qwen model definitions.
- Evidence synthesis: `mlx-uag` commit
  `edef943ab1b6aa81362c4cf4ba44c70dfd2cab04`.
- License: Apache-2.0.
- Qualification: CPU contract tests only; no model execution claim.

Future entries must include source revision, source paths, license, substantive
modifications, tests or benchmarks, and the capability profile affected.

## 2026-09-18 — API execution, durable state and multimodal seams

- LoRA source: local `mlx-lm-unified` revision
  `13eb83388750435bcc751d0b9e33857a38152544`, paths
  `mlx_lm/tuner/lora.py` and `mlx_lm/tuner/utils.py`, repository license MIT.
  Adaptation: explicit `Linear`/`QuantizedLinear` keys only, zero dropout, exact
  tensor coverage, one active reversible session, inflight drain, model
  revision increment, and APCv2/host/compile-cache invalidation. DoRA, implicit
  layer selection, SwitchLinear and full-model tuning are excluded.
- Image/audio source: local `mlx-vlm-qwen4-exp` revision
  `653f1f13e238abb313fd45071bbd04b3de414635`, path `mlx_vlm/utils.py`, MIT.
  Video/control source: local `Rapid-MLX` revision
  `ee101f1bcfb20f343916edd877969aa2a2ac1b98`, path
  `vllm_mlx/mllm_batch_generator.py` and its media helpers, Apache-2.0.
  Adaptation: no implicit network fetch, strict byte/frame bounds, EXIF/RGB
  image normalization, minimum/pixel checks, bounded WAV metadata,
  deterministic capped video frames, content hashes, and an adapter-owned
  execution hook. Current text adapters remain fail-closed.
- Original mlx2 work: atomic tenant-scoped Responses/Files/Batch persistence,
  cursor listings, interrupted-batch recovery, decoder-input representation and
  reranking, allowlisted MCP Streamable HTTP execution, bounded automatic tool
  loops, and pre-byte Chat tool-stream validation.
- Validation: host-only compile plus 7 focused CPU tests and 68 HTTP/API CPU
  tests. Repository pytest collection requires MLX/Metal on this host, so these
  tests were copied unchanged to `/tmp` to bypass the MLX-importing conftest.
  No model, Metal/GPU, remote MCP server, or production media tower was run.

## 2026-09-18 — Gemma 3n video and MiniCPM-o media execution

- Source: local `mlx-vlm-qwen4-exp` revision
  `653f1f13e238abb313fd45071bbd04b3de414635`, paths
  `mlx_vlm/models/gemma3n/processing_gemma3n.py`,
  `mlx_vlm/models/gemma3n/gemma3n.py`,
  `mlx_vlm/models/minicpmo/processing_minicpmo.py`,
  `mlx_vlm/models/minicpmo/minicpmo.py`, and
  `mlx_vlm/models/minicpmo/config.py`; repository license MIT.
- Peer design inputs: TensorRT-LLM's processor/encoder/decoder separation,
  bounded encoder cache and per-modality scheduling; vLLM-Omni's MiniCPM-o
  chat/TTS boundary; Rapid-MLX revision
  `ee101f1bcfb20f343916edd877969aa2a2ac1b98` audio speech route. No peer
  runtime implementation was copied.
- Current-upstream review: Blaizzy `mlx-vlm` main revision
  `e79b0e041677ec4ca5333ba750376bb4e8c434cb` (version 0.7.1, MIT), including
  `mlx_vlm/generate/audio.py`, `mlx_vlm/models/minicpmo/`, Gemma 4's explicit
  video tensors, continuous batching, and the projected vision-feature cache.
  See [UPSTREAM-MLX-VLM-REVIEW-2026-09-18.md](UPSTREAM-MLX-VLM-REVIEW-2026-09-18.md).
- Modifications: timestamped native-video planning over Gemma 3n's vision
  tower with bounded ordered frame forwards, media-bound APCv2 keys, isolated encoder prefill with ordinary batched
  decode afterward, MiniCPM-o global-plus-grid slicing, same-shape bounded
  vision batches, strict-rate PCM normalization and audio chunking, telemetry,
  and an adapter-gated OpenAI `/v1/audio/speech`/`AudioOutput` contract. The
  request fields and six binary response formats were checked against the
  official OpenAI Create speech reference on 2026-09-18.
- Sampling defaults: Gemma 3n declares `top_p=0.95` and `top_k=64` from the
  downloaded snapshot's `generation_config.json`. MiniCPM-o declares
  `temperature=0.5` from the downloaded model card's omni inference example.
  Explicit API request values continue to take precedence.
- Artifact compatibility: the released OpenBMB MiniCPM-o 2.6 snapshot uses a
  Qwen2 text backbone, projection biases, and an implicit attention head
  dimension. Current upstream revision `e79b0e0` wires its MiniCPM-o loader to
  Qwen3-VL instead. mlx2 therefore applies a load-scoped, lock-protected Qwen2
  compatibility normalization, restores upstream globals after loading, and
  still requires strict weight coverage.
- Safety/qualification: current upstream can load MiniCPM-o TTS weights, but
  requires reference-voice audio and generates a chat response rather than
  rendering the API's exact text through a named voice. `output_audio` remains
  unselected rather than misrepresenting that contract. Host-only syntax and
  CPU contract tests are required here. Candidate GPU smokes now exercise real
  weights, but production qualification remains pending and is not implied by
  registry availability.
- Host processor evidence: `scripts/qualify_multimodal_processors.py` loaded
  the real Gemma 3n E2B-it and MiniCPM-o 2.6 tokenizer/image/audio processors
  from downloaded snapshots with Torch's default device fixed to CPU. It did
  not import MLX or load model weights. The retained receipt records three
  ordered Gemma frames as `[3,3,768,768]` and 768 aligned soft tokens; MiniCPM
  produced three image slices, three exact audio chunks (16000/16000/4000
  samples), aligned media bounds and deterministic policy fingerprints. This
  qualifies processor contracts only, not model execution or output quality.
- Candidate GPU evidence:
  `qualification/runs/multimodal-gpu-smoke-20260918/receipt.json` records a
  Gemma 3n native-video request, an identical repeat with 574/575 prompt tokens
  reused by APCv2, a Gemma input-audio request, and a MiniCPM-o combined
  image/audio request. All produced finite responses through the ordinary
  route. These are bounded smokes, not output-quality, concurrency, long-context
  or production qualification.

## 2026-09-18 — Prometheus telemetry contract

- Implementation: original mlx2 code; no peer exporter implementation was
  copied. Public Prometheus/OpenTelemetry conventions and the documented metric
  concepts of oMLX, Rapid-MLX, vLLM, SGLang and TensorRT-LLM were reviewed as
  design inputs.
- Scope: deterministic Prometheus text exposition, cumulative host-only
  histograms, request/token accounting, and bounded translation of APCv2,
  scheduling, speculative, segmented-MTP, cache-capsule, memory, approximate
  state, structured-output and Spomin snapshots.
- Safety: scrapes are non-destructive and perform no device reads; identifiers,
  prompts, paths, fingerprints and free-form diagnostic values are excluded
  from labels.
- Qualification: CPU-only contract, concurrency and HTTP tests. No model was
  loaded and no Metal/GPU execution is claimed.

## 2026-09-15 — Flash-Next serving slice (in progress)

- Source: local `mlx-lm-unified`, revision
  `1e2bc604f71d070bee970c3e7db8b60f7855599b`, Apache-2.0.
- Scope: symbol-level extraction of Flash-Next tensor math and kernels,
  continuous ordinary/self-MTP batching, transformed sampling, APCv2 layer
  descriptors, COW transactions, cache serialization, tokenizer and PLE I/O.
  The exact source modules and retained symbols are recorded in
  `provenance/flashnext.json`.
- Adaptations: package imports relocated to `mlx2.runtime`; no dependency on
  the unified checkout. Historical servers, generic model loaders, CLI paths,
  training, compiled megakernel routes and one-shot host stubbing are excluded.
  APCv2 becomes a standalone engine with mandatory segmented COW storage and
  failed publication on unsupported descriptors, with no legacy APC engine.
- Original work: Flash-Next artifact loader, serving profile, adapter, bounded
  HTTP request lifecycle, CLI, qualification harness and integration tests.
- Upstream copyright notices for adapted source are retained in
  `provenance/NOTICE`; original mlx2 code is not attributed to Apple.
- Validation: pending; implementation alone does not qualify the serving profile.

## 2026-09-15 — Flash-Next integration and qualification updates

- Same source revision and Apache-2.0 license as the extraction above.
- Additional mined modules: `mlx_lm/tool_parsers/qwen3_coder.py`,
  `mlx_lm/tool_parsers/_schema.py`, `mlx_lm/os_memory.py`, and the pure
  self-MTP admission policy from `mlx_lm/server.py`. No old server was copied.
- Adapted tests: APCv2 lifecycle, batched Qwen4 forced acceptance, segmented MTP,
  cache-state oracle helpers and test-only indexed-QSA reference operations.
- Integration changes: fresh per-request sampling RNG; efficient detokenization;
  ordinary/MTP prompt checkpoints; API tool-history normalization; per-request
  observed widths; removal of over-budget scheduler overrides; admitted-subset
  queue ordering; bounded idle admission backoff and scratch reclamation.
- Original implementation: HTTP lifecycle/output channels, artifact/native
  runtime binding, qualification gate, readiness supervision, memory floor,
  benchmark and qualification commands. Original code carries no Apple header.
- `provenance/flashnext.json` records module paths, source revision, adaptations
  and test sources. `provenance/NOTICE` retains mined notices. Results in
  `qualification/` apply only to their recorded runtime/artifact/settings;
  failed and stale candidate evidence does not qualify a later source revision.

## 2026-09-15 — qualified Flash-Next profiles

- Final validation: 102 tests plus 9 forced-acceptance subtests; both MTP2 and
  ordinary real-model HTTP profiles pass, including the mixed-warm admission
  regression and near 16K context reuse. Exact bindings and receipts are in
  `qualification/flash-next-{mtp2,ordinary}.json`.
- Final warm HTTP benchmark: MTP2/ordinary single-request medians 76.55/50.21
  tokens/s; four-request aggregate medians 73.17/104.33 tokens/s. These results
  favor different profiles for interactive versus throughput workloads; no
  general batched-MTP speedup is claimed. See `docs/RESULTS.md` and raw reports.
# Advanced serving parity — 2026-09-15

Additional sampling processors, mechanism diagnostics and focused QSA/shared-state/
promotion tests are mined from local `mlx-lm-unified` revision
`1e2bc604f71d070bee970c3e7db8b60f7855599b` (Apache-2.0), paths
`mlx_lm/sample_utils.py`, `mlx_lm/round_levers.py`, `mlx_lm/models/qwen4*.py`,
and the corresponding `tests/test_qwen4_qsa_indexed*.py`,
`tests/test_qsa_shared_suffix.py`, `tests/test_segmented_physical_promotion.py`,
`tests/test_known_tail_ple_prefetch.py`. Imports are relocated to mlx2; server
soft-reload tests and legacy products are excluded. Validation is recorded in
the parity qualification artifacts. Original client normalization and execution
policy code are mlx2 implementations.

## 2026-09-15 — license classification correction and parity closure

The earlier blanket Apache-2.0 labels above were incorrect. The source root
LICENSE is MIT; individual files explicitly declare Apache-2.0, and the Qwen4
model notes its Apache-2.0 Transformers origin. `provenance/flashnext.json` now
records each source file's license and basis; mined runtime SPDX labels were
corrected accordingly. `provenance/LICENSE.unified-MIT` retains the complete
MIT notice; `provenance/NOTICE` retains copyright attribution. This correction
does not put upstream copyright on original mlx2 code.

Additional selected symbols: SharedPrefixPhysicalPromotionPrequeue and
begin_shared_prefix_physical_promotion from segmented_physical_promotion.py;
indexed chunk-range and private-delta/quantized reference oracles from
models/qwen4_qsa_indexed.py. Same source revision as above. These close the
modern path/test dependency closure without importing a replaced engine.

## 2026-09-15 — model geometry admission and actual mechanism gates

The Flash-Next cache budget is original mlx2 code derived from the adapter's
actual configuration and cache tensor shapes; no generic model forward is
performed during estimation. The existing mined memory controller now accepts
an adapter projection while retaining its reserve/transient policy and actual
resident-byte floor. CPU tests compare the projection to real tiny-model
target/draft allocations and check reserve denial. The source-only distinction
is documented in `docs/FLASHNEXT-PARITY.md`; full-model context validation is
separate. Shared checkpoint attestation, transaction rebind and bounded cache
reclamation are original integration repairs over the mined state contracts.

Root's isolated kernel results are recorded in
`qualification/indexed-kernel-build.json` (62 tests) and
`qualification/optional-epilogue-kernels.json` (8 tests). These qualify only the
exact named native build/shape oracles, not end-to-end serving performance.

The mlx2 adaptation of the mined memory admission callback now accepts an
optional scratch-reclamation callback. Before a non-full plan changes a lane's
route, it clears reclaimable allocator cache and recomputes the plan from fresh
measured headroom. It does not credit hypothetical memory or lower the hard
reserve. CPU regression tests cover recovery and a genuinely constrained queue;
real warm-cohort qualification remains required.

The original mlx2 warm-admission adaptation adds resident-only APC lookup,
lease-preserving pressure eviction, completed allocator reclamation and a
measured full-copy bound for complete warm caches. Existing APCv2 and controller
mechanisms remain rooted in the revisions recorded above. New CPU checks cover
leased generation preservation, no disk restoration before admission, full
warm-copy accounting and delayed memory reclamation. This is a local adaptation,
not a claim that those integration functions were copied from unified.

The callback's original mlx2 pressure adaptation additionally evicts bounded
unleased APC checkpoints before reducing a cohort's route. It compares against
the existing capacity-capped controller plan, reclaims and remeasures each
eviction, and never credits nominal checkpoint bytes. CPU regressions cover
full-cohort recovery, capacity-only limits, exhausted memory, bounded retries,
and a rate-limited neighboring admission that must reclaim allocator pages
before evicting warm state.

The worker-owned hard-admission retry queue is original mlx2 integration code.
It preserves a leased resident lookup across bounded retries rather than
rejecting delayed host-memory recovery immediately. CPU worker tests validate
active progress, admission recovery, cancellation, timeout and shutdown; the
existing controller reserve and disk-before-allocation gate are unchanged.

The optional-promotion pressure preflight is an original mlx2 integration
repair over the mined physical-promotion ticket. The ticket exposes pending
bytes separately after a pressure-only synchronization. A pure policy preview
allows cancellation/drain before APC eviction or lane migration, with bounded
nonblocking host-accounting grace. CPU tests validate continued real ticket
publication, normal overlap, settle/cancel outcomes and the old-order eviction
counterfactual; reserve and qualification gates remain unchanged.

## 2026-09-15 — integrated Muse DFlash2 CPU implementation

- Tensor closure source: local `mlx-uag/worktrees/agnes-vlm-support`, revision
  `8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5`, **MIT**. Selected paths:
  `mlx_vlm/speculative/drafters/dflash2/{dflash2.py,config.py}`,
  `mlx_vlm/speculative/drafters/qwen3_dflash/{dflash.py,config.py}`, and Muse
  post-layer tap semantics from `mlx_vlm/models/muse_glimmer/language.py`.
  Exact attribution, modifications and local helper provenance are in
  [muse-dflash2.json](../provenance/muse-dflash2.json); the complete retained
  notice is [muse-dflash2.NOTICE](../provenance/muse-dflash2.NOTICE).
- Adaptation: self-contained tensor imports, target taps/body-only prefill,
  context-only draft append, batched projections with per-row attention, and
  actual selector proposal distributions. The original mlx2 verifier uses
  acceptance probability `min(1,p/q)` and the positive `p-q` residual. This is
  a distribution-aware extension; upstream tree/exact-match behavior is not
  claimed as implemented parity.
- Original integration: separate external execution lifecycle, request-owned
  RNG, atomic rollback/publication, APCv2 target/draft/RNG disk pairing,
  distinct route capability/receipt, ordinary fallback, reserve-preserving
  memory admission and fair partial-fit scheduling. No legacy APC or server
  is imported; original code has no upstream copyright header.
- Original segmented rotating-cache transactions build on existing mlx2 KV
  classes and attention seams, whose unified source revision is
  `1e2bc604f71d070bee970c3e7db8b60f7855599b` (MIT). Exact reuse boundaries,
  snapshot/replay design and CPU oracles are in
  [segmented-rotating-kv.json](../provenance/segmented-rotating-kv.json).
- Validation before merge: **93 targeted CPU tests**, including actual serving
  reserve callbacks, partial-fit fairness, real APC lease protection, BF16
  cache dtype, ring rollback, sampled-distribution and paired disk/RNG tests.
  The [independent review](ports/MUSE-DFLASH2-REVIEW.md) records initial defects,
  repairs and the exact isolated source identity. Integrated validation is
  separate; no production artifact loading, GPU, peak-memory or speedup
  qualification is implied. The `muse-glimmer-apcv2-dflash2` profile remains
  unqualified and unselected. See [implementation and gaps](ports/MUSE-DFLASH2.md).

The earlier metadata-only Muse provenance describes its original port stage;
this entry and the DFlash2 provenance record the later tensor implementation.

Post-merge CPU validation reported by the integrator: 384 passed, 16 Metal
skips and 59 subtests; focused integration 71 tests; lint and compileall passed.
Combined source: `2ab2c2f8019c388cb81472997f6464f10a6c3a237719334e2b65c041c4840e7d`.
The documentation closeout independently verified this source hash. These CPU
checks do not qualify Muse GPU execution or select its external route.

## 2026-09-15 — North Mini Code ordinary serving slice

- Tensor source: local `mlx-lm-unified` revision
  `1e2bc604f71d070bee970c3e7db8b60f7855599b`,
  `mlx_lm/models/cohere2_moe.py`, MIT. Exact symbols, source hash,
  modifications, artifact identities and validation are in
  [north-mini-code.json](../provenance/north-mini-code.json); the retained
  license is [north-mini-code.NOTICE](../provenance/north-mini-code.NOTICE).
- Adaptation: North-specific sigmoid MoE and parallel-block math uses mlx2
  attention, SwitchGLU and cache classes. The historical local MTP head and
  measured-negative QKV fusion were excluded. The ordinary route declares an
  exact 13-global/36-rotating APCv2 layout.
- Original integration: strict GPU-free artifact inspection, registry dispatch,
  request normalization, Cohere channel/action parsing, absent-MTP refusal,
  diagnostics and qualification documentation. Shared APCv2, scheduler,
  batching and receipt layers have no model-name branch.
- CPU validation: 40 focused adapter/registry tests, including tiny native
  full-versus-split replay and layered-cache offsets. Production weights and
  GPU were not loaded. The route remains pending qualification.

## 2026-09-16 — Qwen3.6 35B-A3B ordinary/APCv2 slice

- Tensor semantics were mined from local `mlx-lm-unified` revision
  `13eb83388750435bcc751d0b9e33857a38152544`, MIT. Exact paths, hashes,
  notices and modifications are in
  [qwen36-35b.json](../provenance/qwen36-35b.json).
- The adapter combines the existing hybrid GDN/cache implementation with the
  existing sparse-MoE block, handles split and fused gate-up checkpoint forms,
  and detects an actual embedded MTP head. Eager/stock MoE is the baseline;
  optimized kernels and compiled decode remain unselected candidates.
- Original mlx2 work adds strict artifact inspection, capability/registry
  wiring, bounded structured-output constraints, batching telemetry and the
  prompt-lookup indexed oracle/policies. Original files carry no Apple header.
- CPU/static focused tests passed. A bounded real-artifact GPU probe and cold/
  warm HTTP/APCv2 smoke passed without swap growth; this evidence is candidate
  validation only and does not select a route.
- Closeout verification passed 621 CPU tests with 16 Metal-only skips and 59
  parameterized subtests at runtime source
  `ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`.
  The wheel and source archive retain the mined MIT notices through the
  project `license-files` manifest. GPU receipts remain bound to their recorded
  earlier sources and are not promoted to current-source qualification.

## 2026-09-17 — compact Qwen4 GDN rollback prototype

- Design input: the public vLLM ReplaySSM reconstruct-at-commit mechanism. No
  vLLM source code was copied. The compact tape/reconstruction Metal source,
  integration, CPU reference and traffic harness are original mlx2 code.
- Scope: `MLX_QWEN4_FUSED_GDN_REPLAY_ROLLBACK=1` selects compact rollback for
  the existing B=1 fused speculative-verify route. The qualified Flash-Next
  adapter profile now selects it by default; direct low-level imports remain
  off unless a profile or caller opts in. It stores activation-precision
  normalized keys plus FP32 correction/decay, retains the final full-accept
  state, reconstructs only partial acceptance, and derives the convolution
  window from the pre-verify checkpoint plus the already-present QKV input.
- Safety: the existing fused snapshot and generic replay implementations remain
  unchanged reference paths. Unsupported geometry, unavailable Metal kernels,
  or failed compile probes return to the generic path and emit bounded counters.
- Validation: CPU reference checks cover every partial acceptance at verify
  widths 3 and 8, convolution restoration, default-off configuration,
  fail-closed acceptance bounds, and static traffic accounting. Apple M5 Max
  qualification then compiled and executed widths 3, 4, and 8, proved exact
  kernel reconstruction at every partial length, and proved exact model-bound
  rollback and continuation logits across 36 GDN layers with nonzero replay
  receipts and zero replay fallbacks. See
  `docs/experiments/QWEN4-GDN-REPLAY-GPU-2026-09-17.md`. After a fresh
  counterbalanced A/B confirmed the result, the route was selected as the
  Flash-Next adapter default. Snapshot and generic replay remain references and
  fail-closed fallbacks. A fresh post-promotion adapter load then observed all
  36 recurrent modules in compact mode and repeated exact widths 3, 4, and 8
  checks with nonzero replay counters and zero fallbacks; see the
  `post-promotion-smoke.json` receipt in the same experiment directory.

## 2026-09-17 — live prompt-lookup route

- The adaptive indexed proposer and exact rollback design were mined from the
  local MIT `mlx-lm-unified` revision
  `13eb83388750435bcc751d0b9e33857a38152544`. Exact source paths,
  modifications and route boundaries are recorded in
  [prompt-lookup-live.json](../provenance/prompt-lookup-live.json); the retained
  license is [LICENSE.unified-MIT](../provenance/LICENSE.unified-MIT).
- mlx2 exposes the mechanism as a distinct APCv2 serving route. Each lane
  target-verifies indexed proposals, restores partial or truncated rounds to
  the exact committed boundary, emits bounded counters and latches to ordinary
  decode when acceptance is poor. The current implementation verifies B1 per
  lane and does not claim cross-request target batching.
- Capability declaration, implementation, qualification and selection remain
  separate. The qualification gate requires observed proposals and rollback;
  ordinary or MTP evidence cannot authorize this route.

## 2026-09-17 — revision-bound Spomin live QSA surgery

- The segmented control plane, epoch manager, and Qwen4 QSA surgery were mined
  from local MIT `mlx-lm-unified` revision
  `13eb83388750435bcc751d0b9e33857a38152544`. Exact paths and modifications are
  recorded in [spomin-live-surgery.json](../provenance/spomin-live-surgery.json);
  the retained license is
  [LICENSE.unified-MIT](../provenance/LICENSE.unified-MIT).
- The implementation retains the uncompacted transcript as a separate,
  immutable proposal plane and edits only a request-private, unquantized B=1
  Qwen4 QSA cache at a revision-matched, quiescent, device-drained barrier.
  Hybrid recurrent/PLE state currently has no compacted-history repair and
  fails closed before a transaction is created. Expected revision, plan,
  capability, and backend refusals are converted into explicit declined
  receipts instead of escaping the request boundary. Active MTP, packed or
  batched caches, replacement summary tokens, missing index ledgers, and
  non-default RoPE also fail closed before mutation.
- The route is implemented and Metal-tested but remains default-off and
  unselected pending real-artifact quality and long-context qualification.

- 2026-09-18: `runtime/spomin_standard_surgery.py` (standard `KVCache` /
  `RotatingKVCache` backend with per-layer RoPE re-phasing) is original mlx2
  code written against the mined control-plane contract above; no additional
  source was imported. Validation: `tests/test_spomin_standard_surgery.py`
  (single-layer exactness against a compacted rebuild for both RoPE layouts and
  NoPE, wrapped-ring re-phasing, refusal before mutation, two-lane decode).

## 2026-09-18 — Xing4.0-29B-A4B port

- **Model** (`src/mlx2/runtime/models/xing4_0.py`): MLA attention, the
  absorbed/expanded gate, `MultiLinear`, the noaux_tc MoE and the
  expert-stacking/`kv_b_proj`-folding sanitize are adapted from local MIT
  `mlx-lm-unified` `mlx_lm/models/deepseek_v3.py` and `mla.py` at
  `13eb83388750435bcc751d0b9e33857a38152544`. The mHC (4-stream Sinkhorn
  hyper-connection) residual, router and stream semantics re-implement the
  Xing4.0 Hugging Face reference `modeling_xing4_0.py` (Apache-2.0, sha256
  `51341dbd…`, not copied). The MTP head follows the DeepSeek-V3 MTP
  definition used by the unmerged vLLM PR 57135 and SGLang PR 39793. mHC
  operands are computed in fp32 as vLLM and SGLang do (HF rounds them to the
  model dtype). Parity fixture: `scripts/xing4_0_reference_fixture.py` →
  `tests/fixtures/xing4_0_tiny`. See
  [xing4-0-model.json](../provenance/xing4-0-model.json).
- **Tokenizer and output parser** (`scripts/xing4_0_tokenizer.py`,
  `src/mlx2/adapters/xing_tokenizer.py`, `src/mlx2/adapters/xing_output.py`):
  tokenizer construction and exact decode are original. Behavioural reference:
  the HF `tokenization_xing4_0.py`, `tokenizer.model` and
  `chat_template.jinja` (Apache-2.0). Tool-call semantics adapted from the vLLM
  PR 57135 tool and reasoning parsers and the SGLang PR 39793 detector
  (Apache-2.0). Validated against the live slow reference on 254,983 strings
  and 634,966 decode checks, all identical. See
  [xing4-0-tokenizer-parser.json](../provenance/xing4-0-tokenizer-parser.json).
- **Adapter, memory budget, converter, segmented MLA, performance kernels**
  (`adapters/xing.py`, `adapters/xing_memory.py`, `scripts/convert_xing4_0.py`,
  `SegmentedBatchKVCache.row_views`, `runtime/models/xing4_0_mhc_metal.py`,
  heads-as-queries absorbed MLA, `structured_output.token_pieces`, the
  turn-closed tool-call rule in `xing_output.py`): original mlx2 code.
  Kernel validation: float64-reference GPU tests in
  `tests/test_xing4_0_model.py` and full-scale KL against the HF reference
  (`docs/ports/XING4-0.md`).

## 2026-09-18 — instance-scoped int8 NAX prefill

- Source: local MIT `mlx-lm-unified`, branch `unified`, path
  `mlx_lm/int8_prefill.py`, last-touching commit
  `aec9b4a1ad6633da4fda1b11a82cc3edce458037` (HEAD
  `13eb83388750435bcc751d0b9e33857a38152544`); a lab-authored overlay
  originally ported from the lab's mlx-vlm tree. Retained license:
  [LICENSE.unified-MIT](../provenance/LICENSE.unified-MIT).
- Kept: the per-row int8 activation quantization kernel, the MPP `matmul2d`
  128×128 int8 GEMM with in-register scaling and bias, the 512-row threshold,
  and the size-1 shared-activation cache.
- Changed: no global `nn.QuantizedLinear.__call__` patch and no env knobs;
  installed per model instance by class swap with a removable handle. Requant
  generalized to 2/3/4/5/6/8-bit affine packing with exact per-channel absmax;
  bf16 `nn.Linear` supported; model-neutral path-based scope plus adapter
  selector; device and compile probe fail closed at startup; the adapter must
  declare the scope; decode/verify row bound checked; APPROXIMATE fidelity, a
  `feature_int8_prefill` qualification check and an APCv2 namespace keyed to
  the policy and kernel revision.
- Validation: `tests/test_int8_prefill.py` (34 CPU tests plus 27 GPU tests
  behind `MLX2_TEST_INT8_NAX=1`, 61/61 on M5 Max). Routed MoE experts stay on
  stock kernels.

## 2026-09-19 — ecosystem probe fixes (ideas only, no code mined)

- `adapters/norm_repair.py`: original mlx2 code. Motivated by omlx#3750 and
  MTPLX#511 (MTP `pre_fc_norm_*` +1 shift ambiguity). The byte hashes were
  taken from the oQ4e-mtp 35B artifact and checked bit for bit against
  `Qwen/Qwen3.6-35B-A3B@995ad96` `model-00026-of-00026.safetensors`.
- `ArraysCache._empty_slot` / `Qwen4ArraysCache` EOS history fill: original.
  The join-must-not-alter-a-row lesson comes from omlx#3703.
- Prompt-lookup allocator reclaim: the cadence idea comes from Ollama v0.34.2
  (periodic buffer-pool release during MLX speculative decode). mlx2 reuses its
  own `ALLOCATOR_RECLAIM_STEP_INTERVAL`.
- Short-request prefill interleave (self-MTP alternation, ordinary one-chunk
  overflow slot): the alternation idea comes from omlx#3726. The implementation
  is original.
- Validation: `tests/test_mtp_norm_repair.py`,
  `tests/test_qwen4_cold_join_ple_history.py`,
  `tests/test_decode_allocator_reclaim.py`,
  `tests/test_short_request_prefill_starvation.py`,
  `tests/test_apc_hits_hybrid_gdn_self_mtp.py` (CPU). No GPU qualification yet.
