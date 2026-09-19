# mlx2

> [!WARNING]
> **Experimental research software.** mlx2 is a lab runtime under active
> development. APIs, on-disk formats, routes and qualification policies change
> without notice, and nothing here is supported for production use. Model
> paths in qualification records and docs refer to the lab machine
> (`~/mlx-models/...`); point them at your own artifacts before running.

A clean inference runtime for Apple Silicon, starting with Qwen4 Flash-Next.
Proven mechanisms are mined with provenance into this project. Serving does
not depend on the unified checkout or a legacy server.

## Serving stack

- OpenAI-compatible chat/completions, SSE streaming, function calls, reasoning,
  seeded sampling, stop strings, cancellation and token log probabilities.
- Responses API text/function items with bounded process-local storage,
  retrieval, deletion and `previous_response_id` continuation; typed text and
  function-call SSE, required/named tools, strict schemas and single-call
  enforcement. UTF-8 files are accepted through the Files API; image/audio,
  hosted tools and MCP tools fail as unavailable on the text-only routes.
- Anthropic-compatible `/v1/messages` and `/v1/messages/count_tokens` reuse the
  same validated chat pipeline, including typed SSE, tools, signed thinking,
  cache-aware usage and fail-closed model tool JSON.
- OpenAI-shaped embeddings plus Cohere/Jina-shaped reranking are exposed only
  through adapter-owned representation hooks. The shipped generation adapters
  return 501 instead of fabricated vectors or scores. Standard Files and JSONL
  Batch lifecycles and vLLM-compatible LoRA control endpoints are bounded and
  tenant-scoped; LoRA activation likewise requires an engine implementation.
- Nonstreaming `n=1..8` sampling reserves the whole cohort before publishing
  jobs and remains bounded by lane capacity and measured physical headroom.
- Regex, JSON-object and strict JSON-schema output constraints fail closed and
  are available only on routes whose qualification exercised them.
- Hermes `options`, `think`, `reasoning_effort` and template thinking controls
  normalize into explicit request controls; unknown or conflicting values fail.
- Continuous batching, bounded queues, adaptive prefill and memory admission
  with a 20 GiB service/driver reserve. Delayed memory recovery can queue a
  request without occupying an execution lane.
- **APCv2 is the only prefix-cache engine.** Layered copy-on-write state,
  segmented execution and revision-bound target/draft snapshots support both
  ordinary decoding and speculation. Leased checkpoints survive pressure
  eviction; idle disk state is process-local scratch.
- Optional APCv2 session tags can park, prefetch, resume and delete exact
  tenant-owned checkpoints across a bounded persistent disk tier and restart
  rescan. They are cache controls, not an alternate conversation store.
- Qualification-only Spomin surgery retains an exact pre-surgery APCv2 shadow
  for repeated-prompt reuse while keeping compacted state out of exact stores.
  Eligible Qwen3.x adapters also expose revision-bound Q8 and K8V4 cache
  operations, default-off until matching route qualification exists.
- Flash-Next persistent MTP2, fused GDN and MoE paths, file-backed/compiled PLE,
  pooled QSA, known-tail prefetch, and gated indexed/shared QSA and asynchronous
  promotion. Policy and mechanism counters show what was selected and used.
- An ordinary reference route using the same model, APCv2 and request lifecycle.

Automatic mechanisms retain their context, shape, state and output-budget
gates. Shared segmented QSA and physical promotion are alternative cache routes
within a cohort. Optional fused indexed epilogues, whole-decode compilation,
megakernels, adaptive draft-depth routing and approximate caches are not all
enabled by default. See the [mechanism matrix](docs/FLASHNEXT-PARITY.md).

## Model and qualification status

| Model | Implemented slice | Serving qualification |
|---|---|---|
| Qwen4 Flash-Next | Optimized native MTP and ordinary reference, shared API/cache/scheduler | Modernized qualification paused; latest 262K rerun interrupted, no complete new profile or benchmark |
| Qwen3.8 27B | Dense hybrid adapter, plain-KV segmented execution, native MTP on complete embedded-head artifacts | CPU/static candidate; no model GPU qualification or deployment |
| Qwen3.6 35B-A3B | Hybrid GDN/MoE ordinary and native-MTP artifact slices, APCv2, bounded fused-GDN trial, prompt-lookup oracle | CPU-tested candidate plus bounded earlier GPU receipts; no matching current-source qualification, selection or deployment |
| Muse Glimmer | Ordinary text adapter and linear external DFlash2 draft/verify with exact per-lane rotating-cache rollback | CPU-tested candidate; no production-weight load, GPU qualification or deployment |

The initial 16K Flash-Next measurements are retained as
[historical results](docs/RESULTS.md). Their four concurrent HTTP requests do
not establish four-wide model execution. New comparisons require observed
compute widths and matching runtime/artifact/settings receipts.

The 2026-09-16 source closeout passed 621 CPU tests, skipped 16 Metal tests and
ran 59 subtests at runtime source
`ad1d6f8d9bf715f1a32c5d12159fb129ec4b62e8066574448f48a784cb11429e`.
This verification closes the repository workset, including the heterogeneous
shared-QSA regression. It does not replace GPU qualification. The latest 262K
Flash-Next run remains interrupted, and Qwen3.6 receipts bind earlier candidate
sources. See [results](docs/RESULTS.md) and the [resume handoff](docs/RESUME.md).

Before GPU testing another model, review its remaining functionality and
optimization gaps: [Qwen3.8 27B](docs/ports/QWEN38-27B.md),
[Qwen3.6 35B-A3B](docs/ports/QWEN36-35B-A3B.md),
[Muse Glimmer](docs/ports/MUSE-GLIMMER.md), and
[Muse DFlash2](docs/ports/MUSE-DFLASH2.md). Muse's host probability verifier and
snapshot/replay rollback are correctness implementations with unmeasured
performance costs.

The [ASPIRE-style asynchronous MTP investigation](docs/experiments/ASPIRE-ASYNC-MTP-CLOSEOUT-2026-09-17.md)
is closed as a negative result for the measured native Qwen3.6 artifact. Real
short- and long-context GPU runs selected ordinary decode over every supported
fixed draft depth; no asynchronous or heterogeneous scheduler was selected.

## Service operation

Host-specific service definitions and deployment receipts are intentionally
kept outside this repository. Discover a locally configured service through
`/v1/models`; `/health` and `/v1/status` expose readiness, qualification
identity, memory, APC leases, scheduler decisions and mechanism counters.
Terminal responses include an `mlx2` route receipt.

## Development and reproduction

The tested environment uses Python 3.12 and local MLX build
`0.32.2.dev20260915+2a817ad94`. Qualification binds source, native binaries,
dependency versions, local artifact identity and serving settings. Changes
require new evidence; normal serving rejects a mismatched receipt.

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/sdk_smoke.py --sdk-python /path/to/sdk-venv/bin/python
.venv/bin/python scripts/qualify_serving.py --output qualification/candidate.json
.venv/bin/python scripts/benchmark_serving.py --rounds 5 --widths 1 2 4 \
  --output qualification/benchmark-candidate.json
```

The last two commands need an exclusively owned running candidate service.
Use the [serving guide](docs/SERVING.md) for startup, artifact requirements,
Hermes controls and qualification. For the tensor-free control core, install
`.[dev]` and run `pytest tests/test_core.py`; tensor/cache tests require MLX.

See [architecture](docs/ARCHITECTURE.md), [results](docs/RESULTS.md),
[API parity](docs/API-PARITY.md), and [provenance](docs/PROVENANCE.md).
