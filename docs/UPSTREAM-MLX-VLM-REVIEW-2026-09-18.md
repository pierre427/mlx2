# Upstream mlx-vlm review — 2026-09-18

Reviewed Blaizzy's `mlx-vlm` main at
`e79b0e041677ec4ca5333ba750376bb4e8c434cb` (package 0.7.1, MIT). The project is
named `mlx-vlm`, not `mlx-vllm`. This review treats upstream as a design and
optional execution dependency; APCv2 and mlx2's request lifecycle remain the
authoritative serving control plane.

## Adopted

- **Multimodal scheduling:** encoder-bearing requests prefill alone, then join
  the ordinary continuous decode batch. Encoder kwargs are single-use and
  later prompt segments return to bounded text prefill.
- **Projected feature reuse:** a bounded LRU caches evaluated vision projections
  at the model embedding boundary. Unlike upstream's path-oriented identity,
  mlx2 keys entries with artifact, content, media order, and processor policy,
  and clears them on model/LoRA revision changes.
- **Explicit processor outputs:** media placeholders, bounds, target sizes and
  encoder tensors stay adapter-owned. The generic scheduler only transports an
  opaque first-prefill payload.
- **MiniCPM-o controls:** artifact-declared image slicing, equal-shape bounded
  vision batches, and exact-rate audio chunking are applied and receipted.

## Adapted rather than copied

- Gemma 4 upstream has a true video token/type contract with separate
  `pixel_values_videos` and `video_position_ids`. Gemma 3n does not. mlx2
  therefore represents a Gemma 3n video as ordered, uniformly sampled,
  timestamp-labelled frames through its existing vision tower. It does not
  invent unsupported Gemma 4 tensor fields.
- Gemma 3n's vision tower already treats frames as an image batch. mlx2 caps
  that forward in ordered chunks and concatenates projected features before
  token fusion, limiting encoder peak without changing frame order.
- APC reuse includes the ordered media fingerprint and is accepted only after
  the cached prefix crosses every media placeholder. A shorter text-only cache
  hit is closed and treated as a miss.
- MiniCPM-o vision batching groups equal processed tensor shapes, caps each
  encoder forward, and restores original image order before fusion.

## Deferred or rejected

- Upstream MiniCPM-o TTS is a spoken-chat pipeline requiring a reference WAV
  and an external vocoder. `/v1/audio/speech` requires exact input rendering
  through a named voice. mlx2 now has the binary `AudioOutput` contract, MIME
  validation, serialization barrier and telemetry, but MiniCPM-o does not
  declare the capability until a voice registry and correctness qualification
  exist.
- Upstream cache classes are not imported. mlx2 remains APCv2-only and retains
  its revision, tenant, semantic, media, and state-layout namespaces.
- No model-family condition was added to the scheduler; all tensor-specific
  behavior remains behind adapters.

## Qualification boundary

This slice is CPU/static-qualified only. It proves request validation, media
planning, cache identity, fail-closed audio dispatch, registry selection and
HTTP wire behavior. It does not claim Metal execution, model output quality,
native Gemma 3n video training semantics, MiniCPM-o TTS compatibility, or GPU
performance.

The retained processor receipt at
`qualification/runs/multimodal-processors-20260918/receipt.json` additionally
executes the downloaded artifacts' real tokenizers and image/audio processors
on CPU. Production route selection still requires the adapter-declared live
checks; the processor receipt is intentionally not accepted as a serving
qualification receipt.
