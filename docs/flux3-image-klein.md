# FLUX 3 Image Klein

`mlx2.adapters.flux3_image.Flux3ImageKleinAdapter` is a direct native MLX
adapter for Black Forest Labs' distilled image model. It supports generation,
single-reference editing, and up to ten reference images. Bounding-box captions
can be passed directly as prompt text in the publisher's format. The adapter
uses the checkpoint's 16-step flow-map sampler without classifier-free guidance.

The image transformer, VAE, and Qwen3-VL text computation execute in MLX.
PyTorch is used only for the reference-compatible seeded CPU noise generator.
The text encoder collects hidden layers 4, 8, 12, 16, 20, 24, 28, and 32,
with right padding to multiples of 80 and the reference code's 4096-token cap.
The Euler state and VAE stay in FP32; the transformer and text features use BF16.

This direct image lane is implemented and unqualified. It is not selected by
the causal serving engine or a hosted image catalog. KV cache, APCv2, token streaming,
and batch widths above one are not applicable. The adapter emits a status/route
receipt and rejects unsupported options. Prompt upsampling, LoRA, and the
CUDA-specific FP8/NVFP4 exports are outside this adapter's capabilities.

## Pinned download

The model repository is
[`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein),
revision `8454017f9ebaada30c998fccd9f67a80c7580f68`. The root BF16 package
contains 21 selected files totalling 20,320,953,181 bytes. Duplicate Diffusers
transformer/VAE exports and quantized variants are excluded. The text encoder
is included; the optional 9B prompt upsampler is separate.

```sh
PYTHONPATH="$PWD/src" python scripts/download_flux3_image_klein.py \
  --output /Users/Shared/models/flux-3-image-klein
```

Transfers resume through Hugging Face's local download cache. The downloader
checks every selected LFS SHA-256 or Git blob hash against the source-controlled
manifest before writing an atomic completion marker. Inspection rechecks the
hashes; the adapter then rejects changed files or extra processor inputs before
loading or using the backend. The optional `flux3-image` extra supplies its
runtime dependencies and the existing pinned mlx-vlm text implementation.

The reference inference code is Apache-2.0; the model weights use the separate
FLUX Kommunity License v1.0, retained as `LICENSE.md` in the download. The text
encoder weights are Apache-2.0. Exact source paths, revision, modifications,
and validation are recorded in `provenance/flux3-image-klein.json`.

## Direct Python use

Use the project's normal GPU ownership window before executing model math:

```python
from pathlib import Path
from mlx2.adapters.flux3_image import Flux3ImageKleinAdapter

adapter = Flux3ImageKleinAdapter("/Users/Shared/models/flux-3-image-klein")
result = adapter.generate_image(
    "A red ceramic teapot on a wooden table, soft daylight, product photograph.",
    width=512, height=512, steps=16, seed=7,
)
Path("teapot.png").write_bytes(result.data)
print(adapter.last_route_receipt)

edited = adapter.edit_image(
    "A blue ceramic teapot on a wooden table, soft daylight, product photograph.",
    ["teapot.png"], width=512, height=512, steps=16, seed=7,
)
Path("blue-teapot.png").write_bytes(edited.data)
```

Sides must be multiples of 16 between 64 and 4096, with at most 4 megapixels
per output. The direct default is 1024×1024, 16 steps, seed 0. Reference image
bytes are snapshotted in memory and hashed for the receipt.

## Focused validation

`scripts/run_flux3_image.py --model PATH --output NEW_DIRECTORY --device cpu
--load-only` checks strict local loading without GPU use. `--inspect-only`
verifies the artifact without importing MLX. Every run refuses a nonempty
evidence directory and writes an atomic receipt, including failures.

For GPU generation/edit probes, run this script under the established
`cpg_job.py` radio wrapper with `--lease --lock --require-radio` and a dedicated
CPG owner lock, then `scripts/run_with_gpu_locks.py` for the paired host locks.
Pass that dedicated lock as `--cpg-owner-lock`; its default is
`/tmp/mlx2-flux3-cpg.lock`. Both wrappers release ownership on exit. The probe
validates the live lease and matching owner receipts and checks thermals before
model loading. Use `--reference PATH` repeatedly for an edit. Receipts label the
run as focused functional smoke and do not promote qualification or defaults.

The CPU tests check artifact/request contracts, four-axis rotary math, BF16
timestep rounding, complete small-shape transformer parity with/without
references, full VAE encode/decode layout parity, and text padding/hidden-state
parity. The independent BFL controls require a clean checkout at the provenance
pin:

```sh
PYTHONPATH="$PWD/src" FLUX3_REFERENCE_ROOT=/path/to/pinned/bfl/source \
  python -m pytest tests/test_flux3_image_cpu.py tests/test_flux3_math_cpu.py
```

Formal qualification and the full publication preflight require their separate
gates and explicit preflight approval under `AGENTS.md`.

Generated media, artifact files, local settings and raw execution receipts are excluded from the public repository. The direct adapter remains unqualified.
