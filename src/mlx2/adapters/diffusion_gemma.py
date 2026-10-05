"""Direct DiffusionGemma candidate, outside the causal-token serving resolver.

The denoising model and sampler remain in the pinned optional mlx-vlm source.
Artifact inspection reads JSON and safetensors headers only. Execution is an
explicit method call and has no serving qualification. See provenance.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_ROOT = Path.home() / "Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_PATHS = ("mlx_vlm/models/diffusion_gemma", "mlx_vlm/generate/diffusion.py",
                "mlx_vlm/generate/dispatch.py", "mlx_vlm/prompt_utils.py")


def _object(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _header(path: Path) -> tuple[dict, bytes]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        length_data = stream.read(8)
        if len(length_data) != 8:
            raise ValueError(f"truncated safetensors shard {path.name}")
        length = struct.unpack("<Q", length_data)[0]
        if not 0 < length <= min(64 << 20, size - 8):
            raise ValueError(f"invalid safetensors header length in {path.name}")
        raw = stream.read(length)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate safetensors tensor {key!r}")
            result[key] = value
        return result
    header = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header in {path.name}")
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length):
            raise ValueError(f"invalid tensor offsets for {name}")
    return header, raw


def inspect_diffusion_gemma(path: str | Path) -> dict:
    """Validate topology and shard/header closure without reading weight data."""
    root = Path(path).expanduser().resolve()
    config = _object(root / "config.json")
    text, vision = config.get("text_config"), config.get("vision_config")
    if (config.get("model_type") != "diffusion_gemma"
            or config.get("architectures") != ["DiffusionGemmaForBlockDiffusion"]
            or config.get("canvas_length") != 256
            or not isinstance(text, dict) or not isinstance(vision, dict)):
        raise ValueError("expected DiffusionGemma block-diffusion artifact")
    if any(text.get(k) != v for k, v in {
        "num_hidden_layers": 30, "hidden_size": 2816, "num_experts": 128,
        "top_k_experts": 8, "sliding_window": 1024, "vocab_size": 262144,
    }.items()):
        raise ValueError("unsupported DiffusionGemma text topology")
    layer_types = text.get("layer_types")
    if (not isinstance(layer_types, list) or len(layer_types) != 30
            or any(kind != ("full_attention" if i % 6 == 5 else "sliding_attention")
                   for i, kind in enumerate(layer_types))):
        raise ValueError("unsupported DiffusionGemma attention pattern")
    if any(vision.get(k) != v for k, v in {
        "num_hidden_layers": 27, "hidden_size": 1152,
        "num_attention_heads": 16, "patch_size": 16,
    }.items()):
        raise ValueError("unsupported DiffusionGemma vision topology")
    generation = _object(root / "generation_config.json")
    if generation.get("max_denoising_steps") != 48:
        raise ValueError("unsupported DiffusionGemma denoising configuration")
    index = _object(root / "model.safetensors.index.json")
    weights = index.get("weight_map")
    shards = {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}
    if not isinstance(weights, dict) or set(weights.values()) != shards:
        raise ValueError("DiffusionGemma index must cover four shards")
    for name in ("tokenizer.json", "processor_config.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing DiffusionGemma processor file {name}")
    required = {
        "model.decoder.embed_tokens.weight": [262144, 704],
        "model.encoder.embed_vision.embedding_projection.weight": [2816, 144],
        "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": [1152, 1152],
    }
    for i in range(30):
        required[f"model.decoder.layers.{i}.experts.gate_up_proj.weight"] = None
        required[f"model.decoder.layers.{i}.experts.down_proj.weight"] = None
        required[f"model.decoder.layers.{i}.router.proj.weight"] = None
    digest = hashlib.sha256()
    for name in ("config.json", "generation_config.json", "model.safetensors.index.json",
                 "tokenizer.json", "processor_config.json", "tokenizer_config.json"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    keys = set()
    sizes = {}
    for shard in sorted(shards):
        file = root / shard
        if not file.is_file():
            raise ValueError(f"missing DiffusionGemma shard {shard}")
        header, raw = _header(file)
        actual = set(header) - {"__metadata__"}
        if keys & actual or any(weights.get(key) != shard for key in actual):
            raise ValueError(f"DiffusionGemma index disagrees with {shard}")
        keys.update(actual)
        for key, shape in required.items():
            if weights.get(key) == shard and (key not in header or
                                           (shape is not None and header[key].get("shape") != shape)):
                raise ValueError(f"missing or mismatched DiffusionGemma tensor {key}")
        size = file.stat().st_size
        sizes[shard] = size
        digest.update(shard.encode())
        digest.update(str(size).encode())
        digest.update(hashlib.sha256(raw).digest())
    if keys != set(weights) or not set(required) <= keys:
        raise ValueError("DiffusionGemma tensor index is incomplete")
    return {"path": root, "fingerprint": digest.hexdigest(), "tensor_count": len(keys),
            "shard_sizes": sizes, "source_revision": SOURCE_REVISION,
            "qualified": False, "selected": False, "lifecycle": "direct-block-diffusion"}


@dataclass(frozen=True, slots=True)
class DiffusionText:
    text: str
    artifact_fingerprint: str
    finish_reason: str | None
    canvas_tokens: int
    denoising_steps: int


class DiffusionGemmaAdapter:
    """Explicit direct candidate; never exposes an ordinary causal route."""

    def __init__(self, path: str | Path, *, backend_factory: Callable | None = None):
        self.artifact = inspect_diffusion_gemma(path)
        self._backend_factory = backend_factory
        self._backend: tuple[Any, Any] | None = None

    def _load(self):
        if self._backend is None:
            if self._backend_factory is not None:
                self._backend = self._backend_factory(self.artifact["path"])
            else:
                from ._direct_mlx_vlm import load_backend
                backend = load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
                self._backend = backend.load(str(self.artifact["path"]),
                                             lazy=False, strict=True,
                                             trust_remote_code=False)
        return self._backend

    def generate_text(self, prompt: str, *, image: str | Path | None = None,
                      video: str | Path | None = None, max_tokens: int = 256,
                      max_denoising_steps: int = 48,
                      sampler: str = "entropy-bound") -> DiffusionText:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("DiffusionGemma prompt must be nonempty")
        if image is not None and video is not None:
            raise ValueError("supply one DiffusionGemma media kind at a time")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("DiffusionGemma max_tokens must be 1..2048")
        if type(max_denoising_steps) is not int or not 1 <= max_denoising_steps <= 48:
            raise ValueError("DiffusionGemma denoising steps must be 1..48")
        if sampler not in {"entropy-bound", "confidence-threshold"}:
            raise ValueError("unsupported DiffusionGemma sampler")
        media = image if image is not None else video
        if media is not None and not Path(media).expanduser().is_file():
            raise ValueError("DiffusionGemma media file is missing")
        model, processor = self._load()
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(processor, model.config, prompt,
                                        num_images=int(image is not None),
                                        video=[str(video)] if video is not None else None,
                                        fps=1.0)
        result = generate(model, processor, formatted,
                          image=str(image) if image is not None else None,
                          video=str(video) if video is not None else None,
                          verbose=False, max_tokens=max_tokens,
                          max_denoising_steps=max_denoising_steps,
                          diffusion_sampler=sampler)
        return DiffusionText(result.text, self.artifact["fingerprint"],
                             result.finish_reason, result.diffusion_canvas_tokens,
                             result.diffusion_denoising_steps)

    def close(self):
        self._backend = None
