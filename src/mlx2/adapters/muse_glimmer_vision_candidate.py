"""Muse Glimmer image candidate using a pinned local vision implementation.

This direct path does not select a serving route. The ordinary mlx2 Muse
adapter continues to own text generation and its existing cache contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

SOURCE_ROOT = Path.home() / "Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_PATHS = (
    "mlx_vlm/models/muse_glimmer",
    "mlx_vlm/generate/dispatch.py",
    "mlx_vlm/prompt_utils.py",
)


def inspect_artifact(model_path: str | Path) -> dict:
    """Check the whole local vision stack without opening weight payloads."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    vision = config.get("vision_config")
    text = config.get("text_config")
    if config.get("model_type") != "muse_glimmer" or not isinstance(vision, dict) or not isinstance(text, dict):
        raise ValueError("expected a Muse Glimmer vision target")
    if any("Draft" in name for name in config.get("architectures", ())):
        raise ValueError("Muse draft is not a vision target")
    if (config.get("image_token_id"), config.get("video_token_id")) != (200092, 200091):
        raise ValueError("unsupported Muse media token contract")
    if (vision.get("num_hidden_layers"), vision.get("hidden_size"),
            vision.get("patch_size"), vision.get("merge_size")) != (50, 1536, 14, 2):
        raise ValueError("unsupported Muse vision tower topology")
    if (text.get("num_hidden_layers"), text.get("hidden_size")) != (52, 6656):
        raise ValueError("unsupported Muse text tower topology")

    index_file = path / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Muse vision target needs an indexed weight map")
    # The original HF checkpoint has a model. prefix; MLX conversions do not.
    keys = {key.removeprefix("model.") for key in weights}
    required = {
        "vision_tower.patch_embedder.patch_embedding.weight",
        "vision_tower.layers.0.attn.q_proj.weight",
        "vision_tower.layers.49.attn.q_proj.weight",
        "vision_adapter.fc1.weight",
        "vision_adapter.fc2.weight",
        "vision_projection.weight",
    }
    if not required <= keys or sum(key.startswith("vision_tower.") for key in keys) < 790:
        raise ValueError("Muse vision tower, adapter, or projection is incomplete")
    if not (path / "chat_template.jinja").is_file() or not (path / "tokenizer.json").is_file():
        raise ValueError("Muse vision target needs a local chat template and tokenizer")

    files = []
    for name in sorted(set(weights.values())):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("unsafe Muse shard path")
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing Muse shard: {name}")
        stat = item.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "processor_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    digest.update(json.dumps(files).encode())
    return {
        "path": str(path), "fingerprint": digest.hexdigest(), "files": files,
        "vision_tensor_count": sum(key.startswith("vision_tower.") for key in keys),
        "vision_candidate": True, "qualified": False, "selected": False,
    }


class MuseGlimmerVisionCandidate:
    """Standalone image generation; no APCv2 or ordinary serving claims."""

    def __init__(self, model_path: str | Path, *, backend_factory: Callable | None = None):
        self.artifact = inspect_artifact(model_path)
        self._backend_factory = backend_factory
        self._backend: tuple[Any, Any] | None = None

    def _load(self):
        if self._backend is None:
            if self._backend_factory is not None:
                self._backend = self._backend_factory(Path(self.artifact["path"]))
            else:
                from ._direct_mlx_vlm import load_backend
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                backend = load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
                self._backend = backend.load(self.artifact["path"], lazy=False,
                                             strict=True, trust_remote_code=False)
        return self._backend

    def generate_response(self, prompt: str, *, images: list[str | Path],
                          max_tokens: int = 256) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Muse vision prompt must be nonempty")
        if "<|patch|>" in prompt or "<|video|>" in prompt:
            raise ValueError("Muse media markers are inserted by the chat template")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("Muse max_tokens must be in 1..4096")
        if not images:
            raise ValueError("Muse vision candidate needs at least one image")
        from ._direct_mlx_vlm import validate_media_paths
        paths = validate_media_paths(images, kind="image")
        model, processor = self._load()
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(processor)
        content = [{"type": "image"} for _ in paths]
        content.append({"type": "text", "text": prompt})
        rendered = processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True,
            reasoning_strength="low",
        )
        if rendered.count("<|patch|>") != len(paths):
            raise ValueError("Muse chat template did not bind every image")
        if not rendered.endswith("<|start|>assistant"):
            raise ValueError("Muse chat template did not end at the assistant header")
        # The template ends at an open assistant header. The no-tool text
        # adapter uses this recipient to request a direct, client-visible reply.
        rendered += " to=user<|message|>"
        from mlx_vlm.generate import generate
        result = generate(model, processor, rendered, image=paths,
                          verbose=False, max_tokens=max_tokens)
        from .muse_glimmer_output import MuseOutputParser
        parser = MuseOutputParser(chat=True)
        events = parser.finish(result.text, result.finish_reason or "length")
        answer = "".join(event.get("content", "") for event in events)
        if not answer.strip():
            raise ValueError("Muse vision output contained no client-visible answer")
        return {
            "text": answer,
            "finish_reason": result.finish_reason,
            "artifact_fingerprint": self.artifact["fingerprint"],
            "route": "muse-glimmer-vision-direct-candidate",
            "qualified": False,
        }

    def close(self):
        self._backend = None
