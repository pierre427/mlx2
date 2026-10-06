"""Direct Qwen3.5 4B/9B/27B/35B vision and optional MTP candidate.

This path uses the pinned local mlx-vlm model and processor. It is separate
from mlx2's ordinary serving registry until media state and draft acceptance
have source-bound qualification receipts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_ROOT = Path.home() / "Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_PATHS = (
    "mlx_vlm/models/qwen3_5", "mlx_vlm/models/qwen3_5_moe",
    "mlx_vlm/generate/dispatch.py", "mlx_vlm/prompt_utils.py",
    "mlx_vlm/speculative/drafters/__init__.py",
)
TOPOLOGIES = {
    "qwen3_5": {(32, 2560): (24, 1024), (32, 4096): (27, 1152),
                (64, 5120): (27, 1152)},
    "qwen3_5_moe": {(40, 2048): (27, 1152)},
}


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    family = config.get("model_type")
    text, vision = config.get("text_config"), config.get("vision_config")
    if family not in TOPOLOGIES or not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("expected Qwen3.5 dense or MoE vision checkpoint")
    topology = (text.get("num_hidden_layers"), text.get("hidden_size"))
    vision_topology = TOPOLOGIES[family].get(topology)
    if vision_topology is None:
        raise ValueError("unsupported Qwen3.5 vision text topology")
    depth, width = vision_topology
    if any(vision.get(key) != value for key, value in {
        "depth": depth, "hidden_size": width, "num_heads": 16, "patch_size": 16,
    }.items()):
        raise ValueError("unsupported Qwen3.5 vision tower topology")
    if (config.get("image_token_id"), config.get("video_token_id")) != (248056, 248057):
        raise ValueError("unsupported Qwen3.5 media token contract")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Qwen3.5 vision checkpoint needs an indexed weight map")
    required = {
        "vision_tower.blocks.0.attn.qkv.weight",
        f"vision_tower.blocks.{depth - 1}.attn.qkv.weight",
    }
    if not required <= weights.keys() or sum("vision_tower." in name for name in weights) < 12 * depth:
        raise ValueError("Qwen3.5 vision tower is incomplete")
    mtp_keys = [name for name in weights if name.startswith(("mtp.", "language_model.mtp."))]
    if mtp_keys and not any(name.endswith("mtp.layers.0.self_attn.q_proj.weight") for name in mtp_keys):
        raise ValueError("Qwen3.5 embedded MTP head is incomplete")
    names = sorted(set(weights.values()))
    files = []
    for name in names:
        if (not isinstance(name, str) or not name.endswith(".safetensors")
                or Path(name).is_absolute() or ".." in Path(name).parts):
            raise ValueError("unsafe Qwen3.5 weight shard path")
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing Qwen3.5 weight shard: {name}")
        stat = item.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    digest.update(json.dumps(files).encode())
    return {
        "path": str(path), "fingerprint": digest.hexdigest(), "files": files,
        "family": family, "vision_tensor_count": sum("vision_tower." in name for name in weights),
        "mtp_tensor_count": len(mtp_keys), "vision_candidate": True,
        "mtp_candidate": bool(mtp_keys), "qualified": False, "selected": False,
    }


class Qwen35VLMAdapter:
    """Explicit image/video generation bridge; no mlx2 serving route."""

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
                backend = load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
                self._backend = backend.load(self.artifact["path"], lazy=False,
                                             strict=True, trust_remote_code=False)
        return self._backend

    def generate_response(self, prompt: str, *, images: list[str | Path] = (),
                          videos: list[str | Path] = (), max_tokens: int = 256,
                          draft_model: str | Path | None = None) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Qwen3.5 VLM prompt must be nonempty")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("Qwen3.5 VLM max_tokens must be in 1..4096")
        if images and videos:
            raise ValueError("provide images or videos, not both")
        paths = [Path(value).expanduser().resolve() for value in (*images, *videos)]
        if any(not path.is_file() for path in paths):
            raise ValueError("Qwen3.5 VLM media file is missing")
        if draft_model is not None and not self.artifact["mtp_candidate"]:
            raise ValueError("artifact has no embedded MTP head")
        draft_path = None
        if draft_model is not None:
            draft_path = Path(draft_model).expanduser().resolve()
            if not draft_path.is_dir() or not (draft_path / "config.json").is_file():
                raise ValueError("MTP draft must be a local extracted artifact")
        model, processor = self._load()
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(processor)
        kwargs = {"max_tokens": max_tokens}
        if draft_path is not None:
            # The pinned source owns MTP extraction, verification and cache
            # rollback. The caller must explicitly supply an extracted draft.
            from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
            draft, kind = load_drafter(str(draft_path), kind="mtp")
            validate_drafter_compatibility(model, draft, kind)
            kwargs.update(draft_model=draft, draft_kind=kind)
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(
            processor, model.config, prompt, num_images=len(paths) if images else 0,
            video=[str(path) for path in paths] if videos else None, fps=1.0,
        )
        result = generate(model, processor, formatted,
                          image=[str(path) for path in paths] if images else None,
                          video=[str(path) for path in paths] if videos else None,
                          verbose=False, **kwargs)
        return {"text": result.text, "finish_reason": result.finish_reason,
                "artifact_fingerprint": self.artifact["fingerprint"],
                "route": "qwen3.5-vlm-direct-candidate",
                "mtp_requested": draft_model is not None}

    def close(self):
        self._backend = None
