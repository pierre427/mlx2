"""Direct Flash-Next VLM and embedded-MTP candidates for the indexed PLE artifact.

Neither candidate changes the ordinary Flash-Next resolver's ple_rows.bin gate.
The vision math and standalone MTP splitter live in pinned optional mlx-vlm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_ROOT = Path.home() / "Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_PATHS = (
    "mlx_vlm/models/qwen4_exp", "mlx_vlm/generate/dispatch.py",
    "mlx_vlm/split_mtp.py", "mlx_vlm/speculative/drafters/qwen4_exp_mtp",
    "mlx_vlm/speculative/drafters/__init__.py", "mlx_vlm/prompt_utils.py",
)


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


def inspect_flash_next_vlm(path: str | Path) -> dict:
    """Inspect the indexed PLE/VLM/MTP checkpoint without loading tensors."""
    root = Path(path).expanduser().resolve()
    config = _object(root / "config.json")
    text, vision = config.get("text_config"), config.get("vision_config")
    if (config.get("model_type") != "qwen4_exp"
            or config.get("architectures") != ["Qwen4ExpForConditionalGeneration"]
            or not isinstance(text, dict) or not isinstance(vision, dict)):
        raise ValueError("expected Flash-Next VLM checkpoint")
    if any(text.get(k) != v for k, v in {
        "num_hidden_layers": 48, "hidden_size": 2560,
        "num_experts": 512, "num_experts_per_tok": 10,
        "mtp_num_hidden_layers": 1, "vocab_size": 248320,
        "full_attention_interval": 4,
    }.items()):
        raise ValueError("unsupported Flash-Next text topology")
    if any(vision.get(k) != v for k, v in {
        "depth": 27, "hidden_size": 1152, "num_heads": 16,
        "out_hidden_size": 2560, "patch_size": 16,
        "spatial_merge_size": 2, "temporal_patch_size": 2,
    }.items()):
        raise ValueError("unsupported Flash-Next vision topology")
    if config.get("ngram_table") is not None:
        raise ValueError("expected indexed Flash-Next PLE table layout")
    index = _object(root / "model.safetensors.index.json")
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Flash-Next VLM has no indexed weights")
    required = {
        "language_model.model.embed_tokens.weight",
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
        "vision_tower.blocks.26.attn.qkv.weight",
        "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.pre_fc_norm_embedding.weight",
    }
    if not required.issubset(weights):
        raise ValueError("Flash-Next index lacks embedded vision or MTP tensors")
    vision_keys = [key for key in weights if key.startswith("vision_tower.")]
    mtp_keys = [key for key in weights if key.startswith("mtp.")]
    ple_keys = [key for key in weights if "ngram_embedding.shard_" in key]
    if (len(vision_keys), len(mtp_keys), len(ple_keys)) != (333, 78, 384):
        raise ValueError("Flash-Next indexed PLE/vision/MTP tensor counts differ")
    names = sorted(set(weights.values()))
    if len(names) != 22:
        raise ValueError("Flash-Next VLM shard count differs")
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("unsafe Flash-Next shard path")
        file = root / name
        if not file.is_file() or file.stat().st_size < 8:
            raise ValueError(f"missing Flash-Next shard {name}")
        stat = file.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing Flash-Next media processor file {name}")
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "preprocessor_config.json"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {"path": root, "fingerprint": digest.hexdigest(),
            "tensor_count": len(weights), "shards": len(names),
            "vision_tensors": len(vision_keys), "mtp_tensors": len(mtp_keys),
            "ple_index_tensors": len(ple_keys), "ple_rows_bin": (root / "ple_rows.bin").is_file(),
            "source_revision": SOURCE_REVISION, "qualified": False, "selected": False}


def _require_source():
    from ._direct_mlx_vlm import load_backend
    return load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)


def split_candidate_mtp(path: str | Path, output: str | Path) -> Path:
    """Offline MTP extraction using the pinned splitter; never selected by serving."""
    artifact = inspect_flash_next_vlm(path)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"candidate MTP output already exists: {target}")
    _require_source()
    from mlx_vlm.split_mtp import split_mtp
    return split_mtp(str(artifact["path"]), str(target), model_type="qwen4_exp")


@dataclass(frozen=True, slots=True)
class VisionText:
    text: str
    artifact_fingerprint: str
    finish_reason: str | None
    mtp_requested: bool


class FlashNextVisionCandidate:
    """Explicit direct image/video path for the indexed PLE VLM artifact."""

    def __init__(self, path: str | Path, *, backend_factory: Callable | None = None):
        self.artifact = inspect_flash_next_vlm(path)
        self._backend_factory = backend_factory
        self._backend: tuple[Any, Any] | None = None

    def _load(self):
        if self._backend is None:
            if self._backend_factory is not None:
                self._backend = self._backend_factory(self.artifact["path"])
            else:
                backend = _require_source()
                self._backend = backend.load(str(self.artifact["path"]), lazy=False,
                                             strict=True, trust_remote_code=False)
        return self._backend

    def generate_text(self, prompt: str, *, image: str | Path | None = None,
                      video: str | Path | None = None, max_tokens: int = 256,
                      draft_model: str | Path | None = None) -> VisionText:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Flash-Next VLM prompt must be nonempty")
        if (image is None) == (video is None):
            raise ValueError("supply exactly one image or video")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("Flash-Next VLM max_tokens must be 1..2048")
        media = Path(image if image is not None else video).expanduser().resolve()
        if not media.is_file():
            raise ValueError("Flash-Next VLM media file is missing")
        if draft_model is not None and not Path(draft_model).expanduser().is_dir():
            raise ValueError("Flash-Next extracted MTP draft is missing")
        model, processor = self._load()
        from ..runtime.chat_templates import secure_model_chat_templates
        secure_model_chat_templates(processor)
        kwargs = {"max_tokens": max_tokens}
        if draft_model is not None:
            from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
            draft, kind = load_drafter(str(Path(draft_model).expanduser().resolve()), kind="mtp")
            validate_drafter_compatibility(model, draft, kind)
            kwargs.update(draft_model=draft, draft_kind=kind)
        from mlx_vlm.generate import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        formatted = apply_chat_template(processor, model.config, prompt,
                                        num_images=int(image is not None),
                                        video=[str(media)] if video is not None else None,
                                        fps=1.0)
        result = generate(model, processor, formatted,
                          image=str(media) if image is not None else None,
                          video=str(media) if video is not None else None,
                          verbose=False, **kwargs)
        return VisionText(result.text, self.artifact["fingerprint"],
                          result.finish_reason, draft_model is not None)

    def close(self):
        self._backend = None
