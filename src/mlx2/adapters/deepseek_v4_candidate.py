"""DeepSeek V4 artifact preflight; execution remains deliberately unavailable.

The local source is an MLX-VLM multimodal model, not an mlx2 text adapter.
HISA, hyper connections, mixed expert pages, MTP, vision, and their cache
ownership have no qualified mlx2 integration.  This module makes the local
artifact identifiable without offering a misleading ordinary route.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from ._direct_mlx_vlm import load_backend
from .deepseek_v4_vision_layout import (
    expand_image_tokens, image_visible, merge_image_embeddings,
)


SOURCE_ROOT = "~/Desktop/mlx-uag/worktrees/agnes-vlm-support"
SOURCE_REVISION = "8a5e704e0fe43cd8654c144c4ecbd4c8aececeb5"
SOURCE_PATHS = (
    "mlx_vlm/models/deepseek_v4", "mlx_vlm/generate/dispatch.py",
    "mlx_vlm/prompt_utils.py", "mlx_vlm/utils.py",
    "mlx_vlm/speculative/drafters/deepseek_v4_mtp",
)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "deepseek_v4":
        raise ValueError("not a DeepSeek V4 artifact")
    expected = {
        "architectures": ["DeepseekV4ForCausalLM"], "num_hidden_layers": 43,
        "hidden_size": 4096, "num_attention_heads": 64,
        "vocab_size": 129280,
        "num_key_value_heads": 1, "n_routed_experts": 256,
        "num_experts_per_tok": 6, "num_hash_layers": 3,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("DeepSeek V4 local topology mismatch")
    vision = {"vision_n_layers": 32, "vision_dim": 1024,
              "vision_n_heads": 16, "vision_inter_dim": 2816,
              "vision_patch_size": 14, "vision_downsample_ratio": 3,
              "vision_max_n_token": 384}
    dspark = {"dspark_block_size": 5,
              "dspark_target_layer_ids": [40, 41, 42],
              "dspark_markov_rank": 256}
    if any(config.get(key) != value for key, value in {**vision, **dspark}.items()):
        raise ValueError("DeepSeek V4 vision/DSpark topology mismatch")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("DeepSeek V4 requires indexed shards")
    required = {"embed.weight", "head.weight", "hc_head_base",
                "vision.patch_embed.proj.weight", "aligner.w1.weight",
                "aligner.w2.weight", "image_start", "image_end",
                "image_newline", "image_pad"}
    for layer in range(43):
        required.update({f"layers.{layer}.attn.wkv.weight",
                         f"layers.{layer}.attn.q_norm.weight"})
    for layer in range(32):
        prefix = f"vision.blocks.{layer}."
        required.update(prefix + name for name in (
            "norm1.weight", "attn.wqkv.weight", "attn.wo.weight",
            "norm2.weight", "mlp.w1.weight", "mlp.w2.weight",
        ))
    if not required <= weight_map.keys():
        raise ValueError("DeepSeek V4 indexed tensor topology is incomplete")
    components = {
        "vision_tensors": sum(key.startswith("vision.") for key in weight_map),
        "image_tokens": sum(key.startswith("image_") for key in weight_map),
        "hyper_connection_tensors": sum("hc_" in key for key in weight_map),
        "mtp_tensors": sum(key.startswith("mtp.") for key in weight_map),
    }
    mtp_layers = {int(key.split(".")[1]) for key in weight_map
                  if key.startswith("mtp.") and key.split(".")[1].isdigit()}
    if any(value == 0 for value in components.values()) or mtp_layers != {0, 1, 2} or config.get("num_nextn_predict_layers") != 3:
        raise ValueError("DeepSeek V4 vision/MTP/hyper-connection components are incomplete")
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("unsafe DeepSeek V4 shard path")
        item = path / name
        if not item.is_file() or not item.resolve().is_relative_to(path):
            raise ValueError(f"missing DeepSeek V4 shard: {name}")
        stat = item.stat(); records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode()); digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {
        "model_type": "deepseek_v4", "vocab_size": config["vocab_size"],
        "indexed_tensors": len(weight_map),
        "shards": len(names), "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
        "components": components, "mtp_layers": sorted(mtp_layers),
        "implementation": "direct-mlx-vlm-text-candidate",
        "serving_route": None,
        "blocker": "mlx2 serving requires HISA, hyper-connection, compressed/rotating/pooling cache lifecycle and mixed expert quantization; pinned source omits vision embedding and drops MTP from target weights",
    }


class DeepSeekV4Candidate:
    """Direct text-only candidate; image/MTP deliberately fail closed."""

    def __init__(self, model_path: str | Path, *, backend=None):
        self.artifact = inspect_artifact(model_path)
        self.backend = backend or load_backend(SOURCE_ROOT, SOURCE_REVISION, SOURCE_PATHS)
        self.model, self.processor = self.backend.load(self.artifact["identity"]["path"], strict=True)
        if getattr(self.model, "model_type", None) != "deepseek_v4":
            raise ValueError("MLX-VLM loaded a different model family")

    def generate(self, prompt: str, *, image=None, mtp: bool = False,
                 max_tokens: int = 128, temperature: float = 0.0) -> dict:
        if image is not None or mtp:
            raise NotImplementedError("pinned DeepSeek V4 MLX-VLM source drops vision and MTP weights")
        if not isinstance(prompt, str) or not 0 < max_tokens <= 4096 or temperature < 0:
            raise ValueError("invalid DeepSeek V4 generation parameters")
        formatted = self.backend.apply_chat_template(self.processor, self.model.config, prompt)
        result = self.backend.generate(model=self.model, processor=self.processor,
                                       prompt=formatted, max_tokens=max_tokens,
                                       temperature=temperature, verbose=False)
        return {"text": result.text, "route_receipt": {
            "model_type": "deepseek_v4", "execution": "direct-mlx-vlm-text-candidate",
            "source_revision": SOURCE_REVISION, "qualification": "pending",
            "modalities": ["text"], "mtp": False,
        }}

    def split_embedded_mtp(self, output: str | Path) -> Path:
        """Prepare the pinned in-zoo drafter; no mlx2 speculative route is implied."""
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise ValueError("MTP output path already exists")
        if not destination.parent.is_dir():
            raise ValueError("MTP output parent directory is missing")
        if getattr(self.backend, "__name__", None) != "mlx_vlm":
            raise RuntimeError("MTP split requires the pinned MLX-VLM backend")
        module = importlib.import_module("mlx_vlm.speculative.drafters.deepseek_v4_mtp.split")
        if not Path(module.__file__).resolve().is_relative_to(Path(SOURCE_ROOT).resolve()):
            raise RuntimeError("MTP splitter did not resolve to pinned source")
        return module.split_deepseek_v4_mtp(
            self.artifact["identity"]["path"], str(destination), force_download=False,
        )

    def prepare_vision_prefill(self, prompt_tokens: list[int], image_token_id: int,
                               images: list[str | Path]):
        """Build the official image span and patches without claiming execution."""
        return expand_image_tokens(prompt_tokens, image_token_id, images,
                                   self.artifact["vocab_size"])

    def load_vision_components(self):
        """Return a separate strict Q4 ViT/aligner candidate and sentinels."""
        from .deepseek_v4_vision_weights import load_vision_components

        return load_vision_components(self.artifact["identity"]["path"])

    def assemble_vision_prefill(self, token_embeddings, prepared, aligned, sentinels,
                                expanded_tokens):
        """Merge image embeddings and return sparse visibility for decoder work."""
        if len(token_embeddings) != len(expanded_tokens):
            raise ValueError("vision prefill token/embedding length mismatch")
        merged = merge_image_embeddings(token_embeddings, prepared, aligned, sentinels)
        visible = image_visible(expanded_tokens, self.artifact["vocab_size"])
        return merged, visible
