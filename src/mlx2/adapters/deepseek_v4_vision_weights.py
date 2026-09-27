"""Strict mixed-Q4 DeepSeek V4 vision weight plan and lazy MLX candidate loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


IMAGE_SENTINELS = ("image_start", "image_pad", "image_newline", "image_end")


@dataclass(frozen=True)
class VisionWeightPlan:
    path: Path
    tensor_shards: dict[str, str]
    quantized_modules: dict[str, dict]


def plan_vision_weights(model_path: str | Path) -> VisionWeightPlan:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    quant = config.get("quantization")
    if config.get("model_type") != "deepseek_v4" or not isinstance(weights, dict) or not isinstance(quant, dict):
        raise ValueError("expected indexed DeepSeek V4 Q4 artifact")
    modules = {name: entry for name, entry in quant.items()
               if name.startswith(("vision.", "aligner."))}
    expected_modules = {"vision.patch_embed.proj", "aligner.w1", "aligner.w2"}
    for layer in range(32):
        expected_modules.update(f"vision.blocks.{layer}.{part}" for part in
                                ("attn.wqkv", "attn.wo", "mlp.w1", "mlp.w2"))
    if set(modules) != expected_modules or any(
        entry != {"group_size": 64, "bits": 4, "mode": "affine"}
        for entry in modules.values()
    ):
        raise ValueError("DeepSeek V4 vision mixed-Q4 module plan differs from checkpoint")
    names = {name for name in weights if name.startswith(("vision.", "aligner."))
             or name in IMAGE_SENTINELS}
    for module in expected_modules:
        if not {module + suffix for suffix in (".weight", ".scales", ".biases")} <= names:
            raise ValueError(f"incomplete Q4 vision module {module}")
    if not set(IMAGE_SENTINELS) <= names or "vision.norm.weight" not in names:
        raise ValueError("missing vision norm or image sentinel")
    for layer in range(32):
        for norm in ("norm1", "norm2"):
            if f"vision.blocks.{layer}.{norm}.weight" not in names:
                raise ValueError("missing vision block norm")
    shards = {name: weights[name] for name in names}
    for shard in set(shards.values()):
        if not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts or not shard.endswith(".safetensors"):
            raise ValueError("unsafe vision shard path")
        if not (path / shard).is_file():
            raise ValueError(f"missing vision shard {shard}")
    return VisionWeightPlan(path, shards, modules)


def materialize_vision_weights(
    plan: VisionWeightPlan,
    *,
    model_factory: Callable[[], object],
    quantize_model: Callable[[object, dict[str, dict]], None],
    read_tensors: Callable[[Path, dict[str, str]], dict],
) -> tuple[object, dict]:
    """Injectable strict assembly; fake readers permit CPU-only contract tests."""
    model = model_factory()
    quantize_model(model, plan.quantized_modules)
    tensors = read_tensors(plan.path, plan.tensor_shards)
    if set(tensors) != set(plan.tensor_shards):
        raise ValueError("vision tensor reader did not return the complete plan")
    sentinels = {name: tensors.pop(name) for name in IMAGE_SENTINELS}
    model.load_weights(list(tensors.items()), strict=True)
    return model, sentinels


def load_vision_components(model_path: str | Path):
    """Lazy structural loader; numerical qualification and decoder join pending."""
    plan = plan_vision_weights(model_path)
    import mlx.core as mx
    import mlx.nn as nn
    from safetensors import safe_open

    from ..runtime.models.deepseek_v4_vision import VisionArgs, VisionComponents

    def read_tensors(path: Path, selected: dict[str, str]) -> dict:
        by_shard: dict[str, list[str]] = {}
        for name, shard in selected.items():
            by_shard.setdefault(shard, []).append(name)
        output = {}
        for shard, names in by_shard.items():
            with safe_open(path / shard, framework="mlx") as stream:
                output.update({name: mx.array(stream.get_tensor(name)) for name in names})
        return output

    def quantize(model, modules):
        nn.quantize(model, group_size=64, bits=4, mode="affine",
                    class_predicate=lambda path, module: modules.get(path, False))

    return materialize_vision_weights(
        plan, model_factory=lambda: VisionComponents(VisionArgs()),
        quantize_model=quantize, read_tensors=read_tensors,
    )
