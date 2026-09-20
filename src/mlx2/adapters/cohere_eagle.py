"""Strict inspection and loading of the North Mini Code Cohere EAGLE drafter.

Header-only inspection runs before any MLX payload load; the target's shared
embedding/vocabulary head is bound only after a strict weight load.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..runtime.drafters.cohere_eagle import CohereEagleConfig, expected_weight_shapes
from .dflash2 import _decode_unique_json
from .draft_artifacts import inspect_files


def inspect_drafter(path, *, target_path=None, target_config=None, block_size=8):
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "draft config.json")
    if target_config is None:
        target_config = json.loads((Path(target_path) / "config.json").read_text())
    text = target_config.get("text_config", target_config)
    args = CohereEagleConfig.from_hf(
        config, num_target_layers=int(text["num_hidden_layers"]), block_size=block_size
    )
    for name in ("hidden_size", "vocab_size"):
        if text[name] != getattr(args, name):
            raise ValueError(f"Cohere EAGLE target {name} mismatch")
    record = inspect_files(
        path, raw, expected=expected_weight_shapes(args),
        dtype=config.get("dtype", config.get("torch_dtype")), label="Cohere EAGLE",
    )
    return {**record, "config": config, "args": args}


def load_drafter(record, target_model):
    import mlx.core as mx

    from ..runtime.drafters.cohere_eagle import CohereEagleDraftModel

    model = CohereEagleDraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(Path(record["path"]) / name)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["inspect_drafter", "load_drafter"]
