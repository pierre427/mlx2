"""Strict inspection and loading of the Laguna XS 2.1 causal DFlash drafter.

Header-only inspection runs before any MLX payload load; the target's
embedding and vocabulary head are bound only after a strict weight load.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..runtime.drafters.laguna_dflash import LagunaDFlashConfig, expected_weight_shapes
from .dflash2 import _decode_unique_json
from .draft_artifacts import inspect_files


def inspect_drafter(path, *, target_path=None, target_config=None):
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "draft config.json")
    args = LagunaDFlashConfig.from_hf(config)
    if target_config is None:
        target_config = json.loads((Path(target_path) / "config.json").read_text())
    text = target_config.get("text_config", target_config)
    for name in ("hidden_size", "vocab_size"):
        if text[name] != getattr(args, name):
            raise ValueError(f"Laguna DFlash target {name} mismatch")
    if text["num_hidden_layers"] != args.num_target_layers:
        raise ValueError("Laguna DFlash target layer count mismatch")
    record = inspect_files(
        path, raw, expected=expected_weight_shapes(args),
        dtype=config.get("dtype", config.get("torch_dtype")), label="Laguna DFlash",
    )
    return {**record, "config": config, "args": args}


def load_drafter(record, target_model):
    import mlx.core as mx

    from ..runtime.drafters.laguna_dflash import LagunaDFlashDraftModel

    model = LagunaDFlashDraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(Path(record["path"]) / name)))
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["inspect_drafter", "load_drafter"]
