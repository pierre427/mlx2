"""Strict inspection and loading of EAGLE-3 (``speculators`` format) drafters.

Header-only inspection runs before any MLX payload load: the draft config is
mapped strictly, reconciled against the target config, and every safetensors
header is checked against the exact schema (bf16/f16/f32 weights, int64
``d2t``, bool ``t2d``).  No pickle and no checkpoint code is ever executed;
``eagle3.py`` shipped beside the weights is not read.  Provenance:
provenance/eagle3-drafter.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..runtime.drafters.eagle3 import ARCHITECTURE, Eagle3Config, expected_weight_schema
from .dflash2 import _decode_unique_json, _read_safetensors_header
from .draft_artifacts import inspect_files

DEFAULT_BLOCK_SIZE = 8


def _has_embed_tokens(path: Path) -> bool:
    index = path / "model.safetensors.index.json"
    if index.exists():
        mapping = _decode_unique_json(index.read_bytes(), "EAGLE-3 weight index")
        return "embed_tokens.weight" in (mapping.get("weight_map") or {})
    _, header, _ = _read_safetensors_header(path / "model.safetensors")
    return "embed_tokens.weight" in header


def inspect_drafter(path, target_path=None, *, target_config=None, block_size=DEFAULT_BLOCK_SIZE):
    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "EAGLE-3 config.json")
    if target_config is None:
        target_config = _decode_unique_json(
            (Path(target_path) / "config.json").read_bytes(), "target config"
        )
    text = target_config.get("text_config", target_config)
    args = Eagle3Config.from_speculators(
        config,
        num_target_layers=int(text["num_hidden_layers"]),
        block_size=block_size,
        has_embed_tokens=_has_embed_tokens(path),
    )
    if text.get("hidden_size") != args.target_hidden_size:
        raise ValueError("EAGLE-3 target hidden_size mismatch")
    if text.get("vocab_size") != args.vocab_size:
        raise ValueError("EAGLE-3 target vocab_size mismatch")
    verifier = ((config.get("speculators_config") or {}).get("verifier") or {})
    declared = verifier.get("architectures")
    if declared is not None and declared != target_config.get("architectures"):
        raise ValueError(
            f"EAGLE-3 drafter was trained for {declared!r}, target is "
            f"{target_config.get('architectures')!r}"
        )
    layer = config["transformer_layer_config"]
    dtype = layer.get("torch_dtype", config.get("torch_dtype", config.get("dtype")))
    schema = expected_weight_schema(args)
    record = inspect_files(
        path, raw,
        expected={name: shape for name, (_, shape) in schema.items()},
        dtype=dtype,
        label="EAGLE-3",
        dtype_overrides={name: code for name, (code, _) in schema.items() if code},
    )
    return {**record, "config": config, "args": args}


def content_revision(record):
    """Metadata content identity (config + tensor headers)."""
    return hashlib.sha256(
        json.dumps([record["config"], record["header_sha256"]], sort_keys=True).encode()
    ).hexdigest()


def load_drafter(record, target_model, *, runtime_quantization=None, draft_attention_windows=None):
    if runtime_quantization is not None:
        raise ValueError("EAGLE-3 runtime quantization is not supported")
    if draft_attention_windows is not None:
        raise ValueError("EAGLE-3 draft_attention_windows are not supported")
    import mlx.core as mx
    import numpy as np

    from ..runtime.drafters.eagle3 import Eagle3DraftModel

    # The fingerprint binds each shard's size and mtime; refuse a shard that
    # changed between header inspection and the payload load.
    for name, size, mtime in record["files"]:
        stat = (Path(record["path"]) / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError("EAGLE-3 shard changed since inspection")
    model = Eagle3DraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(Path(record["path"]) / name)))
    params, d2t, t2d = model.sanitize(weights)
    model.load_weights(list(params.items()), strict=True)
    model.load_vocab_map(
        None if d2t is None else np.asarray(d2t),
        None if t2d is None else np.asarray(t2d),
    )
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    params.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["ARCHITECTURE", "content_revision", "inspect_drafter", "load_drafter"]
