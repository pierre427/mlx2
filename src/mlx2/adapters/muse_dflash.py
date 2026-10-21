"""Strict inspection and loading of Muse Glimmer's original DFlash drafter.

``MuseGlimmerAssistantModel`` checkpoints (BF16, and affine Q8/Q4 exports)
are inspected from safetensors headers only before any MLX payload load:
exact tensor names, dtype classes, shapes, shard agreement and non-overlap.
The fingerprint binds config, shard stats and header digests.  The target's
embedding and logits tail are bound only after a strict weight load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dflash2 import _decode_unique_json, _read_safetensors_header

_FLOAT = {"BF16", "F16", "F32"}


def is_assistant_artifact(path) -> bool:
    try:
        config = json.loads((Path(path).expanduser() / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("architectures") == ["MuseGlimmerAssistantModel"]


def inspect_drafter(path, target):
    from ..runtime.drafters.muse_dflash import MuseDFlashConfig, expected_weights

    path = Path(path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "draft config.json")
    target_config = json.loads((Path(target) / "config.json").read_text())
    text = target_config.get("text_config", target_config)
    args = MuseDFlashConfig.from_hf(config, target_text=text)
    expected = expected_weights(args)
    index = path / "model.safetensors.index.json"
    digest = hashlib.sha256(raw)
    if index.is_file():
        content = index.read_bytes()
        digest.update(content)
        mapping = _decode_unique_json(content, "draft weight index").get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid Muse DFlash weight index")
        names = sorted(set(mapping.values()))
    else:
        mapping, names = None, ["model.safetensors"]
    files, observed, header_digests = [], {}, []
    for name in names:
        if (
            not isinstance(name, str) or "/" in name or "\\" in name
            or name.startswith(".") or not name.endswith(".safetensors")
            or not (path / name).is_file()
        ):
            raise ValueError("Invalid Muse DFlash shard path")
        stat = (path / name).stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
        digest.update(json.dumps(files[-1]).encode())
        header_raw, header, payload = _read_safetensors_header(path / name)
        header_digests.append(hashlib.sha256(header_raw).hexdigest())
        ranges = []
        for key, record in header.items():
            if key == "__metadata__":
                continue
            if key in observed:
                raise ValueError(f"Duplicate Muse DFlash tensor across shards: {key}")
            if mapping is not None and mapping.get(key) != name:
                raise ValueError(f"Muse DFlash weight index/shard mismatch: {key}")
            offsets = record.get("data_offsets")
            if not (isinstance(offsets, list) and len(offsets) == 2
                    and 0 <= offsets[0] <= offsets[1] <= payload):
                raise ValueError(f"Invalid Muse DFlash tensor offsets: {key}")
            ranges.append((offsets[0], offsets[1], key))
            observed[key] = (record.get("dtype"), record.get("shape"))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] > current[0]:
                raise ValueError(f"Overlapping Muse DFlash payloads: {previous[2]}, {current[2]}")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:8]
        extra = sorted(set(observed) - set(expected))[:8]
        raise ValueError(f"Muse DFlash checkpoint schema mismatch (missing={missing}, extra={extra})")
    for key, (kind, shape) in expected.items():
        dtype, observed_shape = observed[key]
        if observed_shape != shape or (dtype not in _FLOAT if kind == "float" else dtype != kind):
            raise ValueError(f"Muse DFlash tensor dtype/shape mismatch: {key}")
    digest.update(json.dumps(header_digests).encode())
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": files,
        "header_sha256": header_digests,
        "config": config,
        "args": args,
    }


def load_drafter(record, target_model):
    import mlx.core as mx
    import mlx.nn as nn

    from ..runtime.drafters.muse_dflash import MuseDFlashDraftModel

    model = MuseDFlashDraftModel(record["args"])
    weights = {}
    for name, _, _ in record["files"]:
        weights.update(mx.load(str(Path(record["path"]) / name)))
    weights = model.sanitize(weights)
    quant = record["args"].quantization
    if quant is not None:
        nn.quantize(
            model,
            group_size=int(quant["group_size"]),
            bits=int(quant["bits"]),
            class_predicate=lambda name, module: isinstance(module, nn.Linear)
            and f"{name}.scales" in weights,
        )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["inspect_drafter", "is_assistant_artifact", "load_drafter"]
