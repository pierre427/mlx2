"""Strict inspection and loading of Inkling 2's bundled DSpark drafter.

The drafter ships inside the target checkpoint as ``mtp.safetensors`` with its
geometry in ``config.json``'s ``mtp_config``.  :func:`inspect_drafter` reads
only JSON and the safetensors header (exact names, BF16 dtypes, shapes,
non-overlapping payloads) and checks the ``hf_quant_config.json`` ``mtp``
group is a plain BF16 cast of exactly those tensors.  :func:`load_drafter`
loads the payload with ``strict=True`` and binds (folds the Markov head into)
the target LM head.  Provenance: provenance/inkling2-dspark.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dflash2 import _decode_unique_json, _read_safetensors_header

DRAFT_FILE = "mtp.safetensors"
DRAFT_DTYPE = "BF16"


def _target_text(target_config):
    if target_config is None:
        return None
    if isinstance(target_config, (str, Path)):
        target_config = json.loads((Path(target_config).expanduser() / "config.json").read_text())
    if isinstance(target_config, dict):
        return target_config.get("text_config", target_config)
    from dataclasses import asdict, is_dataclass

    if is_dataclass(target_config):
        return asdict(target_config)
    raise ValueError("Unsupported Inkling 2 target config")


def _check_quant_group(path, expected):
    quant_path = path / "hf_quant_config.json"
    if not quant_path.is_file():
        return None
    raw = quant_path.read_bytes()
    quant = _decode_unique_json(raw, "hf_quant_config.json")
    groups = (quant.get("mtp") or {}).get("parameter_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("Inkling 2 hf_quant_config.json has no mtp parameter groups")
    seen = []
    for group in groups:
        if (
            group.get("weight_transform") != "cast"
            or group.get("required_activation_quantization") is not None
            or group.get("weight_block_shape") not in ([], None)
            or group.get("compute_dtype") != "bfloat16"
        ):
            raise ValueError("Inkling 2 DSpark tensors must be an unquantized BF16 cast")
        seen.extend(group.get("tensors") or [])
    if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
        raise ValueError("Inkling 2 hf_quant_config mtp group does not match the DSpark schema")
    return hashlib.sha256(raw).hexdigest()


def inspect_drafter(artifact_path, target_config=None):
    """Header-only record of the bundled drafter; raises ValueError on any mismatch.

    ``target_config`` (a config dict, a target ``TextConfig``, or a checkpoint
    path) must agree with the artifact's own text config; ``None`` uses it.
    """
    from ..runtime.drafters.inkling2_dspark import Inkling2DSparkConfig, expected_weights

    path = Path(artifact_path).expanduser().resolve()
    raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(raw, "config.json")
    args = Inkling2DSparkConfig.from_hf(config)
    target = _target_text(target_config)
    if target is not None:
        own = config.get("text_config", config)
        for name in ("model_type", "hidden_size", "vocab_size", "num_hidden_layers"):
            if target.get(name) != own.get(name):
                raise ValueError(f"Inkling 2 DSpark target config mismatch: {name}")
        if Inkling2DSparkConfig.from_hf({**config, "text_config": target}) != args:
            raise ValueError("Inkling 2 DSpark target config mismatch")
    expected = expected_weights(args)
    weights = path / DRAFT_FILE
    if not weights.is_file():
        raise ValueError(f"Inkling 2 DSpark weights missing: {DRAFT_FILE}")
    stat = weights.stat()
    header_raw, header, payload = _read_safetensors_header(weights)
    observed, ranges = {}, []
    for key, record in header.items():
        if key == "__metadata__":
            continue
        offsets = record.get("data_offsets") if isinstance(record, dict) else None
        if not (isinstance(offsets, list) and len(offsets) == 2
                and all(type(o) is int for o in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload):
            raise ValueError(f"Invalid Inkling 2 DSpark tensor offsets: {key}")
        ranges.append((offsets[0], offsets[1], key))
        observed[key] = (record.get("dtype"), record.get("shape"))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"Overlapping Inkling 2 DSpark payloads: {previous[2]}, {current[2]}")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))[:8]
        extra = sorted(set(observed) - set(expected))[:8]
        raise ValueError(f"Inkling 2 DSpark schema mismatch (missing={missing}, extra={extra})")
    for key, shape in expected.items():
        dtype, observed_shape = observed[key]
        if observed_shape != shape or dtype != DRAFT_DTYPE:
            raise ValueError(f"Inkling 2 DSpark tensor dtype/shape mismatch: {key}")
    for start, end, key in ranges:
        size = 2
        for dim in observed[key][1]:
            size *= dim
        if end - start != size:
            raise ValueError(f"Inkling 2 DSpark tensor byte size mismatch: {key}")
    quant_sha = _check_quant_group(path, expected)
    header_sha = hashlib.sha256(header_raw).hexdigest()
    config_sha = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256()
    for part in (config_sha, header_sha, quant_sha or ""):
        digest.update(part.encode())
    return {
        "kind": "inkling2_dspark",
        "path": str(path),
        "files": [(DRAFT_FILE, stat.st_size, stat.st_mtime_ns)],
        "config_sha256": config_sha,
        "header_sha256": header_sha,
        "quant_config_sha256": quant_sha,
        "draft_revision": digest.hexdigest(),
        "args": args,
        "block_size": args.block_size,
        "target_layer_ids": list(args.target_layer_ids),
    }


def load_drafter(record, target_model, *, weights=None, require_bf16=True):
    """Strict load of ``mtp.safetensors`` (or the given ``weights`` mapping).

    ``require_bf16=False`` is for synthetic FP32 test weights only.
    """
    import mlx.core as mx

    from ..runtime.drafters.inkling2_dspark import Inkling2DSparkDraftModel

    model = Inkling2DSparkDraftModel(record["args"])
    if weights is None:
        name, size, mtime = record["files"][0]
        file = Path(record["path"]) / name
        stat = file.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError("Inkling 2 DSpark weights changed since inspection")
        weights = mx.load(str(file))
    weights = model.sanitize(dict(weights))
    for name, value in weights.items():
        if require_bf16 and value.dtype != mx.bfloat16:
            raise ValueError(f"Inkling 2 DSpark tensor is not BF16: {name}")
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model if target_model is None else model.bind(target_model)


__all__ = ["inspect_drafter", "load_drafter"]
