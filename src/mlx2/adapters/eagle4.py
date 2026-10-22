# Copyright 2026 Eagle AI Corp.
# Modified 2026-10-21 by mlx2 contributors: added strict MLX artifact binding
# and loading for the dense Qwen3.8 EAGLE-4 checkpoint.
# SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available
# Eagle 4 Source-Available License (not an OSI-approved open-source license):
# see third_party/eagle4/ (LICENSE, NOTICE, MODIFICATIONS.md, README.md).
"""Strict inspection and MLX loading of dense EAGLE-4 draft artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..runtime.drafters.eagle4_config import (
    ARCHITECTURE,
    Eagle4Config,
    expected_weight_schema,
)
from .dflash2 import (
    _decode_unique_json,
    _file_sha256_with_binding,
    _read_bytes_with_binding,
    _source_binding,
    _verify_source_bindings,
)
from .draft_artifacts import inspect_files


def inspect_drafter(path, target):
    path = Path(path).expanduser().resolve()
    config_raw, config_binding = _read_bytes_with_binding(path / "config.json")
    config = _decode_unique_json(config_raw, "EAGLE-4 config.json")
    target_raw, _ = _read_bytes_with_binding(Path(target) / "config.json")
    target_config = _decode_unique_json(target_raw, "target config.json")
    text = target_config.get("text_config", target_config)
    args = Eagle4Config.from_hf(
        config, num_target_layers=int(text["num_hidden_layers"])
    )
    if int(text["hidden_size"]) != args.target_hidden_size:
        raise ValueError("EAGLE-4 target hidden_size mismatch")
    if int(text["vocab_size"]) != args.vocab_size:
        raise ValueError("EAGLE-4 target vocab_size mismatch")
    record = inspect_files(
        path,
        config_raw,
        expected=expected_weight_schema(args),
        dtype=config.get("dtype", config.get("torch_dtype")),
        label="EAGLE-4",
    )
    if len(record["files"]) != 1 or record["files"][0][0] != "model.safetensors":
        raise ValueError("EAGLE-4 release format expects one model.safetensors file")
    weight = path / "model.safetensors"
    binding = _source_binding(weight)
    weight_hash, _ = _file_sha256_with_binding(weight, binding)
    source_bindings = [
        ("config.json", *config_binding),
        ("model.safetensors", *binding),
    ]
    _verify_source_bindings(path, source_bindings, label="EAGLE-4 artifact")
    return {
        **record,
        "config": config,
        "args": args,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "weight_sha256": [weight_hash],
        "source_bindings": source_bindings,
    }


def legacy_content_revision(record):
    """EAGLE-4 has no prior mlx2 metadata-only revision."""
    return


def content_revision(record):
    weights = record.get("weight_sha256")
    if not isinstance(weights, list) or len(weights) != 1 or len(weights[0]) != 64:
        raise ValueError("EAGLE-4 content revision requires the full payload hash")
    return hashlib.sha256(
        json.dumps(
            [
                "mlx2.eagle4-content.v1",
                record["config"],
                record["header_sha256"],
                weights,
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_runtime_quantization(value):
    if value is not None:
        raise ValueError("EAGLE-4 runtime draft quantization is not implemented")


def load_drafter(
    record, target_model, *, runtime_quantization=None, draft_attention_windows=None,
    eagle4_tree=None,
):
    validate_runtime_quantization(runtime_quantization)
    if draft_attention_windows is not None:
        raise ValueError("EAGLE-4 uses its trained 4096-token sliding window")
    root = Path(record["path"])
    _verify_source_bindings(
        root, record.get("source_bindings"), label="EAGLE-4 artifact"
    )
    config = _decode_unique_json(
        (root / "config.json").read_bytes(), "EAGLE-4 config.json"
    )
    if (
        Eagle4Config.from_hf(config, num_target_layers=record["args"].num_target_layers)
        != record["args"]
    ):
        raise ValueError("EAGLE-4 config changed since inspection")

    import mlx.core as mx

    from ..runtime.drafters.eagle4 import Eagle4DraftModel

    model = Eagle4DraftModel(record["args"], tree=eagle4_tree)
    weights = mx.load(str(root / "model.safetensors"))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    bound = model.bind(target_model)
    _verify_source_bindings(
        root, record.get("source_bindings"), label="EAGLE-4 artifact"
    )
    return bound


__all__ = [
    "ARCHITECTURE",
    "content_revision",
    "inspect_drafter",
    "legacy_content_revision",
    "load_drafter",
    "validate_runtime_quantization",
]
