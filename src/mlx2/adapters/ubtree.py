"""Strict artifact binding and loader for Ling 3 UBTree draft checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..runtime.drafters.ubtree_config import UBTreeConfig
from .dflash2 import (
    _decode_unique_json,
    _file_sha256_with_binding,
    _read_bytes_with_binding,
    _read_safetensors_header,
    _source_binding,
    _verify_source_bindings,
)


def _expected_weight_shapes(args):
    hidden = args.hidden_size
    heads = args.num_attention_heads * args.head_dim
    kv_heads = args.num_key_value_heads * args.head_dim
    shapes = {
        "fc.weight": [hidden, len(args.target_layer_ids) * hidden],
        "hidden_norm.weight": [hidden],
        "norm.weight": [hidden],
        "ubtree_selector.depth_conditioner.correction_log_scale": [
            args.depth_condition_count
        ],
        "ubtree_selector.depth_conditioner.unary_log_scale": [
            args.depth_condition_count
        ],
        "ubtree_selector.hidden_projection.weight": [args.selector_rank, hidden],
        "ubtree_selector.tied_token_factorizer.predecessor_mlp.input_projection.weight": [
            args.factorizer_hidden_width,
            hidden,
        ],
        "ubtree_selector.tied_token_factorizer.predecessor_mlp.output_projection.weight": [
            args.selector_rank,
            args.factorizer_hidden_width,
        ],
        "ubtree_selector.tied_token_factorizer.successor_mlp.input_projection.weight": [
            args.factorizer_hidden_width,
            hidden,
        ],
        "ubtree_selector.tied_token_factorizer.successor_mlp.output_projection.weight": [
            args.selector_rank,
            args.factorizer_hidden_width,
        ],
    }
    for index in range(args.num_hidden_layers):
        prefix = f"layers.{index}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": [hidden],
                prefix + "mlp.down_proj.weight": [hidden, args.intermediate_size],
                prefix + "mlp.gate_proj.weight": [args.intermediate_size, hidden],
                prefix + "mlp.up_proj.weight": [args.intermediate_size, hidden],
                prefix + "post_attention_layernorm.weight": [hidden],
                prefix + "self_attn.k_norm.weight": [args.head_dim],
                prefix + "self_attn.k_proj.weight": [kv_heads, hidden],
                prefix + "self_attn.o_proj.weight": [hidden, heads],
                prefix + "self_attn.q_norm.weight": [args.head_dim],
                prefix + "self_attn.q_proj.weight": [heads, hidden],
                prefix + "self_attn.v_proj.weight": [kv_heads, hidden],
            }
        )
    return shapes


def _validate_headers(files, mapping, args):
    expected = _expected_weight_shapes(args)
    observed = {}
    header_digests = []
    for path in files:
        raw, header, payload_size = _read_safetensors_header(path)
        header_digests.append(hashlib.sha256(raw).hexdigest())
        ranges = []
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(record, dict) or set(record) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid UBTree tensor metadata: {name}")
            shape, offsets = record["shape"], record["data_offsets"]
            if record["dtype"] != "BF16" or not isinstance(shape, list):
                raise ValueError(f"UBTree tensor dtype/shape mismatch: {name}")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise ValueError(f"Invalid UBTree tensor offsets: {name}")
            elements = 1
            for value in shape:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid UBTree tensor shape: {name}")
                elements *= value
            if offsets[1] - offsets[0] != elements * 2:
                raise ValueError(f"UBTree tensor byte size mismatch: {name}")
            if name in observed or (
                mapping is not None and mapping.get(name) != path.name
            ):
                raise ValueError(f"UBTree weight index/shard mismatch: {name}")
            observed[name] = shape
            ranges.append((offsets[0], offsets[1], name))
        for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
            if previous[1] > current[0]:
                raise ValueError("overlapping UBTree tensor payloads")
    if mapping is not None and set(mapping) != set(observed):
        raise ValueError("UBTree index does not match shard headers")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            k for k in set(expected) & set(observed) if expected[k] != observed[k]
        )
        raise ValueError(
            f"UBTree checkpoint schema mismatch (missing={missing}, extra={extra}, wrong_shapes={wrong})"
        )
    return header_digests


def inspect_drafter(path, target):
    root = Path(path).expanduser().resolve()
    config_raw, config_binding = _read_bytes_with_binding(root / "config.json")
    config = _decode_unique_json(config_raw, "UBTree config.json")
    if config.get("architectures") != ["Ling3UBTreeModel"]:
        raise ValueError("expected a Ling3UBTreeModel artifact")
    args = UBTreeConfig.from_dict(config)
    target_root = Path(target).expanduser().resolve()
    target_raw, target_binding = _read_bytes_with_binding(target_root / "config.json")
    target_config = _decode_unique_json(target_raw, "Ling target config.json")
    if target_config.get("model_type") != "bailing_hybrid":
        raise ValueError("UBTree requires a bailing_hybrid target")
    for name in ("hidden_size", "vocab_size"):
        if target_config.get(name) != getattr(args, name):
            raise ValueError(f"UBTree target {name} mismatch")
    if target_config.get("num_hidden_layers") != args.num_target_layers:
        raise ValueError("UBTree target layer count mismatch")

    index = root / "model.safetensors.index.json"
    source_bindings = [("config.json", *config_binding)]
    if index.is_file():
        raw, binding = _read_bytes_with_binding(index)
        document = _decode_unique_json(raw, "UBTree weight index")
        mapping = document.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("invalid UBTree weight index")
        names = sorted(set(mapping.values()))
        source_bindings.append((index.name, *binding))
        index_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        mapping, names, index_sha256 = None, ["model.safetensors"], None
    files, paths, weight_bindings = [], [], []
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
            or not (root / name).is_file()
        ):
            raise ValueError("invalid UBTree shard path")
        item = root / name
        binding = _source_binding(item)
        files.append((name, binding[2], binding[3]))
        paths.append(item)
        weight_bindings.append((name, *binding))
    headers = _validate_headers(paths, mapping, args)
    weight_hashes = [
        _file_sha256_with_binding(path, tuple(binding[1:]))[0]
        for path, binding in zip(paths, weight_bindings, strict=True)
    ]
    source_bindings.extend(weight_bindings)
    _verify_source_bindings(root, source_bindings, label="UBTree artifact")
    return {
        "path": str(root),
        "target_path": str(target_root),
        "config": config,
        "args": args,
        "files": files,
        "header_sha256": headers,
        "weight_sha256": weight_hashes,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "index_sha256": index_sha256,
        "source_bindings": source_bindings,
        "target_source_bindings": [("config.json", *target_binding)],
        "fingerprint": content_revision_parts(config, headers, weight_hashes),
    }


def content_revision_parts(config, headers, weights):
    return hashlib.sha256(
        json.dumps(
            ["mlx2.ubtree-content.v1", config, headers, weights],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def content_revision(record):
    weights = record.get("weight_sha256")
    if not isinstance(weights, list) or any(len(value) != 64 for value in weights):
        raise ValueError("UBTree content revision requires payload hashes")
    return content_revision_parts(record["config"], record["header_sha256"], weights)


def legacy_content_revision(record):
    return hashlib.sha256(
        json.dumps(
            ["mlx2.ubtree-legacy.v1", record["config"], record["header_sha256"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate_runtime_quantization(value):
    if value is not None:
        raise ValueError("runtime quantization is not implemented for UBTree")


def load_drafter(record, target_model, **options):
    if any(value is not None for value in options.values()):
        raise ValueError("UBTree accepts no load-time drafter overrides")
    root = Path(record["path"])
    _verify_source_bindings(root, record["source_bindings"], label="UBTree artifact")
    import mlx.core as mx

    from ..runtime.drafters.ubtree import UBTreeDraftModel
    from ..runtime.ubc_evict import load_shards_evicting

    model = UBTreeDraftModel(record["args"])
    paths = [root / name for name, _, _ in record["files"]]
    weights = model.sanitize(load_shards_evicting(paths))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    bound = model.bind(target_model)
    _verify_source_bindings(root, record["source_bindings"], label="UBTree artifact")
    return bound


__all__ = [
    "content_revision",
    "inspect_drafter",
    "legacy_content_revision",
    "load_drafter",
    "validate_runtime_quantization",
]
