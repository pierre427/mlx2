"""Strict CPU-safe inspection/loading for a trained LiLiCorr companion artifact.

Public pretrained checkpoints are not published as of 2026-10-01. This loader
accepts complete compatible exported artifacts, never a heuristic replacement.
Provenance: provenance/lilicorr.json.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from ..runtime.drafters.base_config import DFlashConfig
from .dflash2 import _DTYPE_BYTES, _decode_unique_json, _read_safetensors_header
from .xpress import XPressConfig


@dataclass
class LiLiCorrConfig(DFlashConfig):
    lilicorr_hidden_size: int = 0
    lilicorr_candidate_topk: int = 8
    lilicorr_num_layers: int = 2
    lilicorr_num_heads: int = 8
    lilicorr_mlp_ratio: float = 2.0
    lilicorr_factor_dim: int = 128
    lilicorr_vector_eps: float = 1e-6
    lilicorr_logit_scale: float = 1.0
    conv_kernel_size: int = 0
    conv_group_size: int = 0

    def __post_init__(self):
        # Shared backbone contract, independently of any XPress head.
        XPressConfig(
            **{k: getattr(self, k) for k in DFlashConfig.__dataclass_fields__},
            xpress_num_passes=1,
        )
        width = self.lilicorr_hidden_size
        if type(width) is not int or width < 0:
            raise ValueError("Invalid LiLiCorr hidden size")
        width = width or self.hidden_size
        k = self.lilicorr_candidate_topk
        if type(k) is not int or not 1 <= k <= min(16, self.vocab_size) or k & (k - 1):
            raise ValueError(
                "LiLiCorr candidate_topk must be a power of two <= 16 and fit vocabulary"
            )
        for name in (
            "lilicorr_num_layers",
            "lilicorr_num_heads",
            "lilicorr_factor_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid LiLiCorr {name}")
        if width % self.lilicorr_num_heads:
            raise ValueError("LiLiCorr head width must divide its attention heads")
        for name in (
            "lilicorr_mlp_ratio",
            "lilicorr_vector_eps",
            "lilicorr_logit_scale",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid LiLiCorr {name}")
        if int(width * self.lilicorr_mlp_ratio) <= 0:
            raise ValueError("Invalid LiLiCorr MLP width")
        for name in ("conv_kernel_size", "conv_group_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("Invalid LiLiCorr convolution geometry")
        if (
            bool(self.conv_kernel_size) != bool(self.conv_group_size)
            or self.conv_group_size
            and self.hidden_size % self.conv_group_size
        ):
            raise ValueError("LiLiCorr convolution requires matched taps/groups")
        if self.conv_kernel_size > self.block_size:
            raise ValueError("LiLiCorr convolution taps exceed trained block")

    @classmethod
    def from_dict(cls, config):
        if not isinstance(config, dict):
            raise ValueError("LiLiCorr config must be an object")  # noqa: TRY004 - artifact validation contract
        if (
            config.get("architectures") != ["LiLiCorrDraftModel"]
            or config.get("model_type") != "qwen3"
        ):
            raise ValueError("Expected Qwen3 LiLiCorrDraftModel artifact")
        if config.get("has_own_lm_head", False) is not False:
            raise ValueError("LiLiCorr owned LM heads are not supported")
        allowed = set(DFlashConfig.__dataclass_fields__) | {
            "architectures",
            "model_type",
            "dflash_config",
            "dtype",
            "torch_dtype",
            "auto_map",
            "bos_token_id",
            "eos_token_id",
            "hidden_act",
            "initializer_range",
            "attention_dropout",
            "max_window_layers",
            "transformers_version",
            "use_cache",
            "use_sliding_window",
            "has_own_lm_head",
            "draft_vocab_size",
        }
        if set(config) - allowed:
            raise ValueError(
                f"Unsupported LiLiCorr config settings: {sorted(set(config) - allowed)}"
            )
        if (
            config.get("hidden_act", "silu") != "silu"
            or config.get("attention_dropout", 0) != 0
            or config.get("use_sliding_window", False) is not False
        ):
            raise ValueError("Unsupported LiLiCorr backbone variant")
        nested = config.get("dflash_config")
        head_fields = {k for k in cls.__dataclass_fields__ if k.startswith("lilicorr_")}
        required_nested = head_fields | {"mask_token_id", "target_layer_ids"}
        if not isinstance(nested, dict) or required_nested - set(nested):
            raise ValueError(
                "LiLiCorr requires all trained head geometry and target taps"
            )
        if set(nested) - (
            required_nested
            | {"lilicorr_enabled", "block_size", "conv_kernel_size", "conv_group_size"}
        ):
            raise ValueError("Unsupported LiLiCorr nested geometry")
        if nested.get("lilicorr_enabled", True) is not True:
            raise ValueError("LiLiCorr must be enabled")
        required = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_target_layers",
            "block_size",
            "layer_types",
            "rms_norm_eps",
            "rope_theta",
        }
        if required - set(config):
            raise ValueError("Missing LiLiCorr backbone geometry")
        if config.get("draft_vocab_size", config["vocab_size"]) != config["vocab_size"]:
            raise ValueError("LiLiCorr requires full target vocabulary")
        for key in ("mask_token_id", "target_layer_ids", "block_size"):
            if key in config and key in nested and config[key] != nested[key]:
                raise ValueError(f"Conflicting LiLiCorr {key}")
        if (
            "dtype" in config
            and "torch_dtype" in config
            and config["dtype"] != config["torch_dtype"]
        ):
            raise ValueError("Conflicting LiLiCorr dtype")
        values = {k: v for k, v in config.items() if k in cls.__dataclass_fields__}
        values.update(
            {k: v for k, v in nested.items() if k in cls.__dataclass_fields__}
        )
        return cls(**values)

    from_hf_dict = from_dict


def expected_weight_shapes(args):
    from .xpress import expected_weight_shapes as backbone_shapes

    # This uses only the common DFlash namespace; XPress head shapes are removed.
    shared = XPressConfig(
        **{k: getattr(args, k) for k in DFlashConfig.__dataclass_fields__},
        xpress_num_passes=1,
    )
    shapes = {
        k: v
        for k, v in backbone_shapes(shared).items()
        if not k.startswith("xpress_head.")
    }
    h = args.lilicorr_hidden_size or args.hidden_size
    model_h = args.hidden_size

    def linear(name, inputs, outputs):
        shapes["lilicorr." + name + ".weight"] = [outputs, inputs]
        shapes["lilicorr." + name + ".bias"] = [outputs]

    if h != model_h:
        linear("token_proj", model_h, h)
    for name in ("pass_hidden_proj", "context_proj"):
        linear(name, model_h, h)
    linear("feature_mlp.up_proj", 5, h)
    linear("feature_mlp.down_proj", h, h)
    shapes.update(
        {
            "lilicorr.feature_norm.weight": [5],
            "lilicorr.feature_norm.bias": [5],
            "lilicorr.slot_embedding": [1, 1, args.block_size - 1, 1, h],
            "lilicorr.rank_embedding": [1, 1, 1, args.lilicorr_candidate_topk, h],
            "lilicorr.relative_slot_bias": [
                args.lilicorr_num_heads,
                2 * args.block_size - 1,
            ],
            "lilicorr.same_slot_bias": [args.lilicorr_num_heads],
            "lilicorr.output_norm.weight": [h],
            "lilicorr.anchor_norm.weight": [h],
        }
    )
    linear("factor_input_proj", 3 * h, h)
    for name in ("out_head", "in_head", "anchor_out_head"):
        linear(name, h, args.lilicorr_factor_dim)
    for i in range(args.lilicorr_num_layers):
        p = f"layers.{i}."
        shapes["lilicorr." + p + "attn_norm.weight"] = [h]
        shapes["lilicorr." + p + "mlp_norm.weight"] = [h]
        shapes["lilicorr." + p + "attn.in_proj_weight"] = [3 * h, h]
        shapes["lilicorr." + p + "attn.in_proj_bias"] = [3 * h]
        linear(p + "attn.out_proj", h, h)
        linear(p + "mlp.up_proj", h, int(h * args.lilicorr_mlp_ratio))
        linear(p + "mlp.down_proj", int(h * args.lilicorr_mlp_ratio), h)
    if args.conv_kernel_size:
        for i in range(args.num_hidden_layers):
            for component in ("attention_conv", "mlp_conv"):
                p = f"layers.{i}.{component}."
                shapes[p + "base_kernel"] = [2, args.conv_kernel_size, model_h]
                shapes[p + "kernel_projection.weight"] = [
                    2 * args.conv_kernel_size * (model_h // args.conv_group_size),
                    model_h,
                ]
    return shapes


def _validate_headers(paths, mapping, args, configured_dtype):
    dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(
        configured_dtype
    )
    if dtype is None:
        raise ValueError("Unsupported LiLiCorr checkpoint dtype")
    expected, observed, digests = expected_weight_shapes(args), {}, []
    for path in paths:
        raw, header, size = _read_safetensors_header(path)
        digests.append(hashlib.sha256(raw).hexdigest())
        ranges = []
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in observed or name not in expected:
                raise ValueError(f"Duplicate or unexpected LiLiCorr tensor: {name}")
            if not isinstance(item, dict) or set(item) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid LiLiCorr tensor metadata: {name}")
            shape, offsets = item["shape"], item["data_offsets"]
            if (
                item["dtype"] != dtype
                or not isinstance(shape, list)
                or any(type(n) is not int for n in shape)
                or shape != expected[name]
            ):
                raise ValueError(f"LiLiCorr tensor dtype/shape mismatch: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size
                or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
            ):
                raise ValueError(f"Invalid LiLiCorr tensor offsets: {name}")
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"LiLiCorr weight index/shard mismatch: {name}")
            observed[name] = shape
            ranges.append((offsets[0], offsets[1]))
        ranges.sort()
        # Safetensors must describe exactly the payload, without holes or tails.
        if (
            not ranges
            or ranges[0][0] != 0
            or ranges[-1][1] != size
            or any(a[1] != b[0] for a, b in pairwise(ranges))
        ):
            raise ValueError(
                "LiLiCorr payload ranges overlap or leave unaccounted bytes"
            )
    if observed != expected or mapping is not None and set(mapping) != set(observed):
        raise ValueError("LiLiCorr checkpoint schema/index mismatch")
    return digests


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_drafter(path, target):
    path = Path(path).expanduser().resolve()
    config_raw = (path / "config.json").read_bytes()
    config = _decode_unique_json(config_raw, "LiLiCorr config")
    args = LiLiCorrConfig.from_dict(config)
    target_config = _decode_unique_json(
        (Path(target) / "config.json").read_bytes(), "target config"
    )
    text = target_config.get("text_config", target_config)
    if text.get("model_type") != "qwen3":
        raise ValueError("LiLiCorr target must be Qwen3")
    for key in (
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rope_theta",
        "rms_norm_eps",
    ):
        if text.get(key) != getattr(args, key):
            raise ValueError(f"LiLiCorr target {key} mismatch")
    if (
        text.get("num_hidden_layers") != args.num_target_layers
        or text.get("rope_scaling") is not None
    ):
        raise ValueError("LiLiCorr target layer/RoPE mismatch")
    digest = hashlib.sha256(config_raw)
    index = path / "model.safetensors.index.json"
    mapping = None
    if index.exists():
        raw = index.read_bytes()
        digest.update(raw)
        mapping = _decode_unique_json(raw, "LiLiCorr index").get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(n, str) for n in mapping.values())
        ):
            raise ValueError("Invalid LiLiCorr weight index")
        names = sorted(set(mapping.values()))
    else:
        names = ["model.safetensors"]
    files, paths = [], []
    for name in names:
        if (
            "/" in name
            or "\\" in name
            or name.startswith(".")
            or not name.endswith(".safetensors")
            or not (path / name).is_file()
        ):
            raise ValueError("Invalid LiLiCorr shard path")
        file = path / name
        stat = file.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))
        paths.append(file)
    headers = _validate_headers(
        paths, mapping, args, config.get("dtype", config.get("torch_dtype"))
    )
    digest.update(json.dumps(files).encode())
    digest.update(json.dumps(headers).encode())
    source_hashes = [_file_sha256(file) for file in paths]
    digest.update(json.dumps(source_hashes).encode())
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": files,
        "header_sha256": headers,
        "weight_sha256": source_hashes,
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "config": config,
        "args": args,
    }


def content_revision(record):
    """Metadata content identity; full tensor hashes belong in acquisition receipts."""
    return hashlib.sha256(
        json.dumps(
            [record["config"], record["header_sha256"], record["weight_sha256"]],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def load_drafter(
    record, target_model, *, runtime_quantization=None, draft_attention_windows=None
):
    if runtime_quantization is not None:
        raise ValueError("LiLiCorr runtime quantization is not supported")
    from ..runtime.drafters.attention_windows import validate_attention_windows

    args = record["args"]
    validate_attention_windows(draft_attention_windows, args.num_hidden_layers)
    if args != LiLiCorrConfig.from_dict(record["config"]):
        raise ValueError("LiLiCorr load configuration differs from inspected source")
    if (
        hashlib.sha256((Path(record["path"]) / "config.json").read_bytes()).hexdigest()
        != record["config_sha256"]
    ):
        raise ValueError("LiLiCorr config changed since inspection")
    # Reconcile the current artifact before importing MLX or allocating payload.
    # Target model identity has already been checked by the serving adapter.
    paths = [Path(record["path"]) / name for name, _, _ in record["files"]]
    for path, (_, size, mtime) in zip(paths, record["files"]):
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError("LiLiCorr artifact changed since inspection")
    if (
        _validate_headers(
            paths,
            None,
            record["args"],
            record["config"].get("dtype", record["config"].get("torch_dtype")),
        )
        != record["header_sha256"]
    ):
        raise ValueError("LiLiCorr header changed since inspection")
    if [_file_sha256(path) for path in paths] != record["weight_sha256"]:
        raise ValueError("LiLiCorr tensor bytes changed since inspection")
    import mlx.core as mx

    from ..runtime.drafters.lilicorr import LiLiCorrDraftModel

    model = LiLiCorrDraftModel(args, draft_attention_windows=draft_attention_windows)
    weights = {}
    for path in paths:
        shard = model.sanitize(mx.load(str(path)))
        mx.eval(list(shard.values()))
        weights.update(shard)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    return model.bind(target_model)
