"""CPU-only metadata inspection and strict loading for Qwen3 XPress artifacts.

No model code or MLX is imported until ``load_drafter`` after source and target
metadata have been reconciled. Provenance: provenance/xpress.json.
"""

import hashlib
import json
import math
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

from ..runtime.drafters.base_config import DFlashConfig
from .dflash2 import _DTYPE_BYTES, _decode_unique_json, _read_safetensors_header


@dataclass
class XPressConfig(DFlashConfig):
    xpress_rank: int = 256
    xpress_mlp_hidden: int = 512
    xpress_num_passes: int = 6
    sample_from_anchor: bool = False

    def __post_init__(self):
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_target_layers",
            "block_size",
            "xpress_rank",
            "xpress_mlp_hidden",
            "xpress_num_passes",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"XPress {key} must be a positive integer")
        if self.xpress_num_passes > self.block_size:
            raise ValueError("XPress passes must not exceed the trained block size")
        if self.block_size < 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Invalid XPress block/head geometry")
        if (
            type(self.mask_token_id) is not int
            or not 0 <= self.mask_token_id < self.vocab_size
        ):
            raise ValueError("Invalid XPress mask token")
        ids = self.target_layer_ids
        if (
            not ids
            or any(
                type(i) is not int or not 0 <= i < self.num_target_layers for i in ids
            )
            or ids != sorted(set(ids))
        ):
            raise ValueError("Invalid XPress target layer taps")
        if self.sample_from_anchor is not False:
            raise ValueError("XPress requires a fixed anchor")
        if (
            self.attention_bias is not False
            or self.layer_types != ["full_attention"] * self.num_hidden_layers
        ):
            raise ValueError("XPress supports unbiased full attention only")
        if (
            self.sliding_window is not None
            or self.rope_scaling is not None
            or self.final_logit_softcapping is not None
        ):
            raise ValueError("Unsupported XPress attention/logit settings")
        if self.runtime_block_size is not None or self.draft_window_size is not None:
            raise ValueError(
                "XPress computes its full trained block without runtime window overrides"
            )
        for key in ("rms_norm_eps", "rope_theta"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid XPress {key}")

    @classmethod
    def from_dict(cls, config):
        if not isinstance(config, dict):
            raise ValueError("XPress config must be an object")  # noqa: TRY004 - artifact validation contract
        allowed = set(cls.__dataclass_fields__) | {
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
            "xpress_block_size",
            "draft_vocab_size",
            "sample_from_anchor",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(f"Unsupported XPress config settings: {sorted(unknown)}")
        if (
            config.get("architectures") != ["Qwen3XPressModel"]
            or config.get("model_type") != "qwen3"
        ):
            raise ValueError("Expected Qwen3XPressModel artifact")
        if (
            config.get("hidden_act", "silu") != "silu"
            or config.get("attention_dropout", 0) != 0
            or config.get("use_sliding_window", False) is not False
        ):
            raise ValueError("Unsupported XPress backbone variant")
        nested = config.get("dflash_config")
        if not isinstance(nested, dict) or set(nested) != {
            "mask_token_id",
            "target_layer_ids",
        }:
            raise ValueError("XPress requires explicit mask token and target taps")
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
            "xpress_rank",
            "xpress_mlp_hidden",
            "xpress_num_passes",
            "xpress_block_size",
        }
        if required - set(config):
            raise ValueError(
                f"Missing XPress config settings: {sorted(required - set(config))}"
            )
        if config["xpress_block_size"] != config["block_size"]:
            raise ValueError("XPress trained block size mismatch")
        if config.get("draft_vocab_size", config["vocab_size"]) != config["vocab_size"]:
            raise ValueError("XPress requires full target vocabulary")
        for key in ("mask_token_id", "target_layer_ids"):
            if key in config and config[key] != nested[key]:
                raise ValueError(f"Conflicting XPress {key} metadata")
        if (
            "dtype" in config
            and "torch_dtype" in config
            and config["dtype"] != config["torch_dtype"]
        ):
            raise ValueError("Conflicting XPress dtype metadata")
        values = {k: v for k, v in config.items() if k in cls.__dataclass_fields__}
        values.update(nested)
        return cls(**values)

    from_hf_dict = from_dict


def expected_weight_shapes(args):
    h, r, b = args.hidden_size, args.xpress_rank, args.block_size
    shapes = {
        "fc.weight": [h, len(args.target_layer_ids) * h],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
        "xpress_head.w1.weight": [args.vocab_size, r],
        "xpress_head.w2.weight": [args.vocab_size, r],
        "xpress_head.down_h.weight": [r, h],
        "xpress_head.down_g.weight": [r, h],
        "xpress_head.in_proj.weight": [r, 3 * r],
        "xpress_head.mix.L": [r, b, b],
        "xpress_head.mlp.gate_proj.weight": [args.xpress_mlp_hidden, r],
        "xpress_head.mlp.up_proj.weight": [args.xpress_mlp_hidden, r],
        "xpress_head.mlp.down_proj.weight": [r, args.xpress_mlp_hidden],
    }
    for i in range(args.num_hidden_layers):
        prefix = f"layers.{i}."
        shapes.update(
            {
                prefix + name: shape
                for name, shape in {
                    "input_layernorm.weight": [h],
                    "post_attention_layernorm.weight": [h],
                    "mlp.gate_proj.weight": [args.intermediate_size, h],
                    "mlp.up_proj.weight": [args.intermediate_size, h],
                    "mlp.down_proj.weight": [h, args.intermediate_size],
                    "self_attn.q_norm.weight": [args.head_dim],
                    "self_attn.k_norm.weight": [args.head_dim],
                    "self_attn.q_proj.weight": [
                        args.num_attention_heads * args.head_dim,
                        h,
                    ],
                    "self_attn.k_proj.weight": [
                        args.num_key_value_heads * args.head_dim,
                        h,
                    ],
                    "self_attn.v_proj.weight": [
                        args.num_key_value_heads * args.head_dim,
                        h,
                    ],
                    "self_attn.o_proj.weight": [
                        h,
                        args.num_attention_heads * args.head_dim,
                    ],
                }.items()
            }
        )
    return shapes


def _validate_headers(paths, mapping, args, configured_dtype):
    dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(
        configured_dtype
    )
    if dtype is None:
        raise ValueError("Unsupported XPress checkpoint dtype")
    expected, observed, digests = expected_weight_shapes(args), {}, []
    for path in paths:
        raw, header, size = _read_safetensors_header(path)
        digests.append(hashlib.sha256(raw).hexdigest())
        ranges = []
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in observed or name not in expected:
                raise ValueError(f"Duplicate or unexpected XPress tensor: {name}")
            if not isinstance(item, dict) or set(item) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid XPress tensor metadata: {name}")
            shape, offsets = item["shape"], item["data_offsets"]
            if (
                item["dtype"] != dtype
                or not isinstance(shape, list)
                or any(type(n) is not int for n in shape)
                or shape != expected[name]
            ):
                raise ValueError(f"XPress tensor dtype/shape mismatch: {name}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size
                or offsets[1] - offsets[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
            ):
                raise ValueError(f"Invalid XPress tensor offsets: {name}")
            if mapping is not None and mapping.get(name) != path.name:
                raise ValueError(f"XPress weight index/shard mismatch: {name}")
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
            raise ValueError("XPress payload ranges overlap or leave unaccounted bytes")
    if observed != expected or mapping is not None and set(mapping) != set(observed):
        raise ValueError("XPress checkpoint schema/index mismatch")
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
    config = _decode_unique_json(config_raw, "XPress config")
    args = XPressConfig.from_dict(config)
    target_config = _decode_unique_json(
        (Path(target) / "config.json").read_bytes(), "target config"
    )
    text = target_config.get("text_config", target_config)
    if text.get("model_type") != "qwen3":
        raise ValueError("XPress target must be Qwen3")
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
            raise ValueError(f"XPress target {key} mismatch")
    if (
        text.get("num_hidden_layers") != args.num_target_layers
        or text.get("rope_scaling") is not None
    ):
        raise ValueError("XPress target layer/RoPE mismatch")
    digest = hashlib.sha256(config_raw)
    index = path / "model.safetensors.index.json"
    mapping = None
    if index.exists():
        raw = index.read_bytes()
        digest.update(raw)
        mapping = _decode_unique_json(raw, "XPress index").get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(not isinstance(n, str) for n in mapping.values())
        ):
            raise ValueError("Invalid XPress weight index")
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
            raise ValueError("Invalid XPress shard path")
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
    record,
    target_model,
    *,
    runtime_quantization=None,
    num_passes=None,
    draft_attention_windows=None,
):
    if runtime_quantization is not None:
        raise ValueError("XPress runtime quantization is not supported")
    from ..runtime.drafters.attention_windows import validate_attention_windows

    args = record["args"]
    validate_attention_windows(draft_attention_windows, args.num_hidden_layers)
    if args != XPressConfig.from_dict(record["config"]):
        raise ValueError("XPress load configuration differs from inspected source")
    if num_passes is not None:
        if type(num_passes) is not int or not 1 <= num_passes <= args.block_size:
            raise ValueError("XPress num_passes must be in [1, trained block size]")
        args = replace(args, xpress_num_passes=num_passes)
    if (
        hashlib.sha256((Path(record["path"]) / "config.json").read_bytes()).hexdigest()
        != record["config_sha256"]
    ):
        raise ValueError("XPress config changed since inspection")
    # Reconcile the current artifact before importing MLX or allocating payload.
    # Target model identity has already been checked by the serving adapter.
    paths = [Path(record["path"]) / name for name, _, _ in record["files"]]
    for path, (_, size, mtime) in zip(paths, record["files"]):
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (size, mtime):
            raise ValueError("XPress artifact changed since inspection")
    if (
        _validate_headers(
            paths,
            None,
            record["args"],
            record["config"].get("dtype", record["config"].get("torch_dtype")),
        )
        != record["header_sha256"]
    ):
        raise ValueError("XPress header changed since inspection")
    if [_file_sha256(path) for path in paths] != record["weight_sha256"]:
        raise ValueError("XPress tensor bytes changed since inspection")
    import mlx.core as mx

    from ..runtime.drafters.xpress import XPressDraftModel

    model = XPressDraftModel(args, draft_attention_windows=draft_attention_windows)
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
