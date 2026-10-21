"""Strict CPU binding for the published RedHat Super VL DSpark head."""
import hashlib
import json
from pathlib import Path

from ..conversion.nvfp4 import read_header, sha256
from ..runtime.drafters.base_config import DFlashConfig
from .dspark import DSparkConfig as GenericDSparkConfig
from .dspark import expected_weight_shapes
from .nemotron35_super_config import language_config

CONFIG_SHA256 = "88462ccd207ac7238f656bcab777cc320ef0a006e66702ac3b09736da6404365"
TAP_CONVENTION = "speculators-hidden-state-index-to-post-layer-v1"


def post_layer_ids(aux_ids, num_layers):
    # Speculators/vLLM indices include embedding state zero and capture
    # decoder output at idx+1. mlx2's target taps index output AFTER layer idx.
    # This head never requests the embedding; do not silently map it to -1.
    if (not isinstance(aux_ids, list) or not aux_ids
            or any(type(index) is not int or not 1 <= index <= num_layers for index in aux_ids)
            or aux_ids != sorted(set(aux_ids))):
        raise ValueError("Unsupported DSpark auxiliary hidden-state indices")
    return [index - 1 for index in aux_ids]


class DSparkConfig(GenericDSparkConfig):
    """Speculators-format (RedHat Super VL) DSpark configuration."""

    @classmethod
    def from_dict(cls, config):
        text = config["transformer_layer_config"]
        if (config.get("architectures") != ["DSparkDraftModel"]
                or config.get("speculators_model_type") != "dspark"
                or config.get("markov_head_type") != "vanilla"
                or config.get("sliding_window_non_causal") is not False
                or text.get("rope_parameters") != {"rope_theta": 10000., "rope_type": "default"}
                or text.get("hidden_act") != "silu" or text.get("attention_dropout") != 0.):
            raise ValueError("Unsupported DSpark source contract")
        fields = {key: value for key, value in text.items() if key in DFlashConfig.__dataclass_fields__}
        fields.update(block_size=config["block_size"] + 1, mask_token_id=config["mask_token_id"],
                      target_layer_ids=post_layer_ids(config["aux_hidden_state_layer_ids"], 88), num_target_layers=88,
                      rope_theta=10000., markov_rank=config["markov_rank"],
                      sample_from_anchor=config["sample_from_anchor"],
                      enable_confidence_head=config["enable_confidence_head"],
                      confidence_head_with_markov=config["confidence_head_with_markov"])
        return cls(**fields)


def inspect_drafter(path, target):
    path = Path(path).expanduser().resolve()
    if sha256(path / "config.json") != CONFIG_SHA256:
        raise ValueError("DSpark config differs from the pinned Super VL release")
    config = json.loads((path / "config.json").read_text())
    args = DSparkConfig.from_dict(config)
    target_config = language_config(json.loads((Path(target) / "config.json").read_text()))
    if (any(target_config[key] != getattr(args, key) for key in ("hidden_size", "vocab_size"))
            or target_config["num_hidden_layers"] != args.num_target_layers):
        raise ValueError("DSpark target feature/embedding contract differs")
    file = path / "model.safetensors"
    if file.is_symlink() or not file.is_file():
        raise ValueError("Missing or escaped DSpark weights")
    header, _ = read_header(file)
    expected = expected_weight_shapes(args)
    actual = {key: entry for key, entry in header.items() if key != "__metadata__"}
    if (set(actual) != set(expected) or any(entry["shape"] != expected[key] or entry["dtype"] != "BF16" for key, entry in actual.items())):
        raise ValueError("DSpark tensor schema differs from the trained head")
    weight_hash, stat = sha256(file), file.stat()
    digest = hashlib.sha256((path / "config.json").read_bytes() + weight_hash.encode())
    digest.update(TAP_CONVENTION.encode())
    return {"path": str(path), "config": config, "args": args,
            "files": [(file.name, stat.st_size, stat.st_mtime_ns)],
            "weight_sha256": weight_hash, "fingerprint": digest.hexdigest(),
            "tap_convention": TAP_CONVENTION}


def load_drafter(record, target_model, *, runtime_quantization=None):
    from .nemotron35_draft_quantization import (
        apply_runtime_quantization,
        normalize_runtime_quantization,
    )

    runtime_quantization = normalize_runtime_quantization(runtime_quantization)
    path = Path(record["path"])
    if sha256(path / "config.json") != CONFIG_SHA256 or DSparkConfig.from_dict(record["config"]) != record["args"]:
        raise ValueError("DSpark config changed since inspection")
    if sha256(path / "model.safetensors") != record["weight_sha256"]:
        raise ValueError("DSpark weights changed since inspection")
    import mlx.core as mx

    from ..runtime.drafters.nemotron35_dspark import DSparkDraftModel
    model = DSparkDraftModel(record["args"])
    weights = mx.load(str(path / "model.safetensors"))
    mx.eval(list(weights.values()))
    model.load_weights(list(weights.items()), strict=True)
    if runtime_quantization is not None:
        apply_runtime_quantization(model, runtime_quantization)
    else:
        model.runtime_weight_quantization = None
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    return model.bind(target_model)
