"""Restricted, reversible LoRA mutation for an idle serving adapter.

Derived from the LoRALinear mechanism in mlx-lm-unified at revision
13eb83388750435bcc751d0b9e33857a38152544.  This serving port deliberately
supports only explicit Linear/QuantizedLinear keys, one active adapter, exact
tensor coverage, and reversible module replacement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def read_lora_config(path):
    path = Path(path).expanduser().resolve()
    config_path = path / "adapter_config.json"
    weights_path = path / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("LoRA directory requires adapter_config.json and adapters.safetensors")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("fine_tune_type", "lora") != "lora":
        raise ValueError("dynamic serving supports LoRA adapters only")
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict):
        raise ValueError(  # noqa: TRY004 - artifact validation is one ValueError API
            "LoRA config requires lora_parameters"
        )
    allowed = {"rank", "scale", "dropout", "keys"}
    if set(parameters) - allowed:
        raise ValueError("unsupported LoRA parameters: " + ", ".join(sorted(set(parameters) - allowed)))
    rank = parameters.get("rank", 8)
    scale = parameters.get("scale", 20.0)
    dropout = parameters.get("dropout", 0.0)
    keys = parameters.get("keys")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 1024:
        raise ValueError("LoRA rank must be an integer from 1 to 1024")
    if not isinstance(scale, (int, float)) or isinstance(scale, bool):
        raise ValueError(  # noqa: TRY004 - artifact validation is one ValueError API
            "LoRA scale must be numeric"
        )
    if dropout != 0 and dropout != 0.0:
        raise ValueError("serving LoRA dropout must be zero")
    if not isinstance(keys, list) or not keys or any(
        not isinstance(key, str) or not key or key.startswith(".") for key in keys
    ):
        raise ValueError("serving LoRA requires a nonempty explicit keys list")
    if len(set(keys)) != len(keys):
        raise ValueError("LoRA keys must be unique")
    return path, {"rank": rank, "scale": float(scale), "keys": tuple(keys)}


@dataclass
class LoRASession:
    name: str
    path: str
    originals: dict

    def restore(self, model):
        from mlx.utils import tree_unflatten

        model.update_modules(tree_unflatten(list(self.originals.items())))


def install_lora(model, *, name, path):
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten, tree_unflatten

    path, config = read_lora_config(path)

    class ServingLoRALinear(nn.Module):
        def __init__(self, linear):
            super().__init__()
            output_dims, packed_input_dims = linear.weight.shape
            input_dims = packed_input_dims
            if isinstance(linear, nn.QuantizedLinear):
                input_dims = packed_input_dims * 32 // linear.bits
            self.linear = linear
            self.scale = config["scale"]
            self.lora_a = mx.zeros((input_dims, config["rank"]))
            self.lora_b = mx.zeros((config["rank"], output_dims))

        def __call__(self, value):
            update = (value @ self.lora_a) @ self.lora_b
            return self.linear(value) + (self.scale * update).astype(value.dtype)

    modules = dict(model.named_modules())
    originals = {}
    replacements = []
    for key in config["keys"]:
        module = modules.get(key)
        if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(  # noqa: TRY004 - the selected key is incompatible
                f"LoRA key {key!r} is not a Linear/QuantizedLinear module"
            )
        originals[key] = module
        replacements.append((key, ServingLoRALinear(module)))
    model.update_modules(tree_unflatten(replacements))
    try:
        weights = mx.load(str(path / "adapters.safetensors"))
        parameters = dict(tree_flatten(model.parameters()))
        expected = {
            f"{key}.{leaf}"
            for key in config["keys"]
            for leaf in ("lora_a", "lora_b")
        }
        if set(weights) != expected:
            missing = sorted(expected - set(weights))
            extra = sorted(set(weights) - expected)
            raise ValueError(f"LoRA tensor coverage mismatch; missing={missing}, extra={extra}")
        for key, value in weights.items():
            if key not in parameters or tuple(value.shape) != tuple(parameters[key].shape):
                expected_shape = tuple(parameters[key].shape) if key in parameters else None
                raise ValueError(
                    f"LoRA tensor {key!r} has shape {tuple(value.shape)}, expected {expected_shape}"
                )
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
    except BaseException:
        model.update_modules(tree_unflatten(list(originals.items())))
        raise
    return LoRASession(name=name, path=str(path), originals=originals)
