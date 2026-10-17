"""Report-guided mixed precision, independent of tensor payloads or MLX."""
import math

from ..adapters.nemotron35_super_config import tensor_schema
from .nvfp4 import DTYPE_BYTES

RECIPE = "nemotron35-super-vl-affine-experts4-protected8-v1"


def precision_for(name, entry, *, expert_bits=4):
    if expert_bits not in (4, 5, 6, 8):
        raise ValueError("Unsupported expert precision")
    if not name.startswith("language_model.") or not name.endswith(".weight"):
        return None
    if ".switch_mlp." in name:
        return {"bits": expert_bits, "group_size": 64, "mode": "affine"}
    # Preserve the report's sensitive latent bridges, embedding and MTP
    # attention/fusion. Norms, routers, convolutions and recurrent scalars
    # remain at source precision. Shared experts receive an 8-bit floor.
    if (len(entry["shape"]) != 2 or ".conv1d." in name or ".gate." in name
            or "latent_proj." in name or ".embeddings." in name
            or name.startswith("language_model.mtp.layers.0.")):
        return None
    return {"bits": 8, "group_size": 64, "mode": "affine"}


def make_plan(config, *, expert_bits=4):
    output, modules, categories = {}, {}, {}
    for name, entry in tensor_schema(config, fused=True).items():
        quant = precision_for(name, entry, expert_bits=expert_bits)
        if quant is None:
            output[name] = dict(entry)
            category = "source_precision"
        else:
            prefix, shape = name.removesuffix(".weight"), entry["shape"]
            bits, group = quant["bits"], quant["group_size"]
            if shape[-1] % group or shape[-1] * bits % 32:
                raise ValueError("Invalid mixed-quant group geometry")
            output[name] = {"shape": [*shape[:-1], shape[-1] * bits // 32], "dtype": "U32"}
            for suffix in (".scales", ".biases"):
                output[prefix + suffix] = {"shape": [*shape[:-1], shape[-1] // group], "dtype": "BF16"}
            modules[prefix] = quant
            category = "routed_experts" if ".switch_mlp." in name else "protected_8bit"
        categories[category] = categories.get(category, 0) + sum(
            math.prod(value["shape"]) * DTYPE_BYTES[value["dtype"]]
            for key, value in output.items() if key == name or key in {name.removesuffix(".weight") + ".scales", name.removesuffix(".weight") + ".biases"}
        )
    return {"recipe": RECIPE, "expert_bits": expert_bits, "schema": output,
            "modules": modules, "tensor_bytes": sum(categories.values()),
            "bytes_by_category": categories, "activation_aware": False,
            "calibration_requirement": "Bound image-text corpus, channel moments and per-expert token counts; no oQe claim before coverage validation",
            "qualified": False, "selected": False}
