"""Adapter-owned runtime quantization for Nemotron external draft models."""

AFFINE_Q8_G64 = "affine_q8_g64"


def normalize_runtime_quantization(value):
    if value is None:
        return None
    if value != AFFINE_Q8_G64:
        raise ValueError("Nemotron draft weight quantization must be affine_q8_g64")
    return value


def apply_runtime_quantization(model, value):
    """Apply deterministic weight-only Q8 to compatible draft matrices.

    The BF16 artifact remains immutable and identity-bound. Small matrices whose
    input width cannot form group-64 blocks retain their checkpoint dtype.
    """
    value = normalize_runtime_quantization(value)
    if value is None:
        model.runtime_weight_quantization = None
        return None

    from mlx import nn

    modules = []

    def predicate(path, module):
        weight = getattr(module, "weight", None)
        accepted = (
            hasattr(module, "to_quantized")
            and weight is not None
            and weight.ndim == 2
            and int(weight.shape[-1]) % 64 == 0
        )
        if accepted:
            modules.append(path)
        return accepted

    nn.quantize(
        model,
        group_size=64,
        bits=8,
        mode="affine",
        class_predicate=predicate,
    )
    receipt = {
        "schema": "mlx2.nemotron35-draft-runtime-quantization.v1",
        "kind": AFFINE_Q8_G64,
        "mode": "affine",
        "bits": 8,
        "group_size": 64,
        "modules": sorted(modules),
        "module_count": len(modules),
        "source_artifact_unchanged": True,
        "qualified": False,
    }
    model.runtime_weight_quantization = receipt
    return receipt
