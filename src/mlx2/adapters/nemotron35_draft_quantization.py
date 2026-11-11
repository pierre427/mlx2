"""Adapter-owned runtime quantization for Nemotron external draft models."""

AFFINE_Q8_G64 = "affine_q8_g64"
# Affine 4-bit, group 64 (MLX default affine mode).  Implemented, not
# qualified, not selected; BF16 stays the default.  GPU A/B pending
# (provenance/nemotron35-draft-affine-q4.json).
AFFINE_Q4_G64 = "affine_q4_g64"

PROFILES = {
    AFFINE_Q8_G64: {"mode": "affine", "bits": 8, "group_size": 64},
    AFFINE_Q4_G64: {"mode": "affine", "bits": 4, "group_size": 64},
}


def normalize_runtime_quantization(value):
    if value is None:
        return None
    if value not in PROFILES:
        raise ValueError(
            "Nemotron draft weight quantization must be one of "
            + ", ".join(sorted(PROFILES))
        )
    return value


def apply_runtime_quantization(model, value):
    """Apply a deterministic weight-only affine profile to compatible matrices.

    The BF16 artifact remains immutable and identity-bound. Small matrices whose
    input width cannot form group-size blocks retain their checkpoint dtype.
    """
    value = normalize_runtime_quantization(value)
    if value is None:
        model.runtime_weight_quantization = None
        return None

    from mlx import nn

    profile = PROFILES[value]
    group_size = profile["group_size"]
    modules = []

    def predicate(path, module):
        weight = getattr(module, "weight", None)
        accepted = (
            hasattr(module, "to_quantized")
            and weight is not None
            and weight.ndim == 2
            and int(weight.shape[-1]) % group_size == 0
        )
        if accepted:
            modules.append(path)
        return accepted

    nn.quantize(
        model,
        group_size=group_size,
        bits=profile["bits"],
        mode=profile["mode"],
        class_predicate=predicate,
    )
    receipt = {
        "schema": "mlx2.nemotron35-draft-runtime-quantization.v1",
        "kind": value,
        "mode": profile["mode"],
        "bits": profile["bits"],
        "group_size": group_size,
        "modules": sorted(modules),
        "module_count": len(modules),
        "source_artifact_unchanged": True,
        "qualified": False,
    }
    model.runtime_weight_quantization = receipt
    return receipt
