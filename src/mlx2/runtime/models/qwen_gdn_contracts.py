"""Qwen-family fused-GDN contracts and legacy configuration translation."""

from .gdn_kernel_contract import (
    FusedGdnContract,
    GdnGeometry,
    GdnQkNormalization,
    resolve_fused_gdn_contract,
)

QWEN4_GDN_CONTRACT = FusedGdnContract(
    "qwen4-exp-gdn-v1", GdnGeometry(16, 48, 128, 128, 4), "sigmoid",
    True, False, GdnQkNormalization.ACTIVATION_L2,
    "batch-hv-dv-dk-f32-accum-v1", ("float32", "float16"),
)
AGNES_GDN_CONTRACT = FusedGdnContract(
    "agnes-gdn-v1", GdnGeometry(16, 48, 128, 128, 4), "swish",
    False, True, GdnQkNormalization.FLOAT_RMS_ASYMMETRIC,
    "batch-hv-dv-dk-f32-accum-v1", ("float32", "float16"),
)
QWEN38_GDN_CONTRACT = FusedGdnContract(
    "qwen38-27b-gdn-v1", GdnGeometry(16, 48, 128, 128, 4), "swish",
    False, True, GdnQkNormalization.FLOAT_RMS_ASYMMETRIC,
    "batch-hv-dv-dk-f32-accum-v1", ("float32", "float16"),
)
QWEN36_GDN_CONTRACT = FusedGdnContract(
    "qwen36-35b-gdn-v1", GdnGeometry(16, 32, 128, 128, 4), "swish",
    False, True, GdnQkNormalization.FLOAT_RMS_ASYMMETRIC,
    "batch-hv-dv-dk-f32-accum-v1", ("float32", "float16"),
)

_LEGACY_QWEN_ARCHITECTURES = {
    "qwen4": QWEN4_GDN_CONTRACT,
    "agnes": AGNES_GDN_CONTRACT,
    "qwen38": QWEN38_GDN_CONTRACT,
    "qwen35": QWEN36_GDN_CONTRACT,
}


def resolve_qwen_gdn_contract(contract, architecture=None):
    """Translate legacy Qwen route strings at the model-owned boundary."""
    legacy = None
    if architecture is not None:
        legacy = _LEGACY_QWEN_ARCHITECTURES.get(architecture)
        if legacy is None:
            return None, f"unsupported Qwen GDN architecture {architecture!r}"
    selected = contract or legacy
    resolved, refusal = resolve_fused_gdn_contract(selected)
    if refusal is not None:
        return resolved, refusal
    if contract is not None and legacy is not None and contract != legacy:
        return None, "fused GDN contract disagrees with legacy Qwen architecture"
    return resolved, None


def uses_activation_dtype_arithmetic(contract: FusedGdnContract) -> bool:
    """Select the existing Qwen kernel variant from explicit arithmetic facts."""
    selected, refusal = resolve_fused_gdn_contract(contract)
    if refusal is not None or selected is None:
        raise ValueError(refusal or "missing fused GDN contract")
    activation_key = (
        "swish", False, True, GdnQkNormalization.FLOAT_RMS_ASYMMETRIC,
    )
    float_decay_key = (
        "sigmoid", True, False, GdnQkNormalization.ACTIVATION_L2,
    )
    if selected.arithmetic_key == activation_key:
        return True
    if selected.arithmetic_key == float_decay_key:
        return False
    raise ValueError(
        f"Qwen fused kernel does not implement arithmetic {selected.arithmetic_key!r}"
    )
