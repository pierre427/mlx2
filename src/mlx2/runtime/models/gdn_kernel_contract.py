"""Model-neutral fused-GDN arithmetic and geometry contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GdnQkNormalization(str, Enum):
    """Arithmetic order used to normalize recurrent query/key features."""

    ACTIVATION_L2 = "activation_l2"
    FLOAT_RMS_ASYMMETRIC = "float_rms_asymmetric"


@dataclass(frozen=True)
class GdnGeometry:
    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int


@dataclass(frozen=True)
class FusedGdnContract:
    identity: str
    geometry: GdnGeometry
    output_gate: str
    decay_input_float32: bool
    beta_input_float32: bool
    qk_normalization: GdnQkNormalization
    state_layout: str
    recurrent_storage_classes: tuple[str, ...]

    @property
    def arithmetic_key(self) -> tuple[str, bool, bool, GdnQkNormalization]:
        """Generic source-selection properties, independent of model names."""
        return (
            self.output_gate,
            self.decay_input_float32,
            self.beta_input_float32,
            self.qk_normalization,
        )

    def refusal(self) -> str | None:
        if not self.identity:
            return "empty fused GDN contract identity"
        if self.output_gate not in ("sigmoid", "swish"):
            return f"unsupported output gate contract {self.output_gate!r}"
        if type(self.decay_input_float32) is not bool:
            return "decay input precision must be an explicit boolean"
        if type(self.beta_input_float32) is not bool:
            return "beta input precision must be an explicit boolean"
        if not isinstance(self.qk_normalization, GdnQkNormalization):
            return f"unsupported QK normalization {self.qk_normalization!r}"
        if (
            min(
                self.geometry.num_key_heads,
                self.geometry.num_value_heads,
                self.geometry.key_head_dim,
                self.geometry.value_head_dim,
                self.geometry.conv_kernel,
            )
            <= 0
        ):
            return f"invalid fused GDN geometry {self.geometry!r}"
        if self.geometry.num_value_heads % self.geometry.num_key_heads:
            return "value-head count is not divisible by key-head count"
        if self.state_layout != "batch-hv-dv-dk-f32-accum-v1":
            return f"unsupported fused GDN state layout {self.state_layout!r}"
        if self.recurrent_storage_classes != ("float32", "float16"):
            return "unsupported fused GDN recurrent storage contract"
        return None


def resolve_fused_gdn_contract(
    contract: FusedGdnContract | None,
) -> tuple[FusedGdnContract | None, str | None]:
    """Validate an explicit model-supplied arithmetic contract."""
    if contract is not None and not isinstance(contract, FusedGdnContract):
        return None, f"invalid fused GDN contract {type(contract).__name__}"
    if contract is None:
        return None, "missing fused GDN contract"
    return contract, contract.refusal()


def arithmetic_key_for_contract(
    contract: FusedGdnContract | None,
) -> tuple[str, bool, bool, GdnQkNormalization]:
    """Return generic compiled arithmetic properties or fail closed."""
    selected, refusal = resolve_fused_gdn_contract(contract)
    if refusal is not None or selected is None:
        raise ValueError(refusal or "missing fused GDN contract")
    return selected.arithmetic_key
