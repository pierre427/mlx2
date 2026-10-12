"""Typed, model-owned contracts for shared routed-MoE kernels.

The shared router/window code is deliberately ignorant of model names.  A
model may enter it only by supplying one of these immutable descriptions of
the eager routing law and physical expert layout.  Missing or inconsistent
descriptions are refusals; callers retain their ordinary/reference body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProjectionLayout = Literal["fused_gate_up", "split_gate_up"]
RowConsumer = Literal["row_exact", "batch_decode", "verify"]


@dataclass(frozen=True, slots=True)
class MoeRoutingContract:
    owner: str
    expert_count: int
    top_k: int
    normalize_topk: bool
    selection_law: Literal["precise_softmax_argpartition"]
    tie_order: Literal["stable_ascending_expert"]
    slot_order: Literal["ascending_selected_probability"]
    activation_dtype: Literal["bfloat16"]
    projection_layouts: frozenset[ProjectionLayout]
    quantization: Literal["affine_q4_group64"]
    expert_kernel_layout: Literal[
        "qmv_down_non512_intermediate", "qmv_fast_down_h2048_i512"
    ]
    shared_expert_form: Literal["dense_swiglu_sigmoid_gate"]
    allowed_row_consumers: frozenset[RowConsumer]

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("MoE routing contract needs an owner")
        if self.expert_count <= 0 or not 0 < self.top_k <= self.expert_count:
            raise ValueError("invalid MoE expert/top-k geometry")
        if not self.projection_layouts:
            raise ValueError("MoE routing contract needs a projection layout")
        unknown = self.allowed_row_consumers - {
            "row_exact", "batch_decode", "verify"
        }
        if unknown:
            raise ValueError(f"unknown MoE row consumers: {sorted(unknown)}")

    def routing_refusal(
        self,
        *,
        expert_count: int,
        top_k: int,
        normalize_topk: bool,
        activation_dtype: str,
        projection_layout: ProjectionLayout | None = None,
        consumer: str | None = None,
    ) -> str | None:
        if expert_count != self.expert_count or top_k != self.top_k:
            return (
                f"declared {self.owner} is E{self.expert_count}/top-{self.top_k}, "
                f"got E{expert_count}/top-{top_k}"
            )
        if normalize_topk is not self.normalize_topk:
            return "routing normalization differs from the declared contract"
        if activation_dtype != self.activation_dtype:
            return (
                f"activation dtype {activation_dtype} differs from "
                f"{self.activation_dtype}"
            )
        if (
            projection_layout is not None
            and projection_layout not in self.projection_layouts
        ):
            return (
                f"projection layout {projection_layout!r} is not allowed by "
                f"{self.owner}"
            )
        if consumer is not None and consumer not in self.allowed_row_consumers:
            return f"row consumer {consumer!r} is not allowed by {self.owner}"
        return None


def __getattr__(name):
    """Compatibility export while definitions remain model-owned."""
    if name in {"QWEN36_E256_K8_ROUTING", "QWEN4_E512_K10_ROUTING"}:
        from . import qwen_moe_contracts

        return getattr(qwen_moe_contracts, name)
    raise AttributeError(name)


def contract_refusal(contract: object) -> str | None:
    if not isinstance(contract, MoeRoutingContract):
        return "missing typed model-owned MoE routing contract"
    if contract.selection_law != "precise_softmax_argpartition":
        return "router selection law is not implemented by the shared kernel"
    if contract.tie_order != "stable_ascending_expert":
        return "router tie order is not implemented by the shared kernel"
    if contract.slot_order != "ascending_selected_probability":
        return "router slot order is not implemented by the shared kernel"
    if contract.quantization != "affine_q4_group64":
        return "expert quantization is not implemented by the shared kernel"
    if contract.shared_expert_form != "dense_swiglu_sigmoid_gate":
        return "shared-expert form is not implemented by the shared kernel"
    if contract.expert_kernel_layout not in {
        "qmv_down_non512_intermediate", "qmv_fast_down_h2048_i512"
    }:
        return "expert kernel layout is not implemented by the shared kernel"
    return None
