# SPDX-License-Identifier: MIT
"""APPROXIMATE Flash-Next MoE soft reap: keep the top N of the 10 routed experts.

Pierre 2026-11-10: try keep 8 / 6 / 4 on Flash-Next the way the DS4.1 Source
soft-reap probe dropped its lowest-weight experts, and measure the quality
cost against any speed gain.

Mechanism.  Every trunk ``SparseMoeBlock`` routes with ``top_k = N`` instead
of 10.  Under ``norm_topk_prob`` (Flash-Next's routing) taking the top N of
the softmax and renormalising over them is exactly "drop the 10 - N lowest
combine weights of the native top-10 and rescale the kept ones to the same
row total" (softmax is monotone, so the top-N set is the top N of the
top-10).  The dropped experts are never gathered: the routed GEMMs run N
slots, not 10.  The shared expert and its gate are unchanged.  The native
MTP head keeps its own exact routing (it only drafts; the reaped target
verifies every committed token).

Arithmetic.  The routed-expert fast paths (fused router, one-launch top-k,
top-k fold, MoE row windows, routed decode, fused expert down) are bound to
the ``qwen4-exp-e512-k10-v1`` routing contract and refuse any other top-k, so
reaped blocks are put on the stock gather arithmetic explicitly instead of
relying on each refusal.  A fair speed comparison therefore needs the exact
``moe_keep_topk_stock: true`` arm (top-10 on the same stock arithmetic) next
to the reaped arms; the default exact arm (fused kernels) is a separate
reference.

Isolation.  Approximate and unqualified.  The route identity carries the keep
count (``batch_config``), and APCv2 reuse and publication are refused for the
whole route (``apcv2_reuse_disabled_reason``), so no reaped state is ever
served to, or read from, an exact model.  Default ``moe_keep_topk = 10`` with
``moe_keep_topk_stock = false`` installs nothing.
"""
from __future__ import annotations

NATIVE_TOP_K = 10


def reap_reason(keep: int) -> str:
    if int(keep) == NATIVE_TOP_K:
        return "moe_keep_topk_stock_control"
    return f"approximate_moe_soft_reap_keep{int(keep)}"


def _trunk_blocks(model):
    layers = model.language_model.model.layers
    return [(index, layer.mlp) for index, layer in enumerate(layers)]


def _counting_switch(switch):
    """Count routed slots per call on this switch instance (engagement)."""
    base = type(switch)
    if getattr(base, "_mlx2_soft_reap_counting", False):
        return
    counting = type(
        f"SoftReapCounting{base.__name__}",
        (base,),
        {
            "_mlx2_soft_reap_counting": True,
            "__call__": _counting_call(base),
        },
    )
    object.__setattr__(switch, "_mlx2_reap_calls", 0)
    object.__setattr__(switch, "_mlx2_reap_slots", 0)
    switch.__class__ = counting


def _counting_call(base):
    def __call__(self, x, indices, *args, **kwargs):
        object.__setattr__(self, "_mlx2_reap_calls", self.__dict__.get("_mlx2_reap_calls", 0) + 1)
        object.__setattr__(
            self, "_mlx2_reap_slots", self.__dict__.get("_mlx2_reap_slots", 0) + int(indices.shape[-1])
        )
        return base.__call__(self, x, indices, *args, **kwargs)

    return __call__


def install(model, keep: int, *, stock: bool):
    """Apply the keep count and stock MoE arithmetic to every trunk block.

    Returns the status record, or None when nothing is selected."""
    keep = int(keep)
    blocks = _trunk_blocks(model)
    native = {int(block.top_k) for _, block in blocks}
    if len(native) != 1:
        raise ValueError(f"soft reap expects one native top-k, found {sorted(native)}")
    native = native.pop()
    if keep == native and not stock:
        return None
    if not 1 <= keep <= native:
        raise ValueError(f"moe_keep_topk {keep} must be within 1..{native} (the native top-k)")
    for _, block in blocks:
        if not block.norm_topk_prob:
            raise ValueError("soft reap needs norm_topk_prob routing (renormalised kept weights)")
        if getattr(block, "shared_folded", False):
            raise ValueError("soft reap does not support a folded shared expert row")
        if block.switch_mlp.__dict__.get("_mlx2_int8_experts") is not None:
            raise ValueError("soft reap is not composed with int8 expert prefill")
        block.force_stock_for_expert_streaming()
        block.set_moe_topk_mode("off")
        block.set_moe_router_mode("stock")
        if getattr(block, "moe_weighted_sum", False):
            block.set_moe_weighted_sum(False)
        block.top_k = keep
        _counting_switch(block.switch_mlp)
    record = {
        "schema": "mlx2.flash-next-soft-reap.v1",
        "implemented": True,
        "qualified": False,
        "approximate": keep != native,
        "keep_top_k": keep,
        "native_top_k": native,
        "arithmetic": "stock_gather",
        "blocks": len(blocks),
        "mtp_head": "native top-10 (drafter only)",
        "apcv2": "reuse refused (route-level)",
        "prefill_handoff": "refused (route-level)",
    }
    return record


def switches(model):
    """The trunk switch modules, captured once at install for diagnostics
    (which must never walk the live module tree)."""
    return tuple(block.switch_mlp for _, block in _trunk_blocks(model))


def status(record, captured):
    if record is None:
        return {"selected": False}
    calls = slots = 0
    for switch in captured:
        calls += switch.__dict__.get("_mlx2_reap_calls", 0)
        slots += switch.__dict__.get("_mlx2_reap_slots", 0)
    return {**record, "selected": True, "switch_calls": calls,
            "routed_slots_per_call": (slots / calls) if calls else None}
