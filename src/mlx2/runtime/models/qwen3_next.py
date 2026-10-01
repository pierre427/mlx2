# SPDX-License-Identifier: MIT
# Adapted from unified; see provenance/.
from __future__ import annotations

_GLUE_LAST_RECEIPT: Optional[dict] = None
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)
from .activations import swiglu
from .base import BaseModelArgs
from .precise_ops import gate_sigmoid
from .qwen4_moe_router import (
    admit_qwen4_moe_router,
    probe_qwen4_moe_router,
    qwen4_moe_router,
)
from . import switch_layers as _switch_layers
from . import qwen4_moe_weighted_sum as _moe_wsum
from . import qwen4_routed_decode as _routed
from . import qwen4_moe_window as _window
from .switch_layers import (
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)

logger = logging.getLogger(__name__)


class MaterializationTooLarge(RuntimeError):
    """A runtime weight materialization does not fit in device memory."""


def _env_flag(name: str, default: bool = False) -> bool:
    """Read one performance flag once, at import time.

    ``default=True`` carries a lever that has been PROMOTED into the shipped
    path: unset means ON, and an operator turns it OFF with ``=0`` instead of
    on with ``=1``.  An unset or empty value takes ``default``; any set value
    is parsed, so ``=0`` reverts a promoted lever.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


_MOE_GATE_COMPILE = _env_flag("MLX_QWEN4_MOE_GATE_COMPILE")
_MOE_ROUTER_KERNEL = _env_flag("MLX_QWEN4_MOE_ROUTER_KERNEL")
_MOE_ROUTER_MODES = ("stock", "fused")
# omlx #3903 port: sorted-order weighted sum for prefill; default off.
_MOE_WEIGHTED_SUM = _moe_wsum.weighted_sum_enabled_from_env()
# omlx #3912 port: one-token routed experts in fewer launches; default off.
_MOE_ROUTED_DECODE = _routed.mode_from_env()
# omlx #4106/#4041/#4052 port: routed MoE row windows and the router top-k
# launch/fold; default off (no consumer, top-k "off").
_MOE_WINDOW_CONSUMERS = _window.consumers_from_env()
_MOE_TOPK_MODE = _window.topk_mode_from_env()
_MOE_GATE_COMPILE_MAX_TOKENS = 8
_COMPILE_GLUE_DEFAULT = False
_COMPILE_GLUE = _env_flag("MLX_QWEN4_COMPILE_GLUE", default=_COMPILE_GLUE_DEFAULT)
_GLUE_COMPILE_CACHE: Dict[Any, Any] = {}
_GLUE_STATS = {"builds": 0, "calls": 0, "fallbacks": 0, "skips": 0}


def compile_glue_enabled() -> bool:
    """Live read of the glue lever, so the toggle needs no module reload."""
    return _COMPILE_GLUE


def _run_glue(key, builder, *args):
    """Run one compiled span, or return ``None`` to mean "stay eager".

    Fail-closed: a span that raises is demoted for the life of the process and
    the caller answers from the eager arithmetic, which is the same math.
    """
    global _GLUE_LAST_RECEIPT
    for arg in args:
        if arg.dtype != mx.bfloat16:
            _GLUE_STATS["skips"] += 1
            return None
    compiled = _GLUE_COMPILE_CACHE.get(key, False)
    if compiled is False:
        compiled = mx.compile(builder(), shapeless=True)
        _GLUE_COMPILE_CACHE[key] = compiled
        _GLUE_STATS["builds"] += 1
    if compiled is None:
        return None
    try:
        out = compiled(*args)
    except Exception as exc:
        _GLUE_COMPILE_CACHE[key] = None
        _GLUE_STATS["fallbacks"] += 1
        _GLUE_LAST_RECEIPT = {"span": repr(key), "error": repr(exc)}
        return None
    _GLUE_STATS["calls"] += 1
    return out


def _build_hyper_gate(hc_count: int):

    def hyper_gate(down):
        return nn.silu(down / hc_count)

    return hyper_gate


def _build_hyper_mix():
    """Mean the already-gated streams. The sigmoid stays with the caller."""

    def hyper_mix(weights, streams):
        return mx.mean(weights * streams, axis=-2)

    return hyper_mix


def _build_inject_apply():

    def inject_apply(residual_streams, branch, inject):
        return residual_streams + branch[..., None, :] * inject[..., None]

    return inject_apply


def _build_moe_combine():
    """``gate`` arrives as the sigmoid output, computed by the caller."""

    def moe_combine(y, gate, shared_y):
        return y + gate * shared_y

    return moe_combine


_MOE_FUSED_GATE_UP = _env_flag("MLX_QWEN4_MOE_FUSED_GATE_UP", default=True)
_MOE_SHARED_IN_GATHER = _env_flag("MLX_QWEN4_MOE_SHARED_IN_GATHER")
_MOE_FUSED_EXPERT_MODES = ("stock", "auto", "scalar", "tile4")


def _fused_expert_mode_from_env() -> str:
    raw = os.environ.get("MLX_QWEN4_FUSED_EXPERT_KERNEL")
    if raw is None or not raw.strip():
        return "auto"
    value = raw.strip().lower()
    if value in {"0", "false", "off", "no", "stock"}:
        return "stock"
    if value in {"1", "true", "on", "yes", "auto"}:
        return "auto"
    if value in {"scalar", "tile4"}:
        return value
    logger.warning("Ignoring invalid MLX_QWEN4_FUSED_EXPERT_KERNEL=%r; using auto", raw)
    return "auto"


_MOE_FUSED_EXPERT_MODE = _fused_expert_mode_from_env()


def _proj_signature(module):
    """Eligibility signature of one expert projection module."""
    if "bias" in module:
        return None
    if isinstance(module, (QuantizedSwitchLinear, nn.QuantizedLinear)):
        return (
            "quantized",
            module.group_size,
            module.bits,
            getattr(module, "mode", "affine"),
            getattr(module, "biases", None) is not None,
        )
    if isinstance(module, (SwitchLinear, nn.Linear)):
        return ("float",)
    return None


def _proj_identity(module):
    """Identity key of every array a projection contributes to a table.

    Scales and biases are included so a partial ``update()`` that replaces
    only them (weight untouched) still invalidates the lazy tables.
    """
    return (
        module["weight"],
        getattr(module, "scales", None),
        getattr(module, "biases", None),
    )


def _proj_table(module):
    """(weight, scales, biases, group_size, bits, mode) view of a projection."""
    if _proj_signature(module)[0] == "quantized":
        return (
            module["weight"],
            module["scales"],
            getattr(module, "biases", None),
            module.group_size,
            module.bits,
            getattr(module, "mode", "affine"),
        )
    return (module["weight"], None, None, None, None, None)


def _concat_tables(tables, axis):
    """Concatenate projection tables along one weight axis, exactly.

    Quantization groups run along the input (K) axis of each output row, so
    concatenation along N (axis -2) or along the expert axis (0) preserves
    every group, scale, and bias byte-for-byte; re-fusing the split
    [gate|up] quantized tensors reproduces the checkpoint's fused layout.

    This is the choke point every fusion site funnels through, so the
    quant-parameter check lives here: parts that differ in mode, bits,
    group_size, or dtype cannot be re-fused without silently corrupting the
    scales, and ``mx.concatenate`` would not object. A future caller that
    forgets to pre-validate inherits this guard by construction (arXiv
    2609.04098 re-derives the fused-scale bug class).
    """
    _assert_concat_compatible(tables)
    weight = mx.concatenate([t[0] for t in tables], axis=axis)
    scales = biases = None
    if tables[0][1] is not None:
        scales = mx.concatenate([t[1] for t in tables], axis=axis)
        if tables[0][2] is not None:
            biases = mx.concatenate([t[2] for t in tables], axis=axis)
    return (weight, scales, biases, *tables[0][3:])


def _assert_concat_compatible(tables):
    """Fail loudly before fusing projection tables that do not match.

    A projection table is ``(weight, scales, biases, group_size, bits, mode)``.
    The valid (all-matching) case is untouched; a mismatch raises ValueError.
    Callers pre-validate today, so this never fires for them -- it exists to
    keep FUTURE fusion sites from concatenating incompatible quantized parts.
    """
    if not tables:
        raise ValueError("_concat_tables: no tables to concatenate")
    head = tables[0]
    head_quant = head[1] is not None
    for i, t in enumerate(tables[1:], start=1):
        if (t[1] is not None) != head_quant:
            raise ValueError(
                f"_concat_tables: cannot fuse quantized and non-quantized projection parts (part 0 quantized={head_quant}, part {i} quantized={t[1] is not None})"
            )
        if t[0].dtype != head[0].dtype:
            raise ValueError(
                f"_concat_tables: weight dtype mismatch (part 0 {head[0].dtype}, part {i} {t[0].dtype})"
            )
        if not head_quant:
            continue
        if (t[3], t[4], t[5]) != (head[3], head[4], head[5]):
            raise ValueError(
                f"_concat_tables: quant parameter mismatch (part 0 group_size/bits/mode={head[3]}/{head[4]}/{head[5]}, part {i}={t[3]}/{t[4]}/{t[5]})"
            )
        if t[1].dtype != head[1].dtype:
            raise ValueError(
                f"_concat_tables: scales dtype mismatch (part 0 {head[1].dtype}, part {i} {t[1].dtype})"
            )
        if (t[2] is None) != (head[2] is None):
            raise ValueError(
                f"_concat_tables: biases present on some parts but not others (part 0 has_biases={head[2] is not None}, part {i} has_biases={t[2] is not None})"
            )
        if t[2] is not None and t[2].dtype != head[2].dtype:
            raise ValueError(
                f"_concat_tables: biases dtype mismatch (part 0 {head[2].dtype}, part {i} {t[2].dtype})"
            )


def table_bytes(table) -> int:
    """Bytes a materialized projection table occupies."""
    return sum((part.nbytes for part in table[:3] if part is not None))


def _fmt_bytes(nbytes: float) -> str:
    """Size text that keeps a small request legible instead of rounding it away."""
    for unit, scale in (("GB", 1000000000.0), ("MB", 1000000.0), ("KB", 1000.0)):
        if abs(nbytes) >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes:.0f} B"


def _working_set_bytes() -> Optional[int]:
    """Recommended working-set size, or None when the device has no such limit.

    Only a POSITIVELY detected absence returns None. A broken API or a
    malformed reply propagates: a memory guard that fails open on its own bugs
    is worse than no guard at all, because it still reads as protection.
    """
    metal = getattr(mx, "metal", None)
    is_available = getattr(metal, "is_available", None)
    if is_available is not None and (not is_available()):
        return None
    info = getattr(mx, "device_info", None) or getattr(metal, "device_info", None)
    if info is None:
        return None
    reported = info()
    if "max_recommended_working_set_size" not in reported:
        return None
    budget = reported["max_recommended_working_set_size"]
    if budget is None or budget <= 0:
        raise ValueError(f"device reported an unusable working set: {budget!r}")
    return budget


def materialization_headroom(active: int, budget: int, headroom: float) -> float:
    """Bytes a NEW allocation may claim beside `active` bytes already held.

    The primary rule is the one this guard has always applied: hold `headroom`
    of the recommended working set back from TOTAL occupancy, leaving
    `budget * (1 - headroom) - active` for the request.

    That rule goes unsatisfiable the moment resident weights alone exceed it.
    A 104.3 GB model in a 120.3 GB working set leaves 102.2 - 104.3 < 0, so
    `active + nbytes <= allowed` is false for EVERY nbytes -- zero included --
    and the guard refuses 20 MB tables that plainly fit. Loading the model
    already spent the reserve; refusing a projection table cannot win it back.

    So when, and only when, the reserve is already gone, fall back to a
    DEGRADED allowance: the same fraction of what is still unclaimed, capped at
    `headroom` of the reserve that should have been there. The cap is the point
    -- uncapped, the allowance leaps from ~0 to 15.3 GB the instant a resident
    model crosses the reserve line, so being slightly bigger would buy a much
    larger claim. Capped, that step is 2.7 GB at the shipped constants: enough
    for the incidental per-layer tables this regime exists to stop refusing,
    and nowhere near a deliberate multi-GB materialization.

    The fallback is reachable only where the primary rule refused a request of
    nothing, so no allocation the primary rule judged -- admitted or refused --
    changes verdict.
    """
    if not 0.0 <= headroom < 1.0:
        raise ValueError(f"headroom must be in [0, 1), got {headroom!r}")
    reserved = budget * (1.0 - headroom) - active
    if reserved >= 0:
        return reserved
    degraded = max(0, budget - active) * (1.0 - headroom)
    return min(degraded, budget * headroom * headroom)


def check_materialization_budget(nbytes: int, what: str, headroom: float = 0.15):
    """Refuse a runtime materialization that does not fit in memory.

    A runtime weight table is a SECOND copy of resident weights, so size it
    against the remaining working-set allowance before building it. Returns the
    recorded estimate and raises MaterializationTooLarge when it does not fit.
    """
    budget = _working_set_bytes()
    if budget is None:
        return {"bytes": nbytes, "checked": False}
    active = mx.get_active_memory()
    allowed = materialization_headroom(active, budget, headroom)
    estimate = {
        "bytes": nbytes,
        "active_bytes": active,
        "budget_bytes": budget,
        "allowed_bytes": allowed,
        "headroom": headroom,
        "checked": True,
        "fits": nbytes <= allowed,
    }
    logger.debug("materialization estimate for %s: %s", what, estimate)
    if not estimate["fits"]:
        raise MaterializationTooLarge(
            f"{what} needs {_fmt_bytes(nbytes)} beside {_fmt_bytes(active)} active; the allowance is {_fmt_bytes(allowed)} of the {_fmt_bytes(budget)} recommended working set"
        )
    return estimate


def switch_layers_sort_min() -> int:
    """Read the sorted-gather threshold at call time so an A/B can switch it."""
    return _switch_layers._GATHER_SORT_MIN_ASSIGNMENTS


class FusedGateUpSwitchGLU(nn.Module):
    """SwitchGLU holding gate and up as one [gate|up] projection.

    One gather matmul of width 2*hidden_dims replaces two of width
    hidden_dims. The halves are split from the OUTPUT, so the weights are
    never copied.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=SwiGLU(),
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: Optional[mx.array] = None,
        variant: str = "scalar",
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= switch_layers_sort_min()
        idx = indices
        inv_order = None
        if do_sort:
            (x, idx, inv_order) = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        routed = _try_routed_decode(self, x, idx, scores, do_sort, variant)
        if routed is not None and routed[0] != "gate_up":
            object.__setattr__(self, "_last_fused_variant", f"routed_{routed[0]}")
            return routed[1]
        if routed is not None:
            hidden = routed[1]
        else:
            gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
            half = self.hidden_dims
            hidden = self.activation(gate_up[..., half:], gate_up[..., :half])
        fused = _try_qwen4_fused_down(
            hidden, idx, scores, self.down_proj, do_sort, variant
        )
        outcome = None
        if fused is not None:
            outcome = _fused_outcome(variant, idx)
        object.__setattr__(self, "_last_fused_variant", outcome)
        if fused is not None:
            return fused
        x = self.down_proj(hidden, idx, sorted_indices=do_sort)
        return _routed_tail(self, x, indices, inv_order, scores, do_sort)


class FusedDownSwitchGLU(SwitchGLU):
    """Stock split gate/up projections with the experimental fused down tail."""

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: Optional[mx.array] = None,
        variant: str = "scalar",
    ) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= switch_layers_sort_min()
        idx = indices
        inv_order = None
        if do_sort:
            (x, idx, inv_order) = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        routed = _try_routed_decode(self, x, idx, scores, do_sort, variant)
        if routed is not None and routed[0] != "gate_up":
            object.__setattr__(self, "_last_fused_variant", f"routed_{routed[0]}")
            return routed[1]
        if routed is not None:
            hidden = routed[1]
        else:
            hidden = self.activation(
                self.up_proj(x, idx, sorted_indices=do_sort),
                self.gate_proj(x, idx, sorted_indices=do_sort),
            )
        fused = _try_qwen4_fused_down(
            hidden, idx, scores, self.down_proj, do_sort, variant
        )
        outcome = None
        if fused is not None:
            outcome = _fused_outcome(variant, idx)
        object.__setattr__(self, "_last_fused_variant", outcome)
        if fused is not None:
            return fused
        x = self.down_proj(hidden, idx, sorted_indices=do_sort)
        return _routed_tail(self, x, indices, inv_order, scores, do_sort)


def _enable_routed_decode(switch_mlp, mode: str) -> None:
    """Attach the omlx #3912/#4113 routed-decode selection and its counters."""
    if mode not in _routed.MODES:
        raise ValueError(f"unknown routed decode mode {mode!r}; expected {_routed.MODES}")
    switch_mlp.routed_decode_mode = mode
    switch_mlp.routed_decode_calls = 0
    switch_mlp.routed_decode_fallbacks = 0
    switch_mlp.routed_decode_last_fallback = None
    # The down half (gate_up_down: served_down; two_launch: down_combine).
    switch_mlp.routed_down_calls = 0
    switch_mlp.routed_down_fallbacks = 0
    switch_mlp.routed_down_last_fallback = None


def _served_down_refusal(switch_mlp, inter, dtype, indices, scores, variant):
    """Why served_down may not stand in for the block's down here, or None.

    It reproduces the tile4 fused down only, so it runs only where the block
    would dispatch tile4 at width 1 on a resident table.
    """
    from .qwen4_fused_moe import admit_qwen4_fused_down, auto_variant

    down = switch_mlp.down_proj
    resolved = auto_variant(1) if variant == "auto" else variant
    if resolved != "tile4":
        return f"block down variant {variant!r} is not tile4"
    if type(down) is not QuantizedSwitchLinear:
        return "down is not a resident QuantizedSwitchLinear"
    admission = admit_qwen4_fused_down(
        _ShapeOnly(indices.shape + (inter,), dtype),
        indices,
        scores,
        down["weight"],
        down["scales"],
        down.get("biases"),
        num_experts=down.num_experts,
        group_size=down.group_size,
        bits=down.bits,
        mode=down.mode,
    )
    if not admission.accepted:
        return f"tile4 admission: {admission.reason}"
    return None


def _try_routed_decode(switch_mlp, x, indices, scores, do_sort, variant="scalar"):
    """``(mode, value)`` from the routed kernels, or None to run the composed body.

    ``gate_up`` returns the SwiGLU hidden in the composed body's shape
    ``indices.shape + (1, inter)``; ``gate_up_down`` and ``two_launch`` return
    the weighted routed sum ``indices.shape[:-1] + (hidden,)``. Only an
    unsorted one-token gather is a candidate, so multi-token forwards decline
    quietly, uncounted. Fused [gate|up] and split gate/up tables are both
    served; which kernel runs follows the switch module's own tables.
    """
    mode = getattr(switch_mlp, "routed_decode_mode", "off")
    if mode == "off" or do_sort or x.size != x.shape[-1]:
        return None
    if mode == "gate_up_down_shared":
        # The block folds the shared expert when it can; reaching the switch
        # means it declined (counted there), so run the routed half alone.
        mode = "gate_up_down"
    if mode in ("two_launch", "gate_up_down") and scores is None:
        mode = "gate_up"
    split = "gate_up_proj" not in switch_mlp
    if split:
        admission = _routed.admit_split_routed_decode(
            x, indices, scores, switch_mlp.gate_proj, switch_mlp.up_proj,
            switch_mlp.down_proj,
        )
    else:
        admission = _routed.admit_routed_decode(
            x, indices, scores, switch_mlp.gate_up_proj, switch_mlp.down_proj
        )
    if admission.accepted and switch_mlp.training:
        admission = _routed.RoutedDecodeAdmission(False, "training")
    if admission.accepted and not _routed.runtime_supported():
        admission = _routed.RoutedDecodeAdmission(False, "Metal runtime unavailable")
    if not admission.accepted:
        switch_mlp.routed_decode_fallbacks += 1
        switch_mlp.routed_decode_last_fallback = admission.reason
        return None
    if split:
        hidden = _routed.split_gate_up_swiglu(
            x, indices, switch_mlp.gate_proj, switch_mlp.up_proj
        )
    else:
        hidden = _routed.gate_up_swiglu(x, indices, switch_mlp.gate_up_proj)
    switch_mlp.routed_decode_calls += 1
    switch_mlp.routed_decode_last_fallback = None
    if mode == "gate_up_down":
        refusal = _served_down_refusal(
            switch_mlp, hidden.shape[-1], hidden.dtype, indices, scores, variant
        )
        if refusal is not None:
            switch_mlp.routed_down_fallbacks += 1
            switch_mlp.routed_down_last_fallback = refusal
            mode = "gate_up"
        else:
            y = _routed.served_down(hidden, indices, scores, switch_mlp.down_proj)
            switch_mlp.routed_down_calls += 1
            switch_mlp.routed_down_last_fallback = None
            return ("gate_up_down", y.reshape(indices.shape[:-1] + (y.shape[-1],)))
    if mode == "two_launch":
        y = _routed.down_combine(hidden, indices, scores, switch_mlp.down_proj)
        switch_mlp.routed_down_calls += 1
        return ("two_launch", y.reshape(indices.shape[:-1] + (y.shape[-1],)))
    return ("gate_up", hidden.reshape(indices.shape + (1, hidden.shape[-1])))


def _shared_fold_refusal(block, x, inds, scores):
    """Why the block may not fold its shared expert into the routed launches."""
    switch_mlp = block.switch_mlp
    if block.shared_folded or block.sharding_group is not None:
        return "shared row folded or sharded"
    if _COMPILE_GLUE:
        return "compiled combine glue selected"
    if not block.fused_expert_kernel_enabled:
        return "fused expert kernel off"
    if "gate_up_proj" in switch_mlp:
        return "fused [gate|up] table (fold is split-table only)"
    if not _routed.runtime_supported():
        return "Metal runtime unavailable"
    admission = _routed.admit_split_routed_decode(
        x, inds, scores, switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj
    )
    if not admission.accepted:
        return admission.reason
    if switch_mlp.training or block.training:
        return "training"
    inter = switch_mlp.gate_proj["weight"].shape[1]
    refusal = _served_down_refusal(
        switch_mlp, inter, x.dtype, inds, scores, block.fused_expert_kernel_mode
    )
    if refusal is not None:
        return refusal
    admission = _routed.admit_shared_fold(block.get("shared_expert"), x.shape[-1], inter)
    return None if admission.accepted else admission.reason


def _try_shared_fold(block, x, inds, scores):
    """The block's output from the two folded launches, or None."""
    if getattr(block.switch_mlp, "routed_decode_mode", "off") != "gate_up_down_shared":
        return None
    if x.size != x.shape[-1]:
        return None  # multi-token forwards decline quietly
    refusal = _shared_fold_refusal(block, x, inds, scores)
    if refusal is not None:
        block.shared_fold_fallbacks += 1
        block.shared_fold_last_fallback = refusal
        return None
    gate_logit = block.shared_expert_gate(x)
    if gate_logit.dtype != x.dtype or gate_logit.size != 1:
        block.shared_fold_fallbacks += 1
        block.shared_fold_last_fallback = "shared gate logit is not one value in the activation dtype"
        return None
    sw = block.switch_mlp
    y = _routed.shared_fold_decode(
        x, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj,
        block.shared_expert, gate_logit,
    )
    block.shared_fold_calls += 1
    block.shared_fold_last_fallback = None
    sw.routed_decode_calls += 1
    sw.routed_down_calls += 1
    return y.reshape(x.shape[:-1] + (y.shape[-1],))


class _ShapeOnly:
    """Shape/dtype stand-in for the routed hidden in a structural admission
    (``admit_qwen4_fused_down`` reads only ``shape`` and ``dtype``)."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.size = 1
        for extent in self.shape:
            self.size *= int(extent)


# --------------------------------------------------------------------------
# Routed MoE row windows and the router top-k launch/fold (omlx #4106 MoE
# half, #4041 verify windows, #4052 top-k fold; qwen4_moe_window). A window
# gives each of 2..17 rows the arithmetic of its own one-token block call:
# router gemv and shared gate through the row-exact qmv kernel, the stock
# routing per row (or the stock-identical routing kernel), routed experts in
# the one-token routed kernels with the token on grid z, and the shared
# expert folded per row (or through the row-exact qmv kernel).
# --------------------------------------------------------------------------
def _enable_moe_window(block, consumers, topk_mode) -> None:
    consumers = frozenset(consumers)
    unknown = consumers - set(_window.CONSUMERS)
    if unknown:
        raise ValueError(f"unknown MoE window consumers {sorted(unknown)}; expected {_window.CONSUMERS}")
    if topk_mode not in _window.TOPK_MODES:
        raise ValueError(f"unknown top-k mode {topk_mode!r}; expected {_window.TOPK_MODES}")
    block.moe_window_consumers = consumers
    block.moe_topk_mode = topk_mode
    block.moe_window_calls = {name: 0 for name in _window.CONSUMERS}
    block.moe_window_rows = 0
    block.moe_window_shared_calls = 0
    block.moe_window_fallbacks = {name: 0 for name in _window.CONSUMERS}
    block.moe_window_last_fallback = None
    block.moe_topk_calls = {"launch": 0, "fold": 0}
    block.moe_topk_fallbacks = 0
    block.moe_topk_last_fallback = None


def _moe_window_consumer(x):
    """Which consumer a multi-row MoE call belongs to, or None (prefill)."""
    from .. import row_exact_verify as _rev
    from .. import verify_scope as _verify_scope

    if _rev.current() is not None:
        return "row_exact"
    if _verify_scope.active():
        return "verify"
    if x.ndim == 3 and x.shape[1] == 1:
        return "batch_decode"
    return None


def _base_linear_ok(layer) -> bool:
    import mlx.nn as _nn

    base = getattr(type(layer), "_mlx2_row_exact_base", type(layer))
    return base is _nn.QuantizedLinear


def _moe_rows_refusal(block, x, rows):
    """Why the block's one-token reference cannot be reproduced row by row
    by the window kernels, or None. Structural; never evaluates."""
    if block.shared_folded or block.sharding_group is not None:
        return "shared row folded or sharded"
    if block.training or block.switch_mlp.training:
        return "training"
    if _COMPILE_GLUE:
        return "compiled combine glue selected"
    if _MOE_GATE_COMPILE:
        return "compiled router selected (one-token reference routes compiled)"
    if block.moe_router_mode == "fused":
        return "fused router kernel selected (one-token reference routes there)"
    if getattr(block.switch_mlp, "routed_decode_mode", "off") == "two_launch":
        return "two_launch selected (one-token reference is stock-down numerics)"
    if not block.fused_expert_kernel_enabled:
        return "fused expert kernel off (one-token reference down is stock)"
    switch_mlp = block.switch_mlp
    if "gate_up_proj" in switch_mlp:
        return "fused [gate|up] table (window is split-table only)"
    hidden = x.shape[-1]
    one_x = _ShapeOnly((hidden,), x.dtype)
    one_idx = _ShapeOnly((1, _window.TOP_K), mx.uint32)
    one_scores = _ShapeOnly((1, _window.TOP_K), x.dtype)
    admission = _routed.admit_split_routed_decode(
        one_x, one_idx, one_scores, switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj
    )
    if not admission.accepted:
        return admission.reason
    inter = switch_mlp.gate_proj["weight"].shape[1]
    refusal = _served_down_refusal(
        switch_mlp, inter, x.dtype, one_idx, one_scores, block.fused_expert_kernel_mode
    )
    if refusal is not None:
        return refusal
    for name in ("gate", "shared_expert_gate"):
        if not _base_linear_ok(block.get(name)):
            return f"{name} is not a QuantizedLinear"
    refusal = _window.admit_router_topk(
        _ShapeOnly((rows, block.num_experts), x.dtype), block.top_k, bool(block.norm_topk_prob)
    )
    if refusal is not None:
        return refusal
    shared = block.get("shared_expert")
    if shared is None or hasattr(shared, "_prefill_counts"):
        return "shared expert missing or tensorfold prefill MLP installed"
    return None


def _shared_fold_ok(block, x) -> bool:
    inter = block.switch_mlp.gate_proj["weight"].shape[1]
    return _routed.admit_shared_fold(block.get("shared_expert"), x.shape[-1], inter).accepted


def _stock_routing(block, gates):
    """The block's own composed routing (the one-token reference)."""
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def _row_projections(window, linears, x):
    """Projections of ``x`` with one-row arithmetic per row (row-exact qmv)."""
    from . import row_exact_qmv as _req

    if len(linears) == 1:
        (y, route) = _req.quantized_linear(linears[0], x)
        outputs = (y,)
    else:
        (outputs, route) = _req.quantized_linears(linears, x)
    if window is not None:
        window.note("projections", route, 1 if route == "group_kernel" else len(linears))
    return outputs


def _run_moe_window(block, x, rows, consumer):
    """The block's output for every row of ``x`` (admitted), each row with
    its one-token block arithmetic."""
    from .. import row_exact_verify as _rev

    window = _rev.current()
    hidden = x.shape[-1]
    flat = x.reshape(rows, hidden)
    (gates, gate_logit) = _row_projections(window, (block.gate, block.shared_expert_gate), flat)
    sw = block.switch_mlp
    mode = block.moe_topk_mode
    if mode == "fold" and rows > _window.topk_fold_max_rows():
        mode = "launch"
    if mode == "launch":
        (inds, scores) = _window.router_topk(gates)
    elif mode == "off":
        (inds, scores) = _stock_routing(block, gates)
    else:
        inds = scores = None
    logits = gates if mode == "fold" else None
    shared = _window.window_shared_enabled() and _shared_fold_ok(block, x)
    if shared:
        y = _window.shared_rows(
            flat, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj,
            block.shared_expert, gate_logit, logits=logits,
        )
        block.moe_window_shared_calls += 1
    else:
        y = _window.routed_rows(
            flat, inds, scores, sw.gate_proj, sw.up_proj, sw.down_proj, logits=logits
        )
        shared_mlp = block.shared_expert
        (g, u) = _row_projections(window, (shared_mlp.gate_proj, shared_mlp.up_proj), flat)
        (shared_y,) = _row_projections(window, (shared_mlp.down_proj,), swiglu(g, u))
    if mode == "fold":
        y = y[0]
    if not shared:
        y = y + gate_sigmoid(gate_logit) * shared_y
    if mode != "off":
        block.moe_topk_calls[mode] += 1
    block.moe_window_calls[consumer] += 1
    block.moe_window_rows += rows
    if window is not None:
        window.note("moe_window", "shared_fold" if shared else "routed", rows)
        window.note("moe_routing", mode, rows)
    return y.reshape(*x.shape[:-1], hidden)


def _try_moe_window(block, x):
    """A multi-row window's block output, or None for the composed body."""
    if not block.moe_window_consumers:
        return None
    hidden = x.shape[-1]
    rows = x.size // hidden
    if rows <= 1:
        return None
    consumer = _moe_window_consumer(x)
    if consumer is None or consumer not in block.moe_window_consumers:
        return None
    if consumer == "batch_decode" and rows > _window.batch_decode_max_rows():
        return None  # a measured loss at wide batches: not a candidate
    if rows > _window.WINDOW_MAX_ROWS:
        refusal = f"{rows} rows > {_window.WINDOW_MAX_ROWS}"
    else:
        refusal = _moe_rows_refusal(block, x, rows)
    if refusal is not None:
        block.moe_window_fallbacks[consumer] += 1
        block.moe_window_last_fallback = refusal
        return None
    block.moe_window_last_fallback = None
    return _run_moe_window(block, x, rows, consumer)


def _topk_refusal(block, x, gates_shape):
    if block.shared_folded or block.sharding_group is not None:
        return "shared row folded or sharded"
    if block.moe_router_mode == "fused":
        return "fused router kernel selected"
    if _MOE_GATE_COMPILE:
        return "compiled router selected"
    return _window.admit_router_topk(
        _ShapeOnly(gates_shape, x.dtype), block.top_k, bool(block.norm_topk_prob)
    )


def _try_topk_fold_decode(block, x):
    """One-token decode with the routing folded into the gate+up launch, or
    None. Needs the routed kernels on both halves (gate_up_down or
    gate_up_down_shared), where the fold replaces the stock routing."""
    if block.moe_topk_mode != "fold" or x.size != x.shape[-1]:
        return None
    sw = block.switch_mlp
    routed_mode = getattr(sw, "routed_decode_mode", "off")
    refusal = None
    if routed_mode not in ("gate_up_down", "gate_up_down_shared"):
        refusal = f"routed decode {routed_mode!r} (fold needs gate_up_down or gate_up_down_shared)"
    if refusal is None:
        refusal = _topk_refusal(block, x, x.shape[:-1] + (block.num_experts,))
    if refusal is None:
        refusal = _moe_rows_refusal(block, x, 1)
    if refusal is not None:
        block.moe_topk_fallbacks += 1
        block.moe_topk_last_fallback = refusal
        return None
    gates = block.gate(x)
    hidden = x.shape[-1]
    flat = x.reshape(1, hidden)
    shared = routed_mode == "gate_up_down_shared" and _shared_fold_ok(block, x)
    if shared:
        gate_logit = block.shared_expert_gate(x)
        (y, _, _) = _window.shared_rows(
            flat, None, None, sw.gate_proj, sw.up_proj, sw.down_proj,
            block.shared_expert, gate_logit, logits=gates,
        )
        block.shared_fold_calls += 1
    else:
        (y, _, _) = _window.routed_rows(
            flat, None, None, sw.gate_proj, sw.up_proj, sw.down_proj, logits=gates
        )
        y = y.reshape(x.shape)
        shared_y = block.shared_expert(x)
        y = y + gate_sigmoid(block.shared_expert_gate(x)) * shared_y
    sw.routed_decode_calls += 1
    sw.routed_down_calls += 1
    block.moe_topk_calls["fold"] += 1
    block.moe_topk_last_fallback = None
    return y.reshape(x.shape)


def _try_topk_launch(block, x, gates):
    """``(inds, scores)`` from the one-launch stock-identical routing for a
    one-token call (top-k mode ``launch``), or None."""
    if block.moe_topk_mode != "launch" or x.size != x.shape[-1]:
        return None
    refusal = _topk_refusal(block, x, gates.shape)
    if refusal is None and gates.dtype != x.dtype:
        refusal = "router logits dtype differs from the activation"
    if refusal is not None:
        block.moe_topk_fallbacks += 1
        block.moe_topk_last_fallback = refusal
        return None
    (inds, scores) = _window.router_topk(gates)
    block.moe_topk_calls["launch"] += 1
    block.moe_topk_last_fallback = None
    lead = gates.shape[:-1]
    return inds.reshape(*lead, _window.TOP_K), scores.reshape(*lead, _window.TOP_K)


def _routed_tail(switch_mlp, x, indices, inv_order, scores, do_sort):
    """Unsort and weight the down-projection rows (stock), or fuse both.

    With ``moe_weighted_sum`` selected on the switch module, an admitted
    sorted prefill tail runs ``qwen4_moe_weighted_sum.moe_weighted_sum`` on
    the sorted rows instead of ``_scatter_unsort`` + ``(x * scores).sum``.
    """
    # Decode and short verify widths (unsorted, or under the row floor)
    # decline quietly: they are not candidates, so they are not fallbacks.
    if (
        getattr(switch_mlp, "moe_weighted_sum", False)
        and scores is not None
        and do_sort
        and indices.size >= _moe_wsum.MIN_ROUTED_ROWS
    ):
        admission = _moe_wsum.admit_moe_weighted_sum(
            x_sorted=x,
            inv_order=inv_order,
            scores=scores,
            indices=indices,
            do_sort=do_sort,
            training=bool(switch_mlp.training),
        )
        if admission.accepted and not _moe_wsum.runtime_supported():
            admission = _moe_wsum.WeightedSumAdmission(False, "Metal runtime unavailable")
        if admission.accepted:
            switch_mlp.moe_weighted_sum_calls += 1
            switch_mlp.moe_weighted_sum_last_fallback = None
            return _moe_wsum.moe_weighted_sum(x, inv_order, scores)
        switch_mlp.moe_weighted_sum_fallbacks += 1
        switch_mlp.moe_weighted_sum_last_fallback = admission.reason
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    x = x.squeeze(-2)
    if scores is not None:
        return (x * scores[..., None]).sum(axis=-2)
    return x


def _enable_moe_weighted_sum(switch_mlp, enabled: bool) -> None:
    """Attach the weighted-sum selection and its counters to a switch module."""
    switch_mlp.moe_weighted_sum = bool(enabled)
    switch_mlp.moe_weighted_sum_calls = 0
    switch_mlp.moe_weighted_sum_fallbacks = 0
    switch_mlp.moe_weighted_sum_last_fallback = None


def _fused_outcome(variant: str, indices) -> str:
    """Name the variant a successful fused dispatch ran, for the receipts."""
    if variant != "auto":
        return variant
    from .qwen4_fused_moe import auto_variant

    return auto_variant(indices.size // indices.shape[-1])


def _try_qwen4_fused_down(
    hidden, indices, scores, down_proj, sorted_indices, variant="scalar"
):
    """Return the fused routed result, or None before dispatch when ineligible.

    A streamed expert table (``StreamedQuantizedSwitchLinear``) is a
    ``QuantizedSwitchLinear`` with no resident ``weight``/``scales``: its rows
    are gathered per step from disk, so the fused kernel has nothing to read
    and the stock path must run.
    """
    if (
        scores is None
        or variant == "stock"
        or sorted_indices
        or down_proj.training
        or (not isinstance(down_proj, QuantizedSwitchLinear))
        or ("bias" in down_proj)
        or ("weight" not in down_proj)
        or ("scales" not in down_proj)
    ):
        return None
    from .qwen4_fused_moe import admit_qwen4_fused_down, auto_variant, qwen4_fused_down

    if hidden.ndim < 3 or hidden.shape[-2] != 1:
        return None
    compact_hidden = hidden.squeeze(-2)
    biases = getattr(down_proj, "biases", None)
    admission = admit_qwen4_fused_down(
        compact_hidden,
        indices,
        scores,
        down_proj["weight"],
        down_proj["scales"],
        biases,
        num_experts=down_proj.num_experts,
        group_size=down_proj.group_size,
        bits=down_proj.bits,
        mode=down_proj.mode,
    )
    if not admission.accepted:
        return None
    if variant == "auto":
        variant = auto_variant(admission.tokens)
    return qwen4_fused_down(
        compact_hidden,
        indices,
        scores,
        down_proj["weight"],
        down_proj["scales"],
        biases,
        num_experts=down_proj.num_experts,
        group_size=down_proj.group_size,
        bits=down_proj.bits,
        mode=down_proj.mode,
        variant=variant,
    )


def _parts(weights: dict, path: str) -> dict:
    """The weight/scales/biases a module path contributes, if present."""
    found = {}
    for suffix in ("weight", "scales", "biases"):
        key = f"{path}.{suffix}"
        if key in weights:
            found[suffix] = weights[key]
    return found


def _drop(weights: dict, path: str):
    for suffix in ("weight", "scales", "biases"):
        weights.pop(f"{path}.{suffix}", None)


def _store(weights: dict, path: str, parts: dict):
    for suffix, value in parts.items():
        weights[f"{path}.{suffix}"] = value


def _concat_parts(parts_list, axis: int):
    """Concatenate matching part sets, or None when they do not match.

    Affine groups run along each row's K axis, so concatenating along the
    output-row axis or the expert axis keeps every packed value, scale and
    bias byte for byte.
    """
    if not all(parts_list):
        return None
    suffixes = set(parts_list[0])
    if any((set(parts) != suffixes for parts in parts_list)):
        return None
    for suffix in suffixes:
        shapes = [parts[suffix].shape for parts in parts_list]
        if len({len(shape) for shape in shapes}) > 1:
            return None
        at = axis % len(shapes[0])
        if len({shape[:at] + shape[at + 1 :] for shape in shapes}) > 1:
            return None
    return {
        suffix: mx.concatenate([parts[suffix] for parts in parts_list], axis=axis)
        for suffix in suffixes
    }


def transform_moe_weights(
    weights: dict, prefixes, *, fuse_gate_up: bool, fold_shared: bool
) -> int:
    """Apply the load-time MoE lever transforms in place.

    ``fuse_gate_up`` keeps (or restores) the shipped [gate|up] tensor.
    ``fold_shared`` appends the shared expert as routed expert index E and
    drops the separate shared tensors. Both consume file-backed checkpoint
    arrays and leave ONE resident tensor per projection, so neither adds a
    second full-size copy. Returns the number of layers transformed.
    """
    if not (fuse_gate_up or fold_shared):
        return 0
    changed = 0
    for prefix in prefixes:
        switch = f"{prefix}.switch_mlp"
        shared = f"{prefix}.shared_expert"
        names = ["gate_up_proj"] if fuse_gate_up else ["gate_proj", "up_proj"]
        routed = {}
        if fuse_gate_up:
            fused = _parts(weights, f"{switch}.gate_up_proj") or _concat_parts(
                [
                    _parts(weights, f"{switch}.gate_proj"),
                    _parts(weights, f"{switch}.up_proj"),
                ],
                axis=-2,
            )
            if fused is None:
                continue
            routed["gate_up_proj"] = fused
        else:
            for name in names:
                routed[name] = _parts(weights, f"{switch}.{name}")
        routed["down_proj"] = _parts(weights, f"{switch}.down_proj")
        if not all(routed.values()):
            continue
        if fold_shared:
            shared_parts = {"down_proj": _parts(weights, f"{shared}.down_proj")}
            if fuse_gate_up:
                shared_parts["gate_up_proj"] = _parts(
                    weights, f"{shared}.gate_up_proj"
                ) or _concat_parts(
                    [
                        _parts(weights, f"{shared}.gate_proj"),
                        _parts(weights, f"{shared}.up_proj"),
                    ],
                    axis=-2,
                )
            else:
                for name in names:
                    shared_parts[name] = _parts(weights, f"{shared}.{name}")
            folded = {}
            for name, parts in routed.items():
                one = shared_parts.get(name)
                if not one:
                    folded = None
                    break
                merged = _concat_parts(
                    [parts, {s: v[None] for (s, v) in one.items()}], axis=0
                )
                if merged is None:
                    folded = None
                    break
                folded[name] = merged
            if folded is None:
                continue
            routed = folded
            for name in ("gate_proj", "up_proj", "gate_up_proj", "down_proj"):
                _drop(weights, f"{shared}.{name}")
        for name in ("gate_proj", "up_proj", "gate_up_proj", "down_proj"):
            _drop(weights, f"{switch}.{name}")
        for name, parts in routed.items():
            _store(weights, f"{switch}.{name}", parts)
        changed += 1
    return changed


@mx.compile
def _select_experts(gates: mx.array, top_k: int, norm_topk_prob: bool):
    gates = mx.softmax(gates, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return (inds, scores)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    shared_expert_intermediate_size: int
    mlp_only_layers: List[int]
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    partial_rotary_factor: float
    max_position_embeddings: int
    head_dim: int
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    full_attention_interval: int = 4
    mtp_num_hidden_layers: int = 0


@partial(mx.compile, shapeless=True)
def _precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)


class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)


class Qwen3NextMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        if hasattr(self, "_prefill_counts"):
            from .tensorfold_prefill import fused_mlp

            fused = fused_mlp(self, x)
            if fused is not None:
                return fused
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        shared_expert_intermediate_size = args.shared_expert_intermediate_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.fused_gate_up = _MOE_FUSED_GATE_UP
        self.shared_folded = (
            _MOE_SHARED_IN_GATHER
            and shared_expert_intermediate_size == intermediate_size
        )
        self.fused_expert_kernel_mode = (
            _MOE_FUSED_EXPERT_MODE if not self.shared_folded else "stock"
        )
        self.moe_router_mode = "fused" if _MOE_ROUTER_KERNEL else "stock"
        self.moe_router_calls = 0
        self.moe_router_fallbacks = 0
        self.moe_router_last_fallback = None
        if self.fused_gate_up:
            switch_cls = FusedGateUpSwitchGLU
        else:
            switch_cls = FusedDownSwitchGLU
        self.switch_mlp = switch_cls(
            dim, intermediate_size, num_experts + (1 if self.shared_folded else 0)
        )
        # The weighted sum needs scores inside the switch module; a folded
        # shared row weights its rows outside it, so it keeps the stock tail.
        self.moe_weighted_sum = _MOE_WEIGHTED_SUM and not self.shared_folded
        _enable_moe_weighted_sum(self.switch_mlp, self.moe_weighted_sum)
        # The routed kernels read the fused [gate|up] table (#3912) or the
        # split gate/up tables (#4113 two-table gate+up); a folded shared row
        # keeps the composed body.
        _enable_routed_decode(
            self.switch_mlp,
            _MOE_ROUTED_DECODE if not self.shared_folded else "off",
        )
        if not self.shared_folded:
            self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        self.shared_fold_calls = 0
        self.shared_fold_fallbacks = 0
        self.shared_fold_last_fallback = None
        self.fused_expert_dispatches = {"scalar": 0, "tile4": 0}
        self.fused_expert_fallbacks = 0
        self.sharding_group = None
        _enable_moe_window(
            self,
            _MOE_WINDOW_CONSUMERS if not self.shared_folded else (),
            _MOE_TOPK_MODE if not self.shared_folded else "off",
        )

    @property
    def fused_expert_kernel_enabled(self):
        return self.fused_expert_kernel_mode != "stock"

    def set_fused_expert_kernel_mode(self, mode: str):
        """Select stock/scalar/tile4 for subsequent forwards without reload."""
        if mode not in _MOE_FUSED_EXPERT_MODES:
            raise ValueError(
                f"unknown fused expert mode {mode!r}; expected one of {_MOE_FUSED_EXPERT_MODES}"
            )
        if mode != "stock" and self.shared_folded:
            raise ValueError("fused expert kernels do not support a folded shared row")
        self.fused_expert_kernel_mode = mode

    def set_moe_weighted_sum(self, enabled: bool) -> bool:
        """Live-toggle the sorted-order weighted-sum tail (omlx #3903 port)."""
        if enabled and self.shared_folded:
            raise ValueError("the weighted-sum tail does not support a folded shared row")
        self.moe_weighted_sum = bool(enabled)
        self.switch_mlp.moe_weighted_sum = bool(enabled)
        return self.moe_weighted_sum

    def set_moe_routed_decode_mode(self, mode: str) -> str:
        """Live-select the one-token routed-expert kernels (omlx #3912/#4113)."""
        if mode not in _routed.MODES:
            raise ValueError(
                f"unknown routed decode mode {mode!r}; expected {_routed.MODES}"
            )
        if mode != "off" and self.shared_folded:
            raise ValueError("routed decode does not support a folded shared row")
        self.switch_mlp.routed_decode_mode = mode
        return mode

    def set_moe_window_consumers(self, consumers) -> frozenset:
        """Live-select who may run a routed row window (omlx #4106/#4041)."""
        consumers = frozenset(consumers)
        if consumers and self.shared_folded:
            raise ValueError("MoE row windows do not support a folded shared row")
        unknown = consumers - set(_window.CONSUMERS)
        if unknown:
            raise ValueError(f"unknown MoE window consumers {sorted(unknown)}; expected {_window.CONSUMERS}")
        self.moe_window_consumers = consumers
        return consumers

    def set_moe_topk_mode(self, mode: str) -> str:
        """Live-select the routing: stock ops, one launch, or folded (#4052)."""
        if mode not in _window.TOPK_MODES:
            raise ValueError(f"unknown top-k mode {mode!r}; expected {_window.TOPK_MODES}")
        if mode != "off" and self.shared_folded:
            raise ValueError("the top-k launch does not support a folded shared row")
        self.moe_topk_mode = mode
        return mode

    def set_moe_router_mode(self, mode: str):
        if mode not in _MOE_ROUTER_MODES:
            raise ValueError(
                f"unknown MoE router mode {mode!r}; expected {_MOE_ROUTER_MODES}"
            )
        self.moe_router_mode = mode

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        windowed = _try_moe_window(self, x)
        if windowed is not None:
            return windowed
        folded = _try_topk_fold_decode(self, x)
        if folded is not None:
            return folded
        gates = self.gate(x)
        router_fused = False
        launched = _try_topk_launch(self, x, gates)
        if launched is not None:
            (inds, scores) = launched
            router_fused = True
        elif self.moe_router_mode == "fused":
            admission = admit_qwen4_moe_router(
                gates, top_k=self.top_k, norm_topk_prob=bool(self.norm_topk_prob)
            )
            if admission.accepted and probe_qwen4_moe_router(gates.dtype):
                (inds, scores) = qwen4_moe_router(gates)
                self.moe_router_calls += 1
                self.moe_router_last_fallback = None
                router_fused = True
            else:
                self.moe_router_fallbacks += 1
                self.moe_router_last_fallback = admission.reason
        if not router_fused and (
            _MOE_GATE_COMPILE
            and gates.size // gates.shape[-1] <= _MOE_GATE_COMPILE_MAX_TOKENS
        ):
            (inds, scores) = _select_experts(
                gates, self.top_k, bool(self.norm_topk_prob)
            )
        elif not router_fused:
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
        folded = _try_shared_fold(self, x, inds, scores)
        if folded is not None:
            return folded
        glue = _COMPILE_GLUE
        if self.shared_folded:
            shared_col = mx.full(
                inds.shape[:-1] + (1,), self.num_experts, dtype=inds.dtype
            )
            rows = self.switch_mlp(x, mx.concatenate([inds, shared_col], axis=-1))
            y = (rows[..., : self.top_k, :] * scores[..., None]).sum(axis=-2)
            shared_y = rows[..., self.top_k, :]
        else:
            if self.fused_expert_kernel_enabled and self.sharding_group is None:
                y = self.switch_mlp(
                    x, inds, scores=scores, variant=self.fused_expert_kernel_mode
                )
                outcome = getattr(self.switch_mlp, "_last_fused_variant", None)
                if outcome in self.fused_expert_dispatches:
                    self.fused_expert_dispatches[outcome] += 1
                elif outcome in ("routed_two_launch", "routed_gate_up_down"):
                    pass  # counted by switch_mlp.routed_down_calls
                else:
                    self.fused_expert_fallbacks += 1
            elif (
                self.moe_weighted_sum
                and self.sharding_group is None
                and inds.size >= _moe_wsum.MIN_ROUTED_ROWS
            ):
                # Prefill widths only: hand the scores to the switch module so
                # its sorted tail can weight rows in place.  ``variant="stock"``
                # keeps the fused-down kernel out of this route.
                y = self.switch_mlp(x, inds, scores=scores, variant="stock")
            else:
                y = self.switch_mlp(x, inds)
                y = (y * scores[..., None]).sum(axis=-2)
            shared_y = self.shared_expert(x)
        gate = gate_sigmoid(self.shared_expert_gate(x))
        combined = None
        if glue:
            combined = _run_glue(
                ("moe_combine",), _build_moe_combine, y, gate, shared_y
            )
        y = y + gate * shared_y if combined is None else combined
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
