# SPDX-License-Identifier: Apache-2.0
"""Capability-owned row-exact building blocks for model adapters.

These mixins preserve the arithmetic of independent one-token projection and
expert calls while allowing an adapter to keep the surrounding verify window
batched.  They do not establish a complete target protocol: attention, state
updates, masks, norms, and logits remain the installing adapter's
responsibility.

Factored from :mod:`mlx2.runtime.models.qwen4_row_exact`; the qwen4 provenance
record continues to cover the original implementation.  This module is an
internal refactor plus the generic class-swap seam used by sibling adapters.

The multi-row forms come from the shared package (:mod:`mlx2.runtime.row_exact`):
each swapped leaf flattens its rows into one ``[1, R, K]`` strip and runs the
package's canary-gated op (``ops.quantized_linear``, ``ops.quantized_linears``,
``ops.dense_linear``, ``ops.per_row``), so a form the installed MLX build does
not keep row-invariant falls back to one one-row call per row.  Routed experts
stay on their own per-row loop (``RowExactSwitchMixin``; see its docstring).
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .. import row_exact_verify as REV
from ..row_exact import ops
from . import row_exact_qmv as REQ

STAGE = "projections"


def row_count(value) -> int:
    """Number of projection rows in ``value`` (all leading dimensions)."""

    return int(value.size // value.shape[-1]) if value.ndim else 1


def _is_strip(x) -> bool:
    return x.ndim == 3 and x.shape[0] == 1


def _strip(x, rows: int):
    """``x`` ([..., K]) as the package's one ``[1, R, K]`` strip."""

    return x if _is_strip(x) else x.reshape(1, rows, x.shape[-1])


def _unstrip(y, x):
    """A strip result ``y`` ([1, R, N]) in the leading shape of ``x``."""

    return y if _is_strip(x) else y.reshape(*x.shape[:-1], y.shape[-1])


def grouped_quantized_linears(window, linears, inputs):
    """Run mixed projection siblings with their row-exact laws.

    GDN input projections can mix quantized q/k/v/z tables with small dense
    b/a tables.  Keep the compatible quantized members in one grouped launch
    and route dense members independently; sending the mixed tuple to
    ``ops.quantized_linears`` makes one dense member demote every sibling to
    the per-row fallback.

    ``window`` is the open verify window the routes are noted on (the package
    notes on ``row_exact_verify.current()``, which must be it).
    """

    compacted = REV.compact_rows(inputs, stage=STAGE)
    if compacted is not None:
        inputs = compacted.values
    rows = row_count(inputs)
    strip = _strip(inputs, rows)
    outputs = [None] * len(linears)
    quantized = [
        (index, module)
        for index, module in enumerate(linears)
        if isinstance(module, nn.QuantizedLinear)
    ]
    if quantized:
        grouped = ops.quantized_linears(
            tuple(module for _, module in quantized), strip, stage=STAGE
        )
        for (index, _), value in zip(quantized, grouped, strict=True):
            outputs[index] = value
    for index, module in enumerate(linears):
        if outputs[index] is not None:
            continue
        if isinstance(module, nn.Linear):
            outputs[index] = ops.dense_linear(module, strip, stage=STAGE)
        else:
            # An undeclared projection stays exact by construction.
            window.note(STAGE, "per_row_unsupported")
            outputs[index] = ops.per_row(module, strip)
    outputs = tuple(_unstrip(y, inputs) for y in outputs)
    return tuple(compacted.restore(y) for y in outputs) if compacted is not None else outputs


class RowExactQuantizedLinearMixin:
    """Route multi-row quantized projections through their one-row law."""
    _mlx2_compacts_valid_rows = True

    def __call__(self, x):
        compacted = REV.compact_rows(x, stage=STAGE)
        if compacted is not None:
            return compacted.restore(self(compacted.values))
        window = REV.current()
        lane = REQ.lane_preparation(self)
        if lane is not None and window is None:
            # Outside a verify window the lane law is simply this layer's law.
            from ..lane import LaneUnsupported

            try:
                return REQ.lane_rows(lane, x)
            except LaneUnsupported:
                pass
        # Inside a window every multi-row form, the lane law included, runs
        # through its canary (the one-row reference is this layer's own call).
        rows = row_count(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        return _unstrip(ops.quantized_linear(self, _strip(x, rows), stage=STAGE), x)


class RowExactLinearMixin:
    """Give every dense projection row its one-row arithmetic.

    The package's broadcast matvec (route ``matvec_rows``) where its canary
    holds on this build; otherwise, and for a biased layer (``nn.Linear``
    fuses the bias through ``addmm``), one one-row call per row.
    """
    _mlx2_compacts_valid_rows = True

    def __call__(self, x):
        compacted = REV.compact_rows(x, stage=STAGE)
        if compacted is not None:
            return compacted.restore(self(compacted.values))
        window = REV.current()
        rows = row_count(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        return _unstrip(ops.dense_linear(self, _strip(x, rows), stage=STAGE), x)


class RowExactMoEMixin:
    """Keep routing batched while routed experts retain one-row arithmetic."""
    _mlx2_compacts_valid_rows = True

    def __call__(self, x):
        compacted = REV.compact_rows(x, stage="moe")
        if compacted is not None:
            return compacted.restore(self(compacted.values))
        window = REV.current()
        rows = row_count(x)
        if window is None or rows <= 1:
            return super().__call__(x)
        if getattr(self, "moe_router_mode", "stock") != "fused":
            from . import qwen3_next as Q3N

            if Q3N._MOE_GATE_COMPILE and rows > Q3N._MOE_GATE_COMPILE_MAX_TOKENS:
                window.fail("moe_router_compiled_one_token_eager_window")
            window.note("moe_router", "batched", rows)
            return super().__call__(x)
        # The fused router kernel is not row-invariant: the whole block runs
        # once per row (the package's reference form).
        out = ops.per_row(super().__call__, _strip(x, rows))
        window.note("moe", "per_row_block", rows)
        return _unstrip(out, x)


class RowExactSwitchMixin:
    """Run routed expert inputs one row at a time inside an exact window.

    Not ``row_exact.ops.routed_experts``: the Qwen switch modules pick their
    kernels by assignment count (the one-token routed-decode and fused-down
    kernels apply the router scores inside the switch call, the sorted gather
    starts at ``switch_layers_sort_min`` assignments), and the package's
    canary probes ``experts(x, indices)`` without scores at 2..8 rows, so it
    would vouch for a different code path than a verify window's rows take.
    """

    def __call__(self, x, indices, scores=None, variant="scalar"):
        window = REV.current()
        rows = row_count(indices)
        if window is None or rows <= 1:
            return super().__call__(x, indices, scores=scores, variant=variant)
        lead = indices.shape[:-1]
        xs = x.reshape(rows, 1, 1, x.shape[-1])
        ids = indices.reshape(rows, 1, 1, indices.shape[-1])
        weights = (
            None if scores is None else scores.reshape(rows, 1, 1, scores.shape[-1])
        )
        call = super().__call__
        out = mx.concatenate(
            [
                call(
                    xs[row],
                    ids[row],
                    scores=None if weights is None else weights[row],
                    variant=variant,
                )
                for row in range(rows)
            ],
            axis=0,
        )
        window.note("moe_experts", "per_row", rows)
        return out.reshape(*lead, *out.shape[2:])


class RowExactGatedDeltaNetProjectionsMixin:
    """Use the grouped one-row law for the four GDN input projections."""

    def _input_projections(self, inputs):
        window = REV.current()
        if window is None or row_count(inputs) <= 1:
            return super()._input_projections(inputs)
        return grouped_quantized_linears(
            window,
            (self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a),
            inputs,
        )


_SUBCLASSES: dict[tuple[type, type], type] = {}


def row_exact_subclass(mixin, cls):
    """Return a cached dynamic subclass carrying row-exact base markers."""

    key = (mixin, cls)
    subclass = _SUBCLASSES.get(key)
    if subclass is None:
        subclass = type(
            "RowExact" + cls.__name__,
            (mixin, cls),
            {"_row_exact_base": cls, "_mlx2_row_exact_base": cls},
        )
        _SUBCLASSES[key] = subclass
    return subclass
