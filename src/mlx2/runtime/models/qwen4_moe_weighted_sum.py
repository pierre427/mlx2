# SPDX-License-Identifier: Apache-2.0
# Structure mined from jundot/omlx 3e2bdb1f (Apache-2.0); re-expressed as a
# JIT kernel.  Reduction order from ddalcu/mlx-serve a453794f (#653, MIT).
# See docs/PROVENANCE.md, provenance/omlx-3903-qwen4-prefill.json and
# provenance/mlx-serve-653-moe-prefill-reduce.json.
"""Sorted-order MoE weighted sum for prefill widths.

For a sorted gather (``do_sort``), the stock ``SwitchGLU`` tail is::

    y = down_proj(hidden, idx, sorted_indices=True)     # (N[+pad], 1, D), sorted
    y = _scatter_unsort(y, inv_order, indices.shape)    # (B, S, K, 1, D)
    y = (y.squeeze(-2) * scores[..., None]).sum(-2)     # (B, S, D)

``moe_weighted_sum`` reads the sorted rows directly through ``inv_order`` and
writes the per-token sum, so neither the unsorted ``(B, S, K, D)`` copy nor
the product intermediate is materialized.  One threadgroup (256 threads) per
token stages that token's K row ids and scores in threadgroup memory, then
strides the hidden axis.  The reduction over K replays MLX's own reduction
of the eager graph (below), so the output is bit-identical to it.

The kernel is a JIT re-expression of omlx's native ``moe_weighted_sum_tiled``
(``omlx/custom_kernels/common/csrc/kernels/quantized_moe.h``, dispatched by
``qwen35_prefill.cpp``; the Python routing is
``omlx/patches/qwen35_moe_weighted_sum.py``).  No omlx C++/Metal text is copied.

Semantics vs. the eager tail (bit-identical)
--------------------------------------------
* Output dtype is ``promote(x.dtype, scores.dtype)``, what the eager product
  produces: bf16 scores -> bf16 output, fp32 scores -> fp32 output.  The
  product and the sum both run in that type ``O``.
* The product is ``O(x) * O(score)``: MLX's ``Multiply`` on two ``O`` values,
  one rounding to ``O`` per term (exact for a bf16 x bf16 product held in
  fp32).
* The eager ``(B, S, K, D).sum(axis=-2)`` is a strided (column) reduction of
  size K and stride D with no other reduced axes.  For K < 32 MLX dispatches
  ``col_reduce_small`` (``strided_reduce_small`` in
  ``mlx/backend/metal/reduce.cpp``) with ``threadgroup_y = min(8, K)``; each
  y-thread folds rows ``j, j + 8, ...`` into its own ``Sum<O>`` partial, which
  starts at ``O(0)``, and thread 0 then folds partials ``1..P-1`` into partial
  0 in order.  Every ``+`` rounds to ``O`` (``Sum<bfloat16_t>`` accumulates
  in bf16).  This kernel replays that order with ``P = min(8, K)``.  It does
  not depend on the token count, so one order covers every prefill width.
  Checked against the installed MLX 0.32.2.dev20260919+39400a0d4 headers
  (``kernels/reduction/reduce_col.h``, ``reduction/ops.h``) and verified bit
  for bit on Metal by ``tests/test_qwen4_moe_weighted_sum_metal.py``.
* The order and the bf16-rounded accumulation come from ddalcu/mlx-serve#653
  (MIT).  omlx's kernel (and this module before 2026-10-01) accumulated in
  fp32 in slot order and rounded once; on Metal that differs in raw bits
  from the eager graph on ~57% of top-10 outputs (a one-partial bf16 sum on
  ~61%), so neither is a substitute for the eager tail.
* At K=8 the eight one-row partials fold in slot order, so the order is a
  plain sequential bf16 sum; at K=10 partials 0 and 1 also hold slots 8, 9.
* ``_gather_sort`` pads the sorted rows past n > 32768 (tail bug
  workaround); ``inv_order`` never points at a pad row, so the kernel reads
  only the real rows -- no special case is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

WEIGHTED_SUM_ENV = "MLX_QWEN4_MOE_WEIGHTED_SUM"
# Flash-Next routes 10 experts, Qwen3.6-35B routes 8.  Both are bit-exact
# against the eager tail on Metal (MLX picks col_reduce_small for any K < 32,
# but only these two are gated).
SUPPORTED_TOP_K = (8, 10)
# ``col_reduce_small`` launches at most 8 y-threads, hence at most 8 partials.
COL_REDUCE_SMALL_PARTIALS = 8
MIN_ROUTED_ROWS = 64
THREADS = 256


def weighted_sum_enabled_from_env() -> bool:
    """``MLX_QWEN4_MOE_WEIGHTED_SUM``; unset or ``0`` means off."""
    raw = os.environ.get(WEIGHTED_SUM_ENV, "0")
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def runtime_supported() -> bool:
    return bool(
        hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
    )


@dataclass(frozen=True)
class WeightedSumAdmission:
    accepted: bool
    reason: str


def admit_moe_weighted_sum(
    *,
    x_sorted: Any,
    inv_order: Any,
    scores: Any,
    indices: Any,
    do_sort: bool,
    training: bool,
) -> WeightedSumAdmission:
    """Structural admission; never evaluates an array."""
    if training:
        return WeightedSumAdmission(False, "training")
    if not do_sort or inv_order is None:
        return WeightedSumAdmission(False, "unsorted gather")
    if scores is None:
        return WeightedSumAdmission(False, "no scores")
    top_k = int(indices.shape[-1])
    if top_k not in SUPPORTED_TOP_K:
        return WeightedSumAdmission(False, f"top_k {top_k}")
    routed = int(indices.size)
    if routed < MIN_ROUTED_ROWS:
        return WeightedSumAdmission(False, f"routed rows {routed} < {MIN_ROUTED_ROWS}")
    if tuple(scores.shape) != tuple(indices.shape):
        return WeightedSumAdmission(False, f"scores shape {tuple(scores.shape)}")
    if int(inv_order.size) != routed:
        return WeightedSumAdmission(False, "inv_order size")
    if x_sorted.ndim != 3 or x_sorted.shape[-2] != 1 or x_sorted.shape[0] < routed:
        return WeightedSumAdmission(False, f"x_sorted shape {tuple(x_sorted.shape)}")
    if x_sorted.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return WeightedSumAdmission(False, f"x dtype {x_sorted.dtype}")
    if scores.dtype not in (x_sorted.dtype, mx.float32):
        return WeightedSumAdmission(False, f"scores dtype {scores.dtype}")
    return WeightedSumAdmission(True, "eligible")


_SOURCE = """
    const uint token = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    threadgroup uint rows[TOPK];
    threadgroup O weights[TOPK];
    if (lid < uint(TOPK)) {
        rows[lid] = uint(inv_order[token * uint(TOPK) + lid]);
        weights[lid] = static_cast<O>(scores[token * uint(TOPK) + lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lid; d < uint(D); d += uint(THREADS)) {
        // MLX col_reduce_small: partial j holds rows j, j + PARTIALS, ...
        O partial[PARTIALS];
        for (int i = 0; i < PARTIALS; ++i) {
            partial[i] = O(0);
        }
        for (int k = 0; k < TOPK; ++k) {
            O xv = static_cast<O>(x_sorted[size_t(rows[k]) * size_t(D) + d]);
            O term = xv * weights[k];
            partial[k % PARTIALS] = term + partial[k % PARTIALS];
        }
        O total = partial[0];
        for (int i = 1; i < PARTIALS; ++i) {
            total = partial[i] + total;
        }
        y[size_t(token) * size_t(D) + d] = total;
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mlx2_qwen4_moe_weighted_sum",
            input_names=["x_sorted", "inv_order", "scores"],
            output_names=["y"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def moe_weighted_sum(
    x_sorted: mx.array,
    inv_order: mx.array,
    scores: mx.array,
    *,
    partials: int | None = None,
) -> mx.array:
    """``sum_k x_sorted[inv_order[t*K+k]] * scores[t, k]`` -> ``scores.shape[:-1] + (D,)``.

    Callers must run ``admit_moe_weighted_sum`` first.  ``partials`` overrides
    the reduction order for tests only; the default replays MLX's.
    """
    top_k = int(scores.shape[-1])
    tokens = int(scores.size) // top_k
    hidden = int(x_sorted.shape[-1])
    if partials is None:
        partials = min(COL_REDUCE_SMALL_PARTIALS, top_k)
    # Admission allows scores in the activation dtype or float32, so the
    # eager product's promotion is one of these two.
    out_dtype = x_sorted.dtype if scores.dtype == x_sorted.dtype else mx.float32
    y = _kernel()(
        inputs=[x_sorted, inv_order.astype(mx.uint32), scores],
        template=[
            ("O", out_dtype),
            ("TOPK", top_k),
            ("PARTIALS", int(partials)),
            ("D", hidden),
            ("THREADS", THREADS),
        ],
        grid=(tokens * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(tokens, hidden)],
        output_dtypes=[out_dtype],
    )[0]
    return y.reshape(*scores.shape[:-1], hidden)


__all__ = [
    "WEIGHTED_SUM_ENV",
    "admit_moe_weighted_sum",
    "moe_weighted_sum",
    "runtime_supported",
    "weighted_sum_enabled_from_env",
]
