# SPDX-License-Identifier: Apache-2.0
# Structure mined from jundot/omlx 3e2bdb1f (Apache-2.0); re-expressed as a
# JIT kernel.  See docs/PROVENANCE.md and provenance/omlx-3903-qwen4-prefill.json.
"""Sorted-order MoE weighted sum for prefill widths.

For a sorted gather (``do_sort``), the stock ``SwitchGLU`` tail is::

    y = down_proj(hidden, idx, sorted_indices=True)     # (N[+pad], 1, D), sorted
    y = _scatter_unsort(y, inv_order, indices.shape)    # (B, S, K, 1, D)
    y = (y.squeeze(-2) * scores[..., None]).sum(-2)     # (B, S, D)

``moe_weighted_sum`` reads the sorted rows directly through ``inv_order`` and
writes the per-token sum, so neither the unsorted ``(B, S, K, D)`` copy nor
the product intermediate is materialized.  One threadgroup (256 threads) per
token stages that token's K row ids and scores in threadgroup memory, then
strides the hidden axis.  Accumulation is fp32, in slot order k = 0..K-1.

The kernel is a JIT re-expression of omlx's native ``moe_weighted_sum_tiled``
(``omlx/custom_kernels/common/csrc/kernels/quantized_moe.h``, dispatched by
``qwen35_prefill.cpp``; the Python routing is
``omlx/patches/qwen35_moe_weighted_sum.py``).  No omlx C++/Metal text is copied.

Semantics vs. the eager tail
----------------------------
* Output dtype is ``promote(x.dtype, scores.dtype)``, what the eager product
  produces: bf16 scores -> bf16 output, fp32 scores -> fp32 output.
* When scores share the activation dtype (the Qwen4 router hands back bf16
  scores: softmax(precise=True) keeps the input dtype), the eager product
  ``x * scores`` rounds each term to bf16 before the sum.  This kernel rounds
  each product to T before the fp32 accumulation (``ROUND_PRODUCT=1``), the
  same boundary mlx2's qualified fused-down kernel mirrors.  **omlx's kernel
  does not** -- it accumulates unrounded fp32 products -- so this is a
  deliberate modification toward the eager graph.
* The eager ``sum`` over the K axis is MLX's reduction; its order on GPU is
  not asserted here.  The GPU gate measures any residual ULP difference.
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
SUPPORTED_TOP_K = 10
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
    if top_k != SUPPORTED_TOP_K:
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
    threadgroup float weights[TOPK];
    if (lid < uint(TOPK)) {
        rows[lid] = uint(inv_order[token * uint(TOPK) + lid]);
        weights[lid] = float(scores[token * uint(TOPK) + lid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lid; d < uint(D); d += uint(THREADS)) {
        float acc = 0.0f;
        for (uint k = 0; k < uint(TOPK); ++k) {
            float xv = float(x_sorted[size_t(rows[k]) * size_t(D) + d]);
            float term = xv * weights[k];
            if (ROUND_PRODUCT) {
                term = float(static_cast<T>(term));
            }
            acc += term;
        }
        y[size_t(token) * size_t(D) + d] = static_cast<O>(acc);
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


def moe_weighted_sum(x_sorted: mx.array, inv_order: mx.array, scores: mx.array) -> mx.array:
    """``sum_k x_sorted[inv_order[t*K+k]] * scores[t, k]`` -> ``scores.shape[:-1] + (D,)``.

    Callers must run ``admit_moe_weighted_sum`` first.
    """
    top_k = int(scores.shape[-1])
    tokens = int(scores.size) // top_k
    hidden = int(x_sorted.shape[-1])
    # Admission allows scores in the activation dtype or float32, so the
    # eager product's promotion is one of these two.
    round_product = int(scores.dtype == x_sorted.dtype)
    out_dtype = x_sorted.dtype if round_product else mx.float32
    y = _kernel()(
        inputs=[x_sorted, inv_order.astype(mx.uint32), scores],
        template=[
            ("T", x_sorted.dtype),
            ("O", out_dtype),
            ("TOPK", top_k),
            ("D", hidden),
            ("THREADS", THREADS),
            ("ROUND_PRODUCT", round_product),
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
