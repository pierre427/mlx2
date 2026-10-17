# SPDX-License-Identifier: Apache-2.0
"""Default-off exact small-row router candidate for Nemotron 3.5 Super."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

NUM_EXPERTS = 512
TOP_K = 22
MAX_ROWS = 16
ROUTED_SCALE = 5.0
_BIAS_DTYPES = (mx.bfloat16, mx.float32)


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str
    rows: int = 0


def admit(
    logits,
    correction_bias,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> Admission:
    shape = tuple(getattr(logits, "shape", ()))
    if len(shape) < 2 or shape[-1] != NUM_EXPERTS:
        return Admission(False, "logits must end in 512")
    rows = logits.size // NUM_EXPERTS
    if not 1 <= rows <= MAX_ROWS:
        return Admission(False, f"rows={rows} outside 1..{MAX_ROWS}", rows)
    if logits.dtype != mx.float32:
        return Admission(False, "logits must be float32", rows)
    if tuple(getattr(correction_bias, "shape", ())) != (NUM_EXPERTS,):
        return Admission(False, "correction bias must be [512]", rows)
    if correction_bias.dtype not in _BIAS_DTYPES:
        return Admission(False, "correction bias must be bfloat16 or float32", rows)
    if top_k != TOP_K:
        return Admission(False, "top-k must be 22", rows)
    if (n_group, topk_group) != (1, 1):
        return Admission(False, "group pruning is unsupported", rows)
    if not norm_topk_prob:
        return Admission(False, "top-k normalization is required", rows)
    if float(routed_scaling_factor) != ROUTED_SCALE:
        return Admission(False, "routed scale must be 5", rows)
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return Admission(False, "Metal kernels are unavailable", rows)
    if mx.default_device() != mx.gpu:
        return Admission(False, "default device is not GPU", rows)
    return Admission(True, "eligible", rows)


_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    constexpr uint PER_LANE = EXPERTS / 32;
    float probability[PER_LANE];
    float corrected[PER_LANE];
    uint expert_id[PER_LANE];

    for (uint i = 0; i < PER_LANE; ++i) {
      const uint expert = i * 32u + lane;
      const float value = logits[size_t(row) * EXPERTS + expert];
      const float p = 1.0f / (1.0f + metal::precise::exp(-value));
      probability[i] = p;
      corrected[i] = p + float(correction_bias[expert]);
      expert_id[i] = expert;
    }

    threadgroup uint selected_expert[TOPK];
    threadgroup float selected_score[TOPK];
    for (uint slot = 0; slot < TOPK; ++slot) {
      float local_value = -INFINITY;
      uint local_expert = 0u;
      for (uint i = 0; i < PER_LANE; ++i) {
        const uint expert = expert_id[i];
        bool used = false;
        for (uint prior = 0; prior < slot; ++prior)
          used = used || expert == selected_expert[prior];
        const float value = corrected[i];
        if (!used && !metal::isnan(value) &&
            (value > local_value ||
             (value == local_value && expert > local_expert))) {
          local_value = value;
          local_expert = expert;
        }
      }
      const float best_value = simd_max(local_value);
      const uint best_expert = simd_max(
          local_value == best_value ? local_expert : 0u);
      if (lane == 0u) {
        selected_expert[slot] = best_expert;
        selected_score[slot] = best_value;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float p = 0.0f;
    if (lane < TOPK) {
      const uint expert = selected_expert[lane];
      const float value = logits[size_t(row) * EXPERTS + expert];
      p = 1.0f / (1.0f + metal::precise::exp(-value));
    }
    const float total = simd_sum(p);
    if (lane < TOPK) {
      const size_t out = size_t(row) * TOPK + lane;
      indices[out] = selected_expert[lane];
      scores[out] = (p / (total + 1.0e-20f)) * 5.0f;
    }
"""

_KERNELS: dict[str, object] = {}


def _kernel(dtype):
    key = str(dtype)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name="mlx2_nemotron35_sigmoid_bias_top22_" + key.replace(".", "_"),
            input_names=["logits", "correction_bias"],
            output_names=["indices", "scores"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kernel
    return kernel


def router_top22(
    logits,
    correction_bias,
    *,
    top_k: int = TOP_K,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALE,
):
    admission = admit(
        logits,
        correction_bias,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )
    if not admission.accepted:
        raise ValueError(f"Nemotron router candidate refused: {admission.reason}")
    shape = logits.shape[:-1] + (TOP_K,)
    indices, scores = _kernel(correction_bias.dtype)(
        inputs=[logits, correction_bias],
        template=[("B", correction_bias.dtype), ("EXPERTS", NUM_EXPERTS), ("TOPK", TOP_K)],
        grid=(32, admission.rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[shape, shape],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, scores
