# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import mlx.core as mx

NUM_EXPERTS = 512
TOP_K = 10
QUALIFIED_TOKEN_WIDTHS: tuple[int, ...] = (1,)
CANDIDATE_TOKEN_WIDTHS: tuple[int, ...] = (3,)


@dataclass(frozen=True)
class RouterAdmission:
    accepted: bool
    reason: str


def _enabled_token_widths(candidate_token_widths: Sequence[int]) -> tuple[int, ...]:
    requested = tuple(candidate_token_widths)
    unsupported = tuple(w for w in requested if w not in CANDIDATE_TOKEN_WIDTHS)
    if unsupported:
        raise ValueError(
            f"unsupported router candidate token widths {unsupported}; "
            f"available candidates are {CANDIDATE_TOKEN_WIDTHS}"
        )
    return tuple(dict.fromkeys((*QUALIFIED_TOKEN_WIDTHS, *requested)))


def admit_qwen4_moe_router(
    gates,
    *,
    top_k: int,
    norm_topk_prob: bool,
    candidate_token_widths: Sequence[int] = (),
):
    if gates.shape == (1, 1, NUM_EXPERTS):
        tokens = 1
    elif gates.shape == (1, 3, NUM_EXPERTS):
        tokens = 3
    else:
        return RouterAdmission(
            False, "gates must be B1/M1/512 or the explicit B1/M3/512 candidate"
        )
    enabled_widths = _enabled_token_widths(candidate_token_widths)
    if tokens not in enabled_widths:
        return RouterAdmission(
            False,
            f"flattened token width M={tokens} is not qualified for the fused "
            f"router (qualified: {QUALIFIED_TOKEN_WIDTHS}; explicit candidates: "
            f"{CANDIDATE_TOKEN_WIDTHS})",
        )
    if gates.dtype != mx.bfloat16:
        return RouterAdmission(False, "gates must be bfloat16")
    if top_k != TOP_K or not norm_topk_prob:
        return RouterAdmission(False, "only normalized top-10 routing")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return RouterAdmission(False, "Metal GPU unavailable")
    return RouterAdmission(True, "eligible")


_HEADER = "\n#include <metal_stdlib>\nusing namespace metal;\n"
_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint token = thread_position_in_grid.z;
    const uint gate_base = token * 512;
    const uint route_base = token * 10;
    threadgroup float local_max[32];
    threadgroup float local_sum[32];
    threadgroup T probabilities[512];

    float values[4];
    float vmax = -INFINITY;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(gates[gate_base + tid * 4 + i]);
        vmax = metal::max(vmax, values[i]);
    }
    if (sg == 0) {
        local_max[lane] = -INFINITY;
        local_sum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(vmax);
    if (lane == 0) local_max[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        vmax = simd_max(local_max[lane]);
        if (lane == 0) local_max[0] = vmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = local_max[0];

    float normalizer = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = metal::fast::exp(values[i] - vmax);
        normalizer += values[i];
    }
    normalizer = simd_sum(normalizer);
    if (lane == 0) local_sum[sg] = normalizer;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        normalizer = simd_sum(local_sum[lane]);
        if (lane == 0) local_sum[0] = normalizer;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normalizer = 1.0f / local_sum[0];
    for (uint i = 0; i < 4; ++i)
        probabilities[tid * 4 + i] = T(values[i] * normalizer);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) return;

    float topv[10];
    uint topi[10];
    for (uint j = 0; j < 10; ++j) {
        topv[j] = -INFINITY;
        topi[j] = 0;
    }

    // Maintain the ten largest values in ascending order. MLX argpartition's
    // selected suffix uses that same order for this production geometry.
    for (uint e = 0; e < 512; ++e) {
        float value = float(probabilities[e]);
        if (value < topv[0]) continue;
        uint pos = 0;
        while (pos < 10 && (value > topv[pos] ||
               (value == topv[pos] && e > topi[pos]))) ++pos;
        if (pos == 0) continue;
        for (uint j = 0; j + 1 < pos; ++j) {
            topv[j] = topv[j + 1];
            topi[j] = topi[j + 1];
        }
        topv[pos - 1] = value;
        topi[pos - 1] = e;
    }

    T probs[10];
    // MLX's length-10 BF16 row reduction accumulates in BF16, rounding after
    // every add. Mirror that order so the normalized router scores are exact.
    T selected_sum = T(0.0f);
    for (uint j = 0; j < 10; ++j) {
        probs[j] = probabilities[topi[j]];
        selected_sum = T(probs[j] + selected_sum);
    }
    for (uint j = 0; j < 10; ++j) {
        indices[route_base + j] = topi[j];
        scores[route_base + j] = T(float(probs[j]) / float(selected_sum));
    }
"""
_KERNEL = mx.fast.metal_kernel(
    name="qwen4_moe_router_m1_m3_candidate",
    input_names=["gates"],
    output_names=["indices", "scores"],
    header=_HEADER,
    source=_SOURCE,
    ensure_row_contiguous=True,
)


def qwen4_moe_router(gates, *, candidate_token_widths: Sequence[int] = ()):
    """Return top-10 routes; non-qualified widths require explicit opt-in."""
    admission = admit_qwen4_moe_router(
        gates,
        top_k=TOP_K,
        norm_topk_prob=True,
        candidate_token_widths=candidate_token_widths,
    )
    if not admission.accepted:
        raise ValueError(f"Qwen4 fused router is not eligible: {admission.reason}")
    tokens = 1
    for extent in gates.shape[:-1]:
        tokens *= extent
    (indices, scores) = _KERNEL(
        inputs=[gates],
        template=[("T", gates.dtype)],
        grid=(128, 1, tokens),
        threadgroup=(128, 1, 1),
        output_shapes=[gates.shape[:-1] + (TOP_K,), gates.shape[:-1] + (TOP_K,)],
        output_dtypes=[mx.uint32, gates.dtype],
    )
    return (indices, scores)


_PROBE_COMPLETE = False
_PROBE_OK = False


def probe_qwen4_moe_router(dtype=mx.bfloat16) -> bool:
    global _PROBE_COMPLETE, _PROBE_OK
    if _PROBE_COMPLETE:
        return _PROBE_OK
    _PROBE_COMPLETE = True
    if dtype != mx.bfloat16 or not mx.metal.is_available():
        return False
    try:
        gates = mx.arange(NUM_EXPERTS, dtype=dtype)[None, None, :]
        (indices, scores) = qwen4_moe_router(gates)
        mx.eval(indices, scores)
        _PROBE_OK = indices.tolist() == [[list(range(502, 512))]]
    except (RuntimeError, ValueError):
        _PROBE_OK = False
    return _PROBE_OK
