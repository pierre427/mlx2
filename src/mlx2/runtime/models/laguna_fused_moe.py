# SPDX-License-Identifier: MIT
"""Shape-locked Metal kernels for Laguna XS 2.1's q8 MoE."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

HIDDEN_SIZE = 2048
EXPERT_HIDDEN_SIZE = 512
NUM_EXPERTS = 256
TOP_K = 8
GROUP_SIZE = 64
BITS = 8
PACK_FACTOR = 4
QUALIFIED_TOKEN_WIDTHS: tuple[int, ...] = ()
CANDIDATE_TOKEN_WIDTHS: tuple[int, ...] = (1, 2, 4, 8)
_INDEX_DTYPES = (mx.int32, mx.uint32)


@dataclass(frozen=True)
class KernelAdmission:
    accepted: bool
    reason: str
    tokens: int = 0


def _shape(value):
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def _enabled_widths(candidate_token_widths):
    requested = tuple(candidate_token_widths)
    unsupported = tuple(width for width in requested if width not in CANDIDATE_TOKEN_WIDTHS)
    if unsupported:
        raise ValueError(
            f"unsupported Laguna fused-MoE token widths {unsupported}; "
            f"candidates are {CANDIDATE_TOKEN_WIDTHS}"
        )
    return tuple(dict.fromkeys((*QUALIFIED_TOKEN_WIDTHS, *requested)))


def admit_laguna_fused_down(
    hidden,
    indices,
    scores,
    down_weight,
    down_scales,
    down_biases,
    *,
    num_experts=NUM_EXPERTS,
    group_size=GROUP_SIZE,
    bits=BITS,
    mode="affine",
    candidate_token_widths=(),
):
    hidden_shape = _shape(hidden)
    if hidden_shape is None or len(hidden_shape) < 3:
        return KernelAdmission(False, "hidden must end in [top_k, 512]")
    if hidden_shape[-2:] != (TOP_K, EXPERT_HIDDEN_SIZE):
        return KernelAdmission(False, "hidden must end in [8, 512]")
    tokens = 1
    for extent in hidden_shape[:-2]:
        tokens *= extent
    if tokens not in _enabled_widths(candidate_token_widths):
        return KernelAdmission(False, f"M={tokens} is not enabled", tokens)
    routed_shape = hidden_shape[:-2] + (TOP_K,)
    if _shape(indices) != routed_shape or _shape(scores) != routed_shape:
        return KernelAdmission(False, "indices/scores must end in top_k=8", tokens)
    if hidden.dtype != mx.bfloat16 or scores.dtype != mx.bfloat16:
        return KernelAdmission(False, "hidden and scores must be bfloat16", tokens)
    if indices.dtype not in _INDEX_DTYPES:
        return KernelAdmission(False, "indices must be int32 or uint32", tokens)
    if num_experts != NUM_EXPERTS:
        return KernelAdmission(False, "only 256 routed experts are supported", tokens)
    if (group_size, bits, mode) != (GROUP_SIZE, BITS, "affine"):
        return KernelAdmission(False, "only affine q8 group-size 64 is supported", tokens)
    expected = (
        (down_weight, (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // PACK_FACTOR)),
        (down_scales, (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // GROUP_SIZE)),
        (down_biases, (NUM_EXPERTS, HIDDEN_SIZE, EXPERT_HIDDEN_SIZE // GROUP_SIZE)),
    )
    for value, wanted in expected:
        if _shape(value) != wanted:
            return KernelAdmission(False, f"q8 table shape {_shape(value)} != {wanted}", tokens)
    if down_weight.dtype != mx.uint32:
        return KernelAdmission(False, "packed q8 weights must be uint32", tokens)
    if down_scales.dtype != mx.bfloat16 or down_biases.dtype != down_scales.dtype:
        return KernelAdmission(False, "q8 scales/biases must be bfloat16", tokens)
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return KernelAdmission(False, "MLX Metal kernels are unavailable", tokens)
    if mx.default_device() != mx.gpu:
        return KernelAdmission(False, "default MLX device is not GPU", tokens)
    return KernelAdmission(True, "eligible", tokens)


_DOWN_SOURCE = r"""
    constexpr uint H = 2048;
    constexpr uint EH = 512;
    constexpr uint TOPK = 8;
    constexpr uint WORDS = 128;
    constexpr uint GROUPS = 8;

    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.y;
    uint token = thread_position_in_grid.z;
    const device uint32_t* packed = down_weight;
    float routed = 0.0f;

#pragma unroll
    for (uint slot = 0; slot < TOPK; ++slot) {
        uint elem = token * TOPK + slot;
        uint expert = uint(indices[elem_to_loc(
            elem, indices_shape, indices_strides, indices_ndim)]);
        size_t wrow = size_t(expert) * H + row;
        const device uint32_t* dw = packed + wrow * WORDS;
        const device W* ds = down_scales + wrow * GROUPS;
        const device W* db = down_biases + wrow * GROUPS;
        const device T* hrow = hidden + size_t(elem) * EH;
        float value = 0.0f;
        for (uint word = lane; word < WORDS; word += 32) {
            uint32_t p = dw[word];
            uint group = word >> 4;
            float scale = float(ds[group]);
            float bias = float(db[group]);
            size_t base = size_t(word) * 4;
            float accum_q = 0.0f;
            float accum_x = 0.0f;
#pragma unroll
            for (uint byte = 0; byte < 4; ++byte) {
                float xv = float(hrow[base + byte]);
                accum_x += xv;
                accum_q += xv * float((p >> (8 * byte)) & 0xFFu);
            }
            value += scale * accum_q + bias * accum_x;
        }
        value = simd_sum(value);
        if (lane == 0) {
            T expert_value = static_cast<T>(value);
            T weighted = static_cast<T>(float(expert_value) * float(scores[
                elem_to_loc(elem, scores_shape, scores_strides, scores_ndim)]));
            routed += float(weighted);
        }
    }
    if (lane == 0)
        out[size_t(token) * H + row] = static_cast<T>(routed);
"""

_down_kernel = None


def _get_down_kernel():
    global _down_kernel
    if _down_kernel is None:
        _down_kernel = mx.fast.metal_kernel(
            name="laguna_q8_down_top8_reduce",
            input_names=[
                "hidden", "down_weight", "down_scales", "down_biases",
                "indices", "scores",
            ],
            output_names=["out"],
            source=_DOWN_SOURCE,
            ensure_row_contiguous=False,
        )
    return _down_kernel


def laguna_fused_down(
    hidden,
    indices,
    scores,
    down_weight,
    down_scales,
    down_biases,
    *,
    num_experts=NUM_EXPERTS,
    group_size=GROUP_SIZE,
    bits=BITS,
    mode="affine",
    candidate_token_widths=(),
):
    # The Metal body uses direct row-major addressing for the exact q8 tables.
    # ``mx.contiguous`` is a no-op for the normal loaded artifact and also keeps
    # unusual views from being admitted with incompatible physical strides.
    hidden = mx.contiguous(hidden)
    down_weight = mx.contiguous(down_weight)
    down_scales = mx.contiguous(down_scales)
    down_biases = mx.contiguous(down_biases)
    admission = admit_laguna_fused_down(
        hidden, indices, scores, down_weight, down_scales, down_biases,
        num_experts=num_experts, group_size=group_size, bits=bits, mode=mode,
        candidate_token_widths=candidate_token_widths,
    )
    if not admission.accepted:
        raise ValueError(f"Laguna fused down is not eligible: {admission.reason}")
    out = _get_down_kernel()(
        inputs=[hidden, down_weight, down_scales, down_biases, indices, scores],
        template=[("T", hidden.dtype), ("W", down_scales.dtype)],
        grid=(32, HIDDEN_SIZE, admission.tokens),
        threadgroup=(32, 1, 1),
        output_shapes=[(admission.tokens, HIDDEN_SIZE)],
        output_dtypes=[hidden.dtype],
    )[0]
    return out.reshape(hidden.shape[:-2] + (HIDDEN_SIZE,))


def admit_laguna_router(
    logits,
    correction_bias,
    *,
    top_k,
    norm_topk_prob,
    use_sigmoid,
    softcap,
    candidate_token_widths=(),
):
    shape = _shape(logits)
    if shape is None or len(shape) < 2 or shape[-1] != NUM_EXPERTS:
        return KernelAdmission(False, "logits must end in 256")
    tokens = 1
    for extent in shape[:-1]:
        tokens *= extent
    if tokens not in _enabled_widths(candidate_token_widths):
        return KernelAdmission(False, f"M={tokens} is not enabled", tokens)
    if logits.dtype != mx.float32:
        return KernelAdmission(False, "router logits must be float32", tokens)
    if _shape(correction_bias) != (NUM_EXPERTS,) or correction_bias.dtype not in (
        mx.bfloat16, mx.float32
    ):
        return KernelAdmission(False, "correction bias must be BF16/F32 [256]", tokens)
    if top_k != TOP_K or not norm_topk_prob or not use_sigmoid or softcap != 0.0:
        return KernelAdmission(False, "requires sigmoid normalized top-8 without softcap", tokens)
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return KernelAdmission(False, "MLX Metal kernels are unavailable", tokens)
    if mx.default_device() != mx.gpu:
        return KernelAdmission(False, "default MLX device is not GPU", tokens)
    return KernelAdmission(True, "eligible", tokens)


_ROUTER_SOURCE = r"""
    constexpr uint EXPERTS = 256;
    constexpr uint TOPK = 8;
    uint tid = thread_position_in_threadgroup.x;
    uint token = thread_position_in_grid.z;
    uint base = token * EXPERTS;
    uint route_base = token * TOPK;
    threadgroup float probabilities[EXPERTS];

#pragma unroll
    for (uint i = 0; i < 4; ++i) {
        uint expert = tid * 4 + i;
        float value = logits[base + expert];
        probabilities[expert] = 1.0f / (1.0f + metal::precise::exp(-value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0) return;

    float topv[TOPK];
    uint topi[TOPK];
    for (uint j = 0; j < TOPK; ++j) {
        topv[j] = -INFINITY;
        topi[j] = 0;
    }
    for (uint expert = 0; expert < EXPERTS; ++expert) {
        float corrected = probabilities[expert] + float(correction_bias[expert]);
        if (corrected < topv[0]) continue;
        uint pos = 0;
        while (pos < TOPK && (corrected > topv[pos] ||
               (corrected == topv[pos] && expert > topi[pos]))) ++pos;
        if (pos == 0) continue;
        for (uint j = 0; j + 1 < pos; ++j) {
            topv[j] = topv[j + 1];
            topi[j] = topi[j + 1];
        }
        topv[pos - 1] = corrected;
        topi[pos - 1] = expert;
    }
    float total = 0.0f;
    for (uint j = 0; j < TOPK; ++j)
        total += probabilities[topi[j]];
    for (uint j = 0; j < TOPK; ++j) {
        indices[route_base + j] = topi[j];
        scores[route_base + j] = probabilities[topi[j]] / total;
    }
"""

_router_kernel_bf16 = None
_router_kernel_float = None


def _get_router_kernel(bias_dtype):
    global _router_kernel_bf16, _router_kernel_float
    name = "laguna_sigmoid_bias_top8_bf16" if bias_dtype == mx.bfloat16 else "laguna_sigmoid_bias_top8_f32"
    target = _router_kernel_bf16 if bias_dtype == mx.bfloat16 else _router_kernel_float
    if target is None:
        target = mx.fast.metal_kernel(
            name=name,
            input_names=["logits", "correction_bias"],
            output_names=["indices", "scores"],
            source=_ROUTER_SOURCE,
            ensure_row_contiguous=True,
        )
        if bias_dtype == mx.bfloat16:
            _router_kernel_bf16 = target
        else:
            _router_kernel_float = target
    return target


def laguna_fused_router(
    logits,
    correction_bias,
    *,
    top_k=TOP_K,
    norm_topk_prob=True,
    use_sigmoid=True,
    softcap=0.0,
    output_dtype=mx.bfloat16,
    candidate_token_widths=(),
):
    admission = admit_laguna_router(
        logits, correction_bias, top_k=top_k,
        norm_topk_prob=norm_topk_prob, use_sigmoid=use_sigmoid, softcap=softcap,
        candidate_token_widths=candidate_token_widths,
    )
    if not admission.accepted:
        raise ValueError(f"Laguna fused router is not eligible: {admission.reason}")
    kernel = _get_router_kernel(correction_bias.dtype)
    indices, scores = kernel(
        inputs=[logits, correction_bias],
        template=[("B", correction_bias.dtype)],
        grid=(64, 1, admission.tokens),
        threadgroup=(64, 1, 1),
        output_shapes=[logits.shape[:-1] + (TOP_K,), logits.shape[:-1] + (TOP_K,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return indices, scores.astype(output_dtype)
