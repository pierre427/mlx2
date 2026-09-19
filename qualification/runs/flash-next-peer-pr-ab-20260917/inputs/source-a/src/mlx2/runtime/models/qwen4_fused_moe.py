# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import mlx.core as mx

HIDDEN_SIZE = 2560
EXPERT_HIDDEN_SIZE = 640
TOP_K = 10
NUM_EXPERTS = 512
GROUP_SIZE = 64
BITS = 4
PACK_FACTOR = 8
_SIMD_WIDTH = 32
_TILE4_SIMDGROUPS = 5
_TILE4_THREADS = _SIMD_WIDTH * _TILE4_SIMDGROUPS
_VARIANTS = ("scalar", "tile4")
QUALIFIED_TOKEN_WIDTHS: tuple[int, ...] = (1,)
CANDIDATE_TOKEN_WIDTHS: tuple[int, ...] = (3,)
AUTO_VARIANT_BY_WIDTH: dict[int, str] = {1: "tile4", 3: "tile4"}
_INDEX_DTYPES = (mx.int32, mx.uint32)


@dataclass(frozen=True)
class FusedMoeAdmission:
    """Result of the exact-shape admission check."""

    accepted: bool
    reason: str
    tokens: int = 0


def _shape(value) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def _dtype_in(value, allowed: Sequence) -> bool:
    return getattr(value, "dtype", None) in allowed


def _enabled_token_widths(candidate_token_widths: Sequence[int]) -> tuple[int, ...]:
    """Return qualified widths plus explicitly enabled experiment candidates."""
    requested = tuple(candidate_token_widths)
    unsupported = tuple(w for w in requested if w not in CANDIDATE_TOKEN_WIDTHS)
    if unsupported:
        raise ValueError(
            f"unsupported fused-down candidate token widths {unsupported}; "
            f"available candidates are {CANDIDATE_TOKEN_WIDTHS}"
        )
    return tuple(dict.fromkeys((*QUALIFIED_TOKEN_WIDTHS, *requested)))


def admit_qwen4_fused_down(
    hidden: mx.array,
    indices: mx.array,
    scores: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array | None,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
    candidate_token_widths: Sequence[int] = (),
) -> FusedMoeAdmission:
    """Admit production geometry plus explicitly requested candidate widths.

    The check is structural. It does not evaluate an MLX array or inspect
    index values, so it cannot add a device synchronization to decode. The
    default remains the qualified production widths only.
    """
    hidden_shape = _shape(hidden)
    if hidden_shape is None or len(hidden_shape) < 3:
        return FusedMoeAdmission(False, "hidden must end in [top_k, 640]")
    if hidden_shape[-2:] != (TOP_K, EXPERT_HIDDEN_SIZE):
        return FusedMoeAdmission(False, "hidden must end in [10, 640]")
    tokens = 1
    for extent in hidden_shape[:-2]:
        tokens *= extent
    enabled_widths = _enabled_token_widths(candidate_token_widths)
    if tokens not in enabled_widths:
        qualified = ", ".join((f"M={w}" for w in QUALIFIED_TOKEN_WIDTHS))
        candidates = ", ".join((f"M={w}" for w in CANDIDATE_TOKEN_WIDTHS))
        return FusedMoeAdmission(
            False,
            f"flattened token width M={tokens} is not qualified for the fused down "
            f"kernel (qualified: {qualified}; explicit candidates: {candidates})",
            tokens,
        )
    routed_shape = hidden_shape[:-2] + (TOP_K,)
    if _shape(indices) != routed_shape or _shape(scores) != routed_shape:
        return FusedMoeAdmission(
            False, "indices and scores must match hidden prefix plus top_k=10", tokens
        )
    if hidden.dtype != mx.bfloat16:
        return FusedMoeAdmission(False, "hidden must be bfloat16", tokens)
    if scores.dtype != hidden.dtype:
        return FusedMoeAdmission(False, "scores dtype must match hidden dtype", tokens)
    if not _dtype_in(indices, _INDEX_DTYPES):
        return FusedMoeAdmission(False, "indices must be int32 or uint32", tokens)
    if num_experts != NUM_EXPERTS:
        return FusedMoeAdmission(False, "only 512 routed experts are supported", tokens)
    if (group_size, bits, mode) != (GROUP_SIZE, BITS, "affine"):
        return FusedMoeAdmission(
            False, "only affine q4 with group_size=64 is supported", tokens
        )
    if down_biases is None:
        return FusedMoeAdmission(False, "affine q4 requires a bias table", tokens)
    packed_width = EXPERT_HIDDEN_SIZE // PACK_FACTOR
    group_width = EXPERT_HIDDEN_SIZE // GROUP_SIZE
    expected = (
        (down_weight, (NUM_EXPERTS, HIDDEN_SIZE, packed_width)),
        (down_scales, (NUM_EXPERTS, HIDDEN_SIZE, group_width)),
        (down_biases, (NUM_EXPERTS, HIDDEN_SIZE, group_width)),
    )
    for value, wanted in expected:
        if _shape(value) != wanted:
            return FusedMoeAdmission(
                False,
                f"packed table shape {_shape(value)} does not match {wanted}",
                tokens,
            )
    if down_weight.dtype != mx.uint32:
        return FusedMoeAdmission(False, "packed q4 weights must be uint32", tokens)
    if down_scales.dtype != mx.bfloat16:
        return FusedMoeAdmission(False, "q4 scales and biases must be bfloat16", tokens)
    if down_biases.dtype != down_scales.dtype:
        return FusedMoeAdmission(
            False, "q4 scales and biases must have one dtype", tokens
        )
    if not hasattr(mx.fast, "metal_kernel") or not mx.metal.is_available():
        return FusedMoeAdmission(False, "MLX Metal kernels are unavailable", tokens)
    if mx.default_device() != mx.gpu:
        return FusedMoeAdmission(False, "the default MLX device is not GPU", tokens)
    return FusedMoeAdmission(True, "eligible", tokens)


def auto_variant(tokens: int) -> str:
    """Variant the ``auto`` policy dispatches at an admitted token width."""
    try:
        return AUTO_VARIANT_BY_WIDTH[tokens]
    except KeyError:
        raise ValueError(
            f"no auto variant for flattened token width M={tokens}; qualified widths are {QUALIFIED_TOKEN_WIDTHS}"
        ) from None


_DOWN_REDUCE_SOURCE = "\n    constexpr uint H = 2560;\n    constexpr uint EH = 640;\n    constexpr uint TOPK = 10;\n    constexpr uint DOWN_WORDS = 80;\n    constexpr uint DOWN_GROUPS = 10;\n\n    uint lane = thread_index_in_simdgroup;\n    uint row = thread_position_in_grid.y;\n    uint token = thread_position_in_grid.z;\n    const device uint32_t* packed = down_weight;\n\n    float routed = 0.0f;\n#pragma unroll\n    for (uint slot = 0; slot < TOPK; ++slot) {\n        // indices/scores/hidden are addressed through their real strides:\n        // the router hands over a strided top-k view, not a packed [M, 10].\n        // hidden's innermost (640) axis is unit-stride by contract.\n        // 32-bit elem_to_loc: these views are tiny (M x 10 x 640 at most),\n        // and the 64-bit overload costs a chain of 64-bit divisions.\n        uint elem = token * TOPK + slot;\n        uint expert = uint(indices[elem_to_loc(\n            elem, indices_shape, indices_strides, indices_ndim)]);\n        size_t wrow = size_t(expert) * H + row;\n        const device uint32_t* dw = packed + wrow * DOWN_WORDS;\n        const device W* ds = down_scales + wrow * DOWN_GROUPS;\n        const device W* db = down_biases + wrow * DOWN_GROUPS;\n        const device T* hrow = hidden + elem_to_loc(\n            elem * EH, hidden_shape, hidden_strides, hidden_ndim);\n\n        float value = 0.0f;\n        for (uint word = lane; word < DOWN_WORDS; word += 32) {\n            uint32_t p = dw[word];\n            uint group = word >> 3;\n            float scale = float(ds[group]);\n            float bias = float(db[group]);\n            size_t hbase = size_t(word) * 8;\n            float accum_q = 0.0f;\n            float accum_x = 0.0f;\n#pragma unroll\n            for (uint nibble = 0; nibble < 8; ++nibble) {\n                float xv = float(hrow[hbase + nibble]);\n                accum_x += xv;\n                accum_q += xv * float((p >> (4 * nibble)) & 0xFu);\n            }\n            value += scale * accum_q + bias * accum_x;\n        }\n        value = simd_sum(value);\n        if (lane == 0) {\n            // Match the two T-valued boundaries in the stock graph.\n            T expert_value = static_cast<T>(value);\n            T weighted_value = static_cast<T>(\n                float(expert_value) * float(scores[elem_to_loc(\n                    elem, scores_shape, scores_strides, scores_ndim)]));\n            routed += float(weighted_value);\n        }\n    }\n    if (lane == 0) {\n        out[size_t(token) * H + row] = static_cast<T>(routed);\n    }\n"
_DOWN_REDUCE_TILE4_SOURCE = "\n    constexpr uint H = 2560;\n    constexpr uint EH = 640;\n    constexpr uint TOPK = 10;\n    constexpr uint DOWN_WORDS = 80;\n    constexpr uint DOWN_GROUPS = 10;\n\n    uint lane = thread_index_in_simdgroup;\n    uint sg = simdgroup_index_in_threadgroup;\n    uint row_base = thread_position_in_grid.y * 4;\n    uint token = thread_position_in_grid.z;\n    uint slot_base = sg * 2;\n    const device uint32_t* packed = down_weight;\n    threadgroup float partials[TOPK * 4];\n\n    float values[8];\n#pragma unroll\n    for (uint i = 0; i < 8; ++i) {\n        values[i] = 0.0f;\n    }\n\n#pragma unroll\n    for (uint local_slot = 0; local_slot < 2; ++local_slot) {\n        uint slot = slot_base + local_slot;\n        // Strided routing views, as in the scalar kernel.\n        uint elem = token * TOPK + slot;\n        uint expert = uint(indices[elem_to_loc(\n            elem, indices_shape, indices_strides, indices_ndim)]);\n        const device T* hrow = hidden + elem_to_loc(\n            elem * EH, hidden_shape, hidden_strides, hidden_ndim);\n\n        for (uint word = lane; word < DOWN_WORDS; word += 32) {\n            size_t hbase = size_t(word) * 8;\n            float xv[8];\n            float accum_x = 0.0f;\n#pragma unroll\n            for (uint nibble = 0; nibble < 8; ++nibble) {\n                xv[nibble] = float(hrow[hbase + nibble]);\n                accum_x += xv[nibble];\n            }\n\n#pragma unroll\n            for (uint local_row = 0; local_row < 4; ++local_row) {\n                uint row = row_base + local_row;\n                size_t wrow = size_t(expert) * H + row;\n                uint32_t p = packed[wrow * DOWN_WORDS + word];\n                uint group = word >> 3;\n                float scale = float(\n                    down_scales[wrow * DOWN_GROUPS + group]);\n                float bias = float(\n                    down_biases[wrow * DOWN_GROUPS + group]);\n                float accum_q = 0.0f;\n#pragma unroll\n                for (uint nibble = 0; nibble < 8; ++nibble) {\n                    accum_q += xv[nibble] *\n                        float((p >> (4 * nibble)) & 0xFu);\n                }\n                values[local_slot * 4 + local_row] +=\n                    scale * accum_q + bias * accum_x;\n            }\n        }\n    }\n\n#pragma unroll\n    for (uint i = 0; i < 8; ++i) {\n        values[i] = simd_sum(values[i]);\n    }\n    if (lane == 0) {\n#pragma unroll\n        for (uint local_slot = 0; local_slot < 2; ++local_slot) {\n            uint slot = slot_base + local_slot;\n            float score = float(scores[elem_to_loc(\n                token * TOPK + slot, scores_shape, scores_strides,\n                scores_ndim)]);\n#pragma unroll\n            for (uint local_row = 0; local_row < 4; ++local_row) {\n                T expert_value = static_cast<T>(\n                    values[local_slot * 4 + local_row]);\n                T weighted_value = static_cast<T>(\n                    float(expert_value) * score);\n                partials[slot * 4 + local_row] = float(weighted_value);\n            }\n        }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (sg == 0 && lane < 4) {\n        float routed = 0.0f;\n#pragma unroll\n        for (uint slot = 0; slot < TOPK; ++slot) {\n            routed += partials[slot * 4 + lane];\n        }\n        out[size_t(token) * H + row_base + lane] = static_cast<T>(routed);\n    }\n"
_scalar_down_reduce_kernel = None
_tile4_down_reduce_kernel = None


def _kernel(variant: str):
    global _scalar_down_reduce_kernel, _tile4_down_reduce_kernel
    if variant == "scalar" and _scalar_down_reduce_kernel is None:
        _scalar_down_reduce_kernel = mx.fast.metal_kernel(
            name="qwen4_q4_down_reduce_scalar",
            input_names=[
                "hidden",
                "down_weight",
                "down_scales",
                "down_biases",
                "indices",
                "scores",
            ],
            output_names=["out"],
            source=_DOWN_REDUCE_SOURCE,
            ensure_row_contiguous=False,
        )
    if variant == "tile4" and _tile4_down_reduce_kernel is None:
        _tile4_down_reduce_kernel = mx.fast.metal_kernel(
            name="qwen4_q4_down_reduce_tile4",
            input_names=[
                "hidden",
                "down_weight",
                "down_scales",
                "down_biases",
                "indices",
                "scores",
            ],
            output_names=["out"],
            source=_DOWN_REDUCE_TILE4_SOURCE,
            ensure_row_contiguous=False,
        )
    return (
        _scalar_down_reduce_kernel if variant == "scalar" else _tile4_down_reduce_kernel
    )


def qwen4_fused_down(
    hidden: mx.array,
    indices: mx.array,
    scores: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    down_biases: mx.array,
    *,
    num_experts: int = NUM_EXPERTS,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    mode: str = "affine",
    variant: str = "scalar",
    candidate_token_widths: Sequence[int] = (),
) -> mx.array:
    """Fuse q4 down projection, router weighting, and top-10 reduction.

    Raise when the exact production geometry (or an explicitly requested
    candidate width) is not present. The integration layer must catch the
    rejected admission before this call and use the stock ``gather_qmm`` path
    instead. Indices must come from the model's trusted
    top-k router. The weight tables must be row-contiguous (they are never
    copied); ``indices``, ``scores`` and ``hidden`` may be strided views, and
    only ``hidden``'s innermost axis must be unit-stride. ``variant`` is
    selected per call, so serving can tune it without reload.
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"unknown Qwen4 fused-down variant {variant!r}; expected one of {_VARIANTS}"
        )
    admission = admit_qwen4_fused_down(
        hidden,
        indices,
        scores,
        down_weight,
        down_scales,
        down_biases,
        num_experts=num_experts,
        group_size=group_size,
        bits=bits,
        mode=mode,
        candidate_token_widths=candidate_token_widths,
    )
    if not admission.accepted:
        raise ValueError(f"Qwen4 fused down is not eligible: {admission.reason}")
    if variant == "scalar":
        grid = (_SIMD_WIDTH, HIDDEN_SIZE, admission.tokens)
        threadgroup = (_SIMD_WIDTH, 1, 1)
    else:
        grid = (_TILE4_THREADS, HIDDEN_SIZE // 4, admission.tokens)
        threadgroup = (_TILE4_THREADS, 1, 1)
    out = _kernel(variant)(
        inputs=[hidden, down_weight, down_scales, down_biases, indices, scores],
        template=[("T", hidden.dtype), ("W", down_scales.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(admission.tokens, HIDDEN_SIZE)],
        output_dtypes=[hidden.dtype],
    )[0]
    return out.reshape(hidden.shape[:-2] + (HIDDEN_SIZE,))
