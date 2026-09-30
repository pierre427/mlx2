# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified and halo-box/strix-llama.cpp; see
# docs/PROVENANCE.md and provenance/qsa-stage1-keys-stationary-2026-09-28.json.
from __future__ import annotations

import math
import os
import threading
from collections import Counter
from functools import lru_cache

import mlx.core as mx

from .qwen4_qsa_nax import nax_kernel_available

from .import_env import snapshot as _import_env_snapshot

_import_env_snapshot(__name__)

_SUPPORTED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)
_MAX_TOPK = 512
_EXACT_BAND_EXTRA = 32
_KEYS_STATIONARY = os.environ.get(
    "MLX_QWEN4_QSA_STAGE1_KEYS_STATIONARY", "0"
).strip().lower() in {"1", "true", "on", "yes"}
_KEYS_STATIONARY_MIN_QUERY = max(
    1,
    int(os.environ.get("MLX_QWEN4_QSA_STAGE1_KEYS_STATIONARY_MIN_QUERY", "64")),
)
_ONEPASS_TOPK = os.environ.get(
    "MLX_QWEN4_QSA_STAGE1_ONEPASS_TOPK", "0"
).strip().lower() in {"1", "true", "on", "yes"}
_ONEPASS_TOPK_MIN_BLOCKS = max(
    1,
    int(os.environ.get("MLX_QWEN4_QSA_STAGE1_ONEPASS_TOPK_MIN_BLOCKS", "2048")),
)
_ONEPASS_TOPK_MAX = 1024
_CANDIDATE_STATS_LOCK = threading.Lock()
_CANDIDATE_STATS = Counter()


def qsa_stage1_candidate_status(*, reset: bool = False) -> dict:
    """Describe opt-in stage-one candidates without touching the device."""
    with _CANDIDATE_STATS_LOCK:
        report = {
            "keys_stationary_configured": bool(_KEYS_STATIONARY),
            "keys_stationary_min_query": int(_KEYS_STATIONARY_MIN_QUERY),
            "keys_stationary_qualification": "not_qualified_benefit",
            "keys_stationary_selected": False,
            "onepass_topk_configured": bool(_ONEPASS_TOPK),
            "onepass_topk_min_blocks": int(_ONEPASS_TOPK_MIN_BLOCKS),
            "onepass_topk_qualification": "rejected_performance",
            "onepass_topk_selected": False,
            "qualification_receipt": (
                "qualification/runs/qsa-stage1-pr91-20260928/qualification.json"
            ),
            "default_producer": "mpp_exact_band",
            "default_selector": "radix_exact",
            "runtime_counts": dict(_CANDIDATE_STATS),
        }
        if reset:
            _CANDIDATE_STATS.clear()
    return report


def _record_candidate_dispatch(name: str) -> None:
    with _CANDIDATE_STATS_LOCK:
        _CANDIDATE_STATS[name] += 1


def qsa_stage1_kernel_available() -> bool:
    """Return whether a Metal custom kernel can run in this process."""
    return mx.metal.is_available() and mx.default_device() == mx.gpu


def qsa_stage1_supported(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> bool:
    """Check static geometry without evaluating device arrays."""
    if not qsa_stage1_kernel_available():
        return False
    if q.ndim != 4 or pooled.ndim != 3 or q_positions.ndim != 2:
        return False
    if q.shape[:2] != q_positions.shape:
        return False
    if q.shape[0] != pooled.shape[0] or q.shape[-1] != pooled.shape[-1]:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or pooled.dtype not in _SUPPORTED_DTYPES:
        return False
    if q_positions.dtype not in (mx.int32, mx.int64):
        return False
    if int(q.shape[1]) <= 0 or int(pooled.shape[1]) <= int(block_topk):
        return False
    return 1 <= int(block_topk) <= _MAX_TOPK and int(compress_ratio) > 0


def qsa_stage1_score_producer(
    q: mx.array, pooled: mx.array, *, block_topk: int = 512
) -> str:
    """Return the selected score producer without dispatching work."""
    if (
        q.shape[0] == pooled.shape[0] == 1
        and q.shape[2:] == (4, 128)
        and (pooled.shape[2] == 128)
        and (q.dtype == pooled.dtype)
        and (q.dtype in (mx.float16, mx.bfloat16))
        and (pooled.shape[1] > int(block_topk) + _EXACT_BAND_EXTRA)
        and nax_kernel_available()
    ):
        if _KEYS_STATIONARY and int(q.shape[1]) >= _KEYS_STATIONARY_MIN_QUERY:
            return "mpp_keys_stationary_exact_band"
        return "mpp_exact_band"
    return "mlx"


def qsa_stage1_selector_producer(*, blocks: int, block_topk: int) -> str:
    """Choose the exact selector without inspecting or evaluating arrays."""
    if (
        _ONEPASS_TOPK
        and int(blocks) >= _ONEPASS_TOPK_MIN_BLOCKS
        and 1 <= int(block_topk) <= min(int(blocks), _ONEPASS_TOPK_MAX)
    ):
        return "onepass_exact"
    return "radix_exact"


def qsa_stage1_route(
    q: mx.array, pooled: mx.array, *, block_topk: int = 512
) -> dict[str, str | None]:
    """Describe the scorer and every selector used by a stage-one call."""
    score_producer = qsa_stage1_score_producer(
        q, pooled, block_topk=block_topk
    )
    uses_exact_band = score_producer.endswith("_exact_band")
    primary_topk = (
        int(block_topk) + _EXACT_BAND_EXTRA
        if uses_exact_band
        else int(block_topk)
    )
    selector = qsa_stage1_selector_producer(
        blocks=int(pooled.shape[1]),
        block_topk=primary_topk,
    )
    refine_selector = (
        qsa_stage1_selector_producer(
            blocks=primary_topk,
            block_topk=int(block_topk),
        )
        if uses_exact_band
        else None
    )
    return {
        "score_producer": score_producer,
        "selector": selector,
        "refine_selector": refine_selector,
    }


_MPP_SCORE_HEADER = "\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n\nconstant constexpr uint QSA_SCORE_HEADS = 4;\nconstant constexpr uint QSA_SCORE_HEAD_DIM = 128;\nconstant constexpr uint QSA_SCORE_QUERY_TILE = 16;\nconstant constexpr uint QSA_SCORE_KEY_TILE = 32;\nconstant constexpr uint QSA_SCORE_QUERIES_PER_SIMDGROUP = 4;\nconstant constexpr uint QSA_SCORE_SIMDGROUPS = 4;\nconstant constexpr uint QSA_SCORE_THREADS = 128;\nconstant constexpr uint QSA_SCORE_K_FRAGMENTS = 8;\nconstant constexpr float QSA_SCORE_SQRT_HEAD_DIM = 11.313708498984761f;\n\ninline short2 qsa_score_nax_coord(ushort lane) {\n    const short qid = short(lane >> 2);\n    const short fragment_row = ((qid & 4) | ((short(lane) >> 1) & 3));\n    const short fragment_col = ((qid & 2) | (short(lane) & 1)) * 4;\n    return short2{fragment_col, fragment_row};\n}\n"
_MPP_SCORE_SOURCE = "\n    const uint tid = thread_position_in_threadgroup.x;\n    const uint simdgroup = tid >> 5;\n    const ushort lane = ushort(tid & 31u);\n    const uint rows = q_shape[1];\n    const uint blocks = pooled_shape[1];\n    const uint key_tiles = (blocks + QSA_SCORE_KEY_TILE - 1u) /\n        QSA_SCORE_KEY_TILE;\n    const uint tile = threadgroup_position_in_grid.x;\n    const uint query_tile = tile / key_tiles;\n    const uint key_tile = tile - query_tile * key_tiles;\n    const uint query0 =\n        query_tile * QSA_SCORE_QUERY_TILE +\n        simdgroup * QSA_SCORE_QUERIES_PER_SIMDGROUP;\n    const uint block0 = key_tile * QSA_SCORE_KEY_TILE;\n\n    threadgroup InT pooled_tile[QSA_SCORE_KEY_TILE * QSA_SCORE_HEAD_DIM];\n    constexpr uint VECTORS_PER_KEY = QSA_SCORE_HEAD_DIM / 4u;\n    constexpr uint TILE_VECTORS = QSA_SCORE_KEY_TILE * VECTORS_PER_KEY;\n    for (uint item = tid; item < TILE_VECTORS; item += QSA_SCORE_THREADS) {\n        const uint key_local = item / VECTORS_PER_KEY;\n        const uint dim4 = item - key_local * VECTORS_PER_KEY;\n        const uint block = block0 + key_local;\n        vec<InT, 4> values = vec<InT, 4>(InT(0));\n        if (block < blocks) {\n            const int64_t source =\n                int64_t(block) * pooled_strides[1] +\n                int64_t(dim4 * 4u) * pooled_strides[2];\n            for (uint elem = 0u; elem < 4u; ++elem) {\n                values[elem] = pooled[\n                    source + int64_t(elem) * pooled_strides[2]];\n            }\n        }\n        const uint destination =\n            key_local * QSA_SCORE_HEAD_DIM + dim4 * 4u;\n        for (uint elem = 0u; elem < 4u; ++elem) {\n            pooled_tile[destination + elem] = values[elem];\n        }\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(\n        16, 32, 16, false, true, true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul;\n    auto left = matmul.get_left_input_cooperative_tensor<InT, InT, float>();\n    auto right = matmul.get_right_input_cooperative_tensor<InT, InT, float>();\n    auto accumulator = matmul.get_destination_cooperative_tensor<\n        decltype(left), decltype(right), float>();\n\n    constexpr short ELEMENTS_PER_FRAGMENT = 8;\n    constexpr short ELEMENT_COLUMNS = 4;\n    constexpr short ELEMENT_ROW_JUMP = 8;\n    const short2 coordinate = qsa_score_nax_coord(lane);\n    for (short item = 0; item < 2 * ELEMENTS_PER_FRAGMENT; ++item) {\n        accumulator[item] = 0.0f;\n    }\n\n    for (uint k_frag = 0u; k_frag < QSA_SCORE_K_FRAGMENTS; ++k_frag) {\n        for (short row_part = 0; row_part < 2; ++row_part) {\n            const uint matrix_row =\n                uint(coordinate.y + row_part * ELEMENT_ROW_JUMP);\n            const uint head = matrix_row / QSA_SCORE_QUERIES_PER_SIMDGROUP;\n            const uint query_local =\n                matrix_row - head * QSA_SCORE_QUERIES_PER_SIMDGROUP;\n            const uint query = query0 + query_local;\n            vec<InT, 4> values = vec<InT, 4>(InT(0));\n            if (query < rows) {\n                const int64_t source =\n                    int64_t(query) * q_strides[1] +\n                    int64_t(head) * q_strides[2] +\n                    int64_t(k_frag * 16u + uint(coordinate.x)) * q_strides[3];\n                for (uint elem = 0u; elem < 4u; ++elem) {\n                    values[elem] = q[\n                        source + int64_t(elem) * q_strides[3]];\n                }\n            }\n            for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {\n                left[row_part * ELEMENT_COLUMNS + elem] = values[elem];\n            }\n        }\n\n        for (short key_half = 0; key_half < 2; ++key_half) {\n            for (short row_part = 0; row_part < 2; ++row_part) {\n                const uint key_local = uint(\n                    key_half * 16 + coordinate.y +\n                    row_part * ELEMENT_ROW_JUMP);\n                const uint source =\n                    key_local * QSA_SCORE_HEAD_DIM +\n                    k_frag * 16u + uint(coordinate.x);\n                const threadgroup vec<InT, 4>* source4 =\n                    reinterpret_cast<const threadgroup vec<InT, 4>*>(\n                        pooled_tile + source);\n                const vec<InT, 4> values = source4[0];\n                for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {\n                    right[\n                        key_half * ELEMENTS_PER_FRAGMENT +\n                        row_part * ELEMENT_COLUMNS + elem] = values[elem];\n                }\n            }\n        }\n        matmul.run(left, right, accumulator);\n    }\n\n    for (short key_half = 0; key_half < 2; ++key_half) {\n        for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {\n            const float head0_or_1 = metal::max(\n                accumulator[key_half * ELEMENTS_PER_FRAGMENT + elem], 0.0f);\n            const float head2_or_3 = metal::max(\n                accumulator[\n                    key_half * ELEMENTS_PER_FRAGMENT +\n                    ELEMENT_COLUMNS + elem],\n                0.0f);\n            const float paired_head1_or_0 =\n                simd_shuffle_xor(head0_or_1, ushort(16));\n            const float paired_head3_or_2 =\n                simd_shuffle_xor(head2_or_3, ushort(16));\n            if ((lane & 16u) == 0u) {\n                const uint query = query0 + uint(coordinate.y);\n                const uint block =\n                    block0 + uint(key_half * 16 + coordinate.x + elem);\n                if (query < rows && block < blocks) {\n                    const float head_sum =\n                        ((head0_or_1 + paired_head1_or_0) + head2_or_3) +\n                        paired_head3_or_2;\n                    scores[size_t(query) * blocks + block] =\n                        head_sum / QSA_SCORE_SQRT_HEAD_DIM;\n                }\n            }\n        }\n    }\n"

_MPP_KEYS_STATIONARY_SOURCE = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint simdgroup = tid >> 5;
    const ushort lane = ushort(tid & 31u);
    const uint rows = q_shape[1];
    const uint blocks = pooled_shape[1];
    const uint key_tile = threadgroup_position_in_grid.x;
    const uint block0 = key_tile * QSA_SCORE_KEY_TILE;

    // Load each pooled-key tile once, then retain it while all query tiles
    // pass through the workgroup.
    threadgroup InT pooled_tile[QSA_SCORE_KEY_TILE * QSA_SCORE_HEAD_DIM];
    constexpr uint VECTORS_PER_KEY = QSA_SCORE_HEAD_DIM / 4u;
    constexpr uint TILE_VECTORS = QSA_SCORE_KEY_TILE * VECTORS_PER_KEY;
    for (uint item = tid; item < TILE_VECTORS; item += QSA_SCORE_THREADS) {
        const uint key_local = item / VECTORS_PER_KEY;
        const uint dim4 = item - key_local * VECTORS_PER_KEY;
        const uint block = block0 + key_local;
        vec<InT, 4> values = vec<InT, 4>(InT(0));
        if (block < blocks) {
            const int64_t source =
                int64_t(block) * pooled_strides[1] +
                int64_t(dim4 * 4u) * pooled_strides[2];
            for (uint elem = 0u; elem < 4u; ++elem) {
                values[elem] = pooled[
                    source + int64_t(elem) * pooled_strides[2]];
            }
        }
        const uint destination =
            key_local * QSA_SCORE_HEAD_DIM + dim4 * 4u;
        for (uint elem = 0u; elem < 4u; ++elem) {
            pooled_tile[destination + elem] = values[elem];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> matmul;
    auto left = matmul.get_left_input_cooperative_tensor<InT, InT, float>();
    auto right = matmul.get_right_input_cooperative_tensor<InT, InT, float>();
    auto accumulator = matmul.get_destination_cooperative_tensor<
        decltype(left), decltype(right), float>();

    constexpr short ELEMENTS_PER_FRAGMENT = 8;
    constexpr short ELEMENT_COLUMNS = 4;
    constexpr short ELEMENT_ROW_JUMP = 8;
    const short2 coordinate = qsa_score_nax_coord(lane);
    const uint query_tiles =
        (rows + QSA_SCORE_QUERY_TILE - 1u) / QSA_SCORE_QUERY_TILE;
    for (uint query_tile = 0u; query_tile < query_tiles; ++query_tile) {
        const uint query0 =
            query_tile * QSA_SCORE_QUERY_TILE +
            simdgroup * QSA_SCORE_QUERIES_PER_SIMDGROUP;
        for (short item = 0; item < 2 * ELEMENTS_PER_FRAGMENT; ++item) {
            accumulator[item] = 0.0f;
        }

        for (uint k_frag = 0u; k_frag < QSA_SCORE_K_FRAGMENTS; ++k_frag) {
            for (short row_part = 0; row_part < 2; ++row_part) {
                const uint matrix_row =
                    uint(coordinate.y + row_part * ELEMENT_ROW_JUMP);
                const uint head =
                    matrix_row / QSA_SCORE_QUERIES_PER_SIMDGROUP;
                const uint query_local =
                    matrix_row - head * QSA_SCORE_QUERIES_PER_SIMDGROUP;
                const uint query = query0 + query_local;
                vec<InT, 4> values = vec<InT, 4>(InT(0));
                if (query < rows) {
                    const int64_t source =
                        int64_t(query) * q_strides[1] +
                        int64_t(head) * q_strides[2] +
                        int64_t(k_frag * 16u + uint(coordinate.x)) *
                            q_strides[3];
                    for (uint elem = 0u; elem < 4u; ++elem) {
                        values[elem] = q[
                            source + int64_t(elem) * q_strides[3]];
                    }
                }
                for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                    left[row_part * ELEMENT_COLUMNS + elem] = values[elem];
                }
            }

            for (short key_half = 0; key_half < 2; ++key_half) {
                for (short row_part = 0; row_part < 2; ++row_part) {
                    const uint key_local = uint(
                        key_half * 16 + coordinate.y +
                        row_part * ELEMENT_ROW_JUMP);
                    const uint source =
                        key_local * QSA_SCORE_HEAD_DIM +
                        k_frag * 16u + uint(coordinate.x);
                    const threadgroup vec<InT, 4>* source4 =
                        reinterpret_cast<const threadgroup vec<InT, 4>*>(
                            pooled_tile + source);
                    const vec<InT, 4> values = source4[0];
                    for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                        right[
                            key_half * ELEMENTS_PER_FRAGMENT +
                            row_part * ELEMENT_COLUMNS + elem] = values[elem];
                    }
                }
            }
            matmul.run(left, right, accumulator);
        }

        for (short key_half = 0; key_half < 2; ++key_half) {
            for (short elem = 0; elem < ELEMENT_COLUMNS; ++elem) {
                const float head0_or_1 = metal::max(
                    accumulator[key_half * ELEMENTS_PER_FRAGMENT + elem], 0.0f);
                const float head2_or_3 = metal::max(
                    accumulator[
                        key_half * ELEMENTS_PER_FRAGMENT +
                        ELEMENT_COLUMNS + elem],
                    0.0f);
                const float paired_head1_or_0 =
                    simd_shuffle_xor(head0_or_1, ushort(16));
                const float paired_head3_or_2 =
                    simd_shuffle_xor(head2_or_3, ushort(16));
                if ((lane & 16u) == 0u) {
                    const uint query = query0 + uint(coordinate.y);
                    const uint block =
                        block0 + uint(key_half * 16 + coordinate.x + elem);
                    if (query < rows && block < blocks) {
                        const float head_sum =
                            ((head0_or_1 + paired_head1_or_0) + head2_or_3) +
                            paired_head3_or_2;
                        scores[size_t(query) * blocks + block] =
                            head_sum / QSA_SCORE_SQRT_HEAD_DIM;
                    }
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _mpp_score_kernel():
    return mx.fast.metal_kernel(
        name="mlx_lm_qwen4_qsa_stage1_mpp_h4d128",
        input_names=["q", "pooled"],
        output_names=["scores"],
        header=_MPP_SCORE_HEADER,
        source=_MPP_SCORE_SOURCE,
        ensure_row_contiguous=False,
    )


@lru_cache(maxsize=1)
def _mpp_keys_stationary_kernel():
    return mx.fast.metal_kernel(
        name="mlx_lm_qwen4_qsa_stage1_mpp_keys_stationary_h4d128",
        input_names=["q", "pooled"],
        output_names=["scores"],
        header=_MPP_SCORE_HEADER,
        source=_MPP_KEYS_STATIONARY_SOURCE,
        ensure_row_contiguous=False,
    )


def _mpp_scores(q: mx.array, pooled: mx.array) -> mx.array:
    rows = int(q.shape[1])
    blocks = int(pooled.shape[1])
    query_tiles = (rows + 15) // 16
    key_tiles = (blocks + 31) // 32
    (scores,) = _mpp_score_kernel()(
        inputs=[q, pooled],
        template=[("InT", q.dtype)],
        grid=(query_tiles * key_tiles * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    return scores


def _mpp_keys_stationary_scores(q: mx.array, pooled: mx.array) -> mx.array:
    rows = int(q.shape[1])
    blocks = int(pooled.shape[1])
    key_tiles = (blocks + 31) // 32
    (scores,) = _mpp_keys_stationary_kernel()(
        inputs=[q, pooled],
        template=[("InT", q.dtype)],
        grid=(key_tiles * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, blocks)],
        output_dtypes=[mx.float32],
    )
    _record_candidate_dispatch("keys_stationary_dispatches")
    return scores


_HEADER = "\n#include <metal_stdlib>\nusing namespace metal;\n\ninline uint qsa_float_order_key(float value) {\n    const uint bits = as_type<uint>(value);\n    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);\n}\n\ninline ulong qsa_composite_key(float score, uint block_id) {\n    return (ulong(qsa_float_order_key(score)) << 32) | ulong(block_id);\n}\n\ninline bool qsa_id_before(\n    uint a_index, bool a_valid, uint b_index, bool b_valid) {\n    if (a_valid != b_valid) {\n        return a_valid;\n    }\n    return a_index < b_index;\n}\n"


@lru_cache(maxsize=16)
def _onepass_select_kernel(topk: int, ratio: int):
    width = 1 << (max(256, topk) - 1).bit_length()
    if width > 1024:
        raise ValueError(f"QSA one-pass threadgroup width {width} is unsupported")
    header = (
        _HEADER
        + f"\nconstant constexpr uint TOP_K = {topk};\n"
        + f"constant constexpr uint RATIO = {ratio};\n"
        + f"constant constexpr uint WIDTH = {width};\n"
        + "constant constexpr uint HISTOGRAM_BINS = 4096;\n"
        + "constant constexpr uint CANDIDATE_CAP = 1024;\n"
    )
    source = r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint blocks = uint(dims[0]);
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t scratch_base = size_t(row) * blocks;

        threadgroup atomic_uint histogram[HISTOGRAM_BINS];
        threadgroup uint candidate_keys[CANDIDATE_CAP];
        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup uchar equal_flags[WIDTH];
        threadgroup atomic_uint candidate_count;
        threadgroup atomic_uint selected_count;
        threadgroup uint selected_bin;
        threadgroup uint selected_prefix;
        threadgroup uint selected_mask;
        threadgroup uint selected_rank;
        threadgroup uint selected_bucket_size;

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0u;
        equal_flags[lane] = 0u;
        if (lane == 0u) {
            selected_bin = 0u;
            selected_prefix = 0u;
            selected_mask = 0u;
            selected_rank = selected_valid > 0u ? selected_valid - 1u : 0u;
            selected_bucket_size = 0u;
            atomic_store_explicit(&candidate_count, 0u, memory_order_relaxed);
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        for (uint bin = lane; bin < HISTOGRAM_BINS; bin += WIDTH) {
            atomic_store_explicit(
                &histogram[bin], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            // First row pass: use the top 12 score bits to locate the bucket
            // containing the k-th largest value.
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const uint key = qsa_float_order_key(
                    score_scratch[scratch_base + block]);
                atomic_fetch_add_explicit(
                    &histogram[key >> 20u], 1u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0u) {
                uint rank = selected_rank;
                for (int bin = 4095; bin >= 0; --bin) {
                    const uint count = atomic_load_explicit(
                        &histogram[uint(bin)], memory_order_relaxed);
                    if (rank < count) {
                        selected_bin = uint(bin);
                        selected_rank = rank;
                        selected_bucket_size = count;
                        selected_prefix = uint(bin) << 20u;
                        selected_mask = 0xfff00000u;
                        break;
                    }
                    rank -= count;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Second row pass: compact the threshold bucket when it fits.
            // Oversized buckets fail closed to row rescans below.
            const bool fits = selected_bucket_size <= CANDIDATE_CAP;
            if (fits) {
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const uint key = qsa_float_order_key(
                        score_scratch[scratch_base + block]);
                    if ((key >> 20u) == selected_bin) {
                        const uint slot = atomic_fetch_add_explicit(
                            &candidate_count, 1u, memory_order_relaxed);
                        if (slot < CANDIDATE_CAP) {
                            candidate_keys[slot] = key;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Resolve the remaining 20 score bits in two 10-bit passes. A
            // compacted bucket stays in threadgroup memory; an overflow
            // rescans only the matching prefix from the source row.
            for (int shift = 10; shift >= 0; shift -= 10) {
                for (uint bin = lane; bin < 1024u; bin += WIDTH) {
                    atomic_store_explicit(
                        &histogram[bin], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint prefix = selected_prefix;
                const uint mask = selected_mask;
                if (fits) {
                    const uint count = atomic_load_explicit(
                        &candidate_count, memory_order_relaxed);
                    for (uint item = lane; item < count; item += WIDTH) {
                        const uint key = candidate_keys[item];
                        if ((key & mask) == prefix) {
                            atomic_fetch_add_explicit(
                                &histogram[(key >> uint(shift)) & 1023u],
                                1u,
                                memory_order_relaxed);
                        }
                    }
                } else {
                    for (uint block = lane; block < valid_count; block += WIDTH) {
                        const uint key = qsa_float_order_key(
                            score_scratch[scratch_base + block]);
                        if ((key & mask) == prefix) {
                            atomic_fetch_add_explicit(
                                &histogram[(key >> uint(shift)) & 1023u],
                                1u,
                                memory_order_relaxed);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) {
                    uint rank = selected_rank;
                    for (int bin = 1023; bin >= 0; --bin) {
                        const uint count = atomic_load_explicit(
                            &histogram[uint(bin)], memory_order_relaxed);
                        if (rank < count) {
                            selected_prefix |= uint(bin) << uint(shift);
                            selected_mask |= 1023u << uint(shift);
                            selected_rank = rank;
                            break;
                        }
                        rank -= count;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint threshold = selected_prefix;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                const uint key = qsa_float_order_key(
                    score_scratch[scratch_base + block]);
                if (key > threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1u;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The reference composite key breaks score ties by larger block
            // ID. Walk chunks from the row tail so the same tied IDs win.
            for (uint chunk_end = valid_count; chunk_end > 0u;) {
                const uint chunk_start =
                    chunk_end > WIDTH ? chunk_end - WIDTH : 0u;
                const uint block = chunk_start + lane;
                equal_flags[lane] = (
                    block < chunk_end &&
                    qsa_float_order_key(score_scratch[scratch_base + block]) ==
                        threshold
                ) ? 1u : 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane == 0u) {
                    uint count = atomic_load_explicit(
                        &selected_count, memory_order_relaxed);
                    const uint chunk_width = chunk_end - chunk_start;
                    for (int offset = int(chunk_width) - 1;
                         offset >= 0 && count < selected_valid;
                         --offset) {
                        if (equal_flags[uint(offset)] != 0u) {
                            exchange_indices[count] =
                                chunk_start + uint(offset);
                            exchange_valid[count] = 1u;
                            count += 1u;
                        }
                    }
                    atomic_store_explicit(
                        &selected_count, count, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                chunk_end = chunk_start;
            }
        }

        // Canonicalize the selected set exactly as the reference selector.
        uint my_index = exchange_indices[lane];
        bool my_valid = exchange_valid[lane] != 0u;
        for (uint sequence = 2u; sequence <= WIDTH; sequence <<= 1u) {
            for (uint stride = sequence >> 1u; stride > 0u; stride >>= 1u) {
                exchange_indices[lane] = my_index;
                exchange_valid[lane] = my_valid ? 1u : 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint partner = lane ^ stride;
                const uint other_index = exchange_indices[partner];
                const bool other_valid = exchange_valid[partner] != 0u;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool is_lower = (lane & stride) == 0u;
                const uint a_index = is_lower ? my_index : other_index;
                const bool a_valid = is_lower ? my_valid : other_valid;
                const uint b_index = is_lower ? other_index : my_index;
                const bool b_valid = is_lower ? other_valid : my_valid;
                const bool lower_wants_before = (lane & sequence) == 0u;
                const bool swap = lower_wants_before
                    ? qsa_id_before(b_index, b_valid, a_index, a_valid)
                    : qsa_id_before(a_index, a_valid, b_index, b_valid);
                if (swap) {
                    my_index = is_lower ? b_index : a_index;
                    my_valid = is_lower ? b_valid : a_valid;
                }
            }
        }

        if (lane < TOP_K) {
            uint output_id = my_index;
            if (lane >= selected_valid) {
                const uint invalid_count = TOP_K - selected_valid;
                output_id = blocks - invalid_count + (lane - selected_valid);
            }
            block_ids[size_t(row) * TOP_K + lane] = output_id;
        }
    """
    return mx.fast.metal_kernel(
        name=f"mlx_lm_qwen4_qsa_stage1_onepass_k{topk}_r{ratio}",
        input_names=["score_scratch", "q_positions", "dims"],
        output_names=["block_ids"],
        header=header,
        source=source,
    )


def _select_scores_onepass(
    scores: mx.array, q_positions: mx.array, *, topk: int, compress_ratio: int
) -> mx.array:
    (rows, blocks) = map(int, scores.shape)
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    kernel = _onepass_select_kernel(int(topk), int(compress_ratio))
    (selected,) = kernel(
        inputs=[scores, q_positions, mx.array([blocks], dtype=mx.int32)],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, int(topk))],
        output_dtypes=[mx.uint32],
    )
    _record_candidate_dispatch("onepass_topk_dispatches")
    return selected


@lru_cache(maxsize=32)
def _stage1_kernel(
    heads: int,
    head_dim: int,
    topk: int,
    ratio: int,
    q_dtype: mx.Dtype,
    pooled_dtype: mx.Dtype,
):
    width = 1 << (max(256, topk) - 1).bit_length()
    if width > 1024:
        raise ValueError(f"QSA stage-one threadgroup width {width} is unsupported")
    header = (
        _HEADER
        + f"\nconstant constexpr uint HEADS = {heads};\nconstant constexpr uint HEAD_DIM = {head_dim};\nconstant constexpr uint TOP_K = {topk};\nconstant constexpr uint RATIO = {ratio};\nconstant constexpr uint WIDTH = {width};\nconstant constexpr uint RADIX_BINS = 256;\nconstant constexpr float SQRT_HEAD_DIM = {math.sqrt(head_dim)!r}f;\n"
    )
    source = "\n        const uint row = threadgroup_position_in_grid.x;\n        const uint lane = thread_position_in_threadgroup.x;\n        const uint blocks = uint(dims[0]);\n        const uint query_rows = uint(dims[1]);\n        const uint batch = row / query_rows;\n        const int qpos = int(q_positions[row]);\n        const int complete_value = (qpos + 1) / int(RATIO);\n        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;\n        const uint valid_count = metal::min(blocks, complete);\n        const uint selected_valid = metal::min(TOP_K, valid_count);\n        const size_t scratch_base = size_t(row) * blocks;\n\n        threadgroup uint exchange_indices[WIDTH];\n        threadgroup uchar exchange_valid[WIDTH];\n        threadgroup atomic_uint radix_histogram[RADIX_BINS];\n        threadgroup atomic_uint selected_count;\n        threadgroup ulong radix_prefix;\n        threadgroup uint radix_rank;\n        threadgroup ulong threshold_key;\n\n        for (uint block = lane; block < valid_count; block += WIDTH) {\n            float score_sum = 0.0f;\n            for (uint head = 0; head < HEADS; ++head) {\n                float dot = 0.0f;\n                const size_t q_base =\n                    (size_t(row) * HEADS + head) * HEAD_DIM;\n                const size_t pooled_base =\n                    (size_t(batch) * blocks + block) * HEAD_DIM;\n                for (uint dim = 0; dim < HEAD_DIM; ++dim) {\n                    dot += float(q[q_base + dim]) *\n                           float(pooled[pooled_base + dim]);\n                }\n                score_sum += metal::max(dot, 0.0f);\n            }\n            score_scratch[scratch_base + block] = score_sum / SQRT_HEAD_DIM;\n        }\n        threadgroup_barrier(\n            mem_flags::mem_threadgroup | mem_flags::mem_device);\n\n        if (lane == 0) {\n            radix_prefix = 0ul;\n            radix_rank = selected_valid > 0 ? selected_valid - 1 : 0;\n            threshold_key = 0xfffffffffffffffful;\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        if (selected_valid > 0) {\n            for (uint pass = 0; pass < 8; ++pass) {\n                if (lane < RADIX_BINS) {\n                    atomic_store_explicit(\n                        &radix_histogram[lane], 0u, memory_order_relaxed);\n                }\n                threadgroup_barrier(mem_flags::mem_threadgroup);\n\n                const uint shift = 56u - pass * 8u;\n                const ulong prefix = radix_prefix;\n                for (uint block = lane; block < valid_count; block += WIDTH) {\n                    const float score = score_scratch[scratch_base + block];\n                    const ulong key = qsa_composite_key(score, block);\n                    const bool prefix_matches = pass == 0 ||\n                        (key >> (shift + 8u)) == prefix;\n                    if (prefix_matches) {\n                        atomic_fetch_add_explicit(\n                            &radix_histogram[uint((key >> shift) & 0xfful)],\n                            1u,\n                            memory_order_relaxed);\n                    }\n                }\n                threadgroup_barrier(mem_flags::mem_threadgroup);\n\n                if (lane == 0) {\n                    uint rank = radix_rank;\n                    uint chosen = 0;\n                    for (int digit = 255; digit >= 0; --digit) {\n                        const uint count = atomic_load_explicit(\n                            &radix_histogram[uint(digit)], memory_order_relaxed);\n                        if (rank < count) {\n                            chosen = uint(digit);\n                            break;\n                        }\n                        rank -= count;\n                    }\n                    radix_prefix = (radix_prefix << 8) | ulong(chosen);\n                    radix_rank = rank;\n                }\n                threadgroup_barrier(mem_flags::mem_threadgroup);\n            }\n            if (lane == 0) {\n                threshold_key = radix_prefix;\n            }\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        exchange_indices[lane] = 0xffffffffu;\n        exchange_valid[lane] = 0;\n        if (lane == 0) {\n            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        if (selected_valid > 0) {\n            const ulong threshold = threshold_key;\n            for (uint block = lane; block < valid_count; block += WIDTH) {\n                const float score = score_scratch[scratch_base + block];\n                if (qsa_composite_key(score, block) >= threshold) {\n                    const uint slot = atomic_fetch_add_explicit(\n                        &selected_count, 1u, memory_order_relaxed);\n                    if (slot < TOP_K) {\n                        exchange_indices[slot] = block;\n                        exchange_valid[slot] = 1;\n                    }\n                }\n            }\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        uint my_index = exchange_indices[lane];\n        bool my_valid = exchange_valid[lane] != 0;\n        for (uint sequence = 2; sequence <= WIDTH; sequence <<= 1) {\n            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {\n                exchange_indices[lane] = my_index;\n                exchange_valid[lane] = my_valid ? 1 : 0;\n                threadgroup_barrier(mem_flags::mem_threadgroup);\n\n                const uint partner = lane ^ stride;\n                const uint other_index = exchange_indices[partner];\n                const bool other_valid = exchange_valid[partner] != 0;\n                threadgroup_barrier(mem_flags::mem_threadgroup);\n\n                const bool is_lower = (lane & stride) == 0;\n                const uint a_index = is_lower ? my_index : other_index;\n                const bool a_valid = is_lower ? my_valid : other_valid;\n                const uint b_index = is_lower ? other_index : my_index;\n                const bool b_valid = is_lower ? other_valid : my_valid;\n                const bool lower_wants_before = (lane & sequence) == 0;\n                const bool b_before_a = qsa_id_before(\n                    b_index, b_valid, a_index, a_valid);\n                const bool a_before_b = qsa_id_before(\n                    a_index, a_valid, b_index, b_valid);\n                const bool swap = lower_wants_before ? b_before_a : a_before_b;\n                if (swap) {\n                    my_index = is_lower ? b_index : a_index;\n                    my_valid = is_lower ? b_valid : a_valid;\n                }\n            }\n        }\n\n        if (lane < TOP_K) {\n            uint output_id = my_index;\n            if (lane >= selected_valid) {\n                const uint invalid_count = TOP_K - selected_valid;\n                output_id = blocks - invalid_count + (lane - selected_valid);\n            }\n            block_ids[size_t(row) * TOP_K + lane] = output_id;\n        }\n    "
    return mx.fast.metal_kernel(
        name=f"mlx_lm_qwen4_qsa_stage1_h{heads}_d{head_dim}_k{topk}_r{ratio}_{str(q_dtype).replace('.', '_')}_{str(pooled_dtype).replace('.', '_')}",
        input_names=["q", "pooled", "q_positions", "dims"],
        output_names=["block_ids", "score_scratch"],
        header=header,
        source=source,
    )


def _select_scores(
    scores: mx.array, q_positions: mx.array, *, topk: int, compress_ratio: int
) -> mx.array:
    """Select score-column IDs with the configured exact selector."""
    (rows, blocks) = map(int, scores.shape)
    if qsa_stage1_selector_producer(blocks=blocks, block_topk=topk) == (
        "onepass_exact"
    ):
        return _select_scores_onepass(
            scores,
            q_positions,
            topk=topk,
            compress_ratio=compress_ratio,
        )
    score_q = mx.ones((rows, 1, 1, 1), dtype=mx.float32)
    score_keys = scores.reshape(rows, blocks, 1)
    score_positions = q_positions.reshape(rows, 1)
    kernel = _stage1_kernel(
        1, 1, int(topk), int(compress_ratio), mx.float32, mx.float32
    )
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    outputs = kernel(
        inputs=[
            score_q,
            score_keys,
            score_positions.astype(mx.int32),
            mx.array([blocks, 1], dtype=mx.int32),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, 1, int(topk)), (rows, blocks)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return outputs[0].reshape(rows, int(topk))


def qsa_stage1_select(
    q: mx.array,
    pooled: mx.array,
    q_positions: mx.array,
    *,
    block_topk: int,
    compress_ratio: int,
) -> mx.array:
    """Return deterministic selected block IDs with shape ``[B,L,K]``."""
    if not qsa_stage1_supported(
        q, pooled, q_positions, block_topk=block_topk, compress_ratio=compress_ratio
    ):
        raise ValueError("unsupported QSA stage-one kernel geometry")
    (batch, length, _, _) = map(int, q.shape)
    blocks = int(pooled.shape[1])
    topk = int(block_topk)
    rows = batch * length
    positions = q_positions.reshape(rows)
    producer = qsa_stage1_route(q, pooled, block_topk=topk)["score_producer"]
    if producer in {"mpp_exact_band", "mpp_keys_stationary_exact_band"}:
        approximate = (
            _mpp_keys_stationary_scores(q, pooled)
            if producer == "mpp_keys_stationary_exact_band"
            else _mpp_scores(q, pooled)
        )
        candidate_count = topk + _EXACT_BAND_EXTRA
        candidate_ids = _select_scores(
            approximate, positions, topk=candidate_count, compress_ratio=compress_ratio
        )
        candidate_keys = mx.take(pooled[0], candidate_ids, axis=0)
        exact = mx.einsum(
            "lhd,lcd->lch", q[0].astype(mx.float32), candidate_keys.astype(mx.float32)
        )
        exact = mx.sum(mx.maximum(exact, 0), axis=-1) / math.sqrt(q.shape[-1])
        candidate_valid = (
            candidate_ids.astype(mx.int32) * int(compress_ratio)
            + int(compress_ratio)
            - 1
            <= positions[:, None]
        )
        exact = mx.where(candidate_valid, exact, -mx.inf)
        all_candidate_positions = mx.full(
            (rows,), candidate_count * int(compress_ratio) - 1, dtype=mx.int32
        )
        selected_slots = _select_scores(
            exact, all_candidate_positions, topk=topk, compress_ratio=compress_ratio
        )
        selected = mx.take_along_axis(candidate_ids, selected_slots, axis=-1)
    else:
        scores = mx.einsum(
            "blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32)
        )
        scores = mx.sum(mx.maximum(scores, 0), axis=-1) / math.sqrt(q.shape[-1])
        selected = _select_scores(
            scores.reshape(rows, blocks),
            positions,
            topk=topk,
            compress_ratio=compress_ratio,
        )
    return selected.reshape(batch, length, topk)
