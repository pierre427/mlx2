# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Exact direct-input selectors for Qwen4 QSA stage-one scores.

These kernels are mechanism candidates.  Policy and route selection remain in
``qwen4_qsa_stage1`` so importing this module never changes a runtime default.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

inline uint qsa_direct_float_order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

inline ulong qsa_direct_composite_key(float score, uint block_id) {
    return (ulong(qsa_direct_float_order_key(score)) << 32) | ulong(block_id);
}

inline bool qsa_direct_id_before(
    uint a_index, bool a_valid, uint b_index, bool b_valid) {
    if (a_valid != b_valid) {
        return a_valid;
    }
    return a_index < b_index;
}
"""


_SORT_AND_STORE = r"""
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
                ? qsa_direct_id_before(b_index, b_valid, a_index, a_valid)
                : qsa_direct_id_before(a_index, a_valid, b_index, b_valid);
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


@lru_cache(maxsize=8)
def _direct8_kernel(topk: int, ratio: int):
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    if width > 1024:
        raise ValueError(f"QSA direct8 threadgroup width {width} is unsupported")
    header = (
        _HEADER
        + f"\nconstant constexpr uint TOP_K = {topk};\n"
        + f"constant constexpr uint RATIO = {ratio};\n"
        + f"constant constexpr uint WIDTH = {width};\n"
        + "constant constexpr uint RADIX_BINS = 256;\n"
    )
    source = (
        r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint blocks = uint(dims[0]);
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t score_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup atomic_uint selected_count;
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;
        threadgroup ulong threshold_key;

        if (lane == 0u) {
            radix_prefix = 0ul;
            radix_rank = selected_valid > 0u ? selected_valid - 1u : 0u;
            threshold_key = 0xfffffffffffffffful;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            for (uint pass = 0u; pass < 8u; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 56u - pass * 8u;
                const ulong prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const ulong key = qsa_direct_composite_key(
                        scores[score_base + block], block);
                    bool prefix_matches = true;
                    if (pass > 0u) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[uint((key >> shift) & 0xfful)],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0u) {
                    uint rank = radix_rank;
                    uint chosen = 0u;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8u) | ulong(chosen);
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0u;
        if (lane == 0u) {
            atomic_store_explicit(&selected_count, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            const ulong threshold = threshold_key;
            for (uint block = lane; block < valid_count; block += WIDTH) {
                if (qsa_direct_composite_key(
                        scores[score_base + block], block) >= threshold) {
                    const uint slot = atomic_fetch_add_explicit(
                        &selected_count, 1u, memory_order_relaxed);
                    if (slot < TOP_K) {
                        exchange_indices[slot] = block;
                        exchange_valid[slot] = 1u;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    """
        + _SORT_AND_STORE
    )
    return mx.fast.metal_kernel(
        name=f"mlx2_qwen4_qsa_direct8_k{topk}_r{ratio}",
        input_names=["scores", "q_positions", "dims"],
        output_names=["block_ids"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=8)
def _direct4_kernel(topk: int, ratio: int):
    width = 1024
    if not 1 <= int(topk) <= width:
        raise ValueError(f"QSA direct4 top-k {topk} is unsupported")
    header = (
        _HEADER
        + f"\nconstant constexpr uint TOP_K = {topk};\n"
        + f"constant constexpr uint RATIO = {ratio};\n"
        + f"constant constexpr uint WIDTH = {width};\n"
        + "constant constexpr uint RADIX_BINS = 256;\n"
        + "constant constexpr uint SIMD_GROUPS = WIDTH / 32;\n"
    )
    source = (
        r"""
        const uint row = threadgroup_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint simd_id = lane / 32u;
        const uint simd_lane = lane % 32u;
        const uint blocks = uint(dims[0]);
        const int qpos = int(q_positions[row]);
        const int complete_value = (qpos + 1) / int(RATIO);
        const uint complete = complete_value > 0 ? uint(complete_value) : 0u;
        const uint valid_count = metal::min(blocks, complete);
        const uint selected_valid = metal::min(TOP_K, valid_count);
        const size_t score_base = size_t(row) * blocks;

        threadgroup uint exchange_indices[WIDTH];
        threadgroup uchar exchange_valid[WIDTH];
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup uint partial_greater[SIMD_GROUPS];
        threadgroup uint partial_ties[SIMD_GROUPS];
        threadgroup uint radix_prefix;
        threadgroup uint radix_rank;
        threadgroup uint threshold_key;
        threadgroup uint total_greater;

        exchange_indices[lane] = 0xffffffffu;
        exchange_valid[lane] = 0u;
        if (lane == 0u) {
            radix_prefix = 0u;
            radix_rank = selected_valid > 0u ? selected_valid - 1u : 0u;
            threshold_key = 0xffffffffu;
            total_greater = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (selected_valid > 0u) {
            for (uint pass = 0u; pass < 4u; ++pass) {
                if (lane < RADIX_BINS) {
                    atomic_store_explicit(
                        &radix_histogram[lane], 0u, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint shift = 24u - pass * 8u;
                const uint prefix = radix_prefix;
                for (uint block = lane; block < valid_count; block += WIDTH) {
                    const uint key = qsa_direct_float_order_key(
                        scores[score_base + block]);
                    bool prefix_matches = true;
                    if (pass > 0u) {
                        prefix_matches = (key >> (shift + 8u)) == prefix;
                    }
                    if (prefix_matches) {
                        atomic_fetch_add_explicit(
                            &radix_histogram[(key >> shift) & 0xffu],
                            1u,
                            memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane == 0u) {
                    uint rank = radix_rank;
                    uint chosen = 0u;
                    for (int digit = 255; digit >= 0; --digit) {
                        const uint count = atomic_load_explicit(
                            &radix_histogram[uint(digit)], memory_order_relaxed);
                        if (rank < count) {
                            chosen = uint(digit);
                            break;
                        }
                        rank -= count;
                    }
                    radix_prefix = (radix_prefix << 8u) | chosen;
                    radix_rank = rank;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lane == 0u) {
                threshold_key = radix_prefix;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reverse-ID raking preserves the selected composite-key tie rule:
        // larger block IDs win at the score threshold.
        const uint segment = (valid_count + WIDTH - 1u) / WIDTH;
        const uint reverse_start = lane * segment;
        const uint reverse_end = metal::min(reverse_start + segment, valid_count);
        uint local_greater = 0u;
        uint local_ties = 0u;
        for (uint reverse = reverse_start; reverse < reverse_end; ++reverse) {
            const uint block = valid_count - 1u - reverse;
            const uint key = qsa_direct_float_order_key(
                scores[score_base + block]);
            local_greater += key > threshold_key ? 1u : 0u;
            local_ties += key == threshold_key ? 1u : 0u;
        }

        uint prefix_greater = metal::simd_prefix_exclusive_sum(local_greater);
        uint prefix_ties = metal::simd_prefix_exclusive_sum(local_ties);
        if (simd_lane == 31u) {
            partial_greater[simd_id] = prefix_greater + local_greater;
            partial_ties[simd_id] = prefix_ties + local_ties;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_id == 0u) {
            const uint group_greater = partial_greater[simd_lane];
            const uint group_ties = partial_ties[simd_lane];
            const uint group_prefix_greater =
                metal::simd_prefix_exclusive_sum(group_greater);
            const uint group_prefix_ties =
                metal::simd_prefix_exclusive_sum(group_ties);
            partial_greater[simd_lane] = group_prefix_greater;
            partial_ties[simd_lane] = group_prefix_ties;
            if (simd_lane == 31u) {
                total_greater = group_prefix_greater + group_greater;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint output_greater = partial_greater[simd_id] + prefix_greater;
        uint output_ties = total_greater + partial_ties[simd_id] + prefix_ties;
        if (local_greater > 0u ||
            (local_ties > 0u && output_ties < selected_valid)) {
            for (uint reverse = reverse_start; reverse < reverse_end; ++reverse) {
                const uint block = valid_count - 1u - reverse;
                const uint key = qsa_direct_float_order_key(
                    scores[score_base + block]);
                if (key > threshold_key) {
                    if (output_greater < selected_valid) {
                        exchange_indices[output_greater] = block;
                        exchange_valid[output_greater] = 1u;
                    }
                    output_greater += 1u;
                } else if (key == threshold_key) {
                    if (output_ties < selected_valid) {
                        exchange_indices[output_ties] = block;
                        exchange_valid[output_ties] = 1u;
                    }
                    output_ties += 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    """
        + _SORT_AND_STORE
    )
    return mx.fast.metal_kernel(
        name=f"mlx2_qwen4_qsa_direct4_k{topk}_r{ratio}",
        input_names=["scores", "q_positions", "dims"],
        output_names=["block_ids"],
        header=header,
        source=source,
    )


def _invoke(
    kernel,
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    width: int,
) -> mx.array:
    rows, blocks = map(int, scores.shape)
    (selected,) = kernel(
        inputs=[
            scores,
            q_positions.astype(mx.int32),
            mx.array([blocks], dtype=mx.int32),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(width, 1, 1),
        output_shapes=[(rows, int(topk))],
        output_dtypes=[mx.uint32],
    )
    return selected


def select_scores_direct8(
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    compress_ratio: int,
) -> mx.array:
    """Run the selected eight-pass composite ordering without a score copy."""
    width = 1 << (max(256, int(topk)) - 1).bit_length()
    return _invoke(
        _direct8_kernel(int(topk), int(compress_ratio)),
        scores,
        q_positions,
        topk=int(topk),
        width=width,
    )


def select_scores_direct4(
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    compress_ratio: int,
) -> mx.array:
    """Run four score passes with deterministic exact threshold tie fill."""
    return _invoke(
        _direct4_kernel(int(topk), int(compress_ratio)),
        scores,
        q_positions,
        topk=int(topk),
        width=1024,
    )
