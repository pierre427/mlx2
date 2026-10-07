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


# --------------------------------------------------------------------------
# GVR-style exact self-sampling multi-threshold selector (research candidate).
#
# Design input: NVIDIA TensorRT-LLM PRs #18702 and #19076 and the "GVR V2:
# Self-Sampling and Multi-Thresholding for Faster Exact Top-K" article
# (Apache-2.0).  No source was copied.  ``qwen4_qsa_gvr_reference`` is the
# step-for-step NumPy mirror; the kernel's per-row diagnostics must equal it.
#
# Per row: (1) sample WIDTH keys at stride valid_count/WIDTH and sort them,
# (2) one coalesced pass counts keys >= four sample-quantile thresholds,
# (3) the highest threshold whose count reaches k and fits CAP is gathered in
# ascending block order, (4) the k-th float key among candidates is found by a
# sort of at most CAP keys and ties are filled from the largest block IDs, so
# the output is already canonical.  Rows with valid_count <= CAP skip (1)-(3).
# Rows where no threshold fits (heavy ties, adversarial data) run the direct8
# eight-pass composite radix inside the same kernel and are flagged.

_GVR_HEADER = r"""
template <uint EPT, bool DOUBLE_BUFFER>
inline void gvr_sort_desc(
    thread uint* v, threadgroup uint* buf, uint lane) {
    constexpr uint N = WIDTH * EPT;
    uint parity = 0u;
    for (uint seq = 2u; seq <= N; seq <<= 1u) {
        for (uint stride = seq >> 1u; stride > 0u; stride >>= 1u) {
            if (stride >= WIDTH) {
                const uint hs = stride / WIDTH;
                for (uint h = 0u; h < EPT; ++h) {
                    const uint p = h ^ hs;
                    if (p > h) {
                        const bool desc = ((lane + h * WIDTH) & seq) == 0u;
                        const uint a = v[h];
                        const uint b = v[p];
                        v[h] = desc ? metal::max(a, b) : metal::min(a, b);
                        v[p] = desc ? metal::min(a, b) : metal::max(a, b);
                    }
                }
            } else if (stride >= 32u) {
                threadgroup uint* base = buf + (DOUBLE_BUFFER ? parity * N : 0u);
                for (uint h = 0u; h < EPT; ++h) {
                    base[lane + h * WIDTH] = v[h];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint h = 0u; h < EPT; ++h) {
                    const uint e = lane + h * WIDTH;
                    const uint other = base[e ^ stride];
                    const bool keep_max =
                        ((e & stride) == 0u) == ((e & seq) == 0u);
                    v[h] = keep_max ? metal::max(v[h], other)
                                    : metal::min(v[h], other);
                }
                if (DOUBLE_BUFFER) {
                    parity ^= 1u;
                } else {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else {
                for (uint h = 0u; h < EPT; ++h) {
                    const uint e = lane + h * WIDTH;
                    const uint other = simd_shuffle_xor(v[h], ushort(stride));
                    const bool keep_max =
                        ((e & stride) == 0u) == ((e & seq) == 0u);
                    v[h] = keep_max ? metal::max(v[h], other)
                                    : metal::min(v[h], other);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Exclusive threadgroup scan of NV per-thread values; simdgroup j scans the
// partials of value j, so NV <= SIMD_GROUPS.
template <uint NV>
inline void gvr_scan(
    thread const uint* value,
    thread uint* exclusive,
    thread uint* total,
    threadgroup uint* partial_in,
    threadgroup uint* partial_out,
    threadgroup uint* totals,
    uint simd_id,
    uint simd_lane) {
    uint local[NV];
    for (uint j = 0u; j < NV; ++j) {
        local[j] = metal::simd_prefix_exclusive_sum(value[j]);
        if (simd_lane == 31u) {
            partial_in[j * SIMD_GROUPS + simd_id] = local[j] + value[j];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < NV) {
        const uint x = simd_lane < SIMD_GROUPS
            ? partial_in[simd_id * SIMD_GROUPS + simd_lane] : 0u;
        const uint prefix = metal::simd_prefix_exclusive_sum(x);
        const uint sum = metal::simd_sum(x);
        if (simd_lane < SIMD_GROUPS) {
            partial_out[simd_id * SIMD_GROUPS + simd_lane] = prefix;
        }
        if (simd_lane == 0u) {
            totals[simd_id] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0u; j < NV; ++j) {
        exclusive[j] = partial_out[j * SIMD_GROUPS + simd_id] + local[j];
        total[j] = totals[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""


_GVR_SOURCE = r"""
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
        device uint* out = block_ids + size_t(row) * TOP_K;
        device uint* row_diag = diagnostics + size_t(row) * 3u;

        threadgroup uint cand_ids[CAP];
        threadgroup uint cand_keys[CAP];
        threadgroup uint sort_buf[SORT_BUF];
        threadgroup uint scan_in[4u * SIMD_GROUPS];
        threadgroup uint scan_out[4u * SIMD_GROUPS];
        threadgroup uint scan_totals[4];
        threadgroup uint shared_word;
        threadgroup atomic_uint radix_histogram[RADIX_BINS];
        threadgroup ulong radix_prefix;
        threadgroup uint radix_rank;

        // Invalid tail: identical fill IDs to the reference selectors.
        for (uint slot = selected_valid + lane; slot < TOP_K; slot += WIDTH) {
            const uint invalid_count = TOP_K - selected_valid;
            out[slot] = blocks - invalid_count + (slot - selected_valid);
        }
        if (selected_valid == 0u) {
            if (lane == 0u) {
                row_diag[0] = 0u;
                row_diag[1] = 0u;
                row_diag[2] = 0u;
            }
            return;
        }

        // Each simdgroup owns one contiguous, coalesced chunk of the row so
        // emission is in ascending block order without a final sort.
        const uint chunk = ((valid_count + SIMD_GROUPS - 1u) / SIMD_GROUPS + 31u)
            & ~31u;
        const uint chunk_start = metal::min(simd_id * chunk, valid_count);
        const uint chunk_end = metal::min(chunk_start + chunk, valid_count);

        uint path = 1u;
        uint picked = 0u;
        uint m = valid_count;
        if (valid_count <= CAP) {
            for (uint block = lane; block < valid_count; block += WIDTH) {
                cand_ids[block] = block;
                cand_keys[block] = qsa_direct_float_order_key(
                    scores[score_base + block]);
            }
        } else {
            path = 2u;
            const uint sample_block =
                uint((ulong(lane) * ulong(valid_count)) / ulong(WIDTH));
            uint sample[1] = {qsa_direct_float_order_key(
                scores[score_base + sample_block])};
            gvr_sort_desc<1, true>(sample, sort_buf, lane);
            sort_buf[lane] = sample[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint thresholds[4];
            for (uint j = 0u; j < 4u; ++j) {
                const uint target =
                    (selected_valid * FRACTIONS[j] + 255u) / 256u;
                const uint rank = uint(
                    (ulong(target) * ulong(WIDTH)) / ulong(valid_count));
                thresholds[j] = sort_buf[metal::clamp(rank, 1u, WIDTH) - 1u];
            }

            uint counts[4] = {0u, 0u, 0u, 0u};
            for (uint block = chunk_start + simd_lane; block < chunk_end;
                 block += 32u) {
                const uint key = qsa_direct_float_order_key(
                    scores[score_base + block]);
                for (uint j = 0u; j < 4u; ++j) {
                    counts[j] += key >= thresholds[j] ? 1u : 0u;
                }
            }
            uint offsets[4];
            uint totals[4];
            gvr_scan<4>(counts, offsets, totals, scan_in, scan_out, scan_totals,
                        simd_id, simd_lane);
            picked = 4u;
            for (uint j = 0u; j < 4u; ++j) {
                if (totals[j] >= selected_valid) {
                    picked = j;
                    break;
                }
            }
            if (picked == 4u || totals[picked] > CAP) {
                path = 3u;
            } else {
                const uint threshold = thresholds[picked];
                m = totals[picked];
                // Lane 0's exclusive prefix is this simdgroup's offset.
                uint position = simd_broadcast_first(offsets[picked]);
                for (uint base = chunk_start; base < chunk_end; base += 32u) {
                    const uint block = base + simd_lane;
                    uint key = 0u;
                    bool take = false;
                    if (block < chunk_end) {
                        key = qsa_direct_float_order_key(
                            scores[score_base + block]);
                        take = key >= threshold;
                    }
                    const uint flag = take ? 1u : 0u;
                    const uint slot =
                        position + metal::simd_prefix_exclusive_sum(flag);
                    if (take) {
                        cand_ids[slot] = block;
                        cand_keys[slot] = key;
                    }
                    position += metal::simd_sum(flag);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (path != 3u) {
            uint tf = 0u;
            if (m != selected_valid) {
                if (m <= WIDTH) {
                    uint v[1] = {lane < m ? cand_keys[lane] : 0u};
                    gvr_sort_desc<1, true>(v, sort_buf, lane);
                    if (lane == selected_valid - 1u) {
                        shared_word = v[0];
                    }
                } else {
                    uint v[CPT];
                    for (uint h = 0u; h < CPT; ++h) {
                        const uint e = lane + h * WIDTH;
                        v[h] = e < m ? cand_keys[e] : 0u;
                    }
                    gvr_sort_desc<CPT, false>(v, sort_buf, lane);
                    for (uint h = 0u; h < CPT; ++h) {
                        if (lane + h * WIDTH == selected_valid - 1u) {
                            shared_word = v[h];
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                tf = shared_word;
            }

            uint local[2] = {0u, 0u};
            for (uint i = 0u; i < CPT; ++i) {
                const uint e = lane * CPT + i;
                if (e < m) {
                    const uint key = cand_keys[e];
                    local[0] += key > tf ? 1u : 0u;
                    local[1] += key == tf ? 1u : 0u;
                }
            }
            uint before[2];
            uint totals[2];
            gvr_scan<2>(local, before, totals, scan_in, scan_out, scan_totals,
                        simd_id, simd_lane);
            // Ties at the k-th key: the largest block IDs (the last ties in
            // ascending order) win, exactly as in the composite-key order.
            const uint skip = totals[1] - (selected_valid - totals[0]);
            uint greater = before[0];
            uint ties = before[1];
            for (uint i = 0u; i < CPT; ++i) {
                const uint e = lane * CPT + i;
                if (e < m) {
                    const uint key = cand_keys[e];
                    if (key > tf) {
                        out[greater + (ties > skip ? ties - skip : 0u)] =
                            cand_ids[e];
                        greater += 1u;
                    } else if (key == tf) {
                        if (ties >= skip) {
                            out[greater + ties - skip] = cand_ids[e];
                        }
                        ties += 1u;
                    }
                }
            }
        } else {
            // Fallback: the direct8 eight-pass composite radix, then an
            // ordered emission of exactly selected_valid keys.
            if (lane == 0u) {
                radix_prefix = 0ul;
                radix_rank = selected_valid - 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
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
                    if (pass == 0u || (key >> (shift + 8u)) == prefix) {
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
            const ulong kth = radix_prefix;
            uint local[1] = {0u};
            for (uint block = chunk_start + simd_lane; block < chunk_end;
                 block += 32u) {
                local[0] += qsa_direct_composite_key(
                    scores[score_base + block], block) >= kth ? 1u : 0u;
            }
            uint before[1];
            uint totals[1];
            gvr_scan<1>(local, before, totals, scan_in, scan_out, scan_totals,
                        simd_id, simd_lane);
            uint position = simd_broadcast_first(before[0]);
            for (uint base = chunk_start; base < chunk_end; base += 32u) {
                const uint block = base + simd_lane;
                const bool take = block < chunk_end &&
                    qsa_direct_composite_key(
                        scores[score_base + block], block) >= kth;
                const uint flag = take ? 1u : 0u;
                const uint slot =
                    position + metal::simd_prefix_exclusive_sum(flag);
                if (take) {
                    out[slot] = block;
                }
                position += metal::simd_sum(flag);
            }
            m = 0u;
        }
        if (lane == 0u) {
            row_diag[0] = path;
            row_diag[1] = m;
            row_diag[2] = picked;
        }
"""


@lru_cache(maxsize=16)
def _gvr_kernel(topk: int, ratio: int, width: int, capacity: int):
    from .qwen4_qsa_gvr_reference import GvrConfig

    config = GvrConfig(int(topk), int(ratio), int(width), int(capacity))
    fractions = ", ".join(f"{value}u" for value in config.fractions256)
    header = (
        _HEADER
        + f"\nconstant constexpr uint TOP_K = {config.topk};\n"
        + f"constant constexpr uint RATIO = {config.ratio};\n"
        + f"constant constexpr uint WIDTH = {config.width};\n"
        + f"constant constexpr uint SIMD_GROUPS = {config.width // 32};\n"
        + f"constant constexpr uint CAP = {config.capacity};\n"
        + f"constant constexpr uint CPT = {config.capacity // config.width};\n"
        + "constant constexpr uint SORT_BUF = "
        + f"{max(config.capacity, 2 * config.width)};\n"
        + "constant constexpr uint RADIX_BINS = 256;\n"
        + f"constant uint FRACTIONS[4] = {{{fractions}}};\n"
        + _GVR_HEADER
    )
    return mx.fast.metal_kernel(
        name=(
            f"mlx2_qwen4_qsa_gvr_k{config.topk}_r{config.ratio}"
            f"_w{config.width}_c{config.capacity}"
        ),
        input_names=["scores", "q_positions", "dims"],
        output_names=["block_ids", "diagnostics"],
        header=header,
        source=_GVR_SOURCE,
    )


GVR_DEFAULT_WIDTH = 1024
GVR_DEFAULT_CAPACITY = 2048


def select_scores_gvr(
    scores: mx.array,
    q_positions: mx.array,
    *,
    topk: int,
    compress_ratio: int,
    width: int = GVR_DEFAULT_WIDTH,
    capacity: int = GVR_DEFAULT_CAPACITY,
    return_diagnostics: bool = False,
):
    """Exact self-sampling multi-threshold selection with in-kernel fallback.

    Diagnostics are ``[rows, 3]`` uint32: path (0 empty, 1 dense, 2 sampled,
    3 direct8 fallback), candidate count, and chosen threshold index.
    """
    rows, blocks = map(int, scores.shape)
    selected, diagnostics = _gvr_kernel(
        int(topk), int(compress_ratio), int(width), int(capacity)
    )(
        inputs=[
            scores,
            q_positions.astype(mx.int32),
            mx.array([blocks], dtype=mx.int32),
        ],
        grid=(rows * int(width), 1, 1),
        threadgroup=(int(width), 1, 1),
        output_shapes=[(rows, int(topk)), (rows, 3)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    if return_diagnostics:
        return selected, diagnostics
    return selected
