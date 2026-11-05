// SPDX-License-Identifier: Apache-2.0
// Original mlx2 scalar-F32 producer/selector prototype. No stock-GEMM parity claim.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#pragma METAL fp math_mode(safe)

template <typename InputT>
inline float selective_score(const device InputT* q, const device InputT* pooled,
                             uint row, uint index, uint rows, uint blocks) {
    const uint batch = row / rows;
    float total = 0.0f;
    for (uint h = 0; h < 4; ++h) {
        float dot = 0.0f;
        for (uint d = 0; d < 128; ++d) {
            float product = float(q[(ulong(row) * 4 + h) * 128 + d]) *
                            float(pooled[(ulong(batch) * blocks + index) * 128 + d]);
            dot = dot + product;
        }
        float positive = isnan(dot) ? dot : (dot > 0.0f ? dot : 0.0f);
        total = total + positive;
    }
    return total / 11.313708498984761f;
}
inline uint selective_order(float value) {
    uint bits = as_type<uint>(value);
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}
inline ulong selective_key(float value, uint index) {
    return (ulong(selective_order(value)) << 32) | ulong(index);
}
inline void selective_heap(thread ulong* heap, thread uint& count, uint required, ulong key) {
    uint at;
    if (count < required) {
        at = count++;
        heap[at] = key;
        while (at > 0) {
            uint parent = (at - 1) / 2;
            if (heap[parent] <= heap[at]) break;
            ulong swap = heap[parent]; heap[parent] = heap[at]; heap[at] = swap;
            at = parent;
        }
    } else if (key > heap[0]) {
        heap[0] = key; at = 0;
        while (2 * at + 1 < count) {
            uint child = 2 * at + 1;
            if (child + 1 < count && heap[child + 1] < heap[child]) ++child;
            if (heap[at] <= heap[child]) break;
            ulong swap = heap[child]; heap[child] = heap[at]; heap[at] = swap;
            at = child;
        }
    }
}

#ifndef MLX_WRAPPED
kernel void selective_qsa(const device INPUT_T* q [[buffer(0)]],
                         const device INPUT_T* pooled [[buffer(1)]],
                         const device int* positions [[buffer(2)]],
                         const device uint* dims [[buffer(3)]],
                         device uint* block_ids [[buffer(4)]],
                         device uint* diagnostics [[buffer(5)]],
                         uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                         uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
#endif
// MLX_BODY
    if (thread_index_in_simdgroup != 0) return;
    const uint row = threadgroup_position_in_grid.x;
    const uint rows = dims[1], blocks = dims[2];
    const long complete = (long(positions[row]) + 1) / long(RATIO);
    const uint valid = uint(clamp(complete, long(0), long(blocks)));
    const uint required = min(uint(TOPK), valid);
    uint count = 0, stored = 0, path = 0;
    ulong heap[TOPK]; uint heap_count = 0;
    if (required > 0) {
        uint sample[SAMPLES];
        const uint ns = min(valid, uint(SAMPLES));
        for (uint s = 0; s < ns; ++s) {
            uint index = uint((ulong(s) * valid) / ns);
            sample[s] = selective_order(selective_score(q, pooled, row, index, rows, blocks));
            uint at = s;
            while (at > 0 && sample[at] > sample[at - 1]) {
                uint swap = sample[at - 1]; sample[at - 1] = sample[at]; sample[at] = swap;
                --at;
            }
        }
        const uint rank = min(ns, max(1u, uint((3ul * required * ns + 2ul * valid - 1) / (2ul * valid))));
        const uint threshold = sample[rank - 1];
        ulong candidates[CAP]; uint counts[4] = {0, 0, 0, 0};
        const uint span = CAP / 4;
        for (uint i = 0; i < valid; ++i) {
            ulong key = selective_key(selective_score(q, pooled, row, i, rows, blocks), i);
            if (uint(key >> 32) >= threshold) {
                uint bank = i % 4, slot = counts[bank]++;
                if (slot < span) { candidates[bank * span + slot] = key; ++stored; }
                ++count;
            }
        }
        uint maximum = max(max(counts[0], counts[1]), max(counts[2], counts[3]));
        if (count >= required && maximum <= span) {
            path = 1;
            for (uint bank = 0; bank < 4; ++bank)
                for (uint i = 0; i < counts[bank]; ++i)
                    selective_heap(heap, heap_count, required, candidates[bank * span + i]);
        } else {
            path = count > CAP ? 2 : (maximum > span ? 3 : 4);
            for (uint i = 0; i < valid; ++i)
                selective_heap(heap, heap_count, required,
                    selective_key(selective_score(q, pooled, row, i, rows, blocks), i));
        }
    }
    uint selected[TOPK];
    for (uint i = 0; i < required; ++i) {
        selected[i] = uint(heap[i]);
        uint at = i;
        while (at > 0 && selected[at] < selected[at - 1]) {
            uint swap = selected[at - 1]; selected[at - 1] = selected[at]; selected[at] = swap;
            --at;
        }
    }
    for (uint i = 0; i < TOPK; ++i)
        block_ids[ulong(row) * TOPK + i] = i < required ? selected[i] : blocks - (TOPK - required) + (i - required);
    diagnostics[ulong(row) * 4 + 0] = path;
    diagnostics[ulong(row) * 4 + 1] = count;
    diagnostics[ulong(row) * 4 + 2] = stored;
    diagnostics[ulong(row) * 4 + 3] = valid;
// MLX_END
#ifndef MLX_WRAPPED
}
#endif
