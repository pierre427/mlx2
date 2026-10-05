#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlx2::paged_kv {
inline uint32_t q1_split_kv_partition(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return 0;
  if (std::string_view(flag) == "128") return 128;
  if (std::string_view(flag) == "256") return 256;
  throw std::invalid_argument("Q1 split KV requires 0, 128 or 256");
}

struct Q1SplitKVPlan {
  uint32_t partition_tokens;
  uint32_t partitions;
  uint32_t scratch_values;
  uint64_t scratch_bytes;
};

inline Q1SplitKVPlan q1_split_kv_plan(uint32_t dim, uint32_t heads,
                                     std::array<uint32_t, 2> visible,
                                     uint32_t partition_tokens) {
  if ((dim != 128 && dim != 256) || heads == 0 || heads > 128 ||
      (partition_tokens != 128 && partition_tokens != 256) ||
      visible[0] == 0 || visible[1] == 0 || visible[0] > 8192 || visible[1] > 8192)
    throw std::invalid_argument("Q1 split KV geometry exceeds bounded B2 plan");
  const uint32_t longest = visible[0] > visible[1] ? visible[0] : visible[1];
  const uint32_t partitions = (longest + partition_tokens - 1) / partition_tokens;
  const uint32_t values = 2 * heads * partitions * (dim + 2);
  return {partition_tokens, partitions, values, uint64_t(values) * sizeof(float)};
}

constexpr const char* kQ1SplitKVSource = R"metal(
#include <metal_stdlib>
using namespace metal;
constant uint Q1_DIM = __Q1_DIM__;
constant uint Q1_PARTITION = __Q1_PARTITION__;

kernel void mlx2_paged_q1_split_partial_fp16(
    device const half* query [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const half* values [[buffer(2)]],
    device const uchar* dependency [[buffer(3)]],
    device const uint* row_span [[buffer(4)]],
    device const uint* row_begin [[buffer(5)]],
    device const uint* query_start [[buffer(6)]],
    device const uint* retained_start [[buffer(7)]],
    device const uint* first_block [[buffer(8)]],
    device const uint* table_begin [[buffer(9)]],
    device const uint* window [[buffer(10)]],
    device const uint* page_ids [[buffer(11)]],
    device float* scratch [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    constant uint& query_heads [[buffer(14)]],
    constant uint& kv_heads [[buffer(15)]],
    constant uint& partitions [[buffer(16)]],
    constant ulong& query_row_stride [[buffer(17)]],
    constant ulong& query_head_stride [[buffer(18)]],
    constant ulong& query_dim_stride [[buffer(19)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint partition = group.x, row = group.y, qh = group.z;
  const uint span = row_span[row];
  const uint upper = query_start[span] + row - row_begin[span] + 1;
  const uint lower = window[span] == 0 ? retained_start[span] :
      metal::max(retained_start[span], upper - metal::min(upper, window[span]));
  // Empty tail partitions and invalid dependencies still initialize scratch
  // and reach both barriers. They contribute zero to the second stage.
  const uint begin = lower + metal::min(upper - lower, partition * Q1_PARTITION);
  const uint end = dependency[0] == 1 ? begin + metal::min(upper - begin, Q1_PARTITION) : begin;
  const uint kvh = qh / (query_heads / kv_heads);
  float q[Q1_DIM / 32], numerator[Q1_DIM / 32] = {0};
  for (uint part = 0; part < Q1_DIM / 32; ++part)
    q[part] = float(query[(size_t)row * query_row_stride +
                          (size_t)qh * query_head_stride +
                          (size_t)(lane + part * 32) * query_dim_stride]);
  float maximum = -INFINITY, denominator = 0.0f;
  for (uint local = stripe; local < end - begin; local += 4) {
    const uint token = begin + local;
    const uint page = page_ids[table_begin[span] + token / 64 - first_block[span]];
    const size_t base = (((size_t)page * kv_heads + kvh) * 64 + (token & 63)) * Q1_DIM;
    float score = 0.0f;
    for (uint part = 0; part < Q1_DIM / 32; ++part)
      score += q[part] * float(keys[base + lane + part * 32]);
    score = simd_sum(score) * scale;
    const float next_maximum = metal::max(maximum, score);
    const float previous = maximum == -INFINITY ? 0.0f : metal::exp(maximum - next_maximum);
    const float current = metal::exp(score - next_maximum);
    maximum = next_maximum;
    denominator = denominator * previous + current;
    for (uint part = 0; part < Q1_DIM / 32; ++part)
      numerator[part] = numerator[part] * previous + current * float(values[base + lane + part * 32]);
  }
  threadgroup float stripe_maximum[4], stripe_denominator[4];
  threadgroup float stripe_numerator[4][Q1_DIM];
  threadgroup float weights[4], merged_maximum, merged_denominator;
  if (lane == 0) {
    stripe_maximum[stripe] = maximum;
    stripe_denominator[stripe] = denominator;
  }
  for (uint part = 0; part < Q1_DIM / 32; ++part)
    stripe_numerator[stripe][lane + part * 32] = numerator[part];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stripe == 0 && lane == 0) {
    float global_maximum = -INFINITY;
    for (uint s = 0; s < 4; ++s)
      global_maximum = metal::max(global_maximum, stripe_maximum[s]);
    float total = 0.0f;
    for (uint s = 0; s < 4; ++s) {
      weights[s] = stripe_denominator[s] > 0.0f ? metal::exp(stripe_maximum[s] - global_maximum) : 0.0f;
      total += stripe_denominator[s] * weights[s];
    }
    merged_maximum = global_maximum;
    merged_denominator = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stripe == 0) {
    const size_t target = (((size_t)row * query_heads + qh) * partitions + partition) * (Q1_DIM + 2);
    if (lane == 0) {
      scratch[target] = merged_maximum;
      scratch[target + 1] = merged_denominator;
    }
    for (uint part = 0; part < Q1_DIM / 32; ++part) {
      const uint d = lane + part * 32;
      float total = 0.0f;
      for (uint s = 0; s < 4; ++s) total += stripe_numerator[s][d] * weights[s];
      scratch[target + 2 + d] = total;
    }
  }
}

kernel void mlx2_paged_q1_split_reduce_fp16(
    device const float* scratch [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& query_heads [[buffer(2)]],
    constant uint& partitions [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.y, qh = group.z;
  const size_t base = ((size_t)row * query_heads + qh) * partitions * (Q1_DIM + 2);
  float maximum = -INFINITY;
  for (uint p = lane; p < partitions; p += 32)
    maximum = metal::max(maximum, scratch[base + (size_t)p * (Q1_DIM + 2)]);
  maximum = simd_max(maximum);
  threadgroup float weights[64], merged_denominator;
  threadgroup float stripe_numerator[4][Q1_DIM];
  if (stripe == 0) {
    float denominator = 0.0f;
    for (uint p = lane; p < partitions; p += 32) {
      const size_t offset = base + (size_t)p * (Q1_DIM + 2);
      const float den = scratch[offset + 1];
      const float weight = den > 0.0f ? metal::exp(scratch[offset] - maximum) : 0.0f;
      weights[p] = weight;
      denominator += den * weight;
    }
    denominator = simd_sum(denominator);
    if (lane == 0) merged_denominator = denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint part = 0; part < Q1_DIM / 32; ++part) {
    const uint d = lane + part * 32;
    float numerator = 0.0f;
    for (uint p = stripe; p < partitions; p += 4)
      numerator += scratch[base + (size_t)p * (Q1_DIM + 2) + 2 + d] * weights[p];
    stripe_numerator[stripe][d] = numerator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stripe == 0)
    for (uint part = 0; part < Q1_DIM / 32; ++part) {
      const uint d = lane + part * 32;
      float total = 0.0f;
      for (uint s = 0; s < 4; ++s) total += stripe_numerator[s][d];
      output[((size_t)row * query_heads + qh) * Q1_DIM + d] =
          merged_denominator > 0.0f ? half(total / merged_denominator) : half(0);
    }
}
)metal";

inline std::string q1_split_kv_source(uint32_t dim, uint32_t partition_tokens) {
  (void)q1_split_kv_plan(dim, 1, {1, 1}, partition_tokens);
  std::string source(kQ1SplitKVSource);
  auto replace = [&](const char* marker, uint32_t value) {
    const auto position = source.find(marker);
    if (position == std::string::npos || source.find(marker, position + 1) != std::string::npos)
      throw std::invalid_argument("Q1 split KV source ABI drifted");
    source.replace(position, std::string(marker).size(), std::to_string(value));
  };
  replace("__Q1_DIM__", dim);
  replace("__Q1_PARTITION__", partition_tokens);
  return source;
}
} // namespace mlx2::paged_kv
