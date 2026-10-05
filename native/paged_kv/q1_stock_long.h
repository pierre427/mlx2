#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "q1_splitkv.h"

namespace mlx2::paged_kv {

inline bool q1_stock_long_requested(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("Q1 stock long switch must be 0 or 1");
}

inline bool q1_stock_long_n20_singleton_requested(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("N20 B1 stock-long selector must be 0 or 1");
}

inline Q1SplitKVPlan q1_stock_long_plan(uint32_t dim, uint32_t heads,
                                        std::array<uint32_t, 2> visible) {
  if ((dim != 128 && dim != 256) || heads == 0 || heads > 128 ||
      visible[0] == 0 || visible[1] == 0 ||
      std::max(visible[0], visible[1]) <= 1024 ||
      std::max(visible[0], visible[1]) > 8192)
    throw std::invalid_argument("Q1 stock long requires bounded B2 vector geometry");
  // Installed MLX 39400a0d4 uses 128 interleaved blocks on architecture 's'
  // at these lengths when GQA has more than four query lanes.
  constexpr uint32_t blocks = 128;
  const uint64_t values = uint64_t(2) * heads * blocks * (dim + 2);
  if (values > UINT32_MAX) throw std::invalid_argument("Q1 stock long scratch overflows");
  return {blocks, blocks, static_cast<uint32_t>(values), values * sizeof(float)};
}

inline Q1SplitKVPlan q1_stock_long_n20_plan(uint32_t dim, uint32_t heads,
                                             const std::vector<uint32_t>& visible) {
  if (dim != 256 || heads != 24 || visible.empty() || visible.size() > 20 ||
      *std::min_element(visible.begin(), visible.end()) == 0 ||
      *std::max_element(visible.begin(), visible.end()) <= 1024 ||
      *std::max_element(visible.begin(), visible.end()) > 8192)
    throw std::invalid_argument("Q1 stock long N20 requires bounded2..20 BF16 D256 H24 rows");
  constexpr uint32_t blocks = 128;
  const uint64_t values = uint64_t(visible.size()) * heads * blocks * (dim + 2);
  if (values > UINT32_MAX || values * sizeof(float) > 64ULL * 1024 * 1024)
    throw std::invalid_argument("Q1 stock long N20 scratch exceeds64MiB");
  return {blocks, blocks, static_cast<uint32_t>(values), values * sizeof(float)};
}

constexpr const char* kQ1StockLongSource = R"metal(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
constant uint Q1_DIM = __Q1_DIM__;
constant uint Q1_BLOCKS = 128;

kernel void mlx2_paged_q1_stock_long_partial_fp16(
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
    constant uint& blocks [[buffer(16)]],
    constant ulong& query_row_stride [[buffer(17)]],
    constant ulong& query_head_stride [[buffer(18)]],
    constant ulong& query_dim_stride [[buffer(19)]],
    constant uint& dense_length [[buffer(20)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block = group.x, row = group.y, qh = group.z;
  const uint span = row_span[row];
  const uint upper = query_start[span] + row - row_begin[span] + 1;
  const uint lower = retained_start[span];
  const uint pad = dense_length - (upper - lower);
  const uint kvh = qh / (query_heads / kv_heads);
  float q[Q1_DIM / 32], numerator[Q1_DIM / 32] = {0};
  for (uint part = 0; part < Q1_DIM / 32; ++part)
    q[part] = scale * float(query[(size_t)row * query_row_stride +
                                   (size_t)qh * query_head_stride +
                                   (size_t)(lane * (Q1_DIM / 32) + part) * query_dim_stride]);
  float maximum = -3.402823466e+38f, denominator = 0.0f;
  // Dense virtual left padding reproduces the stock ragged B2 mask's block
  // residue. Unlike the older contiguous split, each block visits i+=128.
  for (uint dense = block; dense < dense_length; dense += Q1_BLOCKS) {
    if (dependency[0] != 1 || dense < pad) continue;
    const uint token = lower + dense - pad;
    const uint page = page_ids[table_begin[span] + token / 64 - first_block[span]];
    const size_t base = (((size_t)page * kv_heads + kvh) * 64 + (token & 63)) * Q1_DIM;
    float score = 0.0f;
    for (uint part = 0; part < Q1_DIM / 32; ++part)
      score += q[part] * float(keys[base + lane * (Q1_DIM / 32) + part]);
    score = simd_sum(score);
    const float next_maximum = metal::max(maximum, score);
    const float previous = fast::exp(maximum - next_maximum);
    const float current = fast::exp(score - next_maximum);
    maximum = next_maximum;
    denominator = denominator * previous + current;
    for (uint part = 0; part < Q1_DIM / 32; ++part)
      numerator[part] = numerator[part] * previous +
          current * float(values[base + lane * (Q1_DIM / 32) + part]);
  }
  const size_t target = (((size_t)row * query_heads + qh) * blocks + block) * (Q1_DIM + 2);
  if (lane == 0) {
    scratch[target] = maximum;
    scratch[target + 1] = denominator;
  }
  for (uint part = 0; part < Q1_DIM / 32; ++part)
    scratch[target + 2 + lane * (Q1_DIM / 32) + part] = float(half(numerator[part]));
}

kernel void mlx2_paged_q1_stock_long_reduce_fp16(
    device const float* scratch [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& query_heads [[buffer(2)]],
    constant uint& blocks [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.y, qh = group.z;
  const size_t base = ((size_t)row * query_heads + qh) * blocks * (Q1_DIM + 2);
  float maximum = -3.402823466e+38f;
  for (uint b = 0; b < blocks / 32; ++b)
    maximum = metal::max(maximum, scratch[base + (lane + 32 * b) * (Q1_DIM + 2)]);
  maximum = simd_max(maximum);
  float denominator = 0.0f;
  for (uint b = 0; b < blocks / 32; ++b) {
    const size_t pos = base + (lane + 32 * b) * (Q1_DIM + 2);
    denominator += fast::exp(scratch[pos] - maximum) * scratch[pos + 1];
  }
  denominator = simd_sum(denominator);
  threadgroup float outputs[32 * 32];
  for (uint part = 0; part < Q1_DIM / 32; ++part) {
    float numerator = 0.0f;
    for (uint b = 0; b < blocks / 32; ++b) {
      const size_t pos = base + (stripe + 32 * b) * (Q1_DIM + 2);
      numerator += fast::exp(scratch[pos] - maximum) *
          scratch[pos + 2 + lane * (Q1_DIM / 32) + part];
    }
    outputs[lane * 32 + stripe] = numerator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = simd_sum(outputs[stripe * 32 + lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
      output[((size_t)row * query_heads + qh) * Q1_DIM +
             stripe * (Q1_DIM / 32) + part] = half(denominator == 0 ? total : total / denominator);
  }
}
)metal";

inline std::string q1_stock_long_source(uint32_t dim) {
  if (dim != 128 && dim != 256)
    throw std::invalid_argument("Q1 stock long dimension must be 128 or 256");
  std::string source(kQ1StockLongSource);
  constexpr std::string_view marker = "__Q1_DIM__";
  const auto at = source.find(marker);
  if (at == std::string::npos || source.find(marker, at + 1) != std::string::npos)
    throw std::invalid_argument("Q1 stock long source ABI drifted");
  source.replace(at, marker.size(), std::to_string(dim));
  return source;
}

} // namespace mlx2::paged_kv
