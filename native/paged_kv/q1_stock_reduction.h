#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <cstdint>

namespace mlx2::paged_kv {

inline bool use_q1_stock_reduction(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("Q1 stock reduction switch must be 0 or 1");
}

inline bool q1_stock_pair_padding_aligned(uint32_t visible0, uint32_t visible1) {
  const auto gap = visible0 > visible1 ? visible0 - visible1 : visible1 - visible0;
  return visible0 != 0 && visible1 != 0 && gap % 32 == 0;
}

// Default-off singleton arm shares exactly the stock32 source. Only the caller's
// checked short, unwindowed, full-prefix singleton can enter it.
inline bool use_q1_stock_singleton(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("Q1 stock singleton switch must be 0 or 1");
}
inline bool q1_stock_reduction_active(bool requested, bool q1_pair,
                                     bool singleton_requested = false,
                                     bool short_singleton = false) {
  return requested && (q1_pair || (singleton_requested && short_singleton));
}

// This is a bounded, opt-in source variant of the Q1 kernel. The ABI and
// page lookup preceding q[8] are inherited from the checked base source.
// Stream each output component through a 32x32 scratch tile, as MLX's
// sdpa_vector does, rather than storing 32xD partials in threadgroup memory.
inline std::string stock_reduction_source(const char* base) {
  std::string source(base);
  constexpr std::string_view begin = "  float q[8];\n";
  constexpr std::string_view end = "\n}\n";
  const auto first = source.find(begin);
  const auto last = source.rfind(end);
  if (first == std::string::npos || source.find(begin, first + 1) != std::string::npos ||
      last == std::string::npos || last <= first || last + end.size() != source.size() ||
      source.find("kernel void mlx2_paged_attention_read_q1_simd_tile_fp16(") == std::string::npos)
    throw std::invalid_argument("Q1 stock reduction base source ABI drifted");
  source.replace(first, last - first, R"metal(  constexpr uint stripes = 32;
  const uint part_count = dim / 32;
  float q[8];
  float numerator[8] = {0};
  for (uint part = 0; part < part_count; ++part)
    q[part] = scale * float(query[(size_t)row * query_row_stride +
                                  (size_t)qh * query_head_stride +
                                  (size_t)(lane * part_count + part) * query_dim_stride]);
  float maximum = -3.402823466e+38f;
  float denominator = 0.0f;
  // The bounded B2 admission requires equal left-padding modulo 32. Thus
  // logical token stripes match the stock dense ragged mask's virtual slots.
  for (uint token = lower + stripe; token < upper; token += stripes) {
    const uint page = page_ids[table_begin[span] + token / 64 - first_block[span]];
    const size_t base = (((size_t)page * kv_heads + kvh) * 64 + (token & 63)) * dim;
    float score = 0.0f;
    for (uint part = 0; part < part_count; ++part)
      score += q[part] * float(keys[base + lane * part_count + part]);
    score = simd_sum(score);
    const float next_max = metal::max(maximum, score);
    const float previous = fast::exp(maximum - next_max);
    const float current = fast::exp(score - next_max);
    maximum = next_max;
    denominator = denominator * previous + current;
    for (uint part = 0; part < part_count; ++part) {
      const uint d = lane * part_count + part;
      numerator[part] = numerator[part] * previous + current * float(values[base + d]);
    }
  }
  threadgroup float outputs[32 * 32];
  threadgroup float max_scores[32];
  threadgroup float sum_exp_scores[32];
  if (lane == 0) {
    max_scores[stripe] = maximum;
    sum_exp_scores[stripe] = denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float local_max = max_scores[lane];
  const float merged_max = simd_max(local_max);
  const float factor = fast::exp(local_max - merged_max);
  const float total_denominator = simd_sum(sum_exp_scores[lane] * factor);
  for (uint part = 0; part < part_count; ++part) {
    outputs[lane * 32 + stripe] = numerator[part];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total_numerator = simd_sum(outputs[stripe * 32 + lane] * factor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0)
      output[((size_t)row * query_heads + qh) * dim + stripe * part_count + part] =
          half(total_numerator / total_denominator);
  })metal");
  return source;
}

} // namespace mlx2::paged_kv
