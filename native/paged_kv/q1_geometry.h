#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>

namespace mlx2::paged_kv {
// A missing/zero switch preserves the frozen four-stripe reference source.
inline uint32_t q1_simd_stripes(const char* flag) {
  if (!flag || std::string_view(flag) == "0" || std::string_view(flag) == "4") return 4;
  if (std::string_view(flag) == "8") return 8;
  if (std::string_view(flag) == "16") return 16;
  if (std::string_view(flag) == "32") return 32;
  throw std::invalid_argument("Q1 SIMD stripes must be 0, 4, 8, 16 or 32");
}

inline bool q1_geometry_supported(uint32_t dim, uint32_t stripes) {
  // Bound static threadgroup arrays to Apple's 32 KiB baseline. The native
  // pipeline also checks its actual thread and device memory limits.
  const uint64_t bytes = uint64_t(stripes) * (dim + 3) * sizeof(float) + sizeof(float);
  return (dim == 128 || dim == 256) &&
      (stripes == 8 || stripes == 16 || stripes == 32) && bytes <= 32768;
}

inline std::string specialized_q1_source(const char* source, uint32_t dim,
                                         uint32_t stripes) {
  if (!q1_geometry_supported(dim, stripes))
    throw std::invalid_argument("invalid specialized Q1 geometry");
  std::string result(source);
  auto replace = [&](const std::string& from, const std::string& to, size_t expected) {
    size_t position = 0, count = 0;
    while ((position = result.find(from, position)) != std::string::npos) {
      result.replace(position, from.size(), to);
      position += to.size();
      ++count;
    }
    if (count != expected) throw std::invalid_argument("Q1 geometry source ABI drifted");
  };
  replace("constant uint& dim [[buffer(16)]]", "constant uint& dim_bound [[buffer(16)]]", 1);
  replace("  const uint row = group.y;", "  constexpr uint dim = " + std::to_string(dim) +
          ";\n  (void)dim_bound;\n  const uint row = group.y;", 1);
  replace("token += 4", "token += " + std::to_string(stripes), 1);
  replace("[4]", "[" + std::to_string(stripes) + "]", 3);
  replace("][256]", "][" + std::to_string(dim) + "]", 1);
  replace("tile < 4", "tile < " + std::to_string(stripes), 3);
  // Reuse one weight per stripe for denominator and every output channel.
  // All SIMD groups reach the second barrier, including empty tail stripes.
  const std::string before = R"source(  if (stripe == 0) {
    float global_maximum = -INFINITY;
    for (uint tile = 0; tile < STRIPES; ++tile)
      global_maximum = metal::max(global_maximum, partial_maximum[tile]);
    float total_denominator = 0.0f;
    for (uint tile = 0; tile < STRIPES; ++tile)
      if (partial_denominator[tile] > 0.0f)
        total_denominator += partial_denominator[tile] *
            metal::exp(partial_maximum[tile] - global_maximum);
)source";
  std::string expected = before;
  for (size_t at = 0; (at = expected.find("STRIPES", at)) != std::string::npos;)
    expected.replace(at, 7, std::to_string(stripes));
  const std::string after = R"source(  threadgroup float merge_weights[STRIPES];
  threadgroup float merged_denominator;
  if (stripe == 0 && lane == 0) {
    float global_maximum = -INFINITY;
    for (uint tile = 0; tile < STRIPES; ++tile)
      global_maximum = metal::max(global_maximum, partial_maximum[tile]);
    float total_denominator = 0.0f;
    for (uint tile = 0; tile < STRIPES; ++tile) {
      const float weight = partial_denominator[tile] > 0.0f ?
          metal::exp(partial_maximum[tile] - global_maximum) : 0.0f;
      merge_weights[tile] = weight;
      total_denominator += partial_denominator[tile] * weight;
    }
    merged_denominator = total_denominator;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (stripe == 0) {
    const float total_denominator = merged_denominator;
)source";
  std::string replacement = after;
  for (size_t at = 0; (at = replacement.find("STRIPES", at)) != std::string::npos;)
    replacement.replace(at, 7, std::to_string(stripes));
  replace(expected, replacement, 1);
  replace("metal::exp(partial_maximum[tile] - global_maximum);",
          "merge_weights[tile];", 1);
  return result;
}
} // namespace mlx2::paged_kv
