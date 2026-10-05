#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace mlx2::paged_kv {

// A separate selector preserves the validated short inline and long dynamic
// paths. All eight vectors are copied by Metal set_bytes at encode time.
inline bool use_inline_stock_long_metadata(const char* flag, bool stock_long,
                                            uint32_t partition_tokens, uint64_t rows,
                                            size_t spans,
                                            const std::array<size_t, 8>& field_counts) {
  if (flag == nullptr || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) != "1")
    throw std::invalid_argument("stock-long inline metadata switch must be 0 or 1");
  if (!stock_long) return false;
  if (partition_tokens != 128 || rows != 2 || spans != 2)
    throw std::invalid_argument("stock-long inline metadata requires bounded B2 partition geometry");
  for (size_t count : field_counts)
    if (count == 0 || count > 4096 / sizeof(uint32_t))
      throw std::invalid_argument("stock-long inline metadata field exceeds Metal setBytes cap");
  return true;
}

} // namespace mlx2::paged_kv
