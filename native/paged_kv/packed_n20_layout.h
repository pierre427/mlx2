#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mlx2::paged_kv {

template <class Strides>
inline void validate_packed_n20_source_layout(
    uint32_t rows, uint32_t kv_heads, uint32_t dim,
    const Strides& strides, int64_t byte_offset,
    uint64_t backing_bytes) {
  const uint64_t row_values = uint64_t(kv_heads) * dim;
  if (rows == 0 || kv_heads == 0 || dim == 0 || strides.size() != 3 ||
      byte_offset < 0 || uint64_t(byte_offset) % 2 != 0 ||
      (rows != 1 && strides[0] != static_cast<int64_t>(row_values)) ||
      strides[1] != static_cast<int64_t>(dim) || strides[2] != 1)
    throw std::invalid_argument("grouped N20 source must be token-major; singleton row stride is unused");
  // A singleton row never reads stride[0]. The indexed head/channel range is
  // exactly row_values elements even if its unused leading stride is zero.
  const uint64_t offset = static_cast<uint64_t>(byte_offset);
  if (offset > backing_bytes || uint64_t(rows) * row_values > (backing_bytes - offset) / 2)
    throw std::invalid_argument("grouped N20 source exceeds backing bytes");
}

} // namespace mlx2::paged_kv
