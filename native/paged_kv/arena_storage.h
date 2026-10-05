#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mlx2::paged_kv {

// MLX dimensions are signed 32-bit, while an array's byte count and Metal
// buffer offset are size_t. Keep the original flat ABI for small planes and
// represent larger exact-byte buffers as contiguous rows of bytes.
constexpr size_t kArenaLargePlaneRowBytes = 4096;
constexpr size_t kArenaMaxPlaneBytes = size_t{12} << 30;

struct ArenaStoragePlan {
  int32_t rows;
  int32_t columns;
  bool multidimensional;
  size_t bytes;
};

inline size_t arena_max_plane_bytes(size_t metal_max_buffer_bytes) {
  return std::min(metal_max_buffer_bytes, kArenaMaxPlaneBytes);
}

inline ArenaStoragePlan plan_arena_storage(
    size_t plane_bytes, size_t metal_max_buffer_bytes) {
  const size_t limit = arena_max_plane_bytes(metal_max_buffer_bytes);
  if (plane_bytes == 0 || plane_bytes > limit)
    throw std::invalid_argument("plane exceeds bounded Metal buffer capacity");
  if (plane_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return {1, static_cast<int32_t>(plane_bytes), false, plane_bytes};
  if (plane_bytes % kArenaLargePlaneRowBytes != 0 ||
      plane_bytes / kArenaLargePlaneRowBytes >
          static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("large plane requires exact 4096-byte rows");
  return {static_cast<int32_t>(plane_bytes / kArenaLargePlaneRowBytes),
          static_cast<int32_t>(kArenaLargePlaneRowBytes), true, plane_bytes};
}

} // namespace mlx2::paged_kv
