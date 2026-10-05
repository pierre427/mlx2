#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mlx2::paged_kv {

inline bool packed_multirow_write_selected(const char* flag) {
  if (flag == nullptr || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("packed multirow write selector must be 0 or 1");
}

// Return every destination page generation ID used by the proposed rows.
// The caller must hold the corresponding host page-generation leases until
// the native command buffer terminal; this only validates physical IDs.
inline std::vector<uint32_t> validate_packed_multirow_write(
    std::array<uint32_t, 2> counts, std::array<uint32_t, 2> starts,
    std::array<uint32_t, 2> first_blocks, std::array<uint32_t, 2> table_begins,
    const std::vector<uint32_t>& page_ids, uint64_t page_capacity,
    uint32_t source_rows) {
  if (counts[0] == 0 || counts[1] == 0 || counts[0] > 8192 || counts[1] > 8192 ||
      static_cast<uint64_t>(counts[0]) + counts[1] != source_rows ||
      page_ids.empty() || page_ids.size() > 4096 / sizeof(uint32_t) ||
      table_begins[0] != 0 || table_begins[1] == 0 ||
      table_begins[1] >= page_ids.size() || page_capacity == 0)
    throw std::invalid_argument("packed multirow counts or page tables differ");
  std::map<uint32_t, uint32_t> lanes[2];
  std::vector<uint32_t> touched;
  for (uint32_t lane = 0; lane < 2; ++lane) {
    if (static_cast<uint64_t>(starts[lane]) + counts[lane] > 8192 ||
        first_blocks[lane] > starts[lane] / 64)
      throw std::invalid_argument("packed multirow logical range differs");
    const size_t begin = table_begins[lane];
    const size_t end = lane == 0 ? table_begins[1] : page_ids.size();
    if (begin >= end) throw std::invalid_argument("packed multirow empty lane table");
    for (uint32_t row = 0; row < counts[lane]; ++row) {
      const auto logical = starts[lane] + row;
      const auto block = logical / 64 - first_blocks[lane];
      if (begin + block >= end)
        throw std::invalid_argument("packed multirow page table misses a destination");
      const auto page = page_ids[begin + block];
      if (page >= page_capacity || lanes[1 - lane].find(page) != lanes[1 - lane].end() ||
          (lanes[lane].find(page) != lanes[lane].end() && lanes[lane][page] != block))
        throw std::invalid_argument("packed multirow destination aliases another lane or arena bound");
      if (lanes[lane].emplace(page, block).second) touched.push_back(page);
    }
  }
  return touched;
}

} // namespace mlx2::paged_kv
