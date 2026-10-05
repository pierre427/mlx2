#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mlx2::paged_kv {

inline bool packed_n20_requested(const char* flag) {
  if (!flag || std::string_view(flag) == "0") return false;
  if (std::string_view(flag) == "1") return true;
  throw std::invalid_argument("packed N20 selector must be 0 or 1");
}

struct PackedN20WritePlan {
  std::vector<uint32_t> row_begin;
  uint32_t rows = 0;
};

inline PackedN20WritePlan validate_packed_n20_write(
    const std::vector<uint32_t>& counts, const std::vector<uint32_t>& starts,
    const std::vector<uint32_t>& first_blocks,
    const std::vector<uint32_t>& table_begins,
    const std::vector<uint32_t>& page_ids, uint64_t page_capacity,
    uint32_t source_rows) {
  const size_t n = counts.size();
  if (n == 0 || n > 20 || starts.size() != n || first_blocks.size() != n ||
      table_begins.size() != n || page_capacity == 0 ||
      page_ids.empty() || page_ids.size() > 20 * 128 ||
      source_rows == 0 || source_rows > 20 * 8192)
    throw std::invalid_argument("packed N20 write geometry exceeds bounded lanes or pages");
  PackedN20WritePlan plan;
  plan.row_begin.reserve(n);
  std::set<uint32_t> physical_pages;
  uint64_t rows = 0, table = 0;
  for (size_t lane = 0; lane < n; ++lane) {
    const uint64_t end = uint64_t(starts[lane]) + counts[lane];
    if (counts[lane] == 0 || counts[lane] > 8192 || end > 8192 ||
        first_blocks[lane] != 0 || table_begins[lane] != table ||
        rows + counts[lane] > source_rows)
      throw std::invalid_argument("packed N20 write has invalid logical lane range");
    const uint64_t required = (end + 63) / 64;
    if (required == 0 || table + required > page_ids.size())
      throw std::invalid_argument("packed N20 write page table misses a lane");
    for (uint64_t block = 0; block < required; ++block) {
      const uint32_t page = page_ids[table + block];
      if (page >= page_capacity || !physical_pages.insert(page).second)
        throw std::invalid_argument("packed N20 write aliases a physical page");
    }
    plan.row_begin.push_back(static_cast<uint32_t>(rows));
    rows += counts[lane];
    table += required;
  }
  if (rows != source_rows || table != page_ids.size())
    throw std::invalid_argument("packed N20 write has unused source rows or pages");
  plan.rows = source_rows;
  return plan;
}

} // namespace mlx2::paged_kv
