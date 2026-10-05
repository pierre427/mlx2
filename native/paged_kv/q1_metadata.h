#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <string>
#include <stdexcept>

namespace mlx2::paged_kv {
// Only address-space declarations differ. Kernel arithmetic and buffer slots
// remain byte-for-byte identical to the validated ordinary/tile sources.
inline std::string inline_metadata_source(const char* source) {
  std::string result(source);
  const std::string device = "device const uint*";
  const std::string constant = "constant uint*";
  size_t position = 0;
  size_t fields = 0;
  while ((position = result.find(device, position)) != std::string::npos) {
    result.replace(position, device.size(), constant);
    position += constant.size();
    ++fields;
  }
  if (fields != 8)
    throw std::invalid_argument("inline attention metadata requires eight fields");
  return result;
}

// Called only after the existing complete span/page/arena validation. Bound
// every inline buffer to <=24 bytes, far below Metal's 4096-byte setBytes cap.
// A window over a long retained table remains on the dynamic metadata path.
inline bool use_inline_q1_metadata(const char* flag, bool short_q1,
                                   uint64_t rows, size_t spans, size_t pages) {
  return flag != nullptr && std::string_view(flag) == "1" && short_q1 &&
      rows == 2 && spans == 2 && pages >= 2 && pages <= 6;
}
} // namespace mlx2::paged_kv
