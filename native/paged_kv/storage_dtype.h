#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlx2::paged_kv {
enum class StorageDtype { Float16, BFloat16 };

inline StorageDtype storage_dtype_from_name(std::string_view name) {
  if (name == "float16") return StorageDtype::Float16;
  if (name == "bfloat16") return StorageDtype::BFloat16;
  throw std::invalid_argument("paged native storage requires float16 or bfloat16");
}
inline const char* storage_dtype_name(StorageDtype dtype) {
  return dtype == StorageDtype::Float16 ? "float16" : "bfloat16";
}
inline std::string storage_library_suffix(StorageDtype dtype) {
  return dtype == StorageDtype::Float16 ? "" : "_bf16_v1";
}

// Only storage identifiers change. The original FP16 source is returned
// byte-for-byte. Arithmetic stays FP32; raw BF16 copy kernels use ushort so
// every 16-bit payload, including NaNs, is copied without numeric conversion.
inline std::string storage_specialized_source(const char* source, StorageDtype dtype,
                                               bool raw_copy, size_t expected_words) {
  std::string result(source);
  if (dtype == StorageDtype::Float16) return result;
  auto identifier = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
  };
  const std::string replacement = raw_copy ? "ushort" : "bfloat";
  size_t at = 0, count = 0;
  while ((at = result.find("half", at)) != std::string::npos) {
    if ((at == 0 || !identifier(result[at - 1])) &&
        (at + 4 == result.size() || !identifier(result[at + 4]))) {
      result.replace(at, 4, replacement);
      at += replacement.size();
      ++count;
    } else at += 4;
  }
  if (count != expected_words)
    throw std::invalid_argument("paged native storage source ABI drifted");
  return result;
}
} // namespace mlx2::paged_kv
