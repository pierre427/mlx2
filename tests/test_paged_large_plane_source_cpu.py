"""Pure host geometry checks for exact-byte paged arena storage."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "paged_kv"


def test_large_plane_shape_is_exact_and_guarded():
    source = r'''
#include "arena_storage.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
using namespace mlx2::paged_kv;
int main() {
  const size_t hardware = size_t{16} << 30;
  const auto small = plan_arena_storage(8192, hardware);
  assert(!small.multidimensional && small.columns == 8192 && small.bytes == 8192);
  const size_t full_n20 = 4802478080ULL;
  const auto large = plan_arena_storage(full_n20, hardware);
  assert(large.multidimensional && large.columns == 4096);
  assert(size_t(large.rows) * size_t(large.columns) == full_n20);
  assert(large.bytes == full_n20 && large.rows > 0);
  assert(arena_max_plane_bytes(hardware) == (size_t{12} << 30));
  auto refused = [&](size_t bytes, size_t limit) {
    try { (void)plan_arena_storage(bytes, limit); return false; }
    catch (const std::invalid_argument&) { return true; }
  };
  assert(refused(0, hardware));
  assert(refused(full_n20 + 1, hardware));
  assert(refused(full_n20, full_n20 - 1));
  assert(refused((size_t{12} << 30) + 4096, hardware));
}
'''
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test.cpp"
        binary = Path(tmp) / "test"
        src.write_text(source)
        subprocess.run(["clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(NATIVE), str(src), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


def test_native_arena_keeps_exact_byte_abi():
    arena = (NATIVE / "arena.cpp").read_text()
    binding = (NATIVE / "binding.cpp").read_text()
    assert "plan_arena_storage(plane_bytes, metal_device->maxBufferLength())" in arena
    assert "mx::allocator::malloc(plane_bytes)" in arena
    assert "plane_bytes_(plane_bytes)" in arena
    assert "destination_offset > plane_bytes_" in arena
    assert 'module.def("arena_storage_capability"' in binding
    assert 'module.def("plane_bytes"' in binding
