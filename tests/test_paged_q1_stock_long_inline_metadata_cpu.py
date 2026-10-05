"""Native source/ABI checks only; no MLX import or GPU execution."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native/paged_kv"
ARENA = (NATIVE / "arena.cpp").read_text()


class StockLongInline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="q1-long-inline-")
        cls.folder = Path(cls.temp.name)
        probe = cls.folder / "probe.cpp"
        probe.write_text(r'''
#include "q1_stock_long_metadata.h"
#include "q1_stock_long.h"
#include "q1_metadata.h"
#include "storage_dtype.h"
#include <iostream>
#include <string_view>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  try {
    if (argc == 3 && std::string_view(argv[1]) == "flag") {
      const std::array<size_t, 8> fields{2, 2, 2, 2, 2, 2, 2, 256};
      std::cout << use_inline_stock_long_metadata(argv[2], true, 128, 2, 2, fields);
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "large") {
      const std::array<size_t, 8> fields{2, 2, 2, 2, 2, 2, 2, 1025};
      std::cout << use_inline_stock_long_metadata(argv[2], true, 128, 2, 2, fields);
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "old") {
      const std::array<size_t, 8> fields{2, 2, 2, 2, 2, 2, 2, 256};
      std::cout << use_inline_stock_long_metadata(argv[2], false, 128, 2, 2, fields);
      return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "raw") {
      std::cout << q1_stock_long_source(std::stoul(argv[2]));
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "rawspecial") {
      auto source = q1_stock_long_source(std::stoul(argv[2]));
      std::cout << storage_specialized_source(source.c_str(), storage_dtype_from_name(argv[3]), false, 6);
      return 0;
    }
    auto source = inline_metadata_source(q1_stock_long_source(std::stoul(argv[1])).c_str());
    std::cout << storage_specialized_source(source.c_str(), storage_dtype_from_name(argv[2]), false, 6);
  } catch (const std::invalid_argument&) { return 2; }
}
''')
        cls.binary = cls.folder / "probe"
        subprocess.run([shutil.which("c++") or "c++", "-std=c++20", "-Wall", "-Wextra",
                        "-Werror", "-I", str(NATIVE), str(probe), "-o", str(cls.binary)],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, *args):
        return subprocess.run([str(self.binary), *map(str, args)], text=True, capture_output=True)

    def test_strict_dedicated_flag_and_byte_cap(self):
        self.assertEqual(self.run_probe("flag", "0").stdout, "0")
        self.assertEqual(self.run_probe("flag", "1").stdout, "1")
        self.assertEqual(self.run_probe("old", "1").stdout, "0")
        for value in ("true", "10", "-1"):
            self.assertEqual(self.run_probe("flag", value).returncode, 2)
        self.assertEqual(self.run_probe("large", "1").returncode, 2)

    def test_actual_bindings_and_post_dispatch_count(self):
        selector = ARENA.index("const bool inline_stock_long_metadata = use_inline_stock_long_metadata(")
        self.assertGreater(selector, ARENA.index('"paged attention rows or pages are unused"'))
        self.assertIn("if (inline_q1_metadata || inline_stock_long_metadata)", ARENA)
        split = ARENA.index("if (split_plan_.partition_tokens != 0)")
        inline = ARENA.index("if (inline_stock_long_metadata) {", split)
        block = ARENA[inline:ARENA.index("encoder.set_output_array(scratch, 12)", inline)]
        self.assertIn("encoder.set_bytes(items.data(), static_cast<int>(items.size()), static_cast<int>(field + 4))", block)
        self.assertIn("for (int input = 2; input < 10; ++input)", block)
        dispatch = ARENA.index("encoder.dispatch_threadgroups(MTL::Size(split_plan_.partitions", inline)
        self.assertGreater(ARENA.index("record_q1_stock_long_metadata_dispatch()", dispatch), dispatch)
        self.assertIn('"_inline_metadata"', ARENA)
        for path in (NATIVE / "arena.h", NATIVE / "binding.cpp"):
            self.assertIn("q1_stock_long_metadata_dispatch_count", path.read_text())

    def test_kernel_arithmetic_source_differs_only_at_eight_address_spaces(self):
        for dim in (128, 256):
            raw = self.run_probe("rawspecial", dim, "float16")
            self.assertEqual(raw.returncode, 0, raw.stderr)
            inline = self.run_probe(dim, "float16")
            self.assertEqual(inline.returncode, 0, inline.stderr)
            self.assertEqual(inline.stdout,
                             raw.stdout.replace("device const uint*", "constant uint*"))
            self.assertEqual(raw.stdout.count("device const uint*"), 8)

    def test_storage_variants_compile_offline_without_gpu(self):
        if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode:
            self.skipTest("offline Metal compiler unavailable")
        for dtype in ("float16", "bfloat16"):
            for dim in (128, 256):
                with self.subTest(dtype=dtype, dim=dim):
                    result = self.run_probe(dim, dtype)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.count("constant uint*"), 8)
                    self.assertNotIn("device const uint*", result.stdout)
                    source = self.folder / f"{dtype}-{dim}.metal"
                    source.write_text(result.stdout)
                    compiled = subprocess.run(["xcrun", "-sdk", "macosx", "metal",
                                               "-std=metal3.2", "-c", str(source),
                                               "-o", str(source.with_suffix(".air"))],
                                              text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)


if __name__ == "__main__":
    unittest.main()
