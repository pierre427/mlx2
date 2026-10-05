"""Source-bound stock long Q1 contracts; never selects a GPU device."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARENA = (ROOT / "native/paged_kv/arena.cpp").read_text()


class StockLongSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="q1-stock-long-")
        cls.path = Path(cls.folder.name)
        cpp = cls.path / "probe.cpp"
        cpp.write_text(r'''
#include "q1_stock_long.h"
#include "storage_dtype.h"
#include <iostream>
#include <string_view>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  try {
    if (argc == 3 && std::string_view(argv[1]) == "flag") {
      std::cout << q1_stock_long_requested(argv[2]); return 0;
    }
    if (argc == 5 && std::string_view(argv[1]) == "plan") {
      auto p = q1_stock_long_plan(std::stoul(argv[2]), 24,
                                  {uint32_t(std::stoul(argv[3])), uint32_t(std::stoul(argv[4]))});
      std::cout << p.partitions << "," << p.scratch_bytes; return 0;
    }
    auto source = q1_stock_long_source(std::stoul(argv[1]));
    std::cout << storage_specialized_source(source.c_str(),
        storage_dtype_from_name(argv[2]), false, 6);
  } catch (const std::invalid_argument&) { return 2; }
}
''')
        cls.probe = cls.path / "probe"
        subprocess.run([shutil.which("c++") or "c++", "-std=c++20", "-Wall", "-Wextra",
                        "-Werror", "-I", str(ROOT / "native/paged_kv"), str(cpp),
                        "-o", str(cls.probe)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def invoke(self, *args):
        return subprocess.run([str(self.probe), *map(str, args)], capture_output=True, text=True)

    def test_plan_flag_and_bounds(self):
        self.assertEqual(self.invoke("flag", "0").stdout, "0")
        self.assertEqual(self.invoke("flag", "1").stdout, "1")
        self.assertEqual(self.invoke("flag", "true").returncode, 2)
        self.assertEqual(self.invoke("plan", 256, 6929, 6950).stdout, "128,6340608")
        for lengths in ((1024, 1024), (8193, 6950), (0, 6950)):
            self.assertEqual(self.invoke("plan", 256, *lengths).returncode, 2)
        self.assertEqual(self.invoke(64, "float16").returncode, 2)

    def test_old_partition_unchanged_and_physical_proof(self):
        self.assertIn("q1_split_kv_source(dim, partition)", ARENA)
        self.assertIn("q1_stock_long_source(dim)", ARENA)
        self.assertIn("MLX2_PAGED_Q1_STOCK_LONG", ARENA)
        self.assertIn("MLX_SDPA_BLOCKS", ARENA)
        self.assertIn("device.get_architecture().back() != 's'", ARENA)
        self.assertIn("record_q1_stock_long_partial_dispatch()", ARENA)
        self.assertIn("record_q1_stock_long_reduce_dispatch()", ARENA)
        binding = (ROOT / "native/paged_kv/binding.cpp").read_text()
        self.assertIn("q1_stock_long_partial_dispatch_count", binding)
        self.assertIn("q1_stock_long_reduce_dispatch_count", binding)

    def test_fp16_bf16_d128_d256_offline_metal(self):
        if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode:
            self.skipTest("offline Metal compiler unavailable")
        for dtype in ("float16", "bfloat16"):
            for dim in (128, 256):
                with self.subTest(dtype=dtype, dim=dim):
                    result = self.invoke(dim, dtype)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("dense += Q1_BLOCKS", result.stdout)
                    self.assertIn("float(half(numerator[part]))" if dtype == "float16" else
                                  "float(bfloat(numerator[part]))", result.stdout)
                    path = self.path / f"{dtype}-{dim}.metal"
                    path.write_text(result.stdout)
                    compiled = subprocess.run(["xcrun", "-sdk", "macosx", "metal",
                                               "-std=metal3.2", "-c", str(path),
                                               "-o", str(path.with_suffix(".air"))],
                                              text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)


if __name__ == "__main__":
    unittest.main()
