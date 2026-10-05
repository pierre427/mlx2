"""Bounded source and offline compiler checks; no MLX import or GPU dispatch."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ARENA = (ROOT / "native/paged_kv/arena.cpp").read_text()
BASE = re.search(r'constexpr const char\* kAttentionReadQ1TileSource = R"metal\((.*?)\)metal";', ARENA, re.S).group(1)


class StockReductionSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="q1-stock-reduction-")
        cls.path = Path(cls.folder.name)
        src = cls.path / "probe.cpp"
        src.write_text(r'''
#include "q1_stock_reduction.h"
#include "q1_metadata.h"
#include "storage_dtype.h"
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  try {
    if (argc == 3 && std::string_view(argv[1]) == "select") {
      std::cout << use_q1_stock_reduction(argv[2]); return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "padding") {
      std::cout << q1_stock_pair_padding_aligned(
          std::stoul(argv[2]), std::stoul(argv[3])); return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "active") {
      std::cout << q1_stock_reduction_active(std::stoul(argv[2]), std::stoul(argv[3]));
      return 0;
    }
    const std::string base((std::istreambuf_iterator<char>(std::cin)), {});
    auto text = stock_reduction_source(base.c_str());
    text = storage_specialized_source(text.c_str(),
        storage_dtype_from_name(argv[1]), false, 6);
    if (argc == 3) text = inline_metadata_source(text.c_str());
    std::cout << text;
  } catch (const std::invalid_argument&) { return 2; }
}
''')
        cls.probe = cls.path / "probe"
        subprocess.run([shutil.which("c++") or "c++", "-std=c++20", "-Wall", "-Wextra",
                        "-Werror", "-I", str(ROOT / "native/paged_kv"), str(src),
                        "-o", str(cls.probe)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def invoke(self, *args, source=BASE):
        return subprocess.run([str(self.probe), *args], input=source,
                              text=True, capture_output=True)

    def test_selector_and_source_drift_fail_closed(self):
        self.assertEqual(self.invoke("select", "0").stdout, "0")
        self.assertEqual(self.invoke("select", "1").stdout, "1")
        self.assertEqual(self.invoke("select", "yes").returncode, 2)
        self.assertEqual(self.invoke("active", "1", "0").stdout, "0")
        self.assertEqual(self.invoke("active", "1", "1").stdout, "1")
        self.assertEqual(self.invoke("active", "0", "1").stdout, "0")
        for left, right, admitted in ((33, 97, "1"), (32, 96, "1"),
                                      (64, 64, "1"), (63, 129, "0"),
                                      (127, 128, "0"), (0, 64, "0")):
            self.assertEqual(self.invoke("padding", str(left), str(right)).stdout, admitted)
        self.assertEqual(self.invoke("bfloat16", source=BASE.replace("  float q[8];", "  float qq[8];" )).returncode, 2)

    def test_declared_admission_and_dispatch_proof(self):
        self.assertIn("MLX2_PAGED_Q1_STOCK_REDUCTION", ARENA)
        self.assertIn("q1_stock_pair_padding_aligned(visible_pair[0], visible_pair[1])", ARENA)
        self.assertIn("stock_requested, q1_pair && short_q1, singleton_requested, short_singleton", ARENA)
        self.assertIn("stock_reduction ? 32 : requested_stripes", ARENA)
        self.assertIn("pipeline->maxTotalThreadsPerThreadgroup()", ARENA)
        self.assertIn("pipeline->staticThreadgroupMemoryLength()", ARENA)
        self.assertIn("record_q1_stock_reduction_dispatch()", ARENA)
        self.assertIn("q1_stock_reduction_dispatch_count", (ROOT / "native/paged_kv/binding.cpp").read_text())

    def test_fp16_bf16_dynamic_and_inline_compile(self):
        if subprocess.run(["xcrun", "--find", "metal"], capture_output=True).returncode:
            self.skipTest("offline Metal compiler unavailable")
        for dtype in ("float16", "bfloat16"):
            for inline in (False, True):
                with self.subTest(dtype=dtype, inline=inline):
                    generated = self.invoke(dtype, *( ["inline"] if inline else []))
                    self.assertEqual(generated.returncode, 0, generated.stderr)
                    self.assertIn("threadgroup float outputs[32 * 32]", generated.stdout)
                    self.assertIn("fast::exp", generated.stdout)
                    self.assertIn("token += stripes", generated.stdout)
                    self.assertNotIn("partial_numerator", generated.stdout)
                    file = self.path / f"{dtype}-{inline}.metal"
                    file.write_text(generated.stdout)
                    compiled = subprocess.run(["xcrun", "-sdk", "macosx", "metal",
                                               "-std=metal3.2", "-c", str(file),
                                               "-o", str(file.with_suffix(".air"))],
                                              text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)


if __name__ == "__main__":
    unittest.main()
