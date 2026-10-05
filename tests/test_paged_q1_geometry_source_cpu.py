"""Run directly: native source/compiler contracts without MLX or a device."""
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GeometrySourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="q1-geometry-source-")
        cls.path = Path(cls.folder.name)
        cls.source = re.search(
            r'constexpr const char\* kAttentionReadQ1TileSource = R"metal\((.*?)\)metal";',
            (ROOT / "native/paged_kv/arena.cpp").read_text(), re.S).group(1)
        cpp = cls.path / "probe.cpp"
        cpp.write_text(r'''
#include "q1_geometry.h"
#include "q1_metadata.h"
#include <iostream>
#include <iterator>
#include <cstdlib>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  try {
    if (argc == 2) { std::cout << q1_simd_stripes(argv[1]); return 0; }
    const std::string source((std::istreambuf_iterator<char>(std::cin)), {});
    auto text = specialized_q1_source(source.c_str(), std::strtoul(argv[1], nullptr, 10),
                                       std::strtoul(argv[2], nullptr, 10));
    if (argc == 4) text = inline_metadata_source(text.c_str());
    std::cout << text;
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

    def invoke(self, *args, source=None):
        return subprocess.run([str(self.probe), *map(str, args)], input=source,
                              text=True, capture_output=True)

    def test_selector_and_unsupported_memory_geometry(self):
        for flag, stripes in [("0", 4), ("4", 4), ("8", 8), ("16", 16), ("32", 32)]:
            self.assertEqual(self.invoke(flag).stdout, str(stripes))
        for flag in ("", "1", "64", "true"):
            self.assertEqual(self.invoke(flag).returncode, 2)
        for dim, stripes in ((64, 8), (256, 32), (128, 4), (128, 64)):
            self.assertEqual(self.invoke(dim, stripes, source=self.source).returncode, 2)

    def test_original_four_stripe_source_is_unchanged(self):
        self.assertEqual(hashlib.sha256(self.source.encode()).hexdigest(),
                         "71b80b5d43992f5eee7031cfbd4ed964c004268e13f2e6bfdf61bb2e3147897d")

    def test_supported_variants_compile_without_a_device(self):
        metal = subprocess.run(["xcrun", "--find", "metal"], capture_output=True)
        if metal.returncode:
            self.skipTest("offline Metal compiler unavailable")
        for dim, stripes in ((128, 8), (128, 16), (128, 32), (256, 8), (256, 16)):
            for inline in (False, True):
                with self.subTest(dim=dim, stripes=stripes, inline=inline):
                    result = self.invoke(dim, stripes, *(["inline"] if inline else []),
                                         source=self.source)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"partial_numerator[{stripes}][{dim}]", result.stdout)
                    self.assertEqual(result.stdout.count("metal::exp(partial_maximum[tile] - global_maximum)"), 1)
                    file = self.path / f"d{dim}-s{stripes}-{inline}.metal"
                    file.write_text(result.stdout)
                    compiled = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2",
                                               "-c", str(file), "-o", str(file.with_suffix(".air"))],
                                              text=True, capture_output=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)

    def test_source_drift_fails_closed(self):
        self.assertEqual(self.invoke(128, 32, source=self.source.replace("token += 4", "token += 7")).returncode, 2)


if __name__ == "__main__":
    unittest.main()
