"""Compile and execute the actual metadata selector/source adapter without MLX."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "native/paged_kv/arena.cpp"
HASHES = {
    "kAttentionReadSource": "892ae8e2dec62a613005a4ef476d2d3f3b40a28fc5a4c6d3f888f17c6c620b17",
    "kAttentionReadQ1TileSource": "71b80b5d43992f5eee7031cfbd4ed964c004268e13f2e6bfdf61bb2e3147897d",
}


@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    folder = tmp_path_factory.mktemp("inline_metadata")
    program = folder / "probe.cpp"
    program.write_text(r'''
#include "q1_metadata.h"
#include <iostream>
#include <iterator>
#include <cstdlib>
int main(int argc, char** argv) {
  using namespace mlx2::paged_kv;
  if (argc == 6) {
    std::cout << use_inline_q1_metadata(
        std::string_view(argv[1]) == "NULL" ? nullptr : argv[1],
        std::strtoull(argv[2], nullptr, 10) != 0,
        std::strtoull(argv[3], nullptr, 10),
        std::strtoull(argv[4], nullptr, 10),
        std::strtoull(argv[5], nullptr, 10));
    return 0;
  }
  const std::string source((std::istreambuf_iterator<char>(std::cin)), {});
  try { std::cout << inline_metadata_source(source.c_str()); }
  catch (const std::invalid_argument&) { return 2; }
}
''')
    executable = folder / "probe"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "native/paged_kv"), str(program),
                    "-o", str(executable)], check=True, capture_output=True)
    return executable


@pytest.mark.parametrize("flag,short,rows,spans,pages,expected", [
    ("1", 1, 2, 2, 2, "1"), ("1", 1, 2, 2, 6, "1"),
    ("NULL", 1, 2, 2, 2, "0"), ("0", 1, 2, 2, 2, "0"),
    ("true", 1, 2, 2, 2, "0"), ("10", 1, 2, 2, 2, "0"),
    ("1", 0, 2, 2, 2, "0"), ("1", 1, 1, 1, 2, "0"),
    ("1", 1, 3, 3, 3, "0"), ("1", 1, 2, 1, 2, "0"),
    ("1", 1, 2, 2, 1, "0"), ("1", 1, 2, 2, 7, "0"),
    ("1", 1, 2, 2, 4294967295, "0"),
])
def test_actual_cpp_selection_is_default_off_and_bounded(
    helper, flag, short, rows, spans, pages, expected,
):
    result = subprocess.run([str(helper), flag, str(short), str(rows),
                             str(spans), str(pages)], check=True,
                            text=True, capture_output=True)
    assert result.stdout == expected


@pytest.mark.parametrize("name", HASHES)
def test_address_space_adapter_preserves_all_kernel_arithmetic(helper, name):
    source = re.search(r'constexpr const char\* ' + name + r' = R"metal\((.*?)\)metal";',
                       ARENA.read_text(), re.S).group(1)
    # Frozen base b190655ced22 kernel identity, independent of the adapter.
    assert hashlib.sha256(source.encode()).hexdigest() == HASHES[name]
    assert source.count("device const uint*") == 8
    result = subprocess.run([str(helper)], input=source, check=True,
                            text=True, capture_output=True)
    assert result.stdout == source.replace("device const uint*", "constant uint*")
    assert result.stdout.count("constant uint*") == 8


def test_unexpected_metadata_abi_fails_closed(helper):
    result = subprocess.run([str(helper)], input="device const uint* x;",
                            text=True, capture_output=True)
    assert result.returncode == 2


def test_write_dependency_immutable_metadata_and_post_dispatch_counter():
    source = ARENA.read_text()
    assert "const std::vector<std::vector<uint32_t>> inline_metadata_;" in source
    assert "std::vector<mx::array> inputs = {query, dependency};" in source
    binding = source.index("encoder.set_input_array(inputs[1], 3)")
    inline = source.index("if (inline_metadata) {", binding)
    assert "encoder.set_bytes(items.data(), static_cast<int>(items.size())," in source[inline:]
    dispatch = source.index("encoder.dispatch_threads(", inline)
    counter = source.index("arena_->record_q1_metadata_dispatch();", dispatch)
    assert counter > source.index("dispatched->store(true", dispatch)
    # All old span/capacity validation precedes selection and still applies.
    selector = source.index("const bool inline_q1_metadata = split_plan.partition_tokens == 0 && use_inline_q1_metadata(")
    assert selector > source.index('"paged attention rows or pages are unused"')
    assert "short_q1 = short_q1 && count == 1 && visible >= 32 && visible <= 128;" in source
