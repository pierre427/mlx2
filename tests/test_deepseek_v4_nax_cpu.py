"""CPU compilation only: execution requires an independently owned GPU unit."""

import shutil
import subprocess

import pytest

from mlx2.runtime.gguf.nax_metal import NAX_HEADER, NAX_SOURCE


@pytest.mark.parametrize("format,row_bytes", [(8, 272), (10, 84), (16, 66)])
def test_native_gguf_mpp_kernel_compiles_with_cpu_toolchain(
    tmp_path, format, row_bytes
):
    if (
        not shutil.which("xcrun")
        or subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Metal CPU compiler unavailable")
    source = "#include <metal_stdlib>\nusing namespace metal;\n" + NAX_HEADER
    source += "\nkernel void test_gguf_nax(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device float*y [[buffer(2)]],ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],ushort thread_index_in_threadgroup [[thread_index_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n"
    source += (
        f"constexpr int K=256,N=64,TOKENS=17,ROW_BYTES={row_bytes},FORMAT={format};\n"
    )
    source += NAX_SOURCE + "\n}\n"
    path = tmp_path / "packed_nax.metal"
    path.write_text(source)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "packed_nax.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
