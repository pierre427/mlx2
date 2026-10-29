"""CPU-only grouped-kernel compilation and dispatch contracts."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from mlx2.runtime.gguf import cooperative_metal as coop
from mlx2.runtime.gguf.reader import GGUFError


@pytest.mark.parametrize("kind,row_bytes", [(8, 272), (16, 66)])
def test_grouped_metal_compilation(tmp_path, kind, row_bytes):
    if (
        not shutil.which("xcrun")
        or subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Metal compiler unavailable")
    source = "#include <metal_stdlib>\nusing namespace metal;\n" + coop.BLOCK_HEADER
    source += "\nkernel void test_group(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device float*y [[buffer(2)]],ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],ushort thread_index_in_threadgroup [[thread_index_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n"
    source += f"constexpr int K=256,N=33,GROUPS=8,ROW_BYTES={row_bytes},FORMAT={kind},ROWS=4;\n"
    source += coop.GROUPED + "\n}\n"
    p = tmp_path / "grouped.metal"
    p.write_text(source)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            "-c",
            str(p),
            "-o",
            str(tmp_path / "grouped.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_grouped_geometry_and_dispatch(monkeypatch):
    import mlx.core as mx

    prior = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        calls = []

        def kernel(kind):
            def dispatch(**kw):
                calls.append((kind, kw))
                return [mx.zeros(kw["output_shapes"][0], dtype=mx.float32)]

            return dispatch

        monkeypatch.setattr(coop, "_kernel", kernel)
        w = SimpleNamespace(
            ggml_type=8,
            shape=(8, 33, 256),
            row_bytes=272,
            data=mx.zeros((8 * 33 * 272,), dtype=mx.uint8),
        )
        x = mx.zeros((2, 3, 8, 256))
        result = coop.cooperative_grouped(x, w)
        assert result.shape == (2, 3, 8, 33)
        kind, kw = calls.pop()
        assert kind == "grouped"
        assert kw["grid"] == (384, 1, 48)
        assert dict(kw["template"])["GROUPS"] == 8
        assert kw["inputs"][0].shape == (48, 256)
        with pytest.raises(GGUFError, match="axes mismatch"):
            coop.cooperative_grouped(mx.zeros((2, 7, 256)), w)
        with pytest.raises(GGUFError, match="rank3"):
            coop.cooperative_grouped(x, w, rows=3)
    finally:
        mx.set_default_device(prior)
