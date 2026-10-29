"""Grouped NAX independent CPU oracles, geometry capture and offline compilation."""

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import nax_metal
from mlx2.runtime.gguf.reader import GGUFError


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/research"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "grouped_nax_probe_test", directory / "probe_deepseek_v4_grouped_nax.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", [8, 10, 16])
def test_independent_oracle_preserves_groups_and_real_min_offsets(probe, kind):
    x, packed = probe.fixture(1, 256, 32, kind)
    half, full = probe.independent_oracles(x, packed, kind, 32)
    assert half.shape == full.shape == (1, 8, 32)
    assert np.isfinite(half).all() and not np.array_equal(half[:, 0], half[:, 1])
    independent = probe.independent_dequant(packed[3], kind, (32, 256))
    expected = (
        x[:, 3].astype(np.float16).astype(np.float32)
        @ independent.astype(np.float16).astype(np.float32).T
    )
    np.testing.assert_array_equal(half[:, 3], expected)
    if kind == 10:
        blocks = packed.reshape(-1, 84)
        assert np.any(blocks[:, 82:84] != 0)  # Nonzero dmin offset coverage.


def test_full_geometry_memory_bound_is_below_one_gib(probe):
    assert (
        max(probe.working_bound(512, 4096, 1024, 8, kind) for kind in (8, 10, 16))
        < 1 << 30
    )


@pytest.mark.parametrize("leading", [(1,), (16,), (128,), (512,), (1, 16)])
def test_runtime_group_z_geometry_without_gpu(monkeypatch, leading):
    seen = {}

    def fake_kernel(grouped):
        assert grouped

        def invoke(**kwargs):
            seen.update(kwargs)
            return [mx.zeros(kwargs["output_shapes"][0], dtype=mx.float32)]

        return invoke

    # Only guard queries are mocked. Every array is still actually CPU and
    # the Metal kernel factory is replaced, so no GPU work can be submitted.
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M5"})
    monkeypatch.setattr(nax_metal, "_kernel", fake_kernel)
    weight = SimpleNamespace(
        shape=(8, 64, 256),
        ggml_type=8,
        row_bytes=272,
        data=mx.zeros((8 * 64 * 272,), dtype=mx.uint8),
    )
    x = mx.zeros((*leading, 8, 256), dtype=mx.float32)
    y = nax_metal.nax_grouped_prefill(x, weight)
    assert y.shape == (*leading, 8, 64)
    rows = int(np.prod(leading))
    assert seen["grid"] == (128, (rows + 15) // 16, 8)
    assert seen["threadgroup"] == (128, 1, 1)
    assert dict(seen["template"])["GROUPS"] == 8
    assert seen["inputs"][0].shape == (rows, 8, 256)


@pytest.mark.parametrize(
    "shape,kind,input_shape",
    [
        ((8, 64, 256), 0, (1, 8, 256)),
        ((8, 63, 256), 8, (1, 8, 256)),
        ((8, 64, 128), 8, (1, 8, 128)),
        ((8, 64, 256), 8, (1, 7, 256)),
        ((8, 64, 256), 8, (0, 8, 256)),
    ],
)
def test_invalid_geometry_refuses_before_kernel(monkeypatch, shape, kind, input_shape):
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M5"})

    def forbidden(*args):
        raise AssertionError("invalid geometry reached kernel")

    monkeypatch.setattr(nax_metal, "_kernel", forbidden)
    with pytest.raises(GGUFError):
        nax_metal.nax_grouped_prefill(
            mx.zeros(input_shape), SimpleNamespace(shape=shape, ggml_type=kind)
        )


@pytest.mark.parametrize("kind,row_bytes", [(8, 272), (10, 84), (16, 66)])
@pytest.mark.parametrize("rows", [1, 16, 128, 512])
def test_grouped_nax_cpu_metal_compiler(tmp_path, kind, row_bytes, rows):
    if (
        not shutil.which("xcrun")
        or subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Metal CPU compiler unavailable")
    source = "#include <metal_stdlib>\nusing namespace metal;\n" + nax_metal.NAX_HEADER
    source += "\nkernel void fixture(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device float*y [[buffer(2)]],ushort thread_index_in_simdgroup [[thread_index_in_simdgroup]],ushort simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],ushort thread_index_in_threadgroup [[thread_index_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {\n"
    source += f"constexpr int K=256,N=64,GROUPS=8,TOKENS={rows},ROW_BYTES={row_bytes},FORMAT={kind};\n"
    source += nax_metal.GROUPED_NAX_SOURCE + "\n}\n"
    path = tmp_path / "grouped.metal"
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
            str(tmp_path / "grouped.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
