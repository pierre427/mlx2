# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v4_hc_projection as candidate


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("outputs", [4, 24])
@pytest.mark.parametrize("k", [32, 16384])
def test_reference_and_flat_rows(outputs, k):
    rng = np.random.default_rng(32 + outputs)
    x = rng.normal(size=(1, 6, k)).astype(np.float32)
    w = (rng.normal(size=(outputs, k)) / 128).astype(np.float16).astype(np.float32)
    actual = candidate.reference_project(x, w)
    assert actual.shape == (1, 6, outputs)
    for row in range(6):
        np.testing.assert_allclose(
            actual[:, row : row + 1],
            candidate.reference_project(x[:, row : row + 1], w),
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    "shape,weight",
    [
        ((0, 32), (4, 32)),
        ((2, 31), (4, 31)),
        ((2, 32), (5, 32)),
        ((2, 64), (24, 32)),
        ((2, 32), (4, 32, 1)),
    ],
)
def test_invalid_geometry(shape, weight):
    with pytest.raises(ValueError):
        candidate.validate_geometry(
            SimpleNamespace(shape=shape), SimpleNamespace(shape=weight)
        )


def test_cpu_and_dtype_refusal():
    with pytest.raises(RuntimeError, match="owned Metal"):
        candidate.project(mx.zeros((2, 32)), mx.zeros((4, 32)))
    with pytest.raises(ValueError, match="FP32"):
        candidate.project(mx.zeros((2, 32), dtype=mx.float16), mx.zeros((4, 32)))


def test_dispatch_geometry(monkeypatch):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return [mx.zeros(kwargs["output_shapes"][0])]

    monkeypatch.setattr(candidate, "_kernel", lambda: run)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    result = candidate.project(mx.zeros((1, 6, 32)), mx.zeros((24, 32)))
    assert result.shape == (1, 6, 24)
    assert captured["grid"] == (32, 24, 6)
    assert captured["threadgroup"] == (32, 1, 1)
    assert captured["template"] == [("K", 32), ("OUTPUTS", 24)]


@pytest.mark.parametrize("constant_inputs", [(), ("x",), ("weight",), ("x", "weight")])
@pytest.mark.parametrize("outputs,k", [(4, 32), (24, 32), (4, 16384), (24, 16384)])
def test_all_addressspaces_compile(tmp_path, constant_inputs, outputs, k):
    if (
        not shutil.which("xcrun")
        or subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Metal compiler unavailable")
    source = tmp_path / "hc.metal"
    source.write_text(
        candidate.metal_compile_source(
            k=k, outputs=outputs, constant_inputs=constant_inputs
        )
    )
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "hc.air"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
