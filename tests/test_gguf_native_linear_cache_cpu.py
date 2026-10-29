# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import numpy as np
import pytest
from test_gguf_packed_cpu import fixture

from mlx2.runtime.gguf.weights import GGUFWeights


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def provider(tmp_path, kind):
    rng = np.random.default_rng(983)
    value = rng.normal(size=(24, 32)).astype(np.float32)
    if kind == 1:
        raw = value.astype("<f2").tobytes()
    elif kind == 0:
        raw = value.astype("<f4").tobytes()
    else:
        raw = np.array(mx.array(value).astype(mx.bfloat16).view(mx.uint16)).tobytes()
    return GGUFWeights(
        fixture(tmp_path / "native.gguf", raw, kind=kind, shape=(32, 24))
    )


@pytest.mark.parametrize("kind", [0, 1, 30])
@pytest.mark.parametrize("width", [1, 6, 128])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32, mx.bfloat16])
def test_exact_old_law_all_dtypes_widths(tmp_path, kind, width, dtype):
    w = provider(tmp_path, kind)
    try:
        raw = w.array("weight")
        linear = w.linear("weight")
        x = mx.array(np.random.default_rng(width).normal(size=(width, 32))).astype(
            dtype
        )
        expected = x @ raw.T
        actual = linear(x)
        mx.eval(actual, expected)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)), np.array(expected.astype(mx.float32))
        )
        assert raw.dtype == {0: mx.float32, 1: mx.float16, 30: mx.bfloat16}[kind]
        first = w.native_fp32_parameters()["weight"]
        assert w.linear("weight") is not None
        assert w.native_fp32_parameters()["weight"] is first
        assert w.native_linear_fp32_status()["additional_bytes"] == (
            0 if kind == 0 else 24 * 32 * 4
        )
    finally:
        w.close()
    assert w.native_fp32_parameters() == {}


def test_promotion_graph_once_after_preload(tmp_path, monkeypatch):
    w = provider(tmp_path, 1)
    try:
        raw = w.array("weight")
        mx.eval(raw)
        oldgraphs = []
        for i in range(2):
            x = mx.array(np.full((1, 32), i + 1, np.float32))
            y = x @ raw.T
            p = tmp_path / f"old{i}.dot"
            mx.export_to_dot(str(p), y)
            oldgraphs.append(p.read_text())
            mx.eval(y)
        assert all(g.count('label ="AsType"') == 1 for g in oldgraphs)
        linear = w.linear("weight")
        promoted = w.native_fp32_parameters()["weight"]
        assert promoted.shape == (32, 24)
        before = tmp_path / "preload-operand.dot"
        mx.export_to_dot(str(before), promoted, raw_weight=raw)
        graph = before.read_text()
        # The retained operand matches original Transpose→AsType construction,
        # not a base-cast followed by transpose with potentially different layout.
        assert graph.index('label ="Transpose"') < graph.index('label ="AsType"')
        captured = []
        original = mx.eval

        def evaluate(*arrays):
            captured.extend(arrays)
            return original(*arrays)

        monkeypatch.setattr(mx, "eval", evaluate)
        w.preload_native_linear_fp32()
        assert (
            captured and captured[0]["weight"] is w.native_fp32_parameters()["weight"]
        )
        for i in range(2):
            x = mx.array(np.full((1, 32), i + 1, np.float32))
            y = linear(x)
            p = tmp_path / f"cached{i}.dot"
            mx.export_to_dot(str(p), y)
            assert "AsType" not in p.read_text()
            original(y)
    finally:
        w.close()


def test_embedding_and_quantized_not_registered(tmp_path, monkeypatch):
    w = provider(tmp_path, 1)
    try:
        w.embedding(mx.array([[1, 2]], dtype=mx.int32), name="weight")
        assert w.native_fp32_parameters() == {}
    finally:
        w.close()


def test_quantized_linear_stays_packed(tmp_path):
    path = fixture(tmp_path / "quantized.gguf", bytes(34 * 24), kind=8, shape=(32, 24))
    w = GGUFWeights(path)
    try:
        w.linear("weight")
        assert w.native_fp32_parameters() == {}
        assert w._dense == {}
        assert w._packed["weight"].data.nbytes == 34 * 24
    finally:
        w.close()


def test_owned_probe_case_cpu_contract(monkeypatch):
    import importlib.util
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / "scripts/research"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "native_cache_fixture_probe", directory / "probe_gguf_native_linear_cache.py"
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    guards = []
    cases = probe.run_case(32, 24, lambda: guards.append(True))
    assert len(guards) == 4
    assert [c["rows"] for c in cases] == [1, 6, 128]
    assert all(c["bitwise_equal"] and c["numerical"]["passed"] for c in cases)
    assert all(c["cached_shape"] == [32, 24] for c in cases)
    with pytest.raises(SystemExit):
        probe.main(["--output", "unused-evidence-path"])
