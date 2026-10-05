"""CPU tests for actual-boundary capture and bounded logical page export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/varlen_hybrid_fa_boundary_probe.py"
SPEC = importlib.util.spec_from_file_location("hybrid_fa_probe_cpu", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_candidate_callback_keeps_actual_arrays_and_refuses_duplicate():
    probe = MODULE.FABoundaryProbe()
    actual = mx.ones((2, 2, 256), dtype=mx.bfloat16)
    event = {"layer_index": 7, "fa_index": 1, "offsets": (32, 96),
             "queries": actual, "keys": actual, "values": actual,
             "native_attention": actual, "hidden_dtype": mx.bfloat16}
    probe.candidate_callback(event)
    assert probe.candidate[7]["native_attention"] is actual
    with pytest.raises(ValueError, match="duplicated"):
        probe.candidate_callback(event)
    probe.candidate_callback({**event, "layer_index": 11})
    assert set(probe.candidate) == {7}


def test_ordinary_capture_wraps_actual_sdpa_and_requires_all_layers():
    called = []

    def actual(q, k, v, *, cache, scale, mask):
        called.append((q, k, v, mask))
        return q + 1

    module = SimpleNamespace(scaled_dot_product_attention=actual)
    probe = MODULE.FABoundaryProbe()
    q = mx.ones((2, 2, 1, 4), dtype=mx.float16)
    cache = SimpleNamespace(offset=mx.array([33, 97]),
                            left_padding=mx.array([64, 0]))
    with probe.capture_ordinary(module):
        for _ in range(16):
            result = module.scaled_dot_product_attention(
                q, q, q, cache=cache, scale=0.5, mask=mx.array([True]))
            assert bool(mx.array_equal(result, q + 1).item())
    assert module.scaled_dot_product_attention is actual
    assert len(called) == 16 and set(probe.ordinary) == {7, 27}
    assert probe.ordinary[7]["attention"].shape == q.shape
    with pytest.raises(RuntimeError, match="not captured"):
        with MODULE.FABoundaryProbe().capture_ordinary(module):
            module.scaled_dot_product_attention(q, q, q, cache=cache,
                                                 scale=0.5, mask=mx.array([True]))
    assert module.scaled_dot_product_attention is actual


def test_logical_export_reads_only_accepted_page_bytes():
    keys = mx.arange(64 * 2, dtype=mx.float16).reshape(1, 64, 2)
    values = keys + 1
    key_bytes = keys.reshape(-1).view(mx.uint8)
    value_bytes = values.reshape(-1).view(mx.uint8)
    calls = []

    class Backend:
        def diagnostic_read(self, dependency, offset, count, *, permit_diagnostic):
            calls.append((offset, count, permit_diagnostic))
            assert int(dependency[0].item()) == 1
            return key_bytes, value_bytes

    owner = SimpleNamespace(
        _pending=(), _failed=False, writer=SimpleNamespace(
            poisoned=False, backend=Backend()),
        sequence=SimpleNamespace(retained_start=0),
        profile=SimpleNamespace(dtype="float16", kv_heads=1, head_dim=2,
                                page_bytes=256), offset=3,
        accepted_handles=lambda: (SimpleNamespace(page_id=0),))
    got_key, got_value = MODULE._export_logical_native_kv(owner, mx)
    assert tuple(got_key.shape) == (1, 3, 2)
    assert bool(mx.array_equal(got_key, keys[:, :3]).item())
    assert bool(mx.array_equal(got_value, values[:, :3]).item())
    assert calls == [(0, 256, True)]
