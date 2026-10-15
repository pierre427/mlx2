"""Selected-KV memory bound and independent attention law, on real CPU arrays."""

import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_exp as model

spec = importlib.util.spec_from_file_location(
    "qsa_tile_probe", Path(__file__).parents[1] / "scripts/probe_qsa_gather_tiles.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("tile", [1, 2, 3, 8])
def test_selected_gather_tiles_preserve_independent_masked_attention_and_bound(monkeypatch, tile):
    data = probe.fixture(dim=8)
    arrays = probe.mlx_inputs(data, mx.float32)
    original_sdpa, original_gather, original_eval = (
        mx.fast.scaled_dot_product_attention, mx.take_along_axis, mx.eval
    )
    events = []

    def gather(array, indices, *, axis):
        if array.ndim != 5:
            return original_gather(array, indices, axis=axis)
        assert indices.shape[2] <= tile
        result = original_gather(array, indices, axis=axis)
        assert result.shape[0] * result.shape[2] <= tile
        events.append("gather")
        return result

    def sdpa(q, k, v, **kwargs):
        assert q.shape[0] <= tile and k.shape[0] <= tile and v.shape[0] <= tile
        events.append("sdpa")
        return original_sdpa(q, k, v, **kwargs)

    def evaluate(*values):
        events.append("eval")
        return original_eval(*values)

    monkeypatch.setattr(mx, "take_along_axis", gather)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", sdpa)
    monkeypatch.setattr(mx, "eval", evaluate)
    actual = model._gather_qsa_attention(*arrays, scale=8**-0.5, tile_rows=tile)
    expected = probe.reference(data)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-6, rtol=2e-6)
    # A tile must retire before the next gather is constructed, rather than
    # retaining every tile lazily until the final concatenation.
    assert events == ["gather", "gather", "sdpa", "eval"] * (2 * ((5 + tile - 1) // tile))
    assert np.count_nonzero(np.asarray(actual)[0, :, 0]) == 0


@pytest.mark.parametrize("tile", [0, -1, True, 1.5])
def test_invalid_tile_bound_is_refused_before_gather(tile, monkeypatch):
    arrays = probe.mlx_inputs(probe.fixture(dim=8), mx.float32)
    monkeypatch.setattr(mx, "take_along_axis", lambda *a, **k: pytest.fail("invalid bound reached gather"))
    with pytest.raises(ValueError, match="positive integer"):
        model._gather_qsa_attention(*arrays, scale=8**-0.5, tile_rows=tile)


def test_declined_indexed_route_reaches_bounded_gather_and_preserves_output(monkeypatch):
    data = probe.fixture(dim=8)
    arrays = probe.mlx_inputs(data, mx.float32)
    actual_sdpa = mx.fast.scaled_dot_product_attention
    calls = []

    def decline(*args, **kwargs):
        raise model.QSAIndexedProbeDeclined("injected supported fallback")

    def bounded_sdpa(q, k, v, **kwargs):
        calls.append(q.shape[0])
        assert q.shape[0] <= 2
        return actual_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(model, "qwen4_qsa_indexed_attention", decline)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", bounded_sdpa)
    out = model._indexed_qsa_attention_or_gather(
        *arrays, scale=8**-0.5, splits=None, tile_rows=2
    )
    np.testing.assert_allclose(np.asarray(out), probe.reference(data), atol=2e-6, rtol=2e-6)
    assert calls == [2, 2, 1, 2, 2, 1]
