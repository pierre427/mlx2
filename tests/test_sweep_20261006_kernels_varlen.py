"""KR-03 (2026-10-06 sweep): varlen MLP/MoE row compaction declines are
counted, the live-row indices are cached per geometry, and the padding side
is explicit."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import varlen_dense_mlp as v


def _model():
    class M:
        def prefill_row_context(self, lengths, *, width, padding="right"):
            return v.prefill_row_context(self, lengths, width=width, padding=padding)

    model = M()
    return model, v.install(model, v.VarlenSparseMoEPolicy.from_value(True))


def test_geometry_decline_is_counted():
    model, handle = _model()
    with model.prefill_row_context([3, 5], width=8):
        ok = v.compact_rows(mx.zeros((2, 8, 4)), operation="sparse_moe")
        bad = v.compact_rows(mx.zeros((1, 16, 4)), operation="sparse_moe")
    assert ok is not None and bad is None
    counters = v.status(handle)["counters"]
    assert counters["moe_compaction_calls"] == 1
    assert counters["moe_geometry_declines"] == 1


@pytest.mark.parametrize("padding", ["right", "left"])
def test_padding_side_selects_the_live_rows(padding):
    model, _ = _model()
    values = mx.arange(2 * 4 * 1, dtype=mx.float32).reshape(2, 4, 1)
    with model.prefill_row_context([1, 3], width=4, padding=padding):
        packed = v.compact_rows(values, operation="sparse_moe")
        restored = packed.restore(packed.values)
    live = np.array(packed.values).reshape(-1).tolist()
    if padding == "right":
        assert live == [0.0, 4.0, 5.0, 6.0]
    else:
        assert live == [3.0, 5.0, 6.0, 7.0]
    mask = np.zeros((2, 4, 1), dtype=bool)
    flat = mask.reshape(-1)
    flat[[int(x) for x in live]] = True
    want = np.where(mask, np.array(values), 0.0)
    assert np.array_equal(np.array(restored), want)


def test_unknown_padding_side_fails_closed():
    model, _ = _model()
    with pytest.raises(ValueError, match="padding"):
        with model.prefill_row_context([1, 3], width=4, padding="middle"):
            pass


def test_live_row_indices_are_cached_per_geometry():
    v._live_row_indices.cache_clear()
    model, _ = _model()
    for _ in range(3):
        with model.prefill_row_context([2, 4], width=4):
            pass
    info = v._live_row_indices.cache_info()
    assert info.misses == 1 and info.hits == 2
