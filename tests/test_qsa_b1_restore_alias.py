"""A restored single QSA row shares its ledger until its first private write."""

import ctypes
from unittest.mock import patch

import mlx.core as mx
import pytest

from mlx2.runtime.models.qwen4_exp import BatchQSAKVCache, QSAKVCache


def _address(array):
    return ctypes.addressof(ctypes.c_char.from_buffer(memoryview(array)))


def test_qsa_b1_merge_aliases_ledger_then_detaches_on_append():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        source = QSAKVCache()
        keys = mx.arange(128 * 2, dtype=mx.float32).reshape(1, 1, 128, 2)
        source.update_and_fetch(keys, keys)
        source.update_index_keys(
            mx.arange(128 * 64, dtype=mx.float32).reshape(1, 128, 64)
        )
        mx.eval(source.state)
        original_ledger = mx.array(source.index_keys)
        original_keys = mx.array(source.keys)

        batch = BatchQSAKVCache.merge([source])
        mx.eval(batch.state)
        assert _address(batch.index_keys) == _address(source.index_keys)
        assert batch._index_owned is False
        with patch(
            "mlx2.runtime.models.qwen4_exp.record_verify_sync",
            side_effect=AssertionError("B=1 padding was read back from the device"),
        ):
            assert batch.max_left_padding() == 0

        next_kv = mx.full((1, 1, 1, 2), 7, dtype=mx.float32)
        next_index = mx.full((1, 1, 64), 9, dtype=mx.float32)
        batch.update_and_fetch(next_kv, next_kv)
        batch.update_index_keys(next_index)
        mx.eval(batch.state)
        assert _address(batch.index_keys) != _address(source.index_keys)
        assert batch._idx == batch.index_keys.shape[1] == 129
        assert mx.array_equal(source.index_keys, original_ledger).item()
        assert mx.array_equal(source.keys, original_keys).item()
        assert mx.array_equal(batch.index_keys[:, -1:], next_index).item()
        assert mx.array_equal(batch.keys[..., 128:129, :], next_kv).item()

        source.index_keys = source.index_keys[:, :-1]
        with pytest.raises(RuntimeError, match="ledger holds 127 positions"):
            BatchQSAKVCache.merge([source])
    finally:
        mx.set_default_device(previous)


def test_qsa_merge_seeds_host_padding_mirror_for_cold_and_mixed_rows():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        cold = QSAKVCache()
        warm = QSAKVCache()
        row = mx.ones((1, 1, 3, 2), dtype=mx.float32)
        warm.update_and_fetch(row, row)
        warm.update_index_keys(mx.ones((1, 3, 4), dtype=mx.float32))
        single = BatchQSAKVCache.merge([cold])
        mixed = BatchQSAKVCache.merge([cold, warm])
        with patch(
            "mlx2.runtime.models.qwen4_exp.record_verify_sync",
            side_effect=AssertionError("merge padding was read back from the device"),
        ):
            assert single.max_left_padding() == 0
            assert mixed.max_left_padding() == 3
    finally:
        mx.set_default_device(previous)
