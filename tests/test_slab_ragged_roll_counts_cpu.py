"""Slab ragged-trim counters: a residual rewind counts one K/V window roll,
an all-equal rewind counts a cursor-only trim, and neither changes the math."""

import mlx.core as mx

from mlx2.runtime.models.cache import BatchKVCache, KVCache, ragged_roll_counts


def _merged(length, dtype=mx.float32):
    rows = []
    for tag in range(2):
        row = KVCache()
        kv = (mx.arange(length, dtype=dtype) + 100 * tag)[None, None, :, None]
        kv = mx.broadcast_to(kv, (1, 2, length, 4))
        row.update_and_fetch(kv, kv)
        rows.append(row)
    return BatchKVCache.merge(rows)


def test_residual_rewind_counts_one_roll_with_window_bytes():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        ragged_roll_counts(reset=True)
        cache = _merged(8)
        assert cache.trim_ragged([1, 3]) == [1, 3]
        mx.eval(cache.state)
        counts = ragged_roll_counts()
        # Uniform part 1 is a cursor move; the residual [0, 2] rolls [0, 7).
        assert counts["roll_calls"] == 1
        assert counts["rows_rolled"] == 1
        assert counts["window_columns"] == 7
        # K and V: 2 rows x 2 heads x 7 columns x 4 dims x 4 bytes each.
        assert counts["bytes"] == 2 * (2 * 2 * 7 * 4 * 4)
        assert "cursor_only_trims" not in counts
        # The roll itself is unchanged: each row keeps its own valid prefix.
        for index, (base, keep) in enumerate(((0, 7), (100, 5))):
            row = cache.extract(index)
            assert row.offset == keep
            got = row.keys[0, 0, :, 0].tolist()
            assert got == [float(base + i) for i in range(keep)]
    finally:
        mx.set_default_device(previous)
        ragged_roll_counts(reset=True)


def test_uniform_rewind_counts_cursor_only_and_zero_rewind_counts_nothing():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        ragged_roll_counts(reset=True)
        cache = _merged(8)
        cache.trim_ragged([0, 0])
        assert ragged_roll_counts() == {}
        cache.trim_ragged([2, 2])
        assert ragged_roll_counts(reset=True) == {"cursor_only_trims": 1}
        assert ragged_roll_counts() == {}
    finally:
        mx.set_default_device(previous)
        ragged_roll_counts(reset=True)
