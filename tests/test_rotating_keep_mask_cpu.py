"""Retained rotating-cache prefix tokens stay visible during chunked prefill."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime.models.base import create_causal_mask
from mlx2.runtime.models.cache import RotatingKVCache, make_prompt_cache


def _append(cache, start, count):
    values = mx.arange(start, start + count).reshape(1, 1, count, 1)
    cache.update_and_fetch(values, values)


def test_max_kv_size_cache_keeps_prefix_visible_during_chunked_prefill():
    class Model:
        def __init__(self):
            self.layers = [object()]

    cache = make_prompt_cache(Model(), max_kv_size=8)[0]
    assert cache.keep == 4
    _append(cache, 0, 20)

    mask = cache.make_mask(5)

    assert mask.shape == (5, 12)
    assert mx.all(mask[:, :4]).item()
    assert mask.sum(axis=-1).tolist() == [8] * 5


@pytest.mark.parametrize(
    ("size", "keep", "history", "chunk"),
    [
        (2, 1, 0, 3),
        (4, 1, 4, 3),
        (8, 2, 20, 5),
        (8, 7, 19, 10),
    ],
)
def test_keep_mask_matches_independent_retained_token_reference(
    size, keep, history, chunk
):
    cache = RotatingKVCache(max_size=size, keep=keep)
    _append(cache, 0, history)
    mask = cache.make_mask(chunk, return_array=True)
    keys, _ = cache.update_and_fetch(
        mx.arange(history, history + chunk).reshape(1, 1, chunk, 1),
        mx.arange(history, history + chunk).reshape(1, 1, chunk, 1),
    )
    ids = keys[0, 0, :, 0].tolist()
    rows = mask.tolist()

    for row in range(chunk):
        end = history + row + 1
        visible = sorted(token for token, seen in zip(ids, rows[row]) if seen)
        expected = list(range(min(keep, end))) + list(
            range(max(keep, end - (size - keep)), end)
        )
        assert visible == expected


@pytest.mark.parametrize("return_array", [False, True])
def test_explicit_smaller_attention_window_does_not_force_prefix_visible(
    return_array,
):
    cache = RotatingKVCache(max_size=8, keep=2)
    _append(cache, 0, 9)

    mask = cache.make_mask(4, window_size=3, return_array=return_array)
    expected = create_causal_mask(4, offset=7, window_size=3)

    assert mx.array_equal(mask, expected).item()
