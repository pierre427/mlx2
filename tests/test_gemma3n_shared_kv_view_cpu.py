"""Shared layers may read written KV, never allocator capacity."""

import mlx.core as mx

from mlx2.adapters.mlx_vlm import _Gemma3nCacheView, _gemma3n_cache_views
from mlx2.adapters.mlx_vlm import Gemma3nAdapter
from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache


def test_shared_state_excludes_unwritten_capacity_and_keeps_owner_state():
    cache = BatchKVCache([0, 0])
    written = mx.ones((2, 1, 3, 4))
    keys, values = cache.update_and_fetch(written, written)
    assert keys.shape[2] == values.shape[2] == 3
    assert cache.state[0].shape[2] > 3
    view = _Gemma3nCacheView(cache)
    assert view.state[0].shape[2] == view.state[1].shape[2] == 3
    assert mx.array_equal(view.state[0], keys).item()
    assert cache.state[0].shape[2] > 3
    assert view.offset is cache.offset
    cache.update_and_fetch(written[:, :, :1], written[:, :, :1])
    assert view.state[0].shape[2] == 4


def test_rotating_state_preserves_its_existing_ring_view():
    cache = BatchRotatingKVCache(4, [0])
    written = mx.ones((1, 1, 3, 4))
    cache.update_and_fetch(written, written)
    view = _Gemma3nCacheView(cache)
    assert view.state[0] is cache.state[0]
    assert view.state[1] is cache.state[1]


def test_absent_cache_entries_remain_absent():
    assert _gemma3n_cache_views(None) is None
    assert _gemma3n_cache_views([None]) == [None]


def test_adapter_default_uses_one_row_lane_without_overriding_host_or_operator():
    from mlx2.runtime.lane.policy import resolve

    adapter = Gemma3nAdapter.__new__(Gemma3nAdapter)
    defaults = adapter.lane_policy_defaults()
    detected = {"backend": "mpp", "moe": False, "formats": {"bf16": 502}}
    policy = resolve(detected, adapter=defaults)
    assert policy["min_rows"]["bf16"] == 1
    assert policy["min_rows"]["fp16"] == 1
    assert policy["sources"]["min_rows.bf16"].startswith("adapter:")
    assert resolve({**detected, "backend": "simd"}, adapter=defaults)["mode"] == "off"
    assert resolve(detected, adapter=defaults, overrides={"min_rows": {"bf16": 16}})["min_rows"]["bf16"] == 16
