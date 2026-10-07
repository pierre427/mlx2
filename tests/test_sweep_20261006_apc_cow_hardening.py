"""COW-2 and COW-3 hardening (2026-10-06 sweep)."""

import numpy as np
import mlx.core as mx
import pytest

from mlx2.runtime.approximate_kv import (
    KVQuantizationOperation,
    LaneKVState,
    stage_lane_state,
    standard_kv_quantization_operations,
)
from mlx2.runtime.cow_cache import (
    COWCacheError,
    restore_recovery_descriptors,
    snapshot_recovery_descriptors,
)
from mlx2.runtime.models.cache import CacheList, KVCache


def _put(cache, values):
    value = mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)
    cache.update_and_fetch(value, value)
    mx.eval(cache.keys, cache.values)


def test_recovery_restore_refuses_a_prefix_rewritten_after_a_deep_trim():
    # The guard checked only that the fill level was back at the captured
    # level, so trimming to 3 and appending 3 other tokens served
    # [1, 2, 3, 70, 80, 90] as the captured [1..6].
    live = KVCache()
    _put(live, [1, 2, 3, 4, 5, 6])
    snapshot, sidecar, borrowed = snapshot_recovery_descriptors([live])
    live.trim(3)
    _put(live, [70, 80, 90])
    with pytest.raises(COWCacheError, match="rewound"):
        restore_recovery_descriptors(snapshot, sidecar, borrowed)


def test_recovery_restore_still_borrows_after_appends_and_shallow_trims():
    live = KVCache()
    _put(live, [1, 2, 3, 4, 5, 6])
    captured = np.array(live.keys[..., :6, :])
    snapshot, sidecar, borrowed = snapshot_recovery_descriptors([live])
    _put(live, [7, 8, 9])
    live.trim(2)  # a rejected draft: back to 7, still above the level
    restored, _ = restore_recovery_descriptors(snapshot, sidecar, borrowed)
    assert restored[0].offset == 6
    assert np.array_equal(np.array(restored[0].keys[..., :6, :]), captured)


def test_approximate_staging_leaves_a_source_cache_list_exact():
    kv = KVCache()
    value = mx.zeros((1, 1, 8, 64))
    kv.update_and_fetch(value, value)
    source = [CacheList(kv)]
    operation = KVQuantizationOperation(
        "kv_q8",
        standard_kv_quantization_operations()["kv_q8"],
        adapter_fingerprint="fp",
    )
    staged = operation.apply(stage_lane_state(LaneKVState("rev", tuple(source))))
    assert type(source[0].caches[0]) is KVCache
    assert type(staged.planes[0].caches[0]).__name__ == "QuantizedKVCache"
