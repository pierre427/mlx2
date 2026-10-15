"""LFM2.5's single-array recurrent plane survives an APCv2 warm turn."""

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache

mx.set_default_device(mx.cpu)


def _lfm_cache(length):
    recurrent = ArraysCache(1)
    recurrent[0] = mx.arange(12, dtype=mx.float32).reshape(1, 3, 4)
    recurrent.lengths = mx.array([length], dtype=mx.int32)
    recurrent._host_lengths = (recurrent.lengths, [length])
    recurrent.state_checkpoint([length], force=True)
    attention = KVCache()
    values = mx.arange(length, dtype=mx.float32).reshape(1, 1, length, 1)
    attention.update_and_fetch(values, values)
    mx.eval(recurrent.state, attention.state)
    return [recurrent, attention]


def test_single_array_recurrent_multi_turn_warm_hit_reuses_tokens():
    apc = APCv2(max_size=4, layout_name="lfm25-single-array-regression-v1")
    key = APCKey("lfm25-regression")
    first_turn = list(range(32))
    capability = apc.store(key, first_turn, _lfm_cache(len(first_turn)))
    assert capability.stored and capability.exact_prefix

    warm = apc.lookup(key, first_turn + [97, 98, 99])
    try:
        assert warm.hit and warm.hit_kind == "prefix"
        assert warm.cached_tokens == len(first_turn)
        assert warm.remaining_tokens == [97, 98, 99]
        assert len(warm.cache) == 2
        assert isinstance(warm.cache[0], ArraysCache)
        assert len(warm.cache[0].cache) == 1
        assert warm.cache[0].lengths.item() == len(first_turn)
        assert warm.cache[1].offset == len(first_turn)
    finally:
        warm.cache.close()
        apc.close()
