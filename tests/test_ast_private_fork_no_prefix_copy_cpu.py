"""F2: the AST private fork appends into the lane's KV buffers, no prefix copy.

``_copy_ast_private_cache`` sliced K/V to ``offset``, so every round's first
private append concat-grew the whole committed prefix (2x KV peak, O(context)
per round per attention layer).  Append-only caches now lend their buffers to
the fork; the lane's committed prefix is never written, so an aborted fork
leaves the lane exact.  Rotating caches, which overwrite old positions, keep
the private copy.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models import cache as cache_module
from mlx2.runtime.models.cache import ArraysCache, KVCache, RotatingKVCache


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _generator():
    fake = SimpleNamespace(binding="fork-test", scheduler_stats={})
    fake._ast_descriptor_capture = (
        lambda cache, recovery=False: ExternalDraftBatchGenerator._ast_descriptor_capture(
            fake, cache, recovery=recovery
        )
    )
    return fake


def _fork(fake, cache):
    return ExternalDraftBatchGenerator._copy_ast_private_cache(fake, cache)


def _rows(seed, count):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((1, 2, count, 4)).astype(np.float32))


def _lane():
    kv = KVCache()
    kv.update_and_fetch(_rows(1, 100), _rows(2, 100))
    recurrent = ArraysCache(2)
    recurrent.cache = [mx.ones((1, 3)), mx.zeros((1, 3))]
    mx.eval(kv.keys, kv.values)
    return [kv, recurrent]


def test_fork_appends_in_place_without_concatenating_the_prefix(monkeypatch):
    fake = _generator()
    lane = _lane()
    capacity = int(lane[0].keys.shape[2])
    assert capacity == 256 and lane[0].offset == 100
    prefix_k = np.array(lane[0].keys[..., :100, :])
    prefix_v = np.array(lane[0].values[..., :100, :])
    private = _fork(fake, lane)
    assert private[0] is not lane[0] and private[0].offset == 100
    concatenations = []
    original = cache_module.mx.concatenate

    def counted(*args, **kwargs):
        concatenations.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(cache_module.mx, "concatenate", counted)
    # A tree tile: sixteen rows written past the committed offset.
    private[0].update_and_fetch(_rows(3, 16), _rows(4, 16))
    mx.eval(private[0].keys, private[0].values)
    assert concatenations == []
    assert int(private[0].keys.shape[2]) == capacity
    assert private[0].keys is lane[0].keys  # lent, never copied
    # Recurrent state is replaced as list entries on the fork only.
    private[1].cache[0] = mx.full((1, 3), 7.0)
    assert np.array_equal(np.array(lane[1].cache[0]), np.ones((1, 3)))

    # Abort: the fork is simply dropped; the lane's committed view is exact.
    del private
    keys, values = lane[0].keys_and_values()
    assert lane[0].offset == 100
    assert np.array_equal(np.array(keys), prefix_k)
    assert np.array_equal(np.array(values), prefix_v)
    # The lane keeps decoding normally over its own view.
    lane[0].update_and_fetch(_rows(5, 1), _rows(6, 1))
    keys, _ = lane[0].keys_and_values()
    assert np.array_equal(np.array(keys[..., :100, :]), prefix_k)
    assert np.array_equal(np.array(keys[..., 100:, :]), np.array(_rows(5, 1)))


def test_rotating_cache_keeps_a_private_copy():
    fake = _generator()
    lane = RotatingKVCache(max_size=8)
    lane.update_and_fetch(_rows(1, 6), _rows(2, 6))
    mx.eval(lane.keys, lane.values)
    before = np.array(lane.keys)
    private = _fork(fake, [lane])[0]
    assert private.keys is not lane.keys
    private.update_and_fetch(_rows(3, 4), _rows(4, 4))
    mx.eval(private.keys)
    assert np.array_equal(np.array(lane.keys), before)
