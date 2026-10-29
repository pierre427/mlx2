"""Exact prefix commits across ring overwrite and pool growth; no target claim."""

import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_cpu import Weights, config
from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows
from mlx2.adapters.deepseek_v41_external import DeepSeekV41VerifyRows
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model


@pytest.mark.parametrize("prefix", [1, 127, 255])
@pytest.mark.parametrize("accepted", range(7))
@pytest.mark.parametrize("mode", ["serial", "layerwise"])
def test_local_frontier_equals_reference_replay_for_every_prefix(prefix, accepted, mode):
    mx.set_default_device(mx.cpu)
    c = config(sliding_window=4)
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-prefix-commit")
    cache = model.make_cache()
    model(mx.array([(np.arange(prefix)%19).tolist()], mx.uint32), cache)
    original = [s.snapshot() for s in cache]
    tokens = mx.array([[1,2,3,4,5,6]], mx.uint32)
    ref = DeepSeekV4VerifyRows(model, [cache], (4,)).begin([6])
    expected, _ = ref.forward(tokens,taps=False)
    ref.commit([accepted])
    expected_state = [s.snapshot() for s in cache]
    for state, saved in zip(cache, original):
        state.restore(saved)
    tx = DeepSeekV41VerifyRows(model, [cache], (4,)).begin([6])
    model.verify_rows_policy = mode
    actual, _ = tx.forward(tokens,taps=False)
    tx.commit([accepted])
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))
    for state, saved in zip(cache, expected_state):
        assert state.meta_state == saved.metadata
        for a,b in zip(state.state, saved.arrays):
            if a is None or b is None:
                assert a is b
            else:
                np.testing.assert_array_equal(np.array(a), np.array(b))
    assert tx.closed and not tx._locals
    following = model(mx.array([[7]],mx.uint32),cache)
    mx.eval(following)
    for state,saved in zip(cache,expected_state):
        state.restore(saved)
    control = model(mx.array([[7]],mx.uint32),cache)
    mx.eval(control)
    np.testing.assert_array_equal(np.array(following),np.array(control))


def test_failed_prefix_restore_aborts_to_complete_base_state():
    mx.set_default_device(mx.cpu)
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-prefix-abort")
    cache = model.make_cache()
    model(mx.array([[1,2,3]], mx.uint32), cache)
    before = [s.snapshot() for s in cache]
    owner = DeepSeekV41VerifyRows(model, [cache], (4,))
    tx = owner.begin([3])
    tx.forward(mx.array([[4,5,6]], mx.uint32))
    cache[0].offset += 1
    with pytest.raises(ValueError, match="frontier"):
        tx.commit([1])
    assert tx.closed and owner._active is None
    for state,saved in zip(cache,before):
        assert state.meta_state == saved.metadata
        for a,b in zip(state.state,saved.arrays):
            if a is not None:
                np.testing.assert_array_equal(np.array(a),np.array(b))


def test_local_checkpoint_reservation_covers_saved_rows_and_restore_copy():
    from mlx2.adapters.deepseek_v41_external import local_frontier_drafter
    from types import SimpleNamespace

    args = SimpleNamespace(vocab_size=129280, hidden_size=5120, num_hidden_layers=40,
                           sliding_window=128, head_dim=512, kv_source_layer_ids=(2,8,14,20))
    model = SimpleNamespace(args=args,layers=range(40))
    drafter = local_frontier_drafter(model,num_draft=5)
    reserved = drafter.extra_memory_reservation([],append=6,pending_tokens=1,prefill=False)
    assert reserved >= 6*40*128*512*4+6*4*512*8
    assert reserved < 64 << 20
    assert drafter.extra_memory_reservation([],append=512,pending_tokens=0,prefill=True) == 0
