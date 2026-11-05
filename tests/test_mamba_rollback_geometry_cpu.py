"""Synthetic CPU Mamba masked state/rollback contracts; no checkpoint loaded."""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import ArraysCache
from test_nemotron_external_taps_cpu import tiny_model


def _mixer():
    model = tiny_model()
    model.eval()
    return next(layer.mixer for layer in model.layers if layer.block_type == "M")


def _close_cache(actual, expected):
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=2e-6)


def test_leading_pad_forward_keeps_correct_state_without_false_rollback():
    mixer = _mixer()
    rows = mx.random.normal((1, 4, 8))
    cache = ArraysCache(2, left_padding=[2])
    cache.start_speculation()
    out = mixer(rows, mask=cache.make_mask(4), cache=cache)
    reference = ArraysCache(2, left_padding=[2])
    expected = mixer(rows, mask=reference.make_mask(4), cache=reference)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    _close_cache(cache, reference)
    assert not cache._rollbacks
    with pytest.raises(RuntimeError, match="exact rollback"):
        cache.trim(1)


def test_unexplained_mask_invalidates_old_mamba_replay_chain():
    mixer = _mixer()
    cache = ArraysCache(2)
    mixer(mx.random.normal((1, 3, 8)), mask=None, cache=cache)
    cache.start_speculation()
    mixer(mx.random.normal((1, 2, 8)), mask=None, cache=cache)
    assert cache._rollbacks
    marker = cache.rollback_marker()
    reference = ArraysCache(2)
    reference.cache = list(cache)
    rows = mx.random.normal((1, 3, 8))
    mask = mx.array([[True, False, True]])
    out = mixer(rows, mask=mask, cache=cache)
    expected = mixer(rows, mask=mask, cache=reference)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    _close_cache(cache, reference)
    assert not cache._rollbacks
    with pytest.raises(RuntimeError, match="stale epoch"):
        cache.rewind_to_rollback_marker(marker)


def test_ragged_mamba_records_only_valid_tokens_and_restores_all_planes():
    mixer = _mixer()
    prefix = mx.random.normal((2, 3, 8))
    rows = mx.random.normal((2, 5, 8))
    cache = ArraysCache(2)
    mixer(prefix, mask=None, cache=cache)
    cache.start_speculation()
    cache.prepare(lengths=[2, 5])
    mixer(rows, mask=cache.make_mask(5), cache=cache)
    assert cache._rollbacks[-1].depth_list(2) == [2, 5]
    cache.trim_ragged([1, 2])
    cache.finalize()
    for row, keep in enumerate((1, 3)):
        reference = ArraysCache(2)
        mixer(mx.concatenate([prefix[row:row + 1], rows[row:row + 1, :keep]], axis=1),
              mask=None, cache=reference)
        _close_cache(cache.extract(row), reference)


@pytest.mark.parametrize("checkpoint_tape", [False, True])
@pytest.mark.parametrize("keep", [0, 1, 4])
def test_sink_rollback_restores_linear_and_checkpoint_conv_tapes(checkpoint_tape, keep, monkeypatch):
    from mlx2.runtime.models.nemotron_h import Model

    mixer = _mixer()
    if checkpoint_tape:
        ordinary_conv = mixer._conv

        def checkpoint_conv(value, cache, mask, capture_window=False):
            out, padded = ordinary_conv(value, cache, mask, capture_window)
            if capture_window:
                width = mixer.conv_kernel_size - 1
                padded = mx.stack([padded[:, i:i + width]
                                   for i in range(value.shape[1] + 1)], axis=1)
            return out, padded

        monkeypatch.setattr(mixer, "_conv", checkpoint_conv)
    prefix, rows = mx.random.normal((1, 3, 8)), mx.random.normal((1, 5, 8))
    cache, reference = ArraysCache(2), ArraysCache(2)
    mixer(prefix, mask=None, cache=cache)
    mixer(prefix, mask=None, cache=reference)
    sink = []
    mixer(rows, mask=None, cache=cache, ssm_sink=sink)
    Model.rollback_speculative_cache(None, [cache], sink, keep, 5)
    if keep:
        mixer(rows[:, :keep], mask=None, cache=reference)
    _close_cache(cache, reference)
    following = mx.random.normal((1, 1, 8))
    np.testing.assert_allclose(
        np.asarray(mixer(following, mask=None, cache=cache)),
        np.asarray(mixer(following, mask=None, cache=reference)), rtol=1e-5, atol=2e-6,
    )


@pytest.mark.parametrize("keep,block", [(-1, 4), (5, 4), (True, 4), (0, -1)])
def test_sink_rejects_invalid_acceptance_before_mutating_kv(keep, block):
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.nemotron_h import Model

    cache = KVCache()
    cache.update_and_fetch(mx.ones((1, 1, 8, 2)), mx.ones((1, 1, 8, 2)))
    with pytest.raises(ValueError, match="accepted prefix"):
        Model.rollback_speculative_cache(None, [cache], [], keep, block)
    assert cache.offset == 8


def test_sink_rejects_missing_recurrent_tape_before_mutating_kv():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.nemotron_h import Model

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 1, 8, 2)), mx.ones((1, 1, 8, 2)))
    recurrent = ArraysCache(2)
    recurrent.cache = [mx.zeros((1, 2, 16)), mx.zeros((1, 2, 4, 4))]
    with pytest.raises(ValueError, match="ssm_states"):
        Model.rollback_speculative_cache(None, [kv, recurrent], [], 1, 4)
    assert kv.offset == 8


@pytest.mark.parametrize("steps,expected", [(4, [1, 3]), (12, [1, 8, 3])])
def test_empty_state_exact_strip_initializes_with_one_ordinary_token(steps, expected, monkeypatch):
    from mlx2.runtime.models import nemotron_h

    calls = []

    def step(x, a_log, b, c, d, dt, dt_bias, state, limit, mask):
        calls.append(x.shape[1])
        if len(calls) == 1:
            assert state is None
        else:
            assert state is not None
        return x, mx.zeros((1, 2, 4, 4))

    monkeypatch.setattr(nemotron_h, "ssm_update", step)
    x = mx.ones((1, steps, 2, 4))
    b = c = mx.ones((1, steps, 1, 4))
    out, state = nemotron_h._serial_exact_ssm_update(
        x, mx.zeros(2), b, c, mx.ones(2), mx.zeros((1, steps, 2)),
        mx.zeros(2), None, (0., float("inf")), None,
    )
    assert calls == expected
    assert out.shape == x.shape and state.shape == (1, 2, 4, 4)


@pytest.mark.parametrize("width,accepted", [(7, 3), (20, 18), (40, 18)])
def test_masked_exact_strip_matches_ordinary_accepted_prefix_and_continuation(width, accepted):
    from mlx2.runtime.models.nemotron_h import _serial_exact_ssm_update
    from mlx2.runtime.models.ssm import ssm_update

    mx.random.seed(11304)
    x = mx.random.normal((1, width, 2, 4)) * .1
    b, c = [mx.random.normal((1, width, 1, 4)) * .1 for _ in range(2)]
    dt = mx.random.normal((1, width, 2)) - 3
    state = mx.random.normal((1, 2, 4, 4)) * .1
    a_log, d, bias, limit = mx.zeros(2), mx.ones(2), mx.zeros(2), (0., float("inf"))
    mask = mx.arange(width)[None, :] < accepted
    out, candidate = _serial_exact_ssm_update(x, a_log, b, c, d, dt, bias, state, limit, mask)
    expected = state
    ys = []
    for pos in range(accepted):
        y, expected = ssm_update(x[:, pos:pos + 1], a_log, b[:, pos:pos + 1],
                                c[:, pos:pos + 1], d, dt[:, pos:pos + 1], bias, expected, limit)
        ys.append(y)
    np.testing.assert_array_equal(np.asarray(candidate), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(out[:, :accepted]), np.asarray(mx.concatenate(ys, axis=1)))
    next_args = (x[:, :1], a_log, b[:, :1], c[:, :1], d, dt[:, :1], bias)
    actual_y, actual_state = ssm_update(*next_args, candidate, limit)
    expected_y, expected_state = ssm_update(*next_args, expected, limit)
    np.testing.assert_array_equal(np.asarray(actual_y), np.asarray(expected_y))
    np.testing.assert_array_equal(np.asarray(actual_state), np.asarray(expected_state))
