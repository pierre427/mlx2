import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import RotatingKVCache
from mlx2.runtime.rotating_replay import (
    RotatingReplayError,
    RotatingReplayPolicy,
    RotatingReplayTransaction,
)


def values(tokens):
    array = mx.array(tokens, dtype=mx.float32).reshape(1, 1, -1, 1)
    return array, array + 100


def append(cache, tokens):
    keys, vals = values(tokens)
    cache.update_and_fetch(keys, vals)


def temporal(cache):
    return np.asarray(cache._temporal_order(cache.keys))


def seeded():
    cache = RotatingKVCache(max_size=4, keep=0)
    for token in range(1, 7):
        append(cache, [token])
    return cache


def test_undo_replay_commits_only_accepted_prefix_after_ring_wrap():
    cache = seeded()
    reference = seeded()
    tx = RotatingReplayTransaction(
        [cache],
        (7, 8, 9),
        request_id="req",
        state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=4),
    )
    for token in (7, 8, 9):
        append(cache, [token])
    receipt = tx.commit(2, lambda accepted: append(cache, accepted))
    append(reference, [7, 8])
    np.testing.assert_array_equal(temporal(cache), temporal(reference))
    assert cache.offset == reference.offset == 8
    assert receipt["accepted_tokens"] == 2
    assert receipt["replayed"] is True


def test_undo_replay_rolls_back_rejected_proposal_exactly():
    cache = seeded()
    before = temporal(cache).copy()
    tx = RotatingReplayTransaction(
        [cache], (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    append(cache, [7, 8])
    tx.rollback()
    np.testing.assert_array_equal(temporal(cache), before)
    assert cache.offset == 6


def test_undo_replay_is_default_off_and_checks_actual_verify_extent():
    cache = seeded()
    with pytest.raises(RotatingReplayError, match="disabled"):
        RotatingReplayTransaction(
            [cache], (7,), request_id="req", state_revision="r1",
            policy=RotatingReplayPolicy(),
        )
    tx = RotatingReplayTransaction(
        [cache], (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    append(cache, [7])
    with pytest.raises(RotatingReplayError, match="does not match"):
        tx.commit(1, lambda accepted: append(cache, accepted))
    append(cache, [8])
    tx.rollback()


def test_failed_accepted_prefix_replay_restores_the_preverify_ring():
    cache = seeded()
    before = temporal(cache).copy()
    tx = RotatingReplayTransaction(
        [cache], (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    append(cache, [7, 8])

    def fail_after_one(_accepted):
        append(cache, [7])
        raise RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        tx.commit(1, fail_after_one)
    np.testing.assert_array_equal(temporal(cache), before)
    assert cache.offset == 6


def test_multi_cache_rollback_failure_restores_every_verified_cache(monkeypatch):
    caches = [seeded(), seeded()]
    tx = RotatingReplayTransaction(
        caches, (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    for cache in caches:
        append(cache, [7, 8])
    verified = [temporal(cache).copy() for cache in caches]
    original_trim = RotatingKVCache.trim
    calls = 0

    def fail_second(cache, count):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected trim failure")
        return original_trim(cache, count)

    monkeypatch.setattr(RotatingKVCache, "trim", fail_second)
    with pytest.raises(RotatingReplayError, match="not atomic"):
        tx.rollback()
    for cache, expected in zip(caches, verified):
        np.testing.assert_array_equal(temporal(cache), expected)
        assert cache.offset == 8
        assert cache.speculating

    monkeypatch.setattr(RotatingKVCache, "trim", original_trim)
    tx.rollback()
    assert all(cache.offset == 6 for cache in caches)


def test_multi_cache_failed_replay_restores_all_layers_without_trimming(monkeypatch):
    caches = [seeded(), seeded()]
    before = [temporal(cache).copy() for cache in caches]
    tx = RotatingReplayTransaction(
        caches, (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    for cache in caches:
        append(cache, [7, 8])

    def no_trim(*_args, **_kwargs):
        raise RuntimeError("trim must not be needed for replay recovery")

    def fail_after_partial_replay(_accepted):
        append(caches[0], [7])
        append(caches[1], [7])
        monkeypatch.setattr(RotatingKVCache, "trim", no_trim)
        raise RuntimeError("replay failed")

    with pytest.raises(RuntimeError, match="replay failed"):
        tx.commit(1, fail_after_partial_replay)
    for cache, expected in zip(caches, before):
        np.testing.assert_array_equal(temporal(cache), expected)
        assert cache.offset == 6
        assert not cache.speculating


def test_commit_verified_publishes_whole_proposal_without_replay():
    cache = seeded()
    reference = seeded()
    tx = RotatingReplayTransaction(
        [cache], (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    append(cache, [7, 8])
    receipt = tx.commit_verified()
    append(reference, [7, 8])
    np.testing.assert_array_equal(temporal(cache), temporal(reference))
    assert cache.offset == 8 and not cache.speculating
    assert receipt["accepted_tokens"] == receipt["proposed_tokens"] == 2
    assert receipt["replayed"] is False
    with pytest.raises(RotatingReplayError, match="closed"):
        tx.commit_verified()


def test_commit_verified_refuses_a_mismatched_verify_advance():
    cache = seeded()
    tx = RotatingReplayTransaction(
        [cache], (7, 8), request_id="req", state_revision="r1",
        policy=RotatingReplayPolicy(enabled=True, max_proposal_tokens=2),
    )
    append(cache, [7])
    with pytest.raises(RotatingReplayError, match="verify advance"):
        tx.commit_verified()


def test_speculative_trim_in_steps_restores_the_preverify_ring():
    # A partial trim that replays one token must not write into the rollback
    # record it keeps, or a later trim restores a corrupted window.
    cache = seeded()
    for token in (7, 8):
        append(cache, [token])
    before = temporal(cache).copy()
    cache.start_speculation(rollback_window=64)
    append(cache, [100, 101, 102])
    cache.trim(2)
    np.testing.assert_array_equal(temporal(cache)[0, 0, :, 0], [6, 7, 8, 100])
    cache.trim(1)
    np.testing.assert_array_equal(temporal(cache), before)
    assert cache.offset == 8
