"""Compact correction-tape replay and full-snapshot rollback leave identical state.

A tiny synthetic gated-delta layer runs speculative verify blocks of ``k``
drafted rows through a real ``ArraysCache`` twice: once recording the per-step
state/conv snapshots the Qwen3.8 fused snapshot verify records, once recording
the compact tape (keys, corrections, decay) through the real Qwen3.8
state-plane plan (``make_compact_gdn_replay_plan`` / ``accepted_prefix`` /
``cancel``).  After every accept/reject the two caches must hold bitwise the
same conv window and recurrent state, equal to a direct recurrence over the
accepted prefix.  The reconstruction is a CPU transcription of
``qwen4_fused_gdn_verify._RECONSTRUCT_SOURCE`` (same per-step order: decay,
then key x correction), so the property under test is the tape contract and
the cache/plan bookkeeping, not Metal arithmetic (the Metal kernels have their
own parity test in tests/test_qwen4_fused_gdn_verify_metal.py).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen_gdn_contracts import QWEN38_GDN_CONTRACT
from mlx2.runtime.models.qwen38_state_plane import (
    CompactGdnReplayAuthorization,
    QWEN38_STATE_PLANE_OPERATIONS,
)

HK, HV, DK, DV = 2, 4, 8, 8
RATIO = HV // HK
KEEP = QWEN38_GDN_CONTRACT.geometry.conv_kernel - 1
CONV_DIM = 2 * HK * DK + HV * DV


def _bf16(x):
    return mx.array(x).astype(mx.bfloat16)


def _block(rng, steps):
    """Random verify-block inputs: raw qkv rows and the recurrence operands."""
    return {
        "raw": _bf16(rng.normal(size=(1, steps, CONV_DIM))),
        # The kernel stores the normalized key in the activation dtype.
        "keys": _bf16(rng.normal(size=(1, steps, HK, DK)) / np.sqrt(DK)),
        "values": mx.array(rng.normal(size=(1, steps, HV, DV)).astype(np.float32)),
        "decay": mx.array(rng.uniform(0.7, 1.0, size=(1, steps, HV)).astype(np.float32)),
        "beta": mx.array(rng.uniform(0.0, 1.0, size=(1, steps, HV)).astype(np.float32)),
    }


def _value_head_keys(keys_t):
    """(1, HK, DK) -> (1, HV, 1, DK): value head hv reads key head hv // RATIO."""
    return mx.repeat(keys_t, RATIO, axis=1).astype(mx.float32)[:, :, None, :]


def _step(state, key, value, decay, beta):
    """One kernel-order gated-delta step; returns (state, correction)."""
    state = state * decay[:, :, None, None]
    recalled = (state * key).sum(axis=-1)
    correction = (value - recalled) * beta[:, :, None]
    return state + key * correction[..., None], correction


def _cpu_reconstruct(state, keys, corrections, decay, accepted, *, threadgroup_y):
    """CPU transcription of _RECONSTRUCT_SOURCE (decay, then key x correction)."""
    del threadgroup_y
    for t in range(accepted):
        state = state * decay[:, t][:, :, None, None]
        state = state + _value_head_keys(keys[:, t]) * corrections[:, t][..., None]
    return state


def _verify(cache, block, mode):
    """Advance ``cache`` by a speculative block, staging ``mode``'s rollback."""
    raw = block["raw"]
    steps = int(raw.shape[1])
    conv0, state0 = cache[0], cache[1]
    combined = mx.concatenate([conv0, raw], axis=1)
    state, states, corrections = state0, [], []
    for t in range(steps):
        state, correction = _step(
            state,
            _value_head_keys(block["keys"][:, t]),
            block["values"][:, t],
            block["decay"][:, t],
            block["beta"][:, t],
        )
        states.append(state)
        corrections.append(correction)
    final_conv = mx.contiguous(combined[:, steps : steps + KEEP])
    if mode == "snapshots":
        conv_snaps = [mx.contiguous(combined[:, p : p + KEEP]) for p in range(1, steps)]
        state_snaps = states[:-1]
        mx.eval(conv_snaps, state_snaps)

        def rollback(m):
            return [conv_snaps[m - 1], state_snaps[m - 1]]
    else:
        tape = steps - 1
        auth = CompactGdnReplayAuthorization(
            QWEN38_STATE_PLANE_OPERATIONS.provider_revision,
            QWEN38_GDN_CONTRACT.identity,
            steps,
            tape_bytes=1,
            snapshot_bytes_avoided=1,
        )
        plan = QWEN38_STATE_PLANE_OPERATIONS.make_compact_gdn_replay_plan(
            auth,
            cache=cache,
            raw_qkv=raw,
            replay_keys=block["keys"][:, :tape].reshape(1, tape, HK * DK),
            replay_corrections=mx.stack(corrections[:tape], axis=1),
            replay_decay=block["decay"][:, :tape],
            reconstruct=lambda s, k, c, d, m, *, threadgroup_y: _cpu_reconstruct(
                s, k.reshape(1, -1, HK, DK), c, d, m, threadgroup_y=threadgroup_y
            ),
            threadgroup_y=1,
        )
        assert plan.cancel()[0] is conv0 and plan.cancel()[1] is state0
        rollback = plan.accepted_prefix
    cache.record_rollback(steps, rollback, [conv0, state0])
    cache[0], cache[1] = final_conv, states[-1]
    cache.advance(steps)
    return combined, states


def _fresh_cache(rng):
    cache = ArraysCache(2)
    cache[0] = _bf16(rng.normal(size=(1, KEEP, CONV_DIM)))
    cache[1] = mx.array(rng.normal(size=(1, HV, DV, DK)).astype(np.float32))
    cache.start_speculation()
    return cache


def _assert_same(a, b):
    for x, y in zip((a[0], a[1]), (b[0], b[1])):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32)))


@pytest.mark.parametrize("drafted", [1, 2, 3, 4, 8, 16])
def test_every_accept_reject_of_k_drafts_matches_snapshots_bitwise(drafted):
    steps = drafted + 1  # verify block = anchor + k drafts, 2..17 like the kernel
    for accepted in range(0, steps + 1):
        seed = 1000 * drafted + accepted
        caches = {}
        for mode in ("snapshots", "compact"):
            rng = np.random.default_rng(seed)
            cache = _fresh_cache(rng)
            conv0, state0 = cache[0], cache[1]
            combined, states = _verify(cache, _block(rng, steps), mode)
            cache.trim(steps - accepted)
            caches[mode] = cache
        _assert_same(caches["snapshots"], caches["compact"])
        # And both equal the direct recurrence over the accepted prefix.
        expected_state = state0 if accepted == 0 else states[accepted - 1]
        expected_conv = conv0 if accepted == 0 else combined[:, accepted : accepted + KEEP]
        _assert_same(caches["compact"], [expected_conv, expected_state])
        if accepted == 0:
            # A full rejection publishes the untouched checkpoint objects.
            assert caches["compact"][0] is conv0 and caches["compact"][1] is state0


def test_multi_round_decode_and_cross_record_rewind_match_snapshots():
    """Several verify rounds, partial accepts, then a rewind across records."""
    rounds = [(4, 2), (3, 0), (5, 5), (4, 3), (6, 1)]  # (verify width, accepted)
    caches = {}
    for mode in ("snapshots", "compact"):
        rng = np.random.default_rng(7)
        cache = _fresh_cache(rng)
        for steps, accepted in rounds:
            _verify(cache, _block(rng, steps), mode)
            cache.trim(steps - accepted)
            caches.setdefault(mode, []).append((cache[0], cache[1]))
        # Two more blocks fully accepted, then rewind 5 = all of the last (3)
        # plus 2 of the one before (4): the older record replays at m = 2.
        for steps in (4, 3):
            _verify(cache, _block(rng, steps), mode)
        assert cache.trim(5) == 5
        caches[mode].append((cache[0], cache[1]))
    for snap, compact in zip(caches["snapshots"], caches["compact"]):
        _assert_same(snap, compact)


def test_accepted_prefix_outside_tape_is_refused():
    rng = np.random.default_rng(3)
    cache = _fresh_cache(rng)
    _verify(cache, _block(rng, 4), "compact")
    record = cache._rollbacks[-1]
    with pytest.raises(ValueError, match="outside"):
        record.fn(4)
    with pytest.raises(ValueError, match="outside"):
        record.fn(0)
