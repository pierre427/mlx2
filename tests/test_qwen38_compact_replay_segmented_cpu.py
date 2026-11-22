"""Compact GDN replay through the served one-row segmented self-MTP view (CPU).

The Qwen3.8-27B native-MTP route hands every GDN layer a
``SegmentedBatchArraysCache`` view over the lane's ``ArraysCache`` row
(profile ``MLX_LM_SEGMENTED_SELF_MTP=1``), even for a single lane.  This test
drives that exact production stack at the real Qwen3.8 geometry:

* ``Qwen38StatePlaneOperations.authorize_compact_gdn_replay`` on the view
  (the production admission, real tape/snapshot byte accounting);
* ``make_compact_gdn_replay_plan`` / ``accepted_prefix`` / ``cancel``;
* the view's ``record_rollback`` (shared replay + row slicing), ``__setitem__``
  split into the row, ``trim`` -> row ``trim_ragged`` -> re-join, and
  ``resync`` between rounds, as ``hybrid_verify_rows`` does;

against snapshot rollback recorded through the same view and against the
direct recurrence.  What remains test-local: the per-step gated-delta update
(``_step``) and the CPU transcription of the Metal reconstruct kernel
(``_cpu_reconstruct``), so snapshot-versus-compact equality of the *state
values* is largely by construction here.  Metal parity of the replay-verify
and reconstruct kernels with the snapshot kernel rests on
tests/test_qwen4_fused_gdn_verify_metal.py (GPU) and the numpy tape tests in
tests/test_qwen4_gdn_replay_cpu.py.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen_gdn_contracts import QWEN38_GDN_CONTRACT
from mlx2.runtime.models.qwen38_state_plane import QWEN38_STATE_PLANE_OPERATIONS
from mlx2.runtime.segmented_batch_cache import SegmentedBatchArraysCache
from mlx2.runtime.state_plane_operations import StatePlaneOperationRefused

G = QWEN38_GDN_CONTRACT.geometry
HK, HV, DK, DV = G.num_key_heads, G.num_value_heads, G.key_head_dim, G.value_head_dim
RATIO = HV // HK
KEEP = G.conv_kernel - 1
CONV_DIM = 2 * HK * DK + HV * DV
PROVIDER = QWEN38_STATE_PLANE_OPERATIONS


def _bf16(x):
    return mx.array(x.astype(np.float32)).astype(mx.bfloat16)


def _block(rng, steps):
    return {
        "raw": _bf16(rng.normal(size=(1, steps, CONV_DIM))),
        "keys": _bf16(rng.normal(size=(1, steps, HK, DK)) / np.sqrt(DK)),
        "values": mx.array(rng.normal(size=(1, steps, HV, DV)).astype(np.float32)),
        "decay": mx.array(rng.uniform(0.7, 1.0, size=(1, steps, HV)).astype(np.float32)),
        "beta": mx.array(rng.uniform(0.0, 1.0, size=(1, steps, HV)).astype(np.float32)),
    }


def _value_head_keys(keys_t):
    return mx.repeat(keys_t, RATIO, axis=1).astype(mx.float32)[:, :, None, :]


def _step(state, key, value, decay, beta):
    state = state * decay[:, :, None, None]
    recalled = (state * key).sum(axis=-1)
    correction = (value - recalled) * beta[:, :, None]
    return state + key * correction[..., None], correction


def _cpu_reconstruct(state, keys, corrections, decay, accepted, *, threadgroup_y):
    del threadgroup_y
    for t in range(accepted):
        state = state * decay[:, t][:, :, None, None]
        state = state + _value_head_keys(keys[:, t]) * corrections[:, t][..., None]
    return state


def _row(rng):
    row = ArraysCache(2)
    row[0] = _bf16(rng.normal(size=(1, KEEP, CONV_DIM)))
    row[1] = mx.array(rng.normal(size=(1, HV, DV, DK)).astype(np.float32))
    row.start_speculation()
    return row


def _verify(view, block, mode):
    """One speculative block written through ``view`` the way the fused layer does."""
    raw = block["raw"]
    steps = int(raw.shape[1])
    conv0, state0 = view[0], view[1]
    combined = mx.concatenate([conv0, raw], axis=1)
    state, states, corrections = state0, [], []
    for t in range(steps):
        state, correction = _step(
            state, _value_head_keys(block["keys"][:, t]), block["values"][:, t],
            block["decay"][:, t], block["beta"][:, t],
        )
        states.append(state)
        corrections.append(correction)
    mx.eval(states, corrections)
    final_conv = mx.contiguous(combined[:, steps : steps + KEEP])
    if mode == "snapshots":
        conv_snaps = [mx.contiguous(combined[:, p : p + KEEP]) for p in range(1, steps)]
        state_snaps = states[:-1]

        def rollback(m):
            return [conv_snaps[m - 1], state_snaps[m - 1]]
    else:
        # Production admission and plan on the served view.
        auth = PROVIDER.authorize_compact_gdn_replay(view, QWEN38_GDN_CONTRACT, steps)
        assert auth.snapshot_bytes_avoided > auth.tape_bytes > 0
        tape = steps - 1
        plan = PROVIDER.make_compact_gdn_replay_plan(
            auth,
            cache=view,
            raw_qkv=raw,
            replay_keys=block["keys"][:, :tape],
            replay_corrections=mx.stack(corrections[:tape], axis=1),
            replay_decay=block["decay"][:, :tape],
            reconstruct=_cpu_reconstruct,
            threadgroup_y=1,
        )
        assert plan.cancel()[0] is conv0 and plan.cancel()[1] is state0
        rollback = plan.accepted_prefix
    view.record_rollback(steps, rollback, [conv0, state0])
    view[0], view[1] = final_conv, states[-1]
    view.advance(steps)
    return combined, states


def _same(a, b):
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and tuple(x.shape) == tuple(y.shape)
        assert np.array_equal(np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32)))


def _state(cache):
    return [cache[0], cache[1]]


@pytest.mark.parametrize("drafted", [1, 2, 3, 4, 8, 16])
def test_segmented_view_compact_matches_snapshots_and_recurrence(drafted):
    steps = drafted + 1
    for accepted in range(0, steps + 1):
        result = {}
        for mode in ("snapshots", "compact"):
            rng = np.random.default_rng(10_000 * drafted + accepted)
            row = _row(rng)
            view = SegmentedBatchArraysCache([row])
            conv0, state0 = view[0], view[1]
            combined, states = _verify(view, _block(rng, steps), mode)
            view.trim(steps - accepted)
            # The view re-joins from its row: both must hold the same state.
            _same(_state(view), _state(row))
            result[mode] = _state(row)
        _same(result["snapshots"], result["compact"])
        expected = (
            [conv0, state0] if accepted == 0
            else [combined[:, accepted : accepted + KEEP], states[accepted - 1]]
        )
        _same(result["compact"], expected)


def test_segmented_view_multi_round_and_cross_record_rewind():
    rounds = [(4, 2), (3, 0), (5, 5), (4, 3), (9, 1)]
    history = {}
    for mode in ("snapshots", "compact"):
        rng = np.random.default_rng(11)
        row = _row(rng)
        view = SegmentedBatchArraysCache([row])
        for steps, accepted in rounds:
            view.resync(speculating=True)
            _verify(view, _block(rng, steps), mode)
            view.trim(steps - accepted)
            history.setdefault(mode, []).append(_state(row))
        # Two fully accepted blocks, then rewind across both records: all 3 of
        # the newer one plus 2 of the older one (replayed at m = 2).
        for steps in (4, 3):
            view.resync(speculating=True)
            _verify(view, _block(rng, steps), mode)
        assert view.trim(5) == 5
        _same(_state(view), _state(row))
        history[mode].append(_state(row))
    for snap, compact in zip(history["snapshots"], history["compact"]):
        _same(snap, compact)


def test_multi_row_view_is_refused_explicitly_and_rows_untouched():
    rng = np.random.default_rng(5)
    rows = [_row(rng), _row(rng)]
    before = [_state(row) for row in rows]
    view = SegmentedBatchArraysCache(rows)
    with pytest.raises(StatePlaneOperationRefused, match="exactly one row; got 2"):
        PROVIDER.authorize_compact_gdn_replay(view, QWEN38_GDN_CONTRACT, 4)
    for row, (conv, state) in zip(rows, before):
        assert row[0] is conv and row[1] is state
        assert len(row._rollbacks) == 0


def test_segmented_view_with_a_non_exact_row_is_refused():
    rng = np.random.default_rng(6)
    row = _row(rng)
    view = SegmentedBatchArraysCache([row])
    row.cache.append(None)  # a third plane: not the declared two-plane row
    with pytest.raises(StatePlaneOperationRefused, match="exact two-plane ArraysCache row"):
        PROVIDER.authorize_compact_gdn_replay(view, QWEN38_GDN_CONTRACT, 4)


def test_other_cache_types_are_still_refused_with_their_name():
    class Other(ArraysCache):
        pass

    with pytest.raises(StatePlaneOperationRefused, match="got Other"):
        PROVIDER.authorize_compact_gdn_replay(Other(2), QWEN38_GDN_CONTRACT, 4)
