"""Committed journal/receipt ownership, CPU-only; no performance claim."""

import copy
import json

import mlx.core as mx
import pytest
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_ast_tree_cohort_flow_cpu import _cohort

from mlx2.runtime.ast_private_state import freeze_receipt
from mlx2.runtime.cow_cache import (
    CommittedDescriptorPlan,
    restore_recovery_descriptors,
    snapshot_recovery_descriptors,
)
from mlx2.runtime.models.cache import ArraysCache, KVCache, RotatingKVCache


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_plan_refuses_foreign_scope_layout_operations_and_aliases(monkeypatch):
    cache = [KVCache()]
    plan = CommittedDescriptorPlan(cache, "session-a")
    assert plan.capture(cache, "session-b") is None
    assert plan.capture([ArraysCache(2)], "session-a") is None
    assert plan.capture(cache * 2, "session-a") is None
    cache[0].foreign_mutable_state = []
    assert plan.capture(cache, "session-a") is None
    del cache[0].foreign_mutable_state
    monkeypatch.setattr(KVCache, "committed_descriptor_clone", lambda self: self)
    assert plan.capture(cache, "session-a") is None


@pytest.mark.parametrize("kind", [KVCache, ArraysCache, RotatingKVCache])
def test_subclass_does_not_inherit_lean_snapshot_authority(kind):
    class Foreign(kind):
        pass

    item = Foreign(2) if kind in (ArraysCache, RotatingKVCache) else Foreign()
    assert item.committed_descriptor_clone() is None


def test_recurrent_snapshot_owns_containers_and_refuses_open_transaction():
    cache = ArraysCache(2)
    cache.cache = [mx.ones((1, 2)), mx.zeros((1, 2))]
    cache.state_checkpoint([64], force=True)
    clone = cache.committed_descriptor_clone()
    assert clone is not None and clone.cache is not cache.cache
    assert clone._checkpoints is not cache._checkpoints
    assert clone._checkpoints[0] is not cache._checkpoints[0]
    clone._checkpoints[0][0][1].clear()
    assert len(cache._checkpoints[0][0][1]) == 2
    clone.cache[0][:] = 7
    assert mx.array_equal(cache.cache[0], mx.ones((1, 2))).item()
    cache.start_speculation()
    assert cache.committed_descriptor_clone() is None


def test_apcv2_warm_restore_attribution_does_not_disable_lean_snapshots():
    from mlx2.runtime.recurrent_state_codec import decode_prompt_cache

    cache = ArraysCache(2)
    cache.cache = [mx.ones((1, 2)), mx.zeros((1, 2))]
    # The real APCv2 restore hook adds this host-only counter even when no
    # encoded leaves were present. Do not infer schema from cold caches only.
    assert decode_prompt_cache([cache]) == 0
    assert cache._recurrent_codec_restored_leaves == 0
    plan = CommittedDescriptorPlan([cache], "warm-session")
    result = plan.capture([cache], "warm-session", recovery=True)
    assert result is not None
    assert result[0][0]._recurrent_codec_restored_leaves == 0
    cache._recurrent_codec_restored_leaves = []
    assert plan.capture([cache], "warm-session") is None


def test_recovery_and_private_forks_do_not_walk_generic_graph(monkeypatch):
    from mlx2.runtime import cow_cache

    _draft, batch, lanes = _cohort()
    before = [_committed_digest(lane.cache) for lane in lanes]
    try:

        def refuse(*args, **kwargs):
            raise AssertionError("general graph scan reached lean snapshot")

        with monkeypatch.context() as local:
            local.setattr(cow_cache, "_clone_graph", refuse)
            local.setattr(cow_cache, "_tree_nbytes", refuse)
            for lane in lanes:
                frozen, mode = batch._freeze_lane(lane)
                assert mode == "descriptor_cow"
                assert frozen[1] is not lane.cache
                private = batch._copy_ast_private_cache(lane.cache)
                assert _committed_digest(private) == _committed_digest(lane.cache)
        assert batch.scheduler_stats["external_ast_lean_recovery_snapshots"] == 4
        assert batch.scheduler_stats["external_ast_lean_private_forks"] == 4
        assert [_committed_digest(lane.cache) for lane in lanes] == before
    finally:
        batch.close()


def test_borrowed_kv_recovery_matches_reference_and_rewind_refuses():
    cache = KVCache()
    data = mx.ones((1, 1, 3, 2))
    cache.update_and_fetch(data, data)
    plan = CommittedDescriptorPlan([cache], "session")
    clones, borrowed = plan.capture([cache], "session", recovery=True)
    assert clones[0].keys is None
    ordinary, _, _ = snapshot_recovery_descriptors([cache])
    cache.update_and_fetch(data, data)
    restored, _ = restore_recovery_descriptors(clones, None, borrowed)
    assert restored[0].offset == ordinary[0].offset == 3
    assert mx.array_equal(restored[0].keys[..., :3, :], data).item()
    cache.trim(4)
    with pytest.raises(RuntimeError, match="rewound"):
        restore_recovery_descriptors(clones, None, borrowed)


def test_round_builds_each_receipt_once_per_lane_not_per_emitted_token(monkeypatch):
    _draft, batch, lanes = _cohort()
    for lane in lanes:
        lane.sampling["emit_logprobs"] = False
    # Deterministic all-accepted host fixture. Numerical laws themselves are
    # covered by the ordinary/reference tests and the exact GPU oracle.
    monkeypatch.setattr(
        batch,
        "_walk_tree",
        lambda block, sample: (list(range(len(block.tokens) + 1)), [*block.tokens, 0]),
    )
    calls = {}
    for name in (
        "_continuation_receipt",
        "_verification_receipt",
        "_draft_settings_receipt",
        "_proposal_composition_receipt",
        "_target_execution_receipt",
    ):
        original = getattr(batch, name)

        def observed(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(batch, name, observed)
    try:
        batch._round(lanes)
        assert all(count == 4 for count in calls.values())
        assert sum(len(lane.ready) for lane in lanes) > 4
        responses = batch._drain_ready([])
        by_uid = {}
        for response in responses:
            receipt = response.speculative_receipt
            assert copy.deepcopy(receipt) is receipt
            json.dumps(receipt, allow_nan=False)
            with pytest.raises(TypeError, match="immutable"):
                receipt["accepted"] = -1
            by_uid.setdefault(response.uid, []).append(receipt)
        for receipts in by_uid.values():
            assert all(receipt == receipts[0] for receipt in receipts)
            assert all(
                receipt["draft_settings"] is receipts[0]["draft_settings"]
                for receipt in receipts
            )
        snapshot = json.dumps(responses[0].speculative_receipt, sort_keys=True)
        batch._round(lanes)
        assert json.dumps(responses[0].speculative_receipt, sort_keys=True) == snapshot
    finally:
        batch.close()


def test_settings_memo_invalidates_on_change_and_counters_stay_live():
    _draft, batch, _lanes = _cohort()
    try:
        first = batch._draft_settings_receipt()["draft_settings"]
        batch.scheduler_stats["external_proposal_floor_raises"] += 1
        second = batch._draft_settings_receipt()["draft_settings"]
        assert second["proposal_floor_raises"] == first["proposal_floor_raises"] + 1
        batch.speculative_batching = {"changed": [2]}
        third = batch._draft_settings_receipt()["draft_settings"]
        assert third["batching"] == {"changed": [2]}
        batch.speculative_batching["changed"].append(3)
        assert third["batching"] == {"changed": [2]}
        assert batch._draft_settings_receipt()["draft_settings"]["batching"] == {
            "changed": [2, 3]
        }
        nested = freeze_receipt({"items": [{"value": 1}]})
        with pytest.raises(TypeError):
            nested["items"][0]["value"] = 2
    finally:
        batch.close()
