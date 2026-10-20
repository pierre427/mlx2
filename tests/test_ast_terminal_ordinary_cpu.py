"""Terminal AST handoff uses the adapter's real ordinary arithmetic."""

import copy

import mlx.core as mx
import numpy as np
import pytest
from test_ast_mixed_parked_cpu import _prepared

from mlx2.adapters.qwen38_27b import target_verify_row_exact_numerics
from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
    CostPerCommitLaneState,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime import row_exact_verify as REV
from scripts.paired_direct_ab import state_digest


def _committed_digest(caches):
    """Shared near-tie oracle ignores unused allocation-tail storage."""
    return state_digest(
        [
            (
                type(cache).__name__,
                cache.keys_and_values() if isinstance(cache, KVCache) else cache.state,
                cache.meta_state,
            )
            for cache in caches
        ]
    )


def _terminal(monkeypatch):
    target, batch, lanes, _previous = _prepared(monkeypatch, parked_count=0)
    batch._decode_phase(lanes)
    settings = CostPerCommitHandoffSettings(
        enabled=True, ordinary_samples=1, speculative_samples=1, loss_decisions=1
    )
    batch.cost_per_commit_handoff = settings
    for lane in lanes:
        state = CostPerCommitLaneState(
            settings,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
        state.observe(
            kind="ordinary",
            elapsed_ns=100,
            committed_tokens=1,
            execution_width=4,
            lane_uid=lane.uid,
            owner_scope=(batch.binding, lane.session_scope_hash),
        )
        lane.cost_handoff_state = state
    batch._observe_cost_handoff_round(
        lanes,
        [lane.generated - 1 for lane in lanes],
        0,
        ordinary_probe=False,
    )
    assert all(lane.ordinary and lane.cost_handoff_state.handed_off for lane in lanes)
    for lane in lanes:
        lane.ready.clear()
    return target, batch, lanes


def test_terminal_cost_handoff_batches_stock_continuation_without_draft_rejoin(
    monkeypatch,
):
    target, batch, lanes = _terminal(monkeypatch)
    route = target._mlx2_qwen_hybrid_row_exact
    exact_before = route.counts["launches"]
    external_before = batch.scheduler_stats["external_rounds"]
    caches = [copy.deepcopy(lane.cache) for lane in lanes]
    anchors = [lane.anchor for lane in lanes]
    hook = target.ordinary_external_target_forward
    calls = []
    actual_logits = []

    def observed(inputs, cache):
        assert REV.current() is None
        assert all(lane.ordinary for lane in lanes)
        assert all(not getattr(plane, "speculating", False) for plane in cache)
        calls.append(tuple(inputs.shape))
        logits = hook(inputs, cache)
        actual_logits.append(logits)
        return logits

    monkeypatch.setattr(target, "ordinary_external_target_forward", observed)
    for _ in range(3):
        transaction = HybridVerifyRows(caches).begin([1] * 4, plain_single_token=True)
        logits = target(
            mx.array(anchors, dtype=mx.int32)[:, None], cache=transaction.caches
        )
        mx.eval(logits)
        expected = mx.argmax(logits[:, -1, :].astype(mx.float32), axis=-1).tolist()
        transaction.commit([1] * 4)
        batch._decode_phase(lanes)
        np.testing.assert_array_equal(np.asarray(actual_logits[-1]), np.asarray(logits))
        assert [lane.ready[-1].token for lane in lanes] == expected
        for lane, reference in zip(lanes, caches, strict=True):
            assert state_digest(batch._freeze_cache(lane.cache)) == state_digest(
                batch._freeze_cache(reference)
            )
            assert lane.ready[-1].cache_sidecar is None
            assert (
                lane.ready[-1].speculative_receipt["current_execution"]
                == "ordinary_target"
            )
            assert lane.ordinary and lane.cost_handoff_state.handed_off
            lane.ready.clear()
        anchors = expected
    assert calls == [(4, 1)] * 3
    assert route.counts["launches"] == exact_before
    assert route.counts["ordinary_terminal_launches"] == 3
    assert batch.scheduler_stats["external_rounds"] == external_before
    assert (
        batch.scheduler_stats[
            "external_cohort_target_ordinary_native_ordinary_round_rounds"
        ]
        == 3
    )
    schedule = target_verify_row_exact_numerics(target)["terminal_ordinary_handoff"]
    assert schedule["arithmetic"] == "stock_batched_autoregressive"
    assert schedule["speculative_reentry"] is False


@pytest.mark.parametrize("missing", [True, False])
def test_incapable_terminal_hook_preserves_previous_exact_route(monkeypatch, missing):
    target, batch, lanes = _terminal(monkeypatch)
    route = target._mlx2_qwen_hybrid_row_exact
    if missing:
        monkeypatch.delattr(target, "ordinary_external_target_forward")
    else:
        monkeypatch.setattr(target, "ordinary_external_target_protocol", "unknown")
    before = route.counts["launches"]
    batch._decode_phase(lanes)
    assert route.counts["launches"] == before + 1
    assert route.counts["ordinary_terminal_launches"] == 0
    assert all(lane.ready[-1].cache_sidecar is None for lane in lanes)


def test_terminal_handoff_drains_parked_inflight_before_switch(monkeypatch):
    target, batch, lanes, _previous = _prepared(monkeypatch, parked_count=4)
    batch._decode_phase(lanes)
    assert batch._parked_inflight is not None
    batch.disable_speculation(lanes[0].uid)
    assert batch._parked_inflight is None
    assert batch.scheduler_stats["external_parked_pipeline_drains"] == 1
    for lane in lanes[1:]:
        batch.disable_speculation(lane.uid)
    for lane in lanes:
        lane.ready.clear()
    batch._decode_phase(lanes)
    assert target._mlx2_qwen_hybrid_row_exact.counts["ordinary_terminal_launches"] == 1
    assert all(lane.ready[-1].cache_sidecar is None for lane in lanes)
    assert all(lane.ordinary for lane in lanes)


def test_reversible_park_does_not_select_terminal_stock_hook(monkeypatch):
    target, batch, lanes, _previous = _prepared(monkeypatch, parked_count=4)
    batch._decode_phase(lanes)
    batch._decode_phase(lanes)
    assert target._mlx2_qwen_hybrid_row_exact.counts["ordinary_terminal_launches"] == 0
    assert all(not lane.ordinary for lane in lanes)
    batch._abort_parked_inflight()
