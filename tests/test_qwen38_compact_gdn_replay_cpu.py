from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.qwen_gdn_contracts import (
    QWEN36_GDN_CONTRACT,
    QWEN38_GDN_CONTRACT,
)
from mlx2.runtime.models.qwen38_state_plane import (
    CompactGdnReplayAuthorization,
    QWEN38_STATE_PLANE_OPERATIONS,
)
from mlx2.runtime.state_plane_operations import CompactGdnReplayOperations


def _owned_cache() -> ArraysCache:
    geometry = QWEN38_GDN_CONTRACT.geometry
    conv_dim = (
        2 * geometry.num_key_heads * geometry.key_head_dim
        + geometry.num_value_heads * geometry.value_head_dim
    )
    cache = ArraysCache(2)
    cache[0] = mx.broadcast_to(
        mx.zeros((1, 1, 1), mx.bfloat16),
        (1, geometry.conv_kernel - 1, conv_dim),
    )
    cache[1] = mx.broadcast_to(
        mx.zeros((1, 1, 1, 1), mx.float32),
        (
            1,
            geometry.num_value_heads,
            geometry.value_head_dim,
            geometry.key_head_dim,
        ),
    )
    cache.start_speculation()
    return cache


def test_provider_owns_exact_qwen38_state_and_refuses_drift():
    provider = QWEN38_STATE_PLANE_OPERATIONS
    assert isinstance(provider, CompactGdnReplayOperations)
    cache = _owned_cache()
    contract = provider.contract(cache)
    assert contract is not None
    assert contract.fidelity == "exact"
    assert {"checkpoint", "rollback", "private_successor"} <= contract.operations
    assert provider.contract(KVCache()).layout == "qwen38-full-attention-kv-v1"

    receipt = provider.authorize_compact_gdn_replay(
        cache, QWEN38_GDN_CONTRACT, 9
    )
    assert receipt.contract_identity == QWEN38_GDN_CONTRACT.identity
    assert receipt.snapshot_bytes_avoided > receipt.tape_bytes > 0
    with pytest.raises(Exception, match="contract identity drifted"):
        provider.authorize_compact_gdn_replay(cache, QWEN36_GDN_CONTRACT, 9)
    cache.stop_speculation()
    with pytest.raises(Exception, match="active two-plane speculative cache"):
        provider.authorize_compact_gdn_replay(cache, QWEN38_GDN_CONTRACT, 9)


def test_plan_commits_only_accepted_prefix_and_cancellation_keeps_checkpoint():
    provider = QWEN38_STATE_PLANE_OPERATIONS
    auth = CompactGdnReplayAuthorization(
        provider.provider_revision, QWEN38_GDN_CONTRACT.identity, 4, 12, 100
    )
    checkpoint_conv = mx.array([[[10], [11], [12]]], mx.int32)
    checkpoint_state = mx.array([[[[20]]]], mx.float32)
    raw = mx.array([[[30], [31], [32], [33]]], mx.int32)
    tape = mx.zeros((1, 3, 1), mx.float32)
    calls = []

    def reconstruct(state, keys, corrections, decay, accepted, *, threadgroup_y):
        calls.append((accepted, threadgroup_y))
        return state + accepted

    cache = ArraysCache(2)
    cache[0], cache[1] = checkpoint_conv, checkpoint_state
    cache.start_speculation()
    plan = provider.make_compact_gdn_replay_plan(
        auth,
        cache=cache,
        raw_qkv=raw,
        replay_keys=tape,
        replay_corrections=tape,
        replay_decay=tape,
        reconstruct=reconstruct,
        threadgroup_y=8,
    )
    conv, state = plan.accepted_prefix(2)
    assert conv.tolist() == [[[12], [30], [31]]]
    assert state.tolist() == [[[[22.0]]]]
    assert calls == [(2, 8)]
    cancelled = plan.cancel()
    assert cancelled[0] is checkpoint_conv and cancelled[1] is checkpoint_state
    final = [mx.array([[[99]]]), mx.array([[[[88.0]]]])]
    assert plan.commit_full(*final) == final
    with pytest.raises(ValueError, match="outside"):
        plan.accepted_prefix(0)
    with pytest.raises(ValueError, match="outside"):
        plan.accepted_prefix(4)

    # The generic exact rollback owner publishes the accepted-prefix result;
    # a full cancellation takes its immutable checkpoint without reading tape.
    cache.record_rollback(4, plan.accepted_prefix, plan.cancel())
    cache.cache = final
    cache.trim(2)
    assert cache[0].tolist() == [[[12], [30], [31]]]
    assert cache[1].tolist() == [[[[22.0]]]]

    cancelled_cache = ArraysCache(2)
    cancelled_cache[0], cancelled_cache[1] = checkpoint_conv, checkpoint_state
    cancelled_cache.start_speculation()
    cancelled_cache.record_rollback(4, plan.accepted_prefix, plan.cancel())
    cancelled_cache.cache = final
    cancelled_cache.trim(4)
    assert cancelled_cache[0] is checkpoint_conv
    assert cancelled_cache[1] is checkpoint_state


def test_compact_memory_accounting_scales_with_tape_not_state_snapshots():
    cache = _owned_cache()
    short = QWEN38_STATE_PLANE_OPERATIONS.authorize_compact_gdn_replay(
        cache, QWEN38_GDN_CONTRACT, 2
    )
    wide = QWEN38_STATE_PLANE_OPERATIONS.authorize_compact_gdn_replay(
        cache, QWEN38_GDN_CONTRACT, 9
    )
    assert wide.tape_bytes == 8 * short.tape_bytes
    assert wide.snapshot_bytes_avoided > 25_000_000


def test_shared_protocol_is_model_neutral_and_selection_is_default_off():
    import mlx2.runtime.state_plane_operations as shared
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, TARGET_POLICY_KEYS

    protocol_source = inspect.getsource(shared.CompactGdnReplayOperations)
    assert "Qwen" not in protocol_source and "qwen" not in protocol_source
    assert "fused_gdn_compact_replay" in TARGET_POLICY_KEYS
    assert Qwen3827BAdapter.default_fused_gdn_compact_replay is False
    with pytest.raises(ValueError, match="must be boolean"):
        Qwen3827BAdapter(
            "/nonexistent", execution_policy={"fused_gdn_compact_replay": 1}
        )
    with pytest.raises(ValueError, match="requires fused_gdn"):
        Qwen3827BAdapter(
            "/nonexistent",
            execution_policy={
                "fused_gdn": False,
                "fused_gdn_compact_replay": True,
            },
        )


def test_qwen35_compact_source_is_derived_from_typed_profile():
    from mlx2.runtime.models import qwen4_fused_gdn_verify as kernels

    assert "T av = a[t * HV + hv] + dt_bias[hv];" in kernels._QWEN35_REPLAY_SOURCE
    assert "replay_corrections" in kernels._QWEN35_REPLAY_SOURCE
    assert "state_snapshots" not in kernels._QWEN35_REPLAY_SOURCE
