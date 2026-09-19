from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment
from mlx2.runtime.models.qwen4_exp import (
    QSAKVCache,
    TextModel,
    TextModelArgs,
    _apply_rope_positions,
)
from mlx2.runtime.spomin_layer import (
    SpominBackendError,
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominPlanError,
    SpominRevisionError,
    SpominTargetState,
)
from mlx2.runtime.spomin_live_surgery import (
    ServingSpominPolicy,
    SpominLiveSurgeryManager,
)
from mlx2.runtime.spomin_qwen4_surgery import Qwen4SpominSurgeryBackend
from mlx2.runtime.generate import BatchGenerator


def transcript_fixture():
    return TranscriptLedgerPlane(
        tokenizer_identity="test-tokenizer",
        tokenizer_version="1",
        revision="transcript-r1",
        segments=(
            TranscriptLedgerSegment("turn:1", 0, 3, (1, 2, 3)),
            TranscriptLedgerSegment("turn:2", 3, 7, (4, 5, 6, 7)),
            TranscriptLedgerSegment("turn:3", 7, 9, (8, 9)),
            TranscriptLedgerSegment("turn:4", 9, 12, (10, 11, 12)),
        ),
    )


def fixture():
    args = SimpleNamespace(
        head_dim=8,
        partial_rotary_factor=0.5,
        rope_theta=10_000.0,
        rope_scaling={"rope_type": "default"},
    )
    layers = [SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    model = SimpleNamespace(layers=layers, args=args)
    linear = SimpleNamespace(state="uncompacted-recurrent-summary")
    cache = QSAKVCache()
    raw = mx.arange(1 * 2 * 12 * 8, dtype=mx.float32).reshape(1, 2, 12, 8) / 100
    positions = mx.arange(12, dtype=mx.float32)[None, None, :]
    cache.keys = _apply_rope_positions(raw, positions, dims=4, base=10_000.0)
    cache.values = raw + 100
    cache.index_keys = mx.arange(36, dtype=mx.float32).reshape(1, 12, 3)
    cache.offset = 12
    transcript = transcript_fixture()
    state = SpominTargetState(
        revision="target-r1",
        target_tokens=12,
        transcript=transcript,
        visible_segment_ids=tuple(segment.segment_id for segment in transcript.segments),
        has_recurrent_state=False,
    )
    layer = SpominLayer(
        SpominConfig(
            capacity_tokens=16,
            pressure_ratio=0.70,
            target_ratio=0.50,
            protect_recent_segments=0,
            strategy="largest_first",
        )
    )
    return model, linear, cache, raw, state, layer, layer.plan(state)


def test_surgery_gathers_and_rephases_qsa_rows_without_touching_other_layers():
    model, linear, cache, raw, state, layer, plan = fixture()
    old_linear_state = linear.state
    updated = layer.apply(state, plan, Qwen4SpominSurgeryBackend(model, [linear, cache]))
    keep = [0, 1, 2, 7, 8, 9, 10, 11]
    expected_raw = mx.take(raw, mx.array(keep), axis=2)
    expected_keys = _apply_rope_positions(
        expected_raw,
        mx.arange(8, dtype=mx.float32)[None, None, :],
        dims=4,
        base=10_000.0,
    )
    np.testing.assert_allclose(np.asarray(cache.keys), np.asarray(expected_keys), atol=2e-5)
    np.testing.assert_allclose(np.asarray(cache.values), np.asarray(mx.take(raw + 100, mx.array(keep), axis=2)))
    assert cache.offset == 8
    assert cache.index_keys.shape == (1, 8, 3)
    assert linear.state == old_linear_state
    assert updated.visible_segment_ids == ("turn:1", "turn:3", "turn:4")
    assert updated.transcript is state.transcript


def test_live_manager_requires_epoch_and_drained_barrier():
    manager = SpominLiveSurgeryManager(enabled=True)
    model, linear, cache, _, state, _, _ = fixture()
    transaction = manager.prepare(
        request_id="req-1",
        prompt_token_ids=state.transcript.token_ids,
        transcript=state.transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=False,
        cache_is_request_private=True,
    )
    declined = transaction.apply(
        model, [linear, cache], request_quiescent=True, device_work_drained=False
    )
    assert declined["reason"] == "device_work_not_drained"
    assert cache.offset == 12
    applied = transaction.apply(
        model, [linear, cache], request_quiescent=True, device_work_drained=True
    )
    assert applied["status"] == "applied"
    assert applied["selected"] is True
    assert transaction.retained_token_ids == (1, 2, 3, 8, 9, 10, 11, 12)
    assert manager.snapshot()["counts"]["applied"] == 1


def test_live_manager_is_default_off_and_refuses_mtp():
    _, _, _, _, state, _, _ = fixture()
    manager = SpominLiveSurgeryManager()
    assert manager.prepare(
        request_id="req-off",
        prompt_token_ids=state.transcript.token_ids,
        transcript=state.transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=True,
        cache_is_request_private=True,
    ) is None
    enabled = SpominLiveSurgeryManager(enabled=True)
    assert enabled.prepare(
        request_id="req-mtp",
        prompt_token_ids=state.transcript.token_ids,
        transcript=state.transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=True,
        has_recurrent_state=True,
        cache_is_request_private=True,
    ) is None
    assert enabled.snapshot()["counts"]["reason:mtp_state_active"] == 1


def test_serving_policy_is_default_off_and_builds_stable_segment_ledger():
    assert ServingSpominPolicy.from_value(None).enabled is False
    policy = ServingSpominPolicy.from_value(
        {"enabled": True, "capacity_tokens": 16, "segment_tokens": 3}
    )
    transcript = policy.transcript(
        range(1, 13), tokenizer_identity="tokenizer-r1", revision="request-r1"
    )
    assert [segment.token_ids for segment in transcript.segments] == [
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
        (10, 11, 12),
    ]
    assert transcript.token_ids == tuple(range(1, 13))


def test_serving_policy_rejects_implicit_or_unknown_selection():
    for invalid in (True, {"enabled": True, "capacity_tokens": 16, "mystery": 1}):
        try:
            ServingSpominPolicy.from_value(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid serving policy was silently accepted")


def test_serving_route_rejects_unqualified_or_speculative_surgery():
    from mlx2.serving import ServingEngine

    policy = {"enabled": True, "capacity_tokens": 16, "segment_tokens": 3}
    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine("unused", mtp=False, max_lanes=1, spomin_live_surgery=policy)
    with pytest.raises(ValueError, match="incompatible with MTP"):
        ServingEngine(
            "unused",
            mtp=True,
            max_lanes=1,
            qualification_mode=True,
            spomin_live_surgery=policy,
        )


def test_live_manager_fails_closed_when_hybrid_recurrent_state_needs_repair():
    manager = SpominLiveSurgeryManager(enabled=True)
    transcript = transcript_fixture()
    assert manager.prepare(
        request_id="req-hybrid",
        prompt_token_ids=transcript.token_ids,
        transcript=transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=True,
        cache_is_request_private=True,
    ) is None
    snapshot = manager.snapshot()
    assert snapshot["active_epochs"] == 0
    assert snapshot["counts"]["reason:recurrent_state_unrepairable"] == 1


def test_live_manager_close_records_abandoned_epoch_once():
    manager = SpominLiveSurgeryManager(enabled=True)
    transcript = transcript_fixture()
    transaction = manager.prepare(
        request_id="req-abandoned",
        prompt_token_ids=transcript.token_ids,
        transcript=transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=False,
        cache_is_request_private=True,
    )
    assert manager.snapshot()["active_epochs"] == 1

    receipt = transaction.close()
    snapshot = manager.snapshot()
    assert snapshot["active_epochs"] == 0
    assert snapshot["counts"]["declined"] == 1
    assert snapshot["counts"]["reason:closed_without_apply"] == 1
    expected = {
        "schema": "mlx2.spomin-live-surgery.v1",
        "request_id": "req-abandoned",
        "status": "declined",
        "reason": "closed_without_apply",
        "epoch": transaction.epoch.generation,
        "selected": False,
    }
    assert receipt == expected
    assert transaction.receipt == expected
    assert snapshot["recent"][-1] == expected

    assert transaction.close() is None
    assert transaction.receipt == expected
    assert manager.snapshot() == snapshot


def test_live_manager_apply_replays_terminal_receipt_without_double_counting(
    monkeypatch,
):
    manager = SpominLiveSurgeryManager(enabled=True)
    transcript = transcript_fixture()
    transaction = manager.prepare(
        request_id="req-idempotent",
        prompt_token_ids=transcript.token_ids,
        transcript=transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=False,
        cache_is_request_private=True,
    )

    def apply_without_device(_layer, state, plan, _backend):
        return replace(
            state,
            target_tokens=plan.projected_target_tokens,
            visible_segment_ids=tuple(
                segment_id
                for segment_id in state.visible_segment_ids
                if segment_id not in plan.selection.segment_ids
            ),
        )

    monkeypatch.setattr(SpominLayer, "apply", apply_without_device)
    first = transaction.apply(
        SimpleNamespace(), [], request_quiescent=True, device_work_drained=True
    )
    second = transaction.apply(
        SimpleNamespace(), [], request_quiescent=True, device_work_drained=True
    )

    assert second == first
    assert transaction.receipt == first
    snapshot = manager.snapshot()
    assert snapshot["active_epochs"] == 0
    assert snapshot["counts"]["applied"] == 1
    assert snapshot["counts"].get("reason:stale_epoch", 0) == 0
    assert snapshot["recent"] == [first]


def test_live_manager_declines_stale_revision_without_escaping():
    manager = SpominLiveSurgeryManager(enabled=True)
    transcript = transcript_fixture()
    transaction = manager.prepare(
        request_id="req-stale",
        prompt_token_ids=transcript.token_ids,
        transcript=transcript,
        capacity_tokens=16,
        strategy="largest_first",
        has_mtp_state=False,
        has_recurrent_state=False,
        cache_is_request_private=True,
    )
    transaction.state = replace(transaction.state, revision="request:req-stale:changed")
    receipt = transaction.apply(
        SimpleNamespace(), [], request_quiescent=True, device_work_drained=True
    )
    assert receipt["status"] == "declined"
    assert receipt["reason"] == "revision_changed"
    assert "revision changed" in receipt["detail"]
    assert manager.snapshot()["active_epochs"] == 0


def test_live_manager_maps_expected_transaction_errors_to_declined_receipts(monkeypatch):
    transcript = transcript_fixture()
    cases = (
        (SpominCapabilityError, "capability_refused"),
        (SpominRevisionError, "revision_changed"),
        (SpominPlanError, "plan_invalid"),
        (SpominBackendError, "backend_refused"),
    )
    for index, (error_type, expected_reason) in enumerate(cases):
        manager = SpominLiveSurgeryManager(enabled=True)
        transaction = manager.prepare(
            request_id=f"req-error-{index}",
            prompt_token_ids=transcript.token_ids,
            transcript=transcript,
            capacity_tokens=16,
            strategy="largest_first",
            has_mtp_state=False,
            has_recurrent_state=False,
            cache_is_request_private=True,
        )

        def fail_apply(*_args, **_kwargs):
            raise error_type("expected transaction refusal")

        monkeypatch.setattr(SpominLayer, "apply", fail_apply)
        receipt = transaction.apply(
            SimpleNamespace(), [], request_quiescent=True, device_work_drained=True
        )
        assert receipt["status"] == "declined"
        assert receipt["reason"] == expected_reason
        assert receipt["detail"] == "expected transaction refusal"
        assert manager.snapshot()["active_epochs"] == 0


def test_single_attention_layer_continuation_matches_compacted_rebuild():
    args = TextModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        mtp_num_hidden_layers=0,
        rope_parameters={
            "type": "default",
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.5,
        },
    )
    model = TextModel(args)
    full_tokens = mx.array([list(range(1, 13))], dtype=mx.int32)
    compacted_tokens = mx.array([[1, 2, 3, 8, 9, 10, 11, 12]], dtype=mx.int32)
    next_token = mx.array([[13]], dtype=mx.int32)
    surgical_cache = model.make_cache()
    mx.eval(model(full_tokens, surgical_cache))
    _, _, _, _, state, layer, plan = fixture()
    layer.apply(state, plan, Qwen4SpominSurgeryBackend(model, surgical_cache))
    surgical = model(next_token, surgical_cache)
    rebuilt_cache = model.make_cache()
    mx.eval(model(compacted_tokens, rebuilt_cache))
    rebuilt = model(next_token, rebuilt_cache)
    mx.eval(surgical, rebuilt)
    np.testing.assert_allclose(
        np.asarray(surgical), np.asarray(rebuilt), rtol=2e-5, atol=2e-5
    )


def test_batch_generator_runs_surgery_at_isolated_post_prefill_boundary():
    args = TextModelArgs(
        hidden_size=16,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        hc_count=4,
        hc_lowrank=4,
        ple_layer_ids=[],
        ple_embed_dim=16,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        eos_token_id=63,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=4,
        mtp_num_hidden_layers=0,
        rope_parameters={
            "type": "default",
            "rope_theta": 10_000,
            "partial_rotary_factor": 0.5,
        },
    )
    model = TextModel(args)
    manager = SpominLiveSurgeryManager(enabled=True)
    policy = ServingSpominPolicy(
        enabled=True, capacity_tokens=16, segment_tokens=3
    )

    def transform(*, uid, model, prompt_cache, cached_token_ids):
        transcript = policy.transcript(
            cached_token_ids, tokenizer_identity="test", revision=f"request:{uid}"
        )
        transaction = manager.prepare(
            request_id=str(uid),
            prompt_token_ids=cached_token_ids,
            transcript=transcript,
            capacity_tokens=policy.capacity_tokens,
            strategy=policy.strategy,
            has_mtp_state=False,
            has_recurrent_state=False,
            cache_is_request_private=True,
        )
        mx.synchronize()
        receipt = transaction.apply(
            model,
            prompt_cache,
            request_quiescent=True,
            device_work_drained=True,
        )
        return {
            "receipt": receipt,
            "retained_token_ids": transaction.retained_token_ids,
            "prompt_cache": prompt_cache,
        }

    batch = BatchGenerator(
        model,
        completion_batch_size=1,
        prefill_batch_size=1,
        prefill_step_size=32,
        post_prefill_transform=transform,
    )
    (uid,) = batch.insert([list(range(1, 14))], max_tokens=[1])
    prompt, responses = batch.next()
    assert not responses
    assert prompt[-1].end_of_prompt is False
    prompt, responses = batch.next()
    assert prompt[-1].end_of_prompt is True
    receipt = batch.pop_post_prefill_receipt(uid)
    assert receipt["status"] == "applied", receipt.get("detail")
    assert receipt["selected"] is True
    assert receipt["source_tokens"] == 12
    assert receipt["retained_tokens"] == 9
    assert manager.snapshot()["active_epochs"] == 0
