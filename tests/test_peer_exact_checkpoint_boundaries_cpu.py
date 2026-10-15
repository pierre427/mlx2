"""Repeated exact resume crosses an actual ordinary KV allocation boundary."""

import mlx.core as mx
import pytest
from route_harness import patch_host
from test_exact_decode_checkpoint_cpu import _model, _next_output, _run_server

from mlx2.runtime import decode_checkpoint as dc
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import ArraysCache, BatchKVCache
from mlx2.runtime.sample_utils import LaneRNG, draw_key


def _hybrid_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    mx.random.seed(17)
    model = TextModel(TextModelArgs(
        model_type="qwen3_5",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=128,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=512,
    ))
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("sample", [False, True])
def test_repeated_exact_resume_preserves_pending_sample_across_kv_growth(sample):
    model = _model()

    def run(suspend):
        rng = LaneRNG(423)
        sampler = (
            (lambda values: mx.random.categorical(values, key=draw_key(rng)))
            if sample
            else None
        )
        batch = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=64
        )
        (uid,) = batch.insert(
            [[1 + i % 126 for i in range(254)]],
            max_tokens=[8],
            samplers=[sampler],
            lane_rngs=[rng if sample else None],
        )
        rows, geometries = [], []
        try:
            for index in range(8):
                rows.append(_next_output(batch))
                if index == 7:
                    break
                control = batch._generation_batch
                geometries.append(
                    tuple(cache.keys.shape[2] for cache in control.prompt_cache)
                )
                if suspend:
                    pending = control._next_tokens.tolist()
                    draws = rng.draws
                    owner = batch.suspend_ordinary_checkpoint(
                        uid, ("revision", "request")
                    )
                    try:
                        assert not control.prompt_cache
                        assert (
                            batch.resume_ordinary_checkpoint(
                                owner, ("revision", "request")
                            )
                            == uid
                        )
                        assert batch._generation_batch is control
                        assert (
                            control._next_tokens.tolist() == pending
                            and rng.draws == draws
                        )
                    finally:
                        owner.close()
            return rows, geometries, rng.draws
        finally:
            batch.close()

    expected, geometry, draws = run(False)
    actual, restored_geometry, restored_draws = run(True)
    assert len(set(geometry)) > 1
    assert restored_geometry == geometry and restored_draws == draws
    assert [row.token for row in actual] == [row.token for row in expected]
    assert [row.finish_reason for row in actual] == [
        row.finish_reason for row in expected
    ]
    for reference, restored in zip(expected, actual):
        assert mx.array_equal(reference.logprobs, restored.logprobs).item()


@pytest.mark.parametrize("sample", [False, True])
def test_hybrid_repeated_checkpoint_preserves_recurrent_kv_control_and_rng(sample):
    model = _hybrid_model()

    def run(suspend):
        rng = LaneRNG(741)
        sampler = (
            (lambda values: mx.random.categorical(values / 0.8, key=draw_key(rng)))
            if sample
            else None
        )
        batch = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=64,
        )
        (uid,) = batch.insert(
            [[1 + i % 126 for i in range(254)]],
            max_tokens=[10],
            samplers=[sampler],
            lane_rngs=[rng if sample else None],
        )
        rows, control_ids, cache_kinds = [], [], None
        try:
            for index in range(10):
                rows.append(_next_output(batch))
                if index == 9:
                    break
                control = batch._generation_batch
                if cache_kinds is None:
                    cache_kinds = tuple(type(cache) for cache in control.prompt_cache)
                if suspend:
                    pending = mx.array(control._next_tokens)
                    draws = rng.draws
                    contracts = tuple(
                        cache.exact_decode_checkpoint_contract()
                        for cache in control.prompt_cache
                    )
                    owner = batch.suspend_ordinary_checkpoint(
                        uid, ("hybrid-revision", "request")
                    )
                    try:
                        assert all(contract is not None for contract in contracts)
                        assert batch.resume_ordinary_checkpoint(
                            owner, ("hybrid-revision", "request")
                        ) == uid
                        assert batch._generation_batch is control
                        assert mx.array_equal(control._next_tokens, pending).item()
                        assert rng.draws == draws
                        control_ids.append(id(control))
                    finally:
                        owner.close()
            return rows, rng.draws, control_ids, cache_kinds
        finally:
            batch.close()

    expected, expected_draws, _, expected_kinds = run(False)
    actual, actual_draws, controls, actual_kinds = run(True)
    assert expected_kinds == actual_kinds
    assert ArraysCache in actual_kinds and BatchKVCache in actual_kinds
    assert len(controls) == 9 and len(set(controls)) == 1
    assert actual_draws == expected_draws
    assert [row.token for row in actual] == [row.token for row in expected]
    assert [row.finish_reason for row in actual] == [
        row.finish_reason for row in expected
    ]
    assert all(
        mx.array_equal(restored.logprobs, reference.logprobs).item()
        for restored, reference in zip(actual, expected)
    )


@pytest.mark.parametrize("failure", ["cancel", "corrupt"])
def test_hybrid_checkpoint_failure_closes_state_and_executor_recovers(
    monkeypatch, failure,
):
    model = _hybrid_model()
    batch = BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32
    )
    (uid,) = batch.insert(
        [[1 + i % 126 for i in range(190)]], max_tokens=[6]
    )
    owner = None
    try:
        _next_output(batch)
        owner = batch.suspend_ordinary_checkpoint(
            uid, ("hybrid-revision", "failure-request")
        )
        handle = owner.file.handle
        assert any(
            contract[0] == "arrays-cache-v1" for contract in owner.cache_contracts
        )
        if failure == "corrupt":
            original = dc._read_payload
            monkeypatch.setattr(
                dc,
                "_read_payload",
                lambda *args: original(*args)[:-1] + b"x",
            )
            with pytest.raises(ValueError, match="digest"):
                batch.resume_ordinary_checkpoint(
                    owner, ("hybrid-revision", "failure-request")
                )
        owner.close()
        assert owner.closed and owner.control is None and handle.closed
        (replacement,) = batch.insert([[7, 8, 9]], max_tokens=[2])
        assert _next_output(batch).uid == replacement
    finally:
        if owner is not None:
            owner.close()
        batch.close()


def test_hybrid_checkpoint_refuses_speculative_recurrent_state():
    batch = BatchGenerator(
        _hybrid_model(), completion_batch_size=1, prefill_batch_size=1,
        prefill_step_size=32,
    )
    (uid,) = batch.insert([[1 + i % 126 for i in range(96)]], max_tokens=[4])
    try:
        _next_output(batch)
        recurrent = next(
            cache for cache in batch._generation_batch.prompt_cache
            if type(cache) is ArraysCache
        )
        recurrent.start_speculation()
        assert batch.ordinary_checkpoint_block(uid) == (
            "exact_checkpoint_cache_unsupported"
        )
        with pytest.raises(ValueError, match="cache_unsupported"):
            batch.suspend_ordinary_checkpoint(uid, ("hybrid-revision", "request"))
    finally:
        batch.close()


def test_hybrid_checkpoint_refuses_changed_cache_contract_then_retries_exactly():
    batch = BatchGenerator(
        _hybrid_model(), completion_batch_size=1, prefill_batch_size=1,
        prefill_step_size=32,
    )
    (uid,) = batch.insert([[1 + i % 126 for i in range(96)]], max_tokens=[4])
    owner = None
    try:
        _next_output(batch)
        owner = batch.suspend_ordinary_checkpoint(
            uid, ("hybrid-revision", "request")
        )
        contracts = owner.cache_contracts
        changed = list(contracts)
        recurrent_index = next(
            index for index, contract in enumerate(changed)
            if contract[0] == "arrays-cache-v1"
        )
        changed[recurrent_index] = ("arrays-cache-v2",)
        owner.cache_contracts = tuple(changed)
        with pytest.raises(ValueError, match="recurrent-cache contract changed"):
            batch.resume_ordinary_checkpoint(
                owner, ("hybrid-revision", "request")
            )
        assert owner.control is not None and not owner.closed
        owner.cache_contracts = contracts
        assert batch.resume_ordinary_checkpoint(
            owner, ("hybrid-revision", "request")
        ) == uid
        assert _next_output(batch).uid == uid
    finally:
        if owner is not None:
            owner.close()
        batch.close()


def test_hybrid_server_fault_uses_truthful_capability_owned_route_receipt(monkeypatch):
    patch_host(monkeypatch)
    reference, _, _, _ = _run_server(_hybrid_model(), sample=True)
    actual, receipt, counts, job = _run_server(
        _hybrid_model(), sample=True, fault=True
    )
    assert actual == reference
    assert counts["exact_decode_checkpoints"] == counts["exact_decode_resumes"] == 1
    assert receipt["preemption"]["exact_decode_route"] == {
        "implemented": True,
        "selected": True,
        "qualified": False,
        "observed_used": True,
        "scope": "same_executor_ordinary_b1_sdpa_capability_cache_v2",
    }
    assert len(receipt["preemption"]["events"]) == 1
    assert receipt["preemption"]["events"][0]["restored_without_prefill"] is True
    assert job.exact_decode_checkpoint is None
