"""Actual recurrent GDN + KV target state under mixed adaptive verification.

All numerical work runs on CPU. Random tiny XPress weights establish execution
contracts; they do not represent a compatible trained hybrid checkpoint.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from test_qwen38_dflash2_cpu import ordinary_greedy, tiny_target

from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime.drafters.xpress import XPressDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.sample_utils import LaneRNG


def pair(monkeypatch):
    target = tiny_target()
    draft = XPressDraftModel(
        XPressConfig(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=128,
            mask_token_id=127,
            num_target_layers=8,
            target_layer_ids=[1, 6],
            block_size=4,
            layer_types=["full_attention"],
            xpress_rank=8,
            xpress_mlp_hidden=12,
            xpress_num_passes=2,
        )
    ).bind(target)
    original = draft.draft_distributions

    def propose(*args, **kwargs):
        result = original(*args, **kwargs)
        histories = kwargs["processor_histories"]
        draft.adaptive_confidence_features = [
            [40 if history[0] == 1 else -40] * len(tokens)
            for history, tokens in zip(histories, result[0])
        ]
        return result

    monkeypatch.setattr(draft, "draft_distributions", propose)
    policy = {
        "verification_costs": [0.5, 1, 1.4],
        "verification_costs_by_cohort": {"2": [1, 2, 10], "4": [2, 4, 100]},
        "mode": "per_request",
        "min_observations": 1,
    }
    engine = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="tiny-gdn-contract",
        num_draft=2,
        prefill_step_size=4,
        adaptive_verification=policy,
    )
    engine.acceptance_estimator.observe([0, 0], 0, rejected=True)
    engine.acceptance_estimator.rounds = 1
    return target, draft, engine


def assert_state(actual, expected):
    for got, want in zip(actual, expected):
        if isinstance(got, ArraysCache):
            assert not got.speculating and not got._rollbacks
            pairs = zip(got.cache, want.cache)
        else:
            assert got.offset == want.offset
            pairs = zip(got.state, want.state)
        for a, b in pairs:
            if a is None:
                assert b is None
                continue
            if not isinstance(got, ArraysCache):
                a, b = a[..., : got.offset, :], b[..., : want.offset, :]
            np.testing.assert_allclose(
                np.asarray(a), np.asarray(b), atol=2e-5, rtol=2e-5
            )


def prefill_all(engine):
    for lane in engine.lanes.values():
        while lane.remaining:
            engine._prefill(lane)


def test_actual_gdn_ragged_groups_commit_recurrent_prefix_only(monkeypatch):
    target, _, engine = pair(monkeypatch)
    engine.insert([[1, 2, 3], [2, 3, 4]], max_tokens=[7, 7])
    prefill_all(engine)
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    engine._round(list(engine.lanes.values()))
    assert shapes == [(1, 3), (1, 2)]
    for lane, prompt in zip(engine.lanes.values(), [[1, 2, 3], [2, 3, 4]]):
        emitted = [r.token for r in lane.ready]
        assert emitted == ordinary_greedy(target, prompt, len(emitted))
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)
        engine._sidecar(lane).validate("tiny-gdn-contract", len(lane.history))
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 5


def test_actual_gdn_later_group_failure_restores_committed_recurrent_rows(monkeypatch):
    target, _, engine = pair(monkeypatch)
    engine.insert(
        [[1, 2, 3], [2, 3, 4]],
        max_tokens=[7, 7],
        lane_rngs=[LaneRNG(9), LaneRNG(10)],
        sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8}],
    )
    prefill_all(engine)
    lanes = list(engine.lanes.values())
    old = copy.deepcopy([lane.__dict__ for lane in lanes])
    original = target.forward_with_taps
    calls = [0]

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("GDN second group after state write")
        return result

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="GDN second"):
        engine._round(lanes)
    assert calls[0] == 2
    for lane, previous in zip(lanes, old):
        assert lane.history == previous["history"] and not lane.ready
        assert lane.rng.snapshot() == previous["rng"].snapshot()
        assert_state(lane.cache, previous["cache"])
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 0
    monkeypatch.setattr(target, "forward_with_taps", original)
    engine._round(lanes)
    assert all(lane.ready for lane in lanes)


class StopGrammar:
    """Stateful processor with the same provisional-ledger contract as grammars."""

    def __init__(self):
        self.steps = 0

    def __call__(self, tokens, logits):
        self.steps += 1
        return mx.where(mx.arange(logits.shape[-1])[None] == 0, 0.0, -float("inf"))

    def verify_ledger(self):
        return self.steps

    def restore_verify_ledger(self, ledger):
        self.steps = ledger


def test_mixed_sampling_stop_grammar_settles_only_delivered_rows(monkeypatch):
    target, _, engine = pair(monkeypatch)
    engine.stops = {0}
    grammar = StopGrammar()
    engine.insert(
        [[1, 2, 3], [2, 3, 4]],
        max_tokens=[8, 8],
        lane_rngs=[LaneRNG(12), LaneRNG(23)],
        logits_processors=[[grammar], []],
        sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8, "top_k": 16}],
    )
    prefill_all(engine)
    lanes = list(engine.lanes.values())
    engine._round(lanes)
    assert [r.token for r in lanes[0].ready] == [0]
    assert lanes[0].ready[0].finish_reason == "stop"
    # Three target rows were processed, but stop after the first accepted token
    # leaves only one ledger step. Draft-side probes did not touch this owner.
    assert grammar.steps == 1
    for lane in lanes:
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)
        engine._sidecar(lane).validate("tiny-gdn-contract", len(lane.history))
    assert lanes[1].ready


def test_cancellation_and_zero_budget_boundary_preserve_surviving_gdn_lane(monkeypatch):
    target, _, engine = pair(monkeypatch)
    prompts = [[1, 2, 3], [2, 3, 4]]
    engine.insert(prompts, max_tokens=[1, 7])
    prefill_all(engine)
    engine.remove([0], cancelled=True)
    output = []
    final = None
    for _ in range(40):
        _, responses = engine.next()
        output.extend(r.token for r in responses if r.uid == 1)
        for response in responses:
            if response.uid == 1 and response.finish_reason:
                final = response
        if not engine.lanes:
            break
    assert output == ordinary_greedy(target, prompts[1], 7)
    assert final is not None and engine.scheduler_stats["cancelled"] == 1
    final.cache_sidecar.validate("tiny-gdn-contract", len(final.all_tokens))
    assert final.speculative_receipt["current_execution"] == "ordinary_target"


def test_actual_gdn_batches_mixed_temperatures_within_both_depth_groups(monkeypatch):
    from mlx2.runtime.speculative_sampling import softmax

    target, _, engine = pair(monkeypatch)
    prompts = [[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]]
    engine.insert(
        prompts,
        max_tokens=[8] * 4,
        lane_rngs=[LaneRNG(i + 50) for i in range(4)],
        sampling_configs=[
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.8},
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.8},
        ],
    )
    prefill_all(engine)
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    engine._round(list(engine.lanes.values()))
    assert shapes == [(2, 3), (2, 2)]
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 10
    assert engine.scheduler_stats["target_max_width"] == 2
    for uid in (0, 2):
        ready = list(engine.lanes[uid].ready)
        assert [r.token for r in ready] == ordinary_greedy(
            target, prompts[uid], len(ready)
        )
    for uid in (1, 3):
        actual = np.exp(np.asarray(engine.lanes[uid].ready[0].logprobs))
        logits = np.asarray(
            target(mx.array([prompts[uid]]), cache=target.make_cache())[0, -1]
        )
        np.testing.assert_allclose(actual, softmax(logits, 0.8), rtol=2e-5, atol=2e-6)
    for lane in engine.lanes.values():
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)


def test_grammar_and_hidden_steer_cleanup_survive_later_group_failure(monkeypatch):
    from types import SimpleNamespace

    target, _, engine = pair(monkeypatch)
    grammar = StopGrammar()
    engine.stops = {0}
    engine.insert(
        [[1, 2, 3], [2, 3, 4]], max_tokens=[8, 8], logits_processors=[[grammar], []]
    )
    prefill_all(engine)
    owners = []

    def steering(*args):
        owner = SimpleNamespace(steer=None)
        owners.append(owner)
        return owner, object(), []

    monkeypatch.setattr(engine, "_verify_steer", steering)
    original = target.forward_with_taps
    calls = [0]

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("second steer group")
        return result

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="second steer"):
        engine._round(list(engine.lanes.values()))
    assert grammar.steps == 0
    assert len(owners) == 2 and all(owner.steer is None for owner in owners)
    assert all(not lane.ready for lane in engine.lanes.values())


def mamba_pair(monkeypatch):
    from test_nemotron_external_taps_cpu import tiny_model

    target = tiny_model()
    draft = XPressDraftModel(
        XPressConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
            mask_token_id=15,
            num_target_layers=5,
            target_layer_ids=[0, 1, 4],
            block_size=4,
            layer_types=["full_attention"],
            xpress_rank=3,
            xpress_mlp_hidden=5,
            xpress_num_passes=2,
        )
    ).bind(target)
    original = draft.draft_distributions

    def propose(*args, **kwargs):
        result = original(*args, **kwargs)
        draft.adaptive_confidence_features = [
            [40 if history[0] == 1 else -40] * len(tokens)
            for history, tokens in zip(kwargs["processor_histories"], result[0])
        ]
        return result

    monkeypatch.setattr(draft, "draft_distributions", propose)
    engine = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="tiny-mamba-contract",
        num_draft=2,
        prefill_step_size=4,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10], "4": [2, 4, 100]},
            "mode": "per_request",
            "min_observations": 1,
        },
    )
    engine.acceptance_estimator.observe([0, 0], 0, rejected=True)
    engine.acceptance_estimator.rounds = 1
    return target, engine


def test_actual_mamba2_mixed_sampling_batched_depths_preserve_conv_ssm_kv(monkeypatch):
    target, engine = mamba_pair(monkeypatch)
    prompts = [[1, 2, 3], [1, 2], [2, 3, 4], [2, 3]]
    engine.insert(
        prompts,
        max_tokens=[8] * 4,
        lane_rngs=[LaneRNG(i + 61) for i in range(4)],
        sampling_configs=[
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.8},
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.8},
        ],
    )
    prefill_all(engine)
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    engine._round(list(engine.lanes.values()))
    assert shapes == [(2, 3), (2, 2)]
    assert len(target.layers) == 5 and len(target.make_cache()) == 3
    for lane in engine.lanes.values():
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)
        engine._sidecar(lane).validate("tiny-mamba-contract", len(lane.history))
        for item in lane.ready:
            protocol = item.speculative_receipt["target_protocol"]
            assert protocol["single_row_verification"] == "ordinary_tokenwise"
            assert protocol["batched_verification"] == "block_candidate"
            assert protocol["batched_numerical_qualification"] is False
            assert protocol["real_artifact_qualification"] is False
    for uid in (0, 2):
        ready = list(engine.lanes[uid].ready)
        assert [r.token for r in ready] == ordinary_greedy(
            target, prompts[uid], len(ready)
        )


def test_actual_mamba2_group_failure_restores_conv_ssm_and_rng(monkeypatch):
    target, engine = mamba_pair(monkeypatch)
    engine.insert(
        [[1, 2, 3], [2, 3, 4]],
        max_tokens=[8, 8],
        lane_rngs=[LaneRNG(70), LaneRNG(71)],
        sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8}],
    )
    prefill_all(engine)
    lanes = list(engine.lanes.values())
    old = copy.deepcopy([l.__dict__ for l in lanes])
    original = target.forward_with_taps
    calls = [0]

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("Mamba2 second group")
        return result

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="Mamba2 second"):
        engine._round(lanes)
    for lane, previous in zip(lanes, old):
        assert lane.history == previous["history"] and not lane.ready
        assert lane.rng.snapshot() == previous["rng"].snapshot()
        assert_state(lane.cache, previous["cache"])
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 0
    monkeypatch.setattr(target, "forward_with_taps", original)
    engine._round(lanes)
    assert all(lane.ready for lane in lanes)


def test_actual_mamba2_continuous_batch_stop_and_final_budget_boundary(monkeypatch):
    target, engine = mamba_pair(monkeypatch)
    engine.stops = {0}
    grammar = StopGrammar()
    prompts = [[1, 2, 3], [2, 3, 4], [1, 2]]
    ids = engine.insert(
        prompts,
        max_tokens=[8, 7, 1],
        logits_processors=[[grammar], [], []],
        lane_rngs=[LaneRNG(80), LaneRNG(81), LaneRNG(82)],
        sampling_configs=[
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.0},
            {"sampling_temp": 0.8},
        ],
    )
    output = {}
    final = {}
    for _ in range(40):
        _, responses = engine.next()
        for r in responses:
            output.setdefault(r.uid, []).append(r.token)
            if r.finish_reason:
                final[r.uid] = r
        if not engine.lanes:
            break
    assert output[ids[0]] == [0] and grammar.steps == 1
    expected = ordinary_greedy(target, prompts[1], 7)
    if 0 in expected:
        expected = expected[: expected.index(0) + 1]
    assert output[ids[1]] == expected
    assert len(output[ids[2]]) == 1
    for end in final.values():
        end.cache_sidecar.validate("tiny-mamba-contract", len(end.all_tokens))


def test_tensorfold_dispatch_seam_groups_depths_with_cpu_hybrid_substitute(monkeypatch):
    """Exercise real executor dispatch signatures, not native TensorFold kernels."""
    from mlx2.runtime import qwen38_tensorfold
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "tensorfold")
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/tmp/unused-native-seam-test")
    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", "4")
    dispatched = []

    def substitute(
        model,
        token_rows,
        parent_rows,
        caches,
        capture_layers,
        source_root,
        *,
        cached=False,
    ):
        assert source_root == "/tmp/unused-native-seam-test" and not cached
        for tokens, parents in zip(token_rows, parent_rows):
            assert parents == list(range(-1, len(tokens) - 1))
        assert len({id(row) for row in caches}) == len(caches)
        dispatched.append((len(token_rows), len(token_rows[0])))
        tx = HybridVerifyRows(caches).begin([len(row) for row in token_rows])
        logits, hidden = model.forward_with_taps(
            mx.array(token_rows), tx.caches, capture_layers
        )
        return logits, hidden, tx

    def single(
        model, tokens, parents, cache, capture_layers, source_root, *, cached=False
    ):
        return substitute(
            model,
            [tokens],
            [parents],
            [cache],
            capture_layers,
            source_root,
            cached=cached,
        )

    monkeypatch.setattr(qwen38_tensorfold, "forward_many", substitute)
    monkeypatch.setattr(qwen38_tensorfold, "forward", single)
    target, _, engine = pair(monkeypatch)
    engine.insert(
        [[1, 2, 3], [1, 2], [2, 3, 4], [2, 3]],
        max_tokens=[8] * 4,
        sampling_configs=[{"sampling_temp": 0}, {"sampling_temp": 0.8}] * 2,
    )
    prefill_all(engine)
    engine._round(list(engine.lanes.values()))
    assert dispatched == [(2, 3), (2, 2)]
    assert engine.scheduler_stats["external_tensorfold_cohort_rounds"] == 2
    for lane in engine.lanes.values():
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_state(lane.cache, fresh)


def qsa_pair(monkeypatch):
    from qsa_oracle import tiny_args

    from mlx2.runtime.models import qwen4_exp

    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    monkeypatch.setattr(qwen4_exp, "_QSA_APC_SUMMARIES", True)
    target = qwen4_exp.TextModel(
        tiny_args(linear_key_head_dim=8, linear_value_head_dim=8, indexer_budget=4)
    )
    draft = XPressDraftModel(
        XPressConfig(
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            mask_token_id=63,
            num_target_layers=4,
            target_layer_ids=[0, 3],
            block_size=4,
            layer_types=["full_attention"],
            xpress_rank=4,
            xpress_mlp_hidden=8,
            xpress_num_passes=2,
        )
    ).bind(target)
    original = draft.draft_distributions

    def propose(*args, **kwargs):
        result = original(*args, **kwargs)
        draft.adaptive_confidence_features = [
            [40 if history[0] == 1 else -40] * len(tokens)
            for history, tokens in zip(kwargs["processor_histories"], result[0])
        ]
        return result

    monkeypatch.setattr(draft, "draft_distributions", propose)
    engine = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="tiny-qsa-contract",
        num_draft=2,
        prefill_step_size=4,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "verification_costs_by_cohort": {"2": [1, 2, 10], "4": [2, 4, 100]},
            "mode": "per_request",
            "min_observations": 1,
        },
    )
    engine.acceptance_estimator.observe([0, 0], 0, rejected=True)
    engine.acceptance_estimator.rounds = 1
    return target, engine


def assert_qsa_state(actual, expected):
    from test_external_qsa_verify_rows_cpu import _assert_qsa

    assert_state(actual, expected)
    _assert_qsa(actual[-1], expected[-1])


def test_full_qsa_gdn_mixed_sampling_groups_preserve_index_and_pool_planes(monkeypatch):
    target, engine = qsa_pair(monkeypatch)
    prompts = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]
    engine.insert(
        prompts,
        max_tokens=[8] * 4,
        lane_rngs=[LaneRNG(i + 91) for i in range(4)],
        sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8}] * 2,
    )
    prefill_all(engine)
    shapes = []
    original = target.forward_with_taps

    def forward(tokens, *args, **kwargs):
        shapes.append(tuple(tokens.shape))
        return original(tokens, *args, **kwargs)

    monkeypatch.setattr(target, "forward_with_taps", forward)
    engine._round(list(engine.lanes.values()))
    assert shapes == [(2, 3), (2, 2)]
    for lane in engine.lanes.values():
        fresh = target.make_cache()
        mx.eval(target(mx.array([lane.history]), cache=fresh))
        assert_qsa_state(lane.cache, fresh)
        engine._sidecar(lane).validate("tiny-qsa-contract", len(lane.history))
    for uid in (0, 2):
        ready = list(engine.lanes[uid].ready)
        assert [r.token for r in ready] == ordinary_greedy(
            target, prompts[uid], len(ready)
        )
    from mlx2.runtime.speculative_sampling import softmax

    for uid in (1, 3):
        logits = np.asarray(
            target(mx.array([prompts[uid]]), cache=target.make_cache())[0, -1]
        )
        actual = np.exp(np.asarray(engine.lanes[uid].ready[0].logprobs))
        np.testing.assert_allclose(actual, softmax(logits, 0.8), atol=3e-6, rtol=3e-5)


def test_full_qsa_gdn_group_failure_restores_index_pool_recurrent_and_rng(monkeypatch):
    target, engine = qsa_pair(monkeypatch)
    engine.insert(
        [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]],
        max_tokens=[8, 8],
        lane_rngs=[LaneRNG(100), LaneRNG(101)],
        sampling_configs=[{"sampling_temp": 0.0}, {"sampling_temp": 0.8}],
    )
    prefill_all(engine)
    lanes = list(engine.lanes.values())
    old = copy.deepcopy([l.__dict__ for l in lanes])
    original = target.forward_with_taps
    calls = [0]

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("QSA second group")
        return result

    monkeypatch.setattr(target, "forward_with_taps", fail)
    with pytest.raises(RuntimeError, match="QSA second"):
        engine._round(lanes)
    for lane, previous in zip(lanes, old):
        assert lane.history == previous["history"] and not lane.ready
        assert lane.rng.snapshot() == previous["rng"].snapshot()
        assert_qsa_state(lane.cache, previous["cache"])
    assert engine.scheduler_stats["external_adaptive_target_rows"] == 0
    monkeypatch.setattr(target, "forward_with_taps", original)
    engine._round(lanes)
    assert all(lane.ready for lane in lanes)
