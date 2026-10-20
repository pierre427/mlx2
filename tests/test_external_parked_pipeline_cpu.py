"""Pipelined parked decode on a hybrid (Mamba + attention) external route.

A parked session's anchor-only rounds overlap the host read of the previous
round (the PLD plain-round pipeline).  Output, caches and the paired draft
plane must equal the unpipelined route token for token.
"""
import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime import external_speculative as es
from mlx2.runtime.drafters.xpress import XPressDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.nemotron35_super import Model as SuperModel
from mlx2.runtime.models.nemotron_h import Model as LanguageModel
from tests.test_nemotron_external_taps_cpu import tiny_model

PROMPT = [1, 2, 3, 4, 5, 6]
_ROUND = es.ExternalDraftBatchGenerator._round
_READ = es.ExternalDraftBatchGenerator._read_parked


def _models():
    mx.random.seed(44)
    args = tiny_model().args
    model = SuperModel.__new__(SuperModel)
    LanguageModel.__init__(model, args)
    draft = XPressDraftModel(XPressConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=[0, 1, 4], block_size=4,
        layer_types=["full_attention"], xpress_rank=3,
        xpress_mlp_hidden=5, xpress_num_passes=3,
    )).bind(model)
    return model, draft


def _settings(**overrides):
    return {
        "enabled": True, "reversible": True, "ordinary_samples": 2,
        "speculative_samples": 2, "loss_decisions": 1, "window": 4,
        "reprobe_tokens": 6, "reprobe_max_tokens": 12, "settle_rounds": 0,
        **overrides,
    }


def _drain(engine, uid, *, remove_after=None):
    tokens, final = [], None
    for poll in range(400):
        try:
            _, responses = engine.next()
        except RuntimeError as error:
            if "injected" not in str(error):
                raise
            continue
        for response in responses:
            tokens.append(response.token)
            if response.finish_reason:
                final = response
        if remove_after is not None and len(tokens) >= remove_after and engine.lanes:
            engine.remove([uid])
            return tokens, None
        if not engine.lanes:
            return tokens, final
    raise AssertionError("scheduler stalled")


def _run(monkeypatch, *, handoff=None, speculative_ns=lambda _i: 100_000,
         max_tokens=40, stops=(), remove_after=None):
    model, draft = _models()
    clock = {"now": 1, "spec": 0}

    def timed_round(self, cohort, *, forced_depth=None):
        parked = all(es._speculation_parked(lane) for lane in cohort)
        result = _ROUND(self, cohort, forced_depth=forced_depth)
        if forced_depth == 0 or parked:
            clock["now"] += 100
        else:
            clock["now"] += speculative_ns(clock["spec"])
            clock["spec"] += 1
        return result

    def timed_read(self, inflight, rows, following):
        clock["now"] += 100
        return _READ(self, inflight, rows, following)

    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", timed_round)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_read_parked", timed_read)
    kwargs = {} if handoff is None else {"cost_per_commit_handoff": handoff}
    engine = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="parked-pipeline-cpu", num_draft=3,
        prefill_step_size=8, stop_tokens=[[t] for t in stops], **kwargs,
    )
    uid = engine.insert(
        [list(PROMPT)], max_tokens=[max_tokens],
        sampling_configs=[{"sampling_temp": 0.0}],
    )[0]
    tokens, final = _drain(engine, uid, remove_after=remove_after)
    return tokens, final, engine


def test_pipelined_park_matches_unparked_and_unpipelined_output(monkeypatch):
    baseline, _, _ = _run(monkeypatch)
    plain, plain_final, plain_engine = _run(
        monkeypatch, handoff=_settings(pipelined_park=False)
    )
    piped, final, engine = _run(monkeypatch, handoff=_settings())
    assert piped == plain == baseline
    stats = engine.scheduler_stats
    assert stats.get("external_parked_pipelined_rounds", 0) > 10
    assert stats.get("external_parked_pipeline_mismatches", 0) == 0
    assert plain_engine.scheduler_stats.get("external_parked_pipelined_rounds", 0) == 0
    # The final response's paired draft plane and cache equal the unpipelined
    # route's: the deferred context rode the same tail.
    final.cache_sidecar.validate("parked-pipeline-cpu", len(final.all_tokens))
    assert final.all_tokens == plain_final.all_tokens
    from mlx.utils import tree_flatten

    for left, right in zip(final.prompt_cache, plain_final.prompt_cache, strict=True):
        a_leaves = tree_flatten(left.state)
        b_leaves = tree_flatten(right.state)
        assert [key for key, _ in a_leaves] == [key for key, _ in b_leaves]
        for (_, a), (_, b) in zip(a_leaves, b_leaves, strict=True):
            if a is None or b is None:
                assert a is None and b is None
                continue
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_stop_token_inside_the_pipeline_finishes_at_the_same_boundary(monkeypatch):
    baseline, _, _ = _run(monkeypatch)
    stop = baseline[24]
    expected, _, _ = _run(monkeypatch, stops=(stop,))
    tokens, final, engine = _run(monkeypatch, handoff=_settings(), stops=(stop,))
    assert tokens == expected
    assert final.finish_reason == "stop"
    final.cache_sidecar.validate("parked-pipeline-cpu", len(final.all_tokens))
    assert engine._parked_inflight is None


def test_removing_a_lane_mid_flight_rewinds_its_unread_round(monkeypatch):
    tokens, _, engine = _run(monkeypatch, handoff=_settings(), remove_after=20)
    assert len(tokens) >= 20
    assert engine._parked_inflight is None
    assert engine.scheduler_stats.get("external_parked_pipeline_aborts", 0) == 1
    assert not engine.lanes


def test_pipelined_park_reengages_speculation(monkeypatch):
    price = lambda i: 100_000 if i < 2 else 10
    baseline, _, _ = _run(monkeypatch, speculative_ns=price)
    tokens, final, engine = _run(monkeypatch, handoff=_settings(), speculative_ns=price)
    assert tokens == baseline
    reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
    assert reversible["parks"] >= 1 and reversible["reengagements"] >= 1
    assert engine.scheduler_stats.get("external_parked_pipelined_rounds", 0) > 0
    final.cache_sidecar.validate("parked-pipeline-cpu", len(final.all_tokens))


def test_failed_pipelined_read_rewinds_exactly_and_the_lane_continues(monkeypatch):
    baseline, _, _ = _run(monkeypatch)
    commit = es.ExternalDraftBatchGenerator._commit
    state = {"reads": 0, "failed": False}

    def flaky(self, cohort, *args, **kwargs):
        if isinstance(kwargs.get("transaction"), es._CommittedPlainRows):
            state["reads"] += 1
            if state["reads"] == 7 and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("injected pipelined read failure")
        return commit(self, cohort, *args, **kwargs)

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_commit", flaky)
    tokens, final, engine = _run(monkeypatch, handoff=_settings())
    assert state["failed"]
    assert tokens == baseline
    assert engine.scheduler_stats["recovery_checkpoint_restores"] >= 1
    final.cache_sidecar.validate("parked-pipeline-cpu", len(final.all_tokens))


def test_width_growth_drains_seeded_inflight_before_revocation(monkeypatch):
    model, draft = _models()
    engine = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="parked-seed-width-growth-cpu",
        num_draft=3,
        prefill_step_size=8,
        cost_per_commit_handoff=_settings(),
    )
    engine.ast_cost_self_configure = True
    engine._ast_cost_prior = {
        "ordinary_ns_per_committed_token": 100,
        "speculative_ns_per_committed_token": 200,
        "speculative_to_ordinary_cost_ratio": 2,
    }
    uid = engine.insert(
        [list(PROMPT)],
        max_tokens=[40],
        sampling_configs=[{"sampling_temp": 0.0}],
    )[0]
    lane = engine.lanes[uid]
    engine._apply_ast_cost_prior_scope([lane], execution_width=1)
    for _ in range(20):
        engine.next()
        if engine._parked_inflight is not None:
            break
    assert engine._parked_inflight is not None
    assert lane.cost_handoff_state.session_prior_active

    engine.insert(
        [list(PROMPT)],
        max_tokens=[40],
        sampling_configs=[{"sampling_temp": 0.0}],
    )
    engine._decode_phase(list(engine.lanes.values()))
    assert engine._parked_inflight is None
    assert lane.cost_handoff_state.session_prior_active

    # Revocation is now safe: the old B1 transaction and recovery snapshot no
    # longer own the lane state, and the already-booked round remains visible.
    observations = lane.cost_handoff_state.receipt()["observations"]
    engine._apply_ast_cost_prior_scope([lane], execution_width=2)
    receipt = lane.cost_handoff_state.receipt()
    assert receipt["observations"] == observations
    assert receipt["reversible"]["session_seed_active"] is False
    assert receipt["reversible"]["events"][-1]["event"] == "seed_revoke"
