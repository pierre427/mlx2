"""Exact PLD/native-MTP arbitration with real synthetic heads, strictly CPU."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_batched_mtp import _tiny_qwen4_model
from test_parallel_draft_apcv2_batching_cpu import (
    collect,
    pair,
    serving_engine,
)
from test_standard_xpress_serving_cpu import drain, generator, reference, tiny

from mlx2.adapters.external_draft_policy import (
    ExternalDraftAdapterMixin,
    normalize_progressive_external_policy,
    progressive_external_identity,
)
from mlx2.adapters.proposal_sources import native_mtp_source
from mlx2.runtime.proposal_composition import (
    ComposedDraftModel,
    PromptCopySourceRange,
    ProposalCompositionPolicy,
)


def test_progressive_external_policy_is_shared_and_identity_bound():
    policy = normalize_progressive_external_policy(
        {
            "progressive_verification_tile": 3,
            "progressive_multilane_draft_cap": 5,
            "progressive_alternative_branches": 4,
            "progressive_alternative_strategy": "shared_prefix",
            "progressive_alternative_depth_policy": "full",
        },
        num_draft=15,
        minimum_proposals=3,
    )
    assert policy["proposal_composition"] is False
    assert policy["progressive_verification_tile"] == 3
    assert policy["progressive_multilane_draft_cap"] == 5
    assert policy["progressive_alternative_branches"] == 4
    assert policy["progressive_alternative_strategy"] == "shared_prefix"
    assert policy["progressive_alternative_depth_policy"] == "full"
    assert progressive_external_identity(policy) != progressive_external_identity(
        {**policy, "progressive_multilane_draft_cap": 4}
    )


def test_external_adapter_mix_in_selects_and_reports_progressive_policy():
    adapter = ExternalDraftAdapterMixin()
    adapter._parse_external_policy(
        {
            "draft_model": "/unused/drafter",
            "num_draft": 8,
            "progressive_verification_tile": 2,
            "progressive_multilane_draft_cap": 3,
            "progressive_alternative_branches": 4,
            "progressive_alternative_strategy": "shared_prefix",
        },
        family="test",
    )
    adapter._check_num_draft(
        {"args": SimpleNamespace(block_size=16, minimum_proposal_length=3)}
    )
    settings = adapter._external_execution_config(max_lanes=4, prefill_step=512)
    assert adapter.external_policy["proposal_composition"] is False
    assert settings["progressive_verification_tile"] == 2
    assert settings["progressive_multilane_draft_cap"] == 3
    assert settings["progressive_alternative_branches"] == 4
    assert settings["progressive_alternative_strategy"] == "shared_prefix"


def test_shared_external_policy_rejects_unconsumed_fold_key():
    adapter = ExternalDraftAdapterMixin()
    with pytest.raises(ValueError, match="no qualified overrides"):
        adapter._parse_external_policy(
            {"draft_model": "/unused/drafter", "fold": {"window": 4096}},
            family="test",
        )


@pytest.mark.parametrize(
    "policy, kwargs, message",
    [
        ({"progressive_verification_tile": True}, {"num_draft": 8}, "integer"),
        ({"progressive_verification_tile": 8}, {"num_draft": 8}, "below num_draft"),
        ({"progressive_multilane_draft_cap": 3}, {"num_draft": 8}, "requires progressive"),
        ({"progressive_alternative_branches": 2}, {"num_draft": 8}, "requires progressive"),
        (
            {
                "progressive_verification_tile": 2,
                "progressive_multilane_draft_cap": 1,
            },
            {"num_draft": 8, "minimum_proposals": 2},
            "between minimum proposals",
        ),
        (
            {
                "progressive_verification_tile": 2,
                "proposal_composition": {"prompt_lookup": True},
            },
            {"num_draft": 8},
            "cannot combine",
        ),
        (
            {
                "progressive_verification_tile": 2,
                "lilicorr_feedback": {"enabled": True},
            },
            {"num_draft": 8},
            "LiLiCorr feedback",
        ),
    ],
)
def test_progressive_external_policy_refuses_incompatible_values(policy, kwargs, message):
    with pytest.raises(ValueError, match=message):
        normalize_progressive_external_policy(policy, **kwargs)


def test_composition_ranks_committed_score_first_and_span_second():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft, {"ngram_min": 1, "ngram_max": 1}
    )
    short_pld = ("prompt_lookup", [7, 8], "prompt_lookup:match_4")
    full_external = ("external", None, "external")
    scores = {}

    # Cold scores tie, so the longer usable chain wins.
    assert wrapper._select_candidate([short_pld, full_external], 4)[0] == "external"

    # Only committed labels move scores. A stronger PLD score beats length.
    wrapper.observe_proposal_feedback(scores, "prompt_lookup:match_4", 8, 8)
    wrapper.observe_proposal_feedback(scores, "external", 8, 0)
    selected = wrapper._select_candidate([short_pld, full_external], 4, scores)
    assert selected[0] == "prompt_lookup"
    assert wrapper._proposal_score(selected[2], scores) > wrapper._proposal_score("external", scores)


def test_composition_scores_are_lane_local_and_decay_by_proposed_token_mass():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(draft, {"ngram_min": 1, "ngram_max": 1})
    lane_a, lane_b = {}, {}
    wrapper.observe_proposal_feedback(lane_a, "external", 8, 8, 0.5)
    assert lane_b == {}
    assert lane_a["external"] == (8, 8)
    wrapper.observe_proposal_feedback(lane_a, "external", 2, 0, 0.5)
    assert lane_a["external"] == (2.0, 4.0)
    assert lane_b == {}


def test_unselected_score_bucket_ages_and_changed_source_can_compete():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(draft, {"ngram_min": 1, "ngram_max": 1})
    scores = {"prompt_lookup:match_4": (100, 100), "external": (0, 100)}
    stale_copy = ("prompt_lookup", [1, 2], "prompt_lookup:match_4")
    external = ("external", None, "external")
    assert wrapper._select_candidate([stale_copy, external], 2, scores)[0] == "prompt_lookup"
    for _ in range(8):
        wrapper.observe_proposal_feedback(scores, "external", 8, 8, 0.5)
    assert scores["prompt_lookup:match_4"][1] < 1e-12
    assert wrapper._select_candidate([stale_copy, external], 2, scores)[0] == "external"


@pytest.mark.parametrize("counts", [(float("nan"), 1), (2, 1), (-1, 1), (True, 2), (1, float("inf"))])
def test_invalid_score_evidence_cannot_change_source_selection(counts):
    _model, draft = tiny()
    wrapper = ComposedDraftModel(draft, {"ngram_min": 1, "ngram_max": 1})
    scores = {"external": counts}
    with pytest.raises(ValueError, match="score evidence"):
        wrapper._proposal_score("external", scores)
    with pytest.raises(ValueError, match="score evidence"):
        wrapper.observe_proposal_feedback(scores, "external", 2, 2)
    assert scores == {"external": counts}


def test_composition_equal_score_and_span_has_stable_source_order():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft,
        {"ngram_min": 1, "ngram_max": 1, "native_mtp": False},
    )
    selected = wrapper._select_candidate(
        [
            ("external", None, "external"),
            ("prompt_lookup", [4, 5], "prompt_lookup:match_3"),
        ],
        2,
    )
    assert selected[0] == "prompt_lookup"


def test_stochastic_safe_source_choice_is_frozen_before_backend_call(monkeypatch):
    model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft, {"ngram_min": 1, "ngram_max": 1, "min_context_match": 0}
    )
    original = draft.draft_distributions
    scores = {}

    def mutate_lagged_score_after_selection(*args, **kwargs):
        wrapper.observe_proposal_feedback(scores, "external", 100, 100)
        return original(*args, **kwargs)

    monkeypatch.setattr(draft, "draft_distributions", mutate_lagged_score_after_selection)
    history, anchor = [1, 2, 1], 2
    hidden = model.prefill_body(
        mx.array([[1, 2]]), model.make_cache(), [0, 2]
    )
    wrapper.draft_distributions(
        [anchor],
        hidden,
        wrapper.make_cache(),
        2,
        [None],
        [0],
        processor_histories=[history],
        proposal_score_states=[scores],
    )
    assert wrapper.last_proposal_sources == ("prompt_lookup",)
    assert wrapper.last_proposal_score_keys == ("prompt_lookup:match_1",)


def test_trusted_pld_calibrates_cold_bucket_then_returns_to_score_order():
    model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_verified_tokens": 32,
            "trusted_pld_min_match": 1,
        },
    )
    scores = {}
    trust = {}
    wrapper.observe_proposal_feedback(scores, "external", 100, 100)
    history, anchor = [1, 2, 1], 2
    hidden = model.prefill_body(
        mx.array([[1, 2]]), model.make_cache(), [0, 2]
    )

    def propose():
        wrapper.draft_distributions(
            [anchor],
            hidden,
            wrapper.make_cache(),
            2,
            [None],
            [0],
            processor_histories=[history],
            proposal_score_states=[scores],
            proposal_trust_states=[trust],
            proposal_source_ranges=[[PromptCopySourceRange("turn-A", "assistant", 0, len(history) + 1)]],
        )

    propose()
    assert wrapper.last_proposal_sources == ("prompt_lookup",)
    assert wrapper.last_proposal_audits == (True,)
    assert wrapper.last_proposal_periodic_audits == (False,)
    trust_key = wrapper.last_proposal_trust_keys[0]
    assert trust_key.startswith("prompt_lookup:match_1:span_2:source_")
    wrapper.observe_proposal_feedback(scores, "prompt_lookup:match_1", 32, 32)
    wrapper.observe_trusted_pld(
        trust, trust_key, proposed=32, audited=True, accepted=32
    )
    wrapper.observe_proposal_feedback(scores, "external", 100, 100)
    propose()
    assert wrapper.last_proposal_sources == ("external",)
    assert wrapper.last_proposal_audits == (False,)


def test_trusted_pld_requires_a_positive_periodic_audit_when_enabled():
    _model, draft = tiny()
    with pytest.raises(ValueError, match="positive audit interval"):
        ComposedDraftModel(
            draft,
            {
                "prompt_lookup": True,
                "trusted_pld": True,
                "trusted_pld_audit_interval": 0,
            },
        )


@pytest.mark.parametrize("role", [None, "user", "tool", "system", "assistant"])
def test_trusted_copy_requires_declared_assistant_origin_and_cannot_transfer(role):
    model, draft = tiny()
    wrapper = ComposedDraftModel(draft, {
        "ngram_min": 1, "ngram_max": 1, "min_context_match": 0,
        "trusted_pld": True, "trusted_pld_min_match": 1,
        "trusted_pld_min_verified_tokens": 1, "trusted_pld_min_score": 0.5,
    })
    history, anchor = [1, 2, 1], 2
    hidden = model.prefill_body(mx.array([[1, 2]]), model.make_cache(), [0, 2])
    trust = {}

    def propose(origin="turn-A", tokens=history, ranges_override=None):
        ranges = None if role is None else [PromptCopySourceRange(origin, role, 0, 4)]
        wrapper.draft_distributions([anchor], hidden, wrapper.make_cache(), 2,
            [None], [0], processor_histories=[tokens], proposal_trust_states=[trust],
            proposal_source_ranges=[ranges if ranges_override is None else ranges_override])
        return wrapper.last_proposal_trust_keys[0]

    key = propose()
    wrapper.observe_trusted_pld(trust, key, proposed=2, audited=True, accepted=2)
    propose()
    assert wrapper.last_proposal_trusted == (role == "assistant",)
    assert wrapper.last_proposal_audits == (role != "assistant",)
    if role == "assistant":
        other = propose("turn-B")
        assert other != key and wrapper.last_proposal_trusted == (False,)
        assert wrapper.last_proposal_audits == (True,)
        # Equal-sized but unrelated copied content does not inherit labels.
        assert propose(tokens=[7, 2, 7]) != key
        assert wrapper.last_proposal_trusted == (False,)
        # A source range that covers only part of the continuation is excluded.
        propose(ranges_override=[PromptCopySourceRange("turn-A", "assistant", 0, 2)])
        assert wrapper.last_proposal_trusted == (False,)
        # Overlapping declarations are ambiguous even if both say assistant.
        propose(ranges_override=[PromptCopySourceRange("turn-A", "assistant", 0, 4)] * 2)
        assert wrapper.last_proposal_trusted == (False,)
        # Partial overlap must invalidate previously calibrated trust too.
        for overlap_role in ("user", "tool", "system", "assistant"):
            assert propose(ranges_override=[
                PromptCopySourceRange("turn-A", "assistant", 0, 4),
                PromptCopySourceRange("turn-B", overlap_role, 2, 3),
            ]).endswith(":source_unknown")
            assert wrapper.last_proposal_trusted == (False,)
            assert wrapper.last_proposal_audits == (True,)


@pytest.mark.parametrize("role", ["user", "tool", "system", "assistant"])
@pytest.mark.parametrize("start,end", [(2, 3), (0, 2), (2, 4), (1, 3)])
def test_trusted_copy_rejects_any_second_overlapping_declaration(role, start, end):
    # Target lookup maps the copied continuation to full-history [1, 3).
    lookup = SimpleNamespace(last_source=("target", 1, 0, 2), tokens=[1, 2, 1, 2])
    ranges = [PromptCopySourceRange("assistant-turn", "assistant", 0, 4),
              PromptCopySourceRange("another-turn", role, start, end)]
    assert ComposedDraftModel._copy_trust_source(lookup, [2, 1], 3, ranges) is None


def test_trusted_copy_half_open_adjacent_declarations_are_not_overlaps():
    lookup = SimpleNamespace(last_source=("target", 1, 0, 2), tokens=[1, 2, 1, 2])
    owner = PromptCopySourceRange("assistant-turn", "assistant", 1, 3)
    key = ComposedDraftModel._copy_trust_source(lookup, [2, 1], 3, [owner])
    assert key is not None
    assert ComposedDraftModel._copy_trust_source(lookup, [2, 1], 3, [
        PromptCopySourceRange("preceding", "user", 0, 1), owner,
        PromptCopySourceRange("following", "tool", 3, 4),
    ]) == key


@pytest.mark.parametrize("field,value", [
    ("exact_proposed_tokens", "32"), ("exact_accepted_tokens", 32.5),
    ("exact_proposed_tokens", True), ("blind_since_audit", float("nan")),
    ("revoked", 0),
])
def test_trusted_copy_corrupt_lane_evidence_forces_exact_audit(field, value):
    _model, draft = tiny()
    wrapper = ComposedDraftModel(draft, {"trusted_pld": True})
    record = {"exact_proposed_tokens": 32, "exact_accepted_tokens": 32,
              "blind_since_audit": 0, "revoked": False, field: value}
    assert not wrapper._trusted_evidence_ready(record)


def test_trusted_pld_cadence_and_revocation_are_lane_local():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft,
        {
            "prompt_lookup": True,
            "trusted_pld": True,
            "trusted_pld_min_verified_tokens": 4,
            "trusted_pld_audit_interval": 8,
        },
    )
    lane_a = {
        "prompt_lookup:match_4:span_2": {
            "blind_since_audit": 7,
            "blind_rounds": 7,
            "blind_tokens": 14,
            "audit_rounds": 0,
            "audit_proposed_tokens": 0,
            "audit_accepted_tokens": 0,
            "audit_failures": 0,
            "revoked": False,
            "exact_accepted_tokens": 4,
            "exact_proposed_tokens": 4,
        }
    }
    lane_b = {}

    wrapper.observe_trusted_pld(
        lane_a,
        "prompt_lookup:match_4:span_2",
        proposed=2,
        audited=True,
        accepted=1,
    )
    assert lane_a["prompt_lookup:match_4:span_2"]["revoked"]
    assert lane_a["prompt_lookup:match_4:span_2"]["audit_failures"] == 1
    assert lane_a["prompt_lookup:match_4:span_2"]["exact_proposed_tokens"] == 0
    assert lane_b == {}

    wrapper.observe_trusted_pld(
        lane_a,
        "prompt_lookup:match_4:span_2",
        proposed=2,
        audited=True,
        accepted=2,
    )
    assert lane_a["prompt_lookup:match_4:span_2"]["revoked"]
    assert lane_a["prompt_lookup:match_4:span_2"]["exact_proposed_tokens"] == 2
    wrapper.observe_trusted_pld(
        lane_a,
        "prompt_lookup:match_4:span_2",
        proposed=2,
        audited=True,
        accepted=2,
    )
    assert not lane_a["prompt_lookup:match_4:span_2"]["revoked"]
    assert lane_a["prompt_lookup:match_4:span_2"]["exact_proposed_tokens"] == 4


def test_trusted_pld_periodic_audit_schedule_is_per_lane_and_source_bucket():
    model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
            "trusted_pld_audit_interval": 8,
        },
    )
    history, anchor = [1, 2, 1], 2
    hidden = model.prefill_body(
        mx.array([[1, 2]]), model.make_cache(), [0, 2]
    )
    scores = {
        "prompt_lookup:match_1": (64.0, 64.0),
        "external": (0.0, 64.0),
    }

    def propose(trust_state):
        wrapper.draft_distributions(
            [anchor],
            hidden,
            wrapper.make_cache(),
            2,
            [None],
            [0],
            processor_histories=[history],
            proposal_score_states=[dict(scores)],
            proposal_trust_states=[trust_state],
            proposal_source_ranges=[[PromptCopySourceRange("turn-A", "assistant", 0, len(history) + 1)]],
        )
        return (
            wrapper.last_proposal_trusted,
            wrapper.last_proposal_audits,
            wrapper.last_proposal_periodic_audits,
        )

    propose({})
    key = wrapper.last_proposal_trust_keys[0]
    lane_a, lane_b = {}, {}
    wrapper.observe_trusted_pld(
        lane_a, key, proposed=2, audited=True, accepted=2
    )
    wrapper.observe_trusted_pld(
        lane_b, key, proposed=2, audited=True, accepted=2
    )
    assert lane_a[key]["periodic_audit_rounds"] == 0
    for _ in range(7):
        assert propose(lane_a) == ((True,), (False,), (False,))
        wrapper.observe_trusted_pld(
            lane_a, key, proposed=2, audited=False
        )
    # Exactly N-1 blind rounds are permitted; the Nth is exact-audited.
    assert propose(lane_a) == ((False,), (True,), (True,))
    wrapper.observe_trusted_pld(
        lane_a,
        key,
        proposed=2,
        audited=True,
        periodic_audit=True,
        accepted=2,
    )
    assert lane_a[key]["periodic_audit_rounds"] == 1
    assert lane_a[key]["audit_rounds"] == 2
    # Another lane's schedule and another span bucket are independent.
    assert propose(lane_b) == ((True,), (False,), (False,))
    assert lane_b[key]["periodic_audit_rounds"] == 0
    assert "prompt_lookup:match_1:span_1" not in lane_b
    assert not wrapper._trusted_evidence_ready(
        lane_b.get("prompt_lookup:match_1:span_1", {})
    )


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_copy_rows_exact_q_committed_index_and_backend_cache(kind):
    model, draft = pair(kind, [2])
    wrapped = ComposedDraftModel(
        draft,
        {"ngram_min": 1, "ngram_max": 1, "min_context_match": 0},
    )
    histories, anchors = [[1, 2, 3, 1], [4, 5]], [2, 6]
    hidden = model.prefill_body(mx.array([[1, 2], [4, 5]]), model.make_cache(), [0, 2])
    caches = [wrapped.make_cache(), wrapped.make_cache()]
    tokens, laws = wrapped.draft_distributions(
        anchors,
        hidden,
        wrapped.batch_caches(caches),
        2,
        [None, None],
        [0, 0],
        processor_histories=histories,
    )
    assert tokens[0] == [3, 1]
    assert wrapped.last_proposal_sources == ("prompt_lookup", "external")
    assert wrapped.adaptive_confidence_features[0] is None
    assert wrapped.adaptive_confidence_features[1] is not None
    for row, qs in zip(tokens, laws):
        for token, q in zip(row, qs):
            assert q[token] == 1 and np.count_nonzero(q) == 1 and q.sum() == 1
    # Replacement does not leave the external feature cache behind.
    assert all(cache[0].offset == 2 for cache in caches)
    assert draft.stats["backbone_blocks"] == 2
    assert histories == [[1, 2, 3, 1], [4, 5]]


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_real_serving_copy_cold_warm_apcv2_window_and_greedy(kind, monkeypatch):
    model, draft = pair(kind, [2])
    wrapped = ComposedDraftModel(
        draft,
        {"ngram_min": 1, "ngram_max": 1, "min_context_match": 0},
    )
    engine, _ = serving_engine(monkeypatch, kind, model, wrapped)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    try:
        cold = collect(
            engine.submit(
                {"tokens": prompt, "max_tokens": 9, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in cold
        assert cold["tokens"] == reference(model, prompt, 9)
        assert wrapped.composition_stats["prompt_lookup"] > 0
        warm_prompt = prompt + cold["tokens"]
        warm = collect(
            engine.submit(
                {"tokens": warm_prompt, "max_tokens": 5, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in warm
        assert warm["receipt"]["cached_tokens"] == len(warm_prompt) - 1
        assert warm["tokens"] == reference(model, warm_prompt, 5)
    finally:
        engine.close()


def _native_qsa_pair(kind):
    from mlx2.adapters.lilicorr import LiLiCorrConfig
    from mlx2.adapters.xpress import XPressConfig
    from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
    from mlx2.runtime.drafters.xpress import XPressDraftModel

    mx.random.seed(211)
    model = _tiny_qwen4_model()
    args = dict(  # noqa: C408
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        mask_token_id=63,
        num_target_layers=2,
        target_layer_ids=[0, 1],
        block_size=4,
        layer_types=["full_attention"],
    )
    if kind == "xpress":
        draft = XPressDraftModel(
            XPressConfig(
                **args, xpress_rank=4, xpress_mlp_hidden=8, xpress_num_passes=3
            )
        )
    else:
        draft = LiLiCorrDraftModel(
            LiLiCorrConfig(
                **args,
                lilicorr_hidden_size=32,
                lilicorr_candidate_topk=4,
                lilicorr_num_layers=1,
                lilicorr_num_heads=4,
                lilicorr_factor_dim=4,
            )
        )
    return model, draft.bind(model)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_native_qsa_mtp_private_proposals_and_mixed_external_verification(kind):
    model, draft = _native_qsa_pair(kind)
    wrapped = ComposedDraftModel(
        draft,
        {"prompt_lookup": False, "native_mtp": True},
        native_mtp_source=native_mtp_source(model),
    )
    prompts, budgets = [[1, 2, 3], [2, 1, 4, 3, 5]], [9, 7]
    batch = generator(model, wrapped)
    ids = batch.insert(prompts, max_tokens=budgets)
    outputs, finishes = drain(batch)
    for uid, prompt, budget in zip(ids, prompts, budgets):
        assert outputs[uid] == reference(model, prompt, budget)
        finishes[uid].cache_sidecar.validate(
            "tiny-xpress-v1", len(finishes[uid].all_tokens)
        )
        assert finishes[uid].mtp_state is None
    assert wrapped.composition_stats["native_mtp"] > 0
    assert draft.stats["backbone_blocks"] > 0
    assert batch.scheduler_stats["target_max_width"] > 1


def test_native_source_matches_existing_teacher_forced_protocol_and_real_anchor():
    from mlx2.runtime.hybrid_speculative import prepare_self_mtp_lane

    model = _tiny_qwen4_model()
    history, anchor, count = [1, 2, 3, 4], 17, 3
    detached, _ = prepare_self_mtp_lane(
        mx.array(history),
        model,
        uid=1,
        max_tokens=6,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=None,
        num_draft=count,
        sampling_temp=0,
        sampling_top_p=1,
        sampling_top_k=0,
        sampling_min_p=0,
        accept_rule="exact",
        logits_processors=[],
        prefill_step_size=16,
        share_index_state=False,
    )
    h, out = detached.lane.seed_h, []
    for _ in range(count):
        logits, h = model.mtp_step_full_vocab(
            h, mx.array([[anchor]]), detached.caches.draft
        )
        anchor = int(mx.argmax(logits[0, -1]).item())
        out.append(anchor)
    assert native_mtp_source(model)(history, 17, count) == out
    # Independent fresh calls cannot inherit rejected side-head state.
    assert native_mtp_source(model)(history, 17, count) == out


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_native_gdn_mtp_protocol_composes_with_external_state(kind):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    _, template = _native_qsa_pair(kind)
    text = {
        "model_type": "qwen3_5",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
        "partial_rotary_factor": 0.5,
        "rope_parameters": None,
        "max_position_embeddings": 128,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    draft = type(template)(
        replace(template.config, num_target_layers=4, target_layer_ids=[0, 3])
    ).bind(model)
    wrapper = ComposedDraftModel(
        draft,
        {"prompt_lookup": False, "native_mtp": True},
        native_mtp_source=native_mtp_source(model),
    )
    batch = generator(model, wrapper)
    prompts = [[1, 2, 3], [2, 3, 4, 5, 1]]
    ids = batch.insert(prompts, max_tokens=[7, 9])
    outputs, _ = drain(batch)
    for uid, prompt, budget in zip(ids, prompts, [7, 9]):
        assert outputs[uid] == reference(model, prompt, budget)
    assert wrapper.composition_stats["native_mtp"] > 0


def test_native_source_failure_restores_external_transaction_and_retry(monkeypatch):
    from test_batched_mtp import _tree_arrays

    model, draft = _native_qsa_pair("xpress")
    source = native_mtp_source(model)
    wrapper = ComposedDraftModel(
        draft, {"prompt_lookup": False, "native_mtp": True}, native_mtp_source=source
    )
    batch = generator(model, wrapper)
    batch.insert([[1, 2, 3]], max_tokens=[8])
    lane = batch.lanes[0]
    while lane.remaining:
        batch._prefill(lane)
    before = copy.deepcopy(lane.__dict__)

    def fail_after_private_branch(history, anchor, count):
        source(history, anchor, count)
        raise RuntimeError("private native MTP branch failed")

    monkeypatch.setattr(wrapper, "native_mtp_source", fail_after_private_branch)
    with pytest.raises(RuntimeError, match="private native MTP"):
        batch._round([lane])
    assert not lane.ready and lane.history == before["history"]
    for field in ("cache", "draft_cache"):
        for now, old in zip(getattr(lane, field), before[field]):
            if hasattr(now, "offset"):
                assert now.offset == old.offset
            for actual, expected in zip(
                _tree_arrays(now.state), _tree_arrays(old.state)
            ):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(lane.tail), np.asarray(before["tail"]))
    monkeypatch.setattr(wrapper, "native_mtp_source", source)
    output, _ = drain(batch)
    assert output[0] == reference(model, [1, 2, 3], 8)


@pytest.mark.parametrize(
    "value",
    [
        {},
        None,
        {"native_mtp": 1},
        {"lookback": 0},
        {"ngram_min": 4, "ngram_max": 3},
        {"min_context_match": -1},
        {"min_context_match": 65},
        {"max_sources": -1},
        {"prompt_lookup": False},
        {"mystery": True},
    ],
)
def test_composition_policy_fails_closed(value):
    with pytest.raises(ValueError):
        ProposalCompositionPolicy.from_value(value)


def test_unsupported_sources_fail_closed_without_proposal():
    model, draft = tiny()
    with pytest.raises(ValueError, match="native MTP"):
        native_mtp_source(model)
    with pytest.raises(ValueError, match="adapter-owned"):
        ComposedDraftModel(draft, {"native_mtp": True})
    with pytest.raises(ValueError, match="exact proposal-law"):
        ComposedDraftModel(
            SimpleNamespace(proposal_distribution="stochastic"), {"prompt_lookup": True}
        )


def test_dflash_exact_laws_compose_with_pld_without_draft_cache_desync():
    from test_external_dflash2_cpu import tiny as tiny_dflash

    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = tiny_dflash()
    wrapped = ComposedDraftModel(
        draft,
        {"ngram_min": 1, "ngram_max": 1, "min_context_match": 0},
    )
    histories, anchors = [[1, 2, 3, 1], [4, 5]], [2, 6]
    hidden = model.prefill_body(
        mx.array([[1, 2], [4, 5]]), model.make_cache(), [0, 3]
    )
    caches = [wrapped.make_cache(), wrapped.make_cache()]
    tokens, laws = wrapped.draft_distributions(
        anchors,
        hidden,
        wrapped.batch_caches(caches),
        2,
        [RequestRNG(7), RequestRNG(11)],
        [0.8, 0.8],
        processor_histories=histories,
    )

    assert wrapped.proposal_distribution == "stochastic_exact_law"
    assert wrapped.last_proposal_sources == ("prompt_lookup", "external")
    assert tokens[0] == [3, 1]
    for token, law in zip(tokens[0], laws[0]):
        assert law[token] == 1 and np.count_nonzero(law) == 1
    for token, law in zip(tokens[1], laws[1]):
        assert law[token] > 0 and np.count_nonzero(law) > 1
        assert law.sum() == pytest.approx(1)
    # DFlash commits only the supplied target-feature context.  PLD token
    # substitution cannot leave its cache at a different token boundary.
    assert all(cache[0].offset == 2 for cache in caches)


def test_real_dflash_pld_serving_matches_greedy_reference():
    from test_external_dflash2_cpu import (
        drain as drain_dflash,
    )
    from test_external_dflash2_cpu import (
        generator as generator_dflash,
    )
    from test_external_dflash2_cpu import (
        tiny as tiny_dflash,
    )

    model, draft = tiny_dflash()
    wrapped = ComposedDraftModel(draft, {"ngram_min": 1, "ngram_max": 1})
    prompts = [[1, 2, 3, 1, 2, 3, 1, 2], [4, 5, 4, 5, 4]]
    batch = generator_dflash(model, wrapped)
    ids = batch.insert(
        prompts,
        max_tokens=[8, 8],
        sampling_configs=[{"sampling_temp": 0.0}] * 2,
    )
    outputs, finishes = drain_dflash(batch)
    for uid, prompt in zip(ids, prompts):
        cache = model.make_cache()
        expected, context = [], list(prompt)
        for step in range(8):
            logits = model(
                mx.array([context if step == 0 else [context[-1]]]), cache=cache
            )
            token = int(mx.argmax(logits[0, -1]).item())
            expected.append(token)
            context.append(token)
        assert outputs[uid] == expected
        finishes[uid].cache_sidecar.validate("test", len(finishes[uid].all_tokens))
    assert wrapped.composition_stats["prompt_lookup"] > 0
    assert batch.scheduler_stats["target_max_width"] > 1


def test_shared_adapter_binding_selects_reachable_wrapper_and_revision_policy():
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = tiny()
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    fingerprints = []
    for lookback in (64, 128):
        adapter = ExternalDraftAdapterMixin()
        adapter.model = model
        adapter.identity = {"fingerprint": "target-revision"}
        adapter.layout = "target-layout"
        adapter.external_policy = {
            "draft_model": "tiny",
            "num_draft": 2,
            "proposal_composition": {
                "lookback": lookback,
                "ngram_min": 1,
                "ngram_max": 1,
            },
        }
        adapter._bind_external_drafter(
            {"fingerprint": "head-revision"}, lambda record, target: draft, descriptor
        )
        assert isinstance(adapter.draft_model, ComposedDraftModel)
        batch = adapter.create_external_batch()
        ids = batch.insert([[1, 2, 1, 2]], max_tokens=[4])
        outputs, _ = drain(batch)
        assert outputs[ids[0]] == reference(model, [1, 2, 1, 2], 4)
        fingerprints.append(adapter.identity["fingerprint"])
    assert fingerprints[0] != fingerprints[1]


@pytest.mark.parametrize(
    ("min_context_match", "window"),
    [(0, 8 + 2), (5, 8 + 5)],
)
def test_copy_index_covers_only_the_lookback_window(
    monkeypatch, min_context_match, window
):
    """SPEC-04: the per-round index is bounded by lookback, not history.

    The window is ``lookback + max(ngram_max, min_context_match)``: the
    context check reads up to ``min_context_match`` tokens before a source.
    """
    import mlx2.runtime.proposal_composition as composition

    model, draft = pair("xpress", [2])
    wrapped = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 2,
            "lookback": 8,
            "min_context_match": min_context_match,
        },
    )
    built = []
    real = composition.IndexedPromptLookup

    def spy(tokens, **kwargs):
        built.append(list(tokens))
        return real(tokens, **kwargs)

    monkeypatch.setattr(composition, "IndexedPromptLookup", spy)
    history = [1, 2, 3] * 40
    hidden = model.prefill_body(mx.array([[1, 2], [4, 5]]), model.make_cache(), [0, 2])
    caches = [wrapped.make_cache(), wrapped.make_cache()]
    tokens, _laws = wrapped.draft_distributions(
        [1, 6], hidden, wrapped.batch_caches(caches), 2, [None, None], [0, 0],
        processor_histories=[history, [4, 5]],
    )
    assert built == [(history + [1])[-window:], [4, 5, 6]]
    assert tokens[0] == [2, 3]


def test_windowed_lookup_proposes_exactly_what_the_full_index_proposes():
    from mlx2.runtime.prompt_lookup import IndexedPromptLookup

    rng = np.random.default_rng(1)
    for _ in range(1500):
        n = int(rng.integers(5, 300))
        vocab = int(rng.integers(2, 6))
        history = rng.integers(0, vocab, n).tolist()
        lookback = int(rng.integers(1, 120))
        span = int(rng.integers(1, 9))
        ngram_max = int(rng.integers(1, 7))
        ngram_min = int(rng.integers(1, ngram_max + 1))
        # Main's composition defaults add a context check and a source cap;
        # the window must stay exact for both.
        min_context_match = int(rng.choice([0, 0, *range(1, 12)]))
        max_sources = int(rng.choice([0, 0, *range(1, 10)]))
        window = lookback + max(ngram_max, min_context_match)
        full = IndexedPromptLookup(history, ngram_min=ngram_min, ngram_max=ngram_max)
        windowed = IndexedPromptLookup(
            history[-window:], ngram_min=ngram_min, ngram_max=ngram_max
        )
        options = dict(
            lookback=lookback,
            min_context_match=min_context_match,
            max_sources=max_sources,
        )
        assert full.propose(span, **options) == windowed.propose(span, **options)
        # The score key reads the matched n-gram size, which must agree too.
        assert (full.last_source is None) == (windowed.last_source is None)
        if full.last_source is not None:
            assert full.last_source[1] == windowed.last_source[1]


def test_shared_adapter_binding_defaults_local_pld_and_allows_explicit_opt_out():
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = tiny()
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter = ExternalDraftAdapterMixin()
    adapter.model = model
    adapter.identity = {"fingerprint": "target-revision"}
    adapter.layout = "target-layout"
    adapter.external_policy = {"draft_model": "tiny", "num_draft": 2}
    adapter._bind_external_drafter(
        {"fingerprint": "head-revision"}, lambda record, target: draft, descriptor
    )
    assert isinstance(adapter.draft_model, ComposedDraftModel)
    assert adapter.draft_model.policy.as_dict() == {
        "prompt_lookup": True,
        "ngram_min": 3,
        "ngram_max": 6,
        "lookback": 256,
        "min_context_match": 4,
        "max_sources": 8,
        "native_mtp": False,
        "mtp_max_history": 4096,
        "score_decay_per_token": 0.9,
        "trusted_pld": False,
        "trusted_pld_min_score": 0.9,
        "trusted_pld_min_verified_tokens": 32,
        "trusted_pld_min_match": 4,
        "trusted_pld_max_span": 8,
        "trusted_pld_audit_interval": 8,
    }

    disabled = ExternalDraftAdapterMixin()
    disabled.model = model
    disabled.identity = {"fingerprint": "target-revision"}
    disabled.layout = "target-layout"
    disabled.external_policy = {
        "draft_model": "tiny",
        "num_draft": 2,
        "proposal_composition": False,
    }
    disabled._bind_external_drafter(
        {"fingerprint": "head-revision"}, lambda record, target: draft, descriptor
    )
    assert disabled.draft_model is draft


def test_progressive_target_protocol_refuses_before_external_drafter_load():
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = tiny()
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter = ExternalDraftAdapterMixin()
    adapter.model = model
    adapter.external_policy = {"progressive_verification_tile": 1}
    loads = []

    def loader(*args):
        loads.append(args)
        return draft

    with pytest.raises(ValueError, match="row-exact B1 state-promotion"):
        adapter._bind_external_drafter({"fingerprint": "head"}, loader, descriptor)
    assert loads == []


def _enable_tiny_trusted_pld(monkeypatch, model):
    original = model.forward_with_taps

    def forward(*args, last_logits_only=False, **kwargs):
        logits, features = original(*args, **kwargs)
        if last_logits_only:
            logits = logits[:, -1:]
        return logits, features

    monkeypatch.setattr(model, "supports_trusted_pld", True, raising=False)
    monkeypatch.setattr(model, "forward_with_taps", forward)


def test_trusted_pld_unknown_history_always_uses_exact_audit(monkeypatch):
    model, draft = tiny()
    _enable_tiny_trusted_pld(monkeypatch, model)
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
        },
    )
    engine = generator(model, wrapper)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    uid = engine.insert([prompt], max_tokens=[8])[0]
    outputs, finishes = drain(engine)
    assert len(outputs[uid]) == 8 and finishes[uid].finish_reason == "length"
    assert engine.scheduler_stats.get("external_composed_trusted_pld_rounds", 0) == 0
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert wrapper._trusted_stats["audit_rounds"] > 0
    assert outputs[uid] == reference(model, prompt, 8)
    receipt = finishes[uid].speculative_receipt["proposal_composition"]
    assert receipt["ranking"] == receipt["preselection_ranking"]
    assert receipt["postcommit_learning"] is not None
    trust = receipt["trusted_pld"]
    assert trust["scope"] == "request_lane"
    post_trust = receipt["postcommit_learning"]["trusted_pld"]
    assert not any(
        record["blind_rounds"] > 0
        for record in post_trust["sources"].values()
    )
    assert receipt["adapter_diagnostics"]["scope"] == "process_global"


def test_trusted_pld_audit_uses_exact_verifier_and_updates_score():
    model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
            "trusted_pld_audit_interval": 1,
        },
    )
    engine = generator(model, wrapper)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    uid = engine.insert([prompt], max_tokens=[8])[0]
    outputs, finishes = drain(engine)
    assert outputs[uid] == reference(model, prompt, 8)
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert wrapper._trusted_stats["audit_rounds"] > 0
    composition = finishes[uid].speculative_receipt["proposal_composition"]
    assert any(
        key.startswith("prompt_lookup")
        for key in composition["ranking"]["score_counts"]
    )
    postcommit = composition["postcommit_learning"]
    assert "periodic_audit" in postcommit
    assert not any(
        record["periodic_audit_rounds"] > 0
        for record in postcommit["trusted_pld"]["sources"].values()
    )


def test_trusted_pld_span_with_stop_token_is_forced_through_exact_audit(monkeypatch):
    model, draft = tiny()
    _enable_tiny_trusted_pld(monkeypatch, model)
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
        },
    )
    engine = generator(model, wrapper, stop_tokens=[(3,)])
    uid = engine.insert([[1, 2, 3, 1, 2, 3, 1, 2]], max_tokens=[6])[0]
    outputs, finishes = drain(engine)
    assert outputs[uid] and uid in finishes
    assert wrapper._trusted_stats["audit_rounds"] > 0
    assert any(key.startswith("prompt_lookup") for key in finishes[uid].speculative_receipt["proposal_composition"]["ranking"]["score_counts"])
    postcommit = finishes[uid].speculative_receipt["proposal_composition"][
        "postcommit_learning"
    ]
    assert postcommit["audited"]
    assert postcommit["observed_proposed_tokens"] == postcommit[
        "observed_accepted_tokens"
    ]
    record = postcommit["trusted_pld"]["sources"][postcommit["trust_key"]]
    assert not record["revoked"]
    assert record["exact_proposed_tokens"] == record["exact_accepted_tokens"]


def test_round_does_not_mask_run_round_type_error():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    engine = object.__new__(ExternalDraftBatchGenerator)
    engine.continuation_policy = None

    def fail(_cohort, *, forced_depth=None):
        raise TypeError("failure from inside wrapped round")

    engine._run_round = fail
    with pytest.raises(TypeError, match="failure from inside wrapped round"):
        engine._round([])


def test_trusted_pld_in_split_b2_cohort_completes_both_lanes():
    """Trusted PLD is B1-only, judged by the physical cohort width.

    ``_propose`` splits a cohort into draft groups by pending-tail length, so
    a B2 cohort whose lanes committed different counts last round offers the
    composition two singleton groups.  Eligibility judged from one group's
    ``len(anchors)`` marked such a row trusted, and ``_run_round`` raised a
    ValueError out of ``next()``, killing the serving worker.
    """
    from test_external_dflash2_cpu import tiny as tiny_dflash

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    model, draft = tiny_dflash(sliding_window=64)
    assert model.supports_trusted_pld
    wrapper = ComposedDraftModel(
        draft,
        {
            "prompt_lookup": True,
            "ngram_min": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.0,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
        },
    )
    engine = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="trusted-split-b2",
        num_draft=3,
        completion_batch_size=2,
        prefill_step_size=64,
    )
    split_b2 = []
    blind_in_b2 = []
    run_round = engine._run_round

    def observe(cohort, *, forced_depth=None):
        tails = {int(lane.tail.shape[1]) for lane in cohort}
        before = engine.scheduler_stats.get(
            "external_composed_trusted_pld_rounds", 0
        )
        result = run_round(cohort, forced_depth=forced_depth)
        if len(cohort) == 2:
            if len(tails) == 2:
                split_b2.append(sorted(tails))
            blind_in_b2.append(
                engine.scheduler_stats.get("external_composed_trusted_pld_rounds", 0)
                - before
            )
        return result

    engine._run_round = observe
    uids = engine.insert(
        [[1, 2, 3, 4, 5, 6] * 4, [9, 10, 11, 12, 13, 9, 10, 11]],
        max_tokens=[40, 40],
        lane_rngs=[LaneRNG(1), LaneRNG(2)],
    )
    outputs, finishes = {}, {}
    try:
        for _ in range(200):
            _, responses = engine.next()
            for response in responses:
                outputs.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    finishes[response.uid] = response
            if not engine.lanes:
                break
    finally:
        engine.close()
    # The trigger really happened: a B2 cohort split into two draft groups.
    assert split_b2
    assert set(finishes) == set(uids)
    assert all(len(outputs[uid]) == 40 for uid in uids)
    assert all(finishes[uid].finish_reason == "length" for uid in uids)
    # No B2 cohort ever committed a blind (unverified) trusted span.
    assert not any(blind_in_b2)


def test_trusted_pld_on_unsupported_target_is_verified_exactly():
    """A trusted row on a target without trusted-PLD support runs an exact
    audit instead of raising out of ``next()``."""
    model, draft = tiny()
    assert not getattr(model, "supports_trusted_pld", False)
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
        },
    )
    engine = generator(model, wrapper)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    uid = engine.insert([prompt], max_tokens=[8])[0]
    outputs, finishes = drain(engine)
    assert outputs[uid] == reference(model, prompt, 8)
    assert finishes[uid].finish_reason == "length"
    assert engine.scheduler_stats.get("external_composed_trusted_pld_b1_guard_rounds", 0) == 0
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert wrapper._trusted_stats["audit_rounds"] > 0


def test_trusted_pld_per_request_adaptive_round_is_booked_as_exact(monkeypatch):
    """L3-5 (sweep 2026-10-12): the per-request adaptive round verifies every
    row exactly, so a "trusted" copy that reaches it is an audit: it must not
    be booked or receipted as a blind trusted round, and its exact labels
    must feed the lane trust state (where a rejection revokes trust)."""
    from mlx2.runtime import external_speculative as es

    model, draft = tiny()
    original = model.forward_with_taps

    def forward(*args, last_logits_only=False, **kwargs):
        logits, features = original(*args, **kwargs)
        return (logits[:, -1:] if last_logits_only else logits), features

    monkeypatch.setattr(model, "supports_trusted_pld", True, raising=False)
    monkeypatch.setattr(model, "forward_with_taps", forward)
    wrapper = ComposedDraftModel(
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "min_context_match": 0,
            "trusted_pld": True,
            "trusted_pld_min_score": 0.5,
            "trusted_pld_min_verified_tokens": 0,
            "trusted_pld_min_match": 1,
            "trusted_pld_max_span": 2,
        },
    )
    trusted_calls = []
    original_trusted = es.ExternalDraftBatchGenerator._verify_trusted_pld

    def trusted(self, *args):
        trusted_calls.append(1)
        return original_trusted(self, *args)

    monkeypatch.setattr(
        es.ExternalDraftBatchGenerator, "_verify_trusted_pld", trusted
    )
    engine = generator(
        model,
        wrapper,
        adaptive_verification={
            "verification_costs": [0.5, 1, 1.4],
            "mode": "per_request",
            "min_observations": 1,
        },
    )
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    uid = engine.insert([prompt], max_tokens=[8])[0]
    outputs, finishes = drain(engine)
    receipt = finishes[uid].speculative_receipt
    sources = receipt["proposal_composition"]["trusted_pld"]["sources"]
    # Every row went through the exact verifier ...
    assert outputs[uid] == reference(model, prompt, 8)
    assert trusted_calls == []
    assert engine.scheduler_stats["external_adaptive_per_request_rounds"] > 0
    # ... so nothing is booked or receipted as blind trusted execution.
    assert engine.scheduler_stats.get("external_composed_trusted_pld_rounds", 0) == 0
    assert receipt["verification"] == "exact"
    assert receipt.get("approximate") is not True
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert all(record["blind_rounds"] == 0 for record in sources.values())
    # The exact labels reached the lane trust state as audits; a rejected
    # audit revokes trust for that request and source bucket.
    assert wrapper._trusted_stats["audit_rounds"] > 0
    assert engine.scheduler_stats["external_composed_trusted_pld_audit_rounds"] > 0
    failed = [record for record in sources.values() if record["audit_failures"]]
    assert failed
    assert all(record["revoked"] for record in failed)


def test_default_composition_skips_tree_batch_routes():
    """The pinned Qwen3.8 DFlash2 pair defaults to a tree batch-size route;
    composition verifies chains only, so the default must not attach (the
    combination refused at startup: "proposal composition requires chain
    verification")."""
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = tiny()
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter = ExternalDraftAdapterMixin()
    adapter.model = model
    adapter.identity = {"fingerprint": "target-revision"}
    adapter.layout = "target-layout"
    adapter.external_policy = {
        "draft_model": "tiny",
        "num_draft": 2,
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
    }
    adapter._bind_external_drafter(
        {"fingerprint": "head-revision"}, lambda record, target: draft, descriptor
    )
    assert adapter.draft_model is draft
    assert "proposal_composition" not in adapter.external_policy
