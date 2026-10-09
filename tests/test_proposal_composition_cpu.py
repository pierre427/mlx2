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

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
from mlx2.adapters.proposal_sources import native_mtp_source
from mlx2.runtime.proposal_composition import (
    ComposedDraftModel,
    ProposalCompositionPolicy,
)


def test_composition_ranks_committed_score_first_and_span_second():
    _model, draft = tiny()
    wrapper = ComposedDraftModel(
        draft, {"ngram_min": 1, "ngram_max": 1}
    )
    short_pld = ("prompt_lookup", [7, 8], "prompt_lookup:match_4")
    full_external = ("external", None, "external")

    # Cold scores tie, so the longer usable chain wins.
    assert wrapper._select_candidate([short_pld, full_external], 4)[0] == "external"

    # Only committed labels move scores. A stronger PLD score beats length.
    wrapper.observe_proposal_feedback("prompt_lookup:match_4", 8, 8)
    wrapper.observe_proposal_feedback("external", 8, 0)
    selected = wrapper._select_candidate([short_pld, full_external], 4)
    assert selected[0] == "prompt_lookup"
    assert wrapper._proposal_score(selected[2]) > wrapper._proposal_score("external")


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

    def mutate_lagged_score_after_selection(*args, **kwargs):
        wrapper.observe_proposal_feedback("external", 100, 100)
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
        },
    )
    wrapper.observe_proposal_feedback("external", 100, 100)
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
        )

    propose()
    assert wrapper.last_proposal_sources == ("prompt_lookup",)
    assert wrapper.last_proposal_audits == (True,)
    wrapper.observe_proposal_feedback("prompt_lookup:match_1", 32, 32)
    propose()
    assert wrapper.last_proposal_sources == ("external",)
    assert wrapper.last_proposal_audits == (False,)


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
        share_qsa_indices=False,
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
        generator as generator_dflash,
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
        "trusted_pld": False,
        "trusted_pld_min_score": 0.9,
        "trusted_pld_min_verified_tokens": 32,
        "trusted_pld_min_match": 4,
        "trusted_pld_max_span": 8,
        "trusted_pld_audit_interval": 0,
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


def _enable_tiny_trusted_pld(monkeypatch, model):
    original = model.forward_with_taps

    def forward(*args, last_logits_only=False, **kwargs):
        logits, features = original(*args, **kwargs)
        if last_logits_only:
            logits = logits[:, -1:]
        return logits, features

    monkeypatch.setattr(model, "supports_trusted_pld", True, raising=False)
    monkeypatch.setattr(model, "forward_with_taps", forward)


def test_trusted_pld_blindly_commits_copy_and_projects_only_bonus_row(monkeypatch):
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
    assert engine.scheduler_stats["external_composed_trusted_pld_rounds"] > 0
    assert wrapper._trusted_stats["blind_rounds"] > 0
    assert wrapper._trusted_stats["blind_tokens"] >= 2
    assert not any(key.startswith("prompt_lookup") for key in wrapper._score_counts)
    receipt = finishes[uid].speculative_receipt["proposal_composition"]
    assert receipt["trusted_pld"]["blind_rounds"] > 0


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
    outputs, _ = drain(engine)
    assert outputs[uid] == reference(model, prompt, 8)
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert wrapper._trusted_stats["audit_rounds"] > 0
    assert any(key.startswith("prompt_lookup") for key in wrapper._score_counts)


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
    assert engine.scheduler_stats[
        "external_composed_trusted_pld_stop_guard_rounds"
    ] > 0
    assert wrapper._trusted_stats["audit_rounds"] > 0
    assert any(key.startswith("prompt_lookup") for key in wrapper._score_counts)


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

    def observe(cohort):
        tails = {int(lane.tail.shape[1]) for lane in cohort}
        before = engine.scheduler_stats.get(
            "external_composed_trusted_pld_rounds", 0
        )
        result = run_round(cohort)
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
    assert engine.scheduler_stats["external_composed_trusted_pld_b1_guard_rounds"] > 0
    assert wrapper._trusted_stats["blind_rounds"] == 0
    assert wrapper._trusted_stats["audit_rounds"] > 0


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
