"""Complete sequence providers use actual loaded tensor protocols on CPU."""

import copy
import hashlib
import itertools
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_parallel_draft_apcv2_batching_cpu import collect, pair, serving_engine
from test_proposal_composition_cpu import _native_qsa_pair
from test_standard_xpress_serving_cpu import drain, generator, reference

from mlx2.adapters.proposal_path_sources import (
    ExternalContinuationSource,
    NativeMTPContinuationSource,
    SourceContinuation,
    build_continuation_drafter,
)
from mlx2.runtime.proposal_pool import VERIFICATION_CONTRACT
from mlx2.runtime.proposal_providers import (
    ContinuationContext,
    ContinuationDraftModel,
    ContinuationPoolPolicy,
    InsufficientContinuationAlternatives,
    PointMassLaw,
    PromptLookupContinuationSource,
    _continuation_confidence_features,
    continuation_context_revision,
)

PIN = "a" * 64


@pytest.mark.parametrize("minimum_width,expected_launches", [(1, 1), (2, 3)])
def test_progressive_short_trunk_honors_target_packed_minimum_width(
    monkeypatch, minimum_width, expected_launches,
):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, HostDraftRow
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION, ProgressiveTargetRows,
    )

    model, draft = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    model.progressive_external_lane_exact_min_width = minimum_width
    launches = []

    def execute(tokens, caches, layers):
        shape = tuple(tokens.shape)
        assert shape[0] == 1 or shape[1] >= minimum_width
        launches.append(shape)
        logits, features = model.forward_with_taps(tokens, caches, layers)
        return ProgressiveTargetRows(logits, features, PROGRESSIVE_TARGET_PROTOCOL_VERSION,
                                     True, {"algorithm": "cpu-test-executor"})

    monkeypatch.setattr(model, "progressive_external_target_rows", execute, raising=False)
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="short-trunk-capability-regression",
        completion_batch_size=4, num_draft=3, prefill_step_size=64,
        ready_drain="all", progressive_verification_tile=2,
    )
    prompts = [[1, 2, 3], [1, 2, 4], [1, 2, 5]]
    uids = batch.insert(prompts, max_tokens=[8] * 3,
                        sampling_configs=[{"sampling_temp": 0}] * 3)
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
    paths = [reference(model, prompt, 2) for prompt in prompts]
    blocks = [HostDraftRow(path[:1], [PointMassLaw(path[0])], 1) for path in paths]
    result = batch._progressive_verify_many(lanes, blocks, terminal=False)
    assert launches == ([(3, 1)] if minimum_width == 1 else [(1, 1)] * 3)
    assert result[5]["physical_launches"] == expected_launches
    assert result[5]["physical_target_rows"] == 3
    assert [list(decision.emitted) for decision in result[0]] == [path[:1] for path in paths]
    # The narrow shared trunk does not disable packing for legal suffix shapes.
    launches.clear()
    wider = [HostDraftRow(path, [PointMassLaw(t) for t in path], 1) for path in paths]
    wide_result = batch._progressive_verify_many(lanes, wider)
    assert launches == [(3, 3)]
    assert wide_result[5]["physical_launches"] == 1


class _Selection:
    def __init__(self, paths):
        self.paths = tuple(paths)
        self.checked_contracts = []

    def require_verification_contract(self, value):
        self.checked_contracts.append(value)
        assert value == VERIFICATION_CONTRACT


def _ranked_path(tokens, rank, source="external"):
    contributor = SimpleNamespace(
        source_id=source,
        source_revision=(str(rank % 10) or "0") * 64,
        candidate_id=f"{source}:{rank}",
    )
    return SimpleNamespace(
        tokens=tuple(tokens),
        path_id=f"path-{rank}",
        contributors=(contributor,),
        representative=contributor,
    )


def _continuation_bridge(*rows):
    model = ContinuationDraftModel.__new__(ContinuationDraftModel)
    model.last_continuation_selections = tuple(_Selection(row) for row in rows)
    model.last_ordered_alternative_provenance = ()
    return model


def context(model, draft, history=(1, 2), anchor=3, depth=2):
    pending = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    return ContinuationContext(history, anchor, depth, draft.make_cache(), pending)


def test_dflash_tree_source_reports_token_redacted_path_geometry():
    class TreeDraft:
        proposal_distribution = "stochastic_exact_law"
        fold_policy = None
        config = SimpleNamespace(block_size=8)

        @staticmethod
        def propose_tree(_anchors, _pending, _cache, nodes, **_kwargs):
            assert nodes == 15
            return [
                (
                    [10, 11, 12, 13, 14, 15],
                    [-1, -1, 0, 2, 1, 4],
                )
            ]

    source = ExternalContinuationSource(TreeDraft())
    ctx = SimpleNamespace(
        depth=3,
        anchor=7,
        pending_features=None,
        draft_cache=[],
    )
    paths = source(ctx, 15)
    assert [path.tokens for path in paths] == [
        (10, 12, 13),
        (11, 14, 15),
    ]
    assert source.last_diagnostics == {
        "mechanism": "dflash2",
        "node_budget": 15,
        "nodes": 6,
        "roots": 2,
        "leaves": 2,
        "requested_depth": 3,
        "depth_hist": {"1": 2, "2": 2, "3": 2},
        "leaf_depth_hist": {"3": 2},
        "full_depth_nodes": 2,
        "full_depth_leaves": 2,
        "path_selection": "prefix_nondominated",
        "candidate_paths": 6,
        "prefix_dominated_paths_removed": 4,
        "nondominated_paths": 2,
        "offered_paths": 2,
        "offered_depth_hist": {"3": 2},
        "offered_node_ordinals": [3, 5],
    }


def test_dflash_continuation_uses_weight_bound_codebook_cache():
    from test_external_dflash2_cpu import tiny

    model, draft = tiny()
    ctx = context(model, draft)
    source = ExternalContinuationSource(draft)
    first = source(ctx, 15)
    assert source.last_diagnostics["codebook_cache_hit"] is False
    assert source(ctx, 15) == first
    assert source.last_diagnostics["codebook_cache_hit"] is True
    selector = draft.candidate_selector
    selector.predecessor_codebook.weight = selector.predecessor_codebook.weight + 0
    assert source(ctx, 15) == first
    assert source.last_diagnostics["codebook_cache_hit"] is False


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_actual_external_head_beam_returns_complete_unique_paths_private_cache(kind):
    model, draft = pair(kind)
    ctx = context(model, draft)
    source = ExternalContinuationSource(draft)
    before = copy.deepcopy(ctx.draft_cache)
    paths = source(ctx, 15)
    expected_count = (
        15 if kind == "xpress" else draft.config.lilicorr_candidate_topk**ctx.depth
    )
    assert len(paths) == expected_count
    assert len({path.tokens for path in paths}) == len(paths)
    assert all(
        len(path.tokens) == 2 and len(path.confidence_features) == 2 for path in paths
    )
    assert all(np.isfinite(path.ranking_score) for path in paths)
    assert [path.ranking_score for path in paths] == sorted(
        [path.ranking_score for path in paths], reverse=True
    )
    assert all(
        now.offset == old.offset == 0 for now, old in zip(ctx.draft_cache, before)
    )
    again = source(ctx, 15)
    assert paths == again


def test_xpress_beam_matches_bruteforce_refined_block_proxy(monkeypatch):
    model, draft = pair("xpress")
    ctx = context(model, draft, depth=3)
    captured = []
    original = draft.xpress_head.jacobi_refine_greedy

    def capture(*args, **kwargs):
        tokens, logits = original(*args, **kwargs)
        captured.append(np.asarray(logits.astype(mx.float32))[0].astype(np.float64))
        return tokens, logits

    monkeypatch.setattr(draft.xpress_head, "jacobi_refine_greedy", capture)
    paths = ExternalContinuationSource(draft)(ctx, 15)
    logits = captured[0][: ctx.depth]
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    log_probs = np.log(exponent / exponent.sum(axis=-1, keepdims=True))
    all_paths = [
        (sum(log_probs[position, token] for position, token in enumerate(path)), path)
        for path in itertools.product(range(draft.config.vocab_size), repeat=ctx.depth)
    ]
    expected = sorted(all_paths, key=lambda item: (-item[0], item[1]))[:15]
    assert [path.tokens for path in paths] == [path for _, path in expected]
    np.testing.assert_allclose(
        [path.ranking_score for path in paths],
        [score for score, _ in expected],
        atol=1e-10,
    )


@pytest.mark.parametrize("limit", [1, 7, 15, 65536])
def test_partitioned_top_matches_full_vocab_sort_with_cutoff_ties(limit):
    from mlx2.adapters.proposal_path_sources import _top

    rng = np.random.default_rng(58)
    values = rng.normal(size=65536)
    values[[0, 8, 100, 65535]] = 12
    values[[1, 9, 101, 60000, 64000, 65000, 65534]] = 11
    values[[7, 23, 36000]] = -np.inf
    values[[14, 65530]] = np.nan
    finite = np.flatnonzero(np.isfinite(values)).tolist()
    expected = sorted(finite, key=lambda token: (-float(values[token]), token))[:limit]
    assert _top(values, limit) == expected


def test_fixed_xpress_rows_normalize_once_per_position_without_processors(monkeypatch):
    from mlx2.adapters import proposal_path_sources

    model, draft = pair("xpress")
    ctx = context(model, draft, depth=3)
    original = proposal_path_sources._log_probs
    normalized = []

    def capture(values):
        normalized.append(np.array(values, copy=True))
        return original(values)

    monkeypatch.setattr(proposal_path_sources, "_log_probs", capture)
    paths = ExternalContinuationSource(draft)(ctx, 15)
    assert len(paths) == 15 and all(len(path.tokens) == 3 for path in paths)
    assert len(normalized) == ctx.depth


def test_native_mtp_beam_uses_actual_anchor_loaded_head_and_private_branches():
    model, draft = _native_qsa_pair("xpress")
    ctx = context(model, draft, history=(1, 2, 3), anchor=17, depth=2)
    paths = NativeMTPContinuationSource(model)(ctx, 15)
    assert len(paths) == 15 and len({path.tokens for path in paths}) == 15
    assert all(len(path.tokens) == 2 for path in paths)
    assert paths == NativeMTPContinuationSource(model)(ctx, 15)
    assert all(cache.offset == 0 for cache in ctx.draft_cache)
    assert NativeMTPContinuationSource(model, max_history=2)(ctx, 15) == []


def test_pld_enumerates_multiple_committed_matching_continuations():
    policy = ContinuationPoolPolicy.from_value({"ngram_min": 1, "ngram_max": 1})
    ctx = ContinuationContext((1, 2, 3, 1, 4, 5, 1, 6, 7), 1, 2, None, None)
    paths = PromptLookupContinuationSource(policy)(ctx, 15)
    assert {path.tokens for path in paths} == {(2, 3), (4, 5), (6, 7)}
    assert ctx.history == (1, 2, 3, 1, 4, 5, 1, 6, 7)


def test_ranked_continuation_slate_bridges_to_confidence_tree_with_provenance():
    primary = (1, 2, 3, 4)
    paths = [
        _ranked_path(primary, 0, "native_mtp"),
        _ranked_path(primary, 1, "prompt_lookup"),  # full-path duplicate
        _ranked_path((1, 2, 8, 9), 2, "external"),
        _ranked_path((1, 7, 6, 5), 3, "prompt_lookup"),  # wrong trunk
        _ranked_path((1, 2, 6, 7), 4, "prompt_lookup"),
        _ranked_path((1, 2, 5), 6, "external"),  # incompatible depth skipped
    ]
    model = _continuation_bridge(paths)

    alternatives = model.ordered_alternative_proposals(
        [list(primary)],
        branch_count=3,
        strategy="confidence_tree",
        shared_prefix_tokens=(2,),
    )

    assert alternatives == [[[1, 2, 3, 4], [1, 2, 8, 9], [1, 2, 6, 7]]]
    assert model.last_continuation_selections[0].checked_contracts == [
        VERIFICATION_CONTRACT
    ]
    provenance = model.last_ordered_alternative_provenance[0]
    assert [item["rank"] for item in provenance] == [0, 2, 4]
    assert [item["sources"] for item in provenance] == [
        ("native_mtp",),
        ("external",),
        ("prompt_lookup",),
    ]
    assert all(len(item["contributors"][0]) == 3 for item in provenance)


def test_ranked_r8_slate_searches_deeper_exact_fanout():
    primary = (1, 2, 3, 4)
    paths = [
        _ranked_path(primary, 0),
        _ranked_path((1, 2, 8, 9), 1),
        _ranked_path((1, 2, 7, 6), 2),
        _ranked_path((1, 2, 5, 0), 3),
        _ranked_path((1, 2, 4, 8), 4),
        _ranked_path((1, 2, 6, 7), 5),
    ]
    model = _continuation_bridge(paths)

    alternatives = model.ordered_alternative_proposals(
        [list(primary)],
        branch_count=4,
        strategy="confidence_tree",
        shared_prefix_tokens=(1,),
        allow_adaptive_prefix_search=True,
    )

    assert alternatives == [[
        [1, 2, 3, 4],
        [1, 2, 8, 9],
        [1, 2, 7, 6],
        [1, 2, 5, 0],
    ]]
    assert model.last_ordered_alternative_prefix_tokens == (2,)
    assert model.last_ordered_alternative_search_diagnostics == ({
        "attempted_prefixes": (1, 2, 3, 0),
        "feasible_width_by_prefix": ((1, 1), (2, 4), (3, 1), (0, 1)),
        "chosen_prefix": 2,
        "chosen_width": 4,
    },)
    assert [
        item["rank"] for item in model.last_ordered_alternative_provenance[0]
    ] == [0, 1, 2, 3]


@pytest.mark.parametrize("branch_count", [4, 3, 2])
def test_rootless_adaptive_rows_are_contiguous_global_rank_prefix(branch_count):
    primary = (1, 2, 3, 4)
    paths = [
        _ranked_path(primary, 0),
        _ranked_path((8, 8, 7, 6), 1),
        _ranked_path((9, 2, 5, 0), 2),
        _ranked_path((7, 2, 4, 8), 3),
    ][:branch_count]
    model = _continuation_bridge(paths)

    alternatives = model.ordered_alternative_proposals(
        [list(primary)],
        branch_count=branch_count,
        strategy="confidence_tree",
        shared_prefix_tokens=(3,),
        allow_adaptive_prefix_search=True,
    )

    assert alternatives == [[[int(token) for token in path.tokens] for path in paths]]
    assert model.last_ordered_alternative_prefix_tokens == (0,)
    assert [
        item["rank"] for item in model.last_ordered_alternative_provenance[0]
    ] == list(range(branch_count))


def test_rootless_rows_are_distinct_at_the_root_not_overlapping_truncations():
    """F8: native-MTP alternatives are the primary's prefix plus one other
    token; as separate zero-trunk rows they re-verified the shared prefix."""
    primary = (1, 2, 3, 4)
    paths = [
        _ranked_path(primary, 0),
        _ranked_path((1, 2, 9), 1),
        _ranked_path((1, 8), 2),
        _ranked_path((7,), 3),
        _ranked_path((1, 2, 3, 5), 4),
        _ranked_path((6,), 5),
    ]
    model = _continuation_bridge(paths)
    model.last_requested_proposal_length = 4

    alternatives = model.ordered_alternative_proposals(
        [list(primary)],
        branch_count=3,
        strategy="confidence_tree",
        shared_prefix_tokens=(3,),
        allow_adaptive_prefix_search=True,
    )

    # Prefix 3 has only two same-depth children; the rootless fallback keeps
    # one row per root token, in global rank order.
    assert model.last_ordered_alternative_prefix_tokens == (0,)
    assert alternatives == [[[1, 2, 3, 4], [7], [6]]]
    assert [
        item["rank"] for item in model.last_ordered_alternative_provenance[0]
    ] == [0, 3, 5]


def test_rootless_ragged_path_exceeding_requested_depth_fails_closed():
    primary = (1, 2)
    model = _continuation_bridge([
        _ranked_path(primary, 0),
        _ranked_path((8, 7, 6, 5), 1),
    ])
    model.last_requested_proposal_length = 3

    with pytest.raises(
        ValueError,
        match="continuation alternative exceeds the requested depth",
    ):
        model.ordered_alternative_proposals(
            [list(primary)],
            branch_count=2,
            strategy="confidence_tree",
            shared_prefix_tokens=(1,),
            allow_adaptive_prefix_search=True,
        )


def test_ranked_continuation_slate_preserves_paired_family_order():
    primary = (1, 2, 3, 4)
    # Ranking may place the child paths before the second parent.  The bridge
    # still emits B1, B2, B3(B1), B4(B2), as required by staged verification.
    paths = [
        _ranked_path(primary, 0),
        _ranked_path((1, 2, 3, 9), 1),
        _ranked_path((1, 2, 8, 7), 2),
        _ranked_path((1, 2, 8, 4), 3),
    ]
    model = _continuation_bridge(paths)

    alternatives = model.ordered_alternative_proposals(
        [list(primary)],
        branch_count=4,
        strategy="paired_continuations",
        shared_prefix_tokens=2,
    )

    assert alternatives == [
        [[1, 2, 3, 4], [1, 2, 8, 7], [1, 2, 3, 9], [1, 2, 8, 4]]
    ]
    assert [item["rank"] for item in model.last_ordered_alternative_provenance[0]] == [
        0,
        2,
        1,
        3,
    ]


@pytest.mark.parametrize(
    ("paths", "branches", "message"),
    [
        (
            [_ranked_path((9, 2, 3), 0), _ranked_path((1, 8, 3), 1)],
            2,
            "B1 path differs",
        ),
        (
            [_ranked_path((1, 2, 3), 0), _ranked_path((1, 8), 1)],
            2,
            "lacks the requested distinct branches",
        ),
        (
            [_ranked_path((1, 2, 3), 0), _ranked_path((1, 2, 8), 1)],
            2,
            "lacks the requested distinct branches",
        ),
    ],
)
def test_ranked_continuation_bridge_fails_closed_on_invalid_slates(
    paths, branches, message
):
    model = _continuation_bridge(paths)
    with pytest.raises((ValueError, RuntimeError), match=message):
        model.ordered_alternative_proposals(
            [[1, 2, 3]],
            branch_count=branches,
            strategy="distinct_root",
            shared_prefix_tokens=None,
        )


@pytest.mark.parametrize("branch_count", [0, 5])
def test_ranked_continuation_bridge_bounds_physical_branch_count(branch_count):
    model = _continuation_bridge([_ranked_path((1, 2, 3), 0)])
    with pytest.raises(ValueError, match="one through four"):
        model.ordered_alternative_proposals(
            [[1, 2, 3]],
            branch_count=branch_count,
            strategy="distinct_root",
            shared_prefix_tokens=None,
        )


@pytest.mark.parametrize(
    ("probabilities", "depth"),
    [
        ([0.5, 0.5], 2),
        ((0.5,), 2),
        ((float("nan"),), 1),
        ((True,), 1),
        (("0.5",), 1),
        ((-0.1,), 1),
        ((1.1,), 1),
    ],
)
def test_continuation_confidence_fails_closed_without_exact_valid_features(
    probabilities, depth
):
    with pytest.raises(ValueError, match="valid per-token confidence"):
        _continuation_confidence_features(probabilities, depth)


def test_continuation_confidence_log_odds_are_finite_and_bounded():
    assert _continuation_confidence_features((0.0, 0.5, 1.0), 3) == (
        -40.0,
        0.0,
        40.0,
    )


def test_unconfigured_admission_does_not_change_v2_policy_identity():
    policy = ContinuationPoolPolicy.from_value({"ngram_min": 1, "ngram_max": 1})
    assert "admission" not in policy.as_dict()
    configured = ContinuationPoolPolicy.from_value(
        {"admission": {"pld_fallback": False}}
    )
    assert configured.as_dict()["admission"] == {"pld_fallback": False}


def wrapped(model, draft, policy=None):
    return build_continuation_drafter(
        model,
        draft,
        # These fixtures exercise fixed topology, not the cost planner.
        {"ast_planner": False, **(policy or {})},
        target_revision=PIN,
        draft_revision="b" * 64,
        tokenizer_revision=PIN,
        session_revision="c" * 64,
    )


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_bound_wrapper_selects_fifteen_complete_paths_and_advances_context_once(kind):
    model, draft = pair(kind)
    wrapper = wrapped(model, draft, {"ngram_min": 1, "ngram_max": 1})
    histories, anchors = [[1, 2], [1, 2]], [3, 4]
    hidden = model.prefill_body(mx.array(histories), model.make_cache(), [0, 2])
    caches = [draft.make_cache(), draft.make_cache()]
    identities = [
        {
            "request_id": f"request-{row}",
            "round_id": 0,
            "context_revision": continuation_context_revision(
                wrapper.session.session_revision, history, anchor
            ),
        }
        for row, (history, anchor) in enumerate(zip(histories, anchors))
    ]
    tokens, laws = wrapper.draft_distributions(
        anchors,
        hidden,
        draft.batch_caches(caches),
        2,
        [None, None],
        [0, 0],
        processor_histories=histories,
        proposal_contexts=identities,
    )
    assert draft.stats["backbone_blocks"] == 2
    assert all(cache[0].offset == 2 for cache in caches)
    for row, selection in enumerate(wrapper.last_continuation_selections):
        assert 1 <= len(selection.paths) <= 15
        assert all(len(path.tokens) == 2 for path in selection.paths)
        assert tokens[row] == list(selection.paths[0].tokens)
        assert all(
            q == PointMassLaw(token)
            for token, q in zip(tokens[row], laws[row], strict=True)
        )
        wrapper.proposal_pool.discard(selection)
    assert len(wrapper.adaptive_confidence_features) == 2
    assert wrapper.last_continuation_source_diagnostics == ({}, {})
    assert all(
        len(features) == len(tokens[row])
        and np.isfinite(features).all()
        and np.abs(features).max() <= 40
        for row, features in enumerate(wrapper.adaptive_confidence_features)
    )
    for row, provenance in enumerate(
        wrapper.last_continuation_confidence_provenance
    ):
        selected = wrapper.last_continuation_selections[row].paths[0]
        assert provenance["path_id"] == selected.path_id
        assert provenance["score_authority"] == selected.score_authority
        assert provenance["verification_contract"] == VERIFICATION_CONTRACT
        expected = [
            max(
                -40.0,
                min(
                    40.0,
                    np.log(max(probability, np.exp(-40.0)))
                    - np.log(max(1.0 - probability, np.exp(-40.0))),
                ),
            )
            for probability in selected.conditional_acceptance
        ]
        np.testing.assert_allclose(wrapper.adaptive_confidence_features[row], expected)


def test_actual_ranked_pool_drives_confidence_tree_without_changing_primary_law():
    model, draft = pair("xpress")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    history, anchor = [1, 2, 3], 4
    hidden = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    identity = {
        "request_id": "confidence-tree-bridge",
        "round_id": 0,
        "context_revision": continuation_context_revision(
            wrapper.session.session_revision, history, anchor
        ),
    }
    tokens, laws = wrapper.draft_distributions(
        [anchor],
        hidden,
        draft.make_cache(),
        2,
        [None],
        [0],
        processor_histories=[history],
        proposal_contexts=[identity],
    )
    primary = list(tokens[0])
    primary_laws = list(laws[0])

    alternatives = wrapper.ordered_alternative_proposals(
        tokens,
        branch_count=4,
        strategy="confidence_tree",
        shared_prefix_tokens=(1,),
    )[0]

    assert alternatives[0] == primary
    assert len({tuple(path) for path in alternatives}) == 4
    assert all(len(path) == 2 and path[:1] == primary[:1] for path in alternatives)
    assert len(wrapper.adaptive_confidence_features[0]) == len(primary)
    assert len(wrapper.last_ordered_alternative_provenance[0]) == 4
    for token, before, after in zip(primary, primary_laws, laws[0], strict=True):
        assert after == before == PointMassLaw(token)
    wrapper.proposal_pool.discard(wrapper.last_continuation_selections[0])


def test_continuation_backbone_uses_private_retry_stable_rng(monkeypatch):
    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = pair("xpress")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    history, anchor = [1, 2, 3], 4
    hidden = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    identity = {
        "request_id": "private-backbone-rng",
        "round_id": 7,
        "context_revision": continuation_context_revision(
            wrapper.session.session_revision, history, anchor
        ),
    }
    original = draft.draft_distributions
    seen = []

    def consuming(*args, **kwargs):
        seen.append(tuple(rng.uniform() for rng in args[4]))
        return original(*args, **kwargs)

    monkeypatch.setattr(draft, "draft_distributions", consuming)
    authoritative = RequestRNG(91)
    before = authoritative.snapshot()
    outputs = []
    for _ in range(2):
        tokens, laws = wrapper.draft_distributions(
            [anchor],
            hidden,
            draft.make_cache(),
            2,
            [authoritative],
            [0],
            processor_histories=[history],
            proposal_contexts=[identity],
        )
        outputs.append(tokens)
        assert all(isinstance(law, PointMassLaw) for law in laws[0])
        wrapper.proposal_pool.discard(wrapper.last_continuation_selections[0])
    assert authoritative.snapshot() == before
    assert seen[0] == seen[1]
    assert outputs[0] == outputs[1]


def test_empty_complete_path_tree_uses_private_primary_spine(monkeypatch):
    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = pair("xpress")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    monkeypatch.setattr(draft, "proposal_distribution", "stochastic_exact_law")
    monkeypatch.setitem(wrapper.providers, "external", lambda _context, _limit: ())
    history, anchor = [1, 2, 3], 4
    hidden = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    identity = {
        "request_id": "empty-tree-primary-spine",
        "round_id": 0,
        "context_revision": continuation_context_revision(
            wrapper.session.session_revision, history, anchor
        ),
    }
    authoritative = RequestRNG(91)
    before = authoritative.snapshot()
    tokens, laws = wrapper.draft_distributions(
        [anchor],
        hidden,
        draft.make_cache(),
        2,
        [authoritative],
        [0],
        processor_histories=[history],
        proposal_contexts=[identity],
    )
    selection = wrapper.last_continuation_selections[0]
    assert len(selection.paths) == 1
    assert tokens[0] == list(selection.paths[0].tokens)
    assert laws[0] == [PointMassLaw(token) for token in tokens[0]]
    assert authoritative.snapshot() == before
    with pytest.raises(InsufficientContinuationAlternatives):
        wrapper.ordered_alternative_proposals(
            tokens,
            branch_count=2,
            strategy="distinct_root",
            shared_prefix_tokens=(1,),
        )
    wrapper.proposal_pool.discard(selection)


def test_shallow_only_tree_uses_private_full_depth_primary_spine(monkeypatch):
    from mlx2.runtime.speculative_sampling import RequestRNG

    model, draft = pair("xpress")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    monkeypatch.setattr(draft, "proposal_distribution", "stochastic_exact_law")
    monkeypatch.setitem(
        wrapper.providers,
        "external",
        lambda _context, _limit: (
            SourceContinuation((1,), (0.5,), 10.0, "shallow_only"),
        ),
    )
    history, anchor = [1, 2, 3], 4
    hidden = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    identity = {
        "request_id": "shallow-tree-primary-spine",
        "round_id": 0,
        "context_revision": continuation_context_revision(
            wrapper.session.session_revision, history, anchor
        ),
    }
    authoritative = RequestRNG(91)
    before = authoritative.snapshot()
    tokens, laws = wrapper.draft_distributions(
        [anchor],
        hidden,
        draft.make_cache(),
        2,
        [authoritative],
        [0],
        processor_histories=[history],
        proposal_contexts=[identity],
    )
    selection = wrapper.last_continuation_selections[0]
    assert len(selection.paths[0].tokens) == 2
    assert len(selection.paths[1].tokens) == 1
    spine = selection.paths[0].representative
    assert spine.candidate_id == "external:0:private-spine"
    assert spine.ranking_convention == "shallow_only"
    assert spine.ranking_score < selection.paths[1].representative.ranking_score
    assert tokens[0] == list(selection.paths[0].tokens)
    assert laws[0] == [PointMassLaw(token) for token in tokens[0]]
    assert authoritative.snapshot() == before
    wrapper.proposal_pool.discard(selection)


def test_continuation_ast_bridge_settles_one_ticket_and_matches_reference(monkeypatch):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from scripts.paired_direct_ab import state_digest

    model, base = pair("xpress")
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})

    def force_nonprimary_target_winner(context, _limit):
        target = tuple(
            reference(
                model,
                [*context.history, context.anchor],
                context.depth,
            )
        )
        wrong = ((target[0] + 1) % base.config.vocab_size, *target[1:])
        return (
            SourceContinuation(wrong, (0.5,) * context.depth, 2.0, "forced"),
            SourceContinuation(target, (0.5,) * context.depth, 1.0, "forced"),
        )

    monkeypatch.setitem(
        wrapper.providers,
        "external",
        force_nonprimary_target_winner,
    )
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-bridge",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
        progressive_alternative_strategy="distinct_root",
    )
    assert batch._continuation_ast_bridge_geometry(1) == {
        "schema": "mlx2.continuation-ast-bridge-geometry.v1",
        "sessions": 1,
        "branches_per_session": 2,
        "physical_path_rows": 2,
        "tile_rows_per_path": 2,
        "maximum_target_rows_per_launch": 4,
        "private_target_checkpoints": 2,
        "private_provider_cache_rows": 1,
        "live_frontier": False,
    }
    prompt = [1, 2, 3]
    batch.insert([prompt], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 6)
    pool = final[0].speculative_receipt["continuation_pool"]
    assert pool["verification_role"] == "proposal_only_ast_bridge"
    assert pool["observed_used"]
    assert pool["ast_observed_used"]
    assert pool["selected_path"] == 1
    assert pool["selected_physical_path"] == 1
    assert pool["physical_width"] == 2
    assert pool["physical_span"] == 4
    assert not pool["strategy_observed_used"]
    assert pool["ranked_complete_sequences"] == 2
    assert pool["admitted_complete_sequences"] == 2
    assert pool["selected_ast_provenance"]["rank"] == 1
    assert pool["selected_ast_provenance"]["path_id"]
    assert pool["selected_sources"]
    assert pool["provider_stats"]["external"]["selected_rounds"] > 0
    assert pool["last_admission"]
    assert any(
        row["target_selected"] and row["rank"] == 1
        for row in pool["last_candidate_slate"]
    )
    ordinary = ExternalDraftBatchGenerator(
        model,
        draft_model=base,
        binding="continuation-ast-ordinary-control",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
    )
    ordinary_uid = ordinary.insert(
        [prompt],
        max_tokens=[6],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    ordinary.disable_speculation(ordinary_uid)
    ordinary_output, ordinary_final = drain(ordinary)
    assert ordinary_output[ordinary_uid] == output[0]
    for ast_cache, ordinary_cache in zip(
        final[0].prompt_cache,
        ordinary_final[ordinary_uid].prompt_cache,
        strict=True,
    ):
        assert ast_cache.offset == ordinary_cache.offset
        np.testing.assert_allclose(
            np.asarray(ast_cache.keys)[..., : ast_cache.offset, :],
            np.asarray(ordinary_cache.keys)[..., : ordinary_cache.offset, :],
            rtol=0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.asarray(ast_cache.values)[..., : ast_cache.offset, :],
            np.asarray(ordinary_cache.values)[..., : ordinary_cache.offset, :],
            rtol=0,
            atol=2e-6,
        )
    primary_wrapper = wrapped(model, base, {"sources": ["external"]})

    def force_primary_target_winner(context, _limit):
        target = tuple(
            reference(
                model,
                [*context.history, context.anchor],
                context.depth,
            )
        )
        wrong = ((target[0] + 1) % base.config.vocab_size, *target[1:])
        return (
            SourceContinuation(target, (0.5,) * context.depth, 2.0, "forced"),
            SourceContinuation(wrong, (0.5,) * context.depth, 1.0, "forced"),
        )

    monkeypatch.setitem(
        primary_wrapper.providers,
        "external",
        force_primary_target_winner,
    )
    primary = ExternalDraftBatchGenerator(
        model,
        draft_model=primary_wrapper,
        binding="continuation-ast-bridge",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=primary_wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
        progressive_alternative_strategy="distinct_root",
    )
    primary_uid = primary.insert(
        [prompt],
        max_tokens=[6],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    primary_output, primary_final = drain(primary)
    assert primary_output[primary_uid] == output[0]
    final[0].cache_sidecar.validate(
        "continuation-ast-bridge", len(final[0].all_tokens)
    )
    primary_final[primary_uid].cache_sidecar.validate(
        "continuation-ast-bridge",
        len(primary_final[primary_uid].all_tokens),
    )
    assert state_digest(final[0].cache_sidecar) == state_digest(
        primary_final[primary_uid].cache_sidecar
    )
    assert state_digest(final[0].prompt_cache) == state_digest(
        primary_final[primary_uid].prompt_cache
    )
    assert wrapper.proposal_pool.feedback_revision > 0
    assert not wrapper.proposal_pool._pending


def test_continuation_ast_bridge_admission_uses_ast_geometry_not_pool_limit():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    reservations = []
    for limit in (2, 15):
        model, base = pair("xpress")
        model.progressive_external_verify_protocol = (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION
        )
        wrapper = wrapped(
            model,
            base,
            {"sources": ["external"], "limit": limit},
        )
        batch = ExternalDraftBatchGenerator(
            model,
            draft_model=wrapper,
            binding=f"continuation-ast-admission-{limit}",
            completion_batch_size=1,
            num_draft=2,
            prefill_step_size=64,
            ready_drain="all",
            continuation_pool=wrapper.policy,
            continuation_pool_ast_bridge=True,
            progressive_verification_tile=1,
            progressive_alternative_branches=2,
            memory_headroom=lambda: 1 << 60,
        )
        batch.insert(
            [[1, 2, 3]],
            max_tokens=[6],
            sampling_configs=[{"sampling_temp": 0}],
        )
        assert batch._admit(list(batch.lanes.values()), 3)
        assert batch.scheduler_stats["continuation_ast_bridge_geometry"] == (
            batch._continuation_ast_bridge_geometry(1)
        )
        assert batch.scheduler_stats["continuation_ast_bridge_private_bytes"] > 0
        reservations.append(batch.scheduler_stats["reservation_bytes"])

    # The provider may rank 2 or 15 candidate paths, but only the declared AST
    # frontier/checkpoints are resident.  Pool breadth must not multiply the
    # target reservation.
    assert reservations[0] == reservations[1]


def test_continuation_ast_b4_confidence_tail_reports_executed_physical_span(
    monkeypatch,
):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})

    def four_paths(context, _limit):
        target = tuple(
            reference(
                model,
                [*context.history, context.anchor],
                context.depth,
            )
        )

        def changed(delta):
            tokens = list(target)
            tokens[1] = (tokens[1] + delta) % base.config.vocab_size
            return tuple(tokens)

        confidence = (0.9, 0.8, 0.1)
        return (
            SourceContinuation(changed(1), confidence, 4.0, "forced"),
            SourceContinuation(target, confidence, 3.0, "forced"),
            SourceContinuation(changed(2), confidence, 2.0, "forced"),
            SourceContinuation(changed(3), confidence, 1.0, "forced"),
        )

    monkeypatch.setitem(wrapper.providers, "external", four_paths)
    draft_distributions = wrapper.draft_distributions

    def confidence_tail(*args, **kwargs):
        result = draft_distributions(*args, **kwargs)
        wrapper.adaptive_confidence_features = (
            (1.0, 0.5, -2.0),
        )
        return result

    monkeypatch.setattr(wrapper, "draft_distributions", confidence_tail)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-b4-confidence-tail",
        completion_batch_size=1,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
    )
    prompt = [1, 2, 3]
    uid = batch.insert(
        [prompt],
        max_tokens=[4],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    output, final = drain(batch)
    assert output[uid] == reference(model, prompt, 4)
    receipt = final[uid].speculative_receipt
    alternatives = receipt["progressive_verification"]["ordered_alternatives"]
    assert alternatives["last_branch_proposal_counts"] == [3, 3, 2, 2]
    pool = receipt["continuation_pool"]
    assert pool["ast_observed_used"]
    assert pool["ranked_complete_sequences"] == 4
    assert pool["admitted_complete_sequences"] == 4
    assert pool["physical_width"] == 4
    assert pool["physical_span"] == 10
    assert pool["selected_path"] == 1
    assert pool["selected_physical_path"] == 1
    assert pool["selected_ast_provenance"]["rank"] == 1
    assert pool["selected_ast_provenance"]["path_id"]
    assert pool["selected_ast_provenance"]["representative_source"] == "external"
    assert pool["selected_sources"] == [
        ("external", wrapper.source_records["external"].revision)
    ]
    assert not wrapper.proposal_pool._pending
    ordinary = ExternalDraftBatchGenerator(
        model,
        draft_model=base,
        binding="continuation-ast-b4-confidence-tail-ordinary",
        completion_batch_size=1,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
    )
    ordinary_uid = ordinary.insert(
        [prompt],
        max_tokens=[4],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    ordinary.disable_speculation(ordinary_uid)
    ordinary_output, ordinary_final = drain(ordinary)
    assert ordinary_output[ordinary_uid] == output[uid]
    for ast_cache, ordinary_cache in zip(
        final[uid].prompt_cache,
        ordinary_final[ordinary_uid].prompt_cache,
        strict=True,
    ):
        assert ast_cache.offset == ordinary_cache.offset
        np.testing.assert_allclose(
            np.asarray(ast_cache.keys)[..., : ast_cache.offset, :],
            np.asarray(ordinary_cache.keys)[..., : ordinary_cache.offset, :],
            rtol=0,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            np.asarray(ast_cache.values)[..., : ast_cache.offset, :],
            np.asarray(ordinary_cache.values)[..., : ordinary_cache.offset, :],
            rtol=0,
            atol=2e-6,
        )


def test_continuation_ast_confidence_tree_downshifts_and_cross_session_packs(
    monkeypatch,
):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    from mlx2.serving import minimum_tokens_processor
    from scripts.paired_direct_ab import state_digest

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})
    minimum = minimum_tokens_processor(mx, [0], 3, 4)

    def minimum_reference(prompt, length):
        cache, tokens = model.make_cache(), list(prompt)
        output = []
        for step in range(length):
            logits = model(
                mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache
            )
            processed = minimum(mx.array(tokens), logits[:, -1])
            token = int(mx.argmax(processed[0]).item())
            output.append(token)
            tokens.append(token)
        return output

    def forced_prediction(path, _scope):
        probabilities = tuple(0.5 for _ in path.tokens)
        return (
            probabilities,
            float(path.ranking_score),
            None,
            float(path.ranking_score),
            "forced_test_ranking",
        )

    monkeypatch.setattr(wrapper.proposal_pool, "_prediction", forced_prediction)

    def two_compatible_paths(context, _limit):
        target = tuple(
            minimum_reference(
                [*context.history, context.anchor], context.depth
            )
        )
        primary = list(target)
        primary[0] = (primary[0] + 3) % base.config.vocab_size
        target_prefix = target[:2]
        unrelated_a = list(target[:1])
        unrelated_a[0] = (unrelated_a[0] + 1) % base.config.vocab_size
        unrelated_b = list(target[:2])
        unrelated_b[0] = (unrelated_b[0] + 2) % base.config.vocab_size
        def candidate(tokens, confidence, score):
            return SourceContinuation(
                tuple(tokens),
                tuple(confidence),
                score,
                "forced",
            )

        return (
            candidate(target_prefix, (0.999, 0.999), 4.0),
            candidate(primary, (0.8, 0.8, 0.8), 3.0),
            candidate(unrelated_a, (0.5,), 2.0),
            candidate(unrelated_b, (0.2, 0.2), 1.0),
        )

    monkeypatch.setitem(wrapper.providers, "external", two_compatible_paths)
    draft_distributions = wrapper.draft_distributions

    def confidence_tail(*args, **kwargs):
        result = draft_distributions(*args, **kwargs)
        wrapper.adaptive_confidence_features = tuple(
            (1.0, 0.5, -2.0) for _ in result[0]
        )
        return result

    monkeypatch.setattr(wrapper, "draft_distributions", confidence_tail)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-cross-session-downshift",
        completion_batch_size=4,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        progressive_cross_session_pack_rows=8,
        progressive_cross_session_pack_mode="ordered_live_frontier",
    )
    prompts = [[1, 2, 3]] * 4
    assert 0 in reference(model, prompts[0], 4)
    assert 0 not in minimum_reference(prompts[0], 4)

    uids = batch.insert(
        prompts,
        max_tokens=[4] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
        logits_processors=[[minimum] for _ in prompts],
    )
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
    rng_before = {lane.uid: lane.rng.snapshot() for lane in lanes}
    batch._round(lanes)
    output = {lane.uid: [item.token for item in lane.ready] for lane in lanes}
    final = {lane.uid: lane.ready[-1] for lane in lanes}
    for uid, prompt in zip(uids, prompts, strict=True):
        assert output[uid] == minimum_reference(prompt, len(output[uid]))
        alternatives = final[uid].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["last_active_branches"] == 4
        assert alternatives["branch_downshift_rounds"] == 0
        assert alternatives["last_branch_downshift"] is None
        assert alternatives["last_shared_prefix_tokens"] == 0
        assert alternatives["last_topology"] == "rootless_sibling_fanout"
        assert alternatives["last_branch_proposal_counts"] == [3, 2, 1, 2]
        diagnostics = alternatives["last_prefix_diagnostics"]
        assert diagnostics["chosen_prefix"] == 0
        assert diagnostics["chosen_width"] == 4
        assert all(
            width == 1
            for prefix, width in diagnostics["feasible_width_by_prefix"]
            if prefix != 0
        )
        assert (0, 4) in diagnostics["feasible_width_by_prefix"]
        assert alternatives["cross_session_ast_packing"]["observed_used"], (
            {
                key: value
                for key, value in batch.scheduler_stats.items()
                if "progressive" in key or "cohort" in key
            }
        )
        pool = final[uid].speculative_receipt["continuation_pool"]
        assert pool["ast_observed_used"]
        assert pool["selected_path"] == 1
        assert pool["selected_physical_path"] == 1
        assert pool["selected_ast_provenance"]["rank"] == 1
        assert pool["admitted_complete_sequences"] == 4
        assert pool["physical_span"] == 8
        lane = batch.lanes[uid]
        assert lane.rng.snapshot() == rng_before[uid]
    assert batch.scheduler_stats[
        "external_progressive_ast_branch_downshift_rounds"
    ] == 0
    assert batch.scheduler_stats[
        "external_progressive_ast_branch_downshift_lanes"
    ] == 0
    assert batch.scheduler_stats["external_progressive_ast_rootless_rounds"] == 1
    assert batch.scheduler_stats["external_progressive_ast_rootless_lanes"] == 4
    assert batch.scheduler_stats[
        "external_progressive_cross_session_ast_rounds"
    ] == 1
    assert batch.scheduler_stats[
        "external_progressive_cross_session_ast_sessions"
    ] == 4
    assert not wrapper.proposal_pool._pending

    tail_output, completed = drain(batch)
    for lane, prompt in zip(lanes, prompts, strict=True):
        assert tail_output[lane.uid] == minimum_reference(prompt, 4)
        assert state_digest(completed[lane.uid].cache_sidecar) == state_digest(
            batch._sidecar(lane)
        )
    assert not wrapper.proposal_pool._pending

    unsupported = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-unsupported-processor-fallback",
        completion_batch_size=4,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        progressive_cross_session_pack_rows=8,
        progressive_cross_session_pack_mode="ordered_live_frontier",
    )

    def opaque_passthrough(_tokens, value):
        return value

    unsupported_uids = unsupported.insert(
        prompts,
        max_tokens=[4] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
        logits_processors=[[opaque_passthrough] for _ in prompts],
    )
    unsupported_lanes = [unsupported.lanes[uid] for uid in unsupported_uids]
    for lane in unsupported_lanes:
        while lane.anchor is None:
            unsupported._prefill(lane)
    unsupported._round(unsupported_lanes)
    for uid, prompt in zip(unsupported_uids, prompts, strict=True):
        emitted = [item.token for item in unsupported.lanes[uid].ready]
        assert emitted == reference(model, prompt, len(emitted))
        alternatives = unsupported.lanes[uid].ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["rounds"] == 0
        assert alternatives["primary_fallback_reasons"][
            "unsupported_processors"
        ] >= 1
    assert unsupported.scheduler_stats[
        "external_progressive_ast_primary_fallback_unsupported_processors_rounds"
    ] >= 4
    assert not wrapper.proposal_pool._pending

    ordinary = ExternalDraftBatchGenerator(
        model,
        draft_model=base,
        binding="continuation-ast-rootless-top-ranks-ordinary",
        completion_batch_size=4,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
    )
    ordinary_uids = ordinary.insert(
        prompts,
        max_tokens=[4] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
        logits_processors=[[minimum] for _ in prompts],
    )
    for uid in ordinary_uids:
        ordinary.disable_speculation(uid)
    ordinary_output, ordinary_final = drain(ordinary)
    for lane, ordinary_uid in zip(lanes, ordinary_uids, strict=True):
        assert ordinary_output[ordinary_uid] == minimum_reference(prompts[0], 4)
        for ast_cache, ordinary_cache in zip(
            completed[lane.uid].prompt_cache,
            ordinary_final[ordinary_uid].prompt_cache,
            strict=True,
        ):
            assert ast_cache.offset == ordinary_cache.offset
            np.testing.assert_allclose(
                np.asarray(ast_cache.keys)[..., : ast_cache.offset, :],
                np.asarray(ordinary_cache.keys)[..., : ordinary_cache.offset, :],
                rtol=0,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(ast_cache.values)[..., : ast_cache.offset, :],
                np.asarray(ordinary_cache.values)[..., : ordinary_cache.offset, :],
                rtol=0,
                atol=2e-6,
            )


def test_continuation_ast_rootless_dflash_shape_cross_session_packs(monkeypatch):
    """Ranked sibling roots use exact private base states, as in live Muse r7."""

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    from scripts.paired_direct_ab import state_digest

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"], "limit": 8})

    def mixed_topology_paths(context, _limit):
        target = tuple(
            reference(model, [*context.history, context.anchor], context.depth)
        )

        def changed(delta):
            tokens = list(target)
            position = {3: 0, 4: 1, 5: 2}[context.anchor]
            tokens[position] = (
                tokens[position] + delta
            ) % base.config.vocab_size
            return tuple(tokens)

        def changed_at(position, delta):
            tokens = list(target)
            tokens[position] = (
                tokens[position] + delta
            ) % base.config.vocab_size
            return tuple(tokens)

        confidence = (0.9, 0.8, 0.1)
        paths = (
            SourceContinuation(changed(1), confidence, 8.0, "forced"),
            SourceContinuation(target, confidence, 7.0, "forced"),
            SourceContinuation(changed(2), confidence, 6.0, "forced"),
            SourceContinuation(changed(3), confidence, 5.0, "forced"),
            SourceContinuation(changed(4), confidence, 4.0, "forced"),
            SourceContinuation(changed(5), confidence, 3.0, "forced"),
        )
        if context.anchor == 5:
            # The suggested prefix can form B2 from lower-ranked token-1
            # fanout, while the deeper prefix forms B4 from the first four
            # ranked paths. Width must win before prefix preference.
            paths += (
                SourceContinuation(changed_at(1, 1), confidence, 2.0, "forced"),
            )
        return paths

    monkeypatch.setitem(wrapper.providers, "external", mixed_topology_paths)
    draft_distributions = wrapper.draft_distributions

    def confidence_tail(*args, **kwargs):
        result = draft_distributions(*args, **kwargs)
        wrapper.adaptive_confidence_features = tuple(
            (1.0, 0.5, -2.0) for _ in result[0]
        )
        return result

    monkeypatch.setattr(wrapper, "draft_distributions", confidence_tail)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-rootless-live-shape",
        completion_batch_size=4,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        progressive_cross_session_pack_rows=8,
        progressive_cross_session_pack_mode="ordered_live_frontier",
    )
    prompts = [[1, 2, 3], [1, 2, 3], [1, 2, 4], [1, 2, 5]]
    uids = batch.insert(
        prompts,
        max_tokens=[4] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
    )
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
    rng_before = {lane.uid: lane.rng.snapshot() for lane in lanes}
    batch._round(lanes)

    for lane, prompt in zip(lanes, prompts, strict=True):
        assert [item.token for item in lane.ready] == reference(model, prompt, 4)
        alternatives = lane.ready[-1].speculative_receipt[
            "progressive_verification"
        ]["ordered_alternatives"]
        assert alternatives["winner_selection"] == (
            "exact_max_visited_committed_prefix_then_full_rank"
        )
        assert alternatives["fallback_when_none_full"] == "commit_best_exact_visited_prefix"
        expected_rootless = prompt[-1] == 3
        expected_prefix = {3: 0, 4: 1, 5: 2}[prompt[-1]]
        assert alternatives["last_shared_prefix_tokens"] == (
            expected_prefix
        )
        assert alternatives["rootless_rounds"] == int(expected_rootless)
        assert alternatives["last_topology"] == {
            3: "rootless_sibling_fanout",
            4: "shared_trunk",
            5: "adaptive_shared_trunk",
        }[prompt[-1]]
        assert alternatives["adaptive_prefix_rounds"] == int(prompt[-1] in {3, 5})
        assert alternatives["last_prefix_search"] == (
            {"suggested": 1, "active": expected_prefix}
            if prompt[-1] in {3, 5}
            else None
        )
        diagnostics = alternatives["last_prefix_diagnostics"]
        assert diagnostics["chosen_prefix"] == expected_prefix
        assert diagnostics["chosen_width"] == 4
        assert diagnostics["attempted_prefixes"][0] == 1
        assert all(
            len(item) == 2 for item in diagnostics["feasible_width_by_prefix"]
        )
        assert not any(
            key in diagnostics for key in ("tokens", "paths", "path_ids")
        )
        if prompt[-1] == 5:
            assert (1, 2) in diagnostics["feasible_width_by_prefix"]
            assert (2, 4) in diagnostics["feasible_width_by_prefix"]
        assert alternatives["last_active_branches"] == 4
        assert alternatives["cross_session_ast_packing"]["observed_used"]
        geometry = alternatives["geometry_totals"]
        assert geometry["unique_target_rows"] > 0
        assert geometry["unique_target_rows"] == (
            geometry["shared_prefix_target_rows"] + geometry["branch_suffix_target_rows"]
        )
        assert geometry["selected_committed_inputs"] == len(lane.ready)
        pool = lane.ready[-1].speculative_receipt["continuation_pool"]
        assert pool["ast_observed_used"]
        assert pool["selected_path"] == 1
        assert pool["selected_physical_path"] == 1
        assert pool["selected_ast_provenance"]["rank"] == 1
        if expected_rootless:
            assert state_digest(lane.ready[-1].cache_sidecar) == state_digest(
                batch._sidecar(lane)
            )
    assert batch.scheduler_stats["external_progressive_ast_rootless_rounds"] == 1
    assert batch.scheduler_stats["external_progressive_ast_rootless_lanes"] == 2
    assert batch.scheduler_stats["external_progressive_ast_adaptive_prefix_rounds"] == 1
    assert batch.scheduler_stats["external_progressive_ast_adaptive_prefix_lanes"] == 3
    assert batch.scheduler_stats["external_progressive_cross_session_ast_rounds"] == 1
    assert batch.scheduler_stats["external_progressive_cross_session_ast_sessions"] == 4
    assert not wrapper.proposal_pool._pending

    single = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-rootless-live-shape-b1",
        completion_batch_size=1,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
    )
    single_uid = single.insert(
        [prompts[0]],
        max_tokens=[4],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    single_output, single_final = drain(single)
    assert single_output[single_uid] == reference(model, prompts[0], 4)
    single_alternatives = single_final[single_uid].speculative_receipt[
        "progressive_verification"
    ]["ordered_alternatives"]
    assert single_alternatives["last_shared_prefix_tokens"] == 0
    assert single_alternatives["last_topology"] == "rootless_sibling_fanout"
    assert single_alternatives["rounds"] >= 1
    assert single_final[single_uid].speculative_receipt["continuation_pool"][
        "ast_observed_used"
    ]
    assert not wrapper.proposal_pool._pending

    refused = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-rootless-live-shape-refused",
        completion_batch_size=1,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=2,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        progressive_alternative_depth_policy="confidence_tail",
        progressive_ast_rootless_fallback=False,
    )
    refused_uid = refused.insert(
        [prompts[0]],
        max_tokens=[4],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    refused_lane = refused.lanes[refused_uid]
    while refused_lane.anchor is None:
        refused._prefill(refused_lane)
    refused_rng = refused_lane.rng.snapshot()
    refused._round([refused_lane])
    refused_output = [item.token for item in refused_lane.ready]
    assert refused_output == reference(model, prompts[0], len(refused_output))
    refused_receipt = refused_lane.ready[-1].speculative_receipt
    refused_alternatives = refused_receipt["progressive_verification"][
        "ordered_alternatives"
    ]
    assert refused_alternatives["rootless_fallback_allowed"] is False
    assert refused_alternatives["rootless_refusal_rounds"] == 1
    assert refused_alternatives["rounds"] == 0
    assert refused_alternatives["last_active_branches"] == 1
    assert refused_alternatives["last_primary_fallback_reason"] == (
        "rootless_refused"
    )
    assert refused_alternatives["rootless_rounds"] == 0
    assert refused_alternatives["last_prefix_diagnostics"] is None
    assert refused_alternatives["last_proposed_prefix_diagnostics"][
        "chosen_prefix"
    ] == 0
    assert not refused_receipt["continuation_pool"]["ast_observed_used"]
    assert refused.scheduler_stats[
        "external_progressive_ast_rootless_refusal_rounds"
    ] == 1
    assert refused.scheduler_stats[
        "external_progressive_ast_rootless_refusal_sessions"
    ] == 1
    assert refused.scheduler_stats[
        "external_progressive_ast_primary_fallback_rootless_refused_rounds"
    ] == 1
    assert refused.scheduler_stats["external_progressive_ast_rootless_rounds"] == 0
    assert refused_lane.rng.snapshot() == refused_rng
    assert not wrapper.proposal_pool._pending

    ordinary = ExternalDraftBatchGenerator(
        model,
        draft_model=base,
        binding="continuation-ast-rootless-live-shape-ordinary",
        completion_batch_size=4,
        num_draft=3,
        prefill_step_size=64,
        ready_drain="all",
    )
    ordinary_uids = ordinary.insert(
        prompts,
        max_tokens=[4] * 4,
        sampling_configs=[{"sampling_temp": 0}] * 4,
    )
    for uid in ordinary_uids:
        ordinary.disable_speculation(uid)
    ordinary_output, ordinary_final = drain(ordinary)
    for ast_lane, ordinary_uid in zip(lanes, ordinary_uids, strict=True):
        assert ordinary_output[ordinary_uid] == [item.token for item in ast_lane.ready]
        assert ast_lane.rng.snapshot() == rng_before[ast_lane.uid]
        for ast_cache, ordinary_cache in zip(
            ast_lane.ready[-1].prompt_cache,
            ordinary_final[ordinary_uid].prompt_cache,
            strict=True,
        ):
            assert ast_cache.offset == ordinary_cache.offset
            np.testing.assert_allclose(
                np.asarray(ast_cache.keys)[..., : ast_cache.offset, :],
                np.asarray(ordinary_cache.keys)[..., : ordinary_cache.offset, :],
                rtol=0,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                np.asarray(ast_cache.values)[..., : ast_cache.offset, :],
                np.asarray(ordinary_cache.values)[..., : ordinary_cache.offset, :],
                rtol=0,
                atol=2e-6,
            )


@pytest.mark.parametrize("split_groups", [False, True])
@pytest.mark.parametrize("sparse_first", [False, True])
def test_continuation_ast_partial_group_feasibility_is_session_local(
    monkeypatch, split_groups, sparse_first
):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    def build(*, partial):
        model, base = pair("xpress")
        model.progressive_external_verify_protocol = (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION
        )
        wrapper = wrapped(model, base, {"sources": ["external"], "limit": 8})

        def partially_sparse(context, _limit):
            target = tuple(
                reference(model, [*context.history, context.anchor], context.depth)
            )
            confidence = (0.9, 0.8, 0.1)
            if context.anchor == 4:
                return (SourceContinuation(target, confidence, 1.0, "forced"),)

            def changed_root(delta):
                tokens = list(target)
                tokens[0] = (tokens[0] + delta) % base.config.vocab_size
                return tuple(tokens)

            candidates = (
                SourceContinuation(changed_root(1), confidence, 4.0, "forced"),
                SourceContinuation(target, confidence, 3.0, "forced"),
                SourceContinuation(changed_root(2), confidence, 2.0, "forced"),
                SourceContinuation(changed_root(3), confidence, 1.0, "forced"),
            )
            return candidates if partial else candidates[:1]

        monkeypatch.setitem(wrapper.providers, "external", partially_sparse)
        kwargs = {
            "continuation_pool_ast_bridge": True,
            "progressive_verification_tile": 2,
            "progressive_alternative_branches": 4,
            "progressive_alternative_strategy": "confidence_tree",
            "progressive_alternative_depth_policy": "confidence_tail",
            "progressive_cross_session_pack_rows": 8,
            "progressive_cross_session_pack_mode": "ordered_live_frontier",
        }
        batch = ExternalDraftBatchGenerator(
            model,
            draft_model=wrapper,
            binding="continuation-ast-partial-feasibility",
            completion_batch_size=4,
            num_draft=3,
            prefill_step_size=64,
            ready_drain="all",
            continuation_pool=wrapper.policy,
            **kwargs,
        )
        prompts = [[1, 2, 3], [1, 2, 3], [1, 2, 4], [1, 2, 4]]
        if sparse_first:
            prompts.reverse()
        uids = batch.insert(
            prompts,
            max_tokens=[4] * 4,
            sampling_configs=[{"sampling_temp": 0}] * 4,
        )
        lanes = [batch.lanes[uid] for uid in uids]
        for lane in lanes:
            while lane.anchor is None:
                batch._prefill(lane)
        # Exercise the production grouping boundary: one draft group can form
        # rootless B4 while the other cannot form even B2.
        if split_groups:
            for lane in lanes[2:]:
                lane.tail = lane.tail.astype(mx.float16)
        before_rng = {lane.uid: lane.rng.snapshot() for lane in lanes}
        batch._round(lanes)
        return batch, wrapper, prompts, lanes, before_rng

    ast, ast_wrapper, prompts, ast_lanes, ast_rng_before = build(partial=True)
    control, control_wrapper, _, control_lanes, control_rng_before = build(
        partial=False
    )

    for ast_lane, control_lane, prompt in zip(
        ast_lanes, control_lanes, prompts, strict=True
    ):
        expected = reference(ast.model, prompt, len(ast_lane.ready))
        assert [item.token for item in ast_lane.ready] == expected
        assert [item.token for item in control_lane.ready] == expected[:len(control_lane.ready)]
        receipt = ast_lane.ready[-1].speculative_receipt
        branching = prompt[-1] == 3
        assert receipt["continuation_pool"]["ast_observed_used"] is branching
        alternatives = receipt["progressive_verification"]["ordered_alternatives"]
        assert alternatives["last_active_branches"] == (4 if branching else 1)
        assert alternatives["cross_session_ast_packing"]["observed_used"]
        if branching:
            assert alternatives["last_topology"] == "rootless_sibling_fanout"
            assert alternatives["last_selected_branch"] == 2
        else:
            assert alternatives["last_primary_fallback_reason"] == "insufficient_branches"
            assert ast_lane.history == control_lane.history
            # This tiny generic target declares the protocol but does not
            # implement a row-exact executor. Different CPU slab widths differ
            # by <= 2.4e-7 here; test owned prefix/state plumbing, not Metal bits.
            for actual, expected_cache in zip(ast_lane.cache, control_lane.cache, strict=True):
                assert actual.offset == expected_cache.offset
                for name in ("keys", "values"):
                    np.testing.assert_allclose(
                        np.asarray(getattr(actual, name))[..., :actual.offset, :],
                        np.asarray(getattr(expected_cache, name))[..., :expected_cache.offset, :],
                        rtol=0, atol=2e-6,
                    )
            np.testing.assert_allclose(np.asarray(ast_lane.tail), np.asarray(control_lane.tail), rtol=0, atol=2e-6)
        assert ast_lane.rng.snapshot() == ast_rng_before[ast_lane.uid]
        assert control_lane.rng.snapshot() == control_rng_before[control_lane.uid]
    assert ast.scheduler_stats["external_progressive_ast_rootless_rounds"] == 1
    assert ast.scheduler_stats["external_progressive_ast_rootless_lanes"] == 2
    assert not ast_wrapper.proposal_pool._pending
    assert not control_wrapper.proposal_pool._pending


def test_continuation_ast_bridge_refuses_approximate_target_fidelity():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})
    with pytest.raises(
        ValueError,
        match="requires exact progressive ordered alternatives",
    ):
        ExternalDraftBatchGenerator(
            model,
            draft_model=wrapper,
            binding="continuation-ast-approximate-refusal",
            completion_batch_size=1,
            num_draft=2,
            prefill_step_size=64,
            continuation_pool=wrapper.policy,
            continuation_pool_ast_bridge=True,
            progressive_verification_tile=1,
            progressive_alternative_branches=2,
            progressive_verification_fidelity="near_tie_bounded",
        )


def test_continuation_ast_sparse_slate_falls_back_to_exact_primary(monkeypatch):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})

    def sparse(*_args, **_kwargs):
        raise InsufficientContinuationAlternatives("content-dependent sparse slate")

    monkeypatch.setattr(wrapper, "ordered_alternative_proposals", sparse)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-primary-fallback",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
    )
    prompt = [1, 2, 3]
    batch.insert([prompt], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}])
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 6)
    assert batch.scheduler_stats[
        "external_progressive_ast_primary_fallback_rounds"
    ] > 0
    alternatives = final[0].speculative_receipt["progressive_verification"][
        "ordered_alternatives"
    ]
    assert alternatives["primary_fallback_rounds"] > 0
    assert alternatives["primary_fallback_reasons"] == {
        "insufficient_branches": alternatives["primary_fallback_rounds"]
    }
    assert alternatives["last_primary_fallback_reason"] == "insufficient_branches"
    pool = final[0].speculative_receipt["continuation_pool"]
    assert pool["observed_used"]
    assert not pool["ast_observed_used"]
    assert pool["ast_committed_rounds"] == 0
    assert pool["ast_primary_fallback_committed_rounds"] > 0
    assert pool["ranked_complete_sequences"] > 0
    assert pool["admitted_complete_sequences"] == 1
    assert pool["physical_width"] == 1
    assert pool["physical_span"] > 0
    assert not pool["strategy_observed_used"]
    assert pool["selected_path"] == 0
    assert pool["selected_sources"]
    assert pool["selected_ast_provenance"] is None
    assert not wrapper.proposal_pool._pending


@pytest.mark.parametrize("planner", [False, True])
def test_continuation_ast_fault_discards_ticket_and_retry_is_stable(monkeypatch, planner):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    from scripts.paired_direct_ab import state_digest

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"], "ast_planner": planner})
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-fault-retry",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
        progressive_alternative_strategy="distinct_root",
    )
    prompt = [1, 2, 3]
    uid = batch.insert(
        [prompt],
        max_tokens=[6],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    before_history = list(lane.history)
    before_anchor = lane.anchor
    before_rng = lane.rng.snapshot()
    before_target_digest = state_digest(batch._freeze_cache(lane.cache))
    before_draft_digest = state_digest(batch._sidecar(lane))
    verify = batch._progressive_verify_many

    def fault(*_args, **_kwargs):
        raise RuntimeError("injected AST verification fault")

    monkeypatch.setattr(batch, "_progressive_verify_many", fault)
    with pytest.raises(RuntimeError, match="injected AST verification fault"):
        batch._round([lane])
    assert lane.history == before_history
    assert lane.anchor == before_anchor
    assert lane.rng.snapshot() == before_rng
    assert state_digest(batch._freeze_cache(lane.cache)) == before_target_digest
    assert state_digest(batch._sidecar(lane)) == before_draft_digest
    assert not lane.ready
    assert not wrapper.proposal_pool._pending
    assert not getattr(lane, "continuation_ast_committed_rounds", 0)
    if planner:
        assert not batch.ast_execution_planner._survival
        assert not hasattr(lane, "ast_source_budget")
    monkeypatch.setattr(batch, "_progressive_verify_many", verify)
    output, final = drain(batch)
    assert output[uid] == reference(model, prompt, 6)
    if not planner:
        assert final[uid].speculative_receipt["continuation_pool"]["ast_observed_used"]
    assert not wrapper.proposal_pool._pending


def test_continuation_ast_malformed_branch_provenance_fails_before_commit(monkeypatch):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    from scripts.paired_direct_ab import state_digest

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})

    def two_paths(context, _limit):
        target = tuple(
            reference(
                model,
                [*context.history, context.anchor],
                context.depth,
            )
        )
        wrong = ((target[0] + 1) % base.config.vocab_size, *target[1:])
        return (
            SourceContinuation(wrong, (0.5,) * context.depth, 2.0, "forced"),
            SourceContinuation(target, (0.5,) * context.depth, 1.0, "forced"),
        )

    monkeypatch.setitem(wrapper.providers, "external", two_paths)
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-malformed-provenance",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
        progressive_alternative_strategy="distinct_root",
    )
    uid = batch.insert(
        [[1, 2, 3]],
        max_tokens=[6],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    before_history = list(lane.history)
    before_anchor = lane.anchor
    before_rng = lane.rng.snapshot()
    before_target_digest = state_digest(batch._freeze_cache(lane.cache))
    before_draft_digest = state_digest(batch._sidecar(lane))
    propose = batch._propose

    def corrupt_provenance(*args, **kwargs):
        blocks = propose(*args, **kwargs)
        for alternatives in batch._last_alternative_blocks.values():
            for block in alternatives:
                if block.continuation_ast_provenance is not None:
                    block.continuation_ast_provenance["rank"] = (
                        block.continuation_ast_provenance["rank"] + 1
                    ) % len(alternatives)
        return blocks

    monkeypatch.setattr(batch, "_propose", corrupt_provenance)
    with pytest.raises(RuntimeError, match="branch/rank mapping is invalid"):
        batch._round([lane])
    assert lane.history == before_history
    assert lane.anchor == before_anchor
    assert lane.rng.snapshot() == before_rng
    assert state_digest(batch._freeze_cache(lane.cache)) == before_target_digest
    assert state_digest(batch._sidecar(lane)) == before_draft_digest
    assert not wrapper.proposal_pool._pending
    assert not getattr(lane, "continuation_ast_committed_rounds", 0)


def test_continuation_ast_cancellation_leaves_no_pending_ticket():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-cancel",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
    )
    uid = batch.insert([[1, 2, 3]], max_tokens=[6])[0]
    assert batch.remove([uid], cancelled=True) == {}
    assert not batch.lanes
    assert not wrapper.proposal_pool._pending


def test_continuation_ast_processor_uses_exact_primary_chain():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )

    class IdentityProcessor:
        history_pure = True

        def __call__(self, _tokens, logits):
            return logits

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-processor-primary",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
    )
    prompt = [1, 2, 3]
    batch.insert(
        [prompt],
        max_tokens=[6],
        sampling_configs=[{"sampling_temp": 0}],
        logits_processors=[[IdentityProcessor()]],
    )
    output, final = drain(batch)
    assert output[0] == reference(model, prompt, 6)
    assert not final[0].speculative_receipt["progressive_verification"][
        "observed_used"
    ]
    assert final[0].speculative_receipt["continuation_pool"]["observed_used"]
    assert not wrapper.proposal_pool._pending


def test_inventory_admits_resident_mtp_only_when_named_and_refuses_missing_source():
    model, draft = _native_qsa_pair("xpress")
    # A resident head is never silently added: it costs a head pass per round.
    assert set(wrapped(model, draft).providers) == {"external", "prompt_lookup"}
    wrapper = wrapped(
        model, draft, {"sources": ["external", "prompt_lookup", "native_mtp"]}
    )
    assert set(wrapper.providers) == {"external", "prompt_lookup", "native_mtp"}
    model, draft = pair("xpress")
    assert set(wrapped(model, draft).providers) == {"external", "prompt_lookup"}
    with pytest.raises(ValueError, match="native MTP"):
        wrapped(model, draft, {"sources": ["native_mtp"]})
    for missing in ("dpara", "dflash2", "unknown"):
        with pytest.raises(ValueError, match="unavailable"):
            wrapped(model, draft, {"sources": [missing]})


def test_wrong_request_context_refused_before_backbone_cache_advance():
    model, draft = pair("xpress")
    wrapper = wrapped(model, draft)
    ctx = context(model, draft)
    with pytest.raises(ValueError, match="revision mismatch"):
        wrapper.draft_distributions(
            [ctx.anchor],
            ctx.pending_features,
            ctx.draft_cache,
            2,
            [None],
            [0],
            processor_histories=[ctx.history],
            proposal_contexts=[
                {
                    "request_id": "r",
                    "round_id": 0,
                    "context_revision": hashlib.sha256(b"wrong").hexdigest(),
                }
            ],
        )
    assert draft.stats["backbone_blocks"] == 0 and ctx.draft_cache[0].offset == 0


def test_runtime_honors_disabled_pld_fallback():
    from mlx2.runtime.external_speculative import DraftUnavailable

    model, draft = pair("xpress")
    wrapper = wrapped(
        model,
        draft,
        {
            "ngram_min": 1,
            "ngram_max": 1,
            "admission": {"pld_fallback": False},
        },
    )
    wrapper.providers["external"] = lambda _context, _limit: []
    history, anchor = [1, 2, 1, 3], 1
    hidden = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    identity = {
        "request_id": "pld-disabled",
        "round_id": 0,
        "context_revision": continuation_context_revision(
            wrapper.session.session_revision, history, anchor
        ),
    }
    pld_context = ContinuationContext(
        tuple(history), anchor, 2, draft.make_cache(), hidden
    )
    assert wrapper.providers["prompt_lookup"](pld_context, 15)
    with pytest.raises(DraftUnavailable, match="no valid paths"):
        wrapper.draft_distributions(
            [anchor],
            hidden,
            draft.make_cache(),
            2,
            [None],
            [0],
            processor_histories=[history],
            proposal_contexts=[identity],
        )


def test_shared_adapter_pool_feedback_uses_final_binding_and_closes(tmp_path):
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = pair("lilicorr")
    adapter = ExternalDraftAdapterMixin()
    adapter.model, adapter.layout = model, "target-layout"
    adapter.identity = {"fingerprint": PIN}
    adapter.external_policy = {
        "draft_model": "synthetic",
        "num_draft": 2,
        "continuation_pool": {"ngram_min": 1, "ngram_max": 1},
        "lilicorr_feedback": {
            "directory": str(tmp_path),
            "max_examples": 2,
            "min_examples": 2,
            "train_every": 100,
        },
    }
    record = {"fingerprint": "b" * 64, "args": draft.config}
    adapter._check_num_draft(record)
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter._bind_external_drafter(record, lambda record, target: draft, descriptor)
    assert draft.feedback_manager is None
    final = hashlib.sha256(
        (adapter.identity["fingerprint"] + "final-settings").encode()
    ).hexdigest()
    adapter.identity["fingerprint"] = final
    adapter.create_external_batch()
    manager = draft.feedback_manager
    assert manager.binding == final and not manager.closed
    assert adapter.draft_model.feedback_manager is manager
    adapter.close()
    assert manager.closed and manager.child is None


def test_header_policy_refuses_incompatible_feedback_or_continuation_before_load(
    tmp_path,
):
    from types import SimpleNamespace

    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin

    adapter = ExternalDraftAdapterMixin()
    adapter.external_policy = {
        "num_draft": 2,
        "lilicorr_feedback": {"directory": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="LiLiCoRR artifact"):
        adapter._check_num_draft({"args": SimpleNamespace(block_size=4)})
    adapter.external_policy = {"num_draft": 2, "continuation_pool": {}}
    with pytest.raises(ValueError, match="complete-path external head"):
        adapter._check_num_draft(
            {"args": SimpleNamespace(block_size=4), "config": {}}
        )


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_actual_pool_serving_cold_warm_apcv2_window_and_complete_path_counter(
    kind, monkeypatch
):
    model, draft = pair(kind, [2])
    wrapper = wrapped(model, draft, {"ngram_min": 1, "ngram_max": 1})
    engine, batches = serving_engine(monkeypatch, kind, model, wrapper)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    try:
        cold = collect(
            engine.submit(
                {"tokens": prompt, "max_tokens": 8, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in cold
        assert cold["tokens"] == reference(model, prompt, 8)
        assert batches[0].scheduler_stats["external_continuation_sequences"] > 0
        assert batches[0].scheduler_stats["target_max_width"] > 1
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


def test_constrained_target_outside_lili_candidates_falls_back_ordinary_without_labels():
    model, draft = pair("lilicorr")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    prompt = [1, 2, 3]

    class ForceAfterPrompt:
        history_pure = True
        token = None

        def __call__(self, tokens, logits):
            if int(tokens.size) <= len(prompt) or self.token is None:
                return logits
            return mx.where(mx.arange(logits.shape[-1]) == self.token, logits, -mx.inf)

    processor = ForceAfterPrompt()
    batch = generator(model, wrapper)
    batch.insert([prompt], max_tokens=[6], logits_processors=[[processor]])
    lane = batch.lanes[0]
    while lane.remaining:
        batch._prefill(lane)
    current = ContinuationContext(
        tuple(lane.history), lane.anchor, 2, lane.draft_cache, lane.tail
    )
    possible = ExternalContinuationSource(draft)(current, 15)
    first_ids = {path.tokens[0] for path in possible}
    processor.token = next(
        token for token in range(draft.config.vocab_size) if token not in first_ids
    )
    revision = wrapper.proposal_pool.feedback_revision
    output, final = drain(batch)
    assert output[0] == [reference(model, prompt, 1)[0]] + [processor.token] * 5
    assert batch.scheduler_stats["draft_fallbacks"] == 1
    assert final[0].speculative_receipt["ordinary_fallback"]
    assert wrapper.proposal_pool.feedback_revision == revision


def test_empty_model_drafter_round_is_charged_to_the_fallback_only(monkeypatch):
    # Sweep 2026-10-16 SPEC-1: the admitted xpress head returned no path and
    # prompt lookup served as the runtime fallback; only the fallback provider
    # has a fallback round.
    from mlx2.runtime import proposal_pool
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    monkeypatch.setattr(
        proposal_pool, "_SHARED_RANKING_REGISTRY", proposal_pool.ProposalRankingRegistry()
    )
    model, base = pair("xpress")
    draft = wrapped(
        model, base, {"sources": ["external", "prompt_lookup"], "ngram_min": 1, "ngram_max": 1}
    )
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="sweep1016-fallback", num_draft=2,
        prefill_step_size=64, ready_drain="all",
    )
    draft.providers["external"] = lambda _context, _limit: []
    lane = batch.lanes[batch.insert([[1, 2, 1, 2, 1, 2]], max_tokens=[8])[0]]
    while lane.anchor is None:
        batch._prefill(lane)
    batch._round([lane])
    assert lane.continuation_last_admission["runtime_fallback"] == "prompt_lookup"
    stats = lane.continuation_provider_stats
    assert stats["external"]["fallback_rounds"] == 0
    assert stats["prompt_lookup"]["fallback_rounds"] == 1


def test_scoped_continuation_receipt_hides_shared_critic_slate_numbers(monkeypatch):
    # Sweep 2026-10-16 SPEC-2: the slate's acceptance predictions blend the
    # shared ranking critic, which every tenant's committed labels move; a
    # tenant-scoped receipt keeps the slate's shape but not those numbers.
    from mlx2.runtime import proposal_pool
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    monkeypatch.setattr(
        proposal_pool, "_SHARED_RANKING_REGISTRY", proposal_pool.ProposalRankingRegistry()
    )
    model, base = pair("xpress")
    draft = wrapped(model, base, {"sources": ["external"]})
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="sweep1016-slate", num_draft=2,
        prefill_step_size=64, ready_drain="all",
    )
    scoped, public = batch.insert(
        [[1, 2, 3, 4], [1, 2, 3, 4]], max_tokens=[8, 8], session_keys=["b" * 64, None]
    )
    for uid in (scoped, public):
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
    batch._round([batch.lanes[scoped], batch.lanes[public]])

    def slate(uid):
        receipt = batch._continuation_receipt(batch.lanes[uid])
        pool = receipt.get("continuation_pool") or receipt["speculation"]["continuation_pool"]
        return pool["last_candidate_slate"]

    hidden = {"conditional_acceptance", "expected_accepted_tokens", "score"}
    for uid in (scoped, public):  # an unkeyed request gets a private scope
        assert slate(uid) and not hidden & set(slate(uid)[0])
        assert {"rank", "depth", "sources", "admitted", "target_selected"} <= set(slate(uid)[0])
        receipt = batch._continuation_receipt(batch.lanes[uid])
        pool = receipt.get("continuation_pool") or receipt["speculation"]["continuation_pool"]
        for row in pool["provider_stats"].values():
            assert not {"score_sum", "score_observations"} & set(row)
        # Internal stats keep the score for ranking diagnostics.
        assert "score_sum" in next(iter(batch.lanes[uid].continuation_provider_stats.values()))


def test_sampled_request_on_point_mass_ast_bridge_decodes_ordinary_with_receipt():
    """Repro (qualify-ast3 2026-10-18): a ``temperature > 0`` request on the
    Muse continuation-pool AST route failed mid-round ("continuation AST
    point-mass verification is greedy-only", HTTP 500).  Point-mass laws
    verify greedily only, so the sampled lane is admitted to the exact
    ordinary target path instead, labelled in its receipt, and draws the same
    tokens as an ordinary generator with the same seed; a greedy lane beside
    it keeps the AST route."""
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.progressive_external_verify import (
        PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    )
    from mlx2.runtime.sample_utils import LaneRNG

    model, base = pair("xpress")
    model.progressive_external_verify_protocol = PROGRESSIVE_TARGET_PROTOCOL_VERSION
    wrapper = wrapped(model, base, {"sources": ["external"]})
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=wrapper,
        binding="continuation-ast-sampled",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=1,
        progressive_alternative_branches=2,
        progressive_alternative_strategy="distinct_root",
    )
    sampled = {"sampling_temp": 0.8}
    prompt = [1, 2, 3]
    sampled_uid, greedy_uid = batch.insert(
        [prompt, [4, 5, 6]],
        max_tokens=[6, 6],
        lane_rngs=[LaneRNG(4242), LaneRNG(7)],
        sampling_configs=[sampled, {"sampling_temp": 0}],
    )
    output, final = drain(batch)
    # Before the fix the sampled lane was dropped as a LaneFailure here.
    assert [str(f) for f in batch.take_lane_failures()] == []
    receipt = final[sampled_uid].speculative_receipt
    assert receipt["ordinary_fallback"] is True
    assert receipt["current_execution"] == "ordinary_target"
    assert receipt["ordinary_fallback_reason"] == (
        "sampled_request_on_greedy_only_point_mass_ast"
    )
    assert batch.scheduler_stats["external_sampled_point_mass_ordinary_lanes"] == 1
    greedy = final[greedy_uid].speculative_receipt
    assert greedy["ordinary_fallback"] is False
    assert greedy["external_rounds"] > 0
    assert output[greedy_uid] == reference(model, [4, 5, 6], 6)

    ordinary = ExternalDraftBatchGenerator(
        model,
        draft_model=base,
        binding="continuation-ast-sampled-control",
        completion_batch_size=1,
        num_draft=2,
        prefill_step_size=64,
        ready_drain="all",
    )
    uid = ordinary.insert(
        [prompt], max_tokens=[6], lane_rngs=[LaneRNG(4242)], sampling_configs=[sampled]
    )[0]
    ordinary.disable_speculation(uid)
    ordinary_output, _ = drain(ordinary)
    assert ordinary_output[uid] == output[sampled_uid]
