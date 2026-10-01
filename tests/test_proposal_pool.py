"""Original host-only tests of complete-path admission and committed critics."""

import itertools
import json
from dataclasses import replace

import pytest

from mlx2.runtime.proposal_pool import (
    PoolRound,
    ProposalPath,
    ProposalPool,
    ProposalRankingRegistry,
    ProposalSession,
    ProposalSource,
    prefix_tree,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64


def session(target=B):
    return ProposalSession(A, target, C, 1000)


def source(name="alpha", mechanism="xpress", revision=D, target=B):
    return ProposalSource(name, mechanism, revision, A, target, C, 1000)


def path(
    tokens=(1, 2, 3), name="alpha", candidate="p", revision=D, features=(), **kwargs
):
    return ProposalPath(
        candidate, name, revision, A, E, tuple(tokens), tuple(features), **kwargs
    )


def round_(index=0, request="request", scope=None):
    return PoolRound(A, E, request, index, scope)


def pool(sources=None, **kwargs):
    return ProposalPool(session(), [source()] if sources is None else sources, **kwargs)


@pytest.mark.parametrize(
    "patch",
    [
        {"loaded": False},
        {"token_path_support": False},
        {"target_revision": D},
        {"tokenizer_revision": D},
        {"session_revision": D},
        {"vocab_size": 12},
        {"max_depth": 16},
    ],
)
def test_source_admission_fails_closed(patch):
    with pytest.raises(ValueError):
        pool([replace(source(), **patch)])


@pytest.mark.parametrize(
    "patch",
    [
        {"source_id": "missing"},
        {"source_revision": B},
        {"session_revision": B},
        {"context_revision": B},
        {"tokens": (1000,)},
        {"tokens": tuple(range(16))},
    ],
)
def test_path_revisions_geometry_fail_closed(patch):
    with pytest.raises(ValueError):
        pool().select([replace(path(), **patch)], round_())


def test_select_fifteen_complete_sequences_and_prefix_closure():
    p = pool()
    paths = [path((i, *range(40, 54)), candidate=str(i)) for i in range(20)]
    selected = p.select(paths, round_())
    assert len(selected.paths) == 15
    tree = prefix_tree(selected)
    assert len(tree.tokens) == 225
    assert len(tree.path_rows) == 15
    for selected_path, rows in zip(selected.paths, tree.path_rows, strict=True):
        assert tuple(tree.tokens[i] for i in rows) == selected_path.tokens
        assert tuple(tree.parents[i] for i in rows) == (-1, *rows[:-1])
    with pytest.raises(ValueError):
        selected.require_verification_contract("accept_reject_q")


def test_duplicate_sequences_keep_provenance_and_share_common_prefix():
    p = pool([source(), source("beta", "lilicorr", B)])
    selected = p.select(
        [path(), path(name="beta", revision=B), path((1, 2, 4), candidate="other")],
        round_(),
    )
    assert len(selected.paths) == 2
    assert len(selected.paths[0].contributors) == 2
    tree = prefix_tree(selected)
    assert tree.tokens == (1, 2, 3, 4)
    assert tree.parents == (-1, 0, 1, 1)
    assert len(tree.contributors[0]) == 3


def test_cold_features_neutral_and_beam_proxy_only_within_source():
    p = pool([source(), source("beta", "lilicorr", B)])
    candidates = [
        path((1,), features=(40,), ranking_score=-99, ranking_convention="log_proxy"),
        path(
            (2,),
            candidate="q",
            features=(-40,),
            ranking_score=-100,
            ranking_convention="log_proxy",
        ),
        path(
            (3,),
            name="beta",
            revision=B,
            features=(-40,),
            ranking_score=-10000,
            ranking_convention="cos_proxy",
        ),
    ]
    selected = p.select(candidates, round_())
    assert [x.tokens for x in selected.paths] == [(1,), (3,), (2,)]
    assert all(
        x.score == 0.5 and x.score_authority == "declared_neutral_prior"
        for x in selected.paths
    )


def test_censored_labels_and_same_source_prefix_dedup():
    p = pool()
    selected = p.select(
        [
            path((1, 2, 3)),
            path((1, 2, 4), candidate="b"),
            path((1, 9, 8), candidate="c"),
        ],
        round_(),
    )
    result = p.commit_feedback(selected, [1, 2, 7])
    assert result["accepted_labels"] == 2
    assert result["rejected_labels"] == 3
    assert p.observation_counts("alpha")[:4] == ((1, 0), (1, 1), (0, 2), (0, 0))
    with pytest.raises(ValueError):
        p.commit_feedback(selected, [1, 2, 7])
    with pytest.raises(ValueError):
        p.select([path()], round_())


def test_budget_stop_censors_undelivered_suffix_and_discard_has_no_labels():
    p = pool()
    selected = p.select([path()], round_())
    p.discard(selected)
    assert p.feedback_revision == 0
    assert all(pair == (0, 0) for pair in p.observation_counts("alpha"))
    selected = p.select([path()], round_())
    p.commit_feedback(selected, [1])
    assert p.observation_counts("alpha")[:3] == ((1, 0), (0, 0), (0, 0))


def test_ticket_authentication_replay_and_pending_guard():
    p = pool()
    selected = p.select([path()], round_())
    assert p.select([path()], round_()) is selected
    with pytest.raises(ValueError):
        p.select([path((2,))], round_())
    with pytest.raises(ValueError):
        p.discard(replace(selected, score_mode="forged"))
    with pytest.raises(ValueError):
        p.commit_feedback(selected, [1000])
    assert p.feedback_revision == 0
    p.discard(selected)
    with pytest.raises(ValueError):
        p.discard(selected)


def test_costs_explicit_units_and_validation_atomic():
    p = pool(score_mode="expected_accepted_per_cost", cost_units="milliseconds")
    with pytest.raises(ValueError):
        p.select([path()], round_())
    selected = p.select([path(cost=2, cost_units="milliseconds")], round_())
    assert selected.paths[0].score == pytest.approx(0.875 / 2)
    with pytest.raises(ValueError):
        p.commit_feedback(selected, [1], costs={"alpha": 0}, cost_units="milliseconds")
    assert p.feedback_revision == 0
    p.commit_feedback(selected, [1], costs={"alpha": 3}, cost_units="milliseconds")
    selected = p.select([path()], round_(1))
    assert selected.paths[0].estimated_cost == 3


def test_ranker_learns_global_model_session_without_raw_tenants():
    registry = ProposalRankingRegistry()
    p = pool([source(), source("beta", "lilicorr", B)], ranking_registry=registry)
    for i in range(8):
        selected = p.select(
            [path((1,)), path((2,), name="beta", revision=B)], round_(i, scope=D)
        )
        p.commit_feedback(selected, [2])
    selected = p.select(
        [path((1,)), path((2,), name="beta", revision=B)], round_(8, scope=D)
    )
    assert selected.paths[0].representative.source_id == "beta"
    local = registry.predict(session(), source(), D, 0, None)[0]
    other_scope = registry.predict(session(), source(), E, 0, None)[0]
    other_model = registry.predict(session(target=E), source(target=E), E, 0, None)[0]
    assert local < other_scope < other_model < 0.5
    receipt = p.receipt(D)["ranking_registry"]
    assert receipt["session_scope_hash"] == D
    assert (
        next(row for row in receipt["session_counts"] if row["mechanism"] == "xpress")[
            "rejected"
        ]
        == 8
    )
    assert (
        next(row for row in receipt["model_counts"] if row["mechanism"] == "xpress")[
            "rejected"
        ]
        == 8
    )
    assert "tenant" not in json.dumps(receipt)
    assert receipt["paper_lattice"] is False


def test_source_revision_rebind_and_shared_global_prior():
    registry = ProposalRankingRegistry()
    p = pool(ranking_registry=registry)
    selected = p.select([path((1,))], round_(scope=D))
    p.commit_feedback(selected, [1])
    old = registry.predict(session(), source(), E, 0, None)[0]
    new = registry.predict(session(), source(revision=B), E, 0, None)[0]
    assert 0.5 < new < old
    assert registry.receipt(session(), D)["revision"] == 1


def test_anonymous_sessions_unique_explicit_scope_wins():
    p = pool(session_scope_hash=D)
    selected = p.select([path()], round_(scope=E))
    assert selected.round.session_scope_hash == E
    p.discard(selected)
    selected = p.select([path()], round_())
    assert selected.round.session_scope_hash == D
    p.discard(selected)
    p = pool()
    one = p.select([path()], round_(request="one"))
    two = p.select([path()], round_(request="two"))
    assert one.round.session_scope_hash != two.round.session_scope_hash
    with pytest.raises(ValueError):
        round_(scope="raw-user-session")


def test_bounded_registry_eviction_and_global_counts_survive():
    registry = ProposalRankingRegistry(max_models=1, max_sessions=1, max_sources=1)
    first = pool(ranking_registry=registry)
    first.commit_feedback(first.select([path((1,))], round_(scope=D)), [1])
    second = ProposalPool(
        session(target=E), [source(target=E)], ranking_registry=registry
    )
    second.commit_feedback(second.select([path((1,))], round_(scope=E)), [2])
    receipt = registry.receipt(session(target=E), E)
    assert receipt["retained_models"] == receipt["retained_sessions"] == 1
    assert receipt["evictions"]["models"] == receipt["evictions"]["sessions"] == 1
    assert (
        receipt["global_counts"][0]["accepted"]
        == receipt["global_counts"][0]["rejected"]
        == 1
    )
    assert registry.receipt(session(), D)["session_counts"] == []


def test_aliases_identical_artifact_count_once_in_shared_registry():
    p = pool([source(), source("alias")])
    selected = p.select([path((1,)), path((1,), name="alias")], round_(scope=D))
    p.commit_feedback(selected, [1])
    receipt = p.receipt(D)["ranking_registry"]
    assert receipt["global_counts"][0]["accepted"] == 1
    assert receipt["model_counts"][0]["accepted"] == 1


def test_stream_candidate_budget_is_bounded_before_materialization():
    p = pool(max_raw_paths=2)
    with pytest.raises(ValueError):
        p.select(itertools.repeat(path()), round_())


def test_receipt_is_serializable_and_no_current_candidate_confidence_q():
    p = pool()
    selected = p.select([path(features=(2, 2, 2))], round_())
    assert "q" not in selected.as_dict()
    assert p.receipt()["qualified"] is False
    json.dumps(p.receipt(), allow_nan=False)


def test_mixed_source_score_conventions_fail_closed():
    p = pool()
    with pytest.raises(ValueError):
        p.select(
            [
                path((1,), ranking_score=1, ranking_convention="log"),
                path((2,), candidate="b", ranking_score=1, ranking_convention="cos"),
            ],
            round_(),
        )


def test_shared_registry_policy_receipt_reports_actual_prior():
    p = pool(ranking_registry=ProposalRankingRegistry(prior_acceptance=0.25))
    selected = p.select([path((1,))], round_())
    assert selected.paths[0].score == 0.25
    assert p.receipt()["declared_prior_acceptance"] == 0.25


@pytest.mark.parametrize(
    "patch", [{"position": -1}, {"position": 15}, {"label": 1}, {"bucket": 1000}]
)
def test_public_registry_feedback_validation_precedes_mutation(patch):
    registry = ProposalRankingRegistry()
    event = (
        source(),
        patch.get("position", 0),
        patch.get("label", True),
        {patch.get("bucket", 0)},
    )
    with pytest.raises(ValueError):
        registry.commit(session(), D, [event])
    assert registry.revision == 0
    assert registry.receipt(session(), D)["global_counts"] == []


def test_confidence_calibration_requires_prior_committed_same_source_labels():
    p = pool()
    selected = p.select(
        [path((1,), features=(20,)), path((2,), candidate="bad", features=(-20,))],
        round_(),
    )
    p.commit_feedback(selected, [1])
    next_ = p.select(
        [path((3,), features=(-20,)), path((4,), candidate="good", features=(20,))],
        round_(1),
    )
    assert next_.paths[0].tokens == (4,)
    assert next_.paths[0].score > next_.paths[1].score
    assert p.feedback_revision == 1


def test_route_settings_revision_separates_model_and_session_residuals():
    registry = ProposalRankingRegistry()
    p = pool(ranking_registry=registry)
    p.commit_feedback(p.select([path((1,))], round_(scope=D)), [1])
    old = registry.predict(session(), source(), D, 0, None)[0]
    new_session = replace(session(), session_revision=E)
    new_source = replace(source(), session_revision=E)
    new = registry.predict(new_session, new_source, D, 0, None)[0]
    assert 0.5 < new < old
    assert (
        registry.receipt(session(), D)["model_counts"][0]["route_settings_revision"]
        == A
    )


def test_shared_registry_threaded_commits_keep_atomic_counts():
    from concurrent.futures import ThreadPoolExecutor

    registry = ProposalRankingRegistry()

    def commit(index):
        p = pool(ranking_registry=registry)
        selected = p.select([path((1,))], round_(request=str(index)))
        p.commit_feedback(selected, [1 if index % 2 else 2])
        assert p.feedback_revision == 1
        return selected.ranking_revision

    with ThreadPoolExecutor(max_workers=8) as executor:
        revisions = list(executor.map(commit, range(64)))
    receipt = registry.receipt(session())
    assert receipt["revision"] == 64
    assert receipt["global_counts"][0]["accepted"] == 32
    assert receipt["global_counts"][0]["rejected"] == 32
    assert all(0 <= revision < 64 for revision in revisions)
