"""Effective head and verification policies pin critic sources before use."""

import hashlib
import json

import mlx.core as mx
import pytest
from test_parallel_draft_apcv2_batching_cpu import pair

from mlx2.adapters.external_draft_policy import (
    DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION,
    ExternalDraftAdapterMixin,
)
from mlx2.contracts import Capability, ModelDescriptor
from mlx2.runtime.proposal_composition import ProposalCompositionPolicy
from mlx2.runtime.proposal_pool import ProposalRankingRegistry
from mlx2.runtime.proposal_providers import ContinuationPoolPolicy

mx.set_default_device(mx.cpu)

TARGET = "a" * 64
DRAFT = "b" * 64
SCOPE = "c" * 64


def bind(
    *, windows=None, passes=3, depth=2, adaptive=None, pool=True, pool_settings=None
):
    model, drafter = pair("xpress", windows)
    drafter.config.xpress_num_passes = passes
    adapter = ExternalDraftAdapterMixin()
    adapter.model, adapter.layout = model, "target-layout"
    adapter.identity = {"fingerprint": TARGET}
    adapter.external_policy = {"draft_model": "synthetic", "num_draft": depth}
    if pool:
        adapter.external_policy["continuation_pool"] = pool_settings or {}
    if adaptive is not None:
        adapter.external_policy["adaptive_verification"] = adaptive
    record = {"fingerprint": DRAFT, "args": drafter.config}
    adapter._check_num_draft(record)
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter._bind_external_drafter(record, lambda _record, _target: drafter, descriptor)
    return adapter


@pytest.mark.parametrize(
    "settings",
    [
        {"passes": 2},
        {"windows": [2]},
        {"depth": 1},
        {"adaptive": {"verification_costs": [1, 2, 3]}},
        {"adaptive": {"continuation_costs": {"15": [1, 2, 3]}}},
    ],
)
def test_effective_settings_separate_session_and_model_source_critic_entries(settings):
    baseline, changed = bind(), bind(**settings)
    first, second = baseline.draft_model, changed.draft_model
    assert first.session.session_revision != second.session.session_revision
    assert first.session.target_revision == second.session.target_revision == TARGET
    a, b = first.source_records["external"], second.source_records["external"]
    assert a.revision == b.revision == DRAFT  # Actual artifact provenance is retained.
    assert a.session_revision != b.session_revision
    registry = ProposalRankingRegistry()
    assert registry.model_key(first.session) == registry.model_key(second.session)
    assert registry.source_key(a) != registry.source_key(b)
    registry.commit(first.session, SCOPE, [(a, 0, True, ())])
    _p, old_observed = registry.predict(first.session, a, SCOPE, 0, None)
    _p, new_observed = registry.predict(second.session, b, SCOPE, 0, None)
    assert old_observed == 3  # Mechanism, model and request-scope observations.
    assert new_observed == 1  # Only the intentional broad mechanism prior transfers.
    registry.commit(second.session, SCOPE, [(b, 0, False, ())])
    receipt = registry.receipt(second.session, SCOPE)
    assert len(receipt["model_counts"]) == len(receipt["session_counts"]) == 2
    assert {entry["route_settings_revision"] for entry in receipt["model_counts"]} == {
        a.session_revision,
        b.session_revision,
    }


def test_adaptive_defaults_numeric_types_and_mapping_order_have_canonical_identity():
    compact = {
        "verification_costs": [1, 2, 3],
        "draft_cost": 1,
        "verification_costs_by_cohort": {"2": [2, 3, 4]},
    }
    explicit = {
        "verification_costs": [1.0, 2.0, 3.0],
        "draft_cost": 1.0,
        "cohort_size": 1,
        "min_observations": 32,
        "full_depth_interval": 32,
        "min_gain": 0.05,
        "minimum_depth": 1,
        "refit_interval": 100,
        "mode": "cohort",
        "verification_costs_by_cohort": {2: [2.0, 3.0, 4.0]},
        "continuation_costs": {},
    }
    assert (
        bind(adaptive=compact).draft_model.session
        == bind(adaptive=explicit).draft_model.session
    )
    assert bind(adaptive=False).draft_model.session == bind().draft_model.session


def test_raw_receipt_changes_are_bound_before_wrapper_construction(monkeypatch):
    from mlx2.adapters import proposal_path_sources

    snapshots = []
    original = proposal_path_sources.build_continuation_drafter

    def checked(target, drafter, policy, **pins):
        assert not hasattr(drafter, "backend")
        snapshots.append((dict(drafter.receipt_settings), pins["session_revision"]))
        return original(target, drafter, policy, **pins)

    monkeypatch.setattr(proposal_path_sources, "build_continuation_drafter", checked)
    bind(passes=2)
    bind(windows=[2])
    assert snapshots[0][0]["xpress_num_passes"] == 2
    assert snapshots[1][0]["draft_attention_windows"] == [2]
    assert snapshots[0][1] != snapshots[1][1]


@pytest.mark.parametrize("pool", [False, True])
def test_final_route_fingerprint_binds_selected_proposal_arbitration(pool):
    adapter = bind(pool=pool, passes=2, windows=[2])
    composition = (
        json.dumps(
            ContinuationPoolPolicy.from_value({}).as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        if pool
        else json.dumps(
            ProposalCompositionPolicy.from_value(
                DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
            ).as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    expected = hashlib.sha256(
        (TARGET + DRAFT + adapter.EXTERNAL_ROUTE_TAG + composition).encode()
    ).hexdigest()
    assert adapter.identity["fingerprint"] == expected
    assert adapter.identity["target_fingerprint"] == TARGET
    assert adapter._external_draft_revision == DRAFT


def test_dedup_algorithm_pins_enabled_pool_route_session_sources_and_receipt():
    from dataclasses import replace

    adapter = bind()
    policy = ContinuationPoolPolicy.from_value({}).as_dict()
    assert policy["verification_algorithm"] == "stable-truncated-prefix-dedup-v2"
    assert (
        adapter.draft_model.receipt_settings["continuation_pool"][
            "verification_algorithm"
        ]
        == policy["verification_algorithm"]
    )
    old_policy = {
        key: value for key, value in policy.items() if key != "verification_algorithm"
    }
    old_route = hashlib.sha256(
        (
            TARGET
            + DRAFT
            + adapter.EXTERNAL_ROUTE_TAG
            + json.dumps(old_policy, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()
    assert adapter.identity["fingerprint"] != old_route
    old_session = hashlib.sha256(
        json.dumps(
            {
                "schema": "mlx2.continuation-session.v2",
                "target_revision": TARGET,
                "draft_revision": DRAFT,
                "route": adapter.EXTERNAL_ROUTE_TAG,
                "continuation_pool": old_policy,
                "draft_settings": adapter.draft_model.backend.receipt_settings,
                "num_draft": 2,
                "adaptive_verification": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    source = adapter.draft_model.source_records["external"]
    assert source.session_revision == adapter.draft_model.session.session_revision
    assert source.session_revision != old_session
    assert source.revision == DRAFT
    assert ProposalRankingRegistry.source_key(
        source
    ) != ProposalRankingRegistry.source_key(
        replace(source, session_revision=old_session)
    )


def test_explicit_current_algorithm_normalizes_to_default_and_roundtrips():
    default = bind()
    explicit = bind(
        pool_settings={"verification_algorithm": "stable-truncated-prefix-dedup-v2"}
    )
    assert default.identity == explicit.identity
    assert default.draft_model.session == explicit.draft_model.session
    normalized = ContinuationPoolPolicy.from_value({}).as_dict()
    assert ContinuationPoolPolicy.from_value(normalized).as_dict() == normalized


def test_longest_first_exact_prefix_algorithm_is_revision_bound_and_explicit():
    from mlx2.runtime.proposal_providers import LONGEST_FIRST_EXACT_PREFIX

    default = ContinuationPoolPolicy.from_value({})
    cascade = ContinuationPoolPolicy.from_value(
        {"verification_algorithm": LONGEST_FIRST_EXACT_PREFIX}
    )
    assert cascade.verification_algorithm == LONGEST_FIRST_EXACT_PREFIX
    assert cascade.as_dict() != default.as_dict()


@pytest.mark.parametrize(
    "algorithm", ["target-draw-then-prefix-match-v1", "", None, False, 2]
)
def test_unknown_or_legacy_pool_algorithm_fails_before_wrapper_binding(algorithm):
    with pytest.raises(ValueError, match="verification_algorithm"):
        ContinuationPoolPolicy.from_value({"verification_algorithm": algorithm})
