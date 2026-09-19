import pytest

from mlx2 import (
    CacheFingerprint,
    CacheMiss,
    CacheOwner,
    Capability,
    Fidelity,
    InvalidStatePublication,
    ModelDescriptor,
    QualifiedProfile,
    RequestStateTransaction,
    RoutePlanner,
    RouteRequest,
    RouteUnavailable,
    RuntimeTelemetry,
    StateConflict,
    StateIdentity,
    StateManifest,
    StateOperation,
    StatePlane,
)
from mlx2.adapters import QWEN36, QWEN4_FLASH_NEXT


def test_route_requires_declared_and_qualified_capabilities():
    planner = RoutePlanner()
    planner.register_model(QWEN4_FLASH_NEXT)

    with pytest.raises(RouteUnavailable):
        planner.decide(
            RouteRequest(
                "qwen4_exp", "flash-next", frozenset({Capability.CONTINUOUS_BATCH})
            )
        )

    profile = QualifiedProfile(
        name="reference-batch",
        capabilities=frozenset({Capability.TEXT, Capability.CONTINUOUS_BATCH}),
        fidelity=Fidelity.EXACT,
        evidence=("tests/batch/reference.json",),
        implementation="mlx2.adapters.qwen:Qwen4ReferenceBatch",
    )
    planner.register_profile(QWEN4_FLASH_NEXT.key, profile)
    decision = planner.decide(
        RouteRequest(
            "qwen4_exp", "flash-next", frozenset({Capability.CONTINUOUS_BATCH})
        )
    )
    assert decision.profile is profile
    assert "profile=reference-batch" in decision.receipt


def test_profile_cannot_claim_undeclared_capability():
    planner = RoutePlanner()
    planner.register_model(QWEN36)
    profile = QualifiedProfile(
        name="premature-apc",
        capabilities=frozenset({Capability.APC_V2}),
        fidelity=Fidelity.EXACT,
        evidence=("placeholder",),
        implementation="missing",
    )
    with pytest.raises(ValueError, match="undeclared"):
        planner.register_profile(QWEN36.key, profile)


def test_registration_rejects_replacement_and_duplicate_profile_names():
    planner = RoutePlanner()
    planner.register_model(QWEN4_FLASH_NEXT)
    with pytest.raises(ValueError, match="already registered"):
        planner.register_model(QWEN4_FLASH_NEXT)

    profile = QualifiedProfile(
        name="reference",
        capabilities=frozenset({Capability.TEXT}),
        fidelity=Fidelity.EXACT,
        evidence=("evidence.json",),
        implementation="reference",
    )
    planner.register_profile(QWEN4_FLASH_NEXT.key, profile)
    with pytest.raises(ValueError, match="profile name"):
        planner.register_profile(QWEN4_FLASH_NEXT.key, profile)


def test_model_descriptors_separate_ready_contracts_from_pending_bridge():
    assert Capability.APC_V2 in QWEN4_FLASH_NEXT.capabilities
    assert QWEN4_FLASH_NEXT.cache_layout
    assert Capability.APC_V2 not in QWEN36.capabilities
    assert QWEN36.metadata["apc_v2_bridge"] == "pending"


def test_apcv2_descriptor_requires_cache_layout():
    with pytest.raises(ValueError, match="cache layout"):
        ModelDescriptor(
            model_type="broken",
            family="broken",
            variant="base",
            state_planes=frozenset({StatePlane.ATTENTION_KV}),
            capabilities=frozenset({Capability.APC_V2}),
        )


def test_state_transaction_is_revision_bound():
    original = StateManifest(StateIdentity("request-1", 2))
    transaction = RequestStateTransaction(original, StateOperation.APPEND)
    prepared = transaction.prepare({StatePlane.ATTENTION_KV: "next"})
    assert prepared.identity.revision == 3

    changed = StateManifest(StateIdentity("request-1", 3))
    with pytest.raises(StateConflict):
        transaction.commit(changed)


def test_only_compaction_may_publish_non_exact_state():
    original = StateManifest(StateIdentity("request-1", 0))
    transaction = RequestStateTransaction(original, StateOperation.ROLLBACK)
    with pytest.raises(InvalidStatePublication):
        transaction.prepare({}, fidelity=Fidelity.APPROXIMATE)

    unqualified_compact = RequestStateTransaction(original, StateOperation.COMPACT)
    with pytest.raises(InvalidStatePublication, match="qualified compaction"):
        unqualified_compact.prepare({}, fidelity=Fidelity.APPROXIMATE)

    compact_profile = QualifiedProfile(
        name="bounded-compaction",
        capabilities=frozenset({Capability.COMPACTION}),
        fidelity=Fidelity.NUMERICALLY_BOUNDED,
        evidence=("compaction-evidence.json",),
        implementation="compaction",
    )
    compact = RequestStateTransaction(
        original, StateOperation.COMPACT, profile=compact_profile
    )
    prepared = compact.prepare({}, fidelity=Fidelity.NUMERICALLY_BOUNDED)
    assert compact.commit(original) is prepared


def _fingerprint(*, fidelity=Fidelity.EXACT):
    return CacheFingerprint(
        schema_version="mlx2.cache.v1",
        plane=StatePlane.ATTENTION_KV,
        model_revision="model@abc",
        adapter_revision=None,
        configuration_hash="config",
        tokenizer_hash="tok",
        template_hash="template",
        layout="qwen4-exp-layer-segments-v1",
        token_hash="prompt",
        segment_start=0,
        segment_end=8,
        fidelity=fidelity,
    )


def test_cache_identity_rejects_missing_configuration_identity():
    with pytest.raises(ValueError, match="configuration_hash"):
        CacheFingerprint(
            schema_version="mlx2.cache.v1",
            plane=StatePlane.ATTENTION_KV,
            model_revision="model@abc",
            adapter_revision=None,
            configuration_hash="",
            tokenizer_hash="tok",
            template_hash="template",
            layout="layout",
            token_hash="prompt",
            segment_start=0,
            segment_end=8,
        )


def test_invalidation_waits_for_active_lease():
    owner = CacheOwner()
    key = _fingerprint()
    owner.put(key, "cached-state")
    lease = owner.acquire(key)
    assert lease.value == "cached-state"

    assert owner.invalidate(key)
    assert len(owner) == 0
    with pytest.raises(CacheMiss):
        owner.acquire(key)
    lease.close()
    assert len(owner) == 0


def test_cache_fidelity_is_part_of_identity():
    owner = CacheOwner()
    owner.put(_fingerprint(), "exact")
    with pytest.raises(CacheMiss):
        owner.acquire(_fingerprint(fidelity=Fidelity.APPROXIMATE))


def test_telemetry_is_bounded_but_counts_are_cumulative():
    telemetry = RuntimeTelemetry(capacity=2)
    telemetry.emit("route.selected", profile="a")
    telemetry.emit("route.selected", profile="b")
    telemetry.emit("cache.hit")
    assert [event.name for event in telemetry.snapshot()] == [
        "route.selected",
        "cache.hit",
    ]
    assert telemetry.counts()["route.selected"] == 2


def test_telemetry_bounds_distinct_counter_names():
    telemetry = RuntimeTelemetry(capacity=2)
    telemetry.emit("one")
    telemetry.emit("two")
    telemetry.emit("three")
    telemetry.emit("four")
    assert telemetry.counts() == {"one": 1, "two": 1, "__other__": 2}
