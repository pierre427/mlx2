"""CPU policy/ownership tests; host payloads are synthetic, no model qualification."""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from mlx2.runtime.elastic_memory import (
    ElasticHeadroom,
    ElasticMemoryCoordinator,
    MemoryBudgetRefused,
    ReclaimableMemory,
)
from mlx2.runtime.expert_residency import (
    AdaptiveExpertResidency,
    AdaptiveResidencySettings,
)
from mlx2.runtime.weight_stream import (
    ExpertLRU,
    ExpertStreamManager,
    ExpertStreamPlan,
    StreamStats,
)


class Reader:
    def __init__(self, size=10, experts=16):
        self.expert_bytes = size
        self.num_experts = experts
        self.fail = False

    def check_sources(self):
        if self.fail:
            raise RuntimeError("source changed")

    def outstanding_bytes(self):
        return 0

    def iter_many(self, wanted):
        for expert in wanted:
            yield expert, bytes([expert]) * self.expert_bytes

    def materialize(self, value):
        return (value,)


def cache(capacity=2, settings=None, size=10, experts=16):
    return ExpertLRU(
        Reader(size, experts),
        capacity_experts=capacity,
        stats=StreamStats(),
        adaptive_residency=settings or {"enabled": True},
    )


def manager(*, adaptive=True, elastic=True, profile=None):
    settings = AdaptiveResidencySettings(
        enabled=adaptive, profile_path=str(profile) if profile else None
    )
    caches = {
        "a": cache(4, settings=settings, size=10),
        "b": cache(3, settings=settings, size=30, experts=8),
    }
    return ExpertStreamManager(
        plan=ExpertStreamPlan(
            top_k=1,
            ceiling_bytes=130,
            transient_bound_bytes=100,
            manifest_sha256="source-1",
        ),
        stats=StreamStats(),
        pool=None,
        caches=caches,
        adaptive_residency=settings,
        elastic=elastic,
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": 1},
        {"decay": 1},
        {"decay": 0},
        {"decay": math.nan},
        {"hysteresis": -1},
        {"max_swaps": 0},
        {"max_swaps": True},
        {"decay_every": 1.5},
        {"prompt_weight": math.inf},
        {"surprise": True},
        {"profile_path": "x"},
        {"profile_path": ""},
        [],
    ],
)
def test_bad_settings_refused(bad):
    with pytest.raises(ValueError):
        AdaptiveResidencySettings.parse(bad)


def test_hot_residents_survive_one_off_demand_and_topic_change_eventually_wins():
    c = cache()
    for _ in range(15):
        c.acquire([0, 1])
    assert c.acquire([7])[7] == (bytes([7]) * 10,)
    assert set(c._entries) == {0, 1}
    assert c._last_ephemeral == 1
    for _ in range(70):
        c.acquire([7, 8])
    assert set(c._entries) == {7, 8}
    assert c.adaptive.swaps == 2


def test_bounded_swaps_and_decay_are_deterministic():
    settings = AdaptiveResidencySettings(
        enabled=True, max_swaps=1, hysteresis=0, decay_every=2, decay=0.5
    )
    p = AdaptiveExpertResidency(8, settings)
    p.scores = [1, 1, 1, 10, 10, 10, 0, 0]
    assert len(p.retain([0, 1, 2], [3, 4, 5], 3) & {3, 4, 5}) == 1
    p.observe([0, 0])
    p.observe([1], prefill=True)
    assert p.scores[:2] == [1, 2.5]


def test_wide_gather_returns_every_byte_while_bank_is_bounded():
    c = cache()
    result = c.acquire(list(range(16)))
    assert len(result) == 16
    assert all(payload == (bytes([expert]) * 10,) for expert, payload in result.items())
    assert c.resident == 2
    assert c.stats.resident_bytes == 20
    c.note_gather(16)
    assert c.stats.as_dict()["stream_peak_transient_bytes"] <= 320


def test_failed_demand_does_not_publish_partially_read_bank():
    c = cache()
    c.acquire([0, 1])
    before = dict(c._entries)
    c.reader.fail = True
    with pytest.raises(RuntimeError, match="source changed"):
        c.acquire([2, 3])
    assert dict(c._entries) == before
    assert not c._busy


def test_profile_round_trip_rejects_other_source_and_invalid_scores(tmp_path):
    profile = tmp_path / "learned.json"
    a = manager(profile=profile)
    a.caches["a"].acquire([2, 3])
    a.save_residency_profile()
    b = manager(profile=profile)
    assert b.residency_profile_status == "loaded"
    assert b.caches["a"].adaptive.scores == a.caches["a"].adaptive.scores
    data = json.loads(profile.read_text())
    data["scores"]["a"][0] = -1
    profile.write_text(json.dumps(data))
    b = manager(profile=profile)
    assert b.residency_profile_status == "rejected"
    assert not any(b.caches["a"].adaptive.scores)
    data["binding"]["manifest_sha256"] = "other-artifact"
    data["scores"]["a"][0] = 0
    profile.write_text(json.dumps(data))
    assert manager(profile=profile).residency_profile_status == "rejected"


def test_atomic_profile_failure_preserves_previous_file(tmp_path, monkeypatch):
    profile = tmp_path / "learned.json"
    a = manager(profile=profile)
    a.save_residency_profile()
    before = profile.read_bytes()
    import os

    monkeypatch.setattr(
        os, "replace", lambda *_: (_ for _ in ()).throw(OSError("failed"))
    )
    with pytest.raises(OSError):
        a.save_residency_profile()
    assert profile.read_bytes() == before
    assert list(tmp_path.iterdir()) == [profile]


def test_heterogeneous_expert_pools_shrink_and_regrow_without_reading_or_touching_state():
    m = manager()
    for c in m.caches.values():
        c.acquire([0, 1, 2, 3])
    original = {name: dict(c._entries) for name, c in m.caches.items()}
    fixed_transient = m.plan.transient_bound_bytes
    # No free room beside the protected transient: only the expert floor survives.
    m.rebalance_headroom(available_bytes=0, protected_bytes=100)
    assert [c.capacity for c in m.caches.values()] == [1, 1]
    assert m.live_bytes() == 40
    assert m.reserved_bytes() == 140
    assert m.plan.transient_bound_bytes == fixed_transient
    for name, c in m.caches.items():
        assert all(
            value == original[name][expert] for expert, value in c._entries.items()
        )
    m.rebalance_headroom(available_bytes=1000, protected_bytes=0)
    assert [c.capacity for c in m.caches.values()] == [4, 3]
    assert m.live_bytes() == 40  # growth grants capacity, never speculative page-ins


def test_provider_cannot_resize_a_busy_bank():
    m = manager()
    m.caches["a"]._busy = True
    with pytest.raises(RuntimeError, match="acquired"):
        m.rebalance_headroom(available_bytes=0, protected_bytes=100)
    with pytest.raises(RuntimeError, match="resource failed"):
        m.reserved_bytes()


def test_generic_kv_and_workspace_claims_restore_on_error_and_protect_floors():
    events = []

    def provider(name):
        return ReclaimableMemory(10, 100, lambda n: events.append((name, n)) or n)

    pool = ElasticMemoryCoordinator(
        200,
        {"weights": provider("weights"), "scratch": provider("scratch")},
        synchronize=lambda: events.append("sync"),
    )
    pool.set_claim("kv", 80)
    assert sum(r.budget_bytes for r in pool.resources.values()) == 120
    with pytest.raises(RuntimeError, match="encode failed"), pool.borrow("vision", 60):
        assert sum(r.budget_bytes for r in pool.resources.values()) == 60
        with pytest.raises(MemoryBudgetRefused, match="floors"):
            pool.set_claim("another", 100)
        raise RuntimeError("encode failed")
    assert pool.claims == {"kv": 80}
    assert sum(r.budget_bytes for r in pool.resources.values()) == 120
    assert events[0] == "sync"
    pool.set_claim("kv", 0)
    assert sum(r.budget_bytes for r in pool.resources.values()) == 200


def test_resize_is_refused_during_execution_and_from_other_threads():
    pool = ElasticMemoryCoordinator(
        100, {"weights": ReclaimableMemory(10, 100, lambda n: n)}
    )
    with (
        pool.execution_window(),
        pytest.raises(MemoryBudgetRefused, match="execution window"),
    ):
        pool.set_claim("kv", 1)
    with (
        ThreadPoolExecutor(1) as executor,
        pytest.raises(MemoryBudgetRefused, match="another execution thread"),
    ):
        executor.submit(pool.set_claim, "kv", 1).result()


def test_failed_provider_poisoning_does_not_grant_loan():
    def fail(_):
        raise RuntimeError("backend lost")

    pool = ElasticMemoryCoordinator(100, {"weights": ReclaimableMemory(10, 100, fail)})
    with pytest.raises(RuntimeError, match="backend lost"):
        pool.set_claim("kv", 30)
    assert pool.claims == {}
    assert pool.receipt()["failed"]
    with pytest.raises(MemoryBudgetRefused, match="provider failed"):
        pool.set_claim("kv", 0)


@pytest.mark.parametrize("physical_release", [False, True])
def test_admission_uses_measured_reclamation_and_updated_reserve(physical_release):
    from mlx2.serving import ensure_admission_headroom

    state = {"budget": 100, "live": 100, "free": 25}

    def rebalance(**kwargs):
        assert kwargs == {"available_bytes": 25, "protected_bytes": 60}
        state["budget"] = 20
        if physical_release:
            state["live"] = 20
            state["free"] += 80

    headroom = ElasticHeadroom(
        lambda: state["free"],
        lambda: max(0, state["budget"] + 10 - state["live"]),
        rebalance,
    )
    admitted = ensure_admission_headroom(
        70, headroom=headroom, reclaim=lambda: None, evict=lambda: False
    )
    assert admitted is physical_release


def test_serving_policy_and_early_request_forward_both_mechanisms():
    from mlx2.serving import ServingEngine, moe_expert_streaming_policy

    policy = moe_expert_streaming_policy(
        {
            "enabled": True,
            "cache_gib": 2,
            "elastic": True,
            "adaptive_residency": {"enabled": True, "max_swaps": 3},
        }
    )
    engine = ServingEngine.__new__(ServingEngine)
    engine.moe_expert_streaming_policy = policy
    engine.dense_weight_streaming_policy = {"enabled": False}
    engine._preload_admission_limit_bytes = lambda: 10 << 30
    engine._atlas_collector = lambda: None
    request = engine._weight_stream_request()
    assert request.elastic
    assert request.adaptive_residency.max_swaps == 3
    assert "elastic" not in moe_expert_streaming_policy(None)
    for override in (
        {"elastic": 1},
        {"elastic": True, "enabled": False},
        {"elastic": True, "router_hint": "next_layer_reuse"},
        {"adaptive_residency": {"enabled": True}, "prefill_admission": "terminal_hot"},
    ):
        with pytest.raises(ValueError):
            moe_expert_streaming_policy({"enabled": True, "cache_gib": 1, **override})


def test_native_admission_rebalances_live_and_pending_kv_before_using_fresh_measurement():
    from mlx2.runtime.memory_policy import (
        SelfMTPLaneAdmissionController,
        _make_self_mtp_admission_callback,
    )

    state = {"free": 0.0}
    demands = []
    controller = SelfMTPLaneAdmissionController(
        transient_gib_per_lane=0.1, cache_estimator=lambda context: context * 1024
    )

    def rebalance(required):
        demands.append(required)
        state["free"] = required / (1 << 30) + 1

    admit = _make_self_mtp_admission_callback(
        controller,
        free_memory=lambda: state["free"],
        max_draft=2,
        rebalance_memory=rebalance,
    )
    rows = [(1, 100, 2, True, 0.25, 0.5)]
    result = admit(rows)
    assert result[1] == 2
    assert demands[0] >= math.ceil((0.5 + 0.1) * (1 << 30))
    state["free"] = 0
    admit([(1, 100, 2, True, 0.25, 1.0)])
    assert demands[-1] > demands[0]


def test_ordinary_execution_reserves_cache_growth_and_workspace_and_fails_without_physical_headroom():
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from mlx2.serving import Overloaded, prepare_elastic_execution

    demands = []
    state = {"free": 100 << 30}
    headroom = ElasticHeadroom(
        lambda: state["free"],
        lambda: 0,
        lambda **kwargs: demands.append(kwargs["protected_bytes"]),
    )
    controller = SelfMTPLaneAdmissionController(
        transient_gib_per_lane=0.1, cache_estimator=lambda context: context * (1 << 20)
    )
    prepare_elastic_execution(
        headroom, controller, [(100, 100 << 20, True)], workspace_gib=0.5
    )
    first = demands[-1]
    prepare_elastic_execution(
        headroom, controller, [(200, 200 << 20, False)], workspace_gib=1.0
    )
    assert demands[-1] > first
    state["free"] = 0
    with pytest.raises(Overloaded, match="measured expert reclamation"):
        prepare_elastic_execution(headroom, controller, [(200, 200 << 20, False)])


def test_adaptive_trace_is_not_misreported_as_an_exact_lru_baseline(tmp_path):
    from mlx2.runtime.expert_atlas import AtlasCollector, load_trace_sequence

    collector = AtlasCollector(trace_path=tmp_path / "access.trace")
    collector.bind(num_layers=1, num_units=8, prefill_admission="adaptive_residency")
    collector.observe(0, [1, 2])
    collector.close()
    *_, info = load_trace_sequence(tmp_path / "access.trace")
    assert info["baseline"] == "approximate"
    assert info["prefill_admission"] == "adaptive_residency"


def test_admission_grant_replaces_the_old_unfilled_stream_reservation():
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from mlx2.serving import admit_lane_headroom

    state = {"reserve": 2 << 30}
    controller = SelfMTPLaneAdmissionController(
        stream_reserve_gib=lambda: state["reserve"] / (1 << 30)
    )

    def rebalance(**_kwargs):
        state["reserve"] = 1 << 30

    headroom = ElasticHeadroom(lambda: 100 << 30, lambda: state["reserve"], rebalance)
    admitted, _, grant = admit_lane_headroom(
        controller,
        context_tokens=128,
        draft_depth=0,
        cache_gib=0,
        headroom=headroom,
        reclaim=lambda: None,
        evict=lambda: False,
    )
    assert admitted
    assert grant == controller.hard_reserve_gib + controller.lane_gib(128, 0)


def test_counterfactual_replay_matches_the_live_adaptive_bank():
    from mlx2.runtime import weight_stream as ws
    from mlx2.runtime.expert_residency_probe import replay_adaptive

    sequence = (
        [(0, [0, 1])] * 20 + [(0, [7]), (0, [2, 3, 4, 5])] * 3 + [(0, [7, 8])] * 65
    )
    c = cache()
    for index, (_, ids) in enumerate(sequence):
        if index < 20:
            with ws.expert_prefill_admission("intermediate", ("prompt",)):
                c.acquire(ids)
        else:
            c.acquire(ids)
    replay = replay_adaptive(
        sequence, capacities={0: 2}, num_experts=16, decode_start=20, terminal_start=19
    )
    assert replay["hits"] == c.stats.hits
    assert replay["misses"] == c.stats.misses
    assert replay["evictions"] == c.stats.evictions
    assert replay["swaps"] == c.adaptive.swaps


def test_nonregular_profile_is_rejected_without_opening_it(tmp_path):
    import os

    fifo = tmp_path / "profile.fifo"
    os.mkfifo(fifo)
    assert manager(profile=fifo).residency_profile_status == "rejected"


def test_repeated_rounded_budget_claim_does_not_resynchronize():
    events = []
    pool = ElasticMemoryCoordinator(
        100,
        {"weights": ReclaimableMemory(10, 100, lambda n: n // 10 * 10)},
        synchronize=lambda: events.append("sync"),
    )
    pool.set_claim("kv", 31)
    assert pool.resources["weights"].budget_bytes == 60
    pool.set_claim("kv", 31)
    assert len(events) == 1
    with pool.borrow("vision", 10), pytest.raises(MemoryBudgetRefused, match="scoped"):
        pool.set_claim("vision", 0)


def test_executor_stream_synchronizer_precedes_any_expert_release():
    events = []
    m = manager()
    m.bind_execution_synchronizer(lambda: events.append("executor-drained"))
    original = m.caches["a"]._trim_to_capacity

    def trim():
        assert events == ["executor-drained"]
        original()

    m.caches["a"]._trim_to_capacity = trim
    m.rebalance_headroom(available_bytes=0, protected_bytes=100)
    assert events == ["executor-drained"]


def test_close_releases_the_executor_owner_and_refuses_further_loans():
    import gc
    import weakref

    class Owner:
        def synchronize(self):
            pass

    owner = Owner()
    ref = weakref.ref(owner)
    m = manager()
    m.bind_execution_synchronizer(owner.synchronize)
    del owner
    assert ref() is not None
    m.close()
    gc.collect()
    assert ref() is None
    with pytest.raises(RuntimeError, match="not selected"):
        m.rebalance_headroom(available_bytes=1000, protected_bytes=0)
