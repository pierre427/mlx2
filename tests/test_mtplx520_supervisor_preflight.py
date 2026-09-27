"""Pure-CPU admission and failure gates for the external supervisor proposal."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest

from scripts.research.mtplx520_supervisor_preflight import (
    ModelSpec,
    Refused,
    SupervisorPreflight,
)


class FakeClock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeChild:
    def __init__(self, spec):
        self.spec = spec
        self.closed = False
        self.report = {
            "healthy": True,
            "model": spec.model_id,
            "artifact": spec.artifact,
            "runtime": dict(spec.runtime),
            "settings": dict(spec.settings),
            "route_receipt": spec.route_receipt,
            "qualification": "qualified",
            "qualified_capabilities": sorted(spec.capabilities),
            "selected_capabilities": sorted(spec.capabilities),
        }

    def status(self):
        return dict(self.report)

    def close(self):
        self.closed = True


def spec(name):
    return ModelSpec(
        model_id=name,
        artifact=f"artifact-{name}",
        runtime={"source_sha256": f"source-{name}"},
        settings={"route": "self_mtp", "mtp": True},
        route_receipt=f"receipt-{name}",
        resident_bytes=60,
        apc_dir=f"/private/apcv2/{name}",
        capabilities=frozenset({"text", "mtp", "apc_v2"}),
    )


@pytest.fixture
def rig():
    clock = FakeClock()
    children = []

    def spawn(item):
        child = FakeChild(item)
        children.append(child)
        return child

    supervisor = SupervisorPreflight(
        {name: spec(name) for name in ("a", "b")},
        budget_bytes=100,
        spawn=spawn,
        clock=clock,
        api_key="secret",
        idle_ttl_seconds=10,
        evict_to_fit=True,
    )
    return supervisor, clock, children


def admit(supervisor, name="a", **kwargs):
    return supervisor.admit(name, credential="secret", **kwargs)


def test_exact_model_no_alias_or_default_fallback(rig):
    supervisor, _, children = rig
    for invalid in ("unknown", "A", "b/../a", None):
        with pytest.raises(Refused, match="unknown_model"):
            admit(supervisor, invalid)
    assert children == []
    lease = admit(supervisor, "a")
    assert lease.receipt()["requested_model"] == "a"
    assert lease.receipt()["served_model"] == "a"
    assert lease.receipt()["child_route_receipt"] == "receipt-a"
    supervisor.validate_response(
        lease, response_model="a",
        child_receipt={
            "route_receipt": "receipt-a", "route": "self_mtp", "qualification": "qualified"
        },
    )
    assert supervisor.finish(lease)


@pytest.mark.parametrize(
    "model,receipt",
    [
        ("b", {"route_receipt": "receipt-a", "route": "self_mtp", "qualification": "qualified"}),
        ("a", {"route_receipt": "other", "route": "self_mtp", "qualification": "qualified"}),
        ("a", {"route_receipt": "receipt-a", "route": "ordinary", "qualification": "qualified"}),
        ("a", {"route_receipt": "receipt-a", "route": "self_mtp", "qualification": "unqualified"}),
    ],
)
def test_observed_child_receipt_must_match(model, receipt, rig):
    supervisor, _, _ = rig
    lease = admit(supervisor)
    with pytest.raises(Refused, match="child_receipt_mismatch"):
        supervisor.validate_response(lease, response_model=model, child_receipt=receipt)
    assert supervisor.finish(lease)


def test_auth_and_size_reject_before_child_load(rig):
    supervisor, _, children = rig
    reads = []

    def body(limit):
        reads.append(limit)
        return b"x" * min(5, limit)

    with pytest.raises(Refused, match="unauthorized"):
        supervisor.admit("a", credential="wrong", body_reader=body)
    assert reads == [] and children == []
    with pytest.raises(Refused, match="request_too_large"):
        admit(supervisor, body_reader=body, max_body_bytes=4)
    assert reads == [5] and children == []


def test_missing_capability_rejected_before_load(rig):
    supervisor, _, children = rig
    with pytest.raises(Refused, match="unsupported_capability"):
        admit(supervisor, required_capabilities=frozenset({"external_draft"}))
    assert children == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "b"),
        ("artifact", "foreign"),
        ("runtime", {"source_sha256": "foreign"}),
        ("settings", {"route": "ordinary", "mtp": False}),
        ("route_receipt", "foreign"),
        ("qualification", "unqualified"),
        ("healthy", False),
        ("qualified_capabilities", ["text", "apc_v2"]),
    ],
)
def test_child_identity_and_route_must_match(field, value):
    clock = FakeClock()
    children = []

    def spawn(item):
        child = FakeChild(item)
        child.report[field] = value
        children.append(child)
        return child

    supervisor = SupervisorPreflight(
        {"a": spec("a")}, 100, spawn, clock, "secret"
    )
    with pytest.raises(Refused, match="child_.*mismatch"):
        admit(supervisor)
    assert children[0].closed
    assert supervisor._children == {}


def test_apcv2_dirs_must_be_private():
    a, b = spec("a"), spec("b")
    b = ModelSpec(**{**vars(b), "apc_dir": a.apc_dir})
    with pytest.raises(ValueError, match="private APCv2"):
        SupervisorPreflight({"a": a, "b": b}, 100, FakeChild, FakeClock(), "key")


def test_budget_never_evicts_pinned_child(rig):
    supervisor, _, children = rig
    lease = admit(supervisor, "a")
    with pytest.raises(Refused, match="all_eviction_candidates_pinned"):
        admit(supervisor, "b")
    assert len(children) == 1 and not children[0].closed
    assert supervisor.finish(lease)
    other = admit(supervisor, "b")
    assert children[0].closed
    assert other.receipt()["served_model"] == "b"
    assert supervisor.finish(other)


def test_ttl_skips_pin_then_closes_idle(rig):
    supervisor, clock, children = rig
    lease = admit(supervisor)
    clock.now = 20
    assert supervisor.sweep_idle() == ()
    assert not children[0].closed
    assert supervisor.finish(lease)
    clock.now = 31
    assert supervisor.sweep_idle() == ("a",)
    assert children[0].closed


def test_drain_blocks_new_admission_and_waits_for_existing(rig):
    supervisor, _, children = rig
    lease = admit(supervisor)
    assert not supervisor.drain("a")
    with pytest.raises(Refused, match="child_draining_or_failed"):
        admit(supervisor)
    with pytest.raises(Refused, match="child_pinned"):
        supervisor.unload("a")
    assert supervisor.finish(lease)
    supervisor.unload("a")
    assert children[0].closed


def test_draining_pinned_child_still_counts_against_budget(rig):
    supervisor, _, children = rig
    lease = admit(supervisor, "a")
    assert not supervisor.drain("a")
    with pytest.raises(Refused, match="all_eviction_candidates_pinned"):
        admit(supervisor, "b")
    assert len(children) == 1 and not children[0].closed
    assert supervisor.finish(lease)


def test_crash_invalidates_old_stream_and_fences_new_child(rig):
    supervisor, clock, children = rig
    old = admit(supervisor)
    supervisor.crash("a")
    assert children[0].closed
    assert not supervisor.finish(old)
    with pytest.raises(Refused, match="stale_child_generation"):
        supervisor.validate_response(
            old, response_model="a",
            child_receipt={
                "route_receipt": "receipt-a", "route": "self_mtp",
                "qualification": "qualified",
            },
        )
    with pytest.raises(Refused, match="child_restart_backoff"):
        admit(supervisor)
    clock.now = 2
    new = admit(supervisor)
    assert new.generation == old.generation + 1
    assert new.child is not old.child
    assert not supervisor.finish(old)
    assert supervisor._children["a"].pins == 1
    assert supervisor.finish(new)


def test_repeated_crash_and_oom_fail_closed(rig):
    supervisor, clock, _ = rig
    for t in (0, 2, 5):
        clock.now = t
        lease = admit(supervisor)
        supervisor.crash("a")
        assert not supervisor.finish(lease)
    clock.now = 100
    with pytest.raises(Refused, match="child_restart_exhausted"):
        admit(supervisor)
    clock.now = 121
    lease = admit(supervisor)
    supervisor.crash("a", oom=True)
    assert not supervisor.finish(lease)
    with pytest.raises(Refused, match="child_oom_failed"):
        admit(supervisor)


def test_parallel_admission_seats_one_generation_and_pins_both(rig):
    supervisor, _, children = rig
    barrier = Barrier(2)

    def request():
        barrier.wait()
        return admit(supervisor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: request(), range(2)))
    assert first.generation == second.generation == 1
    assert first.child is second.child
    assert len(children) == 1
    assert supervisor._children["a"].pins == 2
    assert supervisor.finish(first)
    with pytest.raises(ValueError, match="already finished"):
        supervisor.finish(first)
    assert supervisor._children["a"].pins == 1
    assert supervisor.finish(second)


def test_drain_before_first_load_is_a_fence(rig):
    supervisor, _, children = rig
    assert supervisor.drain("a")
    with pytest.raises(Refused, match="child_draining_or_failed"):
        admit(supervisor)
    assert children == []
    supervisor.resume("a")
    lease = admit(supervisor)
    assert supervisor.finish(lease)


def test_drained_child_cannot_resume_until_unloaded(rig):
    supervisor, _, _ = rig
    lease = admit(supervisor)
    assert not supervisor.drain("a")
    with pytest.raises(Refused, match="child_must_unload_before_resume"):
        supervisor.resume("a")
    assert supervisor.finish(lease)
    supervisor.unload("a")
    supervisor.resume("a")
    newer = admit(supervisor)
    assert newer.generation > lease.generation
    assert supervisor.finish(newer)


def test_drain_racing_load_sees_pinned_child():
    clock = FakeClock()
    spawn_started = Event()
    allow_spawn = Event()

    def spawn(item):
        spawn_started.set()
        assert allow_spawn.wait(2)
        return FakeChild(item)

    supervisor = SupervisorPreflight(
        {"a": spec("a")}, 100, spawn, clock, "secret"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_admit = pool.submit(admit, supervisor)
        assert spawn_started.wait(2)
        future_drain = pool.submit(supervisor.drain, "a")
        allow_spawn.set()
        lease = future_admit.result(timeout=2)
        assert future_drain.result(timeout=2) is False
    with pytest.raises(Refused, match="child_draining_or_failed"):
        admit(supervisor)
    assert supervisor.finish(lease)
