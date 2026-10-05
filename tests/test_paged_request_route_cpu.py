"""Host-only refusal and selection tests for paged request routing."""

import json
import threading
from dataclasses import replace

import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.paged_atomic_owner import PagedAtomicRequestOwner
from mlx2.runtime.paged_batch_lifecycle import probe_paged_batch_request
from mlx2.runtime.paged_pack_price import PACK_STEP_UNIT, evidence_sha256, load_price
from mlx2.runtime.paged_pack_scheduler import PrefillOffer, PrefillOption, ReservedRows
from mlx2.runtime.paged_request_route import RouteCapability, coordinate_paged_request
from mlx2.runtime.paged_request_transaction import STATE_PLANES, CandidateRequest

PROFILE_ID = "source:artifact:host:kernel:measured"
IDENTITY = {"host": "test-host", "hardware": "cpu", "artifact_sha256": "a" * 64,
            "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
            "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
            "kernel_sha256": "e" * 64}
CONTEXT = (128,)


@pytest.fixture
def measured_price(tmp_path):
    plain = {"request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "route_receipt": {"route": "ordinary"},
             "ordinary_model_forward_calls": 1, "output_token_id": 42}
    paged = {**plain, "route_receipt": {"route": "native_qwen3_paged",
                                           "selected": True, "observed_used": True},
             "ordinary_model_forward_calls": 0, "model_layers": 1,
             "paged_read_calls": 1, "terminal_successes": 1,
             "pending_native_epochs": 0, "retained_pages": 0}
    cases = []
    for reserved in ([], [['verify', 3]], [['verify', 2]],
                     [['verify', 2], ['verify', 3]]):
        for prefill_rows in (0, 2, 3, 4):
            if not reserved and not prefill_rows:
                continue
            step = {"scope": PACK_STEP_UNIT, "reserved": reserved,
                    "prefill_rows": prefill_rows, "context_tokens": list(CONTEXT),
                    "active_lane_count": len(reserved) + bool(prefill_rows),
                    "generator_step": True, "sampled_response": True,
                    "synchronized": True}
            cases.append({"reserved": reserved, "prefill_rows": prefill_rows,
                          "pack_step_ms": [1.0, 1.1, 1.2],
                          "ordinary_pack_step_ms": [1.0, 1.1, 1.2],
                          "complete_native_request_ms": [2.0, 2.1, 2.2],
                          "ordinary_request_ms": [2.0, 2.1, 2.2],
                          "request_proofs": [
                              {"ordinary": {**plain, "scheduler_step": step,
                                            "scheduler_step_ms": 1.0 + i / 10},
                               "paged": {**paged, "scheduler_step": step,
                                         "scheduler_step_ms": 1.0 + i / 10}}
                              for i in range(3)],
                          "kernel_engagement": {"paged_read_calls": 3,
                                                "terminal_successes": 3}})
    path = tmp_path / "price.json"
    data = {"schema": "mlx2.paged-pack-price.v1",
            "status": "measured", "measurement_scope": "complete_request",
            "price_unit": PACK_STEP_UNIT,
            "gpu_executed": True,
            "gpuq_owner": {"session": "test", "lease_id": "test-1", "pid": 1},
            "measured_route": "native_qwen3_paged",
            "identity": IDENTITY, "context_tokens": list(CONTEXT),
            "profile_id": PROFILE_ID, "cases": cases}
    data["complete_request_evidence_sha256"] = evidence_sha256(data)
    path.write_text(json.dumps(data))
    return load_price(path, live_identity=IDENTITY, context_tokens=CONTEXT)


def setup(*, planes=STATE_PLANES, enabled=True):
    owner = PagedAtomicRequestOwner("r1", {p: ["base"] for p in planes},
                                    supported_planes=planes, enabled=enabled)
    capability = RouteCapability(True, False, planes, PROFILE_ID)
    return owner, capability


def coordinate(owner, capability, request, run, *, price, reserved=None,
               offers=(), permit=True, cancelled=lambda: False, capacity=16):
    if reserved is None:
        reserved = (ReservedRows(request.lane_id, "verify", request.proposed_rows, 0, 100),)
    return coordinate_paged_request(
        request, owner, capability, reserved, offers, price=price,
        live_identity=IDENTITY, context_tokens=CONTEXT,
        row_capacity=capacity, free_pages=2, run_candidate=run,
        permit_candidate=permit, cancelled=cancelled,
    )


def runner(request):
    def run(candidate):
        for plane in request.planes:
            candidate.stage(plane, [f"{plane}:{i}" for i in range(request.proposed_rows)])
        return request.proposed_rows
    return run


def test_revision_refusal_releases_lease_backed_snapshot(measured_price):
    class LeaseView:
        revision = "old"
        closed = False

        def close(self):
            self.closed = True

    class LeaseOwner:
        supported_planes = ("kv",)
        atomic_publish = True

        def __init__(self):
            self.view = LeaseView()

        def snapshot(self):
            return self.view

    owner = LeaseOwner()
    request = CandidateRequest(1, "r1", 3, ("kv",))
    receipt = coordinate_paged_request(
        request, owner, RouteCapability(True, False, ("kv",), PROFILE_ID),
        (), (), price=measured_price, live_identity=IDENTITY,
        context_tokens=CONTEXT, row_capacity=16, free_pages=2,
        run_candidate=lambda _: pytest.fail("refused request ran"),
        permit_candidate=True)
    assert receipt.reason == "request_revision_drifted"
    assert owner.view.closed


def test_batch_lifecycle_probe_uses_live_uid_but_never_selects_serving(measured_price):
    generator = BatchGenerator(None)
    lock = threading.RLock()
    try:
        with lock:
            uid = generator.insert([[7] * 128], caches=[[]])[0]
        owner, capability = setup()
        request = CandidateRequest(uid, "r1", 3, STATE_PLANES)
        kwargs = {"price": measured_price, "live_identity": IDENTITY,
                      "context_tokens": CONTEXT, "row_capacity": 16, "free_pages": 2,
                      "run_host_candidate": runner(request)}
        reserved = ()
        offers = (PrefillOffer(uid, (PrefillOption(3, 0),)),)
        disabled = probe_paged_batch_request(generator, lock, request, owner,
                                             capability, reserved, offers, **kwargs)
        assert disabled.reason == "paged_host_probe_disabled"
        assert owner.snapshot().generation == 0
        result = probe_paged_batch_request(generator, lock, request, owner,
                                           capability, reserved, offers,
                                           permit_host_probe=True, **kwargs)
        assert result.generator_stage == "queued"
        assert result.host_probe.published and result.host_probe.observed_used
        assert not result.serving_selected and not result.serving_observed_used
        assert generator._find_uids((uid,))[uid][0] == 0
        with lock:
            generator.remove([uid])
        gone = probe_paged_batch_request(generator, lock, request, owner,
                                         capability, reserved, offers,
                                         permit_host_probe=True, **kwargs)
        assert gone.reason == "request_not_live" and gone.host_probe is None
    finally:
        generator.close()


def test_batch_lifecycle_removal_during_probe_rolls_back_host_state(measured_price):
    generator = BatchGenerator(None)
    lock = threading.RLock()
    try:
        uid = generator.insert([[7] * 128], caches=[[]])[0]
        owner, capability = setup()
        request = CandidateRequest(uid, "r1", 3, STATE_PLANES)
        before = owner.snapshot()

        def remove_after_staging(candidate):
            runner(request)(candidate)
            generator.remove([uid])
            return 3

        result = probe_paged_batch_request(
            generator, lock, request, owner, capability,
            (), (PrefillOffer(uid, (PrefillOption(3, 0),)),),
            price=measured_price, live_identity=IDENTITY,
            context_tokens=CONTEXT, row_capacity=16, free_pages=2,
            run_host_candidate=remove_after_staging, permit_host_probe=True)
        assert result.reason == "cancelled"
        assert result.host_probe.selected and not result.host_probe.published
        assert not result.serving_selected and owner.snapshot() == before
    finally:
        generator.close()


def test_batch_lifecycle_refuses_unbound_context_and_pack(measured_price):
    generator = BatchGenerator(None)
    lock = threading.RLock()
    try:
        uid = generator.insert([[7] * 127], caches=[[]])[0]
        owner, capability = setup()
        request = CandidateRequest(uid, "r1", 3, STATE_PLANES)
        called = []
        kwargs = {"price": measured_price, "live_identity": IDENTITY,
                      "context_tokens": CONTEXT, "row_capacity": 16, "free_pages": 2,
                      "run_host_candidate": lambda _: called.append(True),
                      "permit_host_probe": True}
        offer = (PrefillOffer(uid, (PrefillOption(3, 0),)),)
        result = probe_paged_batch_request(generator, lock, request, owner,
                                           capability, (), offer, **kwargs)
        assert result.reason == "live_context_mismatch"
        kwargs["context_tokens"] = (127,)
        result = probe_paged_batch_request(generator, lock, request, owner,
                                           capability,
                                           (ReservedRows(uid, "verify", 3, 0, 100),),
                                           (), **kwargs)
        assert result.reason == "queued_prefill_shape_mismatch"
        assert not called and owner.snapshot().generation == 0
    finally:
        generator.close()


@pytest.mark.parametrize("override,reason", [
    ({"permit": False}, "paged_route_disabled"),
    ({"price": None}, "source_bound_price_missing"),
    ({"capacity": 1}, "mandatory_decode_capacity"),
])
def test_refusals_do_not_run_candidate_or_change_reference(override, reason, measured_price):
    owner, capability = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    before = owner.snapshot()
    called = []
    options = {"price": measured_price, **override}
    receipt = coordinate(owner, capability, request, lambda _: called.append(1),
                         **options)
    assert receipt.reason == reason and not receipt.selected
    assert not receipt.observed_used and not receipt.published
    assert not called and owner.snapshot() == before


def test_missing_route_plane_revision_and_profile_refuse_before_callback(measured_price):
    owner, cap = setup(planes=("kv", "gdn"))
    request = CandidateRequest(1, "r1", 3, ("kv", "gdn"))
    called = []
    bad = lambda _: called.append(1)
    assert coordinate(owner, RouteCapability(False, False, cap.required_planes, cap.profile_id),
                      request, bad, price=measured_price).reason == "route_not_implemented"
    assert coordinate(owner, RouteCapability(True, False, ("kv",), cap.profile_id),
                      request, bad, price=measured_price).reason == "atomic_state_capability_missing"
    assert coordinate(owner, RouteCapability(True, False, cap.required_planes, "other"),
                      request, bad, price=measured_price).reason == "source_bound_price_missing"
    assert coordinate(owner, cap, CandidateRequest(1, "old", 3, request.planes),
                      bad, price=measured_price).reason == "request_revision_drifted"
    disabled, cap2 = setup(planes=request.planes, enabled=False)
    assert coordinate(disabled, cap2, request, bad, price=measured_price).reason == "atomic_state_capability_missing"
    assert not called and owner.snapshot().generation == 0


def test_unvalidated_price_and_unmeasured_shape_fail_before_callback(measured_price):
    owner, cap = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    called = []
    bad = replace(measured_price, _validation_token=object())
    assert coordinate(owner, cap, request, lambda _: called.append(1),
                      price=bad).reason == "source_bound_price_missing"
    unmeasured = (ReservedRows(1, "verify", 8, 0, 100),)
    eight = CandidateRequest(1, "r1", 8, STATE_PLANES)
    assert coordinate(owner, cap, eight, lambda _: called.append(1),
                      price=measured_price, reserved=unmeasured).reason == "pack_price_or_input_invalid"
    assert not called and owner.snapshot().generation == 0


def test_cancellation_before_and_after_candidate_preserves_public_state(measured_price):
    owner, cap = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    before = owner.snapshot()
    assert coordinate(owner, cap, request, runner(request), price=measured_price,
                      cancelled=lambda: True).reason == "cancelled"
    calls = iter((False, False, False, True))
    result = coordinate(owner, cap, request, runner(request), price=measured_price,
                        cancelled=lambda: next(calls))
    assert result.reason == "cancelled" and result.selected and not result.observed_used
    assert owner.snapshot() == before


@pytest.mark.parametrize("batch", [1, 2])
def test_ragged_b1_b2_selection_reserves_decode_then_prefill(batch, measured_price):
    owner, cap = setup()
    reserved = tuple(ReservedRows(i, "verify", i + 1, 0, 100) for i in range(1, batch + 1))
    offers = (PrefillOffer(9, (PrefillOption(2, 0), PrefillOption(4, 1))),)
    request = CandidateRequest(batch, "r1", batch + 1, STATE_PLANES)
    receipt = coordinate(owner, cap, request, runner(request), price=measured_price,
                         reserved=reserved, offers=offers)
    assert receipt.decision.reason == "mixed_pack"
    assert receipt.decision.prefill_lane_id == 9 and receipt.decision.prefill_rows == 4
    assert receipt.selected and receipt.observed_used and not receipt.qualified
    assert all(len(owner.snapshot().rows(p)) == batch + 2 for p in STATE_PLANES)


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_acceptance_is_one_atomic_boundary(accepted, measured_price):
    owner, cap = setup()
    request = CandidateRequest(1, "r1", 3, STATE_PLANES)
    ready = threading.Event()
    proceed = threading.Event()
    outcome = []
    observations = []

    def run(candidate):
        for plane in STATE_PLANES:
            candidate.stage(plane, [f"{plane}:{i}" for i in range(3)])
        ready.set()
        proceed.wait(timeout=2)
        return accepted

    worker = threading.Thread(target=lambda: outcome.append(coordinate(owner, cap, request, run,
                                                                       price=measured_price)))
    worker.start()
    assert ready.wait(timeout=2)
    assert all(len(owner.snapshot().rows(p)) == 1 for p in STATE_PLANES)
    def observe():
        for _ in range(200):
            snap = owner.snapshot()
            observations.append(tuple(len(snap.rows(p)) for p in STATE_PLANES))

    observer = threading.Thread(target=observe)
    observer.start()
    proceed.set()
    worker.join(timeout=2)
    observer.join(timeout=2)
    assert not worker.is_alive()
    assert observations and all(len(set(lengths)) == 1 for lengths in observations)
    receipt = outcome[0]
    assert receipt.implemented and not receipt.qualified and receipt.selected
    assert receipt.observed_used and receipt.published and receipt.accepted_rows == accepted
    assert all(len(owner.snapshot().rows(p)) == 1 + accepted for p in STATE_PLANES)
