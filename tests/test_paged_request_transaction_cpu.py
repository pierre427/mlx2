"""Host-only acceptance and rollback tests for the paged request seam."""

import pytest

from mlx2.runtime.paged_pack_scheduler import (
    PrefillOffer,
    PrefillOption,
    ReservedRows,
    choose_paged_pack,
)
from mlx2.runtime.paged_request_transaction import (
    CandidateRequest,
    execute_paged_request,
)


class Price:
    profile_id = "source:artifact:host:kernel:measured"

    def estimate_ms(self, reserved, prefill):
        rows = sum(row.rows for row in reserved)
        return 10.0 + rows + (prefill[1].rows * 2 if prefill else 0)


class Prepared:
    def __init__(self, owner, request, accepted):
        self.owner = owner
        self.planes = request.planes
        self.revision = request.revision
        self.accepted = accepted

    def publish(self):
        if self.owner.fail_publish:
            raise RuntimeError("publish refused before pointer swap")
        # One pointer swap publishes the entire multi-plane snapshot.
        next_state = dict(self.owner.public)
        for plane in self.planes:
            next_state[plane] = next_state[plane] + self.accepted
        self.owner.public = next_state
        self.owner.publish_count += 1

    def rollback(self):
        self.owner.rollbacks += 1


class Candidate:
    def __init__(self, owner, request):
        self.owner = owner
        self.request = request

    def prepare(self, accepted_rows):
        self.owner.prepared.append(accepted_rows)
        return Prepared(self.owner, self.request, accepted_rows)

    def rollback(self):
        self.owner.rollbacks += 1


class Owner:
    supported_planes = ("kv", "gdn", "qsa", "mtp")
    atomic_publish = True

    def __init__(self):
        self.public = {plane: 9 for plane in self.supported_planes}
        self.begins = 0
        self.prepared = []
        self.publish_count = 0
        self.rollbacks = 0
        self.fail_publish = False

    def begin(self, request):
        self.begins += 1
        return Candidate(self, request)


def pack(*, deadline=100.0, free_pages=4, permit=True):
    return choose_paged_pack(
        (ReservedRows(1, "decode", 1, 0, deadline),
         ReservedRows(2, "verify", 3, 1, deadline)),
        (PrefillOffer(3, (PrefillOption(8, 1), PrefillOption(16, 2))),),
        price=Price(), row_capacity=24, free_pages=free_pages,
        permit_candidate=permit,
    )


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_verify_publishes_exact_accepted_prefix_across_all_planes(accepted):
    owner = Owner()
    request = CandidateRequest(2, "revision:42", 3, owner.supported_planes)
    receipt = execute_paged_request(pack(), request, owner, lambda _: accepted,
                                    permit_candidate=True)
    assert receipt.reason == "accepted" and receipt.published
    assert receipt.accepted_rows == accepted
    assert receipt.profile_id == Price.profile_id
    assert owner.public == {plane: 9 + accepted for plane in owner.supported_planes}
    assert owner.prepared == [accepted] and owner.publish_count == 1


def test_prefill_offer_respects_decode_deadline_and_age_order():
    chosen = pack(deadline=45)
    assert chosen.prefill_lane_id == 3 and chosen.prefill_rows == 8
    too_tight = pack(deadline=20)
    assert too_tight.reason == "decode_only"
    refused = pack(deadline=10)
    assert not refused.accepted and refused.reason == "mandatory_decode_deadline"
    offers = (PrefillOffer(4, (PrefillOption(8, 0),)),
              PrefillOffer(5, (PrefillOption(8, 0), PrefillOption(12, 0))))
    fair = choose_paged_pack((ReservedRows(1, "decode", 1, 0, 100),),
                             offers, price=Price(), row_capacity=16,
                             free_pages=0, permit_candidate=True)
    assert fair.prefill_lane_id == 4


def test_default_off_unselected_and_unsupported_refuse_before_mutation():
    owner = Owner()
    request = CandidateRequest(3, "revision:42", 16, ("kv",))
    assert execute_paged_request(pack(), request, owner, lambda _: 8).reason == "paged_request_disabled"
    assert execute_paged_request(pack(permit=False), request, owner, lambda _: 8,
                                 permit_candidate=True).reason == "paged_pack_disabled"
    assert execute_paged_request(pack(deadline=20), request, owner, lambda _: 8,
                                 permit_candidate=True).reason == "request_not_selected"
    owner.atomic_publish = False
    assert execute_paged_request(pack(), request, owner, lambda _: 8,
                                 permit_candidate=True).reason == "atomic_state_capability_missing"
    owner.atomic_publish = True
    owner.supported_planes = ("kv",)
    request = CandidateRequest(2, "revision:42", 3, ("kv", "gdn"))
    assert execute_paged_request(pack(), request, owner, lambda _: 3,
                                 permit_candidate=True).reason == "atomic_state_capability_missing"
    assert owner.begins == 0 and owner.public == {plane: 9 for plane in Owner.supported_planes}


def test_cancellation_before_and_after_candidate_rolls_back():
    owner = Owner()
    request = CandidateRequest(2, "revision:42", 3, ("kv", "gdn"))
    assert execute_paged_request(pack(), request, owner, lambda _: 3,
                                 permit_candidate=True, cancelled=lambda: True).reason == "cancelled"
    assert owner.begins == 0
    calls = iter((False, True))
    receipt = execute_paged_request(pack(), request, owner, lambda _: 3,
                                    permit_candidate=True,
                                    cancelled=lambda: next(calls))
    assert receipt.reason == "cancelled" and not receipt.published
    assert owner.rollbacks == 1 and owner.publish_count == 0
    assert owner.public["kv"] == owner.public["gdn"] == 9

    calls = iter((False, False, True))
    receipt = execute_paged_request(pack(), request, owner, lambda _: 2,
                                    permit_candidate=True,
                                    cancelled=lambda: next(calls))
    assert receipt.reason == "cancelled" and not receipt.published
    assert owner.prepared == [2] and owner.rollbacks == 2
    assert owner.publish_count == 0


def test_exception_and_invalid_acceptance_discard_private_state():
    owner = Owner()
    request = CandidateRequest(2, "revision:42", 3, ("kv", "gdn"))

    def fail(_):
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        execute_paged_request(pack(), request, owner, fail, permit_candidate=True)
    with pytest.raises(ValueError, match="integer prefix"):
        execute_paged_request(pack(), request, owner, lambda _: 4,
                              permit_candidate=True)
    owner.fail_publish = True
    with pytest.raises(RuntimeError, match="pointer swap"):
        execute_paged_request(pack(), request, owner, lambda _: 2,
                              permit_candidate=True)
    assert owner.rollbacks == 3 and owner.publish_count == 0
    assert owner.public["kv"] == owner.public["gdn"] == 9


def test_prepared_revision_mismatch_fails_closed():
    owner = Owner()
    request = CandidateRequest(2, "revision:42", 3, ("kv", "qsa", "mtp"))

    class WrongCandidate(Candidate):
        def prepare(self, accepted_rows):
            result = super().prepare(accepted_rows)
            result.revision = "revision:43"
            return result

    owner.begin = lambda request: WrongCandidate(owner, request)
    with pytest.raises(ValueError, match="revision"):
        execute_paged_request(pack(), request, owner, lambda _: 2,
                              permit_candidate=True)
    assert owner.rollbacks == 1 and owner.publish_count == 0
