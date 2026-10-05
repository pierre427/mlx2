"""Pure host checks for the default-off paged pack admission policy."""

import pytest

from mlx2.runtime.paged_pack_scheduler import (
    PrefillOffer, PrefillOption, ReservedRows, choose_paged_pack,
)


class Price:
    profile_id = "m5:artifact:source:kernel:measured-v1"

    def estimate_ms(self, reserved, prefill):
        rows = sum(item.rows for item in reserved) + (prefill[1].rows if prefill else 0)
        # A synthetic kernel regime jump: token count alone is not time.
        return 40.0 + rows * 0.5 + (220.0 if rows >= 154 else 0.0)


def _ready(deadline=200.0):
    return (ReservedRows(7, "decode", 1, 0, deadline),
            ReservedRows(8, "verify", 3, 1, deadline))


def _offers():
    return (PrefillOffer(11, (PrefillOption(64, 1), PrefillOption(128, 2),
                              PrefillOption(153, 3))),)


def test_default_off_and_missing_price_fail_closed():
    assert choose_paged_pack(_ready(), _offers(), price=Price(), row_capacity=256,
                             free_pages=8).reason == "paged_pack_disabled"
    assert choose_paged_pack(_ready(), _offers(), price=None, row_capacity=256,
                             free_pages=8, permit_candidate=True).reason == "source_bound_price_missing"


def test_reserves_decode_and_verify_before_pricing_prefill():
    decision = choose_paged_pack(_ready(), _offers(), price=Price(),
                                 row_capacity=256, free_pages=8, permit_candidate=True)
    assert decision.accepted and decision.reason == "mixed_pack"
    assert decision.prefill_lane_id == 11 and decision.prefill_rows == 128
    assert decision.reserved == _ready()
    assert decision.estimated_ms == 106.0


def test_deadline_or_pages_keep_decoders_moving():
    deadline = choose_paged_pack(_ready(60), _offers(), price=Price(),
                                 row_capacity=256, free_pages=8, permit_candidate=True)
    assert deadline.accepted and deadline.reason == "decode_only"
    pages = choose_paged_pack(_ready(), _offers(), price=Price(),
                              row_capacity=256, free_pages=1, permit_candidate=True)
    assert pages.accepted and pages.reason == "decode_only"
    impossible = choose_paged_pack(_ready(30), _offers(), price=Price(),
                                   row_capacity=256, free_pages=8, permit_candidate=True)
    assert not impossible.accepted and impossible.reason == "mandatory_decode_deadline"


def test_age_order_breaks_equal_progress_ties():
    offers = (PrefillOffer(12, (PrefillOption(64, 0),)),
              PrefillOffer(13, (PrefillOption(64, 0),)))
    decision = choose_paged_pack(_ready(), offers, price=Price(),
                                 row_capacity=128, free_pages=1, permit_candidate=True)
    assert decision.prefill_lane_id == 12


def test_malformed_inputs_and_unpriced_costs_refuse():
    with pytest.raises(ValueError, match="once"):
        choose_paged_pack(_ready(), (PrefillOffer(7, (PrefillOption(1, 0),)),),
                          price=Price(), row_capacity=8, free_pages=1, permit_candidate=True)
    with pytest.raises(ValueError, match="increase"):
        PrefillOffer(1, (PrefillOption(64, 0), PrefillOption(64, 0)))

    class BadPrice(Price):
        def estimate_ms(self, reserved, prefill):
            return float("nan")

    with pytest.raises(ValueError, match="finite"):
        choose_paged_pack(_ready(), _offers(), price=BadPrice(), row_capacity=256,
                          free_pages=8, permit_candidate=True)
