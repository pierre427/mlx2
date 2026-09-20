# SPDX-License-Identifier: MIT
"""Host reserve keyed on the Metal advisory (CPU only; no model, no Metal).

3fa14e7 replaced a flat 20 GiB service/driver reserve with 12.5% / 3.125% of
*physical* RAM, and argued that Metal's ``max_recommended_working_set_size``
must not enter the rule because ``available_execution_bytes`` already applies
it once, so keying the reserve off it too would subtract the OS margin twice.

The premise is right; the conclusion is backwards.  ``host - advisory`` is the
OS margin macOS has already withheld, and it is not a constant fraction:

    128 GiB -> 112.00 GiB advisory   16.00 GiB withheld  12.5%
     36 GiB ->  28.08 GiB advisory    7.92 GiB withheld  22.0%

(measured 2026-09-19).  A reserve of 12.5% of RAM is a *second* OS margin
sized as though macOS always withheld 12.5%.  That is exactly true on the
calibration host and badly wrong on the M3 Pro, where macOS has already taken
22% and admission takes another 12.5% -- charged against the advisory residue,
the term that actually binds once a model is resident.  Muse-Glimmer-30B +
DFlash2 at 21.56 GiB resident leaves 6.14 GiB of residue (measured live), and
the 5.625 GiB reserve took all but 0.52 GiB of it, so serving 429'd on a
two-thirds-empty machine.

The rule here charges the host's non-lane quota (25% of RAM, read off the
calibration host) once: the service reserve supplies only the part the
advisory has not already collected.
"""

import math

import pytest

from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

CAL_HOST_GIB = 128.0
CAL_ADVISORY_GIB = 112.0

M3_HOST_GIB = 36.0
M3_ADVISORY_GIB = 28.08
# Measured on the M3 with Muse-Glimmer-30B-4bit + the DFlash2 external draft
# resident: 21.56 GiB active, and ``execution_headroom`` logged at 6.14 GiB
# steady state across a live probe sequence (range 5.96-6.22).
M3_MUSE_RESIDENT_GIB = 21.56
M3_MUSE_FREE_GIB = 6.14
# The older M3 case from 33d1d5a: a 20 GiB MTP model, 8.336 GiB headroom.
M3_MTP_FREE_GIB = 8.336
# What 3fa14e7's physical-RAM rule produced on that host.
PHYSICAL_RULE_RESERVE_GIB = 5.625
# What this rule produces there: the advisory has collected all but 1.08 GiB
# of the 9.0 GiB quota, so the service floor takes over; the driver term is
# advisory-scaled and does not floor.
M3_SERVICE_GIB = 3.0
M3_DRIVER_GIB = 28.08 * 4.0 / 112.0  # 1.00286
M3_RESERVE_GIB = M3_SERVICE_GIB + M3_DRIVER_GIB  # 4.00286


# --------------------------------------------------------------------------
# HARD REQUIREMENT: the 128 GiB calibration host does not move.
# --------------------------------------------------------------------------


def test_128_gib_host_is_bit_for_bit_todays_calibration():
    """Exact identity, asserted with ``==``, not approx."""
    (service, driver) = C.host_scaled_reserves(CAL_HOST_GIB, CAL_ADVISORY_GIB)
    assert service == 16.0
    assert driver == 4.0
    controller = C(host_memory_gib=CAL_HOST_GIB, advisory_gib=CAL_ADVISORY_GIB)
    assert controller.service_reserve_gib == 16.0
    assert controller.driver_allowance_gib == 4.0
    assert controller.hard_reserve_gib == 20.0
    # The quota constant is read off this host and reproduces it by
    # construction: 16 withheld by Metal + 16 reserved = 32 = 25% of 128.
    assert C.NON_LANE_HOST_QUOTA_FRACTION == 0.25
    assert (
        C.NON_LANE_HOST_QUOTA_FRACTION * CAL_HOST_GIB
        - (CAL_HOST_GIB - CAL_ADVISORY_GIB)
        == 16.0
    )


def test_128_gib_admitted_lane_counts_are_unchanged():
    """The models calibrated on that host admit exactly what they did.

    Flash-Next at its 72.5 GiB PLE operating point and a dense Qwen3.8-27B at
    ~18 GiB, over the contexts the class docstring pins the envelope to.  The
    reserve is identical, so every downstream number is identical; this walks
    the admission path rather than trusting that.
    """
    flash_free = 128.0 - 72.5
    dense_free = 128.0 - 18.0
    for (transient, free) in ((C.K2_TRANSIENT_GIB_PER_LANE, flash_free),
                              (3.1, dense_free)):
        keyed = C(
            host_memory_gib=CAL_HOST_GIB,
            advisory_gib=CAL_ADVISORY_GIB,
            transient_gib_per_lane=transient,
        )
        # The controller as it was before this change: a flat 16 + 4.
        flat = C(
            host_memory_gib=CAL_HOST_GIB,
            service_reserve_gib=16.0,
            driver_allowance_gib=4.0,
            transient_gib_per_lane=transient,
        )
        assert keyed.hard_reserve_gib == flat.hard_reserve_gib == 20.0
        for context in (1024, 4096, 16384, 65536):
            for lanes in (1, 4, 16, 32):
                a = keyed.decide([context] * lanes, free, max_draft=2)
                b = flat.decide([context] * lanes, free, max_draft=2)
                assert a.mtp_indices == b.mtp_indices
                assert a.draft_depths == b.draft_depths
                assert a.stage == b.stage
                assert a.usable_gib == b.usable_gib
        # The docstring's calibration points, unchanged.
        if transient == C.K2_TRANSIENT_GIB_PER_LANE:
            assert len(keyed.decide([1024] * 24, flash_free).mtp_indices) == 16
            assert len(keyed.decide([16384] * 24, flash_free).mtp_indices) == 4


def test_a_bigger_host_is_capped_not_extrapolated():
    # A 512 GiB Mac Studio does not need an 80 GiB reserve, and its advisory
    # residue is enormous, so both terms sit on their caps.
    assert C.host_scaled_reserves(512.0, 460.0) == (16.0, 4.0)
    assert C.host_scaled_reserves(192.0, 172.0) == (16.0, 4.0)


# --------------------------------------------------------------------------
# Why the physical-RAM basis fails, and what replaces it.
# --------------------------------------------------------------------------


def test_the_withheld_share_is_not_a_constant_fraction_of_the_host():
    """The measurement that contradicts 3fa14e7's argument."""
    cal_withheld = (CAL_HOST_GIB - CAL_ADVISORY_GIB) / CAL_HOST_GIB
    m3_withheld = (M3_HOST_GIB - M3_ADVISORY_GIB) / M3_HOST_GIB
    assert cal_withheld == pytest.approx(0.125)
    assert m3_withheld == pytest.approx(0.22)
    # A 12.5%-of-RAM reserve is a second copy of the first number. On the
    # calibration host the copy is exact; on the M3 it over-charges by 76%.
    assert m3_withheld / cal_withheld == pytest.approx(1.76, abs=0.01)


def test_the_physical_rule_is_this_rule_at_the_calibration_ratio():
    """3fa14e7 is not a different policy: it hardcodes a 0.875 advisory."""
    assert C.ADVISORY_RATIO == 0.875
    for host in (16.0, 24.0, 36.0, 64.0, 128.0):
        (service, driver) = C.host_scaled_reserves(host, C.ADVISORY_RATIO * host)
        assert service == pytest.approx(max(3.0, min(16.0, 0.125 * host)))
        assert driver == pytest.approx(max(0.75, min(4.0, 0.03125 * host)))
    # ...which is also what an unmeasurable advisory falls back to, so a
    # failed probe degrades to the previously shipped rule, not to a guess.
    assert C.host_scaled_reserves(M3_HOST_GIB, None) == pytest.approx(
        (4.5, 1.125)
    )


def test_36_gib_host_charges_the_os_margin_once():
    (service, driver) = C.host_scaled_reserves(M3_HOST_GIB, M3_ADVISORY_GIB)
    # 25% of 36 is 9.0; macOS has already withheld 7.92, leaving a 1.08 GiB
    # shortfall, which is under the floor -- so the floor, not the quota,
    # sets the service term on this host.
    quota = C.NON_LANE_HOST_QUOTA_FRACTION * M3_HOST_GIB
    shortfall = quota - (M3_HOST_GIB - M3_ADVISORY_GIB)
    assert quota == pytest.approx(9.0)
    assert shortfall == pytest.approx(1.08)
    assert shortfall < C.MIN_SERVICE_RESERVE_GIB
    assert service == C.MIN_SERVICE_RESERVE_GIB == 3.0
    # The driver term is advisory-scaled and clears its floor on its own.
    assert driver == pytest.approx(M3_DRIVER_GIB)
    assert driver > C.MIN_DRIVER_ALLOWANCE_GIB
    assert service + driver == pytest.approx(M3_RESERVE_GIB, abs=1e-6)
    # Not merely the floors: the driver term is 34% above its own.
    assert service + driver > C.MIN_SERVICE_RESERVE_GIB + C.MIN_DRIVER_ALLOWANCE_GIB
    # ...and strictly less than the physical rule it replaces.
    assert service + driver < PHYSICAL_RULE_RESERVE_GIB


def test_36_gib_admits_the_muse_plus_dflash2_pair_that_429d():
    """The serving failure, as arithmetic.

    Muse-Glimmer-30B-4bit plus the DFlash2 external draft: 21.56 GiB resident
    against a 28.08 GiB advisory.  ``execution_headroom`` on that host was
    measured at 6.14 GiB steady-state during a live external-draft run
    (logged every 3 s over a full probe sequence; it moved between 5.96 and
    6.22 GiB).  The external-draft route subtracts the reserve from that
    reading directly and hands the remainder to the batch, so the reserve is
    the whole difference between serving and not.
    """
    old_usable = M3_MUSE_FREE_GIB - PHYSICAL_RULE_RESERVE_GIB
    new_usable = M3_MUSE_FREE_GIB - M3_RESERVE_GIB
    assert old_usable == pytest.approx(0.515, abs=1e-3)
    assert new_usable == pytest.approx(2.137, abs=1e-3)
    # 4.1x the lane budget, from a reserve that is only 29% smaller: the
    # reserve is subtracted from a residue a resident model has already
    # driven most of the way to zero, which is why over-charging it there
    # costs so much more than the same over-charge costs at 128 GiB.
    assert new_usable / old_usable > 4.0

    contexts = [34]  # the smoke request: 18 prompt + 16 max_tokens
    kwargs = dict(saturation_lane_cap=4)
    old = C(host_memory_gib=M3_HOST_GIB, **kwargs)
    new = C(host_memory_gib=M3_HOST_GIB, advisory_gib=M3_ADVISORY_GIB, **kwargs)
    assert old.hard_reserve_gib == pytest.approx(PHYSICAL_RULE_RESERVE_GIB)
    assert new.hard_reserve_gib == pytest.approx(M3_RESERVE_GIB, abs=1e-6)

    # The external-draft route does not go through ``decide``: it subtracts
    # the reserve from ``execution_headroom`` and hands the remainder to the
    # batch as its whole budget (serving.py, ``create_external_batch``'s
    # ``memory_headroom``).  So on that route the reserve *is* the budget.
    def external_budget(controller, raw_gib=M3_MUSE_FREE_GIB):
        return max(0.0, raw_gib - controller.hard_reserve_gib)

    assert external_budget(old) == pytest.approx(0.515, abs=1e-3)
    assert external_budget(new) == pytest.approx(2.137, abs=1e-3)

    # The self-MTP ladder on the same host and headroom, for the route that
    # does use ``decide``.  The pair resolves to the 1.76 Flash-Next
    # transient (no adapter cache budget), so at the policy's num_draft=4 the
    # transient alone is 2.73 GiB.  The physical rule cannot seat the lane at
    # any speculative depth at all -- even k=1 costs 1.42 GiB against its
    # 0.515 GiB budget -- so the whole cohort queues.  This rule seats k=2.
    assert old.decide(contexts, M3_MUSE_FREE_GIB, max_draft=4).modes == ("queue",)
    seated = new.decide(contexts, M3_MUSE_FREE_GIB, max_draft=4)
    assert seated.modes == ("self_mtp",)
    assert seated.draft_depths == (2,)
    assert seated.estimated_gib <= seated.usable_gib


def test_what_the_36_gib_reserve_still_protects():
    """The reserve shrank; say what is left and against what.

    Two things are being protected and they are protected by different
    terms.  The rest of the machine is protected by the 9.0 GiB quota, of
    which macOS itself enforces 7.92 GiB through the advisory -- that part is
    a hard per-process ceiling and does not depend on this policy being
    right.  What admission still holds back, 4.0 GiB, protects *us* from our
    own estimator: the advisory is a recommendation, so overshooting it
    swaps rather than fails, and ``lane_gib`` is an envelope.
    """
    (service, driver) = C.host_scaled_reserves(M3_HOST_GIB, M3_ADVISORY_GIB)
    withheld_by_metal = M3_HOST_GIB - M3_ADVISORY_GIB
    total = withheld_by_metal + service + driver
    assert total == pytest.approx(11.92, abs=0.01)
    assert total / M3_HOST_GIB == pytest.approx(0.331, abs=0.001)
    # Still more of the host held back, in absolute and relative terms, than
    # the calibration host holds back of itself.
    assert total / M3_HOST_GIB > (
        (CAL_HOST_GIB - CAL_ADVISORY_GIB) + 16.0 + 4.0
    ) / CAL_HOST_GIB
    # The margin the reserve itself supplies is many times the largest
    # unmodelled excursion measured on this class of model: the worst
    # peak-minus-steady cell in provenance/lane-transient-moe.json is
    # 0.281 GiB/lane on the cold-allocator forward.
    assert service + driver > 10 * 0.281
    # It is not a rounding error against the budget it guards, either.
    usable = M3_MUSE_FREE_GIB - (service + driver)
    assert 0.9 < (service + driver) / usable < 2.5


# --------------------------------------------------------------------------
# Fallbacks, floors, overrides, plumbing.
# --------------------------------------------------------------------------


def test_unknown_readings_default_to_the_128_gib_behaviour():
    default = C()
    assert (default.service_reserve_gib, default.driver_allowance_gib) == (16.0, 4.0)
    assert default.hard_reserve_gib == 20.0
    for unknown in (None, 0.0, -8.0, float("nan"), float("inf")):
        assert C.host_scaled_reserves(unknown, unknown) == (16.0, 4.0)
        # One reading is enough; the other is imputed at the calibration
        # ratio, so a half-failed probe still adapts to the host.
        assert C.host_scaled_reserves(M3_HOST_GIB, unknown) == pytest.approx(
            (4.5, 1.125)
        )
        assert C.host_scaled_reserves(unknown, M3_ADVISORY_GIB) == pytest.approx(
            (28.08 / 0.875 * 0.125, M3_DRIVER_GIB)
        )


def test_an_advisory_above_physical_ram_is_not_a_bonus():
    """A nonsense pairing must not inflate the service term."""
    assert C.host_scaled_reserves(36.0, 64.0) == C.host_scaled_reserves(36.0, 36.0)
    (service, _) = C.host_scaled_reserves(36.0, 36.0)
    assert service == pytest.approx(9.0)  # nothing withheld -> full quota


def test_floors_hold_on_an_absurdly_small_host():
    for (host, advisory) in ((8.0, 6.0), (4.0, 3.0), (1.0, 0.7), (0.25, 0.2)):
        (service, driver) = C.host_scaled_reserves(host, advisory)
        assert service == C.MIN_SERVICE_RESERVE_GIB == 3.0
        assert driver == C.MIN_DRIVER_ALLOWANCE_GIB == 0.75
        assert C(
            host_memory_gib=host, advisory_gib=advisory
        ).hard_reserve_gib == pytest.approx(3.75)


def test_the_rule_is_monotone_along_a_fixed_advisory_ratio():
    for ratio in (0.78, 0.875, 0.9):
        sizes = [0.5, 8.0, 16.0, 24.0, 36.0, 64.0, 96.0, 128.0, 256.0]
        reserves = [sum(C.host_scaled_reserves(g, ratio * g)) for g in sizes]
        assert reserves == sorted(reserves)


def test_a_more_generous_advisory_means_a_larger_reserve():
    """Monotone in the advisory at fixed RAM, which is the whole point."""
    at = [sum(C.host_scaled_reserves(36.0, a)) for a in (26.0, 28.08, 30.0, 33.0)]
    assert at == sorted(at)


def test_explicit_overrides_above_the_host_floor_still_work():
    strict = C(host_memory_gib=M3_HOST_GIB, advisory_gib=M3_ADVISORY_GIB,
               service_reserve_gib=9.0, driver_allowance_gib=2.0)
    assert strict.hard_reserve_gib == 11.0
    # The old 16/4 pair remains a legal override on a small host.
    legacy = C(host_memory_gib=M3_HOST_GIB, advisory_gib=M3_ADVISORY_GIB,
               service_reserve_gib=16.0, driver_allowance_gib=4.0)
    assert legacy.hard_reserve_gib == 20.0
    # Exactly at the floor is allowed.
    assert C(host_memory_gib=M3_HOST_GIB, advisory_gib=M3_ADVISORY_GIB,
             service_reserve_gib=M3_SERVICE_GIB,
             driver_allowance_gib=M3_DRIVER_GIB).hard_reserve_gib == pytest.approx(
        M3_RESERVE_GIB
    )


def test_constructor_still_rejects_nonsense():
    with pytest.raises(ValueError):
        C(service_reserve_gib=0.0)
    with pytest.raises(ValueError):
        C(service_reserve_gib=-1.0)
    with pytest.raises(ValueError):
        C(service_reserve_gib=float("nan"))
    with pytest.raises(ValueError):
        C(service_reserve_gib=float("inf"))
    with pytest.raises(ValueError):
        C(driver_allowance_gib=-1.0)
    with pytest.raises(ValueError):
        C(driver_allowance_gib=float("nan"))
    # Below the host floor, on a big host and on a small one.
    with pytest.raises(ValueError, match="host-scaled floor"):
        C(host_memory_gib=128.0, advisory_gib=112.0, service_reserve_gib=15.9)
    with pytest.raises(ValueError, match="host-scaled floor"):
        C(host_memory_gib=M3_HOST_GIB, advisory_gib=M3_ADVISORY_GIB,
          service_reserve_gib=2.99)
    with pytest.raises(ValueError, match="host-scaled floor"):
        C(host_memory_gib=8.0, advisory_gib=6.0, service_reserve_gib=2.9)
    with pytest.raises(ValueError, match="host-scaled floor"):
        C(host_memory_gib=8.0, advisory_gib=6.0, driver_allowance_gib=0.5)


def test_reserve_never_swallows_the_host_whole():
    for (host, advisory) in ((4.0, 3.0), (8.0, 6.2), (16.0, 12.5), (24.0, 19.0),
                             (36.0, 28.08), (64.0, 54.0), (128.0, 112.0)):
        reserve = sum(C.host_scaled_reserves(host, advisory))
        assert 0.0 < reserve < host
        # Anything that could host a servable model keeps a usable share of
        # its advisory. Below that the floors deliberately exceed the whole
        # advisory and the controller fails closed, which is correct: a
        # 4 GiB Mac has no business seating a lane.
        if host >= 16.0:
            assert reserve < 0.35 * advisory


def test_parallel_sample_guard_uses_the_same_reserve(monkeypatch):
    """serving.py carried a second, independent hardcoded 20 GiB."""
    from collections import Counter

    from mlx2 import memory, serving

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.max_lanes = 4
    engine.queued_jobs = 0
    engine.counts = Counter()
    engine.batch_metrics = type("M", (), {"rejected": lambda self, *a: None})()
    monkeypatch.setattr(serving.time, "sleep", lambda _s: None)

    engine._hard_reserve_gib = None
    monkeypatch.setattr(memory, "host_memory_gib", lambda: M3_HOST_GIB)
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: M3_ADVISORY_GIB)
    assert engine.hard_reserve_gib == pytest.approx(M3_RESERVE_GIB, abs=1e-6)
    monkeypatch.setattr(
        memory, "execution_headroom", lambda: int(M3_MTP_FREE_GIB * (1 << 30))
    )
    receipt = engine.admit_parallel_samples(2)
    # 4.0029 + 2*2 = 8.0029 GiB, which the M3's 8.336 GiB of headroom now
    # satisfies -- 3fa14e7's 5.625 + 4 = 9.625 did not, and the original
    # 20 + 4 = 24 could never be met on that host.
    assert receipt["required_headroom_bytes"] == int(
        (M3_RESERVE_GIB + 4) * (1 << 30)
    )
    assert receipt["required_headroom_bytes"] < M3_MTP_FREE_GIB * (1 << 30)

    engine._hard_reserve_gib = None
    monkeypatch.setattr(memory, "host_memory_gib", lambda: CAL_HOST_GIB)
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: CAL_ADVISORY_GIB)
    assert engine.hard_reserve_gib == 20.0
    monkeypatch.setattr(memory, "execution_headroom", lambda: 30 << 30)
    assert engine.admit_parallel_samples(2)["required_headroom_bytes"] == 24 << 30


def test_the_policy_module_stays_pure():
    """The policy must not reach into Metal to size itself."""
    import inspect

    import mlx2.runtime.memory_policy as policy

    source = inspect.getsource(policy)
    assert "mlx.core" not in source
    assert "device_info" not in source
    # Both probes live with the other headroom accounting instead.
    from mlx2.memory import host_memory_gib, metal_advisory_gib

    for probe in (host_memory_gib, metal_advisory_gib):
        measured = probe()
        assert measured is None or (math.isfinite(measured) and measured > 0)


def test_the_probes_agree_with_the_rule_on_this_machine():
    """Whatever host runs the suite, the pair is self-consistent."""
    from mlx2.memory import host_memory_gib, metal_advisory_gib

    (host, advisory) = (host_memory_gib(), metal_advisory_gib())
    (service, driver) = C.host_scaled_reserves(host, advisory)
    assert C.MIN_SERVICE_RESERVE_GIB <= service <= 16.0
    assert C.MIN_DRIVER_ALLOWANCE_GIB <= driver <= 4.0
    if host is not None and advisory is not None:
        # Never reserve more than the advisory residue of an empty machine.
        assert service + driver < advisory
