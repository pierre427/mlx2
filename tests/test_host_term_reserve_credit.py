# SPDX-License-Identifier: MIT
"""The host-available term does not charge the service quota twice (CPU only).

Admission subtracts one hard reserve -- service 16 + driver 4 GiB on the
128 GiB calibration host -- from ``min(host available, advisory - in use)``.
The 16 GiB service share is the host's non-lane quota beyond what the
advisory already withholds for the rest of the machine.  The host-available
term *measures* the rest of the machine, so whenever other processes used
more than the advisory's 16 GiB their pages were taken out of that term and
then the full quota was taken out again.

Readings below are the qualifier's B2 warm ``batch_cohort`` on Flash-Next
native MTP (32K, 4 lanes, 16 GiB cache) on the shared M5 Max, run
``new-served`` attempt 0 of the 2026-10-01 diagnosis
(qualify-709bedf8-uncensored/diagnostics): host available (Mach estimate)
23.28 GiB, physical footprint 78.01 GiB, Metal active 73.17 GiB, so the rest
of the host held 128 - 78.01 - 23.28 = 26.7 GiB.  Admission saw 3.28 GiB
usable and refused the cohort with 34 GiB of advisory residue left.
"""

import mlx.core as mx
import pytest

from mlx2 import memory
from mlx2.runtime import os_memory
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.serving import admit_lane_headroom

GIB = 1 << 30
HOST = 128 * GIB
ADVISORY = 112 * GIB
FAIL_AVAILABLE = 23.28
FAIL_FOOTPRINT = 78.01
FAIL_ACTIVE = 73.17


def _readings(monkeypatch, *, available, footprint, active, cached=0.0, host=HOST, advisory=ADVISORY):
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"memory_size": host, "max_recommended_working_set_size": advisory},
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: int(active * GIB))
    monkeypatch.setattr(mx, "get_cache_memory", lambda: int(cached * GIB))
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: int(footprint * GIB))
    monkeypatch.setattr(memory, "host_available_bytes", lambda _signals=False: int(available * GIB))


def _controller():
    return C(host_memory_gib=128.0, advisory_gib=112.0)


def test_failing_cohort_reading_now_admits_both_lanes(monkeypatch):
    _readings(monkeypatch, available=FAIL_AVAILABLE, footprint=FAIL_FOOTPRINT, active=FAIL_ACTIVE)
    controller = _controller()
    free = memory.execution_headroom(host_signals=True) / GIB
    # Decode-cycle admission of the two warm members (atomic, one depth).
    plan = controller.decide(
        [32579 + 64, 32579 + 64],
        free,
        cache_gib=[1.05, 1.05],
        max_draft=2,
        atomic_cohort=True,
    )
    assert plan.stage == "full", (free, plan)
    assert plan.draft_depths == (2, 2)
    # Lane admission: the second member carries the first's grant.
    granted = 0.0

    def headroom():
        return memory.execution_headroom(host_signals=True) - int(granted * GIB)

    for _member in range(2):
        (admitted, floor, required) = admit_lane_headroom(
            controller,
            context_tokens=32579 + 64,
            draft_depth=2,
            cache_gib=1.05,
            headroom=headroom,
            reclaim=lambda: None,
            evict=lambda: False,
        )
        assert admitted and not floor
        granted += required - controller.hard_reserve_gib


def test_advisory_term_binds_once_the_double_charge_is_removed(monkeypatch):
    _readings(monkeypatch, available=FAIL_AVAILABLE, footprint=FAIL_FOOTPRINT, active=FAIL_ACTIVE)
    free = memory.execution_headroom(host_signals=True) / GIB
    assert free == pytest.approx(112.0 - FAIL_FOOTPRINT, abs=1e-6)


def test_host_floor_still_kept_free_when_the_host_is_short(monkeypatch):
    # The rest of the host holds 44 GiB: host available 6 GiB.  The credit is
    # capped at service - floor, so admission still keeps the 3 GiB service
    # floor plus the 4 GiB driver allowance of real host memory free.
    _readings(monkeypatch, available=6.0, footprint=FAIL_FOOTPRINT, active=FAIL_ACTIVE)
    controller = _controller()
    free = memory.execution_headroom(host_signals=True) / GIB
    usable = free - controller.hard_reserve_gib
    assert usable == pytest.approx(6.0 - C.MIN_SERVICE_RESERVE_GIB - controller.driver_allowance_gib)
    plan = controller.decide([32643], free, cache_gib=[1.05], max_draft=2)
    assert plan.stage == "queue"


def test_nothing_moves_while_the_rest_of_the_host_fits_its_share(monkeypatch):
    # Rest of host exactly the advisory's 16 GiB withheld: both terms agree.
    _readings(monkeypatch, available=128 - 16 - FAIL_FOOTPRINT, footprint=FAIL_FOOTPRINT, active=FAIL_ACTIVE)
    assert memory.execution_headroom() / GIB == pytest.approx(112 - FAIL_FOOTPRINT, abs=1e-6)
    # MLX free-buffer cache still counts against the advisory term.
    _readings(monkeypatch, available=40.0, footprint=70.0, active=70.0, cached=5.0)
    assert memory.execution_headroom() / GIB == pytest.approx(112 - 75.0, abs=1e-6)


def test_small_host_at_its_floor_gets_no_credit():
    assert memory.host_term_reserve_credit_bytes(36 * GIB, int(28.08 * GIB)) == 0


def test_calibration_host_credit_is_service_above_floor():
    assert memory.host_term_reserve_credit_bytes(HOST, ADVISORY) == int(
        (C.SERVICE_RESERVE_GIB - C.MIN_SERVICE_RESERVE_GIB) * GIB
    )


@pytest.mark.parametrize("host,advisory", [(0, ADVISORY), (HOST, 0), (None, ADVISORY)])
def test_unknown_readings_give_no_credit(host, advisory):
    assert memory.host_term_reserve_credit_bytes(host, advisory) == 0


def test_unmeasurable_footprint_still_fails_closed():
    assert memory.available_execution_bytes(
        available=50 * GIB, recommended=ADVISORY, active=0, cached=0, footprint=None,
        host_quota_credit=13 * GIB,
    ) == 0
