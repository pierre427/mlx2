# SPDX-License-Identifier: MIT
"""Measured MoE verify transient (CPU only; no model, no Metal).

``transient_gib_per_lane`` is the peak transient device memory of one lane's
M=(k+1) verify forward, above the resident state that survives the forward --
weights and the K/V + recurrent cache excluded.  It was calibrated at 1.76 on
Flash-Next and measured at 3.1 on a dense Qwen3.8-27B; the dense 3.1 was then
copied onto three small-active MoE adapters it was never measured on.

Measured on an M3 Pro on 2026-09-19 (provenance/lane-transient-moe.json), the
k=2 per-lane transient of two small-active MoE models is 0.015-0.071 GiB
across 1K/4K/16K context x 1/2/4 lanes -- flat, and two orders of magnitude
under 3.1.  ``MOE_TRANSIENT_GIB_PER_LANE`` replaces the misapplied figure.

The first half of this file is the no-regression requirement: the 128 GiB
calibration host must admit exactly the lane counts it admitted before, for
the models calibrated there (Flash-Next at 1.76 and the dense 27B at 3.1).
Those two values, and the arithmetic that consumes them, are untouched.
"""

import json
from pathlib import Path

import pytest

from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C

# The 128 GiB Flash-Next operating point from the class docstring: 72.5 GiB
# resident under PLE offload leaves 55.5 GiB free.
FLASH_NEXT_FREE_GIB = 128.0 - 72.5
# A dense Qwen3.8-27B on the same host: ~18 GiB resident.
DENSE_27B_FREE_GIB = 128.0 - 18.0


def _admitted(controller, contexts, free_gib, **kwargs):
    decision = controller.decide(contexts, free_gib, **kwargs)
    return (len(decision.mtp_indices), decision.stage, decision.draft_depths)


# --------------------------------------------------------------------------
# 128 GiB path: unchanged.
# --------------------------------------------------------------------------


def test_flash_next_calibration_constant_is_untouched():
    """1.76 is Flash-Next's own measurement and was not re-measured."""
    assert C.K2_TRANSIENT_GIB_PER_LANE == 1.76
    assert C(host_memory_gib=128.0).transient_gib_per_lane == 1.76


def test_dense_27b_constant_is_untouched():
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

    assert Qwen38CacheBudget.__dataclass_fields__[
        "transient_gib_per_lane"
    ].default == 3.1


def test_xing_is_left_unmeasured_rather_than_guessed_again():
    """Xing4.0's mHC streams are this exact workspace term and were not measured."""
    from mlx2.adapters.xing_memory import XingCacheBudget

    assert XingCacheBudget.__dataclass_fields__[
        "transient_gib_per_lane"
    ].default == 3.1


@pytest.mark.parametrize("context", [1024, 4096, 16384])
def test_128_gib_flash_next_lane_counts_are_bit_for_bit_unchanged(context):
    """The docstring's calibration: N=16 near 1K, N=4 near 16K.

    Reproduced here from the constants alone, so any later edit to the
    envelope, the reserve or the Flash-Next transient trips this test.
    """
    controller = C(host_memory_gib=128.0)
    assert controller.hard_reserve_gib == 20.0
    usable = FLASH_NEXT_FREE_GIB - 20.0
    assert usable == pytest.approx(35.5)

    per_lane = controller.lane_gib(context, 2)
    expected = 0.44 * (context / 1024.0) + 1.76
    assert per_lane == pytest.approx(expected)

    contexts = [context] * 64
    (count, stage, _) = _admitted(controller, contexts, FLASH_NEXT_FREE_GIB)
    assert stage == "fewer_lanes"
    # The memory envelope alone, before SATURATION_LANE_CAP bites.
    envelope_count = int(usable // per_lane)
    assert count == min(envelope_count, C.SATURATION_LANE_CAP)
    # The calibration the class docstring states: N=16 near 1K, N=4 near 16K.
    assert {1024: 16, 4096: 10, 16384: 4}[context] == count


@pytest.mark.parametrize("context", [1024, 4096, 16384])
def test_128_gib_dense_27b_lane_counts_are_unchanged(context):
    """The dense 27B keeps 3.1, so its admitted counts cannot move."""
    controller = C(host_memory_gib=128.0, transient_gib_per_lane=3.1)
    per_lane = controller.lane_gib(context, 2)
    assert per_lane == pytest.approx(0.44 * (context / 1024.0) + 3.1)
    (count, stage, _) = _admitted(controller, [context] * 64, DENSE_27B_FREE_GIB)
    assert stage == "fewer_lanes"
    usable = DENSE_27B_FREE_GIB - 20.0
    assert count == min(int(usable // per_lane), C.SATURATION_LANE_CAP)
    assert {1024: 16, 4096: 16, 16384: 8}[context] == count


def test_moe_change_cannot_reach_the_flash_next_or_dense_paths():
    """The new constant is only reachable through the adapters that set it."""
    assert C().transient_gib_per_lane == C.K2_TRANSIENT_GIB_PER_LANE
    assert C.MOE_TRANSIENT_GIB_PER_LANE != C.K2_TRANSIENT_GIB_PER_LANE


# --------------------------------------------------------------------------
# The measured MoE value.
# --------------------------------------------------------------------------


def test_moe_constant_matches_the_recorded_measurement():
    record = json.loads(
        (Path(__file__).resolve().parents[1] / "provenance" / "lane-transient-moe.json").read_text()
    )
    assert record["conclusion"]["moe_transient_gib_per_lane"] == C.MOE_TRANSIENT_GIB_PER_LANE
    worst = max(
        cell["transient_per_lane_gib"]
        for model in record["models"].values()
        for cell in model["cells"]
    )
    # Conservative: the charged constant covers every steady-state cell
    # measured, with margin, and also the cold-allocator spike.
    assert worst < C.MOE_TRANSIENT_GIB_PER_LANE
    assert C.MOE_TRANSIENT_GIB_PER_LANE >= 4.0 * worst


def test_moe_adapters_carry_the_measured_value():
    from mlx2.adapters.north_memory import NorthCacheBudget

    assert NorthCacheBudget.__dataclass_fields__[
        "transient_gib_per_lane"
    ].default == C.MOE_TRANSIENT_GIB_PER_LANE


def test_m3_admits_a_full_depth_moe_lane_where_3_1_could_not():
    """The regression the measurement fixes.

    On the 36 GiB M3 Pro (Metal advisory 28.08 GiB) the whole lane budget
    after the reserve is ~4.33 GiB.  A 3.1 GiB/lane transient eats three
    quarters of it, so past ~1K context no lane can be seated at k=2: the
    ladder degrades every lane to k=1 at the cycle boundary and the request
    gate degrades every arrival to the k=0 depth floor.  With the measured
    value the top rung fits at every context the cache envelope allows.

    The reserve is the one from ``host_scaled_reserves`` with the M3's real
    advisory supplied.  When this test was written the reserve was 5.625 GiB
    and the budget 2.71 GiB, and the dense 3.1 exceeded it even at 34 tokens;
    charging the OS margin once (see the class docstring) raises the budget
    to 4.33 GiB, so the two defects now separate cleanly -- the misapplied
    transient no longer breaks the shortest prompts, only real context.
    """
    controller = C(host_memory_gib=36.0, advisory_gib=28.08)
    free = 8.336  # measured execution_headroom with the 20 GiB model resident
    usable = free - controller.hard_reserve_gib
    assert controller.hard_reserve_gib == pytest.approx(4.0029, abs=1e-4)
    assert usable == pytest.approx(4.3331, abs=1e-4)

    dense = C(host_memory_gib=36.0, advisory_gib=28.08, transient_gib_per_lane=3.1)
    assert dense.lane_gib(4096, 2) > usable
    # The ladder from 33d1d5a still seats the lane, but never at full depth:
    # the transient alone outruns the budget, so k=2 is unreachable.
    (count, stage, depths) = _admitted(dense, [4096], free)
    assert (count, stage, depths) == (1, "lower_k", (1,))

    moe = C(
        host_memory_gib=36.0,
        advisory_gib=28.08,
        transient_gib_per_lane=C.MOE_TRANSIENT_GIB_PER_LANE,
    )
    assert moe.lane_gib(4096, 2) < usable
    (count, _, depths) = _admitted(moe, [4096], free)
    assert (count, depths) == (1, (2,))


@pytest.mark.parametrize(
    ("context", "expected"), [(34, 8), (1024, 5), (4096, 2), (16384, 0)]
)
def test_m3_moe_lane_counts_are_set_by_the_cache_envelope_not_the_transient(
    context, expected
):
    """Past ~1K the 0.44 GiB/1K cache envelope, not the transient, binds.

    That is the point of the correction: the workspace term stops being the
    whole budget and the measured cache cost decides, as it does on the
    128 GiB host.  At 16K a single lane's projected cache alone (7.04 GiB)
    exceeds the M3's 4.33 GiB, so the cohort queues -- a cache-envelope
    limit, unaffected by any transient value.
    """
    controller = C(
        host_memory_gib=36.0,
        advisory_gib=28.08,
        transient_gib_per_lane=C.MOE_TRANSIENT_GIB_PER_LANE,
    )
    free = 8.336
    usable = free - controller.hard_reserve_gib
    (count, _, _) = _admitted(controller, [context] * 8, free)
    assert count == expected
    assert count == min(int(usable // controller.lane_gib(context, 2)), 8)
