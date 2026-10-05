"""CPU checks for controlled performance evidence eligibility and phase math."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from varlen_controlled_performance import (
    arm_order,
    paired_token_parity,
    summarize_eligible_pairs,
)
from varlen_performance_phases import (
    PhaseTotals,
    derived_host_phases,
    subtract_phase_snapshots,
)


def test_balanced_arm_order_has_equal_first_position_counts():
    orders = [arm_order(i) for i in range(4)]
    assert orders == [("ordinary", "paged"), ("paged", "ordinary")] * 2
    assert sum(pair[0] == "paged" for pair in orders) == 2
    with pytest.raises(ValueError):
        arm_order(-1)


def test_parity_compares_per_lane_sequences_and_refuses_incomplete():
    plain = {"outputs": {"7": [10, 11], "8": [12, 13]}}
    native = {"outputs": {"19": [10, 11], "20": [12, 13]}}
    assert paired_token_parity(plain, native)
    assert not paired_token_parity(plain, {"outputs": {"19": [10, 11], "20": [12, 14]}})
    with pytest.raises(ValueError):
        paired_token_parity(plain, {"outputs": {"19": [10]}})


def test_host_phase_subtraction_keeps_kernel_time_unknown():
    totals = PhaseTotals()
    for name, ms in (("pack_head_spans", 4), ("pack_metadata", 2),
                     ("append_plus_pack_poll", 10),
                     ("read_submit_plus_terminal_wait", 30),
                     ("read_terminal_wait", 25), ("read_event_poll", 3),
                     ("continuation_next_plus_forward", 50),
                     ("continuation_forward", 40)):
        totals.add(name, ms * 1_000_000)
    derived = derived_host_phases(totals.snapshot())
    assert derived["append_write_and_poll_excluding_pack_wall_ms"] == 6
    assert derived["read_submission_excluding_terminal_wait_wall_ms"] == 5
    assert derived["read_wait_outside_event_poll_wall_ms"] == 22
    assert derived["continuation_sampler_and_response_plus_sync_wall_ms"] == 10
    assert derived["device_kernel_time_ms"] is None


def test_overlapping_phase_accounting_is_rejected():
    with pytest.raises(RuntimeError):
        derived_host_phases({"pack_head_spans": {"wall_ms": 2},
                             "append_plus_pack_poll": {"wall_ms": 1}})


def test_interval_preserves_actual_per_layer_wait_samples():
    totals = PhaseTotals()
    totals.add("read_terminal_wait", 1_100_000)
    before = totals.snapshot()
    totals.add("read_terminal_wait", 2_200_000)
    totals.add("read_submit_plus_terminal_wait", 2_500_000)
    totals.add("read_poll_sleep_actual", 1_070_000)
    totals.add("read_poll_sleep_requested", 1_000_000)
    interval = subtract_phase_snapshots(before, totals.snapshot())
    assert interval["read_terminal_wait"]["samples_ms"] == [2.2]
    phases = derived_host_phases(interval)
    assert phases["read_poll_sleep_count"] == 1
    assert phases["read_poll_sleep_actual_wall_ms"] == 1.07
    assert phases["read_poll_sleep_requested_ms"] == 1


def test_performance_ratio_requires_full_trajectory_parity():
    good = {"ratio_eligible": True,
            "arms": {"ordinary": {"total_request_wall_ms": 10,
                                  "response_ready_wall_ms": 8},
                     "paged": {"total_request_wall_ms": 20,
                               "response_ready_wall_ms": 16}}}
    summary = summarize_eligible_pairs([good, good])
    assert summary["native_over_ordinary"]["complete_request"][
        "median_native_over_ordinary"] == 2
    assert summarize_eligible_pairs([good, {**good, "ratio_eligible": False}])[
        "native_over_ordinary"] is None
