"""The mixed prefill+decode round applies the contended slice floor (sweep
2026-10-02 P4).  The ordinary round lifts a contended slice to
``decode_fairness.slice_floor``; ``_next_mixed`` sized its slice from the
stall bound alone, so with a floor set the two rounds sliced differently."""

from collections import defaultdict

import pytest

from mlx2.runtime import generate
from mlx2.runtime.adaptive_policy import DecodeTimeFairness


class _Stop(Exception):
    pass


class _Scheduler(generate.BatchGenerator):
    def __del__(self):
        pass


def _mixed_slice(slice_floor, *, enabled=True, decode_lanes=2):
    scheduler = _Scheduler.__new__(_Scheduler)
    scheduler.prefill_step_size = 8192
    scheduler.prefill_step_autoscale = False
    scheduler.scheduler_stats = defaultdict(int)
    scheduler.decode_time_fairness = DecodeTimeFairness(
        enabled=enabled, slice_floor=slice_floor
    )
    # 900 tok/s at the 500 ms stall target -> a 448-row stall bound.
    scheduler.decode_time_fairness.best_prefill_tokens_per_second = 900.0
    scheduler._generation_batch = [object()] * decode_lanes
    scheduler._currently_processing = [[[list(range(20000))], 0, 20000, None, 0]]

    class Prompt:
        uids = [7]

        def prompt(self, chunks, forward_fn=None):
            raise _Stop(len(chunks[0]))

    scheduler._prompt_batch = Prompt()
    scheduler._prompt_tokens_counter = 0
    with pytest.raises(_Stop) as stop:
        scheduler._next_mixed()
    return stop.value.args[0], scheduler.decode_time_fairness.counters


def test_mixed_slice_without_floor_is_the_stall_bound():
    rows, counters = _mixed_slice(0)
    assert rows + 2 <= 448
    assert "slice_floor_lifts" not in counters


def test_mixed_slice_is_lifted_to_the_floor():
    rows, counters = _mixed_slice(1024)
    assert 1024 - 64 < rows + 2 <= 1024
    assert counters["slice_floor_lifts"] == 1


def test_mixed_slice_floor_needs_fairness_enabled():
    rows, _ = _mixed_slice(1024, enabled=False)
    assert rows + 2 <= 448
