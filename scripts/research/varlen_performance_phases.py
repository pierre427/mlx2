"""Host-side phase accounting for the default-off Qwen3 native research route.

This is deliberately outside production. Wall spans include GPU waits; no span
is described as kernel execution time without a device timestamp.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class PhaseTotals:
    calls: dict[str, int] = field(default_factory=dict)
    ns: dict[str, int] = field(default_factory=dict)
    samples_ns: dict[str, list[int]] = field(default_factory=dict)

    def add(self, name: str, elapsed_ns: int) -> None:
        if elapsed_ns < 0:
            raise ValueError("negative phase interval")
        self.calls[name] = self.calls.get(name, 0) + 1
        self.ns[name] = self.ns.get(name, 0) + elapsed_ns
        if name in ("read_terminal_wait", "append_plus_pack_poll",
                    "read_poll_sleep_actual", "write_poll_sleep_actual"):
            self.samples_ns.setdefault(name, []).append(elapsed_ns)

    def snapshot(self) -> dict:
        return {name: {"calls": self.calls[name], "wall_ms": self.ns[name] / 1e6,
                       **({"samples_ms": [value / 1e6 for value in self.samples_ns[name]]}
                          if name in self.samples_ns else {})}
                for name in sorted(self.calls)}


def subtract_phase_snapshots(before: dict, after: dict) -> dict:
    """Return a nonnegative interval from cumulative phase observations."""
    delta = {}
    for name, end in after.items():
        start = before.get(name, {"calls": 0, "wall_ms": 0.0, "samples_ms": []})
        calls = end["calls"] - start["calls"]
        wall_ms = end["wall_ms"] - start["wall_ms"]
        if calls < 0 or wall_ms < -1e-6:
            raise RuntimeError("phase counters regressed")
        item = {"calls": calls, "wall_ms": wall_ms}
        if "samples_ms" in end:
            item["samples_ms"] = end["samples_ms"][start["calls"]:]
            if len(item["samples_ms"]) != calls:
                raise RuntimeError("phase sample count differs")
        delta[name] = item
    return delta


@contextmanager
def instrument_native_backend(totals: PhaseTotals):
    """Count nested host spans without changing execution or completion rules."""
    from mlx2.adapters import qwen3_paged_candidate as candidate
    from mlx2.runtime import paged_native_continuation as continuation
    from mlx2.runtime import qwen3_paged_native_backend as backend

    original: list[tuple[object, str, Callable]] = []
    active_poll = ["other"]

    def wrap(target, attribute: str, name: str) -> None:
        previous = getattr(target, attribute)

        def timed(*args, **kwargs):
            previous_poll = active_poll[0]
            if attribute == "append_completed":
                active_poll[0] = "write"
            elif attribute == "_wait_read":
                active_poll[0] = "read"
            start = time.perf_counter_ns()
            try:
                return previous(*args, **kwargs)
            finally:
                totals.add(name, time.perf_counter_ns() - start)
                active_poll[0] = previous_poll

        setattr(target, attribute, timed)
        original.append((target, attribute, previous))

    try:
        class TimedSleepProxy:
            def __getattr__(self, name):
                return getattr(time, name)

            def sleep(self, seconds):
                label = active_poll[0]
                started = time.perf_counter_ns()
                try:
                    return time.sleep(seconds)
                finally:
                    if label in ("read", "write"):
                        totals.add(f"{label}_poll_sleep_actual",
                                   time.perf_counter_ns() - started)
                        totals.add(f"{label}_poll_sleep_requested",
                                   int(seconds * 1e9))

        original.append((backend, "time", backend.time))
        backend.time = TimedSleepProxy()
        wrap(backend, "pack_head_spans", "pack_head_spans")
        wrap(backend.NativeQwen3PagedBackend, "append_completed", "append_plus_pack_poll")
        wrap(backend.NativeQwen3PagedBackend, "read_completed", "read_submit_plus_terminal_wait")
        wrap(backend.NativeQwen3PagedBackend, "_wait_read", "read_terminal_wait")
        wrap(backend, "poll_native_paged_read_events", "read_event_poll")
        wrap(candidate, "prepare_packed_token_read", "pack_metadata")
        wrap(continuation.NativeQwen3Continuation, "_forward_pending", "continuation_forward")
        wrap(continuation.NativeQwen3Continuation, "_next_with_reader", "continuation_next_plus_forward")
        yield totals
    finally:
        for target, attribute, previous in reversed(original):
            setattr(target, attribute, previous)


def derived_host_phases(snapshot: dict) -> dict:
    """Subtract only known nested wall spans; preserve raw overlapping spans."""
    def ms(name):
        return snapshot.get(name, {}).get("wall_ms", 0.0)

    pack = ms("pack_head_spans")
    append = ms("append_plus_pack_poll")
    read = ms("read_submit_plus_terminal_wait")
    wait = ms("read_terminal_wait")
    poll = ms("read_event_poll")
    step = ms("continuation_next_plus_forward")
    forward = ms("continuation_forward")
    if (pack > append + 1e-6 or wait > read + 1e-6 or
            poll > wait + 1e-6 or forward > step + 1e-6):
        raise RuntimeError("nested host-phase spans are inconsistent")
    return {
        "pack_head_spans_wall_ms": pack,
        "pack_metadata_wall_ms": ms("pack_metadata"),
        "append_write_and_poll_excluding_pack_wall_ms": append - pack,
        "read_submission_excluding_terminal_wait_wall_ms": read - wait,
        "read_terminal_wait_wall_ms": wait,
        "read_event_poll_wall_ms": poll,
        "read_wait_outside_event_poll_wall_ms": wait - poll,
        "read_poll_sleep_count": snapshot.get("read_poll_sleep_actual", {}).get("calls", 0),
        "read_poll_sleep_actual_wall_ms": ms("read_poll_sleep_actual"),
        "read_poll_sleep_requested_ms": ms("read_poll_sleep_requested"),
        "write_poll_sleep_count": snapshot.get("write_poll_sleep_actual", {}).get("calls", 0),
        "write_poll_sleep_actual_wall_ms": ms("write_poll_sleep_actual"),
        "write_poll_sleep_requested_ms": ms("write_poll_sleep_requested"),
        "read_terminal_callback_latency_per_layer_ms": snapshot.get(
            "read_terminal_wait", {}).get("samples_ms", []),
        "continuation_sampler_and_response_plus_sync_wall_ms": step - forward,
        "device_kernel_time_ms": None,
        "device_kernel_time_note": "native completion events do not expose device timestamps",
    }


__all__ = ["PhaseTotals", "derived_host_phases", "instrument_native_backend",
           "subtract_phase_snapshots"]
