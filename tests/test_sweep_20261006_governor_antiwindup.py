"""The budget governor does not bank throttle while it has no lane running.

The power reading is host-wide; on a shared host another process can draw
over budget while this engine idles.  Tightening then would throttle this
engine's next round for power it never drew (measured: a B1 decode drawing
20 W under a 25 W budget was paced to 11 tok/s).
"""

from __future__ import annotations

from mlx2.power_governor import PowerGovernor


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _governor():
    clock = Clock()
    gov = PowerGovernor(
        {"mode": "budget", "budget_watts": 25.0},
        max_lanes=4,
        clock=clock,
    )
    return gov, clock


def test_idle_engine_does_not_tighten_on_foreign_power():
    gov, clock = _governor()
    for _ in range(30):
        clock.t += 1.0
        gov.round_hint(())  # no active lanes
        gov.observe(1.0, 60.0)
    assert gov._throttle == 0.0
    assert gov._counts["idle_tighten_skipped"] > 0


def test_active_engine_still_tightens():
    gov, clock = _governor()
    for _ in range(10):
        clock.t += 1.0
        gov.round_hint(("r1",))
        gov.observe(1.0, 60.0)
    assert gov._throttle > 0.0
