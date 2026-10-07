"""KR-05 (2026-10-06 sweep): the default floor-3 sorted-MoE pad changes bits
and is unqualified, so every receipt that saw it run must name it."""

import pytest

from mlx2 import serving
from mlx2.runtime.models import switch_layers as sl


class _Adapter:
    def diagnostics(self):
        return {}


@pytest.fixture
def pad_counters(monkeypatch):
    monkeypatch.setattr(sl, "_RHS_PAD_POLICY", "floor")
    monkeypatch.setattr(sl, "_RHS_PAD_MIN_ROWS_PER_EXPERT", 3)
    monkeypatch.setattr(sl, "rhs_pad_calls", 0)
    monkeypatch.setattr(sl, "rhs_pad_rows", 0)
    monkeypatch.setattr(sl, "pad_choice_counts", {})
    return monkeypatch


def test_default_floor_is_live():
    assert sl._rhs_pad_floor({}) == 3
    sl_floor = sl._RHS_PAD_MIN_ROWS_PER_EXPERT
    try:
        sl._RHS_PAD_MIN_ROWS_PER_EXPERT = 3
        assert sl._rhs_stream_pad(800, 256) == 224
    finally:
        sl._RHS_PAD_MIN_ROWS_PER_EXPERT = sl_floor


def test_engaged_floor_pad_reaches_the_execution_receipt(pad_counters):
    pad_counters.setattr(sl, "rhs_pad_calls", 7)
    pad_counters.setattr(sl, "rhs_pad_rows", 900)
    execution = serving._execution_diagnostics(_Adapter())
    pad = execution["moe_pad"]
    assert pad["policy"] == "floor" and pad["floor_rows_per_expert"] == 3
    assert pad["engaged"] is True and pad["padded_calls"] == 7
    assert pad["qualified"] is False


def test_idle_default_floor_adds_nothing(pad_counters):
    assert "moe_pad" not in serving._execution_diagnostics(_Adapter())


def test_non_default_floor_is_reported_before_it_runs(pad_counters):
    pad_counters.setattr(sl, "_RHS_PAD_MIN_ROWS_PER_EXPERT", 0)
    pad = serving._execution_diagnostics(_Adapter())["moe_pad"]
    assert pad["floor_rows_per_expert"] == 0 and pad["engaged"] is False


def test_reportable_rule_matches_serving(pad_counters):
    assert not sl.moe_pad_reportable(sl.moe_pad_status())
    pad_counters.setattr(sl, "rhs_pad_calls", 1)
    assert sl.moe_pad_reportable(sl.moe_pad_status())
    pad_counters.setattr(sl, "rhs_pad_calls", 0)
    pad_counters.setattr(sl, "_RHS_PAD_POLICY", "adaptive")
    assert sl.moe_pad_reportable(sl.moe_pad_status())


def test_reset_clears_every_counter(pad_counters):
    """rfix-kad 2026-10-07: a reset kept rhs_pad_calls/rows, so every later
    status (and route receipt) stayed engaged=True."""
    pad_counters.setattr(sl, "rhs_pad_calls", 7)
    pad_counters.setattr(sl, "rhs_pad_rows", 900)
    sl._record_pad_choice("uncalibrated_rhs")
    before = sl.moe_pad_status(reset=True)
    assert before["engaged"] and before["padded_calls"] == 7
    after = sl.moe_pad_status()
    assert after["engaged"] is False
    assert (after["padded_calls"], after["pad_rows"], after["choices"]) == (0, 0, {})
    assert not sl.moe_pad_reportable(after)
