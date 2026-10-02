"""Pad-row census arithmetic of scripts/measure_fn_mtp_padding.py (CPU, no model)."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "measure_fn_mtp_padding.py"
_spec = importlib.util.spec_from_file_location("measure_fn_mtp_padding", _PATH)
measure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measure)


def _rec(kind, lengths, width, ms=None):
    lanes = len(lengths)
    return {"kind": kind, "shape": (lanes, width), "lengths": tuple(lengths),
            "padding": tuple(width - n for n in lengths), "ms": ms}


def test_counts_pad_rows_per_kind_and_lane_count():
    records = [
        _rec("verify", [3, 3, 3, 3], 3),
        _rec("verify", [3, 2, 3, 1], 3),
        _rec("draft", [4, 1, 2, 4], 4),
        _rec("draft", [1, 1, 0, 1], 1),
        _rec("verify", [3, 3], 3),
        {"kind": "verify", "shape": (1, 9), "lengths": None, "padding": None, "ms": None},
    ]
    out = measure.pad_census(records)
    assert set(out) == {"verify@B4", "draft@B4", "verify@B2"}
    v4 = out["verify@B4"]
    assert (v4["forwards"], v4["rows"], v4["valid_rows"], v4["pad_rows"]) == (2, 24, 21, 3)
    assert v4["pad_fraction"] == pytest.approx(3 / 24)
    assert v4["width_histogram"] == {3: 2}
    d4 = out["draft@B4"]
    assert (d4["rows"], d4["pad_rows"]) == (20, 6)
    assert d4["width_histogram"] == {1: 1, 4: 1}
    assert out["verify@B2"]["pad_rows"] == 0
    assert "ms_total" not in v4


def test_linear_pad_time_attribution():
    records = [
        _rec("verify", [3, 3, 3, 3], 3, ms=60.0),
        _rec("verify", [3, 3, 3, 0], 3, ms=40.0),  # a quarter of the rows are pad
    ]
    entry = measure.pad_census(records)["verify@B4"]
    assert entry["ms_total"] == pytest.approx(100.0)
    assert entry["ms_median_by_rows"] == {12: pytest.approx(50.0)}
    assert entry["pad_ms_linear_upper_bound"] == pytest.approx(10.0)
