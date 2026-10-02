"""flash_next_options_perf.py: a valueless lever written ``name:off`` must not
switch the lever on (sweep H6)."""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def perf():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("flash_next_options_perf")
    finally:
        sys.path.remove(str(SCRIPTS))


class _Recorder:
    def __init__(self):
        self.calls = []

    def set_fused_gate_inject_enabled(self, value):
        self.calls.append(value)


@pytest.mark.parametrize("value", ["off", "0", "false"])
def test_switch_lever_with_an_off_value_is_refused(perf, value):
    toggles = object.__new__(perf.Toggles)
    toggles.GI = _Recorder()
    with pytest.raises(ValueError, match="takes no value"):
        toggles._apply(f"gate_inject:{value}")
    assert toggles.GI.calls == []


@pytest.mark.parametrize("part", ["gate_inject", "gate_inject:on", "gate_inject:1"])
def test_switch_lever_spellings_that_mean_on_still_switch_it_on(perf, part):
    toggles = object.__new__(perf.Toggles)
    toggles.GI = _Recorder()
    toggles._apply(part)
    assert toggles.GI.calls == [True]
