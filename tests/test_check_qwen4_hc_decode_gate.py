"""check_qwen4_hc_decode.py must fail when its comparisons are vacuous (sweep H5).

When the fused HC path declines, the "fused" arm runs the composed ops and
compares equal to itself.  The script used to report that as bit-identical
and always exited 0.
"""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def gate():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield importlib.import_module("check_qwen4_hc_decode")
    finally:
        sys.path.remove(str(SCRIPTS))


def _report(**overrides):
    report = {
        "exact": {"all_bit_identical": True, "mismatched_cases": 0, "cases": 48,
                  "fused_declined": 0},
        "elementwise_table_mismatches": {"silu_div_hc": 0, "sigmoid": 0,
                                         "inject_gate": 0, "finite_inputs": 65000},
        "chain": {"fused_declined": 0},
        "chain_bit_identical": True,
    }
    report.update(overrides)
    return report


def test_a_clean_gate_passes(gate):
    assert gate.gate_failures(_report()) == []


def test_declined_fused_calls_fail_the_gate(gate):
    vacuous = _report(exact={"all_bit_identical": True, "mismatched_cases": 0,
                             "cases": 48, "fused_declined": 48},
                      chain={"fused_declined": 96})
    failures = gate.gate_failures(vacuous)
    assert any("declined" in f and f.startswith("exact") for f in failures)
    assert any(f.startswith("chain") for f in failures)


def test_mismatches_and_table_and_row_exact_fail_the_gate(gate):
    assert gate.gate_failures(_report(exact={"all_bit_identical": False,
                                             "mismatched_cases": 2, "cases": 48,
                                             "fused_declined": 0}))
    assert gate.gate_failures(_report(elementwise_table_mismatches={"sigmoid": 3}))
    assert gate.gate_failures(_report(row_exact_window={"rows": 10, "all_rows_equal": True,
                                                        "fused_declined": 4}))
    assert gate.gate_failures(_report(chain_bit_identical=False))
    assert gate.gate_failures(_report(exact={"cases": 0, "all_bit_identical": True}))


def test_main_exits_nonzero_on_failures(gate):
    source = (SCRIPTS / "check_qwen4_hc_decode.py").read_text()
    assert 'report["failures"] = gate_failures(report)' in source
    assert "raise SystemExit(1)" in source
