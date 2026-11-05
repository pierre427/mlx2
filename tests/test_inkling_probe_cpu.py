"""A partial or mismatched oracle cannot be used to judge a serving run."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "inkling_probe_contract",
    Path(__file__).resolve().parents[1] / "scripts/probe_inkling.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "change",
    [
        {"complete": False},
        {"rows": {"16k": {}}},
        {"source_revision": "different-commit"},
        {"checkpoint": "different-artifact"},
    ],
)
def test_partial_or_mismatched_reference_refuses(change):
    receipt = {
        "complete": True,
        "rows": {"16k": {}, "32k": {}},
        "checkpoint": "artifact@fingerprint",
        "source_revision": "exact-commit",
    }
    expected = {
        "checkpoint": "artifact@fingerprint",
        "source_revision": "exact-commit",
        "cases": ("16k", "32k"),
    }
    probe.validate_reference_receipt(receipt, **expected)
    with pytest.raises(RuntimeError, match="incomplete or has a different identity"):
        probe.validate_reference_receipt({**receipt, **change}, **expected)
