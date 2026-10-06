"""CPU-only checks for the bounded B1 discriminator's decision logic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_qwen38_tree15_b1_discriminator import first_difference, validate_counters  # noqa: E402


def status(*, tree=0, physical=0, width=0, fallback=0):
    return {"scheduler": {"external_auto_tree_rounds": tree,
                          "external_tensorfold_target_rounds": physical,
                          "external_auto_tree_max_width": width,
                          "draft_fallbacks": fallback}}


def test_first_difference_including_length_mismatch():
    assert first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert first_difference([1, 2], [1, 2, 3]) == 2
    assert first_difference([1, 2], [1, 2]) is None


def test_b1_requires_physical_target_round_and_zero_fallback():
    row = validate_counters(status(), status(tree=3, physical=3, width=1), True)
    assert row["delta"]["external_tensorfold_target_rounds"] == 3
    with pytest.raises(RuntimeError, match="B1 TensorFold"):
        validate_counters(status(), status(tree=3, width=1), True)
    with pytest.raises(RuntimeError, match="draft_fallbacks"):
        validate_counters(status(), status(tree=3, physical=3, width=1, fallback=1), True)


def test_ordinary_must_not_use_tree():
    validate_counters(status(), status(), False)
    with pytest.raises(RuntimeError, match="ordinary route used tree"):
        validate_counters(status(), status(tree=1, physical=1, width=1), False)
