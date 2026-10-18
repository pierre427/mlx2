"""Model x checkpoint x feature review grid: decision indexing and identity rules."""
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_model_feature_grid", ROOT / "scripts" / "build_model_feature_grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(**overrides):
    row = {
        "model": "nemotron_h_omni:base", "checkpoint": "*", "scope": "feature",
        "feature_or_flag": "native_mtp", "implemented": "yes", "qualified": "no",
        "selected": "no", "observed_used": "no", "decision": "candidate",
        "reason": "r", "evidence": "e", "source_revision": "abc12345",
        "artifact_profile": "p",
    }
    row.update(overrides)
    return row


def test_family_and_checkpoint_decisions_coexist():
    builder = _builder()
    checkpoint = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-mlx-affine4-protected8@c9750f0f"
    index = builder.validate_decisions([
        _row(),
        _row(checkpoint=checkpoint, qualified="yes", observed_used="yes"),
    ])
    assert set(index) == {
        ("nemotron_h_omni:base", "*", "feature", "native_mtp"),
        ("nemotron_h_omni:base", checkpoint, "feature", "native_mtp"),
    }


@pytest.mark.parametrize("claim", [{"qualified": "yes"}, {"observed_used": "yes"}])
def test_a_strong_claim_must_name_its_checkpoint(claim):
    with pytest.raises(ValueError, match="checkpoint"):
        _builder().validate_decisions([_row(**claim)])


@pytest.mark.parametrize("revision", ["", "not_recorded"])
def test_a_strong_claim_must_name_its_commit(revision):
    with pytest.raises(ValueError, match="commit"):
        _builder().validate_decisions(
            [_row(checkpoint="m@0123abcd", qualified="yes", source_revision=revision)]
        )


def test_duplicates_and_missing_checkpoint_are_refused():
    builder = _builder()
    with pytest.raises(ValueError, match="duplicate"):
        builder.validate_decisions([_row(), _row()])
    with pytest.raises(ValueError, match="without a checkpoint"):
        builder.validate_decisions([_row(checkpoint="")])


def test_the_committed_decisions_carry_the_checkpoint_dimension():
    path = ROOT / "docs" / "research" / "model-feature-decisions.tsv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and "checkpoint" in rows[0]
    _builder().validate_decisions(rows)
