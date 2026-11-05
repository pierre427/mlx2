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


def test_family_and_checkpoint_decisions_coexist(monkeypatch):
    builder = _builder()
    checkpoint = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-mlx-affine4-protected8@c9750f0f"
    monkeypatch.setattr(builder, "strong_claim_problems", lambda row: [])
    index = builder.validate_decisions([
        _row(),
        _row(checkpoint=checkpoint, qualified="yes", observed_used="yes"),
    ])
    assert set(index) == {
        ("nemotron_h_omni:base", "*", "feature", "native_mtp", "abc12345"),
        ("nemotron_h_omni:base", checkpoint, "feature", "native_mtp", "abc12345"),
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


def test_distinct_source_revisions_preserve_independent_decisions():
    builder = _builder()
    first = _row(checkpoint="m@0123abcd", source_revision="abc12345", decision="candidate")
    second = _row(checkpoint="m@0123abcd", source_revision="def67890", decision="rejected")
    index = builder.validate_decisions([first, second])
    assert len(index) == 2
    assert index[(first["model"], first["checkpoint"], "feature", "native_mtp", "abc12345")] == first
    assert index[(second["model"], second["checkpoint"], "feature", "native_mtp", "def67890")] == second


def test_same_normalized_source_remains_a_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        _builder().validate_decisions([_row(), _row(source_revision=" abc12345 ")])


def test_grid_emits_every_source_revision_without_combining_outcomes(tmp_path, monkeypatch):
    builder = _builder()
    rows = [
        _row(checkpoint="m@0123abcd", source_revision="def67890", decision="rejected", reason="later failed"),
        _row(checkpoint="m@0123abcd", source_revision="abc12345", decision="candidate", reason="earlier observed"),
    ]
    decisions = tmp_path / "decisions.tsv"
    with decisions.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(builder, "DECISIONS", decisions)
    monkeypatch.setattr(builder, "registered_types", lambda: ["nemotron_h_omni"])
    monkeypatch.setattr(builder, "FEATURES", ("native_mtp",))
    monkeypatch.setattr(builder, "server_flags", lambda: set())
    monkeypatch.setattr(builder, "environment_flags", lambda: set())
    target = tmp_path / "grid.tsv"
    builder.main(["--output", str(target)])
    with target.open(newline="") as handle:
        rendered = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rendered) == 3
    assert rendered[0]["checkpoint"] == "*" and rendered[0]["decision"] == "review_needed"
    assert rendered[1:] == sorted(rows, key=lambda row: row["source_revision"])


def test_lora_research_has_an_inventory_slot():
    assert "kimi_linear_supervised_lora_research" in _builder().FEATURES


def test_the_committed_decisions_carry_the_checkpoint_dimension():
    path = ROOT / "docs" / "research" / "model-feature-decisions.tsv"
    if not path.is_file():
        pytest.skip("private material absent: decisions TSV (not exported to the public mirror)")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and "checkpoint" in rows[0]
    _builder().validate_decisions(rows)


@pytest.mark.skipif(not (ROOT / 'docs/research/model-feature-decisions.tsv').is_file(),
                    reason='the decision ledger is private evidence')
def test_source_reviewed_features_have_explicit_inventory_slots():
    builder = _builder()
    with (ROOT / 'docs/research/model-feature-decisions.tsv').open(newline='') as handle:
        source = {row['feature_or_flag'] for row in csv.DictReader(handle, delimiter='\t')
                  if row['model'] == 'deepseek_v41:base' and row['scope'] == 'feature'}
    assert source <= set(builder.FEATURES)
    assert len(builder.FEATURES) == len(set(builder.FEATURES))


def test_invalid_inventory_preserves_existing_grid(tmp_path, monkeypatch):
    module = _builder()
    path = tmp_path / 'grid.tsv'; path.write_text('retained evidence\n')
    decisions = tmp_path / 'decisions.tsv'
    decisions.write_text('model\tcheckpoint\tscope\tfeature_or_flag\timplemented\tqualified\tselected\tobserved_used\tdecision\treason\tevidence\tsource_revision\tartifact_profile\n'
        'missing:base\t*\tfeature\tunknown\tyes\tno\tno\tno\treview\ttest\ttest\tsynthetic\ttest\n')
    monkeypatch.setattr(module, 'DECISIONS', decisions)
    with pytest.raises(ValueError, match='absent from inventory'):
        module.main(['--output', str(path)])
    assert path.read_text() == 'retained evidence\n'


@pytest.mark.parametrize("claim", ["yes_explicit_qualification", "yes_ab_arms_only", "qualified"])
def test_family_level_strong_claim_variants_are_refused(claim):
    with pytest.raises(ValueError, match="checkpoint"):
        _builder().validate_decisions([_row(qualified=claim)])


def test_strong_claim_rejects_unfingerprinted_artifact_and_bogus_commit():
    with pytest.raises(ValueError, match="fingerprint"):
        _builder().validate_decisions([_row(checkpoint="model local", qualified="yes")])
    with pytest.raises(ValueError, match="revision"):
        _builder().validate_decisions([_row(checkpoint="model@0123abcd", qualified="yes", source_revision="zzzz")])
