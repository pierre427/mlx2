import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx2.conversion.importance import ActivationMoments


@pytest.fixture
def merger(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("merge_nemotron35_calibration")


def make_unit(root, first, count, *, expert_count=1024, text_records=()):
    from test_activation_importance_cpu import binding

    path = root / f"unit-{first}"
    (path / "calibration").mkdir(parents=True)
    stats = ActivationMoments({"bank": (2, 3)}, {**binding(), "dependencies": {"mlx": "fixture"}})
    stats.add_reduced("bank", np.full((2, 3), 2 * expert_count, np.float64), np.full(2, expert_count, np.uint64))
    samples = []
    for index in range(first, first + count):
        sample_id = f"image:{index}"
        input_hash = f"{index:064x}"
        stats.record_sample(sample_id, modality="image_text", input_sha256=input_hash)
        samples.append({"sample_id": sample_id, "modality": "image_text", "input_sha256": input_hash, "contamination": []})
    text_samples = []
    if text_records:
        digest = "f" * 64
        stats.record_sample("mtp-text:" + digest, modality="text", input_sha256=digest)
        text_samples.append({
            "sample_id": "mtp-text:" + digest,
            "modality": "text",
            "input_sha256": digest,
            "source_record_ids": list(text_records),
            "contamination": [],
        })
    stats.save(path / "calibration/moments")
    unit = {"state": "completed", "eligible_for_merge": True, "source_unchanged": True,
            "binding": stats.binding, "coverage": stats.coverage(), "image_samples": samples, "text_samples": text_samples}
    campaign = {"status": "completed", "source_unchanged": True, "route": "ordinary", "active_sessions": 1,
                "dependencies_unchanged": True, "dependencies": stats.binding["dependencies"],
                "functional_passed": True, "target": {"fingerprint": "a" * 64},
                "source": {"sha256": "c" * 64}, "calibration_unit": unit}
    (path / "calibration/unit.json").write_text(json.dumps(unit))
    (path / "campaign.json").write_text(json.dumps(campaign))
    return path


def test_units_merge_disjoint_samples_but_reject_failed_or_mutated_execution(merger, tmp_path):
    first, second = make_unit(tmp_path, 0, 2), make_unit(tmp_path, 2, 2)
    kwargs = {"specs": {"bank": (2, 3)}, "artifact": "a" * 64, "corpus": "b" * 64}
    stats = merger.load_unit(first, **kwargs)
    stats.merge(merger.load_unit(second, **kwargs))
    assert len(stats.samples) == 4
    with pytest.raises(ValueError, match="double-count"):
        stats.merge(merger.load_unit(second, **kwargs))
    with pytest.raises(ValueError, match="another source"):
        merger.load_unit(first, **{**kwargs, "artifact": "d" * 64})
    unit_file = second / "calibration/unit.json"
    unit = json.loads(unit_file.read_text())
    unit["state"] = "failed"
    unit_file.write_text(json.dumps(unit))
    with pytest.raises(ValueError):
        merger.load_unit(second, **kwargs)


def test_merge_rejects_record_level_mtp_overlap(merger, tmp_path):
    first = make_unit(tmp_path, 0, 2, text_records=("image:0",))
    second = make_unit(tmp_path, 2, 2, text_records=("image:2",))
    assert merger.text_record_ids(
        json.loads((first / "calibration/unit.json").read_text())
    ) == ("image:0",)
    unit = json.loads((second / "calibration/unit.json").read_text())
    unit["text_samples"][0]["source_record_ids"] = ["image:2", "image:2"]
    with pytest.raises(ValueError, match="disjoint|double-counts"):
        merger.text_record_ids(unit)


def test_merge_rejects_mtp_record_overlap_across_units(merger):
    def unit(image_id):
        return {
            "image_samples": [{"sample_id": image_id}],
            "text_samples": [{"source_record_ids": [image_id]}],
        }

    seen = set()
    merger.claim_text_record_ids(seen, unit("image:7"))
    assert seen == {"image:7"}
    with pytest.raises(ValueError, match="across calibration units"):
        merger.claim_text_record_ids(seen, unit("image:7"))


@pytest.mark.parametrize("count,expert_count,hybrid,fully_weighted", [
    (1024, 1024, True, True), (1023, 1024, False, False), (1024, 1, True, False)])
def test_publication_requires_all_images_and_records_expert_fallback(
        merger, monkeypatch, tmp_path, count, expert_count, hybrid, fully_weighted):
    path = make_unit(tmp_path, 0, count, expert_count=expert_count)
    artifact = {"identity": {"fingerprint": "a" * 64}, "config": {}, "conversion": {"expert_bits": 4}}
    corpus = {"corpus_sha256": "b" * 64}
    monkeypatch.setattr(merger, "inspect_artifact", lambda _: artifact)
    monkeypatch.setattr(merger, "verify_corpus", lambda _: (corpus, [{"sample_id": f"image:{i}"} for i in range(1024)]))
    monkeypatch.setattr(merger, "make_plan", lambda *a, **kw: {})
    monkeypatch.setattr(merger, "statistics_specs", lambda _: {"bank": (2, 3)})
    destination, report = tmp_path / "merged", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [str(merger.__file__), "--model", str(tmp_path / "target"),
        "--corpus", str(tmp_path / "corpus"), "--destination", str(destination), "--report", str(report), str(path)])
    merger.main()
    receipt = json.loads(report.read_text())
    assert receipt["ready_for_hybrid_conversion"] is hybrid
    assert receipt["ready_for_weighted_conversion"] is fully_weighted
    assert destination.exists() is hybrid
    assert receipt["qualified"] is False
    if hybrid:
        assert len(ActivationMoments.load(destination, specs={"bank": (2, 3)}).samples) == 1024
