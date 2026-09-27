"""The audio CLI must fail closed on unreviewed or stale route evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import runpy
from pathlib import Path

import pytest

from mlx2.adapters.nemotron3_diarization import MODEL_REVISION
from mlx2.diarization_qualification import (
    APPROVED_DIARIZATION_HARNESS,
    APPROVED_DIARIZATION_PLAN_SHA256,
    APPROVED_DIARIZATION_PLAN_SHA256S,
    APPROVED_NEMO_SCORER_REVISION,
    load_qualified_diarization_cli,
)


def _receipt():
    return {"schema": "mlx2.diarization-cli-qualification.v2",
            "status": "qualified_cli", "qualification_harness": APPROVED_DIARIZATION_HARNESS,
            "state_operation": "approximate_speaker_cache_v1", "reference_device": "mps",
            "plan_sha256": APPROVED_DIARIZATION_PLAN_SHA256,
            "manifest_sha256": "corpus", "lifecycle_sha256": "lifecycle",
            "nemo_speech_revision": APPROVED_NEMO_SCORER_REVISION,
            "model_revision": MODEL_REVISION, "artifact_sha256": "weights",
            "adapter_source_sha256": "adapter", "settings": {
                "dtype": "float32", "attention": "flash", "threshold": 0.5, "min_frames": 0},
            "runtime": {"mlx": importlib.metadata.version("mlx"),
                        "machine": platform.machine()},
            "profiles": {name: {"candidate_receipt_sha256": "candidate",
                                "official_nemo_scores": {
                                    corpus: {arm: {"der": 0.1} for arm in ("reference", "mlx")}
                                    for corpus in ("voxconverse-v0.3-test", "ami-test-sdm")}}
                         for name in ("offline",)}}


def test_cli_admission_is_exact_to_profile_settings_and_source(tmp_path):
    path = tmp_path / "receipt.json"
    record = _receipt()
    path.write_text(json.dumps(record))
    artifact = {"weight_sha256": "weights", "source_sha256": "adapter"}
    settings = {"artifact": artifact, "profile": "offline", "attention": "flash",
                "dtype": "float32", "threshold": 0.5, "min_frames": 0}
    route = load_qualified_diarization_cli(path, **settings)
    assert route["qualification"] == "qualified_cli"
    assert route["state_operation"] == "approximate_speaker_cache_v1"
    record["plan_sha256"] = hashlib.sha256((Path(__file__).resolve().parents[1] /
        "qualification/plans/nemotron3-diarization-v4-m3.json").read_bytes()).hexdigest()
    path.write_text(json.dumps(record))
    assert load_qualified_diarization_cli(path, **settings)["qualification"] == "qualified_cli"
    record["plan_sha256"] = "unreviewed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks matching qualification"):
        load_qualified_diarization_cli(path, **settings)
    record["plan_sha256"] = APPROVED_DIARIZATION_PLAN_SHA256
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks matching qualification"):
        load_qualified_diarization_cli(path, **(settings | {"profile": "low"}))
    with pytest.raises(ValueError, match="lacks matching qualification"):
        load_qualified_diarization_cli(path, **(settings | {"threshold": 0.6}))
    with pytest.raises(ValueError, match="lacks matching qualification"):
        load_qualified_diarization_cli(path, **(settings | {"artifact": {
            "weight_sha256": "weights", "source_sha256": "changed"}}))
    record["profiles"]["offline"]["official_nemo_scores"].pop("ami-test-sdm")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="lacks matching qualification"):
        load_qualified_diarization_cli(path, **settings)


def test_approved_plan_and_harness_hash_chain_is_current():
    root = Path(__file__).resolve().parents[1]

    def digest(relative_path):
        return hashlib.sha256((root / relative_path).read_bytes()).hexdigest()

    finalizer = runpy.run_path(str(root / "scripts/finalize_nemotron3_qualification.py"))
    assert APPROVED_DIARIZATION_PLAN_SHA256 == digest(
        "qualification/plans/nemotron3-diarization-v3-calibrated.json")
    assert APPROVED_DIARIZATION_PLAN_SHA256S == {
        digest("qualification/plans/nemotron3-diarization-v3-calibrated.json"),
        digest("qualification/plans/nemotron3-diarization-v4-m3.json"),
    }
    assert APPROVED_DIARIZATION_HARNESS["sha256"] == digest(
        "scripts/finalize_nemotron3_qualification.py")
    assert finalizer["APPROVED_CANDIDATE_HARNESS_SHA256"] == digest(
        "scripts/qualify_nemotron3_corpus.py")
    assert finalizer["APPROVED_LIFECYCLE_HARNESS_SHA256"] == digest(
        "scripts/check_nemotron3_lifecycle.py")
