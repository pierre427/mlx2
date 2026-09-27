"""Fail-closed admission of independently scored CLI diarization profiles."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

from .adapters.nemotron3_diarization import MODEL_REVISION

APPROVED_DIARIZATION_HARNESS = {
    "name": "scripts/finalize_nemotron3_qualification.py",
    "sha256": "19b16d75fcea912aa370c53af119ed9d4f4ec1072c5394dbb0c323c60c30fc02",
}
APPROVED_DIARIZATION_PLAN_SHA256 = "fda6393900eb803df5c754861131887b80a2b91d371701c28de1a94cdf2f48d6"
APPROVED_DIARIZATION_PLAN_SHA256S = frozenset({
    APPROVED_DIARIZATION_PLAN_SHA256,
    "7387475a715cc538012922e039ce07c413e1ee6f75cba500edc8cbaea13dc326",
})
APPROVED_NEMO_SCORER_REVISION = "cf724ac337d1ebc7d0dda1e23fb80916f52927a5"
REQUIRED_DIARIZATION_PROFILES = {"offline", "low", "very_low", "ultra_low"}
REQUIRED_DIARIZATION_CORPORA = {"voxconverse-v0.3-test", "ami-test-sdm"}


def load_qualified_diarization_cli(path: str | Path, *, artifact: dict,
                                   profile: str, attention: str, dtype: str,
                                   threshold: float, min_frames: int) -> dict:
    """Return a source-bound route receipt, or refuse unsupported settings."""
    record = json.loads(Path(path).read_text())
    settings = {"dtype": dtype, "attention": attention,
                "threshold": threshold, "min_frames": min_frames}
    profiles = record.get("profiles", {})
    if (record.get("schema") != "mlx2.diarization-cli-qualification.v2" or
            record.get("status") != "qualified_cli" or
            record.get("state_operation") != "approximate_speaker_cache_v1" or
            record.get("reference_device") != "mps" or
            record.get("qualification_harness") != APPROVED_DIARIZATION_HARNESS or
            record.get("plan_sha256") not in APPROVED_DIARIZATION_PLAN_SHA256S or
            record.get("nemo_speech_revision") != APPROVED_NEMO_SCORER_REVISION or
            not record.get("manifest_sha256") or
            not record.get("lifecycle_sha256") or
            record.get("model_revision") != MODEL_REVISION or
            record.get("artifact_sha256") != artifact["weight_sha256"] or
            record.get("adapter_source_sha256") != artifact["source_sha256"] or
            record.get("settings") != settings or
            record.get("runtime", {}).get("mlx") != importlib.metadata.version("mlx") or
            record.get("runtime", {}).get("machine") != platform.machine() or
            not profiles or not set(profiles).issubset(REQUIRED_DIARIZATION_PROFILES) or
            profile not in profiles or
            any(not item.get("candidate_receipt_sha256") or
                set(item.get("official_nemo_scores", {})) != REQUIRED_DIARIZATION_CORPORA or
                any(set(arms) != {"reference", "mlx"}
                    for arms in item["official_nemo_scores"].values())
                for item in profiles.values())):
        raise ValueError("diarization CLI route lacks matching qualification evidence")
    return {"qualification": "qualified_cli", "qualification_receipt": str(Path(path).resolve()),
            "state_operation": record["state_operation"],
            "profile": profile, "attention": attention, "dtype": dtype,
            "artifact_sha256": artifact["weight_sha256"],
            "source_sha256": artifact["source_sha256"]}
