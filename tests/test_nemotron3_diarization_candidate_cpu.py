"""CPU-only source-bound diarization checks; no MLX import or model load."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.nemotron3_diarization import (
    STREAMING_PROFILES, extract_features, inspect_artifact, segments_from_logits,
)


PATH = "/Volumes/T7/models/Nemotron-3-Diarization"

pytestmark = pytest.mark.skipif(
    not all(((Path(PATH) / "config.json").is_file(), (Path(PATH) / "processor_config.json").is_file())),
    reason="optional Nemotron diarization model fixture is absent",
)


def test_artifact_profiles_frontend_and_overlap():
    receipt = inspect_artifact(PATH)
    assert receipt["model_type"] == "nemotron3_diarization"
    assert receipt["qualification"] == "pending"
    assert set(STREAMING_PROFILES) == {"offline", "low", "very_low", "ultra_low"}
    features = extract_features(np.zeros(16000, dtype=np.float32))
    assert features.shape[1] == 128
    logits = np.full((5, 8), -10, dtype=np.float32)
    logits[:3, 0] = 10
    logits[1:, 1] = 10
    assert [item["Speaker"] for item in segments_from_logits(logits)] == [0, 1]


def test_inspection_blocks_real_mlx():
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.nemotron3_diarization import inspect_artifact
inspect_artifact(sys.argv[1])
assert "mlx.core" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, PATH], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

