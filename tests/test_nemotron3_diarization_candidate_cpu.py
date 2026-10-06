"""CPU-only source-bound diarization checks; no MLX import or model load."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.nemotron3_diarization import (
    STREAMING_PROFILES,
    extract_features,
    inspect_artifact,
    segments_from_logits,
)

PATH = "/Volumes/T7/models/Nemotron-3-Diarization"

artifact_required = pytest.mark.skipif(
    not all(((Path(PATH) / "config.json").is_file(), (Path(PATH) / "processor_config.json").is_file())),
    reason="optional Nemotron diarization model fixture is absent",
)


@artifact_required
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


@artifact_required
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
    result = subprocess.run(
        [sys.executable, "-c", script, PATH],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("explicit", (None, "1"))
def test_f32_adapter_applies_process_numerics_before_mlx_and_preserves_explicit_tf32(
    explicit,
):
    script = textwrap.dedent(
        r'''
        import json, os, sys
        value = sys.argv[1]
        if value == "<unset>":
            os.environ.pop("MLX_ENABLE_TF32", None)
        else:
            os.environ["MLX_ENABLE_TF32"] = value
        import mlx2.adapters.nemotron3_diarization as module
        events = []
        real_apply = module.apply_process_numerics
        def observed_apply():
            events.append({
                "mlx_imported": "mlx.core" in sys.modules,
                "before": os.environ.get("MLX_ENABLE_TF32"),
            })
            result = real_apply()
            events[-1]["after"] = os.environ.get("MLX_ENABLE_TF32")
            return result
        module.apply_process_numerics = observed_apply
        try:
            module.Nemotron3DiarizationAdapter(
                "/definitely/missing/nemotron-diarization",
                dtype="float32",
                verify_hash=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing artifact should fail after the import-order probe")
        print(json.dumps(events))
        '''
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-c", script, explicit or "<unset>"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    events = json.loads(result.stdout)
    assert events == [{
        "mlx_imported": False,
        "before": "0" if explicit is None else explicit,
        "after": "0" if explicit is None else explicit,
    }]
