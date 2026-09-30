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


def _naive_features(samples):
    """Frame-by-frame reference: pre-emphasis, 256-sample centre padding,
    one 512-point frame per 10 ms hop, Hann(400) centred, Slaney mel, log."""
    from mlx2.adapters.nemotron3_diarization import _MEL_FILTERS, _WINDOW

    emphasized = samples.astype(np.float32).copy()
    emphasized[1:] -= 0.97 * samples[:-1]
    padded = np.pad(emphasized, (256, 256))
    rows = []
    for index in range(len(samples) // 160):
        frame = padded[index * 160 : index * 160 + 512] * _WINDOW
        spectrum = np.fft.rfft(frame)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        rows.append(np.log(power @ _MEL_FILTERS.T + 2**-24))
    return np.stack(rows)


@pytest.mark.parametrize("tail", [0, 95, 96, 159])
def test_features_keep_the_final_frame_for_every_tail(tail):
    """transformers #49167: a streaming front end that pads each chunk only by
    its hop drops the final STFT frame.  mlx2 pads the whole utterance once
    after pre-emphasis, so every tail keeps ``len // 160`` frames and the last
    one reads the zero-padded end exactly as the frame-by-frame reference."""
    rng = np.random.default_rng(tail)
    samples = rng.standard_normal(64_000 + tail).astype(np.float32) * 0.1
    features = extract_features(samples)
    expected = _naive_features(samples)
    assert features.shape == (len(samples) // 160, 128)
    np.testing.assert_allclose(features, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(features[-1], expected[-1], rtol=1e-4, atol=1e-4)
