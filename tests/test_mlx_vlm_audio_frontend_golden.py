"""Pin the transformers audio front ends the mlx-vlm adapters encode with.

Gemma 3n loads ``Gemma3nAudioFeatureExtractor`` through ``AutoFeatureExtractor``
and MiniCPM-o builds ``WhisperFeatureExtractor``; both come from transformers,
not mlx-vlm.  transformers #48114 moves every audio front end to new
``AudioProcessor`` classes, which a routine upgrade within ``transformers<6``
would pull in.  These golden values were recorded under transformers 5.17.0;
a front end that changes the features fails here before it can serve.
"""

import glob
import os
from types import SimpleNamespace

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def _snapshot(repo):
    found = glob.glob(os.path.join(HUB, f"models--{repo.replace('/', '--')}", "snapshots", "*"))
    if not found:
        pytest.skip(f"{repo} snapshot not present")
    return found[0]


def _wave(n=20_837, sample_rate=16_000):
    # 20,837 samples: not a multiple of any hop, so the tail handling counts.
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * (300 + 900 * t) * t)
        + 0.05 * np.sin(2 * np.pi * 3100 * t)
    ).astype(np.float32)


GOLDEN = {
    "gemma3n": {
        "class": "Gemma3nAudioFeatureExtractor",
        "shape": (1, 128, 128),
        "mask_sum": 128,
        "stats": (-6.13652, 3.74858, -11.51293, 4.18331),
        "probe": (-2.2887, -2.72516, -8.1284),
    },
    "minicpmo": {
        "class": "WhisperFeatureExtractor",
        "shape": (1, 80, 3000),
        "mask_sum": 131,
        "stats": (-0.6166, 0.15489, -0.63545, 1.36455),
        "probe": (1.16087, -0.63545, -0.63545),
    },
}


def _extractor(name):
    if name == "gemma3n":
        return transformers.AutoFeatureExtractor.from_pretrained(_snapshot("google/gemma-3n-E2B-it"))
    return transformers.WhisperFeatureExtractor.from_pretrained(_snapshot("openbmb/MiniCPM-o-2_6"))


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_audio_frontend_matches_the_recorded_features(name):
    extractor, golden = _extractor(name), GOLDEN[name]
    assert type(extractor).__name__ == golden["class"]
    kwargs = {"return_attention_mask": True} if name == "minicpmo" else {}
    out = extractor(_wave(), sampling_rate=16_000, return_tensors="np", **kwargs)
    features = out["input_features"]
    mask = out.get("input_features_mask", out.get("attention_mask"))
    assert tuple(features.shape) == golden["shape"]
    assert int(np.asarray(mask).sum()) == golden["mask_sum"]
    stats = (features.mean(), features.std(), features.min(), features.max())
    np.testing.assert_allclose(stats, golden["stats"], atol=2e-4)
    probe = [features.reshape(-1)[i] for i in (0, 777, 4321)]
    np.testing.assert_allclose(probe, golden["probe"], atol=2e-4)


def test_media_fingerprint_names_the_audio_frontend():
    from mlx2.adapters.mlx_vlm import audio_frontend_identity

    extractor = _extractor("gemma3n")
    identity = audio_frontend_identity(SimpleNamespace(feature_extractor=extractor))
    assert identity["class"].endswith("Gemma3nAudioFeatureExtractor")
    assert identity["transformers"] == transformers.__version__
    assert identity["sampling_rate"] == 16_000 and identity["hop_length"] == 160
    assert audio_frontend_identity(SimpleNamespace()) is None
