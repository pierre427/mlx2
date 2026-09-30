"""Nemotron 3 Diarization log-mel front end; NumPy only, no artifact needed."""

import numpy as np
import pytest

from mlx2.adapters.nemotron3_diarization import extract_features


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
