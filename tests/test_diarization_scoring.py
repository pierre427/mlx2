"""Meaningful overlap, permutation, collar, and corpus-input checks."""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from mlx2.diarization_scoring import read_rttm, score_segments, write_rttm
from scripts.prepare_nemotron3_corpus import collect


def test_overlap_and_speaker_permutation_are_scored_correctly():
    reference = [(0, 2, "A"), (1, 3, "B")]
    predicted = [(0, 2, "other"), (1, 3, "first")]
    score = score_segments(reference, predicted, duration=3)
    assert score["der"] == 0
    assert score["reference_speaker_seconds"] == 4
    assert score["speaker_count_correct"]


def test_der_components_and_collar():
    reference = [(1, 2, "A")]
    predicted = [(0, 1.5, "X"), (1.5, 2.5, "Y")]
    score = score_segments(reference, predicted, duration=3)
    assert score["miss_seconds"] == 0
    assert score["false_alarm_seconds"] == 1.5
    assert score["confusion_seconds"] == pytest.approx(0.5)
    assert score["der"] == pytest.approx(2.0)
    collared = score_segments(reference, [(0.95, 2.05, "X")],
                              duration=3, collar=0.1)
    assert collared["der"] == 0
    assert collared["reference_speaker_seconds"] == pytest.approx(0.8)


def test_uem_excludes_outside_hypothesis():
    score = score_segments([(1, 2, "A")], [(0, 2, "X")],
                           duration=3, uem=[(1, 2)])
    assert score["der"] == 0


def test_rttm_roundtrip_and_manifest_hash_inputs(tmp_path):
    audio_dir, rttm_dir = tmp_path / "audio", tmp_path / "rttm"
    audio_dir.mkdir()
    rttm_dir.mkdir()
    with wave.open(str(audio_dir / "sample.wav"), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(np.zeros(16000, dtype="<i2").tobytes())
    write_rttm(rttm_dir / "sample.rttm", recording_id="sample",
               segments=[{"Start": 0.1, "End": 0.7, "Speaker": 0}])
    assert read_rttm(rttm_dir / "sample.rttm", recording_id="sample") == [
        (0.1, 0.7, "speaker_0")]
    row = collect("fixture", audio_dir, rttm_dir)[0]
    assert json.loads(json.dumps(row))["duration"] == 1.0
    (rttm_dir / "sample.rttm").unlink()
    with pytest.raises(ValueError, match="missing_rttm"):
        collect("fixture", audio_dir, rttm_dir)
