"""CPU admission checks for the 36-row dense-FC coarse-head capture."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts/research/coarse_head_densefc_full_capture.py"
SPEC = importlib.util.spec_from_file_location("coarse_head_densefc_full_capture", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def artifact(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "text_config": {"vocab_size": 248320}, "tie_word_embeddings": False,
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64}}))
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"language_model.lm_head.weight": "a.safetensors",
                       "language_model.mtp.fc.weight": "b.safetensors"}}))
    return model


def test_cpu_preflight_pins_all_strata_without_importing_mlx(monkeypatch, tmp_path):
    model = artifact(tmp_path)
    monkeypatch.setattr(MODULE, "corpus_tokens", lambda _path: np.arange(68547, dtype=np.uint32))
    before = sys.modules.get("mlx.core")
    receipt = MODULE.preflight(model, tmp_path / "rows.npz")
    assert receipt["rows"] == 36
    assert receipt["contexts"] == [128, 8192, 16384, 65536]
    assert receipt["seeds"] == [11, 23, 37]
    assert receipt["raw_logits_bytes"] == 36 * 248320 * 4 * 2
    assert receipt["corpus_token_count"] >= 65536
    assert receipt["capture_code_sha256"] == MODULE.pilot.sha(RUNNER)
    assert sys.modules.get("mlx.core") is before


def test_cpu_preflight_refuses_short_corpus_and_existing_output(monkeypatch, tmp_path):
    model = artifact(tmp_path)
    out = tmp_path / "rows.npz"
    def short_corpus(_path):
        raise ValueError("corpus yielded 100 tokens, need at least 65536")
    monkeypatch.setattr(MODULE, "corpus_tokens", short_corpus)
    with pytest.raises(ValueError, match="corpus yielded"):
        MODULE.preflight(model, out)
    out.touch()
    with pytest.raises(ValueError, match="already exists"):
        MODULE.preflight(model, out)
