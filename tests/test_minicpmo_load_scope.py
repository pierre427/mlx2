"""Released MiniCPM-o loader scope without loading model weights."""

import json
from types import SimpleNamespace

import pytest

from mlx2.adapters.mlx_vlm import _load_model


def test_minicpmo_load_excludes_unserved_tts_and_restores_globals(tmp_path, monkeypatch):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.minicpmo import minicpmo as module
    from mlx_vlm.models.minicpmo.config import ModelConfig

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"language_model.self_attn.q_proj.bias": "weights.safetensors"}})
    )
    seen = {}

    def original_config(cls, params):
        seen.update(params)
        return SimpleNamespace(text_config=SimpleNamespace(rope_traditional=True))

    monkeypatch.setattr(ModelConfig, "from_dict", classmethod(original_config))
    descriptor = ModelConfig.__dict__["from_dict"]
    language_model = module.LanguageModel

    def fake_load(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"lazy": False}  # Keep mlx-vlm's strict default.
        assert module.LanguageModel is not language_model
        return ModelConfig.from_dict({}), object()

    _load_model(fake_load, tmp_path, expected="minicpmo",
                config={"hidden_size": 16, "num_attention_heads": 2})

    assert seen["init_tts"] is False
    assert seen["head_dim"] == 8
    assert seen["attention_bias"] is True
    assert ModelConfig.__dict__["from_dict"] is descriptor
    assert module.LanguageModel is language_model
