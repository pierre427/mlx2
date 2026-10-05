"""Agnes metadata gates without importing MLX or loading model weights."""

import json
import os
import sys
from pathlib import Path

import pytest

from mlx_blocker import block_mlx_imports


ARTIFACT = Path(os.environ.get(
    "MLX2_AGNES_ARTIFACT",
    "~/mlx-models/Agnes-3.0-Flash-Preview-MLX-6bit",
))


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


@pytest.fixture
def indexed_metadata(tmp_path):
    if not ARTIFACT.is_dir():
        pytest.skip(f"local Agnes artifact absent: {ARTIFACT}")
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_bytes((ARTIFACT / name).read_bytes())
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        (tmp_path / name).write_bytes(b"metadata")
    return tmp_path


def test_complete_agnes_text_trunk_accepts_metadata(indexed_metadata):
    from mlx2.adapters.agnes_3_flash import inspect_artifact

    result = inspect_artifact(indexed_metadata)
    assert result["text_weight_count"] == 2725
    assert "mlx.core" not in sys.modules


@pytest.mark.parametrize("missing", [
    "language_model.model.layers.70.delta_attn.dt_bias",
    "language_model.model.layers.70.delta_attn.out_proj.scales",
    "language_model.model.layers.71.global_attn.k_norm.weight",
    "language_model.model.layers.71.mlp.parallel_ffn.down_proj.weight",
    "language_model.model.layers.71.mlp.parallel_ffn.gate_proj.biases",
])
def test_incomplete_text_trunk_fails_before_model_import(indexed_metadata, missing):
    from mlx2.adapters.agnes_3_flash import inspect_artifact

    index_path = indexed_metadata / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop(missing)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="text tensors are incomplete"):
        inspect_artifact(indexed_metadata)
    assert "mlx.core" not in sys.modules


@pytest.mark.parametrize("field,value", [
    ("hidden_act", "gelu"),
    ("tie_word_embeddings", True),
    ("linear_conv_kernel_dim", 8),
])
def test_unsupported_text_math_fails_before_model_import(indexed_metadata, field, value):
    from mlx2.adapters.agnes_3_flash import inspect_artifact

    config_path = indexed_metadata / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"][field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="text topology mismatch"):
        inspect_artifact(indexed_metadata)
    assert "mlx.core" not in sys.modules


def test_unsupported_rotary_math_fails_before_model_import(indexed_metadata):
    from mlx2.adapters.agnes_3_flash import inspect_artifact

    config_path = indexed_metadata / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"]["rope_parameters"]["mrope_interleaved"] = False
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="rotary topology mismatch"):
        inspect_artifact(indexed_metadata)
    assert "mlx.core" not in sys.modules
