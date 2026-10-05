"""Laguna S artifact checks without importing MLX; run with --noconftest."""

import json
import os
import sys
from pathlib import Path

import pytest
from mlx_blocker import block_mlx_imports



@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


S4 = Path(os.environ.get("MLX2_LAGUNA_S4_ARTIFACT", "/Volumes/T7/models/poolside/Laguna-S-2.1-MLX-4bit"))
SBF = Path(os.environ.get("MLX2_LAGUNA_SBF_ARTIFACT", "/Volumes/T7/models/poolside/Laguna-S-2.1-bf16"))


def _fixture(source, destination):
    if not source.is_dir():
        pytest.skip(f"artifact absent: {source}")
    for name in ("config.json", "model.safetensors.index.json"):
        (destination / name).write_bytes((source / name).read_bytes())
    weights = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(weights.values()):
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(b"metadata")


@pytest.mark.parametrize("source,quantized,count", [(S4, True, 1962), (SBF, False, 36769)])
def test_s_artifact_metadata_and_route(source, quantized, count):
    if not source.is_dir():
        pytest.skip(f"artifact absent: {source}")
    from mlx2.adapters.laguna_s21 import LAGUNA_S21, LagunaS21Adapter, inspect_artifact
    from mlx2.contracts import Capability

    result = inspect_artifact(source)
    assert result["quantized"] is quantized
    assert len(result["weight_map"]) == count
    assert result["layers"] == 48 and result["sliding_layers"] == 36
    assert not result["supports_native_mtp"]
    assert Capability.MTP not in LAGUNA_S21.capabilities
    assert Capability.EXTERNAL_DRAFT not in LAGUNA_S21.capabilities
    assert LagunaS21Adapter.default_route == "ordinary"
    assert LagunaS21Adapter.profile_name(False) == "laguna-s21-apcv2-ordinary"
    assert "mlx.core" not in sys.modules


def test_wrong_head_order_fails_before_model_import(tmp_path):
    _fixture(S4, tmp_path)
    from mlx2.adapters.laguna_s21 import inspect_artifact

    config = json.loads((tmp_path / "config.json").read_text())
    config["num_attention_heads_per_layer"][1] = 48
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="attention-head order"):
        inspect_artifact(tmp_path)


def test_bf16_missing_expert_weight_fails(tmp_path):
    _fixture(SBF, tmp_path)
    from mlx2.adapters.laguna_s21 import inspect_artifact

    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"]["model.layers.47.mlp.experts.255.down_proj.weight"]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="ordinary trunk tensors"):
        inspect_artifact(tmp_path)


@pytest.mark.parametrize(
    "source,key",
    [
        (S4, "model.layers.47.mlp.gate.e_score_correction_bias"),
        (SBF, "model.layers.47.mlp.experts.e_score_correction_bias"),
        (S4, "model.layers.47.mlp.shared_expert.up_proj.weight"),
        (SBF, "model.layers.47.mlp.shared_expert.down_proj.weight"),
    ],
)
def test_missing_router_or_shared_expert_tensor_fails_before_model_import(tmp_path, source, key):
    _fixture(source, tmp_path)
    from mlx2.adapters.laguna_s21 import inspect_artifact

    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    del index["weight_map"][key]
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="ordinary trunk tensors"):
        inspect_artifact(tmp_path)
    assert "mlx.core" not in sys.modules


def test_speculative_policy_rejected_before_weight_loading():
    from mlx2.adapters.laguna_s21 import LagunaS21Adapter

    with pytest.raises(ValueError, match="no draft execution policy"):
        LagunaS21Adapter("/nonexistent", execution_policy={"draft_model": "x"})
