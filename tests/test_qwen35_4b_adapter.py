"""CPU-only admission checks for the local Qwen3.5 4B topology."""

import json
import sys
from types import SimpleNamespace

import pytest
from mlx_blocker import block_mlx_imports

from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability
from mlx2.adapters.qwen35_4b import Qwen354BAdapter


def _artifact(path):
    config = {
        "model_type": "qwen3_5",
        "num_hidden_layers": 32,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    (path / "model.safetensors").write_bytes(b"metadata-only test")
    return path


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


def test_4b_dispatch_is_ordinary_and_cpu_safe(tmp_path):
    path = _artifact(tmp_path)
    resolution = inspect_model(path)
    assert resolution.adapter_type.__name__ == "Qwen354BAdapter"
    assert resolution.default_route == "ordinary"
    assert resolution.descriptor.metadata["qualification"] == "pending"
    assert Capability.MTP not in resolution.descriptor.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(path, mtp=True)
    assert "mlx.core" not in sys.modules


def test_4b_requires_exact_topology_and_shard_closure(tmp_path):
    path = _artifact(tmp_path)
    (path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing weight shard"):
        inspect_model(path)
    (path / "model.safetensors").write_bytes(b"metadata-only test")
    config = json.loads((path / "config.json").read_text())
    config["linear_num_value_heads"] = 64
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="topology"):
        inspect_model(path)


def test_4b_budget_receipt_uses_4b_geometry_and_refuses_mtp():
    adapter = object.__new__(Qwen354BAdapter)
    adapter.model = SimpleNamespace(args=SimpleNamespace(text_config={
        "num_hidden_layers": 32, "hidden_size": 2560,
        "num_attention_heads": 16, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
    }))
    assert adapter.cache_budget(mtp=False).as_dict()["schema"] == (
        "qwen35-4b-cache-geometry-v1"
    )
    with pytest.raises(ValueError, match="MTP is not implemented"):
        adapter.cache_budget(mtp=True)
