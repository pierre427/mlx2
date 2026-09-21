"""CPU/static Qwen3.6 port tests; these never load production tensors."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from mlx2.adapters.qwen36_35b import (
    CACHE_LAYOUT,
    Qwen3635BA3BAdapter,
    configure_environment,
    descriptor_for,
    inspect_artifact,
)
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability


CONFIG = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144,
    },
}


def make_artifact(root: Path, *, mtp: bool = False) -> Path:
    (root / "config.json").write_text(json.dumps(CONFIG))
    weights = {"language_model.model.embed_tokens.weight": "model.safetensors"}
    if mtp:
        names = [
            "fc.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate.weight",
            "layers.0.mlp.switch_mlp.gate_proj.weight",
            "layers.0.mlp.switch_mlp.up_proj.weight",
            "layers.0.mlp.switch_mlp.down_proj.weight",
        ]
        weights.update(
            {f"language_model.mtp.{name}": "model.safetensors" for name in names}
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (root / "model.safetensors").write_bytes(b"metadata-only")
    return root


def test_import_is_gpu_free():
    code = """import sys
import mlx2.adapters.qwen36_35b
assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)
assert 'mlx2.runtime.models.qwen36_35b' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ordinary_artifact_registry_and_descriptor():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory))
        artifact = inspect_artifact(root)
        resolved = inspect_model(root)
        assert not artifact["has_mtp"]
        assert resolved.adapter_type is Qwen3635BA3BAdapter
        assert resolved.default_route == "ordinary"
        assert resolved.descriptor.cache_layout == CACHE_LAYOUT
        assert Capability.APC_V2 in resolved.descriptor.capabilities
        assert Capability.MTP not in resolved.descriptor.capabilities


def test_embedded_mtp_requires_complete_sparse_head():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory), mtp=True)
        assert inspect_artifact(root)["has_mtp"]
        resolved = inspect_model(root)
        assert Capability.MTP in resolved.descriptor.capabilities
        # Measured default: native MTP, sound only because the wide-cohort
        # ordinary handoff recovers the batched loss (see the adapter note).
        assert resolved.default_route == "native_mtp"
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"][
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"
        ]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="incomplete"):
            inspect_artifact(root)


def test_wrong_topology_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory))
        config = json.loads((root / "config.json").read_text())
        config["text_config"]["num_experts"] = 128
        (root / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="topology"):
            inspect_artifact(root)


def test_baseline_environment_selects_stock_moe():
    with patch.dict(os.environ, {"MLX_QWEN4_MOE_ROUTER_KERNEL": "1"}):
        profile = configure_environment()
        assert profile["MLX_LM_COMPILED_DECODE"] == "0"
        assert profile["MLX_QWEN36_FUSED_GDN_DECODE"] == "0"
        assert profile["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "0"
        assert profile["MLX_QWEN4_FUSED_EXPERT_KERNEL"] == "stock"
        assert os.environ["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "0"


def test_profile_names_remain_unqualified_candidates():
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter._num_draft = 2
    adapter.descriptor = descriptor_for(has_mtp=False)
    assert adapter.profile_name(False) == "qwen36-35b-a3b-apcv2-ordinary"
    with pytest.raises(ValueError, match="embedded head"):
        adapter.profile_name(True)
    adapter.descriptor = descriptor_for(has_mtp=True)
    assert adapter.profile_name(True) == "qwen36-35b-a3b-apcv2-mtp2"


def test_tensor_module_compiles_and_has_no_apple_header():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/models/qwen36_35b.py"
    source = path.read_text()
    compile(source, str(path), "exec")
    assert "Copyright © 2026 Apple" not in source


def test_fused_gdn_geometry_is_qwen36_specific_and_opt_in():
    import mlx.core as mx
    from mlx2.runtime.models.qwen4_fused_gdn import admit_qwen4_fused_gdn_decode

    bf16 = mx.bfloat16
    admission = admit_qwen4_fused_gdn_decode(
        qkv=mx.zeros((1, 1, 8192), dtype=bf16),
        z=mx.zeros((1, 1, 4096), dtype=bf16),
        b=mx.zeros((1, 1, 32), dtype=bf16),
        a=mx.zeros((1, 1, 32), dtype=bf16),
        conv_state=mx.zeros((1, 3, 8192), dtype=bf16),
        recurrent_state=mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        conv_weight=mx.zeros((8192, 4, 1), dtype=bf16),
        A_log=mx.zeros((32,), dtype=mx.float32),
        dt_bias=mx.zeros((32,), dtype=bf16),
        norm_weight=mx.ones((128,), dtype=bf16),
        mask=None,
        spans=(),
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="swish",
        architecture="qwen35",
    )
    assert admission.accepted

    wrong_architecture = admit_qwen4_fused_gdn_decode(
        qkv=mx.zeros((1, 1, 8192), dtype=bf16),
        z=mx.zeros((1, 1, 4096), dtype=bf16),
        b=mx.zeros((1, 1, 32), dtype=bf16),
        a=mx.zeros((1, 1, 32), dtype=bf16),
        conv_state=mx.zeros((1, 3, 8192), dtype=bf16),
        recurrent_state=mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        conv_weight=mx.zeros((8192, 4, 1), dtype=bf16),
        A_log=mx.zeros((32,), dtype=mx.float32),
        dt_bias=mx.zeros((32,), dtype=bf16),
        norm_weight=mx.ones((128,), dtype=bf16),
        mask=None,
        spans=(),
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="swish",
        architecture="agnes",
    )
    assert not wrong_architecture.accepted
    assert "geometry" in wrong_architecture.reason
