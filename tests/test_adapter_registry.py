"""Metadata dispatch and capability rejection must happen before GPU loading."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability


def artifact(tmp_path, config, weights=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"not loaded")
    if weights is not None:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors" for key in weights}})
        )
    return tmp_path


def test_registry_import_is_gpu_free():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx2.adapters.registry; assert 'mlx.core' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr


def test_muse_dispatch_is_metadata_based(tmp_path):
    path = artifact(tmp_path, {"model_type": "muse_glimmer"})
    result = inspect_model(path)
    assert result.adapter_type.__name__ == "MuseGlimmerAdapter"
    assert result.descriptor.metadata["qualification"] == "pending"
    assert Capability.MTP not in result.descriptor.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(path, mtp=True)


@pytest.mark.parametrize("model_type", ["unknown", "qwen3", None])
def test_unknown_target_fails_closed(tmp_path, model_type):
    path = artifact(tmp_path, {"model_type": model_type})
    with pytest.raises(ValueError, match="No mlx2 adapter"):
        resolve_adapter(path)


def test_drafter_cannot_masquerade_as_target(tmp_path):
    path = artifact(
        tmp_path, {"model_type": "muse_glimmer", "architectures": ["DFlash2DraftModel"]}
    )
    with pytest.raises(ValueError, match="standalone target"):
        resolve_adapter(path)


def test_empty_drafter_metadata_is_still_a_drafter(tmp_path):
    path = artifact(tmp_path, {"model_type": "muse_glimmer", "dflash_config": {}})
    with pytest.raises(ValueError, match="standalone target"):
        resolve_adapter(path)


def test_flash_current_layout_and_actual_mtp_keys(tmp_path):
    path = artifact(tmp_path, {"model_type": "qwen4_exp"}, ["mtp.fc.weight"])
    (path / "ple_rows.bin").write_bytes(b"not loaded")
    assert resolve_adapter(path, mtp=True).__name__ == "FlashNextAdapter"
    artifact(tmp_path, {"model_type": "qwen4_exp"}, ["language_model.weight"])
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(path, mtp=True)


def test_flash_legacy_layout_rejected(tmp_path):
    path = artifact(
        tmp_path,
        {"model_type": "qwen4_exp", "ngram_table": {"x": 1}},
        ["mtp.fc.weight"],
    )
    (path / "ple_rows.bin").write_bytes(b"not loaded")
    with pytest.raises(ValueError, match="current unified artifact"):
        resolve_adapter(path)


def test_qwen_dense_topology_cannot_be_faked_by_model_type(tmp_path):
    path = artifact(
        tmp_path, {"model_type": "qwen3_5", "num_hidden_layers": 1}, ["mtp.fc.weight"]
    )
    with pytest.raises(ValueError, match="topology"):
        resolve_adapter(path)


def test_qwen_ordinary_dispatch_and_mtp_refusal(tmp_path):
    config = {
        "model_type": "qwen3_5",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    path = artifact(tmp_path, config, ["model.embed_tokens.weight"])
    assert resolve_adapter(path).__name__ == "Qwen3827BAdapter"
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(path, mtp=True)


@pytest.mark.parametrize(
    ("model_type", "adapter_name", "required_capabilities"),
    [
        ("gemma3n", "Gemma3nAdapter", {Capability.VISION, Capability.VIDEO, Capability.AUDIO}),
        ("minicpmo", "MiniCPMOAdapter", {Capability.VISION, Capability.AUDIO}),
    ],
)
def test_multimodal_registry_is_metadata_only_and_fail_closed_for_mtp(
    tmp_path, model_type, adapter_name, required_capabilities
):
    path = artifact(
        tmp_path,
        {
            "model_type": model_type,
            "max_position_embeddings": 4096,
        },
    )
    result = inspect_model(path)
    assert result.adapter_type.__name__ == adapter_name
    assert required_capabilities <= result.descriptor.capabilities
    assert result.descriptor.metadata["qualification"] == "pending"
    assert Capability.OUTPUT_AUDIO not in result.descriptor.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        resolve_adapter(path, mtp=True)
