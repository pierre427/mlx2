import json
from types import SimpleNamespace

import pytest

from mlx2.adapters import deepseek_v4_gguf_artifact as artifact
from mlx2.adapters.deepseek_v4_gguf import (
    DESCRIPTOR,
    DeepSeekV4GGUFAdapter,
    validate_execution_policy,
)
from mlx2.adapters.deepseek_v4_gguf_artifact import validate_header
from mlx2.contracts import Capability


def header():
    metadata = {
        "general.architecture": "deepseek4",
        "deepseek4.checkpoint_variant": "vision-exp",
        "deepseek4.block_count": 43,
        "deepseek4.embedding_length": 4096,
        "deepseek4.vocab_size": 129280,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.hash_layer_count": 3,
        "deepseek4.attention.head_count": 64,
        "deepseek4.attention.key_length": 512,
        "deepseek4.attention.value_length": 512,
        "deepseek4.attention.sliding_window": 128,
        "deepseek4.hyper_connection.count": 4,
        "deepseek4.vision.sidecar_required": True,
    }
    tensors = {}
    for layer in range(43):
        for kind in ("gate", "up", "down"):
            down = kind == "down"
            tensors[f"blk.{layer}.ffn_{kind}_exps.weight"] = SimpleNamespace(
                shape=(2048, 4096, 256) if down else (4096, 2048, 256),
                format="Q2_K" if down else "IQ2_XXS",
            )
    tensors.update({f"other{i}": None for i in range(1328 - len(tensors))})
    return metadata, tensors


def test_header_requires_actual_block_format_not_bit_width():
    metadata, tensors = header()
    validate_header(metadata, tensors)
    tensors["blk.0.ffn_gate_exps.weight"].format = "affine2"
    with pytest.raises(ValueError, match="layout mismatch"):
        validate_header(metadata, tensors)


@pytest.mark.parametrize(
    "key,value",
    [
        ("deepseek4.checkpoint_variant", "0731"),
        ("deepseek4.block_count", 42),
        ("deepseek4.attention.key_length", 128),
        ("deepseek4.vision.sidecar_required", 1),
    ],
)
def test_incompatible_variant_and_topology_fail_closed(key, value):
    metadata, tensors = header()
    metadata[key] = value
    with pytest.raises(ValueError, match="topology"):
        validate_header(metadata, tensors)


def test_initial_capabilities_do_not_pretend_vision_or_speculation():
    assert Capability.TEXT in DESCRIPTOR.capabilities
    assert Capability.REASONING in DESCRIPTOR.capabilities
    assert (
        not {
            Capability.VISION,
            Capability.MTP,
            Capability.EXTERNAL_DRAFT,
            Capability.CONTINUOUS_BATCH,
            Capability.TOOLS,
        }
        & DESCRIPTOR.capabilities
    )
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    with pytest.raises(ValueError, match="B1"):
        adapter.execution_config(max_lanes=2, prefill_step=512)


def test_reasoning_mode_and_tool_refusals_precede_template_render():
    adapter = DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    assert adapter.thinking_enabled({"enable_thinking": True})
    assert not adapter.thinking_enabled({"enable_thinking": False})
    with pytest.raises(ValueError, match="boolean"):
        adapter.thinking_enabled({"enable_thinking": "false"})
    with pytest.raises(ValueError, match="tool execution"):
        adapter.prompt_tokens({"messages": [], "tools": [{"type": "function"}]})


def _adapter_config():
    return {
        "model_type": artifact.MODEL_TYPE,
        "mlx2_weight_format": "gguf-packed-v1",
        "gguf_file": artifact.FILENAME,
        "weight_sha256": artifact.WEIGHT_SHA256,
        "tokenizer_reference": artifact.REFERENCE_FILE,
    }


def test_artifact_refuses_symlinked_config_before_loading_weights(tmp_path):
    external = tmp_path / "external.json"
    external.write_text(json.dumps(_adapter_config()))
    (tmp_path / "config.json").symlink_to(external)
    with pytest.raises(ValueError, match="regular local file"):
        artifact.inspect_artifact(tmp_path)


@pytest.mark.parametrize(
    "field,value", [("schema", "other-v1"), ("filename", "other.gguf")]
)
def test_artifact_seal_identity_checked_before_gpu_or_header(tmp_path, field, value):
    (tmp_path / "config.json").write_text(json.dumps(_adapter_config()))
    seal = {"schema": artifact._SCHEMA, "filename": artifact.FILENAME}
    seal[field] = value
    (tmp_path / artifact.INTEGRITY_FILE).write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="schema or filename"):
        artifact.inspect_artifact(tmp_path)


def test_failed_construction_releases_uploaded_weight_owner(monkeypatch):
    closed = []

    def fail_after_weights(self, path):
        self.weights = SimpleNamespace(close=lambda: closed.append(True))
        self.model = object()
        raise RuntimeError("target allocation failed")

    monkeypatch.setattr(DeepSeekV4GGUFAdapter, "_load", fail_after_weights)
    with pytest.raises(RuntimeError, match="target allocation failed"):
        DeepSeekV4GGUFAdapter("unused")
    assert closed == [True]


def test_stale_acquisition_receipt_cannot_reseal_different_payload(
    tmp_path, monkeypatch
):
    weight = tmp_path / artifact.FILENAME
    weight.write_bytes(b"changed payload")
    monkeypatch.setattr(artifact, "WEIGHT_BYTES", weight.stat().st_size)
    receipt = {
        "status": "verified",
        "verified_sha256": artifact.WEIGHT_SHA256,
        "spec": {
            "sha256": artifact.WEIGHT_SHA256,
            "size": weight.stat().st_size,
            "filename": artifact.FILENAME,
        },
    }
    (tmp_path / (artifact.FILENAME + ".receipt.json")).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pinned SHA256"):
        artifact.prepare_artifact(tmp_path, "unused-tokenizer.json")
    assert not (tmp_path / artifact.INTEGRITY_FILE).exists()


@pytest.mark.parametrize(
    "policy",
    [
        {"deepseek_kv": "int8"},
        {"deepseek_embedding": True},
        {"unknown": "enable"},
        {"deepseek_packed": "affine2"},
        {"deepseek_rope_qat": "fp16"},
        {"deepseek_dense_prefill": "auto"},
    ],
)
def test_candidate_policy_cannot_alias_unsupported_math(policy):
    with pytest.raises(ValueError, match="execution policy"):
        validate_execution_policy(policy)


def test_candidates_require_explicit_operator_selection():
    default = validate_execution_policy(None)
    assert default["deepseek_kv"] == "fp32"
    assert default["deepseek_embedding"] == "resident"
    assert (
        default["deepseek_packed"]
        == default["deepseek_hc"]
        == default["deepseek_attention"]
        == default["deepseek_rope_qat"]
        == default["deepseek_dense_prefill"]
        == "reference"
    )
    selected = validate_execution_policy(
        {"deepseek_kv": "fp8_mixed", "deepseek_embedding": "file_rows"}
    )
    assert (
        selected["deepseek_kv"] == "fp8_mixed"
        and selected["deepseek_embedding"] == "file_rows"
    )
    assert validate_execution_policy({"deepseek_rope_qat": "fused_fp32"})["deepseek_rope_qat"] == "fused_fp32"
    assert validate_execution_policy({"deepseek_dense_prefill":"nax_fp16"})["deepseek_dense_prefill"]=="nax_fp16"
