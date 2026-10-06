"""CPU-only checks for the pinned OrcaSAQ GGUF conversion contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
safe_open = pytest.importorskip("safetensors").safe_open

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_orcasaq_qwen38_gguf.py"
SPEC = importlib.util.spec_from_file_location("convert_orcasaq_qwen38_gguf", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def test_mtp_and_recurrent_mapping_have_distinct_target_keys():
    mtp = converter.map_tensor("blk.64.nextn.eh_proj.weight", (5120, 10240))
    assert mtp == ("language_model.mtp.fc.weight", (5120, 10240), "identity")
    recurrent = converter.map_tensor("blk.0.ssm_conv1d.weight", (10240, 4))
    assert recurrent == ("language_model.model.layers.0.linear_attn.conv1d.weight", (10240, 4, 1), "conv_rows")
    with pytest.raises(ValueError):
        converter.map_tensor("blk.64.ssm_a", (48,))
    with pytest.raises(ValueError):
        converter.map_tensor("blk.3.ssm_a", (48,))


def test_all_mtp_keys_satisfy_adapter_inspection_contract():
    source_parts = [
        "attn_norm.weight", "post_attention_norm.weight", "attn_q.weight",
        "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "attn_q_norm.weight", "attn_k_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight", "nextn.eh_proj.weight",
        "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
    ]
    shape_by_part = {
        "attn_norm.weight": (5120,), "post_attention_norm.weight": (5120,),
        "attn_q.weight": (12288, 5120), "attn_k.weight": (1024, 5120),
        "attn_v.weight": (1024, 5120), "attn_output.weight": (5120, 6144),
        "attn_q_norm.weight": (256,), "attn_k_norm.weight": (256,),
        "ffn_gate.weight": (17408, 5120), "ffn_up.weight": (17408, 5120),
        "ffn_down.weight": (5120, 17408), "nextn.eh_proj.weight": (5120, 10240),
        "nextn.enorm.weight": (5120,), "nextn.hnorm.weight": (5120,),
        "nextn.shared_head_norm.weight": (5120,),
    }
    keys = {converter.map_tensor("blk.64." + part, shape_by_part[part])[0]
            for part in source_parts}
    from mlx2.adapters.qwen38_27b import inspect_artifact

    # This is the same required subset as header-only inspect_artifact, which
    # only checks MTP completeness after a full output index is available.
    required = {
        "language_model.mtp.fc.weight", "language_model.mtp.norm.weight",
        "language_model.mtp.pre_fc_norm_embedding.weight",
        "language_model.mtp.pre_fc_norm_hidden.weight",
        "language_model.mtp.layers.0.self_attn.q_proj.weight",
        "language_model.mtp.layers.0.self_attn.k_proj.weight",
        "language_model.mtp.layers.0.self_attn.v_proj.weight",
        "language_model.mtp.layers.0.self_attn.o_proj.weight",
        "language_model.mtp.layers.0.mlp.gate_proj.weight",
        "language_model.mtp.layers.0.mlp.up_proj.weight",
        "language_model.mtp.layers.0.mlp.down_proj.weight",
    }
    assert callable(inspect_artifact)
    assert len(keys) == 15
    assert required <= keys


def test_value_head_inverse_is_bijective_and_preserves_qk():
    destination = [converter.source_row(i, "v_rows_1") for i in range(48)]
    assert sorted(destination) == list(range(48))
    assert destination[:4] == [0, 16, 32, 1]
    assert [converter.source_row(i, "qkv_rows") for i in range(4096)] == list(range(4096))
    assert converter.source_row(4096 + 128, "qkv_rows") == 4096 + 16 * 128
    columns = converter.source_columns(6144)
    assert sorted(columns.tolist()) == list(range(6144))
    assert columns[128] == 16 * 128


def test_stream_writer_recovers_a_log_and_uses_valid_safetensors(tmp_path):
    # GGUF stores -exp(A_log) in tiled head order. A unique value per head
    # catches a sign error or an omitted/inverted head permutation.
    gguf_values = np.asarray([-np.exp(i / 16) for i in range(48)], dtype=np.float32)
    tensor = SimpleNamespace(name="blk.0.ssm_a", data=gguf_values,
                             tensor_type=gguf.GGMLQuantizationType.F32)
    path = tmp_path / "a.safetensors"
    record = converter.write_tensor(tensor, path, "language_model.model.layers.0.linear_attn.A_log",
                                    (48,), "a_log", rows_per_chunk=3)
    assert record["dtype"] == "F32"
    with safe_open(str(path), framework="np") as opened:
        actual = opened.get_tensor("language_model.model.layers.0.linear_attn.A_log")
    expected = np.asarray([converter._source_head(i) / 16 for i in range(48)], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_writer_rejects_nonfinite_values_without_leaving_partial(tmp_path):
    tensor = SimpleNamespace(name="bad", data=np.asarray([float("nan")], dtype=np.float32),
                             tensor_type=gguf.GGMLQuantizationType.F32)
    path = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError, match="nonfinite"):
        converter.write_tensor(tensor, path, "bad", (1,), "identity")
    assert not path.exists()
    assert not (tmp_path / "bad.safetensors.part").exists()


def test_completed_conversion_rejects_current_support_config_mismatch(tmp_path, monkeypatch):
    source = tmp_path / "source.gguf"
    support = tmp_path / "support"
    output = tmp_path / "output"
    support.mkdir()
    output.mkdir()
    shard = output / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"converted")
    old_config = {"model_type": "qwen3_5", "text_config": {"revision": "old"}}
    converter._write_json_atomic(output / "config.json", old_config)
    converter._write_json_atomic(output / "model.safetensors.index.json", {"weight_map": {}})
    layout = {"tensor": {"shape": [1], "type": "F32"}}
    mapping = {"tensor": ("model.tensor", (1,), "identity")}
    monkeypatch.setattr(converter, "TENSOR_COUNT", 1)
    monkeypatch.setattr(converter, "SUPPORT_FILES", ())
    monkeypatch.setattr(
        converter, "inspect_source",
        lambda _source, verify_bytes=False: (object(), layout, "layout", mapping),
    )
    monkeypatch.setattr(
        converter, "_support_config",
        lambda _support: {"model_type": "qwen3_5", "text_config": {"revision": "current"}},
    )
    monkeypatch.setattr(converter, "_check_support_tokenizer", lambda _reader, _support: {})
    receipt = {
        "source_sha256": converter.SOURCE_SHA256,
        "layout_sha256": "layout",
        "tensor_count": 1,
        "files": {
            "tensor": {
                "filename": shard.name,
                "key": "model.tensor",
                "size": shard.stat().st_size,
                "sha256": converter.sha256(shard),
            }
        },
        "support_files": {},
        "config_sha256": converter.sha256(output / "config.json"),
        "index_sha256": converter.sha256(output / "model.safetensors.index.json"),
    }
    (output / "mlx2-gguf-conversion.json").write_text(json.dumps(receipt))

    with pytest.raises(ValueError, match="another support config"):
        converter.convert(source, output, support)


def test_gguf_shifted_norm_is_preserved_for_single_fold(tmp_path):
    # llama.cpp adds +1 to these gammas before writing GGUF. MLX-shaped
    # conv1d weights make mlx2's trunk_raw trigger false on load, so the
    # converter must not subtract or add another unit here.
    values = np.asarray([0.88, 1.03, 1.14], dtype=np.float32)
    tensor = SimpleNamespace(name="blk.0.attn_norm.weight", data=values,
                             tensor_type=gguf.GGMLQuantizationType.F32)
    path = tmp_path / "norm.safetensors"
    converter.write_tensor(tensor, path, "language_model.model.layers.0.input_layernorm.weight",
                           (3,), "identity", rows_per_chunk=2)
    with safe_open(str(path), framework="np") as opened:
        actual = opened.get_tensor("language_model.model.layers.0.input_layernorm.weight")
    np.testing.assert_array_equal(actual, values)
