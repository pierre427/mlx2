import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import pytest

from mlx2.adapters import nemotron35_bolt as bolt


def target_config():
    kinds = (["mamba", "moe"] * 3 + ["mamba", "attention", "moe"]) * 8
    assert len(kinds) == 72
    kinds += ["mamba", "moe"] * 8
    blocks = []
    expert_counts = [240, 304, 368, 432, 496]
    widths = [1536, 2048, 2304, 2560]
    moe_index = 0
    for kind in kinds:
        item = {"block_type": kind}
        if kind == "moe":
            item.update(
                n_routed_experts=expert_counts[moe_index % len(expert_counts)],
                moe_intermediate_size=widths[moe_index % len(widths)],
                num_experts_per_tok=9,
            )
            moe_index += 1
        blocks.append(item)
    groups = {
        "group_0": {"quant_algo": "Q8_0", "group_size": 32, "packing": "ggml", "block_payload_bytes": 34, "targets": [str(i) for i in range(113)]},
        "group_1": {"weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}, "targets": [str(i) for i in range(80)]},
        "group_2": {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml", "block_payload_bytes": 66, "targets": [str(i) for i in range(27728)]},
    }
    return {
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "model_type": "nemotron_h_omni",
        "llm_config": {
            "model_type": "nemotron_h_puzzle", "hidden_size": 4096, "vocab_size": 131072,
            "max_position_embeddings": 262144, "num_nextn_predict_layers": 0,
            "mtp_block_configs": [], "mtp_layers_block_type": [], "layers_block_type": kinds,
            "block_configs": blocks, "mamba_num_heads": 128, "mamba_head_dim": 64,
            "ssm_state_size": 128, "n_groups": 8, "num_attention_heads": 32,
            "num_key_value_heads": 2,
        },
        "vision_config": {"model_type": "radio", "hidden_size": 1280, "num_hidden_layers": 32, "patch_size": 16},
        "projector_hidden_size": 20480, "sound_config": None,
        "quantization_config": {"config_groups": groups},
    }


def test_bolt_geometry_is_owned_and_heterogeneous():
    text = bolt.validate_target_config(target_config())
    assert len(text["block_configs"]) == 88
    assert {b.get("n_routed_experts") for b in text["block_configs"] if b["block_type"] == "moe"} == {240, 304, 368, 432, 496}


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("llm_config", "model_type"), "nemotron_h", "target geometry"),
        (("quantization_config", "config_groups", "group_2", "quant_algo"), "Q2_K", "packing contract"),
    ],
)
def test_bolt_lookalikes_fail_closed(path, value, message):
    config = target_config()
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        bolt.validate_target_config(config)


def test_bolt_lilicorr_binds_exact_taps_and_attention_sink():
    config = {
        "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "hidden_size": 4096,
        "vocab_size": 131072, "num_hidden_layers": 5, "block_size": 8,
        "sliding_window": 1024, "use_sliding_window": True,
        "layer_types": ["sliding_attention"] * 5, "attention_sink_bias": True,
        "dflash_config": {
            "target_layer_ids": [1, 18, 35, 51, 68, 85], "attention_sink_bias": True,
            "lilicorr_candidate_topk": 8, "lilicorr_hidden_size": 1024,
            "conv_kernel_size": 2, "conv_group_size": 16, "swa_window_size": 1024,
        },
        "quantization_config": {"quant_algo": "W4A16_NVFP4", "group_size": 16, "quant_method": "modelopt_fp4"},
    }
    target = {
        "schema": "mlx2.nemotron35-bolt-artifact.v1",
        "identity": {"source_revision": bolt.SOURCE_REVISION, "fingerprint": "0" * 64},
        "text_config": target_config()["llm_config"],
    }
    assert bolt.validate_lilicorr_config(config, target)["num_target_layers"] == 88
    config["dflash_config"]["target_layer_ids"][-1] = 86
    with pytest.raises(ValueError, match="LiLiCorr geometry"):
        bolt.validate_lilicorr_config(config, target)


def test_registry_routes_puzzle_omni_to_bolt(monkeypatch, tmp_path):
    from mlx2.adapters import registry

    module = SimpleNamespace(
        Nemotron35BoltAdapter=type("Bolt", (), {}),
        DESCRIPTOR=object(),
        inspect_artifact=lambda path: {"path": str(path), "identity": {"fingerprint": "x"}},
    )
    seen = []

    def load(name, package):
        seen.append((name, package))
        return module

    monkeypatch.setattr(registry.importlib, "import_module", load)
    result = registry._nemotron_h_omni(tmp_path, {"llm_config": {"model_type": "nemotron_h_puzzle"}})
    assert result.adapter_type.__name__ == "Bolt"
    assert seen == [(".nemotron35_bolt", registry.__package__)]


def test_bolt_descriptor_is_text_only_without_native_mtp():
    from mlx2.contracts import Capability

    caps = bolt.DESCRIPTOR.capabilities
    assert Capability.TEXT in caps and Capability.APC_V2 in caps
    assert Capability.MTP not in caps and Capability.VISION not in caps
    assert bolt.DESCRIPTOR.cache_layout == bolt.CACHE_LAYOUT
    assert bolt.DESCRIPTOR.metadata["qualification"] == "unqualified"
    assert bolt.DESCRIPTOR.metadata["native_mtp"] == "absent"


def test_registry_malformed_llm_config_does_not_raise_attribute_error(monkeypatch, tmp_path):
    from mlx2.adapters import registry

    module = SimpleNamespace(
        Nemotron35SuperAdapter=type("Super", (), {}),
        DESCRIPTOR=object(),
        inspect_artifact=lambda path: {"path": str(path)},
    )
    monkeypatch.setattr(registry.importlib, "import_module", lambda name, package: module)
    result = registry._nemotron_h_omni(tmp_path, {"llm_config": None})
    assert result.adapter_type.__name__ == "Super"


def _write_safetensors(path, entries):
    cursor = 0
    header = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    widths = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    for name, (dtype, shape) in entries.items():
        size = widths[dtype]
        for value in shape:
            size *= value
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        payload.extend(b"\0" * size)
        cursor += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return raw, len(payload)


def test_safetensors_extent_validation_rejects_overlap(tmp_path):
    file = tmp_path / "model.safetensors"
    _write_safetensors(file, {"a": ("BF16", [1]), "b": ("BF16", [1])})
    _, tensors, payload, _ = bolt._validate_safetensors_file(file)
    assert set(tensors) == {"a", "b"}
    assert payload == 4

    raw = json.loads(file.read_bytes()[8 : 8 + struct.unpack("<Q", file.read_bytes()[:8])[0]])
    raw["b"]["data_offsets"] = [0, 2]
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    file.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
    with pytest.raises(ValueError, match="overlaps or has gaps"):
        bolt._validate_safetensors_file(file)


def test_target_inspector_checks_metadata_header_and_shard_closure(monkeypatch, tmp_path):
    config = target_config()
    quantized = {}
    for algorithm, count in (("IQ2_XXS", 27_728), ("Q8_0", 113), ("W4A16_NVFP4", 80)):
        quantized.update(
            {f"module.{algorithm}.{index}": {"quant_algo": algorithm} for index in range(count)}
        )
    quant = {
        "producer": {"name": "modelopt", "version": "0.37.0"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": quantized,
        },
    }
    shard_name = "model-00001-of-00001.safetensors"
    entries = {f"tensor.{index}": ("U8", [1]) for index in range(28_885)}
    mapping = {name: shard_name for name in entries}
    index = {"metadata": {"total_size": len(entries)}, "weight_map": mapping}
    serialized = {}
    for name, value in (
        ("config.json", config),
        ("hf_quant_config.json", quant),
        ("model.safetensors.index.json", index),
    ):
        serialized[name] = json.dumps(value, separators=(",", ":")).encode()
        (tmp_path / name).write_bytes(serialized[name])
    metadata = {"tokenizer.json": b"pinned-tokenizer"}
    for name, raw in metadata.items():
        (tmp_path / name).write_bytes(raw)
    raw_header, payload = _write_safetensors(tmp_path / shard_name, entries)
    shard_size = (tmp_path / shard_name).stat().st_size
    monkeypatch.setattr(bolt, "SOURCE_CONFIG_SHA256", hashlib.sha256(serialized["config.json"]).hexdigest())
    monkeypatch.setattr(bolt, "SOURCE_QUANT_SHA256", hashlib.sha256(serialized["hf_quant_config.json"]).hexdigest())
    monkeypatch.setattr(bolt, "SOURCE_INDEX_SHA256", hashlib.sha256(serialized["model.safetensors.index.json"]).hexdigest())
    monkeypatch.setattr(bolt, "SOURCE_METADATA_SHA256", {name: hashlib.sha256(raw).hexdigest() for name, raw in metadata.items()})
    monkeypatch.setattr(bolt, "SOURCE_HEADER_SET_SHA256", hashlib.sha256(shard_name.encode() + raw_header).hexdigest())
    monkeypatch.setattr(bolt, "SOURCE_TENSOR_BYTES", payload)
    monkeypatch.setattr(bolt, "SOURCE_SHARD_BYTES", shard_size)
    monkeypatch.setattr(bolt, "SOURCE_SHARDS", 1)
    result = bolt.inspect_source_artifact(tmp_path)
    assert result["tensor_count"] == 28_885
    assert result["files"][0][0] == shard_name
    (tmp_path / "tokenizer.json").unlink()
    (tmp_path / "tokenizer.json").symlink_to(tmp_path / "config.json")
    with pytest.raises(ValueError, match="Missing or escaped"):
        bolt.inspect_source_artifact(tmp_path)


def test_lilicorr_inspector_checks_exact_header_and_weight_hash(monkeypatch, tmp_path):
    config = {
        "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3",
        "hidden_size": 4096, "vocab_size": 131072, "num_hidden_layers": 5,
        "block_size": 8, "sliding_window": 1024, "use_sliding_window": True,
        "layer_types": ["sliding_attention"] * 5, "attention_sink_bias": True,
        "dflash_config": {
            "target_layer_ids": [1, 18, 35, 51, 68, 85], "attention_sink_bias": True,
            "lilicorr_candidate_topk": 8, "lilicorr_hidden_size": 1024,
            "conv_kernel_size": 2, "conv_group_size": 16, "swa_window_size": 1024,
        },
        "quantization_config": {
            "quant_algo": "W4A16_NVFP4", "group_size": 16,
            "quant_method": "modelopt_fp4",
        },
    }
    quant = {
        "producer": {"name": "modelopt", "version": "0.0.0"},
        "quantization": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
    }
    config_raw = json.dumps(config, separators=(",", ":")).encode()
    quant_raw = json.dumps(quant, separators=(",", ":")).encode()
    (tmp_path / "config.json").write_bytes(config_raw)
    (tmp_path / "hf_quant_config.json").write_bytes(quant_raw)
    entries = {
        "layers.0.self_attn.attention_sink_bias": ("BF16", [32]),
        "lilicorr.relative_slot_bias": ("BF16", [8, 15]),
        "lilicorr.token_proj.weight": ("U8", [1024, 2048]),
        "lilicorr.token_proj.weight_scale": ("F8_E4M3", [1024, 256]),
        "lilicorr.token_proj.weight_scale_2": ("F32", []),
    }
    for dtype, count in (("BF16", 102), ("F32", 24), ("F8_E4M3", 24), ("U8", 24)):
        entries.update({f"extra.{dtype}.{i}": (dtype, [1]) for i in range(count)})
    file = tmp_path / "model.safetensors"
    raw_header, payload = _write_safetensors(file, entries)
    monkeypatch.setattr(bolt, "LILICORR_CONFIG_SHA256", hashlib.sha256(config_raw).hexdigest())
    monkeypatch.setattr(bolt, "LILICORR_QUANT_SHA256", hashlib.sha256(quant_raw).hexdigest())
    monkeypatch.setattr(bolt, "LILICORR_HEADER_SHA256", hashlib.sha256(raw_header).hexdigest())
    monkeypatch.setattr(bolt, "LILICORR_WEIGHT_SHA256", hashlib.sha256(file.read_bytes()).hexdigest())
    monkeypatch.setattr(bolt, "LILICORR_TENSOR_BYTES", payload)
    target = {
        "schema": "mlx2.nemotron35-bolt-artifact.v1",
        "identity": {"source_revision": bolt.SOURCE_REVISION, "fingerprint": "0" * 64},
        "text_config": target_config()["llm_config"],
    }
    result = bolt.inspect_lilicorr(tmp_path, target_artifact=target)
    assert result["weight_payload_rehashed"] is True
    file.write_bytes(file.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="weights"):
        bolt.inspect_lilicorr(tmp_path, target_artifact=target)


def test_bolt_source_checkpoint_is_not_servable(tmp_path):
    # The mixed GGML/ModelOpt source has no conversion receipt: refuse before
    # any accelerator import, pointing at the converter.
    had_mlx_core = "mlx.core" in sys.modules
    with pytest.raises(ValueError, match="convert_nemotron35_bolt"):
        bolt.Nemotron35BoltAdapter(tmp_path)
    assert ("mlx.core" in sys.modules) is had_mlx_core


def _bolt_provenance():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "provenance" / "nemotron35-bolt.json"
    if not path.is_file():
        pytest.skip("private material absent: Bolt provenance (not exported to the public mirror)")
    return root, json.loads(path.read_text())


def test_bolt_provenance_names_only_real_model_args_fields():
    import re

    from mlx2.runtime.models.nemotron_h import ModelArgs

    _root, record = _bolt_provenance()
    named = {
        name
        for text in record["modifications"]
        for name in re.findall(r"\bModelArgs\.(\w+)", text)
    }
    assert named <= set(ModelArgs.__dataclass_fields__), sorted(
        named - set(ModelArgs.__dataclass_fields__)
    )


def test_bolt_observed_lifecycle_is_recorded_under_its_checkpoint():
    # AGENTS.md: model-path results go in model-feature-decisions.tsv under
    # the exact checkpoint and the mlx2 commit each run came from.
    import csv
    from pathlib import Path

    root, record = _bolt_provenance()
    conversion = record["validation"]["conversion"]
    checkpoint = f"{Path(conversion['artifact']).name}@{conversion['fingerprint'][:8]}"
    units = {
        Path(section["evidence"]).name: section["commit"]
        for section in record["validation"].values()
        if isinstance(section, dict) and "evidence" in section and "commit" in section
    }
    with (root / "docs" / "research" / "model-feature-decisions.tsv").open(newline="") as handle:
        rows = {
            row["feature_or_flag"]: row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["model"] == "nemotron_h_omni:base" and row["checkpoint"] == checkpoint
        }
    observed = {
        name: route for name, route in record["lifecycle"].items() if route["observed_used"]
    }
    assert observed
    for name, route in record["lifecycle"].items():
        assert route["qualified"] is False, name
        for feature in route.get("grid_features", ()):
            row = rows.get(feature)
            assert row is not None, (name, feature)
            assert row["qualified"] == "no", (name, feature)
            if route["selected"]:
                assert row["selected"] == "yes", (name, feature)
            if not route["observed_used"]:
                assert row["observed_used"] != "yes", (name, feature)
    for name, route in observed.items():
        assert route["grid_features"], name
        assert route["evidence"], name
        for unit in route["evidence"]:
            commit = units[unit.split(" ")[0]]
            for feature in route["grid_features"]:
                row = rows[feature]
                assert row["observed_used"] == "yes", (name, feature)
                assert commit in row["source_revision"], (name, feature, commit)
                assert unit.split(" ")[0] in row["evidence"], (name, feature, unit)
