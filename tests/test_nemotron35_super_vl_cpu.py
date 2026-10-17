"""Native Omni namespace/affine recipe against a real tiny hybrid on CPU."""

import hashlib
import json
import os
import struct
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx2.adapters import nemotron35_super_vl as adapter
from mlx2.runtime.models.nemotron_h import Model, ModelArgs

mx.set_default_device(mx.cpu)


def tiny_config():
    c = {
        "model_type": "nemotron_h",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "max_position_embeddings": 1024,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "attention_bias": False,
        "mamba_num_heads": 2,
        "mamba_head_dim": 32,
        "mamba_proj_bias": False,
        "ssm_state_size": 8,
        "conv_kernel": 4,
        "n_groups": 1,
        "mlp_bias": False,
        "layer_norm_epsilon": 1e-5,
        "use_bias": False,
        "use_conv_bias": True,
        "layers_block_type": ["mamba", "moe", "attention"],
        "moe_latent_size": 64,
        "moe_intermediate_size": 128,
        "moe_shared_expert_intermediate_size": 128,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.0,
        "time_step_limit": [0.0, {"__float__": "Infinity"}],
        "num_nextn_predict_layers": 1,
    }
    config = {
        "model_type": "nemotron_h_omni",
        "llm_config": c,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    }
    for name, (_, shape) in adapter.expected_text_tensors(config).items():
        if len(shape) >= 2 and any(
            part in name
            for part in (
                "in_proj.",
                "out_proj.",
                "q_proj.",
                "k_proj.",
                "v_proj.",
                "o_proj.",
                "switch_mlp.",
                "shared_experts.",
                "lm_head.",
            )
        ):
            config["quantization"][name.removesuffix(".weight")] = {
                "bits": 4 if "switch_mlp" in name else 8,
                "group_size": 64,
                "mode": "affine",
            }
    return config


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    config = tiny_config()
    mx.random.seed(284)
    model = Model(ModelArgs.from_dict(adapter.runtime_config(config)))
    # Export uses BF16 source precision, except routing correction biases.
    model.load_weights(
        [
            (
                name,
                value.astype(
                    mx.float32
                    if name.endswith("e_score_correction_bias")
                    else mx.bfloat16
                ),
            )
            for name, value in tree_flatten(model.parameters())
        ]
    )
    quant = config["quantization"]
    nn.quantize(
        model,
        class_predicate=lambda name, module: (
            dict(quant[adapter.TEXT_PREFIX + name])
            if adapter.TEXT_PREFIX + name in quant
            else False
        ),
    )
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        if name.endswith("conv1d.weight"):
            value = mx.moveaxis(value, 1, 2)
        weights[adapter.TEXT_PREFIX + name] = value
    weights["vision_model.unused.weight"] = mx.zeros((3, 4), dtype=mx.bfloat16)
    weights["language_model.mtp.unused.weight"] = mx.zeros((2, 3), dtype=mx.bfloat16)
    file = tmp_path / "model.safetensors"
    mx.save_safetensors(str(file), weights)
    index = {"weight_map": {name: file.name for name in weights}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config_hash = hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest()
    index_hash = hashlib.sha256(
        (tmp_path / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(adapter, "CONFIG_SHA256", config_hash)
    monkeypatch.setattr(adapter, "INDEX_SHA256", index_hash)
    metadata = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        (tmp_path / name).write_text("{}")
        metadata[name] = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(adapter, "TOKENIZER_SHA256", metadata)
    receipt = {
        "schema": "mlx2.nemotron35-super-vl-conversion.v1",
        "recipe": "nemotron35-super-vl-affine-experts4-protected8-v1",
        "source_revision": adapter.SOURCE_REVISION,
        "output_config_sha256": config_hash,
        "output_index_sha256": index_hash,
        "metadata_sha256": metadata,
        "output_shards": [{"file": file.name, "bytes": file.stat().st_size}],
        "tensor_bytes": sum(x.nbytes for x in weights.values()),
    }
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps(receipt))
    return tmp_path, model, config


def test_real_tiny_export_strict_load_matches_original_hybrid_and_chunked_cache(
    artifact,
):
    path, reference, _ = artifact
    record = adapter.inspect_artifact(path)
    actual = adapter.load_text_model(record)
    assert not hasattr(actual, "mtp")
    assert record["load_plan"]["excluded_tensor_count"] == 2
    assert record["load_plan"]["excluded_weight_bytes"] == 36
    assert actual.layers[1].mixer.switch_mlp.fc1.bits == 4
    assert actual.layers[0].mixer.in_proj.bits == 8
    tokens = mx.array([[1, 2, 3, 4]])
    expected = reference(tokens, reference.make_cache())
    got = actual(tokens, actual.make_cache())
    np.testing.assert_array_equal(
        np.asarray(got.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )
    ac, rc = actual.make_cache(), reference.make_cache()
    for part in (tokens[:, :2], tokens[:, 2:3], tokens[:, 3:]):
        logits, taps = actual.forward_with_taps(part, ac, [0, 1, 2])
        ref_logits, ref_taps = reference.forward_with_taps(part, rc, [0, 1, 2])
        np.testing.assert_array_equal(
            np.asarray(logits.astype(mx.float32)),
            np.asarray(ref_logits.astype(mx.float32)),
        )
        np.testing.assert_array_equal(
            np.asarray(taps.astype(mx.float32)), np.asarray(ref_taps.astype(mx.float32))
        )
    assert ac[1].offset == rc[1].offset == 4
    for a, r in zip(ac[0].cache, rc[0].cache):
        np.testing.assert_array_equal(
            np.asarray(a.astype(mx.float32)), np.asarray(r.astype(mx.float32))
        )


def test_raw_schema_matches_real_runtime_parameters(artifact):
    path, _, config = artifact
    with (path / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    expected = adapter.expected_text_tensors(config)
    assert len(expected) > 40
    assert all(
        (header[key]["dtype"], header[key]["shape"]) == shape
        for key, shape in expected.items()
    )


def test_registry_reserves_omni_type_for_full_adapter_and_legacy_refuses_mtp(
    artifact,
):
    from mlx2.adapters.registry import inspect_model, resolve_adapter

    path, _, _ = artifact
    # Automatic dispatch owns this model type through the complete source-bound
    # multimodal conversion.  A valid legacy text-only research export must not
    # be able to select the production cache/router contract.
    with pytest.raises(ValueError, match="Expected Nemotron 3.5 Super VL composition"):
        inspect_model(path)
    with pytest.raises(ValueError, match="Expected Nemotron 3.5 Super VL composition"):
        resolve_adapter(path)

    record = adapter.inspect_artifact(path)
    obj = adapter.Nemotron35SuperVLAdapter.__new__(adapter.Nemotron35SuperVLAdapter)
    obj.artifact = record
    assert obj.default_route == "ordinary"
    with pytest.raises(ValueError, match="native MTP"):
        obj.profile_name(True)


def test_planning_cache_geometry_is_available_without_model_allocation():
    config = tiny_config()
    config["llm_config"]["layers_block_type"] = ["mamba", "moe"] * 40 + [
        "attention"
    ] * 8
    budget = adapter.Nemotron35SuperVLAdapter.planning_cache_budget(config, mtp=False)
    assert budget.recurrent_layers == 40 and budget.attention_layers == 8
    assert budget.mtp_attention_layers == 0
    assert budget.project(4096) > budget.project(2048) > budget.fixed_bytes
    assert budget.as_dict()["transient_bound"] == "conservative-unmeasured-allowance"
    with pytest.raises(ValueError, match="native MTP"):
        adapter.Nemotron35SuperVLAdapter.planning_cache_budget(config, mtp=True)


@pytest.mark.parametrize("mutation", ["shape", "extent", "duplicate"])
def test_inspection_rejects_invalid_header_without_loading_tensors(artifact, mutation):
    path, _, _ = artifact
    file = path / "model.safetensors"
    raw = file.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    key = "language_model.backbone.layers.0.mixer.in_proj.weight"
    if mutation == "shape":
        header[key]["shape"] = list(reversed(header[key]["shape"]))
    elif mutation == "extent":
        header[key]["data_offsets"][0] += 4
    encoded = json.dumps(header).encode()
    if mutation == "duplicate":
        encoded = (
            encoded[:-1]
            + b',"'
            + key.encode()
            + b'":'
            + json.dumps(header[key]).encode()
            + b"}"
        )
    file.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length :])
    with pytest.raises(ValueError, match="dtype/shape|extent|Duplicate JSON"):
        adapter.inspect_artifact(path)


def test_pinned_config_and_conversion_identity_fail_closed(artifact):
    path, _, _ = artifact
    receipt = path / "mlx2-conversion.json"
    data = json.loads(receipt.read_text())
    data["source_revision"] = "foreign"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="conversion identity"):
        adapter.inspect_artifact(path)
    (path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned native export"):
        adapter.inspect_artifact(path)


def test_text_filter_does_not_touch_excluded_lazy_values():
    sentinel = object()
    assert adapter.text_weights(
        {
            "vision_model.bad": sentinel,
            "language_model.mtp.bad": sentinel,
            "language_model.lm_head.weight": sentinel,
        }
    ) == {"lm_head.weight": sentinel}


def test_inspector_import_stays_cpu_metadata_only():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from mlx2.adapters.registry import _nemotron_h_omni; "
                "from mlx2.adapters.nemotron35_super_vl import inspect_artifact; "
                "assert 'mlx.core' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "MLX2_PREFLIGHT_CPU_ONLY": "1"},
    )
    assert result.returncode == 0, result.stderr


def test_default_route_refuses_mtp_and_keeps_external_draft_unselected():
    from mlx2.contracts import Capability

    obj = adapter.Nemotron35SuperVLAdapter.__new__(adapter.Nemotron35SuperVLAdapter)
    assert obj.default_route == "ordinary"
    assert Capability.MTP not in obj.descriptor.capabilities
    with pytest.raises(ValueError, match="native MTP"):
        obj.profile_name(True)
    obj._num_draft = 0
    assert obj.execution_config(max_lanes=1, prefill_step=32)["num_draft"] == 0
    contract = obj.route_capability_contract()
    assert contract["qualification"] == "unqualified"
    assert contract["external_draft"]["implemented"]
    assert not contract["external_draft"]["selected"]
    assert contract["external_draft"]["quantized_exports"] == "refused"


def test_time_step_tag_preserves_zero_floor_and_disables_unbound_mtp():
    config = tiny_config()
    runtime = adapter.runtime_config(config)
    assert runtime["time_step_limit"] == (0, float("inf"))
    assert runtime["num_nextn_predict_layers"] == 0
    assert config["llm_config"]["num_nextn_predict_layers"] == 1
    config["llm_config"]["time_step_limit"] = [float("nan"), 1]
    with pytest.raises(ValueError, match="time-step"):
        adapter.runtime_config(config)
