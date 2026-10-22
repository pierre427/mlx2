"""Strict BF16 execution and fail-closed packed Nemotron LiLiCorr tests."""

import copy
import hashlib
import json
import math
import struct

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters import lilicorr_nemotron as plans
from mlx2.adapters.lilicorr import LiLiCorrConfig


def export_config(variant):
    config = {
        "architectures": ["LiLiCorrDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "num_target_layers": 3,
        "block_size": 8,
        "layer_types": ["sliding_attention"] * 5,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e6,
        "torch_dtype": "bfloat16",
        "has_own_lm_head": True,
        "use_sliding_window": True,
        "sliding_window": 32,
        "dflash_config": {
            "mask_token_id": 99,
            "target_layer_ids": [0, 2],
            "causal": False,
            "projector_type": "lilicorr",
            "lilicorr_enabled": True,
            "lilicorr_candidate_topk": 8,
            "lilicorr_hidden_size": 32,
            "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 8,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 32,
            "lilicorr_vector_eps": 1e-4,
            "lilicorr_logit_scale": 8.0,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "use_swa": True,
            "swa_window_size": 32,
        },
    }
    if variant == "native":
        modules = [
            "fc",
            "lm_head",
            "lilicorr.context_proj",
            "lilicorr.pass_hidden_proj",
            "lilicorr.token_proj",
        ]
        modules.extend(
            f"layers.{i}.mlp.{p}_proj" for i in range(5) for p in ("down", "gate", "up")
        )
        modules.extend(
            f"lilicorr.layers.{i}.{p}"
            for i in range(2)
            for p in ("attn.out_proj", "mlp.down_proj", "mlp.up_proj")
        )
        config["quantization"] = {
            "bits": 4,
            "group_size": 16,
            "mode": "nvfp4",
            "global_scale": "per_tensor_fp32",
            "modules": sorted(modules),
        }
    return config


def write_header(path, schema):
    sizes = {"BF16": 2, "U32": 4, "U8": 1, "F8_E4M3": 1, "F32": 4}
    position, header = 0, {}
    for name, (dtype, shape) in sorted(schema.items()):
        end = position + math.prod(shape) * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [position, end]}
        position = end
    encoded = json.dumps(header).encode()
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + position)
    return position


@pytest.fixture(params=["bf16", "native", "modelopt"])
def exported(tmp_path, monkeypatch, request):
    variant = request.param
    config = export_config(variant)
    raw = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    weight_hash = "0" * 64
    monkeypatch.setattr(plans, "EXPORTS", {digest: (variant, weight_hash)})
    size = write_header(
        tmp_path / "model.safetensors", plans.expected_export_tensors(config, variant)
    )
    (tmp_path / "mlx2-conversion.json").write_text(
        json.dumps(
            {
                "schema": "mlx2.modelopt-nvfp4-import.v1",
                "mode": variant,
                "source_config_sha256": plans.SOURCE_CONFIG_SHA256,
                "source_weight_sha256": plans.SOURCE_WEIGHT_SHA256,
                "output_weight_sha256": weight_hash,
                "global_scale_preserved": True,
                "quantized_matrices": 26,
                "output_tensor_bytes": size,
            }
        )
    )
    target = {
        "text_config": {
            "hidden_size": 64,
            "vocab_size": 128,
            "layers_block_type": ["mamba", "moe", "attention"],
            "num_key_value_heads": 1,
        },
        "identity": {"fingerprint": "target-revision-a"},
        "load_plan": {"resident_weight_bytes": 12345},
    }
    monkeypatch.setattr(plans, "inspect_artifact", lambda _: target)
    return tmp_path, config, target, variant, size


def test_all_export_schemas_bind_target_features_without_claiming_runtime(exported):
    path, config, target, variant, size = exported
    result = plans.inspect_export_plan(path, "target")
    assert result["variant"] == variant
    assert result["tensor_count"] == (125 if variant == "bf16" else 177)
    assert result["target_feature_width"] == 128
    assert result["target_and_draft_weight_bytes"] == 12345 + size
    assert result["artifact_schema_supported"]
    assert not result["runtime_load_supported"]
    assert (
        not result["qualified"]
        and not result["selected"]
        and not result["observed_used"]
    )
    assert not result["weight_payload_rehashed"]
    # GQA geometry is private to each backbone; equality is not compatibility.
    assert config["num_key_value_heads"] != target["text_config"]["num_key_value_heads"]
    with pytest.raises(ValueError, match="owned LM heads"):
        LiLiCorrConfig.from_dict(config)


def test_target_revision_changes_plan_and_target_width_mismatch_refuses(exported):
    path, _, target, _, _ = exported
    before = plans.inspect_export_plan(path, "target")
    target["identity"]["fingerprint"] = "target-revision-b"
    after = plans.inspect_export_plan(path, "target")
    assert before["fingerprint"] != after["fingerprint"]
    target["text_config"]["hidden_size"] = 65
    with pytest.raises(ValueError, match="target feature"):
        plans.inspect_export_plan(path, "target")


def test_missing_owned_head_is_rejected_before_runtime(exported):
    path, config, _, variant, _ = exported
    schema = plans.expected_export_tensors(config, variant)
    del schema["lm_head.weight"]
    write_header(path / "model.safetensors", schema)
    with pytest.raises(ValueError, match="tensor inventory"):
        plans.inspect_export_plan(path, "target")


def test_wrong_scale_or_weight_layout_is_rejected(exported):
    path, config, _, variant, _ = exported
    schema = plans.expected_export_tensors(config, variant)
    name = (
        "lm_head.weight"
        if variant == "bf16"
        else "lm_head.global_scale"
        if variant == "native"
        else "lm_head.weight_scale_2"
    )
    dtype, shape = schema[name]
    schema[name] = (dtype, [*shape, 1])
    write_header(path / "model.safetensors", schema)
    with pytest.raises(ValueError, match="dtype/shape"):
        plans.inspect_export_plan(path, "target")


def test_native_quantization_exclusions_cannot_be_expanded():
    config = export_config("native")
    bad = copy.deepcopy(config)
    bad["quantization"]["modules"].append("layers.0.self_attn.q_proj")
    with pytest.raises(ValueError, match="modules/exclusions"):
        plans.expected_export_tensors(bad, "native")
    schema = plans.expected_export_tensors(config, "native")
    assert schema["layers.0.self_attn.q_proj.weight"] == ("BF16", [64, 64])
    assert schema["lm_head.weight"] == ("U32", [128, 8])
    assert schema["lm_head.scales"] == ("U8", [128, 4])
    assert schema["lm_head.global_scale"] == ("F32", [])


@pytest.fixture
def executable_bf16(tmp_path, monkeypatch):
    from test_nemotron35_super_vl_cpu import tiny_config

    from mlx2.adapters import nemotron35_super_vl as target_adapter
    from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    config = export_config("bf16")
    monkeypatch.setattr(plans, "TRAINED_SLIDING_WINDOW", 32)
    raw = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(raw)
    args = plans.executable_config(config)
    mx.random.seed(811)
    source = LiLiCorrDraftModel(args)
    weights = {
        name: value.astype(mx.bfloat16)
        for name, value in tree_flatten(source.parameters())
    }
    source.load_weights(list(weights.items()), strict=True)
    file = tmp_path / "model.safetensors"
    mx.save_safetensors(str(file), weights)
    weight_hash = hashlib.sha256(file.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(plans, "EXPORTS", {config_hash: ("bf16", weight_hash)})
    _, _, payload_size = plans._read_safetensors_header(file)
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps({
        "schema": "mlx2.modelopt-nvfp4-import.v1",
        "mode": "bf16",
        "source_config_sha256": plans.SOURCE_CONFIG_SHA256,
        "source_weight_sha256": plans.SOURCE_WEIGHT_SHA256,
        "output_weight_sha256": weight_hash,
        "global_scale_preserved": True,
        "quantized_matrices": 26,
        "output_tensor_bytes": payload_size,
    }))
    target_config = tiny_config()
    target = Model(ModelArgs.from_dict(target_adapter.runtime_config(target_config)))
    target_artifact = {
        "text_config": target_adapter.runtime_config(target_config),
        "identity": {"fingerprint": "tiny-nemotron-target"},
        "load_plan": {"resident_weight_bytes": 12345},
    }
    return tmp_path, source, target, target_artifact


def _target_contract(record):
    return {
        "schema": "mlx2.nemotron-lilicorr-target.v1",
        "target_model_type": "nemotron_h_omni",
        "target_fingerprint": record["target_fingerprint"],
        "target_layer_ids": tuple(record["args"].target_layer_ids),
        "target_feature_width": (
            len(record["args"].target_layer_ids) * record["args"].hidden_size
        ),
        "capture": "post-layer-pre-final-norm",
        "ordinary_reference_preserved": True,
        "speculative_rollback": "exact",
        "owned_lm_head": True,
        "owned_final_norm": True,
        "qualified": False,
    }


def _assert_cache_state(actual, expected):
    assert actual.offset == expected.offset
    for left, right in zip(actual.state, expected.state):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_bf16_owned_head_strict_load_preserves_head_norm_and_trained_window(
    executable_bf16,
):
    path, source, target, target_artifact = executable_bf16
    record = plans.inspect_drafter(path, target_artifact=target_artifact)
    tokens = mx.array([[1, 2, 3]])
    ordinary = target(tokens, target.make_cache())
    loaded = plans.load_drafter(
        record, target, target_contract=_target_contract(record)
    )
    assert loaded.embed_tokens is target.embed_tokens
    assert loaded.lm_head is not target.lm_head
    assert loaded.norm is not target.backbone.norm_f
    assert loaded.receipt_settings["owned_lm_head"]
    assert loaded.receipt_settings["trained_sliding_window"] == 32
    assert all(layer.self_attn.is_sliding for layer in loaded.layers)
    assert all(cache.max_size == 31 for cache in loaded.make_cache())
    owned_head = loaded.lm_head
    reset_cache = loaded.reset(target)
    assert loaded.lm_head is owned_head
    assert all(cache.max_size == 31 for cache in reset_cache)
    np.testing.assert_array_equal(
        np.asarray(target(tokens, target.make_cache())), np.asarray(ordinary)
    )
    probe = mx.ones((1, 2, record["args"].hidden_size), dtype=mx.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(loaded._logits(probe).astype(mx.float32)),
        np.asarray(source._logits(probe).astype(mx.float32)),
    )


def test_bf16_pair_restores_target_and_wrapped_draft_state(executable_bf16):
    path, _, target, target_artifact = executable_bf16
    record = plans.inspect_drafter(path, target_artifact=target_artifact)
    draft = plans.load_drafter(
        record, target, target_contract=_target_contract(record)
    )
    target_cache = target.make_cache()
    prefix = mx.array([[1 + index % 126 for index in range(35)]])
    _, taps = target.forward_with_taps(
        prefix, target_cache, record["args"].target_layer_ids
    )
    draft_cache = draft.make_cache()
    draft.append_context(taps, draft_cache)
    assert all(cache.offset == 35 and cache.size() == 31 for cache in draft_cache)
    target_before = copy.deepcopy(target_cache)
    draft_before = copy.deepcopy(draft_cache)
    for cache in [*target_cache, *draft_cache]:
        cache.start_speculation()
    verify = mx.array([[41, 42, 43]])
    _, verify_taps = target.forward_with_taps(
        verify, target_cache, record["args"].target_layer_ids
    )
    draft.append_context(verify_taps, draft_cache)
    mx.eval([cache.state for cache in [*target_cache, *draft_cache]])
    for cache in [*target_cache, *draft_cache]:
        assert cache.trim(3) == 3
        cache.stop_speculation()
    for actual, expected in zip(draft_cache, draft_before):
        _assert_cache_state(actual, expected)
    from test_nemotron_external_taps_cpu import assert_state_equal

    assert_state_equal(target_cache, target_before)


def test_bf16_binding_contract_fails_closed_before_load(executable_bf16):
    path, _, target, target_artifact = executable_bf16
    record = plans.inspect_drafter(path, target_artifact=target_artifact)
    changed = _target_contract(record)
    changed["target_layer_ids"] = (1,)
    with pytest.raises(ValueError, match="target contract"):
        plans.load_drafter(record, target, target_contract=changed)


def test_modelopt_export_fails_closed_and_native_is_executable(exported, monkeypatch):
    quantized_path, _, quantized_target, variant, _ = exported
    monkeypatch.setattr(plans, "TRAINED_SLIDING_WINDOW", 32)
    if variant == "modelopt":
        with pytest.raises(ValueError, match="ModelOpt-packed NVFP4 export"):
            plans.inspect_drafter(
                quantized_path, target_artifact=quantized_target
            )
    else:
        record = plans.inspect_drafter(quantized_path, target_artifact=quantized_target)
        assert record["variant"] == variant and record["runtime_load_supported"]


def test_global_scaled_nvfp4_linear_equals_its_dequantized_law():
    from mlx2.runtime.nvfp4_linear import GlobalScaledNVFP4Linear

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(5)
        weight = mx.random.normal((24, 64))
        scale = mx.array(0.0125, dtype=mx.float32)
        packed, scales = mx.quantize(weight / scale, group_size=16, bits=4, mode="nvfp4")
        layer = GlobalScaledNVFP4Linear(64, 24, bias=True)
        layer.load_weights([("weight", packed), ("scales", scales), ("global_scale", scale),
                            ("bias", mx.arange(24, dtype=mx.float32))], strict=True)
        dense = mx.dequantize(packed, scales, None, group_size=16, bits=4, mode="nvfp4") * scale
        x = mx.random.normal((3, 64))
        np.testing.assert_allclose(np.asarray(layer(x)), np.asarray(x @ dense.T + mx.arange(24)),
                                   rtol=1e-5, atol=1e-4)
        plain = GlobalScaledNVFP4Linear(64, 24)
        plain.load_weights([("weight", packed), ("scales", scales), ("global_scale", scale)])
        # The float32 global scale does not promote a bf16 activation stream.
        assert plain(x.astype(mx.bfloat16)).dtype == mx.bfloat16
    finally:
        mx.set_default_device(previous)


def test_native_nvfp4_export_loads_strictly_as_global_scaled_modules(executable_bf16, monkeypatch):
    from mlx2.runtime.nvfp4_linear import GlobalScaledNVFP4Linear

    path, source, target, target_artifact = executable_bf16
    config = export_config("native")
    modules = config["quantization"]["modules"]
    flat = dict(tree_flatten(source.parameters()))
    native, dense = {}, {}
    for name, value in flat.items():
        module = name.removesuffix(".weight")
        if name.endswith(".weight") and module in modules:
            weight = value.astype(mx.float32)
            scale = mx.max(mx.abs(weight)) / (6.0 * 448.0)
            packed, scales = mx.quantize(weight / scale, group_size=16, bits=4, mode="nvfp4")
            native[name], native[module + ".scales"] = packed, scales
            native[module + ".global_scale"] = scale.astype(mx.float32)
            dense[name] = (mx.dequantize(packed, scales, None, group_size=16, bits=4,
                                         mode="nvfp4") * scale).astype(mx.bfloat16)
        else:
            native[name] = dense[name] = value
    raw = json.dumps(config).encode()
    (path / "config.json").write_bytes(raw)
    file = path / "model.safetensors"
    mx.save_safetensors(str(file), native)
    weight_hash = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(plans, "EXPORTS", {hashlib.sha256(raw).hexdigest(): ("native", weight_hash)})
    _, _, payload_size = plans._read_safetensors_header(file)
    conversion = json.loads((path / "mlx2-conversion.json").read_text())
    conversion.update(mode="native", output_weight_sha256=weight_hash, output_tensor_bytes=payload_size)
    (path / "mlx2-conversion.json").write_text(json.dumps(conversion))
    record = plans.inspect_drafter(path, target_artifact=target_artifact)
    assert record["variant"] == "native" and record["runtime_load_supported"]
    with pytest.raises(ValueError, match="already quantized"):
        plans.load_drafter(record, target, target_contract=_target_contract(record),
                           runtime_quantization="affine_q8_g64")
    loaded = plans.load_drafter(record, target, target_contract=_target_contract(record))
    assert isinstance(loaded.lm_head, GlobalScaledNVFP4Linear)
    assert isinstance(loaded.layers[0].mlp.down_proj, GlobalScaledNVFP4Linear)
    assert loaded.receipt_settings["weight_quantization"]["mode"] == "nvfp4"
    reference = type(source)(source.config)
    reference.load_weights(list(dense.items()), strict=True)
    probe = mx.random.normal((1, 2, record["args"].hidden_size))
    np.testing.assert_allclose(
        np.asarray(loaded.lm_head(probe)), np.asarray(reference.lm_head(probe).astype(mx.float32)),
        rtol=2e-2, atol=2e-2,
    )
