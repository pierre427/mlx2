"""CPU contract for the MLX-VLM NVFP4 export of Nemotron 3.5 Super VL."""

import copy
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters.nemotron35_super_config import (
    NVFP4_QUANTIZATION,
    nvfp4_export_schema,
)
from mlx2.runtime.nvfp4_linear import (
    ScaledNVFP4Linear,
    ScaledNVFP4SwitchLinear,
    install_weight_scale2,
)


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _quantize(weight, scale):
    return mx.quantize(weight / scale, group_size=16, bits=4, mode="nvfp4")


def _dequantize(packed, scales):
    return mx.dequantize(packed, scales, None, group_size=16, bits=4, mode="nvfp4")


def test_scaled_dense_equals_its_dequantized_law(cpu):
    mx.random.seed(11)
    weight = mx.random.normal((24, 64))
    scale = mx.array(0.0125, dtype=mx.float32)
    packed, scales = _quantize(weight, scale)
    layer = ScaledNVFP4Linear(64, 24)
    layer.load_weights(
        [("weight", packed), ("scales", scales), ("weight_scale_2", scale)], strict=True
    )
    x = mx.random.normal((3, 64))
    reference = x @ (_dequantize(packed, scales) * scale).T
    np.testing.assert_allclose(np.asarray(layer(x)), np.asarray(reference), rtol=1e-5, atol=1e-4)
    assert layer(x.astype(mx.bfloat16)).dtype == mx.bfloat16
    assert not isinstance(layer, nn.QuantizedLinear)


@pytest.mark.parametrize("sorted_indices", [False, True])
def test_scaled_switch_applies_each_experts_own_scale(cpu, sorted_indices):
    mx.random.seed(12)
    experts, outputs, inputs = 4, 8, 64
    weights = mx.random.normal((experts, outputs, inputs))
    per_expert = mx.array([0.01, 0.02, 0.5, 3.0], dtype=mx.float32)
    packed, scales = _quantize(weights, per_expert[:, None, None])
    switch = ScaledNVFP4SwitchLinear(inputs, outputs, experts)
    switch.load_weights(
        [("weight", packed), ("scales", scales), ("weight_scale_2", per_expert)], strict=True
    )
    dense = _dequantize(packed, scales) * per_expert[:, None, None]
    if sorted_indices:
        # SwitchMLP's sorted form: [N, 1, D] rows, one expert index per row.
        indices = mx.array([0, 1, 1, 2, 3, 3], dtype=mx.uint32)
        x = mx.random.normal((indices.size, 1, inputs))
        expected = mx.stack([x[n, 0] @ dense[int(e)].T for n, e in enumerate(indices.tolist())])
        got = switch(x, indices, sorted_indices=True)[:, 0]
    else:
        # The unsorted form: [B, L, 1, 1, D] against [B, L, k] indices.
        indices = mx.array([[[0, 3], [2, 1], [3, 3]]], dtype=mx.uint32)
        x = mx.random.normal((1, 3, 1, 1, inputs))
        expected = mx.stack(
            [
                mx.stack([x[0, row, 0, 0] @ dense[int(e)].T for e in indices[0, row].tolist()])
                for row in range(3)
            ]
        )[None]
        got = switch(x, indices)[..., 0, :]
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-4, atol=1e-3)


def test_install_replaces_dense_and_switch_and_refuses_others(cpu):
    from mlx2.runtime.models.switch_layers import SwitchLinear

    model = nn.Module()
    model.layers = [nn.Module()]
    model.layers[0].up = nn.Linear(64, 16, bias=False)
    model.layers[0].fc1 = SwitchLinear(64, 16, 4, bias=False)
    model.layers[0].norm = nn.RMSNorm(64)
    installed = install_weight_scale2(model, ["layers.0.up", "layers.0.fc1"])
    assert installed == {
        "layers.0.up": "ScaledNVFP4Linear",
        "layers.0.fc1": "ScaledNVFP4SwitchLinear",
    }
    assert model.layers[0].fc1.weight.shape == (4, 16, 8)
    assert model.layers[0].fc1.scales.shape == (4, 16, 4)
    with pytest.raises(ValueError, match="not a dense or switch linear"):
        install_weight_scale2(model, ["layers.0.norm"])
    model.layers[0].biased = nn.Linear(64, 16)
    with pytest.raises(ValueError, match="has a bias"):
        install_weight_scale2(model, ["layers.0.biased"])


def _minimal_config():
    layers = ["mamba", "moe"] * 40 + ["attention"] * 8
    return {
        "model_type": "nemotron_h_omni",
        "architectures": ["NemotronH_Omni_Reasoning_V3"],
        "quantization": dict(NVFP4_QUANTIZATION),
        "quantization_config": dict(NVFP4_QUANTIZATION),
        "sound_config": None,
        "downsample_ratio": 0.5,
        "img_context_token_id": 18,
        "projector_hidden_size": 20480,
        "vision_config": {
            "model_type": "radio", "hidden_size": 1280, "num_hidden_layers": 32,
            "num_attention_heads": 16, "patch_size": 16, "num_cls_tokens": 3,
            "num_registers": 7, "summary_idxs": [0, 1], "video_temporal_patch_size": 2,
        },
        "llm_config": {
            "model_type": "nemotron_h", "hidden_size": 4096, "vocab_size": 131072,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 128,
            "n_groups": 8, "conv_kernel": 4, "n_routed_experts": 512,
            "num_experts_per_tok": 22, "moe_latent_size": 1024,
            "moe_intermediate_size": 2688, "moe_shared_expert_intermediate_size": 5376,
            "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"],
            "mamba_ssm_cache_dtype": "float32", "mlp_hidden_act": "relu2",
            "max_position_embeddings": 1048576, "layers_block_type": layers,
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
        },
    }


def test_export_schema_matches_the_published_geometry():
    schema, modules = nvfp4_export_schema(_minimal_config())
    # Pinned export index: 1504 tensors, 41 MoE blocks x 4 NVFP4 projections.
    assert len(schema) == 1504
    assert len(modules) == 164
    p = "language_model.backbone.layers.1.mixer."
    assert schema[p + "switch_mlp.fc1.weight"] == {"shape": [512, 2688, 128], "dtype": "U32"}
    assert schema[p + "switch_mlp.fc1.scales"] == {"shape": [512, 2688, 64], "dtype": "U8"}
    assert schema[p + "switch_mlp.fc1.weight_scale_2"] == {"shape": [512], "dtype": "F32"}
    assert schema[p + "switch_mlp.fc2.weight"] == {"shape": [512, 1024, 336], "dtype": "U32"}
    assert schema[p + "shared_experts.down_proj.scales"] == {"shape": [4096, 336], "dtype": "U8"}
    assert schema[p + "shared_experts.up_proj.weight_scale_2"] == {"shape": [], "dtype": "F32"}
    assert schema["language_model.backbone.layers.0.mixer.conv1d.weight"]["shape"] == [10240, 4, 1]
    assert schema["language_model.backbone.layers.0.mixer.in_proj.weight"]["dtype"] == "BF16"
    assert "mlp1.layers.1.weight" in schema and "vision_final_layernorm.bias" in schema
    assert "vision_model.summary_idxs" not in schema
    assert "mtp.layers.1.mixer.switch_mlp.fc2" in modules
    bad = copy.deepcopy(_minimal_config())
    bad["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine"}
    with pytest.raises(ValueError, match="quantization differs"):
        nvfp4_export_schema(bad)


def test_sanitize_maps_the_mlx_vlm_projector_names():
    from mlx2.runtime.models.nemotron35_super import Model

    model = Model.__new__(Model)
    nn.Module.__init__(model)
    model.args = SimpleNamespace(n_routed_experts=512)
    model.vision_model = SimpleNamespace(
        radio_model=SimpleNamespace(summary_idxs=mx.array([0, 1], dtype=mx.int32))
    )
    weights = {
        "mlp1.layers.0.weight": mx.zeros((4,)),
        "mlp1.layers.1.weight": mx.zeros((4,)),
        "mlp1.layers.3.weight": mx.zeros((4,)),
        "vision_final_layernorm.weight": mx.zeros((4,)),
        "vision_model.radio_model.model.patch_generator.pos_embed": mx.zeros((1,)),
        "language_model.backbone.norm_f.weight": mx.zeros((4,)),
    }
    out = Model.sanitize(model, weights)
    assert set(out) == {
        "vision_projector.mlp1.norm.weight",
        "vision_projector.mlp1.linear1.weight",
        "vision_projector.mlp1.linear2.weight",
        "vision_projector.vision_final_layernorm.weight",
        "vision_model.radio_model.model.patch_generator.pos_embed",
        "vision_model.radio_model.summary_idxs",
        "backbone.norm_f.weight",
    }
    # The vendor names still map the same way.
    vendor = Model.sanitize(model, {"mlp1.1.weight": mx.zeros((4,))})
    assert set(vendor) == {"vision_projector.mlp1.linear1.weight"}


def _scaled_switch_mlp(experts=4, hidden=64, inner=128, seed=21):
    from mlx2.runtime.models.switch_layers import SwitchMLP

    mx.random.seed(seed)
    mlp = SwitchMLP(hidden, inner, experts, activation=nn.ReLU2())
    install_weight_scale2(mlp, ["fc1", "fc2"])
    weights = []
    for name, (n, k) in {"fc1": (inner, hidden), "fc2": (hidden, inner)}.items():
        scale = mx.random.uniform(low=0.001, high=0.05, shape=(experts,))
        packed, scales = _quantize(mx.random.normal((experts, n, k)), scale[:, None, None])
        weights += [(f"{name}.weight", packed), (f"{name}.scales", scales),
                    (f"{name}.weight_scale_2", scale)]
    mlp.load_weights(weights, strict=True)
    return mlp


def test_folded_expert_scales_match_the_applied_scales(cpu):
    from mlx2.runtime.nvfp4_linear import fold_relu2_expert_scales

    mlp = _scaled_switch_mlp()
    x = mx.random.normal((1, 5, 64))
    inds = mx.array([[[0, 3], [1, 2], [3, 3], [2, 0], [1, 1]]], dtype=mx.uint32)
    scores = mx.random.uniform(shape=(1, 5, 2))
    applied = (mlp(x, inds) * scores[..., None]).sum(axis=-2)
    assert fold_relu2_expert_scales([mlp], mx.bfloat16) == 1
    assert mlp.fc1.scale_folded and mlp.fc2.scale_folded
    assert "expert_output_scale" not in dict(mlp.parameters())
    folded = (mlp(x, inds) * (scores * mlp.expert_output_scale()[inds])[..., None]).sum(axis=-2)
    np.testing.assert_allclose(np.asarray(folded), np.asarray(applied), rtol=2e-4, atol=1e-6)


def test_fold_refuses_non_relu2_or_nonpositive_scales(cpu):
    from mlx2.runtime.nvfp4_linear import fold_relu2_expert_scales

    mlp = _scaled_switch_mlp()
    mlp.activation = nn.GELU()
    with pytest.raises(ValueError, match="ReLU2"):
        fold_relu2_expert_scales([mlp], mx.bfloat16)
    mlp = _scaled_switch_mlp()
    mlp.fc1.weight_scale_2 = mx.array([0.01, 0.0, 0.02, 0.03])
    with pytest.raises(ValueError, match="positive"):
        fold_relu2_expert_scales([mlp], mx.bfloat16)
    with pytest.raises(ValueError, match="bfloat16"):
        fold_relu2_expert_scales([_scaled_switch_mlp()], mx.float16)
    with pytest.raises(ValueError, match="multiple of 64"):
        ScaledNVFP4SwitchLinear(32, 8, 4)


def test_vendor_fp8_codes_are_recovered_exactly_and_refused_otherwise(cpu):
    from mlx2.runtime.nvfp4_linear import ScaledFP8Linear, encode_vendor_fp8

    mx.random.seed(31)
    scale = 0.0021
    codes = mx.to_fp8(mx.random.normal((16, 64)) * 40)
    vendor = mx.from_fp8(codes, dtype=mx.float32)
    exported = (vendor * mx.array(scale, dtype=mx.float32)).astype(mx.bfloat16)
    packed, unit = encode_vendor_fp8(exported, scale)
    assert np.array_equal(np.asarray(packed.view(mx.uint8).reshape(16, 64)), np.asarray(codes))
    layer = ScaledFP8Linear(64, 16)
    layer.load_weights([("weight", packed), ("scales", unit),
                        ("weight_scale", mx.array(scale, dtype=mx.float32))], strict=True)
    x = mx.random.normal((3, 64))
    reference = x @ (vendor * scale).T
    np.testing.assert_allclose(np.asarray(layer(x)), np.asarray(reference), rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="not bf16"):
        encode_vendor_fp8(exported, scale * 1.37)
    with pytest.raises(ValueError, match="not bf16"):
        encode_vendor_fp8(mx.random.normal((16, 64)).astype(mx.bfloat16), scale)


def test_vendor_fp8_sidecar_covers_the_exported_dense_projections():
    from mlx2.adapters.nemotron35_super_config import vendor_fp8_scales

    from pathlib import Path

    config_path = (
        Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/config.json"
    )
    if not config_path.is_file():
        pytest.skip("pinned NVFP4 export config is not on this host")
    scales = vendor_fp8_scales()
    schema, _ = nvfp4_export_schema(json.loads(config_path.read_text()))
    dense = {
        name.removeprefix("language_model.").removesuffix(".weight")
        for name, entry in schema.items()
        if entry["dtype"] == "BF16"
        and name.startswith(("language_model.backbone.", "language_model.lm_head"))
        and name.endswith(("in_proj.weight", "out_proj.weight", "q_proj.weight",
                           "k_proj.weight", "v_proj.weight", "o_proj.weight",
                           "lm_head.weight"))
    }
    assert set(scales) == dense and len(scales) == 113
    assert all(0 < value < 1 for value in scales.values())


def test_routing_weights_fold_once_and_pass_through_without_a_table():
    from mlx2.runtime.models.nemotron_h import NemotronHMoE
    from mlx2.runtime.nvfp4_linear import _FoldedScale

    scores = mx.array([[[0.25, 0.75], [0.5, 0.5]]], dtype=mx.float32)
    inds = mx.array([[[0, 2], [1, 2]]], dtype=mx.uint32)
    plain = SimpleNamespace(switch_mlp=SimpleNamespace())
    assert NemotronHMoE._routing_weights(plain, inds, scores) is scores
    table = mx.array([2.0, 3.0, 5.0], dtype=mx.float32)
    folded = SimpleNamespace(switch_mlp=SimpleNamespace(expert_output_scale=_FoldedScale(table)))
    got = NemotronHMoE._routing_weights(folded, inds, scores)
    np.testing.assert_array_equal(np.asarray(got), np.asarray([[[0.5, 3.75], [1.5, 2.5]]]))
    import inspect

    source = inspect.getsource(NemotronHMoE)
    # Both combine sites (ordinary and row-exact) read the folded weights.
    assert source.count("self._routing_weights(inds, scores)") == 2


def test_nvfp4_export_detection_ignores_malformed_or_converted_artifacts(tmp_path):
    from mlx2.adapters.nemotron35_super import _is_nvfp4_export

    (tmp_path / "conversion_info.json").write_text("{not json")
    assert _is_nvfp4_export(tmp_path) is False
    (tmp_path / "conversion_info.json").write_text(json.dumps({"quantization": "nvfp4"}))
    assert _is_nvfp4_export(tmp_path) is False
    (tmp_path / "conversion_info.json").write_text(json.dumps({
        "source_model": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
        "quantization": {"mode": "nvfp4"}}))
    assert _is_nvfp4_export(tmp_path) is True
    (tmp_path / "mlx2-conversion.json").write_text("{}")
    assert _is_nvfp4_export(tmp_path) is False


def test_nax_gather_admits_nvfp4_only_from_the_owned_module():
    from mlx2.runtime.models import moe_nax_gather as nax

    source = __import__("inspect").getsource(nax.try_gather)
    assert "nax_nvfp4_owned" in source
    assert ScaledNVFP4SwitchLinear.nax_nvfp4_owned is True


def test_fp_row_exact_header_and_dispatch_mirror_mlx_one_row_qmv():
    from mlx2.runtime.models import fp_row_exact_qmv as fq

    header = fq._header()
    assert "METAL_FUNC void fp_qmv_fast_impl(" in header
    assert "METAL_FUNC void fp_qmv_impl(" in header
    assert "const constant int& in_vec_size" not in header
    assert "fp_qmv_wide_impl" not in header
    # Real Super VL shapes: FP8 Mamba/attention/lm_head take qmv_fast; the
    # NVFP4 shared down projection (K 5376) takes the general qmv.
    assert fq.fast_layout(4096, 18560, 8) and fq.fast_layout(8192, 4096, 8)
    assert fq.fast_layout(4096, 131072, 8) and fq.fast_layout(4096, 5376, 4)
    assert not fq.fast_layout(5376, 4096, 4)


def test_scaled_modules_fall_back_per_row_off_gpu(cpu):
    from mlx2.runtime.models.nemotron_h import _short_projection
    from mlx2.runtime.nvfp4_linear import ScaledFP8Linear, encode_vendor_fp8
    from mlx2.runtime.row_exact import ops

    scale = 0.003
    codes = mx.to_fp8(mx.random.normal((32, 256)) * 30)
    exported = (mx.from_fp8(codes, dtype=mx.float32) * scale).astype(mx.bfloat16)
    packed, unit = encode_vendor_fp8(exported, scale)
    layer = ScaledFP8Linear(256, 32)
    layer.load_weights([("weight", packed), ("scales", unit),
                        ("weight_scale", mx.array(scale, dtype=mx.float32))], strict=True)
    x = mx.random.normal((1, 3, 256)).astype(mx.bfloat16)
    assert layer.rows_exact(x) == (None, "device")
    rowwise = ops.per_row(layer, x)
    for out in (ops.linear(layer, x, stage="t"), _short_projection(layer, x)[0]):
        assert np.array_equal(np.asarray(out.view(mx.uint16)), np.asarray(rowwise.view(mx.uint16)))
    assert _short_projection(layer, x)[1] == "per_row_device"


def test_fp_row_exact_kernel_mode_switch(monkeypatch):
    from mlx2.runtime.models import fp_row_exact_qmv as fq

    monkeypatch.delenv("MLX2_FP_ROW_EXACT_KERNEL", raising=False)
    assert fq.kernel_mode() == "fused"
    monkeypatch.setenv("MLX2_FP_ROW_EXACT_KERNEL", "loop")
    assert fq.kernel_mode() == "loop"
    monkeypatch.setenv("MLX2_FP_ROW_EXACT_KERNEL", "wide")
    with pytest.raises(ValueError, match="loop or fused"):
        fq.kernel_mode()
    assert "mlx2_fp_qmv_fast_rows" in fq._FUSED_FAST and fq.FUSED_MAX_ROWS == 8


def test_nemotron_routed_decode_switch_and_cpu_decline(cpu, monkeypatch):
    from mlx2.runtime.models import nemotron_routed_decode as rd
    from mlx2.runtime.nvfp4_linear import fold_relu2_expert_scales

    monkeypatch.delenv("MLX2_NEMOTRON_ROUTED_DECODE", raising=False)
    assert rd.enabled() is True
    monkeypatch.setenv("MLX2_NEMOTRON_ROUTED_DECODE", "off")
    assert rd.enabled() is False
    monkeypatch.setenv("MLX2_NEMOTRON_ROUTED_DECODE", "maybe")
    with pytest.raises(ValueError):
        rd.enabled()
    mlp = _scaled_switch_mlp()
    x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)
    inds = mx.array([[[0, 3]]], dtype=mx.uint32)
    assert rd.decline_reason(mlp, x, inds) == "scales_not_folded"
    fold_relu2_expert_scales([mlp], mx.bfloat16)
    # K 64 is MLX's quad one-row layout: declined before the device check.
    assert rd.decline_reason(mlp, x, inds) == "geometry"
    wide = _scaled_switch_mlp(hidden=256, inner=256)
    fold_relu2_expert_scales([wide], mx.bfloat16)
    x = mx.random.normal((1, 1, 256)).astype(mx.bfloat16)
    assert rd.decline_reason(wide, x, inds) == "device"
    assert rd.routed_experts(wide, x, inds) is None
