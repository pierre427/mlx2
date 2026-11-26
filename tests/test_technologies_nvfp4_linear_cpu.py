"""Exact parity of every migrated NVFP4 / FP8 linear copy with the reference.

Each ``_legacy_*`` function is the pre-migration body, verbatim, from the
file named in its docstring (main at d46e7a3ea).
"""
import json
import os
import struct
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime import nvfp4_linear as runtime_nvfp4
from mlx2.runtime.models import modelopt_nvfp4
from mlx2.runtime.technologies import nvfp4_linear as tech
from mlx2.runtime.technologies.nvfp4_linear import NVFP4Spec

DTYPES = (mx.bfloat16, mx.float16, mx.float32)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


# --------------------------------------------------------------------------
# legacy copies (verbatim)
# --------------------------------------------------------------------------

def _legacy_global_scaled_call(self, x):
    """runtime/nvfp4_linear.py GlobalScaledNVFP4Linear.__call__."""
    GROUP_SIZE, BITS = 16, 4
    y = mx.quantized_matmul(
        x, self["weight"], scales=self["scales"], biases=None, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode="nvfp4",
    )
    y = (y.astype(mx.float32) * self["global_scale"]).astype(y.dtype)
    if "bias" in self:
        y = y + self["bias"]
    return y


def _legacy_scale_law(y, scale):
    """runtime/nvfp4_linear.py _scale_law (also inkling2/moe_fast.py)."""
    return (y.astype(mx.float32) * scale).astype(y.dtype)


_LEGACY_COMPILED_SCALE = None


def _legacy_apply_scale(y, scale):
    """runtime/nvfp4_linear.py _apply_scale."""
    global _LEGACY_COMPILED_SCALE

    mode = os.environ.get("MLX2_SCALED_EPILOGUE", "compiled")
    if mode == "eager":
        return _legacy_scale_law(y, scale)
    if mode != "compiled":
        raise ValueError("MLX2_SCALED_EPILOGUE must be eager or compiled")
    if _LEGACY_COMPILED_SCALE is None:
        _LEGACY_COMPILED_SCALE = mx.compile(_legacy_scale_law, shapeless=True)
    return _LEGACY_COMPILED_SCALE(y, scale)


def _legacy_scaled_nvfp4_call(self, x):
    """runtime/nvfp4_linear.py ScaledNVFP4Linear.__call__."""
    GROUP_SIZE, BITS = 16, 4
    y = mx.quantized_matmul(
        x, self["weight"], scales=self["scales"], biases=None, transpose=True,
        group_size=GROUP_SIZE, bits=BITS, mode="nvfp4",
    )
    return _legacy_apply_scale(y, self["weight_scale_2"])


def _legacy_scaled_fp8_call(self, x):
    """runtime/nvfp4_linear.py ScaledFP8Linear.__call__."""
    FP8_GROUP = 32
    y = mx.quantized_matmul(
        x, self["weight"], scales=self["scales"], biases=None, transpose=True,
        group_size=FP8_GROUP, bits=8, mode="mxfp8",
    )
    return _legacy_apply_scale(y, self["weight_scale"])


def _legacy_modelopt_native(self, x):
    """runtime/models/modelopt_nvfp4.py ModelOptNVFP4Linear._native (less counter)."""
    result = mx.quantized_matmul(
        x.astype(mx.float32),
        self.weight,
        scales=self.scales,
        group_size=16,
        bits=4,
        mode="nvfp4",
        transpose=True,
    ) * self.global_scale
    if "bias" in self:
        result = result + self.bias.astype(mx.float32)
    return result.astype(x.dtype)


def _legacy_modelopt_decoded_weight(self):
    """runtime/models/modelopt_nvfp4.py prepare_decoded_weight (decode only)."""
    return mx.dequantize(
        self.weight,
        scales=self.scales,
        group_size=16,
        bits=4,
        mode="nvfp4",
        global_scale=self.global_scale * (448 * 6),
        dtype=mx.bfloat16,
    )


def _legacy_modelopt_decoded_rows(self, x, decoded):
    """runtime/models/modelopt_nvfp4.py __call__ decoded_large_rows branch."""
    result = x @ decoded.T
    if "bias" in self:
        result = result.astype(mx.float32) + self.bias.astype(mx.float32)
        result = result.astype(x.dtype)
    return result


def _legacy_raon_merged_nvfp4(x, weight, scales, column_scale):
    """runtime/models/raon1_mini.py MergedProjection.__call__ nvfp4 branch."""
    y = mx.quantized_matmul(x, weight, scales=scales, biases=None, transpose=True,
                            group_size=16, bits=4, mode="nvfp4")
    y = (y.astype(mx.float32) * column_scale).astype(y.dtype)
    return y


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _nvfp4_arrays(rng, n, k, *, zeros=False):
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.integers(1, 127, (n, k // 16), dtype=np.uint8)
    if zeros:
        packed[0] = 0  # an all-zero row
        scales[1] = 0  # zero block scales
    return mx.array(packed.view("<u4")), mx.array(scales)


def _inputs(rng, shape, dtype, *, ties=False):
    x = rng.normal(size=shape).astype(np.float32)
    if ties:
        x[..., ::2] = x[..., 1::2]  # repeated values (ties)
        x[..., :16] = 0.0
    return mx.array(x).astype(dtype)


def _load(module, weight, scales, scale, bias=None, scale_name="global_scale"):
    items = [("weight", weight), ("scales", scales), (scale_name, mx.array(scale, dtype=mx.float32))]
    if bias is not None:
        items.append(("bias", bias))
    module.load_weights(items, strict=True)
    return module


def _same(a, b):
    mx.eval(a, b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert bool(mx.array_equal(a, b, equal_nan=True))


SCALES = (1.0e-4, 0.125, 3.0)
SHAPES = ((1, 64), (1, 5, 128), (2, 3, 64))


# --------------------------------------------------------------------------
# parity: one test per migrated copy
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_global_scaled_raon_lilicorr_native_parity(dtype, shape, bias, ties):
    rng = np.random.default_rng(hash((str(dtype), shape, bias, ties)) % 2**32)
    k, n = shape[-1], 48
    weight, scales = _nvfp4_arrays(rng, n, k, zeros=True)
    for scale in SCALES:
        module = runtime_nvfp4.GlobalScaledNVFP4Linear(k, n, bias=bias)
        bias_value = mx.array(rng.normal(size=(n,)).astype(np.float32)) if bias else None
        _load(module, weight, scales, scale, bias_value)
        x = _inputs(rng, shape, dtype, ties=ties)
        _same(module(x), _legacy_global_scaled_call(module, x))
        _same(tech.reference(x, weight, scales, module["global_scale"], bias_value,
                             runtime_nvfp4.GLOBAL_SCALED_SPEC),
              _legacy_global_scaled_call(module, x))


def test_global_scaled_init_matches_legacy_parameter_tree():
    module = runtime_nvfp4.GlobalScaledNVFP4Linear(64, 32, bias=True)
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(module.parameters()))
    assert list(params) == ["weight", "scales", "global_scale", "bias"]
    assert params["weight"].dtype == mx.uint32 and params["weight"].shape == (32, 8)
    assert params["scales"].dtype == mx.uint8 and params["scales"].shape == (32, 4)
    assert params["global_scale"].dtype == mx.float32 and params["global_scale"].shape == ()
    assert params["bias"].dtype == mx.float32
    assert module.trainable_parameters() == {}
    with pytest.raises(ValueError, match="multiple of 16"):
        runtime_nvfp4.GlobalScaledNVFP4Linear(24, 8)


@pytest.mark.parametrize("epilogue", ["compiled", "eager"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_weight_scale2_nemotron_super_parity(monkeypatch, epilogue, dtype, shape):
    monkeypatch.setenv("MLX2_SCALED_EPILOGUE", epilogue)
    rng = np.random.default_rng(31)
    k, n = shape[-1], 32
    weight, scales = _nvfp4_arrays(rng, n, k, zeros=True)
    for scale in SCALES:
        module = _load(runtime_nvfp4.ScaledNVFP4Linear(k, n), weight, scales, scale,
                       scale_name="weight_scale_2")
        for ties in (False, True):
            x = _inputs(rng, shape, dtype, ties=ties)
            _same(module(x), _legacy_scaled_nvfp4_call(module, x))


def test_compiled_and_eager_epilogue_agree_bit_for_bit(monkeypatch):
    rng = np.random.default_rng(5)
    for dtype in DTYPES:
        y = _inputs(rng, (3, 7, 40), dtype)
        scale = mx.array(rng.uniform(1e-5, 4, size=(3, 1, 1)).astype(np.float32))
        monkeypatch.setenv("MLX2_SCALED_EPILOGUE", "eager")
        eager = runtime_nvfp4._apply_scale(y, scale)
        monkeypatch.setenv("MLX2_SCALED_EPILOGUE", "compiled")
        compiled = runtime_nvfp4._apply_scale(y, scale)
        _same(eager, compiled)
        _same(compiled, _legacy_apply_scale(y, scale))
    monkeypatch.setenv("MLX2_SCALED_EPILOGUE", "bogus")
    with pytest.raises(ValueError, match="eager or compiled"):
        runtime_nvfp4._apply_scale(y, scale)


@pytest.mark.parametrize("epilogue", ["compiled", "eager"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vendor_fp8_parity(monkeypatch, epilogue, dtype):
    monkeypatch.setenv("MLX2_SCALED_EPILOGUE", epilogue)
    rng = np.random.default_rng(12)
    k, n = 64, 24
    codes = mx.to_fp8(mx.array(rng.normal(size=(n, k)).astype(np.float32)))
    packed = codes.reshape(n, k // 4, 4).view(mx.uint32).reshape(n, k // 4)
    module = runtime_nvfp4.ScaledFP8Linear(k, n)
    assert bool(mx.all(module["scales"] == 127))
    for scale in SCALES:
        module.load_weights([("weight", packed), ("weight_scale", mx.array(scale, dtype=mx.float32))],
                            strict=False)
        for shape in ((1, k), (2, 5, k)):
            x = _inputs(rng, shape, dtype, ties=True)
            _same(module(x), _legacy_scaled_fp8_call(module, x))
    with pytest.raises(ValueError, match="FP8 input width must be a multiple of 32"):
        runtime_nvfp4.ScaledFP8Linear(48, 8)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bias", [False, True])
def test_modelopt_native_lilicorr_ripple_ricursive_parity(dtype, shape, bias):
    rng = np.random.default_rng(77)
    k, n = shape[-1], 40
    weight, scales = _nvfp4_arrays(rng, n, k, zeros=True)
    for scale in SCALES:
        module = modelopt_nvfp4.ModelOptNVFP4Linear(k, n, bias=bias)
        bias_value = (mx.array(rng.normal(size=(n,)).astype(np.float32)).astype(mx.bfloat16)
                      if bias else None)
        _load(module, weight, scales, scale, bias_value)
        for ties in (False, True):
            x = _inputs(rng, shape, dtype, ties=ties)
            _same(module(x), _legacy_modelopt_native(module, x))
            # Ricursive's shared FP32 input path.
            shared = modelopt_nvfp4.shared_fp32_input((module, module), x)
            if shared is not None:
                _same(module(shared).astype(x.dtype),
                      _legacy_modelopt_native(module, shared).astype(x.dtype))
    with pytest.raises(ValueError, match="divide into blocks of 16"):
        modelopt_nvfp4.ModelOptNVFP4Linear(24, 8)


@pytest.mark.parametrize("bias", [False, True])
def test_modelopt_decoded_large_rows_parity(bias):
    rng = np.random.default_rng(91)
    k, n = 64, 32
    weight, scales = _nvfp4_arrays(rng, n, k)
    module = modelopt_nvfp4.ModelOptNVFP4Linear(k, n, bias=bias)
    bias_value = mx.array(rng.normal(size=(n,)).astype(np.float32)).astype(mx.bfloat16) if bias else None
    _load(module, weight, scales, 0.0107421875, bias_value)
    assert module.prepare_decoded_weight(budget_bytes=1 << 20) == n * k * 2
    module.set_dispatch_policy("auto", large_row_threshold=4)
    legacy = _legacy_modelopt_decoded_weight(module)
    _same(module._decoded_weight, legacy)
    for dtype in (mx.bfloat16, mx.float32):
        x = _inputs(rng, (6, k), dtype)
        _same(module(x), _legacy_modelopt_decoded_rows(module, x, legacy))
        assert module._dispatch_counts["decoded_large_rows"] >= 1
        small = _inputs(rng, (3, k), dtype)
        _same(module(small), _legacy_modelopt_native(module, small))


def test_modelopt_fused_b1_fails_closed_off_gpu_and_counts_refusals():
    module = modelopt_nvfp4.ModelOptNVFP4Linear(64, 32)
    module.set_dispatch_policy("fused_b1")
    with pytest.raises(ValueError, match="B1 fused projection refused: device"):
        module(mx.zeros((1, 64), dtype=mx.bfloat16))
    module(mx.zeros((2, 64), dtype=mx.bfloat16))
    assert module._refusal_counts["fused_b1_rows"] == 1


@pytest.mark.parametrize("dtype", DTYPES)
def test_raon_merged_projection_parity(dtype):
    from mlx2.runtime.models.raon1_mini import MergedProjection

    rng = np.random.default_rng(3)
    k = 64
    parts = []
    for n, scale in ((16, 0.5), (24, 1e-3), (8, 2.0)):
        module = runtime_nvfp4.GlobalScaledNVFP4Linear(k, n)
        weight, scales = _nvfp4_arrays(rng, n, k)
        parts.append(_load(module, weight, scales, scale))
    merged = MergedProjection(parts)
    for shape in ((1, k), (1, 4, k)):
        x = _inputs(rng, shape, dtype, ties=True)
        actual = mx.concatenate(merged(x), axis=-1)
        legacy = _legacy_raon_merged_nvfp4(x, merged.weight, merged.scales, merged.column_scale)
        _same(actual, legacy)
        # And column-for-column what each part computes on its own.
        _same(actual, mx.concatenate([p(x) for p in parts], axis=-1))


def test_inkling2_scale_law_parity():
    from mlx2.runtime.models.inkling2 import moe_fast

    rng = np.random.default_rng(4)
    for dtype in DTYPES:
        y = _inputs(rng, (2, 3, 1, 32), dtype, ties=True)
        scale = mx.array(rng.uniform(1e-4, 2, size=(2, 3, 1, 1)).astype(np.float32))
        _same(moe_fast._scale_law(y, scale), _legacy_scale_law(y, scale))
        _same(moe_fast._scale_rows(y, scale), _legacy_scale_law(y, scale))


def test_switch_linear_epilogue_parity(monkeypatch):
    """ScaledNVFP4SwitchLinear's per-expert scale goes through _apply_scale."""
    rng = np.random.default_rng(8)
    experts, n, k = 3, 16, 64
    switch = runtime_nvfp4.ScaledNVFP4SwitchLinear(k, n, experts)
    packed = rng.integers(0, 256, (experts, n, k // 2), dtype=np.uint8)
    scales = rng.integers(1, 127, (experts, n, k // 16), dtype=np.uint8)
    s2 = mx.array(rng.uniform(1e-4, 1, size=(experts,)).astype(np.float32))
    switch.load_weights([("weight", mx.array(packed.view("<u4"))), ("scales", mx.array(scales)),
                         ("weight_scale_2", s2)], strict=True)
    x = _inputs(rng, (4, 1, 1, k), mx.bfloat16)
    idx = mx.array(rng.integers(0, experts, (4, 1, 2)).astype(np.uint32))
    for epilogue in ("compiled", "eager"):
        monkeypatch.setenv("MLX2_SCALED_EPILOGUE", epilogue)
        y = mx.gather_qmm(x, switch["weight"], switch["scales"], None, rhs_indices=idx,
                          transpose=True, group_size=16, bits=4, mode="nvfp4")
        legacy = _legacy_apply_scale(y, s2[idx][..., None, None])
        _same(switch(x, idx), legacy)


# --------------------------------------------------------------------------
# spec, registry, checkpoint
# --------------------------------------------------------------------------

def test_spec_from_config_modelopt_and_mlx_forms():
    ripple = {"quantization_config": {
        "quant_algo": "NVFP4", "quant_method": "modelopt",
        "config_groups": {"group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"]}}}}
    spec = NVFP4Spec.from_config(ripple, global_scale_law="fp32_product")
    assert spec == modelopt_nvfp4.MODELOPT_SPEC
    w4a16 = NVFP4Spec.from_config({"quantization_config": {
        "quant_algo": "W4A16_NVFP4", "group_size": 16, "quant_method": "modelopt"}})
    assert (w4a16.mode, w4a16.group_size, w4a16.input_scale) == ("nvfp4", 16, "absent")
    hf_quant = NVFP4Spec.from_config({"quantization": {"quant_algo": "FP8"}},
                                     global_scale_law="scale_rounded_product",
                                     epilogue="compiled_env", global_scale_name="weight_scale",
                                     unit_block_scale=127)
    assert hf_quant == runtime_nvfp4.VENDOR_FP8_SPEC
    mlx_style = NVFP4Spec.from_config({"quantization": {"mode": "nvfp4", "bits": 4, "group_size": 16}},
                                      global_scale_law="scale_rounded_product",
                                      global_scale_name="global_scale", bias_dtype="float32")
    assert mlx_style == runtime_nvfp4.GLOBAL_SCALED_SPEC
    with pytest.raises(ValueError, match="group size 16"):
        NVFP4Spec.from_config({"quantization": {"mode": "nvfp4", "group_size": 32}})
    with pytest.raises(ValueError, match="compiled epilogue"):
        NVFP4Spec(epilogue="compiled_env")
    with pytest.raises(TypeError):
        NVFP4Spec.from_config({}, mode="nvfp4", not_a_field=1)


@pytest.mark.parametrize("config", [
    {"quantization": {"group_size": 64, "bits": 4}},  # affine
    {"quantization": {"mode": "mxfp4", "bits": 4, "group_size": 32}},
    {},
])
def test_spec_from_config_fails_closed_on_unclassifiable_configs(config):
    with pytest.raises(ValueError, match="cannot derive a block-scaled FP mode"):
        NVFP4Spec.from_config(config)
    # An explicit mode is still honoured when the caller knows it.
    assert NVFP4Spec.from_config({}, mode="nvfp4").mode == "nvfp4"


def test_registry_receipts_and_admission():
    from mlx2.runtime.technologies import dispatch

    dispatch.reset_counters()
    # A spec-only module (no module-owned dispatch policy).
    module = tech.NVFP4Linear(64, 32, spec=modelopt_nvfp4.MODELOPT_SPEC)
    x = mx.zeros((1, 64), dtype=mx.bfloat16)
    _, receipt = tech.project(module, x)
    assert receipt.selected == "reference"
    _, receipt = tech.project(module, x, policy="auto")
    assert receipt.selected == "reference"
    assert receipt.refusals == {"decoded_large_rows": "decoded_not_resident",
                                "fused_b1": "device", "row_tile": "device",
                                "row_tile_row_exact": "device"}
    with pytest.raises(RuntimeError, match="fused_b1"):
        tech.project(module, x, policy="fused_b1")
    snapshot = dispatch.counters_snapshot()
    selections = {(r["technology"], r["operation"], r["path"]) for r in snapshot["selections"]}
    assert ("nvfp4_linear", "project", "reference") in selections
    assert ("nvfp4_linear", "project", "failed_closed") in selections
    codes = {r["code"] for r in snapshot["refusals"] if r["technology"] == "nvfp4_linear"}
    assert codes <= {"decoded_not_resident", "device", "global_scale_law"}, codes
    # The tile kernel's own geometry checks still run (CPU, no device gate).
    assert tech.ROW_TILE.admits(module.spec, module=module, x=x, require_gpu=False) is None
    # Kernels implementing the fp32 law refuse the rounded-product law.
    other = runtime_nvfp4.GlobalScaledNVFP4Linear(64, 32)
    assert tech.FUSED_B1.admits(other.spec, module=other, x=x) == "global_scale_law"
    assert tech.ROW_TILE.admits(other.spec, module=other, x=x, require_gpu=False) == "global_scale_law"


def test_project_on_modelopt_module_uses_its_own_policy_and_counters():
    rng = np.random.default_rng(55)
    k, n = 64, 32
    weight, scales = _nvfp4_arrays(rng, n, k)
    module = _load(modelopt_nvfp4.ModelOptNVFP4Linear(k, n), weight, scales, 0.03125)
    one = _inputs(rng, (1, k), mx.bfloat16)
    many = _inputs(rng, (6, k), mx.bfloat16)

    # native: reference, counted by the module.
    y, receipt = tech.project(module, one)
    assert receipt.selected == "reference" and receipt.refusals == {}
    _same(y, _legacy_modelopt_native(module, one))
    assert module._dispatch_counts["native"] == 1

    # Module auto = decoded large rows only (never fused_b1 / row tiles).
    configured = modelopt_nvfp4.configure_nvfp4_modules(
        module, policy="auto", decoded_budget_bytes=1 << 20, decoded_large_row_threshold=4)
    assert configured["decoded_resident_bytes"] == n * k * 2
    y, receipt = tech.project(module, many, policy="auto")
    assert receipt.selected == "decoded_large_rows"
    _same(y, _legacy_modelopt_decoded_rows(module, many, _legacy_modelopt_decoded_weight(module)))
    y, receipt = tech.project(module, one)
    assert receipt.selected == "reference"
    assert receipt.refusals == {"decoded_large_rows": "rows_below_threshold"}
    status = modelopt_nvfp4.nvfp4_model_status(module)
    assert status["observed_used"] and status["counts"]["decoded_large_rows"] == 1

    # A policy other than the module's own is refused, not silently applied.
    with pytest.raises(ValueError, match="module dispatch policy is 'auto'"):
        tech.project(module, one, policy="fused_b1")

    # fused_b1: one row fails closed off-GPU (module semantics), more rows refuse.
    module.set_dispatch_policy("fused_b1")
    with pytest.raises(ValueError, match="refused: device"):
        tech.project(module, one)
    _, receipt = tech.project(module, many)
    assert receipt.selected == "reference"
    assert receipt.refusals == {"fused_b1": "rows: fused_b1 runs one row only"}


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


@pytest.mark.parametrize("cls,spec_name", [
    (modelopt_nvfp4.ModelOptNVFP4Linear, "global_scale"),
    (runtime_nvfp4.ScaledNVFP4Linear, "weight_scale_2"),
    (runtime_nvfp4.GlobalScaledNVFP4Linear, "global_scale"),
])
def test_modelopt_safetensors_load_through_sanitize(tmp_path, cls, spec_name):
    """A tiny ModelOpt-named checkpoint loads strictly into each module class."""
    rng = np.random.default_rng(21)
    n, k = 32, 64
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    block = rng.integers(1, 127, (n, k // 16), dtype=np.uint8)
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {
        "layers.0.proj.weight": ("U8", packed),
        "layers.0.proj.weight_scale": ("F8_E4M3", block),
        "layers.0.proj.weight_scale_2": ("F32", np.array(0.0625, np.float32)),
        "layers.0.proj.input_scale": ("F32", np.array(1.5, np.float32)),
        "layers.0.norm.weight": ("BF16", np.zeros(k, np.uint16)),
    })

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = cls(k, n)
            self.norm = nn.RMSNorm(k)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    model = Tiny()
    spec = model.layers[0].proj.spec
    if spec.input_scale == "absent":
        spec = replace(spec, input_scale="ignored")
    weights = tech.sanitize_modelopt(mx.load(str(path)), ["layers.0.proj"], spec)
    assert "layers.0.proj.input_scale" not in weights
    assert set(weights) == {"layers.0.proj.weight", "layers.0.proj.scales",
                            f"layers.0.proj.{spec_name}", "layers.0.norm.weight"}
    model.load_weights(list(weights.items()), strict=True)
    proj = model.layers[0].proj
    assert proj["weight"].dtype == mx.uint32 and proj["weight"].shape == (n, k // 8)
    np.testing.assert_array_equal(np.asarray(proj["weight"]), packed.view("<u4"))
    x = _inputs(rng, (2, k), mx.float32)
    from mlx2.conversion.nvfp4 import decode_rows

    np.testing.assert_allclose(np.asarray(proj(x)), np.asarray(x) @ decode_rows(packed, block, 0.0625).T,
                               rtol=3e-5, atol=3e-5)
    with pytest.raises(ValueError, match="input_scale"):
        tech.sanitize_modelopt(mx.load(str(path)), ["layers.0.proj"],
                               replace(spec, input_scale="absent"))


@pytest.mark.parametrize("bad,match", [
    (np.array(0.0, np.float32), "finite and positive"),
    (np.array(-0.5, np.float32), "finite and positive"),
    (np.array(np.inf, np.float32), "finite and positive"),
    (np.array(np.nan, np.float32), "finite and positive"),
    (np.array([0.5, 0.25], np.float32), "scalar"),
    (np.array([0.5], np.float32), "scalar"),
])
def test_sanitize_refuses_bad_tensor_scales(bad, match):
    rng = np.random.default_rng(2)
    weights = {"p.weight": mx.array(rng.integers(0, 256, (8, 32), dtype=np.uint8)),
               "p.weight_scale": mx.array(rng.integers(1, 120, (8, 4), dtype=np.uint8)),
               "p.weight_scale_2": mx.array(bad)}
    with pytest.raises(ValueError, match=match):
        tech.sanitize_modelopt(weights, ["p"], modelopt_nvfp4.MODELOPT_SPEC)


@pytest.mark.parametrize("code", [0x80, 0x81, 0xFF, 0x7F])
def test_sanitize_refuses_negative_or_nan_e4m3_block_scales(code):
    from mlx2.conversion.nvfp4 import decode_e4m3

    block = np.full((8, 4), 56, np.uint8)
    block[3, 2] = code
    with pytest.raises(ValueError, match="finite and nonnegative"):
        decode_e4m3(block)  # the loader rule being matched
    weights = {"p.weight": mx.zeros((8, 32), dtype=mx.uint8),
               "p.weight_scale": mx.array(block),
               "p.weight_scale_2": mx.array(np.float32(0.5))}
    with pytest.raises(ValueError, match="finite and nonnegative"):
        tech.sanitize_modelopt(weights, ["p"], modelopt_nvfp4.MODELOPT_SPEC)
    block[3, 2] = 0x7E  # largest finite code is accepted
    weights["p.weight_scale"] = mx.array(block)
    assert "p.scales" in tech.sanitize_modelopt(weights, ["p"], modelopt_nvfp4.MODELOPT_SPEC)


# --------------------------------------------------------------------------
# proposal measurement: which global-scale law is closer to an fp64 oracle
# --------------------------------------------------------------------------

def _law_errors(dtype, trials=6):
    from mlx2.conversion.nvfp4 import decode_rows

    rng = np.random.default_rng(1234)
    errors = {law: [] for law in tech.GLOBAL_SCALE_LAWS}
    for _ in range(trials):
        n, k = 96, 256
        packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        block = rng.integers(40, 70, (n, k // 16), dtype=np.uint8)
        scale = np.float32(rng.uniform(1e-4, 1e-2))
        x = mx.array(rng.normal(size=(8, k)).astype(np.float32)).astype(dtype)
        oracle = (np.asarray(x.astype(mx.float32)).astype(np.float64)
                  @ decode_rows(packed, block, scale).astype(np.float64).T)
        w, s = mx.array(packed.view("<u4")), mx.array(block)
        for law in tech.GLOBAL_SCALE_LAWS:
            y = tech.reference(x, w, s, mx.array(scale), None, NVFP4Spec(global_scale_law=law))
            got = np.asarray(y.astype(mx.float32)).astype(np.float64)
            errors[law].append(np.abs(got - oracle).mean() / np.abs(oracle).mean())
    return {law: float(np.mean(v)) for law, v in errors.items()}


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_fp32_product_law_is_not_less_accurate(dtype):
    errors = _law_errors(dtype)
    # One rounding (fp32_product) versus two (scale_rounded_product).
    assert errors["fp32_product"] <= errors["scale_rounded_product"], errors
