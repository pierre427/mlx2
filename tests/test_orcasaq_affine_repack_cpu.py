"""Small CPU fixtures for the revision-bound OrcaSAQ affine repack."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("repack_orcasaq_qwen38_mlx_affine",
                                              SCRIPT_DIR / "repack_orcasaq_qwen38_mlx_affine.py")
assert SPEC and SPEC.loader
repack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repack)


def test_affine_profile_creates_exact_module_overrides():
    config = {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 64}}
    files = {
        "quant-a": {"key": "language_model.model.layers.0.mlp.gate_proj.weight", "source_type": "IQ4_XS"},
        "quant-b": {"key": "language_model.model.layers.1.mlp.gate_proj.weight", "source_type": "Q5_K"},
        "quant-c": {"key": "language_model.model.layers.2.mlp.gate_proj.weight", "source_type": "Q6_K"},
        "norm": {"key": "language_model.model.layers.0.input_layernorm.weight", "source_type": "F32"},
    }
    result = repack._quant_config(config, files, repack.DEFAULT_PROFILE)
    policy = result["quantization"]
    assert policy["bits"] == 5
    assert [policy[key.removesuffix(".weight")]["bits"] for key in
            (files[name]["key"] for name in ("quant-a", "quant-b", "quant-c"))] == [5, 6, 8]
    assert "language_model.model.layers.0.input_layernorm" not in policy
    assert "quantization" not in config


@pytest.mark.parametrize("bits", [5, 6, 8])
def test_cpu_affine_shard_roundtrip_and_added_error(tmp_path, bits):
    mx.set_default_device(mx.cpu)
    name = "language_model.model.layers.0.mlp.gate_proj.weight"
    values = mx.sin(mx.arange(256, dtype=mx.float32).reshape(4, 64) * 0.173).astype(mx.bfloat16)
    source_file = tmp_path / "source.safetensors"
    target_file = tmp_path / "target.safetensors"
    mx.save_safetensors(str(source_file), {name: values})
    record = {"key": name, "source_type": "IQ4_XS"}
    entry = repack._pack_one(source_file, target_file, record, bits)
    assert entry["bits"] == bits
    assert entry["sha256"] == repack.sha256(target_file)
    packed = mx.load(str(target_file))
    prefix = name.removesuffix(".weight")
    assert set(packed) == {name, prefix + ".scales", prefix + ".biases"}
    restored = mx.dequantize(packed[name], packed[prefix + ".scales"],
                            packed[prefix + ".biases"], group_size=64, bits=bits, mode="affine")
    mx.eval(restored)
    assert bool(mx.all(mx.isfinite(restored)).item())
    assert float(mx.max(mx.abs(restored - values)).item()) > 0.0


def test_f32_tensor_is_copied_byte_exactly(tmp_path):
    mx.set_default_device(mx.cpu)
    name = "language_model.model.layers.0.input_layernorm.weight"
    source_file = tmp_path / "source.safetensors"
    target_file = tmp_path / "target.safetensors"
    mx.save_safetensors(str(source_file), {name: mx.array([0.875, 1.125], dtype=mx.float32)})
    entry = repack._pack_one(source_file, target_file, {"key": name, "source_type": "F32"}, None)
    assert entry["bits"] is None
    assert source_file.read_bytes() == target_file.read_bytes()


def test_mixed_bits_load_under_adapter_quantize_contract():
    from mlx import nn

    mx.set_default_device(mx.cpu)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 4, bias=False)
            self.b = nn.Linear(64, 4, bias=False)
            self.c = nn.Linear(64, 4, bias=False)

    model = Tiny()
    policy = {name: {"group_size": 64, "bits": bits, "mode": "affine"}
              for name, bits in (("a", 5), ("b", 6), ("c", 8))}
    packed_weights = []
    for name, bits in (("a", 5), ("b", 6), ("c", 8)):
        source = mx.sin(mx.arange(256, dtype=mx.float32).reshape(4, 64) * (0.11 + bits / 100))
        weight, scales, biases = mx.quantize(source, group_size=64, bits=bits, mode="affine")
        packed_weights.extend(((name + ".weight", weight),
                               (name + ".scales", scales),
                               (name + ".biases", biases)))
    nn.quantize(model, group_size=64, bits=5, mode="affine",
                class_predicate=lambda name, module: policy.get(name, False))
    model.load_weights(packed_weights, strict=True)
    for name, bits in (("a", 5), ("b", 6), ("c", 8)):
        module = getattr(model, name)
        assert module.bits == bits
        result = module(mx.ones((1, 64)))
        mx.eval(result)
        assert result.shape == (1, 4)
