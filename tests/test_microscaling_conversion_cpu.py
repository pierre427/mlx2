"""Offline SC artifact conversion: byte preservation, references and failure paths.

Synthetic tensors only; no model checkpoint or GPU inference is exercised.
"""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from mlx2.runtime.microscaling.conversion import ConversionError, Source, convert_artifact
from mlx2.runtime.microscaling.format import from_mxfp4_numpy, metadata_from_mxfp4_scales_numpy


def write_source(path, tensors, config=None):
    path.mkdir()
    offset = 0
    header, buffers = {}, []
    for name, (dtype, array) in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        buffers.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(buffers))
    (path / "config.json").write_text(json.dumps(config or {"model_type": "fixture"}))
    (path / "tokenizer_config.json").write_text('{"chat_template":"fixture"}')
    (path / "LICENSE").write_text("Fixture license bytes\n")
    return path


def bf16(value):
    return (np.asarray(value, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def float_bf16(value):
    return (np.asarray(value, np.uint16).astype(np.uint32) << 16).view(np.float32)


def tensor(source, name):
    t = source.tensors[name]
    dtype = {"U32": "<u4", "U8": "u1", "BF16": "<u2", "F32": "<f4"}[t.dtype]
    return np.frombuffer(t.raw(), dtype=dtype).reshape(t.shape).copy()


def native_decode(words, scales):
    k = words.shape[-1] * 8
    nib = np.stack([(words >> (4 * lane)) & 15 for lane in range(8)], -1).reshape(*words.shape[:-1], k)
    value = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])[nib & 7]
    return np.where(nib & 8, -value, value) * np.repeat(np.exp2(scales.astype(int) - 127), 32, -1)


def test_dense_bf16_standard_bytes_and_unselected_payload(tmp_path):
    rng = np.random.default_rng(61)
    w = bf16(rng.normal(0, .02, (64, 256)))
    norm = bf16(np.linspace(.5, 1.5, 64))
    embedding = bf16(rng.normal(0, .1, (32, 256)))
    src = write_source(tmp_path / "source", {
        "model.layers.0.mlp.gate_proj.weight": ("BF16", w),
        "model.layers.0.input_layernorm.weight": ("BF16", norm),
        "model.embed_tokens.weight": ("BF16", embedding),
        "lm_head.weight": ("BF16", embedding),
    })
    before = hashlib.sha256((src / "model.safetensors").read_bytes()).hexdigest()
    out = tmp_path / "converted"
    result = convert_artifact(src, out, workers=2, rows_per_chunk=7, shard_bytes=20_000)
    output = Source(out)
    words = tensor(output, "model.layers.0.mlp.gate_proj.weight")
    scales = tensor(output, "model.layers.0.mlp.gate_proj.scales")
    packed = from_mxfp4_numpy(words, scales)
    metadata_from_mxfp4_scales_numpy(scales)
    assert packed.shape == w.shape
    decoded = native_decode(words, scales)
    original = float_bf16(w)
    assert np.sum((decoded - original)**2) / np.sum(original**2) < .04
    assert np.array_equal(tensor(output, "model.layers.0.input_layernorm.weight"), norm)
    assert np.array_equal(tensor(output, "model.embed_tokens.weight"), embedding)
    assert np.array_equal(tensor(output, "lm_head.weight"), embedding)
    assert (out / "tokenizer_config.json").read_bytes() == (src / "tokenizer_config.json").read_bytes()
    assert (out / "LICENSE").read_bytes() == (src / "LICENSE").read_bytes()
    assert not (out / ".incomplete").exists()
    assert result["status"] == "complete" and result["converted_modules"] == 1
    assert result["source_unchanged"] and not result["qualified"]
    assert before == hashlib.sha256((src / "model.safetensors").read_bytes()).hexdigest()
    quant = json.loads((out / "config.json").read_text())["quantization"]
    assert quant["model.layers.0.mlp.gate_proj"] == {"bits": 4, "group_size": 32, "mode": "mxfp4"}


def test_numbered_experts_normalize_exactly_and_convert_to_grouped(tmp_path):
    rng = np.random.default_rng(18)
    tensors = {}
    source_values = {}
    for e in range(3):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.0.mlp.experts.{e}.{projection}.weight"
            source_values[key] = bf16(rng.normal(0, .02, (64, 128)))
            tensors[key] = ("BF16", source_values[key])
    attention = bf16(rng.normal(0, .02, (64, 128)))
    tensors["model.layers.0.self_attn.q_proj.weight"] = ("BF16", attention)
    tensors["model.layers.0.mlp.gate.weight"] = ("BF16", bf16(np.ones((3, 128))))
    src = write_source(tmp_path / "source", tensors)
    normalized = tmp_path / "reference"
    receipt = convert_artifact(src, normalized, scope="experts", normalize_only=True)
    reference = Source(normalized)
    assert receipt["converted_modules"] == 0 and receipt["normalized_groups"] == 3
    assert "quantization" not in reference.config
    for projection in ("gate_proj", "up_proj", "down_proj"):
        target = f"model.layers.0.mlp.switch_mlp.{projection}.weight"
        actual = tensor(reference, target)
        assert actual.shape == (3, 64, 128)
        assert np.array_equal(actual, np.stack([source_values[f"model.layers.0.mlp.experts.{e}.{projection}.weight"] for e in range(3)]))
    converted = tmp_path / "candidate"
    receipt = convert_artifact(src, converted, scope="experts", workers=2, rows_per_chunk=16)
    output = Source(converted)
    assert receipt["converted_modules"] == 3
    for projection in ("gate_proj", "up_proj", "down_proj"):
        key = f"model.layers.0.mlp.switch_mlp.{projection}"
        packed = from_mxfp4_numpy(tensor(output, key + ".weight"), tensor(output, key + ".scales"))
        assert packed.shape == (3, 64, 128)
    assert np.array_equal(tensor(output, "model.layers.0.self_attn.q_proj.weight"), attention)
    assert output.tensors["model.layers.0.mlp.gate.weight"].dtype == "BF16"


def test_quantized_source_requires_explicit_diagnostic_opt_in(tmp_path):
    rng = np.random.default_rng(19)
    words = rng.integers(0, 2**32, (64, 32), dtype=np.uint32)
    scales = np.full((64, 4), .0125, np.float32)
    biases = np.full((64, 4), -.1, np.float32)
    key = "model.layers.0.mlp.gate_proj"
    src = write_source(tmp_path / "source", {key + ".weight": ("U32", words),
        key + ".scales": ("F32", scales), key + ".biases": ("F32", biases)},
        {"model_type": "fixture", "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}})
    out = tmp_path / "refused"
    with pytest.raises(ConversionError, match="already-quantized"):
        convert_artifact(src, out)
    assert not out.exists()
    out = tmp_path / "diagnostic"
    result = convert_artifact(src, out, allow_requantization=True)
    output = Source(out)
    assert output.tensors[key + ".scales"].dtype == "U8"
    assert key + ".biases" not in output.tensors
    assert result["projection_stats"][0]["source_requantized"]
    assert result["allow_requantization"] and not result["qualified"]


def test_nonempty_and_nested_destinations_refuse_without_mutation(tmp_path):
    src = write_source(tmp_path / "source", {"model.layers.0.mlp.up_proj.weight": ("F32", np.zeros((64, 128), np.float32))})
    out = tmp_path / "existing"
    out.mkdir(); (out / "precious").write_bytes(b"preserve")
    with pytest.raises(ConversionError, match="nonempty"):
        convert_artifact(src, out)
    assert (out / "precious").read_bytes() == b"preserve"
    for target in (src, src / "nested", tmp_path):
        with pytest.raises(ConversionError, match="separate"):
            convert_artifact(src, target)


def test_failure_retains_incomplete_receipt_and_never_marks_ready(tmp_path):
    weights = np.zeros((64, 128), np.float32); weights[9, 9] = np.nan
    src = write_source(tmp_path / "source", {"model.layers.0.mlp.up_proj.weight": ("F32", weights)})
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="nonfinite"):
        convert_artifact(src, out, workers=2, rows_per_chunk=8)
    receipt = json.loads((out / "mlx2-sc-mxfp4-conversion.json").read_text())
    assert receipt["status"] == "failed" and (out / ".incomplete").exists()
    assert not (out / "model.safetensors.index.json").exists()
    with pytest.raises(ConversionError, match="nonempty"):
        convert_artifact(src, out)


def test_late_gate_failure_leaves_no_loadable_artifact(tmp_path):
    rng = np.random.default_rng(1)
    src = write_source(tmp_path / "source", {
        "model.layers.0.mlp.gate_proj.weight": ("BF16", bf16(rng.normal(0, .02, (64, 256))))})
    out = tmp_path / "out"
    def mutate(_):
        (src / "generation_config.json").write_text("{}")
    with pytest.raises(ConversionError, match="source identity changed"):
        convert_artifact(src, out, progress=mutate)
    assert json.loads((out / "mlx2-sc-mxfp4-conversion.json").read_text())["status"] == "failed"
    assert (out / ".incomplete").exists()
    assert not (out / "config.json").exists()
    assert not (out / "model.safetensors.index.json").exists()
    with pytest.raises(Exception):
        Source(out)


def test_scope_all_skips_non_projection_convolution_weights(tmp_path):
    from mlx2.runtime.microscaling.conversion import plan_tensors
    rng = np.random.default_rng(2)
    src = write_source(tmp_path / "source", {
        "model.layers.0.mlp.gate_proj.weight": ("BF16", bf16(rng.normal(0, .02, (64, 256)))),
        "model.layers.0.linear_attn.in_proj_qkvz.weight": ("BF16", bf16(rng.normal(0, .02, (128, 256)))),
        "model.layers.0.linear_attn.conv1d.weight": ("BF16", bf16(rng.normal(0, .02, (128, 1, 4)))),
    })
    plans = {p.name: p.convert for p in plan_tensors(Source(src), "all")}
    assert plans["model.layers.0.linear_attn.conv1d.weight"] is False
    assert plans["model.layers.0.linear_attn.in_proj_qkvz.weight"] is True
    convert_artifact(src, tmp_path / "out", scope="all")


def test_missing_expert_and_unsupported_geometry_fail_before_output(tmp_path):
    src = write_source(tmp_path / "source", {
        "model.layers.0.mlp.experts.1.up_proj.weight": ("F32", np.zeros((64, 128), np.float32))})
    with pytest.raises(ConversionError, match="contiguous"):
        convert_artifact(src, tmp_path / "bad-experts", scope="experts")
    assert not (tmp_path / "bad-experts").exists()
    src = write_source(tmp_path / "odd-source", {"model.layers.0.mlp.up_proj.weight": ("F32", np.zeros((64, 160), np.float32))})
    with pytest.raises(ConversionError, match="geometry"):
        convert_artifact(src, tmp_path / "bad-geometry")
    assert not (tmp_path / "bad-geometry").exists()


def test_chunk_and_worker_choices_preserve_payloads(tmp_path):
    src = write_source(tmp_path / "source", {"model.layers.0.mlp.up_proj.weight": ("F32", np.random.default_rng(99).normal(size=(128, 256)).astype(np.float32))})
    outputs = []
    for i, (rows, workers) in enumerate(((1, 1), (17, 3))):
        out = tmp_path / str(i)
        convert_artifact(src, out, rows_per_chunk=rows, workers=workers)
        outputs.append(Source(out))
    assert set(outputs[0].tensors) == set(outputs[1].tensors)
    for key in outputs[0].tensors:
        assert np.array_equal(tensor(outputs[0], key), tensor(outputs[1], key))


def test_malformed_quantized_metadata_fails_before_destination_creation(tmp_path):
    key = "model.layers.0.mlp.gate_proj"
    src = write_source(tmp_path / "source", {
        key + ".weight": ("U32", np.zeros((64, 32), np.uint32)),
        key + ".scales": ("F32", np.zeros((63, 4), np.float32)),
        key + ".biases": ("F32", np.zeros((64, 4), np.float32))},
        {"model_type": "fixture", "quantization": {"bits": 4, "group_size": 64}})
    with pytest.raises(ConversionError, match="scale geometry"):
        convert_artifact(src, tmp_path / "bad", allow_requantization=True)
    assert not (tmp_path / "bad").exists()


def test_offline_conversion_can_constrain_native_mxfp4_wide_spread(tmp_path):
    key = 'model.layers.0.mlp.up_proj'
    words = np.full((64, 16), 0x76543210, np.uint32)
    scales = np.broadcast_to(np.array([123, 124, 125, 127], np.uint8), (64, 4)).copy()
    # Runtime remains lossless-only and must refuse these source rows.
    with pytest.raises(ValueError, match='spread'):
        from_mxfp4_numpy(words, scales)
    src = write_source(tmp_path / 'source', {
        key + '.weight': ('U32', words), key + '.scales': ('U8', scales)},
        {'model_type': 'fixture', 'quantization': {'bits': 4, 'group_size': 32, 'mode': 'mxfp4'}})
    out = tmp_path / 'converted'
    receipt = convert_artifact(src, out, allow_requantization=True, rows_per_chunk=9)
    result = Source(out)
    output_scales = tensor(result, key + '.scales')
    metadata_from_mxfp4_scales_numpy(output_scales)
    assert np.max(np.ptp(output_scales.astype(np.int16), axis=-1)) <= 3
    assert receipt['projection_stats'][0]['source_requantized']
    assert np.array_equal(tensor(Source(src), key + '.weight'), words)
    # The decoder implements the standard block scales, not an SC-only
    # shortcut. The resulting source values enter the normal offline codec.
    from mlx2.runtime.microscaling.format import pack_numpy, to_mxfp4_scales_numpy
    reference = pack_numpy(native_decode(words, scales), allow_requantization=True)
    assert np.array_equal(tensor(result, key + '.weight'), reference.codes)
    assert np.array_equal(output_scales, to_mxfp4_scales_numpy(reference))


def test_f64_source_rounding_does_not_pass_through_f32(tmp_path):
    from mlx2.runtime.microscaling.format import pack_numpy, to_mxfp4_scales_numpy
    weights = np.ones((64, 128), np.float64)
    weights[0, 0] = .875 + 1e-10  # F32 would place this exactly on a midpoint.
    reference = pack_numpy(weights, allow_requantization=True)
    narrowed = pack_numpy(weights.astype(np.float32), allow_requantization=True)
    assert not np.array_equal(reference.codes, narrowed.codes)
    key = 'model.layers.0.mlp.up_proj'
    src = write_source(tmp_path / 'source', {key + '.weight': ('F64', weights)})
    out = tmp_path / 'converted'
    convert_artifact(src, out)
    result = Source(out)
    assert np.array_equal(tensor(result, key + '.weight'), reference.codes)
    assert np.array_equal(tensor(result, key + '.scales'), to_mxfp4_scales_numpy(reference))
