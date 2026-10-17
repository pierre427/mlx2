"""Packing, independent stored-precision error and streaming/fusion oracles."""
import hashlib
import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from mlx2.conversion.affine import bf16_float, pack_codes, quantize_rows
from mlx2.conversion.nemotron35 import ShardWriter, fused_destination, tensor_bytes
from mlx2.conversion.nemotron35_plan import make_plan
from mlx2.conversion.nvfp4 import read_header


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_packing_matches_independent_scalar_bits_and_mlx_dequantize(bits):
    mx.set_default_device(mx.cpu)
    codes = (np.arange(128).reshape(2, 64) * 13 + 7) % (1 << bits)
    packed = pack_codes(codes, bits)
    expected = []
    for row in codes:
        stream = sum(int(value) << (index * bits) for index, value in enumerate(row))
        expected.append(list(stream.to_bytes(64 * bits // 8, "little")))
    assert packed.tobytes() == np.array(expected, np.uint8).tobytes()
    actual = mx.dequantize(mx.array(packed), mx.ones((2, 1)), mx.zeros((2, 1)),
                           group_size=64, bits=bits, mode="affine")
    np.testing.assert_array_equal(np.asarray(actual), codes)


@pytest.mark.parametrize("weighted", [False, True])
def test_refit_uses_stored_bf16_and_improves_weighted_error(weighted):
    rng = np.random.default_rng(10)
    values = bf16_float(rng.normal(size=(9, 128)).astype(np.float32))
    values[0, 0] *= 12
    importance = np.geomspace(.01, 50, 128).astype(np.float32) if weighted else None
    packed, scales, biases, receipt = quantize_rows(values, importance=importance)
    actual = np.asarray(mx.dequantize(mx.array(packed), mx.array(scales.astype(np.uint32) << 16).view(mx.float32),
                                      mx.array(biases.astype(np.uint32) << 16).view(mx.float32),
                                      group_size=64, bits=4, mode="affine"))
    assert np.isfinite(actual).all()
    assert receipt["error_after"] <= receipt["error_before"]
    assert receipt["error_after"] < receipt["error_before"] * .99
    if importance is None:
        np.testing.assert_allclose(np.square(actual - values).sum(dtype=np.float64), receipt["error_after"], rtol=1e-5)
    else:
        groups = importance.reshape(-1, 64)
        normalized = (groups / groups.mean(axis=-1, keepdims=True)).reshape(-1)
        np.testing.assert_allclose((np.square(actual - values) * normalized).sum(dtype=np.float64), receipt["error_after"], rtol=1e-5)
        plain, plain_scales, plain_biases, _ = quantize_rows(values)
        baseline = np.asarray(mx.dequantize(mx.array(plain),
            mx.array(plain_scales.astype(np.uint32) << 16).view(mx.float32),
            mx.array(plain_biases.astype(np.uint32) << 16).view(mx.float32), group_size=64, bits=4))
        plain_error = (np.square(baseline - values) * normalized).reshape(9, 2, 64).sum(axis=-1)
        weighted_error = (np.square(actual - values) * normalized).reshape(9, 2, 64).sum(axis=-1)
        assert np.all(weighted_error <= plain_error + 1e-6)
        assert receipt["baseline"] == "unweighted_lsq"


def test_constant_groups_and_invalid_importance():
    values = np.full((2, 64), .75, np.float32)
    packed, _scale, _bias, receipt = quantize_rows(values)
    assert receipt["error_after"] == 0
    assert not np.any(packed)
    for invalid in (np.zeros(64), np.ones(63), np.full(64, np.nan), -np.ones(64)):
        with pytest.raises(ValueError, match="Importance"):
            quantize_rows(values, importance=invalid)


def test_plan_protects_sensitive_tensors_and_includes_all_bytes():
    from pathlib import Path
    config = json.loads((Path(__file__).parent / "fixtures/nemotron35_super_config.json").read_text())
    plan = make_plan(config)
    prefix = "language_model."
    assert plan["schema"][prefix + "backbone.embeddings.weight"]["dtype"] == "BF16"
    assert plan["schema"][prefix + "mtp.layers.0.eh_proj.weight"]["dtype"] == "BF16"
    assert plan["modules"][prefix + "lm_head"]["bits"] == 8
    assert all(value["bits"] >= 8 for name, value in plan["modules"].items() if ".switch_mlp." not in name)
    assert all("latent_proj" not in name and ".gate" not in name for name in plan["modules"])
    assert sum(tensor_bytes(entry) for entry in plan["schema"].values()) == plan["tensor_bytes"]
    assert 75_000_000_000 < plan["tensor_bytes"] < 85_000_000_000


def test_shard_writer_places_experts_into_final_tensor_without_stacking(tmp_path):
    schema = {"moe.fc1.weight": {"shape": [2, 3, 4], "dtype": "U32"},
              "small": {"shape": [2], "dtype": "BF16"}}
    writer = ShardWriter(tmp_path, schema, max_shard_bytes=100)
    first, second = np.arange(12, dtype="<u4"), np.arange(12, 24, dtype="<u4")
    writer.write("moe.fc1.weight", first.nbytes, second.tobytes())
    writer.write("moe.fc1.weight", 0, first.tobytes())
    writer.write("small", 0, b"\0\0\0\0")
    with pytest.raises(ValueError, match="escapes"):
        writer.write("small", 1, b"1234")
    writer.close()
    filename, start, _ = writer.locations["moe.fc1.weight"]
    header, _ = read_header(tmp_path / filename)
    assert header["moe.fc1.weight"]["shape"] == [2, 3, 4]
    with (tmp_path / filename).open("rb") as handle:
        handle.seek(start)
        assert handle.read(96) == np.arange(24, dtype="<u4").tobytes()
    assert fused_destination("language_model.backbone.layers.1.mixer.experts.511.down_proj.weight") == (
        "language_model.backbone.layers.1.mixer.switch_mlp.fc2.weight", 511)


def test_streaming_conversion_fuses_real_payloads_and_hashes_source(monkeypatch, tmp_path):
    from mlx2.conversion import nemotron35
    from mlx2.conversion.nvfp4 import bf16_bits
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{}')
    (source / "model.safetensors.index.json").write_text('{}')
    arrays = {
        f"language_model.backbone.layers.1.mixer.experts.{expert}.up_proj.weight":
        bf16_bits(np.random.default_rng(expert).normal(size=(3, 64)).astype(np.float32))
        for expert in (0, 1)
    }
    arrays["norm.weight"] = bf16_bits(np.array([1., 2.], np.float32))
    header, payload, offset = {}, b"", 0
    for name, array in arrays.items():
        raw = array.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        payload += raw
    raw_header = json.dumps(header).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    file = source / "input.safetensors"
    file.write_bytes(struct.pack('<Q', len(raw_header)) + raw_header + payload)
    info = {"path": file, "tensors": header, "payload": 8 + len(raw_header),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "stat": file.stat()}
    prefix = "language_model.backbone.layers.1.mixer.switch_mlp.fc1"
    schema = {prefix + ".weight": {"shape": [2, 3, 8], "dtype": "U32"},
              prefix + ".scales": {"shape": [2, 3, 1], "dtype": "BF16"},
              prefix + ".biases": {"shape": [2, 3, 1], "dtype": "BF16"},
              "norm.weight": {"shape": [2], "dtype": "BF16"}}
    plan = {"schema": schema, "modules": {prefix: {"bits": 4, "group_size": 64, "mode": "affine"}},
            "recipe": "test", "tensor_bytes": sum(tensor_bytes(v) for v in schema.values()), "bytes_by_category": {}}
    monkeypatch.setattr(nemotron35, "inspect_source", lambda p: ({}, {file.name: info}))
    monkeypatch.setattr(nemotron35, "metadata_snapshot", lambda p: {})
    monkeypatch.setattr(nemotron35, "make_plan", lambda c, **kw: plan)
    destination = tmp_path / "converted"
    receipt = nemotron35.convert_checkpoint(source, destination, reserve_bytes=0)
    assert receipt["source_shards"][0]["sha256"] == info["sha256"]
    assert receipt["activation_aware"] is False
    shard = next(destination.glob('*.safetensors'))
    values = mx.load(str(shard))
    for expert in (0, 1):
        original = (arrays[f"language_model.backbone.layers.1.mixer.experts.{expert}.up_proj.weight"].astype(np.uint32) << 16).view(np.float32)
        packed, scales, biases, _ = quantize_rows(original)
        np.testing.assert_array_equal(np.asarray(values[prefix + ".weight"][expert]), packed)
        np.testing.assert_array_equal(np.asarray(values[prefix + ".scales"][expert].view(mx.uint16)), scales)
        np.testing.assert_array_equal(np.asarray(values[prefix + ".biases"][expert].view(mx.uint16)), biases)
    with pytest.raises(FileExistsError):
        nemotron35.convert_checkpoint(source, destination, reserve_bytes=0)
    from mlx2.adapters.nemotron35_media import PROCESSOR_REVISION
    from mlx2.adapters.nemotron35_super_config import (
        SOURCE_CONFIG_SHA256,
        SOURCE_INDEX_SHA256,
        SOURCE_REVISION,
    )
    from mlx2.conversion.importance import ActivationMoments
    binding = {"source_revision": SOURCE_REVISION, "source_config_sha256": SOURCE_CONFIG_SHA256,
               "source_index_sha256": SOURCE_INDEX_SHA256, "proxy_artifact": "a" * 64,
               "corpus_sha256": "b" * 64, "collector_sha256": "c" * 64,
               "processor_revision": PROCESSOR_REVISION}
    moments = ActivationMoments({prefix: (2, 64)}, binding)
    importance = np.geomspace(.01, 100., 64).astype(np.float32)
    moments.add_reduced(prefix, np.tile(importance * 64, (2, 1)), np.array([64, 64]))
    for index in range(1024):
        moments.record_sample(f"sample-{index}", modality="image_text", input_sha256=f"{index:064x}")
    moments.save(tmp_path / "imatrix")
    calibrated = nemotron35.convert_checkpoint(source, tmp_path / "calibrated", reserve_bytes=0,
                                               importance_path=tmp_path / "imatrix")
    assert calibrated["activation_aware"] is True
    assert calibrated["fully_activation_weighted"] is True
    assert calibrated["calibration"]["coverage"]["complete"] is True
    assert calibrated["errors"][prefix]["error_after"] <= calibrated["errors"][prefix]["error_before"]

    fallback = ActivationMoments({prefix: (2, 64)}, binding)
    fallback.add_reduced(prefix, np.vstack((importance * 64, np.zeros(64))), np.array([64, 0]))
    for index in range(1024):
        fallback.record_sample(f"fallback-{index}", modality="image_text", input_sha256=f"{index:064x}")
    fallback.save(tmp_path / "fallback-imatrix")
    hybrid = nemotron35.convert_checkpoint(source, tmp_path / "hybrid", reserve_bytes=0,
                                           importance_path=tmp_path / "fallback-imatrix")
    policy = hybrid["calibration"]["expert_policy"]
    assert hybrid["activation_aware"] is True and hybrid["fully_activation_weighted"] is False
    assert policy["activation_weighted_experts"] == 1 and policy["fallback_experts"] == 1
    assert policy["modules"][prefix]["activation_weighted_mask"] == [True, False]
    assert policy["modules"][prefix]["observation_counts"] == [64, 0]
    assert policy["modules"][prefix]["fallback_reasons"] == [None, "undercovered"]
    assert hybrid["calibration_mode"] == "hybrid_activation_weighted_q4_with_weight_mse_q4_fallback"
    hybrid_values = mx.load(str(next((tmp_path / "hybrid").glob("*.safetensors"))))
    baseline_values = mx.load(str(next(destination.glob("*.safetensors"))))
    for suffix in ("weight", "scales", "biases"):
        hybrid_value = hybrid_values[f"{prefix}.{suffix}"][1]
        baseline_value = baseline_values[f"{prefix}.{suffix}"][1]
        if suffix != "weight":
            hybrid_value = hybrid_value.view(mx.uint16)
            baseline_value = baseline_value.view(mx.uint16)
        np.testing.assert_array_equal(np.asarray(hybrid_value), np.asarray(baseline_value))
    info["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        nemotron35.convert_checkpoint(source, tmp_path / 'rejected', reserve_bytes=0)
    assert not (tmp_path / 'rejected').exists()
