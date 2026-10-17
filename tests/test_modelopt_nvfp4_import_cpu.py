"""Byte and numerical oracles for NVFP4 import; no model or GPU execution."""
import json
import struct

import numpy as np
import pytest

from mlx2.conversion.nvfp4 import (
    bf16_bits,
    convert_checkpoint,
    decode_e4m3,
    decode_rows,
    pack_group16,
    read_header,
    unpack_group16,
)


def source_artifact(path):
    path.mkdir()
    config = {"architectures": ["LiLiCorrDraftModel"], "quantization_config": {
        "quant_algo": "W4A16_NVFP4", "quant_method": "modelopt_fp4", "group_size": 16}}
    (path / "config.json").write_text(json.dumps(config))
    arrays = {
        "head.weight": ("U8", np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]], np.uint8)),
        "head.weight_scale": ("F8_E4M3", np.array([[1]], np.uint8)),
        "head.weight_scale_2": ("F32", np.array(.0001, "<f4")),
        "head.bias": ("BF16", np.array([0x3f80], "<u2")),
    }
    header, payload = {}, b""
    for name, (dtype, array) in arrays.items():
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    raw = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return arrays


def tensor(path, name, dtype):
    header, start = read_header(path)
    item = header[name]
    with path.open("rb") as handle:
        handle.seek(start + item["data_offsets"][0])
        data = handle.read(item["data_offsets"][1] - item["data_offsets"][0])
    return np.frombuffer(data, dtype=dtype).reshape(item["shape"])


def test_all_fp4_codes_and_two_level_scale():
    packed = np.array([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]], np.uint8)
    # E4M3 code 1 is 2^-9, intentionally below the fold-to-FP8 safe range.
    actual = decode_rows(packed, np.array([[1]], np.uint8), np.float32(.0001))
    expected = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    np.testing.assert_array_equal(actual[0], expected * np.float32(2**-9) * np.float32(.0001))
    assert np.count_nonzero(actual) == 14


def test_e4m3_boundaries_and_invalid_scales():
    np.testing.assert_array_equal(decode_e4m3([0, 1, 8, 56, 126]), [0, 2**-9, 2**-6, 1, 448])
    for value in (127, 128, 255):
        with pytest.raises(ValueError, match="scales"):
            decode_e4m3([value])


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_tensor_global_scale(scale):
    with pytest.raises(ValueError, match="tensor scale"):
        decode_rows(np.zeros((1, 8), np.uint8), np.ones((1, 1), np.uint8), scale)


def test_bf16_rounding_ties_to_even():
    values = np.array([0x3f808000, 0x3f818000], "<u4").view("<f4")
    np.testing.assert_array_equal(bf16_bits(values), [0x3f80, 0x3f82])


@pytest.mark.parametrize("rows,tile_rows", [(1, 8), (8, 8), (10, 8), (17, 16)])
def test_persistent_group16_pack_round_trips_source_bytes(rows, tile_rows):
    rng = np.random.default_rng(rows + tile_rows)
    packed = rng.integers(0, 256, (rows, 40), dtype=np.uint8)
    scales = rng.integers(0, 127, (rows, 5), dtype=np.uint8)
    packed_tiles, scale_tiles, logical_rows = pack_group16(
        packed, scales, tile_rows=tile_rows
    )
    actual_packed, actual_scales = unpack_group16(
        packed_tiles, scale_tiles, logical_rows
    )
    np.testing.assert_array_equal(actual_packed, packed)
    np.testing.assert_array_equal(actual_scales, scales)
    assert packed_tiles.shape[-1] == 8
    assert scale_tiles.shape == packed_tiles.shape[:-1]


def test_persistent_pack_refuses_non_group16_geometry():
    with pytest.raises(ValueError, match="group-16"):
        pack_group16(np.zeros((2, 7), np.uint8), np.zeros((2, 1), np.uint8))


@pytest.mark.parametrize("mode", ["bf16", "native"])
def test_complete_atomic_conversion_preserves_source(tmp_path, mode):
    source, output = tmp_path / "source", tmp_path / "output"
    arrays = source_artifact(source)
    before = (source / "model.safetensors").read_bytes()
    receipt = convert_checkpoint(source, output, mode=mode, reserve_bytes=0)
    assert (source / "model.safetensors").read_bytes() == before
    assert receipt["global_scale_preserved"] and not receipt["qualified"]
    assert receipt["source_weight_sha256"] != receipt["output_weight_sha256"]
    weight = output / "model.safetensors"
    if mode == "native":
        np.testing.assert_array_equal(tensor(weight, "head.weight", "<u4").view(np.uint8), arrays["head.weight"][1])
        np.testing.assert_array_equal(tensor(weight, "head.scales", np.uint8), arrays["head.weight_scale"][1])
        assert tensor(weight, "head.global_scale", "<f4") == np.float32(.0001)
    else:
        expected = bf16_bits(decode_rows(arrays["head.weight"][1], arrays["head.weight_scale"][1], np.float32(.0001)))
        np.testing.assert_array_equal(tensor(weight, "head.weight", "<u2"), expected)
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, output, reserve_bytes=0)


def test_truncated_payload_refused_before_output(tmp_path):
    source = tmp_path / "source"
    source_artifact(source)
    path = source / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="span"):
        convert_checkpoint(source, tmp_path / "output", reserve_bytes=0)
    assert not (tmp_path / "output").exists()
