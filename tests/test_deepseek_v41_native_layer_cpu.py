"""Bounded CPU format conversion; no MLX import, model load or GPU execution."""

import importlib.util
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def converter():
    path = Path(__file__).resolve().parents[1] / "scripts/research/deepseek_v41_native_layer.py"
    spec = importlib.util.spec_from_file_location("v41_native_layer_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source(ggml_type, shape, *, scales=None):
    elements, block_bytes = (256, 50) if ggml_type == 19 else (32, 34)
    blocks = math.prod(shape) // elements
    raw = np.random.default_rng(41025).integers(0, 256, (blocks, block_bytes), np.uint8)
    if scales is None:
        scales = np.resize(np.array([.03125, .5, 1., 65504., 2**-24], np.float16), blocks)
    raw[:, :2] = np.asarray(scales, dtype="<f2").reshape(-1, 1).view(np.uint8)
    return raw.reshape(-1)


@pytest.mark.parametrize("ggml_type,shape", [(19, (256,)), (19, (3, 512)), (19, (2, 3, 256)),
                                           (8, (32,)), (8, (2, 96)), (8, (2, 3, 64))])
def test_leading_axes_native_packing_and_exact_weight_values(converter, ggml_type, shape):
    raw = source(ggml_type, shape)
    arrays = converter.convert_affine(raw, ggml_type, shape)
    spec = converter.affine_spec(ggml_type, shape)
    assert [x.shape for x in arrays] == [spec["packed_shape"], spec["scales_shape"], spec["biases_shape"]]
    assert [x.dtype for x in arrays] == [np.uint32, np.float32, np.float32]
    assert all(x.flags.c_contiguous and not np.shares_memory(x, raw) for x in arrays)
    assert sum(x.nbytes for x in arrays) == spec["output_bytes"]
    proof = converter.validate_affine(raw, ggml_type, shape, arrays, require_bitwise=True)
    assert proof["value_exact"] and proof["bitwise_exact"] and proof["signed_zero_differences"] == 0
    before = [x.copy() for x in arrays]
    raw.fill(255)
    assert all(np.array_equal(x, y) for x, y in zip(before, arrays))


def test_every_iq1_codebook_entry_scale_and_delta_against_existing_decoder(converter):
    count = len(converter.IQ1_GRID)
    raw = np.zeros((count, 50), np.uint8)
    raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), np.uint8)
    index = np.arange(count, dtype=np.uint16)
    raw[:, 2:34] = (index & 255).astype(np.uint8)[:, None]
    upper = index >> 8
    high = (upper[:, None] | upper[:, None] << 3 | upper[:, None] << 6 | upper[:, None] << 9)
    high = high | np.arange(8, dtype=np.uint16)[None, :] << 12
    high = high | (np.arange(8, dtype=np.uint16)[None, :] % 2) << 15
    raw[:, 34:] = high.astype("<u2").view(np.uint8)
    proof = converter.validate_affine(raw, 19, (count, 256), require_bitwise=True)
    assert proof["elements"] == count * 256 and proof["bitwise_exact"]


def test_all_q8_codes_pack_low_lane_first(converter):
    raw = source(8, (256,))
    raw.reshape(-1, 34)[:, 2:] = np.arange(256, dtype=np.uint8).reshape(-1, 32)
    packed, scales, biases = converter.convert_affine(raw, 8, (256,))
    expected = (np.arange(256, dtype=np.uint8) ^ np.uint8(128)).view("<u4")
    assert np.array_equal(packed, expected)
    assert converter.validate_affine(raw, 8, (256,), (packed, scales, biases), require_bitwise=True)["bitwise_exact"]


@pytest.mark.parametrize("ggml_type,shape", [(19, (2, 256)), (8, (2, 32))])
def test_signed_zero_is_reported_without_claiming_bitwise_losslessness(converter, ggml_type, shape):
    raw = source(ggml_type, shape, scales=np.array([0., -0.], np.float16))
    if ggml_type == 8:
        raw.reshape(-1, 34)[:, 2:] = np.array([0, 1, 255, 128], np.uint8).repeat(8)
    proof = converter.validate_affine(raw, ggml_type, shape)
    assert proof["value_exact"] and not proof["bitwise_exact"] and proof["signed_zero_differences"] > 0
    with pytest.raises(ValueError, match="signed-zero"):
        converter.validate_affine(raw, ggml_type, shape, require_bitwise=True)


def test_negative_nonzero_iq1_scales_remain_bitwise_exact(converter):
    raw = source(19, (2, 256), scales=np.array([-.03125, -65504.], np.float16))
    assert converter.validate_affine(raw, 19, (2, 256), require_bitwise=True)["bitwise_exact"]


@pytest.mark.parametrize("ggml_type,shape", [(19, (2, 3, 512)), (8, (2, 3, 64))])
def test_chunk_iterator_streams_to_memmaps_without_source_copy(converter, tmp_path, ggml_type, shape):
    raw = source(ggml_type, shape)
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(raw.tobytes())
    mapped = np.memmap(source_path, mode="r", dtype=np.uint8)
    spec = converter.affine_spec(ggml_type, shape)
    outputs = [np.lib.format.open_memmap(tmp_path / f"{name}.npy", mode="w+", dtype=dtype, shape=spec[key])
               for name, dtype, key in (("packed", np.uint32, "packed_shape"),
                                        ("scales", np.float32, "scales_shape"),
                                        ("biases", np.float32, "biases_shape"))]
    chunks = 0
    for chunk in converter.iter_affine_chunks(memoryview(mapped), ggml_type, shape,
                                              rows_per_chunk=5, max_chunk_elements=2 * shape[-1]):
        assert 0 < chunk.row_stop - chunk.row_start <= 2
        for output, value in zip(outputs, (chunk.packed, chunk.scales, chunk.biases)):
            output.reshape(-1, output.shape[-1])[chunk.row_start:chunk.row_stop] = value
        chunks += 1
    assert chunks == 3
    for output in outputs:
        output.flush()
    assert converter.validate_affine(raw, ggml_type, shape, outputs, require_bitwise=True)["bitwise_exact"]
    for output, expected in zip(outputs, converter.convert_affine(raw, ggml_type, shape)):
        assert np.array_equal(output, expected)


@pytest.mark.parametrize("ggml_type", [8, 19])
@pytest.mark.parametrize("scale", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_source_scales_fail_closed(converter, ggml_type, scale):
    width = 32 if ggml_type == 8 else 256
    raw = source(ggml_type, (width,), scales=np.array([scale], np.float16))
    with pytest.raises(ValueError, match="finite"):
        converter.convert_affine(raw, ggml_type, (width,))


def test_shape_buffer_and_allocation_boundaries(converter):
    raw = source(19, (256,))
    for ggml_type, shape in ((True, (256,)), (1, (256,)), (19, (True, 256)), (19, (0, 256)),
                             (19, (255,)), (8, (33,))):
        with pytest.raises(ValueError):converter.convert_affine(raw, ggml_type, shape)
    for bad in (raw[:-1], raw.view(np.int8), np.tile(raw, 2)[::2], [0] * len(raw)):
        with pytest.raises(ValueError):converter.convert_affine(bad, 19, (256,))
    for limit in (True, 0, 255):
        with pytest.raises(ValueError):converter.convert_affine(raw, 19, (256,), max_elements=limit)
    with pytest.raises(ValueError, match="one tensor row"):
        list(converter.iter_affine_chunks(raw, 19, (256,), max_chunk_elements=255))
    with pytest.raises(ValueError, match="rows_per_chunk"):
        list(converter.iter_affine_chunks(raw, 19, (256,), rows_per_chunk=False))
    arrays = list(converter.convert_affine(raw, 19, (256,)))
    arrays[1] = arrays[1].astype(np.float16)
    with pytest.raises(ValueError, match="shape/dtype"):
        converter.dequantize_affine(*arrays, 19, (256,))


def test_validation_rejects_changed_weight_value(converter):
    raw = source(19, (256,))
    arrays = converter.convert_affine(raw, 19, (256,))
    arrays[0][0] ^= np.uint32(1)
    with pytest.raises(ValueError, match="changed finite source weight"):
        converter.validate_affine(raw, 19, (256,), arrays)


@pytest.mark.skipif(not os.environ.get("MLX2_V41_AFFINE_GGUF"), reason="explicit actual-artifact CPU sample path required")
def test_actual_layer2_first_middle_last_blocks(converter, record_property):
    """Explicit model-path probe: bounded header/index plus <=18 source blocks."""
    from mlx2.adapters.deepseek_v41_artifact import FILENAME, inspect_artifact
    from mlx2.runtime.gguf.reader import GGUFFile
    path = Path(os.environ["MLX2_V41_AFFINE_GGUF"]).resolve()
    admitted = inspect_artifact(path.parent)
    assert path.name == FILENAME
    before = path.stat()
    proofs = []
    with GGUFFile(path) as gguf:
        tensors = [gguf.tensors[f"blk.2.ffn_{kind}_exps.weight"] for kind in ("gate", "up", "down")]
        tensors += [t for name, t in sorted(gguf.tensors.items()) if name.startswith("blk.2.") and t.ggml_type == 8][:3]
        assert len(tensors) == 6
        for tensor in tensors:
            elements, block_bytes = (256, 50) if tensor.ggml_type == 19 else (32, 34)
            count = tensor.nbytes // block_bytes
            for index in (0, count // 2, count - 1):
                raw = os.pread(gguf._file.fileno(), block_bytes, gguf.data_offset + tensor.offset + index * block_bytes)
                proofs.append(converter.validate_affine(raw, tensor.ggml_type, (elements,)))
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    assert all(proof["value_exact"] for proof in proofs)
    record_property("actual_source_path", str(path))
    record_property("checkpoint", f"{FILENAME}@{admitted['identity']['fingerprint']}")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    record_property("source_revision", revision)
    for source_file in (Path(converter.__file__), root / "src/mlx2/runtime/gguf/blocks.py",
                        root / "src/mlx2/runtime/gguf/iq1_codebook.py"):
        record_property(f"source_sha256:{source_file.relative_to(root)}",
                        hashlib.sha256(source_file.read_bytes()).hexdigest())
    record_property("source_blocks", len(proofs))
    record_property("signed_zero_differences", sum(proof["signed_zero_differences"] for proof in proofs))
