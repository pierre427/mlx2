"""CPU format contracts; checkpoint=n/a (synthetic), source=working revision.

Pinned author-code digest fixture checks byte compatibility independently of
our decode implementation. No model load, MLX import, or GPU dispatch occurs.
"""
from dataclasses import FrozenInstanceError
import hashlib

import numpy as np
import pytest

from mlx2.runtime.microscaling.format import (
    PackedSCWeight,
    SC_FORMAT,
    SCFormat,
    SCFormatError,
    decode_numpy,
    from_mxfp4_numpy,
    metadata_from_mxfp4_scales_numpy,
    pack_numpy,
    to_mxfp4_scales_numpy,
)


def test_pinned_author_codec_bytes_match():
    # mxfp4-int8-m5 71ab38d3ad7e94279368accd0e880df90a63ae1c,
    # e2e/sc_format.py quantize(), NumPy default_rng(20261103).
    rng = np.random.default_rng(20261103)
    w = rng.standard_normal((3, 512)) * .02 * rng.lognormal(0, .7, (1, 512))
    packed = pack_numpy(w, allow_requantization=True)
    expected = {
        "codes": "886777a6dae94ac213898abc76697c38ba10ae0a245730b2dfd92884a8cc524e",
        "bcodes": "c829a4d10def6bf27bd181da0b96dc9844396906830b10411612702a88a4abea",
        "row_scales": "051f26a1dab9adbb359d5f1e2afdc900a99bdfe0121d5ee639e0103534d3e9c7",
        "operand": "40c95ec78bdef57baec4001a88bde12863a44826431fbf42effbd9a7329d00b2",
    }
    arrays = {"codes": packed.codes, "bcodes": packed.bcodes,
              "row_scales": packed.row_scales, "operand": decode_numpy(packed)}
    for name, a in arrays.items():
        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == expected[name]


@pytest.mark.parametrize("shape", [(5, 640), (3, 5, 640)])
def test_native_mxfp4_relayout_preserves_values_and_integer_dot(shape):
    # Deliberately exercise every code and exponent shift; compare directly to
    # the standard FP4 codebook rather than round-tripping through our decoder.
    words = np.resize(np.array([0x76543210, 0xFEDCBA98], np.uint32), (*shape[:-1], shape[-1] // 8))
    scales = np.broadcast_to(np.resize(np.array([123, 124, 125, 126], np.uint8), shape[-1] // 32),
                             (*shape[:-1], shape[-1] // 32)).copy()
    packed = from_mxfp4_numpy(words, scales)
    assert np.array_equal(to_mxfp4_scales_numpy(packed), scales)
    nibble = np.stack([(words >> (4 * i)) & 15 for i in range(8)], -1).reshape(shape)
    codebook = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    reference = np.where(nibble & 8, -1, 1) * codebook[nibble & 7]
    reference *= np.repeat(np.exp2(scales.astype(int) - 127), 32, axis=-1)
    q = decode_numpy(packed)
    assert np.array_equal(q * packed.row_scales[..., None], reference)
    assert np.abs(q.astype(np.int16)).max() <= 96
    x = np.resize(np.array([-127, -1, 0, 1, 127], np.int64), shape[-1])
    exact_dot = np.sum(x * q.astype(np.int64), axis=-1)
    assert np.array_equal(exact_dot * packed.row_scales, np.sum(x * reference, axis=-1))


def test_zero_blocks_can_normalize_without_requantizing_nonzero_weights():
    words = np.zeros((2, 16), np.uint32)
    words[0, :4] = 0x77777777
    scales = np.array([[124, 0, 254, 63], [0, 254, 127, 253]], np.uint8)
    packed = from_mxfp4_numpy(words, scales)
    assert np.array_equal(packed.codes, words)
    q = decode_numpy(packed)
    assert np.all(q[0, :32] * packed.row_scales[0] == .75)
    assert np.count_nonzero(q[:, 32:]) == 0
    assert np.count_nonzero(q[1]) == 0
    assert packed.stats["zero_blocks_normalized"] == 6
    with pytest.raises(SCFormatError, match="spread"):
        metadata_from_mxfp4_scales_numpy(scales)


@pytest.mark.parametrize("shape", [(4, 512), (2, 4, 512)])
def test_offline_conversion_handles_dead_blocks_outliers_and_zero_rows(shape):
    rng = np.random.default_rng(12)
    w = rng.standard_normal(shape) * .02
    w[..., :32] *= 100
    w[..., 32:64] = 0
    w.reshape(-1, shape[-1])[0] = 0
    packed = pack_numpy(w, allow_requantization=True)
    standard_scales = to_mxfp4_scales_numpy(packed)
    strict_bc, strict_ws = metadata_from_mxfp4_scales_numpy(standard_scales)
    assert np.array_equal(strict_bc, packed.bcodes)
    assert np.array_equal(strict_ws, packed.row_scales)
    again = from_mxfp4_numpy(packed.codes, standard_scales)
    assert np.array_equal(decode_numpy(again), decode_numpy(packed))
    assert np.all(decode_numpy(packed).reshape(-1, shape[-1])[0] == 0)
    assert np.max(np.ptp(standard_scales.astype(int), axis=-1)) <= 3
    assert packed.stats["constrained_rows"] > 0


def test_load_rejects_spread_and_nan_without_changing_source():
    words = np.full((1, 16), 0x77777777, np.uint32)
    scales = np.array([[127, 127, 127, 131]], np.uint8)
    original_words, original_scales = words.copy(), scales.copy()
    with pytest.raises(SCFormatError, match="spread"):
        from_mxfp4_numpy(words, scales)
    assert np.array_equal(words, original_words) and np.array_equal(scales, original_scales)
    scales[0, 0] = 255
    with pytest.raises(SCFormatError, match="nonfinite"):
        from_mxfp4_numpy(words, scales)
    with pytest.raises(SCFormatError, match="nonfinite"):
        metadata_from_mxfp4_scales_numpy(scales)


def test_weight_payload_does_not_alias_mutable_inputs():
    words = np.full((1, 16), 0x12345678, np.uint32)
    scales = np.full((1, 4), 127, np.uint8)
    packed = from_mxfp4_numpy(words, scales)
    expected = packed.codes.copy()
    words[:] = 0
    scales[:] = 0
    assert np.array_equal(packed.codes, expected)
    with pytest.raises(ValueError):
        packed.codes[0, 0] = 0
    with pytest.raises(TypeError):
        packed.stats["conversion"] = "requantized"
    with pytest.raises(FrozenInstanceError):
        packed.shape = (2, 128)
    assert packed.codes.flags.c_contiguous and packed.bcodes.flags.c_contiguous


@pytest.mark.parametrize("bad", [np.zeros((1, 128), np.int8), np.full((1, 128), np.nan), np.full((1, 128), np.inf)])
def test_offline_requantization_is_explicit_and_finite(bad):
    with pytest.raises(SCFormatError, match="allow_requantization"):
        pack_numpy(bad)
    with pytest.raises(SCFormatError):
        pack_numpy(bad, allow_requantization=True)


def test_shape_dtype_scale_and_accumulator_contracts():
    with pytest.raises(SCFormatError, match="multiple"):
        pack_numpy(np.zeros((1, 32)), allow_requantization=True)
    with pytest.raises(SCFormatError, match="INT32"):
        metadata_from_mxfp4_scales_numpy(np.zeros((1, 5508), np.uint8))
    with pytest.raises(SCFormatError, match="uint32"):
        from_mxfp4_numpy(np.zeros((1, 16), np.uint64), np.zeros((1, 4), np.uint8))
    with pytest.raises(SCFormatError, match="shape"):
        from_mxfp4_numpy(np.zeros((1, 16), np.uint32), np.zeros((1, 8), np.uint8))
    with pytest.raises(SCFormatError, match="powers"):
        PackedSCWeight(np.zeros((1, 16), np.uint32), np.zeros((1, 1), np.uint8), np.array([.3], np.float32), (1, 128))
    with pytest.raises(SCFormatError, match="finite E8M0"):
        PackedSCWeight(np.zeros((1, 16), np.uint32), np.full((1, 1), 255, np.uint8), np.array([2.**126], np.float32), (1, 128))
    with pytest.raises(SCFormatError, match="only"):
        SCFormat(block=16)
    assert SC_FORMAT.max_reduction == 176138


def test_finite_source_cannot_publish_overflowing_quantized_weights():
    # Round-to-nearest can move a finite F32 maximum onto 4 * 2**126,
    # which is above the largest finite F32. Reject the resulting artifact.
    with pytest.raises(SCFormatError, match='finite float32'):
        pack_numpy(np.full((1, 128), np.finfo(np.float32).max), allow_requantization=True)
    with pytest.raises(SCFormatError, match='finite float32'):
        from_mxfp4_numpy(np.full((1, 16), 0x77777777, np.uint32),
                          np.full((1, 4), 254, np.uint8))
    # The metadata-only runtime check conservatively rejects high exponents
    # without reading the weight payload; ordinary handling can remain.
    with pytest.raises(SCFormatError, match='finite float32'):
        metadata_from_mxfp4_scales_numpy(np.full((1, 4), 253, np.uint8))
    safe = pack_numpy(np.full((1, 128), 2.**125), allow_requantization=True)
    assert np.isfinite(decode_numpy(safe).astype(np.float32) * safe.row_scales[:, None]).all()
