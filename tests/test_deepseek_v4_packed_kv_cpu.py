"""Synthetic CPU checks, no checkpoint execution or GPU dispatch."""
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v4_packed_kv import pack_fp8_kv, unpack_fp8_kv


def test_fp8_all_finite_codes_and_physical_storage():
    codes = np.arange(127, dtype=np.uint32)
    exponent, fraction = (codes >> 3) & 15, codes & 7
    lut = np.where(exponent == 0, fraction / 512,
                   (1 + fraction / 8) * 2. ** (exponent.astype(float) - 7)).astype(np.float32)
    x = np.zeros((2, 512), np.float32)
    x[:, :127] = lut
    x[1, :127] *= -1
    x[:, 127:448] = 448  # Force scale=1 in every group.
    for start in range(0, 448, 64):
        x[:, start + 63] = 448
    x[:, 448:] = np.arange(64, dtype=np.float32) / 13
    packed = pack_fp8_kv(x, xp=np)
    np.testing.assert_array_equal(unpack_fp8_kv(packed, xp=np), x)
    assert packed.nbytes == 2 * (448 + 7 * 4 + 64 * 4)
    assert packed.nbytes < x.nbytes / 2
    assert packed.rope.flags.owndata
    assert not np.shares_memory(packed.rope, x)


def test_cpu_mlx_roundtrip_matches_forward_qat():
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import simulate_fp8_kv
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(72)
        x = rng.normal(size=(1, 19, 512)).astype(np.float32)
        x *= np.geomspace(1e-6, 1e4, 19).astype(np.float32)[None, :, None]
        x[:, 0, :448] = 0
        a = mx.array(x)
        packed = pack_fp8_kv(a, xp=mx)
        restored = unpack_fp8_kv(packed, xp=mx)
        np.testing.assert_array_equal(np.array(restored), np.array(simulate_fp8_kv(a)))
        cpu = unpack_fp8_kv(pack_fp8_kv(x, xp=np), xp=np)
        np.testing.assert_array_equal(cpu, np.array(restored))
    finally:
        mx.set_default_device(old)


def test_reject_wrong_dtype_or_dimension():
    for value in (np.ones((2, 512), np.float16), np.ones((2, 128), np.float32)):
        with pytest.raises(ValueError):
            pack_fp8_kv(value, xp=np)

def test_all_codes_and_even_midpoint_ties_against_independent_lut():
    code = np.arange(127)
    exp, frac = code // 8, code % 8
    lut = np.where(exp == 0, frac / 512,
                   (1 + frac / 8) * 2. ** (exp - 7)).astype(np.float32)
    samples = np.concatenate((lut, (lut[:-1] + lut[1:]) / 2))
    x = np.full((len(samples), 512), 448, dtype=np.float32)
    x[:, 0] = samples
    x[:, 1] = -samples
    packed = pack_fp8_kv(x, xp=np)
    expected = []
    for s in samples:
        delta = abs(lut - s)
        nearest = np.flatnonzero(delta == delta.min())
        expected.append(next((int(i) for i in nearest if i % 2 == 0), int(nearest[0])))
    np.testing.assert_array_equal(packed.codes[:, 0], expected)
    np.testing.assert_array_equal(packed.codes[:, 1], np.array(expected) | 128)
