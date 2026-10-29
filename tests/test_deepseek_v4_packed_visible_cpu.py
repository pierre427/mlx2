"""Direct packed attention codec/visible-set/shape/offline compiler contracts."""

import shutil
import subprocess
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v4_packed_visible_attention as candidate
from mlx2.runtime.models.deepseek_v4_packed_kv import (
    PackedKV,
    pack_fp8_kv,
    unpack_fp8_kv,
)
from mlx2.runtime.models.deepseek_v4_visible_attention import (
    reference_visible_attention,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def fixture(ratio=4, offset=13, length=3, topk=3):
    rng = np.random.default_rng(77 + ratio)
    q = rng.normal(0, 0.2, (1, 2, length, 512)).astype(np.float32)
    count = 0 if ratio == 0 else (offset + length) // ratio
    local = rng.normal(0, 0.3, (1, 1, min(offset, 3) + length, 512)).astype(np.float32)
    pool = (
        None
        if not count
        else pack_fp8_kv(rng.normal(0, 0.3, (1, count, 512)).astype(np.float32), xp=np)
    )
    indices = (
        None
        if topk == 0 or not count
        else np.stack(
            [rng.choice(count, min(topk, count), replace=False) for _ in range(length)]
        )[None].astype(np.uint32)
    )
    return q, local, pool, indices, np.array([0.2, -0.7], np.float32)


@pytest.mark.parametrize(
    "ratio,offset,topk",
    [(0, 9, 0), (4, 13, 0), (4, 13, 3), (128, 509, 0), (4, 4093, 512)],
)
def test_packed_visible_exact_reference_rows_and_values(ratio, offset, topk):
    q, local, packed, indices, sinks = fixture(ratio, offset, topk=topk)
    out = candidate.reference_packed_visible_attention(
        q, local, packed, indices, sinks, offset=offset, window=4, ratio=ratio
    )
    unpacked = None if packed is None else unpack_fp8_kv(packed, xp=np)
    expected = reference_visible_attention(
        q, local, unpacked, indices, sinks, offset=offset, window=4, ratio=ratio
    )
    np.testing.assert_array_equal(out, expected)


def test_all_code_values_match_exact_e4_decode_and_rope_tail():
    codes = np.resize(np.arange(256, dtype=np.uint8), (1, 2, 448))
    scales = np.array([[[2.0 ** (g - 8) for g in range(7)]]] * 2, np.float32).reshape(
        1, 2, 7
    )
    rope = np.arange(128, dtype=np.float32).reshape(1, 2, 64) / 7
    actual = unpack_fp8_kv(PackedKV(codes, scales, rope), xp=np)
    expected = np.empty_like(actual)
    for row in range(2):
        for column in range(448):
            code = int(codes[0, row, column])
            exponent = (code >> 3) & 15
            fraction = code & 7
            value = np.float32(
                fraction / 512
                if exponent == 0
                else np.ldexp(1 + fraction / 8, exponent - 7)
            )
            expected[0, row, column] = (
                np.float32(-value if code & 128 else value)
                * scales[0, row, column // 64]
            )
    expected[..., 448:] = rope
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("bad", [-1, 4, "duplicate"])
def test_invalid_indices_fail_closed_cpu_reference(bad):
    q, local, packed, indices, sinks = fixture()
    indices = indices.astype(np.int32)
    indices[0, 0, 0] = indices[0, 0, 1] if bad == "duplicate" else bad
    with pytest.raises(ValueError, match="index|indices"):
        candidate.reference_packed_visible_attention(
            q, local, packed, indices, sinks, offset=13, window=4, ratio=4
        )


@pytest.mark.parametrize("change", ["layout", "scales", "rope", "count", "offset"])
def test_metadata_shape_contract_refuses_malformed_pool(change):
    q, local, packed, indices, sinks = fixture()
    if change == "layout":
        packed = replace(packed, layout="other")
    if change == "scales":
        packed = replace(packed, scales=packed.scales[:, :, :6])
    if change == "rope":
        packed = replace(packed, rope=packed.rope[:, :, :63])
    if change == "count":
        packed = PackedKV(packed.codes[:, :3], packed.scales[:, :3], packed.rope[:, :3])
    offset = 2**31 if change == "offset" else 13
    with pytest.raises(ValueError):
        candidate.validate_packed_geometry(
            q, local, packed, indices, sinks, offset, 4, 4
        )


def test_dispatch_never_unpacks_pool_and_preserves_compact_planes(monkeypatch):
    data = fixture()
    q, local, packed, indices, sinks = data
    packed = PackedKV(
        *(mx.array(x) for x in (packed.codes, packed.scales, packed.rope))
    )
    seen = {}

    def no_unpack(*a, **kw):
        raise AssertionError("whole pool unpacked")

    monkeypatch.setattr(candidate, "unpack_fp8_kv", no_unpack)
    monkeypatch.setattr(candidate, "self_check", lambda: None)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)

    def kernel():
        def call(**kw):
            seen.update(kw)
            return [mx.zeros(kw["output_shapes"][0])]

        return call

    monkeypatch.setattr(candidate, "_kernel", kernel)
    out = candidate.packed_visible_attention(
        mx.array(q),
        mx.array(local),
        packed,
        mx.array(indices),
        mx.array(sinks),
        offset=13,
        window=4,
        ratio=4,
    )
    assert out.shape == (1, 2, 3, 512)
    assert seen["inputs"][2].shape == (4, 448) and seen["inputs"][2].dtype == mx.uint8
    assert seen["inputs"][3].shape == (4, 7) and seen["inputs"][4].shape == (4, 64)
    assert candidate.workspace_bytes(heads=64, length=8, topk=512) < 2 << 20


def test_cpu_entry_refuses_gpu_and_empty_local_contract():
    q, local, packed, indices, sinks = fixture()
    packed = PackedKV(
        *(mx.array(x) for x in (packed.codes, packed.scales, packed.rope))
    )
    with pytest.raises(RuntimeError, match="owned Metal"):
        candidate.packed_visible_attention(
            mx.array(q),
            mx.array(local),
            packed,
            mx.array(indices),
            mx.array(sinks),
            offset=13,
            window=4,
            ratio=4,
        )


@pytest.mark.parametrize(
    "constant_inputs",
    [
        (),
        ("scales",),
        ("rope",),
        ("codes",),
        ("scales", "rope"),
        ("codes", "scales"),
        ("codes", "rope"),
        ("codes", "scales", "rope"),
    ],
)
def test_metal_source_compiles_with_offline_cpu_toolchain(tmp_path, constant_inputs):
    if (
        not shutil.which("xcrun")
        or subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Metal compiler unavailable")
    path = tmp_path / "packed_attention.metal"
    path.write_text(candidate.metal_compile_source(constant_inputs=constant_inputs))
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal3.2",
            "-c",
            str(path),
            "-o",
            str(tmp_path / "packed_attention.air"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
