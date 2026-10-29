import struct

import numpy as np
import pytest

from mlx2.runtime.gguf import GGUFError, GGUFFile, decode_blocks


@pytest.fixture(autouse=True)
def cpu_device():
    import mlx.core as mx

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def string(s):
    data = s.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(path, payload, kind=8, shape=(32,), offset=0):
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += string("general.architecture") + struct.pack("<I", 8) + string("test")
    header += (
        string("weight")
        + struct.pack("<I", len(shape))
        + struct.pack("<" + "Q" * len(shape), *shape)
        + struct.pack("<IQ", kind, offset)
    )
    path.write_bytes(header + b"\0" * ((-len(header)) % 32) + payload)
    return path


def test_index_views_keep_packed_bytes(tmp_path):
    payload = struct.pack("<e", 0.5) + bytes(range(32))
    with GGUFFile(fixture(tmp_path / "a.gguf", payload)) as f:
        assert f.metadata["general.architecture"] == "test"
        assert f.tensors["weight"].format == "Q8_0"
        assert f.inventory()["formats"]["Q8_0"]["packed_bytes"] == 34
        v = f.packed_view("weight")
        assert v.readonly and bytes(v) == payload
        v.release()


@pytest.mark.parametrize(
    "kind,shape,offset,payload",
    [
        (8, (33,), 0, b"\0" * 68),
        (99, (32,), 0, b"\0" * 34),
        (8, (32,), 1, b"\0" * 34),
        (8, (32,), 0, b"\0"),
    ],
)
def test_reject_invalid_index(tmp_path, kind, shape, offset, payload):
    with pytest.raises(GGUFError):
        GGUFFile(fixture(tmp_path / "a.gguf", payload, kind, shape, offset))


def test_q8_signed_scale():
    q = np.arange(-16, 16, dtype=np.int8)
    np.testing.assert_array_equal(
        decode_blocks(struct.pack("<e", 0.25) + q.tobytes(), 8),
        q.astype(np.float32) * 0.25,
    )


def test_file_embedding_factory_never_uploads_full_table(tmp_path):
    import mlx.core as mx

    from mlx2.runtime.gguf.weights import GGUFWeights

    rows = np.arange(24, dtype=np.float16).reshape(6, 4)
    path = fixture(tmp_path / "embedding.gguf", rows.tobytes(), kind=1, shape=(4, 6))
    weights = GGUFWeights(path, embedding_policy="file_rows")
    try:
        weights.prepare_embedding("weight")
        reader = weights._embedding_rows["weight"]
        assert not weights._dense and not weights._packed
        ids = np.array([[5, 1, 5], [2, 0, 1]], dtype=np.int32)
        result = weights.embedding(mx.array(ids), name="weight")
        np.testing.assert_array_equal(
            np.array(result).view(np.uint16), rows[ids].view(np.uint16)
        )
        assert not weights._dense and not weights._packed
    finally:
        weights.close()
    assert reader.closed


def test_q2_layout_scales_minima_and_four_bitplanes():
    # Every group's expected quant is controlled independently; row order must
    # follow 16-value sub-blocks rather than contiguous packed 2-bit nibbles.
    scales = bytes((i % 15 + 1) | ((i % 7) << 4) for i in range(16))
    qs = bytearray(64)
    expected = []
    for group in range(16):
        q = group % 4
        for k in range(16):
            qs[32 * (group // 8) + 16 * (group & 1) + k] |= q << (
                ((group // 2) & 3) * 2
            )
        expected.extend(
            [0.5 * (scales[group] & 15) * q - 0.25 * (scales[group] >> 4)] * 16
        )
    data = scales + qs + struct.pack("<ee", 0.5, 0.25)
    np.testing.assert_array_equal(
        decode_blocks(data, 10), np.array(expected, dtype=np.float32)
    )


def test_iq2_grid_sign_and_scale():
    # Grid 0 is all8; grid1 has low byte43. Sign index1=>0x81.
    group = struct.pack("<II", 1, 1 | (3 << 28))
    data = struct.pack("<e", 2.0) + group * 8
    out = decode_blocks(data, 16).reshape(8, 32)
    expected = np.full((8, 32), 14.0, dtype=np.float32)
    expected[:, 0] = -75.25
    expected[:, 7] = -14.0
    np.testing.assert_array_equal(out, expected)


def test_reference_budget_and_truncation():
    with pytest.raises(GGUFError):
        decode_blocks(bytes(66), 16, max_elements=255)
    with pytest.raises(GGUFError):
        decode_blocks(bytes(65), 16)


def test_native_factory_preserves_dtype_and_linear(tmp_path):
    import mlx.core as mx

    from mlx2.runtime.gguf.weights import GGUFWeights

    values = np.arange(64, dtype=np.float16).reshape(2, 32)
    path = fixture(tmp_path / "native.gguf", values.tobytes(), kind=1, shape=(32, 2))
    with GGUFFile(path) as f:
        assert f.tensors["weight"].numpy_shape == (2, 32)
    w = GGUFWeights(path, max_dense_elements=1)
    try:
        assert w.array("weight").dtype == mx.float16
        assert w.linear("weight")(mx.ones((1, 32), mx.float16)).tolist() == [
            [496.0, 1520.0]
        ]
    finally:
        w.close()


def test_native_integer_routing_table(tmp_path):
    import mlx.core as mx

    from mlx2.runtime.gguf.weights import GGUFWeights

    path = fixture(
        tmp_path / "ints.gguf", np.arange(32, dtype="<i4").tobytes(), kind=26
    )
    w = GGUFWeights(path)
    try:
        assert w.array("weight").dtype == mx.int32
        assert w.array("weight").tolist() == list(range(32))
    finally:
        w.close()


def test_grouped_native_explicit_axes(tmp_path):
    import mlx.core as mx

    from mlx2.runtime.gguf.weights import GGUFWeights

    values = np.arange(128, dtype=np.float16).reshape(4, 32)
    w = GGUFWeights(
        fixture(tmp_path / "group.gguf", values.tobytes(), kind=1, shape=(32, 4))
    )
    try:
        actual = w.grouped(mx.ones((1, 2, 32), mx.float16), "weight", groups=2)
        assert actual.shape == (1, 2, 2)
        assert actual.tolist() == [[[496.0, 1520.0], [2544.0, 3568.0]]]
    finally:
        w.close()
