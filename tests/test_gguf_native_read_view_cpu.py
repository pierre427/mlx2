"""APFS read-view payload oracle with an independent safetensors reader."""

import hashlib
import struct

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from mlx2.runtime.gguf.native_read_view import prepare_read_view, validate_read_view
from mlx2.runtime.gguf.reader import GGUFError
from mlx2.runtime.gguf.weights import GGUFWeights


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def test_native_view_preserves_payload_and_refuses_mutation(tmp_path):
    source = tmp_path / "source.gguf"
    dest = tmp_path / "packed.safetensors"
    tensors = [
        ("k3", (256, 2), 11, bytes(range(220))),
        ("native", (3, 2), 0, np.arange(6, dtype="<f4").tobytes()),
        ("half", (3, 2), 1, np.arange(6, dtype="<f2").tobytes()),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    header += string("padding") + struct.pack("<I", 8) + string(" " * 8192)
    body = bytearray()
    for name, shape, kind, data in tensors:
        body.extend(b"\0" * (-len(body) % 32))
        offset = len(body)
        body.extend(data)
        header += string(name) + struct.pack("<IQQIQ", 2, *shape, kind, offset)
    source.write_bytes(header + b"\0" * (-len(header) % 32) + body)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        prepare_read_view(source, dest, source_sha256=digest)
    except OSError as error:
        pytest.skip(f"APFS clone unavailable: {error}")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with safe_open(dest, framework="numpy") as reader:
        for name, _shape, _kind, data in tensors:
            assert reader.get_tensor(name).tobytes() == data
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        weights = GGUFWeights(source, native_read_view=dest, source_sha256=digest)
        try:
            np.testing.assert_array_equal(
                np.array(weights.packed("k3").data),
                np.frombuffer(tensors[0][3], dtype=np.uint8),
            )
            for name in ("native", "half"):
                assert weights.array(name).shape == (2, 3)
                np.testing.assert_array_equal(
                    np.array(weights.array(name)), np.arange(6).reshape(2, 3)
                )
        finally:
            weights.close()
    finally:
        mx.set_default_device(previous)
    with dest.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(GGUFError, match="changed"):
        validate_read_view(source, dest, source_sha256=digest)
