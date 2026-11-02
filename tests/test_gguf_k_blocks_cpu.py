"""Independent same-format raw-byte tests; no model or GPU execution."""

import numpy as np
import pytest

from mlx2.runtime.gguf.blocks import decode_blocks
from mlx2.runtime.gguf.k_blocks import SIZES
from mlx2.runtime.gguf.packed_metal import row_bytes
from mlx2.runtime.gguf.reader import GGUFError


@pytest.mark.parametrize("kind", SIZES)
def test_raw_blocks_match_gguf_reference(kind):
    gguf = pytest.importorskip("gguf")
    from gguf.quants import dequantize

    count, size = SIZES[kind]
    rng = np.random.default_rng(906 + kind)
    data = rng.integers(0, 256, (97, size), dtype=np.uint8)
    offsets = {2: (0,), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,)}
    for offset in offsets[kind]:
        data[:, offset : offset + 2] = (
            rng.uniform(-1, 1, (97, 1)).astype("<f2").view(np.uint8)
        )
    actual = decode_blocks(data.tobytes(), kind)
    reference = dequantize(data, gguf.GGMLQuantizationType(kind)).reshape(-1)
    np.testing.assert_array_equal(actual, reference)
    assert row_bytes(kind, count * 3) == size * 3
    with pytest.raises(GGUFError, match="truncated"):
        decode_blocks(data.tobytes()[:-1], kind)
    with pytest.raises(GGUFError, match="budget"):
        decode_blocks(data.tobytes(), kind, max_elements=count)
    with pytest.raises(GGUFError, match="aligned"):
        row_bytes(kind, count - 1)
