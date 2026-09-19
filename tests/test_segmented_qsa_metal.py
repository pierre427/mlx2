"""Opt-in GPU coverage for the isolated segmented shared-prefix QSA probe."""

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.segmented_qsa_metal import segmented_shared_prefix_attention_metal
from mlx2.runtime.segmented_qsa_reference import materialized_row_attention_reference


pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_GPU_TESTS") != "1",
    reason="set MLX2_RUN_GPU_TESTS=1 while holding the GPU lease",
)


def _case(batch, query, *, base=1024, suffix=31, dim=128, seed=19):
    mx.set_default_device(mx.gpu)
    mx.random.seed(seed + batch * 11 + query)
    hq, hkv = 8, 2
    lengths = tuple(0 if row == 0 else 1 + (row * 13) % suffix for row in range(batch))
    arrays = (
        mx.random.normal((batch, hq, query, dim), dtype=mx.float16),
        mx.random.normal((1, hkv, base, dim), dtype=mx.float16),
        mx.random.normal((1, hkv, base, dim), dtype=mx.float16),
        mx.random.normal((batch, hkv, suffix, dim), dtype=mx.float16),
        mx.random.normal((batch, hkv, suffix, dim), dtype=mx.float16),
    )
    return arrays, lengths


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("query", [1, 4])
def test_metal_probe_matches_materialized_reference(batch, query):
    (q, bk, bv, sk, sv), lengths = _case(batch, query)
    scale = 1.0 / math.sqrt(q.shape[-1])
    actual, engaged, receipt = segmented_shared_prefix_attention_metal(
        q,
        bk,
        bv,
        sk,
        sv,
        mx.array(lengths, dtype=mx.uint32),
        scale=scale,
        splits=32,
    )
    expected, _ = materialized_row_attention_reference(
        q,
        bk,
        bv,
        [sk[row : row + 1, :, :length] for row, length in enumerate(lengths)],
        [sv[row : row + 1, :, :length] for row, length in enumerate(lengths)],
        scale=scale,
    )
    mx.eval(actual, expected, engaged)
    assert int(engaged.item()) == 1
    assert receipt.mechanism == "metal_shared_prefix_partition_v1"
    assert receipt.shared_base_row_read_proxy == 1
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=2.0e-3,
        rtol=2.0e-3,
    )


def test_metal_probe_fails_closed_on_unsupported_batch():
    (q, bk, bv, sk, sv), lengths = _case(3, 1)
    with pytest.raises(ValueError, match="batch must be"):
        segmented_shared_prefix_attention_metal(
            q,
            bk,
            bv,
            sk,
            sv,
            mx.array(lengths, dtype=mx.uint32),
            scale=1.0 / math.sqrt(q.shape[-1]),
        )


def test_indexed_shared_set_matches_materialized_reference():
    (q, bk, bv, sk, sv), lengths = _case(4, 1, base=4096, suffix=31, dim=256)
    indices = mx.arange(0, 4096, 4, dtype=mx.uint32)
    scale = 1.0 / math.sqrt(q.shape[-1])
    actual, engaged, _ = segmented_shared_prefix_attention_metal(
        q,
        bk,
        bv,
        sk,
        sv,
        mx.array(lengths, dtype=mx.uint32),
        scale=scale,
        splits=64,
        base_indices=indices,
    )
    selected_k = mx.take(bk, indices, axis=2)
    selected_v = mx.take(bv, indices, axis=2)
    expected, _ = materialized_row_attention_reference(
        q,
        selected_k,
        selected_v,
        [sk[row : row + 1, :, :length] for row, length in enumerate(lengths)],
        [sv[row : row + 1, :, :length] for row, length in enumerate(lengths)],
        scale=scale,
    )
    mx.eval(actual, expected, engaged)
    assert int(engaged.item()) == 1
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        atol=2.0e-3,
        rtol=2.0e-3,
    )
