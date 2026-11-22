"""Inkling 2 W4A4 fake activation quantization on Metal (GPU; run under the GPU lock).

The CPU tests pin the quantizers bitwise to an independent NumPy reference.
Here the same quantizers run on the GPU: Metal division and the four-over-six
block-error sums may round differently, and their reduction order could
depend on the row count.  Bitwise against the NumPy reference is expected;
any mismatch is reported with the blocks involved.  Row-count invariance
(one row alone equals the same row inside a 512-row batch) must hold
exactly, or cold/warm and B1/BN runs would drift for a reason that is not
the model.
"""

import numpy as np
import mlx.core as mx
import pytest

from test_inkling2_act_quant_cpu import _edge_rows, _random, ref_mxfp8, ref_nvfp4

from mlx2.runtime.models.inkling2 import act_quant as aq

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def _inputs():
    return np.concatenate([_random((64, 2304), 0), _edge_rows().repeat(36, axis=1)])


def test_nvfp4_gpu_matches_numpy_reference():
    x = _inputs()
    got = np.array(aq.fake_quant_nvfp4_act(mx.array(x)))
    want = ref_nvfp4(x)[0]
    bad = np.argwhere((got != want).reshape(*got.shape[:-1], -1, 16).any(-1))
    assert bad.size == 0, f"{len(bad)} blocks differ, first {bad[:5].tolist()}"


def test_mxfp8_gpu_matches_numpy_reference():
    x = _inputs()
    assert np.array_equal(np.array(aq.fake_quant_mxfp8_act(mx.array(x))), ref_mxfp8(x))


@pytest.mark.parametrize("fn", [aq.fake_quant_nvfp4_act, aq.fake_quant_mxfp8_act])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("width", [2304, 1152])
def test_row_count_invariant(fn, dtype, width):
    x = mx.array(_random((512, width), 5)).astype(dtype)
    batch = fn(x)
    for row in (0, 1, 255, 511):
        assert mx.array_equal(fn(x[row:row + 1]), batch[row:row + 1])
    # The sorted MoE layout: [N, 1, width] rows.
    assert mx.array_equal(fn(x[:, None, :]), batch[:, None, :])
