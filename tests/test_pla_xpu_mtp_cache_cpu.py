"""Packed-cache rollback invariants mined from XPU #21, in mlx2's ABI."""
import mlx.core as mx
import numpy as np
import pytest
from test_short_gdn_rollback import _layer

from mlx2.runtime.models import qwen3_5
from mlx2.runtime.models.cache import ArraysCache


@pytest.mark.parametrize("device_accept", [False, True])
def test_repeated_mixed_acceptance_preserves_exact_conv_tail(monkeypatch, device_accept):
    monkeypatch.setattr(qwen3_5, "_GDN_ARRAY_ACCEPT", device_accept)
    mx.random.seed(21)
    layer = _layer(); layer.eval()
    cache = ArraysCache(2)
    prefix = mx.random.normal((3, 4, 32))
    layer(prefix, cache=cache)
    raw_prefix = layer.in_proj_qkv(prefix)
    raw = [raw_prefix[i:i+1] for i in range(3)]
    replay = [prefix[i:i+1] for i in range(3)]
    cache.start_speculation()
    for width, keep in ((5, (0, 2, 5)), (2, (2, 0, 1)), (4, (1, 4, 0))):
        inputs = mx.random.normal((3, width, 32))
        projected = layer.in_proj_qkv(inputs)
        layer(inputs, cache=cache)
        cache.trim_ragged([width - k for k in keep])
        for row, accepted in enumerate(keep):
            raw[row] = mx.concatenate([raw[row], projected[row:row+1, :accepted]], axis=1)
            replay[row] = mx.concatenate([replay[row], inputs[row:row+1, :accepted]], axis=1)
            np.testing.assert_array_equal(np.array(cache[0][row:row+1]), np.array(raw[row][:, -3:]))
            oracle = ArraysCache(2)
            layer(replay[row], cache=oracle)
            np.testing.assert_allclose(np.array(cache[1][row:row+1]), np.array(oracle[1]), rtol=2e-5, atol=2e-6)
