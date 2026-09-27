"""Real MLX CPU parity for the shared-prefix MLA research callable."""

import mlx.core as mx
import numpy as np

from mlx2.runtime.models.xing_mla_hybrid_mlx import hybrid_shared_prefix_attention_mlx
from mlx2.runtime.models.xing_mla_research import hybrid_shared_prefix_attention


def test_hybrid_mlx_matches_independent_numpy_oracle():
    assert mx.default_device() == mx.cpu
    rng = np.random.default_rng(20260926)
    b, h, l, dn, rank, dr, dv, prefix, suffix = 2, 3, 2, 4, 6, 4, 5, 7, 3

    def normal(shape):
        return rng.normal(size=shape).astype(np.float32)

    arguments = {
        "query_nope": normal((b, h, l, dn)),
        "query_rope": normal((b, h, l, dr)),
        "shared_latent": normal((prefix, rank)),
        "shared_rope": normal((prefix, dr)),
        "suffix_latent": normal((b, suffix, rank)),
        "suffix_rope": normal((b, suffix, dr)),
        "embed_weight": normal((h, rank, dn)),
        "unembed_weight": normal((h, dv, rank)),
        "query_positions": np.tile(np.arange(prefix + suffix - l, prefix + suffix), (b, 1)).astype(np.int32),
        "shared_positions": np.arange(prefix, dtype=np.int32),
        "suffix_positions": np.tile(np.arange(prefix, prefix + suffix), (b, 1)).astype(np.int32),
        "scale": 1 / np.sqrt(dn + dr),
        "tile_size": 3,
    }
    expected = hybrid_shared_prefix_attention(**arguments).output
    mlx_arguments = {key: mx.array(value) if isinstance(value, np.ndarray) else value for key, value in arguments.items()}
    actual = hybrid_shared_prefix_attention_mlx(**mlx_arguments)
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-5, atol=2e-5)
