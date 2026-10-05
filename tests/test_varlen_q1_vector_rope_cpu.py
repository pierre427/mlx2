"""Bitwise Q1 RoPE proof before enabling any native experiment."""

from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
import numpy as np
import pytest

from mlx2.runtime.models.standard_decoder import Model


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("offsets", [(32, 96), (63, 127), (63, 65), (64, 128)])
def test_vector_offset_q1_rope_matches_two_lane_calls_bitwise(dim, offsets):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        q = mx.array(np.linspace(-.9, .9, 2 * 4 * dim,
                                 dtype=np.float16).reshape(2, 4 * dim))
        k = mx.array(np.linspace(.8, -.8, 2 * 2 * dim,
                                 dtype=np.float16).reshape(2, 2 * dim))
        v = mx.array(np.linspace(-.2, .7, 2 * 2 * dim,
                                 dtype=np.float16).reshape(2, 2 * dim))
        real_rope = nn.RoPE(dim, traditional=False, base=1000000.0)
        rope_offsets = []

        def rope(value, *, offset):
            rope_offsets.append(offset)
            return real_rope(value, offset=offset)

        attention = SimpleNamespace(
            n_heads=4, n_kv_heads=2, head_dim=dim,
            q_proj=lambda _: q, k_proj=lambda _: k, v_proj=lambda _: v,
            q_norm=lambda x: x, k_norm=lambda x: x,
            rope=rope)
        model = SimpleNamespace(layers=(SimpleNamespace(
            input_layernorm=lambda x: x, self_attn=attention),))
        hidden = mx.zeros((2, 4 * dim), dtype=mx.float16)
        ordinary = Model.paged_project(model, 0, hidden, (1, 1), offsets)
        assert len(rope_offsets) == 4
        vector = Model.paged_project(
            model, 0, hidden, (1, 1), offsets, vector_q1_rope=True)
        assert len(rope_offsets) == 6 and all(
            type(offset) is mx.array and tuple(offset.shape) == (2,)
            for offset in rope_offsets[-2:])
        mx.eval(*ordinary, *vector)
        assert all(bool(mx.array_equal(left, right).item())
                   for left, right in zip(ordinary, vector))
        assert tuple(array.shape for array in vector) == (
            (2, 4, dim), (2, 2, dim), (2, 2, dim))
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("counts,offsets", [((2, 1), (32, 96)),
                                           ((1, 1), (-1, 96)),
                                           ((1, 1), (32, 2**31))])
def test_vector_offset_q1_rope_refuses_unbounded_or_ragged_rows(counts, offsets):
    model = SimpleNamespace(layers=(SimpleNamespace(
        input_layernorm=lambda x: x,
        self_attn=SimpleNamespace(
            n_heads=1, n_kv_heads=1, head_dim=128,
            q_proj=lambda _: mx.zeros((2, 128), dtype=mx.float16),
            k_proj=lambda _: mx.zeros((2, 128), dtype=mx.float16),
            v_proj=lambda _: mx.zeros((2, 128), dtype=mx.float16),
            q_norm=lambda x: x, k_norm=lambda x: x)),))
    with pytest.raises(ValueError, match="two bounded one-row lanes"):
        Model.paged_project(model, 0, mx.zeros((2, 128), dtype=mx.float16),
                            counts, offsets, vector_q1_rope=True)
