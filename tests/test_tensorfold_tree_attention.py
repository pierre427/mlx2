"""Masked tree SDPA and batched row RoPE against the per-node reference law.

The CPU backend picks shape-dependent kernels (SDPA at short key lengths,
RMSNorm across row counts), so these CPU checks bound the difference at
rounding level; GPU bit parity is measured on the target, not here.
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.tensorfold_qwen38 import lane_multi, stream_attention


def _kv(prefix, width, heads=2, dim=256, seed=0):
    mx.random.seed(seed)
    keys = mx.random.normal((1, heads, prefix + width + 3, dim)).astype(mx.bfloat16)
    values = mx.random.normal((1, heads, prefix + width + 3, dim)).astype(mx.bfloat16)
    return keys, values


@pytest.mark.parametrize("parents", [
    [-1],
    [-1, 0, 1, 2, 3],                       # chain: causal mask
    [-1, 0, 1, 0, 3, 4, 0, 6],              # three branches from the anchor
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14],   # four_path15-shaped
])
@pytest.mark.parametrize("prefix", [0, 37])
def test_masked_tree_sdpa_matches_ordinary_per_node_law(parents, prefix):
    with mx.stream(mx.cpu):
        heads, kv_heads = 8, 2
        plan = stream_attention.Plan([parents], [prefix], heads, kv_heads)
        keys, values = _kv(prefix, len(parents), kv_heads)
        queries = mx.random.normal((1, heads, len(parents), 256)).astype(mx.bfloat16)
        reference = stream_attention.ordinary_tree_sdpa(queries, [(keys, values)], 0.0625, plan)
        masked = stream_attention.masked_tree_sdpa(queries, [(keys, values)], 0.0625, plan)
        assert masked.shape == reference.shape
        assert mx.allclose(reference, masked, atol=2e-2, rtol=0).item()


def test_masked_tree_sdpa_keeps_streams_independent():
    with mx.stream(mx.cpu):
        heads, kv_heads = 4, 2
        parents = [[-1, 0, 0, 1], [-1, 0, 1]]
        starts = [5, 11]
        plan = stream_attention.Plan(parents, starts, heads, kv_heads)
        kv = [_kv(5, 4, kv_heads, seed=1), _kv(11, 3, kv_heads, seed=2)]
        queries = mx.random.normal((1, heads, 7, 256)).astype(mx.bfloat16)
        reference = stream_attention.ordinary_tree_sdpa(queries, kv, 0.0625, plan)
        masked = stream_attention.masked_tree_sdpa(queries, kv, 0.0625, plan)
        assert mx.allclose(reference, masked, atol=2e-2, rtol=0).item()
        # A wrong ancestor set is a large error, not rounding.
        wrong = stream_attention.Plan([[-1, 0, 1, 2], [-1, 0, 1]], starts, heads, kv_heads)
        assert not mx.allclose(
            reference, stream_attention.masked_tree_sdpa(queries, kv, 0.0625, wrong), atol=2e-2, rtol=0
        ).item()


class _Attn:
    num_key_value_heads = 2

    def __init__(self):
        self.q_norm = nn.RMSNorm(64)
        self.k_norm = nn.RMSNorm(64)
        self.rope = nn.RoPE(64, base=1_000_000)


def test_batched_row_rope_matches_per_row_calls(monkeypatch):
    with mx.stream(mx.cpu):
        attn = _Attn()
        L = 6
        queries = mx.random.normal((1, L, 4, 64))
        keys = mx.random.normal((1, L, 2 * 64))
        values = mx.random.normal((1, L, 2 * 64))
        positions = [40, 41, 42, 41, 42, 43]
        monkeypatch.setattr(lane_multi, "_BATCHED_ROPE", [True])
        batched = lane_multi._ordinary_qkv_rows(attn, queries, keys, values, positions)
        monkeypatch.setattr(lane_multi, "_BATCHED_ROPE", [False])
        rows = lane_multi._ordinary_qkv_rows(attn, queries, keys, values, positions)
        for left, right in zip(batched, rows):
            assert left.shape == right.shape
            assert mx.allclose(left, right, atol=1e-5, rtol=0).item()
