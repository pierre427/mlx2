"""ArraysCache.make_mask returns None for unpadded, unbounded rows (2026-09-23).

An all-true mask computes the same as no mask, and ``None`` keeps the packed
GDN kernel eligible; a merged B=1 ordinary-route cache used to get an
all-true (1, N) array and fall back to the masked kernel.
"""
import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import ArraysCache


def _cache(left_padding, lengths=None):
    c = ArraysCache(2, left_padding=left_padding)
    if lengths is not None:
        c.lengths = mx.array(lengths)
    return c


@pytest.mark.parametrize("pad", [[0], [0, 0, 0]])
def test_unpadded_rows_get_no_mask(pad):
    assert _cache(pad).make_mask(16) is None


def test_padded_rows_keep_the_mask():
    mask = _cache([0, 3]).make_mask(8)
    assert mask is not None
    assert mask.tolist() == [[True] * 8, [False] * 3 + [True] * 5]


def test_bounded_rows_keep_the_mask():
    mask = _cache([0], lengths=[5]).make_mask(8)
    assert mask.tolist() == [[True] * 5 + [False] * 3]


def test_padding_consumed_by_advance_drops_the_mask():
    c = _cache([0, 3])
    assert c.make_mask(4) is not None
    c.advance(4)  # both rows are past their padding now
    assert c.make_mask(4) is None


def test_no_left_padding_attribute_path_unchanged():
    c = ArraysCache(2)
    assert c.make_mask(8) is None


def test_gdn_layer_output_is_the_same_with_mask_or_none():
    """The semantics the shortcut relies on: all-true mask == no mask."""
    from mlx2.runtime.models.qwen3_5 import GatedDeltaNet, TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=4096,
    )
    mx.random.seed(0)
    layer = GatedDeltaNet(args)
    x = mx.random.normal((1, 12, 64))
    a = layer(x, mx.ones((1, 12), dtype=mx.bool_), ArraysCache(2))
    b = layer(x, None, ArraysCache(2))
    assert mx.allclose(a, b, atol=1e-6).item()
