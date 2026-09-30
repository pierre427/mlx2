"""The QSA indexer rotates by the same rope as attention (FreeToken #573)."""
import mlx.core as mx
import pytest

from mlx2.runtime.models.qwen4_exp import QSAIndexer, _apply_rope_positions
from mlx2.runtime.models.rope_utils import initialize_rope
from qsa_oracle import tiny_args

YARN = {
    "rope_type": "yarn",
    "rope_theta": 10000,
    "partial_rotary_factor": 0.5,
    "factor": 4,
    "original_max_position_embeddings": 16,
}


def _attention_and_indexer(args, offset=5, length=40):
    """(attention rope, indexer rope) of one random (B, H, L, D) tensor."""
    indexer = QSAIndexer(args)
    dims = indexer.rotary_dim
    rope = initialize_rope(
        dims, base=args.rope_theta, traditional=False,
        scaling_config=args.rope_scaling,
        max_position_embeddings=args.max_position_embeddings,
    )
    x = mx.random.normal((1, 2, length, 8), key=mx.random.key(0))
    attention = rope(x, offset=offset)
    positions = mx.arange(offset, offset + length)[None, :, None]
    indexed = _apply_rope_positions(
        x.transpose(0, 2, 1, 3), positions, dims, indexer.rope_theta, indexer.rope_scaling
    ).transpose(0, 2, 1, 3)
    return indexer, x, attention, indexed


def test_plain_rope_keeps_the_exact_unscaled_table():
    indexer, _, attention, indexed = _attention_and_indexer(tiny_args())
    assert indexer.rope_scaling is None
    assert mx.allclose(attention, indexed, atol=1e-5).item()


def test_yarn_indexer_matches_attention_rotation():
    args = tiny_args(rope_parameters=dict(YARN))
    indexer, x, attention, indexed = _attention_and_indexer(args)
    assert indexer.rope_scaling is not None
    assert mx.allclose(attention, indexed, atol=1e-5).item()
    # Before, the indexer ignored the scaling: its rotation differed.
    unscaled = _apply_rope_positions(
        x.transpose(0, 2, 1, 3), mx.arange(5, 45)[None, :, None],
        indexer.rotary_dim, indexer.rope_theta,
    ).transpose(0, 2, 1, 3)
    assert not mx.allclose(attention, unscaled, atol=1e-3).item()


def test_linear_scaling_matches_attention_rotation():
    args = tiny_args(rope_parameters={
        "rope_type": "linear", "rope_theta": 10000, "partial_rotary_factor": 0.5,
        "factor": 2.0,
    })
    _, _, attention, indexed = _attention_and_indexer(args)
    assert mx.allclose(attention, indexed, atol=1e-5).item()


def test_unsupported_scaling_fails_at_construction():
    args = tiny_args(rope_parameters={
        "rope_type": "dynamic", "rope_theta": 10000, "partial_rotary_factor": 0.5,
        "factor": 2.0,
    })
    with pytest.raises(ValueError, match="QSA indexer does not support"):
        QSAIndexer(args)
