"""CPU-only numerical and safety checks for Xing MLA research candidates."""

import numpy as np
import pytest

from mlx2.runtime.models.xing_mla_research import (
    ReuseResearchContract,
    compare_reuse_to_recomputed,
    hybrid_shared_prefix_attention,
    propose_shifted_reuse,
    rotate_rope_delta,
)
from mlx2.runtime.models.xing_mla_reuse_probe import probe_shifted_reuse_on_model


def _case(*, batch=3, prefix=5, suffix=4, queries=2):
    rng = np.random.default_rng(1842)
    heads, nope, rank, rope, value = 3, 4, 6, 4, 5
    return {
        "query_nope": rng.normal(size=(batch, heads, queries, nope)),
        "query_rope": rng.normal(size=(batch, heads, queries, rope)),
        "shared_latent": rng.normal(size=(prefix, rank)),
        "shared_rope": rng.normal(size=(prefix, rope)),
        "suffix_latent": rng.normal(size=(batch, suffix, rank)),
        "suffix_rope": rng.normal(size=(batch, suffix, rope)),
        "embed_weight": rng.normal(size=(heads, rank, nope)),
        "unembed_weight": rng.normal(size=(heads, value, rank)),
        "query_positions": np.tile(np.arange(prefix + suffix - queries, prefix + suffix), (batch, 1)),
        "shared_positions": np.arange(prefix),
        "suffix_positions": np.tile(np.arange(prefix, prefix + suffix), (batch, 1)),
        "scale": 1 / np.sqrt(nope + rope),
    }


def _ordinary_expanded_attention(kwargs):
    q = kwargs["query_nope"]
    qr = kwargs["query_rope"]
    shared = kwargs["shared_latent"]
    suffix = kwargs["suffix_latent"]
    ew = kwargs["embed_weight"]
    uw = kwargs["unembed_weight"]
    batch = q.shape[0]
    all_latent = np.concatenate((np.broadcast_to(shared, (batch, *shared.shape)), suffix), axis=1)
    all_rope = np.concatenate(
        (np.broadcast_to(kwargs["shared_rope"], (batch, *kwargs["shared_rope"].shape)), kwargs["suffix_rope"]),
        axis=1,
    )
    all_positions = np.concatenate(
        (np.broadcast_to(kwargs["shared_positions"], (batch, len(shared))), kwargs["suffix_positions"]),
        axis=1,
    )
    keys = np.einsum("bsr,hrd->bhsd", all_latent, ew)
    values = np.einsum("bsr,hvr->bhsv", all_latent, uw)
    scores = np.einsum("bhld,bhsd->bhls", q, keys)
    scores += np.einsum("bhld,bsd->bhls", qr, all_rope)
    scores *= kwargs["scale"]
    visible = all_positions[:, None, None, :] <= kwargs["query_positions"][:, None, :, None]
    scores = np.where(visible, scores, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        probabilities = np.where(np.isfinite(scores), np.exp(scores - maximum), 0.0)
    denominator = probabilities.sum(axis=-1, keepdims=True)
    probabilities = np.divide(
        probabilities, denominator, out=np.zeros_like(probabilities), where=denominator > 0
    )
    return np.einsum("bhls,bhsv->bhlv", probabilities, values)


@pytest.mark.parametrize("prefix,suffix,queries,tile", [(5, 4, 2, 2), (0, 7, 3, 3), (7, 0, 3, 2), (1, 1, 1, 1)])
def test_hybrid_matches_ordinary_expanded_attention(prefix, suffix, queries, tile):
    kwargs = _case(prefix=prefix, suffix=suffix, queries=queries)
    got = hybrid_shared_prefix_attention(**kwargs, tile_size=tile)
    np.testing.assert_allclose(got.output, _ordinary_expanded_attention(kwargs), atol=1e-12, rtol=1e-12)
    assert got.shared_prefix_tiles == (prefix + tile - 1) // tile
    assert got.private_suffix_tiles == (suffix + tile - 1) // tile


def test_hybrid_causal_visibility_and_all_masked_rows():
    kwargs = _case()
    kwargs["query_positions"] = np.array([[0, 2], [3, 4], [0, 8]])
    kwargs["shared_positions"] = np.arange(5) + 10
    kwargs["suffix_positions"] = np.tile(np.arange(4) + 20, (3, 1))
    got = hybrid_shared_prefix_attention(**kwargs, tile_size=2)
    np.testing.assert_array_equal(got.output, 0)
    np.testing.assert_allclose(got.output, _ordinary_expanded_attention(kwargs))


def test_hybrid_rejects_invalid_geometry_and_tile():
    kwargs = _case()
    with pytest.raises(ValueError, match="geometry"):
        hybrid_shared_prefix_attention(**dict(kwargs, suffix_rope=kwargs["suffix_rope"][:, :, :-1]))
    with pytest.raises(ValueError, match="tile_size"):
        hybrid_shared_prefix_attention(**kwargs, tile_size=0)


def _contract():
    return ReuseResearchContract(
        experiment_id="cpu-oracle", model_revision="checkpoint-abc",
        source_context_hash="prefix-a", target_context_hash="prefix-b", enabled=True,
    )


def test_delta_rotation_matches_direct_rotation():
    rng = np.random.default_rng(3)
    raw = rng.normal(size=(4, 8))
    old = np.array([0, 1, 9, 20])
    new = np.array([5, 8, 12, 3])
    rotated_old = rotate_rope_delta(raw, old, theta=10000.0)
    rotated_new = rotate_rope_delta(raw, new, theta=10000.0)
    np.testing.assert_allclose(
        rotate_rope_delta(rotated_old, new - old, theta=10000.0),
        rotated_new, atol=1e-12,
    )
    denominators = np.array([1.0, 4.0, 12.0, 93.0])
    scaled_old = rotate_rope_delta(raw, old, frequency_denominators=denominators)
    scaled_new = rotate_rope_delta(raw, new, frequency_denominators=denominators)
    np.testing.assert_allclose(
        rotate_rope_delta(scaled_old, new - old, frequency_denominators=denominators),
        scaled_new, atol=1e-12,
    )


def test_shifted_reuse_is_offline_approximate_and_context_error_is_detected():
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(3, 6))
    rope = rng.normal(size=(3, 4))
    args = {
        "source_latent": latent, "source_rotated_rope": rope,
        "source_positions": np.array([2, 3, 4]), "target_positions": np.array([10, 11, 12]),
        "source_token_ids": np.array([7, 8, 9]), "target_token_ids": np.array([7, 8, 9]),
        "theta": 10000.0, "contract": _contract(),
    }
    candidate = propose_shifted_reuse(**args)
    assert candidate.fidelity == "approximate_candidate_only"
    assert not candidate.latent.flags.writeable
    assert compare_reuse_to_recomputed(
        candidate, recomputed_target_latent=latent,
        recomputed_target_rope=candidate.rotated_rope, atol=1e-12,
    ).within_tolerance
    changed = latent.copy()
    changed[1, 2] += 0.1
    comparison = compare_reuse_to_recomputed(
        candidate, recomputed_target_latent=changed,
        recomputed_target_rope=candidate.rotated_rope, atol=1e-3,
    )
    assert not comparison.within_tolerance
    assert comparison.latent_max_abs_error == pytest.approx(0.1)


def test_shifted_reuse_fails_closed_without_contract_or_identical_tokens():
    with pytest.raises(ValueError, match="explicit offline opt-in"):
        ReuseResearchContract("x", "model", "a", "b")
    args = {
        "source_latent": np.zeros((1, 2)), "source_rotated_rope": np.zeros((1, 4)),
        "source_positions": np.array([0]), "target_positions": np.array([1]),
        "source_token_ids": np.array([1]), "target_token_ids": np.array([2]),
        "theta": 10000.0, "contract": _contract(),
    }
    with pytest.raises(ValueError, match="identical token"):
        propose_shifted_reuse(**args)
    with pytest.raises(TypeError, match="contract"):
        propose_shifted_reuse(**dict(args, target_token_ids=np.array([1]), contract=None))


def test_model_probe_enforces_bounds_before_mlx_work():
    args = {
        "source_prefix_ids": np.array([1, 2]), "target_prefix_ids": np.array([1, 3, 4]),
        "shared_chunk_ids": np.arange(33), "continuation_ids": np.array([5]),
        "model_revision": "fixture",
    }
    with pytest.raises(ValueError, match="shared_chunk_ids"):
        probe_shifted_reuse_on_model(object(), **args)
    with pytest.raises(ValueError, match="model_revision"):
        probe_shifted_reuse_on_model(
            object(), **dict(args, shared_chunk_ids=np.array([6]), model_revision="")
        )
