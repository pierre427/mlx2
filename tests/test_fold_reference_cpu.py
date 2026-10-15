"""Source-equation gates for the default-off FOLD component probe."""

import numpy as np
import pytest

from mlx2.runtime.drafters.fold_reference import (
    PROBE_CONTEXTS,
    FOLDPolicy,
    _rotate_neox,
    _softmax,
    fold_block,
    side_cache_bytes,
)


def tensors(context, block=3, heads=2, dim=8):
    rng = np.random.default_rng(context + block)
    return [
        rng.normal(size=shape)
        for shape in (
            (block, heads, dim),
            (context, heads, dim),
            (context, heads, dim),
            (block, heads, dim),
            (block, heads, dim),
        )
    ]


def full_attention(q, kc, vc, kb, vb, inv_freq, scale):
    block, context = q.shape[0], kc.shape[0]
    keys = np.concatenate((kc, kb))
    values = np.concatenate((vc, vb))
    qr = _rotate_neox(q, np.arange(context, context + block), inv_freq)
    kr = _rotate_neox(keys, np.arange(context + block), inv_freq)
    return np.stack(
        [
            np.einsum(
                "hn,nhd->hd",
                _softmax(np.einsum("hd,nhd->hn", qr[row], kr) * scale),
                values,
            )
            for row in range(block)
        ]
    )


def test_policy_and_memory_admission_are_explicit():
    assert FOLDPolicy.from_value(None) is None
    assert FOLDPolicy.from_value({"window": 2048}).as_dict() == {
        "window": 2048,
        "alpha": 2.0,
        "sliding_layers": "window",
    }
    with pytest.raises(ValueError):
        FOLDPolicy(window=0)
    with pytest.raises(ValueError):
        FOLDPolicy(window=1, alpha=0.9)
    policy = FOLDPolicy(window=2048)
    assert side_cache_bytes(
        policy, max_context=131072, layers=4, kv_heads=8, head_dim=128
    ) == 2_147_483_648
    assert PROBE_CONTEXTS == (32768, 65536, 131072)


def test_fold_reduces_exactly_to_full_attention_before_far_memory_exists():
    q, kc, vc, kb, vb = tensors(context=4)
    inv = np.asarray([1.0, 0.1, 0.01, 0.001])
    policy = FOLDPolicy(window=4, alpha=2.0)
    got = fold_block(q, kc, vc, kb, vb, policy=policy, inv_freq=inv, scale=0.5)
    expected = full_attention(q, kc, vc, kb, vb, inv, 0.5)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_far_memory_changes_only_after_the_window_boundary():
    q, kc, vc, kb, vb = tensors(context=12)
    inv = np.asarray([1.0, 0.1, 0.01, 0.001])
    fold = fold_block(
        q,
        kc,
        vc,
        kb,
        vb,
        policy=FOLDPolicy(window=4, alpha=2.0),
        inv_freq=inv,
        scale=0.5,
    )
    full = full_attention(q, kc, vc, kb, vb, inv, 0.5)
    assert np.isfinite(fold).all()
    assert not np.allclose(fold, full)
