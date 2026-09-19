"""CPU-only oracle coverage for the experimental shared-prefix QSA math."""

import math

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.segmented_qsa_reference import (  # noqa: E402
    materialized_row_attention_reference,
    shared_prefix_segmented_attention,
)


def _arrays(*, batch=4, query=3, base=11, suffixes=(0, 1, 5, 9), seed=73):
    mx.random.seed(seed)
    hq, hkv, dim = 8, 2, 8
    q = mx.random.normal((batch, hq, query, dim), dtype=mx.float32)
    bk = mx.random.normal((1, hkv, base, dim), dtype=mx.float32)
    bv = mx.random.normal((1, hkv, base, dim), dtype=mx.float32)
    sk = [mx.random.normal((1, hkv, length, dim)) for length in suffixes]
    sv = [mx.random.normal((1, hkv, length, dim)) for length in suffixes]
    return q, bk, bv, sk, sv


def _run(arrays, **kwargs):
    q, bk, bv, sk, sv = arrays
    reference, reference_receipt = materialized_row_attention_reference(
        q, bk, bv, sk, sv, scale=1.0 / math.sqrt(q.shape[-1]), **kwargs
    )
    actual, actual_receipt = shared_prefix_segmented_attention(
        q, bk, bv, sk, sv, scale=1.0 / math.sqrt(q.shape[-1]), **kwargs
    )
    mx.eval(reference, actual)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(reference), atol=2.0e-6, rtol=2.0e-5
    )
    return actual, actual_receipt, reference_receipt


@pytest.mark.parametrize("batch,suffixes", [(1, (0,)), (2, (0, 7)), (4, (9, 1, 0, 4))])
def test_ragged_and_empty_suffix_matches_materialized_oracle(batch, suffixes):
    _, receipt, reference = _run(_arrays(batch=batch, suffixes=suffixes))
    assert receipt.shared_base_calls == 1
    assert receipt.private_segment_calls == sum(length > 0 for length in suffixes)
    assert receipt.materialized_row_calls == 0
    assert receipt.base_kv_row_read_proxy == 1
    assert reference.materialized_row_calls == batch
    assert reference.base_kv_row_read_proxy == batch


def test_gqa_mapping_matches_direct_oracle():
    output, receipt, _ = _run(_arrays(batch=2, suffixes=(3, 8)))
    assert output.shape == (2, 8, 3, 8)
    assert receipt.query_heads == 8
    assert receipt.kv_heads == 2


def test_additive_masks_include_fully_masked_segment():
    arrays = _arrays(batch=2, query=2, base=7, suffixes=(3, 5))
    base_mask = mx.zeros((2, 1, 2, 7), dtype=mx.float32)
    base_mask = mx.concatenate(
        (base_mask[:1], mx.full((1, 1, 2, 7), -mx.inf)), axis=0
    )
    suffix_masks = [
        mx.array([[[[0.0, -mx.inf, 0.0], [0.0, 0.0, -mx.inf]]]]),
        mx.zeros((1, 1, 2, 5), dtype=mx.float32),
    ]
    _run(arrays, base_mask=base_mask, suffix_masks=suffix_masks)


def test_boolean_broadcast_mask_matches_independent_numpy_oracle():
    arrays = _arrays(batch=2, query=2, base=7, suffixes=(3, 5), seed=101)
    q, bk, bv, sk, sv = arrays
    base_mask = mx.array([[[[True, True, False, True, True, True, False]] * 2]])
    suffix_masks = [
        mx.array([[[[True, False, True], [True, True, False]]]]),
        mx.array([[[[True, True, False, True, True]] * 2]]),
    ]
    actual, _, _ = _run(
        arrays, base_mask=base_mask, suffix_masks=suffix_masks
    )
    qn, bkn, bvn = np.asarray(q), np.asarray(bk), np.asarray(bv)
    expected_rows = []
    for row in range(2):
        keys = np.concatenate((bkn, np.asarray(sk[row])), axis=2)
        values = np.concatenate((bvn, np.asarray(sv[row])), axis=2)
        repeat = qn.shape[1] // keys.shape[1]
        keys = np.repeat(keys, repeat, axis=1)
        values = np.repeat(values, repeat, axis=1)
        scores = qn[row : row + 1] @ keys.swapaxes(-1, -2)
        scores *= 1.0 / math.sqrt(qn.shape[-1])
        allowed = np.concatenate(
            (np.asarray(base_mask, dtype=bool), np.asarray(suffix_masks[row], dtype=bool)),
            axis=-1,
        )
        scores = np.where(allowed, scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected_rows.append(weights @ values)
    expected = np.concatenate(expected_rows, axis=0)
    np.testing.assert_allclose(
        np.asarray(actual), expected, atol=2.0e-6, rtol=2.0e-5
    )


def test_adversarial_logits_are_stable():
    q, bk, bv, sk, sv = _arrays(batch=2, query=2, base=5, suffixes=(2, 4))
    q = q * 900.0
    bk = bk * 700.0
    sk = [value * -800.0 for value in sk]
    actual, _, _ = _run((q, bk, bv, sk, sv))
    assert bool(mx.all(mx.isfinite(actual)).item())


def test_shared_base_is_not_mutated():
    arrays = _arrays(batch=4, suffixes=(1, 2, 3, 4))
    _, bk, bv, _, _ = arrays
    before_k, before_v = np.asarray(bk).copy(), np.asarray(bv).copy()
    _run(arrays)
    np.testing.assert_array_equal(np.asarray(bk), before_k)
    np.testing.assert_array_equal(np.asarray(bv), before_v)


def test_invalid_geometry_fails_closed():
    q, bk, bv, sk, sv = _arrays(batch=2, suffixes=(1, 2))
    with pytest.raises(ValueError, match="cover every query row"):
        shared_prefix_segmented_attention(
            q, bk, bv, sk[:1], sv[:1], scale=1.0
        )
    with pytest.raises(ValueError, match="query/key geometry"):
        shared_prefix_segmented_attention(
            q,
            bk,
            bv,
            sk,
            sv,
            scale=1.0,
            base_mask=mx.ones((2, 1, 2, 11), dtype=mx.bool_),
        )
