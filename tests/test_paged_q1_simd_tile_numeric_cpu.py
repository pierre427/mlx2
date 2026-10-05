"""CPU oracle for the default-off four-stripe Q1 Metal reduction.

This checks stable partial combination and page-table boundaries. Actual Metal
compilation, terminal ownership and model parity require a separate GPU gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _paged_values(retained: int, visible: int, dim: int, window: int):
    end = retained + visible
    lower = max(retained, end - min(end, window)) if window else retained
    first_block = retained // 64
    page_ids = tuple(range(5, 5 + (end - 1) // 64 - first_block + 1))
    rng = np.random.default_rng(retained + visible + dim + window)
    query = rng.normal(0, 0.2, dim).astype(np.float16).astype(np.float32)
    keys = rng.normal(0, 0.2, (len(page_ids), 64, dim)).astype(np.float16)
    values = rng.normal(0, 0.2, (len(page_ids), 64, dim)).astype(np.float16)

    def load(token: int):
        page_index = token // 64 - first_block
        assert 0 <= page_index < len(page_ids)
        return (keys[page_index, token & 63].astype(np.float32),
                values[page_index, token & 63].astype(np.float32))

    return query, load, lower, end


def _online(query, load, tokens):
    maximum = -math.inf
    denominator = 0.0
    numerator = np.zeros_like(query)
    for token in tokens:
        key, value = load(token)
        score = float(np.dot(query, key)) / math.sqrt(len(query))
        next_max = max(maximum, score)
        previous = 0.0 if maximum == -math.inf else math.exp(maximum - next_max)
        current = math.exp(score - next_max)
        numerator = numerator * previous + current * value
        denominator = denominator * previous + current
        maximum = next_max
    return maximum, denominator, numerator


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("retained,visible,window", [
    (0, 32, 0), (0, 63, 0), (0, 64, 0), (0, 65, 0), (0, 128, 0),
    (64, 63, 0), (64, 65, 0), (128, 128, 64), (65, 128, 65),
])
def test_four_stripe_partial_reduction_matches_dense_oracle(
    dim: int, retained: int, visible: int, window: int,
) -> None:
    query, load, lower, end = _paged_values(retained, visible, dim, window)
    stripes = [_online(query, load, range(lower + stripe, end, 4))
               for stripe in range(4)]
    global_max = max(partial[0] for partial in stripes)
    normalizers = [math.exp(partial[0] - global_max) for partial in stripes]
    denominator = sum(partial[1] * factor
                      for partial, factor in zip(stripes, normalizers))
    numerator = sum((partial[2] * factor
                     for partial, factor in zip(stripes, normalizers)),
                    np.zeros_like(query))
    actual = (numerator / denominator).astype(np.float16)

    keys, values = zip(*(load(token) for token in range(lower, end)))
    scores = np.stack(keys).astype(np.float64) @ query.astype(np.float64)
    scores /= math.sqrt(dim)
    weights = np.exp(scores - scores.max())
    expected = ((weights @ np.stack(values).astype(np.float64)) /
                weights.sum()).astype(np.float16)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)
