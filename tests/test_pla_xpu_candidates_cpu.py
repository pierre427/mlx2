"""Source-independent routing, decode preservation and KDA recurrence gates."""
from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import gated_delta as gd
from mlx2.runtime.models import gdn_chunk32 as chunk
from mlx2.runtime.models import moe_nax_gather as moe
from mlx2.runtime.models import qwen4_qsa_indexed_merge as qsa


def test_candidate_never_takes_ordinary_decode():
    q = mx.zeros((1, 1, 1, 128)); v = mx.zeros((1, 1, 8, 128))
    g = mx.ones((1, 1, 8)); state = mx.zeros((1, 8, 128, 128))
    stats = Counter()
    assert gd._chunked_prefill(q, q, v, g, g, state, None, 32, stats) is None
    assert stats == {"fallback_short_sequence": 1}


def test_compact_tile_is_explicit_and_pinned_plan_wins(monkeypatch):
    ordinary = moe._plan(81920, 512, 2560, 640)
    monkeypatch.setenv("MLX_QWEN4_COMPACT_MOE_PREFILL", "1")
    assert moe._plan(81920, 512, 2560, 640).bm == 64
    assert ordinary.bm == 128
    assert moe._plan(81921, 512, 2560, 640) == ordinary
    monkeypatch.setenv("MLX2_MOE_NAX_GATHER_PLAN", "seg,96,128,0,0")
    assert moe._plan(81920, 512, 2560, 640).bm == 96


def test_qsa_prefetch_keeps_rounding_and_rejects_unknown_source_seam():
    for source in (qsa._SOURCE, qsa._SOURCE_GATED):
        out = qsa.prefetch_source(source)
        assert "base + j < P" in out
        assert "volatile float state_o_product" in out
        assert "volatile float chunk_o_product" in out
    with pytest.raises(ValueError, match="seam"):
        qsa.prefetch_source("unknown source")


@pytest.mark.parametrize("length", [17, 33, 65])
def test_vector_kda_chunk_matches_independent_fp64_reference(length):
    rng = np.random.default_rng(12)
    h = 4
    q, k, v = (rng.normal(size=(1, length, h, 128)).astype(np.float32) / 12 for _ in range(3))
    g = rng.uniform(.7, 1, size=q.shape).astype(np.float32)
    g[:, length // 2, 0, :64] = 0
    beta = rng.uniform(.1, .8, size=q.shape[:-1]).astype(np.float32)
    state = rng.normal(size=(1, h, 128, 128)).astype(np.float32) / 10
    current = state.astype(np.float64); expected = []
    for i in range(length):
        current *= g[:, i, :, None, :]
        delta = (v[:, i] - np.sum(current * k[:, i, :, None, :], axis=-1)) * beta[:, i, :, None]
        current += delta[..., None] * k[:, i, :, None, :]
        expected.append(np.sum(current * q[:, i, :, None, :], axis=-1))
    y, result = chunk.chunk32(*(mx.array(a) for a in (q, k, v, g, beta, state)))
    np.testing.assert_allclose(np.array(y), np.stack(expected, axis=1), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(np.array(result), current, atol=2e-6, rtol=2e-5)
