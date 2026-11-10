"""Opt-in packed indexer-key decode-once lever (after sglang#49068), on Metal.

Under native packed KV, ``Attention.select`` decodes the fp4 indexer keys once
per query-row tile. ``MLX2_DS41_PACKED_INDEX_DECODE_ONCE`` decodes each key
chunk once per call and reuses it across tiles. The arms must agree bit for
bit (logits and every cache plane), and the counters must show that the lever
engaged (reuses > 0, fewer decodes) in the on arm and not in the off arm.

``index_topk=2`` and one candidate block make selection depend on the score
values (with a top-k wider than the history every position is kept and a
wrong-but-finite score would go unseen), and 2-key chunks give every call
several chunks. Synthetic weights only: no checkpoint, speed or qualification
claim.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config

from mlx2.runtime.models import deepseek_v41 as v41
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (owned GPU window) for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


_REAL_TOPK, _REAL_BLOCKS = v41.stable_pool_topk, v41.candidate_blocks


def _bits(x):
    mx.eval(x)
    return np.array(x.astype(mx.float32)).tobytes()


def _run(monkeypatch, decode_once):
    c = config(head_dim=32, index_head_dim=32, simulate_quantization=True,
               activation_dtype="bfloat16", sliding_window=3,
               index_topk=2, candidate_topk_blocks=1)
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@decode-once",
                             index_policy="reference", selected_kv_policy="decode_reuse",
                             kv_storage_policy="native_packed")
    monkeypatch.setattr(v41, "PACKED_INDEX_DECODE_ONCE", decode_once)
    # One query row per tile, so every multi-row call has several tiles.
    monkeypatch.setattr(v41, "INDEX_SCORE_SCRATCH_BYTES", 1)
    # Several key chunks per call, so the per-chunk reuse key is exercised.
    monkeypatch.setattr(v41, "PACKED_INDEX_KEY_CHUNK", 2)
    # Record every indexer score tensor (selection alone can hide a uniformly
    # rescaled score, since it only ranks).
    scores = []

    def topk(score, *a, **k):
        scores.append(_bits(score))
        return _REAL_TOPK(score, *a, **k)

    def blocks(score, *a, **k):
        scores.append(_bits(score))
        return _REAL_BLOCKS(score, *a, **k)

    monkeypatch.setattr(v41, "stable_pool_topk", topk)
    monkeypatch.setattr(v41, "candidate_blocks", blocks)
    cache = model.make_cache()
    logits = []
    for chunk in ([1, 2, 3, 4, 5, 6, 7, 8], [9], [10, 11, 12, 13, 14, 15], [16, 17]):
        logits.append(model(mx.array([chunk]), cache, logits_to_keep=0))
    return logits, cache, dict(model.scheduled_kv_calls), scores


def test_decode_once_is_bit_identical_and_engages(monkeypatch):
    off_logits, off_cache, off_calls, off_scores = _run(monkeypatch, False)
    on_logits, on_cache, on_calls, on_scores = _run(monkeypatch, True)
    assert off_scores and len(off_scores) == len(on_scores)
    assert off_scores == on_scores, "indexer scores differ between arms"
    for a, b in zip(off_logits, on_logits):
        assert _bits(a) == _bits(b)
    for a, b in zip(off_cache, on_cache):
        for name in ("keys", "pooled", "index_keys"):
            av, bv = getattr(a, name), getattr(b, name)
            assert (av is None) == (bv is None), name
            if av is not None:
                assert _bits(av) == _bits(bv), name
    # Mechanism: the off arm decoded per tile and never reused; the on arm
    # reused chunks and decoded strictly fewer.
    assert off_calls.get("packed_index_decoded_chunks", 0) > 0, off_calls
    assert off_calls.get("packed_index_reused_chunks", 0) == 0, off_calls
    assert on_calls.get("packed_index_reused_chunks", 0) > 0, on_calls
    assert on_calls["packed_index_decoded_chunks"] < off_calls["packed_index_decoded_chunks"]
    assert (on_calls["packed_index_decoded_chunks"] + on_calls["packed_index_reused_chunks"]
            == off_calls["packed_index_decoded_chunks"])
    assert on_calls.get("packed_index_resident_max_rows", 0) > v41.PACKED_INDEX_KEY_CHUNK
    assert "packed_index_resident_max_rows" not in off_calls
