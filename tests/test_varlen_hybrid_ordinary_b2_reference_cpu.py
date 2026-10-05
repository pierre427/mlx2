"""CPU proofs for the stock merged-cache hybrid research comparator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import ArraysCache, KVCache


SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/varlen_hybrid_ordinary_b2_reference.py"
SPEC = importlib.util.spec_from_file_location("hybrid_ordinary_b2_reference", SOURCE)
REF = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
import sys
sys.modules[SPEC.name] = REF
SPEC.loader.exec_module(REF)


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def rows():
    result = []
    for length in (32, 96):
        kv = KVCache()
        kv.keys = mx.ones((1, 2, length, 4), dtype=mx.float16)
        kv.values = mx.ones((1, 2, length, 4), dtype=mx.float16)
        kv.offset = length
        recurrent = ArraysCache(2)
        recurrent.cache = [mx.ones((1, 3, 4), dtype=mx.float16),
                           mx.ones((1, 3, 4), dtype=mx.float32)]
        result.append((kv, recurrent))
    return tuple(result)


class DummyModel:
    def model(self, tokens, *, cache):
        assert tuple(tokens.shape) == (2, 1)
        assert len(cache) == 2
        cache[0].offset = cache[0].offset + 1
        return tokens.astype(mx.float32)[..., None]

    def logits(self, hidden):
        return hidden


def merged():
    return REF.OrdinaryHybridB2Reference.from_row_caches(
        DummyModel(), rows(), full_attention_layers=(0,),
        recurrent_layers=(1,), expected_offsets=(32, 96), mx=mx)


def test_stock_merge_preserves_ragged_offsets_and_recurrent_rows():
    reference = merged()
    kv, recurrent = reference.merged_cache
    assert tuple(int(x.item()) for x in kv.offset) == (32, 96)
    assert tuple(int(x.item()) for x in kv.left_padding) == (64, 0)
    assert tuple(kv.keys.shape) == (2, 2, 96, 4)
    assert all(value.shape[0] == 2 for value in recurrent.cache)
    logits = reference.forward_one((7, 9))
    assert tuple(logits.shape) == (2, 1)
    assert tuple(float(x.item()) for x in logits[:, 0]) == (7.0, 9.0)


def test_slot_comparator_reports_absolute_relative_and_mixed_tolerance():
    reference = merged()
    private = []
    for lane in range(2):
        cache = ArraysCache(2)
        cache.cache = [mx.array(reference.merged_cache[1].cache[0][lane:lane + 1]),
                       mx.array(reference.merged_cache[1].cache[1][lane:lane + 1])]
        private.append((cache,))
    exact = REF.compare_recurrent_slots(tuple(private), reference, atol=0.0, rtol=0.0)
    assert exact["passed"] and exact["max_abs"] == exact["max_rel"] == 0.0
    assert len(exact["slots"]) == 4
    private[0][0].cache[1] = private[0][0].cache[1] + 0.5
    drift = REF.compare_recurrent_slots(tuple(private), reference, atol=0.1, rtol=0.1)
    assert not drift["passed"] and drift["max_abs"] == 0.5
    assert drift["max_rel"] == 0.5
    assert (drift["slots"][1]["lane"], drift["slots"][1]["slot"]) == (0, 1)


def test_refuses_nonfinite_state_and_malformed_row_boundary():
    reference = merged()
    private = []
    for lane in range(2):
        cache = ArraysCache(2)
        cache.cache = [mx.array(reference.merged_cache[1].cache[0][lane:lane + 1]),
                       mx.array(reference.merged_cache[1].cache[1][lane:lane + 1])]
        private.append((cache,))
    private[1][0].cache[0] = private[1][0].cache[0] * mx.array(float("nan"), dtype=mx.float16)
    with pytest.raises(ValueError, match="nonfinite"):
        REF.compare_recurrent_slots(tuple(private), reference, atol=0.1)
    malformed = rows()
    malformed[0][0].offset = 31
    with pytest.raises(ValueError, match="prompt boundary"):
        REF.OrdinaryHybridB2Reference.from_row_caches(
            DummyModel(), malformed, full_attention_layers=(0,),
            recurrent_layers=(1,), expected_offsets=(32, 96), mx=mx)
