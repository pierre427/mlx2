"""Lockstep multirow chain verification equals ordinary token-major rows; CPU only."""

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config, hash_metadata

from mlx2.adapters import deepseek_v41_dspark_chain as chain
from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows
from mlx2.adapters.deepseek_v41_external import DeepSeekV41VerifyRows
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher

pytest.importorskip("greenlet")


class RecordingExecutor:
    """Per-row originals (exact on CPU), recording each served site group."""

    def __init__(self):
        self.groups = []

    def run(self, parks):
        self.groups.append((parks[0].key, len(parks)))
        return [p.original(*p.args, **p.kwargs) for p in parks]


def _model(engram=False, **kwargs):
    mx.set_default_device(mx.cpu)
    if not engram:
        c = config(sliding_window=4, **kwargs)
        return DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-lockstep")
    c = config(sliding_window=4, engram_layer_ids=(1, 3), engram_num_embeddings=(48, 72),
               engram_compressed_vocab_size=5, **kwargs)
    weights = Weights(c)
    rng = np.random.default_rng(1989)
    for layer in c.engram_layer_ids:
        weights.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones((4, 8))
        weights.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones((4, 8))
        weights.data[f"blk.{layer}.engram_kv.weight"] = mx.array(rng.normal(0, .0004, (40, 6144)).astype(np.float32))
    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(19)]

    class Rows:
        def prefetch(self, ids):
            pass

        def read(self, ids):
            return np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256)).astype(np.float32).copy()
    return DeepSeekV41Model(c, weights, identity="synthetic@v41-lockstep-engram",
                            hasher=EngramHasher(metadata), engram_tables={i: Rows() for i in c.engram_layer_ids})


def _state(cache):
    return [(s.meta_state, [None if a is None else np.array(a) for a in s.state]) for s in cache]


def _assert_state(a, b):
    for (meta_a, arrays_a), (meta_b, arrays_b) in zip(a, b):
        assert meta_a == meta_b
        for x, y in zip(arrays_a, arrays_b):
            assert (x is None) == (y is None)
            if x is not None:
                np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("prefix", [1, 3, 6, 127])
@pytest.mark.parametrize("width", [2, 4, 6])
def test_lockstep_rows_equal_token_major_taps_and_state(engram, prefix, width):
    model = _model(engram)
    cache = model.make_cache()
    model(mx.array([(np.arange(prefix) % 19).tolist()], mx.uint32), cache)
    base = [s.snapshot() for s in cache]
    tokens = mx.array([[1, 2, 3, 4, 5, 6][:width]], mx.uint32)
    reference = [model.forward_with_taps(tokens[:, i:i + 1], cache, (1, 4)) for i in range(width)]
    expected_logits = mx.concatenate([r[0] for r in reference], 1)
    expected_taps = mx.concatenate([r[1] for r in reference], 1)
    expected_state = _state(cache)
    for state, saved in zip(cache, base):
        state.restore(saved)
    recorder = RecordingExecutor()
    logits, taps, frontiers = chain.verify(model, tokens, cache, (1, 4), executor=recorder)
    mx.eval(logits, taps)
    np.testing.assert_array_equal(np.array(expected_logits), np.array(logits))
    np.testing.assert_array_equal(np.array(expected_taps), np.array(taps))
    _assert_state(expected_state, _state(cache))
    assert len(frontiers) == width - 1 and all(len(row) == len(cache) for row in frontiers)
    # Every served post-attention site covered all rows at once.
    assert recorder.groups and all(count == width for _, count in recorder.groups)
    kinds = {key[0] for key, _ in recorder.groups}
    assert {"grouped", "gate_up", "linear"} <= kinds
    names = {str(key[1]) for key, _ in recorder.groups}
    assert any(n.endswith(".attn_q_b.weight") and "indexer" not in n for n in names)
    assert not any("indexer" in n or "compressor" in n for n in names)
    assert model._weight_executions == 0


@pytest.mark.parametrize("accepted", range(7))
@pytest.mark.parametrize("taps", [False, True])
def test_lockstep_local_frontier_commit_equals_replay(accepted, taps):
    model = _model()
    cache = model.make_cache()
    model(mx.array([(np.arange(5) % 19).tolist()], mx.uint32), cache)
    original = [s.snapshot() for s in cache]
    tokens = mx.array([[1, 2, 3, 4, 5, 6]], mx.uint32)
    ref = DeepSeekV4VerifyRows(model, [cache], (4,)).begin([6])
    expected, expected_features = ref.forward(tokens, taps=taps)
    ref.commit([accepted])
    expected_state = _state(cache)
    for state, saved in zip(cache, original):
        state.restore(saved)
    model.verify_rows_policy = "lockstep_multirow_v1"
    tx = DeepSeekV41VerifyRows(model, [cache], (4,)).begin([6])
    actual, features = tx.forward(tokens, taps=taps)
    tx.commit([accepted])
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))
    if taps:
        np.testing.assert_array_equal(np.array(expected_features), np.array(features))
    else:
        assert features is None
    _assert_state(expected_state, _state(cache))
    following = model(mx.array([[7]], mx.uint32), cache)
    mx.eval(following)


def test_lockstep_refuses_unsupported_profiles_and_restores_weights():
    model = _model()
    cache = model.make_cache()
    model(mx.array([[1, 2]], mx.uint32), cache)
    model.kv_storage_policy = "native_packed"
    with pytest.raises(ValueError, match="native packed"):
        chain.verify(model, mx.array([[3, 4]], mx.uint32), cache)
    model.kv_storage_policy = "reference"
    model.weights.evaluate_block = lambda *a, **k: None
    with pytest.raises(ValueError, match="block transactions"):
        chain.verify(model, mx.array([[3, 4]], mx.uint32), cache)
    del model.weights.evaluate_block

    class Failing(RecordingExecutor):
        def run(self, parks):
            raise RuntimeError("synthetic executor failure")
    with pytest.raises(RuntimeError, match="synthetic"):
        chain.verify(model, mx.array([[3, 4]], mx.uint32), cache, executor=Failing())
    assert not {"linear", "grouped", "gate_up"} & set(vars(model.weights))
    assert model._weight_executions == 0
    with pytest.raises(ValueError, match="one to six"):
        chain.verify(model, mx.array([[1] * 7], mx.uint32), model.make_cache())


def test_grouped_woa_declines_mxfp8_storage_before_reading_bf16_coefficient():
    class MXFP8Weights:
        source_woa_pp_policy = "reference"
        source_woa_storage_policy = "mxfp8_gemv_v1"
        source_bf16_loads_policy = "direct_f32"
        source_resident_bank_session = None

        def array(self, name):
            raise AssertionError("MXFP8 WoA has no resident BF16 coefficient")

    executor = chain.SourceRowsExecutor(MXFP8Weights(), {})
    parks = [chain._Park("grouped", "blk.0.attn_output_a.weight", (mx.zeros((1, 1, 8, 4)), "blk.0.attn_output_a.weight"),
                         {"groups": 8}, None) for _ in range(2)]
    with pytest.raises(chain._Decline):
        executor._grouped(parks)
