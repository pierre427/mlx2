"""Tiny CPU ordinary/CED dependency-state oracle; no Source checkpoint/GPU."""
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41_ced_bounded import (
    BoundedCedExecutor, CedResult, CedExecutionError, OrdinaryRequired, PROFILE)
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from test_deepseek_v41_cpu import config, Weights, hash_metadata
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def model(*, activation_dtype='float32', window=4):
    c = config(sliding_window=window, activation_dtype=activation_dtype)
    w = Weights(c); calls = []
    original = w.moe
    def moe(x, ids, scores, prefix):
        calls.append((prefix, x.shape[0])); return original(x, ids, scores, prefix)
    w.moe = moe
    return DeepSeekV41Model(c, w, identity='synthetic-ced-model@seed567'), calls


def compare_states(reference, actual, *, atol=3e-6, rtol=3e-5):
    for ref, test in zip(reference, actual):
        assert ref.meta_state[2:] == test.meta_state[2:]
        for a, b in zip(ref.state, test.state):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)
        test.validate_state()
        assert not test.pending


@pytest.mark.parametrize('length', [1, 3, 4, 5, 17, 31])
@pytest.mark.parametrize('activation_dtype', ['float32', 'bfloat16'])
def test_bounded_prefix_preserves_logits_all_state_and_continuation(length, activation_dtype):
    ordinary, _ = model(activation_dtype=activation_dtype)
    candidate, rows = model(activation_dtype=activation_dtype)
    executor = BoundedCedExecutor(candidate, encoder_layers=3, reference_chunk_rows=1,
                                  numerical_profile=PROFILE)
    ids = mx.array((np.arange(length)[None] % 19), mx.int32)
    reference_cache = ordinary.make_cache()
    expected = ordinary.prefill_forward(ids, reference_cache)
    result = executor.prefill(ids)
    assert isinstance(result, CedResult) and result.logits.shape == (1, 1, 19)
    np.testing.assert_allclose(np.asarray(result.logits), np.asarray(expected[:, -1:]), atol=3e-6, rtol=3e-5)
    compare_states(reference_cache, result.cache)
    assert result.receipt['final_offsets'] == [length]*5
    assert result.receipt['caller_cache_mutated'] is False
    assert result.receipt['source_numerical_equivalence_claimed'] is False
    boundary = result.receipt['stages'][3]
    assert boundary['full_boundary_pool_prepared'] and boundary['attention_rows'] == length
    assert result.cache[3].pooled.shape[1] == length
    assert result.model.identity != ordinary.identity
    with pytest.raises(ValueError, match='identity'):
        ordinary(mx.array([[7]], mx.int32), result.cache)
    for token in (7, 8, 9):
        x = mx.array([[token]], mx.int32)
        expected = ordinary(x, reference_cache)
        actual = result.model(x, result.cache)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-6, rtol=3e-5)
    compare_states(reference_cache, result.cache)
    if length > 4:
        assert ('blk.3', 4) in rows and ('blk.4', 1) in rows


def test_alignment_is_explicit_and_cache_profile_is_bound():
    base, _ = model()
    a = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    b = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=8, numerical_profile=PROFILE)
    assert a.identity != b.identity
    result = b.prefill(mx.array((np.arange(31)[None] % 19), mx.int32), output_rows=2)
    assert result.logits.shape == (1, 2, 19)
    assert all(s['output_first'] % 8 == 0 for s in result.receipt['stages'][3:])


@pytest.mark.parametrize('kwargs', [dict(prompt_logprobs=True), dict(all_hidden_states=True),
                                  dict(output_rows=0), dict(output_rows=33)])
def test_unsupported_requests_choose_ordinary_without_mutation(kwargs):
    base, rows = model()
    executor = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    result = executor.prefill(mx.ones((1, 31), mx.int32), **kwargs)
    assert isinstance(result, OrdinaryRequired) and result.ordinary_required and not result.mutated_cache
    assert not rows and executor._model is None


def test_larger_prefix_resumed_cache_and_unsupported_topology_refuse_early():
    base, rows = model()
    executor = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    assert isinstance(executor.prefill(SimpleNamespace(ndim=2, shape=(1, 1025))), OrdinaryRequired)
    cache = base.make_cache()
    assert executor.prefill(mx.array([[1]], mx.int32), cache=cache).reason.startswith('caller_cache')
    assert all(s.offset == 0 and s.keys is None for s in cache)
    base.weights.source_row_packet_policy = 'row_packet_v1'
    assert executor.prefill(mx.array([[1]], mx.int32)).reason.startswith('row_packet')
    assert not rows and executor._model is None


def test_integer_dtype_and_generic_prefill_capability_refuse_early():
    base, rows = model()
    executor = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    assert executor.prefill(mx.ones((1, 3), mx.float32)).reason == 'integer_prompt_tokens_required'
    base.weights.dense_prefill_policy = 'unadmitted_tiled'
    assert executor.prefill(mx.ones((1, 3), mx.int32)).reason == 'generic_prefill_context_unsupported'
    assert not rows and executor._model is None


def test_failure_keeps_private_state_and_leases_until_explicit_drain(monkeypatch):
    base, _ = model()
    executor = BoundedCedExecutor(base, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    original = base.weights.moe
    def fail(x, ids, scores, prefix):
        if prefix == 'blk.4': raise RuntimeError('injected private decoder failure')
        return original(x, ids, scores, prefix)
    monkeypatch.setattr(base.weights, 'moe', fail)
    with pytest.raises(CedExecutionError, match='before publishing') as failure:
        executor.prefill(mx.array((np.arange(31)[None] % 19), mx.int32))
    retained = failure.value.retained_cache
    assert len(retained) == 5 and not failure.value.published
    assert base._weight_executions == executor._model._weight_executions == 0
    assert all(s._in_forward for s in retained)
    assert isinstance(executor.prefill(mx.array([[1]], mx.int32)), OrdinaryRequired)
    with pytest.raises(RuntimeError, match='drain failed'):
        executor.drain_failed(lambda *values: (_ for _ in ()).throw(RuntimeError('drain failed')))
    assert executor._retained_failure and not retained[0]._closed
    executor.drain_failed(mx.eval)
    assert not executor._retained_failure and all(s._closed for s in retained)


def test_explicit_profile_and_integer_configuration_are_required():
    base, _ = model()
    for kwargs in (dict(numerical_profile='reference'), dict(encoder_layers=0),
                   dict(encoder_layers=True), dict(reference_chunk_rows=0)):
        values = dict(encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE); values.update(kwargs)
        with pytest.raises(ValueError): BoundedCedExecutor(base, **values)


def engram_model(*, fail_context_end=False):
    c = config(sliding_window=4, engram_layer_ids=(1, 2),
               engram_num_embeddings=(48, 72), engram_compressed_vocab_size=5)
    weights = Weights(c); rng = np.random.default_rng(1989); events = []
    for layer in c.engram_layer_ids:
        weights.data[f'blk.{layer}.engram_q_norm.weight'] = mx.ones((4, 8))
        weights.data[f'blk.{layer}.engram_k_norm.weight'] = mx.ones((4, 8))
        weights.data[f'blk.{layer}.engram_kv.weight'] = mx.array(rng.normal(0, .0004, (40, 6144)).astype(np.float32))
    class Rows:
        def prefetch(self, ids): pass
        def read(self, ids):
            return np.broadcast_to(np.asarray(ids)[..., None]/100, (*ids.shape, 256)).astype(np.float32).copy()
    def begin(keys): events.append('begin')
    def end():
        events.append('end')
        if fail_context_end: raise RuntimeError('injected atlas context end failure')
    weights.expert_stream = SimpleNamespace(atlas=object(), begin_context=begin, end_context=end)
    metadata = hash_metadata(); metadata['deepseek41.engram.token_map'] = [i % 5 for i in range(19)]
    target = DeepSeekV41Model(c, weights, identity='synthetic-ced-engram@seed1989',
        hasher=EngramHasher(metadata), engram_tables={i: Rows() for i in c.engram_layer_ids})
    return target, events


def test_encoder_engram_history_and_atlas_contexts_follow_full_prefix():
    ordinary, reference_events = engram_model()
    candidate, events = engram_model()
    executor = BoundedCedExecutor(candidate, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    ids = mx.array((np.arange(17)[None] % 19), mx.int32)
    cache = ordinary.make_cache(); expected = ordinary.prefill_forward(ids, cache)
    result = executor.prefill(ids)
    np.testing.assert_allclose(np.asarray(result.logits), np.asarray(expected[:, -1:]), atol=3e-6, rtol=3e-5)
    compare_states(cache, result.cache)
    assert events == reference_events == ['begin', 'end']
    assert result.cache[0].hash_tail.tolist() == cache[0].hash_tail.tolist()


def test_context_cleanup_failure_never_publishes_and_restores_execution_counters():
    candidate, events = engram_model(fail_context_end=True)
    executor = BoundedCedExecutor(candidate, encoder_layers=3, reference_chunk_rows=1, numerical_profile=PROFILE)
    with pytest.raises(CedExecutionError, match='context end failure') as failure:
        executor.prefill(mx.array((np.arange(17)[None] % 19), mx.int32))
    assert not failure.value.published and events == ['begin', 'end']
    assert candidate._weight_executions == executor._model._weight_executions == 0
    assert executor._retained_failure
    executor.drain_failed(mx.eval)
