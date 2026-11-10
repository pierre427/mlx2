"""Synthetic layer-major Source prefill contracts; no checkpoint, GPU or timing."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config, hash_metadata

from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_layer_major import (
    CHUNK_ROWS, KEY, MAX_WAVE_ROWS, POLICY, WAVE_ROW_BYTES, WAVE_SAFETY_BYTES,
    layer_major_model_class, receipt, wave_rows)
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_engram import EngramHasher


def bf16_config(**kw):
    return config(hidden_size=32, head_dim=32, q_lora_rank=32, qk_rope_head_dim=16,
                  index_head_dim=32, o_lora_rank=16, moe_intermediate_size=32,
                  simulate_quantization=True, activation_dtype='bfloat16', **kw)


class Rows:
    def __init__(self):
        self.pending = None
        self.reads = 0

    def prefetch(self, ids):
        if self.pending is not None and not np.array_equal(self.pending, np.asarray(ids)):
            raise ValueError('Engram prefetch window is already owned')
        self.pending = np.asarray(ids).copy()

    def read(self, ids):
        self.prefetch(ids)
        self.pending = None
        self.reads += 1
        return np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256)).astype(np.float32).copy()


def pair(c, *, engram=False, chunk=3):
    def weights():
        w = Weights(c)
        w.source_prefill_order_policy = POLICY
        if engram:
            rng = np.random.default_rng(1989)
            for layer in c.engram_layer_ids:
                w.data[f'blk.{layer}.engram_q_norm.weight'] = mx.ones((4, c.hidden_size))
                w.data[f'blk.{layer}.engram_k_norm.weight'] = mx.ones((4, c.hidden_size))
                w.data[f'blk.{layer}.engram_kv.weight'] = mx.array(
                    rng.normal(0, 0.0004, (5 * c.hidden_size, 6144)).astype(np.float32))
        return w
    extra = {}
    if engram:
        metadata = hash_metadata()
        metadata['deepseek41.engram.token_map'] = [i % 5 for i in range(c.vocab_size)]
        extra = lambda: dict(hasher=EngramHasher(metadata), engram_tables={i: Rows() for i in c.engram_layer_ids})
    else:
        extra = lambda: {}
    model = layer_major_model_class()(c, weights(), identity='synthetic-layer-major@1', **extra())
    model.source_layer_major_rows = chunk
    ref = DeepSeekV41Model(c, weights(), identity='synthetic-layer-major@1', **extra())
    return model, ref


def chunk_major(ref, ids, cache, chunk):
    return mx.concatenate([ref(ids[:, s:s + chunk], cache) for s in range(0, ids.shape[1], chunk)], 1)


def assert_states_equal(a, b):
    for x, y in zip(a, b):
        assert x.offset == y.offset and x._pool_count == y._pool_count
        for p, q in zip(x.state, y.state):
            if p is None:
                assert q is None
            else:
                np.testing.assert_array_equal(np.asarray(p).view(np.uint8), np.asarray(q).view(np.uint8))


@pytest.mark.parametrize('rows', [4, 7, 9, 17])
@pytest.mark.parametrize('kind', ['f32', 'bf16'])
def test_wave_is_bitwise_chunk_major_with_compressed_consumers(rows, kind):
    c = config() if kind == 'f32' else bf16_config()
    model, ref = pair(c)
    ids = mx.array((np.arange(rows) * 5 % c.vocab_size)[None], mx.int32)
    cache, rc = model.make_cache(), ref.make_cache()
    actual = model(ids, cache)
    expected = chunk_major(ref, ids, rc, 3)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32))
    assert_states_equal(cache, rc)
    assert model.scheduled_kv_calls['source_layer_major_waves'] == 1
    assert model.scheduled_kv_calls['source_layer_major_chunks'] == -(-rows // 3)
    # Continuation after a wave decodes identically.
    nxt = mx.array([[3]], mx.int32)
    np.testing.assert_array_equal(np.asarray(model(nxt, cache)).view(np.uint32),
                                  np.asarray(ref(nxt, rc)).view(np.uint32))


def test_warm_prefix_wave_and_engram_tail_match_chunk_major():
    c = config(engram_layer_ids=(1, 3), engram_num_embeddings=(48, 72), engram_compressed_vocab_size=5)
    model, ref = pair(c, engram=True)
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], mx.int32)
    cache, rc = model.make_cache(), ref.make_cache()
    a = model(ids[:, :2], cache)  # ordinary single chunk, then a warm wave
    b = model(ids[:, 2:], cache)
    expected = mx.concatenate([ref(ids[:, :2], rc), chunk_major(ref, ids[:, 2:], rc, 3)], 1)
    np.testing.assert_array_equal(np.asarray(mx.concatenate([a, b], 1)).view(np.uint32),
                                  np.asarray(expected).view(np.uint32))
    assert_states_equal(cache, rc)
    assert cache[0].hash_tail.tolist() == rc[0].hash_tail.tolist()
    assert all(t.pending is None for t in model.engram_tables.values())


def test_checkpoint_row_guard_narrows_wave_head():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    c = config()
    model, ref = pair(c)
    ids = mx.array([list(range(1, 9))], mx.int32)
    cache, rc = model.make_cache(), ref.make_cache()
    with prefill_checkpoint_rows([8], width=8) as request:
        out = model(ids, cache)
    full = chunk_major(ref, ids, rc, 3)
    assert request.applied == 1 and out.shape == (1, 1, full.shape[-1])
    np.testing.assert_allclose(np.asarray(out[0, 0]), np.asarray(full[0, -1]), rtol=1e-5, atol=1e-6)
    keep, _ = model._forward(mx.array([[1, 2, 3, 4]], mx.int32), model.make_cache(), logits_to_keep=2)
    assert keep.shape[1] == 2


def test_invalid_late_token_and_unsupported_wave_refuse_before_mutation():
    c = config()
    model, _ = pair(c)
    cache = model.make_cache()
    with pytest.raises(ValueError, match='vocabulary'):
        model(mx.array([[1, 2, 3, 4, 5, 99]], mx.int32), cache)
    assert all(s.offset == 0 and not s._in_forward for s in cache)
    with pytest.raises(ValueError, match='taps'):
        model.forward_with_taps(mx.array([[1, 2, 3, 4]], mx.int32), cache, (1,))
    model.weights.source_prefill_order_policy = 'reference'
    with pytest.raises(ValueError, match='owner/profile'):
        model(mx.array([[1, 2, 3, 4]], mx.int32), cache)
    model.weights.source_prefill_order_policy = POLICY
    model.weights.source_resident_bank_session = object()
    with pytest.raises(ValueError, match='idle ordinary'):
        model(mx.array([[1, 2, 3, 4]], mx.int32), cache)
    assert all(s.offset == 0 for s in cache)


def test_partial_failure_leaves_cache_pending():
    c = config()
    model, _ = pair(c)
    cache = model.make_cache()
    original = model.weights.moe
    calls = []

    def broken(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise MemoryError('actual allocator failure')
        return original(*a, **kw)
    model.weights.moe = broken
    with pytest.raises(MemoryError):
        model(mx.array([[1, 2, 3, 4, 5]], mx.int32), cache)
    assert any(s._in_forward for s in cache)


def test_policy_default_choice_and_fail_closed_combinations():
    assert KEY not in source_policy()
    assert source_policy({KEY: POLICY})[KEY] == POLICY
    with pytest.raises(ValueError, match='invalid Source policy'):
        source_policy({KEY: 'expert_major'})
    # dependency_exact CED now composes (test_deepseek_v41_source_layer_major_ced_cpu).
    assert source_policy({KEY: POLICY, 'deepseek_ced': 'dependency_exact'})['deepseek_ced'] == 'dependency_exact'
    for extra in ({'deepseek_vision': 'enabled'},
                  {'deepseek_source_prefill_partition': 'canonical_m1_v1'},
                  {'deepseek_kv_storage': 'native_packed'},
                  {'deepseek_source_frame_reuse': 'prefill_v1', 'deepseek_source_prefill_batch': 'gate_up',
                   'deepseek_source_expert_math': 'routed_fused_fp32'},
                  {'draft_model': 'prompt_lookup'}):
        with pytest.raises(ValueError):
            source_policy({KEY: POLICY, **extra})


def test_identity_salt_receipt_and_model_binding():
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.kernel_policy = {}
    a.weights = SimpleNamespace()
    a.numerics_revision = a.profile = a.numeric_profile = 'base'
    a._bind_prefill_order_identity()
    reference = (a.numerics_revision, a.profile, a.numeric_profile)
    assert reference == ('base',) * 3 and a.weights.source_prefill_order_policy == 'reference'
    a.kernel_policy = {KEY: POLICY}
    a._bind_prefill_order_identity()
    assert a.numerics_revision != 'base' and 'implementation=' in a.numerics_revision
    assert a.profile.startswith('base:layer-major-') and a.numeric_profile.endswith('layer_major_v1_unqualified')
    assert receipt(POLICY)['apcv2_reuse_enabled'] is False and receipt('reference')['policy'] == 'reference'
    instance = layer_major_model_class()
    assert issubclass(instance, DeepSeekV41Model) and instance.source_layer_major_rows == CHUNK_ROWS


def test_execution_config_admits_waves_only_under_policy(monkeypatch):
    seen = []

    def base(self, *, max_lanes, prefill_step):
        if not 1 <= prefill_step <= 1024:
            raise ValueError('V4.1 requires physical width1..1 and prefill chunks1..1024')
        seen.append(prefill_step)
        return {'prefill_step_size': prefill_step}
    monkeypatch.setattr(DeepSeekV41MLXAdapter, 'execution_config', base)
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.kernel_policy = {}
    with pytest.raises(ValueError, match='1..1024'):
        a.execution_config(max_lanes=1, prefill_step=4096)
    a.kernel_policy = {KEY: POLICY}
    result = a.execution_config(max_lanes=1, prefill_step=8192)
    assert result['prefill_step_size'] == 8192 and seen[-1] == 1024
    assert result['apcv2_reuse_disabled_reason'] == 'adaptive_prefill_partitions_not_qualified'
    assert a.execution_config(max_lanes=1, prefill_step=512)['prefill_step_size'] == 512
    with pytest.raises(ValueError):
        a.execution_config(max_lanes=1, prefill_step=MAX_WAVE_ROWS + 1)


def test_wave_memory_advice_is_bounded_and_falls_back_to_chunk():
    w = SimpleNamespace(source_memory_advice={'workspace_bytes': 1 << 30})
    room = lambda rows: WAVE_SAFETY_BYTES + (1 << 30) + rows * WAVE_ROW_BYTES
    assert wave_rows(w, 8192, 1024, 200 << 30) == 8192
    assert wave_rows(w, 3000, 1024, 200 << 30) == 2048
    assert wave_rows(w, 16384 * 4, 1024, 1 << 50) == MAX_WAVE_ROWS
    assert wave_rows(w, 8192, 1024, room(3 * 1024 + 7)) == 3072
    assert wave_rows(w, 8192, 1024, room(1500)) == 1024
    assert w.source_layer_major_advice['fallback'] == 'insufficient_live_memory'
    assert wave_rows(w, 8192, 1024, None) == 1024
    assert w.source_layer_major_advice['fallback'] == 'availability_unknown'
    assert wave_rows(w, 8192, 512, 200 << 30) == 512
    assert w.source_layer_major_advice['fallback'] == 'chunk_reduced_by_memory_advice'


def test_weights_recommendation_keeps_reference_and_widens_only_under_policy(monkeypatch):
    w = object.__new__(SourceV41Weights)
    w.expert_stream = None
    w.source_prefill_order_policy = 'reference'
    monkeypatch.setattr(SourceV41Weights, '_source_available_bytes', lambda self: 200 << 30)
    assert w.recommended_prefill_rows(4096) == 1024 and w.recommended_prefill_rows(300) == 300
    w.source_prefill_order_policy = POLICY
    assert w.recommended_prefill_rows(4096) == 4096 and w.recommended_prefill_rows(700) == 700
    assert w.source_layer_major_chunk_rows(context_tokens=10) == 1024
    monkeypatch.setattr(SourceV41Weights, '_source_available_bytes', lambda self: None)
    assert w.recommended_prefill_rows(4096) == 1024
