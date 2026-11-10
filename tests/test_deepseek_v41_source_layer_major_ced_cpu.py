"""Synthetic CPU contracts for exact CED inside Source layer-major waves.

No checkpoint, GPU or timing. The reference is chunk-major dependency CED
driven exactly as the prompt loop drives it (one lookahead scope per chunk,
step = the wave's sub-chunk size).
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_layer_major import (
    DEFAULT_WAVE_ROWS, KEY, POLICY, layer_major_model_class, receipt)
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.prefill_lookahead import StateOnlyPrefillOutput, prefill_lookahead


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)

    def forbidden(*args, **kwargs):
        raise AssertionError('synthetic CPU test attempted Metal')
    monkeypatch.setattr(mx.fast, 'metal_kernel', forbidden)
    yield
    mx.set_default_device(previous)


def _config(**kwargs):
    # Barrier 3 (last KV source), decoder layers 4..7, window 3: the first
    # decoder layer's cone is 3 + 3 * 2 = 9 rows.
    values = dict(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
                  kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7), sliding_window=3)
    values.update(kwargs)
    return config(**values)


def _bf16(**kwargs):
    return _config(hidden_size=32, head_dim=32, q_lora_rank=32, qk_rope_head_dim=16, index_head_dim=32,
                   o_lora_rank=16, moe_intermediate_size=32, simulate_quantization=True,
                   activation_dtype='bfloat16', **kwargs)


def _pair(c, *, chunk, state_only=True):
    def weights():
        w = Weights(c)
        w.source_prefill_order_policy = POLICY
        return w
    options = dict(ced_policy='dependency_exact', ced_state_only=state_only)
    wave = layer_major_model_class()(c, weights(), identity='synthetic-lm-ced@1', **options)
    wave.source_layer_major_rows = chunk
    ref = DeepSeekV41Model(c, weights(), identity='synthetic-lm-ced@1', **options)
    return wave, ref


def _prompt(model, cache, ids, step, *, after=0):
    """The prompt loop: one lookahead scope per plain chunk of one call."""
    outputs = []
    for first in range(0, ids.shape[1], step):
        chunk = ids[:, first:first + step]
        with prefill_lookahead([after + ids.shape[1] - first - chunk.shape[1]], step=step):
            outputs.append(model.prefill_forward(chunk, cache))
    return outputs


def _bits(x):
    return np.asarray(x).view(np.uint32)


def _assert_states_equal(a, b):
    for x, y in zip(a, b):
        assert x.offset == y.offset and x._pool_count == y._pool_count
        assert x.ced_local_deferred == y.ced_local_deferred
        for p, q in zip(x.state, y.state):
            assert (p is None) == (q is None)
            if p is not None:
                np.testing.assert_array_equal(np.asarray(p).view(np.uint8), np.asarray(q).view(np.uint8))


def _ids(c, rows, salt=3):
    return mx.array([[(7 * i + salt) % c.vocab_size for i in range(rows)]], mx.int32)


@pytest.mark.parametrize('cfg', [_config, _bf16])
@pytest.mark.parametrize('rows,chunk', [(40, 4), (37, 5), (23, 3), (12, 4), (17, 6)])
@pytest.mark.parametrize('state_only', [True, False])
def test_ced_wave_is_bitwise_chunk_major_ced(cfg, rows, chunk, state_only):
    c = cfg()
    wave, ref = _pair(c, chunk=chunk, state_only=state_only)
    ids = _ids(c, rows)
    wc, rc = wave.make_cache(), ref.make_cache()
    # One serving-style slice: nothing follows the wave in this call.
    with prefill_lookahead([0], step=rows):
        actual = wave.prefill_forward(ids, wc)
    expected = _prompt(ref, rc, ids, chunk)
    assert not isinstance(actual, StateOnlyPrefillOutput)
    np.testing.assert_array_equal(_bits(actual), _bits(expected[-1]))
    _assert_states_equal(wc, rc)
    assert not any(s.pending for s in wc)
    stats, rstats = wave.scheduled_kv_calls, ref.scheduled_kv_calls
    for key in ('ced_state_only_chunks', 'ced_prefill_calls', 'ced_skipped_layer_rows'):
        assert stats.get(key, 0) == rstats.get(key, 0), key
    in_wave = sum(isinstance(o, StateOnlyPrefillOutput) for o in expected)
    assert stats.get('source_layer_major_ced_state_only_chunks', 0) == in_wave
    assert in_wave == (sum(rows - min(rows, s + chunk) >= 9 for s in range(0, rows, chunk))
                       if state_only and chunk >= 3 else 0)
    assert stats['source_layer_major_ced_waves'] == 1 and stats['source_layer_major_chunks'] == -(-rows // chunk)
    token = mx.array([[1]], mx.int32)
    np.testing.assert_array_equal(_bits(wave(token, wc)), _bits(ref(token, rc)))


def test_in_wave_state_only_engages_without_outer_lookahead():
    """Chunk-major state-only needs rows after the *call*; a wave supplies them."""
    c = _config()
    wave, ref = _pair(c, chunk=4)
    ids = _ids(c, 24)
    actual = wave.prefill_forward(ids, wave.make_cache())  # no lookahead scope at all
    assert not isinstance(actual, StateOnlyPrefillOutput)
    assert wave.scheduled_kv_calls['source_layer_major_ced_state_only_chunks'] == 3  # ends 4, 8, 12
    rc = ref.make_cache()
    for first in range(0, 24, 4):  # chunk-major serving slices: no lookahead
        out = ref.prefill_forward(ids[:, first:first + 4], rc)
    assert ref.scheduled_kv_calls.get('ced_state_only_chunks', 0) == 0
    # Same logits up to row-shape rounding; the CED identity is not row-exact.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(out), rtol=1e-5, atol=1e-6)


def test_two_waves_of_one_call_match_chunk_major_and_whole_wave_can_be_state_only():
    c = _config()
    wave, ref = _pair(c, chunk=4)
    ids = _ids(c, 32, salt=5)
    wc, rc = wave.make_cache(), ref.make_cache()
    outputs = []
    for first in (0, 16):  # prompt loop with step 16 = two waves of one call
        with prefill_lookahead([32 - first - 16], step=16):
            outputs.append(wave.prefill_forward(ids[:, first:first + 16], wc))
    assert isinstance(outputs[0], StateOnlyPrefillOutput)  # 16 rows follow: whole wave parks
    assert any(s.ced_local_deferred for s in wc[4:]) is False  # refilled by wave 2
    expected = _prompt(ref, rc, ids, 4)
    np.testing.assert_array_equal(_bits(outputs[-1]), _bits(expected[-1]))
    _assert_states_equal(wc, rc)


def test_plain_call_wave_under_ced_model_stays_full_depth_reference_wave():
    c = _config()
    wave, ref = _pair(c, chunk=4)
    ids = _ids(c, 10)
    wc, rc = wave.make_cache(), ref.make_cache()
    actual = wave(ids, wc)
    expected = mx.concatenate([ref(ids[:, s:s + 4], rc) for s in range(0, 10, 4)], 1)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert 'source_layer_major_ced_waves' not in wave.scheduled_kv_calls


def test_engram_above_barrier_and_bad_options_refuse_before_mutation():
    from dataclasses import replace
    c = _config()
    wave, _ = _pair(c, chunk=4)
    cache = wave.make_cache()
    # Planning reads only the config's Engram placement; it refuses first.
    wave.args = replace(c, engram_layer_ids=(5,), engram_num_embeddings=(7,))
    with pytest.raises(ValueError, match='Engram layers at or below'):
        wave._forward(_ids(c, 10), cache, logits_to_keep=1, dependency_prefill=True)
    assert all(s.offset == 0 and not s._in_forward for s in cache)
    wave, _ = _pair(c, chunk=4)
    with pytest.raises(ValueError, match='final-row'):
        wave._forward(_ids(c, 10), wave.make_cache(), logits_to_keep=0, dependency_prefill=True)
    wave.ced_policy = 'reference'
    with pytest.raises(ValueError, match='dependency_exact'):
        wave._forward(_ids(c, 10), wave.make_cache(), logits_to_keep=1, dependency_prefill=True)


def test_policy_admits_only_dependency_ced_and_salts_identity():
    parsed = source_policy({KEY: POLICY, 'deepseek_ced': 'dependency_exact',
                            'deepseek_ced_state_only': 'after_cone_v1'})
    assert parsed['deepseek_ced'] == 'dependency_exact' and parsed[KEY] == POLICY
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.weights = SimpleNamespace()
    a.numerics_revision = a.profile = a.numeric_profile = 'base'
    a.kernel_policy = {KEY: POLICY}
    a._bind_prefill_order_identity()
    plain = a.numerics_revision
    a.numerics_revision = a.profile = a.numeric_profile = 'base'
    a.kernel_policy = {KEY: POLICY, 'deepseek_ced': 'dependency_exact', 'deepseek_ced_state_only': 'after_cone_v1'}
    a._bind_prefill_order_identity()
    assert a.numerics_revision != plain and ':ced=dependency_exact:ced_state_only=after_cone_v1' in a.numerics_revision
    value = receipt(POLICY, ced='dependency_exact', counters={'source_layer_major_ced_waves': 2, 'other': 1})
    assert value['ced'] == 'dependency_exact' and 'implemented_untested_unqualified' in value['ced_composition']
    assert value['counters'] == {'source_layer_major_ced_waves': 2}
    assert receipt(POLICY)['ced_composition'] == 'off'


def test_prefill_step_default_only_under_layer_major():
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.kernel_policy = {}
    from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter
    assert a.prefill_step_default() == DeepSeekV4GGUFAdapter.prefill_step_default(a) == 512
    a.kernel_policy = {KEY: 'reference'}
    assert a.prefill_step_default() == 512
    a.kernel_policy = {KEY: POLICY}
    assert a.prefill_step_default() == DEFAULT_WAVE_ROWS == 16384
