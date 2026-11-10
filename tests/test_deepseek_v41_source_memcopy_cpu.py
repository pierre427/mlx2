"""CPU memory-copy contracts for the DS4.1 Source dense fallback.

deepseek_source_bf16_dense=resident_f32_v1 must hand the generic F32 matmul
the same operand values the reference widens per call (bit-identical
products), widen each coefficient once (no per-call 7.5 MiB router expansion),
and never reuse a widened copy of a retired or foreign coefficient.
"""
import numpy as np
import pytest
import mlx.core as mx

from mlx2.adapters.deepseek_v41_source import SOURCE_POLICY_CHOICES, source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights

ROUTER = 'blk.7.ffn_gate_inp.weight'


def _weights(policy, dense):
    w = SourceV41Weights.__new__(SourceV41Weights)
    w.source_bf16_loads_policy = 'direct_f32'
    w.source_bf16_dense_policy = policy
    w._quant = {}
    w._dense = dict(dense)
    w.scheduled_dense_calls = {}
    w._ensure_dense = lambda: None
    w._ready = lambda: None
    return w


def _bf16(rng, shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)


def _bits(value):
    mx.eval(value)
    return np.asarray(value).view(np.uint32)


def test_policy_is_explicit_and_reference_by_default():
    assert SOURCE_POLICY_CHOICES['deepseek_source_bf16_dense'] == ('reference', 'resident_f32_v1')
    assert 'deepseek_source_bf16_dense' not in source_policy()
    assert source_policy({'deepseek_source_bf16_dense': 'resident_f32_v1'})['deepseek_source_bf16_dense'] == 'resident_f32_v1'
    with pytest.raises(ValueError):
        source_policy({'deepseek_source_bf16_dense': 'f32'})


@pytest.mark.parametrize('rows', [1, 3, 64])
def test_router_products_bit_identical_and_widened_once(rows):
    rng = np.random.default_rng(rows)
    # Real Source router geometry: BF16 [384, 5120].
    weight = _bf16(rng, (384, 5120))
    mx.eval(weight)
    with mx.stream(mx.cpu):
        reference = _weights('reference', {ROUTER: weight})
        resident = _weights('resident_f32_v1', {ROUTER: weight})
        for step in range(3):
            x = mx.array(rng.standard_normal((1, rows, 5120)).astype(np.float32))
            expected = reference.linear(ROUTER)(x)
            actual = resident.linear(ROUTER)(x)
            np.testing.assert_array_equal(_bits(actual), _bits(expected))
    counts = resident.scheduled_dense_calls
    assert counts['source_bf16_dense_f32_widened'] == 1
    assert counts['source_bf16_dense_f32_reused'] == 2
    assert counts['source_bf16_dense_f32_bytes_avoided'] == 2 * 384 * 5120 * 4
    assert 'source_bf16_dense_f32_widened' not in reference.scheduled_dense_calls


def test_resident_operand_is_one_buffer_and_removes_the_per_call_allocation():
    rng = np.random.default_rng(0)
    weight = _bf16(rng, (384, 5120))
    x = mx.array(rng.standard_normal((1, 1, 5120)).astype(np.float32))
    mx.eval(weight, x)
    with mx.stream(mx.cpu):
        w = _weights('resident_f32_v1', {ROUTER: weight})
        first = w._f32_operand(ROUTER)
        mx.eval(first)
        address = np.asarray(first).ctypes.data
        second = w._f32_operand(ROUTER)
        assert second is first and np.asarray(second).ctypes.data == address
        # A call after warm-up allocates only its [1, 1, 384] output.
        mx.eval(w.linear(ROUTER)(x))
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(w.linear(ROUTER)(x))
        resident_peak = mx.get_peak_memory() - base
        reference = _weights('reference', {ROUTER: weight})
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        mx.eval(reference.linear(ROUTER)(x))
        reference_peak = mx.get_peak_memory() - base
    assert reference_peak >= 384 * 5120 * 4
    assert resident_peak < 64 * 1024


def test_reloaded_coefficient_is_widened_again_never_stale():
    rng = np.random.default_rng(1)
    old, new = _bf16(rng, (32, 64)), _bf16(rng, (32, 64))
    name = 'blk.20.attn_compressor_kv.weight'
    x = mx.array(rng.standard_normal((1, 1, 64)).astype(np.float32))
    with mx.stream(mx.cpu):
        w = _weights('resident_f32_v1', {name: old})
        mx.eval(w.linear(name)(x))
        w._dense[name] = new  # park/activate reloads new array objects
        actual = w.linear(name)(x)
        expected = x @ new.astype(mx.float32).T
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert w.scheduled_dense_calls['source_bf16_dense_f32_widened'] == 2


def test_head_f32_and_unlisted_coefficients_keep_reference_widening():
    rng = np.random.default_rng(2)
    head = _bf16(rng, (16, 8))
    f32 = mx.array(rng.standard_normal((16, 8)).astype(np.float32))
    with mx.stream(mx.cpu):
        w = _weights('resident_f32_v1', {'output.weight': head, 'blk.2.attn_compressor_gate.weight': f32})
        assert w._f32_operand('output.weight') is not w._f32_operand('output.weight')
        # F32 coefficients are already the operand: astype is a no-op view.
        assert w._f32_operand('blk.2.attn_compressor_gate.weight').dtype == mx.float32
    assert w.__dict__.get('_dense_f32', {}) == {}
    assert 'source_bf16_dense_f32_widened' not in w.scheduled_dense_calls
