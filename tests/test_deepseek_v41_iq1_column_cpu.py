"""CPU value-law and offline compile gates for unselected column IQ1 GEMV."""

import inspect
import shutil
import subprocess

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import v41_iq1_column as column
from mlx2.runtime.gguf import v41_iq1_splash, v41_iq1_tiled as tiled
from mlx2.runtime.gguf.iq1_codebook import IQ1_GRID
from mlx2.runtime.gguf.reader import GGUFError

mx.set_default_device(mx.cpu)


def test_column_float4_decode_preserves_every_iq1_grid_lane_order():
    words = np.array(v41_iq1_splash.NIBBLE_GRID, np.uint32)
    lo = ((words[:, None] >> np.array([0, 8, 16, 24], np.uint32)) & 15).astype(np.int32)-1
    hi = ((words[:, None] >> np.array([4, 12, 20, 28], np.uint32)) & 15).astype(np.int32)-1
    actual = np.concatenate([lo, hi], axis=1)
    expected = np.array(IQ1_GRID, '<u8').view(np.int8).reshape(2048, 8)
    np.testing.assert_array_equal(actual, expected)
    # Four independently selected grids become contiguous chunks0..7, with
    # low and high halves interleaved exactly as the Metal float4[8] vector.
    indices = (np.arange(2048)[:, None]*4+np.arange(4)) % 2048
    vector = actual[indices].reshape(2048, 8, 4)
    np.testing.assert_array_equal(vector.reshape(2048, 32), expected[indices].reshape(2048, 32))


def test_f32_vector_values_never_round_weights_to_fp16():
    half = np.arange(65536, dtype=np.uint16).view(np.float16)
    d = half[np.isfinite(half)].astype(np.float32)
    odd = np.arange(1, 16, 2, dtype=np.float32)
    value = np.array([-1, 0, 1], np.float32)
    saw_half_loss = False
    for delta in (-.125, .125):
        scalar = (d[:, None, None]*odd[None, :, None])*(value[None, None, :]+np.float32(delta))
        # Vector decoder computes F32scale before broadcasting multiply, using
        # the same binary-rational delta and untouched half scale conversion.
        scale = d[:, None]*odd[None, :]
        vector = scale[:, :, None]*(value+np.float32(delta))
        np.testing.assert_array_equal(vector.view(np.uint32), scalar.view(np.uint32))
        with np.errstate(over='ignore'):
            saw_half_loss |= bool(np.any(vector.astype(np.float16).astype(np.float32) != scalar))
    assert saw_half_loss
    assert 'thread float4 (&v)[8]' in column.HEADER
    assert 'scale*(float4(lo)+delta)' in column.HEADER
    assert 'half4' not in column.HEADER


@pytest.mark.parametrize('width', [256, 2304, 5120])
def test_prefetch_and_half_d_hoist_visit_original_groups_and_blocks(width):
    group_address, scale_address = 0, 0
    for group in range(width//32):
        assert group_address == group*64
        assert scale_address == (group//8)*64
        # Decode uses current payload and current blockscale before loading
        # the next group; d changes only at the next256-value boundary.
        if group+1 < width//32:
            group_address = (group+1)*64
            if (group+1) % 8 == 0:
                scale_address = ((group+1)//8)*64
    for kind in ('gate', 'down'):
        _, source = column.kernel_spec(kind)
        assert source.index('v41_column_decode(') < source.index('if(group+1<groups)')
        assert 'if((group+1)%8==0)' in source


@pytest.mark.parametrize('kind', ['gate', 'down'])
def test_guard_geometry_numerical_stores_and_distinct_reduction_are_explicit(kind):
    _, source = column.kernel_spec(kind)
    assert 'row=thread_position_in_grid.x' in source
    assert source.index('row>=N') < source.index('ids[first]')
    assert source.index('expert>=EXPERTS') < source.index('slot_map[expert]')
    assert source.index('slot>=capacity') < source.index('v41_column_payload(')
    assert 'simd_sum' not in source
    assert 'for(uint group=0;group<groups;group++)' in source
    assert 'for(uint i=0;i<8;i++)' in source
    assert '#pragma clang fp reassociate(off)' in column.HEADER
    if kind == 'gate':
        assert 'gsum=gsum+sg; usum=usum+su;' in source
        assert 'gsum=min(v41_bf16(gsum),10.0f); usum=clamp(v41_bf16(usum),-10.0f,10.0f);' in source
        assert 'v41_bf16((gsum/(1.0f+exp(-gsum)))*usum*scores[first])' in source
    else:
        assert 'total=total+sum;' in source and 'v41_bf16(total)' in source
    assert column.REVISION != tiled.REVISION
    assert column.NUMERIC_PROFILE == 'native_iq1_s_tiled_column_f32'


@pytest.mark.parametrize('rows', [1, 2, 16])
def test_column_dispatch_keeps_bank_and_gpu_routes_without_host_sync(monkeypatch, rows):
    x = mx.zeros((rows, 256), mx.float32)
    ids = mx.zeros((rows, 6), mx.uint32)
    scores = mx.full(ids.shape, 1/6, mx.float32)
    slots = mx.array([1, tiled.MISSING_SLOT, 0], mx.uint32)
    planes = tuple(mx.zeros((size,), mx.uint8) for size in tiled.bank_plane_sizes(2, 256, 256))
    calls = []
    def kernel(kind):
        def call(**kw):
            calls.append((kind, kw))
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw['output_shapes'], kw['output_dtypes'])]
        return call
    def forbidden(*args, **kwargs):
        raise AssertionError('hot column path attempted host conversion/synchronization')
    monkeypatch.setattr(column, 'kernel', kernel)
    monkeypatch.setattr(tiled, 'kernel', kernel)
    monkeypatch.setattr(mx, 'eval', forbidden)
    monkeypatch.setattr(mx, 'synchronize', forbidden)
    monkeypatch.setattr(np, 'array', forbidden)
    monkeypatch.setattr(np, 'asarray', forbidden)
    counters = {}
    out = column.moe(x, ids, scores, planes, slots, capacity=2,
                     hidden_size=256, intermediate_size=256, counters=counters)
    assert out.shape == x.shape and counters == {'column_f32_calls': 1}
    assert [kind for kind, _ in calls] == ['gate', 'down', 'exact_combine']
    for kind, kw in calls[:2]:
        assert kw['inputs'][1] is ids and kw['inputs'][2] is slots
        assert kw['grid'] == (256, rows*6, 1) and kw['threadgroup'] == (128, 1, 1)
        assert dict(kw['template'])['EXPERTS'] == 3
    assert all(a is b for a, b in zip(calls[0][1]['inputs'][4:], planes[:6]))
    assert all(a is b for a, b in zip(calls[1][1]['inputs'][3:], planes[6:]))
    source = inspect.getsource(column.moe)
    assert 'np.' not in source and 'mx.eval' not in source


def test_empty_bank_and_unsupported_prefill_fail_safely(monkeypatch):
    ids = mx.zeros((1, 1), mx.uint32)
    x, scores = mx.zeros((1, 256)), mx.ones((1, 1))
    slots = mx.array([tiled.MISSING_SLOT], mx.uint32)
    planes = tuple(mx.array([], mx.uint8) for _ in range(9))
    def forbidden(*args, **kwargs):
        raise AssertionError('empty bank bound a kernel')
    monkeypatch.setattr(column, 'kernel', forbidden)
    out = column.moe(x, ids, scores, planes, slots, capacity=0, hidden_size=256, intermediate_size=256)
    np.testing.assert_array_equal(np.array(out), 0)
    np.testing.assert_array_equal(np.array(column.missing_routes(ids, slots, 0)), True)
    with pytest.raises(GGUFError, match='selected prefill'):
        column.moe(x, ids, scores, planes, slots, capacity=0, hidden_size=256, intermediate_size=256, prefill=True)
    with pytest.raises(GGUFError, match='geometry'):
        column.moe(x, ids, scores, planes[:-1], slots, capacity=0, hidden_size=256, intermediate_size=256)


def test_column_capacity_growth_uses_physical_runtime_shapes(monkeypatch):
    ids = mx.array([[0, 1, 2, 3, 384, 0xffffffff]], mx.uint32)
    x, scores = mx.zeros((1, 256)), mx.ones(ids.shape)
    templates = {}
    def kernel(kind):
        def call(**kw):
            current = tuple(kw['template'])
            assert 'CAPACITY' not in dict(current)
            if kind in templates:
                assert current == templates[kind]
            templates[kind] = current
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw['output_shapes'], kw['output_dtypes'])]
        return call
    monkeypatch.setattr(column, 'kernel', kernel)
    monkeypatch.setattr(tiled, 'kernel', kernel)
    for capacity in (1, 2, 383, 384):
        slots = mx.array([0, capacity-1, capacity, tiled.MISSING_SLOT]+[tiled.MISSING_SLOT]*380, mx.uint32)
        planes = tuple(mx.zeros((n,), mx.uint8) for n in tiled.bank_plane_sizes(capacity, 256, 256))
        column.moe(x, ids, scores, planes, slots, capacity=capacity,
                   hidden_size=256, intermediate_size=256)
    for kind, plane in (('gate', 'gqs'), ('down', 'wqs')):
        source = column.kernel_spec(kind)[1]
        assert f'{plane}_shape[0]/(ulong(N)*K/8)' in source
        assert 'CAPACITY' not in source
    assert set(templates) == {'gate', 'down', 'exact_combine'}


@pytest.mark.parametrize('kind', ['gate', 'down'])
@pytest.mark.parametrize('constant_inputs', [False, True])
def test_offline_column_metal_compiles_without_gpu(tmp_path, kind, constant_inputs):
    if not shutil.which('xcrun') or subprocess.run(['xcrun', '--find', 'metal'], capture_output=True).returncode:
        pytest.skip('offline Metal compiler unavailable')
    source = tmp_path/(kind+'.metal')
    source.write_text(column.metal_compile_source(kind, constant_inputs=constant_inputs))
    result = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.0', '-c', str(source),
                             '-o', str(source.with_suffix('.air'))], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
