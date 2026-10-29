"""Exhaustive decoder algebra and CPU-only dispatch/offline compiler checks."""

import hashlib
import shutil
import subprocess

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import v41_iq1_splash as candidate
from mlx2.runtime.gguf import v41_row_exact as base
from mlx2.runtime.gguf.iq1_codebook import IQ1_GRID
from mlx2.runtime.gguf.packed_metal import PackedMatrix
from mlx2.runtime.gguf.reader import GGUFError, GGUFTensor

mx.set_default_device(mx.cpu)


def test_every_grid_lane_matches_original_and_pinned_splash_grid():
    old = np.array(IQ1_GRID, dtype='<u8').view(np.int8).reshape(2048, 8).astype(np.int32)
    new = np.array(candidate.NIBBLE_GRID, dtype='<u4')
    # Generated from pinned Splash QuantTables.h kIQ1SGrid, not a network read.
    assert hashlib.sha256(new.tobytes()).hexdigest() == '29bb0f4450fab4aa61d9fa9bc62a06b4a129acf31ebf38266adb6a62ef424117'
    shifts = np.array([(lane % 4)*8+(lane//4)*4 for lane in range(8)], np.uint32)
    actual = ((new[:, None] >> shifts) & 15).astype(np.int32)-1
    np.testing.assert_array_equal(actual, old)
    assert new.nbytes == 8192 and 'constant ulong' not in candidate.HEADER


def test_all_finite_half_scales_odd_factors_values_and_deltas_preserve_f32_bits():
    scales = np.arange(65536, dtype=np.uint16).view(np.float16)
    scales = scales[np.isfinite(scales)].astype(np.float32)
    factors = np.arange(1, 16, 2, dtype=np.float32)[None, :, None]
    values = np.array([-1, 0, 1], np.float32)[None, None, :]
    for delta in (-.125, .125):
        expected = (scales[:, None, None]*factors)*(values+np.float32(delta))
        # The generated nibble representation changes only integer decoding.
        codes = np.array([0, 1, 2], np.uint32)
        decoded = (codes.astype(np.int32)-1).astype(np.float32)[None, None, :]
        actual = (scales[:, None, None]*factors)*(decoded+np.float32(delta))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize('kind', ['gate', 'down'])
@pytest.mark.parametrize('variant', candidate.VARIANTS)
def test_source_preserves_original_accumulator_and_epilogue(kind, variant):
    names, source = candidate.kernel_spec(kind, 10, variant)
    old_names, old = base.kernel_spec(kind, 10, False)
    assert names == old_names
    # The final SIMD reduction, BF16 stores, clamp and route-score placement
    # are exactly the original source, not an independently similar formula.
    marker = 'for(uint r=0;r<ROWS;r++) {\n    float '
    assert source.split(marker, 1)[1] == old.split(marker, 1)[1]
    body = ('sg[r]+=a*wg; su[r]+=a*wu;' if kind == 'gate'
            else 'sums[r]+=x[size_t(first+r)*K+k]*weight;')
    assert body in source and body in old
    assert 'simd_sum' in source and 'gguf_value(' not in source
    if variant == 'nibble':
        assert 'for(uint k=lane;k<K;k+=32)' in source
    else:
        assert 'const uint k=block*256+group*32+lane;' in source
        assert source.index('gguf_half(p') < source.index('for(uint group=0;group<8;group++)')
        assert source.index('if(group+1<8)') < source.index(body)
        # New loop visits exactly each original lane's k sequence, not a
        # reordered group reduction or a separate partial sum.
        for width in (256, 2304, 5120):
            for lane in range(32):
                sequence = [block*256+group*32+lane for block in range(width//256) for group in range(8)]
                assert sequence == list(range(lane, width, 32))


@pytest.mark.parametrize('rows', [1, 2, 16, 1024])
@pytest.mark.parametrize('variant', candidate.VARIANTS)
def test_dispatch_retains_separate_weights_route_order_and_default_geometry(monkeypatch, rows, variant):
    planes = []
    for kind in ('gate', 'up', 'down'):
        tensor = GGUFTensor(kind, (256, 256, 1), 19, 0, 256*50)
        planes.append(tuple(PackedMatrix(mx.zeros((tensor.nbytes,), mx.uint8), tensor) for _ in range(11)))
    x = mx.zeros((rows, 256), mx.float32)
    ids = np.arange(rows*6, dtype=np.uint32).reshape(rows, 6) % 11
    scores = mx.ones(ids.shape, mx.float32)
    calls, orders = [], []
    def kernel(kind, count, actual_variant):
        assert actual_variant == variant
        def call(**kw):
            values = dict(kw['template'])
            assert values['ROWS'] == (1 if rows == 1 else 4)
            assert values['SIMDS'] == 4
            payload = kw['inputs'][4:] if kind == 'gate' else kw['inputs'][2:]
            assert len(payload) == count*(2 if kind == 'gate' else 1)
            assert all(v.ndim == 1 and v.dtype == mx.uint8 for v in payload)
            if kind == 'gate':
                orders.extend(np.array(kw['inputs'][1]).tolist())
            calls.append((kind, count))
            return [mx.zeros(kw['output_shapes'][0], mx.float32)]
        return call
    def combine(**kw):
        inverse = np.array(kw['inputs'][1])
        np.testing.assert_array_equal(np.array(orders)[inverse], np.arange(ids.size))
        return [mx.zeros(kw['output_shapes'][0], mx.float32)]
    monkeypatch.setattr(candidate, 'kernel', kernel)
    monkeypatch.setattr(base, 'combine_kernel', lambda: combine)
    monkeypatch.setattr(candidate, 'self_check', lambda *args: None)
    result = candidate.moe(x, ids, scores, *planes, variant=variant)
    assert result.shape == x.shape
    assert calls and all(count <= 10 for _, count in calls)


def test_unsupported_variant_broadcast_and_cpu_first_use_fail_closed():
    with pytest.raises(GGUFError, match='unsupported'):
        candidate.kernel_spec('gate', 6, 'other')
    for kw in ({'variant': 'other'}, {'broadcast': True}):
        with pytest.raises(GGUFError, match='scalar decoder'):
            candidate.moe(None, None, None, (), (), (), **kw)
    with pytest.raises(GGUFError, match='owned GPU'):
        candidate.self_check('nibble')


@pytest.mark.parametrize('kind', ['gate', 'down'])
@pytest.mark.parametrize('variant', candidate.VARIANTS)
@pytest.mark.parametrize('reuse,count', [(1, 6), (4, 10)])
def test_offline_metal_compiles_both_variants_and_binding_geometries(tmp_path, kind, variant, reuse, count):
    if not shutil.which('xcrun') or subprocess.run(['xcrun', '--find', 'metal'], capture_output=True).returncode:
        pytest.skip('offline Metal compiler unavailable')
    source = tmp_path/'candidate.metal'
    source.write_text(candidate.metal_compile_source(kind, count, variant, reuse=reuse))
    result = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-std=metal3.2', '-c', str(source),
                             '-o', str(tmp_path/'candidate.air')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
