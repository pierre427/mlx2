"""Source direct BF16 coefficient geometry and lifetime tests; no GPU oracle."""
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.gguf import source_dense as dense


def array(shape, dtype):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def inputs(kind, rows):
    g, n, k = dense.SHAPES[kind]
    return (array((1, rows, g, k) if kind == 'wo_a' else (1, rows, k), mx.float32),
            array((g*n, k), mx.bfloat16))


@pytest.mark.parametrize('kind', ('wo_a', 'head'))
@pytest.mark.parametrize('rows', (1, 2, 3, 4, 7, 31, 32, 33, 63, 64, 65, 99, 100, 127, 128, 129, 1024))
def test_shapes_and_no_global_coefficient_copy(kind, rows):
    x, weight = inputs(kind, rows)
    p = dense._geometry(x, weight, kind, 'auto', dense.DEFAULT_TILE)
    assert dense.supports(x, weight, kind, check_device=False)
    assert p['global_f32_coefficient_copy_bytes'] == 0
    assert p['coefficient_storage_bytes'] == weight.shape[0]*weight.shape[1]*2
    assert p['output_dtype'] == 'float32' and p['weight_dtype'] == 'bfloat16'
    assert p['grid'][2] == p['groups'] and p['output_shape'][1] == rows


@pytest.mark.parametrize('kind', ('wo_a', 'head'))
def test_gemv_row_local_partition_matches_pinned_law(kind):
    p = dense.plan(kind, 1)
    bm, bn, sm, sn, tm, tn = p['gemv_geometry']
    assert (bn, sm, sn, tm, tn) == (1, 1, 32, 4, 4)
    owners = Counter()
    for group in range(p['grid'][0]//p['threadgroup'][0]):
        for sg in range(bm):
            for row in range(tm):
                owners[group*bm*tm+sg*tm+row] += 1
    assert owners == Counter(range(p['n']))
    ks = Counter()
    for lane in range(sn):
        for base in range(0, p['k'], sn*tn):
            for i in range(tn):
                ks[base+lane*tn+i] += 1
    assert ks == Counter(range(p['k']))


@pytest.mark.parametrize('rows', (1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 99, 100, 129, 1024))
def test_nax_masked_rows_have_exactly_one_owner(rows):
    p = dense.plan('wo_a', rows, algorithm='nax')
    bm, bn, bk, wm, wn = p['tile']
    sm = bm//wm
    counts = Counter()
    for tile in range(p['grid'][1]):
        for row_sg in range(wm):
            start = tile*bm+row_sg*sm
            live = max(0, min(sm, rows-start))
            counts.update(range(start, start+live))
    assert counts == Counter(range(rows))
    assert p['k'] % bk == 0 and p['n'] % bn == 0


@pytest.mark.parametrize('kind,rows,algorithm,tile', [
    ('bad', 1, 'auto', dense.DEFAULT_TILE), ('head', 0, 'auto', dense.DEFAULT_TILE),
    ('head', 1025, 'auto', dense.DEFAULT_TILE), ('head', True, 'auto', dense.DEFAULT_TILE),
    ('head', 2, 'gemv', dense.DEFAULT_TILE), ('wo_a', 1, 'wrong', dense.DEFAULT_TILE),
    ('wo_a', 100, 'nax', (32, 32, 128, 2, 2)),
    ('wo_a', 100, 'nax', (64, 64, 48, 2, 2)),
    ('wo_a', 100, 'nax', (64, 64, 128, 3, 2)),
    ('wo_a', 100, 'nax', [64, 64, 128, 2, 2]),
])
def test_unadmitted_plan_refused(kind, rows, algorithm, tile):
    with pytest.raises(ValueError, match='Source dense'):
        dense.plan(kind, rows, algorithm=algorithm, tile=tile)


@pytest.mark.parametrize('change', ('x_dtype', 'weight_dtype', 'weight_shape', 'x_ndim', 'batch', 'groups', 'width'))
def test_bad_geometry_refuses_before_device_or_dispatch(change, monkeypatch):
    x, w = inputs('wo_a', 100)
    if change == 'x_dtype': x.dtype = mx.bfloat16
    elif change == 'weight_dtype': w.dtype = mx.float32
    elif change == 'weight_shape': w.shape = (8191,4096)
    elif change == 'x_ndim': x.ndim = 3
    elif change == 'batch': x.shape = (2,100,8,4096)
    elif change == 'groups': x.shape = (1,100,7,4096)
    elif change == 'width': x.shape = (1,100,8,4095)
    def forbidden(): raise AssertionError('invalid geometry touched GPU/device')
    monkeypatch.setattr(dense, 'device_supported', forbidden)
    assert not dense.supports(x,w,'wo_a')
    assert dense.maybe_project(x,w,'wo_a') is None


@pytest.mark.parametrize('kind,rows', (('wo_a',1),('head',1),('wo_a',100),('head',100)))
def test_candidate_dispatch_retains_original_weight_object(kind, rows, monkeypatch):
    x,w = inputs(kind,rows)
    seen = [];result = object()
    def kernel(**kwargs):
        seen.append(kwargs)
        return [result]
    monkeypatch.setattr(dense,'device_supported',lambda:True)
    monkeypatch.setattr(dense,'_kernel',lambda *args:kernel)
    assert dense.project(x,w,kind) is result
    assert seen[0]['inputs'][0] is x and seen[0]['inputs'][1] is w
    assert seen[0]['output_dtypes'] == [mx.float32]
    assert seen[0]['output_shapes'] == [dense.plan(kind,rows)['output_shape']]


def test_device_refusal_returns_none_without_allocating(monkeypatch):
    x,w = inputs('head',1)
    monkeypatch.setattr(dense,'device_supported',lambda:False)
    monkeypatch.setattr(dense,'_kernel',lambda *a:pytest.fail('unsupported device compiled kernel'))
    assert dense.maybe_project(x,w,'head') is None
    with pytest.raises(ValueError,match='M5 Max'):
        dense.project(x,w,'head')


@pytest.mark.parametrize('kind', ('wo_a','head'))
def test_generated_gemv_retains_float_intermediates_and_direct_bfloat_loads(kind):
    spec = dense.kernel_spec(kind,1)
    assert spec['input_names'] == ['x','weight']
    assert 'const device bfloat* mat' in spec['header']
    assert 'thread T inter[TN]' in spec['header']
    assert 'result[tm] += inter[tn] * v_coeff[tn]' in spec['header']
    assert 'simd_shuffle_down(result[tm], sn)' in spec['header']
    assert 'load_unsafe(mat, inter' in spec['header']
    assert 'device float* dst=' in spec['source']


def test_nax_uses_float_fragments_with_only_b_pointer_storage_changed():
    spec = dense.kernel_spec('wo_a',100)
    assert 'const device bfloat* B,' in spec['header']
    assert 'NAXTile<T, RB, CB> Btile;' in spec['header']
    assert 'Btile.load(B + B_offset, ldb)' in spec['header']
    assert 'NAXTile<float,TM,TN> result' in spec['source']
    assert 'if(block_row+64<=M)' in spec['source']
    assert 'if(live_m>0) result.store_safe' in spec['source']
    assert 'if(live_m>0) {' not in spec['source']
    assert '#include "mlx/' not in spec['header']
