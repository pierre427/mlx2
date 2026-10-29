"""Pinned F32/MXFP8 projection source/geometry/offline compiler, no GPU."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_fp8_projection_tiles as p


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**kw):raise AssertionError('CPU test attempted Metal execution')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('kind',p.SHAPES)
@pytest.mark.parametrize('sg',[2,4,8])
@pytest.mark.parametrize('results',[4,8])
def test_vector_virtual_stock_ownership_is_disjoint_complete(kind,sg,results):
    d=p.plan(kind,1,simdgroups=sg,results_per_simd=results)
    owned=[]
    for group in range(d['threadgroups'][1]):
        for physical_simd in range(sg):
            virtual_group=group*(sg//2)+physical_simd//2;virtual_sg=physical_simd%2
            first=virtual_group*(2*results)+virtual_sg*results
            owned.extend(range(first,first+results))
    assert owned==list(range(d['n']))
    assert d['threadgroup']==(32,sg,1)
    assert d['k_step']==256 and d['k_steps']*256==d['k']
    assert d['threadgroup_bytes']==0 and d['full_weight_dequantization_bytes']==0


@pytest.mark.parametrize('kind',p.SHAPES)
@pytest.mark.parametrize('tile',p.TILES)
def test_matrix_ownership_and_resource_bounds(kind,tile):
    d=p.plan(kind,99,tile=tile);bm,bn=tile
    assert d['threadgroups']==(d['n']//bn,(99+bm-1)//bm,1)
    assert d['k_step']==8 and d['k_steps']*8==d['k']
    assert d['threadgroup']==(32,2,2)
    assert d['threadgroup_bytes']==(bm+bn)*36*4<32<<10
    assert d['output_shape']==(1,99,d['n'])
    assert d['logical_weight_tile_read_bytes']==d['n']*d['k']*((99+bm-1)//bm)
    assert d['coefficient_repack_bytes']==0


@pytest.mark.parametrize('kind,rows,options',[
    ('bad',1,{}),('q_b',2,{}),('q_b',98,{}),('q_b',True,{}),('q_b',1,{'tile':(64,32)}),
    ('q_b',99,{'simdgroups':4}),('q_b',99,{'results_per_simd':8}),
    ('q_b',1,{'simdgroups':3}),('q_b',1,{'results_per_simd':16}),('q_b',99,{'tile':(64,64)}),
    ('q_b',99,{'fused_boundary':1})])
def test_unsupported_shape_law_refuses(kind,rows,options):
    with pytest.raises(ValueError):p.plan(kind,rows,**options)


def test_raw_fast_helper_is_verbatim_and_fused_changes_only_store():
    from mlx2.runtime.gguf.source_prefill_legacy_tiles import _installed
    _,headers=_installed();text=(headers/'fp_quantized.h').read_text()
    begin=text.index('template <\n    typename T,\n    int group_size,\n    int bits,\n    bool partial_rows = false,')
    end=text.index('\ntemplate <typename T, int group_size, int bits, bool has_global_scale = false>\nMETAL_FUNC void fp_qmv_impl(',begin)
    helper=text[begin:end]
    assert helper in p._header('qmv_fast',False)
    fused=p._header('qmv_fast',True)
    assert helper in fused.replace('y[row] = static_cast<T>(mlx2_source_fp8_store::owner_bf16(result[row]));','y[row] = static_cast<T>(result[row]);')
    assert 'result[row] += qdot<U, values_per_thread, bits>'in helper
    assert 'result[row] = simd_sum(result[row]);'in helper


def test_fused_matrix_changes_only_final_transform_parameter():
    raw=p._header('qmm_t',False);fused=p._header('qmm_t',True)
    changed='BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded, float, mlx2_source_fp8_store::OwnerBF16>;'
    original='BlockMMA<T, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;'
    assert fused.replace(changed,original)==raw
    assert 'for (short kk = 0; kk < BK; kk += kFragSize)'in raw


@pytest.mark.parametrize('kind,rows',[('q_b',1),('wo_b',1),('q_b',99),('wo_b',99)])
def test_geometry_original_codes_and_input_boundaries(kind,rows):
    k,n=p.SHAPES[kind]
    x=SimpleNamespace(ndim=3,shape=(1,rows,k),dtype=mx.float32)
    q=SimpleNamespace(shape=(n,k//4),dtype=mx.uint32);s=SimpleNamespace(shape=(n,k//32),dtype=mx.uint8)
    assert p.geometry(x,q,s,kind)['kind']==kind
    x.dtype=mx.bfloat16
    with pytest.raises(ValueError):p.geometry(x,q,s,kind)
    x.dtype=mx.float32;s.dtype=mx.float16
    with pytest.raises(ValueError):p.geometry(x,q,s,kind)


def test_cpu_gate_prevents_kernel_and_arrays_retain_identity(monkeypatch):
    k,n=p.SHAPES['q_b'];x=SimpleNamespace(ndim=3,shape=(1,1,k),dtype=mx.float32)
    q=SimpleNamespace(shape=(n,k//4),dtype=mx.uint32);s=SimpleNamespace(shape=(n,k//32),dtype=mx.uint8)
    with pytest.raises(ValueError,match='caller-owned'):p.project(x,q,s,'q_b')
    from mlx2.runtime.gguf import source_dense_legacy
    monkeypatch.setattr(source_dense_legacy,'_device_supported',lambda:True)
    seen={};output=object()
    def fn(**kwargs):seen.update(kwargs);return[output]
    monkeypatch.setattr(p,'_kernel',lambda *args:fn)
    assert p.project(x,q,s,'q_b',simdgroups=4)is output
    assert all(a is b for a,b in zip(seen['inputs'],(x,q,s)))
    assert seen['output_dtypes']==[mx.float32]


CASES=[(kind,1,dict(simdgroups=sg,results_per_simd=r,fused_boundary=fused))
    for kind in p.SHAPES for sg,r in ((2,4),(4,4),(8,4),(2,8),(4,8),(8,8))for fused in (False,True)]
CASES += [(kind,99,dict(tile=tile,fused_boundary=fused))for kind in p.SHAPES for tile in p.TILES for fused in (False,True)]


@pytest.mark.parametrize('kind,rows,options',CASES)
def test_offline_only_compile_actual_dimensions(kind,rows,options,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    file=tmp_path/'probe.metal';file.write_text(p.compilation_source(kind,rows,**options))
    include=Path(mx.__file__).parent/'include'
    run=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.2','-I',str(include),'-c',str(file),'-o',str(tmp_path/'probe.air')],capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stdout+run.stderr
