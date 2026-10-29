"""Experimental Source split-K geometry/source/Metal compilation; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import numpy as np
import mlx.core as mx
import pytest
from mlx2.runtime.gguf import source_prefill_nax as nax
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*args,**kwargs):raise AssertionError('CPU component test reached GPU')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('rows',[13,16,31,32,33,64,95,96])
def test_split_partitions_cover_physical_k_without_rounding(rows):
    p=nax.plan(rows)
    assert p['partitions']==(5 if rows<=32 else 2)
    ranges=[range(s,s+p['k_partition']) for s in p['partition_starts']]
    assert [v for r in ranges for v in r]==list(range(5120))
    assert p['k_partition']%64==0
    assert p['partial_shape']==(p['partitions'],rows,2304)
    assert p['partial_bytes']==p['partitions']*rows*2304*4
    assert p['partial_bytes']<=2_000_000
    assert p['threadgroup_bytes']==9216 and p['full_weight_repack_bytes']==0
    assert p['input_bindings']==4


@pytest.mark.parametrize('rows',[0,1,12,97,1024,True,13.0,'13'])
def test_outside_verified_legacy_split_domain_refused(rows):
    with pytest.raises(ValueError):nax.plan(rows)


def packed(kind='original'):
    layout=canonical_layout(kind);types={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    fields={t.name:SimpleNamespace(shape=t.shape,dtype=types[t.dtype]) for t in layout.tensors}
    backing=SimpleNamespace(shape=(layout.nbytes,),ndim=1,dtype=mx.uint8)
    return PackedExpert(backing,layout,fields)


@pytest.mark.parametrize('projection',['gate','up'])
def test_one_backing_and_projection_offsets_are_retained(projection):
    e=packed();x=SimpleNamespace(ndim=2,shape=(99,5120),dtype=mx.float32)
    rows=SimpleNamespace(ndim=1,shape=(33,),dtype=mx.uint32)
    p,offsets=nax.prepare_inputs(x,e,rows,projection)
    assert offsets==tuple(e.layout.tensor(projection+'.'+s).offset for s in ('qweight','scales'))
    assert p['partitions']==2


@pytest.mark.parametrize('change',['kind','projection','empty','width','dtype','rowsdtype','rowsndim','m12','m97','backing','typed'])
def test_input_contract_refusals_before_gpu(change):
    e=packed();x=SimpleNamespace(ndim=2,shape=(99,5120),dtype=mx.float32)
    rows=SimpleNamespace(ndim=1,shape=(33,),dtype=mx.uint32);projection='gate'
    if change=='kind':e=packed('q1')
    elif change=='projection':projection='down'
    elif change=='empty':x.shape=(0,5120)
    elif change=='width':x.shape=(99,5121)
    elif change=='dtype':x.dtype=mx.bfloat16
    elif change=='rowsdtype':rows.dtype=mx.int32
    elif change=='rowsndim':rows.ndim=2
    elif change=='m12':rows.shape=(12,)
    elif change=='m97':rows.shape=(97,)
    elif change=='backing':e.backing.shape=(1,)
    else:e['gate.scales'].dtype=mx.float32
    with pytest.raises(ValueError):nax.prepare_inputs(x,e,rows,projection)


def test_source_keeps_physical_strides_but_limits_reduction_extent():
    s=nax.kernel_spec(33);h=s['header']
    assert 'const int K_it = k_count / G::kBK;' in h
    assert 'q.advance(w_row * K_g + k_begin / Q::kGroup);' in h
    assert 'w + w_row * K_w + k_begin * Q::kBits / 8' in h
    assert 'rmap[row_start + r] * uint(K)' in h
    assert 'const int K_w = K * Q::kBits / 8;' in h
    assert 'const int K_g = K / Q::kGroup;' in h
    assert 'x += k_begin;' in h
    assert 'const device float* xn' in h and 'device float* y' in h
    assert 'A.frag_at(r, kk)[h * 4 + c] = T(v[c]);' in h
    assert 'NAXTile<float, kTM, kTN> Dtile;' in h
    assert 'partials+part*33*2304' in s['source']
    assert '2304,5120,part*2560,2560' in s['source']
    assert s['output_names']==['partials']


def test_boundary_rounds_only_after_f32_partition_sum(monkeypatch):
    # A BF16 midpoint partial must survive until reduction. Rounding each
    # partial first would yield zero, while the correct reduced result is2^-8.
    e=packed();x=mx.zeros((13,5120),mx.float32);rows=mx.arange(13,dtype=mx.uint32)
    values=np.zeros((5,13,2304),np.float32)
    values[0]=np.float32(1+2**-8);values[1]=np.float32(-1)
    partials=mx.array(values);seen={}
    def fake(**kw):seen.update(kw);return [partials]
    monkeypatch.setattr(nax,'_kernel',lambda n:fake)
    from mlx2.runtime.gguf import source_mxfp4_pair
    from mlx2 import process_env
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    monkeypatch.setattr(process_env,'tf32_enabled',lambda:False)
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)  # Actual default remains CPU.
    rounded,raw,actual=nax.project(x,e,rows,return_partials=True)
    assert actual is partials and seen['inputs'][0] is x and seen['inputs'][2] is e.backing
    assert np.all(np.asarray(raw)==np.float32(2**-8))
    assert np.array_equal(np.asarray(rounded),np.asarray(raw))
    assert seen['output_dtypes']==[mx.float32] and seen['output_shapes']==[(5,13,2304)]


def test_device_refuses_cpu_before_creating_kernel():
    with pytest.raises(ValueError,match='caller-owned'):
        nax.project(mx.zeros((13,5120)),packed(),mx.arange(13,dtype=mx.uint32))


@pytest.mark.parametrize('rows',[13,16,32,33,64,96])
def test_offline_metal4_compilation_only(rows,tmp_path):
    if not shutil.which('xcrun'):pytest.skip('Metal compiler unavailable')
    s=nax.kernel_spec(rows)
    text='#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']+'''
kernel void probe(const device float* x [[buffer(0)]],
const device uint* rows_map [[buffer(1)]],const device uchar* backing [[buffer(2)]],
const device uint* layout_offsets [[buffer(3)]],device float* partials [[buffer(4)]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
'''+s['source']+'\n}\n'
    path=tmp_path/f'm{rows}.metal';path.write_text(text)
    include=Path(mx.__file__).parent/'include'
    run=subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-I',str(include),'-c',str(path),'-o',str(tmp_path/'probe.air')],capture_output=True,text=True,timeout=45)
    assert run.returncode==0,run.stdout+run.stderr
