"""Direct DOWN metadata/source/ownership guards; no GPU or MLX backend import."""
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS

import numpy as np
import pytest

from mlx2.runtime.gguf import source_down_worklist as down
from mlx2.runtime.gguf import source_mxfp4_pair as pair
from mlx2.runtime.gguf import source_down_prefill as old
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout,ExpertLayout,TensorSlice
from mlx2.adapters import deepseek_v41_source_buffers as buffers


@pytest.fixture(autouse=True)
def pinned_installed_source(monkeypatch):
    root=Path('~/Desktop/mlx2/.venv/lib/python3.12/site-packages/mlx')
    if not root.is_dir():pytest.skip('Pinned installed MLX headers unavailable; no GPU is used')
    monkeypatch.setattr(pair,'installed_root',lambda:root)


@pytest.mark.parametrize('kind,counts',[('original',(m,)*e)for m in range(1,13)for e in (1,3,15)]+[('q1',(1,2,3)),('q1',(3,)*15),('q1',(1,)*15)])
def test_each_global_row_has_one_device_owner(kind,counts):
    p=down.plan(kind,counts);seen=[]
    if kind=='original':
        for job in range(p['grid'][1]):
            slot,group=divmod(job,p['vector_groups'])
            for v in range(p['vecs']):
                if group*p['vecs']+v<counts[slot]:seen.append(slot*counts[slot]+group*p['vecs']+v)
    else:
        assert tuple(p['owner_slots'])==tuple(s for s,m in enumerate(counts)for _ in range(m))
        assert not p['owner_slots'].flags.writeable
        seen=list(range(p['grid'][1]))
    assert sorted(seen)==list(range(sum(counts)))
    assert p['scratch_bytes']==0 and p['threadgroup']==(64,1,1)
    assert p['grid'][0]//64*p['tile']==5120
    assert p['output_bytes']==sum(counts)*5120*4
    assert p['input_bindings']+2<=19


@pytest.mark.parametrize('kind,counts',[('other',(1,)),('original',()),('original',(1,2)),('original',(13,)),('q1',(4,)),('q1',(0,)),('q1',(True,)),('q1',(1.,)),('q1',(1,)*16)])
def test_bad_geometry_fails_closed(kind,counts):
    with pytest.raises(ValueError):down.plan(kind,counts)


@pytest.mark.parametrize('kind,rows',[('original',m)for m in range(1,13)]+[('q1',1)])
@pytest.mark.parametrize('raw',[False,True])
def test_direct_sink_no_shared_memory_or_barrier_and_stock_dot_preserved(kind,rows,raw):
    s=down.kernel_spec(kind,rows,15,return_raw=raw)
    text=s['header']+'\n'+s['source']
    assert 'threadgroup float partials'not in text and 'threadgroup T* y'not in text
    assert 'threadgroup_barrier'not in text and 'atomic'not in text
    assert 'bfloat('in s['header']
    assert ('device T* raw'in s['header']or'device float* raw'in s['header'])==raw
    assert s['total_buffer_bindings']<=19
    assert 'jobs'not in s['input_names']and'layout_offsets'not in s['input_names']
    if kind=='q1':
        reference=old.kernel_spec('q1',1,15)['header']
        first='    float accumulators[';last='    for (uint row = 0; row < 4; ++row) {\n        accumulators[row] = simd_sum'
        a=reference[reference.index(first):reference.index(last)]
        b=s['header'][s['header'].index(first):s['header'].index(last)]
        assert a==b
        assert 'const uint slot=owner_slots[job], row_base=job'in s['source']
    else:
        reference=old.kernel_spec('original',rows,15)['header']
        if rows==1:
            start='    int k = 0;';end='    for (int row = 0; row < results_per_simdgroup; row++) {\n      result[row] = simd_sum'
            # Both small-output and normal-output dot/reduction programs stay.
            assert s['header'].count('qdot_safe<U, values_per_thread, bits>')==reference.count('qdot_safe<U, values_per_thread, bits>')
            a=reference[reference.rindex(start):reference.rindex(end)]
            b=s['header'][s['header'].rindex(start):s['header'].rindex(end)]
        else:
            start='  // Each lane reduces a strided subset';end='  if (k_lane == 0 && out_row < out_vec_size)'
            a=reference[reference.index(start):reference.index(end)]
            b=s['header'][s['header'].index(start):s['header'].index(end)]
        assert a==b


def test_q1_route_counts_do_not_specialize_shader_or_pipeline():
    a=down.plan('q1',(1,2,3));b=down.plan('q1',(3,1,1))
    assert a['rows']!=b['rows']and a['local_rows']==b['local_rows']==1
    sa=down.kernel_spec('q1',a['local_rows'],a['experts'])
    sb=down.kernel_spec('q1',b['local_rows'],b['experts'])
    assert sa['header']==sb['header']and sa['source']==sb['source']


@pytest.fixture
def fake_mx(monkeypatch):
    mx=ModuleType('mlx.core');mx.float32='float32';mx.uint32='uint32';mx.uint8='uint8';mx.float16='float16'
    mx.normalized=[];mx.uploaded=[]
    def contiguous(x):mx.normalized.append(x);return x
    def array(x,dtype):mx.uploaded.append((x,dtype));return NS(host=x,dtype=dtype)
    def forbidden(*a,**k):raise AssertionError('Candidate read back/evaluated/copied weight inputs')
    mx.contiguous=contiguous;mx.array=array;mx.eval=forbidden;mx.synchronize=forbidden;mx.concatenate=forbidden
    package=ModuleType('mlx');package.core=mx
    monkeypatch.setitem(sys.modules,'mlx',package);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    return mx


def packed(kind,mx):
    layout=canonical_layout(kind);types={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    class Array:
        ndim=1
        shape=(layout.nbytes,)
        dtype=mx.uint8
    return PackedExpert(Array(),layout,
        {t.name:NS(shape=t.shape,dtype=types[t.dtype])for t in layout.tensors},
        _compact_issuer=buffers._COMPACT_UPLOAD_ISSUER)


@pytest.mark.parametrize('kind,counts',[('q1',(1,3,2)),('original',(9,9,9))])
@pytest.mark.parametrize('raw',[False,True])
def test_dispatch_only_normalizes_bounded_middle_and_borrows_all_backings(fake_mx,monkeypatch,kind,counts,raw):
    mx=fake_mx;experts=tuple(packed(kind,mx)for _ in counts);x=NS(shape=(sum(counts),2304),dtype=mx.float32)
    submitted=[];keys=[]
    def factory(*args):
        keys.append(args)
        def call(**kw):
            submitted.append(kw)
            return [NS(shape=(sum(counts),5120))for _ in range(2 if raw else 1)]
        return call
    monkeypatch.setattr(down,'_kernel',factory);monkeypatch.setattr(pair,'_device_supported',lambda:True)
    value=down.pp_down(x,experts,counts,return_raw=raw)
    outputs=value if raw else(value,)
    assert all(v.shape==(sum(counts),5120)for v in outputs)
    args=submitted[0]['inputs'];assert args[0]is x
    assert all(a is e.backing for a,e in zip(args[-len(experts):],experts))
    assert mx.normalized==[x]and len(mx.uploaded)==int(kind=='q1')
    assert keys==[(kind,counts[0]if kind=='original'else 1,len(counts),raw,'canonical')]
    if kind=='q1':np.testing.assert_array_equal(mx.uploaded[0][0],[0,1,1,1,2,2])


@pytest.mark.parametrize('change',['dtype','middle','backing','typed','mixed','plain','noncanonical','unproven'])
def test_prepare_refusals_before_launch(fake_mx,change):
    mx=fake_mx;experts=[packed('original',mx)for _ in range(2)];x=NS(shape=(2,2304),dtype=mx.float32)
    if change=='dtype':x.dtype=mx.float16
    elif change=='middle':x.shape=(2,2305)
    elif change=='backing':experts[0].backing.shape=(1,)
    elif change=='typed':experts[0]['down.scales'].dtype=mx.float16
    elif change=='mixed':experts[1]=packed('q1',mx)
    elif change=='plain':experts[0]=dict(experts[0])
    elif change=='unproven':experts[0]=PackedExpert(experts[0].backing,experts[0].layout,dict(experts[0]))
    else:
        original=experts[0];cursor=0;ts=[]
        for t in reversed(original.layout.tensors):
            ts.append(TensorSlice(t.name,cursor,t.dtype,t.shape));cursor+=t.nbytes
        experts[0]=PackedExpert(original.backing,ExpertLayout('original',tuple(ts)),dict(original),_compact_issuer=buffers._COMPACT_UPLOAD_ISSUER)
    with pytest.raises(ValueError):down.prepare_inputs(x,experts,(1,1))
    assert not mx.normalized and not mx.uploaded


def test_device_refusal_and_boolean_raw_do_not_launch(fake_mx,monkeypatch):
    mx=fake_mx;e=packed('q1',mx);x=NS(shape=(1,2304),dtype=mx.float32)
    monkeypatch.setattr(pair,'_device_supported',lambda:False)
    with pytest.raises(ValueError,match='caller-owned'):down.pp_down(x,[e],[1])
    with pytest.raises(ValueError,match='boolean'):down.pp_down(x,[e],[1],return_raw=1)
    assert not mx.normalized and not mx.uploaded


def test_canonical_descriptor_is_cached_without_arrays():
    down._canonical_layout.cache_clear()
    a=down._canonical_layout('original');b=down._canonical_layout('original')
    assert a is b and down._canonical_layout.cache_info().hits==1
    assert down._canonical_layout.cache_info().maxsize==2
    assert down._kernel.cache_info().maxsize>=390


def test_compact_capability_exact_weak_owner_and_no_global_registry(fake_mx):
    import gc,weakref
    e=packed('q1',fake_mx);backing=e.backing
    assert e.has_compact_backing
    assert isinstance(e._compact_backing,weakref.ReferenceType)
    unproven=PackedExpert(backing,e.layout,dict(e))
    forged=PackedExpert(backing,e.layout,dict(e),_compact_issuer=object())
    assert not unproven.has_compact_backing and not forged.has_compact_backing
    e._backing=NS(shape=backing.shape,ndim=1,dtype=backing.dtype)
    assert not e.has_compact_backing
    del backing,unproven,forged;gc.collect()
    assert e._compact_backing()is None and not e.has_compact_backing


def test_actual_materialize_issues_capability_only_after_fresh_host_upload(fake_mx,monkeypatch):
    mx=fake_mx;calls=[]
    class Array:
        def __init__(self,shape,dtype):self.shape=shape;self.ndim=len(shape);self.dtype=dtype
        def __getitem__(self,part):return Array((part.stop-part.start,),self.dtype)
        def view(self,dtype):return Array(self.shape,dtype)
        def reshape(self,shape):return Array(shape,self.dtype)
    def upload(data):
        assert isinstance(data,np.ndarray)and data.dtype==np.uint8 and data.flags.c_contiguous
        calls.append(data);return Array(data.shape,mx.uint8)
    monkeypatch.setattr(mx,'array',upload)
    layout=canonical_layout('q1');slab=np.zeros(layout.nbytes,np.uint8)
    host=buffers.HostExpert(slab,layout)
    e=buffers.materialize(host)
    assert e.has_compact_backing and calls==[host.data]
    assert e._compact_backing()is e.backing
    assert all(e[t.name].shape==t.shape for t in layout.tensors)


def test_native_q1_physical_layout_is_distinct_bounded_and_not_repacked(fake_mx):
    mx=fake_mx;e=packed('q1',mx);layout=down._native_q1_layout()
    native=PackedExpert(e.backing,layout,dict(e),_compact_issuer=buffers._COMPACT_UPLOAD_ISSUER)
    d,experts=down.prepare_inputs(NS(shape=(2,2304),dtype=mx.float32),[native],[2])
    assert d['layout_variant']=='native_q1_v1'and experts[0]is native
    s=down.kernel_spec('q1',1,1,layout_variant=d['layout_variant'])
    assert 'backing+0)'in s['source']and'backing+4792320)'in s['source']and'backing+4423680)'in s['source']
    assert layout.tensor('down.qweight').offset==0
    assert layout.tensor('down.scales').offset==4792320
    assert layout.tensor('down.biases').offset==4423680
    assert not mx.normalized and not mx.uploaded
    with pytest.raises(ValueError):
        down.prepare_inputs(NS(shape=(2,2304),dtype=mx.float32),[native,e],[1,1])
    with pytest.raises(ValueError):down.kernel_spec('original',1,1,layout_variant='native_q1_v1')


def test_real_native_parser_inventory_order_has_same_physical_capability(fake_mx):
    mx=fake_mx;e=packed('q1',mx);native=down.native_q1_layout()
    loader_layout=ExpertLayout('q1',tuple(native.tensor(t.name)for t in canonical_layout('q1').tensors))
    assert loader_layout!=native # borrow_q1_packed uses schema inventory order
    issued=PackedExpert(e.backing,loader_layout,dict(e),_compact_issuer=buffers._COMPACT_UPLOAD_ISSUER)
    d,experts=down.prepare_inputs(NS(shape=(3,2304),dtype=mx.float32),[issued],[3])
    assert d['layout_variant']=='native_q1_v1'and experts[0]is issued
    assert down._layout_signature(loader_layout)==down._layout_signature(native)
