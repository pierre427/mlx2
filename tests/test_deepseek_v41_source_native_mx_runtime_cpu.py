"""Dynamic Source native runtime metadata/capability contracts; no GPU/driver."""
from dataclasses import fields, replace
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

from mlx2.runtime.gguf import source_native_mx_runtime as r
from mlx2.adapters.deepseek_v41_source_buffers import HostExpert, PackedExpert, canonical_layout


class Array:
    def __init__(self,x,dtype=None):self.a=np.asarray(x,dtype=dtype)
    @property
    def shape(self):return self.a.shape
    @property
    def dtype(self):return self.a.dtype
    @property
    def ndim(self):return self.a.ndim
    def astype(self,dtype):return Array(self.a.astype(dtype))
    def __array__(self,*args,**kwargs):raise AssertionError('Runtime readback')
    def item(self):raise AssertionError('Runtime scalar readback')


@pytest.fixture
def mx(monkeypatch):
    calls=[];m=ModuleType('mlx.core');parent=ModuleType('mlx');parent.core=m
    m.array=Array;m.float32=np.dtype('float32');m.float16=np.dtype('float16')
    m.uint32=np.dtype('uint32');m.uint8=np.dtype('uint8')
    m.contiguous=lambda x:(calls.append(('contiguous',x)),Array(np.ascontiguousarray(x.a)))[1]
    m.eval=lambda *v:(_ for _ in ()).throw(AssertionError('Runtime eval'))
    def construct(**spec):
        calls.append(('construct',spec))
        def invoke(**kw):
            calls.append(('invoke',kw));return[Array(np.zeros(kw['output_shapes'][0],np.float32))]
        return invoke
    m.fast=SimpleNamespace(metal_kernel=construct)
    monkeypatch.setitem(sys.modules,'mlx',parent);monkeypatch.setitem(sys.modules,'mlx.core',m)
    r._kernel.cache_clear()
    return m,calls


def host(low=119,high=123):
    layout=canonical_layout('original');slab=np.zeros(layout.nbytes,np.uint8)
    for t in layout.tensors:
        if t.name.endswith('scales'):
            slab[t.offset:t.offset+t.nbytes]=low
            slab[t.offset+t.nbytes-1]=high
    return HostExpert(slab,layout)


def attestation(h):return r.attest_host(h,source_fingerprint='a'*64,source_seals_sha256='b'*64,layer=0,expert=38)


def expert(mx):
    m,_=mx;layout=canonical_layout('original')
    return PackedExpert(Array(np.zeros(layout.nbytes,np.uint8)),layout,
        {t.name:Array(np.zeros(t.shape,np.dtype(t.dtype)))for t in layout.tensors})


@pytest.mark.parametrize('rows',[1,3,15,16,17,31,32,34,98,99])
@pytest.mark.parametrize('projection',['gate','up','down'])
def test_runtime_m_dispatch_grid_and_bounded_tails(rows,projection):
    p=r.plan(projection,rows)
    assert p['dispatch_eligible'] is (rows>=16)
    assert p['grid']==(p['n'],(rows+15)//16,1)
    assert p['output_shape']==(rows,p['n'])
    assert p['kernel_geometry']==r.SHAPES[projection]
    assert (p['grid'][1]-1)*16 < rows <= p['grid'][1]*16
    assert p['full_weight_repack_bytes']==p['full_weight_dequantization_bytes']==0


@pytest.mark.parametrize('projection,rows',[('gate',0),('gate',100),('gate',True),('gate',16.),('bad',34),([],34)])
def test_runtime_unbounded_metadata_refused(projection,rows):
    with pytest.raises(ValueError):r.plan(projection,rows)


def test_source_has_dynamic_not_row_jit_variants():
    for p in ('gate','up','down'):
        s=r.kernel_spec(p)
        assert 'dynamic_extent' in s['source'] and 'AExt{rows}' in s['source'] and 'CExt{rows}' in s['source']
        assert 'int(layout_offsets_rows[2])' in s['source']
        assert 'return;' not in s['source']
        assert '16,32,' in s['source'] and 'false,true,false' in s['source']
        assert s['source'].index('*it=0.0f') < s['source'].index('op.run(')
        assert s['compile_options']=={'math_mode':'safe'}
    assert r.kernel_spec('gate')==r.kernel_spec('up')


def test_gate_up_and_all_runtime_rows_share_one_kernel_cache(mx):
    m,calls=mx;e=expert(mx);b=r.bind_expert(e,attestation(host()))
    for rows in (16,17,31,32,34,98,99):
        half=r.prepare_half(Array(np.full((rows,5120),1.0001,np.float32)))
        r.project_half(half,e,b,projection='gate');r.project_half(half,e,b,projection='up')
        r.project_half(r.prepare_half(Array(np.ones((rows,2304),np.float32))),e,b,projection='down')
    constructors=[v for v in calls if v[0]=='construct']
    assert len(constructors)==2
    assert r._kernel.cache_info().currsize==2
    assert all('m16'not in v[1]['name']and'm99'not in v[1]['name']for v in constructors)
    invokes=[v[1]for v in calls if v[0]=='invoke']
    assert [int(v['inputs'][2].a[2])for v in invokes]==[row for row in (16,17,31,32,34,98,99)for _ in range(3)]
    assert all(v['inputs'][1]is e.backing for v in invokes)
    assert invokes[0]['inputs'][2].a[:2].tolist()==[0,5898240]
    assert invokes[1]['inputs'][2].a[:2].tolist()==[6266880,12165120]


def test_half_conversion_is_explicit_changed_numeric_profile_without_readback(mx):
    m,calls=mx;x=Array(np.full((17,5120),1.0001,np.float32))
    half=r.prepare_half(x)
    assert half.dtype==np.float16
    assert not np.array_equal(half.a.astype(np.float32),x.a)
    assert r.plan('gate',17)['activation_consumption']=='float16_from_float32_rounding_allowed'
    assert [v[0]for v in calls]==['contiguous']


@pytest.mark.parametrize('rows,k,dtype',[(1,5120,np.float32),(15,5120,np.float32),(100,5120,np.float32),(17,100,np.float32),(17,5120,np.float16)])
def test_prepare_half_refuses_outside_source_bounded_domain(mx,rows,k,dtype):
    with pytest.raises(ValueError):r.prepare_half(Array(np.zeros((rows,k),dtype)))


@pytest.mark.parametrize('lo,hi,eligible',[(104,140,True),(119,123,True),(103,140,False),(104,141,False),(0,255,False)])
def test_attestation_conservative_scale_proof_and_sha(lo,hi,eligible):
    import json
    h=host(lo,hi);a=attestation(h)
    assert a.eligible is eligible
    assert a.payload_sha256==hashlib.sha256(h.data).hexdigest()
    assert a.scale_domains==tuple((p,lo,hi)for p in ('gate','up','down'))
    assert json.loads(json.dumps(a.receipt()))['eligible']is eligible


def test_analytical_scale_domain_exhaustive_e2m1_codes_finite_lossless_half():
    codes=np.array([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],np.float32)
    scales=np.exp2(np.arange(104,141,dtype=np.float32)-127)
    decoded=codes[:,None]*scales[None,:]
    restored=decoded.astype(np.float16).astype(np.float32)
    assert np.isfinite(restored).all()
    assert decoded.tobytes()==restored.tobytes()
    assert np.float32(.5)*np.exp2(np.float32(103-127))!=np.float32(np.float16(np.float32(.5)*np.exp2(np.float32(103-127))))
    assert np.float32(6)*np.exp2(np.float32(141-127))>np.finfo(np.float16).max


@pytest.mark.parametrize('kwargs',[{'layer':True},{'layer':40},{'expert':384},{'source_fingerprint':'x'*64},{'source_seals_sha256':'A'*64}])
def test_loader_identity_attestation_refuses_bad_source_fields(kwargs):
    params=dict(source_fingerprint='a'*64,source_seals_sha256='b'*64,layer=0,expert=38);params.update(kwargs)
    with pytest.raises(ValueError):r.attest_host(host(),**params)


def test_metadata_token_does_not_keep_expert_or_backing_alive(mx):
    import gc
    e=expert(mx);e_ref=weakref.ref(e.backing);b=r.bind_expert(e,attestation(host()))
    assert all(type(value)is int for _,value in b.tensor_object_ids)
    assert all(getattr(b,field.name)is not e and getattr(b,field.name)is not e.backing for field in fields(b))
    del e;gc.collect()
    assert e_ref()is None


@pytest.mark.parametrize('tamper',['expert','backing','view','coefficient','seal'])
def test_binding_and_coefficient_refusal_before_native(mx,tamper):
    m,calls=mx;e=expert(mx);b=r.bind_expert(e,attestation(host()));half=Array(np.zeros((34,5120),np.float16))
    if tamper=='expert':e=expert(mx)
    if tamper=='backing':e._backing=Array(e.backing.a.copy())
    if tamper=='view':e._tensors['gate.scales']=Array(e['gate.scales'].a.copy())
    if tamper=='coefficient':b=replace(b,attestation=attestation(host(0,255)))
    if tamper=='seal':b=replace(b,_seal=None)
    with pytest.raises(ValueError):r.project_half(half,e,b)
    assert not any(v[0]=='construct'for v in calls)


def test_smallm_and_mismatched_k_refuse_dispatch(mx):
    m,calls=mx;e=expert(mx);b=r.bind_expert(e,attestation(host()))
    for shape in ((15,5120),(34,2304)):
        with pytest.raises(ValueError):r.project_half(Array(np.zeros(shape,np.float16)),e,b)
    assert not any(v[0]=='construct'for v in calls)
