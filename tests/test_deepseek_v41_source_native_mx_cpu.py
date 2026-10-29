"""Native component API/representation fixtures; fake MLX, no GPU or driver."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'src/mlx2/runtime/gguf/source_native_mx.py'
spec=importlib.util.spec_from_file_location('mlx2.runtime.gguf._native_mx_cpu',path)
n=importlib.util.module_from_spec(spec);sys.modules[spec.name]=n;spec.loader.exec_module(n)


@pytest.mark.parametrize('projection,k,out',[('gate',5120,2304),('up',5120,2304),('down',2304,5120)])
@pytest.mark.parametrize('rows',[1,3,5,9,16,32,34,98])
def test_accepted_tiles_cover_actual_rows_without_early_return(projection,k,out,rows):
    p=n.plan(projection,rows);s=n.kernel_spec(projection,rows)
    assert p['output_shape']==(rows,out) and p['k']==k
    assert p['threadgroups']==(out//32,(rows+15)//16,1)
    assert p['padded_rows']>=rows and p['padded_rows']-rows<16
    assert p['full_weight_repack_bytes']==p['full_weight_dequantization_bytes']==0
    assert s['compile_options']=={'math_mode':'safe'} and not s['ensure_row_contiguous']
    assert 'return;'not in s['source']
    assert 'false, true, false, matmul2d_descriptor::mode::multiply' in s['source']
    assert s['source'].index('*it=0.0f')<s['source'].index('op.run(ta, tw, acc)')
    assert 'const device half* x_half' in n.compilation_source(projection,rows)
    assert 'device const'not in s['source']


@pytest.mark.parametrize('projection,rows,tile',[
    ('other',34,(16,32)),([],34,(16,32)),('down',True,(16,32)),
    ('down',34.0,(16,32)),('down',0,(16,32)),('down',99,(16,32)),
    ('down',34,[16,32]),('down',34,(16.0,32)),('down',34,(8,64))])
def test_unbounded_or_ambiguous_geometry_refused(projection,rows,tile):
    with pytest.raises(ValueError):n.plan(projection,rows,tile=tile)


BF16=object()


class Array:
    def __init__(self,x,dtype=None):
        self.a=np.asarray(x,dtype=dtype);self._logical_dtype=None
    @property
    def dtype(self):return self._logical_dtype if self._logical_dtype is not None else self.a.dtype
    @property
    def ndim(self):return self.a.ndim
    @property
    def shape(self):return self.a.shape
    @property
    def strides(self):return tuple(v//self.a.dtype.itemsize for v in self.a.strides)
    @property
    def nbytes(self):return self.a.nbytes
    def astype(self,dtype):
        if dtype is BF16:
            bits=self.a.astype(np.float32).view(np.uint32)
            result=Array(((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32))
            result._logical_dtype=BF16
            return result
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):return Array(self.a.astype(dtype))
    def view(self,dtype):return Array(self.a.view(dtype))
    def __eq__(self,other):return Array(self.a==other.a)
    def item(self):return self.a.item()


@pytest.fixture
def mx(monkeypatch):
    calls=[];m=ModuleType('mlx.core');package=ModuleType('mlx');package.core=m
    m.array=Array;m.float32=np.dtype('float32');m.float16=np.dtype('float16')
    m.uint32=np.dtype('uint32');m.uint8=np.dtype('uint8')
    m.bfloat16=BF16
    m.isfinite=lambda x:Array(np.isfinite(x.a));m.all=lambda x:Array(np.all(x.a))
    m.contiguous=lambda x:Array(np.ascontiguousarray(x.a))
    m.eval=lambda *args:calls.append(('eval',len(args)))
    m.gpu=object();m.default_device=lambda:m.gpu
    def metal_kernel(**kwargs):
        calls.append(('construct',kwargs))
        def invoke(**args):
            calls.append(('invoke',args));return[Array(np.zeros(args['output_shapes'][0],np.float32))]
        return invoke
    m.fast=SimpleNamespace(metal_kernel=metal_kernel)
    monkeypatch.setitem(sys.modules,'mlx',package);monkeypatch.setitem(sys.modules,'mlx.core',m)
    n._kernel.cache_clear()
    return m,calls


def input_array(rows=34,k=2304):
    values=np.array([0.,-0.,.5,-.5,2**-24,65504.],np.float32)
    return Array(np.resize(values,(rows,k)))


def test_actual_input_gate_preserves_all_bits_and_serializable_receipt(mx):
    import json
    _,calls=mx;x=input_array();p=n.prepare_input(x)
    assert p.raw is x and p.half.dtype==np.float16
    assert np.array_equal(x.a.view(np.uint32),p.half.a.astype(np.float32).view(np.uint32))
    assert p.receipt['finite'] and p.receipt['float_half_float_bits_exact']
    assert p.receipt['signed_zero_preserved']
    json.dumps(p.receipt,allow_nan=False)
    assert calls==[('eval',3)]


@pytest.mark.parametrize('value',[np.float32(.1),np.float32(1e-30),np.float32(65520),np.float32(np.nan),np.float32(np.inf)])
def test_lossy_overflow_underflow_nonfinite_input_refused_before_native(value,mx):
    _,calls=mx;x=input_array();x.a[0,0]=value
    with pytest.raises(ValueError,match='changes F32'):n.prepare_input(x)
    assert not any(x[0]in ('construct','invoke')for x in calls)


def test_wrong_input_never_casts(mx):
    _,calls=mx
    for x in [Array(np.zeros((34,2304),np.float16)),
              Array(np.zeros((2,2304),np.float32))]:
        with pytest.raises(ValueError):n.prepare_input(x)
    assert not calls


def test_strided_input_is_explicitly_materialized_without_public_strides_API(mx):
    x=Array(np.zeros((34,4608),np.float32)[:,::2]);p=n.prepare_input(x)
    assert p.half.a.flags.c_contiguous and p.receipt['contiguous_half_explicit']
    assert 'x.strides'not in path.read_text() and 'half.strides'not in path.read_text()


def expert(kind='original'):
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
    layout=canonical_layout(kind)
    backing=Array(np.broadcast_to(np.uint8(0),(layout.nbytes,)))
    return PackedExpert(backing,layout,{})


def test_project_borrows_exact_backing_and_keeps_owner_outside(mx):
    m,calls=mx;p=n.prepare_input(input_array());e=expert()
    # Array constructor's dtype overload matches MLX only for this metadata.
    class Factory(type):
        def __instancecheck__(cls,value):return isinstance(value,Array)
    class MetadataArray(metaclass=Factory):
        def __new__(cls,value,dtype=None):return Array(np.asarray(value,dtype=dtype))
    m.array=MetadataArray
    result=n.project(p,e,projection='down')
    invoke=next(value for name,value in calls if name=='invoke')
    assert invoke['inputs'][0] is p.half and invoke['inputs'][1] is e.backing
    assert invoke['inputs'][2].a.tolist()==[12533760,18432000]
    assert invoke['output_dtypes']==[np.dtype('float32')]
    assert result.shape==(34,5120)
    assert 'bfloat'not in n.kernel_spec('down',34)['source']


@pytest.mark.parametrize('bad',['forged','q1','wrong_k','changed_half','cpu'])
def test_bad_admission_or_backing_refuses_before_kernel(bad,mx):
    m,calls=mx;p=n.prepare_input(input_array());e=expert()
    if bad=='forged':p=n.PreparedInput(p.raw,p.half,p.receipt,object())
    if bad=='q1':e=expert('q1')
    if bad=='wrong_k':p=n.prepare_input(input_array(k=5120))
    if bad=='changed_half':p=n.PreparedInput(p.raw,Array(p.half.a.astype(np.float32)),p.receipt,n._SEAL)
    if bad=='cpu':m.default_device=lambda:object()
    with pytest.raises(ValueError):n.project(p,e,projection='down')
    assert not any(value[0]in('construct','invoke')for value in calls)


@pytest.mark.parametrize('tile',n.TILES)
@pytest.mark.parametrize('rows',[1,5,34,98])
def test_geometry_descriptor_origins_grid_and_cache_are_bound(tile,rows,mx):
    _,calls=mx;p=n.plan('down',rows,tile=tile);s=n.kernel_spec('down',rows,tile=tile)
    bm,bn=tile
    assert p['threadgroups']==(5120//bn,(rows+bm-1)//bm,1)
    assert p['grid']==(5120//bn*32,(rows+bm-1)//bm,1)
    assert p['tile']==tile and p['padded_rows']==(rows+bm-1)//bm*bm
    assert f'    {bm}, {bn}, 2304, false, true, false,' in s['source']
    assert s['source'].count(f'threadgroup_position_in_grid.y * {bm}')==2
    assert s['source'].count(f'threadgroup_position_in_grid.x * {bn}')==2
    assert n._kernel('down',rows,tile)is n._kernel('down',rows,tile)
    assert len([c for c in calls if c[0]=='construct'])==1


def test_default_standalone_kernel_is_byte_identical_to_first_compiled_control():
    import hashlib
    assert hashlib.sha256(n.compilation_source('down',34).encode()).hexdigest()==\
        '2707cc8826ea3aa97b17273636baff9d7fcbe899d7672c33a78b9a4d5bc87fc9'
    assert n.plan('down',34)==n.plan('down',34,tile=(16,32))


def test_distinct_geometry_cannot_reuse_another_pipeline_cache_key(mx):
    _,calls=mx
    kernels=[n._kernel('down',34,tile)for tile in n.TILES]
    assert len({id(k)for k in kernels})==len(n.TILES)
    names=[value['name']for name,value in calls if name=='construct']
    assert len(set(names))==len(n.TILES)


@pytest.mark.parametrize('rounded',[False,True])
def test_admitted_stage_has_fresh_cast_and_no_diagnostic_readbacks(rounded,mx,monkeypatch):
    m,calls=mx;admitted=n.prepare_input(input_array());e=expert();calls.clear()
    def forbidden(*a,**kw):raise AssertionError('hidden diagnostic evaluation/readback')
    for name in ['eval','all','isfinite']:monkeypatch.setattr(m,name,forbidden)
    monkeypatch.setattr(Array,'item',forbidden)
    original=Array.astype;casts=[]
    def cast(x,dtype):
        y=original(x,dtype);casts.append((x,dtype,y));return y
    monkeypatch.setattr(Array,'astype',cast)
    results=[n.project_from_admitted_raw(admitted,e,projection='down',tile=(16,64),rounded=rounded)for _ in range(2)]
    invokes=[value for name,value in calls if name=='invoke']
    assert len(invokes)==2 and invokes[0]['inputs'][0]is not invokes[1]['inputs'][0]
    assert all(value['inputs'][0]is not admitted.half and value['inputs'][1]is e.backing for value in invokes)
    assert len([v for v in casts if v[0]is admitted.raw and v[1]==m.float16])==2
    assert len([v for v in casts if v[1]is BF16])==(2 if rounded else 0)
    assert len([v for v in casts if v[1]==m.float32])==(2 if rounded else 0)
    assert all(result.shape==(34,5120)for result in results)
    assert all(value['grid']==(2560,3,1)for value in invokes)


@pytest.mark.parametrize('bad',['raw_replacement','half_replacement','receipt_false','receipt_array',
    'receipt_bool_as_int','receipt_shape_float','receipt_extra','forged','rounded','tile','projection'])
def test_admitted_stage_capability_and_arguments_refuse_before_cast(bad,mx,monkeypatch):
    m,calls=mx;p=n.prepare_input(input_array());e=expert();calls.clear()
    kwargs=dict(projection='down',tile=(8,32),rounded=True)
    if bad=='raw_replacement':p=n.PreparedInput(input_array(),p.half,p.receipt,p._seal,p._admission)
    if bad=='half_replacement':p=n.PreparedInput(p.raw,Array(p.half.a.copy()),p.receipt,p._seal,p._admission)
    if bad=='receipt_false':p.receipt['finite']=False
    if bad=='receipt_array':p.receipt['finite']=Array(True)
    if bad=='receipt_bool_as_int':p.receipt['finite']=1
    if bad=='receipt_shape_float':p.receipt['shape'][0]=34.0
    if bad=='receipt_extra':p.receipt['other']=1
    if bad=='forged':p=n.PreparedInput(p.raw,p.half,p.receipt,object(),p._admission)
    if bad=='rounded':kwargs['rounded']=1
    if bad=='tile':kwargs['tile']=(8,64)
    if bad=='projection':kwargs['projection']='other'
    def forbidden(*a,**kw):raise AssertionError('cast/eval/readback before refusal')
    monkeypatch.setattr(Array,'astype',forbidden);monkeypatch.setattr(Array,'item',forbidden)
    monkeypatch.setattr(m,'eval',forbidden)
    with pytest.raises(ValueError):n.project_from_admitted_raw(p,e,**kwargs)
    assert not calls
