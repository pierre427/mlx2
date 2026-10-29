"""Prepared mapping invariants and independent F32 oracle; fake backend only."""
import gc
from dataclasses import replace
import sys
from types import ModuleType

import numpy as np
import pytest

from mlx2.runtime.gguf import source_prefill_combine as c
from mlx2.adapters import deepseek_v41_source_math as math
from mlx2.adapters.deepseek_v41_source_prefill import plan_jobs


class Array:
    def __init__(self,data,dtype='f32'):
        self.data=data;self.dtype=dtype;self.shape=data.shape;self.ndim=data.ndim


@pytest.fixture
def backend(monkeypatch):
    mx=ModuleType('mlx.core');mx.float32='f32';mx.uint32='u32';mx.uploads=[];mx.calls=[]
    def upload(value,dtype):
        assert not value.flags.writeable
        mx.uploads.append(value);return Array(value.copy(),dtype)
    def forbidden(*a,**kw):raise AssertionError('Unexpected readback/sync/concatenate')
    mx.array=upload;mx.eval=forbidden;mx.synchronize=forbidden;mx.concatenate=forbidden
    package=ModuleType('mlx');package.core=mx
    monkeypatch.setitem(sys.modules,'mlx',package);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    def factory(count,first):
        def kernel(**kw):
            args=kw['inputs'];live=args[:count];refs=args[-1].data
            result=np.zeros(kw['output_shapes'][0],np.float32)if first else args[count].data.copy()
            for row in range(len(result)):
                for ref in refs[row]:
                    if ref:result[row]=np.add(result[row],live[(int(ref)>>16)-1].data[int(ref)&65535],dtype=np.float32)
            mx.calls.append((count,first,kw));return[Array(result)]
        return kernel
    monkeypatch.setattr(c,'_kernel',factory)
    return mx


def problem(rows=99,part_count=31):
    rng=np.random.default_rng(411026+rows)
    ids=rng.integers(0,61,(rows,6),dtype=np.uint32);ids[::3,2]=ids[::3,0]
    raw=rng.integers(-500,500,(rows,6,9)).astype(np.float32)/32
    raw[:,:,0]=np.resize(np.array([2**25,1,-2**25,1,-0.,0.],np.float32),(rows,6))
    raw[:,:,1]=-0.;order=np.argsort(ids.ravel(),kind='stable')
    unique=np.unique(ids);ordinary=[raw.reshape(-1,9)[order][ids.ravel()[order]==e]for e in unique]
    parts=[];descriptors=[]
    for slot in reversed(range(min(part_count,len(unique)))):
        picked=range(slot,len(unique),min(part_count,len(unique)))
        picked=tuple(picked)
        parts.append(Array(np.concatenate([ordinary[j]for j in picked])))
        descriptors.append(tuple((int(unique[j]),len(ordinary[j]))for j in picked))
    result=np.zeros((rows,9),np.float32)
    for row in range(rows):
        for slot in sorted(range(6),key=lambda s:int(ids[row,s])):
            result[row]=np.add(result[row],raw[row,slot],dtype=np.float32)
    return ids,order,parts,descriptors,result


@pytest.mark.parametrize('rows',[1,7,99,1024])
@pytest.mark.parametrize('chunk',[1,3,28])
def test_prepared_exact_order_duplicates_cancellation_and_immutable_carry(backend,monkeypatch,rows,chunk):
    ids,order,parts,d,expected=problem(rows)
    packet=c.prepare_mapped(parts,ids,order,d,chunk_parts=chunk)
    assert not packet.values.flags.writeable and not packet.order.flags.writeable
    assert packet.values is not ids and packet.order is not order
    assert packet.metadata_bytes==sum(v.refs.nbytes for v in packet.chunks)
    assert all(not v.refs.flags.writeable for v in packet.chunks)
    uploaded=len(backend.uploads)
    monkeypatch.setattr(c,'mapped_plan',lambda *a,**k:pytest.fail('Prepared consumer replanned'))
    a=c.combine_mapped_prepared(parts,packet);b=c.combine_mapped_prepared(parts,packet)
    assert a is not b and len(backend.uploads)==uploaded
    assert a.data.tobytes()==b.data.tobytes()==expected.tobytes()
    assert all(len(kw['inputs'])+1<=31 for _,_,kw in backend.calls)
    # Changing the caller's writable routing array cannot change its snapshot.
    ids[:]=0;order[:]=0
    assert c.combine_mapped_prepared(parts,packet).data.tobytes()==expected.tobytes()


def test_default_mapping_builds_offsets_once_and_never_discarded_contiguous_plan(monkeypatch):
    ids,order,parts,d,_=problem();calls=[];ordinary=math.ordered_offsets
    def wrapped(*a):calls.append(a);return ordinary(*a)
    monkeypatch.setattr(math,'ordered_offsets',wrapped)
    monkeypatch.setattr(c,'plan',lambda *a,**k:pytest.fail('Mapped planner built discarded chunks'))
    chunks=c.mapped_plan(ids,order,d)
    assert len(calls)==1 and chunks


def test_existing_readonly_route_plan_borrowed_without_offsets_rebuild(backend,monkeypatch):
    ids,order,parts,d,expected=problem()
    route=plan_jobs(ids,{int(e):'q1'for e in np.unique(ids)},reuse=True)
    monkeypatch.setattr(math,'ordered_offsets',lambda *a:pytest.fail('Borrowed offsets reconstructed'))
    packet=c.prepare_mapped(parts,route.values,route.order,d,route_plan=route)
    assert packet.values is route.values and packet.order is route.order and packet.borrowed_route_plan
    assert c.combine_mapped_prepared(parts,packet).data.tobytes()==expected.tobytes()


@pytest.mark.parametrize('change',['identity','writable','duplicate','crossrow','reverse','shape','dtype','missing'])
def test_bad_authoritative_route_map_refused_before_metadata_upload(backend,change):
    ids,order,parts,d,_=problem()
    route=plan_jobs(ids,{int(e):'q1'for e in np.unique(ids)},reuse=True)
    values=route.values;borrowed_order=route.order;offsets=route.combine_offsets.copy()
    if change=='identity':values=values.copy();values.flags.writeable=False
    elif change=='duplicate':offsets[:]=0
    elif change=='crossrow':offsets[[0,1]]=offsets[[1,0]]
    elif change=='reverse':offsets=offsets[:,::-1].copy()
    elif change=='shape':offsets=offsets[:-1]
    elif change=='dtype':offsets=offsets.astype(np.int64)
    if change!='writable':offsets.flags.writeable=False
    route=replace(route,combine_offsets=None if change=='missing'else offsets)
    with pytest.raises(ValueError):c.prepare_mapped(parts,values,borrowed_order,d,route_plan=route)
    assert not backend.uploads and not backend.calls


@pytest.mark.parametrize('change',['newpart','permutation','shape','dtype','seal','writable'])
def test_prepared_consumer_refuses_wrong_frame_parts_before_kernel(backend,change):
    ids,order,parts,d,_=problem();packet=c.prepare_mapped(parts,ids,order,d)
    if change=='newpart':parts[0]=Array(parts[0].data.copy())
    elif change=='permutation':parts[0],parts[1]=parts[1],parts[0]
    elif change=='shape':parts[0].shape=(1,9)
    elif change=='dtype':parts[0].dtype='bf16'
    elif change=='seal':packet=replace(packet,_seal=object())
    else:packet.values.flags.writeable=True
    with pytest.raises(ValueError):c.combine_mapped_prepared(parts,packet)
    assert not backend.calls


def test_packet_does_not_keep_activation_parts_alive(backend):
    ids,order,parts,d,_=problem();packet=c.prepare_mapped(parts,ids,order,d)
    refs=packet.part_refs;del parts;gc.collect()
    assert all(ref()is None for ref in refs)
    assert c._kernel.__name__=='factory' # no global tensor/packet cache introduced


def test_bad_part_extent_and_weak_ownership_fail_before_upload(backend):
    from types import SimpleNamespace
    ids,order,parts,d,_=problem()
    parts[0].shape=(parts[0].shape[0]+1,9)
    with pytest.raises(ValueError):c.prepare_mapped(parts,ids,order,d)
    ids,order,parts,d,_=problem()
    parts=[SimpleNamespace(shape=p.shape,ndim=p.ndim,dtype=p.dtype)for p in parts]
    with pytest.raises(ValueError,match='weak-referenceable'):c.prepare_mapped(parts,ids,order,d)
    assert not backend.uploads
