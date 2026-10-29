"""Bounded down worklists and mapped exact-sum CPU oracles; no GPU."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.gguf import source_down_prefill as down
from mlx2.runtime.gguf import source_prefill_combine as combine
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
from test_deepseek_v41_source_prefill_combine_cpu import _problem,_reference


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*args,**kwargs):raise AssertionError('CPU test reached Metal')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('kind,counts', [('original',(r,)*e) for r in (1,2,3,7,12) for e in (1,3,15)]+[('q1',(1,3,7,15,99)),('q1',(1024,)*6)])
def test_worklist_covers_each_input_and_output_once(kind,counts):
    p=down.plan(kind,counts);seen=[]
    for slot,group,base in p['jobs']:
        slot,group,base=map(int,(slot,group,base))
        assert base==sum(counts[:slot])
        seen.extend(base+group*p['vecs']+v for v in range(p['vecs']) if group*p['vecs']+v<counts[slot])
    assert sorted(seen)==list(range(sum(counts)))
    assert p['input_bindings']+2<=20 and p['scratch_bytes']<=80
    assert p['grid'][0]//64*p['tile']==5120
    assert p['threadgroup']==(64,1,1)


@pytest.mark.parametrize('kind,counts',[('other',(1,)),('original',(13,)),('original',(1,2)),('original',()),('q1',(0,)),('q1',(True,)),('q1',(1.5,)),('q1',(1,)*16),('q1',(1024,)*7)])
def test_bad_plan_refused(kind,counts):
    with pytest.raises(ValueError):down.plan(kind,counts)


def packed(kind):
    layout=canonical_layout(kind);types={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    fields={t.name:SimpleNamespace(shape=t.shape,dtype=types[t.dtype])for t in layout.tensors}
    backing=SimpleNamespace(shape=(layout.nbytes,),ndim=1,dtype=mx.uint8)
    return PackedExpert(backing,layout,fields)


@pytest.mark.parametrize('kind,counts',[('original',(1,1,1)),('original',(7,7,7)),('q1',(1,3,99))])
def test_prepare_binds_one_immutable_backing_and_exact_physical_offsets(kind,counts):
    experts=[packed(kind)for _ in counts]
    middle=SimpleNamespace(shape=(sum(counts),2304),dtype=mx.float32)
    p,offsets,retained=down.prepare_inputs(middle,experts,counts)
    assert all(a is b for a,b in zip(experts,retained))
    assert offsets.dtype==np.uint32 and offsets.shape==(len(counts),3)
    for i,e in enumerate(experts):
        assert offsets[i,0]==e.layout.tensor('down.qweight').offset
        assert offsets[i,1]==e.layout.tensor('down.scales').offset
        assert offsets[i,2]==(e.layout.tensor('down.biases').offset if kind=='q1' else 0)
    assert p['rows']==sum(counts)


@pytest.mark.parametrize('change',['kind','middle','backing','typed','plain'])
def test_bad_payload_geometry_refuses(change):
    experts=[packed('original'),packed('original')];x=SimpleNamespace(shape=(2,2304),dtype=mx.float32)
    if change=='kind':experts[1]=packed('q1')
    elif change=='middle':x.shape=(2,2303)
    elif change=='backing':experts[0].backing.shape=(1,)
    elif change=='typed':experts[0]['down.scales'].dtype=mx.float16
    else:experts[0]=dict(experts[0])
    with pytest.raises(ValueError):down.prepare_inputs(x,experts,(1,1))


@pytest.mark.parametrize('kind,rows',[('original',1),('original',2),('original',3),('original',7),('original',12),('q1',1)])
def test_source_has_disjoint_segment_stores_and_no_subtotal(kind,rows):
    s=down.kernel_spec(kind,rows,15,return_raw=True)
    assert s['total_buffer_bindings']==20
    assert len(s['input_names'])==18 and s['output_names']==['output','raw']
    assert 'output[index]=float(bfloat(value))' in s['source']
    assert 'raw[index]=value' in s['source']
    assert 'atomic' not in s['source'] and 'total' not in s['source']
    assert 'threadgroup_barrier' in s['source']
    if kind=='original' and rows>1:
        assert 'fp_qmv_wide_to_shared_impl' in s['header']
        assert 'y[v * 4 + out_row - tid.y * 4]' in s['header']
        assert 'constant int K = 2304;' in s['header'] and 'constant int N = 5120;' in s['header']


def _bucketed(rows=99,topk=6,parts_count=31):
    ids,order,raw,ordinary=_problem(rows,topk)
    unique=np.unique(ids);parts=[];descriptors=[]
    for i in reversed(range(min(parts_count,len(unique)))):
        selected=list(range(i,len(unique),min(parts_count,len(unique))))
        parts.append(np.concatenate([ordinary[j]for j in selected]))
        descriptors.append(tuple((int(unique[j]),len(ordinary[j]))for j in selected))
    return ids,order,raw,parts,descriptors


def replay(chunks,parts,shape):
    value=np.zeros(shape,np.float32)
    for chunk in chunks:
        live=[parts[i]for i in chunk['parts']]
        for row in range(shape[0]):
            for ref in chunk['refs'][row]:
                if ref:value[row]=np.add(value[row],live[(int(ref)>>16)-1][int(ref)&65535],dtype=np.float32)
    return value


@pytest.mark.parametrize('rows',[1,7,99,1024])
@pytest.mark.parametrize('budget',[1,2,7,28])
def test_bucket_mapping_preserves_individual_global_ID_order(rows,budget):
    ids,order,raw,parts,descriptors=_bucketed(rows)
    chunks=combine.mapped_plan(ids,order,descriptors,chunk_parts=budget)
    assert replay(chunks,parts,(rows,raw.shape[-1])).tobytes()==_reference(ids,raw).tobytes()
    assert all(c['total_buffer_bindings']<=31 for c in chunks)
    assert all(a['last_expert']<b['first_expert']for a,b in zip(chunks,chunks[1:]))
    assert sum(np.count_nonzero(c['refs'])for c in chunks)==ids.size


def test_group_subtotal_cancellation_would_be_wrong_but_refs_are_exact():
    ids=np.array([[2,0,1]],np.uint32);order=np.argsort(ids.ravel(),kind='stable')
    parts=[np.array([[2**25],[-2**25]],np.float32),np.array([[1]],np.float32)]
    descriptors=[((0,1),(2,1)),((1,1),)]
    for budget in (1,2,28):
        chunks=combine.mapped_plan(ids,order,descriptors,chunk_parts=budget)
        assert replay(chunks,parts,(1,1))[0,0]==0
    assert np.float32(np.float32(parts[0][0,0]+parts[0][1,0])+parts[1][0,0])==1


@pytest.mark.parametrize('change',['duplicate','count','missing','empty','invalidid'])
def test_mapped_descriptor_refusals(change):
    ids,order,raw,parts,d=_bucketed()
    d=[list(g)for g in d]
    if change=='duplicate':d[0].append(d[0][0])
    elif change=='count':e,n=d[0][0];d[0][0]=(e,n+1)
    elif change=='missing':d.pop()
    elif change=='empty':d.append([])
    else:e,n=d[0][0];d[0][0]=(384,n)
    with pytest.raises(ValueError):combine.mapped_plan(ids,order,d)


def test_mapped_dispatch_keeps_buffers_and_carries_not_group_subtotals(monkeypatch):
    ids,order,raw,np_parts,descriptors=_bucketed(99,6,31)
    parts=[mx.array(p)for p in np_parts];calls=[]
    def factory(n,first):
        def kernel(**kw):
            args=kw['inputs'];refs=np.array(args[-1]);live=[np.array(a)for a in args[:n]]
            result=np.zeros((99,9),np.float32)if first else np.array(args[n]).copy()
            for row in range(99):
                for ref in refs[row]:
                    if ref:result[row]=np.add(result[row],live[(int(ref)>>16)-1][int(ref)&65535],dtype=np.float32)
            calls.append(kw);return[mx.array(result)]
        return kernel
    monkeypatch.setattr(combine,'_kernel',factory)
    def forbidden(*a,**kw):raise AssertionError('mapped combine concatenated part payloads')
    monkeypatch.setattr(mx,'concatenate',forbidden)
    value=combine.combine_mapped(parts,ids,order,descriptors,chunk_parts=7)
    assert np.array(value).tobytes()==_reference(ids,raw).tobytes()
    assert all(len(c['inputs'])+1<=31 for c in calls)


def test_down_dispatch_borrows_backings_and_middle_without_completion_or_concat(monkeypatch):
    from mlx2.runtime.gguf import source_mxfp4_pair
    experts=[packed('q1')for _ in range(3)];counts=(1,3,7)
    middle=mx.zeros((11,2304),mx.float32);submitted=[]
    def kernel(**kw):
        submitted.append(kw)
        return[mx.zeros((11,5120),mx.float32),mx.zeros((11,5120),mx.float32)]
    monkeypatch.setattr(down,'_kernel',lambda *args:kernel)
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    def forbidden(*args,**kwargs):raise AssertionError('down unexpectedly copied/fenced producer or weights')
    monkeypatch.setattr(mx,'concatenate',forbidden);monkeypatch.setattr(mx,'eval',forbidden)
    monkeypatch.setattr(mx,'synchronize',forbidden)
    output,raw=down.pp_down(middle,experts,counts,return_raw=True)
    assert output.shape==raw.shape==(11,5120)
    assert submitted[0]['inputs'][0]is middle
    assert all(submitted[0]['inputs'][i+3]is e.backing for i,e in enumerate(experts))
    np.testing.assert_array_equal(np.array(submitted[0]['inputs'][1]),down.plan('q1',counts)['jobs'])
    assert submitted[0]['threadgroup']==(64,1,1)


def test_down_cpu_refusal_has_no_fallback(monkeypatch):
    from mlx2.runtime.gguf import source_mxfp4_pair
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:False)
    with pytest.raises(ValueError,match='owned M5'):
        down.pp_down(mx.zeros((1,2304)),[packed('original')],(1,))
