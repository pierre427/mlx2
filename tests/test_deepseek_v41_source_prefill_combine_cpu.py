"""Independent direct-part reference/ownership tests; no Metal execution."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import source_prefill_combine as combine


def _problem(rows,topk,width=9):
    rng=np.random.default_rng(22500+rows+topk)
    ids=rng.integers(0,61,(rows,topk),dtype=np.uint32)
    if topk>=3:ids[::3,2]=ids[::3,0] # duplicate experts retain separate route scores
    raw=rng.integers(-500,500,(rows,topk,width)).astype(np.float32)/32
    raw[:,:,0]=np.resize(np.array([2**25,1,-2**25,1,-0.,0.],np.float32),(rows,topk))
    raw[:,:,1]=-0.
    raw[:,:,2]=np.resize(np.array([2**-120,-2**-120,2**-126,-2**-126],np.float32),(rows,topk))
    order=np.argsort(ids.ravel(),kind='stable')
    sorted_raw=raw.reshape(-1,width)[order]
    sorted_ids=ids.ravel()[order]
    cuts=(np.flatnonzero(np.diff(sorted_ids))+1).tolist()
    parts=list(np.split(sorted_raw,cuts))
    return ids,order,raw,parts


def _reference(ids,raw):
    result=np.zeros((len(ids),raw.shape[-1]),np.float32)
    for row in range(len(ids)):
        for slot in sorted(range(ids.shape[1]),key=lambda i:int(ids[row,i])):
            result[row]=np.add(result[row],raw[row,slot],dtype=np.float32)
    return result


def _replay(chunks,parts,shape):
    result=np.zeros(shape,np.float32)
    for chunk in chunks:
        live=parts[chunk['begin']:chunk['end']]
        for row in range(len(result)):
            for encoded in chunk['refs'][row]:
                if encoded==0:continue
                index=(int(encoded)>>16)-1;local=int(encoded)&0xffff
                result[row]=np.add(result[row],live[index][local],dtype=np.float32)
    return result


@pytest.mark.parametrize('rows,topk',[(1,1),(1,6),(7,6),(99,6),(1024,6)])
@pytest.mark.parametrize('chunk_parts',[1,2,7,28])
def test_refs_preserve_independent_ordered_sum_across_chunks(rows,topk,chunk_parts):
    ids,order,raw,parts=_problem(rows,topk)
    plan=combine.plan(ids,order,[len(p) for p in parts],chunk_parts=chunk_parts)
    actual=_replay(plan,parts,(rows,raw.shape[-1]))
    assert actual.tobytes()==_reference(ids,raw).tobytes()
    assert all(p['total_buffer_bindings']<=31 for p in plan)
    assert sum(np.count_nonzero(p['refs']) for p in plan)==ids.size
    assert np.all(np.signbit(actual[:,1])==np.signbit(_reference(ids,raw)[:,1]))
    starts=np.cumsum([0]+[len(p) for p in parts[:-1]])
    for chunk in plan:
        for row in range(rows):
            slots=sorted(range(topk),key=lambda s:int(ids[row,s]))
            for rank,slot in enumerate(slots):
                ref=int(chunk['refs'][row,rank])
                if not ref:continue
                global_part=chunk['begin']+(ref>>16)-1
                global_offset=int(starts[global_part])+(ref&0xffff)
                assert order[global_offset]==row*topk+slot


def test_provider_style_immutable_dispatch_has_no_concatenate_or_zero_array(monkeypatch):
    ids,order,raw,np_parts=_problem(99,6)
    with mx.stream(mx.cpu):
        parts=[mx.array(p) for p in np_parts]
        calls=[];outputs=[]
        def factory(count,first):
            def kernel(**kw):
                args=kw['inputs'];refs=np.array(args[-1]);bound=args[:count]
                initial=np.zeros((99,9),np.float32) if first else np.array(args[count])
                # Independent of planner extents: kernel sees only its direct
                # bound buffers plus encoded rows and immutable previous output.
                value=initial.copy()
                for row in range(99):
                    for ref in refs[row]:
                        if ref:
                            value[row]=np.add(value[row],np.array(bound[(int(ref)>>16)-1])[int(ref)&65535],dtype=np.float32)
                out=mx.array(value);outputs.append(out)
                calls.append((count,first,kw))
                return [out]
            return kernel
        def forbidden(*a,**kw):raise AssertionError('avoidable MLX copy/zero/Metal reached')
        monkeypatch.setattr(combine,'_kernel',factory)
        monkeypatch.setattr(mx,'concatenate',forbidden)
        monkeypatch.setattr(mx,'zeros',forbidden)
        monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
        actual=combine.combine(parts,ids,order,chunk_parts=28);mx.eval(actual)
        assert np.array(actual).tobytes()==_reference(ids,raw).tobytes()
        cursor=0
        for i,(count,first,kw) in enumerate(calls):
            assert first==(i==0)
            assert all(a is b for a,b in zip(kw['inputs'][:count],parts[cursor:cursor+count]))
            if not first:assert kw['inputs'][count] is outputs[i-1]
            assert len(kw['inputs'])+1<=31
            assert kw['output_shapes']==[(99,9)] and kw['output_dtypes']==[mx.float32]
            cursor+=count
        assert cursor==len(parts) and actual is outputs[-1]


@pytest.mark.parametrize('count',[1,7,28])
@pytest.mark.parametrize('first',[True,False])
def test_kernel_binding_budget_exact_add_order_and_absent_skip(count,first):
    spec=combine.kernel_spec(count,first)
    assert len(spec['input_names'])+len(spec['output_names'])==count+2+(not first)
    assert len(set(spec['input_names']))==len(spec['input_names'])
    assert ('previous' in spec['input_names'])==(not first)
    assert spec['source'].count('case ')==count
    assert 'if(ref==0) continue;' in spec['source']
    assert 'loop unroll(disable)' in spec['source']
    assert spec['compile_options']=={'math_mode':'safe'}
    assert '#pragma clang fp contract(off)' in spec['header']
    assert ('float total=0.0f' in spec['source'])==first


@pytest.mark.parametrize('change',[
    'bad_dtype','flat_values','empty_values','too_many_rows','too_many_slots','expert_out_of_range',
    'flat_order','float_order','short_order','negative_order','out_of_range_order','duplicate_order',
    'unsorted_order','unstable_tie','empty_parts','zero_part','huge_part','float_part','boolean_part',
    'wrong_extent','zero_chunk','big_chunk','boolean_chunk',
])
def test_bad_plans_refuse_before_any_kernel(change,monkeypatch):
    ids=np.array([[1,0,1]],np.uint32);order=np.array([1,0,2]);counts=[1,2];chunk=28
    if change=='bad_dtype':ids=ids.astype(np.int32)
    elif change=='flat_values':ids=ids.ravel()
    elif change=='empty_values':ids=ids[:0]
    elif change=='too_many_rows':ids=np.zeros((1025,1),np.uint32)
    elif change=='too_many_slots':ids=np.zeros((1,7),np.uint32)
    elif change=='expert_out_of_range':ids[0,0]=384
    elif change=='flat_order':order=order.reshape(1,-1)
    elif change=='float_order':order=order.astype(np.float32)
    elif change=='short_order':order=order[:2]
    elif change=='negative_order':order=np.array([-1,0,1])
    elif change=='out_of_range_order':order=np.array([0,1,3])
    elif change=='duplicate_order':order=np.array([0,0,1])
    elif change=='unsorted_order':order=np.array([0,1,2])
    elif change=='unstable_tie':order=np.array([1,2,0])
    elif change=='empty_parts':counts=[]
    elif change=='zero_part':counts=[0,3]
    elif change=='huge_part':counts=[65537]
    elif change=='float_part':counts=[1.,2]
    elif change=='boolean_part':counts=[True,2]
    elif change=='wrong_extent':counts=[1,1]
    elif change=='zero_chunk':chunk=0
    elif change=='big_chunk':chunk=29
    elif change=='boolean_chunk':chunk=True
    monkeypatch.setattr(combine,'_kernel',lambda *a:pytest.fail('invalid plan reached Metal'))
    with pytest.raises(ValueError):combine.plan(ids,order,counts,chunk_parts=chunk)


@pytest.mark.parametrize('change',['dtype','rank','width','empty'])
def test_bad_output_arrays_refuse_without_launch(change,monkeypatch):
    def fake(shape,dtype):return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype)
    parts=[fake((1,9),mx.float32),fake((2,9),mx.float32)]
    if change=='dtype':parts[1]=fake((2,9),mx.bfloat16)
    elif change=='rank':parts[1]=fake((18,),mx.float32)
    elif change=='width':parts[1]=fake((2,8),mx.float32)
    else:parts=[]
    monkeypatch.setattr(combine,'_kernel',lambda *a:pytest.fail('bad output reached kernel'))
    with pytest.raises(ValueError):
        combine.combine(parts,np.array([[1,0,1]],np.uint32),np.array([1,0,2]))
