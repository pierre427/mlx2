"""CPU/source contracts only; no actual Metal numerical or throughput oracle."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
from mlx2.runtime.gguf import source_gate_up_fusion as gu
from mlx2.runtime.gguf import source_gate_up_prefill as pp


def arr(shape, dtype):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def packed(kind):
    layout=canonical_layout(kind)
    return PackedExpert(arr((layout.nbytes,), mx.uint8), layout, {})


def fixture(kind='original', counts=(7,7,7)):
    return (arr((99,5120),mx.float32),[packed(kind) for _ in counts],
            arr((sum(counts),),mx.uint32),arr((sum(counts),),mx.float32),counts)


@pytest.mark.parametrize('rows',range(1,13))
def test_original_preserves_local_stock_geometry(rows):
    p=pp.descriptor('original',(rows,)*15);stock=gu.descriptor('original',rows)
    assert p['grid']==(*stock['grid'][:2],15)
    assert p['offsets']==tuple(i*rows for i in range(16))
    assert p['output_shape']==(15*rows,2304)
    assert p['logical_input_bindings']==20
    assert p['threadgroup_middle_bytes']==stock['threadgroup_middle_bytes']
    assert p['coefficient_copy_bytes']==p['raw_gate_up_global_bytes']==0


def test_q1_variable_row_segments_and_bounds():
    p=pp.descriptor('q1',(1,7,99,1024))
    assert p['offsets']==(0,1,8,107,1131)
    assert p['grid']==(72*256,1024,4)
    assert p['output_shape']==(1131,2304)
    assert p['logical_input_bindings']==9


@pytest.mark.parametrize('kind,counts',[
 ('original',()),('original',(1,)*16),('original',(1,2)),('original',(13,)),
 ('q1',(0,)),('q1',(1025,)),('q1',(True,)),('bad',(1,))])
def test_unproven_plan_refused(kind,counts):
    with pytest.raises(ValueError):pp.descriptor(kind,counts)


@pytest.mark.parametrize('kind,counts', [('original',(1,)*15),('original',(7,)*3),('original',(12,)*2),('q1',(1,7,99))])
def test_generated_sources_keep_one_unchanged_dot_core_and_offset_all_outputs(kind,counts):
    args=fixture(kind,counts);key,plan=pp._geometry(*args)
    spec=pp.kernel_spec(key,exp_spelling='metal::precise::exp',return_projections=True)
    ordinary=gu.kernel_spec(kind,selected_rows=counts[0] if kind=='original' else 1,
                           exp_spelling='metal::precise::exp',return_projections=True)
    assert spec['header']==ordinary['header']
    assert spec['input_names']==['x','rows','scores','offsets','inverse']+[f'expert{i}' for i in range(len(counts))]
    assert len(spec['input_names'])<=20
    assert spec['output_names']==['out','raw_gate','raw_up']
    assert spec['source'].count('switch(expert_slot)')==1
    assert spec['source'].count('case ')==len(counts)
    # Restore only indexing; every raw dot operation and epilogue stays verbatim.
    body=spec['source'].split('default: return;\n}\n',1)[1]
    if kind=='original' and counts[0]>1:
        body=body.replace('rows[row_base+min(vec0+v,GU_ROWS-1)]','rows[min(vec0+v,GU_ROWS-1)]')
        body=body.replace('scores[row_base+vec0+v]','scores[vec0+v]')
        body=body.replace('(row_base+vec0+sg)*2304','(vec0+sg)*2304')
        body=body.replace('(row_base+vec0+v)*2304','(vec0+v)*2304')
    else:
        body=body.replace('rows[row_base+selected_row]','rows[selected_row]')
        body=body.replace('scores[row_base+selected_row]','scores[selected_row]')
        body=body.replace('(row_base+selected_row)*2304','selected_row*2304')
    assert body==ordinary['source']
    if kind=='q1':assert 'if(threadgroup_position_in_grid.y>=row_count) return;' in spec['source']
    assert pp.supports(*args)


@pytest.mark.parametrize('change',['xshape','xdtype','rowshape','rowdtype','scores','plain','mixed','backing','countlength'])
def test_invalid_geometry_never_reaches_compilation(change,monkeypatch):
    x,experts,rows,scores,counts=fixture()
    if change=='xshape':x=arr((99,5121),mx.float32)
    elif change=='xdtype':x=arr((99,5120),mx.bfloat16)
    elif change=='rowshape':rows=arr((21,1),mx.uint32)
    elif change=='rowdtype':rows=arr((21,),mx.int32)
    elif change=='scores':scores=arr((21,),mx.bfloat16)
    elif change=='plain':experts[1]={}
    elif change=='mixed':experts[1]=packed('q1')
    elif change=='backing':experts[1]._backing=arr((1,),mx.uint8)
    elif change=='countlength':experts.pop()
    monkeypatch.setattr(pp,'_kernel',lambda *a:pytest.fail('invalid shape compiled'))
    assert not pp.supports(x,experts,rows,scores,counts)
    with pytest.raises(ValueError):pp.pp_middle(x,experts,rows,scores,counts)


def test_dispatch_binds_shared_input_and_original_owned_backings(monkeypatch):
    args=fixture('q1',(1,7,99));seen=[];outputs=(object(),object(),object())
    def kernel(key,spelling,diagnostic):
        assert all(isinstance(v,(int,str))for item in key[0]for v in item)
        assert key[1]==1 and spelling=='metal::precise::exp' and diagnostic
        def call(**kw):seen.append(kw);return outputs
        return call
    monkeypatch.setattr(pp,'_kernel',kernel)
    with mx.stream(mx.cpu):
        assert pp.pp_middle(*args,exp_spelling='metal::precise::exp',return_projections=True)==outputs
    call=seen[0];assert call['inputs'][:3]==[args[0],args[2],args[3]]
    assert call['inputs'][3].tolist()==[0,1,8,107]
    assert all(a is e.backing for a,e in zip(call['inputs'][5:],args[1]))
    assert call['output_shapes']==[(107,2304)]*3
    assert call['grid']==(72*256,99,3)


def test_q1_runtime_counts_not_in_compilation_key():
    assert pp._geometry(*fixture('q1',(1,7,99)))[0]==pp._geometry(*fixture('q1',(99,2,4)))[0]


def test_no_backing_retained_by_key():
    import gc,weakref
    class Backing:
        def __init__(self,n):self.shape=(n,);self.dtype=mx.uint8
    layout=canonical_layout('original');backing=Backing(layout.nbytes);reference=weakref.ref(backing)
    expert=PackedExpert(backing,layout,{})
    key,_=pp._geometry(arr((99,5120),mx.float32),[expert],arr((7,),mx.uint32),arr((7,),mx.float32),(7,))
    del expert,backing;gc.collect();assert reference() is None
    assert key[1]==7


@pytest.mark.parametrize('badkey',[((),1),((('original',0,1,2,-1),),1),((('original',0,1,2,3),('q1',0,1,2,3,4,5)),1)])
def test_bad_source_key_refused(badkey):
    with pytest.raises((ValueError,IndexError)):pp.kernel_spec(badkey)


def test_kernel_cache_covers_canonical_shapes_without_holding_backings():
    assert pp._kernel.cache_parameters()=={'maxsize':256,'typed':False}
    # Twelve original local-row laws times fifteen possible batch lengths,
    # plus fifteen Q1 batch lengths. Q1 runtime row counts do not specialize.
    assert 12*pp.MAX_EXPERTS+pp.MAX_EXPERTS < 256
