"""CPU-only geometry, ownership and mapped-dispatch gates; no Metal oracle."""
from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.gguf import source_gate_up_fusion as fused


def array(shape,dtype):
    return SimpleNamespace(shape=shape,ndim=len(shape),dtype=dtype)


def inputs(kind="q1",selected_rows=3,token_rows=99):
    x=array((token_rows,5120),mx.float32)
    rows=array((selected_rows,),mx.uint32)
    scores=array((selected_rows,),mx.float32)
    expert={}
    for p in ("gate","up"):
        expert[p+".qweight"]=array((2304,160 if kind=="q1" else 640),mx.uint32)
        expert[p+".scales"]=array((2304,80 if kind=="q1" else 160),mx.float16 if kind=="q1" else mx.uint8)
        if kind=="q1":expert[p+".biases"]=array((2304,80),mx.float16)
    return x,rows,scores,expert


@pytest.mark.parametrize("kind,rows",[("original",1),("q1",1),("q1",7),("q1",99),("q1",1024)])
def test_one_owner_per_output_and_complete_qat_groups(kind,rows):
    plan=fused.descriptor(kind,rows)
    owners=Counter()
    groups=[]
    for tile in range(plan['grid'][0]//plan['threadgroup'][0]):
        channels=[tile*32+sg*4+component for sg in range(8) for component in range(4)]
        groups.append(channels)
        owners.update(channels)
    assert owners==Counter(range(2304))
    assert all(len(g)==32 and max(g)//32==min(g)//32 for g in groups)
    assert plan['threadgroup_middle_bytes']==128
    assert plan['raw_gate_up_global_bytes']==0
    assert plan['grid'][1]==rows and plan['output_shape']==(rows,2304)


@pytest.mark.parametrize("kind,rows",[("original",0),("original",13),("original",99),("q1",0),("q1",1025),("q1",True),("bad",1)])
def test_unproven_or_invalid_geometry_refused(kind,rows):
    with pytest.raises(ValueError,match="Source gate/up fusion"):
        fused.descriptor(kind,rows)


@pytest.mark.parametrize("kind",["original","q1"])
def test_mapped_dispatch_binds_original_input_without_gather(kind,monkeypatch):
    x,rows,scores,expert=inputs(kind,1 if kind=="original" else 7)
    calls=[];result=object()
    def kernel(**kwargs):
        calls.append(kwargs)
        return [result]
    monkeypatch.setattr(fused,"_kernel",lambda *args:kernel)
    with mx.stream(mx.cpu):
        actual=fused.middle(x,rows,scores,expert,exp_spelling="metal::precise::exp")
    assert actual is result
    call=calls[0]
    assert call['inputs'][:3]==[x,rows,scores]
    assert all(a is b for a,b in zip(call['inputs'][3:-1],expert.values()))
    assert call['output_shapes']==[(rows.shape[0],2304)]
    assert call['output_dtypes']==[mx.float32]


@pytest.mark.parametrize("change",["x_width","x_type","row_type","score_shape","score_type","q_shape","q_type","scale_type","missing_bias","original_many_rows"])
def test_bad_input_refused_before_kernel(change,monkeypatch):
    x,rows,scores,expert=inputs("original" if change=="original_many_rows" else "q1",13 if change=="original_many_rows" else 3)
    if change=="x_width":x=array((99,5119),mx.float32)
    elif change=="x_type":x=array((99,5120),mx.bfloat16)
    elif change=="row_type":rows=array((3,),mx.int32)
    elif change=="score_shape":scores=array((3,1),mx.float32)
    elif change=="score_type":scores=array((3,),mx.bfloat16)
    elif change=="q_shape":expert['up.qweight']=array((2303,160),mx.uint32)
    elif change=="q_type":expert['up.qweight']=array((2304,160),mx.int32)
    elif change=="scale_type":expert['up.scales']=array((2304,80),mx.float32)
    elif change=="missing_bias":expert.pop('up.biases')
    def forbidden(*args,**kwargs):raise AssertionError("invalid input reached Metal")
    monkeypatch.setattr(fused,"_kernel",forbidden)
    assert not fused.supports(x,rows,scores,expert)
    with pytest.raises(ValueError,match="Source gate/up fusion"):
        fused.middle(x,rows,scores,expert)


def test_sigmoid_spelling_is_closed_set_and_source_has_no_raw_output():
    spec=fused.kernel_spec('q1',exp_spelling='metal::precise::exp')
    assert spec['output_names']==['out']
    assert 'metal::precise::exp' in spec['header']
    assert '__EXP__' not in spec['header']
    assert 'threadgroup float weighted[32]' in spec['source']
    assert spec['source'].count('out[')==1
    with pytest.raises(ValueError,match="sigmoid spelling"):
        fused.kernel_spec('q1',exp_spelling='fast::exp')


def packed(kind):
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
    layout=canonical_layout(kind)
    return PackedExpert(array((layout.nbytes,),mx.uint8),layout,{})


@pytest.mark.parametrize("count",range(1,7))
@pytest.mark.parametrize("formats",["original","q1","mixed"])
def test_batch_planes_binding_budget_and_payload_only_output(count,formats):
    kinds=[("q1" if i%2 else "original") if formats=="mixed" else formats for i in range(count)]
    experts=[packed(kind) for kind in kinds]
    x=array((1,5120),mx.float32);scores=array((6,),mx.float32);slots=array((count,),mx.uint32)
    key=fused._batch_geometry(x,experts,scores,slots)
    assert tuple(item[0] for item in key)==tuple(kinds)
    assert all(isinstance(v,(int,str)) for item in key for v in item)
    plan=fused.batch_descriptor(count)
    assert plan['grid'][2]==count and plan['output_shape']==(count,2304)
    assert plan['logical_input_bindings']==count+4<=10
    assert plan['expert_backing_bindings']==count
    assert fused.batch_supports(x,experts,scores,slots)
    spec=fused.batch_kernel_spec(key,exp_spelling="metal::precise::exp")
    assert len(spec['input_names'])==plan['logical_input_bindings']
    assert spec['input_names'][4:]==[f"expert{i}" for i in range(count)]
    assert spec['output_names']==['out'] and 'rows[' not in spec['source']
    assert spec['source'].count('case ')==count
    assert spec['source'].count('out[expert_slot*2304+tile*32+lane]')==count
    assert spec['source'].count('scores[slots[expert_slot]]')==count


def test_batch_dispatch_uses_one_backing_per_expert_and_metadata_key(monkeypatch):
    experts=[packed('original'),packed('q1'),packed('original')]
    x=array((1,5120),mx.float32);scores=array((6,),mx.float32);slots=array((3,),mx.uint32)
    seen=[];calls=[];result=object()
    def get_kernel(key,spelling):
        seen.append((key,spelling))
        def call(**kwargs):
            calls.append(kwargs)
            return [result]
        return call
    monkeypatch.setattr(fused,'_batch_kernel',get_kernel)
    with mx.stream(mx.cpu):
        assert fused.batch_middle(x,experts,scores,slots,exp_spelling='metal::precise::exp') is result
    call=calls[0]
    assert call['inputs'][:3]==[x,scores,slots]
    assert all(a is e.backing for a,e in zip(call['inputs'][4:],experts))
    assert call['output_shapes']==[(3,2304)]
    assert all(isinstance(v,(int,str)) for item in seen[0][0] for v in item)


@pytest.mark.parametrize("change",["empty","too_many","plain_mapping","backing_shape","backing_dtype","x_rows","score_shape","score_dtype","slot_count","slot_dtype"])
def test_batch_refuses_invalid_geometry_before_compilation(change,monkeypatch):
    experts=[packed('q1'),packed('original')]
    x=array((1,5120),mx.float32);scores=array((6,),mx.float32);slots=array((2,),mx.uint32)
    if change=='empty':experts=[];slots=array((0,),mx.uint32)
    elif change=='too_many':experts=[packed('q1') for _ in range(7)];slots=array((7,),mx.uint32)
    elif change=='plain_mapping':experts[0]={}
    elif change=='backing_shape':experts[0]._backing=array((1,),mx.uint8)
    elif change=='backing_dtype':experts[0]._backing=array((experts[0].layout.nbytes,),mx.uint32)
    elif change=='x_rows':x=array((2,5120),mx.float32)
    elif change=='score_shape':scores=array((6,1),mx.float32)
    elif change=='score_dtype':scores=array((6,),mx.bfloat16)
    elif change=='slot_count':slots=array((1,),mx.uint32)
    elif change=='slot_dtype':slots=array((2,),mx.int32)
    def forbidden(*args,**kwargs):raise AssertionError('invalid batch reached Metal')
    monkeypatch.setattr(fused,'_batch_kernel',forbidden)
    assert not fused.batch_supports(x,experts,scores,slots)
    with pytest.raises(ValueError,match="Source batched gate/up"):
        fused.batch_middle(x,experts,scores,slots)


def test_batch_kernel_metadata_is_closed_and_has_no_expert_cache_references():
    with pytest.raises(ValueError,match="layout metadata"):
        fused.batch_kernel_spec((('q1',0,1),))
    with pytest.raises(ValueError,match="layout metadata"):
        fused.batch_kernel_spec((('original',0,1,2,-1),))
    with pytest.raises(ValueError,match="sigmoid spelling"):
        fused.batch_kernel_spec((('original',0,1,2,3),),exp_spelling='other::exp')


def test_batch_layout_cache_deduplicates_descriptors_but_rechecks_backing():
    experts=[packed('original' if i%2==0 else 'q1') for i in range(6)]
    x=array((1,5120),mx.float32);scores=array((6,),mx.float32);slots=array((6,),mx.uint32)
    fused._batch_layout.cache_clear()
    try:
        first=fused._batch_geometry(x,experts,scores,slots)
        assert fused._batch_geometry(x,experts,scores,slots)==first
        stats=fused._batch_layout.cache_info()
        assert stats.misses==2 and stats.hits==10 and stats.currsize==2
        experts[0]._backing=array((1,),mx.uint8)
        assert not fused.batch_supports(x,experts,scores,slots)
    finally:
        fused._batch_layout.cache_clear()


def test_batch_layout_cache_never_retains_expert_or_backing():
    import gc
    import weakref
    from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
    class Backing:
        def __init__(self,n):self.shape=(n,);self.dtype=mx.uint8
    layout=canonical_layout('q1');backing=Backing(layout.nbytes)
    expert=PackedExpert(backing,layout,{})
    reference=weakref.ref(backing)
    fused._batch_layout.cache_clear()
    try:
        fused._batch_geometry(array((1,5120),mx.float32),[expert],array((1,),mx.float32),array((1,),mx.uint32))
        assert fused._batch_layout.cache_info().currsize==1
        del expert,backing
        gc.collect()
        assert reference() is None
    finally:
        fused._batch_layout.cache_clear()


def test_batch_layout_cache_has_a_fixed_bound():
    from itertools import islice,permutations
    from mlx2.adapters.deepseek_v41_source_buffers import ExpertLayout,TensorSlice,canonical_layout
    base=canonical_layout('original')
    fused._batch_layout.cache_clear()
    try:
        for order in islice(permutations(base.tensors),40):
            cursor=0;rows=[]
            for t in order:
                rows.append(TensorSlice(t.name,cursor,t.dtype,t.shape));cursor+=t.nbytes
            fused._batch_layout(ExpertLayout('original',tuple(rows)))
        stats=fused._batch_layout.cache_info()
        assert stats.maxsize==32 and stats.currsize==32 and stats.misses==40
    finally:
        fused._batch_layout.cache_clear()


@pytest.mark.parametrize("rows,vecs",[(2,2),(3,3),(4,4),(5,5),(6,3),(7,4),(8,4),(9,5),(10,5),(11,4),(12,4)])
def test_wide_plan_preserves_stock_virtual_rows_and_tails(rows,vecs):
    plan=fused.descriptor('original',rows)
    assert plan['vecs']==vecs and plan['vector_groups']==(rows+vecs-1)//vecs
    assert plan['threadgroup_middle_bytes']==vecs*32*4
    assert plan['output_phases']==2 and plan['grid'][1]==plan['vector_groups']
    owners=Counter()
    for tile in range(72):
        for group in range(plan['vector_groups']):
            for phase in range(2):
                for sg in range(8):
                    for lane in (0,16):
                        channel=tile*32+phase*16+sg*2+lane//16
                        stock_tg=channel//4;stock_sg=(channel%4)//2;stock_sg_row=channel%2
                        assert stock_tg==tile*8+phase*4+sg//2
                        assert stock_sg==sg%2 and stock_sg_row==lane//16
                        for v in range(vecs):
                            selected=group*vecs+v
                            read=min(selected,rows-1)
                            assert 0<=read<rows
                            if selected<rows:owners[(selected,channel)]+=1
    assert len(owners)==rows*2304 and set(owners.values())=={1}


@pytest.mark.parametrize("kind,rows",[('original',1),('original',2),('original',7),('original',12),('q1',7)])
def test_optional_raw_projection_sinks_precede_bf16_epilogue(kind,rows):
    plain=fused.kernel_spec(kind,selected_rows=rows)
    diagnostic=fused.kernel_spec(kind,selected_rows=rows,return_projections=True)
    assert plain['output_names']==['out']
    assert diagnostic['output_names']==['out','raw_gate','raw_up']
    assert 'raw_gate[' not in plain['source'] and 'raw_up[' not in plain['source']
    for name in ('raw_gate[','raw_up['):
        assert diagnostic['source'].count(name)==1
        assert diagnostic['source'].index(name)<diagnostic['source'].rindex('source_gu_weighted(')
    assert '__ROWS__' not in diagnostic['source'] and '__VECS__' not in diagnostic['source']


def test_single_and_batched_diagnostic_return_three_arrays(monkeypatch):
    expected=(object(),object(),object());calls=[]
    def kernel(**kwargs):
        calls.append(kwargs);return list(expected)
    monkeypatch.setattr(fused,'_kernel',lambda *args:kernel)
    monkeypatch.setattr(fused,'_batch_kernel',lambda *args:kernel)
    x,rows,scores,expert=inputs('original',7)
    with mx.stream(mx.cpu):
        assert fused.middle(x,rows,scores,expert,return_projections=True)==expected
        experts=[packed('original'),packed('q1')]
        assert fused.batch_middle(array((1,5120),mx.float32),experts,array((6,),mx.float32),
                                  array((2,),mx.uint32),return_projections=True)==expected
    assert calls[0]['output_shapes']==[(7,2304)]*3
    assert calls[1]['output_shapes']==[(2,2304)]*3
    assert all(c['output_dtypes']==[mx.float32]*3 for c in calls)


def test_diagnostic_flag_requires_a_real_boolean():
    with pytest.raises(ValueError,match="diagnostic flag"):
        fused.kernel_spec('q1',return_projections=1)
    with pytest.raises(ValueError,match="diagnostic flag"):
        fused.batch_kernel_spec((('original',0,1,2,3),),return_projections=1)
