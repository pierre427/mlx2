"""CPU operand/control-flow oracles; actual fused GPU body/model remain untested."""
from contextlib import contextmanager
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import mlx.core as mx

from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters import deepseek_v41_source_ffn_input as helper
from mlx2.runtime.models.deepseek_v41 import Block
from mlx2.runtime.models import deepseek_v41_quant as qat

KEY='deepseek_source_ffn_input'
P='blk.0'
G=P+'.ffn_gate_shexp.weight'
U=P+'.ffn_up_shexp.weight'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:(_ for _ in()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def owner():
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.source_ffn_input_policy=helper.POLICY;w.expert_math_policy='routed_fused_fp32'
    w.source_dense_qat_policy='fused_fp32';w.shared_policy='reference'
    w.source_completion_policy='bounded_2'  # Reuse must not force GPU blocking.
    w.artifact={'args':SimpleNamespace(hidden_size=5120,activation_dtype='bfloat16',simulate_quantization=True,num_experts_per_tok=2)}
    w._quant={name:(mx.zeros((64,1280),mx.uint32),mx.ones((64,160),mx.uint8))for name in(G,U)}
    w.scheduled_dense_calls={};w.scheduled_expert_calls={}
    w._ensure_dense=w._ready=lambda:None;w._closed=w._parked=False
    return w


def values(rows=1):
    # Exact BF16-grid values near E4M3 ties, zero, large/small scale blocks.
    pattern=np.array([0.,-0.,.25,-.25,1.,1.0625,1.1875,-1.3125,448.,450.,2**-40,-2**-40,2**40,-2**40,.5,-.5],np.float32)
    x=np.resize(pattern,(1,rows,5120))
    return mx.array(x).astype(mx.bfloat16).astype(mx.float32)


def test_default_is_reference_source_only_and_prerequisites():
    assert KEY not in source_policy()
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:'reference'})[KEY]=='reference'
    for opts in({KEY:helper.POLICY},{KEY:helper.POLICY,'deepseek_source_dense_qat':'fused_fp32'}):
        with pytest.raises(ValueError,match='requires'):source_policy(opts)
    opts={KEY:helper.POLICY,'deepseek_source_dense_qat':'fused_fp32','deepseek_source_expert_math':'routed_fused_fp32'}
    assert source_policy(opts)[KEY]==helper.POLICY
    with pytest.raises(ValueError):execution_policy(opts)


@pytest.mark.parametrize('bad',[None,True,1,{},[],'bad'])
def test_policy_malformed_refused(bad):
    with pytest.raises(ValueError):source_policy({KEY:bad})


@pytest.mark.parametrize('rows',[1,99])
def test_lazy_exact_rank_views_and_role_counters(rows):
    w=owner();x=values(rows);t=w.ffn_input_token(x,P);calls=[]
    expected=qat.quantize(x,'fp8');mx.eval(expected)
    assert t is not None and not calls
    def prepare():calls.append('routed');return qat.quantize(t.flat_input,'fp8')
    routed=t.routed(w,t.flat_input,P,prepare)
    shared=t.shared(w,x,G,U);mx.eval(routed,shared)
    assert calls==['routed']
    assert np.asarray(routed).reshape(expected.shape).tobytes()==np.asarray(expected).tobytes()
    assert np.asarray(shared).tobytes()==np.asarray(expected).tobytes()
    assert w.scheduled_dense_calls=={'source_ffn_input_qat_materialized':1,'source_ffn_input_qat_calls_avoided':1}
    t.close();assert t.raw_input is None and t.flat_input is None
    # Consumed graph outputs remain valid after the CPU token releases its refs.
    mx.eval(shared);assert np.asarray(shared).tobytes()==np.asarray(expected).tobytes()


@pytest.mark.parametrize('shape,dtype',[((1,2,5120),mx.float32),((2,1,5120),mx.float32),((1,1,4096),mx.float32),((1,1,5120),mx.bfloat16)])
def test_unsupported_factory_retains_ordinary_path(shape,dtype):
    w=owner();assert w.ffn_input_token(mx.zeros(shape,dtype),P) is None
    assert w.scheduled_dense_calls=={'source_ffn_input_reference_fallback':1}


def test_factory_reference_is_noop_before_inspecting_operand():
    w=owner();w.source_ffn_input_policy='reference'
    assert w.ffn_input_token(object(),object()) is None


@pytest.mark.parametrize('field,bad',[('source_dense_qat_policy','reference'),('expert_math_policy','reference'),('shared_policy','fused'),('expert_io_policy','overlap')])
def test_invalid_selected_configuration_refuses(field,bad):
    w=owner();setattr(w,field,bad)
    with pytest.raises(ValueError,match='configuration'):w.ffn_input_token(values(),P)


@pytest.mark.parametrize('bad',['blk.40','blk.00','blk.-1','other.0',None])
def test_alias_capability_refusal(bad):
    assert owner().ffn_input_token(values(),bad) is None


def test_nonquantized_or_replaced_shared_pair_refused():
    w=owner();x=values();t=w.ffn_input_token(x,P)
    w._quant[G]=(mx.ones((64,1280),mx.uint32),w._quant[G][1])
    with pytest.raises(ValueError,match='changed'):t.routed(w,t.flat_input,P,lambda:None)
    w._quant.pop(U);assert w.ffn_input_token(x,P) is None


def test_canonical_identity_role_expiry_and_aliases():
    w=owner();x=values();t=w.ffn_input_token(x,P)
    with pytest.raises(ValueError,match='role'):t.shared(w,x,G,U)
    with pytest.raises(ValueError,match='operand'):t.routed(w,mx.array(t.flat_input),P,lambda:None)
    with pytest.raises(ValueError,match='operand'):t.routed(w,t.flat_input,'blk.1',lambda:None)
    with pytest.raises(AttributeError):t.raw_input=mx.array(x)
    t.routed(w,t.flat_input,P,lambda:qat.quantize(t.flat_input,'fp8'))
    with pytest.raises(ValueError,match='role'):t.routed(w,t.flat_input,P,lambda:None)
    with pytest.raises(ValueError,match='operand'):t.shared(w,mx.array(x),G,U)
    with pytest.raises(ValueError,match='role'):t.shared(w,x,U,G)
    t.shared(w,x,G,U)
    with pytest.raises(ValueError,match='role'):t.shared(w,x,G,U)
    t.close();t.close()
    with pytest.raises(ValueError,match='closed'):t.shared(w,x,G,U)


def test_foreign_provider_and_thread_refuse_without_materializing():
    w=owner();t=w.ffn_input_token(values(),P)
    with pytest.raises(ValueError,match='owner'):t.routed(owner(),t.flat_input,P,lambda:None)
    with ThreadPoolExecutor(max_workers=1)as pool:
        f=pool.submit(t.routed,w,t.flat_input,P,lambda:None)
        with pytest.raises(ValueError,match='thread'):f.result()


def test_nested_frames_keep_independent_canonical_inputs_without_provider_stash():
    w=owner();x=values();y=mx.array(x);a=w.ffn_input_token(x,P);b=w.ffn_input_token(y,P)
    a.routed(w,a.flat_input,P,lambda:qat.quantize(a.flat_input,'fp8'))
    b.routed(w,b.flat_input,P,lambda:qat.quantize(b.flat_input,'fp8'))
    with pytest.raises(ValueError,match='operand'):a.shared(w,y,G,U)
    b.shared(w,y,G,U);b.close();result=a.shared(w,x,G,U);a.close();mx.eval(result)
    assert not any(isinstance(v,helper.FfnInputToken)for v in vars(w).values())
    assert w.scheduled_dense_calls['source_ffn_input_qat_materialized']==2
    assert w.scheduled_dense_calls['source_ffn_input_qat_calls_avoided']==2


@pytest.mark.parametrize('bad',[mx.bfloat16,'raises'])
def test_failed_materialization_has_no_shared_role_or_avoided_counter(bad):
    w=owner();t=w.ffn_input_token(values(),P)
    def callback():
        if bad=='raises':raise RuntimeError('materialization failure')
        return t.flat_input.astype(bad)
    with pytest.raises((ValueError,RuntimeError)):t.routed(w,t.flat_input,P,callback)
    assert not w.scheduled_dense_calls
    with pytest.raises(ValueError,match='role'):t.shared(w,t.raw_input,G,U)
    t.close()


def test_constructor_rejects_incompatible_selected_config_before_resources():
    with pytest.raises(ValueError,match='requires fused'):
        SourceV41Weights({},expert_cache_bytes=1,ffn_input_policy=helper.POLICY)
    with pytest.raises(ValueError,match='canonical'):
        SourceV41Weights({},expert_cache_bytes=1,ffn_input_policy=helper.POLICY,dense_qat_policy='fused_fp32',expert_math_policy='routed_fused_fp32')


@pytest.mark.parametrize('field,value',[('_closed',True),('_parked',True),('expert_math_policy','reference')])
def test_changed_lifetime_fences_before_readback_or_expert_window(field,value):
    w=owner();x=values();t=w.ffn_input_token(x,P);setattr(w,field,value)
    w.expert_stream=SimpleNamespace()
    with pytest.raises(ValueError,match='changed'):
        w.moe(t.flat_input,object(),object(),P,ffn_input_token=t)


def build_block(monkeypatch,rows,enabled,fail=None):
    """Actual Block/moe/shared/gate_up methods with CPU primitive substitutes."""
    from mlx2.runtime.models.deepseek_v4 import limited_swiglu
    from mlx2.runtime.gguf import source_middle_fusion,source_q1,source_down_fusion
    from mlx2.adapters import deepseek_v41_source_math as sm
    w=owner();w.source_ffn_input_policy=helper.POLICY if enabled else'reference'
    rng=np.random.default_rng(71);x=mx.array(rng.normal(size=(1,rows,5120)).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
    events=[];tokens=[];bank={};dense={};w.source_prefill_batch_policy='reference'
    for e in range(2):
        bank[e]={}
        for kind in('gate','up','down'):
            n,k=(5120,64)if kind=='down'else(64,5120)
            q=mx.zeros((n,k//(32 if e else 8)),mx.uint32);s=mx.ones((n,k//(64 if e else 32)),mx.float16 if e else mx.uint8)
            bank[e][kind+'.qweight']=q;bank[e][kind+'.scales']=s
            if e:bank[e][kind+'.biases']=s
            dense[id(q)]=mx.array(rng.normal(0,.01,size=(n,k)).astype(np.float32))
    for key in(G,U):dense[id(w._quant[key][0])]=mx.array(rng.normal(0,.01,size=(64,5120)).astype(np.float32))
    down=mx.array(rng.normal(0,.02,size=(5120,64)).astype(np.float32))
    @contextmanager
    def window():
        assert not w.expert_stream._active;w.expert_stream._active=True;events.append('window_enter')
        try:yield
        finally:w.expert_stream._active=False;events.append('window_exit')
    w.expert_stream=SimpleNamespace(_active=False,packed_buffers=False,stats={'lookups':0},window=window,
        acquire=lambda prefix,e:bank[e],observe_routes=lambda *a:events.append('readback'))
    def q(value,kind):
        events.append(('qat',value.shape,w.expert_stream._active))
        if fail=='routed':raise RuntimeError('routed failure')
        return qat.quantize(value,kind)
    def qmm(value,packed,*,scales,mode):
        if fail=='shared'and id(packed)in{id(w._quant[G][0]),id(w._quant[U][0])}:raise RuntimeError('shared failure')
        return value@dense[id(packed)].T
    monkeypatch.setattr(qat,'fused_quantize',q)
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    monkeypatch.setattr(source_q1,'matmul',lambda x,q,s,b:x@dense[id(q)].T)
    monkeypatch.setattr(source_q1,'paired_matmul_mapped',lambda x,rows,gq,gs,gb,uq,us,ub:
        (x[rows]@dense[id(gq)].T,x[rows]@dense[id(uq)].T))
    rounded=lambda v:v.astype(mx.bfloat16).astype(mx.float32)
    def fused_down(middles,experts,ids):
        result=mx.zeros((1,5120),mx.float32)
        for _,middle,e in sorted(zip(ids,middles,experts),key=lambda x:x[0]):
            result=result+rounded(middle@dense[id(e['down.qweight'])].T)
        return result
    monkeypatch.setattr(source_down_fusion,'down_combine',fused_down)
    monkeypatch.setattr(source_middle_fusion,'middle',lambda g,u,s,slots:qat.quantize(rounded(limited_swiglu(rounded(g),rounded(u))*s[slots,None]),'fp8'))
    def combine(parts,values,order):
        offsets=sm.ordered_offsets(values,order);flat=mx.concatenate(parts,axis=0);result=mx.zeros((rows,5120),mx.float32)
        for j in range(values.shape[1]):result=result+flat[mx.array(offsets[:,j],mx.uint32)]
        return result
    monkeypatch.setattr(sm,'ordered_combine',combine)
    factory=w.ffn_input_token
    def capture_token(*a):
        t=factory(*a)
        if t is not None:tokens.append(t)
        return t
    w.ffn_input_token=capture_token
    block=object.__new__(Block);block.w=w;block.c=w.artifact['args'];block.c.swiglu_limit=10;block.prefix=P+'.'
    block.model=SimpleNamespace(round=rounded,linear=lambda a,n:rounded(a@down.T))
    block.attention_half=lambda *a:(x,x,None,None,None)
    ids=mx.array(np.tile(np.array([[1,0]],np.uint32),(rows,1)))
    scores=mx.array(np.tile(np.array([[.25,.75]],np.float32),(rows,1)))
    def router(value):assert value is x;events.append('router_raw');return ids,scores
    block.router=router;block.finish=lambda h,r,s,*a:rounded(r+s)
    return block,w,x,events,tokens


@pytest.mark.parametrize('rows',[1,99])
def test_actual_block_and_provider_chain_one_less_input_qat_same_values(monkeypatch,rows):
    ordinary,w,x,events,_=build_block(monkeypatch,rows,False)
    expected,_=ordinary(x,None,None,None,{})
    mx.eval(expected);baseline=list(events)
    candidate,w,x,events,tokens=build_block(monkeypatch,rows,True)
    result,_=candidate(x,None,None,None,{})
    mx.eval(result)
    assert np.asarray(result).tobytes()==np.asarray(expected).tobytes()
    assert events[0]=='router_raw'
    assert sum(isinstance(v,tuple)and v[0]=='qat'for v in baseline)==2
    assert [v for v in events if isinstance(v,tuple)and v[0]=='qat']==[('qat',(rows,5120),True)]
    assert tokens[0].raw_input is None and tokens[0].flat_input is None
    assert w.scheduled_dense_calls['source_dense_gate_up_qat_reused']==1
    assert w.scheduled_dense_calls['source_ffn_input_qat_calls_avoided']==1
    assert w.scheduled_expert_calls['official_source_input_qat']==1
    assert events.index('window_enter')<next(i for i,v in enumerate(events)if isinstance(v,tuple))<events.index('window_exit')


@pytest.mark.parametrize('fail',['routed','shared'])
def test_actual_block_failure_closes_frame_and_releases_window(monkeypatch,fail):
    block,w,x,events,tokens=build_block(monkeypatch,1,True,fail)
    with pytest.raises(RuntimeError,match='failure'):block(x,None,None,None,{})
    assert not w.expert_stream._active
    assert len(tokens)==1 and tokens[0].raw_input is None and tokens[0].flat_input is None


def test_identity_binding_reference_unchanged_and_optin_explicit():
    a=object.__new__(DeepSeekV41SourceAdapter);a.profile='base';a.numerics_revision='numbers';a.kernel_policy={}
    a._bind_ffn_input_identity();assert(a.profile,a.numerics_revision)==('base','numbers')
    a.kernel_policy={KEY:helper.POLICY};a._bind_ffn_input_identity()
    assert 'ffninput-'in a.profile and helper.implementation_identity()in a.numerics_revision
