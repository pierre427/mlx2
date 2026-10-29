"""PP Q1 M1 policy/control/ownership tests; GPU arithmetic proof is separate."""
from types import SimpleNamespace
import numpy as np
import pytest
import mlx.core as mx

from mlx2.adapters import deepseek_v41_source_prefill_down as helper
from mlx2.adapters.deepseek_v41_source import source_policy, DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.adapters.deepseek_v41_source_buffers import HostExpert, canonical_layout, materialize
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.runtime.gguf import source_q1_down_owner, source_mxfp4_pair
from test_deepseek_v41_source_prefill_integration_cpu import case, run, expected
from test_deepseek_v41_source_batch_provider_cpu import _case, _expected

KEY='deepseek_source_prefill_down'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**kw):raise AssertionError('CPU test attempted Metal')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def args():
    return SimpleNamespace(hidden_size=5120,moe_intermediate_size=2304,n_routed_experts=384,
        num_hidden_layers=40,num_experts_per_tok=6,activation_dtype='bfloat16',simulate_quantization=True)


def packed(kind='q1'):
    layout=canonical_layout(kind)
    return materialize(HostExpert(np.zeros(layout.nbytes,np.uint8),layout))


def options():
    return {KEY:helper.POLICY,'deepseek_source_prefill_batch':'gate_up',
            'deepseek_source_expert_math':'routed_fused_fp32'}


def test_default_reference_adapter_only_and_required_profiles():
    assert KEY not in source_policy()
    assert source_policy({KEY:'reference'})[KEY]=='reference'
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy(options())[KEY]==helper.POLICY
    with pytest.raises(ValueError):execution_policy(options())
    for change in ({'deepseek_source_prefill_batch':'reference'},
                   {'deepseek_source_prefill_batch':'gate_up_down'},
                   {'deepseek_source_expert_math':'reference'}):
        with pytest.raises(ValueError,match='requires'):source_policy({**options(),**change})


@pytest.mark.parametrize('bad',[None,True,1,{},[],'bad'])
def test_bad_policy_refused(bad):
    with pytest.raises(ValueError):source_policy({KEY:bad})


@pytest.mark.parametrize('field,value', [('hidden_size',4096),('moe_intermediate_size',2048),
    ('n_routed_experts',256),('num_hidden_layers',41),('num_experts_per_tok',8),
    ('activation_dtype','float32'),('simulate_quantization',False)])
def test_canonical_configuration_guard(field,value):
    config=args();assert helper.configuration_supported(config)
    setattr(config,field,value);assert not helper.configuration_supported(config)
    with pytest.raises(ValueError,match='canonical'):
        SourceV41Weights({'args':config},expert_cache_bytes=75_000_000_000,
            expert_policy={'mode':'hot_original_q1'},expert_math_policy='routed_fused_fp32',
            prefill_batch_policy='gate_up',prefill_down_policy=helper.POLICY)


def test_no_q1_profile_and_bad_batch_refused_before_load():
    for policy in (None,{'mode':'original_lfu'}):
        with pytest.raises(ValueError,match='canonical'):
            SourceV41Weights({'args':args()},expert_cache_bytes=75_000_000_000,
                expert_policy=policy,expert_math_policy='routed_fused_fp32',
                prefill_batch_policy='gate_up',prefill_down_policy=helper.POLICY)
    with pytest.raises(ValueError,match='requires'):
        SourceV41Weights({},expert_cache_bytes=75_000_000_000,
            expert_math_policy='routed_fused_fp32',prefill_batch_policy='gate_up_down',prefill_down_policy=helper.POLICY)


@pytest.mark.parametrize('change',['none','rows','width','dtype','original','plain','scale','backing','decode'])
def test_exact_payload_geometry_and_device_guards(change,monkeypatch):
    data=packed();x=mx.zeros((1,2304),mx.float32);prefill_rows=99
    if change=='rows':x=mx.zeros((2,2304),mx.float32)
    elif change=='width':x=mx.zeros((1,2303),mx.float32)
    elif change=='dtype':x=x.astype(mx.bfloat16)
    elif change=='original':data=packed('original')
    elif change=='plain':data=dict(data)
    elif change=='scale':data._tensors['down.scales']=mx.zeros((5120,36),mx.float32)
    elif change=='backing':data._backing=mx.zeros((1,),mx.uint8)
    elif change=='decode':prefill_rows=1
    assert helper.eligible(x,data,prefill_rows=prefill_rows,check_device=False) is (change=='none')
    # CPU declines even a valid canonical payload, without a device probe.
    assert helper.eligible(x,data,prefill_rows=prefill_rows) is False
    if change=='none':
        monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
        monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:False)
        assert helper.eligible(x,data,prefill_rows=99) is False
        monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
        assert helper.eligible(x,data,prefill_rows=99) is True


def test_helper_preserves_exact_inputs_and_already_owned_return(monkeypatch):
    data=packed();x=mx.zeros((1,2304),mx.float32);calls=[];sentinel=object();counts={}
    monkeypatch.setattr(helper,'eligible',lambda *a,**kw:True)
    def project(value,q,s,b):calls.append((value,q,s,b));return sentinel
    monkeypatch.setattr(source_q1_down_owner,'project',project)
    out=helper.maybe_project(x,data,policy=helper.POLICY,expert_math='routed_fused_fp32',
        prefill_batch='gate_up',prefill_rows=99,counters=counts)
    assert out is sentinel and calls==[(x,data['down.qweight'],data['down.scales'],data['down.biases'])]
    assert counts=={helper.COUNTER:1}
    assert helper.maybe_project(object(),object(),policy='reference',expert_math='bad',prefill_batch='bad',prefill_rows=1,counters={}) is None


def test_identity_binds_policy_and_implementation_without_default_drift(monkeypatch):
    ids=[]
    for selected,revision in [(False,'first'),(True,'first'),(True,'second')]:
        adapter=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter)
        adapter.kernel_policy={KEY:helper.POLICY if selected else 'reference'}
        adapter.numerics_revision='source-reference';adapter.profile='profile-reference'
        monkeypatch.setattr(helper,'implementation_identity',lambda:revision)
        adapter._bind_prefill_down_identity()
        ids.append((adapter.numerics_revision,adapter.profile))
    assert ids[0]==('source-reference','profile-reference') and len(set(ids))==3


def test_reviewed_kernel_pin_and_metadata_only_layout_cache(monkeypatch):
    helper.implementation_identity.cache_clear()
    assert len(helper.implementation_identity())==64
    assert helper._q1_layout() is helper._q1_layout()
    monkeypatch.setattr(helper,'SOURCE_PINS',{'source_q1_down_owner.py':'bad'})
    helper.implementation_identity.cache_clear()
    with pytest.raises(RuntimeError,match='re-audit'):helper.implementation_identity()
    helper.implementation_identity.cache_clear()


class Owned:
    def __init__(self,value):self.value=value
    def astype(self,*args,**kwargs):raise AssertionError('Redundant owner cast on fused DOWN output')


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('kind',['mixed','q1'])
def test_real_moe_provider_keeps_owned_output_and_order_without_extra_acquires(monkeypatch,owned,kind):
    ids=[[9,0,7,2,6,1],[9,0,7,6,5,1],[9,0,7,6,5,3]]
    c=case(monkeypatch,ids,owned=owned,kind=kind);c.w.source_prefill_down_policy=helper.POLICY
    identity=lambda d:next(e for e,v in c.bank.items()if v is d)
    selected=[];owned_values=[];passed_parts=[]
    def eligible(x,data,*,prefill_rows,**kw):
        assert prefill_rows==3 and c.store._active and len(c.store._leased)==len(c.bank)
        return x.shape==(1,2304)and data.layout.kind=='q1'
    monkeypatch.setattr(helper,'eligible',eligible)
    def project(x,q,s,b):
        e=next(e for e,d in c.bank.items()if d['down.qweight']is q)
        selected.append(e)
        value=mx.repeat(((e+1)*x[:,0])[:,None],5120,axis=1).astype(mx.bfloat16).astype(mx.float32)
        result=Owned(value);owned_values.append(result);return result
    monkeypatch.setattr(source_q1_down_owner,'project',project)
    before=assembly.ordered_combine
    def combine(parts,values,order):
        passed_parts.extend(parts)
        return before([p.value if isinstance(p,Owned)else p for p in parts],values,order)
    monkeypatch.setattr(assembly,'ordered_combine',combine)
    result=run(c)
    want=[e for e,d in c.bank.items()if d.layout.kind=='q1'and np.count_nonzero(c.ids==e)==1]
    assert selected==want and all(any(p is o for p in passed_parts)for o in owned_values)
    assert result.tobytes()==expected(c).tobytes()
    assert c.acquired==[('blk.0',e)for e in sorted(c.bank)]
    assert c.w.scheduled_expert_calls[helper.COUNTER]==len(want)
    assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==len(c.bank)-len(want)
    assert c.w.scheduled_expert_calls['official_source_prefill_assembly_parts']==len(c.bank)
    assert all(e not in selected for e,name,m in c.calls['projections']if name=='down')
    if owned:assert not c.store._active and not c.store._leased and len(c.fences)==1
    else:assert c.store._active and len(c.store._leased)==len(c.bank)and not c.fences


@pytest.mark.parametrize('owned',[True,False])
def test_candidate_failure_retains_or_drains_existing_lease(monkeypatch,owned):
    c=case(monkeypatch,[[0,1,2,3,4,5],[0,1,2,3,4,6]],owned=owned,kind='q1')
    c.w.source_prefill_down_policy=helper.POLICY
    monkeypatch.setattr(helper,'eligible',lambda x,data,**kw:x.shape==(1,2304))
    def fail(*a,**kw):
        assert c.store._active and c.store._leased
        raise RuntimeError('candidate submit failure')
    monkeypatch.setattr(source_q1_down_owner,'project',fail)
    with pytest.raises(RuntimeError,match='candidate submit'):run(c)
    assert helper.COUNTER not in c.w.scheduled_expert_calls
    if owned:assert not c.store._active and not c.store._leased and len(c.fences)==1
    else:assert c.store._active and c.store._leased and not c.fences


def test_decode_fastpath_never_calls_prefill_owner_policy(monkeypatch):
    c=_case(monkeypatch);c.w.source_prefill_down_policy=helper.POLICY;c.w.source_prefill_batch_policy='gate_up'
    def forbidden(*a,**kw):raise AssertionError('Decode reached PP-only owner policy')
    monkeypatch.setattr(helper,'maybe_project',forbidden)
    value=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(value)
    assert np.array(value).tobytes()==_expected(c).tobytes()
    assert len(c.calls['down'])==1 and helper.COUNTER not in c.w.scheduled_expert_calls
