"""Tiny Q1 integration orchestration/ownership only; no GPU/model payload."""
from types import SimpleNamespace as NS
import weakref
import numpy as np
import pytest
import mlx.core as mx

from mlx2.adapters import deepseek_v41_source_prefill_down as helper
from mlx2.adapters import deepseek_v41_source_buffers as buffers
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.runtime.gguf import source_down_prefill,source_prefill_combine,source_gate_up_prefill
from test_deepseek_v41_source_prefill_integration_cpu import case,run,expected
from test_deepseek_v41_source_batch_provider_cpu import _case,_expected

KEY='deepseek_source_prefill_down'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    prior=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('CPU test reached Metal'))
    yield
    mx.set_default_device(prior)


def options():return{KEY:helper.TINY_POLICY,'deepseek_source_prefill_batch':'gate_up','deepseek_source_expert_math':'routed_fused_fp32'}


def test_policy_is_explicit_alternative_and_native_profile_refuses():
    assert KEY not in source_policy()
    assert source_policy(options())[KEY]==helper.TINY_POLICY
    assert source_policy({**options(),KEY:helper.POLICY})[KEY]==helper.POLICY
    for changed in ({'deepseek_source_prefill_batch':'reference'},{'deepseek_source_prefill_batch':'gate_up_down'},
                    {'deepseek_source_expert_math':'reference'},{'deepseek_source_native_mx':'original_pp_half_v1'}):
        with pytest.raises(ValueError):source_policy({**options(),**changed})


@pytest.mark.parametrize('counts,expected',[((),None),((1,2,3),(3,6)),((1,2,3,4,9),(3,6)),((4,5),(0,0))])
def test_prefix_is_whole_sorted_job_prefix(counts,expected):
    offsets=tuple(np.cumsum((0,*counts)).tolist())
    batch=NS(kind='q1',row_counts=counts,offsets=offsets,job_indices=tuple(range(len(counts))))
    if expected is None:
        with pytest.raises(ValueError):helper.tiny_prefix(batch,prefill_rows=99)
    else:assert helper.tiny_prefix(batch,prefill_rows=99)==expected
    assert helper.tiny_prefix(batch,prefill_rows=1)==(0,0)
    batch.kind='original';assert helper.tiny_prefix(batch,prefill_rows=99)==(0,0)


@pytest.mark.parametrize('change',['unsorted','offset','duplicate','bool','big','negativejob'])
def test_prefix_malformed_ownership_fails_closed(change):
    b=NS(kind='q1',row_counts=(1,2,3),offsets=(0,1,3,6),job_indices=(2,1,0))
    if change=='unsorted':b.row_counts=(2,1,3)
    elif change=='offset':b.offsets=(0,1,4,6)
    elif change=='duplicate':b.job_indices=(0,0,1)
    elif change=='bool':b.row_counts=(True,2,3)
    elif change=='big':b.row_counts=(1,2,1025)
    else:b.job_indices=(-1,0,1)
    with pytest.raises(ValueError):helper.tiny_prefix(b,prefill_rows=99)


class Owned:
    def __init__(self,value):self.value=value;self.shape=value.shape
    def __getitem__(self,slice):return Owned(self.value[slice])
    def astype(self,*a,**k):raise AssertionError('Tiny output got a redundant owning cast')


def routes(counts):
    flat=np.concatenate([np.full(n,e,np.uint32)for e,n in counts])
    assert len(flat)%6==0
    return flat.reshape(-1,6)[:,::-1].copy()


def install_tiny(c,monkeypatch,*,failure=False):
    # Backend stays CPU. Only the pure metadata eligibility predicate is mocked
    # to name GPU; every arithmetic provider is a CPU oracle and Metal forbidden.
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    monkeypatch.setattr(helper,'tiny_implementation_identity',lambda:'reviewed')
    c.w.source_prefill_down_policy=helper.TINY_POLICY
    for data in c.bank.values():
        class Backing:
            ndim=1;dtype=mx.uint8;shape=(data.layout.nbytes,)
        data._backing=Backing();data._compact_issuer=buffers._COMPACT_UPLOAD_ISSUER
        data._compact_backing=weakref.ref(data._backing)
    captured=[];views=[];packed=[]
    before_gu=source_gate_up_prefill.pp_middle
    def gu(*a,**k):
        out=before_gu(*a,**k);packed.append(out);return out
    monkeypatch.setattr(source_gate_up_prefill,'pp_middle',gu)
    def down(x,data,counts):
        assert c.store._active and len(c.store._leased)==len(c.bank)
        if failure:raise RuntimeError('tiny batch submission failure')
        eids=[next(e for e,v in c.bank.items()if v is d)for d in data]
        captured.append((x,tuple(data),tuple(counts),tuple(eids)))
        offsets=np.cumsum((0,*counts));parts=[]
        for e,a,b in zip(eids,offsets[:-1],offsets[1:]):
            parts.append(mx.repeat(((e+1)*x[a:b,0])[:,None],5120,axis=1).astype(mx.bfloat16).astype(mx.float32))
        return Owned(mx.concatenate(parts,axis=0))
    monkeypatch.setattr(source_down_prefill,'pp_down',down)
    original=assembly.ordered_combine
    def combine(parts,values,order,**kw):
        views.extend(parts)
        return original([p.value if isinstance(p,Owned)else p for p in parts],values,order)
    monkeypatch.setattr(assembly,'ordered_combine',combine)
    monkeypatch.setattr(source_prefill_combine,'combine_mapped',lambda *a,**k:pytest.fail('Tiny used mapped assembly'))
    monkeypatch.setattr(helper,'maybe_project',lambda *a,**k:pytest.fail('Tiny invoked old M1 owner policy'))
    return captured,views,packed


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('kind',['mixed','q1'])
@pytest.mark.parametrize('shape',['partial','all','none','many'])
def test_actual_moe_restores_views_and_preserves_fallbacks_and_leases(monkeypatch,owned,kind,shape):
    if shape=='partial':ids=routes([(0,5),(1,4),(2,3),(3,2),(4,1),(5,1),(6,1),(7,1)])
    elif shape=='all':ids=routes([(9,3),(1,2),(7,1),(5,3),(3,2),(11,1)])
    elif shape=='none':ids=routes([(0,4),(1,4),(2,4)])
    else:ids=np.arange(36,dtype=np.uint32).reshape(6,6)[:,::-1].copy()
    c=case(monkeypatch,ids,owned=owned,kind=kind)
    captured,views,packed=install_tiny(c,monkeypatch)
    result=run(c)
    assert result.tobytes()==expected(c).tobytes()
    tiny={e for e,data in c.bank.items()if data.layout.kind=='q1'and np.count_nonzero(c.ids==e)<=3}
    assert {e for *_,eids in captured for e in eids}==tiny
    assert c.acquired==[('blk.0',e)for e in sorted(c.bank)]
    assert [e for e,name,_ in c.calls['projections']if name=='down']==[e for e in sorted(c.bank)if e not in tiny]
    assert len(views)==len(c.bank)and sum(isinstance(v,Owned)for v in views)==len(tiny)
    assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==len(c.bank)-len(tiny)
    assert c.w.scheduled_expert_calls['official_source_prefill_down_batches']==len(captured)
    if tiny:
        assert c.w.scheduled_expert_calls[helper.TINY_COUNTER+'jobs']==len(tiny)
        assert c.w.scheduled_expert_calls[helper.TINY_COUNTER+'rows']==sum(np.count_nonzero(c.ids==e)for e in tiny)
        assert c.w.scheduled_expert_calls[helper.TINY_COUNTER+'output_views']==len(tiny)
    else:assert helper.TINY_COUNTER+'jobs'not in c.w.scheduled_expert_calls
    for activation,data,counts,eids in captured:
        batch=next(v for v in c.calls['prefill']if all(e in v[0]for e in eids))
        assert counts==batch[3][:len(counts)]and all(n<=3 for n in counts)
        owner=packed[c.calls['prefill'].index(batch)]
        if len(counts)==len(batch[3]):assert activation is owner
        else:assert np.array(activation).tobytes()==np.array(owner[:sum(counts)]).tobytes()
    if owned:assert not c.store._active and not c.store._leased and len(c.fences)==1
    else:assert c.store._active and c.store._leased and not c.fences


@pytest.mark.parametrize('owned',[True,False])
def test_batch_failure_preserves_existing_completion_cleanup(monkeypatch,owned):
    c=case(monkeypatch,np.arange(12,dtype=np.uint32).reshape(2,6),owned=owned,kind='q1')
    install_tiny(c,monkeypatch,failure=True)
    with pytest.raises(RuntimeError,match='tiny batch submission'):run(c)
    assert helper.TINY_COUNTER+'batches'not in c.w.scheduled_expert_calls
    if owned:assert not c.store._active and not c.store._leased and len(c.fences)==1
    else:assert c.store._active and c.store._leased and not c.fences


def test_decode_never_calls_prefix_or_tiny_kernel(monkeypatch):
    c=_case(monkeypatch);c.w.source_prefill_down_policy=helper.TINY_POLICY;c.w.source_prefill_batch_policy='gate_up'
    monkeypatch.setattr(helper,'tiny_prefix',lambda *a,**k:pytest.fail('Decode reached PP tiny prefix'))
    monkeypatch.setattr(helper,'maybe_tiny_project',lambda *a,**k:pytest.fail('Decode reached PP tiny DOWN'))
    output=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(output)
    assert np.array(output).tobytes()==_expected(c).tobytes()
    assert len(c.calls['down'])==1


def test_selected_policy_identity_differs_and_source_pin_is_enforced(monkeypatch):
    identities=[]
    for policy,identity in [('reference','a'),(helper.POLICY,'a'),(helper.TINY_POLICY,'a'),(helper.TINY_POLICY,'b')]:
        a=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter);a.kernel_policy={KEY:policy}
        a.numerics_revision='reference';a.profile='reference'
        monkeypatch.setattr(helper,'implementation_identity',lambda:identity)
        monkeypatch.setattr(helper,'tiny_implementation_identity',lambda:identity)
        a._bind_prefill_down_identity();identities.append((a.numerics_revision,a.profile))
    assert len(set(identities))==4


def test_tiny_selection_cannot_fallthrough_to_old_owner_helper():
    assert helper.maybe_project(object(),object(),policy=helper.TINY_POLICY,expert_math='routed_fused_fp32',
        prefill_batch='gate_up',prefill_rows=99,counters={})is None


def test_unproven_batch_falls_back_to_ordinary_jobs_without_lost_middle(monkeypatch):
    c=case(monkeypatch,np.arange(12,dtype=np.uint32).reshape(2,6),kind='q1')
    captured,views,_=install_tiny(c,monkeypatch)
    c.bank[0]._compact_issuer=None
    assert run(c).tobytes()==expected(c).tobytes()
    assert not captured and not any(isinstance(p,Owned)for p in views)
    assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==12


def test_tiny_kernel_source_pin_is_enforced(monkeypatch):
    helper.tiny_implementation_identity.cache_clear()
    assert len(helper.tiny_implementation_identity())==64
    monkeypatch.setattr(helper,'TINY_SOURCE_PINS',{'source_down_prefill.py':'bad'})
    helper.tiny_implementation_identity.cache_clear()
    try:
        with pytest.raises(RuntimeError,match='re-audit'):helper.tiny_implementation_identity()
    finally:helper.tiny_implementation_identity.cache_clear()


def test_dense_and_adapter_receipts_bind_selected_tiny_contract_and_counters(monkeypatch):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
    monkeypatch.setattr(helper,'tiny_implementation_identity',lambda:'tiny-hash')
    monkeypatch.setattr(helper,'implementation_identity',lambda:'old-m1-hash')
    monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.source_prefill_down_policy=helper.TINY_POLICY
    w.scheduled_expert_calls={helper.TINY_COUNTER+'jobs':12,'unrelated':5}
    w.expert_math_policy='routed_fused_fp32';w.source_qakv_policy='reference';w.source_q_b_prefill_policy='reference'
    dense=w.dense_receipt()
    assert dense['source_prefill_down_implementation_sha256']=='tiny-hash'
    assert dense['source_prefill_down_contract']==helper.TINY_CONTRACT
    assert dense['source_prefill_down_counters']=={helper.TINY_COUNTER+'jobs':12}
    monkeypatch.setattr(DeepSeekV41MLXAdapter,'diagnostics',lambda self:{})
    a=DeepSeekV41SourceAdapter.__new__(DeepSeekV41SourceAdapter);a.weights=w;a.kernel_policy={KEY:helper.TINY_POLICY}
    a.model=NS(source_hc_boundary_policy='reference',index_policy='reference',selected_kv_policy='reference',
        source_attention_math_policy='reference',scheduled_hc_calls={},hc_coeff_policy='reference')
    receipt=a.diagnostics()
    assert receipt['source_prefill_down_implementation_sha256']=='tiny-hash'
    assert receipt['source_prefill_down_contract']==helper.TINY_CONTRACT
    assert receipt['source_prefill_down_counters']==dense['source_prefill_down_counters']


def test_direct_constructor_identity_branch_uses_tiny_hash(monkeypatch):
    import ast
    from pathlib import Path
    import mlx2.adapters.deepseek_v41_source_weights as module
    tree=ast.parse(Path(module.__file__).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SourceV41Weights')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
    branch=next(n for n in init.body if isinstance(n,ast.If)
        and ast.unparse(n.test)=="prefill_down_policy != 'reference'"
        and any(isinstance(x,ast.Assign)and any('compute_policy'in ast.unparse(t)for t in x.targets)for x in n.body))
    monkeypatch.setattr(helper,'tiny_implementation_identity',lambda:'tiny-hash')
    monkeypatch.setattr(helper,'implementation_identity',lambda:pytest.fail('Tiny used old identity'))
    owner=NS(compute_policy='base')
    scope={'self':owner,'prefill_down_policy':helper.TINY_POLICY,'__package__':'mlx2.adapters'}
    exec(compile(ast.Module(body=[branch],type_ignores=[]),module.__file__,'exec'),scope)
    assert owner.compute_policy=='base:source_prefill_down=q1_tiny_batch_v1:tiny-hash'
