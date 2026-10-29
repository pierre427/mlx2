"""PP bank central dispatch/policy contracts; synthetic mx.cpu, no model/GPU."""
from contextlib import nullcontext
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU kernel forbidden'))
    yield
    mx.set_default_device(old)


def weights(rows=128):
    w=SourceV41Weights.__new__(SourceV41Weights)
    w.artifact={'args':SimpleNamespace(hidden_size=8,num_experts_per_tok=6)}
    w._ready=lambda:None;w.scheduled_expert_calls={}
    w.expert_stream=SimpleNamespace()
    w.source_resident_gather_policy='resident_bank_v1'
    w.source_resident_prefill_policy='expert_major_v1'
    calls=[]
    w.source_resident_bank_session=SimpleNamespace(active=True,replaying=False,max_token_rows=rows,
        moe=lambda *args:calls.append(args)or'gpu-routes')
    return w,calls


def test_explicit_pp_policy_requires_bank_and_existing_math():
    assert 'deepseek_source_resident_prefill'not in source_policy()
    with pytest.raises(ValueError,match='resident_bank_v1'):
        source_policy({'deepseek_source_resident_prefill':'expert_major_v1'})
    with pytest.raises(ValueError,match='routed_fused_fp32'):
        source_policy({'deepseek_source_resident_prefill':'expert_major_v1',
                       'deepseek_source_resident_gather':'resident_bank_v1'})
    value=source_policy({'deepseek_source_resident_prefill':'expert_major_v1',
        'deepseek_source_resident_gather':'resident_bank_v1','deepseek_source_expert_math':'routed_fused_fp32'})
    assert value['deepseek_source_resident_prefill']=='expert_major_v1'


@pytest.mark.parametrize('rows',[1,2,99,128])
def test_pp_early_dispatch_preserves_rows_and_never_materializes_host_routes(monkeypatch,rows):
    from mlx2.runtime.models import deepseek_v41_quant as q
    prepared=[]
    monkeypatch.setattr(q,'fused_quantize',lambda x,kind:prepared.append((x,kind))or x)
    w,calls=weights()
    assert w.moe(mx.ones((rows,8),mx.bfloat16),mx.zeros((rows,6),mx.uint32),
                 mx.ones((rows,6),mx.float32),'blk.3.ffn')=='gpu-routes'
    assert calls[0][0].shape==(rows,8)and calls[0][0].dtype==mx.float32
    assert calls[0][1].shape==calls[0][2].shape==(rows*6,)
    assert prepared[0][1]=='fp8'
    assert not hasattr(w,'source_route_host_materializations')
    assert w.scheduled_expert_calls['official_source_resident_bank_gpu_routes']==1


@pytest.mark.parametrize('limit,rows',[(1,2),(128,129)])
def test_decode_bound_and_pp_bound_fail_before_qat_or_dispatch(monkeypatch,limit,rows):
    from mlx2.runtime.models import deepseek_v41_quant as q
    monkeypatch.setattr(q,'fused_quantize',lambda *args:pytest.fail('invalid geometry prepared'))
    w,calls=weights(limit)
    if limit==1:del w.source_resident_bank_session.max_token_rows
    with pytest.raises(ValueError,match='geometry'):
        w.moe(mx.ones((rows,8)),mx.zeros((rows,6),mx.uint32),mx.ones((rows,6)),'blk.0.ffn')
    assert calls==[]


def test_preparation_blocks_all_forward_and_moe_entry():
    w,calls=weights();w.expert_stream._resident_bank_preparing={'generation':1}
    with pytest.raises(ValueError,match='preparation'):w.execution_scope_for_rows(99)
    with pytest.raises(ValueError,match='preparation'):
        w.moe(mx.ones((1,8)),mx.zeros((1,6),mx.uint32),mx.ones((1,6)),'blk.0.ffn')
    assert calls==[]


def test_unprepared_pp_refuses_but_private_replay_remains_ordinary():
    w,_=weights();w.source_resident_bank_session=None
    with pytest.raises(ValueError,match='explicitly prepared'):w.execution_scope_for_rows(99)
    w.source_resident_bank_session=SimpleNamespace(active=False,replaying=True)
    w._ordinary_execution_policy=lambda:nullcontext('ordinary')
    with w.execution_scope_for_rows(99)as value:assert value=='ordinary'


def test_pp_prepare_delegates_incremental_builder_only_explicitly(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_prefill_session as pp
    w,_=weights();w.source_resident_bank_session=None;seen=[]
    monkeypatch.setattr(pp,'prepare_prefill_session',lambda *a,**kw:seen.append((a,kw))or'pp-session')
    assert w.prepare_resident_bank(((0,1),),ownercheck=lambda:True,prefill=True)=='pp-session'
    assert seen[0][1]['max_bank_bytes']==3<<30
    assert seen[0][1]['budget_bytes']==75_000_000_000
    w.source_resident_prefill_policy='reference'
    with pytest.raises(ValueError,match='explicit expert-major policy'):
        w.prepare_resident_bank(((0,1),),ownercheck=lambda:True,prefill=True)
    assert len(seen)==1
