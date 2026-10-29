"""CPU fake-provider Source pressure routing and completed-lease ownership."""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters.deepseek_v41_source_memory import ORIGINAL_EXPERT_BYTES
from test_deepseek_v41_source_batch_provider_cpu import _case, _expected


@pytest.fixture(autouse=True)
def forbid_gpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def adaptive_case(monkeypatch, *, rows=1, available=0):
    ids=np.tile(np.array([[9,1,7,2,6,0]],np.uint32),(rows,1))
    c=_case(monkeypatch,owned=True,ids=ids)
    stream=c.store
    stream.adaptive_residency_enabled=True;stream.maximum_cache_bytes=75_000_000_000
    stream.minimum_cache_bytes=ORIGINAL_EXPERT_BYTES;stream._cache_bytes=0
    stream._adaptive_staging=ORIGINAL_EXPERT_BYTES;stream._adaptive_available=lambda:available
    stream._completion_collector=None;released=[];updates=[]
    def update(**kwargs):
        return dict(enabled=True,availability_known=available is not None,measured_available_bytes=available,
                    effective_payload_target_bytes=ORIGINAL_EXPERT_BYTES)
    stream.adaptive_residency_update=update
    def target(value,*,working_set_experts):
        updates.append((value,working_set_experts))
        return dict(effective_payload_target_bytes=max(ORIGINAL_EXPERT_BYTES,value-ORIGINAL_EXPERT_BYTES),
                    imminent_expert_window_bytes=working_set_experts*ORIGINAL_EXPERT_BYTES)
    stream.set_adaptive_cache_target=target
    old_acquire=stream.acquire
    def acquire(prefix,expert):
        result=old_acquire(prefix,expert)
        stream._leased.pop(expert)
        stream._leased[(int(prefix.split('.')[1]),expert)]=result
        return result
    stream.acquire=acquire
    def release(keys,*,roots):
        assert stream._active and stream._completion_collector is None
        assert keys[0] in stream._leased and len(stream._leased)==1
        mx.eval(*roots)
        assert np.isfinite(np.asarray(roots[0])).all()
        released.append((keys,np.array(roots[0])))
        del stream._leased[keys[0]]
    stream.release_completed_experts=release
    def qmm(value,weight,*,scales,mode):
        expert=next(k for k,data in c.bank.items()if data['down.qweight']is weight)
        return mx.repeat(((expert+1)*value[:,0])[:,None],5120,axis=1)
    monkeypatch.setattr(mx,'quantized_matmul',qmm)
    c.w.source_prefill_batch_policy='gate_up'
    c.w.source_frame_reuse_policy='reference'
    return c,released,updates


@pytest.mark.parametrize('rows',[1,2,7])
def test_pressure_uses_one_complete_expert_and_preserves_rows_scores_and_order(monkeypatch,rows):
    c,released,updates=adaptive_case(monkeypatch,rows=rows,available=0)
    c.w._prefill_batched_gate_up=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Bulk under pressure'))
    result=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(result)
    np.testing.assert_array_equal(np.array(result),_expected(c))
    assert [keys[0][1]for keys,_ in released]==sorted(c.bank)
    assert all(part.shape[0]==rows for _,part in released)
    assert all(count==1 for _,count in updates)
    assert not c.calls['batch'] and not c.calls['down']
    assert c.w.scheduled_expert_calls['source_adaptive_completed_expert_parts']==len(c.bank)
    assert c.w.scheduled_expert_calls['source_adaptive_sequential_windows']==1
    assert not c.store._leased


def test_measured_room_retains_existing_fused_decode(monkeypatch):
    c,released,updates=adaptive_case(monkeypatch,available=5<<30)
    result=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(result)
    np.testing.assert_array_equal(np.array(result),_expected(c))
    assert not released and c.calls['batch'] and c.calls['down']
    assert updates[-1][1]==6
    assert c.w.scheduled_expert_calls['source_adaptive_bulk_windows']==1


def test_adaptive_scope_disables_key_epoch_collector_without_changing_reference(monkeypatch):
    c,_,_=adaptive_case(monkeypatch)
    c.w.source_completion_policy='bounded_2';c.w._source_completion=None
    with c.w.execution_scope():
        assert c.w._source_completion is None
    assert c.w.scheduled_expert_calls['source_adaptive_ordinary_completion_executions']==1
    c.w._source_completion=SimpleNamespace(active=True)
    with pytest.raises(ValueError,match='active completion'):c.w.execution_scope()
    c.w._source_completion=None;c.store._completion_collector=object()
    plan=c.w._adaptive_expert_plan(np.asarray(c.ids),'blk.0')
    assert plan['completion_pressure_deferred'] and plan['expert_strategy']!='sequential_expert'
    assert c.w._source_adaptive_force_ordinary
    assert c.w.scheduled_expert_calls['source_adaptive_pressure_deferred_to_next_forward']==1


@pytest.mark.parametrize('available,expected',[(None,1),(0,1),(4<<30,128)])
def test_prefill_advice_uses_live_room_and_unknown_defaults_to_one(monkeypatch,available,expected):
    c,_,_=adaptive_case(monkeypatch,available=available)
    assert c.w.recommended_prefill_rows(128,context_tokens=4096)==expected
    assert c.w.source_memory_advice['complete_layer_minimum_bytes']==0
    assert c.w.source_memory_advice['fixed_host_reserve_bytes']==0
    assert c.w.source_memory_advice['dense_estimate_is_admission_requirement']is False


def test_unknown_callback_failure_selects_minimum_without_throwing(monkeypatch):
    c,_,_=adaptive_case(monkeypatch)
    c.store._adaptive_available=lambda:(_ for _ in ()).throw(RuntimeError('telemetry unavailable'))
    assert c.w.recommended_prefill_rows(128)==1


def test_high_room_preserves_bounded_before_any_expert_acquire(monkeypatch):
    c,_,_=adaptive_case(monkeypatch,available=32<<30)
    c.w.source_completion_policy='bounded_2';c.w._source_completion=None
    calls=[]
    c.w._ordinary_execution_policy=lambda:(calls.append(True) or nullcontext())
    with c.w.execution_scope_for_rows(128):pass
    assert calls==[True]
    assert c.w.source_completion_memory_advice['worst_unique_experts']==384
    assert c.w.source_completion_memory_advice['selected_completion']=='bounded_2'
    assert c.w.scheduled_expert_calls['source_adaptive_bounded_completion_executions']==1
    c.w._source_adaptive_force_ordinary=True
    with c.w.execution_scope_for_rows(1):pass
    assert calls==[True]  # a mid-forward pressure event forces the next scope ordinary


def test_large_prefill_worst_case_selects_ordinary_despite_room_for_six(monkeypatch):
    c,_,_=adaptive_case(monkeypatch,available=1<<30)
    c.w.source_completion_policy='bounded_2';c.w._source_completion=None
    c.w._ordinary_execution_policy=lambda:(_ for _ in ()).throw(AssertionError('bounded under pressure'))
    with c.w.execution_scope_for_rows(100):pass
    assert c.w.source_completion_memory_advice['selected_completion']=='reference'
