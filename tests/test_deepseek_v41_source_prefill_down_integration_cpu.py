"""Provider down-batch/mapped-assembly ownership, with CPU kernel oracles.

Actual down-dot parity is the separate GPU component proof. These fixtures
reuse metadata-only PackedExperts and the real NativeExpertStore.window.
"""
import numpy as np
import mlx.core as mx
import pytest

from test_deepseek_v41_source_prefill_integration_cpu import case as gu_case,expected,run
from mlx2.runtime.gguf import source_gate_up_prefill as gu
from mlx2.runtime.gguf import source_down_prefill as down
from mlx2.runtime.gguf import source_prefill_combine as assembly
from mlx2.runtime.gguf import source_middle_fusion as middle
from mlx2.adapters import deepseek_v41_mlx_weights as native


def case(monkeypatch,ids,*,owned=True,kind='mixed',cancellation=False):
    c=gu_case(monkeypatch,ids,owned=owned,kind=kind)
    c.w.source_prefill_batch_policy='gate_up_down'
    c.w.source_prefill_wide_dtype='reference'
    c.calls['down_batches']=[];c.calls['mapped']=[];c.calls['mapped_kernels']=[]
    c.producers=[]
    identity=lambda obj:next(e for e,data in c.bank.items()if data is obj)
    existing_gu=gu.pp_middle
    def gate_up(*args,**kwargs):
        value=existing_gu(*args,**kwargs)
        c.producers.append(value)
        return value
    monkeypatch.setattr(gu,'pp_middle',gate_up)
    def down_batch(activation,experts,counts):
        assert c.store._active and len(c.store._leased)==len(c.bank)
        # Eligible jobs must keep the exact producer array, without per-expert
        # slicing and re-concatenation before launching their batch of down dots.
        assert any(activation is value for value in c.producers)
        eids=[identity(data)for data in experts]
        c.calls['down_batches'].append((eids,tuple(counts),activation))
        offsets=np.cumsum([0,*counts]);parts=[]
        for e,a,b in zip(eids,offsets[:-1],offsets[1:]):
            part=mx.repeat(((e+1)*activation[int(a):int(b),0])[:,None],5120,axis=1)
            if cancellation:
                scalar={0:2**25,1:1,2:-2**25}.get(e,0)
                part=mx.concatenate((mx.full((int(b-a),1),scalar,mx.float32),part[:,1:]),1)
            parts.append(part.astype(mx.bfloat16).astype(mx.float32))
        return mx.concatenate(parts,0)
    monkeypatch.setattr(down,'pp_down',down_batch)
    # Execute the real mapped planner, but independently replay its submitted
    # immutable buffer references on CPU instead of compiling a Metal kernel.
    def factory(count,first):
        def kernel(**kw):
            args=kw['inputs'];refs=np.array(args[-1]);live=[np.array(v)for v in args[:count]]
            total=np.zeros((len(c.ids),5120),np.float32)if first else np.array(args[count]).copy()
            for row in range(len(c.ids)):
                for encoded in refs[row]:
                    if encoded:
                        total[row]=np.add(total[row],live[(int(encoded)>>16)-1][int(encoded)&65535],dtype=np.float32)
            c.calls['mapped_kernels'].append((count,first,kw))
            return[mx.array(total)]
        return kernel
    monkeypatch.setattr(assembly,'_kernel',factory)
    actual_mapped=assembly.combine_mapped
    def mapped(parts,values,order,part_experts):
        c.calls['mapped'].append((tuple(parts),values.copy(),order.copy(),tuple(tuple(g)for g in part_experts)))
        return actual_mapped(parts,values,order,part_experts)
    monkeypatch.setattr(assembly,'combine_mapped',mapped)
    return c


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('kind',['mixed','q1'])
def test_batch_down_preserves_job_segments_score_slots_and_stable_combine(monkeypatch,owned,kind):
    ids=[[9,0,7,2,6,1],[9,0,7,6,5,1],[9,0,7,6,5,3]]
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids,owned=owned,kind=kind)
        actual=run(c)
        assert actual.tobytes()==expected(c).tobytes()
        assert c.acquired==[('blk.0',e)for e in sorted(c.bank)]
        assert len(c.calls['qat'])==len(c.calls['mapped'])==1
        assert not c.calls['combine'] and not c.calls['projections']
        assert len(c.calls['prefill'])==len(c.calls['down_batches'])
        for gu_call,down_call in zip(c.calls['prefill'],c.calls['down_batches']):
            assert gu_call[0]==down_call[0] and gu_call[3]==down_call[1]
        parts,values,order,groups=c.calls['mapped'][0]
        assert [n for group in groups for _,n in group]==[n for _,counts,_ in c.calls['down_batches']for n in counts]
        assert sorted(e for group in groups for e,_ in group)==sorted(c.bank)
        assert not np.array_equal(np.array([e for group in groups for e,_ in group]),np.sort(np.unique(c.ids)))
        assert c.w.scheduled_expert_calls['official_source_prefill_down_batches']==len(c.calls['down_batches'])
        assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==0
        assert c.w.scheduled_expert_calls['official_source_prefill_assembly_parts']==len(parts)
        assert c.store.stats['lookups']==1
        if owned:
            assert len(c.fences)==1 and not c.store._active and not c.store._leased
        else:
            assert c.store._active and len(c.store._leased)==len(c.bank) and not c.fences


def test_no_physical_group_subtotal_can_change_cancellation_order(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[2,0,1,3,4,5],[2,0,6,3,4,5],[2,0,6,3,4,5]],cancellation=True)
        actual=run(c);want=expected(c);want[:,0]=0
        assert actual.tobytes()==want.tobytes()
        parts,_,_,groups=c.calls['mapped'][0]
        # In token0, original0 and original2 share one physical batch while
        # Q1 expert1 is another. Summing those group subtotals would return1.
        subtotals=[]
        for part,group in zip(parts,groups):
            total=np.float32(0);start=0
            for e,n in group:
                if e in c.ids[0]:total=np.float32(total+np.array(part)[start,0])
                start+=n
            subtotals.append(total)
        assert np.float32(sum(subtotals))==1


def test_whole_original_wide_fallback_remains_one_stock_down_job(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6))
        assert run(c).tobytes()==expected(c).tobytes()
        assert c.calls['projections']==[(0,'gate',13),(0,'up',13),(0,'down',13)]
        assert c.w.scheduled_expert_calls['official_source_prefill_fallback_jobs']==1
        assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==1
        parts,_,_,groups=c.calls['mapped'][0]
        assert groups[-1]==((0,13),)
        assert sorted(e for group in groups for e,_ in group)==[0,1,2]
        assert all(0 not in eids for eids,*_ in c.calls['down_batches'])


def test_over15_q1_backings_keep_batch_segments_and_one_global_sum(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,np.arange(36,dtype=np.uint32).reshape(6,6)[:,::-1].copy(),kind='q1')
        assert run(c).tobytes()==expected(c).tobytes()
        assert [len(eids)for eids,*_ in c.calls['down_batches']]==[15,15,6]
        assert not c.calls['projections']
        assert c.w.scheduled_expert_calls['official_source_prefill_assembly_parts']==3
        assert len(c.calls['mapped'])==1


def test_all_fallback_jobs_do_not_create_empty_down_batch(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,np.zeros((3,6),np.uint32))
        assert run(c).tobytes()==expected(c).tobytes()
        assert not c.calls['prefill'] and not c.calls['down_batches']
        assert c.calls['projections']==[(0,'gate',18),(0,'up',18),(0,'down',18)]
        assert c.w.scheduled_expert_calls['official_source_prefill_down_batches']==0
        assert c.calls['mapped'][0][3]==(((0,18),),)


@pytest.mark.parametrize('failure',['acquire','gate_up','fallback_middle','batch_down','fallback_down','mapped'])
@pytest.mark.parametrize('owned',[True,False])
def test_failed_pipeline_keeps_all_submitted_backings_until_completion(monkeypatch,failure,owned):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6),owned=owned)
        def fail(*args,**kwargs):
            assert c.store._active and c.store._leased
            c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
            raise RuntimeError('injected '+failure)
        if failure=='acquire':
            original=c.store.acquire;c.store.acquire=lambda p,e:fail()if e==1 else original(p,e)
        elif failure=='gate_up':monkeypatch.setattr(gu,'pp_middle',fail)
        elif failure=='fallback_middle':monkeypatch.setattr(middle,'middle',fail)
        elif failure=='batch_down':monkeypatch.setattr(down,'pp_down',fail)
        elif failure=='mapped':monkeypatch.setattr(assembly,'combine_mapped',fail)
        else:
            original=mx.quantized_matmul
            def qmm(value,weight,*a,**kw):
                return fail()if weight is c.bank[0]['down.qweight']else original(value,weight,*a,**kw)
            monkeypatch.setattr(mx,'quantized_matmul',qmm)
        with pytest.raises(RuntimeError,match='injected'):run(c)
        assert c.store.stats['lookups']==0
        assert c.w.scheduled_expert_calls=={'official_source_input_qat':1}
        if owned:
            assert len(c.fences)==1 and c.fences[0]
            assert not c.store._active and not c.store._leased
            assert not c.store._pending_host and not c.store._pending_uploads
        else:
            assert c.store._active and c.store._leased and not c.fences
            assert c.store._pending_host and c.store._pending_uploads


def test_failed_completion_fence_quarantines_all_backing_owners(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]]*2)
        c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
        def refuse():
            assert len(c.store._leased)==len(c.bank)
            raise RuntimeError('injected fence refusal')
        c.store._fence=refuse
        try:
            with pytest.raises(RuntimeError,match='fence refusal'):run(c)
            assert c.store._active and len(c.store._leased)==len(c.bank)
            assert c.store._pending_host and c.store._pending_uploads
            assert id(c.store)in native._UPLOAD_QUARANTINE
        finally:
            mx.synchronize();c.store._leased.clear();c.store._pending_host.clear()
            c.store._pending_uploads.clear();c.store._active=False;native._release_upload(c.store)


def test_no_prefill_down_or_mapped_path_at_decode_width_one(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]])
        run(c)
        assert not c.calls['prefill'] and not c.calls['down_batches'] and not c.calls['mapped']
        assert c.w.scheduled_expert_calls['official_source_down_combine_fused']==1


def test_final_output_eval_failure_retains_leases_until_owned_fence(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]]*2)
        def failed(*args):
            assert len(c.store._leased)==len(c.bank) and c.store._active
            c.store._pending_host['sentinel']=object()
            raise RuntimeError('injected final evaluation')
        monkeypatch.setattr(mx,'eval',failed)
        with pytest.raises(RuntimeError,match='final evaluation'):
            c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert not c.store._leased and not c.store._active and not c.store._pending_host
        assert len(c.fences)==1 and c.store.stats['lookups']==1
        assert c.w.scheduled_expert_calls['official_source_prefill_down_batches']>0
