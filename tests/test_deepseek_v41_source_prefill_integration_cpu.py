"""CPU Source prefill provider orchestration; actual dot parity is separate.

Reuse the existing metadata-only PackedExpert/real NativeExpertStore.window
scaffold. Tag each expert contribution independently of batching so bucket
reordering, score-slot mistakes and lost duplicate calls change the oracle.
"""
import numpy as np
import pytest
import mlx.core as mx

from test_deepseek_v41_source_batch_provider_cpu import _case
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.runtime.gguf import source_gate_up_prefill as pp
from mlx2.runtime.gguf import source_middle_fusion as middle, source_q1


def case(monkeypatch,ids,*,owned=True,kind='mixed'):
    c=_case(monkeypatch,owned=owned,kind=kind,ids=ids)
    c.w.source_prefill_batch_policy='gate_up'
    c.store.kind=lambda key:c.bank[key[1]].layout.kind
    c.calls['prefill']=[];c.calls['fallback_middle']=[];c.calls['projections']=[]
    identity=lambda obj:next(e for e,value in c.bank.items()if value is obj)
    def prefill(x,experts,rows,scores,counts,**kwargs):
        assert c.store._active and len(c.store._leased)==len(c.bank)
        eids=[identity(e)for e in experts]
        c.calls['prefill'].append((eids,np.array(rows),np.array(scores),tuple(counts),kwargs))
        offsets=np.cumsum([0,*counts]);parts=[]
        for e,a,b in zip(eids,offsets[:-1],offsets[1:]):
            parts.append(mx.repeat(((e+1)*scores[a:b])[:,None],2304,axis=1))
        return mx.concatenate(parts,axis=0)
    monkeypatch.setattr(pp,'pp_middle',prefill)
    def project(value,weight,*args,**kwargs):
        e,name=next((e,name)for e,data in c.bank.items()for name in ('gate','up','down')
                    if data[name+'.qweight']is weight)
        c.calls['projections'].append((e,name,value.shape[0]))
        if name=='down':return mx.repeat(((e+1)*value[:,0])[:,None],5120,axis=1)
        return mx.full((value.shape[0],2304),e+1,mx.float32)
    monkeypatch.setattr(source_q1,'matmul',project)
    monkeypatch.setattr(mx,'quantized_matmul',project)
    def fallback(gate,up,scores,slots):
        c.calls['fallback_middle'].append((np.array(gate),np.array(slots)))
        return gate*scores[slots,None]
    monkeypatch.setattr(middle,'middle',fallback)
    return c


def expected(c):
    # Independent Source per-expert BF16 boundary, then stable F32 sum.
    scores=np.array(c.scores);output=[]
    for row,ids in enumerate(c.ids):
        total=np.float32(0)
        for slot in sorted(range(6),key=lambda s:int(ids[s])):
            factor=np.float32(int(ids[slot])+1)
            raw=np.float32(factor*np.float32(factor*scores[row,slot]))
            bits=int(raw.view(np.uint32))
            value=np.uint32((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)
            total=np.float32(total+value)
        output.append(np.full(5120,total,np.float32))
    return np.stack(output)


def run(c):
    result=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(result)
    return np.array(result)


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('kind',['mixed','q1'])
def test_bucketed_gu_restores_expert_order_scores_and_ordinary_down(monkeypatch,owned,kind):
    # Ascending expert order differs from row-count bucket order: expert0 has
    # R3, expert2 R1, and odd Q1 IDs have differing row counts in one batch.
    ids=[[9,0,7,2,6,1],[9,0,7,6,5,1],[9,0,7,6,5,3]]
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids,owned=owned,kind=kind)
        actual=run(c)
        assert actual.tobytes()==expected(c).tobytes()
        assert c.acquired==[('blk.0',e)for e in sorted(c.bank)]
        assert len(c.calls['qat'])==1 and len(c.calls['combine'])==1
        assert c.w.scheduled_expert_calls['official_source_prefill_batched_jobs']==len(c.bank)
        assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==len(c.bank)
        assert c.w.scheduled_expert_calls['official_source_prefill_fallback_jobs']==0
        assert c.w.scheduled_expert_calls['official_source_prefill_gu_batches']==len(c.calls['prefill'])
        assert c.w.scheduled_expert_calls['official_source_input_gather_bytes_avoided']==c.ids.size*5120*4
        assert c.store.stats['lookups']==1
        seen=[]
        for eids,rows,scores,counts,kw in c.calls['prefill']:
            seen.extend(eids);offsets=np.cumsum([0,*counts])
            assert kw=={'exp_spelling':'metal::precise::exp'}
            for e,a,b in zip(eids,offsets[:-1],offsets[1:]):
                slots=np.flatnonzero(c.ids.ravel()==e)
                assert np.array_equal(rows[a:b],slots//6)
                assert np.array_equal(scores[a:b],np.array(c.scores).ravel()[slots])
        assert sorted(seen)==sorted(c.bank)
        assert c.calls['projections']==[(e,'down',int(np.sum(c.ids==e)))for e in sorted(c.bank)]
        if owned:assert len(c.fences)==1 and not c.store._active and not c.store._leased
        else:assert not c.fences and c.store._active and len(c.store._leased)==len(c.bank)


def test_original_wide_fallback_is_whole_expert_not_split_into_changed_dot_laws(monkeypatch):
    # Repeated routes are legal to the CPU planner and make13calls to one
    # original expert; its stock dot path must see13rows in one invocation.
    ids=np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6)
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids)
        assert run(c).tobytes()==expected(c).tobytes()
        assert c.calls['projections'][:2]==[(0,'gate',13),(0,'up',13)]
        assert c.calls['projections'][2:]==[(0,'down',13),(1,'down',3),(2,'down',2)]
        assert len(c.calls['fallback_middle'])==1
        assert c.calls['fallback_middle'][0][1].tolist()==list(range(13))
        assert all(0 not in ids for ids,*_ in c.calls['prefill'])
        assert c.w.scheduled_expert_calls['official_source_prefill_fallback_jobs']==1
        assert c.w.scheduled_expert_calls['official_source_prefill_batched_jobs']==2
        assert c.w.scheduled_expert_calls['official_source_input_gather_bytes_avoided']==5*5120*4


def test_more_than15_q1_objects_chunk_without_losing_job_order(monkeypatch):
    ids=np.arange(36,dtype=np.uint32).reshape(6,6)[:,::-1].copy()
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids,kind='q1')
        assert run(c).tobytes()==expected(c).tobytes()
        assert [len(v[0])for v in c.calls['prefill']]==[15,15,6]
        assert c.w.scheduled_expert_calls['official_source_prefill_gu_batches']==3
        assert [e for e,name,n in c.calls['projections']]==list(range(36))


@pytest.mark.parametrize('failure',['acquire','prefill','fallback','down','combine'])
@pytest.mark.parametrize('owned',[True,False])
def test_failures_keep_all_live_leases_until_owned_fence_or_external_completion(monkeypatch,failure,owned):
    ids=np.array([0]*13+[1]*3+[2]*2,np.uint32).reshape(3,6)
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids,owned=owned)
        def fail(*args,**kwargs):
            assert c.store._active and c.store._leased
            c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
            raise RuntimeError('injected '+failure)
        if failure=='acquire':
            original=c.store.acquire
            c.store.acquire=lambda p,e:fail()if e==1 else original(p,e)
        elif failure=='prefill':monkeypatch.setattr(pp,'pp_middle',fail)
        elif failure=='fallback':monkeypatch.setattr(middle,'middle',fail)
        elif failure=='combine':monkeypatch.setattr(assembly,'ordered_combine',fail)
        else:
            original=mx.quantized_matmul
            def qmm(value,weight,*a,**kw):
                if weight is c.bank[0]['down.qweight']:return fail()
                return original(value,weight,*a,**kw)
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


def test_failed_fence_retains_all_expert_and_pending_owners_until_successful_cleanup(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]]*2)
        c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
        def bad_fence():
            assert len(c.store._leased)==len(c.bank)
            raise RuntimeError('injected fence failure')
        c.store._fence=bad_fence
        try:
            with pytest.raises(RuntimeError,match='fence failure'):run(c)
            assert c.store._active and len(c.store._leased)==len(c.bank)
            assert c.store._pending_host and c.store._pending_uploads
            assert id(c.store)in native._UPLOAD_QUARANTINE
        finally:
            # CPU test has no asynchronous GPU consumers; explicit successful
            # completion precedes releasing the real quarantine owner.
            mx.synchronize();c.store._leased.clear();c.store._pending_host.clear()
            c.store._pending_uploads.clear();c.store._active=False
            native._release_upload(c.store)


@pytest.mark.parametrize('admission',['unpacked','device','invalid_kind'])
def test_admission_refusal_occurs_before_any_expert_acquire(monkeypatch,admission):
    from mlx2.runtime.gguf import source_mxfp4_pair
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]]*2)
        if admission=='unpacked':c.store.packed_buffers=False
        elif admission=='device':monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:False)
        else:c.store.kind=lambda key:'unknown'
        with pytest.raises(ValueError):run(c)
        assert not c.acquired and not c.store._leased
        assert not c.calls['prefill'] and c.store.stats['lookups']==0


def test_all_wide_original_jobs_use_whole_fallback_without_empty_batch_launch(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,np.zeros((3,6),np.uint32))
        assert run(c).tobytes()==expected(c).tobytes()
        assert not c.calls['prefill']
        assert c.calls['projections']==[(0,'gate',18),(0,'up',18),(0,'down',18)]
        assert c.w.scheduled_expert_calls['official_source_prefill_gu_batches']==0
        assert c.w.scheduled_expert_calls['official_source_prefill_fallback_jobs']==1
        assert c.w.scheduled_expert_calls['official_source_input_gather_bytes_avoided']==0


def test_output_evaluation_failure_keeps_leases_until_real_window_fence(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,[[0,1,2,3,4,5]]*2)
        def bad_eval(*outputs):
            assert c.store._active and len(c.store._leased)==len(c.bank)
            c.store._pending_host['sentinel']=object()
            raise RuntimeError('injected output evaluation')
        monkeypatch.setattr(mx,'eval',bad_eval)
        with pytest.raises(RuntimeError,match='output evaluation'):
            c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert len(c.fences)==1 and len(c.fences[0])==len(c.bank)
        assert not c.store._active and not c.store._leased and not c.store._pending_host
        # Work was scheduled before evaluation failed; preserve those counters.
        assert c.store.stats['lookups']==1
        assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==6
