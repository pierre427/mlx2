"""Source batch provider control/ownership oracles; no Metal or model math.

Heavy kernels are replaced by independently score-tagged CPU outputs. Exact
projection arithmetic is checked by the separately owned actual-weight GPU
probes. Metadata-only packed objects allocate no expert weight payload here.
"""
from collections import OrderedDict
import threading
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertStore
from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert,canonical_layout
from mlx2.adapters.deepseek_v41_source_lfu import LFUIndex,LFUSourceExpertStore,ORIGINAL_BYTES,Q1_BYTES
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters import deepseek_v41_source_math as assembly
from mlx2.runtime.gguf import source_down_fusion as down,source_gate_up_fusion as gu
from mlx2.runtime.gguf import source_middle_fusion as middle,source_mxfp4_pair,source_q1
from mlx2.runtime.models import deepseek_v41_quant as qat


def _packed(kind):
    layout=canonical_layout(kind)
    types={'<u4':mx.uint32,'|u1':mx.uint8,'<f2':mx.float16}
    fields={t.name:SimpleNamespace(shape=t.shape,ndim=len(t.shape),dtype=types[t.dtype],nbytes=t.nbytes)
            for t in layout.tensors}
    backing=SimpleNamespace(shape=(layout.nbytes,),ndim=1,dtype=mx.uint8,nbytes=layout.nbytes)
    return PackedExpert(backing,layout,fields)


def _case(monkeypatch,*,owned=True,packed=True,ids=None,kind='mixed'):
    ids=np.array([[9,1,7,2,6,0]] if ids is None else ids,np.uint32)
    rows=len(ids);scores=mx.array(np.tile([.125,.25,.5,1,2,4],(rows,1)),mx.float32)
    x=mx.zeros((rows,5120),mx.float32)
    bank={int(e):_packed('q1' if kind=='q1' or int(e)%2 else 'original') for e in np.unique(ids)}
    identity=lambda obj:next(e for e,value in bank.items() if value is obj)
    acquired=[];observed=[];fences=[];calls={'qat':[],'batch':[],'single':[],'down':[],'plain_down':[],'combine':[]}
    store=SimpleNamespace(_active=not owned,_error=None,_lock=threading.RLock(),
        _leased={},_pending_uploads={},_pending_host={},packed_buffers=packed,
        stats={'lookups':0,'windows':0})
    store._ready=lambda:None
    store._fence=lambda:fences.append(tuple(store._leased))
    store.window=MethodType(NativeExpertStore.window,store)
    store.observe_routes=lambda p,v:observed.append((p,v.copy()))
    def acquire(prefix,e):
        assert store._active and type(e) is int and 0<=e<384
        acquired.append((prefix,e));store._leased[e]=bank[e]
        return bank[e]
    store.acquire=acquire
    w=object.__new__(SourceV41Weights);w._ready=lambda:None
    w.artifact={'args':SimpleNamespace(hidden_size=5120,num_experts_per_tok=6)}
    w.expert_math_policy='routed_fused_fp32';w.scheduled_expert_calls={};w.expert_stream=store
    def no_metal(*args,**kwargs):raise AssertionError('CPU provider oracle reached Metal')
    monkeypatch.setattr(mx.fast,'metal_kernel',no_metal)
    monkeypatch.setattr(source_mxfp4_pair,'_device_supported',lambda:True)
    monkeypatch.setattr(middle,'exp_spelling',lambda:'metal::precise::exp')
    def quantize(value,kind):calls['qat'].append((value,kind));return value
    monkeypatch.setattr(qat,'fused_quantize',quantize)
    def batch(value,experts,flat_scores,slots,**kwargs):
        calls['batch'].append((value,tuple(experts),np.array(flat_scores),np.array(slots)))
        return mx.stack([mx.full((2304,),(identity(e)+1)*flat_scores[int(s)],mx.float32)
                         for e,s in zip(experts,np.array(slots))])
    monkeypatch.setattr(gu,'batch_middle',batch)
    def single(value,token_rows,selected_scores,expert,**kwargs):
        calls['single'].append((np.array(token_rows),np.array(selected_scores),identity(expert)))
        return mx.repeat(((identity(expert)+1)*selected_scores)[:,None],2304,axis=1)
    monkeypatch.setattr(gu,'middle',single)
    def down_sum(middles,experts,expert_ids):
        calls['down'].append((tuple(middles),tuple(experts),tuple(expert_ids)))
        total=mx.zeros((1,5120),mx.float32)
        for index in sorted(range(len(expert_ids)),key=lambda i:expert_ids[i]):
            v=middles[index][0,0]*(expert_ids[index]+1)
            total=total+mx.full((1,5120),v,mx.float32)
        return total
    monkeypatch.setattr(down,'down_combine',down_sum)
    def plain_down(value,weight,scales,biases):
        e=next(k for k,obj in bank.items() if obj['down.qweight'] is weight)
        calls['plain_down'].append(e)
        return mx.repeat(((e+1)*value[:,0])[:,None],5120,axis=1)
    monkeypatch.setattr(source_q1,'matmul',plain_down)
    def combine(parts,values,order):
        calls['combine'].append((values.copy(),order.copy()))
        flat=mx.concatenate(parts,axis=0);inverse={int(slot):i for i,slot in enumerate(order)}
        outputs=[]
        for row in range(len(values)):
            total=mx.zeros((5120,),mx.float32)
            for slot in sorted(range(6),key=lambda s:int(values[row,s])):
                total=total+flat[inverse[row*6+slot]]
            outputs.append(total)
        return mx.stack(outputs)
    monkeypatch.setattr(assembly,'ordered_combine',combine)
    return SimpleNamespace(w=w,store=store,bank=bank,x=x,ids=ids,scores=scores,calls=calls,
        acquired=acquired,observed=observed,fences=fences)


def _expected(case):
    scores=np.array(case.scores);out=[]
    for row,ids in enumerate(case.ids):
        value=np.float32(0)
        for slot in sorted(range(len(ids)),key=lambda s:int(ids[s])):
            value=np.float32(value+np.float32((int(ids[slot])+1)**2*scores[row,slot]))
        out.append(np.full(5120,value,np.float32))
    return np.stack(out)


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('kind',['mixed','q1'])
def test_three_stage_batch_preserves_sorted_experts_and_original_score_slots(monkeypatch,owned,kind):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch,owned=owned,kind=kind)
        output=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(output)
        assert np.array(output).tobytes()==_expected(c).tobytes()
        assert len(c.calls['qat'])==len(c.calls['batch'])==len(c.calls['down'])==1
        assert not c.calls['single'] and not c.calls['plain_down'] and not c.calls['combine']
        _,experts,scores,slots=c.calls['batch'][0]
        assert list(slots)==list(np.argsort(c.ids.ravel(),kind='stable'))
        assert [c.bank[e] for e in sorted(c.bank)]==list(experts)
        assert np.array_equal(scores,np.array(c.scores).ravel())
        assert c.calls['down'][0][2]==tuple(sorted(c.bank))
        assert c.acquired==[('blk.0',e) for e in sorted(c.bank)]
        assert len(c.observed)==1 and c.store.stats['lookups']==1
        assert c.w.scheduled_expert_calls['official_source_gate_up_middle_batch_fused']==1
        assert c.w.scheduled_expert_calls['official_source_down_combine_fused']==1
        assert c.w.scheduled_expert_calls['official_source_input_gather_bytes_avoided']==6*5120*4
        if owned:
            assert len(c.fences)==1 and not c.store._active and not c.store._leased
        else:
            assert not c.fences and c.store._active and len(c.store._leased)==6


@pytest.mark.parametrize('packed_flag,plain_object',[(False,False),(True,True)])
def test_missing_packed_capability_declines_batch_without_extra_lease_objects(monkeypatch,packed_flag,plain_object):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch,packed=packed_flag,kind='q1')
        if plain_object:
            # A normal tensor mapping is a valid fallback expert, but does not
            # carry the validated backing/layout needed by the batch kernel.
            c.bank[7]=dict(c.bank[7])
        output=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(output)
        assert np.array(output).tobytes()==_expected(c).tobytes()
        assert not c.calls['batch'] and len(c.calls['single'])==6 and len(c.calls['down'])==1
        assert len(c.calls['qat'])==1
        assert set(c.acquired)=={('blk.0',e) for e in c.bank}
        assert not c.store._leased and len(c.fences)==1


@pytest.mark.parametrize('ids',[
    [[3,1,3,2,1,2]],
    [[9,1,7,2,6,0],[9,1,7,2,6,0]],
])
def test_duplicate_and_multirow_routes_keep_safe_per_expert_path(monkeypatch,ids):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch,ids=ids,kind='q1')
        output=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(output)
        assert np.array(output).tobytes()==_expected(c).tobytes()
        assert not c.calls['batch'] and not c.calls['down']
        assert len(c.calls['combine'])==1 and len(c.calls['qat'])==1
        assert len(c.calls['single'])==len(c.bank)
        assert len(c.calls['plain_down'])==len(c.bank)
        for token_rows,selected_scores,e in c.calls['single']:
            flat=np.flatnonzero(c.ids.ravel()==e)
            assert np.array_equal(token_rows,flat//6)
            assert np.array_equal(selected_scores,np.array(c.scores).ravel()[flat])


@pytest.mark.parametrize('failure',['acquire','gate_up','down'])
def test_batch_failures_fence_and_release_owned_window_without_completed_calls(monkeypatch,failure):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch,kind='q1')
        def fail(*a,**kw):
            c.store._pending_host['temporary']=object()
            c.store._pending_uploads['temporary']={}
            raise RuntimeError('injected '+failure)
        if failure=='acquire':
            original=c.store.acquire
            def acquire(prefix,e):
                if e==6:fail()
                return original(prefix,e)
            c.store.acquire=acquire
        elif failure=='gate_up':monkeypatch.setattr(gu,'batch_middle',fail)
        else:monkeypatch.setattr(down,'down_combine',fail)
        with pytest.raises(RuntimeError,match='injected'):c.w.moe(c.x,c.ids,c.scores,'blk.0')
        assert len(c.fences)==1 and not c.store._active
        assert not c.store._leased and not c.store._pending_host and not c.store._pending_uploads
        assert c.store._error and c.store.stats['lookups']==0
        # The input-QAT launch was already scheduled before a downstream
        # failure. No gate/up, down or completed-MoE launch is counted.
        assert c.w.scheduled_expert_calls=={'official_source_input_qat':1}


@pytest.mark.parametrize('malformed',['ids_dtype','ids_shape','x_shape','scores_shape','ticket'])
def test_malformed_route_refuses_before_acquisition_or_math(monkeypatch,malformed):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch);ids=c.ids;x=c.x;scores=c.scores;ticket=None
        if malformed=='ids_dtype':ids=ids.astype(np.int32)
        elif malformed=='ids_shape':ids=ids.ravel()
        elif malformed=='x_shape':x=mx.zeros((1,5119),mx.float32)
        elif malformed=='scores_shape':scores=scores[:,:5]
        else:ticket=object()
        with pytest.raises(ValueError):c.w.moe(x,ids,scores,'blk.0',ticket=ticket)
        assert not c.acquired and not c.calls['qat'] and not c.fences


def test_batch_misses_obey_existing_lfu_ceiling_and_preserve_leases(monkeypatch):
    with mx.stream(mx.cpu):
        c=_case(monkeypatch)
        s=c.store;hot={e for e in c.bank if e%2==0}
        s.hot=frozenset((0,e) for e in hot);s.q1_root='/q1';s.index=LFUIndex()
        s.policy={'hot_bytes':len(hot)*ORIGINAL_BYTES,
                  'budget_bytes':len(hot)*ORIGINAL_BYTES+3*Q1_BYTES+ORIGINAL_BYTES}
        s.payload_ceiling=s.policy['budget_bytes']-ORIGINAL_BYTES
        dummy={e:_packed('q1') for e in (20,21,22)}
        s._cache=OrderedDict({(0,e):obj for e,obj in {**{e:c.bank[e] for e in hot},**dummy}.items()})
        s._cache_bytes=s.payload_ceiling;s.pool_bytes={'original':len(hot)*ORIGINAL_BYTES,'q1':3*Q1_BYTES}
        s.stats['evictions']=0;s.kind=lambda key:'original' if key in s.hot else 'q1'
        for key in s._cache:s.index.touch(key)
        peaks=[]
        def acquire(prefix,e):
            key=(0,e);c.acquired.append((prefix,e))
            if key not in s._cache:
                LFUSourceExpertStore._evict_for(s,key,Q1_BYTES)
                s._cache[key]=c.bank[e];s._cache_bytes+=Q1_BYTES;s.pool_bytes['q1']+=Q1_BYTES
            s.index.touch(key);s._leased[key]=s._cache[key]
            peaks.append(s._cache_bytes+ORIGINAL_BYTES)
            return s._cache[key]
        s.acquire=acquire
        output=c.w.moe(c.x,c.ids,c.scores,'blk.0');mx.eval(output)
        assert np.array_equal(np.array(output),_expected(c))
        assert s.stats['evictions']==3
        assert max(peaks)<=s.policy['budget_bytes']
        assert s._cache_bytes==s.payload_ceiling and not s._leased
        assert set(s._cache)=={(0,e) for e in c.bank}
