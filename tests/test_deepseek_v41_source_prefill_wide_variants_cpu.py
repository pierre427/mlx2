"""Alternate wide geometry/fused-consumer provider contracts, CPU tags only."""
import numpy as np
import mlx.core as mx
import pytest

from test_deepseek_v41_source_prefill_wide_integration_cpu import case as base_case
from test_deepseek_v41_source_prefill_integration_cpu import expected,run
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.gguf import source_prefill_legacy_tiles as legacy

VARIANTS=('legacy_tiles_16x64','legacy_tiles_fused','legacy_tiles_fused_16x64')


def case(monkeypatch,policy,*,batch_down=False,owned=True,ids=None):
    c=base_case(monkeypatch,batch_down=batch_down,owned=owned,ids=ids)
    c.w.source_prefill_wide_math=policy
    c.calls['variant_project']=[];c.calls['variant_middle']=[]
    original=legacy.batch_project
    def projection(*args,**kwargs):
        assert policy=='legacy_tiles_16x64','Fused provider must not materialize projections'
        tile=kwargs.pop('tile',None);assert tile==(16,64)
        c.calls['variant_project'].append((tile,kwargs.get('projection'),args[0]))
        return original(*args,**kwargs)
    monkeypatch.setattr(legacy,'batch_project',projection)
    identity=lambda obj:next(e for e,data in c.bank.items()if data is obj)
    def fused(gathered,experts,counts,scores,slots,*,exp_spelling,tile=(32,32)):
        assert '_fused'in policy
        assert exp_spelling=='metal::precise::exp'
        assert tile==((16,64)if policy.endswith('_16x64')else(32,32))
        assert scores.shape==(c.ids.size,) and scores.dtype==mx.float32
        assert slots.shape==(sum(counts),) and slots.dtype==mx.uint32
        assert c.store._active and len(c.store._leased)==len(c.bank)
        eids=tuple(identity(e)for e in experts);offsets=np.cumsum([0,*counts]);selected=scores[slots]
        c.calls['variant_middle'].append((eids,tuple(counts),scores,np.array(slots),tile,gathered))
        parts=[]
        for e,a,b in zip(eids,offsets[:-1],offsets[1:]):
            np.testing.assert_array_equal(np.array(slots)[a:b],np.flatnonzero(c.ids.ravel()==e))
            parts.append(mx.repeat(((e+1)*selected[int(a):int(b)])[:,None],2304,axis=1))
        return mx.concatenate(parts,axis=0)
    monkeypatch.setattr(legacy,'batch_middle',fused)
    return c


@pytest.mark.parametrize('policy',VARIANTS)
@pytest.mark.parametrize('batch_down',[False,True])
def test_variants_keep_scores_occurrences_restoration_and_down_contract(monkeypatch,policy,batch_down):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,policy,batch_down=batch_down)
        assert run(c).tobytes()==expected(c).tobytes()
        counters=c.w.scheduled_expert_calls
        assert counters['official_source_prefill_wide_jobs']==5
        assert counters['official_source_prefill_wide_gu_batches']==3
        assert counters['official_source_prefill_fallback_jobs']==1
        assert counters['official_source_prefill_ordinary_down_jobs']==(6 if batch_down else 7)
        assert [(e,p,n)for e,p,n in c.calls['projections']if p!='down']==[(6,'gate',97),(6,'up',97)]
        if '_fused'in policy:
            assert not c.calls['variant_project']and not c.calls['wide']
            assert len(c.calls['variant_middle'])==3
            assert len(c.calls['fallback_middle'])==1  # M97 only.
            assert counters['official_source_prefill_wide_projection_reductions']==0
            assert counters['official_source_prefill_wide_fused_middle_batches']==3
            scores=c.calls['variant_middle'][0][2]
            assert all(row[2]is scores for row in c.calls['variant_middle'])
            for eids,counts,_,slots,_,gathered in c.calls['variant_middle']:
                np.testing.assert_array_equal(np.array(gathered)[:,0],slots//6)
        else:
            assert len(c.calls['variant_project'])==6 and not c.calls['variant_middle']
            assert len(c.calls['fallback_middle'])==4
            assert counters['official_source_prefill_wide_projection_reductions']==6
            assert counters['official_source_prefill_wide_fused_middle_batches']==0
            for gate,up in zip(c.calls['variant_project'][::2],c.calls['variant_project'][1::2]):
                assert gate[2]is up[2]
        assert counters['official_source_prefill_wide_16x64_batches']==(3 if policy.endswith('_16x64')else 0)
        assert not c.store._active and not c.store._leased


@pytest.mark.parametrize('policy',VARIANTS)
@pytest.mark.parametrize('owned',[True,False])
def test_variant_failure_retains_owner_until_completion(monkeypatch,policy,owned):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,policy,owned=owned)
        def fail(*a,**kw):
            assert c.store._active and len(c.store._leased)==len(c.bank)
            c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
            raise RuntimeError('injected variant failure')
        monkeypatch.setattr(legacy,'batch_middle'if '_fused'in policy else'batch_project',fail)
        with pytest.raises(RuntimeError,match='variant failure'):run(c)
        assert c.w.scheduled_expert_calls=={'official_source_input_qat':1}
        if owned:
            assert c.fences and not c.store._active and not c.store._leased
            assert not c.store._pending_host and not c.store._pending_uploads
        else:
            assert not c.fences and c.store._active and c.store._leased and c.store._pending_host


@pytest.mark.parametrize('policy',VARIANTS)
def test_variants_do_not_dispatch_for_small_rows_or_m1(monkeypatch,policy):
    for ids in ([[0,1,2,3,4,5]]*2,[[0,1,2,3,4,5]]):
        with mx.stream(mx.cpu):
            c=case(monkeypatch,policy,ids=ids)
            assert run(c).tobytes()==expected(c).tobytes()
            assert not c.calls['variant_middle']and not c.calls['variant_project']


@pytest.mark.parametrize('policy',('legacy_tiles',*VARIANTS))
def test_all_wide_options_require_same_closed_policy_dependencies(policy):
    key='deepseek_source_prefill_wide_math'
    base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
    assert source_policy({**base,key:policy})[key]==policy
    for bad in ({key:policy},{**base,key:policy,'deepseek_source_prefill_wide_dtype':'bf16_p1'}):
        with pytest.raises(ValueError):source_policy(bad)
    for extra in ({'expert_policy':None},{'expert_policy':{},'prefill_wide_dtype':'bf16_p1'}):
        with pytest.raises(ValueError):
            SourceV41Weights({},expert_cache_bytes=1,expert_math_policy='routed_fused_fp32',
                prefill_batch_policy='gate_up',prefill_wide_math=policy,**extra)
