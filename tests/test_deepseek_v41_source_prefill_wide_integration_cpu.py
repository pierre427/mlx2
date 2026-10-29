"""Whole-wide provider wiring/ownership with CPU projection tags, not GPU math."""
import numpy as np
import mlx.core as mx
import pytest

from test_deepseek_v41_source_prefill_integration_cpu import case as gu_case,expected,run
from test_deepseek_v41_source_prefill_down_integration_cpu import case as down_case
from mlx2.adapters.deepseek_v41_source import source_policy,DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.runtime.gguf import source_prefill_legacy_tiles as legacy,source_middle_fusion as middle


def routes():
    values=np.concatenate([np.full(n,e,np.uint32)for e,n in ((0,13),(2,33),(4,65),(6,97),(8,17),(10,20),(1,349))])
    assert len(values)==594
    return values.reshape(99,6)


def case(monkeypatch,*,owned=True,batch_down=False,ids=None):
    c=(down_case if batch_down else gu_case)(monkeypatch,routes()if ids is None else ids,owned=owned)
    c.w.source_prefill_wide_math='legacy_tiles';c.w.source_prefill_wide_dtype='reference'
    c.calls['wide']=[]
    # Independent row tags make the one shared gathered activation observable.
    c.x=mx.repeat(mx.arange(len(c.ids),dtype=mx.float32)[:,None],5120,axis=1)
    identity=lambda obj:next(e for e,data in c.bank.items()if data is obj)
    def project(gathered,experts,counts,*,projection='gate',return_partials=False):
        assert c.store._active and len(c.store._leased)==len(c.bank)
        assert not return_partials and gathered.dtype==mx.float32
        eids=tuple(identity(e)for e in experts)
        c.calls['wide'].append((projection,eids,tuple(counts),gathered,np.array(gathered)[:,0]))
        parts=[mx.full((n,2304),e+1,mx.float32)for e,n in zip(eids,counts)]
        return mx.concatenate(parts,axis=0).astype(mx.bfloat16).astype(mx.float32)
    monkeypatch.setattr(legacy,'batch_project',project)
    return c


@pytest.mark.parametrize('owned',[True,False])
@pytest.mark.parametrize('batch_down',[True,False])
def test_wide_batches_share_one_gather_and_restore_ordinary_down_and_expert_sum(monkeypatch,owned,batch_down):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,owned=owned,batch_down=batch_down)
        assert run(c).tobytes()==expected(c).tobytes()
        assert len(c.calls['wide'])==6  # Three compatible batches, gate + up.
        flattened=[]
        for gate,up in zip(c.calls['wide'][::2],c.calls['wide'][1::2]):
            assert gate[0]=='gate' and up[0]=='up' and gate[1:3]==up[1:3]
            assert gate[3]is up[3]
            eids,counts=gate[1:3];flattened.extend(eids)
            expected_rows=np.concatenate([np.flatnonzero(c.ids.ravel()==e)//6 for e in eids])
            np.testing.assert_array_equal(gate[4],expected_rows)
            assert sum(counts)==len(expected_rows)
        assert sorted(flattened)==[0,2,4,8,10] and 6 not in flattened
        # Only M97 uses ordinary gate/up; all wide down remains ordinary even
        # when the small jobs select the separate batched-down policy.
        ordinary=c.calls['projections']
        assert [(e,p,n)for e,p,n in ordinary if p!='down']==[(6,'gate',97),(6,'up',97)]
        assert [(e,n)for e,p,n in ordinary if p=='down']==[(e,int(np.sum(c.ids==e)))for e in sorted(c.bank)if not batch_down or e%2==0]
        counters=c.w.scheduled_expert_calls
        assert counters['official_source_prefill_wide_jobs']==5
        assert counters['official_source_prefill_wide_gu_batches']==3
        assert counters['official_source_prefill_wide_projection_batches']==6
        assert counters['official_source_prefill_wide_projection_reductions']==6
        assert counters['official_source_prefill_wide_gather_logical_bytes']==148*5120*4
        assert counters['official_source_prefill_wide_partial_logical_bytes']==2*(2*(33+65)+5*(13+17+20))*2304*4
        assert counters['official_source_prefill_fallback_jobs']==1
        assert counters['official_source_prefill_ordinary_down_jobs']==(6 if batch_down else 7)
        if owned:assert len(c.fences)==1 and not c.store._active and not c.store._leased
        else:assert c.store._active and len(c.store._leased)==len(c.bank) and not c.fences


def test_reference_flag_preserves_existing_fallback_flow_without_wide_calls(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch);c.w.source_prefill_wide_math='reference'
        assert run(c).tobytes()==expected(c).tobytes()
        assert not c.calls['wide']
        assert len([1 for e,p,n in c.calls['projections']if p=='gate'])==6
        assert not any(k.startswith('official_source_prefill_wide_')for k in c.w.scheduled_expert_calls)


def test_optin_without_eligible_wide_jobs_keeps_small_prefill_and_m1_decode(monkeypatch):
    for ids in ([[0,1,2,3,4,5]]*2,[[0,1,2,3,4,5]]):
        with mx.stream(mx.cpu):
            c=case(monkeypatch,ids=ids)
            assert run(c).tobytes()==expected(c).tobytes()
            assert not c.calls['wide']
            if len(ids)>1:assert c.w.scheduled_expert_calls['official_source_prefill_wide_jobs']==0
            else:assert 'official_source_prefill_wide_jobs'not in c.w.scheduled_expert_calls


@pytest.mark.parametrize('failure',['gate','up','middle'])
@pytest.mark.parametrize('owned',[True,False])
def test_wide_pipeline_errors_do_not_drop_owners_before_completion(monkeypatch,failure,owned):
    with mx.stream(mx.cpu):
        c=case(monkeypatch,owned=owned)
        def fail():
            assert c.store._active and len(c.store._leased)==len(c.bank)
            c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
            raise RuntimeError('injected wide '+failure)
        if failure=='middle':
            original=middle.middle
            monkeypatch.setattr(middle,'middle',lambda *a,**kw:fail()if a[0].shape[0]>12 else original(*a,**kw))
        else:
            original=legacy.batch_project
            monkeypatch.setattr(legacy,'batch_project',lambda *a,**kw:fail()if kw['projection']==failure else original(*a,**kw))
        with pytest.raises(RuntimeError,match='injected wide'):run(c)
        assert c.w.scheduled_expert_calls=={'official_source_input_qat':1}
        assert c.store.stats['lookups']==0
        if owned:
            assert len(c.fences)==1 and c.fences[0]
            assert not c.store._active and not c.store._leased and not c.store._pending_host
        else:
            assert not c.fences and c.store._active and c.store._leased and c.store._pending_host


def test_failed_fence_after_wide_graph_retains_all_backing_and_staging_owners(monkeypatch):
    with mx.stream(mx.cpu):
        c=case(monkeypatch);c.store._pending_host['sentinel']=object()
        def bad_fence():
            assert len(c.store._leased)==len(c.bank)
            raise RuntimeError('wide fence failure')
        c.store._fence=bad_fence
        try:
            with pytest.raises(RuntimeError,match='wide fence'):run(c)
            assert c.store._active and len(c.store._leased)==len(c.bank)
            assert c.store._pending_host and id(c.store)in native._UPLOAD_QUARANTINE
        finally:
            mx.synchronize();c.store._leased.clear();c.store._pending_host.clear()
            c.store._pending_uploads.clear();c.store._active=False;native._release_upload(c.store)


def test_policy_is_explicit_identity_material_with_closed_options_and_dependencies():
    key='deepseek_source_prefill_wide_math'
    assert key not in source_policy() and key in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
    assert source_policy({**base,key:'legacy_tiles'})[key]=='legacy_tiles'
    for bad in (None,True,1,[],'unknown'):
        with pytest.raises(ValueError):source_policy({**base,key:bad})
    for bad in ({key:'legacy_tiles'}, {**base,'deepseek_source_prefill_batch':'reference',key:'legacy_tiles'},
                {**base,'deepseek_source_prefill_wide_dtype':'bf16_p1',key:'legacy_tiles'}):
        with pytest.raises(ValueError):source_policy(bad)


@pytest.mark.parametrize('options',[
 {'prefill_batch_policy':'reference','expert_policy':{'mode':'original_lfu'}},
 {'prefill_batch_policy':'gate_up','expert_policy':None},
 {'prefill_batch_policy':'gate_up','expert_policy':{'mode':'original_lfu'},'prefill_wide_dtype':'bf16_p1'},
 {'prefill_batch_policy':'gate_up','expert_policy':{'mode':'original_lfu'},'expert_math_policy':'reference'},
])
def test_weight_constructor_refuses_invalid_wide_combinations_before_artifact_load(options):
    args={'expert_math_policy':'routed_fused_fp32','prefill_wide_math':'legacy_tiles',**options}
    with pytest.raises(ValueError):SourceV41Weights({},expert_cache_bytes=1,**args)


def test_wide_policy_is_returned_in_weight_receipt_and_policy_identity(monkeypatch):
    import hashlib,json
    with mx.stream(mx.cpu):
        c=case(monkeypatch,ids=[[0,1,2,3,4,5]]*2)
        monkeypatch.setattr(native.NativeV41Weights,'dense_receipt',lambda self:{})
        assert c.w.dense_receipt()['source_prefill_wide_math']=='legacy_tiles'
        base={'deepseek_source_expert_math':'routed_fused_fp32','deepseek_source_prefill_batch':'gate_up'}
        ordinary=source_policy(base)
        tiled=source_policy({**base,'deepseek_source_prefill_wide_math':'legacy_tiles'})
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        assert digest(ordinary)!=digest(tiled)
