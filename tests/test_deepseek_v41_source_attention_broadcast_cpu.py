"""Synthetic Source attention policy/dataflow/state tests; no GPU/model claim."""
import hashlib
import json
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models import deepseek_v41 as v41
from mlx2.adapters.deepseek_v41_source import source_policy, SOURCE_POLICY_CHOICES, DeepSeekV41SourceAdapter
from mlx2.adapters.deepseek_v41_mlx import execution_policy
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from test_deepseek_v41_cpu import config, Weights
from test_deepseek_v41_kv_fastpaths_cpu import _selector

KEY='deepseek_source_attention_math'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))
    yield
    mx.set_default_device(old)


def test_source_policy_default_optin_prerequisite_and_native_refusal():
    assert KEY not in source_policy()
    assert SOURCE_POLICY_CHOICES[KEY]==('reference','all_retained_broadcast','omlx_nax_bf16_online_v1')
    assert KEY in DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert source_policy({KEY:'reference'})[KEY]=='reference'
    for value in (None,True,1,[],{},'bad'):
        with pytest.raises(ValueError):source_policy({KEY:value})
    for index in (None,'reference'):
        p={KEY:'all_retained_broadcast'}
        if index is not None:p['deepseek_index_policy']=index
        with pytest.raises(ValueError,match='requires finite_all_retained'):source_policy(p)
    good={KEY:'all_retained_broadcast','deepseek_index_policy':'finite_all_retained'}
    assert source_policy(good)[KEY]=='all_retained_broadcast'
    with pytest.raises(ValueError):execution_policy(good)


@pytest.mark.parametrize('option,index', [('unknown','finite_all_retained'),('all_retained_broadcast','reference')])
def test_model_rejects_bad_attention_policy_before_weights(option,index):
    with pytest.raises(ValueError,match='attention'):
        v41.DeepSeekV41Model(config(),object(),identity='synthetic@broadcast',index_policy=index,source_attention_math_policy=option)


def _model(policy='reference'):
    c=config(index_topk=512,candidate_topk_blocks=2048)
    return v41.DeepSeekV41Model(c,Weights(c),identity='synthetic@source-broadcast',
                               index_policy='finite_all_retained',selected_kv_policy='decode_reuse',
                               source_attention_math_policy=policy)


def test_identity_profile_dense_receipt_and_apc_state_isolation(monkeypatch):
    a,b=_model(),_model('all_retained_broadcast')
    assert a.identity!=b.identity
    assert ':source_attention_math='not in a.identity
    assert ':source_attention_math=all_retained_broadcast'in b.identity
    cache=a.make_cache()
    with pytest.raises(ValueError,match='identity'):b(mx.array([[1]],mx.int32),cache)
    ordinary=source_policy({'deepseek_index_policy':'finite_all_retained'})
    broadcast=source_policy({'deepseek_index_policy':'finite_all_retained',KEY:'all_retained_broadcast'})
    digest=lambda p:hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    assert digest(ordinary)!=digest(broadcast)  # Existing adapter constructor/profile and binding use this hash.
    from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
    monkeypatch.setattr(NativeV41Weights,'dense_receipt',lambda self:{})
    w=SourceV41Weights.__new__(SourceV41Weights);w.source_attention_math_policy='all_retained_broadcast'
    assert w.dense_receipt()['source_attention_math']=='all_retained_broadcast'


def test_whole_forward_prefill_exact_and_decode_stays_reference(monkeypatch):
    ref,fast=_model(),_model('all_retained_broadcast');rc,fc=ref.make_cache(),fast.make_cache()
    original=v41.selected_attention;calls=[]
    def selected(*a,**kw):
        calls.append((a[0].shape[2],kw.get('trusted_broadcast',False)))
        return original(*a,**kw)
    monkeypatch.setattr(v41,'selected_attention',selected)
    ids=mx.array([[1,2,3,4,5,6,7]],mx.int32)
    expected=ref(ids,rc);calls.clear();actual=fast(ids,fc)
    np.testing.assert_array_equal(np.array(expected).view(np.uint32),np.array(actual).view(np.uint32))
    assert [b for _,b in calls]==[False,True,True,True,True]
    assert fast.scheduled_kv_calls['source_all_retained_broadcast_attention']==4
    assert fast.scheduled_kv_calls['source_attention_math_reference']==1
    for a,b in zip(rc,fc):
        for x,y in zip(a.state,b.state):
            if x is not None:np.testing.assert_array_equal(np.array(x),np.array(y))
    calls.clear();expected=ref(mx.array([[8]],mx.int32),rc);calls.clear();actual=fast(mx.array([[8]],mx.int32),fc)
    np.testing.assert_array_equal(np.array(expected).view(np.uint32),np.array(actual).view(np.uint32))
    assert [b for _,b in calls]==[False]*5
    assert fast.scheduled_kv_calls['decode_gather_build']>0


@pytest.mark.parametrize('mutation',['remove','ids','count','length','frontier','generation'])
def test_missing_or_stale_trusted_marker_falls_back(monkeypatch,mutation):
    model=_model('all_retained_broadcast');original_select=v41.Attention.select;original_attn=v41.selected_attention;calls=[]
    def select(obj,*a,**kw):
        result=original_select(obj,*a,**kw)
        if obj.layer==1:
            shared=a[4]
            marker=shared['_all_retained_chronological']
            if mutation=='remove':shared.pop('_all_retained_chronological')
            elif mutation=='ids':shared['indices']=mx.asarray(shared['indices'],copy=True)
            else:
                slot={'count':1,'length':2,'frontier':3,'generation':4}[mutation]
                changed=list(marker);changed[slot]+=1;shared['_all_retained_chronological']=tuple(changed)
        return result
    def selected(*a,**kw):calls.append(kw.get('trusted_broadcast',False));return original_attn(*a,**kw)
    monkeypatch.setattr(v41.Attention,'select',select);monkeypatch.setattr(v41,'selected_attention',selected)
    model(mx.array([[1,2,3,4,5]],mx.int32),model.make_cache())
    assert calls[:3]==[False,False,False]
    assert calls[3:]==[True,True]


def test_selector_marks_only_finite_fastpath_and_invalidates_before_failure():
    a,s,shared,x,qr,start,_=_selector(512,policy='finite_all_retained')
    a.select(x,qr,s,start,shared)
    assert shared['_all_retained_chronological']==(id(shared['indices']),512,3,start,1)
    # The score-derived selector does not inherit a marker from an earlier generation.
    b,s2,_,x2,qr2,start2,_=_selector(513,policy='finite_all_retained')
    b.linear=lambda *a:(_ for _ in ()).throw(RuntimeError('fixture projection'))
    with pytest.raises(RuntimeError,match='fixture'):b.select(x2,qr2,s2,start2,shared)
    assert '_all_retained_chronological'not in shared


@pytest.mark.parametrize('rows,count,gathered',[(1,8,False),(2,513,False),(1,8,True)])
def test_low_level_unsupported_shapes_keep_gather_and_decode_reuse(monkeypatch,rows,count,gathered):
    rng=np.random.default_rng(14);d=8
    q=mx.array(rng.normal(size=(1,2,rows,d)).astype(np.float32))
    local=mx.array(rng.normal(size=(1,1,4,d)).astype(np.float32));mask=mx.ones((1,1,rows,4),mx.bool_)
    pooled=mx.array(rng.normal(size=(1,count,d)).astype(np.float32))
    ids=mx.broadcast_to(mx.arange(count,dtype=mx.uint32)[None,None],(1,rows,count))
    valid=mx.ones(ids.shape,mx.bool_);sinks=mx.zeros((2,),mx.float32)
    selected=v41._selected_rows;seen=[]
    monkeypatch.setattr(v41,'_selected_rows',lambda *a:seen.append(a)or selected(*a))
    kwargs={'gathered':selected(pooled,ids)}if gathered else{}
    expected=v41.selected_attention(q,local,mask,pooled,ids,valid,sinks,**kwargs);seen.clear()
    actual=v41.selected_attention(q,local,mask,pooled,ids,valid,sinks,trusted_broadcast=True,**kwargs)
    np.testing.assert_array_equal(np.array(expected).view(np.uint32),np.array(actual).view(np.uint32))
    assert bool(seen)is(not gathered)


def test_broadcast_changes_only_row_materialization_retains_tile_plan(monkeypatch):
    rng=np.random.default_rng(9);rows,count,d,heads=7,5,8,2
    q=mx.array(rng.normal(size=(1,heads,rows,d)).astype(np.float32));pool=mx.array(rng.normal(size=(1,count,d)).astype(np.float32))
    local=mx.array(rng.normal(size=(1,1,rows,d)).astype(np.float32));mask=mx.tril(mx.ones((rows,rows),mx.bool_))[None,None]
    ids=mx.broadcast_to(mx.arange(count,dtype=mx.uint32)[None,None],(1,rows,count));valid=mx.ones(ids.shape,mx.bool_)
    sinks=mx.array([.1,-.2],mx.float32)
    # Tiny workspace forces multiple tiles, including a tail.
    args=(q,local,mask,pool,ids,valid,sinks)
    expected=v41.selected_attention(*args,workspace_bytes=1600)
    monkeypatch.setattr(v41,'_selected_rows',lambda *a:(_ for _ in ()).throw(AssertionError('gather forbidden')))
    actual=v41.selected_attention(*args,workspace_bytes=1600,trusted_broadcast=True)
    np.testing.assert_array_equal(np.array(expected).view(np.uint32),np.array(actual).view(np.uint32))
