"""CPU-only submission observations over real tiny model equations; no checkpoint."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from test_deepseek_v41_cpu import config,Weights

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))


def run(monkeypatch,*,policy='per_layer_async_v1',active=True,rows=1,eval_interval=1,session=True,layers=5):
    c=config(num_hidden_layers=layers,compress_ratios=(0,2,2)+(1,)*(layers-3));weights=Weights(c);model=DeepSeekV41Model(c,weights,identity='synthetic-bank-submission@seed567')
    cache=model.make_cache();model.eval_interval=0
    before=model(mx.array([[1,2,3]],mx.int32),cache);mx.eval(before,[row.state for row in cache])
    retained=[];weights.evaluate_dense_window=lambda *values:retained.append(values)
    bank=SimpleNamespace(active=active,submission_policy=policy,stats={})
    if session:weights.source_resident_bank_session=bank
    calls=[];monkeypatch.setattr(mx,'async_eval',lambda *values:calls.append(values))
    # Only the Python eligibility predicate sees gpu; all arrays/operations use
    # the actual CPU default stream. No shader, Metal call or GPU array runs.
    monkeypatch.setattr(mx,'default_device',lambda:mx.gpu)
    model.eval_interval=eval_interval
    logits,_=model._forward_impl(mx.array([list(range(4,4+rows))],mx.int32),cache,logits_to_keep=1)
    mx.eval(logits,[row.state for row in cache])
    return model,cache,logits,bank,calls,retained


def test_whole_forward_defers_all_layers_but_retains_every_frontier(monkeypatch):
    model,cache,logits,bank,calls,retained=run(monkeypatch,policy='whole_forward_v1')
    assert not calls and bank.stats=={'layer_async_deferred':5}
    assert len(retained)==7  # embedding, each of5blocks, final head
    assert all(row.offset==4 and not row.pending for row in cache)
    assert np.isfinite(np.asarray(logits)).all()


@pytest.mark.parametrize('kwargs',[{},dict(active=False,policy='whole_forward_v1'),
    dict(session=False,policy='whole_forward_v1'),dict(rows=3,policy='whole_forward_v1')])
def test_default_inactive_nonbank_and_prefill_keep_layer_submissions(monkeypatch,kwargs):
    model,cache,logits,bank,calls,retained=run(monkeypatch,**kwargs)
    assert len(calls)==5 and len(retained)==7
    if kwargs.get('session',True)and kwargs.get('active',True)and kwargs.get('rows',1)==1:
        assert bank.stats=={'layer_async_submitted':5}
    else:assert not bank.stats


def test_eval_disabled_remains_disabled(monkeypatch):
    _,_,_,bank,calls,retained=run(monkeypatch,eval_interval=0,policy='whole_forward_v1')
    assert not calls and not bank.stats and len(retained)==7


def test_submission_policy_changes_no_cpu_logits_or_final_cache(monkeypatch):
    a=run(monkeypatch,policy='per_layer_async_v1')
    mx.set_default_device(mx.cpu)
    b=run(monkeypatch,policy='whole_forward_v1')
    np.testing.assert_array_equal(np.asarray(a[2]),np.asarray(b[2]))
    for left,right in zip(a[1],b[1]):
        assert left.meta_state==right.meta_state
        for x,y in zip(left.state,right.state):
            assert (x is None)==(y is None)
            if x is not None:np.testing.assert_array_equal(np.asarray(x),np.asarray(y))


@pytest.mark.parametrize('policy,submitted,deferred',[
    ('layer_group_2_v1',20,20),('layer_group_4_v1',10,30)])
def test_grouped_actual40layer_submission_counts(monkeypatch,policy,submitted,deferred):
    model,cache,logits,bank,calls,retained=run(monkeypatch,policy=policy,layers=40)
    assert len(calls)==submitted
    assert bank.stats=={'layer_async_submitted':submitted,'layer_async_deferred':deferred}
    assert len(retained)==42 and all(row.offset==4 and not row.pending for row in cache)
    assert np.isfinite(np.asarray(logits)).all()


@pytest.mark.parametrize('policy',['layer_group_2_v1','layer_group_4_v1'])
def test_grouped_submission_preserves_cpu_logits_and_all_cache(monkeypatch,policy):
    a=run(monkeypatch,policy='per_layer_async_v1')
    mx.set_default_device(mx.cpu)
    b=run(monkeypatch,policy=policy)
    np.testing.assert_array_equal(np.asarray(a[2]),np.asarray(b[2]))
    for left,right in zip(a[1],b[1]):
        assert left.meta_state==right.meta_state
        for x,y in zip(left.state,right.state):
            assert (x is None)==(y is None)
            if x is not None:np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
