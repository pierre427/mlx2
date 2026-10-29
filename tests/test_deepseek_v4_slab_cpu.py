"""Causal-slab transaction and exact compressed rollback, synthetic CPU only."""
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)

from mlx2.adapters.deepseek_v4_slab import DeepSeekV4CausalSlabVerifyRows
from mlx2.runtime.drafters.deepseek_v4_dspark import DSparkContractError
from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model
from test_deepseek_v4_forward_cpu import config,SmallWeights


class ObservedModel(DeepSeekV4Model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.widths=[]
        self.fail=False
    def forward_with_taps(self,inputs,*args,**kwargs):
        self.widths.append(inputs.shape[1])
        result=super().forward_with_taps(inputs,*args,**kwargs)
        if self.fail:
            raise RuntimeError('injected slab mutation failure')
        return result


def prepared():
    c=config((4,128));model=ObservedModel(c,SmallWeights(c),identity='synthetic-slab-cpu')
    cache=model.make_cache()
    out=model(mx.array(np.arange(126)[None]%c.vocab_size,dtype=mx.int32),cache)
    mx.eval(out,[x.state for x in cache])
    return model,cache


def equal_state(a,b,*,exact=True):
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_state(x,y,exact=exact)
    elif a is None or b is None:
        assert a is b
    elif exact:
        np.testing.assert_array_equal(np.array(a),np.array(b))
    else:
        np.testing.assert_allclose(np.array(a),np.array(b),rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('accepted',range(7))
def test_slab_crosses_ratio4_and128_boundaries_and_replays_exact_accepted_prefix(accepted):
    model,cache=prepared(); expected_model,expected=prepared()
    owner=DeepSeekV4CausalSlabVerifyRows(model,[cache],(0,1))
    tx=owner.begin([6]);inputs=mx.array([[1,2,3,4,5,6]],dtype=mx.int32)
    logits,taps=tx.forward(inputs)
    mx.eval(logits,taps,[x.state for x in cache])
    assert logits.shape==(1,6,19) and taps.shape==(1,6,16)
    assert model.widths==[6] # one genuine target slab, not six Python forwards
    tx.commit([accepted]);mx.eval([x.state for x in cache])
    for i in range(accepted):
        out,hidden=expected_model.forward_with_taps(inputs[:,i:i+1],expected,(0,1))
        mx.eval(out,hidden,[x.state for x in expected])
    assert all(x.offset==126+accepted for x in cache)
    if accepted<6:
        assert model.widths==[6]+[1]*accepted
    else:
        assert model.widths==[6]
    for a,b in zip(cache,expected):
        assert a.meta_state==b.meta_state
        # Partial commits restore and replay exactly. Full-slab vs sequential
        # shape equivalence here is a CPU tolerance diagnostic, not GPU parity.
        equal_state(a.state,b.state,exact=accepted<6)
    assert tx.closed and owner._active is None
    assert owner.receipt_settings['scheduled_slab_forwards']==1
    assert owner.receipt_settings['logical_batch_width']==owner.receipt_settings['physical_batch_width']==1
    assert owner.receipt_settings['maximum_causal_target_rows']==owner.receipt_settings['scheduled_max_causal_target_rows']==6
    assert not owner.receipt_settings['packed_tree_ast']


def test_causal_slab_future_tokens_do_not_change_earlier_rows_or_hc_taps():
    first,cache=prepared();second,other=prepared()
    a=DeepSeekV4CausalSlabVerifyRows(first,[cache],(0,1)).begin([6])
    b=DeepSeekV4CausalSlabVerifyRows(second,[other],(0,1)).begin([6])
    al,at=a.forward(mx.array([[1,2,3,4,5,6]],dtype=mx.int32))
    bl,bt=b.forward(mx.array([[1,2,3,8,9,10]],dtype=mx.int32))
    mx.eval(al,at,bl,bt)
    np.testing.assert_array_equal(np.array(al[:,:3]),np.array(bl[:,:3]))
    np.testing.assert_array_equal(np.array(at[:,:3]),np.array(bt[:,:3]))
    a.abort();b.abort()


def test_slab_failure_restores_all_planes_and_closes_owner():
    model,cache=prepared();saved=[x.snapshot() for x in cache];model.fail=True
    owner=DeepSeekV4CausalSlabVerifyRows(model,[cache],(0,1));tx=owner.begin([6])
    with pytest.raises(RuntimeError,match='injected'):
        tx.forward(mx.array([[1,2,3,4,5,6]],dtype=mx.int32))
    for actual,old in zip(cache,saved):
        assert actual.offset==126
        restored=type(actual).from_state(actual.state,actual.meta_state)
        restored.restore(old)
        equal_state(actual.state,restored.state)
    assert tx.closed and owner._active is None


def test_slab_invalid_width_repetition_and_foreign_shape_fail_closed():
    model,cache=prepared();owner=DeepSeekV4CausalSlabVerifyRows(model,[cache],(0,1))
    for lengths in ([0],[7],[True],[1,1]):
        with pytest.raises(DSparkContractError):owner.begin(lengths)
    tx=owner.begin([2])
    with pytest.raises(DSparkContractError):tx.forward(mx.array([[1]],dtype=mx.int32))
    tx.forward(mx.array([[1,2]],dtype=mx.int32))
    with pytest.raises(DSparkContractError):tx.forward(mx.array([[1,2]],dtype=mx.int32))
    tx.abort()
    assert all(x.offset==126 for x in cache)


def test_config_generic_intermediate_geometry_is_actual_expert_width():
    c=config()
    assert c.intermediate_size==c.moe_intermediate_size==6
