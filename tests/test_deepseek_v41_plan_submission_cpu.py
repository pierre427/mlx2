"""Injected scheduling/error oracle and tiny CPU model; never GPU execution."""
import numpy as np
import pytest
import mlx.core as mx

from mlx2.adapters.deepseek_v41_source_submission import shared_before_route


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **k: pytest.fail('GPU forbidden'))


def test_shared_graph_is_submitted_before_any_router_materialization():
    events = []
    shared, routed = object(), object()
    counters = {}
    def build_shared():
        events.append('shared'); return shared
    def submit(value):
        assert value is shared; events.append('submit')
    def build_routed():
        events.append('host-ids'); return routed
    assert shared_before_route(build_shared, build_routed, submit=submit,
                              counters=counters) == (routed, shared)
    assert events == ['shared', 'submit', 'host-ids']
    assert counters == {'source_shared_first_submissions': 1}


@pytest.mark.parametrize('failure', ['shared', 'submit', 'routed'])
def test_errors_propagate_to_existing_owner_without_running_later_stages(failure):
    events=[]; counters={}
    def stage(name):
        def call(*args):
            events.append(name)
            if name == failure: raise RuntimeError(name)
            return object()
        return call
    with pytest.raises(RuntimeError, match=failure):
        shared_before_route(stage('shared'), stage('routed'), submit=stage('submit'),
                            counters=counters)
    order=['shared','submit','routed']
    assert events == order[:order.index(failure)+1]
    assert counters.get('source_shared_first_submissions',0)==int(failure=='routed')


def test_tiny_cpu_model_shared_first_preserves_outputs_and_future_state(monkeypatch):
    from test_deepseek_v41_cpu import config, Weights
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    c=config()
    ordinary=DeepSeekV41Model(c,Weights(c),identity='synthetic@shared-order')
    weights=Weights(c);weights.source_submission_policy='shared_first_v1'
    weights.scheduled_dense_calls={}
    candidate=DeepSeekV41Model(c,weights,identity='synthetic@shared-order')
    events=[]
    monkeypatch.setattr(mx,'async_eval',lambda *v:events.append('submit'))
    a,b=ordinary.make_cache(),candidate.make_cache()
    for ids in ([[1,2,3]],[[4]],[[5,6]]):
        tokens=mx.array(ids,mx.int32)
        np.testing.assert_array_equal(np.asarray(ordinary(tokens,a)),np.asarray(candidate(tokens,b)))
        for x,y in zip(a,b):
            assert x.meta_state==y.meta_state
            for p,q in zip(x.state,y.state):
                if p is None: assert q is None
                else: np.testing.assert_array_equal(np.asarray(p),np.asarray(q))
    assert len(events)==3*c.num_hidden_layers
    assert weights.scheduled_dense_calls['source_shared_first_submissions']==len(events)


def test_policy_is_explicit_and_native_adapter_refuses_it():
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.adapters.deepseek_v41_mlx import execution_policy
    assert 'deepseek_source_submission' not in source_policy()
    assert source_policy({'deepseek_source_submission':'shared_first_v1'})['deepseek_source_submission']=='shared_first_v1'
    with pytest.raises(ValueError): source_policy({'deepseek_source_submission':'unknown'})
    with pytest.raises(ValueError): execution_policy({'deepseek_source_submission':'shared_first_v1'})


def test_shared_first_refuses_routed_first_token_reuse_before_load():
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    with pytest.raises(ValueError,match='reference FFN input'):
        source_policy({'deepseek_source_submission':'shared_first_v1',
            'deepseek_source_ffn_input':'shared_routed_qat_reuse',
            'deepseek_source_dense_qat':'fused_fp32','deepseek_source_expert_math':'routed_fused_fp32'})
    with pytest.raises(ValueError,match='reference FFN input'):
        SourceV41Weights({},expert_cache_bytes=75_000_000_000,
            submission_policy='shared_first_v1',ffn_input_policy='shared_routed_qat_reuse')
