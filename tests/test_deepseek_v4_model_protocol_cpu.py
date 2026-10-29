"""Production post-load inspection seam; no affine alias or GPU execution."""
import pytest
import mlx.core as mx
import mlx.nn as nn
mx.set_default_device(mx.cpu)
from test_deepseek_v4_forward_cpu import config,SmallWeights
from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Model


def model():
    c=config();return DeepSeekV4Model(c,SmallWeights(c),identity='protocol-synthetic-cpu')


def test_native_model_detection_and_lane_install_are_honest_noops(monkeypatch):
    from mlx2.runtime.lane import policy,installer,matmul
    from mlx2.serving import row_exact_target_mutation_guard
    from types import SimpleNamespace
    target=model()
    assert list(target.named_modules())==[('',target)]
    assert all(not isinstance(m,(nn.Linear,nn.QuantizedLinear)) for _,m in target.named_modules())
    monkeypatch.setattr(matmul,'backend',lambda:'mpp')
    detected=policy.detect(target,{'model_type':'deepseek4_gguf'})
    assert detected['formats']=={} # native GGUF is not MLX affine2/4/8bit
    actual_geometry=policy.detect(target,{'n_routed_experts':target.args.n_routed_experts})
    assert actual_geometry['moe'] and actual_geometry['formats']=={}
    resolved=policy.resolve(detected,family='deepseek-v4',mode='auto')
    monkeypatch.setattr(installer,'available',lambda:True)
    monkeypatch.setattr(installer,'backend',lambda:'mpp')
    monkeypatch.setattr(installer,'prepare',lambda *a:(_ for _ in ()).throw(AssertionError('native GGUF must never enter affine lane preparation')))
    row_exact_target_mutation_guard(SimpleNamespace(model=target),lane_policy=resolved)
    receipt=installer.apply_policy(target,resolved)
    assert not receipt['covered'] and not receipt['groups']


def test_postload_eval_parameters_cache_and_layer_protocol_cpu():
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities
    target=model()
    assert target.eval() is target
    assert isinstance(target.parameters(),dict)
    assert len(target.layers)==target.args.num_hidden_layers
    cache=make_prompt_cache(target)
    assert len(cache)==len(target.layers)
    capabilities=inspect_apc_capabilities(cache)
    assert capabilities is not None
    assert all(c.identity==target.identity and c.offset==0 for c in cache)


@pytest.mark.parametrize("greedy_contract", [False,True])
def test_tiny_native_model_reaches_real_serving_engine_ready_and_generates_cpu(monkeypatch,tmp_path,greedy_contract):
    from route_harness import patch_host,make_adapter,collect
    from mlx2 import serving
    from mlx2.runtime import weight_residency,generation_stream
    from mlx2.adapters.deepseek_v4_gguf import DeepSeekV4GGUFAdapter,split_execution_policy,CACHE_LAYOUT
    mx.set_default_device(mx.cpu)
    patch_host(monkeypatch)
    monkeypatch.setattr(weight_residency,'wire_serving_weights',lambda:{'fixture':'CPU, no GPU wiring'})
    monkeypatch.setattr(serving.ServingEngine,'_start_residency_heartbeat',lambda self:None)
    monkeypatch.setattr(mx.metal,'is_available',lambda:False)
    monkeypatch.setattr(generation_stream,'generation_stream',mx.new_thread_local_stream(mx.cpu))
    if hasattr(mx,'set_wired_limit'):
        monkeypatch.setattr(mx,'set_wired_limit',lambda *a:(_ for _ in ()).throw(AssertionError('CPU seam must not change wired limit')))
    harness=make_adapter(None,config().vocab_size)
    class Adapter(DeepSeekV4GGUFAdapter):
        tokenizer=harness.tokenizer
        max_context=512
        def __init__(self,path,execution_policy=None):
            # Match real serving construction: create all lazy constants on
            # the generation worker's bound CPU stream, never a foreign one.
            target=model();target.weights._dense=target.weights.data
            for key in ('scheduled_expert_calls','scheduled_grouped_calls','scheduled_dense_calls'):
                setattr(target.weights,key,{})
            self.model=target;self.weights=target.weights;self.identity={'fingerprint':'synthetic-protocol-cpu'}
            self.kernel_policy,self.external_policy=split_execution_policy(execution_policy)
            self.draft_model=None;self.external_record=self.external_source=None
            self.layout=CACHE_LAYOUT;self.config=vars(target.args);self.metadata={};self.environment={}
        prompt_tokens=harness.prompt_tokens
        output_parser=harness.output_parser
        def close(self):pass
    if greedy_contract:
        from dataclasses import replace
        from mlx2.sampling_contract import GREEDY_REFUSAL_SAMPLING_CONTRACT,status_sampling_contract
        from mlx2.adapters.deepseek_v4_gguf import EXTERNAL_SAMPLING
        Adapter.descriptor=replace(Adapter.descriptor,metadata={**Adapter.descriptor.metadata,
            'sampling_contract':GREEDY_REFUSAL_SAMPLING_CONTRACT})
        Adapter.sampling_defaults=EXTERNAL_SAMPLING
    engine=serving.ServingEngine(str(tmp_path),adapter_factory=Adapter,mtp=False,max_lanes=1,max_inflight=1,
        prefill_step=16,max_context=512,cache_bytes=64<<20,residency_keep_seconds=0)
    try:
        assert engine.ready.wait(10),engine.error
        assert engine.error is None,engine.error
        status=engine.status();assert status['state']=='ready' and status['qualification']=='unqualified'
        if greedy_contract:assert status_sampling_contract(status)==GREEDY_REFUSAL_SAMPLING_CONTRACT
        else:assert 'sampling_contract' not in status['settings']
        result=collect(engine.submit({'tokens':[1,2,3,4,5,6,7,8,9], 'max_tokens':2,'temperature':0}),timeout=10)
        assert result.get('error') is None,result
        assert len(result['tokens'])==2
        warm=collect(engine.submit({'tokens':[1,2,3,4,5,6,7,8,9], 'max_tokens':2,'temperature':0}),timeout=10)
        assert warm.get('error') is None,warm
        assert warm['tokens']==result['tokens']
        assert result['receipt']['cached_tokens']==0 and warm['receipt']['cached_tokens']>0
    finally:engine.close()
