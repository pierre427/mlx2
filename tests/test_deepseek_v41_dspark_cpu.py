"""Synthetic CPU DSpark binding/Markov contracts; no artifact qualification."""
from copy import deepcopy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_dspark import (
    SOURCE_FINGERPRINT,
    DSparkConfig,
)
from mlx2.runtime.drafters.deepseek_v41_dspark import ranked_markov_paths


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    def forbidden(*a,**k):raise AssertionError('DSpark CPU gate reached GPU factory')
    monkeypatch.setattr(mx.fast,'metal_kernel',forbidden)
    yield
    mx.set_default_device(old)


def test_config_rejects_foreign_and_missing_head():
    base={'identity': {'fingerprint':SOURCE_FINGERPRINT},'config': {'text_config':{'num_nextn_predict_layers': 3,
        'dspark_block_size': 5,'dspark_target_layer_ids': [37,38,39],'dspark_noise_token_id': 128799,
        'dspark_markov_rank': 256,'dspark_n_routed_experts': 128,'dspark_num_experts_per_tok': 3}},
        'args': SimpleNamespace(hidden_size=5120,num_hidden_layers=40,n_routed_experts=384,num_experts_per_tok=6,
            vocab_size=129280,hc_mult=4)}
    assert DSparkConfig.from_artifact(base).target_layers==(37,38,39)
    for field,value in [('num_nextn_predict_layers',2),('dspark_block_size',6),('dspark_n_routed_experts',384),
                        ('dspark_target_layer_ids',[36,37,38])]:
        bad=deepcopy(base);bad['config']['text_config'][field]=value
        with pytest.raises(ValueError,match='geometry'):DSparkConfig.from_artifact(bad)
    bad=deepcopy(base);bad['identity']['fingerprint']='foreign-sidecar'
    with pytest.raises(ValueError,match='exact official'):DSparkConfig.from_artifact(bad)


def test_ranked_markov_paths_use_three_distinct_full_roots_and_own_predecessors():
    vocab=11
    base=mx.array(np.stack([np.arange(vocab,dtype=np.float32)*.01 for _ in range(5)]))
    hidden=mx.zeros((5,2),mx.float32)
    seen=[]
    def lookup(token):
        seen.append(token)
        return mx.array([float(token)],mx.float32)
    def project(embed):
        token=int(embed.item());out=np.full((1,vocab),-100.,np.float32);out[0,(token+1)%vocab]=20.
        return mx.array(out)
    paths,confidence=ranked_markov_paths(base,hidden,2,lookup=lookup,project=project,
        confidence=lambda x:mx.sum(x,axis=-1,keepdims=True))
    assert len(paths)==3 and all(len(p)==5 for p in paths)
    assert len({p[0]for p in paths})==3
    assert paths[0]==(3,4,5,6,7)
    for path in paths:
        assert all(path[d]==(path[d-1]+1)%vocab for d in range(1,5))
    assert len(confidence)==3 and all(len(x)==5 for x in confidence)
    assert len(seen)==13 # One shared root bias plus4conditional steps perbranch.


def test_forbidden_root_never_shortens_three_by_five():
    x=mx.zeros((5,9),mx.float32)
    kwargs={'lookup': lambda t:mx.array([t],mx.float32),'project': lambda x:mx.zeros((1,9)),
        'confidence': lambda x:mx.zeros((1,1))}
    paths,_=ranked_markov_paths(x,mx.zeros((5,2)),1,forbidden_root=(0,1,2,3),**kwargs)
    assert len(paths)==3 and all(len(p)==5 and p[0]not in {0,1,2,3}for p in paths)
    with pytest.raises(ValueError,match='enough'):ranked_markov_paths(x,mx.zeros((5,2)),1,forbidden_root=range(8),**kwargs)


@pytest.mark.parametrize('rows,branches,seed',[(4,3,0),(5,2,0),(5,3,99)])
def test_refuse_other_tile_depth_or_invalid_anchor(rows,branches,seed):
    with pytest.raises(ValueError):ranked_markov_paths(mx.zeros((rows,9)),mx.zeros((rows,2)),seed,
        lookup=lambda t:None,project=lambda x:None,confidence=lambda x:None,branches=branches)


def test_nonfinite_markov_or_confidence_refused():
    with pytest.raises(ValueError,match='Nonfinite'):
        ranked_markov_paths(mx.zeros((5,9)),mx.zeros((5,2)),0,lookup=lambda t:mx.zeros((1,)),
            project=lambda x:mx.full((1,9),mx.nan),confidence=lambda x:mx.zeros((1,1)))


def test_branch_safe_forbidden_ids_apply_at_each_depth():
    x=mx.array(np.tile(np.arange(9,dtype=np.float32),(5,1)))
    seen=[]
    def forbidden(depth):
        seen.append(depth);return (8,)if depth else (8,7)
    paths,_=ranked_markov_paths(x,mx.zeros((5,2)),1,lookup=lambda t:mx.array([t],mx.float32),
        project=lambda x:mx.zeros((1,9)),confidence=lambda x:mx.zeros((1,1)),forbidden_at_depth=forbidden)
    assert {p[0]for p in paths}=={4,5,6}
    assert all(p[1:]==(7,7,7,7)for p in paths)
    assert seen==[0,1,2,3,4,1,2,3,4,1,2,3,4]


def test_serving_factory_binds_generic_fixed3x5_ast_and_refuses_missing_oracle(tmp_path,monkeypatch):
    import json

    from test_deepseek_v41_ast_target_cpu import fixture

    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    from mlx2.adapters.deepseek_v41_dspark_ast import SourceDSparkASTBinding
    model,_,_=fixture()
    backbone=SimpleNamespace(config=SimpleNamespace(target_layers=(2,3,4),hidden_size=8,vocab_size=19),
        model=model,target=model,identity='cpu-synthetic-head',record={'fingerprint':'b'*64},
        receipt=lambda:{'qualified':False},close=lambda:None)
    monkeypatch.setattr(artifact_head,'load_source_dspark',lambda *a,**k:backbone)
    adapter=SimpleNamespace(model=model,identity={'fingerprint':'a'*64},tokenizer_receipt={'cpu_test':True})
    binding=SourceDSparkASTBinding(adapter,check_live=lambda:True)
    with pytest.raises(ValueError,match='oracle'):binding.create_external_batch()
    doc=binding.oracle_template()
    assert not any(doc['checks'].values())and doc['numerical_classification']=='unclassified'
    doc.update(receipt_sha256='c'*64,numerical_classification='raw_bits_exact')
    doc['checks']=dict.fromkeys(doc['checks'],True)
    path=tmp_path/'cpu-test-only-oracle.json';path.write_text(json.dumps(doc))
    binding.oracle_path=path
    binding.provider.with_ast_oracle(path,artifact=binding.target_artifact,capture_layers=(37,38,39),target_binding=binding.target_binding)
    batch=binding.create_external_batch()
    try:
        assert batch.num_draft==5
        assert batch.ast_tree_tile_rows==16
    finally:batch.close();binding.close()


@pytest.mark.parametrize('position_law',[False,True,'width_dependent'])
def test_real_six_plane_source_cache_runs_generic_ast_and_ordinary_continuation(tmp_path,monkeypatch,position_law):
    from test_deepseek_v41_ast_target_cpu import PATHS, _clone, fixture, same_cache
    from test_standard_xpress_serving_cpu import reference

    import mlx2.adapters.deepseek_v41_ast_target as target
    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    from mlx2.adapters.deepseek_v41_dspark_ast import (
        SourceDSparkASTBinding,
        _copy_source_cache,
    )
    from mlx2.runtime.ast_planner import (
        AlternativeExecutionBackend,
        NumericalEquivalenceOracle,
    )
    class CPUProvider(target.SourceASTTargetTreeExecutor):
        def __init__(self,model,**kwargs):super().__init__(model,cpu_reference=True,**kwargs)
        def ast_execution_backend(self,profile):
            self.ast_input_widths=(16,)
            return AlternativeExecutionBackend('target_tree_single_session',profile.binding,profile.device,profile.build,
                self.source_revision_token,(16,),profile.row_costs,
                oracle=NumericalEquivalenceOracle(profile.binding,profile.device,profile.build,self.source_revision_token,'CPU_TEST_ONLY',True))
        def forward(self,model,inputs,parents,cache,layers,*,cached=True):
            base=cache[0].offset
            logits,features,transaction=super().forward(model,inputs,parents,cache,layers,cached=cached)
            if position_law:
                depth=[]
                for node,parent in enumerate(parents):depth.append(0 if parent<0 else depth[parent]+1)
                selected=[(int(token)*3+base+depth[node]+1)%19 for node,token in enumerate(inputs)]
                logits=mx.where(mx.arange(19)[None,None,:]==mx.array(selected)[None,:,None],20.,-20.)
            return logits,features,transaction
    model,_,cache=fixture()
    if position_law:
        original=model._forward_impl
        def dependent(inputs,cache=None,**kwargs):
            before=cache[0].offset
            logits,features=original(inputs,cache,**kwargs)
            if logits is not None:
                width=logits.shape[1]
                start=inputs.shape[1]-width
                selected=(inputs[:,start:]*3+mx.arange(before+start+1,before+inputs.shape[1]+1)[None])%19
                if position_law=='width_dependent'and inputs.shape[1]>1:selected=(selected+7)%19
                logits=mx.where(mx.arange(19)[None,None,:]==selected[:,:,None],20.,-20.)
            return logits,features
        monkeypatch.setattr(model,'_forward_impl',dependent)
    before=[_clone(item)for item in cache]
    private=_copy_source_cache(cache)
    same_cache(private,before)
    assert all(not hasattr(item,'values')for item in private)
    model(mx.array([[3]],mx.uint32),private)
    same_cache(cache,before)
    def record_features(hidden,position,history):
        main=hidden[:,:,:8]
        if main.shape[1]:history.append(main,(main,main,main),position)
    backbone=SimpleNamespace(config=SimpleNamespace(target_layers=(2,3,4),hidden_size=8,vocab_size=19),
        model=model,target=model,identity='cpu-synthetic-head',record={'fingerprint':'b'*64},
        receipt=lambda:{'qualified':False},close=lambda:None,record_features=record_features,
        propose_paths=lambda anchor,history,**kwargs:(PATHS,((0.,)*5,)*3))
    monkeypatch.setattr(artifact_head,'load_source_dspark',lambda *a,**k:backbone)
    monkeypatch.setattr(target,'SourceASTTargetTreeExecutor',CPUProvider)
    adapter=SimpleNamespace(model=model,identity={'fingerprint':'a'*64},tokenizer_receipt={'cpu_test':True})
    prompt=[1,2,3,4,5,6,7,8,9]
    expected=reference(model,prompt,10)
    if position_law=='width_dependent':
        whole_expected=expected
        matched=model.make_cache();expected=[]
        model.prefill_body(mx.array([prompt[:-1]],mx.uint32),matched,(2,3,4))
        for step in range(10):
            token=prompt[-1]if step==0 else expected[-1]
            logits,_=model.forward_with_taps(mx.array([[token]],mx.uint32),matched,(2,3,4))
            expected.append(int(mx.argmax(logits[0,-1]).item()))
        assert whole_expected!=expected
    binding=SourceDSparkASTBinding(adapter,check_live=lambda:True)
    binding.oracle_path='CPU_TEST_ONLY'
    batch=binding.create_external_batch()
    try:
        batch.insert([prompt],max_tokens=[10],sampling_configs=[{'sampling_temp':0}])
        actual=[];receipts=[];handoff=False
        for _ in range(64):
            _,responses=batch.next()
            actual.extend(int(response.token)for response in responses)
            receipts.extend(response.speculative_receipt for response in responses if response.speculative_receipt)
            if binding.provider.launches and not handoff:
                for uid in batch.lanes:batch.disable_speculation(uid)
                handoff=True
            if not batch.lanes:break
        else:raise AssertionError('Source generic AST CPU request stalled')
        assert actual==expected
        assert binding.provider.launches==1
        assert batch.scheduler_stats['external_source_ast_six_plane_private_forks']==1
        assert any(row.get('ast_tree_tiles',{}).get('observed_used')is True for row in receipts)
    finally:batch.close();binding.close()


def test_private_source_fork_preserves_all_six_planes_and_rejects_uncommitted_or_foreign():
    from mlx2.adapters.deepseek_v41_dspark_ast import _copy_source_cache
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    cache=DeepSeekV41LayerCache(2,xp=mx,window_size=2,head_dim=8,index_head_dim=4,
        identity='CPU_SOURCE_SIX_PLANES',source=True,history=True,compressed_vocab=5)
    cache.offset=3
    cache.state=(mx.ones((1,1,2,8)),mx.ones((1,1,8))*2,mx.ones((1,1,8))*3,
                 mx.ones((1,1,8))*4,mx.ones((1,1,4))*5,mx.array([1,2,3],mx.int32))
    cache.validate_state()
    cloned=_copy_source_cache([cache])[0]
    assert cloned.meta_state==cache.meta_state and len(cloned.state)==6
    for left,right in zip(cache.state,cloned.state):
        assert left is not right
        np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
    cloned.keys[0,0,0,0]=99
    cloned.hash_tail[0]=4
    assert float(cache.keys[0,0,0,0].item())==1 and int(cache.hash_tail[0].item())==1
    cache._in_forward=True
    with pytest.raises(ValueError,match='committed'):_copy_source_cache([cache])
    with pytest.raises(ValueError,match='six-plane'):_copy_source_cache([SimpleNamespace(keys=None,values=None)])


@pytest.mark.parametrize('failure_site', ['provider', 'continuation', 'oracle'])
def test_failed_public_ast_configure_restores_existing_and_absent_target_contract(monkeypatch, failure_site):
    from test_deepseek_v41_ast_target_cpu import fixture

    import mlx2.adapters.deepseek_v41_ast_target as target
    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    import mlx2.adapters.proposal_path_sources as paths
    from mlx2.adapters.deepseek_v41_dspark_ast import _AST_TARGET_ATTRIBUTES
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter

    model, _, _ = fixture()
    original_rows = object()
    model.progressive_external_target_rows = original_rows
    model.progressive_external_exact_max_lanes = 7
    before = {name: vars(model)[name] for name in _AST_TARGET_ATTRIBUTES if name in vars(model)}
    events = []
    backbone = SimpleNamespace(config=SimpleNamespace(target_layers=(2, 3, 4), hidden_size=8, vocab_size=19),
        model=model, target=model, identity='cpu-transactional-head', record={'fingerprint': 'b' * 64},
        receipt=lambda: {'qualified': False}, close=lambda: events.append('head-close'))
    monkeypatch.setattr(artifact_head, 'load_source_dspark', lambda *a, **k: backbone)

    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + failure_site + ' admission failure')

    if failure_site == 'provider':
        monkeypatch.setattr(target, 'SourceASTTargetTreeExecutor', fail)
    elif failure_site == 'continuation':
        monkeypatch.setattr(paths, 'build_continuation_drafter', fail)
    else:
        monkeypatch.setattr(target.SourceASTTargetTreeExecutor, 'with_ast_oracle', fail)
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    adapter.model, adapter.weights = model, model.weights
    adapter.kernel_policy = {}
    adapter.identity = {'fingerprint': 'a' * 64}
    adapter.tokenizer_receipt = {'cpu_test': True}
    adapter_before = dict(vars(adapter))
    with pytest.raises(RuntimeError, match='injected ' + failure_site):
        adapter.configure_dspark_ast(oracle_path='injected-oracle', check_live=lambda: True)
    assert vars(adapter) == adapter_before
    assert {name: vars(model)[name] for name in _AST_TARGET_ATTRIBUTES if name in vars(model)} == before
    assert model.progressive_external_target_rows is original_rows
    assert events == ['head-close']
    assert '_source_ast_bank_manager' not in vars(model)
    assert '_source_ast_bank_manager' not in vars(model.weights)


def test_ast_close_restores_original_target_contract_and_is_idempotent(monkeypatch):
    from test_deepseek_v41_ast_target_cpu import fixture

    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    from mlx2.adapters.deepseek_v41_dspark_ast import (
        _AST_TARGET_ATTRIBUTES,
        SourceDSparkASTBinding,
    )

    model, _, _ = fixture()
    original = object()
    model.progressive_external_target_rows = original
    model.progressive_external_exact_max_lanes = 7
    model._source_ast_bank_manager = None
    before = {name: vars(model)[name] for name in (*_AST_TARGET_ATTRIBUTES, '_source_ast_bank_manager')
              if name in vars(model)}
    events = []
    backbone = SimpleNamespace(config=SimpleNamespace(target_layers=(2, 3, 4), hidden_size=8, vocab_size=19),
        model=model, target=model, identity='cpu-close-head', record={'fingerprint': 'b' * 64},
        receipt=lambda: {'qualified': False}, close=lambda: events.append('head-close'))
    monkeypatch.setattr(artifact_head, 'load_source_dspark', lambda *a, **k: backbone)
    adapter = SimpleNamespace(model=model, identity={'fingerprint': 'a' * 64}, tokenizer_receipt={'cpu_test': True})
    binding = SourceDSparkASTBinding(adapter, check_live=lambda: True)
    assert model.progressive_external_exact_max_lanes == 1
    assert model.progressive_external_target_rows is not original
    binding.close()
    binding.close()
    assert {name: vars(model)[name] for name in (*_AST_TARGET_ATTRIBUTES, '_source_ast_bank_manager')
            if name in vars(model)} == before
    assert model.progressive_external_target_rows is original
    assert events == ['head-close']
    with pytest.raises(ValueError, match='closed'):
        binding.create_external_batch()


def test_failed_ast_configuration_quarantines_uncertain_cleanup_and_restores_contract(monkeypatch):
    from test_deepseek_v41_ast_target_cpu import fixture

    import mlx2.adapters.deepseek_v41_ast_target as target
    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    from mlx2.adapters.deepseek_v41_dspark_ast import (
        _AST_TARGET_ATTRIBUTES,
        SourceDSparkASTBinding,
    )
    from mlx2.adapters.deepseek_v41_mlx_weights import _UPLOAD_QUARANTINE

    model, _, _ = fixture()
    before = {name: vars(model)[name] for name in _AST_TARGET_ATTRIBUTES if name in vars(model)}
    def failed_close():
        raise RuntimeError('injected uncertain head retirement')
    backbone = SimpleNamespace(config=SimpleNamespace(target_layers=(2, 3, 4), hidden_size=8, vocab_size=19),
        model=model, target=model, identity='cpu-quarantined-head', record={'fingerprint': 'b' * 64},
        close=failed_close)
    monkeypatch.setattr(artifact_head, 'load_source_dspark', lambda *a, **k: backbone)
    def failed_provider(*args, **kwargs):
        raise ValueError('injected provider validation')
    monkeypatch.setattr(target, 'SourceASTTargetTreeExecutor', failed_provider)
    adapter = SimpleNamespace(model=model, identity={'fingerprint': 'a' * 64}, tokenizer_receipt={'cpu_test': True})
    previous = set(_UPLOAD_QUARANTINE)
    owner = None
    try:
        with pytest.raises(ValueError, match='provider validation') as error:
            SourceDSparkASTBinding(adapter, check_live=lambda: True)
        added = set(_UPLOAD_QUARANTINE) - previous
        assert len(added) == 1
        owner = _UPLOAD_QUARANTINE[added.pop()]
        assert owner.model is model and owner.backbone is backbone
        assert any('uncertain head retirement' in note for note in error.value.__notes__)
        assert {name: vars(model)[name] for name in _AST_TARGET_ATTRIBUTES if name in vars(model)} == before
    finally:
        backbone.close = lambda: None
        if owner is not None:
            owner.close()
    assert set(_UPLOAD_QUARANTINE) == previous


def test_ast_head_plan_reserves_eager_tensors_and_work_without_cache_maximum():
    from test_deepseek_v41_embedded_dspark_cpu import artifact

    from mlx2.adapters.deepseek_v41_dspark import source_dspark_admission_plan
    from mlx2.adapters.deepseek_v41_dspark_artifact import dspark_workspace_bound

    source = artifact()
    source['identity']['fingerprint'] = SOURCE_FINGERPRINT
    small = source_dspark_admission_plan(source, expert_cache_bytes=512 << 20)
    large = source_dspark_admission_plan(source, expert_cache_bytes=1 << 30)
    assert small['expert_cache_maximum_reserved'] is False
    assert small['reservation_bytes'] == large['reservation_bytes']
    assert small['imminent_head_expert_progress_bytes'] > 0
    assert small['workspace_bound_bytes'] == (
        dspark_workspace_bound(source['args']) + small['imminent_head_expert_progress_bytes']
    )
    assert small['reservation_bytes'] == sum(small[name] for name in (
        'payload_bytes', 'prepared_bound_bytes', 'staging_bound_bytes', 'workspace_bound_bytes'
    ))


@pytest.mark.parametrize('failure_site', ['before', 'after', None])
def test_ast_head_loader_measures_both_sides_and_restores_owned_headroom(monkeypatch, failure_site):
    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    import mlx2.adapters.deepseek_v41_source_weights as providers
    import mlx2.runtime.drafters.deepseek_v41_dspark as runtime
    from mlx2 import memory

    events = []
    target_stream = SimpleNamespace(_adaptive_headroom=7, _adaptive_protected_bytes=11,
        adaptive_residency_enabled=True)

    def reclaim(*, available_bytes, protected_bytes):
        events.append('reclaim')
        target_stream._adaptive_protected_bytes = protected_bytes

    target_stream.rebalance_headroom = reclaim
    target = SimpleNamespace(weights=SimpleNamespace(artifact={}, expert_stream=target_stream,
        bounds={'admission_limit_bytes': 10000, 'imminent_working_reserve_bytes': 5}))
    record = {'artifact': {}, 'config': SimpleNamespace(top_k=3)}
    plan = {'payload_bytes': 100, 'prepared_bound_bytes': 30, 'staging_bound_bytes': 20,
        'workspace_bound_bytes': 60, 'reservation_bytes': 210}
    monkeypatch.setattr(artifact_head, 'inspect_dspark', lambda source: record)
    monkeypatch.setattr(artifact_head, 'source_dspark_admission_plan', lambda *a, **k: plan)
    monkeypatch.setattr(mx, 'get_active_memory', lambda: 0)

    def available(**kwargs):
        return 0 if ((failure_site == 'before' and 'construct' not in events)
                     or (failure_site == 'after' and 'ready' in events)) else 10000

    monkeypatch.setattr(memory, 'host_available_bytes', available)

    class Weights:
        def __init__(self, artifact, **kwargs):
            events.append('construct')
            self.expert_stream = SimpleNamespace(configure_adaptive_residency=self.configure)

        def configure(self, **kwargs):
            assert kwargs['headroom_bytes'] == plan['workspace_bound_bytes']

        def close(self):
            events.append('head-close')

    original_close = runtime.SourceDSparkBackbone.close

    class Backbone:
        close = original_close

        def __init__(self, model, weights, record, **kwargs):
            self.weights, self._closed = weights, False
            events.append('ready')

        def _live(self):
            pass

    monkeypatch.setattr(providers, 'SourceV41Weights', Weights)
    monkeypatch.setattr(runtime, 'SourceDSparkBackbone', Backbone)
    if failure_site:
        with pytest.raises(MemoryError, match='headroom'):
            artifact_head.load_source_dspark(target, check_live=lambda: True)
        assert ('construct' in events) is (failure_site == 'after')
        assert events.count('head-close') == int(failure_site == 'after')
    else:
        head = artifact_head.load_source_dspark(target, check_live=lambda: True)
        assert head.admission['after_load']['loaded'] == ['dspark_ast']
        assert target_stream._adaptive_headroom == 60
        assert target_stream._adaptive_protected_bytes == 65
        head.close()
        head.close()
        assert events.count('head-close') == 1
    assert target_stream._adaptive_headroom == 7
    assert target_stream._adaptive_protected_bytes == 11
