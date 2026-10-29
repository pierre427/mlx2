"""Synthetic integrated16-row Source/APCv2 contracts; no GPU admission."""
import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_ast_target_cpu import fixture, PATHS
from test_deepseek_v41_ast_batched_attention_cpu import clone, compare
from mlx2.adapters.deepseek_v41_ast_batched_attention import LayerAncestrySegments, PARENTS
from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
from mlx2.runtime.cow_cache import freeze_prompt_cache
from mlx2.runtime.cache_planes import CachePlaneKind
from mlx2.adapters.deepseek_v41_ast_batched_target import PROFILE


def compare_private_payload(left,right):
    assert left.ast_private_profile==PROFILE
    assert left.meta_state[0]!=right.meta_state[0]  # Honest separate state profile.
    assert left.meta_state[1:]==right.meta_state[1:]
    assert left._pool_count==right._pool_count and left._capacity==right._capacity
    for a,b in zip(left.state,right.state):
        assert(a is None)==(b is None)
        if a is not None:np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=8e-6,rtol=8e-5)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def run(model,cache,paths=PATHS):
    return execute_batched_tree(model,cache,3,paths,(2,3,4),
                                cpu_reference=True,expert_law='ordinary_m16_v1')


def test_allnodes_logits_features_cache_ordinary_ancestry_and_selected_materialization_only(monkeypatch):
    model,weights,cache=fixture();original=[clone(row)for row in cache]
    tokens=(3,)+tuple(v for path in PATHS for v in path)
    frontiers=[];expected=[]
    for node,parent in enumerate(PARENTS):
        local=[clone(row)for row in(original if parent<0 else frontiers[parent])]
        expected.append(model._forward_impl(mx.array([[tokens[node]]],mx.int32),local,
                                            logits_to_keep=0,capture_layers=(2,3,4)))
        frontiers.append(local)
    calls=[];implementation=LayerAncestrySegments.materialize
    def materialize(self,node):
        calls.append((self,node));return implementation(self,node)
    monkeypatch.setattr(LayerAncestrySegments,'materialize',materialize)
    moe=[];ordinary=weights.moe
    def grouped(x,ids,scores,prefix):
        moe.append((x.shape,ids.shape));return ordinary(x,ids,scores,prefix)
    monkeypatch.setattr(weights,'moe',grouped)
    result=run(model,cache)
    assert calls==[]
    assert moe==[((16,8),(16,2))]*5
    assert result.receipt['physical_target_invocations']==1
    assert result.receipt['materialized_node_frontiers_during_forward']==0
    assert result.receipt['attention_rows_per_call']==16
    for node in range(16):
        np.testing.assert_allclose(np.asarray(result.logits[:,node:node+1]),np.asarray(expected[node][0]),atol=8e-6,rtol=8e-5)
        np.testing.assert_allclose(np.asarray(result.features[:,node:node+1]),np.asarray(expected[node][1]),atol=8e-6,rtol=8e-5)
        for actual,reference in zip(result.transaction.diagnostic_cache(result.transaction.paths[node]),frontiers[node]):
            compare_private_payload(actual,reference)
    calls.clear()
    with pytest.raises(ValueError,match='exact publication admission'):
        result.transaction.commit_paths([(0,6,7,8)])
    assert calls==[]
    selected=result.transaction.diagnostic_cache((0,6,7,8))
    assert len(calls)==5 and {node for _,node in calls}=={8}
    result.transaction.abort()
    assert result.transaction.closed
    assert model._mlx2_ds41_ast_active is None
    for a,b in zip(selected,frontiers[8]):compare_private_payload(a,b)
    for a,b in zip(cache,original):compare(a,b)


def test_sibling_poison_does_not_change_other_branch_and_abort_releases_no_state():
    model,_,cache=fixture();saved=[clone(row)for row in cache]
    first=run(model,cache);logits=np.asarray(first.logits).copy();features=np.asarray(first.features).copy()
    first.transaction.abort()
    second=run(model,cache,(PATHS[0],(18,)*5,(17,)*5))
    np.testing.assert_array_equal(np.asarray(second.logits)[:,:6],logits[:,:6])
    np.testing.assert_array_equal(np.asarray(second.features)[:,:6],features[:,:6])
    with pytest.raises(ValueError,match='root-connected'):second.transaction.diagnostic_cache((0,1,7))
    second.transaction.abort()
    for a,b in zip(cache,saved):compare(a,b)


def test_scope_factory_failure_restores_active_count_and_live_cache(monkeypatch):
    model,weights,cache=fixture();saved=[clone(row)for row in cache]
    def refused(rows):raise ValueError('synthetic scope refusal')
    monkeypatch.setattr(weights,'execution_scope_for_rows',refused,raising=False)
    with pytest.raises(ValueError,match='scope refusal'):run(model,cache)
    assert model._weight_executions==0 and model._mlx2_ds41_ast_active is None
    for a,b in zip(cache,saved):compare(a,b)


@pytest.mark.parametrize('invalidation',('owner','plane','closed_branch'))
def test_actual_apcv2_frozen_segment_branch_invalidates_selected_publication(invalidation):
    model,_,cache=fixture()
    frozen,_=freeze_prompt_cache(cache,key='synthetic-source@seed567',tokens=(1,2),
                                cache_type='deepseek_v41',layer_segments=True)
    branch=frozen.branch();owner=frozen.cow_owner
    try:
        result=run(model,branch)
        assert result.receipt['prefix_ownership']=='apcv2_descriptor_snapshot'
        assert result.receipt['apcv2_lineage_bound']
        assert owner.pin_count>0
        if invalidation=='owner':frozen.close()
        elif invalidation=='closed_branch':branch.close()
        else:
            owner.invalidate_plane(CachePlaneKind.ATTENTION_RING,'synthetic integration invalidation')
        with pytest.raises(ValueError,match='changed|closed|stale|incomplete'):
            result.transaction.diagnostic_cache((0,))
        result.transaction.abort()
    finally:
        branch.close();frozen.close()


def test_live_prefix_descriptor_change_refuses_commit_without_donation():
    model,_,cache=fixture();result=run(model,cache)
    cache[0].keys=mx.asarray(cache[0].keys,copy=True)
    with pytest.raises(ValueError,match='changed'):result.transaction.diagnostic_cache((0,))
    result.transaction.abort()


def test_existing_strict_engram_window_hash_tail_and_ordinary_ancestry_oracle(monkeypatch):
    # Reapply the original executor's strict pending-window/complete-history
    # oracle to the new batched executor, including its injected failure leg.
    import test_deepseek_v41_ast_target_cpu as original_oracle
    def batched(*args,**kwargs):
        return execute_batched_tree(*args,**kwargs,expert_law='ordinary_m16_v1')
    def private_same_cache(left,right):
        # Diagnostic frontier metadata intentionally carries its unadmitted
        # native profile. Keep the complete six-plane/history payload oracle
        # while asserting that this is not ordinary metadata equivalence.
        for actual,reference in zip(left,right):
            compare_private_payload(actual,reference)
    monkeypatch.setattr(original_oracle,'execute_tree',batched)
    monkeypatch.setattr(original_oracle,'same_cache',private_same_cache)
    original_oracle.test_engram_hash_tail_and_attention_input_features_follow_only_ancestry()


def test_failed_completion_retains_shared_owners_and_poisons_target(monkeypatch):
    model,weights,cache=fixture();saved=[clone(row)for row in cache]
    def fail(*args,**kwargs):raise RuntimeError('synthetic expert failure')
    def failed_fence(*args,**kwargs):raise RuntimeError('synthetic completion failure')
    monkeypatch.setattr(weights,'moe',fail);monkeypatch.setattr(mx,'eval',failed_fence)
    with pytest.raises(RuntimeError,match='expert failure'):run(model,cache)
    assert model._mlx2_ds41_ast_poisoned
    roots,base,segments,transaction=model._mlx2_ds41_ast_failed_roots
    assert roots and len(base)==5 and len(segments)==5 and segments[0]is not None
    assert transaction.closed and model._weight_executions==0
    for a,b in zip(cache,saved):compare(a,b)


def test_dspark_descriptor_private_fork_retains_one_cow_lease_and_closes_exactly_once(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_dspark_ast import SourceDSparkASTBinding
    import mlx2.adapters.deepseek_v41_dspark as artifact_head
    from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
    model,_,cache=fixture()
    backbone=SimpleNamespace(config=SimpleNamespace(target_layers=(2,3,4),hidden_size=8,vocab_size=19),
        model=model,target=model,identity='synthetic-descriptor-head',record={'fingerprint':'b'*64},
        receipt=lambda:{'qualified':False},close=lambda:None)
    monkeypatch.setattr(artifact_head,'load_source_dspark',lambda *a,**k:backbone)
    adapter=SimpleNamespace(model=model,identity={'fingerprint':'a'*64},tokenizer_receipt={'CPU':True})
    # This cache-ownership fixture models an already prepared target. Actual
    # adapters bind these laws before loading/cache creation; late binding is
    # intentionally refused by the production constructor.
    model._mlx2_source_ast_target_laws=('ordinary_m16_v1','fully_batched_v1')
    binding=SourceDSparkASTBinding(adapter,check_live=lambda:True,forward_law='fully_batched_v1')
    doc=binding.oracle_template();doc.update(receipt_sha256='c'*64,numerical_classification='raw_bits_exact')
    doc['checks']=dict.fromkeys(doc['checks'],True)
    path=tmp_path/'synthetic-only-not-artifact-admission.json';path.write_text(json.dumps(doc))
    binding.oracle_path=path
    binding.provider.with_ast_oracle(path,artifact=binding.target_artifact,capture_layers=(37,38,39),target_binding=binding.target_binding)
    batch=binding.create_external_batch()
    frozen,_=freeze_prompt_cache(cache,key='synthetic-dspark-private@seed567',tokens=(1,2),
                                cache_type='deepseek_v41',layer_segments=True)
    owner=frozen.cow_owner;live=frozen.branch();private=None
    try:
        assert owner.pin_count==1
        def forbidden(*a,**k):raise AssertionError('six-plane deep copy during descriptor private fork')
        with monkeypatch.context()as guard:
            guard.setattr(DeepSeekV41LayerCache,'from_state',forbidden)
            private=batch._copy_ast_private_cache(live)
        assert private.cow_owner is owner and private.cow_generation==live.cow_generation
        assert owner.pin_count==2
        assert private.cow_prep_telemetry['descriptor_aliases']>0
        assert private.cow_prep_telemetry['avoided_copy_bytes']>0
        assert len(private)==len(model.layers)  # One prefix clone, not16 frontiers.
        for a,b in zip(live,private):compare(a,b)
        private.close();private.close()
        assert owner.pin_count==1 and not live.closed
        frozen.close()
        assert not owner.source_released  # The original live branch still pins it.
        live.close()
        assert owner.pin_count==0 and owner.source_released
    finally:
        if private is not None:private.close()
        live.close();frozen.close();batch.close();binding.close()


def test_lazy_frontier_cannot_launder_private_profile_through_snapshot():
    model,_,cache=fixture();result=run(model,cache)
    try:
        view=result.frontiers[0][0]
        assert view.ast_private_profile==PROFILE
        assert ':private:'+PROFILE in view.meta_state[0]
        with pytest.raises(Exception,match='private'):view.snapshot()
    finally:result.transaction.abort()


def test_private_same_profile_checkpoint_can_branch_without_becoming_publishable():
    from mlx2.runtime.checkpoint_publication import require_checkpoint_publishable, CheckpointPublicationUnsafe
    from mlx2.adapters.deepseek_v41_ast_batched_target import _snapshot, PROFILE
    model, _, cache = fixture()
    first = run(model, cache)
    private = first.transaction.diagnostic_cache((0, 6))
    first.transaction.abort()
    revision = first.transaction.private_source_revision
    recovered = _snapshot(private, PROFILE,revision)
    assert all(row.ast_private_profile == PROFILE for row in recovered)
    assert all(row.ast_private_revision == revision for row in recovered)
    with pytest.raises(ValueError, match='another numerical profile'):
        _snapshot(private, 'foreign-profile',revision)
    with pytest.raises(ValueError, match='another numerical profile'):
        _snapshot(private, PROFILE,'foreign-source-revision')
    second = run(model, recovered)
    successor = second.transaction.diagnostic_cache((0, 1))
    second.transaction.abort()
    assert successor[0].offset == private[0].offset + 2
    with pytest.raises(CheckpointPublicationUnsafe, match='private'):
        require_checkpoint_publishable(successor)
    with pytest.raises(ValueError, match='exact publication admission'):
        second.transaction.prepared_cache((0,))
