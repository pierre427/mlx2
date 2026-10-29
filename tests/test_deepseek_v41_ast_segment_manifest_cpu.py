"""Private APCv2 addresses and parent invalidation; no model qualification."""
import mlx.core as mx
import pytest

from test_deepseek_v41_cpu import config,Weights
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.cow_cache import freeze_prompt_cache
from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
from mlx2.adapters.deepseek_v41_ast_segment_manifest import build_layer_manifest,checkpoint_manifest


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def test_private_layer_addresses_bind_real_parent_fullsource_span_without_node_copies(monkeypatch):
    c=config(sliding_window=2);model=DeepSeekV41Model(c,Weights(c),identity='synthetic-segment-manifest@567')
    cache=model.make_cache();model._forward_impl(mx.array([[1,2,3,4,5,6]],mx.int32),cache)
    frozen,_=freeze_prompt_cache(cache,key='source-segments@567',tokens=range(6),cache_type='deepseek_v41',layer_segments=True)
    owner=frozen.cow_owner;branch=frozen.branch();result=None
    try:
        result=execute_batched_tree(model,branch,3,((4,5,6,7,8),(9,10,11,12,13),(14,15,16,17,18)),
                                    (2,3,4),cpu_reference=True,expert_law='ordinary_m16_v1')
        for descriptor in result.transaction.segments:
            monkeypatch.setattr(descriptor,'materialize',lambda *a: (_ for _ in ()).throw(AssertionError('node copy')))
        kwargs=dict(lineage_id=owner.lineage_id,generation=owner.generation,
                    source_revision_token=result.transaction.provider_revision_token,parent_segments=owner.segment_manifest.segments)
        source=build_layer_manifest(result.transaction.segments[1],1,**kwargs)
        local=build_layer_manifest(result.transaction.segments[0],0,**kwargs)
        assert source.parent_segments and source.parent_segments[0].key.token_start==4
        assert source.prefix.key.token_start==0 and source.prefix.key.token_stop==6
        assert local.prefix.key.token_start==4
        assert source.checkpoints[5].key.token_start==0 and source.checkpoints[5].key.token_stop==12
        assert local.checkpoints[5].key.token_start==10 and local.checkpoints[5].key.token_stop==12
        assert len(set(item.key for item in source.checkpoints))==16
        assert source==build_layer_manifest(result.transaction.segments[1],1,**kwargs)
        different=build_layer_manifest(result.transaction.segments[1],1,**{**kwargs,'generation':owner.generation+1})
        assert source.prefix.fingerprint!=different.prefix.fingerprint
        with pytest.raises(ValueError,match='invalidated'):different.assert_parent_current(owner)
        assert not source.receipt()['published_to_APCv2']and not source.receipt()['exact_publication_admitted']
        import json
        selected=checkpoint_manifest(result.transaction,(0,6,7,8))
        json.dumps(selected)
        assert selected['selected_node']==8 and selected['node_materializations']==0
        assert selected['layers'][1]['selected_checkpoint']['token_start']==0
        assert selected['layers'][0]['selected_checkpoint']['token_start']==8
        assert selected['layers'][1]['parent_segments'][0]['role']=='window'
        with pytest.raises(ValueError,match='root-connected'):
            checkpoint_manifest(result.transaction,(0,1,7))
        old=branch[0].window_size;branch[0].window_size+=1
        with pytest.raises(ValueError,match='changed'):
            checkpoint_manifest(result.transaction,(0,))
        branch[0].window_size=old
        source.assert_parent_current(owner)
        owner.invalidate_segment(source.parent_segments[0].key,'synthetic source checkpoint invalidation')
        with pytest.raises(ValueError,match='invalidated'):source.assert_parent_current(owner)
        with pytest.raises(ValueError,match='generation or segment changed'):
            checkpoint_manifest(result.transaction,(0,))
    finally:
        if result is not None:result.transaction.abort()
        branch.close();frozen.close()


def _profile_fixture():
    from types import SimpleNamespace
    from mlx2.runtime.cache_planes import CacheLayerSegment,CachePlaneFingerprint,CachePlaneKind,CacheSegmentKey
    key=CacheSegmentKey(CachePlaneKind.ATTENTION_RING,0,0,6,10,'window')
    parent=CacheLayerSegment(key,CachePlaneFingerprint.from_fields(key.kind,schema_version=3,fixture='profile-bound-parent'),32)
    base=SimpleNamespace(identity='synthetic-manifest-profile@1',ratio=0,head_dim=8,index_head_dim=4,
        source=False,history=False,window_size=4,compressed_vocab=32,offset=10,state=(),ast_private_profile=None)
    descriptor=SimpleNamespace(base=base,assert_current=lambda:None,
        offset=lambda node:11+node,local_rows=tuple((0,1,2,3)for _ in range(16)))
    return descriptor,parent


def test_same_parent_native_and_exact_profiles_have_distinct_bound_fingerprints():
    from mlx2.adapters.deepseek_v41_ast_segment_manifest import PROFILE,M1_PROFILE,KNOWN_PROFILES
    from mlx2.adapters.deepseek_v41_ast_batched_target import PROFILES
    descriptor,parent=_profile_fixture()
    kwargs=dict(lineage_id='profile-lineage',generation=7,source_revision_token='same-source-revision',parent_segments=(parent,))
    native=build_layer_manifest(descriptor,0,**kwargs)
    exact=build_layer_manifest(descriptor,0,**kwargs,numerical_profile=M1_PROFILE)
    assert set(KNOWN_PROFILES)==set(PROFILES.values())
    assert native.numerical_profile==native.receipt()['numerical_profile']==PROFILE
    assert exact.numerical_profile==exact.receipt()['numerical_profile']==M1_PROFILE
    from dataclasses import replace
    with pytest.raises(ValueError,match='foreign numerical profile'):
        replace(native,numerical_profile=M1_PROFILE)
    assert native.parent_segments==exact.parent_segments
    assert native.prefix.key==exact.prefix.key and native.prefix.fingerprint!=exact.prefix.fingerprint
    assert all(a.key==b.key and a.fingerprint!=b.fingerprint for a,b in zip(native.checkpoints,exact.checkpoints))
    assert not exact.receipt()['published_to_APCv2']and not exact.receipt()['exact_publication_admitted']


@pytest.mark.parametrize('profile',[None,'','foreign-profile',[],False])
def test_unknown_numerical_profiles_refused(profile):
    descriptor,_=_profile_fixture()
    with pytest.raises(ValueError,match='unknown numerical profile'):
        build_layer_manifest(descriptor,0,lineage_id='profile-lineage',generation=7,
            source_revision_token='same-source-revision',numerical_profile=profile)


def test_transaction_receipt_uses_real_exact_profile_at_every_level_and_foreign_base_refused():
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_ast_segment_manifest import M1_PROFILE,PROFILE
    descriptor,parent=_profile_fixture()
    owner=SimpleNamespace(lineage_id='profile-lineage',branchable=lambda expected_generation:expected_generation==7,
        segment_manifest=SimpleNamespace(segments=(parent,)))
    transaction=SimpleNamespace(profile=M1_PROFILE,_validate=lambda path:tuple(path),cow_owner=owner,
        cow_generation=7,provider_revision_token='same-source-revision',segments=(descriptor,),_exact_publication_admitted=False)
    result=checkpoint_manifest(transaction,(0,))
    assert result['numerical_profile']==M1_PROFILE
    assert result['layers'][0]['numerical_profile']==M1_PROFILE
    assert not result['published_to_APCv2']and not result['exact_publication_admitted']
    transaction.profile='unknown'
    with pytest.raises(ValueError,match='unknown numerical profile'):checkpoint_manifest(transaction,(0,))
    descriptor.base.ast_private_profile=M1_PROFILE
    with pytest.raises(ValueError,match='foreign private numerical profile'):
        build_layer_manifest(descriptor,0,lineage_id='profile-lineage',generation=7,
            source_revision_token='same-source-revision',numerical_profile=PROFILE)
