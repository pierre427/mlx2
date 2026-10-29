"""Optional APC expert metadata; synthetic CPU cache state, no model/GPU."""
from dataclasses import FrozenInstanceError,replace
import json
import mlx.core as mx
import pytest

from mlx2.runtime.expert_residency_hints import (
    BINDING_FIELDS,ExpertResidencyHints,MAX_HINT_BYTES,prefix_token_digest)
from mlx2.runtime.apc_v2 import APCKey,APCv2
from mlx2.runtime.cache_planes import CachePlaneKind
from mlx2.runtime.models.cache import KVCache


def hints(tokens=(1,2,3,4),**changes):
    fields=dict(checkpoint='synthetic-artifact@42',source_revision='runtime-source-A',
        quantization='original-fp4-q1-map-A',layout='packed-layer-layout-A',
        resource_abi='synthetic-bank.v1',snapshot_revision='prefix-snapshot-A',
        token_digest=prefix_token_digest(tokens),covered_tokens=len(tokens),
        layer_expert_counts=(8,16),experts=((0,1),(1,4)))
    return ExpertResidencyHints(**{**fields,**changes})


def state(size):
    cache=KVCache();values=mx.arange(size,dtype=mx.float32).reshape(1,1,size,1)
    cache.update_and_fetch(values,values);mx.eval(cache.state);return cache


def key(revision='runtime-source-A'):
    return APCKey('synthetic-artifact@42',revision=revision,adapter='source',
        tokenizer_fingerprint='tokens',cache_layout_fingerprint='layout',
        semantic_fingerprint=('text-token-v1','tenant','test'))


def persistent(path):
    return APCv2(max_size=8,max_bytes=1<<20,layout_name='layout',
        idle_disk_seconds=180,idle_disk_dir=str(path),persist_dir=str(path),
        persist_identity=key(),persist_semantic_namespace='tenant')


@pytest.mark.parametrize('name',BINDING_FIELDS)
def test_every_resource_revision_mismatch_declines_without_mutating_hint(name):
    value=hints();expected=dict(value.binding);expected[name]+='-other'
    assert value.for_restore(4,expected)is None
    assert value.for_restore(4,value.binding)is value


def test_snapshot_identity_distinct_from_artifact_and_trim_is_explicit_superset():
    value=hints();short=value.for_restore(2)
    assert short.coverage=='superset'and short.covered_tokens==4
    assert short.snapshot_revision==value.snapshot_revision and short.token_digest==value.token_digest
    assert short.for_store((1,2))is None
    assert value.for_restore(5)is None
    assert replace(value,phase='speculative').for_restore(4)is None
    assert replace(value,snapshot_revision='another-prefix').binding==value.binding
    assert replace(value,phase='prefill').for_store((1,2,3,4))is not None


@pytest.mark.parametrize('changes',[
    dict(checkpoint=lambda:None),dict(resource_abi=object()),dict(experts=((0,8),)),
    dict(experts=((2,0),)),dict(experts=((1,4),(0,1))),dict(experts=((0,1),(0,1))),
    dict(experts=((True,1),)),dict(layer_expert_counts=(0,)),dict(layer_expert_counts=[8,16]),
    dict(phase='decode'),dict(token_digest='wrong'),dict(covered_tokens=True),
    dict(phase='resident_snapshot'),dict(coverage='advisory'),dict(truncated=True),dict(truncated=1),
    dict(snapshot_revision=''),dict(experts=tuple((0,i)for i in range(4097)),layer_expert_counts=(8192,))])
def test_only_bounded_scalar_id_metadata_allowed(changes):
    with pytest.raises((ValueError,TypeError)):hints(**changes)


def test_manifest_roundtrip_immutable_and_byte_limit():
    value=hints();payload=value.to_manifest()
    assert ExpertResidencyHints.from_manifest(json.loads(json.dumps(payload)))==value
    assert value.nbytes==len(json.dumps(payload,sort_keys=True,separators=(',',':')).encode())
    assert value.nbytes<MAX_HINT_BYTES
    with pytest.raises(FrozenInstanceError):value.experts=()
    assert ExpertResidencyHints.from_manifest({'schema':'future-version'})is None
    assert ExpertResidencyHints.from_manifest({**payload,'experts':[[0,999]]})is None
    assert ExpertResidencyHints.from_manifest({**payload,'huge':'x'*MAX_HINT_BYTES})is None
    old=dict(payload);old.pop('truncated')
    assert ExpertResidencyHints.from_manifest(old)==value


@pytest.mark.parametrize('truncated',[False,True])
def test_resident_inventory_is_advisory_not_a_committed_path_set(truncated):
    value=hints(phase='resident_snapshot',coverage='advisory',truncated=truncated)
    assert value.for_store((1,2,3,4))is value
    restored=value.for_restore(2,value.binding)
    assert restored is value and restored.coverage=='advisory'
    assert restored.phase=='resident_snapshot'and restored.truncated is truncated
    assert restored.covered_tokens==4 and restored.snapshot_revision=='prefix-snapshot-A'
    assert restored.token_digest==prefix_token_digest((1,2,3,4))
    assert restored.for_store((1,2))is None
    assert ExpertResidencyHints.from_manifest(value.to_manifest())==value
    stale=dict(value.binding);stale['resource_abi']='replaced-bank-ABI'
    assert value.for_restore(2,stale)is None


def test_nonexhaustive_resident_inventory_obeys_the_same_finite_key_and_byte_caps():
    value=hints(phase='resident_snapshot',coverage='advisory',truncated=True,
        layer_expert_counts=(8192,),experts=tuple((0,i)for i in range(4096)))
    assert value.nbytes<=MAX_HINT_BYTES and len(value.experts)==4096
    with pytest.raises(ValueError,match='finite metadata bound'):
        replace(value,experts=tuple((0,i)for i in range(4097)))


def test_apc_store_restore_branch_trim_independent_invalidation_and_clear():
    apc=APCv2(max_size=8,max_bytes=1<<20,layout_name='layout');tokens=[1,2,3,4];value=hints()
    assert apc.store(key(),tokens,[state(4)],expert_hints=value).stored
    hit=apc.lookup(key(),tokens+[5],expert_hint_binding=value.binding)
    assert hit.hit and hit.expert_hints==value and hit.cache.cow_metadata.expert_hints==value
    branch=hit.cache.fork();branch[0].trim(2)
    assert branch.cow_metadata.expert_hints.coverage=='superset'
    assert branch.cow_metadata.expert_hints.covered_tokens==4
    branch.close()  # A rejected private branch cannot mutate the stored hint.
    again=apc.lookup(key(),tokens+[9]);assert again.expert_hints==value;again.cache.close()
    shorter=apc.lookup(key(),[1,2,99]);assert shorter.hit and shorter.cached_tokens==2
    assert shorter.expert_hints.coverage=='superset'and shorter.expert_hints.covered_tokens==4
    shorter.cache.close()
    hit.cache.cow_owner.invalidate_plane(CachePlaneKind.EXPERT_HINTS,'resource_abi_changed')
    stale=apc.lookup(key(),tokens+[8]);assert stale.hit and stale.expert_hints is None
    assert stale.expert_hints_miss_reason=='absent_or_declined'
    stale.cache.close();hit.cache.close()
    apc.clear();assert apc.resident_nbytes()==0
    assert not apc.lookup(key(),tokens+[5]).hit
    apc.close()


def test_stale_or_speculative_optional_hints_never_block_valid_kv_and_no_wrong_branch_publish():
    tokens=[1,2,3,4];apc=APCv2(max_size=8,max_bytes=1<<20,layout_name='layout')
    for value in(hints(phase='speculative'),hints(token_digest=prefix_token_digest((8,9,10,11))),None):
        assert apc.store(key(),tokens,[state(4)],expert_hints=value).stored
        hit=apc.lookup(key(),tokens+[5]);assert hit.hit and hit.expert_hints is None;hit.cache.close()
    assert apc.store(key(),tokens,[state(4)],expert_hints=hints()).stored
    wrong=dict(hints().binding);wrong['quantization']='other'
    hit=apc.lookup(key(),tokens+[5],expert_hint_binding=wrong)
    assert hit.hit and hit.expert_hints is None and hit.expert_hints_miss_reason=='binding_mismatch_or_stale'
    hit.cache.close();assert not apc.lookup(key('new-source'),tokens+[5]).hit
    apc.close()


def test_metadata_bytes_charged_and_bounded_by_existing_apc_budget():
    value=hints();tokens=[1,2,3,4]
    a=APCv2(max_size=8,max_bytes=1<<20,layout_name='layout')
    a.store(key(),tokens,[state(4)]);plain=a.resident_nbytes()
    a.store(key(),tokens,[state(4)],expert_hints=value)
    assert a.resident_nbytes()==plain+value.nbytes
    assert a._trie.get(key(),tokens).prompt_cache.cow_owner.plane_stats()['expert_hints']['logical_bytes']==value.nbytes
    a.clear();assert a.resident_nbytes()==0;a.close()
    small=APCv2(max_size=8,max_bytes=plain,layout_name='layout')
    small.store(key(),tokens,[state(4)],expert_hints=value)
    assert small.resident_nbytes()<=plain
    hit=small.lookup(key(),tokens+[5]);assert hit.hit and hit.expert_hints is None;hit.cache.close()
    small.close()


def test_disk_placeholder_host_hint_budget_is_finite_and_optional(tmp_path):
    apc=persistent(tmp_path);tokens=[1,2,3,4]
    apc._expert_hint_metadata_limit=hints().nbytes
    apc.store(key(),tokens,[state(4)],expert_hints=hints(),session_tag=('test','first'))
    apc.park_session('test','first',ttl_seconds=60)
    other=[5,6,7,8]
    assert apc.store(key(),other,[state(4)],expert_hints=hints(tuple(other))).stored
    hit=apc.lookup(key(),other+[9]);assert hit.hit and hit.expert_hints is None;hit.cache.close()
    assert apc.expert_hint_metadata_stats['logical_bytes']==hints().nbytes
    apc.clear();assert apc.expert_hint_metadata_stats['logical_bytes']==0;apc.close()


def test_optional_hint_plane_lease_never_owns_a_physical_bank_or_blocks_model_clear():
    apc=APCv2(max_size=8,max_bytes=1<<20,layout_name='layout');tokens=[1,2,3,4]
    apc.store(key(),tokens,[state(4)],expert_hints=hints())
    owner=apc._trie.get(key(),tokens).prompt_cache.cow_owner
    plane=owner.plane_manifest._owners[CachePlaneKind.EXPERT_HINTS]
    lease=owner.plane_manifest.try_adopt(CachePlaneKind.EXPERT_HINTS,plane.fingerprint)
    # The only leased payload is immutable IDs. No bank/tensor/closer is
    # retained; a resource manager can retire physical weights independently.
    assert lease.payload==hints()and not hasattr(lease.payload,'close')
    apc.clear();assert apc.resident_nbytes()==0
    assert lease.payload==hints();lease.close();apc.close()


@pytest.mark.parametrize('corrupt',[False,True])
def test_persistence_roundtrip_and_bad_optional_metadata_preserve_valid_kv(tmp_path,corrupt):
    tokens=[1,2,3,4];first=persistent(tmp_path)
    first.store(key(),tokens,[state(4)],expert_hints=hints(),session_tag=('test','session'))
    assert first.park_session('test','session',ttl_seconds=60)['state']=='disk'
    first.close()
    manifest=next(tmp_path.glob('*.manifest.json'));payload=json.loads(manifest.read_text())
    assert payload['expert_hints']==hints().to_manifest()
    if corrupt:
        payload['expert_hints']['experts']=[[0,99999]];manifest.write_text(json.dumps(payload))
    restored=persistent(tmp_path);hit=restored.lookup(key(),tokens+[5])
    assert hit.hit and hit.cached_tokens==4
    assert hit.expert_hints==(None if corrupt else hints())
    assert hit.cache[0].keys[0,0,:4,0].tolist()==[0.,1.,2.,3.]
    hit.cache.close();restored.clear();restored.close()


@pytest.mark.parametrize('truncated',[False,True])
def test_advisory_resident_inventory_persists_without_becoming_committed(tmp_path,truncated):
    tokens=[1,2,3,4];value=hints(phase='resident_snapshot',coverage='advisory',truncated=truncated)
    first=persistent(tmp_path)
    first.store(key(),tokens,[state(4)],expert_hints=value,session_tag=('test','inventory'))
    first.park_session('test','inventory',ttl_seconds=60);first.close()
    restored=persistent(tmp_path);hit=restored.lookup(key(),[1,2,99],expert_hint_binding=value.binding)
    assert hit.hit and hit.cached_tokens==2
    assert hit.expert_hints==value and hit.expert_hints.phase=='resident_snapshot'
    assert hit.expert_hints.coverage=='advisory'and hit.expert_hints.truncated is truncated
    assert hit.cache.cow_metadata.expert_hints==value
    hit.cache.close();restored.clear();restored.close()
