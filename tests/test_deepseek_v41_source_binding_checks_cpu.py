"""Fresh original Source file seals on every LFU acquire; CPU fixtures only."""
import threading
from types import SimpleNamespace

import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters.deepseek_v41_source_artifact import checked_file
from mlx2.adapters.deepseek_v41_source_weights import expert_names
from mlx2.runtime.immutable_weight_pages import file_stamp


def fixture(tmp_path,*,leased=False,file_count=1):
    paths=[]
    for i in range(file_count):
        p=tmp_path/f'shard{i}.safetensors';p.write_bytes(b'stat-only CPU fixture');paths.append(p)
    s=lfu.LFUSourceExpertStore.__new__(lfu.LFUSourceExpertStore)
    s._lock=threading.RLock();s._active=True;s._ready=lambda:None;s.q1_root=None
    keys=[(0,e)for e in range(2)];s.tables={'blk.0':keys};s.root=tmp_path
    s.artifact={'tensors':{name:{'file':paths[i%file_count].name}for key in keys
                for i,name in enumerate(expert_names(*key))},
                'seals':{p.name:file_stamp(p.stat())for p in paths}}
    arrays={key:{'owner':object()}for key in keys};s._cache=dict(arrays)
    s._leased={keys[0]:arrays[keys[0]]}if leased else {}
    s._pending_uploads={};s._pending_host={};s.index=lfu.LFUIndex()
    s.stats={'cache_hits':0,'original_hits':0,'demand_wait_ns':0};s.policy={}
    s._completion_collector=None
    return s,arrays,paths


@pytest.mark.parametrize('leased',[True,False])
@pytest.mark.parametrize('file_count',[1,2,3,6])
def test_every_unique_shard_checked_once_in_first_appearance_order_before_hit_return(tmp_path,monkeypatch,leased,file_count):
    s,arrays,paths=fixture(tmp_path,leased=leased,file_count=file_count);events=[]
    def check(root,name,seal):
        events.append(('check',name))
        assert seal is s.artifact['seals'][name]
        return checked_file(root,name,seal)
    monkeypatch.setattr(lfu,'checked_file',check)
    s._completion_collector=lambda key:events.append(('collector',key))
    assert s.acquire('blk.0',0)is arrays[(0,0)]
    assert events==[('check',p.name)for p in paths]+[('collector',(0,0))]
    assert s._active and s._leased[(0,0)]is arrays[(0,0)]
    assert s.stats['cache_hits']==(0 if leased else 1)


def test_checks_are_never_memoized_across_calls_experts_or_hit_kinds(tmp_path,monkeypatch):
    s,arrays,paths=fixture(tmp_path);calls=[]
    def check(root,name,seal):calls.append(name);return checked_file(root,name,seal)
    monkeypatch.setattr(lfu,'checked_file',check)
    assert s.acquire('blk.0',0)is arrays[(0,0)]  # cached
    assert s.acquire('blk.0',0)is arrays[(0,0)]  # already leased
    assert s.acquire('blk.0',1)is arrays[(0,1)]  # same shard, another expert
    assert calls==[paths[0].name]*3
    assert s.stats['cache_hits']==2 and len(s._leased)==2


@pytest.mark.parametrize('leased',[True,False])
@pytest.mark.parametrize('mutation',['content','symlink','missing'])
def test_changed_cached_or_already_leased_source_refuses_without_releasing_owners(tmp_path,leased,mutation):
    s,arrays,paths=fixture(tmp_path,leased=leased,file_count=2);events=[]
    s._completion_collector=lambda key:events.append(key)
    if mutation=='content':paths[1].write_bytes(b'changed source seal')
    elif mutation=='symlink':paths[1].unlink();paths[1].symlink_to(paths[0])
    else:paths[1].unlink()
    with pytest.raises((ValueError,FileNotFoundError)):s.acquire('blk.0',0)
    assert not events and s._active and s._cache[(0,0)]is arrays[(0,0)]
    assert s._cache[(0,1)]is arrays[(0,1)]
    if leased:assert s._leased[(0,0)]is arrays[(0,0)]
    else:assert not s._leased
    assert not s._pending_host and not s._pending_uploads


@pytest.mark.parametrize('defect',['missing_last_tensor','missing_file_field','missing_seal','bad_path'])
def test_every_original_tensor_and_every_unique_seal_must_resolve_on_each_call(tmp_path,defect):
    s,arrays,paths=fixture(tmp_path,leased=True)
    last=expert_names(0,0)[-1]
    if defect=='missing_last_tensor':del s.artifact['tensors'][last]
    elif defect=='missing_file_field':s.artifact['tensors'][last]={}
    elif defect=='missing_seal':del s.artifact['seals'][paths[0].name]
    else:s.artifact['tensors'][last]['file']='../outside.safetensors';s.artifact['seals']['../outside.safetensors']={}
    with pytest.raises((KeyError,ValueError)):s.acquire('blk.0',0)
    assert s._active and s._leased[(0,0)]is arrays[(0,0)]


def test_new_acquire_verifies_unique_shards_before_starting_bounded_loader(tmp_path,monkeypatch):
    s,arrays,paths=fixture(tmp_path,file_count=2);s._cache={};events=[]
    s._evict_for=lambda key,size:events.append(('evict',key))
    def check(root,name,seal):events.append(('check',name));return checked_file(root,name,seal)
    monkeypatch.setattr(lfu,'checked_file',check)
    def load(function,artifact,names):
        assert function is lfu.read_tensors and artifact is s.artifact
        assert tuple(names)==expert_names(0,0)
        events.append(('loader',(0,0)))
        raise RuntimeError('bounded loader sentinel; no payload read')
    s.pool=SimpleNamespace(call=load)
    with pytest.raises(RuntimeError,match='loader sentinel'):s.acquire('blk.0',0)
    assert events==[('check',p.name)for p in paths]+[('evict',(0,0)),('loader',(0,0))]
    assert s._active and not s._leased and not s._pending_uploads and not s._pending_host
    assert s.stats['demand_wait_ns']>0


def test_invalid_ownership_refuses_before_any_file_access(tmp_path,monkeypatch):
    s,*_=fixture(tmp_path);s._active=False
    monkeypatch.setattr(lfu,'checked_file',lambda *a:pytest.fail('ownership refusal touched files'))
    with pytest.raises(ValueError,match='outside ownership'):s.acquire('blk.0',0)
