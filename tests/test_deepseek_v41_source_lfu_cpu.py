"""CPU-only LFU and explicit precision/profile boundary tests."""
import numpy as np
import pytest
from mlx2.adapters.deepseek_v41_source_lfu import LFUIndex,parse_policy,ORIGINAL_BYTES,Q1_BYTES
from mlx2.adapters.deepseek_v41_source import source_policy

def test_lfu_counts_tokens_and_preserves_active_lease():
    index=LFUIndex();a,b,c=(0,0),(0,1),(1,0)
    index.observe(0,np.array([[0,1],[0,1],[0,1]],np.uint32));index.observe(1,np.array([[0]],np.uint32))
    for key in (a,b,c):index.touch(key)
    assert index.victim({a:1,b:1,c:1},{c:1},lambda k:True)==a
    assert index.victim({b:1,c:1},{},lambda k:True)==c

def test_lfu_refreshes_old_heap_frequency_and_lru_ties():
    index=LFUIndex();a,b=(0,0),(0,1)
    index.observe(0,np.array([[0,1]],np.uint32));index.touch(a);index.touch(b)
    index.observe(0,np.array([[0,0,0]],np.uint32))
    assert index.victim({a:1,b:1},{},lambda k:True)==b
    with pytest.raises(MemoryError):index.victim({a:1},{a:1},lambda k:True)

def test_original_and_q1_profiles_do_not_silently_mix():
    original={'mode':'original_lfu','budget_bytes':90_000_000_000}
    assert source_policy({'deepseek_source_experts':original})['deepseek_source_experts']==original
    assert 'deepseek_source_experts' not in source_policy()
    with pytest.raises(ValueError):parse_policy({**original,'q1_root':'/tmp/q1'})
    with pytest.raises(ValueError):parse_policy({**original,'budget_bytes':90_000_000_001})
    with pytest.raises(ValueError):parse_policy({'mode':'hot_original_q1','budget_bytes':90_000_000_000,'hot_bytes':75_000_000_000,'q1_root':'relative','hotset_path':'/tmp/hot'})
    assert ORIGINAL_BYTES==3*5120*2304*(.5+1/32)
    assert Q1_BYTES==3*5120*2304*(1/8+4/64)

def test_lfu_byte_eviction_never_drops_active_lease():
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore
    store=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    a,b,c=(0,0),(0,1),(0,2)
    store.index=LFUIndex();store.index.observe(0,np.array([[0,1,1]],np.uint32))
    for k in [a,b]:store.index.touch(k)
    store._cache={k:{'weight':SimpleNamespace(nbytes=ORIGINAL_BYTES)} for k in [a,b]}
    store._leased={a:store._cache[a]};store._cache_bytes=2*ORIGINAL_BYTES;store.pool_bytes={'original':2*ORIGINAL_BYTES,'q1':0}
    store.payload_ceiling=2*ORIGINAL_BYTES;store.q1_root=None;store.stats={'evictions':0}
    store._evict_for(c,ORIGINAL_BYTES)
    assert set(store._cache)=={a} and store._cache_bytes+ORIGINAL_BYTES<=store.payload_ceiling

def test_atlas_atomic_merge_does_not_double_count(tmp_path):
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore,read_atlas
    store=LFUSourceExpertStore.__new__(LFUSourceExpertStore);store.index=LFUIndex();store.binding='test-source'
    store.atlas_path=tmp_path/'atlas.json';store._atlas_saved=store.index.counts.copy()
    store.index.observe(0,np.array([[1,1,2]],np.uint32));store.index.touch((0,1));store.index.touch((0,2))
    store.persist_atlas();first=read_atlas(store.atlas_path,store.binding)
    store.persist_atlas();assert read_atlas(store.atlas_path,store.binding)==first
    store.index.observe(0,np.array([[1]],np.uint32));store.persist_atlas()
    counts={r['expert']:r['token_selections'] for r in read_atlas(store.atlas_path,store.binding)['counts']}
    assert counts=={1:3,2:1}
    with pytest.raises(ValueError):read_atlas(store.atlas_path,'different-source')

def test_startup_preheat_populates_hot_and_dynamic_pools_by_rank():
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore
    store=LFUSourceExpertStore.__new__(LFUSourceExpertStore);store.index=LFUIndex()
    store.files={(l,e):{} for l in range(40) for e in range(384)}
    store.index.counts[(39,383)]=10;store.index.counts[(38,382)]=9;store.index.counts[(37,381)]=8
    store.hot=frozenset({(39,383),(38,382)});store.q1_root='/q1'
    store.policy={'hot_bytes':2*ORIGINAL_BYTES,'budget_bytes':3*ORIGINAL_BYTES+3*Q1_BYTES}
    store.payload_ceiling=store.policy['budget_bytes']-ORIGINAL_BYTES
    selected=store.preheat_selection()
    assert set(selected[:2])==store.hot
    assert selected[2]==(37,381) and len(selected)==5
    assert sum(ORIGINAL_BYTES if k in store.hot else Q1_BYTES for k in selected)==store.payload_ceiling

@pytest.mark.parametrize('fail', [False, True])
def test_empty_atlas_preheat_fills_pools_and_releases_window(monkeypatch, fail):
    import threading
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore
    from mlx2.runtime import os_memory
    store=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    store.index=LFUIndex();store.files={(0,0):{},(1,0):{},(2,0):{}}
    store.hot=frozenset({(0,0)});store.q1_root='/q1'
    store.policy={'hot_bytes':ORIGINAL_BYTES,'budget_bytes':2*ORIGINAL_BYTES+2*Q1_BYTES}
    store.payload_ceiling=store.policy['budget_bytes']-ORIGINAL_BYTES
    store.preheat_receipt={'requested':True,'performed':False}
    store._cache_bytes=0;store._leased={};store._pending_uploads={};store._pending_host={}
    store._lock=threading.RLock();store._active=False;store._error=None;store.stats={'windows':0}
    store._ready=lambda:None;fences=[];store._fence=lambda:fences.append(True)
    store.counters=lambda:dict(store.stats)
    monkeypatch.setattr(os_memory,'host_memory_snapshot',lambda:SimpleNamespace(available_bytes=1<<40))
    calls=[]
    def acquire(prefix,expert):
        key=(int(prefix.split('.')[1]),expert);calls.append(key)
        store._leased[key]={};store._pending_uploads[key]={};store._pending_host[key]={}
        if fail and len(calls)==2:raise OSError('injected read failure')
        store._cache_bytes+=ORIGINAL_BYTES if key in store.hot else Q1_BYTES
    store.acquire=acquire
    if fail:
        with pytest.raises(OSError,match='injected'):store.preheat()
        assert not store.preheat_receipt['performed']
    else:
        result=store.preheat()
        assert result['performed'] and result['original_objects']==1 and result['q1_objects']==2
        assert result['unseen_fill_objects']==3 and result['payload_bytes']==store.payload_ceiling
        assert len(calls)==3
    assert fences==[True] and not store._active
    assert not store._leased and not store._pending_uploads and not store._pending_host
    assert not store.index.counts

def test_mixed_eviction_preserves_hot_and_leased_cold_objects():
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore
    store=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    hot,a,b,new=(0,0),(0,1),(0,2),(0,3)
    store.index=LFUIndex();store.q1_root='/q1';store.hot=frozenset({hot})
    store.policy={'hot_bytes':ORIGINAL_BYTES,'budget_bytes':2*ORIGINAL_BYTES+2*Q1_BYTES}
    store.payload_ceiling=store.policy['budget_bytes']-ORIGINAL_BYTES
    store._cache={k:{'weight':SimpleNamespace(nbytes=ORIGINAL_BYTES if k==hot else Q1_BYTES)} for k in (hot,a,b)}
    for k in store._cache:store.index.touch(k)
    store._leased={a:store._cache[a]};store._cache_bytes=store.payload_ceiling
    store.pool_bytes={'original':ORIGINAL_BYTES,'q1':2*Q1_BYTES};store.stats={'evictions':0}
    store._evict_for(new,Q1_BYTES)
    assert set(store._cache)=={hot,a}
    assert store.pool_bytes=={'original':ORIGINAL_BYTES,'q1':Q1_BYTES}
    assert store._cache_bytes+Q1_BYTES==store.payload_ceiling
    store._leased[hot]=store._cache[hot]
    with pytest.raises(MemoryError):store._evict_for(new,2*Q1_BYTES)

def test_q1_native_file_seal_reads_and_detects_modification(tmp_path):
    import hashlib
    from safetensors.numpy import save_file
    from mlx2.adapters.deepseek_v41_source_lfu import q1_read
    from mlx2.adapters.deepseek_v41_mlx_weights import _stamp
    arrays={}
    for kind in ('gate','up','down'):
        n,k=(5120,2304) if kind=='down' else (2304,5120)
        arrays[kind+'.qweight']=np.zeros((n,k//32),np.uint32)
        for field in ('scales','biases'):arrays[kind+'.'+field]=np.zeros((n,k//64),np.float16)
    path=tmp_path/'l00-e000.safetensors';save_file(arrays,str(path))
    record={'file':path.name,'size':path.stat().st_size,'stat':_stamp(path.stat()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    actual=q1_read(tmp_path,record)
    assert sum(v.nbytes for v in actual.values())==Q1_BYTES
    assert all(np.array_equal(actual[k],v) for k,v in arrays.items())
    with path.open('r+b') as f:f.seek(-1,2);f.write(b'\x01')
    with pytest.raises(ValueError,match='changed after admission'):q1_read(tmp_path,record)
