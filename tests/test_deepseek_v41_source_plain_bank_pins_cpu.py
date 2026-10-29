"""Plain Source cache ownership; no GPU/model/provider initialization."""
from collections import OrderedDict
from types import SimpleNamespace
import pytest
from mlx2.adapters.deepseek_v41_source_weights import SourceExpertStore


def fixture():
    store=object.__new__(SourceExpertStore)
    store._cache=OrderedDict(((0,n),SimpleNamespace(allocation_nbytes=100))for n in range(3))
    store._cache_bytes=300;store.stats={'evictions':0};store._leased={};store._pending_uploads={}
    store._resident_bank_pinned_keys=frozenset({(0,0),(0,1)})
    return store


def test_plain_source_adaptive_victim_never_selects_one_shared_bank_sibling():
    store=fixture()
    assert store._adaptive_victim()==(0,2)
    store._leased[(0,2)]=store._cache[(0,2)]
    assert store._adaptive_victim()is None
    store._leased.clear();store._pending_uploads[(0,2)]={}
    assert store._adaptive_victim()is None
    assert store._cache_bytes==300


def test_plain_source_direct_eviction_guard_preserves_pins_and_accounting():
    store=fixture()
    with pytest.raises(ValueError,match='individually evict'):
        store._adaptive_evict_key((0,0))
    assert store._cache_bytes==300 and len(store._cache)==3 and store.stats['evictions']==0
    assert store._adaptive_evict_key((0,2))==100
    assert store._cache_bytes==200 and set(store._cache)==set(store._resident_bank_pinned_keys)
