"""An immutable precision map is separate from adaptive resident capacity."""
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_lfu import (
    LFUIndex, LFUSourceExpertStore, ORIGINAL_BYTES, Q1_BYTES, parse_policy,
)
from mlx2.adapters.deepseek_v41_source_weights import SourceExpertStore


def policy(budget=32<<30):
    return {'mode':'hot_original_q1','budget_bytes':budget,'hot_bytes':60_000_000_000,
            'q1_root':'/models/source-q1','hotset_path':'/models/hot60-map.json'}


@pytest.mark.parametrize('budget',[32<<30,48<<30,75_000_000_000])
def test_adaptive_capacity_does_not_rewrite_precision_map(budget):
    value=policy(budget)
    # Pool split is a residency default; the precision map is untouched.
    expected={**value,'residency':'split_v1'}
    assert parse_policy(value)==expected
    assert source_policy({'deepseek_source_experts':value})['deepseek_source_experts']==expected


def test_legacy_parser_preserves_fixed_pool_validation():
    with pytest.raises(ValueError,match='fixed-pool budget split'):
        parse_policy(policy(),adaptive=False)
    assert parse_policy(policy(75_000_000_000),adaptive=False)==policy(75_000_000_000)


def test_precision_map_still_has_historical_positive_bound():
    for invalid in (True,0,-1,90_000_000_000):
        with pytest.raises(ValueError,match='precision-map budget'):
            parse_policy({**policy(),'hot_bytes':invalid})
    with pytest.raises(ValueError,match='residency mode'):
        parse_policy(policy(),adaptive=1)


def test_legacy_transition_rejects_before_mutating_residency(monkeypatch):
    value=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    value.policy=policy();value._adaptive_residency_enabled=True
    calls=[]
    monkeypatch.setattr(SourceExpertStore,'configure_adaptive_residency',lambda *a,**k:calls.append((a,k)))
    with pytest.raises(ValueError,match='fixed-pool budget split'):
        value.configure_adaptive_residency(False)
    assert value._adaptive_residency_enabled is True and not calls


def test_valid_legacy_transition_keeps_original_pool_contract(monkeypatch):
    value=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    value.policy=policy(75_000_000_000);calls=[]
    value.cache_budget_bytes=75_000_000_000
    # An adaptive eviction previously lowered the fixed-pool ceiling.
    value.payload_ceiling=4*ORIGINAL_BYTES
    def configure(self,enabled,**options):
        calls.append((enabled,options));return {'enabled':enabled}
    monkeypatch.setattr(SourceExpertStore,'configure_adaptive_residency',configure)
    assert value.configure_adaptive_residency(False)=={'enabled':False}
    assert calls==[(False,{'staging_bytes':ORIGINAL_BYTES,'staging_within_cache':True})]
    assert value.payload_ceiling==75_000_000_000-ORIGINAL_BYTES


def test_adaptive_eviction_respects_shared_cap_without_changing_formats():
    value=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    value.policy=policy();a,b,new=(0,0),(0,1),(0,2)
    value.hot=frozenset({a,new});value.q1_root='/q1'
    value._adaptive_residency_enabled=True
    value.maximum_cache_bytes=2*ORIGINAL_BYTES+Q1_BYTES
    value._adaptive_payload_ceiling=value.maximum_cache_bytes-ORIGINAL_BYTES
    value.adaptive_residency_update=lambda **kwargs:None
    value.index=LFUIndex();value.index.observe(0,np.array([[0,1,1]],np.uint32))
    for key in (a,b):value.index.touch(key)
    value._cache={a:{'w':SimpleNamespace(nbytes=ORIGINAL_BYTES)},b:{'w':SimpleNamespace(nbytes=Q1_BYTES)}}
    value._cache_bytes=ORIGINAL_BYTES+Q1_BYTES
    value._leased={};value._pending_uploads={}
    value.pool_bytes={'original':ORIGINAL_BYTES,'q1':Q1_BYTES};value.stats={'evictions':0}
    original_map=value.hot
    value._evict_for(new,ORIGINAL_BYTES)
    assert value.hot is original_map and value.policy['hot_bytes']==60_000_000_000
    assert value.kind(a)==value.kind(new)=='original' and value.kind(b)=='q1'
    assert value._cache_bytes+ORIGINAL_BYTES<=value.maximum_cache_bytes-ORIGINAL_BYTES
    assert value.stats['evictions']==1
