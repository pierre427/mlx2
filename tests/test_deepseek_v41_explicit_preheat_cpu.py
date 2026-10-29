"""Explicit adaptive warmup stays bounded and never invents unseen routes."""
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import threading

import pytest

from mlx2.adapters.deepseek_v41_source_lfu import (
    LFUIndex, LFUSourceExpertStore, ORIGINAL_BYTES, Q1_BYTES,
)


def store(*, explicit=True, available=1<<40, budget=20*ORIGINAL_BYTES):
    value=LFUSourceExpertStore.__new__(LFUSourceExpertStore)
    value.policy={} if explicit is None else {'preheat':explicit}
    value.preheat_receipt={'requested':True,'performed':False}
    value._adaptive_residency_enabled=True
    value.index=LFUIndex();value.index.counts=Counter({(0,0):2,(1,0):3,(2,0):0})
    value.index.last={(0,0):2,(1,0):1}
    value.files={(0,0):{},(1,0):{},(2,0):{}}
    value.hot=frozenset({(0,0)});value.q1_root='/q1'
    value._cache={};value._cache_bytes=0
    value._leased={};value._pending_uploads={};value._pending_host={}
    value._lock=threading.RLock();value._active=False;value._ready=lambda:None
    value._adaptive_staging=ORIGINAL_BYTES;value._adaptive_working_set=6
    value._adaptive_headroom=value._adaptive_protected_bytes=0
    value.minimum_cache_bytes=ORIGINAL_BYTES;value.cache_budget_bytes=budget
    value._adaptive_available=available if callable(available) else lambda:available
    value.stats={'cache_misses':0};value.counters=lambda:dict(value.stats)
    value.calls=[];value.windows=[]
    @contextmanager
    def window():
        assert not value._active
        value._active=True;value.windows.append('open')
        try:yield
        finally:value._active=False;value.windows.append('closed')
    value.window=window
    def acquire(prefix,expert):
        assert value._active
        key=(int(prefix[4:]),expert);value.calls.append(key)
        value._cache[key]=object()
        value._cache_bytes+=ORIGINAL_BYTES if key in value.hot else Q1_BYTES
        value.stats['cache_misses']+=1
    value.acquire=acquire
    return value


def test_explicit_warmup_uses_observed_frequency_and_fenced_windows():
    value=store();counts=value.index.counts.copy()
    result=value.preheat()
    assert value.calls==[(1,0),(0,0)]
    assert value.windows==['open','closed','open','closed']
    assert result['performed'] and result['objects']==result['loaded_objects']==2
    assert result['original_objects']==result['q1_objects']==1
    assert result['unseen_fill_objects']==result['deferred_objects']==0
    assert result['resource_lease'] is False and result['precision_unchanged']
    assert result['counter_delta']['cache_misses']==2
    assert value.index.counts==counts


@pytest.mark.parametrize('explicit',[None,False])
def test_adaptive_atlas_does_not_implicitly_fill_cache(explicit):
    value=store(explicit=explicit)
    result=value.preheat()
    assert not result['performed'] and not value.calls


@pytest.mark.parametrize('available',[None,-1,True])
def test_unknown_availability_defers_without_loading(available):
    value=store(available=available)
    result=value.preheat()
    assert result['reason']=='live_availability_unknown'
    assert result['deferred_objects']==2 and not value.calls


def test_live_availability_is_remeasured_between_uploads():
    readings=iter([1<<40,0]);value=store(available=lambda:next(readings))
    result=value.preheat()
    assert value.calls==[(1,0)] and result['deferred_objects']==1
    assert result['reason']=='live_availability_or_working_headroom'


def test_operator_maximum_retains_staging_and_next_working_set():
    value=store(budget=7*ORIGINAL_BYTES+Q1_BYTES)
    result=value.preheat()
    assert value.calls==[(1,0)]
    assert result['reason']=='configured_maximum_or_working_headroom'
    assert result['deferred_objects']==1


def test_existing_warm_objects_need_no_reallocation():
    value=store(available=0)
    value._cache[(0,0)]=value._cache[(1,0)]=object()
    value._cache_bytes=ORIGINAL_BYTES+Q1_BYTES
    result=value.preheat()
    assert result['performed'] and result['objects']==2 and result['loaded_objects']==0
    assert not value.calls and result['deferred_objects']==0


def test_empty_atlas_does_not_fabricate_route_experts():
    value=store();value.index.counts.clear()
    result=value.preheat()
    assert not result['performed'] and result['reason']=='no_observed_atlas_experts'
    assert result['requested_objects']==0 and not value.calls


def test_active_owner_is_refused_before_warmup():
    value=store();value._active=True
    with pytest.raises(ValueError,match='idle expert provider'):value.preheat()
    assert not value.calls


def test_read_error_propagates_and_completion_window_closes():
    value=store()
    def fail(*args):raise OSError('read failed')
    value.acquire=fail
    with pytest.raises(OSError,match='read failed'):value.preheat()
    assert value.windows==['open','closed'] and not value._active
    assert not value.preheat_receipt['performed']


def test_partial_read_failure_retains_partial_inventory_in_receipt():
    value=store();original=value.acquire
    def fail_second(prefix,expert):
        if value.calls:raise OSError('second read failed')
        original(prefix,expert)
    value.acquire=fail_second
    with pytest.raises(OSError,match='second read failed'):value.preheat()
    result=value.preheat_receipt
    assert result['performed'] is False and result['reason']=='preheat_failed'
    assert result['objects']==result['loaded_objects']==result['deferred_objects']==1
    assert result['counter_delta']['cache_misses']==1
    assert result['error']['type']=='OSError'
