"""Admission eviction keeps the prefixes queued requests are about to hit.

GPU 2026-10-28 (Flash-Next default route, 4-lane warm burst at 4K): warm
requests are admitted one at a time, and each admission's pressure eviction
took the oldest unleased entries, which were the next queued requests'
prompt boundaries (1/4 warm hits, memory_pressure_evictions +61).  CPU only.
"""

import threading
from types import SimpleNamespace as NS

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.cache import KVCache
from mlx2.serving import queued_prefix_protector


def _store(apc, key, tokens):
    kv = KVCache()
    value = mx.zeros((1, 1, len(tokens), 64))
    kv.update_and_fetch(value, value)
    mx.eval(kv.state)
    assert apc.store(key, list(tokens), [kv]).stored


def _job(tokens, *, tenant="default", cancelled=False):
    event = threading.Event()
    if cancelled:
        event.set()
    return NS(admission_tokens=list(tokens), tenant_id=tenant, request={}, cancelled=event)


def test_protected_entries_stay_and_unprotected_ones_go_first():
    apc = APCv2(max_size=8, layout_name="protect-queued")
    key = apc.key("m")
    old, queued = tuple(range(8)), tuple(range(100, 108))
    _store(apc, key, old)
    _store(apc, key, queued)
    protect = lambda k, tokens: k == key and tuple(tokens) == queued  # noqa: E731
    assert apc.evict_oldest_unleased(protect=protect)
    assert apc.lookup(key, list(queued) + [1]).cached_tokens == len(queued)
    assert apc.lookup(key, list(old) + [1]).cached_tokens == 0
    # Only protected entries left: eviction makes no progress rather than
    # dropping what a queued request is about to look up.
    assert not apc.evict_oldest_unleased(protect=lambda k, tokens: True)


def test_protector_matches_prefixes_of_live_waiting_prompts_only():
    key_for = lambda job: ("k", job.tenant_id)  # noqa: E731
    jobs = [_job(range(10)), _job(range(50, 60), cancelled=True), _job([])]
    protect = queued_prefix_protector(jobs, key_for)
    assert protect(("k", "default"), tuple(range(9)))  # a boundary the prompt extends
    assert protect(("k", "default"), tuple(range(10)))
    assert not protect(("k", "default"), tuple(range(11)))  # longer than the prompt
    assert not protect(("k", "other"), tuple(range(9)))  # another namespace
    assert not protect(("k", "default"), tuple(range(50, 59)))  # cancelled job
    assert queued_prefix_protector([_job([]), _job(range(3), cancelled=True)], key_for) is None
