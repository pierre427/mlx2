"""A plain landing at ``len - 1`` hides a draft checkpoint only for self-MTP.

``lookup(require_sidecar=True)`` let a sidecar-less hit at ``len - 1`` beat
the deepest sidecar ancestor because the self-MTP route decodes that one
token plainly.  The external-draft route cannot use a target-only hit at
any depth (it re-prefills the whole transcript), so the same exception
turned a reusable draft checkpoint into a cold miss there.
"""

import mlx.core as mx

from mlx2.runtime import apc_v2
from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import ArraysCache, KVCache


def _kv(length, seed=0):
    cache = KVCache()
    value = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def _rec(length):
    cache = ArraysCache(1)
    cache[0] = mx.ones((1, 4), dtype=mx.float32) * length
    cache.lengths = mx.array([length], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [length])
    mx.eval(cache.state)
    return cache


def _sidecar(covered):
    return MTPAPCSidecar(
        ([_kv(covered - 1, seed=7)], mx.ones((1, 1, 4))), covered_tokens=covered
    )


def _shadowed(layout):
    apc = APCv2(max_size=8, layout_name=layout)
    key = APCKey("m")
    prompt = list(range(400))
    assert apc.store(
        key, prompt[:100], [_rec(100), _kv(100)], sidecar=_sidecar(100)
    ).stored
    assert apc.store(key, prompt[:399], [_rec(399), _kv(399)]).stored
    return apc, key, prompt


def test_route_without_plain_fallback_gets_the_sidecar_ancestor():
    apc, key, prompt = _shadowed("external-plain-v1")
    hit = apc.lookup(
        key, prompt, require_sidecar=True, allow_target_only_plain=False
    )
    assert hit.sidecar is not None and hit.cached_tokens == 100
    hit.cache.close()


def test_self_mtp_plain_landing_still_wins():
    apc, key, prompt = _shadowed("external-plain-v2")
    hit = apc.lookup(key, prompt, require_sidecar=True)
    assert hit.sidecar is None and hit.cached_tokens == 399
    hit.cache.close()


def _lookup_kwargs(monkeypatch, external):
    from route_harness import (
        make_engine, make_external_engine, patch_host, run, tiny_qwen38_mtp,
    )

    seen = []
    original = apc_v2.APCv2.lookup

    def lookup(self, key, tokens, **kwargs):
        seen.append(kwargs)
        return original(self, key, tokens, **kwargs)

    monkeypatch.setattr(apc_v2.APCv2, "lookup", lookup)
    patch_host(monkeypatch)
    if external:
        from test_sweep_20261006_apc_speculative_reuse import _kv_target_external_pair

        model, draft, vocab = _kv_target_external_pair()
        engine = make_external_engine(model, draft, vocab)
    else:
        model, vocab = tiny_qwen38_mtp()
        engine = make_engine(model, vocab, mtp=True)
    try:
        result = run(engine, {"tokens": [3, 4, 5, 6], "max_tokens": 2, "temperature": 0})
    finally:
        engine.close()
    assert "error" not in result, result
    assert seen
    return seen


def test_serving_routes_declare_their_plain_fallback(monkeypatch):
    for kwargs in _lookup_kwargs(monkeypatch, external=True):
        assert kwargs["require_sidecar"] is True
        assert kwargs["allow_target_only_plain"] is False


def test_self_mtp_route_keeps_its_plain_fallback(monkeypatch):
    for kwargs in _lookup_kwargs(monkeypatch, external=False):
        assert kwargs["require_sidecar"] is True
        assert kwargs["allow_target_only_plain"] is True
