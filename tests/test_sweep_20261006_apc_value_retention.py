"""Opt-in value-aware APCv2 retention (2026-10-06 sweep).

The default order evicts by categorical rank, then recency: an interior
checkpoint without a hit goes first and a committed prompt boundary last, so
one-off prompt boundaries outlive the shared tool-schema and document
checkpoints later requests would resume from.  ``apc_retention_policy:
"value"`` evicts the entry with the least saved prefill per resident byte
first: decayed hit rate x (C(depth) - C(deepest stored ancestor)) / unique
bytes from the snapshot ledger.
"""

import random

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2, apc_retention_policy
from mlx2.runtime.models.cache import ArraysCache, KVCache

BYTES_PER_TOKEN = 2 * 8 * 4


def _kv(length):
    cache = KVCache()
    cache.step = 16
    value = mx.zeros((1, 1, length, 8))
    cache.update_and_fetch(value, value)
    mx.eval(cache.state)
    return cache


def _rec():
    cache = ArraysCache(1)
    cache[0] = mx.zeros((1, 64))
    mx.eval(cache[0])
    return cache


def _workload(seed, count=240):
    """Repeated tool schemas, long documents, a forked conversation, and
    one-off prompts, each with the interior cuts a turn-aware planner makes."""
    rng = random.Random(seed)
    cursor = [1000]

    def fresh(length):
        tokens = list(range(cursor[0], cursor[0] + length))
        cursor[0] += length
        return tokens

    schemas = [fresh(300) for _ in range(3)]
    documents = [fresh(1500) for _ in range(4)]
    fork = fresh(600)
    branches = [fresh(120) for _ in range(3)]
    requests = []
    for _ in range(count):
        draw = rng.random()
        if draw < 0.35:
            requests.append(((300,), rng.choice(schemas) + fresh(20)))
        elif draw < 0.6:
            document = documents[rng.choice([0, 0, 1, 1, 2, 3])]
            requests.append(((1500,), document + fresh(30)))
        elif draw < 0.8:
            requests.append(((600, 720), fork + rng.choice(branches) + fresh(25)))
        else:
            requests.append(((), fresh(rng.randrange(800, 2000))))
    return requests


def _replay(policy, budget_tokens, requests):
    clock = [0.0]
    apc = APCv2(
        max_size=10_000, max_interior_entries=10_000,
        max_bytes=budget_tokens * BYTES_PER_TOKEN, layout_name="replay",
        now_fn=lambda: clock[0], retention_policy=policy,
    )
    key = APCKey("m")
    avoided = 0
    for cuts, tokens in requests:
        clock[0] += 1.0
        hit = apc.lookup(key, tokens)
        avoided += hit.cached_tokens
        if hit.cache is not None:
            hit.cache.close()
        for cut in cuts:
            if cut > hit.cached_tokens:
                apc.store(
                    key, tokens[:cut], [_rec(), _kv(cut)],
                    retention_role="interior_checkpoint",
                )
        apc.store(
            key, tokens[:-1], [_rec(), _kv(len(tokens) - 1)],
            retention_role="committed_prompt_boundary",
        )
    return avoided, apc.apc_stats["retention"]


@pytest.mark.parametrize("budget_tokens", [3000, 6000, 12000])
def test_value_retention_avoids_more_prefill_at_equal_byte_budget(budget_tokens):
    requests = _workload(0)
    lru, lru_stats = _replay(None, budget_tokens, requests)
    value, value_stats = _replay("value", budget_tokens, requests)
    print(
        f"budget {budget_tokens} tokens: avoided prefill tokens LRU {lru}, "
        f"value {value}; {value_stats}"
    )
    assert lru_stats == {
        "policy": {"policy": "lru"}, "value_evictions": 0,
        "value_reordered_evictions": 0,
    }
    assert value > 2 * lru
    assert value_stats["value_evictions"] > 0
    assert value_stats["value_reordered_evictions"] > 0


def test_nested_chain_is_valued_by_its_increment_over_the_ancestor():
    clock = [0.0]
    apc = APCv2(max_size=8, layout_name="value-chain", now_fn=lambda: clock[0],
                retention_policy="value")
    key = APCKey("m")
    prompt = list(range(2000))
    for depth in (1000, 1100):
        assert apc.store(key, prompt[:depth], [_rec(), _kv(depth)]).stored
    shallow = apc._trie.get(key, prompt[:1000])
    deep = apc._trie.get(key, prompt[:1100])
    value = lambda depth, entry: apc._entry_retention_value_locked(  # noqa: E731
        key, prompt[:depth], entry, clock[0]
    )
    # Both carry the same prior; the deep entry saves only its 100 tokens
    # beyond the stored 1000-token ancestor.
    assert value(1000, shallow) > 5 * value(1100, deep)
    # Hits raise the decayed rate; a withdrawn hit takes its share back.
    hit = apc.lookup(key, prompt)
    assert hit.cached_tokens == 1100
    hit.cache.close()
    raised = value(1100, deep)
    apc.discard_lookup_credit(hit, "test")
    assert value(1100, deep) < raised


def test_policy_validation():
    assert apc_retention_policy(None) is None
    assert apc_retention_policy("lru") is None
    assert apc_retention_policy("value")["half_life_seconds"] == 600.0
    assert apc_retention_policy(
        {"policy": "value", "half_life_seconds": 30, "attention_tokens": 4096}
    ) == {"policy": "value", "half_life_seconds": 30.0, "attention_tokens": 4096}
    for bad in ("fifo", {"policy": "value", "half_life_seconds": 0},
                {"policy": "value", "attention_tokens": 1.5},
                {"policy": "value", "decay": 1}):
        with pytest.raises(ValueError):
            apc_retention_policy(bad)


def test_policy_round_trips_through_serving_settings(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    default = make_engine(model, vocab, mtp=False)
    try:
        assert "apc_retention_policy" not in default.snapshot["settings"]
    finally:
        default.close()
    engine = make_engine(
        model, vocab, mtp=False,
        execution_policy={"apc_retention_policy": {"policy": "value", "half_life_seconds": 60}},
    )
    try:
        run(engine, {"tokens": list(range(1, 40)), "max_tokens": 2, "temperature": 0})
        settings = dict(engine.snapshot["settings"])
        retention = engine.apc.apc_stats["retention"]
    finally:
        engine.close()
    expected = {"policy": "value", "half_life_seconds": 60.0, "attention_tokens": 8192}
    assert settings["apc_retention_policy"] == expected
    assert retention["policy"] == expected
