"""``PromptTrie.entries`` mirrors the trie exactly, so APCv2 can count
resident entries without re-walking every stored token path."""

import random

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import KVCache, PromptTrie


def _walk_all(trie):
    found = []

    def visit(model, node, path):
        if "__value__" in node:
            found.append((model, list(path), node["__value__"]))
        for tok, child in node.items():
            if tok != "__value__":
                visit(model, child, path + [tok])

    for model, root in trie._trie.items():
        visit(model, root, [])
    return found


def test_registry_tracks_add_pop_and_pop_prefixes():
    trie = PromptTrie()
    rng = random.Random(3)
    live = {}
    for step in range(400):
        tokens = [rng.randrange(4) for _ in range(rng.randrange(1, 7))]
        op = rng.random()
        if op < 0.5:
            trie.add("m", tokens, f"v{step}")
            live[tuple(tokens)] = f"v{step}"
        elif op < 0.8 and live:
            key = rng.choice(list(live))
            trie.pop("m", list(key))
            del live[key]
        elif live:
            # pop_prefixes walks a stored path; pick one so the walk exists.
            base = list(rng.choice(list(live)))
            popped = trie.pop_prefixes("m", base)
            for length, _value in popped:
                del live[tuple(base[:length])]
        registry = sorted((tuple(t), v) for _m, t, v in trie.entries())
        assert registry == sorted(live.items())
        assert registry == sorted((tuple(t), v) for _m, t, v in _walk_all(trie))


def _state(n, seed=0):
    cache = KVCache()
    values = mx.arange(seed, seed + n, dtype=mx.float32).reshape(1, 1, n, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def test_resident_count_matches_the_lru_walk_under_churn():
    key = APCKey("artifact-a", revision="r", adapter="a", tokenizer_fingerprint="t",
                 cache_layout_fingerprint="layout-a")
    apc = APCv2(max_size=4, max_bytes=1 << 24, layout_name="layout-a")
    rng = random.Random(11)
    for step in range(60):
        n = rng.randrange(8, 64)
        tokens = [rng.randrange(50) for _ in range(n)]
        apc.store(key, tokens, [_state(n, seed=step)])
        if rng.random() < 0.3:
            hit = apc.lookup(key, tokens + [99])
            if hit.hit and hit.cache is not None:
                hit.cache.close()
        with apc._apc_lock:
            walked = sum(
                bool(entry.prompt_cache)
                for _k, _t, entry in apc._entry_records_locked()
            )
            assert apc._resident_entry_count_locked() == walked
            assert apc._resident_entry_count_locked(interior=False) + \
                apc._resident_entry_count_locked(interior=True) == walked
    apc.close()
