"""Warm-hit admission estimates scale by allocated capacity, not cached tokens."""

import json
from pathlib import Path

import mlx.core as mx

from mlx2.runtime.apc_v2 import APCv2, _walk_cache_entries
from mlx2.runtime.models.xing4_0 import Model, ModelArgs
from mlx2.serving import warm_cache_copy_gib

FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"


def test_short_shared_prefix_of_a_long_entry_is_not_projected_huge():
    # GPU 2026-09-18: a prompt sharing only its chat-template head with a
    # 2k-token checkpoint was charged ~(entry bytes x 8200 / 3) and every
    # request 429'd on an idle server.
    model = Model(ModelArgs.from_dict(json.loads((FIXTURE / "config.json").read_text())))
    model.load_weights(list(model.sanitize(mx.load(str(FIXTURE / "weights.safetensors"))).items()), strict=True)
    apc = APCv2(layout_name=model.apc_v2_layout)
    key = apc.key("t", revision="r")
    vocab = model.args.vocab_size  # MLX gathers are unchecked: stay in range
    stored = [5, 6, 7] + [20 + (i * 37) % (vocab - 20) for i in range(500)]
    cache = model.make_cache()
    for start in range(0, len(stored), 256):
        model(mx.array([stored[start : start + 256]]), cache=cache)
    mx.eval([c.state for c in cache])
    apc.store(key, stored, cache)
    try:
        for prompt in ([5, 6, 7, 99, 98, 97], stored[:300] + [1, 2, 4]):
            hit = apc.lookup(key, prompt)
            assert hit.cache is not None and hit.cached_tokens > 0
            size = sum(int(getattr(row, "nbytes", 0)) for row in _walk_cache_entries(hit.cache))
            estimate = warm_cache_copy_gib(hit, context_tokens=8200, prefill_step=2048, mtp=False)
            # Never more than the entry's bytes scaled to 8200 of its >=503 slots.
            assert estimate * (1 << 30) <= size * 8200 / 503 * 1.01, (hit.cached_tokens, estimate)
            hit.cache.close()
    finally:
        apc.clear()
