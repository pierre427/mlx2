"""GQA-folded attention for Ripple: same keys per row as stock SDPA.

Not bit-identical (different reduction schedule): component checks use an
FP32 tolerance; route checks require exact greedy parity on the tiny FP32
model through DFlash2, prompt lookup and continuous batching.
"""
import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models.base import create_attention_mask, scaled_dot_product_attention
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.models.ricursive import gqa_folded_attention


def rand(*shape, seed):
    return mx.array(np.random.default_rng(seed).normal(size=shape).astype(np.float32))


def close(a, b):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("kind", ["sliding", "global"])
def test_folded_matches_stock_with_served_masks(rows, kind):
    window = 6
    cache = RotatingKVCache(window, keep=0) if kind == "sliding" else KVCache()
    for step in range(4):  # several appends so the rotating buffer wraps
        cache.update_and_fetch(rand(1, 2, 5, 8, seed=step), rand(1, 2, 5, 8, seed=50 + step))
    q = rand(1, 10, rows, 8, seed=99)
    mask = create_attention_mask(mx.zeros((1, rows, 8)), cache, window_size=window if kind == "sliding" else None)
    k, v = cache.update_and_fetch(rand(1, 2, rows, 8, seed=7), rand(1, 2, rows, 8, seed=8))
    expected = scaled_dot_product_attention(q, k, v, cache=None, scale=8 ** -0.5, mask=mask)
    close(gqa_folded_attention(q, k, v, scale=8 ** -0.5, mask=mask), expected)


def test_folded_batched_mask_and_per_head_fallback():
    q, k, v = rand(2, 4, 3, 8, seed=1), rand(2, 2, 9, 8, seed=2), rand(2, 2, 9, 8, seed=3)
    mask = mx.array(np.random.default_rng(4).random((2, 1, 3, 9)) > 0.3)
    mask = mx.logical_or(mask, mx.arange(9)[None, None, None, :] == 8)  # no fully masked row
    close(gqa_folded_attention(q, k, v, scale=0.3, mask=mask),
          mx.fast.scaled_dot_product_attention(q, k, v, scale=0.3, mask=mask))
    per_head = mx.ones((2, 4, 3, 9), dtype=mx.bool_)
    assert gqa_folded_attention(q, k, v, scale=0.3, mask=per_head) is None


def fold(model):
    model.set_gqa_folded_attention(16)
    return model


def test_model_toggle_counts_and_greedy_parity():
    from test_ripple_pld_cpu import greedy, tiny

    model = tiny()
    reference = greedy(model, [3, 7, 11, 5, 9, 2, 8], 14)
    fold(model)
    assert greedy(model, [3, 7, 11, 5, 9, 2, 8], 14) == reference
    assert model.gqa_fold_calls() > 0
    model.set_gqa_folded_attention(0)
    with pytest.raises(ValueError):
        model.set_gqa_folded_attention(100)


def test_folded_dflash2_pld_and_batching_keep_greedy_parity():
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.pld import PromptLookupBatchGenerator
    from test_external_dflash2_cpu import drain
    from test_ripple_dflash2_cpu import draft_args
    from test_ripple_pld_cpu import greedy, tiny

    model = tiny()
    prompts = [[3, 7, 11, 5, 9, 2, 8, 1, 4], [5, 1, 6]]
    expected = [greedy(model, p, 14) for p in prompts]
    fold(model)
    mx.random.seed(11)
    draft = DFlash2DraftModel(draft_args()).bind(model)
    mx.eval(draft.parameters())
    external = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test", num_draft=3, prefill_step_size=4)
    ids = external.insert(prompts, max_tokens=[14, 14], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, _ = drain(external)
    assert [got[i] for i in ids] == expected
    lookup = PromptLookupBatchGenerator(model, prefill_step_size=8,
                                        prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 4})
    seed = [3, 7, 11, 5]
    prompt = seed + greedy(model, seed, 10) + seed
    uid = lookup.insert([prompt], max_tokens=[16], caches=[model.make_cache()])[0]
    emitted = []
    for _ in range(300):
        _, responses = lookup.next()
        emitted.extend(r.token for r in responses if r.uid == uid)
        if any(r.finish_reason for r in responses):
            break
    model.set_gqa_folded_attention(0)
    reference = greedy(model, prompt, 16)
    assert emitted == reference
    fold(model)
    batch = BatchGenerator(model, max_tokens=14, prefill_step_size=3, completion_batch_size=4, prefill_batch_size=4)
    uids = batch.insert(prompts, max_tokens=[14, 14])
    out = {u: [] for u in uids}
    for _ in range(400):
        responses = batch.next()
        responses = responses[1] if isinstance(responses, tuple) else responses
        for r in responses:
            out[r.uid].append(r.token)
        if all(len(t) >= 14 for t in out.values()):
            break
    assert [out[u][:14] for u in uids] == expected


def test_folded_attention_engages_on_external_verify_transactions():
    """DFlash2 verify runs on segmented transaction caches (row_views)."""
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from test_external_dflash2_cpu import drain
    from test_ripple_dflash2_cpu import draft_args
    from test_ripple_pld_cpu import greedy, tiny

    model = tiny()
    prompt = [3, 7, 11, 5, 9, 2, 8, 1, 4]
    expected = greedy(model, prompt, 14)
    fold(model)
    mx.random.seed(11)
    draft = DFlash2DraftModel(draft_args()).bind(model)
    mx.eval(draft.parameters())
    rows_seen = []
    original = model.layers[0].self_attn._folded_rows

    def spy(q, cache, mask, scale):
        rows_seen.append(q.shape[2])
        return original(q, cache, mask, scale)

    model.layers[0].self_attn._folded_rows = spy
    generator = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test", num_draft=3, prefill_step_size=4)
    uid = generator.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
    got, _ = drain(generator)
    assert got[uid] == expected
    assert rows_seen and max(rows_seen) > 1
