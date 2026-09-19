"""Prompt lookup verifies several lanes in one target forward, exactly."""
import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator


def _north(seed=11):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=4, intermediate_size=8,
        prefix_dense_intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32, num_experts=4,
        num_experts_per_tok=2, first_k_dense_replace=1, sliding_window=6,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    ))
    model.eval()
    return model


PROMPTS = [
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],      # repetitive: long proposals
    [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # nothing to look up
    [3, 3, 3, 3, 3, 3, 3, 3, 3],
]


def _run(model, *, batched, tokens=24):
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=len(PROMPTS), prefill_step_size=5,
        prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 3, "adaptive": False,
                       "deferred_admission": False, "batched_verify": batched},
    )
    uids = generator.insert(PROMPTS, max_tokens=[tokens] * len(PROMPTS))
    out, finals = {uid: [] for uid in uids}, {}
    for _ in range(2000):
        _prompts, responses = generator.next()
        for response in responses:
            out[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == len(uids):
            break
    return [out[uid] for uid in uids], [finals[uid] for uid in uids], generator.scheduler_stats


def _greedy(model, prompt, tokens):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    produced = []
    for _ in range(tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        produced.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return produced


def test_batched_verify_is_exact_against_per_lane_and_plain_greedy():
    model = _north()
    batched, batched_finals, stats = _run(model, batched=True)
    per_lane, per_lane_finals, legacy = _run(model, batched=False)
    assert batched == per_lane
    for prompt, produced in zip(PROMPTS, batched):
        assert produced == _greedy(model, prompt, 24)
    # It really shared forwards, with ragged proposals and rejected tails.
    assert stats["pld_batched_rounds"] > 0 and stats["pld_batched_max_width"] == len(PROMPTS)
    assert stats["pld_batched_lanes"] > stats["pld_batched_rounds"]
    assert stats["pld_proposed"] > 0 and stats["pld_rollbacks"] > 0
    assert legacy["pld_batched_rounds"] == 0
    # Published caches are complete and free of any speculation epoch.
    for final in batched_finals:
        assert all(int(c.offset) == len(final.all_tokens) for c in final.prompt_cache)
        assert not any(getattr(c, "speculating", False) for c in final.prompt_cache)
        assert {type(c) for c in final.prompt_cache} == {KVCache, RotatingKVCache}
    assert max(r.execution_width for r in batched_finals) > 1


def test_batched_verify_continues_from_a_published_cache():
    """The cache a batched lane finishes with is an exact prefix for the next turn."""
    model = _north()
    _tokens, finals, _stats = _run(model, batched=True, tokens=10)
    final = finals[0]
    follow = [1, 2, 3]
    logits = model(mx.array([follow]), cache=final.prompt_cache)
    reference_cache = model.make_cache()
    reference = model(mx.array([final.all_tokens + follow]), cache=reference_cache)
    assert int(mx.argmax(logits[0, -1]).item()) == int(mx.argmax(reference[0, -1]).item())


def test_batched_verify_policy_is_validated_and_yields_to_rotating_replay():
    with pytest.raises(ValueError, match="batched_verify must be a boolean"):
        PromptLookupBatchGenerator.validate_policy({"batched_verify": "yes"})
    model = _north()
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=2, prefill_step_size=5,
        prompt_lookup={"num_draft": 4, "rotating_replay": True},
    )
    generator.insert(PROMPTS[:2], max_tokens=[8, 8])
    for _ in range(200):
        _prompts, responses = generator.next()
        if not generator.lanes:
            break
    assert generator.scheduler_stats["pld_batched_rounds"] == 0


def test_transaction_snapshots_survive_uncopyable_cow_state_and_skip_the_payload():
    """Caches restored from APCv2 carry lock-holding bookkeeping; rollback must
    neither deep-copy it nor copy (or even reference) the append-only payload."""
    import threading

    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows, _snapshot_cache

    def row(tokens):
        plain, ring = KVCache(), RotatingKVCache(max_size=4)
        for cache in (plain, ring):
            cache._cow_owner = threading.Lock()  # deepcopy would raise here
            for step in range(tokens):
                value = mx.full((1, 2, 1, 4), float(step))
                cache.update_and_fetch(value, value)
        return [plain, ring]

    rows = [row(9), row(6)]
    assert _snapshot_cache(rows[0][0]) == ("offset", 9, 0)
    before = [[mx.array(c._temporal_order(c.keys)) if type(c) is RotatingKVCache
               else mx.array(c.keys[..., : c.offset, :]) for c in r] for r in rows]
    transaction = SegmentedKVRows(rows).begin(lengths=[3, 2])
    block = mx.full((2, 2, 3, 4), 99.0)
    for view in transaction.caches:
        view.make_mask(3)
        view.update_and_fetch(block, block)
    transaction.commit(accepted_lengths=[1, 0])
    assert [int(c.offset) for c in rows[0]] == [10, 10] and [int(c.offset) for c in rows[1]] == [6, 6]
    # The fully rejected lane is bit-identical to its pre-round state.
    plain, ring = rows[1]
    assert mx.array_equal(plain.keys[..., :6, :], before[1][0]).item()
    assert mx.array_equal(ring._temporal_order(ring.keys), before[1][1]).item()
    assert float(rows[0][0].keys[0, 0, 9, 0]) == 99.0
