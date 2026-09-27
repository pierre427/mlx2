import mlx.core as mx

from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.pld import PromptLookupBatchGenerator


class _PatternModel:
    def __init__(self, *, reject=False):
        self.reject = reject

    def __call__(self, tokens, *, cache):
        values = tokens.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(values, values)
        vocabulary = 8
        predicted = (
            mx.full(tokens.shape, 4, dtype=mx.int32)
            if self.reject
            else mx.where(tokens == 1, 2, 1)
        )
        return mx.where(
            mx.arange(vocabulary)[None, None, :] == predicted[..., None],
            20.0,
            -20.0,
        )


def _run(model, maximum):
    generator = PromptLookupBatchGenerator(
        model,
        prefill_step_size=16,
        prompt_lookup={"num_draft": 2, "ngram_min": 2, "ngram_max": 2},
    )
    uid = generator.insert(
        [[1, 2, 1, 2]],
        max_tokens=[maximum],
        caches=[[KVCache()]],
    )[0]
    prompts, responses = generator.next()
    assert not responses and prompts[0].uid == uid and prompts[0].end_of_prompt
    emitted = []
    final = None
    while final is None:
        _, responses = generator.next()
        emitted.extend(response.token for response in responses)
        final = next(
            (response for response in responses if response.finish_reason), None
        )
    return emitted, final


def test_live_prompt_lookup_accepts_exact_indexed_continuation():
    emitted, final = _run(_PatternModel(), 3)
    assert emitted == [1, 2, 1]
    assert final.speculative_receipt["execution"] == "prompt_lookup_verify"
    assert final.speculative_receipt["accepted"] == 2
    assert final.speculative_receipt["proposed"] == 2
    assert final.prompt_cache[0].offset == len(final.all_tokens) == 6


def test_recent_prompt_segments_include_exact_apcv2_prefix_tokens():
    model = _PatternModel()
    cache = [KVCache()]
    model(mx.array([[1, 2, 1, 2]], dtype=mx.uint32), cache=cache)
    generator = PromptLookupBatchGenerator(
        model, prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
            "recent_prompt_segments": 2, "prompt_segment_tokens": 3,
            "adaptive": False,
        },
    )
    uid = generator.insert(
        [[1, 2, 1, 2]], all_tokens=[[1, 2, 1, 2]],
        max_tokens=[3], caches=[cache],
    )[0]
    assert generator.lanes[uid].proposer.indexed_start == 2
    emitted, final = [], None
    while final is None:
        _prompts, responses = generator.next()
        emitted.extend(response.token for response in responses)
        final = next((response for response in responses if response.finish_reason), None)
    assert emitted == [1, 2, 1]
    assert final.speculative_receipt["recent_prompt_segments"] == 2
    assert final.speculative_receipt["prompt_segment_tokens"] == 3
    assert final.speculative_receipt["accepted"] > 0


def test_live_prompt_lookup_rejection_rewinds_to_committed_boundary():
    emitted, final = _run(_PatternModel(reject=True), 2)
    assert emitted == [4, 4]
    assert final.speculative_receipt["accepted"] == 0
    assert final.prompt_cache[0].offset == len(final.all_tokens) == 5


def test_deferred_admission_probes_on_plain_path_then_activates():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2,
            "ngram_min": 2,
            "ngram_max": 2,
            "deferred_admission": True,
            "admission_window": 1,
            "admission_gate": 1.0,
            "admission_reprobe_interval": 1,
            "adaptive_warmup": 100,
        },
    )
    uid = generator.insert(
        [[1, 2, 1, 2]], max_tokens=[4], caches=[[KVCache()]]
    )[0]
    generator.next()
    _, first = generator.next()
    assert first[0].uid == uid and not first[0].from_draft
    _, second = generator.next()
    receipts = [response.speculative_receipt for response in second]
    assert receipts[-1]["admission_activations"] == 1
    # The activation happens only after the ordinary probe commits; the next
    # closed boundary is the first one allowed to run a verify span.
    _, third = generator.next()
    assert any(response.from_draft for response in third)


def test_deferred_admission_probe_stride_keeps_plain_rounds_unprobed():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
            "deferred_admission": True, "admission_probe_stride": 3,
            "admission_window": 2, "admission_gate": 1.0,
            "adaptive_warmup": 100,
        },
    )
    generator.insert([[1, 2, 1, 2]], max_tokens=[8], caches=[[KVCache()]])
    generator.next()  # prefill
    _, first = generator.next()  # token 0 is probed
    assert first[-1].speculative_receipt["admission_probe_tokens"] == 1
    _, second = generator.next()  # token 1 is ordinary without lookup
    assert second[-1].speculative_receipt["admission_probe_tokens"] == 1
    assert second[-1].speculative_receipt["lookup_calls"] == 1
    _, third = generator.next()  # token 2 is ordinary without lookup
    assert third[-1].speculative_receipt["admission_probe_tokens"] == 1
    _, fourth = generator.next()  # token 3 is probed; window can activate
    assert fourth[-1].speculative_receipt["admission_probe_tokens"] == 2
    assert fourth[-1].speculative_receipt["admission_activations"] == 1


def test_cost_aware_latch_uses_shadow_copies_and_keeps_exact_output():
    policy = {
        "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
        "cost_aware_admission": True, "cost_shadow_span": 2,
        "cost_probe_stride": 2, "cost_shadow_window": 2,
        "cost_plain_rounds": 4, "cost_explore_rounds": 2,
    }
    generator = PromptLookupBatchGenerator(
        _PatternModel(), prefill_step_size=16, prompt_lookup=policy,
    )
    generator.insert([[1, 2] * 6], max_tokens=[20], caches=[[KVCache()]])
    emitted, final = [], None
    while final is None:
        _prompts, responses = generator.next()
        emitted.extend(response.token for response in responses)
        final = next((response for response in responses if response.finish_reason), None)
    assert emitted == [1, 2] * 10
    assert final.prompt_cache[0].offset == len(final.all_tokens)
    latch = final.speculative_receipt["cost_latch"]
    assert latch["shadow_trials"] >= 2 and latch["shadow_matches"] >= 4
    assert latch["activations"] >= 1
    assert latch["plain_ns"] > 0 and latch["plain_tokens"] >= 4
    assert latch["verify_ns"] > 0 and latch["verify_tokens"] >= 1
    assert generator.scheduler_stats["pld_parked_direct_rounds"] >= 4


def test_cost_aware_latch_leaves_novel_lane_parked_with_sparse_lookups():
    class NovelModel:
        def __call__(self, tokens, *, cache):
            values = tokens.astype(mx.float32)[:, None, :, None]
            cache[0].update_and_fetch(values, values)
            predicted = (tokens + 1) % 32
            return mx.where(
                mx.arange(32)[None, None, :] == predicted[..., None], 20.0, -20.0
            )

    generator = PromptLookupBatchGenerator(
        NovelModel(), prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
            "cost_aware_admission": True, "cost_shadow_span": 2,
            "cost_probe_stride": 4, "cost_plain_rounds": 4,
        },
    )
    generator.insert([list(range(1, 13))], max_tokens=[12], caches=[[KVCache()]])
    final = None
    while final is None:
        _prompts, responses = generator.next()
        final = next((response for response in responses if response.finish_reason), None)
    receipt = final.speculative_receipt
    assert receipt["execution"] == "ordinary_target"
    assert receipt["lookup_calls"] <= 3
    assert receipt["cost_latch"]["state"] == "parked"
    assert receipt["cost_latch"]["activations"] == 0
    assert generator.scheduler_stats["pld_recovery_checkpoint_captures"] == 1
    assert generator.scheduler_stats["pld_parked_direct_rounds"] == 12


def test_cost_latch_uses_ordinary_b1_full_attention_mask():
    class MaskSensitiveModel:
        def __call__(self, tokens, *, cache):
            mask = cache[0].make_mask(
                tokens.shape[1], return_array=False, window_size=None,
            )
            values = tokens.astype(mx.float32)[:, None, :, None]
            cache[0].update_and_fetch(values, values)
            chosen = 2 if isinstance(mask, mx.array) else 1
            return mx.where(
                mx.arange(8)[None, None, :] == chosen, 20.0, -20.0,
            ) * mx.ones((*tokens.shape, 1))

    cache = [KVCache()]
    generator = PromptLookupBatchGenerator(
        MaskSensitiveModel(), prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "cost_aware_admission": True,
            "cost_shadow_span": 2, "cost_reprobe_interval": 0,
        },
    )
    generator.insert([[3, 4, 5]], max_tokens=[2], caches=[cache],
                     prompt_lookup_configs=[{"memory_max_draft": 0}])
    assert cache[0].make_mask(3, return_array=False, window_size=None) == "causal"
    mask = cache[0].make_mask(1, return_array=False, window_size=None)
    reference = BatchKVCache([0]).make_mask(1, return_array=False, window_size=None)
    assert mask.shape == reference.shape == (1, 1, 1, 1)
    assert bool(mx.array_equal(mask, reference))
    tokens, final = [], None
    while final is None:
        _, responses = generator.next()
        tokens.extend(response.token for response in responses)
        final = next((response for response in responses if response.finish_reason), None)
    assert tokens == [2, 2]
    assert final.speculative_receipt["proposed"] == 0
    assert final.speculative_receipt["ordinary_b1_mask_calls"] >= 2


def test_cost_latch_does_not_activate_when_admission_cannot_fit_shadow_span():
    generator = PromptLookupBatchGenerator(
        _PatternModel(), prefill_step_size=16,
        prompt_lookup={
            "num_draft": 4, "ngram_min": 2, "ngram_max": 2,
            "cost_aware_admission": True, "cost_shadow_span": 4,
            "cost_reprobe_interval": 0,
        },
    )
    generator.insert([[1, 2] * 6], max_tokens=[12], caches=[[KVCache()]],
                     prompt_lookup_configs=[{"memory_max_draft": 0}])
    final = None
    while final is None:
        _prompts, responses = generator.next()
        final = next((response for response in responses if response.finish_reason), None)
    receipt = final.speculative_receipt
    assert receipt["proposed"] == receipt["lookup_calls"] == 0
    assert receipt["cost_latch"]["activations"] == 0
    assert receipt["cost_latch"]["memory_blocked_rounds"] == 12


def test_cost_latch_keeps_concurrent_lanes_at_physical_width_one():
    generator = PromptLookupBatchGenerator(
        _PatternModel(), completion_batch_size=2, prefill_step_size=16,
        prompt_lookup={
            "num_draft": 2, "ngram_min": 2, "ngram_max": 2,
            "cost_aware_admission": True, "cost_shadow_span": 2,
            "cost_probe_stride": 2, "cost_shadow_window": 2,
            "cost_plain_rounds": 4, "cost_explore_rounds": 2,
        },
    )
    uids = generator.insert(
        [[1, 2] * 6, [1, 2] * 6], max_tokens=[20, 20],
        caches=[[KVCache()], [KVCache()]],
    )
    tokens = {uid: [] for uid in uids}
    finals = {}
    for _ in range(100):
        _, responses = generator.next()
        for response in responses:
            tokens[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == 2:
            break
    assert len(finals) == 2
    assert all(tokens[uid] == [1, 2] * 10 for uid in uids)
    assert all(finals[uid].speculative_receipt["target_width"] == 1 for uid in uids)
    assert all(
        finals[uid].speculative_receipt["cost_width1_cohort_splits"] > 0
        for uid in uids
    )
    assert generator.scheduler_stats["pld_cost_width1_cohort_splits"] > 0
    assert generator.scheduler_stats["pld_batched_max_width"] <= 1
    assert generator.scheduler_stats["pld_proposed"] > 0


def test_cliff_aware_pld_extends_past_the_configured_plateau():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=32,
        prompt_lookup={
            "num_draft": 8,
            "ngram_min": 2,
            "ngram_max": 2,
            "cliff_aware_span": True,
        },
    )
    prompt = [1, 2] * 10
    generator.insert([prompt], max_tokens=[18], caches=[[KVCache()]])
    while True:
        _prompts, responses = generator.next()
        if responses:
            receipt = responses[-1].speculative_receipt
            assert max(map(int, receipt["verify_span_hist"])) == 16
            assert receipt["span_extend_cycles"] == 1
            break


def test_pld_accept_histogram_decomposes_the_accepted_aggregate():
    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=32,
        prompt_lookup={"num_draft": 8, "ngram_min": 2, "ngram_max": 2},
    )
    prompt = [1, 2] * 10
    generator.insert([prompt], max_tokens=[18], caches=[[KVCache()]])
    while True:
        _prompts, responses = generator.next()
        if responses:
            receipt = responses[-1].speculative_receipt
            hist = {int(k): int(v) for k, v in receipt["verify_accept_hist"].items()}
            # One entry per verify round, including plain rounds at accept 0.
            assert sum(hist.values()) == receipt["cycles"]
            assert sum(hist.values()) == sum(
                int(v) for v in receipt["verify_span_hist"].values()
            )
            # First moment is exactly the route's existing accepted aggregate.
            assert sum(k * v for k, v in hist.items()) == receipt["accepted"]
            break


def test_warm_apc_continuation_on_prompt_lookup_server_matches_cold(monkeypatch):
    # A warm APCv2 hit hands the lane a COW branch whose cache objects carry
    # lock-holding segment tokens; the prompt boundary snapshot deep-copied
    # them and killed the generation worker on the first continuation.
    from types import SimpleNamespace

    from mlx2 import memory, serving
    from mlx2.contracts import Capability
    from mlx2.runtime import os_memory
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs
    from mlx2.serving import ServingEngine
    from test_approximate_kv_serving import make_adapter, run

    monkeypatch.delenv("MLX_LM_EXTERNAL_ROUND_COW", raising=False)
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    mx.random.seed(11)
    model = Model(
        ModelArgs(
            hidden_size=16, head_dim=4, num_hidden_layers=4, intermediate_size=8,
            prefix_dense_intermediate_size=24, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=128, num_experts=4,
            num_experts_per_tok=2, first_k_dense_replace=1, sliding_window=6,
            layer_types=["full_attention"] + ["sliding_attention"] * 3,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    adapter = make_adapter(model, operations=None)
    adapter.descriptor = SimpleNamespace(
        capabilities=frozenset({Capability.PROMPT_LOOKUP})
    )
    first = list(range(1, 40)) + [50, 51]
    second = first + [60, 61]

    def engine():
        instance = ServingEngine(
            "tiny", adapter_factory=adapter, qualification_mode=True,
            mtp=False, prompt_lookup=True,
        )
        assert instance.ready.wait(30), instance.error
        return instance

    cold_engine = engine()
    try:
        cold, cold_receipt = run(cold_engine, second, max_tokens=4)
    finally:
        cold_engine.close()
    warm_engine = engine()
    try:
        run(warm_engine, first, max_tokens=4)
        warm, warm_receipt = run(warm_engine, second, max_tokens=4)
        alive, error = warm_engine.thread.is_alive(), warm_engine.error
    finally:
        warm_engine.close()
    assert alive and error is None
    assert cold_receipt["cached_tokens"] == 0 and warm_receipt["cached_tokens"] > 0
    assert warm == cold


def test_prompt_lookup_insert_refuses_inputs_it_cannot_apply():
    # insert swallowed arbitrary keyword arguments, so neural-concept (or
    # multimodal) prefill inputs were silently dropped while serving still
    # reported the bridge as applied.
    import pytest

    generator = PromptLookupBatchGenerator(
        _PatternModel(),
        prefill_step_size=16,
        prompt_lookup={"num_draft": 2, "ngram_min": 2, "ngram_max": 2},
    )
    for refused in (
        {"prefill_inputs": [{"deep_concept_memory": {"keys": 1}}]},
        {"apc_interior_positions": [(2,)]},
        {"state_boundaries": [("rolling", 2)]},
    ):
        with pytest.raises(ValueError, match="prompt lookup"):
            generator.insert([[1, 2, 1, 2]], max_tokens=[4], caches=[[KVCache()]], **refused)
    with pytest.raises(TypeError):
        generator.insert([[1, 2, 1, 2]], max_tokens=[4], caches=[[KVCache()]], mtp_states=[None])
    assert not generator.lanes
    # The serving seam's neutral values are still accepted.
    (uid,) = generator.insert(
        [[1, 2, 1, 2]], max_tokens=[4], caches=[[KVCache()]], lane_rngs=[None],
        prefill_inputs=[None], apc_interior_positions=[()],
    )
    assert uid in generator.lanes
