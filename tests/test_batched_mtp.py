# Adapted from unified tests/test_batched_self_mtp_qwen4.py at 1e2bc604, MIT.
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from mlx2.runtime.hybrid_speculative import (
    advance_self_mtp_prefill,
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.models.cache import make_prompt_cache
from mlx2.runtime.models.qwen4_exp import (
    BatchQSAKVCache,
    Model,
    ModelArgs,
    QSAKVCache,
    Qwen4ArraysCache,
)
from mlx2.runtime.sample_utils import LaneRNG


def _tiny_qwen4_model():
    text_config = dict(
        model_type="qwen4_exp_text",
        hidden_size=32,
        intermediate_size=0,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        mtp_num_hidden_layers=1,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
        },
    )
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    mx.eval(model.parameters())
    return model


def _tree_arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tree_arrays(item)


def _cache_offsets(caches):
    offsets = []
    for cache in caches:
        if not hasattr(cache, "offset"):
            continue
        offset = cache.offset
        if isinstance(offset, mx.array):
            offset = int(offset.item())
        offsets.append((type(cache).__name__, int(offset)))
    return offsets


def _prepare_lane(model, uid, prompt, *, share_qsa_indices=False):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=8,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(700 + uid),
        num_draft=2,
        sampling_temp=0.8,
        sampling_top_p=1.0,
        sampling_top_k=8,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_qsa_indices=share_qsa_indices,
    )[0]


def _forced_cycle(model, prompts, accepts, uids=None):
    # Each lane's first token is sampled from LaneRNG(700 + uid); a single-lane
    # oracle must reuse the batched lane's uid so the seeds — and thus the
    # sampled first token — match. Default keeps positional uids for the batch.
    uids = list(range(len(prompts))) if uids is None else uids
    lanes = [_prepare_lane(model, uid, prompt) for uid, prompt in zip(uids, prompts)]
    batch = attach_self_mtp_lanes(model, None, lanes)
    pending = iter(accepts)

    def force(logprobs, _draft_lps, _drafts, _temperature, *, rng=None):
        accepted = next(pending)
        return accepted, int(mx.argmax(logprobs[accepted]).item())

    with patch(
        "mlx2.runtime.hybrid_speculative._batched_residual_verify", side_effect=force
    ):
        proposal = propose_batched_self_mtp(model, batch)
    commit_batched_self_mtp(
        batch,
        proposal,
        emitted_counts=[len(row) for row in proposal.outputs],
        terminal=[False] * len(prompts),
    )
    batch, detached = detach_self_mtp_lanes(model, batch, list(range(len(prompts))))
    return detached


class TestBatchedSelfMTPQSA(unittest.TestCase):
    def test_sliced_fresh_prefill_matches_monolithic_and_ordinary_target(self):
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            mx.random.seed(71)
            model = _tiny_qwen4_model()
            prompt = mx.array([1, 7, 3, 9, 2, 8, 4, 6, 5], mx.uint32)

            ordinary_cache = make_prompt_cache(model)
            ordinary_logits = model(prompt[None], cache=ordinary_cache)[0, -1]
            ordinary_lp = ordinary_logits.astype(mx.float32)
            ordinary_lp = ordinary_lp - mx.logsumexp(ordinary_lp, keepdims=True)

            common = {
                "model": model,
                "uid": 11,
                "max_tokens": 8,
                "lane_rng": None,
                "num_draft": 2,
                "sampling_temp": 0.0,
                "sampling_top_p": 1.0,
                "sampling_top_k": 0,
                "sampling_min_p": 0.0,
                "accept_rule": "residual",
                "logits_processors": [],
                "prefill_step_size": 4,
                "share_qsa_indices": False,
            }
            monolithic_boundary = {}
            monolithic, monolithic_first = prepare_self_mtp_lane(
                prompt,
                prompt_cache=None,
                mtp_state=None,
                prompt_boundary_out=monolithic_boundary,
                **common,
            )

            remaining = prompt
            prompt_cache = None
            mtp_state = None
            while int(remaining.size) > 5:
                remaining, prompt_cache, mtp_state, processed = (
                    advance_self_mtp_prefill(
                        remaining,
                        model,
                        prompt_cache=prompt_cache,
                        mtp_state=mtp_state,
                        max_tokens=4,
                    )
                )
                self.assertEqual(processed, 4)
            sliced_boundary = {}
            sliced, sliced_first = prepare_self_mtp_lane(
                remaining,
                prompt_cache=prompt_cache,
                mtp_state=mtp_state,
                prompt_boundary_out=sliced_boundary,
                **common,
            )

            mx.eval(
                ordinary_lp,
                monolithic_first.logprobs,
                sliced_first.logprobs,
                monolithic.lane.seed_h,
                sliced.lane.seed_h,
            )
            self.assertEqual(monolithic_first.token, sliced_first.token)
            self.assertEqual(
                sliced_first.token, int(mx.argmax(ordinary_lp).item())
            )
            np.testing.assert_allclose(
                np.asarray(monolithic_first.logprobs),
                np.asarray(ordinary_lp),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(sliced_first.logprobs),
                np.asarray(monolithic_first.logprobs),
                rtol=1e-5,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(sliced.lane.seed_h),
                np.asarray(monolithic.lane.seed_h),
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertEqual(
                _cache_offsets(sliced.caches.target),
                _cache_offsets(monolithic.caches.target),
            )
            self.assertEqual(
                _cache_offsets(sliced.caches.draft),
                _cache_offsets(monolithic.caches.draft),
            )
            self.assertEqual(
                sliced_boundary["covered_tokens"],
                monolithic_boundary["covered_tokens"],
            )
        finally:
            mx.set_default_device(previous_device)

    def test_batched_proposal_reuses_per_lane_topk_and_disarms_cycle(self):
        mx.random.seed(43)
        model = _tiny_qwen4_model()
        lanes = [
            _prepare_lane(model, uid, prompt, share_qsa_indices=True)
            for uid, prompt in enumerate(([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]))
        ]
        batch = attach_self_mtp_lanes(model, None, lanes)
        qsa_caches = [
            cache for cache in batch.caches.draft if isinstance(cache, BatchQSAKVCache)
        ]
        self.assertTrue(qsa_caches)

        mtp_step = model.mtp_step
        calls = 0

        def checked_mtp_step(hidden, tokens, cache):
            nonlocal calls
            if calls == 1:
                self.assertTrue(
                    all(item._mtp_shared_topk is not None for item in qsa_caches)
                )
                self.assertTrue(
                    all(item._mtp_shared_topk.shape[0] == 2 for item in qsa_caches)
                )
            result = mtp_step(hidden, tokens, cache)
            calls += 1
            return result

        model.mtp_step = checked_mtp_step
        propose_batched_self_mtp(model, batch)

        self.assertEqual(calls, 2)
        self.assertTrue(all(cache._mtp_shared_topk is None for cache in qsa_caches))
        self.assertTrue(all(not cache._mtp_share_topk for cache in qsa_caches))

    def test_shared_topk_uses_each_rows_last_valid_query(self):
        cache = BatchQSAKVCache([0, 0, 0])
        cache.prepare(right_padding=[0, 2, 1])
        selected = mx.array(
            [
                [[0, 1], [2, 3], [4, 5], [6, 7]],
                [[10, 11], [12, 13], [98, 98], [99, 99]],
                [[20, 21], [22, 23], [24, 25], [99, 99]],
            ],
            dtype=mx.uint32,
        )

        got = cache.last_valid_query(selected)
        mx.eval(got)

        np.testing.assert_array_equal(
            np.asarray(got),
            np.asarray([[6, 7], [12, 13], [24, 25]], dtype=np.uint32),
        )

    def test_shared_topk_refuses_a_fully_padded_row(self):
        cache = BatchQSAKVCache([0, 0])
        cache.prepare(right_padding=[0, 3])
        selected = mx.zeros((2, 3, 2), dtype=mx.uint32)

        with self.assertRaisesRegex(ValueError, "one valid query"):
            cache.last_valid_query(selected)

    def test_ragged_head_trim_releases_cycle_and_aligns_qsa_ledger(self):
        cache = BatchQSAKVCache([0, 0])
        cache.keys = mx.arange(2 * 1 * 6 * 2).reshape(2, 1, 6, 2)
        cache.values = cache.keys + 100
        cache._idx = 6
        cache.offset = mx.array([6, 6])
        cache.left_padding = mx.array([0, 0])
        # A two-step shared-QSA cycle appends the first raw key only.
        cache.index_keys = mx.arange(2 * 5 * 3).reshape(2, 5, 3)
        cache._mtp_share_topk = True
        cache._mtp_shared_topk = mx.array([[0, 1], [1, 2]], dtype=mx.uint32)

        drops = cache.trim_ragged([2, 1])
        mx.eval(cache.state)

        self.assertEqual(drops, [2, 1])
        self.assertEqual(cache._idx, 5)
        self.assertEqual(cache.offset.tolist(), [4, 5])
        self.assertEqual(cache.left_padding.tolist(), [1, 0])
        self.assertEqual(cache.index_keys.shape[1], cache._idx)
        self.assertIsNone(cache._mtp_shared_topk)
        self.assertFalse(cache._mtp_share_topk)
        first, second = cache.extract(0), cache.extract(1)
        self.assertEqual(first.offset, 4)
        self.assertEqual(first.index_keys.shape[1], first.offset)
        self.assertEqual(second.offset, 5)
        self.assertEqual(second.index_keys.shape[1], second.offset)


class TestQwen4ForcedAcceptanceCacheEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(41)
        cls.model = _tiny_qwen4_model()

    def assert_cache_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertIs(type(got), type(want))
            got_arrays = list(_tree_arrays(got.state))
            want_arrays = list(_tree_arrays(want.state))
            self.assertEqual(len(got_arrays), len(want_arrays))
            for left, right in zip(got_arrays, want_arrays):
                # numpy has no bfloat16 buffer format; the production caches are
                # bf16, so upcast to fp32 for the comparison (bit-identical bf16
                # values upcast identically — this does not loosen the check).
                np.testing.assert_allclose(
                    np.asarray(left.astype(mx.float32)),
                    np.asarray(right.astype(mx.float32)),
                    rtol=2e-4,
                    atol=2e-5,
                )
            if isinstance(got, QSAKVCache):
                self.assertEqual(got.offset, want.offset)
                np.testing.assert_allclose(
                    np.asarray(got.index_keys.astype(mx.float32)),
                    np.asarray(want.index_keys.astype(mx.float32)),
                    rtol=2e-4,
                    atol=2e-5,
                )

    def test_all_forced_acceptance_vectors_match_independent_lanes(self):
        prompts = ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])
        for accepts in product(range(3), repeat=2):
            with self.subTest(accepts=accepts):
                batched = _forced_cycle(self.model, prompts, accepts)
                singles = [
                    _forced_cycle(self.model, [prompt], [accepted], uids=[i])[0]
                    for i, (prompt, accepted) in enumerate(zip(prompts, accepts))
                ]
                for got, want in zip(batched, singles):
                    self.assert_cache_equal(got.caches.target, want.caches.target)
                    self.assert_cache_equal(got.caches.draft, want.caches.draft)
                    qwen4 = [
                        cache
                        for cache in got.caches.target
                        if isinstance(cache, Qwen4ArraysCache)
                    ]
                    self.assertEqual(len(qwen4), 1)
                    self.assertEqual(len(qwen4[0].cache), 4)
                    self.assertTrue(all(slot is not None for slot in qwen4[0].cache))
                    qsa = [
                        cache
                        for cache in got.caches.target + got.caches.draft
                        if isinstance(cache, QSAKVCache)
                    ]
                    self.assertTrue(qsa)
                    self.assertTrue(all(cache.index_keys is not None for cache in qsa))


def test_mtp_warm_prefix_without_draft_state_is_refused_at_insert():
    """An ordinary-route checkpoint has no paired draft state.  Inserting it
    into an unmarked self-MTP lane used to raise only inside ``next`` (killing
    the whole batch); the request boundary must refuse it instead."""
    import pytest

    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.sample_utils import LaneRNG

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _tiny_qwen4_model()
        cache = make_prompt_cache(model)
        prefix = [1, 2, 3, 4, 5]
        model(mx.array([prefix], mx.uint32), cache=cache)
        mx.eval([c.state for c in cache])
        gen = BatchGenerator(
            model, self_mtp={"num_draft": 2, "persistent": True}, prefill_step_size=4
        )
        try:
            with pytest.raises(ValueError, match="target-only ordinary fallback"):
                gen.insert(
                    [[6, 7, 8]], max_tokens=[8], caches=[cache], all_tokens=[prefix],
                    mtp_states=[None], lane_rngs=[LaneRNG(3)],
                    self_mtp_configs=[{"sampling_temp": 0.0}],
                )
            # A cold prompt (no warm prefix) still inserts without draft state.
            uids = gen.insert(
                [prefix + [6, 7, 8]], max_tokens=[8], mtp_states=[None],
                lane_rngs=[LaneRNG(3)], self_mtp_configs=[{"sampling_temp": 0.0}],
            )
            assert len(uids) == 1
        finally:
            gen.close()
    finally:
        mx.set_default_device(previous_device)


def test_depth_zero_fast_path_matches_ordinary_and_reenters_exactly():
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(922)
        model = _tiny_qwen4_model()
        prompt = [1, 7, 3, 9, 2, 8, 4]
        ordinary = BatchGenerator(
            model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=32,
        )
        mtp = BatchGenerator(
            model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 1,
            },
            adaptive_mtp_depth={"current_depth": 0},
        )
        ordinary.insert([prompt], max_tokens=[10])
        mtp.insert(
            [prompt],
            max_tokens=[10],
            lane_rngs=[LaneRNG(17)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        outputs = [[], []]
        draft_calls = 0
        real_mtp_step = model.mtp_step

        def counted_mtp_step(*args, **kwargs):
            nonlocal draft_calls
            draft_calls += 1
            return real_mtp_step(*args, **kwargs)

        counting = switched = False
        for _ in range(100):
            for index, batch in enumerate((ordinary, mtp)):
                _, responses = batch.next()
                outputs[index].extend(response.token for response in responses)
            if not counting and len(outputs[1]) >= 1:
                model.mtp_step = counted_mtp_step
                counting = True
            if not switched and len(outputs[1]) >= 4:
                # Preparation has finished. K=0 itself must not execute the
                # MTP head; bounded counters also prove the direct path engaged.
                assert draft_calls == 0
                assert mtp.scheduler_stats["self_mtp_zero_fast_rounds"] >= 1
                assert (
                    mtp.scheduler_stats["self_mtp_zero_draft_forwards_skipped"]
                    == mtp.scheduler_stats["self_mtp_zero_fast_rounds"]
                )
                mtp._generation_batch.adaptive_depth_policy.current_depth = 1
                switched = True
            if min(map(len, outputs)) >= 10:
                break
        assert switched and draft_calls > 0
        assert outputs[1] == outputs[0]
        ordinary.close()
        mtp.close()
    finally:
        mx.set_default_device(previous_device)


def test_greedy_self_mtp_applies_fly_and_records_relaxed_accept():
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(933)
        model = _tiny_qwen4_model()
        (detached, _) = prepare_self_mtp_lane(
            mx.array([1, 7, 3, 9], mx.uint32),
            model,
            uid=4,
            max_tokens=8,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=LaneRNG(704),
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="residual",
            logits_processors=[],
            prefill_step_size=8,
            share_qsa_indices=False,
            fly_verification={
                "enabled": True,
                "entropy_threshold": 0.5,
                "window": 1,
                "min_prob": 0.01,
            },
        )
        batch = attach_self_mtp_lanes(model, None, [detached])
        with patch(
            "mlx2.runtime.speculative_sampling.apply_fly_relaxation",
            return_value=(2, 1),
        ) as relaxed:
            proposal = propose_batched_self_mtp(model, batch)
        assert relaxed.call_count == 1
        assert proposal.accepted_lengths == (2,)
        assert proposal.relaxed_accepts == (1,)
        assert batch.lanes[0].relaxed_accepts == 0
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(proposal.outputs[0])],
            terminal=[False],
        )
        assert batch.lanes[0].relaxed_accepts == 1
    finally:
        mx.set_default_device(previous_device)


def test_self_mtp_draft_processor_probe_is_side_effect_free():
    from mlx2.runtime.hybrid_speculative import _lane_mtp_draft_logprobs

    class StatefulProcessor:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tokens, logits):
            self.calls += 1
            return logits

    processor=StatefulProcessor()
    lane=SimpleNamespace(
        logits_processors=[processor],token_prefix=mx.array([1,2],mx.uint32),
        cur=3,logprob_transform=None,sampling_temp=0.0,
    )
    result=_lane_mtp_draft_logprobs(lane,mx.arange(64,dtype=mx.float32),[])
    mx.eval(result)
    assert processor.calls==0


def test_self_mtp_min_tokens_and_grammar_complete_on_cpu():
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.serving import minimum_tokens_processor
    from mlx2.structured_output import StructuredOutputProcessor

    class Tokenizer:
        eos_token_ids = [0]
        vocab_size = 64

        def decode(self, tokens, **_kwargs):
            return "".join(chr(65 + int(token)) for token in tokens if int(token) != 0)

    class Bs:
        pattern = None

        @staticmethod
        def canonicalize(value):
            return value

        @staticmethod
        def fullmatch(value, *, partial=False, timeout=None):
            del timeout
            valid = all(char == "B" for char in value)
            return object() if valid and (partial or len(value) >= 2) else None

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        mx.random.seed(923)
        model = _tiny_qwen4_model()
        prompt = [1, 7, 3]
        processors = [
            minimum_tokens_processor(mx, [0], len(prompt), 2),
            StructuredOutputProcessor(Tokenizer(), len(prompt), Bs()),
        ]
        generator = BatchGenerator(
            model,
            completion_batch_size=1,
            prefill_batch_size=1,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 2,
            },
        )
        generator.insert(
            [prompt],
            max_tokens=[6],
            lane_rngs=[LaneRNG(11)],
            logits_processors=[processors],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        emitted = []
        terminal = None
        for _ in range(30):
            _, responses = generator.next()
            emitted.extend(response.token for response in responses)
            terminal = next(
                (response for response in responses if response.finish_reason),
                terminal,
            )
            if terminal is not None:
                break
        assert terminal is not None
        assert emitted[:2] == [1, 1]
        assert emitted[-1] == 0
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous_device)


def test_depth_zero_fast_path_honours_true_batched_off(monkeypatch):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime import segmented_self_mtp as segmented

    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP","0")
    previous_device=mx.default_device();mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(922);model=_tiny_qwen4_model()
        generator=BatchGenerator(
            model,completion_batch_size=2,prefill_batch_size=2,prefill_step_size=32,
            self_mtp={"num_draft":2,"persistent":True,
                      "segment_aware_live_tip":True,"segment_aware_cohort_size":2},
            adaptive_mtp_depth={"current_depth":0},
        )
        generator.insert(
            [[1,7,3,9,2],[4,5,6,2,8]],max_tokens=[6,6],
            lane_rngs=[LaneRNG(1),LaneRNG(2)],
            self_mtp_configs=[{"sampling_temp":0.0}]*2,
        )
        before=dict(segmented.segmented_self_mtp_stats())
        final=[]
        for _ in range(60):
            _,responses=generator.next();final.extend(responses)
            if len(final)>=12:
                break
        after=segmented.segmented_self_mtp_stats()
        assert after["true_batched_requests"]-before.get("true_batched_requests",0)==0
        terminal=[response for response in final if response.finish_reason]
        assert len(terminal)==2
        assert all(response.mtp_receipt["observed_compute_widths"]==[1]
                   for response in terminal)
        generator.close()
    finally:
        mx.set_default_device(previous_device)
