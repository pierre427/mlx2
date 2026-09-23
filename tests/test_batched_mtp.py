# Adapted from unified tests/test_batched_self_mtp_qwen4.py at 1e2bc604, MIT.
import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

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


def _prepare_lane(model, uid, prompt, *, share_qsa_indices=False, max_tokens=8):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=max_tokens,
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


def _forced_cycles(model, prompts, accepts_per_cycle, uids):
    """Run several forced-acceptance self-MTP cycles; return tokens and batch."""
    lanes = [
        _prepare_lane(model, uid, prompt, max_tokens=1000)
        for uid, prompt in zip(uids, prompts)
    ]
    batch = attach_self_mtp_lanes(model, None, lanes)
    tokens = [[] for _ in prompts]
    for accepts in accepts_per_cycle:
        pending = iter(accepts)

        def force(logprobs, _draft_lps, _drafts, _temperature, *, rng=None):
            accepted = next(pending)
            return accepted, int(mx.argmax(logprobs[accepted]).item())

        with patch(
            "mlx2.runtime.hybrid_speculative._batched_residual_verify",
            side_effect=force,
        ):
            proposal = propose_batched_self_mtp(model, batch)
        commit_batched_self_mtp(
            batch,
            proposal,
            emitted_counts=[len(row) for row in proposal.outputs],
            terminal=[False] * len(prompts),
        )
        for row, output in enumerate(proposal.outputs):
            tokens[row].extend(item.token for item in output)
    return tokens, batch


def test_physical_ragged_cycles_keep_padding_bounded_and_tokens_exact():
    """X3-4: rows that keep rejecting different amounts on the physical route.

    The cohort never runs ``filter()`` while its membership is stable, so the
    padding a ragged trim adds used to stay: under alternating rejections the
    shared width grew by a column per cycle while the rows did not. Every
    lane's tokens must still equal the same lane run alone.
    """
    mx.random.seed(41)
    model = _tiny_qwen4_model()
    prompts = ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])
    cycles = 16
    accepts = [(2, 0) if cycle % 2 == 0 else (0, 2) for cycle in range(cycles)]
    tokens, batch = _forced_cycles(model, prompts, accepts, uids=[0, 1])
    physical = [
        cache
        for cache in batch.caches.target + batch.caches.draft
        if isinstance(cache, BatchQSAKVCache)
    ]
    assert physical
    for cache in physical:
        offsets = cache.offset.tolist()
        # Only the length difference between the rows remains as padding.
        assert cache._idx == max(offsets), (cache._idx, offsets)
        assert min(cache.left_padding.tolist()) == 0
    for row, prompt in enumerate(prompts):
        alone, _ = _forced_cycles(
            model, [prompt], [(pair[row],) for pair in accepts], uids=[row]
        )
        assert tokens[row] == alone[0]


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
            adaptive_mtp_depth={},
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
        mtp_terminal = None
        for _ in range(100):
            for index, batch in enumerate((ordinary, mtp)):
                _, responses = batch.next()
                outputs[index].extend(response.token for response in responses)
                if index == 1:
                    mtp_terminal = next(
                        (
                            response
                            for response in responses
                            if response.finish_reason is not None
                        ),
                        mtp_terminal,
                    )
            if not counting and len(outputs[1]) >= 1:
                model.mtp_step = counted_mtp_step
                mtp._generation_batch._adaptive_admitted_cap = 0
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
                mtp._generation_batch._adaptive_admitted_cap = 2
                switched = True
            if min(map(len, outputs)) >= 10:
                break
        assert switched and draft_calls > 0
        assert outputs[1] == outputs[0]
        assert mtp_terminal is not None
        adaptive = mtp_terminal.mtp_receipt["adaptive_depth"]
        assert adaptive["cost_model"]["buckets"]["1"][
            "goodput_tokens_per_second"
        ]
        assert all(
            row["elapsed_seconds"] > 0 and row["committed"] >= 0
            for row in adaptive["trace"]
        )
        ordinary.close()
        mtp.close()
    finally:
        mx.set_default_device(previous_device)


def test_wide_mtp_cohort_handoff_matches_ordinary_greedy_continuation():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    ordinary = handoff = None
    try:
        mx.random.seed(924)
        model = _tiny_qwen4_model()
        prompts = [[1, 7, 3, 9, 2], [4, 5, 6, 2, 8]]
        ordinary = BatchGenerator(
            model,
            completion_batch_size=2,
            prefill_batch_size=2,
            prefill_step_size=32,
        )
        handoff = BatchGenerator(
            model,
            completion_batch_size=2,
            prefill_batch_size=2,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2,
                "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 2,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 1}
            ),
        )
        ordinary_uids = ordinary.insert(prompts, max_tokens=[8, 8])
        handoff_uids = handoff.insert(
            prompts,
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(31), LaneRNG(32)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )

        def finish(generator, uids):
            output = {uid: [] for uid in uids}
            terminal = {}
            for _ in range(100):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        terminal[response.uid] = response
                if len(terminal) == len(uids):
                    return output, terminal
            raise AssertionError("generator did not terminate")

        ordinary_output, _ = finish(ordinary, ordinary_uids)
        handoff_output, terminal = finish(handoff, handoff_uids)
        assert [ordinary_output[uid] for uid in ordinary_uids] == [
            handoff_output[uid] for uid in handoff_uids
        ]
        assert handoff.scheduler_stats["mtp_ordinary_handoff_events"] == 1
        assert handoff.scheduler_stats["mtp_ordinary_handoff_lanes"] == 2
        assert all(
            response.mtp_receipt["route"] == "ordinary_after_mtp_handoff"
            and response.mtp_receipt["mtp_ordinary_handoff"]["engaged"]
            and response.mtp_receipt["mtp_ordinary_handoff"]["one_way"]
            for response in terminal.values()
        )
        assert not handoff._generation_batch._ordinary_handoff_latched
        (later_uid,) = handoff.insert(
            [prompts[0]],
            max_tokens=[4],
            lane_rngs=[LaneRNG(33)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        _, later_terminal = finish(handoff, [later_uid])
        assert later_terminal[later_uid].mtp_receipt["route"] == ("segmented_self_mtp")
    finally:
        if ordinary is not None:
            ordinary.close()
        if handoff is not None:
            handoff.close()
        mx.set_default_device(previous_device)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_zero_commit_width_lock_handoff_preserves_target_state_and_rng(temperature):
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import draw_key

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    ordinary = handoff = None
    try:
        mx.random.seed(934)
        model = _tiny_qwen4_model()
        late_prompt = [3, 4, 5, 6, 7]
        seed = 93
        reference_rng = LaneRNG(seed)

        def reference_sampler(logprobs):
            if temperature <= 0:
                return mx.argmax(logprobs, axis=-1)
            token = mx.random.categorical(
                logprobs[0] / temperature, key=draw_key(reference_rng)
            )
            return token[None]

        reference_sampler.batch_groupable = False
        ordinary = BatchGenerator(
            model, completion_batch_size=3, prefill_batch_size=3,
            prefill_step_size=32,
        )
        (ordinary_uid,) = ordinary.insert(
            [late_prompt], max_tokens=[16], samplers=[reference_sampler],
            lane_rngs=[reference_rng],
        )
        handoff = BatchGenerator(
            model, completion_batch_size=3, prefill_batch_size=3,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2, "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 3,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 2}
            ),
        )
        handoff.insert(
            [[1, 8, 12, 14, 2], [4, 5, 6, 2, 8]],
            max_tokens=[32, 32],
            lane_rngs=[LaneRNG(91), LaneRNG(92)],
            self_mtp_configs=[{"sampling_temp": temperature}] * 2,
        )
        for _ in range(20):
            handoff.next()
            if handoff._generation_batch._segmented_compute_width_locked:
                break
        else:
            raise AssertionError("initial segmented cohort never locked its width")

        late_rng = LaneRNG(seed)
        (late_uid,) = handoff.insert(
            [late_prompt], max_tokens=[16], lane_rngs=[late_rng],
            self_mtp_configs=[{
                "sampling_temp": temperature,
                "sampling_top_p": 1.0,
                "sampling_top_k": 0,
            }],
        )
        ordinary_tokens = []
        handoff_tokens = []
        handoff_from_draft = []
        ordinary_terminal = handoff_terminal = None
        for _ in range(100):
            _, ordinary_responses = ordinary.next()
            _, handoff_responses = handoff.next()
            for response in ordinary_responses:
                ordinary_tokens.append(response.token)
                if response.finish_reason:
                    ordinary_terminal = response
            for response in handoff_responses:
                if response.uid != late_uid:
                    continue
                handoff_tokens.append(response.token)
                handoff_from_draft.append(response.from_draft)
                if response.finish_reason:
                    handoff_terminal = response
            if ordinary_terminal is not None and handoff_terminal is not None:
                break
        assert ordinary_terminal is not None and handoff_terminal is not None
        assert handoff_tokens[0] == ordinary_tokens[0]
        if temperature <= 0:
            assert handoff_tokens == ordinary_tokens
        assert len(handoff_tokens) == 16
        assert not any(handoff_from_draft)
        assert handoff_terminal.rng_draws == reference_rng.draws
        assert bool(mx.array_equal(handoff_terminal.lane_rng.key, reference_rng.key))
        receipt = handoff_terminal.mtp_receipt
        boundary = receipt["mtp_ordinary_handoff"]
        assert boundary["committed_tokens_before_handoff"] == 0
        assert boundary["prepared_initial_token_pending"] is True
        assert boundary["prepared_initial_token_from_draft"] is False
        assert boundary["mtp_observed_compute_widths_before_handoff"] == []
        assert boundary["stats_before_handoff"]["draft_proposed"] == 0
        assert receipt["stats"]["draft_proposed"] == 0
        assert receipt["ordinary_compute_widths"]
    finally:
        if ordinary is not None:
            ordinary.close()
        if handoff is not None:
            handoff.close()
        mx.set_default_device(previous_device)




@pytest.mark.parametrize("segmented", [False, True])
def test_midstream_handoff_after_accepted_drafts_matches_greedy(segmented):
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    ordinary = handoff = None
    try:
        mx.random.seed(924)
        model = _tiny_qwen4_model()
        prompts = [[1, 8, 12, 14, 2], [4, 5, 6, 2, 8]]
        seeds = [31, 32]

        def ordinary_sampler(seed):
            del seed
            def sample(logprobs):
                return mx.argmax(logprobs, axis=-1)

            sample.batch_groupable = False
            return sample

        ordinary = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2,
            prefill_step_size=32,
        )
        config = {"num_draft": 2, "persistent": True}
        if segmented:
            config.update(
                {"segment_aware_live_tip": True, "segment_aware_cohort_size": 2}
            )
        handoff = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2,
            prefill_step_size=32, self_mtp=config,
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 1}
            ),
        )
        (ordinary_uid0,) = ordinary.insert(
            [prompts[0]], max_tokens=[32],
            samplers=[ordinary_sampler(seeds[0])],
        )
        (handoff_uid0,) = handoff.insert(
            [prompts[0]], max_tokens=[32], lane_rngs=[LaneRNG(seeds[0])],
            self_mtp_configs=[{
                "sampling_temp": 0.0, "sampling_top_p": 1.0,
                "sampling_top_k": 8,
            }],
        )
        ordinary_output = {ordinary_uid0: []}
        handoff_output = {handoff_uid0: []}
        for _ in range(40):
            for generator, output in (
                (ordinary, ordinary_output), (handoff, handoff_output)
            ):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
            lanes = handoff._generation_batch.state.lanes
            if lanes and lanes[0].stats.draft_accepted > 0:
                break
        else:
            raise AssertionError("self-MTP did not accept a draft before handoff")

        (ordinary_uid1,) = ordinary.insert(
            [prompts[1]], max_tokens=[16],
            samplers=[ordinary_sampler(seeds[1])],
        )
        (handoff_uid1,) = handoff.insert(
            [prompts[1]], max_tokens=[16], lane_rngs=[LaneRNG(seeds[1])],
            self_mtp_configs=[{
                "sampling_temp": 0.0, "sampling_top_p": 1.0,
                "sampling_top_k": 8,
            }],
        )
        ordinary_output[ordinary_uid1] = []
        handoff_output[handoff_uid1] = []
        ordinary_terminal = {}
        handoff_terminal = {}
        for _ in range(200):
            for generator, output, terminal in (
                (ordinary, ordinary_output, ordinary_terminal),
                (handoff, handoff_output, handoff_terminal),
            ):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        terminal[response.uid] = response
            if len(ordinary_terminal) == len(handoff_terminal) == 2:
                break
        assert [ordinary_output[ordinary_uid0], ordinary_output[ordinary_uid1]] == [
            handoff_output[handoff_uid0], handoff_output[handoff_uid1]
        ]
        receipt = handoff_terminal[handoff_uid0].mtp_receipt
        assert receipt["stats"]["draft_accepted"] > 0
        assert receipt["mtp_ordinary_handoff"][
            "committed_tokens_before_handoff"
        ] > 0
        assert receipt["ordinary_compute_widths"]
        assert receipt["observed_compute_widths"]
    finally:
        if ordinary is not None:
            ordinary.close()
        if handoff is not None:
            handoff.close()
        mx.set_default_device(previous_device)


def test_promoted_qwen4_handoff_matches_plain_from_committed_prefix(monkeypatch):
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda _device: new_stream(mx.cpu))
    handoff = reference = None
    try:
        mx.random.seed(925)
        model = _tiny_qwen4_model()
        prompt = [1, 8, 12, 14, 2]
        handoff = BatchGenerator(
            model, completion_batch_size=3, prefill_batch_size=3,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2, "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 3,
                "segment_aware_async_qsa_promotion": True,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 2}
            ),
        )
        uids = handoff.insert(
            [prompt, [4, 5, 6, 2, 8]], max_tokens=[20, 20],
            lane_rngs=[LaneRNG(41), LaneRNG(42)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )
        emitted = {uid: [] for uid in uids}
        for _ in range(20):
            _, responses = handoff.next()
            for response in responses:
                emitted[response.uid].append(response.token)
            batch = handoff._generation_batch
            if not batch.segmented_live_tip and emitted[uids[0]]:
                break
        else:
            raise AssertionError("segmented QSA state did not promote")

        recurrent = next(
            cache for cache in batch.state.caches.target
            if isinstance(cache, Qwen4ArraysCache)
        )
        assert recurrent.ple_history_fill is not None
        prefix = prompt + list(emitted[uids[0]])
        before_handoff = len(emitted[uids[0]])

        handoff.insert(
            [[3, 4, 5, 6, 7]], max_tokens=[12],
            lane_rngs=[LaneRNG(43)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        terminal = None
        for _ in range(100):
            _, responses = handoff.next()
            for response in responses:
                if response.uid != uids[0]:
                    continue
                emitted[uids[0]].append(response.token)
                if response.finish_reason:
                    terminal = response
            if terminal is not None:
                break
        assert terminal is not None
        assert terminal.mtp_receipt["mtp_ordinary_handoff"]["engaged"]

        reference = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=32,
        )
        (reference_uid,) = reference.insert(
            [prefix], max_tokens=[20 - before_handoff]
        )
        expected = []
        for _ in range(100):
            _, responses = reference.next()
            for response in responses:
                expected.append(response.token)
                if response.finish_reason:
                    break
            if responses and responses[-1].finish_reason:
                break
        assert emitted[uids[0]][before_handoff:] == expected
    finally:
        if handoff is not None:
            handoff.close()
        if reference is not None:
            reference.close()
        mx.set_default_device(previous_device)


@pytest.mark.parametrize("segmented", [False, True])
def test_seeded_sampling_state_survives_midstream_handoff(segmented):
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generators = []
    try:
        mx.random.seed(926)
        model = _tiny_qwen4_model()

        def run_once():
            config = {"num_draft": 2, "persistent": True}
            if segmented:
                config.update({
                    "segment_aware_live_tip": True,
                    "segment_aware_cohort_size": 2,
                })
            generator = BatchGenerator(
                model, completion_batch_size=2, prefill_batch_size=2,
                prefill_step_size=32, self_mtp=config,
                mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                    {"enabled": True, "max_mtp_width": 1}
                ),
            )
            generators.append(generator)
            (first_uid,) = generator.insert(
                [[1, 8, 12, 14, 2]], max_tokens=[20],
                lane_rngs=[LaneRNG(51)],
                self_mtp_configs=[{
                    "sampling_temp": 0.8, "sampling_top_p": 1.0,
                    "sampling_top_k": 8,
                }],
            )
            output = {first_uid: []}
            for _ in range(4):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
            lane = generator._generation_batch.state.lanes[0]
            draws_before = lane.rng.draws
            assert draws_before > 0
            (second_uid,) = generator.insert(
                [[4, 5, 6, 2, 8]], max_tokens=[12],
                lane_rngs=[LaneRNG(52)],
                self_mtp_configs=[{
                    "sampling_temp": 0.8, "sampling_top_p": 1.0,
                    "sampling_top_k": 8,
                }],
            )
            output[second_uid] = []
            terminal = {}
            for _ in range(100):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        terminal[response.uid] = response
                if len(terminal) == 2:
                    break
            assert len(terminal) == 2
            assert terminal[first_uid].lane_rng is not None
            assert terminal[first_uid].rng_draws > draws_before
            return (
                output[first_uid], output[second_uid],
                terminal[first_uid].rng_draws,
                terminal[second_uid].rng_draws,
            )

        assert run_once() == run_once()
    finally:
        for generator in generators:
            generator.close()
        mx.set_default_device(previous_device)


def test_handoff_latch_expires_below_threshold_while_plain_work_remains():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        mx.random.seed(927)
        model = _tiny_qwen4_model()
        generator = BatchGenerator(
            model, completion_batch_size=3, prefill_batch_size=3,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2, "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 3,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 2}
            ),
        )
        generator.insert(
            [[1, 8, 12, 14, 2], [4, 5, 6, 2, 8], [3, 4, 5, 6, 7]],
            max_tokens=[3, 3, 8],
            lane_rngs=[LaneRNG(61), LaneRNG(62), LaneRNG(63)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 3,
        )
        for _ in range(10):
            generator.next()
            if (len(generator._plain_fallback_batch) == 1
                    and not generator._generation_batch._ordinary_handoff_latched):
                break
        else:
            raise AssertionError("handoff latch did not expire below threshold")
        assert len(generator._plain_fallback_batch) == 1

        (later_uid,) = generator.insert(
            [[9, 8, 7, 6, 5]], max_tokens=[4],
            lane_rngs=[LaneRNG(64)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        terminal = None
        for _ in range(30):
            _, responses = generator.next()
            terminal = next(
                (response for response in responses
                 if response.uid == later_uid and response.finish_reason),
                terminal,
            )
            if terminal is not None:
                break
        assert terminal is not None
        assert terminal.mtp_receipt["route"] == "segmented_self_mtp"
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous_device)


def test_handoff_batches_cache_migration_once(monkeypatch):
    import mlx2.runtime.generate as generate_module
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        mx.random.seed(928)
        generator = BatchGenerator(
            _tiny_qwen4_model(), completion_batch_size=2,
            prefill_batch_size=2, prefill_step_size=32,
            self_mtp={"num_draft": 2, "persistent": True},
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 1}
            ),
        )
        generator.insert(
            [[1, 8, 12, 14, 2], [4, 5, 6, 2, 8]],
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(71), LaneRNG(72)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )
        generator.next()
        generator.next()

        original_merge = generate_module._merge_caches
        merge_widths = []

        def observed_merge(caches):
            merge_widths.append(len(caches))
            return original_merge(caches)

        monkeypatch.setattr(generate_module, "_merge_caches", observed_merge)
        generator.next()
        assert merge_widths == [2]
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous_device)


def test_width_lock_and_starved_fallbacks_migrate_in_one_plain_batch(
    monkeypatch,
):
    import mlx2.runtime.generate as generate_module
    from mlx2.runtime.generate import (
        WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN,
        BatchGenerator,
    )

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        mx.random.seed(929)
        generator = BatchGenerator(
            _tiny_qwen4_model(), completion_batch_size=2,
            prefill_batch_size=2, prefill_step_size=32,
            self_mtp={
                "num_draft": 2, "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 2,
            },
        )
        generator.insert(
            [[1, 8, 12, 14, 2], [4, 5, 6, 2, 8]],
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(81), LaneRNG(82)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )
        generator.next()
        generator.next()
        batch = generator._generation_batch

        late = batch._detach_packages([1])[0]
        batch._segmented_compute_width_locked = True
        for _ in range(WIDTH_LOCK_DEFERRALS_BEFORE_PLAIN + 1):
            batch._attach_packages([late])
            if batch._plain_ready:
                break
            late = batch._paused.pop(1)
        assert [package.detached.lane.uid
                for package in batch._plain_ready] == [1]

        starved = batch._detach_packages([0])[0]
        batch._paused[0] = starved
        assert batch.demote_oldest_paused_to_plain() == 0
        assert [package.detached.lane.uid
                for package in batch._plain_ready] == [1, 0]

        original_merge = generate_module._merge_caches
        merge_widths = []

        def observed_merge(caches):
            merge_widths.append(len(caches))
            return original_merge(caches)

        monkeypatch.setattr(generate_module, "_merge_caches", observed_merge)
        assert generator._migrate_plain_fallbacks() == []
        assert merge_widths == [2]
        assert len(generator._plain_fallback_batch) == 2

        terminal = {}
        for _ in range(20):
            _, responses = generator.next()
            for response in responses:
                if response.finish_reason:
                    terminal[response.uid] = response
            if len(terminal) == 2:
                break
        assert len(terminal) == 2
        assert all(response.lane_rng is not None
                   for response in terminal.values())
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous_device)


def test_handoff_preserves_logits_processor_and_partial_stop_state():
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.generate import BatchGenerator, StopSequenceMatcher

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    ordinary = handoff = None
    try:
        mx.random.seed(925)
        model = _tiny_qwen4_model()
        prompt = [1, 7, 3, 9, 2]

        def force_sequence(tokens, logits):
            index = max(int(tokens.shape[-1]) - len(prompt), 0)
            token = (11, 12, 13)[min(index, 2)]
            return mx.where(
                mx.arange(logits.shape[-1]) == token,
                mx.zeros_like(logits),
                -1e9,
            )

        ordinary = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2,
            prefill_step_size=32,
        )
        handoff = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2,
            prefill_step_size=32,
            self_mtp={
                "num_draft": 2, "persistent": True,
                "segment_aware_live_tip": True,
                "segment_aware_cohort_size": 2,
            },
            mtp_ordinary_handoff=MTPOrdinaryHandoffPolicy.from_value(
                {"enabled": True, "max_mtp_width": 1}
            ),
        )
        matcher = StopSequenceMatcher([[11, 12]])
        (ordinary_uid,) = ordinary.insert(
            [prompt], max_tokens=[8], logits_processors=[[force_sequence]],
            stop_matchers=[matcher],
        )
        (handoff_uid,) = handoff.insert(
            [prompt], max_tokens=[8], logits_processors=[[force_sequence]],
            stop_matchers=[matcher], lane_rngs=[LaneRNG(41)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        ordinary_tokens = []
        handoff_tokens = []
        while not handoff_tokens:
            _, ordinary_responses = ordinary.next()
            _, handoff_responses = handoff.next()
            ordinary_tokens.extend(r.token for r in ordinary_responses)
            handoff_tokens.extend(r.token for r in handoff_responses)
        handoff.insert(
            [[4, 5, 6, 2, 8]], max_tokens=[8],
            lane_rngs=[LaneRNG(42)],
            self_mtp_configs=[{"sampling_temp": 0.0}],
        )
        ordinary_reason = handoff_reason = None
        for _ in range(40):
            _, ordinary_responses = ordinary.next()
            _, handoff_responses = handoff.next()
            ordinary_tokens.extend(r.token for r in ordinary_responses)
            handoff_tokens.extend(
                r.token for r in handoff_responses if r.uid == handoff_uid
            )
            ordinary_reason = next(
                (r.finish_reason for r in ordinary_responses
                 if r.uid == ordinary_uid and r.finish_reason),
                ordinary_reason,
            )
            handoff_reason = next(
                (r.finish_reason for r in handoff_responses
                 if r.uid == handoff_uid and r.finish_reason),
                handoff_reason,
            )
            if ordinary_reason and handoff_reason:
                break
        assert handoff_tokens == ordinary_tokens == [11, 12]
        assert handoff_reason == ordinary_reason == "stop"
    finally:
        if ordinary is not None:
            ordinary.close()
        if handoff is not None:
            handoff.close()
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
