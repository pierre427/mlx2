"""device_round lazy drafts for the DFlash-family drafters, CPU only.

Pins, per drafter (plain DFlash v1, Kimi DFlash, XPress, causal and
bidirectional DSpark, LiLiCorr, DSpine): ``draft_lazy`` returns the tokens
of the host ``draft_distributions`` for one lane and leaves the same draft
cache (and RNG) state, also across a rejected-tail commit; the engine with
``device_round {lazy_draft, decide}`` decodes exactly what the option-off
engine decodes at one and two lanes, with the same proposed and accepted
counts on a fixture with real accepts and rejects (a two-lane cohort is
declined ``lazy_draft_width``, every drafter here declaring
``draft_lazy_width_one``, and served again once it drops to one lane);
proposal composition over XPress serves device rounds with its history;
sampled drafts verify with the recorded law; and drafters that cannot serve
a lazy round (UBTree,
a subclass that changes the host draft, LiLiCorr feedback capture, DSpine
processed lanes) decline with a counted reason.  No checkpoint is loaded:
every model is a seeded tiny CPU model.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.speculative_device_round import DECLINE_REASONS
from mlx2.runtime.speculative_sampling import RequestRNG

BOTH = {"lazy_draft": True, "decide": True}
PROC = {"lazy_draft": True, "decide": True, "processors": True}
SAMPLED = {"lazy_draft": True, "decide": True, "sampled_drafts": True}


# -- fixtures (seeded tiny models; built once per call) ---------------------------


def _plain(kimi=False):
    from test_dflash_v1_cpu import draft_config, target

    from mlx2.adapters import dflash_v1
    from mlx2.runtime.drafters.kimi_dflash import KimiDFlashDraftModel
    from mlx2.runtime.drafters.plain_dflash import PlainDFlashDraftModel

    model = target()
    mx.random.seed(31)
    cls = KimiDFlashDraftModel if kimi else PlainDFlashDraftModel
    draft = cls(dflash_v1.parse_config(draft_config()))
    draft.eval()
    mx.eval(draft.parameters())
    return model, draft.bind(model), 3


def _xpress():
    from test_standard_xpress_serving_cpu import tiny

    model, draft = tiny()
    return model, draft, 2


def _causal_dspark():
    from test_standard_xpress_serving_cpu import tiny

    from mlx2.adapters.nemotron35_dspark import DSparkConfig
    from mlx2.runtime.drafters.dspark import DSparkDraftModel

    model, _ = tiny()
    mx.random.seed(41)
    draft = DSparkDraftModel(DSparkConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, vocab_size=9, block_size=5, mask_token_id=8,
        target_layer_ids=[0, 2], num_target_layers=3,
        layer_types=["sliding_attention"] * 2, sliding_window=4, markov_rank=2))
    head = draft.markov_head
    head.markov_w1.weight = mx.random.normal(head.markov_w1.weight.shape)
    head.markov_w2.weight = mx.random.normal(head.markov_w2.weight.shape)
    mx.eval(draft.parameters())
    return model, draft.bind(model), 3


def _bidirectional_dspark():
    from test_qwen38_dspark_cpu import tiny_pair

    target, draft = tiny_pair()
    return target, draft, 4


def _lilicorr():
    from test_lilicorr_serving_cpu import pair

    model, draft = pair()
    return model, draft, 2


def _dspine():
    from test_dspine_drafter_cpu import hybrid_pair

    _, target, draft = hybrid_pair()
    return target, draft, 4


DRAFTERS = {
    "plain_dflash": _plain,
    "kimi_dflash": lambda: _plain(kimi=True),
    "xpress": _xpress,
    "dspark_causal": _causal_dspark,
    "dspark_bidirectional": _bidirectional_dspark,
    "lilicorr": _lilicorr,
    "dspine": _dspine,
}
PROCESSED = [name for name in DRAFTERS if name != "dspine"]


def _vocab(draft):
    return int(draft.config.vocab_size)


# Prompts on which each seeded tiny pair both accepts and rejects drafts
# (a random tiny drafter mostly misses; these were found by search).
PROMPTS = {
    "plain_dflash": ([6, 10, 3], [2, 8, 6]),
    "kimi_dflash": ([6, 10, 3], [2, 8, 6]),
    "xpress": ([3, 7, 4, 0], [3, 4, 4, 7, 6, 0, 7]),
    "dspark_causal": ([1, 4, 1, 7], [6, 0, 7, 4, 3, 1]),
    "dspark_bidirectional": ([3, 114, 106, 49, 55, 77], [120, 63, 70, 29, 44, 29]),
    "lilicorr": ([3, 7, 4, 0], [1, 4, 1, 7]),
    "dspine": ([72, 108, 102, 97], [60, 83, 48, 100, 26, 12]),
}


def _presence(prompt, penalty=1.5):
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=penalty, presence_context_size=20,
                                  penalty_generation_start=len(prompt))


def _decode(name, *, lanes=1, steps=24, holder=None, sampling=None, processed=False, **options):
    target, draft, depth = DRAFTERS[name]()
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding=f"tiny-{name}", num_draft=depth,
        prefill_step_size=4, **options)
    prompts = [list(p) for p in PROMPTS[name][:lanes]]
    try:
        extra = {"logits_processors": [_presence(p) for p in prompts]} if processed else {}
        uids = batch.insert(prompts, max_tokens=[steps] * lanes,
                            sampling_configs=[sampling or {"sampling_temp": 0}] * lanes, **extra)
        got = {uid: [] for uid in uids}
        rngs = {}
        for _ in range(600):
            for uid, lane in batch.lanes.items():
                rngs[uid] = lane.rng.snapshot()
            _, responses = batch.next()
            for response in responses:
                if response.uid in got:
                    got[response.uid].append(response.token)
            if not batch.lanes:
                break
        else:
            raise AssertionError("tiny scheduler stalled")
        stats = dict(batch.scheduler_stats)
        stats["_rng"] = [rngs[uid] for uid in uids]
        stats["_draft"] = dict(draft.stats)
        return [got[uid] for uid in uids], stats
    finally:
        if holder is not None:
            holder.update(batch.scheduler_stats)
        batch.close()


# -- drafter level: lazy tokens and cache state equal the host draft ------------


def _state(cache):
    arrays = []
    for entry in cache:
        state = entry.state
        arrays.append((int(entry.offset), [np.asarray(part).tolist() for part in state]
                       if isinstance(state, (list, tuple)) else np.asarray(state).tolist()))
    return arrays


def _host_kwargs(draft, processors, history):
    kwargs = {}
    if processors:
        kwargs.update(logits_processors=[processors], processor_histories=[history])
    elif getattr(draft, "requires_processor_histories", False):
        kwargs["processor_histories"] = [history]
    return kwargs


def _lazy_kwargs(draft, processors, history, rng):
    kwargs = {}
    if getattr(draft, "draft_lazy_consumes_rng", False):
        kwargs["rng"] = rng
    if processors:
        kwargs.update(logits_processors=processors, processor_history=history)
    elif getattr(draft, "draft_lazy_history", False):
        kwargs["processor_history"] = history
    return kwargs


def _width(draft):
    return int(draft.config.hidden_size) * len(draft.config.target_layer_ids)


@pytest.mark.parametrize("processed", [False, True])
@pytest.mark.parametrize("name", list(DRAFTERS))
def test_draft_lazy_equals_host_draft_and_cache_state(name, processed):
    if processed and name not in PROCESSED:
        pytest.skip("DSpine applies no draft processors (declined; see below)")
    _, draft, depth = DRAFTERS[name]()
    history = [5, 2, 2, 7, 1]
    processors = _presence(history[:2], penalty=2.5) if processed else []
    anchor = 3
    cache_a, cache_b = draft.make_cache(), draft.make_cache()
    rng_a, rng_b = RequestRNG(5), RequestRNG(5)
    for step, rows in enumerate((5, 2, 1)):
        hidden = mx.random.normal((1, rows, _width(draft)), key=mx.random.key(10 + step))
        for count in sorted({1, depth}):
            # Each draft runs on a copy: the committed context is what the
            # last call below appends.
            import copy

            a, b = copy.deepcopy(cache_a), copy.deepcopy(cache_b)
            tokens, _laws = draft.draft_distributions(
                [anchor], hidden, draft.batch_caches([a]), count, [rng_a], [0.0],
                **_host_kwargs(draft, processors, history))
            lazy = draft.draft_lazy(anchor, hidden, b, count,
                                    **_lazy_kwargs(draft, processors, history, rng_b))
            assert lazy.dtype == mx.int32 and lazy.shape == (count,)
            assert [int(t) for t in lazy.tolist()] == tokens[0], (step, count)
            assert _state(a) == _state(b)
            assert rng_a.snapshot() == rng_b.snapshot()
        # Commit this step's context in both caches the same way, then trim a
        # rejected tail (the commit after a partial accept).
        draft.draft_distributions([anchor], hidden, draft.batch_caches([cache_a]), depth,
                                  [rng_a], [0.0], **_host_kwargs(draft, processors, history))
        draft.draft_lazy(anchor, hidden, cache_b, depth,
                         **_lazy_kwargs(draft, processors, history, rng_b))
        if rows > 1:
            for entry_a, entry_b in zip(cache_a, cache_b, strict=True):
                if entry_a.is_trimmable():
                    entry_a.trim(1)
                    entry_b.trim(1)
        assert _state(cache_a) == _state(cache_b)
        history.append(anchor)
        anchor = (anchor + 4) % (_vocab(draft) - 1)


# -- engine level: device round output equals the option-off engine --------------


@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("name", list(DRAFTERS))
def test_engine_greedy_output_and_counts_unchanged(name, lanes):
    off, off_stats = _decode(name, lanes=lanes)
    on, stats = _decode(name, lanes=lanes, device_round=BOTH)
    assert on == off
    assert stats["_rng"] == off_stats["_rng"]
    rounds = stats["external_device_round_rounds"]
    if lanes > 1:
        # draft_lazy_width_one: a multi-lane cohort keeps the host round (the
        # host's batched draft is not proven bitwise per-lane on the GPU); a
        # cohort that drops to one lane (one request done) is served again.
        assert stats[DECLINE_REASONS].get("lazy_draft_width", 0) > 0, stats[DECLINE_REASONS]
    else:
        assert rounds > 0
        assert stats["external_device_round_host_reads"] == rounds
        assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]
    # Real accepts and rejects, proposed and accepted exactly as on the host.
    assert 0 < off_stats["accepted_proposals"] < off_stats["proposed_tokens"]
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


@pytest.mark.parametrize("name", list(DRAFTERS))
def test_lazy_draft_alone_is_exact(name):
    off, _ = _decode(name)
    on, stats = _decode(name, device_round={"lazy_draft": True})
    assert on == off and stats["external_device_round_rounds"] > 0


@pytest.mark.parametrize("name", PROCESSED)
def test_presence_penalty_lane_keeps_the_device_round_exactly(name):
    off, off_stats = _decode(name, processed=True)
    on, stats = _decode(name, processed=True, device_round=PROC)
    assert on == off and stats["_rng"] == off_stats["_rng"]
    assert stats["external_device_round_rounds"] > 0
    assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


def test_dspine_processed_lane_declines_counted():
    from mlx2.runtime.drafters.dspine import DSpineDraftModel

    # The host chain refuses draft processors, so a processed lane fails on
    # either path; the device round declines it (counted) before drafting.
    assert DSpineDraftModel.draft_lazy_processors is False
    stats = {}
    with pytest.raises(ValueError, match="does not apply logits processors"):
        _decode("dspine", processed=True, device_round=PROC, holder=stats)
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("lazy_draft_processors", 0) > 0


@pytest.mark.parametrize("name", list(DRAFTERS))
def test_sampled_drafts_serve_sampled_lanes(name):
    sampling = {"sampling_temp": 0.8, "seed": 3}
    first, stats = _decode(name, sampling=sampling, device_round=SAMPLED)
    again, _ = _decode(name, sampling=sampling, device_round=SAMPLED)
    assert first == again and len(first[0]) == 24
    assert stats["external_device_round_rounds"] > 0
    # exact_verification unset: the block rule (DEFAULT ON, Pierre 2026-11-27).
    assert stats["external_device_round_decide_block_lanes"] > 0
    assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]


def test_point_mass_record_is_the_point_mass_law():
    from mlx2.runtime.drafters.dflash_base import lazy_point_mass_record

    recorded = []

    def sampler(logits, temp=0.7):
        scaled = logits.astype(mx.float32) * (1.0 / temp)
        recorded.append(scaled)
        return mx.random.categorical(scaled, key=mx.random.key(len(recorded))).astype(mx.int32)

    tokens = mx.array([2, 0, 5], dtype=mx.int32)
    assert lazy_point_mass_record(sampler, tokens, 7) is tokens
    q = mx.concatenate(recorded, axis=0)
    law = np.asarray(mx.exp(q - mx.logsumexp(q, axis=-1, keepdims=True)))
    np.testing.assert_array_equal(law, np.eye(7)[[2, 0, 5]])


# -- declines -----------------------------------------------------------------------


def test_ubtree_declares_no_lazy_draft():
    from mlx2.runtime.drafters.ubtree import UBTreeDraftModel

    assert UBTreeDraftModel.draft_lazy is None
    assert not callable(getattr(UBTreeDraftModel, "draft_lazy", None))


def test_lazy_drafts_live_on_the_families_not_the_shared_base():
    """``DFlashDraftModel`` declares none, so DFlash2 and other subclasses
    never inherit a lazy draft that mirrors another family's host draft."""
    from mlx2.runtime.drafters.dflash_base import DFlashDraftModel
    from mlx2.runtime.drafters.dspark import (
        BidirectionalDSparkDraftModel,
        DSparkDraftModel,
    )
    from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
    from mlx2.runtime.drafters.xpress import XPressDraftModel

    assert not hasattr(DFlashDraftModel, "draft_lazy")
    for cls in (DSparkDraftModel, LiLiCorrDraftModel):
        assert cls.draft_lazy is not XPressDraftModel.draft_lazy
        assert cls.draft_lazy_history is False
    assert BidirectionalDSparkDraftModel.draft_lazy is DSparkDraftModel.draft_lazy


@pytest.mark.parametrize("name", list(DRAFTERS))
def test_a_subclass_that_changes_the_host_draft_declines_counted(name):
    target, draft, depth = DRAFTERS[name]()
    original = type(draft).draft_distributions

    class Changed(type(draft)):
        def draft_distributions(self, *args, **kwargs):
            return original(self, *args, **kwargs)

    draft.__class__ = Changed
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding=f"tiny-{name}", num_draft=depth,
        prefill_step_size=4, device_round=BOTH)
    try:
        batch.insert([list(PROMPTS[name][0])], max_tokens=[8],
                     sampling_configs=[{"sampling_temp": 0}])
        for _ in range(200):
            batch.next()
            if not batch.lanes:
                break
        stats = batch.scheduler_stats
    finally:
        batch.close()
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("lazy_draft_host_override", 0) > 0


def test_lilicorr_feedback_capture_declines_counted():
    from mlx2.runtime.proposal_composition import LazyDecline

    _, draft, _ = _lilicorr()
    draft.feedback_manager = object()
    with pytest.raises(LazyDecline) as declined:
        draft.draft_lazy(3, mx.zeros((1, 2, _width(draft))), draft.make_cache(), 2)
    assert declined.value.reason == "lilicorr_feedback_capture"


def test_xpress_without_a_history_declines():
    from mlx2.runtime.proposal_composition import LazyDecline

    _, draft, _ = _xpress()
    with pytest.raises(LazyDecline) as declined:
        draft.draft_lazy(3, mx.zeros((1, 2, _width(draft))), draft.make_cache(), 2)
    assert declined.value.reason == "xpress_without_history"


def test_composed_xpress_serves_device_rounds_and_equals_option_off():
    # ComposedDraftModel forwards the committed history to a backend that
    # declares ``draft_lazy_history`` (XPress's predecessor token) even
    # without processors, as its host path passes ``processor_histories``.
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    def run(**options):
        target, draft, depth = _xpress()
        draft = ComposedDraftModel(draft, {"prompt_lookup": True})
        batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="tiny-xpress",
                                            num_draft=depth, prefill_step_size=4, **options)
        try:
            uids = batch.insert([list(PROMPTS["xpress"][0])], max_tokens=[16],
                                sampling_configs=[{"sampling_temp": 0}])
            got = []
            for _ in range(600):
                _, responses = batch.next()
                got += [r.token for r in responses if r.uid == uids[0]]
                if not batch.lanes:
                    break
            return got, dict(batch.scheduler_stats), dict(draft.composition_stats)
        finally:
            batch.close()

    off, off_stats, off_composition = run()
    on, stats, composition = run(device_round=BOTH)
    assert on == off
    assert stats["external_device_round_rounds"] > 0
    assert stats["external_device_round_declines"] == 0, stats[DECLINE_REASONS]
    assert composition == off_composition
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


def test_lazy_drafts_refuse_wide_batches():
    for build in DRAFTERS.values():
        _, draft, _ = build()
        with pytest.raises(ValueError, match="one lane"):
            draft.draft_lazy(3, mx.zeros((2, 2, _width(draft))), draft.make_cache(), 1,
                             processor_history=[1])
