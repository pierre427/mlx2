"""``device_round`` lazy drafts of the standalone drafters, CPU only.

Tiny random weights (no real artifacts).  Per drafter (Laguna DFlash, Muse
DFlash assistant, Hy4-flash DFly2): ``draft_lazy`` returns the host
``draft_distributions`` tokens and leaves the same draft-cache state, stats
and lane RNG; a ``device_round {lazy_draft, decide}`` engine decodes what the
option-off engine decodes with the same proposed/accepted counts, with real
accepts and rejects; declines (two-lane dense-block cohorts, tree and AST
engines, Muse DSpark, a decide-only KV-sharing drafter, composition over a
KV-sharing drafter) are counted and keep the host output.
"""

from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

import test_hy_v4_flash_cpu as hy
import test_laguna_dflash_cpu as laguna
import test_muse_dflash_assistant_cpu as muse

from mlx2.runtime.drafters.dflash2 import DRAFT_LAZY_HOST_REDO
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.speculative_device_round import DECLINE_REASONS
from mlx2.runtime.speculative_sampling import RequestRNG

BOTH = {"lazy_draft": True, "decide": True}
PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


# -- builders -------------------------------------------------------------------


def _favour(draft, attr, vocab, token):
    """Bias a dense-head drafter toward one token: accepts where the target
    emits it, rejects elsewhere."""
    head = getattr(draft, attr)
    object.__setattr__(
        draft, attr,
        lambda hidden: head(hidden) + mx.where(mx.arange(vocab) == token, 1e4, 0.0),
    )


def laguna_pair():
    target, draft = laguna.tiny()
    favourite = Counter(laguna.greedy_reference(target, PROMPT, 24)).most_common(1)[0][0]
    _favour(draft, "lm_head", 48, favourite)
    return target, draft


def muse_pair():
    target, draft = muse.tiny()
    favourite = Counter(muse.greedy_reference(target, PROMPT, 24)).most_common(1)[0][0]
    _favour(draft, "_logits", muse.VOCAB, favourite)
    return target, draft


PAIRS = {"laguna": laguna_pair, "muse": muse_pair}


def drain(batch, uids):
    got = {uid: [] for uid in uids}
    for _ in range(600):
        result = batch.next()
        responses = result[1] if isinstance(result, tuple) else result
        for response in responses:
            got[response.uid].append(int(response.token))
        if not batch.lanes:
            return [got[uid] for uid in uids]
    raise AssertionError("scheduler stalled")


def decode(build, *, prompts=(PROMPT,), steps=20, sampling=None, num_draft=4, **options):
    target, draft = build()
    batch = ExternalDraftBatchGenerator(target, draft_model=draft, binding="test",
                                        num_draft=num_draft, prefill_step_size=3, **options)
    try:
        uids = batch.insert([list(p) for p in prompts], max_tokens=[steps] * len(prompts),
                            sampling_configs=[sampling or {"sampling_temp": 0}] * len(prompts))
        return drain(batch, uids), dict(batch.scheduler_stats), dict(draft.stats)
    finally:
        batch.close()


def assert_served_exactly(off, on):
    (off_out, off_stats, off_draft), (on_out, stats, on_draft) = off, on
    assert on_out == off_out
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_declines"] == 0 and stats[DECLINE_REASONS] == {}
    assert stats["external_device_round_host_reads"] == rounds
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key
    # Real accepts and rejects.
    assert 0 < stats["accepted_proposals"] < stats["proposed_tokens"]
    assert on_draft == off_draft


def _cache_state(cache):
    return [(entry.offset, [np.asarray(x) for x in entry.state]) for entry in cache]


def _assert_same_cache(a, b):
    for (oa, xa), (ob, xb) in zip(_cache_state(a), _cache_state(b), strict=True):
        assert oa == ob
        for x, y in zip(xa, xb, strict=True):
            np.testing.assert_array_equal(x, y)


# -- dense-block drafters (Laguna DFlash, Muse DFlash) ---------------------------


def _features(name, target, draft, prompt):
    return target.prefill_body(mx.array([prompt]), target.make_cache(),
                               draft.config.target_layer_ids)


@pytest.mark.parametrize("name", sorted(PAIRS))
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_dense_block_lazy_equals_host(name, depth):
    target, draft = PAIRS[name]()
    features = _features(name, target, draft, PROMPT[:6])
    cache_a, cache_b = draft.make_cache(), draft.make_cache()
    rng_a, rng_b = RequestRNG(3), RequestRNG(3)
    tokens, _ = draft.draft_distributions([7], features, draft.batch_caches([cache_a]), depth,
                                          [rng_a], [0.0])
    host_stats = dict(draft.stats)
    lazy = draft.draft_lazy(7, features, cache_b, depth, rng=rng_b)
    assert lazy.dtype == mx.int32 and lazy.shape == (depth,)
    assert [int(t) for t in lazy.tolist()] == tokens[0]
    _assert_same_cache(cache_a, cache_b)
    assert rng_a.snapshot() == rng_b.snapshot() and rng_a.draws == depth
    assert draft.stats["block_drafts"] == host_stats["block_drafts"] + 1
    # Rounds after a rejection: a one-row commit, then another draft.
    more = features[:, -1:]
    tokens, _ = draft.draft_distributions([3], more, draft.batch_caches([cache_a]), depth,
                                          [rng_a], [0.0])
    lazy = draft.draft_lazy(3, more, cache_b, depth, rng=rng_b)
    assert [int(t) for t in lazy.tolist()] == tokens[0]
    _assert_same_cache(cache_a, cache_b)


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_lazy_with_processors_equals_host(name):
    from mlx2.runtime.sample_utils import make_logits_processors

    target, draft = PAIRS[name]()
    features = _features(name, target, draft, PROMPT[:6])
    processors = make_logits_processors(presence_penalty=1e5)
    history = list(PROMPT[:6])
    tokens, _ = draft.draft_distributions(
        [7], features, draft.batch_caches([draft.make_cache()]), 4, [RequestRNG(1)], [0.0],
        logits_processors=[processors], processor_histories=[history])
    lazy = draft.draft_lazy(7, features, draft.make_cache(), 4, rng=RequestRNG(1),
                            logits_processors=processors, processor_history=history)
    assert [int(t) for t in lazy.tolist()] == tokens[0]
    # The penalty moved the favoured token off its repeats (processors engaged).
    plain = draft.draft_lazy(7, features, draft.make_cache(), 4)
    assert [int(t) for t in plain.tolist()] != tokens[0]


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_dead_position_asks_for_a_host_redo(name):
    target, draft = PAIRS[name]()
    features = _features(name, target, draft, PROMPT[:6])
    history = list(PROMPT[:6])

    def dead_after_first(tokens, logits):
        if tokens.shape[0] > len(history) + 1:
            return mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return logits

    dead_after_first.probe = dead_after_first
    # The host loop ends this lane's draft after one token.
    tokens, _ = draft.draft_distributions(
        [7], features, draft.batch_caches([draft.make_cache()]), 3, [RequestRNG(1)], [0.0],
        logits_processors=[[dead_after_first]], processor_histories=[history])
    assert len(tokens[0]) == 1
    lazy = draft.draft_lazy(7, features, draft.make_cache(), 3,
                            logits_processors=[dead_after_first], processor_history=history)
    assert [int(t) for t in lazy.tolist()] == [DRAFT_LAZY_HOST_REDO] * 3


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_lazy_refusals_move_no_state(name):
    from mlx2.runtime.proposal_composition import LazyDecline

    target, draft = PAIRS[name]()
    features = _features(name, target, draft, PROMPT[:6])
    cache = draft.make_cache()
    with pytest.raises(LazyDecline):
        draft.draft_lazy(mx.array([7]), features, cache, 2)
    with pytest.raises(ValueError, match="one lane"):
        draft.draft_lazy(7, mx.concatenate([features, features]), cache, 2)
    assert all(entry.offset == 0 for entry in cache)
    assert draft.stats["block_drafts"] == 0


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_device_round_greedy_equals_host(name):
    off = decode(PAIRS[name])
    assert_served_exactly(off, decode(PAIRS[name], device_round=BOTH))
    assert_served_exactly(off, decode(PAIRS[name], device_round={"lazy_draft": True}))


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_presence_penalty_lane_equals_host(name):
    sampling = {"sampling_temp": 0, "presence_penalty": 1.5}
    off = decode(PAIRS[name], sampling=sampling)
    on = decode(PAIRS[name], sampling=sampling, device_round={**BOTH, "processors": True})
    assert on[0] == off[0]
    assert on[1]["external_device_round_rounds"] > 0
    assert on[1][DECLINE_REASONS] == {}
    for key in ("proposed_tokens", "accepted_proposals"):
        assert on[1][key] == off[1][key], key


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_two_lane_cohorts_decline_counted(name):
    prompts = (PROMPT, [9, 2, 5, 4])
    off = decode(PAIRS[name], prompts=prompts)
    on = decode(PAIRS[name], prompts=prompts, device_round=BOTH)
    assert on[0] == off[0]
    reasons = on[1][DECLINE_REASONS]
    assert reasons.get("lazy_draft_width", 0) > 0
    # ``decide`` alone then meets the host round's dense one-hot laws (the
    # pre-existing host verifier keeps them).
    assert set(reasons) <= {"lazy_draft_width", "non_point_mass_proposals"}
    assert on[1]["external_device_round_declines"] == sum(reasons.values())


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_dense_block_sampled_drafts_run_on_the_device(name):
    sampling = {"sampling_temp": 0.8}
    build = muse._spiked_pair if name == "muse" else PAIRS[name]
    on = decode(build, sampling=sampling,
                device_round={**BOTH, "sampled_drafts": True})
    stats = on[1]
    assert len(on[0][0]) == 20
    assert stats["external_device_round_decide_ratio_lanes"] > 0
    assert stats["accepted_proposals"] > 0
    point = decode(build, sampling=sampling, device_round=BOTH)[1]
    assert point["external_device_round_decide_point_mass_lanes"] > 0


def test_muse_tree_engine_declines_counted(monkeypatch):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    off = decode(muse_pair)
    on = decode(muse_pair, device_round=BOTH)
    assert on[0] == off[0]
    stats = on[1]
    assert stats["external_device_round_rounds"] == 0
    assert set(stats[DECLINE_REASONS]) <= {"tree_topology"}
    assert stats["external_device_round_declines"] == sum(stats[DECLINE_REASONS].values())


def test_muse_dspark_does_not_inherit_the_assistant_lazy_draft():
    from mlx2.runtime.drafters.muse_dflash import MuseDFlashDraftModel
    from mlx2.runtime.drafters.muse_dspark import MuseDSparkDraftModel

    assert callable(MuseDFlashDraftModel.draft_lazy)
    assert MuseDSparkDraftModel.draft_lazy is None  # counted drafter_without_lazy_draft


def test_research_and_wrapped_drafters_never_reach_the_device_round_directly():
    from mlx2.runtime.drafters.dpara import MDFlashDParaDraftModel
    from mlx2.runtime.drafters.inkling2_dspark import Inkling2DSparkDraftModel
    from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal

    # DPara is a single-request research model and Inkling 2 DSpark reaches
    # the external engine only through its wrapper (which drafts lazily).
    for cls in (MDFlashDParaDraftModel, Inkling2DSparkDraftModel):
        assert not hasattr(cls, "draft_distributions") and not hasattr(cls, "draft_lazy")
    assert callable(Inkling2DSparkExternal.draft_lazy)


# -- Hy4-flash DFly2 (KV-sharing; target caches passed) ------------------------


@pytest.fixture(scope="module")
def hy_tiny():
    config = hy.tiny_config()
    W = hy.checkpoint(config)
    model, head = hy.build(config, W)
    return config, W, model, head


def _hy_favoured(model, head, prompt, count=24):
    """Bias the head toward the target's most frequent greedy token."""
    favourite = Counter(hy._greedy(model, prompt, count)).most_common(1)[0][0]
    step = head.step_logits
    vocab = model.args.vocab_size
    object.__setattr__(head, "step_logits", lambda *a, **k: step(*a, **k) + mx.where(
        mx.arange(vocab) == favourite, 1e4, 0.0))
    return step


def hy_decode(hy_tiny, *, prompts=(hy.TOKENS[:13],), steps=16, sampling=None, favour=True,
              compose=False, **options):
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel

    _, _, model, head = hy_tiny
    model.record_layers = frozenset(head.index_layers)
    step = _hy_favoured(model, head, list(prompts[0])) if favour else None
    try:
        draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
        if compose:
            from mlx2.runtime.proposal_composition import ComposedDraftModel

            draft = ComposedDraftModel(draft, {"prompt_lookup": True})
        batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                            num_draft=3, prefill_step_size=4, **options)
        try:
            uids = batch.insert([list(p) for p in prompts], max_tokens=[steps] * len(prompts),
                                sampling_configs=[sampling or {"sampling_temp": 0}] * len(prompts))
            return drain(batch, uids), dict(batch.scheduler_stats), dict(draft.stats)
        finally:
            batch.close()
    finally:
        model.record_layers = frozenset()
        if step is not None:
            object.__setattr__(head, "step_logits", step)


def test_hy4_dfly_lazy_equals_host(hy_tiny):
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel

    _, _, model, head = hy_tiny
    prompt = hy.TOKENS[:13]
    model.record_layers = frozenset(head.index_layers)
    try:
        cache = model.make_cache()
        hidden = model.hidden(mx.array([prompt]), cache)
    finally:
        model.record_layers = frozenset()
    bonus = int(mx.argmax(model.logits(hidden[:, -1]), -1).item())
    draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
    for depth in (1, 3):
        cache_a, cache_b = draft.make_cache(), draft.make_cache()
        tokens, _ = draft.draft_distributions(
            [bonus], hidden, draft.batch_caches([cache_a]), depth, [RequestRNG(1)], [0.0],
            target_caches=[cache])
        lazy = draft.draft_lazy(bonus, hidden, cache_b, depth, target_caches=[cache])
        assert lazy.dtype == mx.int32 and lazy.shape == (depth,)
        assert [int(t) for t in lazy.tolist()] == tokens[0]
        _assert_same_cache(cache_a, cache_b)
    with pytest.raises(ValueError, match="target cache"):
        draft.draft_lazy(bonus, hidden, draft.make_cache(), 2)


def test_hy4_dfly_device_round_greedy_equals_host(hy_tiny):
    off = hy_decode(hy_tiny)
    assert_served_exactly(off, hy_decode(hy_tiny, device_round=BOTH))
    assert_served_exactly(off, hy_decode(hy_tiny, device_round={"lazy_draft": True}))
    assert off[0][0] == hy._greedy(hy_tiny[2], hy.TOKENS[:13], 16)


def test_hy4_dfly_lazy_per_lane_equals_a_two_row_host_draft(hy_tiny):
    # The host chain engine verifies DFly2 at width one only (its target cache
    # has no segmented batch form; multi-lane DFly2 is the AST route), but the
    # host draft loops per row, so each lane's lazy draft is that row's.
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel

    _, _, model, head = hy_tiny
    caches, hiddens, bonuses = [], [], []
    model.record_layers = frozenset(head.index_layers)
    try:
        for prompt in (hy.TOKENS[:13], hy.TOKENS[4:17]):
            cache = model.make_cache()
            hidden = model.hidden(mx.array([prompt]), cache)
            caches.append(cache)
            hiddens.append(hidden)
            bonuses.append(int(mx.argmax(model.logits(hidden[:, -1]), -1).item()))
    finally:
        model.record_layers = frozenset()
    draft = Hy4DFlyDraftModel(model, head, num_draft=3, revision="test")
    host = [draft.make_cache(), draft.make_cache()]
    tokens, _ = draft.draft_distributions(
        bonuses, mx.concatenate(hiddens), draft.batch_caches(host), 3,
        [RequestRNG(1), RequestRNG(2)], [0.0, 0.0], target_caches=caches)
    for row in range(2):
        lazy_cache = draft.make_cache()
        lazy = draft.draft_lazy(bonuses[row], hiddens[row], lazy_cache, 3,
                                target_caches=[caches[row]])
        assert [int(t) for t in lazy.tolist()] == tokens[row]
        _assert_same_cache(host[row], lazy_cache)


def test_hy4_dfly_one_token_prompt_falls_back_like_the_host(hy_tiny):
    off = hy_decode(hy_tiny, prompts=([9],), steps=6, favour=False)
    on = hy_decode(hy_tiny, prompts=([9],), steps=6, favour=False, device_round=BOTH)
    assert on[0] == off[0] == [hy._greedy(hy_tiny[2], [9], 6)]


def test_hy4_dfly_host_only_rounds_decline_target_caches(hy_tiny):
    # A prompt-copy round would bypass the lazy draft; it keeps the host path.
    off = hy_decode(hy_tiny)
    on = hy_decode(hy_tiny, prompt_copy_round=True)
    assert on[0] == off[0]
    reasons = on[1][DECLINE_REASONS]
    assert reasons.get("target_caches", 0) > 0 and set(reasons) == {"target_caches"}


def test_hy4_dfly_tree_engine_declines_counted(hy_tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    options = dict(dynamic_singleton_tree=True, dynamic_tree_max_width=1,
                   target_tree_executor_provider=Hy4ChainTreeExecutor.create(),
                   tree_node_budget_by_lanes={1: 7}, ast_topology="drafter")
    off = hy_decode(hy_tiny, **options)
    options["target_tree_executor_provider"] = Hy4ChainTreeExecutor.create()
    on = hy_decode(hy_tiny, device_round=BOTH, **options)
    assert on[0] == off[0]
    stats = on[1]
    assert stats["external_device_round_rounds"] == 0
    assert stats["external_device_round_declines"] == sum(stats[DECLINE_REASONS].values())
    # The tree engine's ordinary fallbacks reach the runner as forced depths.
    assert set(stats[DECLINE_REASONS]) <= {"tree_topology", "ast", "target_tree_provider",
                                           "dynamic_tree", "forced_depth"}


def test_composed_hy4_dfly_declines_composition_target_caches(hy_tiny):
    # ComposedDraftModel forwards the backend's ``requires_target_caches`` and
    # ``draft_lazy_target_caches``, but its ``draft_lazy`` does not forward
    # ``target_caches``: the round declines at admission instead of failing.
    off = hy_decode(hy_tiny, compose=True)
    on = hy_decode(hy_tiny, compose=True, device_round=BOTH)
    assert on[0] == off[0]
    reasons = on[1][DECLINE_REASONS]
    assert set(reasons) == {"composition_target_caches"} and reasons["composition_target_caches"] > 0
    assert on[1]["external_device_round_rounds"] == 0
    assert on[1]["external_device_round_declines"] == reasons["composition_target_caches"]
    assert on[2] == off[2]


def test_untested_multi_lane_drafters_declare_width_one():
    # No test covers multi-lane device rounds of these drafters, so a wider
    # cohort keeps the host round (counted ``lazy_draft_width``).
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.drafters.inkling_mtp import CascadedMTPDraftModel
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

    for cls in (Hy4DFlyDraftModel, CascadedMTPDraftModel, PromptLookupDraftModel):
        assert cls.draft_lazy_width_one is True, cls.__name__
