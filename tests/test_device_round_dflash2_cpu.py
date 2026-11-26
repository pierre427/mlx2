"""device_round v2 slice 1a-1: DFlash2 lazy greedy drafts, CPU only.

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.7.
Pins: ``DFlash2DraftModel.draft_lazy`` returns the tokens of the unprocessed
``draft_distributions`` path for one lane and consumes the same RNG draws; the
engine with ``device_round {lazy_draft, decide}`` decodes exactly what the
option-off engine decodes, with one host read per round; sampled lanes, wider
cohorts and proposal composition decline with counted reasons.
"""

import copy

import mlx.core as mx
import pytest
from test_external_dflash2_cpu import generator, tiny

from mlx2.runtime.speculative_device_round import DECLINE_REASONS
from mlx2.runtime.speculative_sampling import RequestRNG

mx.set_default_device(mx.cpu)

BOTH = {"lazy_draft": True, "decide": True}
PROMPT = [3, 9, 14, 2, 27, 5, 11, 20, 8, 17, 6, 25]


def _decode(steps=12, sampling=None, prompts=(PROMPT,), **options):
    model, draft = tiny()
    batch = generator(model, draft, **options)
    try:
        uids = batch.insert([list(p) for p in prompts], max_tokens=[steps] * len(prompts),
                            sampling_configs=[sampling or {"sampling_temp": 0}] * len(prompts))
        got = {uid: [] for uid in uids}
        rngs = {}
        for _ in range(600):
            for uid, lane in batch.lanes.items():
                rngs[uid] = lane.rng.snapshot()  # last state before the lane retires
            _, responses = batch.next()
            for r in responses:
                if r.uid in got:
                    got[r.uid].append(r.token)
            if not batch.lanes:
                break
        stats = dict(batch.scheduler_stats)
        stats["_rng"] = rngs
        return [got[uid] for uid in uids], stats
    finally:
        batch.close()


def test_draft_lazy_equals_unprocessed_draft_distributions():
    _, draft = tiny()
    for depth in (1, 2, 3):
        cache_a, cache_b = draft.make_cache(), draft.make_cache()
        hidden = mx.random.normal((1, 5, draft.config.hidden_size * len(draft.config.target_layer_ids)),
                                  key=mx.random.key(depth))
        rng_a, rng_b = RequestRNG(7), RequestRNG(7)
        tokens, _laws = draft.draft_distributions(
            [4], hidden, draft.batch_caches([cache_a]), depth, [rng_a], [0.0])
        lazy = draft.draft_lazy(4, hidden, cache_b, depth, rng=rng_b)
        assert lazy.dtype == mx.int32 and lazy.shape == (depth,)
        assert [int(t) for t in lazy.tolist()] == tokens[0]
        assert rng_a.snapshot() == rng_b.snapshot()  # the same uniforms consumed


def test_draft_lazy_refuses_a_sampler_and_wide_batches():
    _, draft = tiny()
    hidden = mx.zeros((2, 3, draft.config.hidden_size * len(draft.config.target_layer_ids)))
    with pytest.raises(ValueError, match="one lane"):
        draft.draft_lazy(4, hidden, draft.make_cache(), 2)
    with pytest.raises(ValueError, match="greedy only"):
        draft.draft_lazy(4, hidden[:1], draft.make_cache(), 2, lambda logits: logits)


def test_engine_greedy_output_unchanged_with_one_read_per_round():
    off, off_stats = _decode()
    on, stats = _decode(device_round=BOTH)
    assert on == off
    # The lane RNG ends where the option-off engine leaves it (draft uniforms
    # plus the greedy verifier's schedule).
    assert list(stats["_rng"].values()) == list(off_stats["_rng"].values())
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_host_reads"] == rounds
    # ``decide`` also serves rounds outside the runner (the zero-depth final round).
    assert stats["external_device_round_decide_greedy_lanes"] >= rounds
    assert stats["external_device_round_declines"] == 0


def test_lazy_draft_alone_is_also_exact():
    off, _ = _decode()
    on, stats = _decode(device_round={"lazy_draft": True})
    assert on == off and stats["external_device_round_rounds"] > 0


def test_sampled_lane_declines_counted():
    sampling = {"sampling_temp": 0.7, "seed": 3}
    off, _ = _decode(sampling=sampling)
    # lazy_draft alone: a declined round is the host path, token for token.
    on, stats = _decode(sampling=sampling, device_round={"lazy_draft": True})
    assert on == off
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("lazy_draft_sampled_lane", 0) > 0
    # With decide, v1's device decision of host-drafted point-mass rounds
    # (here the zero-depth final round) keeps the law, not the tokens (section 8 answer 5).
    _, stats = _decode(sampling=sampling, device_round=BOTH)
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("lazy_draft_sampled_lane", 0) > 0


def test_wide_cohort_declines_counted():
    prompts = (PROMPT, list(reversed(PROMPT)))
    off, _ = _decode(prompts=prompts)
    on, stats = _decode(prompts=prompts, device_round=BOTH)
    assert on == off
    assert stats[DECLINE_REASONS].get("lazy_draft_width", 0) > 0


def test_a_scored_wrapper_without_a_lazy_composition_declines():
    from mlx2.runtime.speculative_device_round import DeviceRound

    owner = copy.copy(generator(*tiny()))
    owner.draft = type("Composed", (), {"requires_proposal_score_states": True})()
    assert DeviceRound._engine_reason(owner) == "proposal_composition"


def test_round_probe_sees_fewer_host_syncs_per_round():
    from mlx2.runtime import round_probe

    round_probe.uninstall()
    round_probe.reset()
    assert round_probe.install()
    try:
        _decode()
        off = round_probe.status()["routes"]
        round_probe.reset()
        _decode(device_round=BOTH)
        on = round_probe.status()["routes"]
    finally:
        round_probe.uninstall()
        round_probe.reset()
    host = off["external:DFlash2DraftModel:chain"]["per_round"]
    device = on["external:DFlash2DraftModel:device_round"]["per_round"]
    print("syncs/round host", host["syncs"], "device", device["syncs"])
    assert device["syncs"] < host["syncs"]


# -- slice 1a-2: stateless logits processors (presence penalty) ------------------

PROC = {"lazy_draft": True, "decide": True, "processors": True}


def _presence(prompt, penalty=1.5):
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=penalty, presence_context_size=20,
                                  penalty_generation_start=len(prompt))


def _decode_processed(steps=14, make=_presence, **options):
    model, draft = tiny()
    batch = generator(model, draft, **options)
    try:
        uid = batch.insert([list(PROMPT)], max_tokens=[steps],
                           sampling_configs=[{"sampling_temp": 0}],
                           logits_processors=[make(PROMPT)])[0]
        got, rng = [], None
        for _ in range(600):
            if uid in batch.lanes:
                rng = batch.lanes[uid].rng.snapshot()
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        return got, rng, dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_processed_draft_lazy_equals_processed_draft_distributions():
    _, draft = tiny()
    history = [5, 9, 9, 3, 12, 4]
    processors = _presence(history[:2], penalty=2.5)
    for depth in (1, 3):
        cache_a, cache_b = draft.make_cache(), draft.make_cache()
        hidden = mx.random.normal((1, 4, draft.config.hidden_size * len(draft.config.target_layer_ids)),
                                  key=mx.random.key(10 + depth))
        rng_a, rng_b = RequestRNG(5), RequestRNG(5)
        tokens, _laws = draft.draft_distributions(
            [9], hidden, draft.batch_caches([cache_a]), depth, [rng_a], [0.0],
            logits_processors=[processors], processor_histories=[history])
        lazy = draft.draft_lazy(9, hidden, cache_b, depth, rng=rng_b,
                                logits_processors=processors, processor_history=history)
        assert [int(t) for t in lazy.tolist()] == tokens[0]
        assert rng_a.snapshot() == rng_b.snapshot()


def test_presence_penalty_lane_keeps_the_device_round_exactly():
    off, off_rng, _ = _decode_processed()
    on, on_rng, stats = _decode_processed(device_round=PROC)
    assert on == off and on_rng == off_rng
    assert stats["external_device_round_rounds"] > 0
    assert stats["external_device_round_host_reads"] == stats["external_device_round_rounds"]
    assert stats["external_device_round_declines"] == 0


def test_processors_need_the_switch():
    off, _, _ = _decode_processed()
    on, _, stats = _decode_processed(device_round=BOTH)
    assert on == off
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("logits_processors", 0) > 0


def test_stateful_or_undeclared_processors_still_decline():
    from mlx2.runtime.sample_utils import make_logits_processors

    def frequency(prompt):
        return make_logits_processors(frequency_penalty=0.7, penalty_generation_start=len(prompt))

    off, _, _ = _decode_processed(make=frequency)
    on, _, stats = _decode_processed(make=frequency, device_round=PROC)
    assert on == off
    assert stats["external_device_round_rounds"] == 0
    assert stats[DECLINE_REASONS].get("logits_processors", 0) > 0


def test_host_redo_restores_and_runs_the_round_on_the_host(monkeypatch):
    from mlx2.runtime.drafters import dflash2

    off, off_rng, _ = _decode_processed()
    real = dflash2.DFlash2DraftModel.draft_lazy
    calls = []

    def flaky(self, *args, **kwargs):
        tokens = real(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == 2:  # the second round asks for a host redo
            return mx.full(tokens.shape, dflash2.DRAFT_LAZY_HOST_REDO, dtype=mx.int32)
        return tokens

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", flaky)
    on, on_rng, stats = _decode_processed(device_round=PROC)
    assert on == off and on_rng == off_rng
    assert stats[DECLINE_REASONS].get("host_redo", 0) == 1
    assert stats["external_device_round_rounds"] > 0


def test_round_probe_presence_lane_syncs():
    from mlx2.runtime import round_probe

    round_probe.uninstall()
    round_probe.reset()
    assert round_probe.install()
    try:
        _decode_processed()
        host = round_probe.status()["routes"]["external:DFlash2DraftModel:chain"]["per_round"]
        round_probe.reset()
        _decode_processed(device_round=PROC)
        device = round_probe.status()["routes"]["external:DFlash2DraftModel:device_round"]["per_round"]
    finally:
        round_probe.uninstall()
        round_probe.reset()
    print("presence lane syncs/round host", host["syncs"], "device", device["syncs"])
    assert device["syncs"] < host["syncs"]


# -- slice 1a-1b: proposal composition (lazy external rows) ----------------------

REPEATS = [3, 9, 14, 2, 27, 5, 3, 9, 14, 2, 27, 5, 3, 9, 14]


def _decode_composed(prompt=PROMPT, steps=14, processors=None, policy=None, **options):
    from mlx2.adapters.external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    model, draft = tiny()
    composed = ComposedDraftModel(draft, policy or dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION))
    batch = generator(model, composed, **options)
    try:
        uid = batch.insert([list(prompt)], max_tokens=[steps],
                           sampling_configs=[{"sampling_temp": 0}],
                           logits_processors=[processors(prompt)] if processors else None)[0]
        got, state = [], None
        for _ in range(600):
            if uid in batch.lanes:
                lane = batch.lanes[uid]
                state = (lane.rng.snapshot(), copy.deepcopy(lane.proposal_score_counts),
                         copy.deepcopy(lane.proposal_trust_states),
                         copy.deepcopy(getattr(lane, "proposal_composition_counts", {})))
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        return got, state, dict(batch.scheduler_stats)
    finally:
        batch.close()


def test_composed_external_rows_keep_the_device_round_exactly():
    off, off_state, _ = _decode_composed()
    on, on_state, stats = _decode_composed(device_round=BOTH)
    assert on == off
    assert on_state == off_state  # RNG, lagged scores, trust and source counters
    assert stats["external_device_round_rounds"] > 0
    assert stats.get("external_composed_external_rounds", 0) > 0


def test_composed_copy_selection_declines_to_the_host_path():
    off, off_state, _ = _decode_composed(prompt=REPEATS)
    on, on_state, stats = _decode_composed(prompt=REPEATS, device_round=BOTH)
    assert on == off and on_state == off_state
    reasons = stats[DECLINE_REASONS]
    assert reasons.get("composition_selected_prompt_lookup", 0) > 0
    assert stats.get("external_composed_prompt_lookup_rounds", 0) > 0


def test_composed_presence_lane_with_processors_switch():
    off, off_state, _ = _decode_composed(processors=_presence)
    on, on_state, stats = _decode_composed(processors=_presence, device_round=PROC)
    assert on == off and on_state == off_state
    assert stats["external_device_round_rounds"] > 0


def test_composed_lazy_draft_refuses_a_backend_without_one():
    from mlx2.runtime.proposal_composition import ComposedDraftModel, LazyDecline

    class Backend:
        proposal_distribution = "deterministic_point_mass"
        draft_lazy = None

    wrapper = object.__new__(ComposedDraftModel)
    wrapper.backend = Backend()
    with pytest.raises(LazyDecline, match="without_lazy_draft"):
        wrapper.draft_lazy(1, None, None, 2)


def test_restored_composed_rounds_are_not_counted_twice(monkeypatch):
    from mlx2.adapters.external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION
    from mlx2.runtime.drafters import dflash2
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    def run(**options):
        model, draft = tiny()
        composed = ComposedDraftModel(draft, dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION))
        batch = generator(model, composed, **options)
        try:
            uid = batch.insert([list(PROMPT)], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
            got = []
            for _ in range(600):
                _, responses = batch.next()
                got += [r.token for r in responses if r.uid == uid]
                if not batch.lanes:
                    break
            return got, dict(composed.composition_stats), dict(batch.scheduler_stats)
        finally:
            batch.close()

    off, off_counts, _ = run()
    real = dflash2.DFlash2DraftModel.draft_lazy
    calls = []

    def flaky(self, *args, **kwargs):
        tokens = real(self, *args, **kwargs)
        calls.append(1)
        return (mx.full(tokens.shape, dflash2.DRAFT_LAZY_HOST_REDO, dtype=mx.int32)
                if len(calls) == 2 else tokens)

    monkeypatch.setattr(dflash2.DFlash2DraftModel, "draft_lazy", flaky)
    on, on_counts, stats = run(device_round=BOTH)
    assert on == off and on_counts == off_counts
    assert stats[DECLINE_REASONS].get("host_redo", 0) == 1


def test_composition_declines_without_lazy_draft_or_with_long_copies():
    from mlx2.adapters.external_draft_policy import DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION

    _, _, stats = _decode_composed(device_round={"decide": True})
    assert stats[DECLINE_REASONS].get("proposal_composition", 0) == 0  # decide-only never runs the runner
    policy = {**DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION, "prompt_lookup_max_span": 24}
    try:
        on, _, stats = _decode_composed(policy=policy, device_round=BOTH)
    except ValueError as error:  # the tiny DFlash2 refuses spans past its block
        pytest.skip(f"backend refuses prompt_lookup_max_span: {error}")
    assert stats[DECLINE_REASONS].get("composition_long_copies", 0) > 0



def test_prompt_copy_rounds_never_bypass_composition():
    """A copy-only device runtime on a composed route keeps composition's host path."""
    off, off_state, _ = _decode_composed(prompt=REPEATS)
    on, on_state, stats = _decode_composed(prompt=REPEATS, prompt_copy_round={"max_tokens": 12})
    assert on == off and on_state == off_state
    assert stats[DECLINE_REASONS].get("proposal_composition", 0) > 0
    assert stats.get("external_composed_prompt_copy_rounds", 0) == 0


def test_submit_drafts_is_exact_and_needs_lazy_draft():
    from mlx2.runtime.speculative_device_round import DeviceRoundPolicy

    with pytest.raises(ValueError, match="submit_drafts requires lazy_draft"):
        DeviceRoundPolicy.from_value({"decide": True, "submit_drafts": True})
    off, off_rng, _ = _decode_processed()
    on, on_rng, stats = _decode_processed(device_round={**PROC, "submit_drafts": True})
    assert on == off and on_rng == off_rng
    assert stats["external_device_round_rounds"] > 0
