"""Tree route at B2/B4: parked lanes run one batched, pipelined plain step.

h2h r2c (2026-10-21, Qwen3.6-35B-A3B + DFlash2, M5): a parked B4 cohort's
plain step cost 24-30 ms against the ordinary route's ~10 ms B4 step.  The
scheduler rotates its service order every poll, and the parked pipeline
compared the rotated cohort with the dispatched round's lane order, so every
poll drained the pipeline and restarted it: one synchronous batched step per
two polls.  Each round also rebuilt its hybrid owner, re-joining every
recurrent state, and each deferred-context cap stopped the pipeline for a
depth-zero external flush round.

Sessions that parked or re-probed at different times also split every
decode phase into a speculative round beside an unpipelined plain step of
their peers; at width two or more the cohort now re-probes together and
adopts one aggregate park/speculate decision.

A tiny hybrid Qwen3.6 MoE target (GDN + attention) with a DFlash2 drafter
runs the tree route with lanes at different offsets.  The lanes park; the
parked cohort must then stay on one pipelined batched plain step (owner kept,
context flushed inline), and every lane's greedy output must equal plain
greedy decoding of its own prompt.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_qwen38_dflash2_cpu import ordinary_greedy
from test_tree_route_parked_plain_cpu import MAX_TOKENS, _pair, _run, _settings

# Different lengths: every lane decodes at its own offset (left-padded
# batched views, per-lane recurrent state).
PROMPTS = (
    [3, 9, 27, 81, 15, 44, 2],
    [5, 6, 70, 71, 72],
    [9, 8, 7],
    [1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17],
)
WIDE = {"dynamic_tree_max_width": 4, "tree_node_budget_by_lanes": {1: 15, 2: 7, 3: 4, 4: 3}}


@pytest.mark.parametrize("width", [2, 4])
def test_parked_cohort_runs_one_pipelined_batched_plain_step(monkeypatch, width):
    prompts = PROMPTS[:width]
    target, _draft = _pair()
    expected = [ordinary_greedy(target, prompt, MAX_TOKENS) for prompt in prompts]
    outputs, finals, batch, _, _ = _run(
        monkeypatch,
        handoff=_settings(reprobe_tokens=400, reprobe_max_tokens=400),
        speculative_ns=lambda _i: 100_000,
        prompts=prompts,
        **WIDE,
    )
    assert outputs == expected
    stats = batch.scheduler_stats
    for final in finals:
        reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
        assert reversible["parks"] >= 1
        final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))
    pipelined = stats.get("external_parked_pipelined_rounds", 0)
    assert pipelined >= MAX_TOKENS // 2
    # One pipeline for the parked cohort, not a restart per poll.
    assert stats.get("external_parked_pipeline_starts", 0) <= 3
    assert stats.get("external_parked_pipeline_drains", 0) <= 2
    assert stats.get("external_parked_pipeline_mismatches", 0) == 0
    # Every parked round was one batched target forward over the cohort.
    assert stats["external_cohort_target_batched_parked_plain_rounds"] >= pipelined
    # The cohort's hybrid owner (and its joined views) is kept across rounds;
    # it is rebuilt only when the cohort changes (a lane parks or finishes).
    assert stats.get("external_parked_owner_reuses", 0) >= pipelined - 2 * width
    assert batch._parked_inflight is None


def test_parked_cohort_flushes_deferred_context_inline(monkeypatch):
    """The deferred draft context is appended inside the pipeline at its cap."""
    prompts = PROMPTS[:2]
    target, _draft = _pair()
    expected = [ordinary_greedy(target, prompt, MAX_TOKENS) for prompt in prompts]
    outputs, finals, batch, _, _ = _run(
        monkeypatch,
        handoff=_settings(
            reprobe_tokens=400, reprobe_max_tokens=400, deferred_context_cap=8
        ),
        speculative_ns=lambda _i: 100_000,
        prompts=prompts,
        **WIDE,
    )
    assert outputs == expected
    stats = batch.scheduler_stats
    assert stats.get("external_parked_context_flushes", 0) >= 4
    assert stats.get("external_cost_parked_flush_rounds", 0) == 0
    assert stats.get("external_parked_pipeline_starts", 0) <= 3
    for final in finals:
        final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))


def test_parked_lane_reprobes_after_inline_flushes(monkeypatch):
    """A re-probe after inline flushes consumes a paired draft plane."""
    target, _draft = _pair()
    expected = ordinary_greedy(target, PROMPTS[0], MAX_TOKENS)
    price = lambda i: 100_000 if i < 2 else 10
    tokens, final, batch, reengaged_at, _ = _run(
        monkeypatch,
        handoff=_settings(
            reprobe_tokens=20, reprobe_max_tokens=20, deferred_context_cap=8
        ),
        speculative_ns=price,
    )
    assert tokens == expected
    stats = batch.scheduler_stats
    assert stats.get("external_parked_context_flushes", 0) >= 2
    assert stats.get("external_cost_parked_flush_rounds", 0) == 0
    reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
    assert reversible["reengagements"] >= 1
    assert reengaged_at is not None
    final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))


@pytest.mark.parametrize("width", [2, 4])
def test_losing_cohort_parks_and_reprobes_together(monkeypatch, width):
    """At width >= 2 the cohort parks, re-probes and re-parks as one.

    A lone re-probing (or re-engaged) session splits every decode phase into
    its one-lane speculative round beside an unpipelined plain step of its
    parked peers.  Aligned re-probes and the aggregate cohort decision keep
    the whole cohort on one route.
    """
    prompts = PROMPTS[:width]
    target, _draft = _pair()
    expected = [ordinary_greedy(target, prompt, MAX_TOKENS) for prompt in prompts]
    outputs, finals, batch, _, _ = _run(
        monkeypatch,
        handoff=_settings(reprobe_tokens=6, reprobe_max_tokens=12),
        speculative_ns=lambda _i: 100_000,
        prompts=prompts,
        **WIDE,
    )
    assert outputs == expected
    stats = batch.scheduler_stats
    assert stats.get("external_decode_parked_split_phases", 0) <= 1
    receipts = [
        final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
        for final in finals
    ]
    assert all(receipt["parks"] == 1 for receipt in receipts)
    assert all(receipt["reengagements"] == 0 for receipt in receipts)
    # Every session re-probed in the same rounds as its peers.
    assert len({receipt["reprobes"] for receipt in receipts}) == 1
    assert receipts[0]["reprobes"] >= 2
    for final in finals:
        final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))


def test_cohort_decision_adoption_is_exact_reversible_state():
    from mlx2.runtime.cost_per_commit_handoff import (
        CostPerCommitHandoffSettings,
        CostPerCommitLaneState,
    )

    settings = CostPerCommitHandoffSettings.from_value(
        _settings(reprobe_tokens=4, reprobe_max_tokens=32)
    )

    def state(uid):
        lane = CostPerCommitLaneState(settings, lane_uid=uid, owner_scope="s")
        for _ in range(settings.ordinary_samples):
            lane.observe(kind="ordinary", elapsed_ns=100, committed_tokens=1,
                         execution_width=2, lane_uid=uid, owner_scope="s")
        return lane

    speculating, parked = state(1), state(2)
    assert speculating.ordinary_baseline() == 100.0
    assert parked.begin_cohort_reprobe() is None  # not parked
    event = parked.adopt_cohort_decision(
        park=True, ordinary_cost=100.0, speculative_cost=300.0, reprobe_after=16
    )
    assert event["event"] == "park" and event["reason"] == "cohort_aggregate"
    assert parked.speculation_paused and parked.reprobe_after_tokens == 16
    # A parked session joins a peer's re-probe once, then probes normally.
    assert parked.begin_cohort_reprobe()["reason"] == "cohort_aligned"
    assert not parked.speculation_paused
    assert parked.begin_cohort_reprobe() is None
    # A lost cohort probe re-parks it as a failed re-probe with backoff.
    event = parked.adopt_cohort_decision(
        park=True, ordinary_cost=100.0, speculative_cost=300.0
    )
    assert event["event"] == "repark" and parked.reprobe_after_tokens == 32
    receipt = parked.receipt()["reversible"]
    assert receipt["failed_reprobes"] == 1 and receipt["cohort_reprobes"] == 1
    # A winning cohort re-engages a parked session and resets its backoff.
    event = parked.adopt_cohort_decision(
        park=False, ordinary_cost=100.0, speculative_cost=50.0
    )
    assert event["event"] == "reengage" and not parked.speculation_paused
    assert parked.reprobe_after_tokens == 4
    # Already on the cohort's route: nothing to adopt.
    assert speculating.adopt_cohort_decision(
        park=False, ordinary_cost=100.0, speculative_cost=50.0
    ) is None
    assert parked.receipt()["reversible"]["cohort_decisions"] == 3


def test_width_where_the_tree_lost_parks_the_next_cohort_directly(monkeypatch):
    """A second cohort at a width that already lost starts parked.

    The first cohort measures and parks at width 4; the generator records
    that the tree lost at that width.  The next four requests park as soon
    as all four decode (no ordinary probes, settle or speculative samples at
    that width), build their baselines from the pipelined plain steps, and
    still re-probe with the cohort later.  Output stays exact.
    """
    import test_tree_route_parked_plain_cpu as fixture
    from mlx2.runtime import external_speculative as es
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    from test_dflash_tree_bridge import _install_reference_cohort

    target, draft = _pair()
    clock = {"now": 1}

    def timed_round(self, cohort, *, forced_depth=None):
        result = fixture._ROUND(self, cohort, forced_depth=forced_depth)
        parked = forced_depth == 0 or all(es._speculation_parked(l) for l in cohort)
        clock["now"] += 100 if parked else 100_000
        return result

    def timed_read(self, inflight, rows, following, **kwargs):
        clock["now"] += 100
        return fixture._READ(self, inflight, rows, following, **kwargs)

    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", timed_round)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_read_parked", timed_read)
    batch = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="tree-parked-plain-cpu",
        num_draft=3, prefill_step_size=4, completion_batch_size=4,
        dynamic_singleton_tree=True,
        target_tree_executor_provider=Qwen38TargetTreeExecutor.create(),
        cost_per_commit_handoff=_settings(reprobe_tokens=12, reprobe_max_tokens=48),
        **WIDE,
    )
    _install_reference_cohort(batch, monkeypatch, [])

    def wave():
        uids = batch.insert(
            [list(prompt) for prompt in PROMPTS],
            max_tokens=[MAX_TOKENS] * len(PROMPTS),
            sampling_configs=[{"sampling_temp": 0.0}] * len(PROMPTS),
        )
        outputs = {uid: [] for uid in uids}
        finals = {}
        tree_rounds = batch.scheduler_stats.get("external_auto_tree_rounds", 0)
        for _ in range(1200):
            _, responses = batch.next()
            for response in responses:
                outputs[response.uid].append(response.token)
                if response.finish_reason:
                    finals[response.uid] = response
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        rounds = batch.scheduler_stats.get("external_auto_tree_rounds", 0) - tree_rounds
        return [outputs[uid] for uid in uids], [finals[uid] for uid in uids], rounds

    expected = [ordinary_greedy(target, prompt, MAX_TOKENS) for prompt in PROMPTS]
    first, _finals, first_tree_rounds = wave()
    assert first == expected
    assert batch._width_park_priors.get(4) is not None
    assert batch.scheduler_stats.get("external_cost_width_prior_parks", 0) == 0
    second, finals, second_tree_rounds = wave()
    assert second == expected
    stats = batch.scheduler_stats
    assert stats.get("external_cost_width_prior_parks", 0) >= 1
    for final in finals:
        events = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]["events"]
        reasons = [event.get("reason") for event in events if event["event"] == "park"]
        assert reasons and reasons[0] in ("width_prior", "cost_per_committed_token", "cohort_aggregate")
        final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))
    assert any(
        event.get("reason") == "width_prior"
        for final in finals
        for event in final.speculative_receipt["cost_per_commit_handoff"]["reversible"]["events"]
    )
    # The second cohort skipped (most of) the speculative measurement phase.
    assert second_tree_rounds < first_tree_rounds


def _observe(state, kind, elapsed, committed=1):
    return state.observe(
        kind=kind, elapsed_ns=elapsed, committed_tokens=committed,
        execution_width=1, lane_uid=state.lane_uid, owner_scope="s",
    )


def _parked_then_reengaged():
    from mlx2.runtime.cost_per_commit_handoff import (
        CostPerCommitHandoffSettings,
        CostPerCommitLaneState,
    )

    settings = CostPerCommitHandoffSettings.from_value(
        _settings(reprobe_tokens=6, reprobe_max_tokens=48)
    )
    state = CostPerCommitLaneState(settings, lane_uid=1, owner_scope="s")
    for _ in range(2):
        _observe(state, "ordinary", 100)
    _observe(state, "speculative", 1000)
    assert _observe(state, "speculative", 1000)["event"] == "park"
    events = [_observe(state, "ordinary", 100) for _ in range(6)]
    assert events[-1]["event"] == "reprobe"
    assert _observe(state, "speculative", 10) is None  # catch-up round
    _observe(state, "speculative", 10)
    assert _observe(state, "speculative", 10)["event"] == "reengage"
    assert state.reprobe_after_tokens == 6
    return state


def test_short_reengagement_parks_with_doubled_backoff():
    """A re-engagement that parks before paying back backs off again."""
    state = _parked_then_reengaged()
    event = _observe(state, "speculative", 1000)
    assert event["event"] == "park" and event["flap_damped"] is True
    assert state.reprobe_after_tokens == 12
    assert state.receipt()["reversible"]["flap_damped_parks"] == 1


def test_long_reengagement_keeps_the_reset_backoff():
    state = _parked_then_reengaged()
    for _ in range(6):
        assert _observe(state, "speculative", 10) is None
    # The stale baseline owes one fresh ordinary round before a decision.
    assert _observe(state, "ordinary", 100) is None
    event = None
    for _ in range(8):
        event = _observe(state, "speculative", 10_000)
        if event is not None:
            break
    assert event["event"] == "park" and "flap_damped" not in event
    assert state.reprobe_after_tokens == 6
