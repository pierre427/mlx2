"""Device-decided speculative round and prompt-copy rounds, CPU only.

``ExternalDraftBatchGenerator(device_round=..., prompt_copy_round=...)``
(runtime/speculative_device_round.py, runtime/prompt_copy_round.py; MTPLX
ingest 2026-11-23).  Pins: both options default off with no receipt change;
greedy output equal to the option-off engine and to ordinary decode at DSpark
depths 3, 4 and 7 (rings wrapped) with every round's proposals unchanged;
one host read per lazy+decide round; the sampled laws (point mass, ratio,
residual) by direct sampling; filtered requests decline without ``nucleus``
and engage with it; the prompt-copy backoff rule.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_inkling2_dspark_cpu import _cpu, pair  # noqa: F401
from test_inkling2_dspark_predraft_cpu import (  # noqa: F401
    _ordinary,
    _prompt,
    _run,
    agreeing,
)

from mlx2.runtime.drafters.inkling2_dspark_external import Inkling2DSparkExternal
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, _block_row
from mlx2.runtime.prompt_copy_round import (
    LaneCopyState,
    NgramIndex,
    PromptCopyPolicy,
    PromptCopyRuntime,
)
from mlx2.runtime.speculative_device_round import (
    COUNTERS,
    DECLINE_REASONS,
    DeviceRound,
    DeviceRoundPolicy,
)
from mlx2.runtime.speculative_sampling import RequestRNG

LAZY = {"lazy_draft": True}
DECIDE = {"decide": True}
BOTH = {"lazy_draft": True, "decide": True}


def _engine(target, draft, num_draft, **options):
    return ExternalDraftBatchGenerator(
        target, draft_model=Inkling2DSparkExternal(
            draft, num_draft=min(num_draft, 7), engine_depth=num_draft if num_draft > 7 else None),
        binding="test", num_draft=num_draft, prefill_step_size=16, **options,
    )


def _record(batch):
    """Every committed round's proposal rows, in order (both paths commit)."""
    seen = []
    commit = batch._commit

    def wrapped(cohort, decisions, features, *, blocks, **kwargs):
        seen.append([None if b is None else list(_block_row(b, None)[0]) for b in blocks])
        return commit(cohort, decisions, features, blocks=blocks, **kwargs)

    batch._commit = wrapped
    return seen


def _decode(target, draft, prompt, num_draft, steps, sampling=None, **options):
    batch = _engine(target, draft, num_draft, **options)
    try:
        proposals = _record(batch)
        uid = batch.insert([prompt], max_tokens=[steps],
                           sampling_configs=[sampling or {"sampling_temp": 0}])[0]
        got = _run(batch, uid)
        return got, proposals, dict(batch.scheduler_stats)
    finally:
        batch.close()


# -- default off ---------------------------------------------------------------
def test_options_default_off_and_validate(pair):
    target, draft, _ = pair
    batch = _engine(target, draft, 3)
    try:
        assert batch.device_round is None
        assert not any(key in batch.scheduler_stats for key in COUNTERS)
        assert DECLINE_REASONS not in batch.scheduler_stats
        assert not any(key.startswith("external_prompt_copy") for key in batch.scheduler_stats)
    finally:
        batch.close()
    assert DeviceRoundPolicy.from_value(None) is None
    assert DeviceRoundPolicy.from_value(False) is None
    assert DeviceRoundPolicy.from_value({}) is None
    assert DeviceRoundPolicy.from_value({"decide": False}) is None
    assert PromptCopyPolicy.from_value(None) is None
    assert PromptCopyPolicy.from_value({"enabled": False}) is None
    default = PromptCopyPolicy.from_value(True)
    assert (default.ngram_min, default.max_tokens, default.ema_floor) == (6, 24, 0.35)
    for bad in ({"decide": 1}, {"sampled_drafts": True}, {"nucleus": True}, {"other": True}, []):
        with pytest.raises(ValueError):
            DeviceRoundPolicy.from_value(bad)
    for bad in ({"ngram_min": 1}, {"max_tokens": 0}, {"ema_floor": 2}, {"x": 1}, 3):
        with pytest.raises(ValueError):
            PromptCopyPolicy.from_value(bad)


def test_adapter_policy_keys_are_generic_and_validated():
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin

    host = SimpleNamespace()
    parse = ExternalDraftAdapterMixin._parse_external_policy
    assert parse(host, {"draft_model": "x", "device_round": BOTH,
                        "prompt_copy_round": True}, family="T")
    with pytest.raises(ValueError):
        parse(host, {"draft_model": "x", "device_round": {"sampled_drafts": True}}, family="T")
    with pytest.raises(ValueError):
        parse(host, {"draft_model": "x", "prompt_copy_round": {"ngram_min": 0}}, family="T")


def test_inkling2_prompt_copy_raises_the_num_draft_ceiling_only_when_set():
    from mlx2.adapters.inkling2 import Inkling2Adapter

    record = {"args": SimpleNamespace(block_size=8)}
    off, on = Inkling2Adapter.__new__(Inkling2Adapter), Inkling2Adapter.__new__(Inkling2Adapter)
    off.external_policy = {"draft_model": "embedded_dspark"}
    on.external_policy = {"draft_model": "embedded_dspark", "prompt_copy_round": {"max_tokens": 24}}
    assert not off._prompt_copy_long() and on._prompt_copy_long()
    assert off._num_draft_ceiling(record) == 7
    assert on._num_draft_ceiling(record) == 24


# -- greedy exactness, depths 3 / 4 / 7 -------------------------------------------
@pytest.mark.parametrize("num_draft", [3, 4, 7])
@pytest.mark.parametrize("weights", ["random", "agreeing"])
@pytest.mark.parametrize("options", [LAZY, DECIDE, BOTH], ids=["lazy", "decide", "both"])
def test_greedy_equals_off_and_ordinary_with_rings_wrapped(pair, agreeing, weights, num_draft,
                                                          options):
    target, draft = pair[:2] if weights == "random" else agreeing
    prompt = _prompt(101)
    steps = 32
    expect = _ordinary(target, prompt, steps)
    off, off_props, off_stats = _decode(target, draft, prompt, num_draft, steps)
    on, on_props, stats = _decode(target, draft, prompt, num_draft, steps, device_round=options)
    assert off == expect and on == expect
    assert on_props == off_props
    for key in ("accepted_proposals", "proposed_tokens", "external_rounds"):
        assert stats[key] == off_stats[key]
    assert stats["external_device_round_declines"] == 0
    rounds = stats["external_device_round_rounds"]
    if options.get("lazy_draft"):
        assert rounds > 0 and stats["external_device_round_lazy_draft_lanes"] == rounds
        # One host read per round: drafts, decision and verify together.
        assert stats["external_device_round_host_reads"] == rounds
    else:
        assert rounds == 0
    if options.get("decide"):
        assert stats["external_device_round_decide_greedy_lanes"] > 0
    else:
        assert stats["external_device_round_decide_greedy_lanes"] == 0
    if weights == "agreeing":
        assert stats["accepted_proposals"] > 0


def test_greedy_rng_schedule_matches_host_verifier(agreeing):
    """The lane RNG advances exactly as ``verify_greedy_proposals`` does."""
    target, draft = agreeing
    prompt = _prompt(7)
    draws = {}
    for name, options in (("off", {}), ("on", {"device_round": BOTH})):
        batch = _engine(target, draft, 4, **options)
        try:
            uid = batch.insert([prompt], max_tokens=[20], sampling_configs=[{"sampling_temp": 0}])[0]
            for _ in range(200):
                batch.next()
                if uid in batch.lanes:
                    draws[name] = batch.lanes[uid].rng.draws
                if not batch.lanes:
                    break
        finally:
            batch.close()
    assert draws["on"] == draws["off"] > 0


# -- sampled ---------------------------------------------------------------------
@pytest.mark.parametrize("options", [BOTH, {**BOTH, "sampled_drafts": True}, DECIDE],
                         ids=["point_mass", "ratio", "decide_only"])
def test_sampled_rounds_engage_and_finish(agreeing, options):
    target, draft = agreeing
    sampling = {"sampling_temp": 1.0}
    got, _props, stats = _decode(target, draft, _prompt(5), 4, 24, sampling, device_round=options)
    assert len(got) == 24
    # Sampled drafts with exact_verification unset take the block rule
    # (DEFAULT ON, Pierre 2026-11-27).
    key = ("external_device_round_decide_block_lanes" if options.get("sampled_drafts")
           else "external_device_round_decide_point_mass_lanes")
    assert stats[key] > 0
    assert stats["external_device_round_declines"] == 0


def test_filtered_sampling_declines_without_nucleus_and_engages_with_it(agreeing):
    target, draft = agreeing
    sampling = {"sampling_temp": 0.8, "top_p": 0.9, "top_k": 5}
    _, _, stats = _decode(target, draft, _prompt(5), 4, 12, sampling, device_round=BOTH)
    assert stats[DECLINE_REASONS].get("sampling_filters", 0) > 0
    assert stats["external_device_round_nucleus_lanes"] == 0
    got, _, stats = _decode(target, draft, _prompt(5), 4, 12, sampling,
                            device_round={**BOTH, "nucleus": True})
    assert len(got) == 12 and stats["external_device_round_nucleus_lanes"] > 0
    got, _, stats = _decode(target, draft, _prompt(5), 4, 12, {"sampling_temp": 0.8, "top_p": 0.9},
                            device_round={**BOTH, "nucleus": True, "sampled_drafts": True})
    assert len(got) == 12 and stats["external_device_round_nucleus_lanes"] > 0


def test_incapable_drafter_declines_counted(pair):
    target, draft, _ = pair

    class Plain(Inkling2DSparkExternal):
        draft_lazy = None

    batch = ExternalDraftBatchGenerator(target, draft_model=Plain(draft, num_draft=3),
                                        binding="test", num_draft=3, prefill_step_size=16,
                                        device_round=LAZY)
    try:
        uid = batch.insert([_prompt(3)], max_tokens=[8], sampling_configs=[{"sampling_temp": 0}])[0]
        assert len(_run(batch, uid)) == 8
        reasons = batch.scheduler_stats[DECLINE_REASONS]
        assert reasons.get("drafter_without_lazy_draft", 0) > 0
        assert batch.scheduler_stats["external_device_round_rounds"] == 0
    finally:
        batch.close()


# -- the sampled laws, by direct sampling ----------------------------------------------
def _law_plan(policy, logp, drafts, seed, q=None, temp=1.0):
    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value(policy)
    lane = SimpleNamespace(sampling={"sampling_temp": temp, "emit_logprobs": False},
                           rng=RequestRNG(0))
    row = SimpleNamespace(lane=lane, count=len(drafts), drafts=mx.array(drafts, dtype=mx.int32),
                          q=q, ratio=q is not None, seed=seed, host=None)
    plan = runtime._lane_plan(SimpleNamespace(mx=mx), row, mx.array(logp))
    out = [int(v) for v in np.asarray(plan.packet).tolist()]
    return out[0], out[2]


def test_point_mass_device_law_is_the_target_law():
    """First emitted token ~ p_0; accepted with probability p_0(draft)."""
    p0 = np.array([0.5, 0.2, 0.2, 0.1])
    p1 = np.array([0.1, 0.1, 0.1, 0.7])
    logits = np.log(np.stack([p0, p1])).astype(np.float32)
    n, first, accepts = 3000, np.zeros(4), 0
    for seed in range(n):
        accepted, last = _law_plan(BOTH, logits, [0], seed)
        token = 0 if accepted else last
        first[token] += 1
        accepts += accepted
    assert np.all(np.abs(first / n - p0) < 4 * np.sqrt(p0 * (1 - p0) / n) + 1e-3)
    assert abs(accepts / n - p0[0]) < 4 * np.sqrt(0.25 / n)


def test_ratio_device_law_is_the_target_law():
    """Sampled drafts from q, min(1, p/q) acceptance, residual resample:
    first emitted token ~ p_0 and acceptance rate sum(min(p, q))."""
    p0 = np.array([0.5, 0.2, 0.2, 0.1])
    q0 = np.array([0.1, 0.6, 0.2, 0.1])
    bonus = np.array([0.25, 0.25, 0.25, 0.25])
    logits = np.log(np.stack([p0, bonus])).astype(np.float32)
    logq = mx.array(np.log(q0)[None].astype(np.float32))
    n, first, accepts = 4000, np.zeros(4), 0
    for seed in range(n):
        keys = mx.random.split(mx.random.key(seed), 4)
        draft = int(mx.random.categorical(logq[0], key=keys[0]).item())
        accepted, last = _law_plan({**BOTH, "sampled_drafts": True}, logits, [draft], seed, q=logq)
        first[draft if accepted else last] += 1
        accepts += accepted
    assert np.all(np.abs(first / n - p0) < 4 * np.sqrt(p0 * (1 - p0) / n) + 1e-3)
    expect = np.minimum(p0, q0).sum()
    assert abs(accepts / n - expect) < 4 * np.sqrt(expect * (1 - expect) / n)


# -- prompt copy ----------------------------------------------------------------------
def test_ngram_index_finds_the_longest_backward_match():
    corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 10, 11]
    index = NgramIndex(3, 6)
    index.sync(corpus)
    index._corpus = corpus
    context = corpus + [9, 2, 3, 4, 5, 6]  # the prompt, then the output so far
    pos, ext = index.find(lambda i: context[i], len(context), max_pos=len(corpus))
    # "4 5 6" occurs twice; both extend back over "2 3" (ext 2); the most
    # recent wins ties.
    assert (pos, ext) == (15, 2)
    assert corpus[pos] == 10


@pytest.mark.parametrize("options", [DECIDE, BOTH])
def test_verify_trace_covers_device_decisions(pair, monkeypatch, tmp_path, options):  # noqa: F811
    import json

    from mlx2.runtime import external_speculative

    target, draft, _ = pair
    trace = tmp_path / "verify.jsonl"
    with trace.open("w") as handle:
        monkeypatch.setattr(external_speculative, "_VERIFY_TRACE_PATH", str(trace))
        monkeypatch.setattr(external_speculative, "_VERIFY_TRACE_STATE",
                            {"forward": 0, "handle": handle})
        _, rounds, stats = _decode(target, draft, _prompt(12, length=8), 3, 4, device_round=options)
    assert stats["external_device_round_decide_rounds"] > 0
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [event["inputs"][1:] for event in events if len(event["inputs"]) > 1] == [
        row[0] for row in rounds if row[0]]
    assert [event["forward"] for event in events] == list(range(1, len(events) + 1))


def test_prompt_end_does_not_consume_the_copy_candidate_budget():
    corpus = [1, 2, 3, 1, 2]
    index = NgramIndex(2, 2, max_candidates=1)
    index.sync(corpus)
    index._corpus = corpus
    assert index.find(corpus.__getitem__, len(corpus), max_pos=len(corpus)) == (2, 0)


def test_declined_lazy_round_restores_copy_bookkeeping(pair, monkeypatch):  # noqa: F811
    from mlx2.runtime.proposal_composition import LazyDecline

    def decline(*args, **kwargs):
        raise LazyDecline("test_decline")

    target, draft, _ = pair
    batch = _engine(target, draft, 3, device_round=BOTH, prompt_copy_round=True)
    try:
        monkeypatch.setattr(batch.draft, "draft_lazy", decline)
        uid = batch.insert([list(range(21, 40))], max_tokens=[5])[0]
        lane = batch.lanes[uid]
        _run(batch, uid)
        assert batch.scheduler_stats[DECLINE_REASONS]["test_decline"] > 0
        assert batch.scheduler_stats["external_device_round_rounds"] == 0
        assert batch.scheduler_stats["external_prompt_copy_probes"] == 0
        assert batch.scheduler_stats["external_prompt_copy_decline_reasons"] == {}
        assert lane.prompt_copy_state.probes == 0
    finally:
        batch.close()


def test_prompt_copy_backoff_suspends_and_doubles():
    runtime = PromptCopyRuntime(PromptCopyPolicy.from_value({"backoff_min": 64, "backoff_max": 256}))
    owner = SimpleNamespace(scheduler_stats={})
    runtime.install_counters(owner.scheduler_stats)
    lane = SimpleNamespace(generated=100, prompt_copy_state=LaneCopyState(backoff=64))
    for _ in range(3):
        runtime.observe(owner, lane, 8, 0)
    state = lane.prompt_copy_state
    assert state.suspend_until == 164 and state.backoff == 128 and state.suspensions == 1
    for _ in range(3):
        runtime.observe(owner, lane, 8, 0)
    assert state.backoff == 256 and state.suspensions == 2
    runtime.observe(owner, lane, 8, 8)
    assert state.backoff == 64
    assert owner.scheduler_stats["external_prompt_copy_suspensions"] == 2


@pytest.mark.parametrize("options", [{}, {"device_round": BOTH}], ids=["copy_only", "copy_lazy"])
def test_prompt_copy_rounds_engage_and_stay_greedy_exact(agreeing, options):
    """A periodic prompt gives the suffix a prompt n-gram at once, so copy
    rounds run; output stays ordinary decode's, with copies past the
    drafter's 7-token block (num_draft 12, DSpark clamped to 7)."""
    target, draft = agreeing
    unit = _prompt(11, length=9)
    prompt = (unit * 5)[:44]
    steps = 40
    expect = _ordinary(target, prompt, steps)
    off, _, _ = _decode(target, draft, prompt, 7, steps)
    on, props, stats = _decode(target, draft, prompt, 12, steps,
                               prompt_copy_round={"max_tokens": 12, "probation_tokens": 8},
                               **options)
    assert off == expect and on == expect
    assert stats["external_prompt_copy_rounds"] > 0
    assert stats["external_prompt_copy_proposed_tokens"] > 0
    assert any(len(row[0]) > 0 for row in props if row[0] is not None)


# -- review 2026-11-23: nucleus miss, laws in the engine, depth > 1, bonus, long copies -----
def _owner():
    return SimpleNamespace(mx=mx, scheduler_stats={}, stops=set())


@pytest.mark.parametrize("law", ["token", "block"])
@pytest.mark.parametrize("q", [[np.nan, 0.0], [-np.inf, -np.inf], [-np.inf, 0.0]])
def test_sampled_device_verification_refuses_invalid_proposal(law, q):
    from mlx2.runtime.external_speculative import LaneFailure

    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value(BOTH)
    owner = _owner()
    owner.exact_verification = law
    lane = SimpleNamespace(uid=1, processors=[], rng=RequestRNG(0),
                           sampling={"sampling_temp": 1.0})
    row = SimpleNamespace(lane=lane, count=1, host=None, drafts=mx.array([0]),
                          q=mx.array([q]), ratio=True, seed=1)
    logits = mx.zeros((1, 2, 2))
    plans = runtime._plan(owner, [lane], [row], logits)
    with pytest.raises(LaneFailure, match="proposal law"):
        runtime._finish(owner, plans, logits)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("draft,raw_last,valid", [
    (1, 7.0, True),       # unreachable: use the raw EOS-only row
    (1, -np.inf, False),  # unreachable: even the raw row is invalid
    (0, 7.0, False),      # reachable: the processor masks its only token
])
def test_minimum_tokens_checks_only_reachable_processed_laws(temperature, draft, raw_last, valid):
    from mlx2.runtime.external_speculative import LaneFailure
    from mlx2.serving import minimum_tokens_processor

    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value({**BOTH, "processors": True})
    processor = minimum_tokens_processor(mx, [2], prompt_tokens=1, minimum=9)
    lane = SimpleNamespace(uid=1, processors=[processor], rng=RequestRNG(0),
                           history=[], anchor=0, sampling={"sampling_temp": temperature})
    row = SimpleNamespace(lane=lane, count=1, host=[draft], drafts=mx.array([draft]),
                          ratio=False, seed=1)
    logits = mx.array([[[3.0, -np.inf, 2.0], [-np.inf, -np.inf, raw_last]]])
    owner = _owner()
    assert runtime._processors_ok(lane)
    plans = runtime._plan(owner, [lane], [row], logits)
    if valid:
        decision, = runtime._finish(owner, plans, logits)
        assert decision.accepted == 0 and decision.emitted == [0]
        assert plans[0].response[1].tolist() == [-np.inf, -np.inf, 0.0]
    else:
        with pytest.raises(LaneFailure, match="target law"):
            runtime._finish(owner, plans, logits)


def test_nucleus_window_miss_completes_with_the_dense_kept_set():
    """Zero logits over 4096 tokens: every score ties, so the certified top-512
    window cannot decide top-p 0.95 and the round redoes the dense transform.
    The redo plan must replace the original (``_run`` reads ``drafts_host``
    from the list) and draw only from the dense kept set."""
    from mlx2.runtime import sample_utils

    runtime = DeviceRound.__new__(DeviceRound)
    runtime.policy = DeviceRoundPolicy.from_value({**BOTH, "nucleus": True})
    logits = mx.zeros((1, 2, 4096))
    dense = sample_utils.make_transformed_logprobs(1.0, top_p=0.95)(logits[0, :1])[0]
    kept = set(np.flatnonzero(np.isfinite(np.asarray(dense))).tolist())
    assert 0 < len(kept) < 4096
    for seed in range(12):
        owner = _owner()
        lane = SimpleNamespace(uid=1, processors=[], rng=RequestRNG(seed),
                               sampling={"sampling_temp": 1.0, "top_p": 0.95})
        row = SimpleNamespace(lane=lane, count=1, host=[4095],
                              drafts=mx.array([4095], dtype=mx.int32),
                              q=None, ratio=False, seed=None)
        plans = runtime._plan(owner, [lane], [row], logits)
        mx.eval(*[plan.packet for plan in plans])
        decisions = runtime._finish(owner, plans, logits)
        assert [plan.drafts_host for plan in plans] == [[4095]]  # what ``_run`` reads
        assert owner.scheduler_stats["external_device_round_nucleus_window_misses"] == 1
        assert np.array_equal(np.isfinite(np.asarray(plans[0].response[0])),
                              np.isfinite(np.asarray(dense)))
        assert all(token in kept for token in decisions[0].emitted)


def test_ratio_law_at_depth_two_and_bonus_after_full_acceptance():
    """Two drafted positions: P(accept >= 1) = sum min(p0, q0) and, given
    that, the second emitted token ~ p1.  With q == p every draft is
    accepted and the last token is the bonus row's draw."""
    p0, q0 = np.array([0.5, 0.2, 0.2, 0.1]), np.array([0.1, 0.6, 0.2, 0.1])
    p1, q1 = np.array([0.1, 0.1, 0.4, 0.4]), np.array([0.25, 0.25, 0.25, 0.25])
    bonus = np.array([0.7, 0.1, 0.1, 0.1])
    logits = np.log(np.stack([p0, p1, bonus])).astype(np.float32)
    policy = {**BOTH, "sampled_drafts": True}

    def run(q_rows, seed):
        logq = mx.array(np.log(np.stack(q_rows)).astype(np.float32))
        keys = mx.random.split(mx.random.key(seed), 6)
        drafts = [int(mx.random.categorical(logq[j], key=keys[j]).item()) for j in range(2)]
        accepted, last = _law_plan(policy, logits, drafts, seed, q=logq)
        return drafts, accepted, last

    n, past_first, second = 4000, 0, np.zeros(4)
    for seed in range(n):
        drafts, accepted, last = run([q0, q1], seed)
        if accepted >= 1:
            past_first += 1
            second[drafts[1] if accepted == 2 else last] += 1
    expect = np.minimum(p0, q0).sum()
    assert abs(past_first / n - expect) < 4 * np.sqrt(expect * (1 - expect) / n)
    freq = second / past_first
    assert np.all(np.abs(freq - p1) < 4 * np.sqrt(p1 * (1 - p1) / past_first) + 1e-3)
    last_tokens = np.zeros(4)
    for seed in range(2000):
        _drafts, accepted, last = run([p0, p1], seed)
        assert accepted == 2
        last_tokens[last] += 1
    assert np.all(np.abs(last_tokens / 2000 - bonus)
                  < 4 * np.sqrt(bonus * (1 - bonus) / 2000) + 1e-3)


def test_greedy_bonus_after_full_acceptance():
    rows = np.array([[0, 5, 0, 0], [0, 0, 5, 0], [9, 0, 0, 0]], np.float32)
    assert _law_plan(BOTH, rows, [1, 2], 0, temp=0.0) == (2, 0)
    assert _law_plan(BOTH, rows, [1, 3], 0, temp=0.0) == (1, 2)


def _forced(bonus_of):
    """A tiny target whose next-token law is pinned by ``bonus_of(input
    token)`` (a [V] float32 vector added to its output rows), with a
    random-weight DSpark drafter bound to it (its proposals ignore the pin)."""
    from test_inkling2_dspark_cpu import TEXT, _config, _random

    from mlx2.runtime.drafters.inkling2_dspark import (
        Inkling2DSparkConfig,
        Inkling2DSparkDraftModel,
    )
    from mlx2.runtime.models.inkling2 import Model, TextConfig

    class Forced(Model):
        def _pin(self, inputs, rows):
            tail = inputs[:, -rows.shape[1]:]
            width = rows.shape[-1]  # the unpadded vocabulary
            return rows + mx.array(np.stack([[bonus_of(int(t))[:width] for t in row]
                                             for row in np.asarray(tail).tolist()]))

        def __call__(self, inputs=None, cache=None, **kwargs):
            return self._pin(inputs, super().__call__(inputs, cache, **kwargs))

        def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
            rows, features = super().forward_with_taps(inputs, cache, capture_layers,
                                                       body_only=body_only)
            return (None if rows is None else self._pin(inputs, rows)), features

    target = Forced(TextConfig.from_dict(TEXT))
    _random(target, 0)
    draft = Inkling2DSparkDraftModel(Inkling2DSparkConfig.from_hf(_config()))
    _random(draft, 1)
    draft.bind(target)
    return target, draft


LAW_TOKENS, LAW = (20, 21, 22), np.array([0.6, 0.3, 0.1])


@pytest.fixture(scope="module")
def pinned_law():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    vector = np.zeros(512, np.float32)
    vector[list(LAW_TOKENS)] = 80.0 + np.log(LAW)
    pair = _forced(lambda token: vector)
    mx.set_default_device(old)
    return pair


@pytest.mark.parametrize("options", [{}, BOTH, {**BOTH, "sampled_drafts": True}],
                         ids=["host", "device_point_mass", "device_ratio"])
def test_in_engine_sampled_law_matches_the_target(pinned_law, options):
    """Temperature 1 against a target whose law is pinned to three tokens
    (0.6 / 0.3 / 0.1, independent of context): every emitted token, across
    requests with different seeds, follows it; depth 4, so ratio rounds
    run past the first position."""
    target, draft = pinned_law
    # The token-wise ratio law (block has its own law test below).
    batch = _engine(target, draft, 4, **({"device_round": options, "exact_verification": "token"}
                                         if options else {}))
    counts, tokens = np.zeros(3), 0
    try:
        for request in range(60):
            uid = batch.insert([_prompt(request, 12)], max_tokens=[6],
                               sampling_configs=[{"sampling_temp": 1.0}])[0]
            for token in _run(batch, uid):
                counts[LAW_TOKENS.index(token)] += 1
                tokens += 1
        stats = dict(batch.scheduler_stats)
    finally:
        batch.close()
    freq = counts / tokens
    assert np.all(np.abs(freq - LAW) < 4 * np.sqrt(LAW * (1 - LAW) / tokens))
    if options.get("sampled_drafts"):
        assert stats["external_device_round_decide_ratio_lanes"] > 0
    elif options:
        assert stats["external_device_round_decide_point_mass_lanes"] > 0


@pytest.fixture(scope="module")
def cycle():
    """Greedy target that continues the 40-token cycle 100..139."""
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    table = {}
    for token in range(512):
        vector = np.zeros(512, np.float32)
        vector[100 + (token - 100 + 1) % 40 if 100 <= token < 140 else 100] = 80.0
        table[token] = vector
    pair = _forced(table.__getitem__)
    mx.set_default_device(old)
    return pair


@pytest.mark.parametrize("options", [{}, {"device_round": BOTH}], ids=["copy_only", "copy_lazy"])
def test_prompt_copy_longer_than_the_drafter_block_is_proposed_and_accepted(cycle, options):
    """The prompt holds the cycle once and restarts it; the target continues
    it, so the copy lane proposes prompt spans past DSpark's 7-token block and
    they land (DSpark itself, unpinned, almost never does)."""
    target, draft = cycle
    prompt = [*range(100, 140), *range(100, 110)]
    steps = 40
    expect = _ordinary(target, prompt, steps)
    assert expect[:3] == [110, 111, 112]
    batch = _engine(target, draft, 24, prompt_copy_round={"max_tokens": 24}, **options)
    seen = []
    commit = batch._commit

    def wrapped(cohort, decisions, features, *, blocks, **kwargs):
        for block, decision in zip(blocks, decisions):
            if block is not None and getattr(block, "proposal_source", None) == "prompt_copy":
                seen.append((len(block.tokens), decision.accepted))
        return commit(cohort, decisions, features, blocks=blocks, **kwargs)

    batch._commit = wrapped
    try:
        uid = batch.insert([prompt], max_tokens=[steps], sampling_configs=[{"sampling_temp": 0}])[0]
        got = _run(batch, uid)
        stats = dict(batch.scheduler_stats)
    finally:
        batch.close()
    assert got == expect
    assert any(length > 7 and accepted > 7 for length, accepted in seen), seen
    assert stats["external_prompt_copy_accepted_tokens"] > 7


def test_copy_only_engine_installs_decline_counters_and_propose_books_nothing(cycle):
    """Copy-only: runner declines land in installed counters; ``propose``
    alone counts nothing (only ``book`` does, for a round the runner serves)."""
    target, draft = cycle
    batch = _engine(target, draft, 24, prompt_copy_round=True)
    try:
        assert batch.scheduler_stats["external_device_round_declines"] == 0
        assert batch.scheduler_stats[DECLINE_REASONS] == {}
        uid = batch.insert([[*range(100, 140), *range(100, 110)]], max_tokens=[30],
                           sampling_configs=[{"sampling_temp": 0}])[0]
        lane = batch.lanes[uid]
        for _ in range(20):  # prefill slices until the anchor is set
            if lane.anchor is not None:
                break
            batch.next()
            lane = batch.lanes[uid]
        before = dict(batch.scheduler_stats)
        probes = batch.device_round.copy._state(lane).probes
        outcome = batch.device_round.copy.propose(batch, lane)
        assert outcome.tokens and outcome.probed
        assert batch.scheduler_stats["external_prompt_copy_probes"] == before["external_prompt_copy_probes"]
        assert batch.scheduler_stats[
            "external_prompt_copy_decline_reasons"] == before["external_prompt_copy_decline_reasons"]
        assert lane.prompt_copy_state.probes == probes
        batch.device_round.copy.book(batch, lane, outcome)
        assert batch.scheduler_stats["external_prompt_copy_probes"] == before["external_prompt_copy_probes"] + 1
    finally:
        batch.close()


@pytest.mark.parametrize("setting,reason", [
    (("engine", "pairwise_selection", "batched"), "pairwise_selection"),
    (("draft", "requires_proposal_contexts", True), "proposal_contexts"),
    (("draft", "requires_target_caches", True), "target_caches"),
])
def test_engine_reasons_decline_unsupported_shapes(pair, setting, reason):
    target, draft, _ = pair
    batch = _engine(target, draft, 3, device_round=LAZY)
    try:
        where, name, value = setting
        setattr(batch if where == "engine" else batch.draft, name, value)
        assert DeviceRound._engine_reason(batch) == reason
    finally:
        batch.close()


def test_copies_are_capped_to_the_lane_exact_width():
    from mlx2.runtime.external_speculative import PROGRESSIVE_FIDELITY_EXACT

    owner = SimpleNamespace(model=SimpleNamespace(progressive_external_target_rows=object()),
                            progressive_verification_fidelity=PROGRESSIVE_FIDELITY_EXACT)
    assert DeviceRound._lane_exact_copy_cap(owner, [1, 2]) == 15
    assert DeviceRound._lane_exact_copy_cap(owner, [1]) is None
    owner.progressive_verification_fidelity = None
    assert DeviceRound._lane_exact_copy_cap(owner, [1, 2]) is None


# -- exact block verification on the device ------------------------------------
BLOCK = {**BOTH, "sampled_drafts": True}


def test_block_verification_rounds_decide_on_the_device_with_the_target_law(pinned_law):
    """``exact_verification: "block"`` with sampled drafts: ratio rounds take
    the device block rule (no decline), the route receipt names exact block
    verification, and every emitted token follows the pinned target law."""
    target, draft = pinned_law
    batch = _engine(target, draft, 4, device_round=BLOCK, exact_verification="block")
    counts, tokens, receipts = np.zeros(3), 0, []
    try:
        for request in range(60):
            uid = batch.insert([_prompt(request, 12)], max_tokens=[6],
                               sampling_configs=[{"sampling_temp": 1.0}])[0]
            lane = batch.lanes[uid]
            for token in _run(batch, uid):
                counts[LAW_TOKENS.index(token)] += 1
                tokens += 1
            receipts.append(getattr(lane, "block_verified_rounds", 0))
        stats = dict(batch.scheduler_stats)
    finally:
        batch.close()
    freq = counts / tokens
    assert np.all(np.abs(freq - LAW) < 4 * np.sqrt(LAW * (1 - LAW) / tokens))
    assert stats["external_device_round_decide_block_lanes"] > 0
    assert stats.get("external_device_round_decide_ratio_lanes", 0) == 0
    assert stats["external_device_round_declines"] == 0
    assert sum(receipts) == stats["external_device_round_decide_block_lanes"]


def _sampled_rule_stats(target, draft, **options):
    batch = _engine(target, draft, 4, **options)
    try:
        for request in range(8):
            uid = batch.insert([_prompt(request, 12)], max_tokens=[6],
                               sampling_configs=[{"sampling_temp": 1.0}])[0]
            list(_run(batch, uid))
        return dict(batch.scheduler_stats), batch.device_round.ratio_rule(batch)
    finally:
        batch.close()


def test_unset_exact_verification_is_block_for_sampled_device_rounds(pinned_law):
    """DEFAULT ON (Pierre 2026-11-27): with no ``exact_verification`` the
    device round's ratio rows take the block rule; an explicit "token" opts
    out; a device commit keeps the token rule (block under it unvalidated)."""
    target, draft = pinned_law
    stats, rule = _sampled_rule_stats(target, draft, device_round=BLOCK)
    assert rule == "block"
    assert stats["external_device_round_decide_block_lanes"] > 0
    assert stats.get("external_device_round_decide_ratio_lanes", 0) == 0
    assert stats["external_device_round_declines"] == 0
    stats, rule = _sampled_rule_stats(target, draft, device_round=BLOCK, exact_verification="token")
    assert rule == "token"
    assert stats["external_device_round_decide_ratio_lanes"] > 0
    assert stats.get("external_device_round_decide_block_lanes", 0) == 0
    batch = _engine(target, draft, 4, device_round={**BLOCK, "device_commit": True})
    try:
        assert batch.device_round.ratio_rule(batch) == "token"
        assert batch.exact_verification == "token"  # the host verifier is unchanged
    finally:
        batch.close()


def test_block_verification_keeps_greedy_and_point_mass_rules(agreeing):
    """Greedy rows under block verification are the greedy rule (bitwise the
    token-wise device round), and point-mass rows keep their rule: neither
    declines, as the host never sends those rows to the block verifier."""
    target, draft = agreeing
    plain, _, _ = _decode(target, draft, _prompt(5), 4, 20, device_round=BOTH)
    block, _, stats = _decode(target, draft, _prompt(5), 4, 20, device_round=BOTH,
                              exact_verification="block")
    assert block == plain
    assert stats["external_device_round_declines"] == 0
    _, _, stats = _decode(target, draft, _prompt(5), 4, 20, {"sampling_temp": 1.0},
                          device_round=BOTH, exact_verification="block")
    assert stats["external_device_round_decide_point_mass_lanes"] > 0
    assert stats["external_device_round_declines"] == 0
