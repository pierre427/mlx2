"""Copy-drafts inside the self-MTP verify transaction (rm01, default off)."""
import copy
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_batched_mtp import _tiny_qwen4_model  # noqa: E402

from mlx2.runtime.copy_draft import (  # noqa: E402
    CopyDraftPolicy,
    CopyDraftState,
    cohort_copy_cap,
    verify_point_mass_by_sampling,
)
from mlx2.runtime.sample_utils import LaneRNG  # noqa: E402

CYCLE = [11, 23, 5, 42, 17, 30, 8, 51, 2, 36]
_NEXT = {token: CYCLE[(index + 1) % len(CYCLE)] for index, token in enumerate(CYCLE)}


# -- pure policy/state -------------------------------------------------------


@pytest.mark.parametrize(
    "value,error",
    [
        ({"mystery": 1}, "unknown self_mtp_copy_draft"),
        ({"enabled": 1}, "enabled must be boolean"),
        ({"enabled": True, "ngram_min": 4, "ngram_max": 3}, "ngram_max"),
        ({"enabled": True, "probe_span": 9, "max_span": 8}, "probe_span"),
        ({"enabled": True, "verify_row_cost": -1}, "nonnegative"),
        ("yes", "object, boolean or null"),
    ],
)
def test_copy_policy_rejects_invalid_values(value, error):
    with pytest.raises(ValueError, match=error):
        CopyDraftPolicy.from_value(value)


def test_copy_policy_default_off_and_round_trip():
    assert CopyDraftPolicy.from_value(None).enabled is False
    assert CopyDraftPolicy.from_value(False).enabled is False
    policy = CopyDraftPolicy.from_value({"enabled": True, "max_span": 16})
    assert CopyDraftPolicy.from_value(policy.as_dict()) == policy
    assert CopyDraftPolicy.from_value(True).enabled is True
    with pytest.raises(ValueError, match="enabled"):
        CopyDraftState(CopyDraftPolicy())


def test_lookup_prefers_longest_suffix_then_most_recent_source():
    policy = CopyDraftPolicy(enabled=True, ngram_min=2, ngram_max=4)
    # "1 2 3" -> 9 early, "7 1 2 3" -> 4 later (longer match wins).
    state = CopyDraftState(policy, [7, 1, 2, 3, 4, 5, 0, 1, 2, 3, 9, 9, 7, 1, 2, 3])
    assert state.lookup(3) == [4, 5, 0]
    state.observe([4])
    assert state.lookup(2) == [5, 0]
    # No match: empty.
    assert CopyDraftState(policy, [1, 2, 3, 4]).lookup(4) == []


def test_sizer_is_a_congestion_window_and_gate_declines_then_reprobes():
    policy = CopyDraftPolicy(
        enabled=True, max_span=8, probe_span=2, min_samples=2, reprobe_interval=3
    )
    stream = CYCLE * 2
    state = CopyDraftState(policy, stream)
    widths = []
    for _ in range(4):
        span, decision = state.plan(head_depth=2, cap=100)
        assert decision == "copy"
        widths.append(len(span))
        state.record(
            copy_span=len(span), head_depth=0, accepted=len(span),
            emitted=len(span) + 1, committed=span + [_NEXT[span[-1]]],
        )
    assert widths == [2, 4, 8, 8]
    # A short round falls back to 1.5x measured acceptance (floor probe).
    state.record(copy_span=8, head_depth=0, accepted=3, emitted=4, committed=[1])
    assert state.width == 5
    state.record(copy_span=5, head_depth=0, accepted=0, emitted=1, committed=[1])
    assert state.width == 2

    # Losing copies against a strong head: the gate declines, then re-probes.
    gate = CopyDraftState(policy, stream)
    for _ in range(4):
        gate.record(copy_span=2, head_depth=0, accepted=0, emitted=1, committed=[])
        gate.record(copy_span=0, head_depth=2, accepted=2, emitted=3, committed=[])
    decisions = [gate.plan(head_depth=2, cap=8)[1] for _ in range(3)]
    assert decisions == ["declined", "declined", "probe"]
    assert gate.gate_declines == 2 and gate.probe_rounds == 1


def test_deepcopy_is_a_watermark_and_restore_truncates_the_shared_store():
    policy = CopyDraftPolicy(enabled=True, ngram_min=2, ngram_max=3)
    state = CopyDraftState(policy, [1, 2, 3, 1, 2])
    snapshot = copy.deepcopy(state)
    assert snapshot._store is state._store  # O(1): no index copy
    assert state.lookup(2) == [3, 1]
    state.record(copy_span=2, head_depth=0, accepted=2, emitted=3, committed=[3, 1, 9])
    assert state.index_tokens == 8 and state.copy_rounds == 1
    restored = copy.deepcopy(snapshot)
    assert restored.index_tokens == 5 and restored.copy_rounds == 0
    # First read truncates the shared store (and its index) to the watermark.
    assert restored.lookup(3) == [3, 1, 2]
    assert len(restored._store.tokens) == 5
    assert all(
        start + size <= 5
        for size, table in restored._store.index.items()
        for starts in table.values()
        for start in starts
    )


def test_cohort_cap_bounds_batched_padding():
    # Default: single-lane rounds copy, cohorts do not (GPU 2026-09-19).
    policy = CopyDraftPolicy(enabled=True, max_span=8)
    assert policy.batched_max_span == 0
    assert cohort_copy_cap(policy, lanes=1, head_depths=[2]) == 8
    assert cohort_copy_cap(policy, lanes=3, head_depths=[2, 1, 0]) == 0
    head_depth_cap = CopyDraftPolicy(enabled=True, max_span=8, batched_max_span=None)
    assert cohort_copy_cap(head_depth_cap, lanes=3, head_depths=[2, 1, 0]) == 2
    assert cohort_copy_cap(head_depth_cap, lanes=2, head_depths=[0, 0]) == 1
    wide = CopyDraftPolicy(enabled=True, max_span=8, batched_max_span=4)
    assert cohort_copy_cap(wide, lanes=4, head_depths=[2]) == 4


def test_point_mass_sampling_law_is_exact():
    """Accept d with p(d); otherwise emit p restricted to != d (renormalised)."""
    rng = np.random.default_rng(5)
    p = np.array([0.5, 0.3, 0.2])
    n = 20000
    first = np.empty(n, dtype=np.int64)
    accepted = 0
    for i in range(n):
        sampled = rng.choice(3, size=2, p=p)
        n_accept, token = verify_point_mass_by_sampling([0], sampled)
        accepted += n_accept
        first[i] = 0 if n_accept else token
    emitted = np.bincount(first, minlength=3) / n
    assert 0.5 * np.abs(emitted - p).sum() < 0.02
    assert abs(accepted / n - p[0]) < 0.02
    with pytest.raises(ValueError, match="bonus"):
        verify_point_mass_by_sampling([1, 2], [1, 2])


# -- end-to-end on the tiny model -------------------------------------------


def _copying_model(seed=922):
    """Tiny model whose target law follows a fixed token cycle.

    The MTP head is untouched (its drafts are mostly wrong), so any speedup
    must come from copied spans, and exactness is checked against copy-off.
    """
    mx.random.seed(seed)
    model = _tiny_qwen4_model()
    real_backbone = model.mtp_backbone
    real_logits = model.logits
    last = {}
    table = np.full(64, CYCLE[0], dtype=np.int64)
    for token, successor in _NEXT.items():
        table[token] = successor
    table_mx = mx.array(table)

    def backbone(inputs, cache=None):
        last["tokens"] = inputs
        return real_backbone(inputs, cache=cache)

    def logits(hidden):
        base = real_logits(hidden)
        # Only the target head (called right after a trunk forward) is
        # biased; the MTP head's own logits stay those of the tiny model.
        tokens = last.pop("tokens", None)
        if tokens is None:
            return base
        tokens = tokens[:, -hidden.shape[1] :]
        bias = mx.eye(64)[table_mx[tokens]] * 30.0
        return base + bias.astype(base.dtype)

    model.mtp_backbone = backbone
    model.logits = logits
    return model


def _run(model, prompt, *, max_tokens, copy=None, segmented=True, lanes=1,
         temp=0.0, adaptive=None, prompts=None):
    from mlx2.runtime.generate import BatchGenerator

    self_mtp = {"num_draft": 2, "persistent": True}
    if segmented:
        self_mtp.update(
            {"segment_aware_live_tip": True, "segment_aware_cohort_size": lanes}
        )
    kwargs = {}
    if copy is not None:
        kwargs["copy_draft"] = copy
    if adaptive is not None:
        kwargs["adaptive_mtp_depth"] = adaptive
    generator = BatchGenerator(
        model,
        completion_batch_size=lanes,
        prefill_batch_size=lanes,
        prefill_step_size=32,
        self_mtp=self_mtp,
        **kwargs,
    )
    prompts = prompts or [prompt]
    generator.insert(
        prompts,
        max_tokens=[max_tokens] * len(prompts),
        lane_rngs=[LaneRNG(17 + i) for i in range(len(prompts))],
        self_mtp_configs=[{"sampling_temp": temp}] * len(prompts),
    )
    outputs = {}
    receipts = {}
    for _ in range(400):
        _, responses = generator.next()
        for response in responses:
            outputs.setdefault(response.uid, []).append(response.token)
            receipt = getattr(response, "mtp_receipt", None)
            if receipt is not None:
                receipts[response.uid] = receipt
        if outputs and all(len(v) >= max_tokens for v in outputs.values()) and len(outputs) == len(prompts):
            break
    stats = dict(generator.scheduler_stats)
    generator.close()
    return [outputs[uid] for uid in sorted(outputs)], stats, receipts


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


PROMPT = [1, 7] + CYCLE + CYCLE[:4]


@pytest.mark.parametrize("segmented", [True, False])
def test_greedy_copy_mtp_is_exact_and_copies_beyond_head_depth(cpu, segmented):
    model = _copying_model()
    baseline, base_stats, _ = _run(model, PROMPT, max_tokens=40, segmented=segmented)
    copied, stats, receipts = _run(
        model, PROMPT, max_tokens=40, segmented=segmented,
        copy={"enabled": True, "max_span": 8},
    )
    # Exactness: copy-on output is token-identical to copy-off self-MTP.
    assert copied == baseline
    assert baseline[0][:5] == [CYCLE[4], CYCLE[5], CYCLE[6], CYCLE[7], CYCLE[8]]
    # Mechanism ran: copied spans were verified and accepted...
    assert stats["self_mtp_copy_rounds"] > 0
    assert stats["self_mtp_copy_accepted_tokens"] > 0
    # ...some wider than the head's depth 2 (the sizer climbed).
    assert stats["self_mtp_copy_proposed_tokens"] > 2 * stats["self_mtp_copy_rounds"]
    assert "self_mtp_copy_rounds" not in base_stats
    receipt = next(iter(receipts.values()))
    copy_receipt = receipt["copy_draft"]
    assert copy_receipt["enabled"] and copy_receipt["copy_rounds"] > 0
    assert copy_receipt["verification"] == "exact"
    assert receipt["stats"]["retrieval_accepted"] == copy_receipt["copy_accepted"]
    # Head-only acceptance stays head-only.
    assert receipt["stats"]["draft_proposed"] == copy_receipt["head_proposed"]
    # Fewer target verify rounds than copy-off for the same tokens.
    assert (
        copy_receipt["copy_rounds"] + copy_receipt["head_rounds"]
        < receipt["stats"]["total_emitted"] - 1
    )


def _ordinary_copying_model():
    """The copying model with the same cycle bias on the ordinary path."""
    model = _copying_model()
    table = np.full(64, CYCLE[0], dtype=np.int64)
    for token, successor in _NEXT.items():
        table[token] = successor
    table_mx = mx.array(table)
    base = type(model)

    class OrdinaryCopying(base):
        def __call__(self, inputs, cache=None, input_embeddings=None):
            logits = base.__call__(self, inputs, cache, input_embeddings)
            bias = mx.eye(64)[table_mx[inputs]] * 30.0
            return logits + bias.astype(logits.dtype)

    model.__class__ = OrdinaryCopying
    return model


def _run_until_stop(model, *, stop, copy=None, segmented=True, max_tokens=60):
    from mlx2.runtime.generate import BatchGenerator

    kwargs = {}
    if copy is not None:
        self_mtp = {"num_draft": 2, "persistent": True}
        if segmented:
            self_mtp.update(
                {"segment_aware_live_tip": True, "segment_aware_cohort_size": 1}
            )
        kwargs.update(self_mtp=self_mtp, copy_draft=copy)
    generator = BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32,
        stop_tokens=[[stop]], **kwargs,
    )
    insert = {"max_tokens": [max_tokens]}
    if copy is not None:
        insert.update(
            lane_rngs=[LaneRNG(17)], self_mtp_configs=[{"sampling_temp": 0.0}]
        )
    generator.insert([PROMPT], **insert)
    tokens, finish = [], None
    try:
        for _ in range(400):
            _, responses = generator.next()
            for response in responses:
                tokens.append(response.token)
                finish = response.finish_reason or finish
            if finish:
                break
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    return tokens, finish, stats


@pytest.mark.parametrize("segmented", [True, False])
def test_stop_inside_a_copied_span_after_copy_rounds_matches_ordinary(cpu, segmented):
    """A stop inside a copied span keeps every pending MTP pair.

    Copy rows run the head at depth 0, so undrafted pairs pile up across
    consecutive copy rounds.  The terminal commit used to replace them with
    this round's pairs only, and the detach replay then failed validation.
    """
    model = _copying_model()
    ordinary = _ordinary_copying_model()
    multi_round_stops = 0
    for stop in CYCLE:
        expected = _run_until_stop(ordinary, stop=stop)
        got = _run_until_stop(
            model, stop=stop, segmented=segmented,
            copy={"enabled": True, "max_span": 8},
        )
        assert got[:2] == expected[:2], stop
        assert got[1] == "stop"
        multi_round_stops += got[2].get("self_mtp_copy_rounds", 0) >= 2
    assert multi_round_stops >= 5


class _LetterTokenizer:
    eos_token_ids = [0]
    vocab_size = 64

    def decode(self, tokens, **_kwargs):
        return "".join(chr(65 + int(token)) for token in tokens if int(token) != 0)


class _RunOfBs:
    """Grammar: one or more ``B`` (token 1), complete at two or more."""

    pattern = None

    @staticmethod
    def canonicalize(value):
        return value

    @staticmethod
    def fullmatch(value, *, partial=False, timeout=None):
        legal = all(char == "B" for char in value)
        return object() if legal and (partial or len(value) >= 2) else None


@pytest.mark.parametrize("route", ["ordinary", "mtp", "mtp-copy", "segmented-copy"])
def test_copied_token_the_grammar_forbids_does_not_latch_a_failure(cpu, route):
    """A rejected copied token must not reach the real structured processor.

    The prompt repeats ``1 1`` followed by ``1`` (the copy source), but the
    cycle bias makes the copied continuation illegal.  Verification rejects
    it; the processor used to see the rows after it, latch NO_CONTINUATION,
    and the serving layer returned 502.
    """
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.structured_output import StructuredOutputProcessor

    prompt = [2, 1, 1, 1, 9, 9, 3, 7, 1, 1]
    processor = StructuredOutputProcessor(_LetterTokenizer(), len(prompt), _RunOfBs())
    kwargs = {}
    insert = {"max_tokens": [12], "logits_processors": [[processor]]}
    if route == "ordinary":
        model = _ordinary_copying_model()
    else:
        model = _copying_model()
        kwargs["self_mtp"] = {"num_draft": 2, "persistent": True}
        insert.update(
            lane_rngs=[LaneRNG(3)], self_mtp_configs=[{"sampling_temp": 0.0}]
        )
        if route == "segmented-copy":
            kwargs["self_mtp"].update(
                {"segment_aware_live_tip": True, "segment_aware_cohort_size": 1}
            )
        if route.endswith("copy"):
            kwargs["copy_draft"] = {"enabled": True, "ngram_min": 2, "ngram_max": 3}
    generator = BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32,
        stop_tokens=[[0]], **kwargs,
    )
    tokens, finish = [], None
    try:
        generator.insert([prompt], **insert)
        for _ in range(60):
            _, responses = generator.next()
            for response in responses:
                tokens.append(response.token)
                finish = response.finish_reason or finish
            if finish:
                break
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    assert processor.failure is None
    assert (tokens, finish) == ([1, 1, 0], "stop")
    if route.endswith("copy"):
        assert stats["self_mtp_copy_rounds"] > 0


def test_default_off_receipt_and_counters_unchanged(cpu):
    model = _copying_model()
    _, stats, receipts = _run(model, PROMPT, max_tokens=12)
    assert not any(key.startswith("self_mtp_copy_") for key in stats)
    assert all("copy_draft" not in receipt for receipt in receipts.values())


def test_sampled_copy_mtp_engages_under_the_point_mass_law(cpu):
    model = _copying_model()
    outputs, stats, receipts = _run(
        model, PROMPT, max_tokens=40, temp=0.7, copy={"enabled": True}
    )
    assert len(outputs[0]) >= 40
    assert stats["self_mtp_copy_rounds"] > 0
    assert stats["self_mtp_copy_accepted_tokens"] > 0
    # The cycle bias dominates, so sampled text still follows the cycle.
    follows = sum(_NEXT.get(a) == b for a, b in zip(outputs[0], outputs[0][1:]))
    assert follows >= 35


def test_two_lane_cohort_mixes_copy_and_head_rows_exactly(cpu, monkeypatch):
    from mlx2.runtime import hybrid_speculative

    model = _copying_model()
    other = [3, 4, 60, 61, 62, 63, 59]
    solo = [
        _run(model, prompt, max_tokens=24)[0][0] for prompt in (PROMPT, other)
    ]
    rounds = []
    real_plan = hybrid_speculative._plan_copy_drafts

    def spy(lanes, head_depths):
        rows, decisions = real_plan(lanes, head_depths)
        rounds.append((len(lanes), tuple(head_depths), tuple(len(r) for r in rows)))
        return rows, decisions

    monkeypatch.setattr(hybrid_speculative, "_plan_copy_drafts", spy)
    batched, stats, receipts = _run(
        model, None, max_tokens=24, lanes=2, prompts=[PROMPT, other],
        copy={"enabled": True, "batched_max_span": None},
    )
    assert batched == solo
    assert stats["self_mtp_copy_rounds"] > 0
    two_lane = [r for r in rounds if r[0] == 2]
    # Ragged verify: some two-lane round mixes a copy row with a head row or
    # two copy rows, and none pads wider than the cohort's head depth.
    assert any(any(spans) for _, _, spans in two_lane)
    for _, depths, spans in two_lane:
        assert max(spans) <= max(max(depths), 1)


def test_default_policy_refuses_cohort_copies_but_copies_solo(cpu):
    """batched_max_span=0: the lever is single-lane by default (GPU 2026-09-19)."""
    model = _copying_model()
    other = [3, 4, 60, 61, 62, 63, 59]
    _, solo_stats, _ = _run(model, PROMPT, max_tokens=24, copy={"enabled": True})
    assert solo_stats["self_mtp_copy_rounds"] > 0
    _, cohort_stats, _ = _run(
        model, None, max_tokens=24, lanes=2, prompts=[PROMPT, other],
        copy={"enabled": True},
    )
    assert cohort_stats["self_mtp_copy_rounds"] == 0


def test_history_tokens_are_indexed_for_apc_hits(cpu):
    policy = CopyDraftPolicy(enabled=True)
    from mlx2.runtime.generate import BatchGenerator

    model = _copying_model()
    generator = BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=32,
        self_mtp={"num_draft": 2, "persistent": True}, copy_draft=policy,
    )
    try:
        # The lane is built from history + prompt; history alone holds the
        # source n-gram (a prefix-cache hit supplies it as history).
        captured = []
        real = CopyDraftState.__init__

        def spy(self, policy, context=()):
            captured.append(list(context))
            real(self, policy, context)

        CopyDraftState.__init__ = spy
        try:
            generator.insert([PROMPT], max_tokens=[4], lane_rngs=[LaneRNG(1)],
                             self_mtp_configs=[{"sampling_temp": 0.0}])
            for _ in range(20):
                generator.next()
                if captured:
                    break
        finally:
            CopyDraftState.__init__ = real
        assert captured and captured[0][: len(PROMPT)] == PROMPT
        assert len(captured[0]) == len(PROMPT) + 1  # plus the first token
    finally:
        generator.close()


def test_zero_depth_cohort_still_copies_and_depth_policy_sees_head_only(cpu):
    model = _copying_model()
    baseline, _, _ = _run(model, PROMPT, max_tokens=30, adaptive={"current_depth": 0})
    copied, stats, _ = _run(
        model, PROMPT, max_tokens=30, adaptive={"current_depth": 0},
        copy={"enabled": True},
    )
    assert copied == baseline
    assert stats["self_mtp_copy_rounds"] > 0


def test_segmented_abort_restores_copy_state(cpu):
    from mlx2.runtime.hybrid_speculative import (
        abort_batched_self_mtp,
        attach_segmented_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    model = _copying_model()
    (detached, first) = prepare_self_mtp_lane(
        mx.array(PROMPT, mx.uint32), model, uid=3, max_tokens=30,
        prompt_cache=None, mtp_state=None, lane_rng=LaneRNG(3), num_draft=2,
        sampling_temp=0.0, sampling_top_p=1.0, sampling_top_k=0,
        sampling_min_p=0.0, accept_rule="residual", logits_processors=[],
        prefill_step_size=32, share_qsa_indices=False,
    )
    policy = CopyDraftPolicy(enabled=True)
    detached.lane.copy_draft = CopyDraftState(policy, PROMPT + [first.token])
    batch = attach_segmented_self_mtp_lanes(model, None, [detached])
    lane = batch.lanes[0]
    # Commit a couple of rounds so the copy path is live.
    for _ in range(3):
        proposal = propose_batched_self_mtp(model, batch)
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=[len(proposal.outputs[0])], terminal=[False]
        )
    before = (lane.copy_draft.index_tokens, lane.copy_draft.copy_rounds, lane.cur)
    proposal = propose_batched_self_mtp(model, batch)
    assert proposal.copy_spans[0] > 0
    abort_batched_self_mtp(batch, proposal)
    lane = batch.lanes[0]
    assert (lane.copy_draft.index_tokens, lane.copy_draft.copy_rounds, lane.cur) == before
    # The restored lane proposes the same copy again and commits cleanly.
    again = propose_batched_self_mtp(model, batch)
    assert again.copy_spans == proposal.copy_spans
    assert again.outputs[0][0].token == proposal.outputs[0][0].token
    commit_batched_self_mtp(
        batch, again, emitted_counts=[len(again.outputs[0])], terminal=[False]
    )
    assert lane.copy_draft.index_tokens == before[0] + len(again.outputs[0])


# -- serving / qualification -------------------------------------------------


def _fake_serving(monkeypatch, captured):
    from types import SimpleNamespace as NS

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2, generate, os_memory

    class APC:
        def __init__(self, **_kwargs):
            self.apc_stats = {}

        def key(self, *_args, **_kwargs):
            return "key"

        def spill_idle_entries(self):
            pass

        def clear(self):
            pass

    class Batch:
        def __init__(self, *_args, **kwargs):
            self.scheduler_stats = {}
            captured.update(kwargs)
            self.lanes = {}

        def close(self):
            pass

    class Adapter:
        max_context = 64
        layout = "fake-layout"
        model = None
        tokenizer = NS(vocab_size=32, eos_token_ids=[])

        def __init__(self, _path):
            self.identity = {"fingerprint": "fake"}
            self.environment = {}

        def profile_name(self, _mtp):
            return "fake-mtp"

        def execution_config(self, **_kwargs):
            return {"num_draft": 2}

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "source"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    return Adapter


def test_serving_propagates_copy_policy_and_refuses_non_mtp_routes(monkeypatch):
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="unknown self_mtp_copy_draft"):
        ServingEngine("unused", execution_policy={"self_mtp_copy_draft": {"x": 1}})
    captured = {}
    adapter = _fake_serving(monkeypatch, captured)
    selected = {"enabled": True, "max_span": 12}
    engine = ServingEngine(
        "fake", adapter_factory=adapter, qualification_mode=True, mtp=True,
        max_lanes=2, max_inflight=4,
        execution_policy={"self_mtp_copy_draft": selected},
    )
    try:
        assert engine.ready.wait(5)
        assert engine.error is None
        settings = engine.snapshot["settings"]["self_mtp_copy_draft"]
        assert settings["enabled"] is True and settings["max_span"] == 12
        assert captured["copy_draft"].max_span == 12
    finally:
        engine.close()

    captured.clear()
    ordinary = ServingEngine(
        "fake", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=2, max_inflight=4,
        execution_policy={"self_mtp_copy_draft": {"enabled": True}},
    )
    try:
        ordinary.ready.wait(5)
        assert "native self-MTP" in str(ordinary.error)
    finally:
        ordinary.close()

    captured.clear()
    default = ServingEngine(
        "fake", adapter_factory=adapter, qualification_mode=True, mtp=True,
        max_lanes=2, max_inflight=4,
    )
    try:
        assert default.ready.wait(5) and default.error is None
        assert "copy_draft" not in captured
        assert default.snapshot["settings"]["self_mtp_copy_draft"]["enabled"] is False
    finally:
        default.close()


def test_selected_copy_draft_requires_observed_copies():
    from mlx2.qualification import required_feature_checks
    from scripts.qualify_serving import feature_observations

    base = {
        "mtp": True, "speculation": "self_mtp", "execution_policy": {},
        "environment": {}, "max_context": 1024,
    }
    assert "feature_self_mtp_copy_draft" not in required_feature_checks(base)
    assert "feature_self_mtp_copy_draft" in required_feature_checks(
        {**base, "self_mtp_copy_draft": {"enabled": True}}
    )
    assert feature_observations(
        {"scheduler": {"self_mtp_copy_rounds": 0}, "recent_receipts": []}
    )["self_mtp_copy_draft"] == 0
    assert feature_observations(
        {"scheduler": {"self_mtp_copy_rounds": 7}, "recent_receipts": []}
    )["self_mtp_copy_draft"] == 7


def test_prometheus_exports_copy_counters_under_their_own_mechanism():
    from mlx2 import prometheus

    assert prometheus._scheduler_mechanism("self_mtp_copy_rounds") == "self_mtp_copy_draft"
    assert prometheus._scheduler_mechanism("self_mtp_zero_fast_rounds") == "self_mtp"
    for key in (
        "self_mtp_copy_rounds", "self_mtp_copy_proposed_tokens",
        "self_mtp_copy_accepted_tokens", "self_mtp_copy_gate_declines",
    ):
        assert key in prometheus._SCHEDULER_EVENTS


def test_ab_harness_verdict_enforces_go_no_go():
    from scripts.ab_copy_mtp import verdict

    def cell(arm, corpus, width, rate, sha="x"):
        return {"arm": arm, "corpus": corpus, "width": width, "temperature": 0.0,
                "decode_tok_s_mean": rate, "rows": [{"output_sha256": sha}]}

    good = [cell("off", "prose", 1, 20.0), cell("on", "prose", 1, 19.9),
            cell("off", "code", 1, 20.0), cell("on", "code", 1, 25.0)]
    result = verdict(good, [1], [0.0])
    assert result["go"] and result["greedy_output_mismatches"] == 0
    bad = [cell("off", "prose", 1, 20.0), cell("on", "prose", 1, 18.0),
           cell("off", "code", 1, 20.0), cell("on", "code", 1, 20.5, sha="y")]
    result = verdict(bad, [1], [0.0])
    assert not result["go"] and len(result["reasons"]) == 2
    assert result["greedy_output_mismatches"] == 1
    # Peak memory budget: +0.5 GiB on the on-arm fails the verdict.
    heavy = [dict(c, metal_peak_bytes=(10 << 30) + (c["arm"] == "on") * (3 << 28)) for c in good]
    result = verdict(heavy, [1], [0.0])
    assert not result["go"] and result["peak_memory"]["delta_gib"] == 0.75
    light = [dict(c, metal_peak_bytes=(10 << 30) + (c["arm"] == "on") * (1 << 28)) for c in good]
    assert verdict(light, [1], [0.0])["go"]


def test_ab_harness_ordinary_arm_and_native_mtp_flag():
    from scripts.ab_copy_mtp import server_route_args, verdict

    assert server_route_args("ord", None, True) == ["--ordinary"]
    assert server_route_args("on", "p.json", True) == ["--native-mtp", "--execution-policy", "p.json"]
    assert server_route_args("off", "p.json", False) == ["--execution-policy", "p.json"]

    def cell(arm, corpus, rate, sha="x"):
        return {"arm": arm, "corpus": corpus, "width": 1, "temperature": 0.0,
                "decode_tok_s_mean": rate, "rows": [{"output_sha256": sha}]}

    cells = [cell("off", "code", 20.0), cell("on", "code", 25.0), cell("ord", "code", 22.0, sha="z"),
             cell("off", "prose", 20.0), cell("on", "prose", 20.0), cell("ord", "prose", 20.0)]
    result = verdict(cells, [1], [0.0])
    code = result["cells"]["code/B1/t0.0"]
    assert result["go"] and code["ratio"] == 1.25
    assert abs(code["on_vs_ord"] - 25.0 / 22.0) < 1e-9
    assert result["greedy_output_mismatches"] == 0
    assert result["greedy_output_mismatches_vs_ordinary"] == 1


def test_logits_processors_reject_forbidden_copied_tokens_exactly(cpu):
    """A constraint the prompt violates: copies of it must be refused exactly."""
    from mlx2.runtime.generate import BatchGenerator

    forbidden = CYCLE[3]
    mask = mx.array([-1e9 if token == forbidden else 0.0 for token in range(64)])

    def forbid(tokens, logits):
        return logits + mask.astype(logits.dtype)

    def run(copy):
        model = _copying_model()
        kwargs = {"copy_draft": copy} if copy else {}
        generator = BatchGenerator(
            model, completion_batch_size=1, prefill_batch_size=1,
            prefill_step_size=32,
            self_mtp={"num_draft": 2, "persistent": True,
                      "segment_aware_live_tip": True, "segment_aware_cohort_size": 1},
            **kwargs,
        )
        generator.insert([PROMPT], max_tokens=[30], lane_rngs=[LaneRNG(5)],
                         logits_processors=[[forbid]],
                         self_mtp_configs=[{"sampling_temp": 0.0}])
        out = []
        for _ in range(200):
            _, responses = generator.next()
            out.extend(response.token for response in responses)
            if len(out) >= 30:
                break
        stats = dict(generator.scheduler_stats)
        generator.close()
        return out, stats

    baseline, _ = run(None)
    copied, stats = run({"enabled": True})
    assert copied == baseline
    assert forbidden not in copied
    assert stats["self_mtp_copy_rounds"] > 0
