"""Host-only lifecycle checks for the default-off native generator lane."""

from contextlib import contextmanager, nullcontext
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.qwen3_paged_candidate import Qwen3PackedCandidate
from mlx2.runtime.generate import BatchGenerator, GenerationBatch, StopSequenceMatcher
from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
from mlx2.runtime.paged_native_batch_lifecycle import NativeQueuedHandoffPreparation
from mlx2.runtime.paged_native_batch_lifecycle import (
    prepare_queued_native_first_response, run_research_native_queued_qwen3,
)
from mlx2.runtime.paged_request_transaction import CandidateRequest


def _owner(offset=3, generation=1):
    owner = object.__new__(NativeAtomicRequestOwner)
    owner.supported_planes = ("kv",)
    owner._enabled = True
    owner.snapshot = lambda: nullcontext(SimpleNamespace(
        revision="r1", generation=generation, offset=offset,
        layer_owners=(object(), object())))
    return owner


def _prepared(uid):
    return NativeQueuedHandoffPreparation(
        uid, "r1", 1, (7, 8, 9), object(),
        "native_continuation_not_installed")


def _healthy_writer():
    return SimpleNamespace(poisoned=False, pending_epochs=(),
                           ledger=SimpleNamespace(pending_count=0))


class _FakeContinuation:
    closed = 0
    stepped = 0

    def __init__(self, **values):
        self.uid = values["uid"]

    def next(self):
        self.stepped += 1
        return GenerationBatch.Response(
            self.uid, 11, None, "length", None, [7, 8, 9],
            mtp_receipt={"route": "native_qwen3_paged", "observed_used": True})

    def close(self):
        self.closed += 1

    def reap(self):
        pass

    @property
    def can_release(self):
        return bool(self.closed)


def test_native_install_is_default_off_and_refusals_keep_ordinary_queue(monkeypatch):
    from mlx2.runtime import paged_native_continuation

    monkeypatch.setattr(paged_native_continuation, "NativeQwen3Continuation",
                        _FakeContinuation)
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        owner = _owner()
        candidate = Qwen3PackedCandidate(None, SimpleNamespace(
            read_submissions=1, terminal_successes=1))
        args = (_prepared(uid), owner, candidate, RLock())
        assert generator.install_native_queued(*args)["reason"] == "native_serving_disabled"
        assert generator.install_native_queued(*args, permit_native=True,
                                               cancelled=lambda: True)["reason"] == "cancelled"
        assert generator._find_uids((uid,))[uid][0] == 0
        assert generator.install_native_queued(_prepared(uid + 1), owner, candidate,
                                               RLock(), permit_native=True)["reason"] == "queued_uid_drifted"
        assert generator.install_native_queued(*args, permit_native=True)["reason"] == "native_installed"
        assert generator._find_uids((uid,))[uid][0] == 4
        assert generator.install_native_queued(*args, permit_native=True)["reason"] == "native_preparation_incomplete"
        prompts, responses = generator.next()
        assert [response.uid for response in prompts] == [uid]
        assert len(responses) == 1 and responses[0].uid == uid
        assert responses[0].mtp_receipt["observed_used"]
        assert uid not in generator._find_uids((uid,))
    finally:
        generator.close()


def test_explicit_research_graph_b2_uses_one_live_generator_step(monkeypatch):
    from mlx2.runtime import paged_native_graph_group
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation

    candidate = SimpleNamespace(_research_staged_graph=True)
    lanes = []
    for uid in (7, 8):
        lane = object.__new__(NativeQwen3Continuation)
        lane.uid = uid
        lane.candidate = candidate
        lane.owner = SimpleNamespace(supported_planes=("kv",))
        lane.research_only = True
        lane.closed = False
        lane._first_logits = None
        lane._pending_token = uid
        lanes.append(lane)
    calls = []

    def fake_graph(pair):
        calls.append(tuple(lane.uid for lane in pair))
        return tuple(GenerationBatch.Response(
            lane.uid, lane.uid + 1, None, None, None, None,
            mtp_receipt={"route": "native_qwen3_paged_graph_research",
                         "selected": False, "observed_used": False},
            execution_width=2) for lane in pair)

    monkeypatch.setattr(paged_native_graph_group, "run_research_graph_b2", fake_graph)
    generator = BatchGenerator(None)
    try:
        generator._native_continuations.update((lane.uid, lane) for lane in lanes)
        prompts, responses = generator.next()
        assert not prompts and calls == [(7, 8)]
        assert [response.uid for response in responses] == [7, 8]
        assert all(response.execution_width == 2 and
                   not response.mtp_receipt["selected"] for response in responses)
    finally:
        generator._native_continuations.clear()
        generator.close()


def test_selected_b2_generator_requires_both_ready_lanes(monkeypatch):
    from mlx2.runtime import paged_native_graph_group
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation

    candidate = SimpleNamespace(_research_staged_graph=True, _serving_b2=True)
    lanes = []
    for uid in (7, 8):
        lane = object.__new__(NativeQwen3Continuation)
        lane.uid = uid
        lane.candidate = candidate
        lane.owner = SimpleNamespace(supported_planes=("kv",))
        lane.research_only = False
        lane.closed = False
        lane._first_logits = None
        lane._pending_token = uid
        lane.close = lambda: None
        lane.reap = lambda: None
        lanes.append(lane)
    monkeypatch.setattr(paged_native_graph_group, "run_research_graph_b2",
                        lambda pair: tuple(GenerationBatch.Response(
                            lane.uid, lane.uid + 1, None, None, None, None,
                            mtp_receipt={"route": "native_qwen3_paged_b2",
                                         "selected": True, "observed_used": True,
                                         "physical_dispatches": {
                                             "grouped_q1_writes": 2,
                                             "native_write_dispatches": 0,
                                             "q1_tile_dispatches": 2}},
                            execution_width=2) for lane in pair))
    generator = BatchGenerator(None)
    try:
        generator._native_continuations.update((lane.uid, lane) for lane in lanes)
        _, responses = generator.next()
        assert [response.uid for response in responses] == [7, 8]
        assert all(response.execution_width == 2 and
                   response.mtp_receipt["observed_used"] and
                   response.mtp_receipt["physical_dispatches"]["grouped_q1_writes"] == 2
                   for response in responses)
        generator.remove([8])
        _, responses = generator.next()
        assert not responses
        assert generator.take_lane_failures() == [
            {"uid": 7, "reason": "serving_b2_partner_missing"}]
    finally:
        generator._native_continuations.clear()
        generator.close()


def test_selected_b2_generator_emits_first_tokens_before_shared_ready_step(monkeypatch):
    from mlx2.runtime import paged_native_graph_group
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation

    candidate = SimpleNamespace(_research_staged_graph=True, _serving_b2=True)
    lanes = []
    for uid in (7, 8):
        lane = object.__new__(NativeQwen3Continuation)
        lane.uid = uid
        lane.candidate = candidate
        lane.owner = SimpleNamespace(supported_planes=("kv",))
        lane.research_only = False
        lane.closed = False
        lane._first_logits = object()
        lane._pending_token = None
        lane.close = lambda: None
        lane.reap = lambda: None

        def first_step(lane=lane):
            assert lane._first_logits is not None
            lane._first_logits = None
            lane._pending_token = lane.uid + 1
            return GenerationBatch.Response(
                lane.uid, lane.uid + 1, None, None, None, None,
                mtp_receipt={"route": "native_qwen3_paged_b2",
                             "selected": True, "observed_used": False})

        lane.next = first_step
        lanes.append(lane)
    paired = []

    def packed(pair):
        paired.append(tuple(lane.uid for lane in pair))
        return tuple(GenerationBatch.Response(
            lane.uid, lane.uid + 2, None, "length", None, None,
            mtp_receipt={"route": "native_qwen3_paged_b2",
                         "selected": True, "observed_used": True,
                         "native_read_delta": 2,
                         "physical_dispatches": {"grouped_q1_writes": 2,
                                                  "native_write_dispatches": 0,
                                                  "q1_tile_dispatches": 2}},
            execution_width=2) for lane in pair)

    monkeypatch.setattr(paged_native_graph_group, "run_research_graph_b2", packed)
    generator = BatchGenerator(None)
    try:
        generator._native_continuations.update((lane.uid, lane) for lane in lanes)
        _, first = generator.next()
        assert [response.uid for response in first] == [7, 8]
        assert all(response.mtp_receipt["observed_used"] is False for response in first)
        assert not generator.take_lane_failures() and not paired
        _, second = generator.next()
        assert [response.uid for response in second] == [7, 8]
        assert paired == [(7, 8)]
        assert all(response.execution_width == 2 and
                   response.mtp_receipt["observed_used"] for response in second)
        assert not generator.take_lane_failures()
    finally:
        generator._native_continuations.clear()
        generator.close()


def test_selected_b2_missing_partner_never_emits_even_first_token():
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation

    lane = object.__new__(NativeQwen3Continuation)
    lane.uid = 7
    lane.candidate = SimpleNamespace(_research_staged_graph=True, _serving_b2=True)
    lane.owner = SimpleNamespace(supported_planes=("kv",))
    lane.research_only = False
    lane.closed = False
    lane._first_logits = object()
    lane._pending_token = None
    lane.next = lambda: pytest.fail("orphaned B2 lane emitted first response")
    lane.close = lambda: None
    lane.reap = lambda: None
    generator = BatchGenerator(None)
    try:
        generator._native_continuations[7] = lane
        _, responses = generator.next()
        assert responses == []
        assert generator.take_lane_failures() == [
            {"uid": 7, "reason": "serving_b2_partner_missing"}]
    finally:
        generator._native_continuations.clear()
        generator.close()


def test_native_install_cancel_and_remove_close_lane_without_emitting(monkeypatch):
    from mlx2.runtime import paged_native_continuation

    class Tracked(_FakeContinuation):
        instances = []

        def __init__(self, **values):
            super().__init__(**values)
            self.instances.append(self)

    monkeypatch.setattr(paged_native_continuation, "NativeQwen3Continuation", Tracked)
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        result = generator.install_native_queued(
            _prepared(uid), _owner(), Qwen3PackedCandidate(None, SimpleNamespace(
                read_submissions=1, terminal_successes=1)), RLock(),
            permit_native=True)
        assert result["selected"] and not result["observed_used"]
        generator.remove([uid])
        assert Tracked.instances[0].closed == 1
        assert uid not in generator._find_uids((uid,))
        prompts, responses = generator.next()
        assert not prompts and not responses
    finally:
        generator.close()


def test_warm_apcv2_prefix_handoff_preserves_complete_prompt(monkeypatch):
    from mlx2.runtime import paged_native_continuation

    captured = {}

    class Warm(_FakeContinuation):
        def __init__(self, **values):
            captured.update(values)
            super().__init__(**values)

    monkeypatch.setattr(paged_native_continuation, "NativeQwen3Continuation", Warm)
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[3]], caches=[[]], all_tokens=[[1, 2]])[0]
        owner = _owner(offset=3)
        candidate = Qwen3PackedCandidate(None, SimpleNamespace(
            read_submissions=1, terminal_successes=1))
        prepared = NativeQueuedHandoffPreparation(
            uid, "r1", 1, (1, 2, 3), object(),
            "native_continuation_not_installed")
        result = generator.install_native_queued(
            prepared, owner, candidate, RLock(), permit_native=True)
        assert result["selected"]
        assert captured["prompt_tokens"] == (1, 2, 3)
        assert captured["apcv2_restored_tokens"] == 2
    finally:
        generator.close()


def test_cancel_after_first_native_response_retains_owner_until_retired(monkeypatch):
    from mlx2.runtime import paged_native_continuation

    class Delayed(_FakeContinuation):
        instances = []

        def __init__(self, **values):
            super().__init__(**values)
            self.released = False
            self.instances.append(self)

        @property
        def can_release(self):
            return self.released

        def next(self):
            self.stepped += 1
            return GenerationBatch.Response(
                self.uid, 11, None, None, None, None,
                mtp_receipt={"route": "native_qwen3_paged", "observed_used": True})

    monkeypatch.setattr(paged_native_continuation, "NativeQwen3Continuation", Delayed)
    generator = BatchGenerator(None)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]])[0]
        result = generator.install_native_queued(
            _prepared(uid), _owner(), Qwen3PackedCandidate(None, SimpleNamespace(
                read_submissions=1, terminal_successes=1)), RLock(),
            permit_native=True)
        assert result["selected"]
        _, responses = generator.next()
        assert len(responses) == 1 and responses[0].finish_reason is None
        generator.remove([uid])
        lane = Delayed.instances[0]
        assert lane.closed == 1 and lane in generator._native_retiring
        assert uid not in generator._find_uids((uid,))
        lane.released = True
        generator._reap_native_retiring()
        assert not generator._native_retiring
    finally:
        generator.close()


def test_generator_close_retains_native_owner_when_close_raises(monkeypatch):
    from mlx2.runtime import generate

    class Poisoned:
        def close(self):
            raise RuntimeError("terminal pending")

        def reap(self):
            raise RuntimeError("terminal pending")

        @property
        def can_release(self):
            return False

    lane = Poisoned()
    generator = BatchGenerator(None)
    generator._native_continuations[7] = lane
    generator.close()
    assert lane in generate._NATIVE_ORPHANED_RETIREMENTS
    generate._NATIVE_ORPHANED_RETIREMENTS.remove(lane)


def test_poisoned_native_reader_fails_one_lane_and_retains_cleanup_owner():
    class Failed:
        def next(self):
            raise RuntimeError("native public arena is poisoned")

        def close(self):
            raise RuntimeError("terminal proof pending")

        def reap(self):
            raise RuntimeError("terminal proof pending")

        @property
        def can_release(self):
            return False

    class Healthy:
        def next(self):
            return GenerationBatch.Response(
                8, 2, None, "length", None, [1, 2],
                mtp_receipt={"route": "native_qwen3_paged", "observed_used": True})

        def close(self):
            pass

        def reap(self):
            pass

        @property
        def can_release(self):
            return True

    generator = BatchGenerator(None)
    failed = Failed()
    generator._native_continuations.update({7: failed, 8: Healthy()})
    try:
        prompts, responses = generator.next()
        assert not prompts and [response.uid for response in responses] == [8]
        assert generator.take_lane_failures() == [
            {"uid": 7, "reason": "native public arena is poisoned"}]
        assert failed in generator._native_retiring
    finally:
        from mlx2.runtime import generate
        generator.close()
        generate._NATIVE_ORPHANED_RETIREMENTS.remove(failed)


def test_ordinary_prefill_drops_native_only_rng_reference():
    generator = BatchGenerator(None)
    try:
        rng = object()
        uid = generator.insert([[7, 8, 9]], caches=[[]], lane_rngs=[rng])[0]
        assert generator._native_lane_rngs[uid] is rng
        generator._make_batch(1)
        assert uid not in generator._native_lane_rngs
    finally:
        generator.close()


def test_native_continuation_samples_and_forwards_accepted_token_without_device(monkeypatch):
    from mlx2.runtime import generate, paged_native_continuation

    numpy_mx = SimpleNamespace(
        float32=np.float32, uint32=np.uint32, array=np.array,
        isfinite=np.isfinite, eval=lambda *_: None,
        logsumexp=lambda x, axis, keepdims: np.log(
            np.sum(np.exp(x), axis=axis, keepdims=keepdims)))
    monkeypatch.setattr(paged_native_continuation, "mx", numpy_mx)
    monkeypatch.setattr(generate, "mx", numpy_mx)

    events = []

    class Branch:
        layers = (object(), object())

        def prepare(self, accepted):
            events.append(("prepare", accepted))
            return self

        def publish(self):
            events.append(("publish",))

        def rollback(self):
            events.append(("rollback",))

    owner = _owner()
    reader = {"open": False, "acquired": 0, "released": 0}

    @contextmanager
    def live_snapshot():
        assert not reader["open"]
        reader["open"] = True
        reader["acquired"] += 1
        try:
            yield SimpleNamespace(revision="r1", generation=1, offset=3,
                                  layer_owners=(object(), object()))
        finally:
            reader["open"] = False
            reader["released"] += 1

    owner.snapshot = live_snapshot
    owner.begin = lambda request: Branch()
    owner.close = lambda: events.append(("close",))
    owner.reap_retired = lambda: events.append(("reap", reader["open"]))
    owner.reap_quarantine = lambda: 0
    backend = SimpleNamespace(read_submissions=2, terminal_successes=2,
                              writer=_healthy_writer())
    candidate = Qwen3PackedCandidate(SimpleNamespace(layers=(object(), object())), backend)

    def forward(lanes, **_):
        assert reader["open"], "native forward lost its public reader lease"
        backend.read_submissions += 2
        backend.terminal_successes += 2
        return (np.array([[0.0, 2.0, 0.0]], dtype=np.float32),
                {"simulated_or_backend_reads": 2})

    candidate.forward = forward
    contexts = []
    rng = SimpleNamespace(draws=0)

    def processor(context, logits):
        contexts.append(tuple(int(token) for token in context))
        return logits

    def sampler(logprobs):
        rng.draws += 1
        return np.argmax(logprobs, axis=-1)

    continuation = paged_native_continuation.NativeQwen3Continuation(
        uid=4, revision="r1", prompt_tokens=(7, 8, 9),
        first_logits=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        owner=owner, candidate=candidate, maximum=3,
        sampler=sampler, processors=[processor],
        matcher=StopSequenceMatcher([[1]]), lane_rng=rng,
    )
    first = continuation.next()
    assert not reader["open"] and reader["acquired"] == reader["released"] == 1
    assert first.token == 2 and first.finish_reason is None
    assert continuation.tokens == [7, 8, 9]
    second = continuation.next()
    assert not reader["open"] and reader["acquired"] == reader["released"] == 2
    assert second.token == 1 and second.finish_reason == "stop"
    assert second.all_tokens == [7, 8, 9, 2]
    assert contexts == [(7, 8, 9), (7, 8, 9, 2)]
    assert second.rng_draws == 2 and second.lane_rng is rng
    assert second.mtp_receipt["native_read_calls"] == 4
    assert second.mtp_receipt["terminal_successes"] == 4
    assert second.mtp_receipt["native_reader_generation"] == 1
    assert second.mtp_receipt["native_reader_offset"] == 3
    assert second.mtp_receipt["native_reader_lease"] == "held_through_token_step"
    assert events.count(("reap", False)) == 2
    assert events[1:3] == [("prepare", 1), ("publish",)]
    continuation.close()
    assert events[-2:] == [("close",), ("reap", False)]


def test_native_live_reader_drift_fails_before_sampling(monkeypatch):
    from mlx2.runtime.paged_native_continuation import NativeQwen3Continuation

    owner = _owner(offset=2)
    candidate = Qwen3PackedCandidate(
        SimpleNamespace(layers=(object(), object())),
        SimpleNamespace(read_submissions=2, terminal_successes=2))
    sampled = []
    lane = NativeQwen3Continuation(
        uid=4, revision="r1", prompt_tokens=(7, 8, 9),
        first_logits=np.array([0.0, 1.0], dtype=np.float32),
        owner=owner, candidate=candidate, maximum=1,
        sampler=lambda _: sampled.append(True), processors=[],
        matcher=StopSequenceMatcher([]))
    with pytest.raises(RuntimeError, match="public reader state drifted"):
        lane.next()
    assert not sampled


@pytest.mark.parametrize("prefix", [(), (5, 6)])
def test_research_permit_executes_private_prompt_without_serving_selection(monkeypatch, prefix):
    from mlx2.runtime import generate, paged_native_continuation

    numpy_mx = SimpleNamespace(
        float32=np.float32, uint32=np.uint32, array=np.array,
        isfinite=np.isfinite, eval=lambda *_: None,
        argmax=np.argmax,
        logsumexp=lambda x, axis, keepdims: np.log(
            np.sum(np.exp(x), axis=axis, keepdims=keepdims)),
        stream=lambda _: nullcontext(),
    )
    public = SimpleNamespace(revision="r1", generation=0, offset=len(prefix),
                             layer_owners=(object(), object()))
    owner = _owner(offset=len(prefix), generation=0)
    owner.snapshot = lambda: nullcontext(public)
    # The real owner exposes a read-only retirement proof. This host-only
    # fixture replaces that proof for the duration of the test.
    monkeypatch.setattr(NativeAtomicRequestOwner, "fully_retired",
                        property(lambda _: True), raising=False)
    owner.close = lambda: None
    owner.reap_retired = lambda: 0
    owner.reap_quarantine = lambda: 0

    class Branch:
        layers = (object(), object())

        def prepare(self, accepted):
            assert accepted == 3
            return self

        def publish(self):
            public.generation, public.offset = 1, len(prefix) + 3

        def rollback(self):
            pass

    owner.begin = lambda request: Branch()
    backend = SimpleNamespace(read_submissions=0, terminal_successes=0,
                              writer=_healthy_writer())
    model = SimpleNamespace(layers=(object(), object()))
    candidate = Qwen3PackedCandidate(model, backend)

    def forward(lanes, **_):
        backend.read_submissions += 2
        backend.terminal_successes += 2
        return np.array([[0., 2.], [0., 2.], [0., 2.]], dtype=np.float32), {}

    candidate.forward = forward
    generator = BatchGenerator(model, max_tokens=1)
    try:
        uid = generator.insert([[7, 8, 9]], caches=[[]],
                               all_tokens=[list(prefix)])[0]
        monkeypatch.setattr(paged_native_continuation, "mx", numpy_mx)
        monkeypatch.setattr(generate, "mx", numpy_mx)
        lock = RLock()
        request = CandidateRequest(uid, "r1", 3, ("kv",))
        assert run_research_native_queued_qwen3(
            generator, lock, request, owner, candidate).reason == "research_permit_required"
        probe = run_research_native_queued_qwen3(
            generator, lock, request, owner, candidate, research_permit=True)
        assert probe.research_executed and not probe.serving_selected
        assert not probe.serving_observed_used
        prepared = prepare_queued_native_first_response(generator, lock, owner, probe)
        installed = generator.install_native_queued(
            prepared, owner, candidate, lock, permit_native=True,
            research_only=True)
        assert installed["reason"] == "native_installed"
        assert not installed["selected"] and installed["research_only"]
        prompts, responses = generator.next()
        assert len(prompts) == len(responses) == 1, generator.take_lane_failures()
        receipt = responses[0].mtp_receipt
        assert receipt["research_executed"]
        assert not receipt["selected"] and not receipt["observed_used"]
        assert receipt["native_read_calls"] == receipt["terminal_successes"] == 2
        assert receipt["apcv2_restored_tokens"] == len(prefix)
    finally:
        generator.close()
