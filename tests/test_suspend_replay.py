"""Preempt-and-replay under memory pressure (item 8, CPU only).

Two harnesses: a tiny real hybrid model through the real ``BatchGenerator`` and
APCv2 (replays must be token-identical to an uninterrupted run), and a scripted
fake batch for the scheduling state machine (stall trigger, admission pause,
replay cap, drain, non-preemptible classes).
"""

import threading
import time
from types import SimpleNamespace as NS

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2 import memory, serving
from mlx2.memory_preemption import (
    MAX_REPLAYS_PER_JOB,
    choose_preemption_victim,
    decode_replay_block,
    memory_preemption_policy,
    preemption_block,
    replay_lane_rng,
)
from mlx2.prometheus import _ENGINE_EVENTS, _OPTIONAL_ENGINE_EVENTS
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.runtime.os_memory import PressureLevel
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.serving import Job, ServingEngine

POLICY = {"memory_preemption": {"enabled": True}}


# --- policy and eligibility ------------------------------------------------


def test_policy_defaults_off_and_validates():
    assert memory_preemption_policy(None) == {
        "enabled": False, "stall_seconds": 60.0, "on_pressure": True,
    }
    assert memory_preemption_policy({"enabled": True, "stall_seconds": 5})[
        "stall_seconds"
    ] == 5.0
    for bad in (
        [], {"enabled": 1}, {"on_pressure": "yes"}, {"stall_seconds": 0},
        {"stall_seconds": True}, {"stall_seconds": float("inf")}, {"cap": 3},
    ):
        with pytest.raises(ValueError):
            memory_preemption_policy(bad)


def _job(**overrides):
    job = Job(request=dict(overrides.pop("request", {})))
    job.preemption_prompt = [1, 2, 3]
    job.generated_token_ids = []
    for name, value in overrides.items():
        setattr(job, name, value)
    return job


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({}, None),
        # Decode has started: no route can rebuild decode-produced state by
        # prefilling it, so the lane is refused rather than replayed.
        ({"completion_tokens": 4}, "decode_state_not_reconstructible"),
        ({"preemption_prompt": None}, "untracked"),
        ({"preemptions": MAX_REPLAYS_PER_JOB}, "replay_cap"),
        ({"request": {"batch_cohort": {"id": "c"}}}, "batch_cohort"),
        ({"fanout_group": "g"}, "parallel_samples"),
        ({"parallel_sample": True}, "parallel_samples"),
        ({"cache_capsule": object()}, "cache_capsule"),
        ({"request": {"_mlx2_prefill_inputs": object()}}, "multimodal"),
        ({"request": {"_mlx2_media_fingerprint": "m"}}, "multimodal"),
        ({"approximate_kv_applied": True}, "approximate"),
        ({"spomin_receipt": {"status": "applied"}}, "approximate"),
        # Decode-only blocks do not stop a prefill-phase restart.
        ({"decode_replay_block": "steering"}, None),
        ({"decode_replay_block": "steering", "completion_tokens": 1}, "steering"),
        # ... and with no per-job reason recorded, the general one applies.
        ({"completion_tokens": 1}, "decode_state_not_reconstructible"),
    ],
)
def test_preemption_eligibility(overrides, reason):
    assert preemption_block(_job(**overrides)) == reason


def test_decode_replay_is_blocked_on_every_route():
    """Replay rebuilds delivered tokens by prefilling; that is not decode.

    Measured on CPU: next-token logits after eight tokens produced by decode
    differ from the same eight replayed as one prefill by up to 1.6e-06
    (dense) and 7.7e-07 (hybrid GDN/MTP).  Greedy hides it until the
    perturbation exceeds the top-2 margin, then the argmax flips.  The GPU
    requeue saw exactly that: the ordinary arm diverged from the first
    replayed token, the self-MTP arm matched by margin luck.
    """
    assert decode_replay_block() == "decode_state_not_reconstructible"
    # A lane that has decoded nothing is still preemptible: there is no
    # decode-produced state to rebuild.
    assert preemption_block(_job()) is None
    assert preemption_block(_job(completion_tokens=1)) == decode_replay_block()


def test_victim_is_youngest_preemptible_by_first_admission():
    old = _job(started=1.0)
    young_blocked = _job(started=3.0, request={"batch_cohort": {"id": "c"}})
    young = _job(started=2.0, preemptions=1)
    assert choose_preemption_victim({0: old, 1: young_blocked, 2: young}) == 2
    assert choose_preemption_victim({1: young_blocked}) is None
    # A stall never parks work older than the lane it is making room for.
    lanes = {0: old, 2: young}
    assert choose_preemption_victim(lanes, stalled=[2]) == 2
    young.preemptions = MAX_REPLAYS_PER_JOB
    assert choose_preemption_victim(lanes, stalled=[2]) is None
    assert choose_preemption_victim(lanes) == 0


def test_replay_rng_continues_the_lane_stream():
    lane = LaneRNG(11)
    for _ in range(5):
        lane.next_key()
    replay = replay_lane_rng(LaneRNG, 11, 5)
    assert replay.draws == 5
    assert mx.array_equal(replay.next_key(), lane.next_key()).item()


def test_counters_are_exported_and_absent_from_default_status():
    names = {
        "memory_preemptions", "memory_preemptions_stall",
        "memory_preemptions_pressure", "memory_preemptions_fault",
        "preempted_replays",
        "memory_preemption_drain_cancellations",
    }
    # Default-off: the mapping lives in the optional table, so a server that
    # never enabled preemption emits no zero series for these.
    assert names <= set(_OPTIONAL_ENGINE_EVENTS)
    assert not (names & set(_ENGINE_EVENTS))
    assert {
        _OPTIONAL_ENGINE_EVENTS[name][0] for name in names
    } == {"memory_preemption"}


# --- tiny real model: replay is token-identical ----------------------------


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=3,
        full_attention_interval=4, mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=256,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


class Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"{int(token)} "

    def finalize(self):
        pass


class Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


class Tokenizer:
    vocab_size = 128
    eos_token_ids = []

    @property
    def detokenizer(self):
        return Detok()


class HistoryBias:
    """A history-pure processor: its mask is a function of the token history."""

    history_pure = True

    def __init__(self, prompt_length):
        self.prompt_length = prompt_length

    def __call__(self, tokens, logits):
        generated = int(tokens.shape[-1]) - self.prompt_length
        if generated % 3:
            return logits
        # Strong enough to decide the token, so a wrong prompt length shows.
        bias = mx.zeros((logits.shape[-1],))
        bias[(generated * 7 + 1) % logits.shape[-1]] = 30.0
        return logits + bias


def make_adapter(model, *, processors=None):
    class Adapter:
        max_context = 256
        identity = {"fingerprint": "tiny-hybrid"}
        environment = {}
        layout = "tiny-layout"
        tokenizer = Tokenizer()

        def __init__(self, _path):
            self.model = model

        def profile_name(self, _mtp):
            return "tiny-ordinary"

        def execution_config(self, **_kwargs):
            return {"num_draft": 0}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    if processors is not None:
        Adapter.request_logits_processors = (
            lambda self, request, prompt_length: processors(request, prompt_length)
        )
    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def collect(job):
    tokens = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            tokens.extend(int(t) for t in event["delta"]["content"].split())
        if "finish_reason" in event:
            return tokens, event["receipt"]


A_PROMPT = list(range(1, 30))
B_PROMPT = list(range(40, 61))


def run_pair(model, *, trigger=None, processors=None, sampling=None, policy=POLICY):
    """Run A (long) and B together; ``trigger(b_job)`` says when to press."""
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, processors=processors),
        qualification_mode=True, mtp=False, max_lanes=2, prefill_step=8,
        execution_policy=policy,
    )
    try:
        assert engine.ready.wait(60), engine.error
        sampling = sampling or {"temperature": 0}
        a = engine.submit({"tokens": A_PROMPT, "max_tokens": 24, **sampling})
        while a.uid is None and not a.events.qsize():
            time.sleep(0.001)
        b = engine.submit({"tokens": B_PROMPT, "max_tokens": 12, **sampling})
        if trigger is not None:
            fired = threading.Event()

            def level():
                if not fired.is_set() and trigger(b):
                    return PressureLevel.CRITICAL
                if engine.counts["memory_preemptions"]:
                    fired.set()
                return PressureLevel.NORMAL

            engine.memory_pressure_level = level
        a_tokens, a_receipt = collect(a)
        b_tokens, b_receipt = collect(b)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert not engine.error
    return (a_tokens, b_tokens), (a_receipt, b_receipt), counts


def decode_trigger(after):
    return lambda job: job.uid is not None and job.completion_tokens >= after


def prefill_trigger(job):
    return job.uid is not None and job.completion_tokens == 0


def test_greedy_prefill_replay_is_token_identical(host):
    """Prefill-phase replay is exact by construction: nothing was decoded."""
    model = tiny_model()
    reference, ref_receipts, _ = run_pair(model)
    assert "preemption" in ref_receipts[1] and ref_receipts[1]["preemption"] is None
    outputs, receipts, counts = run_pair(model, trigger=prefill_trigger)
    assert outputs == reference
    assert len(outputs[1]) == 12
    assert counts["memory_preemptions"] == counts["memory_preemptions_pressure"] == 1
    assert counts["preempted_replays"] == 1
    assert counts["admitted"] == 2  # a replay is not a second admission
    preemption = receipts[1]["preemption"]
    assert preemption["replays"] == 1
    (event,) = preemption["events"]
    assert event["trigger"] == "pressure" and event["phase"] == "prefill"
    assert receipts[1]["prompt_tokens"] == len(B_PROMPT)
    assert receipts[1]["completion_tokens"] == 12
    assert receipts[0]["preemption"] is None


def test_decode_phase_lane_is_never_preempted(host):
    """Pressure at decode leaves the lane alone rather than replaying it."""
    model = tiny_model()
    reference, _ref_receipts, _ = run_pair(model)
    outputs, receipts, counts = run_pair(model, trigger=decode_trigger(4))
    # Untouched, so still exact -- but by not preempting, not by replaying.
    assert outputs == reference
    assert counts["memory_preemptions"] == 0
    assert counts["preempted_replays"] == 0
    assert receipts[1]["preemption"] is None


@pytest.mark.parametrize("after_tokens", [0])
def test_injected_fault_preempts_one_lane_for_qualification(host, after_tokens):
    """``mlx_fault`` is how a qualification harness observes the mechanism."""
    model = tiny_model()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model), qualification_mode=True,
        mtp=False, max_lanes=2, prefill_step=8, execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 12, "temperature": 0}
        reference, _ = collect(engine.submit(dict(request)))
        tokens, receipt = collect(engine.submit(dict(
            request, mlx_fault={"kind": "memory_preempt", "after_tokens": after_tokens},
        )))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert not engine.error
    assert tokens == reference
    assert counts["memory_preemptions_fault"] == counts["preempted_replays"] == 1
    (event,) = receipt["preemption"]["events"]
    assert event["trigger"] == "fault"
    assert event["phase"] == "prefill"
    assert event["committed_tokens"] >= after_tokens


class _StatefulProcessor:
    """No ``history_pure`` attribute, so P5 assumes it is stateful."""

    def __call__(self, tokens, logits):
        return logits


@pytest.mark.parametrize("after_tokens", [1, 5, 8])
def test_a_nonzero_threshold_is_evaluated_then_declined(host, after_tokens):
    """The threshold works; the decode phase is what refuses.

    The two are worth separating.  `never_reached` would mean the trigger
    never ran at N -- a defect.  Naming the decode block means the trigger
    ran exactly at N and the eligibility rule refused, which is the intended
    behaviour now that decode-phase replay is known to be inexact.
    """
    model = tiny_model()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model), qualification_mode=True,
        mtp=False, max_lanes=2, prefill_step=8, execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 24, "temperature": 0}
        reference, _ = collect(engine.submit(dict(request)))
        tokens, receipt = collect(engine.submit(dict(
            request, mlx_fault={"kind": "memory_preempt", "after_tokens": after_tokens},
        )))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert not engine.error
    # Never preempted, so trivially unchanged -- and the run says so.
    assert tokens == reference
    assert counts["memory_preemptions_fault"] == counts["preempted_replays"] == 0
    assert counts["memory_preemption_fault_declined"] == 1
    assert counts["memory_preemption_fault_unfired"] == 1
    assert receipt["preemption"]["fault_unfired"] == "decode_state_not_reconstructible"


def test_a_fault_that_cannot_fire_is_reported_not_swallowed(host):
    """An ineligible decode lane declines the fault -- loudly.

    The eligibility rule is correct: a lane that cannot replay exactly must
    not be preempted.  What must not happen is the injected fault silently
    doing nothing, because a harness then cannot tell "mechanism observed"
    from "mechanism declined" and writes a gate that passes vacuously.
    """
    model = tiny_model()
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(
            model, processors=lambda request, prompt_length: [_StatefulProcessor()]
        ),
        qualification_mode=True, mtp=False, max_lanes=2, prefill_step=8,
        execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 24, "temperature": 0}
        _tokens, receipt = collect(engine.submit(dict(
            request, mlx_fault={"kind": "memory_preempt", "after_tokens": 8},
        )))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert not engine.error
    # Declined, so no preemption happened ...
    assert counts["memory_preemptions_fault"] == counts["preempted_replays"] == 0
    # ... and that is visible rather than indistinguishable from success.
    assert counts["memory_preemption_fault_declined"] == 1
    assert counts["memory_preemption_fault_unfired"] == 1
    assert receipt["preemption"]["replays"] == 0
    assert receipt["preemption"]["fault_unfired"] == "decode_state_not_reconstructible"


def _bench_was_preempted():
    """Import the bench's gate predicate without running the bench."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_suspend_replay.py"
    spec = importlib.util.spec_from_file_location("_bench_suspend_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.was_preempted


def test_gate_is_not_satisfied_by_a_declined_fault(host):
    """The "was preempted" gate must reject the evidence that it was not.

    Recording the decline in the ``preemption`` receipt made a gate written as
    ``preemption is not None`` pass on a request that was never preempted:
    making a thing observable and making it asserted are different steps, and
    the first can quietly undo the second.  This drives a real declined fault
    through the engine and asserts the bench predicate rejects it.
    """
    was_preempted = _bench_was_preempted()
    model = tiny_model()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model), qualification_mode=True,
        mtp=False, max_lanes=2, prefill_step=8, execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 24, "temperature": 0}
        _tokens, declined = collect(engine.submit(dict(
            request, mlx_fault={"kind": "memory_preempt", "after_tokens": 8},
        )))
    finally:
        engine.close()
    # The field is populated -- which is exactly why presence is not the test.
    assert declined["preemption"] is not None
    assert not was_preempted(declined)

    # And a real preemption still satisfies it.
    model = tiny_model()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model), qualification_mode=True,
        mtp=False, max_lanes=2, prefill_step=8, execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        request = {"tokens": B_PROMPT, "max_tokens": 24, "temperature": 0}
        _tokens, fired = collect(engine.submit(dict(
            request, mlx_fault={"kind": "memory_preempt", "after_tokens": 0},
        )))
    finally:
        engine.close()
    assert was_preempted(fired)
    assert not was_preempted({"preemption": None})


def test_sampled_prefill_replay_is_token_identical(host):
    """Sampled lanes replay exactly in prefill; the lane RNG carries over."""
    model = tiny_model()
    sampling = {"temperature": 0.9, "seed": 1234}
    reference, _, _ = run_pair(model, sampling=sampling)
    outputs, receipts, counts = run_pair(
        model, trigger=prefill_trigger, sampling=sampling
    )
    assert counts["preempted_replays"] == 1
    assert receipts[1]["preemption"]["events"][0]["phase"] == "prefill"
    assert outputs == reference


def test_processor_lanes_replay_in_prefill_and_never_in_decode(host):
    """P5 no longer decides preemptibility; the decode phase does.

    `history_pure` used to separate replayable decode lanes from
    unreplayable ones.  Decode replay is now refused for every lane, so the
    distinction no longer gates anything here: both processor kinds replay
    in prefill and neither is touched in decode.
    """
    model = tiny_model()

    def pure(request, prompt_length):
        return [HistoryBias(prompt_length)]

    def stateful(request, prompt_length):
        return [lambda tokens, logits: logits]

    for processors in (pure, stateful):
        reference, _, _ = run_pair(model, processors=processors)
        outputs, receipts, counts = run_pair(
            model, trigger=prefill_trigger, processors=processors
        )
        assert counts["preempted_replays"] == 1
        assert outputs == reference
        assert receipts[1]["preemption"]["events"][0]["phase"] == "prefill"

        outputs, receipts, counts = run_pair(
            model, trigger=decode_trigger(3), processors=processors
        )
        assert counts["memory_preemptions"] == 0
        assert outputs == reference
        assert receipts[1]["preemption"] is None


def test_default_off_receipts_and_settings_are_unchanged(host):
    model = tiny_model()
    _, receipts, counts = run_pair(model, policy=None)
    assert all("preemption" not in receipt for receipt in receipts)
    assert "memory_preemptions" not in counts
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model), qualification_mode=True,
        mtp=False, execution_policy=POLICY,
    )
    try:
        assert engine.ready.wait(60), engine.error
        assert engine.status()["settings"]["memory_preemption"]["enabled"] is True
    finally:
        engine.close()


def test_refuses_approximate_kv_combination(host):
    from mlx2.runtime.approximate_kv import standard_kv_quantization_operations

    adapter = make_adapter(tiny_model())
    adapter.approximate_kv_operations = (
        lambda self: standard_kv_quantization_operations(group_size=32)
    )
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=1, execution_policy=POLICY,
        approximate_kv={"operation": "kv_k8v4", "enabled": True},
    )
    engine.thread.join(30)
    assert "incompatible" in (engine.error or "")


# --- scripted batch: scheduling state machine --------------------------------


@pytest.fixture
def scripted(monkeypatch):
    """Fake APC/batch.  ``state['progress']`` names request tags whose lanes
    emit a token per cycle; every other lane stalls."""
    state = dict(lookups=[], inserts=[], branches=[], progress=set(), tags={})

    class Branch(list):
        def __init__(self, tokens):
            super().__init__()
            self.tokens = tokens
            self.closed = 0
            state["branches"].append(self)

        def close(self):
            self.closed += 1

    class APC:
        def __init__(self, **kw):
            self.apc_stats = {}

        def key(self, *a, **kw):
            return "key"

        def lookup(self, key, tokens, **kw):
            state["lookups"].append(list(tokens))
            return NS(cache=Branch(list(tokens)), cached_tokens=len(tokens) - 1,
                      remaining_tokens=[tokens[-1]], sidecar=None,
                      miss_reason=None, retention_role=None)

        def store(self, *a, **kw):
            pass

        def spill_idle_entries(self):
            pass

        def evict_oldest_unleased(self):
            return False

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *a, **kw):
            self.lanes = {}
            self.uid = 0

        def insert(self, prompts, max_tokens, all_tokens, **kw):
            uid = self.uid
            self.uid += 1
            tokens = list(all_tokens[0]) + list(prompts[0])
            state["inserts"].append((tokens, max_tokens[0]))
            self.lanes[uid] = dict(
                tag=state["tags"][tuple(tokens[:3])], left=max_tokens[0],
                history=list(tokens), prefilled=False,
            )
            return [uid]

        def next(self):
            # Prefill always completes; only decode is gated by ``progress``.
            prompts = []
            for uid, lane in self.lanes.items():
                if not lane["prefilled"]:
                    lane["prefilled"] = True
                    prompts.append(NS(uid=uid, end_of_prompt=True))
            responses = []
            for uid, lane in list(self.lanes.items()):
                if lane["tag"] not in state["progress"]:
                    continue
                token = 100 + len(lane["history"])
                lane["history"].append(token)
                lane["left"] -= 1
                done = lane["left"] == 0
                responses.append(NS(
                    uid=uid, token=token, execution_width=len(self.lanes),
                    finish_reason="length" if done else None, mtp_state=None,
                    all_tokens=list(lane["history"]), prompt_cache=[],
                    mtp_receipt=None, logprobs=None,
                ))
                if done:
                    del self.lanes[uid]
            time.sleep(0.002)
            return prompts, responses

        def pop_prompt_boundary(self, uid):
            return None

        def remove(self, uids):
            for uid in uids:
                self.lanes.pop(uid, None)

        def close(self):
            pass

    class Adapter:
        max_context = 2000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=1000, eos_token_ids=[], detokenizer=None)

        def __init__(self, path):
            Adapter.tokenizer.detokenizer = Detok()

        def profile_name(self, mtp):
            return "fake"

        def execution_config(self, **kw):
            return {"num_draft": 0}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, request):
            return Parser()

        def request_logits_processors(self, request, *, prompt_length):
            # Opt in per request: P5 (item 12) marks the built-in processors
            # history_pure, including logit_bias, so a test that needs a
            # genuinely unreplayable lane must supply one.
            return (
                [_StatefulProcessor()] if request.get("_stateful_processor") else ()
            )

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engines = []

    def start(stall_seconds=0.15, **kwargs):
        engine = ServingEngine(
            "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
            execution_policy={"memory_preemption": {
                "enabled": True, "stall_seconds": stall_seconds,
            }},
            **kwargs,
        )
        engines.append(engine)
        assert engine.ready.wait(5), engine.error
        return engine

    def submit(engine, tag, *, max_tokens=4, **request):
        tokens = [len(state["tags"]) * 10 + i for i in (1, 2, 3)]
        state["tags"][tuple(tokens)] = tag
        return engine.submit({"tokens": tokens, "max_tokens": max_tokens,
                              "temperature": 0, **request})

    yield state, start, submit
    for engine in engines:
        engine.close()


def consume(job):
    """Drain a long-running lane's events so its bounded queue never fills."""
    seen = []

    def run():
        while True:
            event = job.events.get()
            seen.append(event)
            if "error" in event or "finish_reason" in event:
                return

    threading.Thread(target=run, daemon=True).start()
    return seen


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


def test_stall_preempts_youngest_and_pauses_admission_until_blocker_progresses(scripted):
    state, start, submit = scripted
    engine = start(stall_seconds=0.5, max_lanes=2)
    # Nothing decodes while the stall deadline runs.  Letting ``a`` decode
    # here made the test racy: with 150 tokens at ~2ms per cycle it could
    # finish and leave before the deadline, dropping the lane count to one,
    # and preemption needs more than one active lane.  Both lanes now sit in
    # the prefill phase (decode-phase lanes are refused outright), so the
    # victim is deterministically the youngest, ``b``.
    state["progress"] = set()
    a = submit(engine, "a", max_tokens=150)
    wait_for(lambda: a.uid is not None)
    b = submit(engine, "b", max_tokens=6)
    wait_for(lambda: b.uid is not None)
    wait_for(lambda: engine.counts["memory_preemptions"] == 1, timeout=10)
    assert b.preempted and b.preemptions == 1
    committed = list(b.generated_token_ids)
    assert committed == []
    # The replay prefix is leased: prompt + delivered tokens, not closed.
    assert state["lookups"][-1] == [11, 12, 13] + committed
    lease = state["branches"][-1]
    assert lease.closed == 0 and b.cache_branch is lease
    # Admission is paused: c queues behind the pending replay.  Rendezvous on
    # serving cycles actually elapsing rather than on a fixed sleep -- a sleep
    # races the loop under load and, when it loses, asserts that c has no lane
    # for the wrong reason.
    c = submit(engine, "c", max_tokens=1)
    cycles = engine.counts["cycles"]
    wait_for(lambda: engine.counts["cycles"] >= cycles + 3, timeout=10)
    assert len(state["inserts"]) == 2 and c.uid is None
    assert not [e for e in list(a.events.queue) if "error" in e]
    # The stalled blocker progresses, which releases the pending replay.
    state["progress"] = {"a"}
    wait_for(lambda: engine.counts["preempted_replays"] == 1, timeout=10)
    # Then b decodes its replay and c attaches behind it.
    state["progress"] = {"a", "b", "c"}
    for job in (a, b, c):
        tokens, receipt = collect(job)
    replay_tokens, replay_max = state["inserts"][2]
    assert replay_tokens == [11, 12, 13] + committed
    assert replay_max == 6 - len(committed)
    assert state["inserts"][3][0][:3] == [21, 22, 23]
    assert b.completion_tokens == 6 and lease.closed == 1
    # The replay attached from the lease taken at preemption (whatever APCv2
    # returned: a prompt boundary today, a rolling checkpoint with P1) and
    # did not look up again.
    assert [lookup[:3] for lookup in state["lookups"]].count([11, 12, 13]) == 2
    assert engine.counts["memory_preemptions_stall"] == 1
    assert engine.counts["preempted_replays"] == 1


def test_replay_cap_then_todays_429(scripted):
    state, start, submit = scripted
    engine = start(max_lanes=2)
    state["progress"] = {"a"}
    a = submit(engine, "a", max_tokens=1900)
    a_events = consume(a)
    wait_for(lambda: a.uid is not None)
    # b never decodes: it is the youngest preemptible lane and yields itself,
    # replays, stalls again, and after the cap fails as it does today.
    b = submit(engine, "b", max_tokens=4)
    result = b.events.get(timeout=10)
    # ``a`` must still be decoding: a blocker that exhausted its budget and
    # left would drop the active count below two and disable preemption
    # entirely, which would otherwise look like a preemption bug.
    assert a.uid is not None, "blocker finished before the test completed"
    assert result["status"] == 429
    assert "did not permit progress" in result["error"]
    assert b.preemptions == MAX_REPLAYS_PER_JOB
    assert engine.counts["memory_preemptions"] == MAX_REPLAYS_PER_JOB
    assert engine.counts["preempted_replays"] == MAX_REPLAYS_PER_JOB
    assert a.preemptions == 0 and not [e for e in a_events if "error" in e]
    b_leases = [br for br in state["branches"] if br.tokens[:3] == [11, 12, 13]]
    assert len(b_leases) == 1 + MAX_REPLAYS_PER_JOB
    assert all(branch.closed == 1 for branch in b_leases)
    a.cancelled.set()


def test_lone_or_non_preemptible_lanes_fail_as_before(scripted):
    state, start, submit = scripted
    engine = start(max_lanes=2)
    lone = submit(engine, "lone", max_tokens=4)
    assert lone.events.get(timeout=10)["status"] == 429
    assert engine.counts["memory_preemptions"] == 0
    # Both lanes have decoded, so neither may be replayed and the starved
    # lane fails exactly as it did before preemption existed.
    state["progress"] = {"a", "b"}
    a = submit(engine, "a", max_tokens=1900)
    consume(a)
    wait_for(lambda: a.completion_tokens >= 1)
    b = submit(engine, "b", max_tokens=1900)
    wait_for(lambda: b.completion_tokens >= 1)
    state["progress"] = {"a"}
    result = b.events.get(timeout=10)
    while "error" not in result:
        result = b.events.get(timeout=10)
    assert a.uid is not None, "blocker finished before the test completed"
    assert result["status"] == 429
    assert b.decode_replay_block == "decode_state_not_reconstructible"
    assert engine.counts["memory_preemptions"] == 0
    a.cancelled.set()


def test_drain_cancels_pending_replay_with_503(scripted):
    state, start, submit = scripted
    engine = start(stall_seconds=0.5, max_lanes=2)
    # Same determinism rule as the stall test: nothing decodes while the
    # deadline runs, so no lane can finish and leave.
    state["progress"] = set()
    a = submit(engine, "a", max_tokens=150)
    wait_for(lambda: a.uid is not None)
    b = submit(engine, "b", max_tokens=6)
    wait_for(lambda: b.uid is not None)
    wait_for(lambda: b.preempted, timeout=10)
    engine.quiesce(suspend=False, drain_timeout_seconds=30)
    result = b.events.get(timeout=5)
    while "error" not in result:
        result = b.events.get(timeout=5)
    assert result["status"] == 503 and "draining" in result["error"]
    assert engine.counts["memory_preemption_drain_cancellations"] == 1
    assert state["branches"][-1].closed == 1
    state["progress"] = {"a"}
    assert collect(a)[1]["completion_tokens"] == 150
    assert engine.wait_for_quiesce(5)


def test_qualification_requires_observed_preemption_only_when_enabled():
    from mlx2.qualification import required_feature_checks

    for route in ("ordinary", "external_draft", "prompt_lookup"):
        base = {"speculation": route, "execution_policy": {}, "environment": {}}
        assert "feature_memory_preemption" not in required_feature_checks(base)
        enabled = {**base, "memory_preemption": {"enabled": True}}
        assert "feature_memory_preemption" in required_feature_checks(enabled)
