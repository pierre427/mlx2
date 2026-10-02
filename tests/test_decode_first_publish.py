"""Decode-first publication (port of mlx-vlm #1630), CPU.

Off by default.  On, ``BatchGenerator.next`` returns a round's decode tokens
before that round's prefill phase runs (the phase runs at the start of the
next call); the device work and its order are unchanged, so greedy tokens
match the default.  ``all`` mode also shares one prefill token budget across
rows prefilling in the same round.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime import generate as G
from mlx2.runtime.adaptive_policy import DECODE_FIRST_ENV, DecodeFirstPublish
from mlx2.runtime.sample_utils import LaneRNG


def tiny_model():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 16,
    )
    mx.random.seed(7)
    m = TextModel(args)
    m.eval()
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(DECODE_FIRST_ENV, raising=False)


# ---------------------------------------------------------------- policy


def test_policy_off_by_default_and_parsing():
    off = DecodeFirstPublish.from_value(None)
    assert not off.enabled and off.mode({}) == "off"
    assert DecodeFirstPublish.from_value(False).mode({}) == "off"
    on = DecodeFirstPublish.from_value(True)
    assert on.mode({}) == "all"
    order = DecodeFirstPublish.from_value({"shared_prefill_budget": False})
    assert order.enabled and order.mode({}) == "order"
    capped = DecodeFirstPublish.from_value({"prefill_token_budget": 512})
    assert capped.as_dict() == {
        "enabled": True, "shared_prefill_budget": True, "prefill_token_budget": 512,
    }
    with pytest.raises(ValueError, match="unknown decode_first"):
        DecodeFirstPublish.from_value({"budget": 3})
    with pytest.raises(ValueError, match="positive integer"):
        DecodeFirstPublish.from_value({"prefill_token_budget": 0})
    with pytest.raises(ValueError, match="boolean"):
        DecodeFirstPublish.from_value({"enabled": "yes"})
    with pytest.raises(ValueError, match="boolean or an object"):
        DecodeFirstPublish.from_value(3)


def test_env_kill_switch_and_forcing():
    on = DecodeFirstPublish.from_value(True)
    assert on.mode({DECODE_FIRST_ENV: "0"}) == "off"
    assert on.counters["kill_switch_rounds"] == 1
    off = DecodeFirstPublish()
    assert off.mode({DECODE_FIRST_ENV: "order"}) == "order"
    assert off.mode({DECODE_FIRST_ENV: "1"}) == "all"
    assert off.mode({DECODE_FIRST_ENV: "garbage"}) == "off"
    assert "kill_switch_rounds" not in off.counters


@pytest.mark.parametrize(
    ("prompt_lookup", "backend"), [(True, None), (False, "external_draft")]
)
def test_serving_refuses_decode_first_off_batch_generator_routes(
    monkeypatch, prompt_lookup, backend
):
    from mlx2 import serving
    from test_apc_interior_route_selection import _UnsupportedInteriorAdapter

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})

    class Adapter(_UnsupportedInteriorAdapter):
        pass

    Adapter.backend = backend
    engine = serving.ServingEngine(
        "fixture",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        prompt_lookup=prompt_lookup,
        execution_policy={"decode_first": True},
    )
    try:
        engine.thread.join(5)
        assert not engine.ready.is_set()
        assert "decode_first requires" in (engine.error or "")
    finally:
        engine.close()


def test_serving_parses_decode_first_at_construction():
    from mlx2 import serving

    with pytest.raises(ValueError, match="unknown decode_first"):
        serving.ServingEngine(
            "fixture",
            adapter_factory=lambda *_a, **_k: None,
            execution_policy={"decode_first": {"order_only": True}},
        )


def test_decode_first_counters_export_under_their_mechanism():
    from mlx2.prometheus import PrometheusBuilder, _add_scheduler

    builder = PrometheusBuilder()
    _add_scheduler(builder, {"decode_first_published_rounds": 3,
                             "decode_first_budget_deferred_rows": 2})
    text = builder.render()
    assert 'mechanism="decode_first"' in text


# ---------------------------------------------------------- instrumentation


class Trace:
    def __init__(self, monkeypatch, mtp=False):
        self.events = []
        trace = self
        decode_cls = G.MTPGenerationBatch if mtp else G.GenerationBatch
        real_decode = decode_cls.next

        def decode(batch):
            out = real_decode(batch)
            if out:
                trace.events.append("decode")
            return out

        monkeypatch.setattr(decode_cls, "next", decode)
        if mtp:
            for name in ("_advance_mtp_prefill", "_make_mtp_batch"):
                real = getattr(G.BatchGenerator, name)

                def wrapped(gen, *args, _real=real, **kwargs):
                    trace.events.append("prefill")
                    return _real(gen, *args, **kwargs)

                monkeypatch.setattr(G.BatchGenerator, name, wrapped)
        else:
            real_prompt = G.PromptProcessingBatch.prompt

            def prompt(batch, tokens, **kwargs):
                if any(len(t) for t in tokens):
                    trace.events.append(("prefill", tuple(len(t) for t in tokens)))
                return real_prompt(batch, tokens, **kwargs)

            monkeypatch.setattr(G.PromptProcessingBatch, "prompt", prompt)

    def mark(self, tokens):
        self.events.append(("return", tokens))


def drive(gen, trace, *, arrivals, max_calls=400):
    """Run ``next`` until every lane finishes; insert late arrivals."""
    tokens, calls, done = {}, 0, set()
    pending = dict(arrivals)
    expected = None
    while calls < max_calls:
        for at in [k for k in pending if k <= calls]:
            for item in pending.pop(at):
                for uid in gen.insert(**item):
                    tokens.setdefault(uid, [])
        expected = len(tokens)
        prompts, responses = gen.next()
        calls += 1
        for r in responses:
            tokens[r.uid].append(int(r.token))
            if r.finish_reason:
                done.add(r.uid)
        trace.mark(len(responses))
        if not pending and len(done) == expected:
            break
    assert len(done) == expected, "lanes did not finish"
    return tokens


def work_order(events):
    return [e if isinstance(e, str) else e[0] for e in events
            if not (isinstance(e, tuple) and e[0] == "return")]


def same_device_work(a, b):
    """Same prefill slices in the same order and the same decode steps.

    A request inserted between two calls is seen by the pending prefill
    phase, which runs before the next decode step, so its first slice can
    move one decode step earlier than in the default; nothing else moves.
    """
    slices = lambda ev: [e for e in ev if not isinstance(e, str)
                         and e[0] != "return"] + [
        e for e in ev if e == "prefill"]
    return (slices(a) == slices(b)
            and work_order(a).count("decode") == work_order(b).count("decode"))


def ordinary(model, **kwargs):
    return G.BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=2,
        prefill_step_size=16, **kwargs,
    )


def ordinary_arrivals():
    return {
        0: [dict(prompts=[list(range(2, 12))], max_tokens=[24])],
        3: [dict(prompts=[list(range(3, 70))], max_tokens=[6])],
    }


def calls_returning_tokens_beside_prefill(events):
    """For each next() call that returned tokens: did a prefill run after
    the call's last decode step and before it returned?"""
    out, since = [], []
    for e in events:
        if isinstance(e, tuple) and e[0] == "return":
            kinds = [x if isinstance(x, str) else x[0] for x in since]
            if e[1] and "decode" in kinds:
                last = len(kinds) - 1 - kinds[::-1].index("decode")
                out.append("prefill" in kinds[last + 1:])
            since = []
        else:
            since.append(e)
    return out


# ------------------------------------------------------------- ordinary


def test_default_returns_decode_tokens_after_the_prefill_slice(model, monkeypatch):
    trace = Trace(monkeypatch)
    gen = ordinary(model)
    try:
        drive(gen, trace, arrivals=ordinary_arrivals())
    finally:
        gen.close()
    flags = calls_returning_tokens_beside_prefill(trace.events)
    # Today: a round's decode tokens wait for that round's prefill slice.
    assert any(flags)
    assert not any(k.startswith("decode_first_") for k in gen.scheduler_stats)


@pytest.mark.parametrize("value", [{"shared_prefill_budget": False}, "env-order"])
def test_decode_first_returns_tokens_before_prefill_same_work_and_tokens(
    model, monkeypatch, value
):
    ref_trace = Trace(monkeypatch)
    gen = ordinary(model)
    try:
        ref = drive(gen, ref_trace, arrivals=ordinary_arrivals())
    finally:
        gen.close()
    monkeypatch.undo()
    trace = Trace(monkeypatch)
    if value == "env-order":
        monkeypatch.setenv(DECODE_FIRST_ENV, "order")
        gen = ordinary(model)
    else:
        gen = ordinary(model, decode_first=value)
    try:
        got = drive(gen, trace, arrivals=ordinary_arrivals())
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    assert got == ref
    # Same device work: only return points (and arrival alignment) moved.
    assert same_device_work(trace.events, ref_trace.events)
    flags = calls_returning_tokens_beside_prefill(trace.events)
    assert flags and not any(flags)
    assert stats["decode_first_published_rounds"] > 0
    assert stats["decode_first_prefill_phases_resumed"] > 0
    assert "decode_first_budget_split_rounds" not in stats


def test_shared_budget_splits_the_slice_across_rows(model, monkeypatch):
    arrivals = {
        0: [dict(prompts=[list(range(2, 12))], max_tokens=[30])],
        2: [dict(prompts=[list(range(3, 60)), list(range(5, 61))],
                 max_tokens=[4, 4])],
    }
    ref_trace = Trace(monkeypatch)
    gen = ordinary(model)
    try:
        drive(gen, ref_trace, arrivals=arrivals)
    finally:
        gen.close()
    monkeypatch.undo()
    trace = Trace(monkeypatch)
    gen = ordinary(model, decode_first=True)
    try:
        drive(gen, trace, arrivals=arrivals)
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    two_row = lambda ev: [e[1] for e in ev if isinstance(e, tuple)
                          and e[0] == "prefill" and len(e[1]) == 2]
    assert any(max(w) == 16 for w in two_row(ref_trace.events))
    # Two rows share the 16-token slice: 8 tokens each, padded tile <= 16.
    assert two_row(trace.events)
    assert all(max(w) <= 8 for w in two_row(trace.events))
    assert stats["decode_first_budget_split_rounds"] > 0


def test_prefill_token_budget_caps_a_single_row(model, monkeypatch):
    trace = Trace(monkeypatch)
    gen = ordinary(model, decode_first={"prefill_token_budget": 4})
    try:
        drive(gen, trace, arrivals={0: [dict(prompts=[list(range(2, 30))],
                                             max_tokens=[3])]})
    finally:
        gen.close()
    widths = [e[1] for e in trace.events if isinstance(e, tuple) and e[0] == "prefill"]
    assert widths and all(max(w) <= 4 for w in widths)


def test_kill_switch_finishes_pending_phase_then_runs_whole_rounds(model, monkeypatch):
    trace = Trace(monkeypatch)
    gen = ordinary(model, decode_first={"shared_prefill_budget": False})
    try:
        uid = gen.insert([list(range(2, 12))], max_tokens=[40])[0]
        for _ in range(4):
            gen.next()
        gen.insert([list(range(3, 70))], max_tokens=[4])
        _p, rs = gen.next()
        assert rs and gen._decode_first_pending is not None
        monkeypatch.setenv(DECODE_FIRST_ENV, "0")
        before = len(trace.events)
        gen.next()
        kinds = work_order(trace.events[before:])
        # The pending prefill phase ran first, then one whole round.
        assert kinds[0] == "prefill" and kinds[1:3] == ["decode", "prefill"]
        assert gen._decode_first_pending is None
        assert gen.scheduler_stats["decode_first_kill_switch_rounds"] >= 1
        assert uid is not None
    finally:
        gen.close()


def test_close_drops_a_pending_phase(model):
    gen = ordinary(model, decode_first=True)
    gen.insert([list(range(2, 12))], max_tokens=[40])
    for _ in range(4):
        gen.next()
    gen.insert([list(range(3, 70))], max_tokens=[4])
    gen.next()
    pending = gen._decode_first_pending
    assert pending is not None
    gen.close()
    assert gen._decode_first_pending is None
    assert pending[0].gi_frame is None  # generator closed


def test_mixed_round_is_not_split(model, monkeypatch):
    """A fused prompt+decode forward has no decode-only publication point."""
    gen = ordinary(model, decode_first=True)
    try:
        monkeypatch.setattr(gen, "_mixed_round_ready", lambda: True)
        fused = SimpleNamespace(uid=0, finish_reason=None)
        monkeypatch.setattr(gen, "_next_mixed", lambda: ([], [fused]))
        assert gen.next() == ([], [fused])
        assert gen._decode_first_pending is None
        assert gen.scheduler_stats["decode_first_fused_rounds"] == 1
    finally:
        gen.close()


# ---------------------------------------------------------------- self-MTP


def mtp(model, **kwargs):
    return G.BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=1,
        prefill_step_size=32, self_mtp={"num_draft": 2, "persistent": True},
        **kwargs,
    )


def mtp_item(prompt, max_tokens, seed):
    return dict(prompts=[prompt], max_tokens=[max_tokens], lane_rngs=[LaneRNG(seed)],
                self_mtp_configs=[{"sampling_temp": 0.0}])


def mtp_arrivals():
    return {
        0: [mtp_item(list(range(2, 12)), 30, 1)],
        3: [mtp_item(list(range(3, 100)), 6, 2)],
    }


def test_self_mtp_decode_first_same_tokens_and_work_order(model, monkeypatch):
    ref_trace = Trace(monkeypatch, mtp=True)
    gen = mtp(model)
    try:
        ref = drive(gen, ref_trace, arrivals=mtp_arrivals())
    finally:
        gen.close()
    assert any(calls_returning_tokens_beside_prefill(ref_trace.events))
    monkeypatch.undo()
    trace = Trace(monkeypatch, mtp=True)
    gen = mtp(model, decode_first={"shared_prefill_budget": False})
    try:
        got = drive(gen, trace, arrivals=mtp_arrivals())
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    assert got == ref
    assert same_device_work(trace.events, ref_trace.events)
    flags = calls_returning_tokens_beside_prefill(trace.events)
    assert flags and not any(flags)
    assert stats["decode_first_published_rounds"] > 0


def test_self_mtp_shared_budget_defers_burst_prefills(model, monkeypatch):
    arrivals = {
        0: [mtp_item(list(range(2, 12)), 40, 1)],
        3: [mtp_item(list(range(3 + i, 25 + i)), 4, 2 + i) for i in range(3)],
    }
    prepared = []
    real = G.BatchGenerator._make_mtp_batch

    def counting(gen, n):
        prepared.append(n)
        return real(gen, n)

    monkeypatch.setattr(G.BatchGenerator, "_make_mtp_batch", counting)
    ref_trace = Trace(monkeypatch, mtp=True)
    gen = mtp(model)
    try:
        ref = drive(gen, ref_trace, arrivals=arrivals)
    finally:
        gen.close()
    ref_prepared = list(prepared)
    del prepared[:]
    gen = mtp(model, decode_first={"prefill_token_budget": 24})
    try:
        got = drive(gen, Trace(monkeypatch, mtp=True), arrivals=arrivals)
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    # Default prepares the whole 3-prompt burst (22 tokens each) in one call;
    # a 24-token shared budget admits one per round.
    assert max(ref_prepared) == 3
    assert max(prepared) == 1
    assert stats["decode_first_budget_deferred_rows"] >= 2
    # The burst rows keep their greedy tokens (per-lane MTP state).
    assert sorted(map(tuple, got.values())) == sorted(map(tuple, ref.values()))
