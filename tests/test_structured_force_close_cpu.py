"""A stop while reasoning is open turns into the reasoning close (option D).

A structured request on a thinking model defers its grammar past the
reasoning close.  A model that stopped before closing used to end the request
with an empty answer reported as ``enforced: true`` (2026-10-14 survey,
decision 1).  ``ReasoningForceCloseProcessor`` gives the stop's probability to
the next release-marker token instead; the grammar then constrains the answer.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2 import structured_output
from mlx2.structured_output import ThinkingBudgetProcessor, structured_receipt
from test_structured_deferral import (  # noqa: F401 - pytest fixture
    EOS,
    HELLO,
    SPLIT_MARKER,
    THINK_CLOSE,
    THINKING_JSON_REQUEST,
    _collect,
    scripted_engine,
)

mx.set_default_device(mx.cpu)

VOCAB = 8
PROMPT = [1, 1]


def _row(**values):
    row = np.zeros(VOCAB, dtype=np.float32)
    for index, value in values.items():
        row[int(index[1:])] = value
    return mx.array(row)[None]


def _call(processor, generated, logits, probe=False):
    tokens = mx.array(PROMPT + list(generated), dtype=mx.uint32)
    method = processor.probe if probe else processor
    return np.asarray(method(tokens, logits))[0]


def _finite(row):
    return set(np.flatnonzero(np.isfinite(row)).tolist())


# -- the processor ---------------------------------------------------------------


def test_greedy_stop_while_reasoning_writes_the_close_marker():
    processor = structured_output.ReasoningForceCloseProcessor(2, (5,), (0,), temperature=0)
    # The model would stop: only the close marker remains.
    assert _finite(_call(processor, [3], _row(t0=4.0, t2=1.0))) == {5}
    # The model would not stop: its argmax is untouched, the stop is removed.
    out = _call(processor, [3], _row(t0=1.0, t2=4.0))
    assert int(np.argmax(out)) == 2 and not np.isfinite(out[0])
    # Reasoning already closed: the processor masks nothing.
    closed = _call(processor, [3, 5], _row(t0=4.0))
    assert np.isfinite(closed).all() and int(np.argmax(closed)) == 0


def test_multi_token_marker_is_completed_token_by_token():
    marker = (5, 6, 7)
    processor = structured_output.ReasoningForceCloseProcessor(2, marker, (0,), temperature=0)
    stop = _row(t0=9.0)
    assert _finite(_call(processor, [3], stop)) == {5}
    # History mode: once the marker is being written it is finished, whatever
    # the model would write next.
    assert _finite(_call(processor, [3, 5], _row(t3=9.0))) == {6}
    assert _finite(_call(processor, [3, 5, 6], _row(t3=9.0))) == {7}
    assert np.isfinite(_call(processor, [3, *marker], stop)).all()
    # Sampled rows complete it the same way.
    sampled = structured_output.ReasoningForceCloseProcessor(2, marker, (0,), temperature=1.0)
    assert _finite(_call(sampled, [3, 5], _row(t3=9.0))) == {6}


@pytest.mark.parametrize("temperature", [0.3, 1.0, 1.7])
def test_sampled_law_is_the_original_law_with_the_stop_moved_to_the_marker(temperature):
    processor = structured_output.ReasoningForceCloseProcessor(2, (5,), (0, 4), temperature=temperature)
    raw = _row(t0=1.2, t1=0.3, t2=-0.4, t4=0.9, t5=0.1, t6=2.0)
    out = _call(processor, [3], raw)

    def law(values):
        scaled = np.asarray(values, dtype=np.float64) / temperature
        weights = np.exp(scaled - scaled[np.isfinite(scaled)].max())
        weights[~np.isfinite(scaled)] = 0.0
        return weights / weights.sum()

    before, after = law(np.asarray(raw)[0]), law(out)
    expected = before.copy()
    expected[5] += expected[0] + expected[4]
    expected[[0, 4]] = 0.0
    np.testing.assert_allclose(after, expected, atol=1e-6)


def test_a_marker_an_earlier_processor_removed_is_not_reinstated():
    processor = structured_output.ReasoningForceCloseProcessor(2, (5,), (0,), temperature=1.0)
    raw = _row(t0=3.0, t1=1.0, t5=-np.inf)
    np.testing.assert_array_equal(_call(processor, [3], raw), np.asarray(raw)[0])
    greedy = structured_output.ReasoningForceCloseProcessor(2, (5,), (0,), temperature=0)
    np.testing.assert_array_equal(_call(greedy, [3], raw), np.asarray(raw)[0])


def test_self_addressed_reasoning_keeps_the_recipient_choice_and_tool_stops():
    # Muse: separator (6,), self header (7,), release (6, 4, 5).  Only stop
    # rows write the release; a message to anybody else (a tool call) ends
    # reasoning without it, and its stop stays a stop.
    processor = structured_output.ReasoningForceCloseProcessor(
        2, (6, 4, 5), (0,), temperature=0, reasoning_message=((6,), (7,))
    )
    stop = _row(t0=9.0)
    assert _finite(_call(processor, [7, 3], stop)) == {6}
    assert _finite(_call(processor, [7, 3, 6], stop)) == {4}
    # After a natural separator the model may address itself again.
    choice = _call(processor, [7, 3, 6], _row(t0=1.0, t7=5.0))
    assert int(np.argmax(choice)) == 7
    told_tool = _call(processor, [7, 3, 6, 2, 3], stop)
    assert np.isfinite(told_tool).all() and int(np.argmax(told_tool)) == 0


def test_an_explicit_budget_and_the_force_close_never_close_twice():
    budget = ThinkingBudgetProcessor(2, 2, SPLIT_MARKER)
    force = structured_output.ReasoningForceCloseProcessor(2, SPLIT_MARKER, (EOS,), temperature=0)
    generated = []
    for _ in range(8):
        logits = mx.zeros((1, 22))
        logits = logits + mx.where(mx.arange(22) == EOS, 9.0, 0.0)
        context = mx.array(PROMPT + generated, dtype=mx.uint32)
        for processor in (budget, force):
            logits = processor(context, logits)
        generated.append(int(mx.argmax(logits[0]).item()))
    # The stop at the first row starts the close; the budget at row 2 finds
    # the marker in progress and only continues it; nothing closes twice.
    assert generated[:3] == list(SPLIT_MARKER)
    assert sum(
        tuple(generated[i : i + 3]) == SPLIT_MARKER for i in range(len(generated))
    ) == 1


def test_probe_rows_leave_no_record_and_settle_reads_the_committed_stream():
    processor = structured_output.ReasoningForceCloseProcessor(2, (5, 6), (0,), temperature=0)
    stop = _row(t0=9.0)
    _call(processor, [3], stop, probe=True)
    assert processor.settle([3, 5, 6])["force_close_tokens"] == 0
    _call(processor, [3], stop)
    _call(processor, [3, 5], stop)
    _call(processor, [3, 5, 6, 2], stop)  # an unused row past the end
    receipt = processor.settle([3, 5, 6])
    assert receipt == {
        "reasoning_closed": True,
        "reasoning_force_closed": True,
        "force_close_tokens": 2,
        "force_close_eos_share_max": 1.0,
    }
    # A close the model wrote itself is not a forced one.
    natural = structured_output.ReasoningForceCloseProcessor(2, (5,), (0,), temperature=0)
    _call(natural, [3], _row(t5=9.0))
    assert natural.settle([3, 5])["reasoning_force_closed"] is False


def test_receipt_stops_reporting_an_unclosed_reasoning_as_enforced():
    from mlx2.structured_output import make_structured_processor
    from test_structured_deferral import _tokenizer

    structured = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"},
        defer_until=(THINK_CLOSE,),
    )
    structured(mx.array([1, 1, 2, 3]), mx.zeros((1, 22)))
    receipt = structured_receipt(structured, 2)
    assert receipt["enforced"] is False
    assert receipt["unenforced_reason"] == "reasoning_not_closed"


# -- serving (scripted model: the grammar, parser and receipts are real) --------


def _structured(final):
    return final["receipt"]["request_controls"]["structured_output"]


@pytest.mark.parametrize(
    "marker, script",
    [
        ((THINK_CLOSE,), [2, 3, 4, EOS]),
        # "</" "think" ">": the stop writes "</", history mode the rest.
        (SPLIT_MARKER, [2, 3, 4, EOS]),
    ],
    ids=["single_token", "multi_token"],
)
def test_stop_while_reasoning_force_closes_and_the_answer_is_constrained(
    scripted_engine, marker, script
):
    build, state = scripted_engine
    state["marker"] = marker
    engine = build(declare_marker=True)
    state["script"] = script
    reasoning, content, final = _collect(engine.submit(dict(THINKING_JSON_REQUEST)))
    assert "error" not in final, final
    assert reasoning == "Let me think"
    assert isinstance(json.loads(content), dict), content
    receipt = _structured(final)
    assert receipt["enforced"] is True and receipt["reasoning_force_closed"] is True
    assert receipt["force_close_tokens"] == len(marker)
    counts = engine.status()["counts"]
    assert counts["structured_reasoning_force_closed"] == 1
    assert counts["structured_reasoning_unclosed"] == 0


def test_max_tokens_after_the_force_close_finishes_length_with_an_honest_receipt(
    scripted_engine,
):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, 4, EOS]
    request = {**THINKING_JSON_REQUEST, "max_tokens": 5}
    reasoning, content, final = _collect(engine.submit(request))
    assert final["finish_reason"] == "length"
    assert reasoning == "Let me think"
    receipt = _structured(final)
    assert receipt["reasoning_force_closed"] is True and receipt["enforced"] is True


def test_reasoning_still_open_at_max_tokens_is_not_reported_as_enforced(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, 4, 3, 4]
    request = {**THINKING_JSON_REQUEST, "max_tokens": 3}
    _, content, final = _collect(engine.submit(request))
    assert final["finish_reason"] == "length" and content == ""
    receipt = _structured(final)
    assert receipt["enforced"] is False
    assert receipt["unenforced_reason"] == "reasoning_not_closed"
    assert receipt["reasoning_force_closed"] is False
    assert engine.status()["counts"]["structured_reasoning_unclosed"] == 1


@pytest.mark.parametrize(
    "request_update",
    [
        {"response_format": None},  # no grammar
        {"enable_thinking": False},  # thinking off: the grammar binds at once
    ],
    ids=["no_grammar", "thinking_off"],
)
def test_force_close_is_not_installed_without_a_deferred_grammar(
    scripted_engine, request_update
):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, EOS] if request_update.get("enable_thinking", True) else [7, 11, EOS]
    request = {**THINKING_JSON_REQUEST, **request_update}
    if request.get("response_format") is None:
        request.pop("response_format")
    _, _, final = _collect(engine.submit(request))
    assert final["finish_reason"] == "stop", final
    assert engine.status()["counts"]["structured_reasoning_force_closed"] == 0
    structured = final["receipt"]["request_controls"]["structured_output"]
    assert structured is None or "reasoning_force_closed" not in structured


def test_budget_forced_close_is_not_doubled_by_the_force_close(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True)
    state["script"] = [2, 3, 4, 4, EOS, EOS]
    request = {**THINKING_JSON_REQUEST, "thinking_budget": 3, "thinking_budget_mode": "history"}
    reasoning, content, final = _collect(engine.submit(request))
    assert "error" not in final, final
    assert "</think>" not in content and isinstance(json.loads(content), dict)
    receipt = final["receipt"]["request_controls"]
    assert receipt["thinking_budget_fired"] is True
    assert receipt["structured_output"]["reasoning_force_closed"] is False


# -- every speculative route (tiny real models) -------------------------------

ROUTE_EOS = {"qwen": (125,), "muse": (37, 14)}  # ids each model reaches while reasoning


@pytest.fixture(scope="module")
def route_models():
    from route_harness import tiny_qwen38_mtp
    from test_speculative_grammar_route_parity_cpu import _wide_selector_dflash2

    target, vocab = tiny_qwen38_mtp()
    return {"qwen": (target, vocab), "muse": _wide_selector_dflash2()}


def _route_engine(models, route, eos, close):
    from route_harness import make_engine, make_external_engine
    from test_speculative_grammar_route_parity_cpu import SELF_MTP_ROUTES

    kw = dict(eos=eos, close_id=close, max_lanes=2)
    if route == "external_dflash2":
        muse, draft, vocab = models["muse"]
        return make_external_engine(muse, draft, vocab, num_draft=3, **kw)
    if route == "muse_ordinary":
        muse, _, vocab = models["muse"]
        return make_engine(muse, vocab, mtp=False, **kw)
    target, vocab = models["qwen"]
    if route == "ordinary":
        return make_engine(target, vocab, mtp=False, **kw)
    if route == "prompt_lookup":
        return make_engine(target, vocab, mtp=False, prompt_lookup=True, num_draft=4, **kw)
    return make_engine(target, vocab, **SELF_MTP_ROUTES[route], **kw)


def _route_serve(models, route, close, *, temperature=0.0, seed=None, sampling=None):
    from route_harness import collect
    from test_speculative_grammar_route_parity_cpu import GRAMMAR, PROMPTS

    family = "muse" if route in ("external_dflash2", "muse_ordinary") else "qwen"
    engine = _route_engine(models, route, ROUTE_EOS[family], close)
    try:
        jobs = [
            engine.submit({
                "tokens": prompt, "messages": [{"role": "user", "content": "x"}],
                "max_tokens": 16, "temperature": temperature, "grammar": GRAMMAR,
                **({"seed": seed + index} if seed is not None else {}),
                **(sampling or {}),
            })
            for index, prompt in enumerate(PROMPTS)
        ]
        results = [collect(job) for job in jobs]
        assert engine.thread.is_alive() and engine.error is None, route
        return results
    finally:
        engine.close()


def _route_answer(tokens, close):
    close = tuple(close) if isinstance(close, tuple) else (close,)
    for index in range(len(tokens) - len(close) + 1):
        if tuple(tokens[index : index + len(close)]) == close:
            from route_harness import PIECES

            return "".join(PIECES[token] for token in tokens[index + len(close):])
    return None


def _speculative_routes():
    from test_speculative_grammar_route_parity_cpu import SELF_MTP_ROUTES

    return (*SELF_MTP_ROUTES, "prompt_lookup", "external_dflash2")


@pytest.mark.parametrize("close", [127, (124, 127)], ids=["single_token", "multi_token"])
@pytest.mark.parametrize("route", _speculative_routes())
def test_greedy_force_close_is_identical_on_every_speculative_route(
    monkeypatch, route_models, route, close
):
    import re

    from route_harness import patch_host
    from test_speculative_grammar_route_parity_cpu import GRAMMAR, _engaged

    patch_host(monkeypatch)
    base = "muse_ordinary" if route == "external_dflash2" else "ordinary"
    reference = _route_serve(route_models, base, close)
    results = _route_serve(route_models, route, close)
    for want, got in zip(reference, results):
        assert got.get("error") is None, (route, got.get("error"))
        assert got["tokens"] == want["tokens"], route
        structured = got["receipt"]["request_controls"]["structured_output"]
        assert structured == want["receipt"]["request_controls"]["structured_output"]
        assert structured["reasoning_force_closed"] is True, route
        answer = _route_answer(got["tokens"], close)
        assert answer is not None
        if got["finish"] == "stop":
            assert re.fullmatch(GRAMMAR, answer), (route, answer)
    if route != "ordinary_handoff":
        assert _engaged(route, results), route


@pytest.mark.parametrize(
    "sampling",
    [{}, {"top_k": 3}, {"top_p": 0.8, "min_p": 0.05}],
    ids=["unfiltered", "top_k", "top_p_min_p"],
)
@pytest.mark.parametrize("route", ("ordinary", "muse_ordinary", *_speculative_routes()))
def test_sampled_force_close_answers_satisfy_the_grammar(
    monkeypatch, route_models, route, sampling
):
    import re

    from route_harness import patch_host
    from test_speculative_grammar_route_parity_cpu import GRAMMAR

    patch_host(monkeypatch)
    closed = 0
    for seed in (7, 31, 53, 71, 97, 113):
        for result in _route_serve(
            route_models, route, 127, temperature=1.0, seed=seed, sampling=sampling
        ):
            assert result.get("error") is None, (route, result.get("error"))
            structured = result["receipt"]["request_controls"]["structured_output"]
            if sampling:
                assert structured["sampling_filters"] == "force_close_processor"
            answer = _route_answer(result["tokens"], 127)
            if answer is None:
                assert structured["enforced"] is False
                continue
            closed += 1
            assert structured["enforced"] is True
            if result["finish"] == "stop":
                assert re.fullmatch(GRAMMAR, answer), (route, seed, answer)
    assert closed


# -- HTTP surfaces: the receipt reaches clients under ``mlx2`` --------------------


def _http(engine):
    import contextlib
    import threading
    from http.server import ThreadingHTTPServer

    from mlx2.server import handler_for

    @contextlib.contextmanager
    def serve():
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    return serve()


def _post(base, path, body):
    from urllib.request import Request, urlopen

    request = Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    from urllib.error import HTTPError

    try:
        with urlopen(request) as response:
            return json.load(response)
    except HTTPError as error:
        raise AssertionError(f"{path}: {error.code} {error.read().decode()}") from None


def test_chat_and_responses_surfaces_report_the_force_close(scripted_engine):
    build, state = scripted_engine
    state["thinking_from_effort"] = True
    engine = build(declare_marker=True)
    model = engine.status()["model"]
    state["script"] = [2, 3, 4, EOS]
    with _http(engine) as base:
        chat = _post(base, "/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": True, "response_format": {"type": "json_object"},
            "max_tokens": 16, "temperature": 0,
        })
        state["script"] = [2, 3, 4, EOS]
        responses = _post(base, "/v1/responses", {
            "model": model, "input": "hi", "reasoning": {"effort": "low"},
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": 16, "temperature": 0,
        })
    message = chat["choices"][0]["message"]
    assert message["reasoning_content"] == "Let me think"
    assert isinstance(json.loads(message["content"]), dict)
    assert chat["choices"][0]["finish_reason"] == "stop"
    for payload in (chat, responses):
        structured = payload["mlx2"]["request_controls"]["structured_output"]
        assert structured["reasoning_force_closed"] is True
        assert structured["enforced"] is True
    texts = [
        part["text"]
        for item in responses["output"] if item["type"] == "message"
        for part in item["content"] if part["type"] == "output_text"
    ]
    assert isinstance(json.loads("".join(texts)), dict)
    reasoning = [item for item in responses["output"] if item["type"] == "reasoning"]
    assert reasoning and "Let me think" in json.dumps(reasoning)


def test_anthropic_forced_tool_after_a_stop_while_thinking(scripted_engine):
    """Messages has no response format; its structured receipt is the forced
    tool grammar, deferred past thinking like any other."""
    from test_structured_deferral import TOOL_CALL

    build, state = scripted_engine
    engine = build(declare_marker=True, execution_policy={"constrained_tool_grammar": True})
    model = engine.status()["model"]
    state["script"] = [2, 3, 4, EOS, TOOL_CALL, EOS]
    with _http(engine) as base:
        reply = _post(base, "/v1/messages", {
            "model": model, "max_tokens": 32,
            "messages": [{"role": "user", "content": "add"}],
            "thinking": {"type": "enabled", "budget_tokens": 16},
            "tool_choice": {"type": "any"},
            "tools": [{
                "name": "sum", "description": "add",
                "input_schema": {
                    "type": "object", "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            }],
        })
    kinds = [block["type"] for block in reply["content"]]
    assert kinds == ["thinking", "tool_use"], reply["content"]
    assert reply["content"][0]["thinking"] == "Let me think"
    assert reply["content"][1]["name"] == "sum"
    structured = reply["mlx2"]["request_controls"]["structured_output"]
    assert structured["kind"] == "tool_choice" and structured["enforced"] is True
    assert structured["reasoning_force_closed"] is True


# -- the request's sampling filters (Codex review of b99fb0c90) -----------------


def _filtered_law(raw, temperature, *, top_p=0.0, top_k=0, min_p=0.0):
    """What ``make_sampler`` draws from: filters at temperature 1, then T."""
    from mlx2.runtime.sample_utils import make_transformed_logprobs

    transform = make_transformed_logprobs(
        temperature, top_p=top_p, top_k=top_k, min_p=min_p
    )
    return np.exp(np.asarray(transform(raw), dtype=np.float64))[0]


def _served_law(processor, generated, raw, temperature):
    """The processor's row, sampled at the temperature alone (no filter)."""
    from mlx2.runtime.sample_utils import make_transformed_logprobs

    out = processor(mx.array(PROMPT + list(generated), dtype=mx.uint32), raw)
    return np.exp(np.asarray(make_transformed_logprobs(temperature)(out), dtype=np.float64))[0]


def test_top_k_one_never_moves_a_stop_the_sampler_could_not_draw():
    """Codex counterexample: EOS (0) and the marker (5) each rank below A (2),
    but their merged probability would beat it.  With ``top_k=1`` ordinary
    sampling always draws A, so nothing may move."""
    raw = _row(t0=1.5, t2=2.0, t5=1.5)
    processor = structured_output.ReasoningForceCloseProcessor(
        2, (5,), (0,), temperature=1.0, top_k=1
    )
    assert processor.owns_sampling_filters
    law = _served_law(processor, [3], raw, 1.0)
    np.testing.assert_allclose(law, _filtered_law(raw, 1.0, top_k=1), atol=1e-7)
    assert law[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "filters",
    [
        {"top_k": 1}, {"top_k": 3}, {"top_p": 0.6}, {"top_p": 0.9},
        {"min_p": 0.3}, {"min_p": 0.05}, {"top_k": 4, "top_p": 0.8, "min_p": 0.1},
    ],
    ids=lambda filters: ",".join(f"{k}={v}" for k, v in filters.items()),
)
@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.6])
@pytest.mark.parametrize(
    "values",
    [
        {"t0": 1.5, "t2": 2.0, "t5": 1.5, "t6": 0.2},  # EOS and marker tie below A
        {"t0": 2.5, "t1": 0.4, "t2": 2.0, "t5": -1.0, "t6": 1.0},  # EOS on top, marker in the tail
        {"t0": -0.5, "t2": 2.0, "t4": 1.9, "t5": 0.3},  # EOS in the tail
    ],
    ids=["tied_below", "eos_top", "eos_tail"],
)
def test_sampled_law_is_the_filtered_law_with_the_kept_stop_moved_to_the_marker(
    filters, temperature, values
):
    raw = _row(**values)
    processor = structured_output.ReasoningForceCloseProcessor(
        2, (5,), (0, 4), temperature=temperature, **filters
    )
    ordinary = _filtered_law(raw, temperature, **filters)
    expected = ordinary.copy()
    expected[5] += expected[0] + expected[4]
    expected[[0, 4]] = 0.0
    np.testing.assert_allclose(
        _served_law(processor, [3], raw, temperature), expected, atol=1e-6
    )
    # Once reasoning has closed the row is exactly the ordinary filtered law.
    np.testing.assert_allclose(
        _served_law(processor, [3, 5], raw, temperature), ordinary, atol=1e-6
    )


def _counterexample_bias(models, family, prompt, close):
    """logit_bias placing token A just above EOS and the marker at the first
    row, each below A alone but above it merged (``log 2 > 0.3``)."""
    if family == "qwen":
        model = models["qwen"][0]
        eos = ROUTE_EOS["qwen"][0]
    else:
        model = models["muse"][0]
        eos = ROUTE_EOS["muse"][0]
    out = model(mx.array([prompt]))
    out = out[0] if isinstance(out, tuple) else out
    raw = out[0, -1].astype(mx.float32).tolist()
    a = 111
    return {str(a): 10.3 - raw[a], str(eos): 10.0 - raw[eos], str(close): 10.0 - raw[close]}


@pytest.mark.parametrize("route", ("ordinary", "muse_ordinary", *_speculative_routes()))
def test_top_k_one_sampling_equals_greedy_on_every_route(monkeypatch, route_models, route):
    """The Codex counterexample end to end: ``top_k=1`` draws the argmax, so a
    sampled request must produce the greedy tokens; before the fix the merged
    stop made the marker win on every route."""
    from route_harness import collect, patch_host
    from test_speculative_grammar_route_parity_cpu import GRAMMAR, PROMPTS

    patch_host(monkeypatch)
    family = "muse" if route in ("external_dflash2", "muse_ordinary") else "qwen"
    prompt = PROMPTS[0]
    bias = _counterexample_bias(route_models, family, prompt, 127)
    engine = _route_engine(route_models, route, ROUTE_EOS[family], 127)
    base = {
        "tokens": prompt, "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 8, "grammar": GRAMMAR, "logit_bias": bias,
    }
    try:
        greedy = collect(engine.submit({**base, "temperature": 0}))
        sampled = [
            collect(engine.submit({**base, "temperature": 1.0, "top_k": 1, "seed": seed}))
            for seed in (3, 4)
        ]
    finally:
        engine.close()
    assert greedy["tokens"][0] == 111
    for result in sampled:
        assert result.get("error") is None, result.get("error")
        assert result["tokens"] == greedy["tokens"], route
