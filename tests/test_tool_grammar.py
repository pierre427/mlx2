"""Item 12: tool-call grammar during generation (auto alternation, quantifiers,
tools + JSON answer, streaming) and the P5 processor properties."""

import contextlib
import json
import threading
from http.server import ThreadingHTTPServer

import mlx.core as mx
import numpy as np
import pytest
import regex
from test_serving_contract import (
    TOOLS,
    FakeEngine,
    Job,
    handler_for,
    post,
    post_response,
)
from test_structured_deferral import (
    BLANK,
    EOS,
    HELLO,
    MALFORMED_TOOL_CALL,
    PIECES,
    SPLIT_MARKER,
    THINK_CLOSE,
    TOOL_CALL,
    _tokenizer,
)
from test_structured_deferral import scripted_engine as _scripted_engine_fixture

from mlx2.tool_grammar import plan_tool_grammar, text_excluding


def _row(generated, prompt=(1, 1)):
    return mx.array(list(prompt) + list(generated), dtype=mx.uint32)


def _passes_through(processor, generated):
    logits = mx.array(np.linspace(-1.0, 1.0, len(PIECES), dtype=np.float32))[None]
    out = processor(_row(generated), logits)
    return bool(np.array_equal(np.array(out), np.array(logits)))


def _histories():
    body = [2, 3, 4, 13, 12]
    yield []
    for cut in range(1, len(body) + 1):
        yield body[:cut]
    for marker in ((THINK_CLOSE,), SPLIT_MARKER):
        for cut in range(len(body) + 1):
            yield body[:cut] + list(marker)
            yield body[:cut] + list(marker) + [7]


def _assert_dormant_is_sound(processor):
    """``dormant`` may be conservative, but never claims a masking row."""
    seen = set()
    for generated in _histories():
        dormant = processor.dormant(_row(generated))
        seen.add(dormant)
        if dormant:
            assert _passes_through(processor, generated), generated
    return seen


def test_p5_structured_processor_is_dormant_only_while_deferred():
    from mlx2.structured_output import make_structured_processor

    processor = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"},
        defer_until=(THINK_CLOSE,),
    )
    assert processor.history_pure is True
    assert _assert_dormant_is_sound(processor) == {True, False}
    assert processor.dormant(_row([2, 3])) is True
    assert processor.dormant(_row([2, THINK_CLOSE])) is False
    # Side-effect free: probing dormancy past the marker does not activate.
    processor(_row([2]), mx.zeros((1, len(PIECES))))
    processor.dormant(_row([2, THINK_CLOSE]))
    assert processor.constraining is False
    undeferred = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"}
    )
    assert undeferred.dormant(_row([])) is False
    blocking = make_structured_processor(
        _tokenizer(), 2, response_format={"type": "json_object"},
        defer_until=(THINK_CLOSE,), block_eos_while_deferred=True,
    )
    assert blocking.dormant(_row([2])) is False


def test_p5_thinking_budget_and_guard_dormancy():
    from mlx2.structured_output import ThinkingBudgetProcessor
    from mlx2.thinking_guard import ThinkingGuard

    budget = ThinkingBudgetProcessor(2, 3, SPLIT_MARKER)
    assert budget.history_pure is True
    assert _assert_dormant_is_sound(budget) == {True, False}
    assert budget.dormant(_row([2, 3])) is True
    assert budget.dormant(_row([2, 3, 4])) is False
    budget.fired = "untouched"
    budget.dormant(_row([2, 3, *SPLIT_MARKER]))
    assert budget.fired == "untouched"

    guard = ThinkingGuard(2, (THINK_CLOSE,), budget=2)
    assert guard.history_pure is True
    assert _assert_dormant_is_sound(guard) == {True, False}


def test_p5_stateless_and_adapter_processors_declare_history_purity():
    from mlx2.adapters.muse_glimmer import MuseRecipientProcessor
    from mlx2.adapters.north_mini_code import NorthActionProcessor
    from mlx2.runtime.sample_utils import make_logits_processors
    from mlx2.serving import minimum_tokens_processor

    processors = make_logits_processors(
        logit_bias={3: 1.0}, repetition_penalty=1.1, presence_penalty=0.5,
        frequency_penalty=0.5, penalty_generation_start=2,
    )
    assert processors and all(p.history_pure is True for p in processors)
    minimum = minimum_tokens_processor(mx, [EOS], 2, 3)
    assert minimum.history_pure is True
    assert _assert_dormant_is_sound(minimum) == {True, False}
    assert NorthActionProcessor(prompt_length=2, action_open=(7,)).history_pure
    muse = MuseRecipientProcessor(2, [(2, 3)])
    assert muse.history_pure is True
    assert muse.dormant(_row([2, 3, 4])) is True
    assert muse.dormant(_row([2])) is False
    assert _passes_through(muse, [2, 3, 4])


# ---------------------------------------------------------------------------
# Grammar composition: shapes and quantifiers over the real adapter builders.

SUM = {
    "type": "function",
    "function": {
        "name": "sum",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        },
    },
}
LOOSE = {"type": "function", "function": {"name": "sum", "parameters": {}}}
QWEN_CALL = "<tool_call>\n<function=sum>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"


def _qwen_accessor(request):
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

    return constrained_tool_grammar(
        request["tools"], request["tool_choice"],
        parallel_tool_calls=request.get("parallel_tool_calls", True),
    )


def _language(request, **kw):
    pattern, status, receipt = plan_tool_grammar(
        request, _qwen_accessor, open_marker="<tool_call>", **kw
    )
    assert status == "engaged", status
    return regex.compile(rf"(?:{pattern})"), receipt


@pytest.mark.parametrize("literal", ["<tool_call>", "<|START_ACTION|>", "<atem:function_calls>", "x", "abab"])
def test_text_excluding_matches_exactly_the_texts_without_the_opener(literal):
    import itertools

    compiled = regex.compile(rf"(?:{text_excluding(literal)})")
    alphabet = sorted(set(literal)) + ["z"]
    for length in range(7):
        for chars in itertools.product(alphabet, repeat=length):
            text = "".join(chars)
            assert bool(compiled.fullmatch(text)) == (literal not in text), text
    for text in (literal[:-1], "a" + literal[:-1] + literal[:-1], literal[:-1] + "\n"):
        assert bool(compiled.fullmatch(text)) == (literal not in text)
    assert not compiled.fullmatch("hello " + literal + " world")


@pytest.mark.parametrize(
    "tool_choice, parallel, calls, one, two, text, text_then_call",
    [
        ("required", True, "+", True, True, False, False),
        ("required", False, "1", True, False, False, False),
        ({"type": "function", "function": {"name": "sum"}}, True, "+", True, True, False, False),
        ("auto", True, "*", True, True, True, True),
        ("auto", False, "?", True, False, True, True),
    ],
)
def test_quantifiers_per_tool_choice(tool_choice, parallel, calls, one, two, text, text_then_call):
    language, receipt = _language(
        {"tools": [SUM], "tool_choice": tool_choice, "parallel_tool_calls": parallel}
    )
    assert receipt["calls"] == calls
    assert bool(language.fullmatch(QWEN_CALL)) is one
    assert bool(language.fullmatch(QWEN_CALL + "\n" + QWEN_CALL)) is two
    assert bool(language.fullmatch("The answer is 4.")) is text
    assert bool(language.fullmatch("Let me call it.\n" + QWEN_CALL)) is text_then_call
    assert bool(language.fullmatch("")) is text  # zero calls only under auto
    # Always rejected: malformed calls, undeclared tools, strict violations.
    for bad in (
        "<tool_call><function=missing></function></tool_call>",
        QWEN_CALL.replace("sum", "sub"),
        QWEN_CALL.replace("\n1\n", "\none\n"),
        "text " + QWEN_CALL.replace("\n1\n", "\n1.5\n"),
    ):
        assert not language.fullmatch(bad), bad
    if calls == "?":
        assert not language.fullmatch(QWEN_CALL + "then " + QWEN_CALL)


def test_tools_combined_with_a_json_answer():
    schema = {
        "type": "json_schema",
        "json_schema": {
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }
    for answer in ({"type": "json_object"}, schema):
        for tools in ([SUM], [LOOSE]):  # non-strict tools engage with an answer
            language, receipt = _language(
                {"tools": tools, "tool_choice": "auto", "response_format": answer},
                leading_whitespace=True,
            )
            assert receipt == {"shape": "calls_or_answer", "calls": "*"}
            assert language.fullmatch('{"a": 1}')
            assert language.fullmatch('\n\n{"a": 1}')
            assert language.fullmatch(QWEN_CALL)
            assert language.fullmatch("\n\n" + QWEN_CALL)
            assert not language.fullmatch("The answer is 1.")
            assert not language.fullmatch('{"a": 1}' + QWEN_CALL)
    # Forced calls ignore the answer: a tool call is the only admissible output.
    language, receipt = _language(
        {"tools": [SUM], "tool_choice": "required", "response_format": schema}
    )
    assert receipt == {"shape": "calls", "calls": "+"}
    assert language.fullmatch(QWEN_CALL) and not language.fullmatch('{"a": 1}')


def test_plan_disabled_and_skip_reasons():
    assert plan_tool_grammar({"tools": [LOOSE]}, _qwen_accessor)[1] == "disabled"
    assert plan_tool_grammar({"tools": [SUM], "tool_choice": "none"}, _qwen_accessor)[1] == "disabled"
    assert plan_tool_grammar({"messages": []}, _qwen_accessor)[1] == "disabled"
    assert plan_tool_grammar(
        {"tools": [SUM], "grammar": "x"}, _qwen_accessor, open_marker="<tool_call>"
    )[1] == "skipped_request_combination"
    assert plan_tool_grammar(
        {"tools": [SUM], "min_tokens": 2}, _qwen_accessor, open_marker="<tool_call>"
    )[1] == "skipped_request_combination"
    assert plan_tool_grammar({"tools": [SUM]}, None)[1] == "skipped_adapter_unsupported"
    # auto text needs the adapter's opener; a JSON answer does not.
    assert plan_tool_grammar({"tools": [SUM]}, _qwen_accessor)[1] == "skipped_adapter_unsupported"
    assert plan_tool_grammar(
        {"tools": [SUM], "response_format": {"type": "json_object"}}, _qwen_accessor
    )[1] == "engaged"
    union = {
        "type": "function",
        "function": {
            "name": "u", "strict": True,
            "parameters": {
                "type": "object",
                "properties": {"v": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
                "required": ["v"], "additionalProperties": False,
            },
        },
    }
    assert plan_tool_grammar(
        {"tools": [union]}, _qwen_accessor, open_marker="<tool_call>"
    )[1] == "skipped_grammar_unrepresentable"


@pytest.mark.parametrize("adapter", ["qwen", "north", "muse"])
@pytest.mark.parametrize("parallel", [True, False])
def test_every_shipped_adapter_grammar_compiles_to_the_exact_automaton(adapter, parallel):
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.adapters.muse_glimmer_output import constrained_tool_grammar as muse
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.adapters.north_output import constrained_tool_grammar as north
    from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar as qwen
    from mlx2.structured_automaton import automaton_for

    builder, cls = {
        "qwen": (qwen, FlashNextAdapter),
        "north": (north, NorthMiniCodeAdapter),
        "muse": (muse, MuseGlimmerAdapter),
    }[adapter]

    def accessor(request):
        return builder(
            request["tools"], request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    for extra in ({}, {"response_format": {"type": "json_object"}}):
        pattern, status, _ = plan_tool_grammar(
            {"tools": [SUM], "parallel_tool_calls": parallel, **extra},
            accessor,
            open_marker=cls.tool_call_open_marker,
            leading_whitespace=True,
        )
        assert status == "engaged"
        automaton_for(regex.compile(rf"(?:{pattern})"))


# ---------------------------------------------------------------------------
# Serving: a real ServingEngine loop over the scripted model.

# The real ServingEngine loop over the scripted model of the deferral tests.
scripted_engine = _scripted_engine_fixture

AUTO_POLICY = {"constrained_tool_grammar": True, "constrained_tool_grammar_auto": True}


def _auto_engine(build, **kw):
    engine = build(declare_marker=True, execution_policy=AUTO_POLICY, **kw)
    type(engine.adapter).tool_call_open_marker = "<tool_call>"
    return engine


def _run_request(engine, request):
    reasoning, content, calls = [], [], []
    job = engine.submit(request)
    while True:
        event = job.events.get(timeout=10)
        if "delta" in event:
            reasoning.append(event["delta"].get("reasoning_content", ""))
            content.append(event["delta"].get("content", ""))
            calls.extend(event["delta"].get("tool_calls", ()))
        if "finish_reason" in event or "error" in event:
            return "".join(reasoning), "".join(content), calls, event


def _request(**kw):
    return {
        "messages": [{"role": "user", "content": "add"}],
        "enable_thinking": False,
        "tools": [SUM],
        "max_tokens": 6,
        "temperature": 0,
        "top_k": 5,
        **kw,
    }


def test_strict_auto_yields_text_or_a_valid_call(scripted_engine):
    build, state = scripted_engine
    engine = _auto_engine(build)
    state["script"] = [HELLO, EOS]
    _, content, calls, final = _run_request(engine, _request())
    assert "error" not in final, final
    assert content == "hello" and calls == []
    tool_choice = final["receipt"]["request_controls"]["tool_choice"]
    assert tool_choice["decode_grammar"] == "engaged"
    assert tool_choice["grammar"] == {"shape": "text_or_calls", "calls": "*"}
    assert final["receipt"]["request_controls"]["structured_output"]["kind"] == "tool_choice"

    state["script"] = [TOOL_CALL, EOS]
    _, content, calls, final = _run_request(engine, _request())
    assert "error" not in final, final
    assert [call["function"]["name"] for call in calls] == ["sum"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    # The model's favourite token is a malformed call: the grammar removes it
    # and the lane answers in text instead.
    state["script"] = [MALFORMED_TOOL_CALL, EOS]
    _, content, calls, final = _run_request(engine, _request())
    assert "error" not in final, final
    assert calls == [] and content.startswith("hello")
    assert "<tool_call>" not in content
    assert engine.counts["constrained_tool_grammar_engagements"] == 3
    assert engine.counts["structured_output_failures"] == 0


def test_single_call_auto_admits_at_most_one_call(scripted_engine):
    build, state = scripted_engine
    engine = _auto_engine(build)
    state["script"] = [TOOL_CALL, TOOL_CALL, EOS]
    _, _, calls, final = _run_request(
        engine, _request(parallel_tool_calls=False)
    )
    assert "error" not in final, final
    assert len(calls) == 1
    assert final["receipt"]["request_controls"]["tool_choice"]["grammar"] == {
        "shape": "text_or_calls", "calls": "?",
    }


def test_tools_with_a_json_answer_serve_either_branch(scripted_engine):
    build, state = scripted_engine
    engine = _auto_engine(build)
    request = _request(tools=[LOOSE], response_format={"type": "json_object"}, max_tokens=8)
    state["script"] = [7, 8, 9, 10, 11, EOS]
    _, content, calls, final = _run_request(engine, request)
    assert "error" not in final, final
    assert json.loads(content) == {"a": 1} and calls == []
    assert final["receipt"]["request_controls"]["tool_choice"]["grammar"] == {
        "shape": "calls_or_answer", "calls": "*",
    }
    state["script"] = [TOOL_CALL, EOS]
    _, content, calls, final = _run_request(engine, request)
    assert "error" not in final, final
    assert [call["function"]["name"] for call in calls] == ["sum"]
    # Plain text ("hello" is the model's favourite) is outside the language.
    state["script"] = []
    _, content, calls, final = _run_request(engine, request)
    assert "hello" not in content

    # Forced calls plus a response_format were skipped before; now engaged.
    state["script"] = [TOOL_CALL, EOS]
    _, _, calls, final = _run_request(
        engine, {**request, "tools": [SUM], "tool_choice": "required"}
    )
    assert "error" not in final, final
    assert len(calls) == 1
    assert final["receipt"]["request_controls"]["tool_choice"]["decode_grammar"] == "engaged"


def test_thinking_then_strict_auto_call(scripted_engine):
    build, state = scripted_engine
    engine = _auto_engine(build)
    state["script"] = [2, 3, THINK_CLOSE, BLANK, TOOL_CALL, EOS]
    reasoning, _, calls, final = _run_request(
        engine, _request(enable_thinking=True, max_tokens=8)
    )
    assert "error" not in final, final
    assert reasoning == "Let me"
    assert [call["function"]["name"] for call in calls] == ["sum"]
    structured = final["receipt"]["request_controls"]["structured_output"]
    assert structured["deferred"] is True


def test_default_policy_is_unchanged_and_skips_are_counted(scripted_engine):
    build, state = scripted_engine
    engine = build(declare_marker=True, execution_policy={"constrained_tool_grammar": True})
    state["script"] = [HELLO, EOS]
    _, content, _, final = _run_request(engine, _request())
    tool_choice = final["receipt"]["request_controls"]["tool_choice"]
    assert tool_choice["decode_grammar"] == "skipped_strict_auto"
    assert "grammar" not in tool_choice
    settings = engine.status()["settings"]
    assert "constrained_tool_grammar_auto" not in settings
    assert "tool_grammar_streaming" not in settings

    # Enabled but the adapter declares no opener: counted skip, no grammar.
    engine = build(declare_marker=True, execution_policy=AUTO_POLICY)
    state["script"] = [HELLO, EOS]
    _, content, _, final = _run_request(engine, _request())
    assert content == "hello"
    assert final["receipt"]["request_controls"]["tool_choice"][
        "decode_grammar"
    ] == "skipped_adapter_unsupported"
    assert engine.counts["constrained_tool_grammar_skips"] == 1
    assert engine.status()["settings"]["constrained_tool_grammar_auto"] is True
    # Non-strict auto without an answer format is not a tool contract at all.
    _, _, _, final = _run_request(engine, _request(tools=[LOOSE]))
    assert final["receipt"]["request_controls"]["tool_choice"]["decode_grammar"] == "disabled"
    assert engine.counts["constrained_tool_grammar_skips"] == 1


def test_extension_policies_require_the_base_grammar():
    from mlx2.serving import ServingEngine

    for name in ("constrained_tool_grammar_auto", "tool_grammar_streaming"):
        with pytest.raises(ValueError, match=f"{name} requires constrained_tool_grammar"):
            ServingEngine("unused", execution_policy={name: True})
        with pytest.raises(ValueError, match=f"{name} must be boolean"):
            ServingEngine("unused", execution_policy={name: 1})


# ---------------------------------------------------------------------------
# HTTP streaming of grammar-engaged tool calls (tool_grammar_streaming).


def _call(index, name="weather", city="Toronto"):
    return {
        "index": index,
        "id": f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"city": city})},
    }


class GrammarEngine(FakeEngine):
    def __init__(self, *, streaming=True, engaged="engaged", events=()):
        super().__init__()
        self.streaming, self.engaged, self.script = streaming, engaged, list(events)

    def status(self):
        status = super().status()
        status["settings"] = {"constrained_tool_grammar": True}
        if self.streaming:
            status["settings"]["tool_grammar_streaming"] = True
        return status

    def submit(self, request, *, tenant_id="default"):
        self.job = Job(request)
        self.job.tenant_id = tenant_id
        self.job.prompt_tokens, self.job.completion_tokens = 8, 3
        self.job.tool_grammar_status = self.engaged
        for event in self.script:
            self.job.events.put(event)
        return self.job


@contextlib.contextmanager
def _serve(engine):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _records(response):
    return [
        json.loads(line[6:])
        for line in response.read().decode().splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


FINISH = {"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}}


@pytest.mark.parametrize("streaming, engaged", [(True, "engaged"), (False, "engaged"), (True, "skipped_strict_auto")])
def test_chat_streams_engaged_calls_as_they_complete(streaming, engaged):
    events = [
        {"delta": {"content": "Checking."}},
        {"delta": {"tool_calls": [_call(0)]}},
        {"delta": {"tool_calls": [_call(1, city="Oslo")]}},
        FINISH,
    ]
    engine = GrammarEngine(streaming=streaming, engaged=engaged, events=events)
    with _serve(engine) as base, post(
        base, stream=True, tools=TOOLS, tool_choice="required"
    ) as response:
        records = _records(response)
    deltas = [record["choices"][0].get("delta", {}) for record in records]
    calls = [call for delta in deltas for call in delta.get("tool_calls", ())]
    # Streamed or buffered, the calls reassemble to the same validated set.
    assert [json.loads(call["function"]["arguments"]) for call in calls] == [
        {"city": "Toronto"}, {"city": "Oslo"},
    ]
    assert [call["index"] for call in calls] == [0, 1]
    assert records[-1]["choices"][0]["finish_reason"] == "tool_calls"
    if streaming and engaged == "engaged":
        # One chunk per model event, in model order.
        assert [bool(delta.get("tool_calls")) for delta in deltas[:3]] == [False, True, True]
        assert engine.counts["constrained_tool_grammar_streams"] == 1
    else:
        # Buffered: everything arrives in one validated chunk.
        assert len(deltas[0]["tool_calls"]) == 2 and deltas[0]["content"] == "Checking."
        assert engine.counts["constrained_tool_grammar_streams"] == 0


def test_chat_stream_contract_violation_after_streaming_is_an_sse_error():
    events = [{"delta": {"tool_calls": [_call(0), _call(1)]}}, FINISH]
    engine = GrammarEngine(events=events)
    with _serve(engine) as base, post(
        base, stream=True, tools=TOOLS, tool_choice="required",
        parallel_tool_calls=False,
    ) as response:
        assert response.status == 200
        records = _records(response)
    assert records[0]["choices"][0]["delta"]["tool_calls"]
    assert "error" in records[-1] and "finish_reason" not in str(records[-1])


def test_chat_stream_mid_generation_grammar_failure_is_an_sse_error():
    events = [
        {"delta": {"content": "Checking."}},
        {"error": "structured output failed closed: dead end", "status": 502},
    ]
    engine = GrammarEngine(events=events)
    with _serve(engine) as base, post(
        base, stream=True, tools=TOOLS, tool_choice="auto"
    ) as response:
        records = _records(response)
    assert records[0]["choices"][0]["delta"]["content"] == "Checking."
    assert records[-1]["error"]["message"].startswith("structured output failed closed")


RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "weather",
        "parameters": TOOLS[0]["function"]["parameters"],
        "strict": True,
    }
]


@pytest.mark.parametrize("text_first", [True, False])
def test_responses_stream_function_calls_on_arrival(text_first):
    text = {"delta": {"content": "Checking."}}
    calls = [{"delta": {"tool_calls": [_call(0)]}}, {"delta": {"tool_calls": [_call(1, city="Oslo")]}}]
    events = ([text] + calls if text_first else calls + [text]) + [FINISH]
    engine = GrammarEngine(events=events)
    with _serve(engine) as base, post_response(
        base, stream=True, store=False, tools=RESPONSES_TOOLS, input="weather?",
    ) as response:
        records = _records(response)
    types = [record["type"] for record in records]
    completed = records[-1]["response"]
    assert types[-1] == "response.completed"
    # Function calls were opened before generation finished: their argument
    # deltas precede the message's text done event.
    first_args = types.index("response.function_call_arguments.delta")
    assert first_args < types.index("response.output_text.done")
    # Every streamed index agrees with the final payload order.
    output = completed["output"]
    for record in records:
        if record["type"] in {
            "response.output_item.added", "response.output_item.done",
        }:
            assert output[record["output_index"]]["id"] == record["item"]["id"]
        elif "item_id" in record:
            assert output[record["output_index"]]["id"] == record["item_id"]
    assert [item["type"] for item in output] == (
        ["message", "function_call", "function_call"]
        if text_first
        else ["function_call", "function_call", "message"]
    )
    # output_item.done for a call is sent only after the terminal contract.
    done = [r for r in records if r["type"] == "response.output_item.done"]
    assert len(done) == 3
    last_text = max(i for i, t in enumerate(types) if t == "response.output_text.delta")
    assert all(last_text < records.index(r) for r in done
               if r["item"]["type"] == "function_call")
    assert sum(t == "response.function_call_arguments.delta" for t in types) == 2
