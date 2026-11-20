"""Hy4-flash tool-call parsing (CPU, no model)."""

import json
from pathlib import Path

import pytest

from mlx2.adapters.hy_v4_flash_output import HYV4OutputParser, output_parser, tool_constraint
from mlx2.runtime.tool_parsers.hy_v4 import detect_suffix, make_parser

S = ":opensource"
CALLS, CALLS_END = f"<tool_calls{S}>", f"</tool_calls{S}>"
CALL, CALL_END = f"<tool_call{S}>", f"</tool_call{S}>"
K, K_END, V, V_END = f"<arg_key{S}>", f"</arg_key{S}>", f"<arg_value{S}>", f"</arg_value{S}>"
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"},
    "filters": {"type": "object"}, "tags": {"type": "array"}, "score": {"type": "number"},
    "note": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}}},
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}]
TEMPLATE = Path.home() / "Models/Hy4-flash/chat_template.jinja"


def call(name, **args):
    body = name + "".join(f"{K}{k}{K_END}{V}{v}{V_END}" for k, v in args.items())
    return CALL + body + CALL_END


def run(text, *, chunks=None, request=None, **kwargs):
    request = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS, **(request or {})}
    parser = output_parser(request, suffix=S) if not kwargs else HYV4OutputParser(
        suffix=S, chat=True, tools=TOOLS, **kwargs)
    events = []
    pieces = [text] if chunks is None else [text[i:i + chunks] for i in range(0, len(text), chunks)]
    for i, piece in enumerate(pieces):
        events += parser.push(piece, final=i == len(pieces) - 1)
    content = "".join(e.get("content", "") for e in events)
    calls = [c for e in events for c in e.get("tool_calls", ())]
    return content, calls


def test_typed_values_follow_schema_and_template_encoding():
    parse = make_parser(S)
    body = call("  search \n", query="  two words \n", limit="5", exact="true",
                filters='{"a": [1, 2]}', tags='["x"]', score="2.5", note="null")[len(CALL):-len(CALL_END)]
    out = parse(body, TOOLS)
    assert out["name"] == "search"
    assert out["arguments"] == {"query": "  two words \n", "limit": 5, "exact": True,
                                "filters": {"a": [1, 2]}, "tags": ["x"], "score": 2.5, "note": None}


def test_argument_names_keep_whitespace_function_name_is_stripped():
    out = make_parser(S)(f" now {K} spaced key {K_END}{V}1{V_END}", TOOLS)
    assert out["name"] == "now" and list(out["arguments"]) == [" spaced key "]


def test_value_typing_without_and_with_a_declared_schema():
    parse = make_parser(S)
    # An undeclared property defaults to a string (vLLM: missing type = string).
    assert parse(f"now{K}a{K_END}{V}[1, 2]{V_END}", TOOLS)["arguments"] == {"a": "[1, 2]"}
    # A declared non-string type decodes JSON.
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {
        "a": {"type": ["array", "string"]}, "b": {"oneOf": [{"type": "integer"}, {"type": "string"}]}}}}}]
    assert parse(f"f{K}a{K_END}{V}[1, 2]{V_END}{K}b{K_END}{V}7{V_END}", tools)["arguments"] == {"a": [1, 2], "b": 7}


@pytest.mark.parametrize("chunks", [None, 1, 3, 17])
def test_stream_any_chunking_gives_the_same_calls(chunks):
    text = ("Let me look." + CALLS + call("search", query="mlx", limit="2") + call("now") + CALLS_END)
    content, calls = run(text, chunks=chunks)
    assert content == "Let me look."
    assert [c["function"]["name"] for c in calls] == ["search", "now"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "mlx", "limit": 2}
    assert [c["index"] for c in calls] == [0, 1]


def test_one_delta_with_many_tags_equals_tag_by_tag():
    # Speculative/AST decode commits several tokens per step (vllm#55550).
    text = CALLS + call("search", query="a") + call("search", query="b") + CALLS_END
    strip = lambda result: (result[0], [{**c, "id": None} for c in result[1]])
    assert strip(run(text)) == strip(run(text, chunks=1))


def test_wrapper_markers_never_reach_content():
    content, calls = run("Done." + CALLS + call("now") + CALLS_END + " ")
    assert CALLS not in content and CALLS_END not in content and len(calls) == 1


def test_nameless_call_fails_closed_or_falls_back_to_text():
    text = CALLS + CALL + f"{K}q{K_END}{V}x{V_END}" + CALL_END + CALLS_END
    with pytest.raises(ValueError):
        run(text)
    content, calls = run(text, request={"_tolerant_tool_markers": True})
    assert not calls and f"{K}q{K_END}" in content


def test_undeclared_tool_is_refused_and_auto_has_no_grammar():
    with pytest.raises(ValueError, match="undeclared"):
        run(CALLS + call("delete_everything") + CALLS_END)
    assert tool_constraint({"tools": TOOLS, "tool_choice": "auto"}) is None
    assert tool_constraint({"tools": TOOLS, "tool_choice": "none"}) is None


def test_suffix_detection():
    assert detect_suffix({"<tool_call:abc>": 5}) == ":abc"
    assert detect_suffix({}, "{%- set HYTK = ':opensource' %}") == ":opensource"
    assert detect_suffix({}) == ""


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="Hy4-flash chat template not on this host")
def test_round_trip_through_the_checkpoint_chat_template():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"a": 0}, unk_token="a")))
    tok.chat_template = TEMPLATE.read_text()
    args = {"query": "  café <b> \n", "limit": 3, "filters": {"k": [1, None]}, "exact": False}
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": args}}]}]
    text = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, reasoning_effort="no_think")
    # The tools prompt shows the format too; take the assistant's own block.
    start = text.rindex(CALLS)
    segment = text[start : text.index(CALLS_END, start) + len(CALLS_END)]
    _, calls = run(segment)
    assert [c["function"]["name"] for c in calls] == ["search"]
    assert json.loads(calls[0]["function"]["arguments"]) == args


def test_degenerate_argument_runs_parse_in_linear_time():
    import time

    parse = make_parser(S)
    body = "now" + f"{K}k{K_END}" * 8000
    start = time.perf_counter()
    with pytest.raises(ValueError):
        parse(body, TOOLS)
    assert time.perf_counter() - start < 0.5


def test_deeply_nested_value_is_handled_not_crashing():
    deep = "[" * 100000 + "]" * 100000
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {"a": {"type": "array"}}}}}]
    assert make_parser(S)(f"f{K}a{K_END}{V}{deep}{V_END}", tools)["arguments"]["a"] == deep


def test_local_refs_resolve_like_vllm():
    params = {"$defs": {"N": {"type": "integer"}, "M": {"type": "object"}},
              "properties": {"n": {"$ref": "#/$defs/N"},
                             "m": {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]},
                             "x": {"$ref": "https://example.com/s.json"}}}
    tools = [{"type": "function", "function": {"name": "f", "parameters": params}}]
    out = make_parser(S)(f"f{K}n{K_END}{V}5{V_END}{K}m{K_END}{V}{{\"x\": 1}}{V_END}{K}x{K_END}{V}5{V_END}", tools)
    assert out["arguments"] == {"n": 5, "m": {"x": 1}, "x": "5"}


def test_type_names_are_case_sensitive_like_vllm_and_numbers_finite():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"properties": {
        "a": {"type": "String"}, "b": {"type": "number"}}}}}]
    out = make_parser(S)(f"f{K}a{K_END}{V}123{V_END}{K}b{K_END}{V}1e999{V_END}", tools)
    assert out["arguments"] == {"a": 123, "b": "1e999"}


def test_partial_name_rules_match_the_full_parser():
    names = make_parser(S).partial_function_names
    assert names(f"{CALL}search{K}q") == ["search"]
    with pytest.raises(ValueError):
        names(f"{CALL}a<b{K}q")


def test_template_suffix_in_double_quotes():
    assert detect_suffix({}, '{%- set HYTK = ":x1" %}') == ":x1"


# --- Forced tool_choice grammar -------------------------------------------

import regex  # noqa: E402

from mlx2.adapters.hy_v4_flash_output import reasoning_effort, thinking_enabled  # noqa: E402

THINK, THINK_END = f"<think{S}>", f"</think{S}>"
FORCED_TOOLS = [
    {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"},
        "score": {"type": "number"}, "mode": {"type": "string", "enum": ["fast", "deep"]},
        "note": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "filters": {"type": "object"}},
        "required": ["query"]}}},
    {"type": "function", "function": {"name": "now"}},
]


def forced(choice="required", *, parallel=True, tools=FORCED_TOOLS, suffix=S):
    grammar = tool_constraint({"tools": tools, "tool_choice": choice, "parallel_tool_calls": parallel},
                              suffix=suffix)
    return regex.compile(f"(?:{grammar})")


def block(*calls):
    return CALLS + "".join(calls) + CALLS_END


@pytest.mark.parametrize("text", [
    block(call("search", query="mlx")),
    block(call("search", query="  two words \n<b>", limit="5", exact="false", score="2.5",
               mode="deep", note="null", filters='{"a": [1]}')),
    block(call("search", query="", limit="-3", note="7")),
    block(call("now"), call("search", query="x")),
    block(call("now", anything="goes")),
])
def test_required_grammar_accepts_valid_blocks(text):
    assert forced().fullmatch(text)
    # Every accepted block parses to declared calls with schema-typed values.
    _, calls = run(text, request={"tools": FORCED_TOOLS, "tool_choice": "required"})
    assert calls


@pytest.mark.parametrize("text", [
    "Sure." + block(call("search", query="x")),          # forced: the block is the whole answer
    block(),                                              # no call
    block(call("delete", query="x")),                     # undeclared function
    block(call("search")),                                # required argument missing
    block(call("search", limit="5", query="x")),          # declared order
    block(call("search", query="x", query2="y")),         # undeclared key on a declared schema
    block(call("search", query="x", limit="five")),       # integer
    block(call("search", query="x", limit="2.5")),
    block(call("search", query="x", exact="yes")),        # boolean
    block(call("search", query="x", score="1e999")),      # non-finite number
    block(call("search", query="x", mode="slow")),        # enum
    block(call("search", query="x", note="null ")),
    block(call("search", query=f"a{V_END}b")),            # a value cannot close itself early
    block(call("search", query=f"a{CALL_END}b")),
    block(call("search", query=f"a{K}b")),
    CALLS + call(" search", query="x") + CALLS_END,       # the name is exact
    CALLS + call("search", query="x"),                    # unclosed wrapper
    block(call("search", query="x")) + "trailing",
])
def test_required_grammar_rejects_invalid_blocks(text):
    assert not forced().fullmatch(text)


def test_named_choice_and_parallel_bound():
    named = forced({"type": "function", "function": {"name": "now"}})
    assert named.fullmatch(block(call("now")))
    assert named.fullmatch(block(call("now"), call("now")))
    assert not named.fullmatch(block(call("search", query="x")))
    single = forced(parallel=False)
    assert single.fullmatch(block(call("now")))
    assert not single.fullmatch(block(call("now"), call("now")))
    # A partial block stays live (what the token mask needs at each step).
    assert forced().fullmatch(CALLS + CALL + "sea", partial=True)
    assert not forced().fullmatch(CALLS + CALL + "nope", partial=True)


def test_grammar_refuses_what_the_wire_cannot_carry():
    with pytest.raises(ValueError, match="selects no declared function"):
        forced({"type": "function", "function": {"name": "missing"}})
    for name in ("a<b", " padded"):
        with pytest.raises(ValueError):
            forced(tools=[{"type": "function", "function": {"name": name}}])
    with pytest.raises(ValueError):
        forced(tools=[{"type": "function", "function": {"name": "f", "parameters": {
            "properties": {f"k{K_END}": {"type": "string"}}, "required": [f"k{K_END}"]}}}])
    # strict schemas the grammar cannot bound fail closed instead of decoding loosely
    for prop in ({"type": "object"}, {"type": "string", "maxLength": 4}, {"type": ["string", "integer"]}):
        with pytest.raises(ValueError, match="strict"):
            forced(tools=[{"type": "function", "function": {"name": "f", "strict": True, "parameters": {
                "type": "object", "properties": {"a": prop}, "required": ["a"],
                "additionalProperties": False}}}])
    # ... also when the constraint sits in a union branch or a $ref target,
    # which the parser's type resolution drops.
    for prop, defs in (({"anyOf": [{"type": "string", "maxLength": 2}, {"type": "null"}]}, {}),
                       ({"anyOf": [{"type": "string", "enum": ["x"]}, {"type": "null"}]}, {}),
                       ({"anyOf": [{"type": "integer", "minimum": 0}]}, {}),
                       ({"$ref": "#/$defs/S"}, {"S": {"type": "string", "maxLength": 2}}),
                       ({"$ref": "https://example.com/s.json"}, {})):
        with pytest.raises(ValueError, match="strict"):
            forced(tools=[{"type": "function", "function": {"name": "f", "strict": True, "parameters": {
                "type": "object", "$defs": defs, "properties": {"a": prop}, "required": ["a"],
                "additionalProperties": False}}}])
    strict = forced(tools=[{"type": "function", "function": {"name": "f", "strict": True, "parameters": {
        "type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"},
                                         "e": {"type": ["string", "null"], "enum": ["a", None]},
                                         "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
        "required": ["n", "s", "e", "u"], "additionalProperties": False}}}])
    assert strict.fullmatch(block(call("f", n="3", s="x", e="a", u="null")))
    assert strict.fullmatch(block(call("f", n="3", s="x", e="null", u="4")))
    assert not strict.fullmatch(block(call("f", n="3", s="x", e="b", u="4")))
    assert not strict.fullmatch(block(call("f", n="3")))
    # An integer enum holding a float is not an integer enum.
    mixed = forced(tools=[{"type": "function", "function": {"name": "f", "parameters": {
        "properties": {"i": {"type": "integer", "enum": [1, 2.0]}}, "required": ["i"]}}}])
    assert mixed.fullmatch(block(call("f", i="1"))) and not mixed.fullmatch(block(call("f", i="2.0")))


def test_unsuffixed_wire_and_token_automaton():
    from mlx2.structured_automaton import automaton_for

    plain = forced(suffix="")
    assert plain.fullmatch("<tool_calls><tool_call>now</tool_call></tool_calls>")
    assert not plain.fullmatch("<tool_calls><tool_call>search<arg_key>query</arg_key>"
                               "<arg_value>a</arg_value>b</arg_value></tool_call></tool_calls>")
    # Free keys of a property-less schema cannot close themselves early.
    assert plain.fullmatch("<tool_calls><tool_call>now<arg_key>a</arg_key>"
                           "<arg_value>v</arg_value></tool_call></tool_calls>")
    assert not plain.fullmatch("<tool_calls><tool_call>now<arg_key>a</arg_key>b</arg_key>"
                               "<arg_value>v</arg_value></tool_call></tool_calls>")
    # The exact token automaton compiles the grammar (no lookaround, under
    # the state caps) for the suffixed and the plain wire.
    for suffix in (S, ""):
        automaton_for(forced(suffix=suffix))


def test_grammar_admits_the_block_the_template_renders():
    template = _toy_template_tokenizer()
    if template is None:
        pytest.skip("Hy4-flash chat template not on this host")
    args = {"query": "  café <b> \n", "limit": 3, "exact": False, "note": None, "filters": {"k": [1, None]}}
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "search", "arguments": args}}]}]
    text = template.apply_chat_template(messages, tools=FORCED_TOOLS, tokenize=False, reasoning_effort="no_think")
    start = text.rindex(CALLS)
    segment = text[start: text.index(CALLS_END, start) + len(CALLS_END)]
    assert forced().fullmatch(segment)
    _, calls = run(segment, request={"tools": FORCED_TOOLS, "tool_choice": "required"})
    assert json.loads(calls[0]["function"]["arguments"]) == args


def test_forced_choice_parser_fails_closed_on_malformed_output():
    request = {"tools": TOOLS, "tool_choice": "required", "_tolerant_tool_markers": True}
    with pytest.raises(ValueError):
        run(CALLS + CALL + f"{K}q{K_END}{V}x{V_END}" + CALL_END + CALLS_END, request=request)


# --- Reasoning channel ------------------------------------------------------


def test_reasoning_is_opt_in_and_fails_closed_on_unknown_levels():
    chat = {"messages": [{"role": "user", "content": "x"}]}
    assert reasoning_effort(chat) == "no_think" and not thinking_enabled(chat)
    assert reasoning_effort({**chat, "enable_thinking": True}) == "high"
    with pytest.raises(ValueError, match="removed upstream"):
        reasoning_effort({**chat, "enable_thinking": True, "reasoning_effort": "max"})
    assert reasoning_effort({**chat, "reasoning_effort": "high"}) == "high"
    assert reasoning_effort({**chat, "enable_thinking": False, "reasoning_effort": "high"}) == "no_think"
    assert reasoning_effort({**chat, "enable_thinking": False, "reasoning_effort": "none"}) == "no_think"
    assert not thinking_enabled({"prompt": "x", "enable_thinking": True})
    for bad in ({"reasoning_effort": "medium", "enable_thinking": True},
                {"reasoning_effort": "low"}, {"reasoning_effort": 40},
                {"reasoning_effort": "none", "enable_thinking": True}):
        with pytest.raises(ValueError):
            reasoning_effort({**chat, **bad})
        with pytest.raises(ValueError):
            output_parser({**chat, **bad}, suffix=S)
        assert thinking_enabled({**chat, **bad}) is False


def reason(text, *, chunks=None, request=None):
    request = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS,
               "reasoning_effort": "high", "enable_thinking": True, **(request or {})}
    parser = output_parser(request, suffix=S)
    events = []
    pieces = [text] if chunks is None else [text[i:i + chunks] for i in range(0, len(text), chunks)]
    for i, piece in enumerate(pieces):
        events += parser.finish(piece, "stop") if i == len(pieces) - 1 else parser.push(piece)
    return ("".join(e.get("reasoning_content", "") for e in events),
            "".join(e.get("content", "") for e in events),
            [c for e in events for c in e.get("tool_calls", ())], parser)


@pytest.mark.parametrize("chunks", [None, 1, 2, 5, 13])
def test_reasoning_split_streams_the_same_at_any_chunking(chunks):
    # Tool markup inside reasoning is thinking about a call, never a call.
    text = (f"Plan: maybe {CALLS}{call('now')}{CALLS_END}, or <think> plain."
            f"{THINK_END}Answer.{block(call('search', query='mlx', limit='2'))}")
    reasoning, content, calls, _ = reason(text, chunks=chunks)
    assert reasoning == f"Plan: maybe {CALLS}{call('now')}{CALLS_END}, or <think> plain."
    assert content == "Answer."
    assert [c["function"]["name"] for c in calls] == ["search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "mlx", "limit": 2}


@pytest.mark.parametrize("chunks", [None, 1, 4])
def test_reasoning_then_forced_block_parses(chunks):
    # The deferred grammar binds after </think:S>: reasoning, then the block.
    text = "think it through" + THINK_END + block(call("now"))
    reasoning, content, calls, _ = reason(text, chunks=chunks, request={"tool_choice": "required"})
    assert (reasoning, content) == ("think it through", "")
    assert [c["function"]["name"] for c in calls] == ["now"]
    assert forced().fullmatch(text[len("think it through") + len(THINK_END):])


def test_unclosed_reasoning_stays_reasoning():
    reasoning, content, calls, _ = reason("still thinking about " + CALL)
    assert reasoning == "still thinking about " + CALL and content == "" and not calls


@pytest.mark.parametrize("chunks", [None, 1, 3])
def test_no_think_output_is_all_content_and_generic_think_tags_are_text(chunks):
    content, calls = run("<think>not a marker</think> done", chunks=chunks)
    assert content == "<think>not a marker</think> done" and not calls
    # The prompt already closed the think block (vLLM no_think): a leading
    # <think:S> is text, never a reasoning channel.
    text = THINK + "hidden" + THINK_END + "visible"
    request = {"messages": [{"role": "user", "content": "x"}]}
    parser = output_parser(request, suffix=S)
    pieces = [text] if chunks is None else [text[i:i + chunks] for i in range(0, len(text), chunks)]
    events = [e for i, p in enumerate(pieces) for e in parser.push(p, final=i == len(pieces) - 1)]
    assert "".join(e.get("content", "") for e in events) == text
    assert not any("reasoning_content" in e for e in events)


@pytest.mark.parametrize("chunks", [None, 1, 3])
def test_stop_strings_apply_to_the_answer_not_the_reasoning(chunks):
    request = {"stop": ["STOP"]}
    reasoning, content, _, parser = reason(f"I could say STOP here{THINK_END}Fine. STOP more", chunks=chunks,
                                           request=request)
    assert reasoning == "I could say STOP here" and content == "Fine. " and parser.stop_sequence == "STOP"
    # A stop string split across the think close and the answer.
    reasoning, content, _, parser = reason(f"r{THINK_END}abST" + "OPcd", chunks=chunks, request=request)
    assert (reasoning, content, parser.stop_sequence) == ("r", "ab", "STOP")


def test_stop_string_cutting_a_call_after_reasoning_is_a_requested_stop():
    reasoning, content, calls, parser = reason(
        f"r{THINK_END}" + block(call("search", query="aXYb")), request={"stop": ["XY"]})
    assert reasoning == "r" and not calls
    assert parser.stop_truncated_tool_call == {"function": "search"}


def test_reasoning_counters_feed_diagnostics():
    from mlx2.adapters.hy_v4_flash_output import FeatureCounters

    stats = FeatureCounters()
    parser = output_parser({"messages": [], "reasoning_effort": "high", "enable_thinking": True},
                           suffix=S, stats=stats)
    parser.push("hmm")
    parser.push(THINK_END + "ok", final=True)
    output_parser({"messages": []}, suffix=S, stats=stats).push(THINK + "x" + THINK_END, final=True)
    tool_constraint({"tools": TOOLS, "tool_choice": "required"}, suffix=S, stats=stats)
    assert stats.snapshot() == {"reasoning_requests": 1, "reasoning_emitted": 1, "tool_grammar_hook_calls": 1}


# --- Template rendering through the adapter ---------------------------------


def _toy_template_tokenizer():
    if not TEMPLATE.is_file():
        return None
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"a": 0}, unk_token="a")))
    tok.chat_template = TEMPLATE.read_text()
    tok.add_tokens([THINK, THINK_END], special_tokens=True)
    return tok


def _template_adapter():
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter

    tok = _toy_template_tokenizer()
    if tok is None:
        pytest.skip("Hy4-flash chat template not on this host")
    adapter = object.__new__(HYV4FlashAdapter)
    adapter.tokenizer = tok
    adapter.tool_suffix = S
    from mlx2.adapters.hy_v4_flash_output import FeatureCounters

    adapter.feature_stats = FeatureCounters()
    return adapter


ASSISTANT = f"<｜hy_start{S}｜>assistant<｜hy_middle{S}｜>"


@pytest.mark.parametrize("extra, effort, tail", [
    ({}, "no_think", ASSISTANT + THINK + THINK_END),
    ({"enable_thinking": False}, "no_think", ASSISTANT + THINK + THINK_END),
    ({"enable_thinking": True}, "high", ASSISTANT + THINK),
    ({"reasoning_effort": "high", "enable_thinking": True}, "high", ASSISTANT + THINK),
])
def test_template_renders_the_selected_reasoning_level(extra, effort, tail):
    adapter = _template_adapter()
    request = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS, **extra}
    text = adapter.render_prompt(request)
    assert text.endswith(tail)
    assert f"<｜reasoning_mode{S}｜>reasoning_effort:{effort}" in text
    assert adapter.thinking_enabled(request) is (effort != "no_think")
    assert adapter.output_parser(request).channel == ("content" if effort == "no_think" else "reasoning_content")


def test_template_history_keeps_reasoning_only_when_thinking():
    adapter = _template_adapter()
    messages = [{"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1", "reasoning_content": "secret plan"},
                {"role": "user", "content": "q2"}]
    assert "secret plan" not in adapter.render_prompt({"messages": messages})
    # preserved_thinking is on with tools: past reasoning goes back in.
    thinking = adapter.render_prompt({"messages": messages, "tools": TOOLS, "reasoning_effort": "high",
                                      "enable_thinking": True})
    assert THINK + "secret plan" + THINK_END + "a1" in thinking


def test_thinking_close_token_is_one_token_for_deferred_grammars():
    adapter = _template_adapter()
    ids = adapter.thinking_close_token_ids()
    assert ids is not None and len(ids) == 1
    assert adapter.tokenizer.decode(list(ids)) == THINK_END
    # Special marker tokens the grammar spells are declared so the mask
    # admits them; markers the toy vocab lacks are not.
    adapter.tokenizer.add_tokens([CALLS, CALL], special_tokens=True)
    declared = {adapter.tokenizer.decode([i]) for i in adapter.structured_special_token_ids()}
    assert declared == {CALLS, CALL, THINK_END}
    adapter.tool_suffix = ":other"
    assert adapter.thinking_close_token_ids() is None
