"""Ripple 65B Preview tool calling and reasoning channels, CPU only.

Uses the real tokenizer and chat template of the local oQ4e artifact
(tokenizer files only; weights are never opened).  No model decode runs:
the outputs below are hand-written in the template's own assistant format.
"""

import json
from pathlib import Path

import pytest

from mlx2.adapters.ripple import DESCRIPTOR, RippleAdapter
from mlx2.contracts import Capability
from mlx2.runtime.tool_parsers.hermes_json import parse_tool_call, partial_function_names

MODEL = Path.home() / "Models/Ripple-65B-Preview-oQ4e"
FIXTURES = Path(__file__).parent / "fixtures/ripple"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string"}},
                   "required": ["city"]}}}
BASH = {"type": "function", "function": {
    "name": "execute_bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                           "required": ["command"], "additionalProperties": False}}}
TOOLS = [WEATHER, BASH]
USER = [{"role": "user", "content": "Weather in Paris and Lyon?"}]


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer.json").is_file():
        pytest.skip("local Ripple tokenizer files are not present")
    from transformers import PreTrainedTokenizerFast

    from mlx2.runtime.tokenizer_integrity import repair_loaded_tokenizer
    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

    # The adapter's own tokenizer construction (RippleAdapter._load), minus weights.
    raw = PreTrainedTokenizerFast.from_pretrained(MODEL, local_files_only=True)
    raw.chat_template = (MODEL / "chat_template.jinja").read_text()
    repair_loaded_tokenizer(raw, MODEL)
    return TokenizerWrapper(raw, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[100265])


@pytest.fixture(scope="module")
def adapter(tokenizer):
    adapter = object.__new__(RippleAdapter)
    adapter.tokenizer = tokenizer
    return adapter


def _request(**extra):
    return {"messages": USER, "tools": TOOLS, **extra}


def _run(parser, chunks, finish_reason="stop"):
    events = []
    for chunk in chunks[:-1]:
        events += parser.push(chunk)
    events += parser.finish(chunks[-1] if chunks else "", finish_reason)
    reasoning = "".join(e.get("reasoning_content", "") for e in events)
    content = "".join(e.get("content", "") for e in events)
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]), c["index"])
             for e in events for c in e.get("tool_calls", ())]
    return reasoning, content, calls


def _token_chunks(tokenizer, text):
    """The streaming detokenizer's segments for ``text``, as served."""
    detok = tokenizer.detokenizer
    detok.reset()
    chunks = []
    for token in tokenizer.encode(text, add_special_tokens=False):
        detok.add_token(token)
        chunks.append(detok.last_segment)
    detok.finalize()
    chunks.append(detok.last_segment)
    assert "".join(chunks) == text
    return chunks


def _all_ways(adapter, tokenizer, request, text, finish_reason="stop"):
    """Non-streaming (one push), token-streamed and char-streamed must agree."""
    whole = _run(adapter.output_parser(request), [text], finish_reason)
    tokens = _run(adapter.output_parser(request), _token_chunks(tokenizer, text), finish_reason)
    chars = _run(adapter.output_parser(request), list(text) + [""], finish_reason)
    assert whole == tokens == chars
    return whole


def call_block(name, arguments):
    return "\n<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


# -- descriptor and template ------------------------------------------------

def test_descriptor_declares_tools_without_qualification():
    assert Capability.TOOLS in DESCRIPTOR.capabilities
    assert Capability.REASONING in DESCRIPTOR.capabilities
    assert Capability.GRAMMAR not in DESCRIPTOR.capabilities
    assert DESCRIPTOR.metadata["qualified"] is False


def test_fixture_template_matches_artifact_template(tokenizer):
    assert (FIXTURES / "chat_template.jinja").read_text() == (MODEL / "chat_template.jinja").read_text()


def test_template_renders_tools_calls_and_tool_responses(adapter):
    history = USER + [
        {"role": "assistant", "content": "", "reasoning_content": "Two cities.",
         "tool_calls": [
             {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
             {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Lyon <x>"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "18C"},
        {"role": "tool", "tool_call_id": "b", "content": "20C"},
    ]
    text = adapter.render_prompt({"messages": history, "tools": TOOLS})
    assert text.startswith("<|im_start|>system\n# Tools\n")
    assert "<tools>\n" + json.dumps(WEATHER, ensure_ascii=False) in text
    assert '"name": "execute_bash"' in text
    assert ("<|im_start|>assistant\n<think>\nTwo cities.\n</think>\n"
            '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
            '\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lyon <x>"}}\n</tool_call><|im_end|>\n') in text
    assert ("<|im_start|>user\n<tool_response>\n18C\n</tool_response>"
            "\n<tool_response>\n20C\n</tool_response><|im_end|>\n") in text
    assert text.endswith("<|im_start|>assistant\n")
    # prompt_tokens encodes the same rendering.
    assert adapter.prompt_tokens({"messages": history, "tools": TOOLS}) == adapter.tokenizer.encode(
        text, add_special_tokens=False)


def test_tool_choice_none_renders_no_tools_and_parses_no_calls(adapter, tokenizer):
    request = _request(tool_choice="none")
    text = adapter.render_prompt(request)
    assert "<tools>" not in text and text == adapter.render_prompt({"messages": USER})
    output = "Plain answer." + call_block("get_weather", {"city": "Paris"})
    assert _all_ways(adapter, tokenizer, request, output) == ("", output, [])


def test_thinking_off_is_still_refused_with_tools(adapter):
    with pytest.raises(ValueError, match="thinking-off"):
        adapter.output_parser(_request(enable_thinking=False))
    with pytest.raises(ValueError, match="thinking-off"):
        adapter.render_prompt(_request(enable_thinking=False))


def test_messages_translation_omits_the_off_flag_ripple_would_refuse(adapter):
    from mlx2.anthropic_compat import anthropic_request_to_chat

    assert RippleAdapter.thinking_off_supported is False
    chat = anthropic_request_to_chat(
        {"model": "ripple", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        thinking_off_supported=RippleAdapter.thinking_off_supported,
    )
    assert "enable_thinking" not in chat
    assert adapter.render_prompt(chat) == adapter.render_prompt({"messages": chat["messages"]})


# -- calls -------------------------------------------------------------------

def test_single_call_without_content(adapter, tokenizer):
    output = call_block("get_weather", {"city": "Paris"}).lstrip("\n")
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "", "", [("get_weather", {"city": "Paris"}, 0)])


def test_content_then_parallel_calls_drops_framing_newlines(adapter, tokenizer):
    output = ("Checking both." + call_block("get_weather", {"city": "Paris"})
              + call_block("get_weather", {"city": "Lyon", "unit": "C"}))
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "", "Checking both.",
        [("get_weather", {"city": "Paris"}, 0), ("get_weather", {"city": "Lyon", "unit": "C"}, 1)])


def test_content_newlines_that_are_text_survive(adapter, tokenizer):
    output = "Line one.\n\nLine two.\n"
    assert _all_ways(adapter, tokenizer, _request(), output) == ("", output, [])
    # Only the one newline that frames the call is dropped.
    output = "Line one.\n" + call_block("execute_bash", {"command": "pwd"})
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "", "Line one.\n", [("execute_bash", {"command": "pwd"}, 0)])


def test_reasoning_then_calls(adapter, tokenizer):
    output = ("<think>\nI need the weather.\nCall it.\n</think>\n"
              + call_block("get_weather", {"city": "Paris"}))
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "I need the weather.\nCall it.", "", [("get_weather", {"city": "Paris"}, 0)])


def test_reasoning_content_and_call_as_template_renders_history(adapter, tokenizer):
    # The assistant turn exactly as the template writes it, minus "<|im_start|>assistant\n".
    output = "<think>\nThink.\n</think>\nSure." + call_block("execute_bash", {"command": "ls"})
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "Think.", "Sure.", [("execute_bash", {"command": "ls"}, 0)])


@pytest.mark.parametrize("tools", [False, True])
def test_reasoning_split_with_and_without_tools(adapter, tokenizer, tools):
    request = _request() if tools else {"messages": USER}
    output = "<think>\nStep one.\n\nStep two.\n</think>\nThe answer is 4."
    assert _all_ways(adapter, tokenizer, request, output) == (
        "Step one.\n\nStep two.", "The answer is 4.", [])


@pytest.mark.parametrize("tools", [False, True])
def test_output_without_think_is_all_content(adapter, tokenizer, tools):
    request = _request() if tools else {"messages": USER}
    output = "\nNo reasoning here; </think> is literal.\n"
    assert _all_ways(adapter, tokenizer, request, output) == ("", output, [])


def test_unterminated_reasoning_stays_reasoning(adapter, tokenizer):
    output = "<think>\nStill thinking\n"
    assert _all_ways(adapter, tokenizer, {"messages": USER}, output, "length") == (
        "Still thinking\n", "", [])


def test_tool_markup_inside_reasoning_is_not_a_call(adapter, tokenizer):
    output = "<think>\nMaybe" + call_block("get_weather", {"city": "Paris"}) + "\n</think>\nNo call needed."
    reasoning, content, calls = _all_ways(adapter, tokenizer, _request(), output)
    assert calls == [] and content == "No call needed."
    assert reasoning.startswith("Maybe\n<tool_call>")


def test_quoted_closer_inside_json_string(adapter, tokenizer):
    arguments = {"command": "echo '</tool_call>' > x"}
    output = call_block("execute_bash", arguments)
    assert _all_ways(adapter, tokenizer, _request(), output) == ("", "", [("execute_bash", arguments, 0)])


def test_fenced_example_is_not_a_call(adapter, tokenizer):
    output = "Example:\n```\n<tool_call>\n{}\n</tool_call>\n```\n" + call_block("get_weather", {"city": "Nice"})
    reasoning, content, calls = _all_ways(adapter, tokenizer, _request(), output)
    assert content == "Example:\n```\n<tool_call>\n{}\n</tool_call>\n```\n"  # one framing newline dropped
    assert calls == [("get_weather", {"city": "Nice"}, 0)]
    # The fence's closing newline is the call's framing newline: still closes the fence.
    output = "Example:\n```\nx\n```" + call_block("get_weather", {"city": "Nice"})
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "", "Example:\n```\nx\n```", [("get_weather", {"city": "Nice"}, 0)])


def test_parallel_tool_calls_false_keeps_first_call(adapter, tokenizer):
    request = _request(parallel_tool_calls=False)
    output = call_block("get_weather", {"city": "Paris"}) + call_block("get_weather", {"city": "Lyon"})
    parser = adapter.output_parser(request)
    assert _run(parser, [output])[2] == [("get_weather", {"city": "Paris"}, 0)]
    assert parser.tool_call_constraint_truncations == 1


def test_named_and_required_tool_choice_constrain_parser(adapter):
    named = adapter.output_parser(_request(tool_choice={"type": "function", "function": {"name": "get_weather"}}))
    required = adapter.output_parser(_request(tool_choice="required"))
    auto = adapter.output_parser(_request())
    assert named.constrained_tools and required.constrained_tools and not auto.constrained_tools


# -- malformed calls fail closed ---------------------------------------------

@pytest.mark.parametrize("body", [
    "not json",
    '{"name": "get_weather"}',
    '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}',
    '{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "x"}',
    '{"name": "unknown_tool", "arguments": {}}',
    '{"name": "get_weather", "arguments": {}}',
    '{"name": "execute_bash", "arguments": {"command": "ls", "extra": 1}}',
    '{"name": "get_weather", "arguments": {"city": NaN}}',
    '[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
])
def test_malformed_call_fails_closed(adapter, tokenizer, body):
    output = "<tool_call>\n" + body + "\n</tool_call>"
    for chunks in ([output], _token_chunks(tokenizer, output)):
        with pytest.raises(ValueError):
            _run(adapter.output_parser(_request()), chunks)


def test_tolerant_markers_return_malformed_call_as_text(adapter):
    output = "Hi\n<tool_call>\nnot json\n</tool_call>"
    parser = adapter.output_parser(_request(_tolerant_tool_markers=True))
    assert _run(parser, [output]) == ("", output, [])
    assert parser.tool_call_parse_fallbacks == 1
    with pytest.raises(ValueError):  # a forced choice is never tolerant
        _run(adapter.output_parser(_request(_tolerant_tool_markers=True, tool_choice="required")), [output])


def test_incomplete_call_fails_closed_and_length_cut_is_recorded(adapter):
    output = 'Hi\n<tool_call>\n{"name": "get_weather", "arguments": {"ci'
    with pytest.raises(ValueError, match="incomplete"):
        _run(adapter.output_parser(_request()), [output])
    parser = adapter.output_parser(_request())
    assert _run(parser, [output], "length") == ("", "Hi", [])
    assert parser.truncated_tool_call == {"function": "get_weather", "cause": "max_tokens"}


def test_stop_string_cut_inside_call(adapter):
    parser = adapter.output_parser(_request(stop=["Lyon"]))
    output = 'A\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lyon"}}\n</tool_call>'
    assert _run(parser, [output]) == ("", "A", [])
    assert parser.stop_truncated_tool_call == {"function": "get_weather"}


# -- the JSON body grammar ---------------------------------------------------

def test_hermes_json_parser_contract():
    assert parse_tool_call(' \n{"name": "get_weather", "arguments": {"city": "P"}}\n', TOOLS) == {
        "name": "get_weather", "arguments": {"city": "P"}}
    with pytest.raises(ValueError) as error:
        parse_tool_call('{"name": "execute_bash", "arguments": {"command": "echo ', TOOLS)
    assert getattr(error.value, "tool_call_close_quoted", False)
    with pytest.raises(ValueError) as error:
        parse_tool_call('{"name": "execute_bash", "arguments": {"command": 1', TOOLS)
    assert not getattr(error.value, "tool_call_close_quoted", False)
    assert partial_function_names("") == []
    assert partial_function_names('\n{"name": "get_wea') == []
    assert partial_function_names('\n{"name": "get_weather", "argu') == ["get_weather"]
    with pytest.raises(ValueError):
        partial_function_names("get_weather(")
    assert parse_tool_call.partial_function_names is partial_function_names


def test_external_draft_routes_share_the_parser(adapter):
    # DFlash2 chain and AST tree routes detokenize accepted tokens into the
    # same parser; neither route removes TOOLS from the descriptor.
    from mlx2.qualification import effective_route_capabilities

    for external in (False, True):
        selected = effective_route_capabilities(DESCRIPTOR.capabilities, mtp=False, external_draft=external,
                                                prompt_lookup=False, max_lanes=1)
        assert Capability.TOOLS in selected
    drafted = object.__new__(RippleAdapter)
    drafted.tokenizer, drafted.draft_model = adapter.tokenizer, object()
    assert type(drafted.output_parser(_request())) is type(adapter.output_parser(_request()))


# -- review follow-ups ------------------------------------------------------

@pytest.mark.parametrize("tools", [False, True])
def test_newline_run_after_think_close_is_dropped(adapter, tokenizer, tools):
    # Live Ripple output writes "</think>\n\n" before an answer.
    request = _request() if tools else {"messages": USER}
    assert _all_ways(adapter, tokenizer, request, "<think>\nR\n</think>\n\nfoo") == ("R", "foo", [])
    assert _all_ways(adapter, tokenizer, request, "<think>\nR\n</think>\nfoo") == ("R", "foo", [])
    assert _all_ways(adapter, tokenizer, request, "<think>\nR\n</think>\n\nfoo\n\nbar")[1] == "foo\n\nbar"


def test_reasoning_then_tool_only_turn_drops_both_framing_newlines(adapter, tokenizer):
    # Tool-only turn after reasoning: "</think>\n" + "" + "\n<tool_call>...".
    output = "<think>\nPlan.\n</think>\n" + call_block("get_weather", {"city": "Paris"})
    assert _all_ways(adapter, tokenizer, _request(), output) == (
        "Plan.", "", [("get_weather", {"city": "Paris"}, 0)])


def test_dropped_parallel_call_keeps_its_framing_newline_out_of_content(adapter, tokenizer):
    one = call_block("get_weather", {"city": "Paris"})
    output = "A" + one + "\nB" + one
    capped = _all_ways(adapter, tokenizer, _request(parallel_tool_calls=False), output)
    free = _all_ways(adapter, tokenizer, _request(), output)
    assert capped[1] == free[1] == "A\nB"
    assert len(capped[2]) == 1 and len(free[2]) == 2


def test_operator_sampling_defaults_and_no_thinking_override():
    from mlx2.sampling_defaults import resolve_sampling, vendor_sampling

    declared = vendor_sampling(RippleAdapter, {"messages": USER})
    assert declared.model == "ricursiveAI/Ripple-65B-Preview"
    sampling, receipt = resolve_sampling({"messages": USER}, declared, thinking=False)
    assert (sampling["temperature"], sampling["top_p"], sampling["top_k"]) == (0.7, 0.95, 20)
    assert sampling["min_p"] == 0.0
    override, _ = resolve_sampling({"messages": USER, "temperature": 0, "top_k": 0}, declared, thinking=False)
    assert (override["temperature"], override["top_p"], override["top_k"]) == (0, 0.95, 0)
    # The model decides per turn whether to open <think>: the engine must not
    # arm thinking paths for every chat request.
    assert not hasattr(RippleAdapter, "thinking_enabled")


# -- implicit reasoning (no <think>, bare </think>) --------------------------

KNAVES = ("We need answer. Need solve.\n\nCase 1: B knight, so C knave.\nNeed final.\n</think>\n\n"
          "C is a **knave**.")


@pytest.mark.parametrize("tools", [True, False])
def test_implicit_reasoning_closed_by_bare_think_close(adapter, tokenizer, tools):
    request = _request() if tools else {"messages": USER}
    assert _all_ways(adapter, tokenizer, request, KNAVES) == (
        "We need answer. Need solve.\n\nCase 1: B knight, so C knave.\nNeed final.", "C is a **knave**.", [])
    parser = adapter.output_parser(request)
    parser.finish(KNAVES, "stop")
    assert parser.implicit_reasoning is True


def test_implicit_reasoning_then_tool_call(adapter, tokenizer):
    text = ("The user wants weather in Paris. Call the tool.\n</think>\n\nChecking."
            + call_block("get_weather", {"city": "Paris"}))
    assert _all_ways(adapter, tokenizer, _request(), text) == (
        "The user wants weather in Paris. Call the tool.", "Checking.", [("get_weather", {"city": "Paris"}, 0)])


@pytest.mark.parametrize("text", [
    "The user manual says to restart it.",  # opener, never closed: an answer
    "We need more data before deciding.\n\nTwo reasons follow.",
    "We",  # undecided prefix at the end of the output
    "Answer first.\n</think>\nstill content",  # bare close without an opener stays literal
])
def test_held_or_unmatched_output_stays_content(adapter, tokenizer, text):
    assert _all_ways(adapter, tokenizer, {"messages": USER}, text) == ("", text, [])


def test_opener_then_tool_call_before_any_close_is_content(adapter, tokenizer):
    text = "We need the weather." + call_block("get_weather", {"city": "Paris"})
    assert _all_ways(adapter, tokenizer, _request(), text) == (
        "", "We need the weather.", [("get_weather", {"city": "Paris"}, 0)])


def test_stop_string_before_close_releases_held_text_as_content(adapter, tokenizer):
    text = "We need X. STOP then more\n</think>\nanswer"
    request = {"messages": USER, "stop": ["STOP"]}
    assert _all_ways(adapter, tokenizer, request, text) == ("", "We need X. ", [])


def test_implicit_reasoning_cut_by_max_tokens_is_content(adapter, tokenizer):
    text = "We need answer. Let's search memory. Let's"
    assert _all_ways(adapter, tokenizer, {"messages": USER}, text, "length") == ("", text, [])


def test_explicit_think_block_unchanged(adapter, tokenizer):
    text = "<think>\nWe need answer.\n</think>\n\nRome."
    assert _all_ways(adapter, tokenizer, {"messages": USER}, text) == ("We need answer.", "Rome.", [])


def test_held_pushes_are_handed_back_as_reasoning_tokens(adapter):
    parser = adapter.output_parser({"messages": USER})
    close_end = KNAVES.index("</think>") + len("</think>")
    for char in KNAVES[:close_end - 1]:
        assert parser.push(char) == []
    assert parser.take_deferred_reasoning_tokens() == 0
    parser.push(KNAVES[close_end - 1])
    # Every held push was reasoning; the releasing push is counted by the loop.
    assert parser.take_deferred_reasoning_tokens() == close_end - 1
    assert parser.take_deferred_reasoning_tokens() == 0
    held_content = adapter.output_parser({"messages": USER})
    _run(held_content, list("The user manual says so.") + [""])
    assert held_content.take_deferred_reasoning_tokens() == 0


def test_opener_needs_a_word_boundary(adapter, tokenizer):
    text = "The username field is required.\n</think>\nliteral"
    assert _all_ways(adapter, tokenizer, {"messages": USER}, text) == ("", text, [])
