"""Kimi K2 wire tool calls: parser, streaming markers, template inputs (CPU)."""
import json

import pytest

from mlx2.adapters.kimi_linear_output import KimiOutputParser, make_output_parser, tool_constraint
from mlx2.runtime.tool_parsers.kimi import UnclosedJSONString, parse_tool_call, partial_function_names

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
             "type": "object", "properties": {"city": {"type": "string"}}}}},
         {"type": "function", "function": {"name": "search", "parameters": {
             "type": "object", "properties": {"q": {"type": "string"}}}}}]
REQUEST = {"messages": [{"role": "user", "content": "x"}], "tools": TOOLS}
TWO_CALLS = ('Checking.<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0'
             '<|tool_call_argument_begin|>{"city": "Paris <|tool_call_end|> x"}<|tool_call_end|>'
             '<|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{"q":"mlx"}'
             '<|tool_call_end|><|tool_calls_section_end|>')


def run(parser, text, step):
    events = []
    for i in range(0, len(text), step):
        events += parser.push(text[i:i + step])
    return events + parser.finish("", "stop")


def test_call_body_names_and_arguments():
    assert parse_tool_call('functions.get_weather:0<|tool_call_argument_begin|>{"city":"Paris"}') == {
        "name": "get_weather", "arguments": {"city": "Paris"}}
    assert parse_tool_call("search<|tool_call_argument_begin|>") == {"name": "search", "arguments": {}}
    assert parse_tool_call("ns.tools.lookup:12<|tool_call_argument_begin|>{}")["name"] == "lookup"
    for bad in ("get_weather", "functions.:0<|tool_call_argument_begin|>{}",
                "f<|tool_call_argument_begin|>[1]", "f<|tool_call_argument_begin|>{nope}"):
        with pytest.raises(ValueError):
            parse_tool_call(bad)
    with pytest.raises(UnclosedJSONString):
        parse_tool_call('f<|tool_call_argument_begin|>{"a": "cut here')
    assert partial_function_names("functions.search:1") == []
    assert partial_function_names("functions.search:1<|tool_call_argument_begin|>{") == ["search"]


@pytest.mark.parametrize("step", [1, 3, 7, 1000])
def test_streamed_parallel_calls_quoted_closer_and_hidden_section(step):
    parser = make_output_parser(REQUEST)
    events = run(parser, TWO_CALLS, step)
    content = "".join(e.get("content", "") for e in events)
    calls = [c for e in events for c in e.get("tool_calls", [])]
    assert content == "Checking." and "section" not in content
    assert [c["function"]["name"] for c in calls] == ["get_weather", "search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris <|tool_call_end|> x"}
    assert parser.tool_count == 2 and parser.tool_call_parse_fallbacks == 0
    assert parser.section_markers_stripped == 2


def test_plain_text_and_held_marker_prefix_release_at_finish():
    parser = make_output_parser({"messages": [{"role": "user", "content": "x"}]})
    events = run(parser, "price <|tool", 4)
    assert "".join(e.get("content", "") for e in events) == "price <|tool"


def test_forced_tool_choice_and_reasoning_fail_closed():
    assert tool_constraint(REQUEST) is None
    with pytest.raises(ValueError):
        tool_constraint({**REQUEST, "tool_choice": "required"})
    with pytest.raises(ValueError):
        make_output_parser({**REQUEST, "enable_thinking": True})
    assert isinstance(make_output_parser(REQUEST), KimiOutputParser)


def test_adapter_declares_tools_and_renders_them_through_the_template():
    from mlx2.adapters.kimi_linear import DESCRIPTOR, KimiLinearAdapter
    from mlx2.contracts import Capability
    assert Capability.TOOLS in DESCRIPTOR.capabilities
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
            seen.update(messages=messages, tools=tools)
            return [1, 2] if tokenize else "rendered"
    adapter = KimiLinearAdapter.__new__(KimiLinearAdapter)
    adapter.tokenizer = Tokenizer()
    history = {**REQUEST, "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "functions.search:0", "type": "function",
            "function": {"name": "search", "arguments": '{"q": "mlx"}'}}]},
        {"role": "tool", "tool_call_id": "functions.search:0", "content": "ok"}]}
    assert adapter.prompt_tokens(history) == [1, 2]
    assert seen["tools"] == TOOLS
    assert seen["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"q": "mlx"}
    adapter.render_prompt({**REQUEST, "tool_choice": "none"})
    assert seen["tools"] is None


def _tools(*names):
    return [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
            for n in names]


def _calls(events):
    return [c for e in events for c in e.get("tool_calls", [])]


def test_dotted_declared_name_is_read_whole_from_the_call_id():
    # The template renders functions.<name>:<idx>; a declared "web.search"
    # must not be cut to its last segment and refused as undeclared.
    request = {"messages": [{"role": "user", "content": "x"}], "tools": _tools("web.search", "search")}
    text = ('<|tool_calls_section_begin|><|tool_call_begin|>functions.web.search:0'
            '<|tool_call_argument_begin|>{"q":"mlx"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>{}<|tool_call_end|>'
            '<|tool_calls_section_end|>')
    calls = _calls(run(make_output_parser(request), text, 5))
    assert [c["function"]["name"] for c in calls] == ["web.search", "search"]
    tools = _tools("web.search")
    assert parse_tool_call("functions.web.search:3<|tool_call_argument_begin|>{}", tools)["name"] == "web.search"
    # Another prefix still resolves to the declared suffix, as before.
    assert parse_tool_call("api.web.search:3<|tool_call_argument_begin|>{}", tools)["name"] == "web.search"
    assert partial_function_names("functions.web.search:0<|tool_call_argument_begin|>{", tools) == ["web.search"]


@pytest.mark.parametrize("step", [1, 4, 1000])
def test_section_marker_quoted_in_a_json_argument_is_kept(step):
    request = {"messages": [{"role": "user", "content": "x"}], "tools": _tools("echo")}
    payload = "say <|tool_calls_section_end|> and <|tool_calls_section_begin|> literally"
    text = ('Ok.<|tool_calls_section_begin|><|tool_call_begin|>functions.echo:0'
            '<|tool_call_argument_begin|>' + json.dumps({"s": payload})
            + '<|tool_call_end|><|tool_calls_section_end|>')
    parser = make_output_parser(request)
    events = run(parser, text, step)
    assert "".join(e.get("content", "") for e in events) == "Ok."
    assert json.loads(_calls(events)[0]["function"]["arguments"])["s"] == payload
    assert parser.section_markers_stripped == 2


def test_history_call_ids_render_as_kimi_function_ids():
    # Emitted ids are call_<hex>; the template writes the id after
    # <|tool_call_begin|>, so history must carry functions.<name>:<idx>.
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
            seen.update(messages=messages)
            return "rendered"
    adapter = KimiLinearAdapter.__new__(KimiLinearAdapter)
    adapter.tokenizer = Tokenizer()
    call = lambda cid, name: {"id": cid, "type": "function", "function": {"name": name, "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_aa", "search"), call("call_bb", "web.search")]},
        {"role": "tool", "tool_call_id": "call_bb", "content": "two"},
        {"role": "tool", "tool_call_id": "call_aa", "content": "one"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_cc", "search")]},
        {"role": "tool", "tool_call_id": "call_cc", "content": "three"},
        {"role": "tool", "tool_call_id": "unknown", "content": "kept"}]
    request = {"messages": messages, "tools": _tools("search", "web.search")}
    adapter.render_prompt(request)
    rendered = seen["messages"]
    assert [c["id"] for c in rendered[1]["tool_calls"]] == ["functions.search:0", "functions.web.search:1"]
    assert [m["tool_call_id"] for m in rendered[2:4]] == ["functions.web.search:1", "functions.search:0"]
    assert rendered[4]["tool_calls"][0]["id"] == "functions.search:2"
    assert rendered[5]["tool_call_id"] == "functions.search:2"
    assert rendered[6]["tool_call_id"] == "unknown"
    assert messages[1]["tool_calls"][0]["id"] == "call_aa"  # the request is not mutated


def _k2_layout_tokenizer(tmp_path, *, with_tool_markers):
    # A synthetic vocabulary of the K2 base size (163584 ranks) with the
    # staged artifacts' added_tokens_decoder: the REAP config omits the five
    # tool-call markers that the 48B q8 config names.
    import base64

    from tokenizers import AddedToken

    from mlx2.runtime.kimi_tokenizer import TikTokenTokenizer
    ranks = [bytes([i]) for i in range(256)]
    ranks += [b"\xff\xfe" + i.to_bytes(3, "big") for i in range(163584 - 256)]
    vocab = tmp_path / "tiktoken.model"
    vocab.write_text("".join(f"{base64.b64encode(t).decode()} {r}\n" for r, t in enumerate(ranks)))
    names = {163584: "[BOS]", 163585: "[EOS]", 163586: "<|im_end|>", 163587: "<|im_user|>",
             163588: "<|im_assistant|>", 163590: "<|start_header_id|>", 163591: "<|end_header_id|>",
             163593: "[EOT]", 163594: "<|im_system|>", 163601: "<|im_middle|>",
             163838: "[UNK]", 163839: "[PAD]"}
    if with_tool_markers:
        names.update({163595: "<|tool_calls_section_begin|>", 163596: "<|tool_calls_section_end|>",
                      163597: "<|tool_call_begin|>", 163598: "<|tool_call_argument_begin|>",
                      163599: "<|tool_call_end|>"})
    decoder = {i: AddedToken(n, special=True) for i, n in names.items()}
    return TikTokenTokenizer(str(vocab), bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]",
                             pad_token="[PAD]", added_tokens_decoder=decoder)


@pytest.mark.parametrize("with_tool_markers", [False, True])
def test_kimi_tokenizer_names_k2_tool_markers_and_stays_inside_the_model_vocab(tmp_path, with_tool_markers):
    from mlx2.runtime.tool_parsers.kimi import ARGUMENT_BEGIN, CALL_BEGIN, CALL_END, SECTION_BEGIN, SECTION_END
    tok = _k2_layout_tokenizer(tmp_path, with_tool_markers=with_tool_markers)
    for token_id, marker in zip(range(163595, 163600),
                                (SECTION_BEGIN, SECTION_END, CALL_BEGIN, ARGUMENT_BEGIN, CALL_END)):
        assert tok.decode([token_id]) == marker
        assert tok.encode(marker) == [token_id]
    # 256 reserved ids fill the model's 163840-row embedding exactly.
    assert tok.n_words == 163840
    assert max(tok.special_tokens.values()) == 163839
    assert all(i < 163840 for i in tok.encode("<|reserved_token_163840|><|reserved_token_163841|>"))


def test_marker_overlay_lists_only_the_ids_a_config_leaves_unnamed(tmp_path):
    from mlx2.adapters.kimi_linear import tokenizer_marker_overlay
    named = {"163584": {"content": "[BOS]"}, "163597": {"content": "<|tool_call_begin|>"}}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": named}))
    overlay = tokenizer_marker_overlay(tmp_path)
    assert sorted(overlay) == ["163595", "163596", "163598", "163599"]
    assert overlay["163599"] == "<|tool_call_end|>"


ECHO_CALL = ('Ok.<|tool_calls_section_begin|><|tool_call_begin|>functions.echo:0'
             '<|tool_call_argument_begin|>{"s": "hi"}<|tool_call_end|><|tool_calls_section_end|>')


@pytest.mark.parametrize("step", [1, 4, 10_000])
@pytest.mark.parametrize("stop", ["section", "tool_calls", "<|", "call_begin", "argument", "|>"])
def test_stop_strings_inside_wire_markers_do_not_cut_a_call(stop, step):
    # The K2 markers are structure: a client stop string that only occurs
    # inside a marker spelling must not stop the answer or cut the call.
    request = {"messages": [{"role": "user", "content": "x"}], "tools": _tools("echo"), "stop": [stop]}
    parser = make_output_parser(request)
    events = run(parser, ECHO_CALL, step)
    assert "".join(e.get("content", "") for e in events) == "Ok."
    assert [json.loads(c["function"]["arguments"]) for c in _calls(events)] == [{"s": "hi"}]
    assert not parser.stopped and parser.stop_sequence is None


@pytest.mark.parametrize("step", [1, 4, 10_000])
def test_stop_string_in_answer_text_still_stops(step):
    request = {"messages": [{"role": "user", "content": "x"}], "tools": _tools("echo"), "stop": ["STOP"]}
    parser = make_output_parser(request)
    events = run(parser, "Ok. STOP now" + ECHO_CALL, step)
    assert "".join(e.get("content", "") for e in events) == "Ok. "
    assert not _calls(events) and parser.stopped and parser.stop_sequence == "STOP"
    # Text inside a call body is still matched, as for every family.
    parser = make_output_parser({**request, "stop": ["hi"]})
    events = run(parser, ECHO_CALL, step)
    assert not _calls(events) and parser.stop_sequence == "hi"
    assert parser.truncated_tool_call["cause"] == "stop_sequence"


def test_served_ids_continue_kimi_numbering_and_history_ids_are_repaired():
    """Hermes regression: OpenAI-style call_ ids rendered into Kimi's template
    taught the model to head a new call with an id (502 undeclared tool)."""
    from mlx2.adapters.kimi_linear_output import normalize_call_ids
    from mlx2.output import OutputParser
    history = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_4038758cb3f84810b9d55f4d", "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "call_4038758cb3f84810b9d55f4d", "content": "18C"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "functions.search:1", "type": "function", "function": {"name": "search", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "functions.search:1", "content": "ok"}]
    fixed = normalize_call_ids(history)
    assert fixed[1]["tool_calls"][0]["id"] == "functions.get_weather:0"
    assert fixed[2]["tool_call_id"] == "functions.get_weather:0"
    assert fixed[3]["tool_calls"][0]["id"] == "functions.search:1" and fixed[4]["tool_call_id"] == "functions.search:1"
    assert history[1]["tool_calls"][0]["id"].startswith("call_")  # caller's request untouched
    parser = make_output_parser({"messages": history, "tools": TOOLS})
    calls = [c for e in run(parser, TWO_CALLS, 5) for c in e.get("tool_calls", [])]
    assert [c["id"] for c in calls] == ["functions.get_weather:2", "functions.search:3"]
    # The shared parser keeps OpenAI-style ids for every other wire.
    assert OutputParser()._tool_call_id({"name": "x"}, 0).startswith("call_")


def test_template_never_sees_openai_style_ids():
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
            seen["messages"] = messages
            return "rendered"
    adapter = KimiLinearAdapter.__new__(KimiLinearAdapter)
    adapter.tokenizer = Tokenizer()
    adapter.render_prompt({**REQUEST, "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc", "type": "function",
            "function": {"name": "search", "arguments": '{"q": "mlx"}'}}]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "ok"}]})
    assert "call_abc" not in json.dumps(seen["messages"])
    assert seen["messages"][2]["tool_call_id"] == "functions.search:0"
