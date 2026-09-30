"""GPT-OSS harmony channels: parser and prompt routing (CPU only, no MLX)."""

import pytest

from mlx2.adapters.gpt_oss import GPT_OSS, GPT_OSS_PUZZLE, GptOssAdapter, GptOssPuzzleAdapter
from mlx2.adapters.gpt_oss_output import HarmonyOutputParser
from mlx2.contracts import Capability

REASONED = (
    "<|channel|>analysis<|message|>17+25=42.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>42"
)


def run(parser, text, *, chunk=1, finish="stop"):
    events = []
    pieces = [text[i:i + chunk] for i in range(0, len(text), chunk)] or [""]
    for piece in pieces[:-1]:
        events += parser.push(piece)
    events += parser.finish(pieces[-1], finish)
    out = {}
    for event in events:
        for key, value in event.items():
            out[key] = out.get(key, "") + value
    return out


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_analysis_goes_to_reasoning_and_final_to_content(chunk):
    out = run(HarmonyOutputParser(), REASONED, chunk=chunk)
    assert out == {"reasoning_content": "17+25=42.", "content": "42"}


@pytest.mark.parametrize("chunk", [1, 1000])
def test_hidden_reasoning_is_dropped(chunk):
    out = run(HarmonyOutputParser(show_reasoning=False), REASONED, chunk=chunk)
    assert out == {"content": "42"}


def test_channel_reports_reasoning_until_the_final_message():
    parser = HarmonyOutputParser()
    assert parser.channel == "reasoning_content"
    parser.push("<|channel|>analysis<|message|>think")
    assert parser.channel == "reasoning_content"
    parser.push("<|end|><|start|>assistant<|channel|>final<|message|>4")
    assert parser.channel == "content"


def test_stop_strings_apply_to_the_final_channel_only():
    text = ("<|channel|>analysis<|message|>maybe STOP here<|end|>"
            "<|start|>assistant<|channel|>final<|message|>answer STOP tail")
    parser = HarmonyOutputParser(stops=["STOP"])
    out = run(parser, text, chunk=2)
    assert out == {"reasoning_content": "maybe STOP here", "content": "answer "}
    assert parser.stopped and parser.stop_sequence == "STOP"


@pytest.mark.parametrize("chunk", [1, 1000])
def test_direct_final_prompt_starts_in_the_answer(chunk):
    out = run(HarmonyOutputParser(start_in_final=True), "SUM=1237 WORD=garnet", chunk=chunk)
    assert out == {"content": "SUM=1237 WORD=garnet"}


def test_direct_final_answer_ends_at_end_of_message():
    # Seen from the Puzzle model when the final channel is forced.
    text = "42.<|end|><|start|>assistant<|channel|>analysis<|message|>The sum is 42."
    out = run(HarmonyOutputParser(start_in_final=True, show_reasoning=False), text)
    assert out == {"content": "42."}


def test_output_without_harmony_framing_is_kept_as_the_answer():
    out = run(HarmonyOutputParser(), "Plain answer without a header.")
    assert out == {"content": "Plain answer without a header."}


@pytest.mark.parametrize("chunk", [1, 1000])
def test_a_message_marker_without_a_channel_header_is_answer_text(chunk):
    # Codex review: the text before <|message|> is not a header here.
    out = run(HarmonyOutputParser(), "Plain <|message|> answer", chunk=chunk)
    assert out == {"content": "Plain <|message|> answer"}


def test_headers_with_recipient_and_constraint_annotations_parse():
    text = ("<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{}<|end|>"
            "<|start|>assistant<|channel|>final<|message|>done")
    parser = HarmonyOutputParser()
    assert run(parser, text, chunk=3) == {"reasoning_content": "{}", "content": "done"}
    assert parser.headers == ["commentary", "final"]


def test_length_finish_inside_analysis_has_no_content():
    out = run(HarmonyOutputParser(), "<|channel|>analysis<|message|>still thinking", finish="length")
    assert out == {"reasoning_content": "still thinking"}


def test_raw_completions_pass_through():
    out = run(HarmonyOutputParser(chat=False, stops=["\n"]), REASONED + "\nmore")
    assert out == {"content": REASONED}


class _Tokenizer:
    """Records template kwargs; tokens are characters."""

    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        self.kwargs = kwargs
        effort = kwargs.get("reasoning_effort", "medium")
        return list(f"[Reasoning: {effort}]<|start|>assistant")

    def encode(self, text, add_special_tokens=False):
        return list(text)


def adapter(cls):
    value = object.__new__(cls)
    value.tokenizer = _Tokenizer()
    return value


CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def prompt(value, request):
    return "".join(value.prompt_tokens(request))


def test_both_descriptors_declare_reasoning():
    assert Capability.REASONING in GPT_OSS.capabilities
    assert Capability.REASONING in GPT_OSS_PUZZLE.capabilities
    assert Capability.TOOLS not in GPT_OSS_PUZZLE.capabilities


def test_stock_gpt_oss_answers_directly_by_default():
    value = adapter(GptOssAdapter)
    assert not value.thinking_enabled(CHAT)
    assert prompt(value, CHAT).endswith("<|start|>assistant<|channel|>final<|message|>")
    assert value.output_parser(CHAT).channel == "content"


def test_stock_gpt_oss_reasons_on_request():
    value = adapter(GptOssAdapter)
    for request in ({**CHAT, "enable_thinking": True}, {**CHAT, "reasoning_effort": "high"}):
        assert value.thinking_enabled(request)
        assert prompt(value, request).endswith("<|start|>assistant")
    assert "Reasoning: high" in prompt(value, {**CHAT, "reasoning_effort": "high"})


def test_puzzle_reasons_by_default_and_never_skips_analysis():
    value = adapter(GptOssPuzzleAdapter)
    assert value.thinking_enabled(CHAT)
    assert prompt(value, CHAT) == "[Reasoning: medium]<|start|>assistant"
    off = {**CHAT, "enable_thinking": False}
    assert not value.thinking_enabled(off)
    # Thinking off still opens analysis, briefly, and the parser hides it.
    assert prompt(value, off) == "[Reasoning: low]<|start|>assistant"
    parser = value.output_parser(off)
    assert run(parser, REASONED) == {"content": "42"}


@pytest.mark.parametrize("effort,level,shown", [
    ("none", "low", False), ("minimal", "low", True), ("medium", "medium", True),
    ("xhigh", "high", True), ("ultra", "high", True),
])
def test_server_reasoning_efforts_map_to_template_levels(effort, level, shown):
    request = {**CHAT, "reasoning_effort": effort}
    puzzle = adapter(GptOssPuzzleAdapter)
    assert puzzle.thinking_enabled(request) is shown
    assert f"Reasoning: {level}" in prompt(puzzle, request)
    stock = adapter(GptOssAdapter)
    assert prompt(stock, request).endswith("<|channel|>final<|message|>") is (not shown)


def test_invalid_reasoning_effort_fails_closed():
    with pytest.raises(ValueError, match="reasoning_effort"):
        adapter(GptOssPuzzleAdapter).prompt_tokens({**CHAT, "reasoning_effort": "extreme"})


def test_raw_prompt_is_not_templated():
    value = adapter(GptOssPuzzleAdapter)
    assert value.prompt_tokens({"prompt": "abc"}) == ["a", "b", "c"]
    assert not value.thinking_enabled({"prompt": "abc"})


class _HarmonyTokenizer(_Tokenizer):
    """Harmony specials and ``assistant``/``final`` as single ids."""

    VOCAB = {"<|end|>": 200007, "<|start|>": 200006, "assistant": 173781,
             "<|channel|>": 200005, "final": 17196, "<|message|>": 200008}

    def encode(self, text, add_special_tokens=False):
        import re
        pieces = re.findall(r"<\|\w+\|>|assistant|final|.", text, re.DOTALL)
        return [self.VOCAB.get(piece, ord(piece[0])) for piece in pieces]

    def decode(self, ids):
        names = {value: key for key, value in self.VOCAB.items()}
        return "".join(names.get(token, chr(token)) for token in ids)


def test_harmony_final_switch_is_the_declared_close_marker():
    for cls in (GptOssAdapter, GptOssPuzzleAdapter):
        value = object.__new__(cls)
        value.tokenizer = _HarmonyTokenizer()
        assert value.thinking_close_token_ids() == (200007, 200006, 173781, 200005, 17196, 200008)
    # A tokenizer that splits the switch differently declares nothing.
    value = adapter(GptOssPuzzleAdapter)
    assert value.thinking_close_token_ids() is None


@pytest.mark.parametrize("request_fields,budget", [
    ({"enable_thinking": False, "max_tokens": 64}, 29),
    # The six-token switch and an answer still fit (codex review).
    ({"enable_thinking": False, "max_tokens": 20}, 7),
    ({"enable_thinking": False, "max_tokens": 8}, 1),
    ({"enable_thinking": False, "max_tokens": 4096}, 512),
    ({"enable_thinking": False}, 512),
    ({"reasoning_effort": "none", "max_tokens": 100}, 47),
    ({}, 0),                       # visible reasoning: the operator's budget applies
    ({"enable_thinking": True}, 0),
])
def test_puzzle_bounds_hidden_reasoning_by_max_tokens(request_fields, budget):
    assert adapter(GptOssPuzzleAdapter).hidden_thinking_budget({**CHAT, **request_fields}) == budget


def test_stock_gpt_oss_has_no_hidden_reasoning_to_bound():
    value = adapter(GptOssAdapter)
    assert value.hidden_thinking_budget({**CHAT, "enable_thinking": False, "max_tokens": 64}) == 0
    assert value.hidden_thinking_budget({"prompt": "x"}) == 0


def test_gpt_oss_declares_the_juice_soft_landing_nudge():
    from mlx2.adapters.gpt_oss import _NUDGE_TEXT

    value = object.__new__(GptOssPuzzleAdapter)
    value.tokenizer = _HarmonyTokenizer()
    ids = value.thinking_nudge_token_ids()
    assert ids and value.tokenizer.decode(list(ids)) == _NUDGE_TEXT
    assert "final answer" in _NUDGE_TEXT
