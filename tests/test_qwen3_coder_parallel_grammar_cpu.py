"""The parallel Qwen tool-call grammar admits the template's final newline (Nemotron 3.5 Super, 2026-10-18)."""
import re

from mlx2.runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

TOOLS = [{"type": "function", "function": {
    "name": "write_file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}}]
CALL = "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"


def _full(pattern, text):
    return re.fullmatch(pattern, text, re.S) is not None


def test_parallel_grammar_ends_after_the_template_newline():
    pattern = constrained_tool_grammar(TOOLS, {"type": "function", "function": {"name": "write_file"}})
    # The model writes </tool_call>\n before its end-of-turn token; that must end the block,
    # not force another call (which made greedy decoding repeat the call to max_tokens).
    assert _full(pattern, CALL + "\n")
    assert _full(pattern, CALL)
    assert _full(pattern, CALL + "\n" + CALL + "\n")
    assert not _full(pattern, CALL + "\n\n")


def test_single_call_grammar_is_unchanged():
    pattern = constrained_tool_grammar(TOOLS, "required", parallel_tool_calls=False)
    assert _full(pattern, CALL)
    assert not _full(pattern, CALL + "\n")
