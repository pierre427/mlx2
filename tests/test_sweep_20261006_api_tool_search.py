"""Claude Code tool search over Anthropic Messages (vLLM #57693, #59718).

With tool search on, Claude Code declares tools with ``defer_loading`` and
surfaces them later through ``tool_addition``/``tool_removal`` blocks in
mid-conversation ``role: "system"`` messages.  Under agent compat these
resolve into the tool list the chat template sees; without it they keep
failing closed.
"""

from collections import Counter

import pytest

from mlx2.agent_compat import AgentCompat
from mlx2.anthropic_compat import anthropic_request_to_chat

SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}
BASH = {"name": "Bash", "input_schema": {"type": "object"}}
WEATHER = {"name": "get_weather", "defer_loading": True, "input_schema": SCHEMA}


def _run(body, *, compat=True, counts=None, count_tokens=False):
    body = {"model": "m", "max_tokens": 64, **body}
    return anthropic_request_to_chat(
        body,
        agent_compat=AgentCompat(enabled=compat),
        counts=counts,
        count_tokens=count_tokens,
    )


def _system(*blocks):
    return {"role": "system", "content": list(blocks)}


def _ref(kind, name):
    return {"type": kind, "tool": {"type": "tool_reference", "name": name}}


def _names(result):
    return [tool["function"]["name"] for tool in result.get("tools", ())]


ASK = {"role": "user", "content": "weather in Paris?"}


def test_deferred_tool_is_withheld_until_added():
    counts = Counter()
    result = _run({"tools": [BASH, WEATHER], "messages": [ASK]}, counts=counts)
    assert _names(result) == ["Bash"]
    assert counts["agent_compat_tools_deferred"] == 1
    added = _run(
        {
            "tools": [BASH, WEATHER],
            "messages": [
                ASK,
                _system(
                    {"type": "text", "text": "The following tools just became available."},
                    _ref("tool_addition", "get_weather"),
                ),
            ],
        },
        counts=counts,
    )
    assert _names(added) == ["Bash", "get_weather"]
    assert "defer_loading" not in added["tools"][1]["function"]
    assert counts["agent_compat_tool_additions"] == 1
    # The text part is folded into the user turn; the block itself is not.
    assert added["messages"] == [
        {
            "role": "user",
            "content": "weather in Paris?\n\nThe following tools just became available.",
        }
    ]


def test_tool_only_system_message_renders_nothing_in_place():
    result = _run(
        {
            "tools": [WEATHER],
            "messages": [ASK, _system(_ref("tool_addition", "get_weather"))],
        }
    )
    assert _names(result) == ["get_weather"]
    assert result["messages"] == [{"role": "user", "content": "weather in Paris?"}]


def test_removal_withdraws_and_readdition_restores():
    removed = _run(
        {
            "tools": [BASH, {**WEATHER, "defer_loading": False}],
            "messages": [ASK, _system(_ref("tool_removal", "get_weather"))],
        }
    )
    assert _names(removed) == ["Bash"]
    readded = _run(
        {
            "tools": [BASH, {**WEATHER, "defer_loading": False}],
            "messages": [
                ASK,
                _system(_ref("tool_removal", "get_weather")),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"},
                _system(_ref("tool_addition", "get_weather")),
            ],
        }
    )
    assert _names(readded) == ["Bash", "get_weather"]


def test_tool_definition_by_value_is_declared_and_offered():
    result = _run(
        {
            "tools": [BASH],
            "messages": [
                ASK,
                _system(
                    {
                        "type": "tool_addition",
                        "tool": {
                            "type": "tool_definition",
                            "definition": {
                                "name": "db_query",
                                "description": "Run SQL.",
                                "input_schema": {"type": "object"},
                            },
                        },
                    }
                ),
            ],
        }
    )
    assert _names(result) == ["Bash", "db_query"]
    assert result["tools"][1]["function"]["description"] == "Run SQL."


def test_no_tool_changes_render_exactly_as_before():
    body = {"tools": [BASH], "messages": [ASK]}
    assert _run(body) == _run(body, compat=False)


@pytest.mark.parametrize(
    "block, message",
    [
        (_ref("tool_addition", "missing"), "tool_reference_unresolved"),
        (_ref("tool_removal", "missing"), "tool_reference_unresolved"),
        (
            {"type": "tool_addition", "tool": {"type": "mcp_tool_reference",
                                               "server_name": "c", "name": "x"}},
            "MCP",
        ),
        (
            {"type": "tool_addition", "tool": {"type": "tool_definition",
                                               "definition": {"type": "mcp_toolset",
                                                              "mcp_server_name": "c"}}},
            "MCP",
        ),
    ],
)
def test_unresolvable_tool_changes_are_client_errors(block, message):
    with pytest.raises(ValueError, match=message):
        _run({"tools": [BASH], "messages": [ASK, _system(block)]})


def test_tool_changes_only_in_compat_system_messages():
    with pytest.raises(ValueError, match="defer_loading"):
        _run({"tools": [WEATHER], "messages": [ASK]}, compat=False)
    with pytest.raises(ValueError, match="unsupported user content block"):
        _run({"tools": [WEATHER], "messages": [
            {"role": "user", "content": [_ref("tool_addition", "get_weather")]}
        ]})
    with pytest.raises(ValueError, match="system block"):
        _run({"tools": [WEATHER], "system": [_ref("tool_addition", "get_weather")],
              "messages": [ASK]})


def test_all_tools_withheld_keeps_auto_and_rejects_forced_choice():
    # Default/auto/none stay usable with no tool offered (vLLM #59718).
    for choice in ({"type": "auto", "disable_parallel_tool_use": True}, {"type": "none"}):
        result = _run({"tools": [WEATHER], "tool_choice": choice, "messages": [ASK]})
        assert "tools" not in result
        assert "tool_choice" not in result and "parallel_tool_calls" not in result
    for choice in ({"type": "any"}, {"type": "tool", "name": "get_weather"}):
        for count_tokens in (False, True):
            with pytest.raises(ValueError, match="currently offered"):
                body = {"tools": [WEATHER], "tool_choice": choice, "messages": [ASK]}
                if count_tokens:
                    body.pop("max_tokens", None)
                _run(body, count_tokens=count_tokens)


def test_named_choice_of_removed_tool_rejected_while_others_remain():
    with pytest.raises(ValueError, match="currently offered"):
        _run(
            {
                "tools": [BASH, {**WEATHER, "defer_loading": False}],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "messages": [ASK, _system(_ref("tool_removal", "get_weather"))],
            }
        )


def test_explicit_empty_tools_still_reaches_validation():
    assert _run({"tools": [], "messages": [ASK]})["tools"] == []
