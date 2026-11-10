# SPDX-License-Identifier: Apache-2.0
"""Strict Hermes-style JSON tool-call body: ``{"name": ..., "arguments": {...}}``.

Original mlx2 code.  The grammar is the one a chat template writes between
``<tool_call>`` and ``</tool_call>`` when it renders a call as
``{"name": <tojson>, "arguments": <tojson>}`` (Ripple 65B Preview's source
``chat_template.jinja``).  Only that one form is accepted: exactly one JSON
object with a string ``name`` and an object ``arguments`` and no other
member.  The Xing parser (``adapters/xing_output.parse_tool_block``) also
reads a JSON body but additionally admits tag-parameter and bare-name forms
that this template never writes, so a wire that only renders JSON uses this
stricter reader and fails anything else closed.

``tojson`` leaves ``<`` raw, so a JSON string may quote ``</tool_call>``.
When the body the output parser cut ends inside a JSON string the error
carries ``tool_call_close_quoted`` and the call runs on to a later closer.
"""

import json

from ._partial_json import partial_json_name

_MEMBERS = frozenset({"name", "arguments"})


class UnclosedJSONString(ValueError):
    """The body ends inside a JSON string: the closer it was cut at is quoted."""

    tool_call_close_quoted = True


def _ends_inside_string(text):
    inside = escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif inside and char == "\\":
            escaped = True
        elif char == '"':
            inside = not inside
    return inside


def _reject_constant(value):
    raise ValueError(f"JSON tool call uses the non-JSON constant {value}")


class DuplicateJSONKey(ValueError):
    """An object repeats a member; ``json.loads`` would keep the last silently."""


def _unique_members(pairs):
    # A repeated "name" would call the last one while partial_function_names
    # had streamed the first; a repeated argument would drop a value unseen
    # (sweep 2026-11-06 RIPPLE-5). The template's tojson never repeats a key.
    value = {}
    for key, member in pairs:
        if key in value:
            raise DuplicateJSONKey(f"JSON tool call repeats the member {key!r}")
        value[key] = member
    return value


def parse_tool_call(text, tools=None):
    """Parse one tool-call body; raise ``ValueError`` on anything else."""
    body = text.strip()
    if not body:
        raise ValueError("Empty JSON tool call")
    try:
        value = json.loads(body, parse_constant=_reject_constant, object_pairs_hook=_unique_members)
    except json.JSONDecodeError:
        if _ends_inside_string(body):
            raise UnclosedJSONString("JSON tool call ends inside a string") from None
        raise ValueError("Malformed JSON tool call") from None
    except RecursionError:
        raise ValueError("JSON tool call nests too deeply") from None
    if not isinstance(value, dict):
        raise ValueError("JSON tool call is not an object")
    if set(value) != _MEMBERS:
        raise ValueError('JSON tool call must have exactly the members "name" and "arguments"')
    name, arguments = value["name"], value["arguments"]
    if not isinstance(name, str) or not name:
        raise ValueError("JSON tool call needs a nonempty string name")
    if not isinstance(arguments, dict):
        # The template renders arguments with ``tojson`` of an object and its
        # system prompt says the field "must be a JSON object, not a string".
        raise ValueError("JSON tool-call arguments must be a JSON object")
    definitions = {
        tool["function"]["name"]: tool["function"]
        for tool in tools or ()
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    }
    if name not in definitions:
        raise ValueError("Model called an undeclared tool")
    schema = definitions[name].get("parameters") or {}
    if isinstance(schema, dict):
        if any(key not in arguments for key in schema.get("required") or ()):
            raise ValueError("JSON tool call is missing a required parameter")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False and any(key not in properties for key in arguments):
            raise ValueError("JSON tool call contains an undeclared parameter")
    return {"name": name, "arguments": arguments}


def partial_function_names(partial, tools=None):
    """The function a cut-short body had committed to (see ``stop_cut_record``).

    The top-level ``name`` once its string closed; nothing before that.
    Raises ``ValueError`` for a body that is not the start of a JSON object.
    """
    body = partial.strip()
    if not body:
        return []
    if not body.startswith("{"):
        raise ValueError("JSON tool call does not start with an object")
    name = partial_json_name(body, context="JSON tool call")
    return [name] if name is not None else []


parse_tool_call.partial_function_names = partial_function_names
