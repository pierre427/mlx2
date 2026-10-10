# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; see provenance/laguna-xs21.json and .NOTICE.
"""Parser for Poolside Laguna XML-like tool calls."""

import ast
import json
import re

from ._schema import json_native

_ARG_KEY_OPEN = "<arg_key>"
_ARG_KEY_CLOSE = "</arg_key>"
_ARG_VALUE_OPEN = "<arg_value>"
_ARG_VALUE_CLOSE = "</arg_value>"
# The template puts a newline between the pieces of a call; the reference
# parser also accepts a literal backslash-n there.
_SEPARATOR = re.compile(r"(?:\\n|\s)*")
_JSON_OPEN = re.compile(r'\s*[\[{"]')
_JSON_CLOSER = re.compile(r"</arg_value>|</tool_call>")
_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)


class UnclosedJSONString(ValueError):
    """The text ends inside a string of an argument's JSON value.

    The template writes non-string arguments with ``tojson``, which leaves
    ``<`` raw, so a JSON string may quote ``</tool_call>``.  The output parser
    cut the call at that closer, so the call runs on to a later one.
    """

    tool_call_close_quoted = True


def _declared_types(tool_name, argument_name, tools):
    """The JSON types the argument's schema declares, empty when it declares none."""
    for tool in tools or ():
        function = tool.get("function", {})
        if function.get("name") != tool_name:
            continue
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        schema = properties.get(argument_name) if isinstance(properties, dict) else None
        # A boolean subschema (``true`` admits anything, ``false`` nothing) is
        # valid JSON Schema but declares no type, like a missing one.
        if not isinstance(schema, dict):
            return ()
        kind = schema.get("type")
        if isinstance(kind, str):
            return (kind,)
        return tuple(kind) if isinstance(kind, list) else ()
    return ()


def _deserialize(value):
    """JSON, then a Python literal, else the raw text.

    A decoded value the arguments cannot carry as written (a tuple, a
    non-string key, a non-finite float, a set) stays the raw text too.
    """
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        try:
            decoded = ast.literal_eval(value)
        except Exception:  # noqa: BLE001 - any failure means "not a literal"
            # An unhashable dict key or set member raises TypeError.
            return value
    return decoded if json_native(decoded) else value


def _json_value_end(text, start):
    """Where the closer after the JSON value that starts at ``start`` is.

    Outside a string ``<`` is not JSON, so the value ends at the first
    ``</arg_value>`` or ``</tool_call>`` outside every string.  Returns -1
    when none follows, and raises ``UnclosedJSONString`` when the text ends
    inside a string.
    """
    position = start
    while True:
        closer = _JSON_CLOSER.search(text, position)
        close = closer.start() if closer is not None else -1
        quote = text.find('"', position, close if close >= 0 else len(text))
        if quote < 0:
            return close
        string = _JSON_STRING.match(text, quote)
        if string is None:
            raise UnclosedJSONString("Laguna argument ends inside a JSON string")
        position = string.end()


def _parse_arguments(name, body, position, tools):
    """The arguments of the ``<arg_key>`` pairs from ``position`` to the end.

    The template writes string arguments raw and every other argument with
    ``tojson``.  A raw value ends at the first ``</arg_value>``: a closer the
    string itself contains cannot be told from markup, so what follows it is
    left over and the call is malformed rather than silently truncated.  An
    argument whose declared type is not a string and whose text opens a JSON
    object, array or string ends at the first closer outside its strings,
    provided that text decodes as JSON.
    """
    arguments = {}
    while True:
        position = _SEPARATOR.match(body, position).end()
        if position == len(body):
            return arguments
        if not body.startswith(_ARG_KEY_OPEN, position):
            raise ValueError("Malformed Laguna tool-call arguments")
        key_start = position + len(_ARG_KEY_OPEN)
        key_end = body.find(_ARG_KEY_CLOSE, key_start)
        if key_end < 0:
            raise ValueError("Malformed Laguna tool-call arguments")
        key = body[key_start:key_end].strip()
        if "<" in key or ">" in key:
            # A repeated <arg_key> (vLLM #56093) would otherwise name the
            # argument "<arg_key>content".
            raise ValueError("Malformed Laguna tool-call argument name")
        position = _SEPARATOR.match(body, key_end + len(_ARG_KEY_CLOSE)).end()
        if not body.startswith(_ARG_VALUE_OPEN, position):
            raise ValueError("Malformed Laguna tool-call arguments")
        start = position + len(_ARG_VALUE_OPEN)
        end = body.find(_ARG_VALUE_CLOSE, start)
        types = _declared_types(name, key, tools)
        string_type = "string" in types
        if types and not string_type and _JSON_OPEN.match(body, start):
            close = _json_value_end(body, start)
            if close >= 0 and not body.startswith(_ARG_VALUE_CLOSE, close):
                close = -1  # a </tool_call> outside the value's strings
            if close >= 0 and _JSON_CLOSER.search(body, start, close):
                # The value quotes a closer.  Only JSON puts one there.
                try:
                    json.loads(body[start:close])
                except ValueError:
                    raise ValueError("Malformed Laguna JSON argument") from None
            end = close
        if end < 0:
            # The call ended inside the value.  A raw value never runs past
            # its first closer, so a </tool_call> the string itself contains
            # fails closed instead of serving the call without it.
            raise ValueError("Unclosed Laguna argument value")
        # The template writes a string value raw: its leading indentation and
        # trailing newlines belong to it.  Only a decoded value sheds the
        # whitespace around it; text that decodes as nothing stays as written.
        value = body[start:end]
        if not string_type:
            text = value.strip()
            decoded = _deserialize(text)
            value = value if decoded is text else decoded
        arguments[key] = value
        position = end + len(_ARG_VALUE_CLOSE)


def _tool_call_bodies(text):
    """What ``re.findall(r"<tool_call>(.*?)</tool_call>")`` finds, in linear time.

    The regex rescans the rest of the text from every unclosed opener, so a
    run of degenerate output repeating ``<tool_call>`` took quadratic time
    (10.7 s at 20k repeats; vLLM #54678 is the same class).
    """
    open_tag, close_tag = "<tool_call>", "</tool_call>"
    bodies, position = [], 0
    while (start := text.find(open_tag, position)) >= 0:
        end = text.find(close_tag, start + len(open_tag))
        if end < 0:
            break
        bodies.append(text[start + len(open_tag) : end])
        position = end + len(close_tag)
    return bodies


def parse_tool_call(text, tools=None):
    """Parse one or more complete ``<tool_call>`` blocks.

    The output parser passes the text between ``<tool_call>`` and
    ``</tool_call>``; only text that itself opens with ``<tool_call>`` is
    split into blocks, so a JSON string that quotes a whole block stays part
    of its value.
    """
    bodies = None
    if text.lstrip().startswith("<tool_call>"):
        bodies = _tool_call_bodies(text)
    if not bodies:
        bodies = [text]
    calls = []
    for body in bodies:
        body = body.strip()
        name_match = re.search(r"^(.*?)<arg_key>", body, re.DOTALL)
        name = (name_match.group(1) if name_match else body.split("\n", 1)[0]).strip()
        arguments = {}
        if name_match:
            arguments = _parse_arguments(
                name, body, name_match.end() - len(_ARG_KEY_OPEN), tools
            )
        calls.append({"name": name, "arguments": arguments})
    return calls[0] if len(calls) == 1 else calls


def partial_function_names(partial, tools=None):
    """The function a cut-short tool-call body had committed to, if any.

    The reader behind ``OutputParser``'s stop-cut record (see
    :func:`mlx2.output.stop_cut_record`), by :func:`parse_tool_call`'s own
    two rules: the name is everything before the first ``<arg_key>``,
    stripped, whatever characters it holds; a block with no argument marker
    names its function on its first line.  Either delimiter commits the
    name (a cut after the first newline reads as the parser would read the
    block closed there); before both nothing is committed.  Raises
    ``ValueError`` when the marker came with no name before it, which the
    parser serves as a call to the empty name and rejects.
    """
    # A nested block, as the parser splits one.
    body = partial.lstrip().removeprefix("<tool_call>").lstrip()
    position = body.find(_ARG_KEY_OPEN)
    if position >= 0:
        name = body[:position].strip()
        if not name:
            raise ValueError("Laguna tool call names no function")
        return [name]
    line, newline, _rest = body.partition("\n")
    if not newline:
        return []
    return [line.strip()]


parse_tool_call.partial_function_names = partial_function_names
