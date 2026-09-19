# SPDX-License-Identifier: MIT
# Mined from local mlx-lm-unified; see provenance/laguna-xs21.json and .NOTICE.
"""Parser for Poolside Laguna XML-like tool calls."""

import ast
import json
import re


def _is_string_type(tool_name, argument_name, tools):
    for tool in tools or ():
        function = tool.get("function", {})
        if function.get("name") != tool_name:
            continue
        schema = (function.get("parameters") or {}).get("properties", {}).get(argument_name, {})
        kind = schema.get("type")
        return kind == "string" or (isinstance(kind, list) and "string" in kind)
    return False


def _deserialize(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value


def parse_tool_call(text, tools=None):
    """Parse one or more complete ``<tool_call>`` blocks."""
    bodies = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not bodies:
        bodies = [text]
    calls = []
    for body in bodies:
        body = body.strip()
        name_match = re.search(r"^(.*?)<arg_key>", body, re.DOTALL)
        name = (name_match.group(1) if name_match else body.split("\n", 1)[0]).strip()
        arguments = {}
        for match in re.finditer(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            body,
            re.DOTALL,
        ):
            key, value = match.group(1).strip(), match.group(2).strip()
            arguments[key] = value if _is_string_type(name, key, tools) else _deserialize(value)
        calls.append({"name": name, "arguments": arguments})
    return calls[0] if len(calls) == 1 else calls
