"""Incremental JSON object name reader shared by tool grammars.

Original mlx2 code; extraction of the tested Xing reader, with a configurable
field. Nested argument members never identify the enclosing function.
"""

import json
import re

_JSON_STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)

_JSON_SPACE = " \t\n\r"
_JSON_LITERAL = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null"
)
_JSON_RUN = re.compile(r"[-+.0-9A-Za-z]+")


def partial_json_name(text, *, field="name", context="JSON tool call"):
    """The top-level ``name`` member of a JSON object cut short.

    Walks the object as ``json.loads`` would read it, member by member, and
    decodes each string with ``json.loads`` (escapes and ``\\uXXXX`` included).
    Returns the decoded ``name`` once its closing quote has been read (the
    last top-level ``name`` wins, as it does for ``json.loads``), ``None``
    while the member has not been reached or its string is still open, and
    raises ``ValueError`` when the member is not a string, when a repeated
    ``name`` member is cut inside its string (the earlier one may not be the
    name the call ends with), or when the text so far is not JSON the parser
    would accept.
    """
    stack, expect, key, name = [], "value", None, None
    position, length = 0, len(text)
    while True:
        while position < length and text[position] in _JSON_SPACE:
            position += 1
        if position >= length:
            return name
        char = text[position]
        if expect in ("key", "key_or_close"):
            if char == "}" and expect == "key_or_close":
                stack.pop()
                expect, position = ("comma_or_close" if stack else "end"), position + 1
                continue
            if char != '"':
                raise ValueError(f"Malformed {context}")
            match = _JSON_STRING.match(text, position)
            if match is None:
                return name  # cut inside a key
            key = json.loads(match.group(0)) if len(stack) == 1 else None
            expect, position = "colon", match.end()
            continue
        if expect == "colon":
            if char != ":":
                raise ValueError(f"Malformed {context}")
            expect, position = "value", position + 1
            continue
        if expect in ("value", "value_or_close"):
            if char == "]" and expect == "value_or_close":
                stack.pop()
                expect, position = ("comma_or_close" if stack else "end"), position + 1
                continue
            top_name = len(stack) == 1 and stack[0] == "{" and key == field
            if char in "{[":
                if top_name:
                    raise ValueError(f"{context} needs a string {field}")
                stack.append(char)
                expect = "key_or_close" if char == "{" else "value_or_close"
                position += 1
                continue
            if char == '"':
                match = _JSON_STRING.match(text, position)
                if match is None:
                    # Cut inside the string.  Inside a ``name`` that repeats
                    # an earlier one, the name the call ends with is unknown
                    # and the earlier one is not "no name yet".
                    if top_name and name is not None:
                        raise ValueError(f"{context} repeats its {field}")
                    return None if top_name else name
                value = json.loads(match.group(0))
                if top_name:
                    name = value
                position = match.end()
            else:
                run = _JSON_RUN.match(text, position)
                if run is None:
                    raise ValueError(f"Malformed {context}")
                if top_name:
                    raise ValueError(f"{context} needs a string {field}")
                if run.end() == length:
                    return name  # cut inside a literal
                if _JSON_LITERAL.fullmatch(run.group(0)) is None:
                    raise ValueError(f"Malformed {context}")
                position = run.end()
            expect = "comma_or_close" if stack else "end"
            continue
        if expect == "comma_or_close":
            if char == ",":
                expect = "key" if stack[-1] == "{" else "value"
            elif char == "}" and stack[-1] == "{" or char == "]" and stack[-1] == "[":
                stack.pop()
                expect = "comma_or_close" if stack else "end"
            else:
                raise ValueError(f"Malformed {context}")
            position += 1
            continue
        raise ValueError(f"Malformed {context}")  # text after the object
