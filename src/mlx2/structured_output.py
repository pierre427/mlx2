"""Fail-closed regex and JSON-schema token constraints for batched decoding.

Two engines enforce a compiled constraint.  The exact token-level automaton
(``structured_automaton``) is used whenever the pattern is in its compilable
subset; the ``regex`` partial-match scanner in this module is the fallback for
everything else and for ``MLX2_STRUCTURED_AUTOMATON=0``.  With thinking
enabled the processor defers the grammar past the adapter's thinking-close
marker (``defer_until``).
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict

import regex

from .runtime.tool_parsers._schema import (
    executable_schema,
    schema_value_matches,
    string_length_bounds,
)


# Insignificant whitespace is bounded for the same reason digit runs are: it is
# always admissible, so a model that cannot emit the token it wants can sit in
# it until the token cap (observed on GPU: North after ``"coastal":``).  32 is
# a newline plus 2-space indentation 15 levels deep; the bound only narrows
# formatting, never which JSON values are reachable.
_WS = r"[\x20\x09\x0a\x0d]{0,32}"
_STRING = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
# JSON places no bound on a number's length, and a greedy model that reaches a
# digit run can stay in it until the token cap (observed on GPU: a population
# field filled with zeros).  Bound every digit run: 19 integer digits covers
# int64, 18 fraction digits exceeds float64 precision, 3 exponent digits its
# range.
_INTEGER = r"-?(?:0|[1-9][0-9]{0,18})"
_NUMBER = _INTEGER + r"(?:\.[0-9]{1,18})?(?:[eE][+-]?[0-9]{1,3})?"
_FINITE_NUMBER = _INTEGER + r"(?:\.[0-9]{1,18})?(?:[eE][+-]?[0-9]{1,2})?"
_MATCH_TIMEOUT_SECONDS = 0.01
# Wall-clock budget for computing one token's admissible set.  The walk below
# runs on the generation thread, so an overrun stalls every lane; past this
# budget the request fails closed instead.
_ALLOWED_BUDGET_SECONDS = 2.0
_BUDGET_CHECK_INTERVAL = 256
# Sampling: stop examining tokens once the unexamined mass is below this
# fraction of the admissible mass already admitted.  Structured routes are
# qualified as numerically bounded, not exact; the tail of a 248K-token
# distribution is fat enough that float32 resolution would mean a full scan.
# Greedy decoding is exact regardless: the first admissible token in logit
# order is the argmax of the fully masked row.
_MASS_RESOLUTION = 1e-4
# Scanner pool cost model: dispatch/collect overhead plus a calibrated
# per-token rate; small tails are cheaper to finish in-process.
_POOL_FIXED_SECONDS = 0.03
_POOL_CALIBRATION_MIN = 8192
_POOL_MIN_TOKENS = 2048
# Tokens sampled from the unexamined tail to estimate its admissible share.
_TAIL_SAMPLE = 512
# A sample that found no admissible tail token cannot resolve below this
# (its upper confidence end); the walk then stops and records the bound.
_TAIL_UNRESOLVED = 1.0 / _TAIL_SAMPLE + 1e-12
# Share of the vocabulary after which the walk stops with a recorded bound.
_MAX_EXAMINED_FRACTION = 0.25
# Pure temperature sampling stops once the estimated omitted admissible mass
# is below this fraction of the admitted mass.
_BOUND_STOP = 1e-3
_COMPLETE_KEY = "__complete__"
_DEBUG = __import__("os").environ.get("MLX2_STRUCTURED_DEBUG") == "1"
_MAX_PREFIX_CACHE = 256
_TOKEN_PIECES_ATTRIBUTE = "_mlx2_structured_token_pieces_v2"
_TOKEN_INDEX_ATTRIBUTE = "_mlx2_structured_token_index_v2"
_FAILURE_HISTORY_TOKENS = 64
_FAILURE_TOP_LOGITS = 8
_FAILURE_TEXT_CHARS = 64
_FAILURE_BYTES = 32


def vocabulary_bound(tokenizer):
    """One past the highest token id this tokenizer can produce.

    ``tokenizer.vocab_size`` is the *base* vocabulary only: it excludes every
    token added after training.  Those added ids are exactly the wire markers
    the adapters' tool grammars are written against -- ``<tool_call>`` is id
    248058 on Flash-Next against a base size of 248044, ``<|START_ACTION|>``
    is 255014 against 255000 on North, ``<|message|>`` is 200023 against
    200000 on Muse.  A piece table cut at the base size leaves those ids with
    an empty piece, so no grammar can ever admit them and the model has to
    spell the marker out one ordinary token at a time (or dead-end).

    The bound used here is the tokenizer's whole id space: ``len(tokenizer)``
    (base plus added), raised to cover any added id that sits past it.  It is
    deliberately *not* the model's ``config.vocab_size`` / ``lm_head`` row
    count, which is padded above the tokenizer on every model checked
    (248320 vs 248077, 262144 vs 255032): those extra rows decode to nothing
    and must stay inadmissible.  The logits row is the wider array of the two
    on every model checked, so a mask built from this bound is padded with
    ``False`` out to the row's width; a row narrower than the tokenizer cuts
    the mask instead, since an id past the row cannot be sampled at all.
    Either way the mask is exactly as wide as the row (see ``_mask``).
    """
    bound = int(getattr(tokenizer, "vocab_size", 0) or 0)
    try:
        bound = max(bound, len(tokenizer))
    except TypeError:
        # Fake tokenizers in tests, and any wrapper without ``__len__``.
        pass
    added = getattr(tokenizer, "get_added_vocab", None)
    if callable(added):
        try:
            ids = list(added().values())
        except Exception:  # noqa: BLE001 - an unusable index is not a failure
            ids = []
        if ids:
            bound = max(bound, max(int(token) for token in ids) + 1)
    return bound


def token_pieces(tokenizer, vocab_size):
    """The text each token id adds when it follows other generated text.

    A SentencePiece decoder strips a word-boundary space at the start of a
    decode, so an isolated decode under-reports pieces such as ``"▁▁"`` (two
    spaces in context, one alone).  A mask built from isolated pieces then
    admits a whitespace token that overruns a bounded whitespace run and
    dead-ends the grammar.  Each piece is therefore measured after a plain
    anchor token; on byte-level vocabularies this equals the isolated decode.
    """

    def decode(ids):
        return tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    anchor = None
    try:
        encoded = list(tokenizer.encode("a", add_special_tokens=False))
        if encoded and decode(encoded[-1:]).endswith("a"):
            anchor = int(encoded[-1])
    except Exception:  # noqa: BLE001 - fall back to isolated pieces
        anchor = None
    base = decode([anchor]) if anchor is not None else None
    pieces = []
    for token in range(vocab_size):
        isolated = decode([token])
        if base is not None:
            joined = decode([anchor, token])
            if joined.startswith(base):
                pieces.append(joined[len(base):])
                continue
        pieces.append(isolated)
    return tuple(pieces)


def recursive_json_object_pattern():
    """Embeddable recursive JSON-object reference plus its regex definitions."""
    definitions = (
        rf"(?(DEFINE)(?P<value>{_WS}(?:{_STRING}|{_NUMBER}|true|false|null|"
        rf"(?&object)|(?&array)){_WS})(?P<object>\{{{_WS}(?:{_STRING}{_WS}:"
        rf"{_WS}(?&value)(?:,{_WS}{_STRING}{_WS}:{_WS}(?&value))*)?{_WS}\}})"
        rf"(?P<array>\[{_WS}(?:(?&value)(?:,{_WS}(?&value))*)?{_WS}\]))"
    )
    return "(?&object)", definitions


def _require_schema_keys(schema, allowed):
    unknown = set(schema) - set(allowed)
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unsupported JSON schema keywords: {names}")


def _json_literal(value):
    try:
        return json.dumps(value, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("JSON schema literals must be valid finite JSON values") from exc


def _schema_pattern(schema, depth=0, *, finite_numbers=False):
    if depth > 16 or not isinstance(schema, dict):
        raise ValueError("JSON schema must be an object with nesting depth at most 16")
    if "anyOf" in schema:
        _require_schema_keys(schema, {"anyOf"})
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not 1 <= len(branches) <= 8:
            raise ValueError("JSON schema anyOf must contain 1 to 8 schemas")
        return "(?:" + "|".join(
            _schema_pattern(branch, depth + 1, finite_numbers=finite_numbers)
            for branch in branches
        ) + ")"
    if "oneOf" in schema:
        raise ValueError("JSON schema oneOf is unsupported because overlapping branches are ambiguous")
    if "enum" in schema:
        _require_schema_keys(schema, {"enum", "type"})
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > 256:
            raise ValueError("JSON schema enum must contain 1 to 256 values")
        if "type" in schema and any(
            not schema_value_matches(value, {"type": schema["type"]})
            for value in enum
        ):
            raise ValueError("JSON schema enum values must match its declared type")
        return "(?:" + "|".join(regex.escape(_json_literal(v)) for v in enum) + ")"
    if "const" in schema:
        _require_schema_keys(schema, {"const", "type"})
        if "type" in schema:
            if not schema_value_matches(schema["const"], {"type": schema["type"]}):
                raise ValueError("JSON schema const must match its declared type")
        return regex.escape(_json_literal(schema["const"]))
    kind = schema.get("type")
    if isinstance(kind, list):
        _require_schema_keys(schema, {"type"})
        if not kind or len(kind) > 8:
            raise ValueError("JSON schema type union is invalid")
        return "(?:" + "|".join(
            _schema_pattern(
                {**schema, "type": item}, depth + 1,
                finite_numbers=finite_numbers,
            )
            for item in kind
        ) + ")"
    if kind == "string":
        _require_schema_keys(schema, {"type", "minLength", "maxLength"})
        if set(schema) == {"type"}:
            return _STRING
        minimum, maximum = string_length_bounds(schema)
        # Counting decoded JSON characters across escape spellings requires a
        # much larger automaton. A canonical unescaped subset is exact: every
        # admitted spelling decodes to a string within the declared bounds.
        return rf'"[^"\\\x00-\x1f]{{{minimum},{maximum}}}"'
    if kind in ("number", "integer"):
        _require_schema_keys(schema, {"type"})
        return (_FINITE_NUMBER if finite_numbers else _NUMBER) if kind == "number" else _INTEGER
    if kind == "boolean":
        _require_schema_keys(schema, {"type"})
        return r"(?:true|false)"
    if kind == "null":
        _require_schema_keys(schema, {"type"})
        return "null"
    if kind == "array":
        _require_schema_keys(schema, {"type", "items"})
        if "items" not in schema:
            raise ValueError("JSON schema arrays require an items schema")
        item = _schema_pattern(
            schema.get("items", {}), depth + 1,
            finite_numbers=finite_numbers,
        )
        return rf"\[{_WS}(?:{item}(?:{_WS},{_WS}{item})*)?{_WS}\]"
    if kind == "object" or "properties" in schema:
        _require_schema_keys(
            schema, {"type", "properties", "required", "additionalProperties"}
        )
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or len(properties) > 64:
            raise ValueError("JSON schema supports at most 64 object properties")
        if (
            not isinstance(required, list)
            or len(required) != len(set(required))
            or set(required) - set(properties)
            or any(not isinstance(name, str) for name in properties)
        ):
            raise ValueError("JSON schema required keys must name properties")
        if schema.get("additionalProperties", False) not in (False, None):
            raise ValueError("additionalProperties must be false for constrained output")
        # Canonical schema order makes the automaton bounded and deterministic.
        parts = []
        optional_seen = False
        for name, subschema in properties.items():
            value = _schema_pattern(
                subschema, depth + 1, finite_numbers=finite_numbers
            )
            pair = regex.escape(json.dumps(name)) + _WS + ":" + _WS + value
            if name in required:
                if optional_seen:
                    raise ValueError("required properties must precede optional properties")
                parts.append((pair, True))
            else:
                optional_seen = True
                parts.append((pair, False))
        required_parts = [part for part, required_flag in parts if required_flag]
        optional_parts = [part for part, required_flag in parts if not required_flag]
        if len(optional_parts) > 8:
            raise ValueError("JSON schema supports at most 8 optional properties")
        separator = f"{_WS},{_WS}"
        # Optional properties keep canonical order, each present or absent.
        # Build that language as a chain of optional groups rather than by
        # enumerating every subset: the 2**n alternation made the compiled
        # pattern so large that each partial match cost tens of microseconds,
        # which the per-token vocabulary walk multiplied into seconds.
        if required_parts:
            body = separator.join(required_parts) + "".join(
                f"(?:{separator}{part})?" for part in optional_parts
            )
        else:
            # No leading comma before the first present optional, so start the
            # chain at each possible first property in turn.
            chains = []
            for index, part in enumerate(optional_parts):
                tail = "".join(
                    f"(?:{separator}{later})?" for later in optional_parts[index + 1 :]
                )
                chains.append(part + tail)
            body = "(?:" + "|".join(chains) + ")?" if chains else ""
        return rf"\{{{_WS}{body}{_WS}\}}"
    raise ValueError(f"unsupported JSON schema type {kind!r}")



def _canonical_json_prefix(prefix, schema=None):
    """Shortest text with the same grammar state as ``prefix``.

    The compiled JSON grammars re-parse the whole prefix on every partial
    match, and the admissible set inside a string value is the same for every
    position of that value.  This lexer keeps every structural character,
    every key, every number/literal and the escape state verbatim, and drops
    the content of string values whose language is the unconstrained
    ``_STRING`` (all strings for ``json_object``; ``{"type": "string"}``
    properties and array items for a schema).  Anything the lexer does not
    understand (a malformed prefix) is returned unchanged, which is always
    sound because the caller then matches the original text.
    """
    if schema is not None and not isinstance(schema, dict):
        return prefix
    out = []
    # Stack of frames: ("obj", schema, last_key, expecting) / ("arr", item_schema)
    stack = []
    index = 0
    length = len(prefix)
    while index < length:
        char = prefix[index]
        if char == '"':
            # Keys select schema properties, so they are state under a
            # schema; every string is free under the recursive json_object
            # grammar; a value is free when its schema is a plain string.
            is_key = bool(stack and stack[-1][0] == "obj" and stack[-1][3] == "key")
            if schema is None:
                free = True
            elif is_key:
                free = False
            else:
                free = _string_value_is_free(stack)
            # Scan the string body.
            end = index + 1
            body = []
            escaped = False
            hex_left = 0
            closed = False
            while end < length:
                current = prefix[end]
                if hex_left:
                    if current not in "0123456789abcdefABCDEF":
                        return prefix
                    hex_left -= 1
                    body.append(current)
                    end += 1
                    continue
                if escaped:
                    if current == "u":
                        hex_left = 4
                    elif current not in '"\\/bfnrt':
                        return prefix
                    escaped = False
                    body.append(current)
                    end += 1
                    continue
                if current == "\\":
                    escaped = True
                    body.append(current)
                    end += 1
                    continue
                if current == '"':
                    closed = True
                    break
                if ord(current) < 0x20:
                    return prefix
                body.append(current)
                end += 1
            text = "".join(body)
            if free:
                # Content is not state; only an unfinished escape is.
                tail = ""
                if hex_left:
                    tail = text[-(2 + (4 - hex_left)) :]
                elif escaped:
                    tail = "\\"
                out.append('"' + tail + ('"' if closed else ""))
            else:
                out.append('"' + text + ('"' if closed else ""))
            if closed and is_key:
                frame = stack[-1]
                stack[-1] = (frame[0], frame[1], text, "colon")
            index = end + (1 if closed else 0)
            if not closed:
                break
            continue
        out.append(char)
        if char == "{":
            stack.append(("obj", _enter_schema(stack, schema), None, "key"))
        elif char == "[":
            stack.append(("arr", _enter_schema(stack, schema)))
        elif char in "}]":
            if not stack:
                return prefix
            stack.pop()
            if stack and stack[-1][0] == "obj":
                frame = stack[-1]
                stack[-1] = (frame[0], frame[1], frame[2], "sep")
        elif char == ":":
            if not stack or stack[-1][0] != "obj":
                return prefix
            frame = stack[-1]
            stack[-1] = (frame[0], frame[1], frame[2], "value")
        elif char == ",":
            if not stack:
                return prefix
            if stack[-1][0] == "obj":
                frame = stack[-1]
                stack[-1] = (frame[0], frame[1], frame[2], "key")
        index += 1
    return "".join(out)


def _canonical_json_object_prefix(prefix):
    """Depth-bounded canonical text for the recursive ``json_object`` grammar.

    Only the open path is state: for each unclosed container, whether it has
    completed members and what its current member is expecting.  Whitespace,
    string contents, completed siblings and the values of closed members are
    not.  A malformed prefix is returned unchanged (sound: the caller then
    matches the original text).
    """
    # Frames: ["obj", has_members, state] with state in key|colon|value|sep,
    #         ["arr", has_items, state] with state in value|sep.
    # ``scalar`` holds an in-progress number/literal; ``string`` an open
    # string's real state (escape tail) or None.
    stack = []
    scalar = None
    string = None  # (escaped, hex_left, tail)
    index = 0
    length = len(prefix)
    root_closed = False

    def value_done():
        nonlocal root_closed
        if stack:
            stack[-1][1] = True
            stack[-1][2] = "sep"
        else:
            root_closed = True

    while index < length:
        char = prefix[index]
        if string is not None:
            escaped, hex_left, tail = string
            if hex_left:
                if char not in "0123456789abcdefABCDEF":
                    return prefix
                string = (False, hex_left - 1, tail + char)
            elif escaped:
                if char == "u":
                    string = (False, 4, "\\u")
                elif char in '"\\/bfnrt':
                    string = (False, 0, "")
                else:
                    return prefix
            elif char == "\\":
                string = (True, 0, "\\")
            elif char == '"':
                string = None
                if stack and stack[-1][0] == "obj" and stack[-1][2] == "key":
                    stack[-1][2] = "colon"
                else:
                    value_done()
            elif ord(char) < 0x20:
                return prefix
            index += 1
            continue
        if scalar is not None:
            if char in "0123456789.eE+-" or char.isalpha():
                scalar += char
                index += 1
                continue
            scalar = None
            value_done()
            # fall through: ``char`` is structural or whitespace
        if char in " \t\n\r":
            index += 1
            continue
        if char == '"':
            if stack and stack[-1][0] == "obj" and stack[-1][2] not in ("key", "value"):
                return prefix
            if stack and stack[-1][0] == "arr" and stack[-1][2] != "value":
                return prefix
            string = (False, 0, "")
        elif char == "{" or char == "[":
            if root_closed or (
                stack
                and not (
                    (stack[-1][0] == "obj" and stack[-1][2] == "value")
                    or (stack[-1][0] == "arr" and stack[-1][2] == "value")
                )
            ):
                return prefix
            stack.append(["obj", False, "key"] if char == "{" else ["arr", False, "value"])
        elif char == "}" or char == "]":
            if not stack:
                return prefix
            frame = stack[-1]
            if char == "}" and (frame[0] != "obj" or frame[2] not in ("key", "sep")):
                return prefix
            if char == "]" and (frame[0] != "arr" or frame[2] not in ("value", "sep")):
                return prefix
            if frame[2] == "key" and frame[1]:
                return prefix  # trailing comma
            if frame[0] == "arr" and frame[2] == "value" and frame[1]:
                return prefix
            stack.pop()
            value_done()
        elif char == ":":
            if not stack or stack[-1][0] != "obj" or stack[-1][2] != "colon":
                return prefix
            stack[-1][2] = "value"
        elif char == ",":
            if not stack or stack[-1][2] != "sep":
                return prefix
            stack[-1][2] = "key" if stack[-1][0] == "obj" else "value"
        else:
            if stack and not (
                (stack[-1][0] == "obj" and stack[-1][2] == "value")
                or (stack[-1][0] == "arr" and stack[-1][2] == "value")
            ):
                return prefix
            scalar = char
        index += 1

    if root_closed:
        # A complete root object admits nothing further, whatever it held.
        return "{}" if len(prefix) >= 2 and not stack and scalar is None and string is None else prefix
    if not stack and (scalar is not None or string is not None):
        return prefix  # a bare scalar at the root is not an object
    out = []
    for frame in stack:
        kind, has_members, state = frame
        if kind == "obj":
            out.append("{")
            if state == "key":
                if has_members:
                    out.append('"":"",')
            elif state == "colon":
                out.append('""')
            elif state == "value":
                out.append('"":')
            else:  # sep
                out.append('"":""')
        else:
            out.append("[")
            if state == "value":
                if has_members:
                    out.append('"",')
            else:
                out.append('""')
    if scalar is not None:
        out.append(scalar)
    if string is not None:
        escaped, hex_left, tail = string
        out.append('"' + tail)
    canonical = "".join(out)
    return canonical if len(canonical) <= len(prefix) else prefix


def _enter_schema(stack, schema):
    """Schema of the container being opened, given the frame stack."""
    if schema is None:
        return None
    if not stack:
        return schema
    frame = stack[-1]
    if frame[0] == "obj":
        properties = frame[1].get("properties", {}) if isinstance(frame[1], dict) else {}
        return properties.get(frame[2]) if isinstance(properties, dict) else None
    if frame[0] == "arr":
        return frame[1].get("items") if isinstance(frame[1], dict) else None
    return None


def _string_value_is_free(stack):
    """Whether the string value at the current position is a plain string."""
    if not stack:
        return False
    frame = stack[-1]
    if frame[0] == "obj":
        if frame[3] != "value" or not isinstance(frame[1], dict):
            return False
        properties = frame[1].get("properties", {})
        target = properties.get(frame[2]) if isinstance(properties, dict) else None
    else:
        target = frame[1].get("items") if isinstance(frame[1], dict) else None
    return isinstance(target, dict) and target.get("type") == "string" and set(target) == {"type"}


class _Constraint:
    """A compiled grammar plus the prefix canonicalizer that is sound for it."""

    __slots__ = ("pattern", "schema", "canonical", "kind")

    def __init__(self, pattern, schema=None, canonical=None, kind="grammar"):
        self.pattern = pattern
        self.schema = schema
        self.canonical = canonical
        self.kind = kind

    def fullmatch(self, *args, **kwargs):
        return self.pattern.fullmatch(*args, **kwargs)

    def canonicalize(self, prefix):
        if self.canonical is None:
            return prefix
        return self.canonical(prefix, self.schema)


def compile_constraint(response_format=None, grammar=None, *, leading_whitespace=False):
    """Compile a request's constraint.

    ``leading_whitespace`` is used when the grammar is deferred past a
    thinking-close marker: chat templates put whitespace between the marker
    and the answer (Qwen emits a blank line), so the JSON grammars then admit
    insignificant JSON whitespace before the root value.  A raw ``grammar``
    regex is the client's exact language and is never widened.
    """
    lead = _WS if leading_whitespace else ""
    if grammar is not None:
        if not isinstance(grammar, str) or not grammar or len(grammar) > 4096:
            raise ValueError("grammar must be a nonempty regex of at most 4096 characters")
        try:
            return _Constraint(regex.compile(rf"(?:{grammar})"), kind="grammar")
        except regex.error as exc:
            raise ValueError(f"invalid grammar regex: {exc}") from exc
    if response_format is None or response_format == {"type": "text"}:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "json_object" and set(response_format) == {"type"}:
        # Recursive JSON value grammar, narrowed to an object at the root.
        root, definitions = recursive_json_object_pattern()
        pattern = lead + root + definitions
        return _Constraint(
            regex.compile(pattern),
            None,
            lambda text, _schema: _canonical_json_object_prefix(text),
            "json_object",
        )
    if kind == "json_schema" and set(response_format) <= {"type", "json_schema"}:
        wrapper = response_format.get("json_schema")
        if not isinstance(wrapper, dict) or set(wrapper) - {"name", "description", "schema", "strict"}:
            raise ValueError("json_schema wrapper is invalid")
        if wrapper.get("strict", True) is not True or not isinstance(wrapper.get("schema"), dict):
            raise ValueError("json_schema requires strict:true and a schema object")
        schema = executable_schema(wrapper["schema"])
        return _Constraint(
            regex.compile(lead + _schema_pattern(schema)),
            schema,
            _canonical_json_prefix,
            "json_schema",
        )
    raise ValueError("response_format supports text, json_object, or strict json_schema")


def _build_piece_index(pieces, excluded):
    """Sort the usable vocabulary pieces so shared prefixes are contiguous.

    ``excluded`` holds the ids never admitted by their text: the terminals
    (admitted by id) and the other special tokens.
    """
    order = sorted(
        (
            token
            for token, piece in enumerate(pieces)
            if token not in excluded and piece and "\ufffd" not in piece
        ),
        key=lambda token: pieces[token],
    )
    return (tuple(pieces[token] for token in order), tuple(order))



# ---------------------------------------------------------------------------
# Parallel scanners.  ``regex`` holds the GIL while matching, so exact scans of
# the vocabulary tail run in a spawn-based process pool.  Workers hold the
# vocabulary pieces once and compile each grammar on first use.

_WORKER_PIECES = None
_WORKER_PATTERNS = {}


def _scanner_init(pieces):
    global _WORKER_PIECES
    _WORKER_PIECES = pieces


def _scanner_scan(pattern_source, pattern_flags, prefix, tokens, timeout, wall_deadline):
    """Admissible ``tokens`` under ``prefix``; ``None`` once ``wall_deadline`` passes.

    Aborting in the worker keeps an overrun from leaving stale shards that
    later scans would queue behind.
    """
    pattern = _WORKER_PATTERNS.get((pattern_source, pattern_flags))
    if pattern is None:
        pattern = _WORKER_PATTERNS[(pattern_source, pattern_flags)] = regex.compile(
            pattern_source, pattern_flags
        )
    admitted = []
    count = len(_WORKER_PIECES)
    for index, token in enumerate(tokens):
        if index % 64 == 0 and time.time() > wall_deadline:
            return None
        if token >= count:
            continue  # logits are padded beyond the tokenizer vocabulary
        piece = _WORKER_PIECES[token]
        if not piece or "\ufffd" in piece:
            continue
        try:
            if pattern.fullmatch(prefix + piece, partial=True, timeout=timeout) is not None:
                admitted.append(token)
        except TimeoutError:
            continue
    return admitted


class _ScannerPool:
    """Lazily started pool of regex scanner processes bound to one vocabulary.

    Built on ``ProcessPoolExecutor`` rather than ``multiprocessing.Pool``: an
    idle Pool worker holds the task-queue read lock while blocked in recv, so
    a worker killed from outside left the parent's atexit finalizer waiting on
    that lock forever, with the model still resident.  The executor's shutdown
    path handles broken workers and ``shutdown_scanner_pools`` runs it from
    ``ServingEngine.close`` and at interpreter exit.
    """

    def __init__(self, pieces, workers):
        import multiprocessing

        self.workers = int(workers)
        self.pieces = pieces
        self._executor = None
        self._failed = False
        self._context = multiprocessing.get_context("spawn")
        # Measured wall seconds per scanned token (EMA), optimistic to start.
        self.seconds_per_token = 2e-6

    def available(self):
        if self._failed or self.workers <= 0:
            return False
        if self._executor is None:
            try:
                from concurrent.futures import ProcessPoolExecutor

                self._executor = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=self._context,
                    initializer=_scanner_init,
                    initargs=(self.pieces,),
                )
            except Exception:  # noqa: BLE001 - fall back to the in-process walk
                self._failed = True
                return False
        return True

    def scan(self, pattern, prefix, tokens, budget_seconds):
        """Exact admissibility of ``tokens`` under ``prefix`` or None on overrun."""
        if not self.available() or not tokens:
            return [] if not tokens else None
        shards = max(1, min(self.workers * 2, len(tokens) // 512 or 1))
        size = -(-len(tokens) // shards)
        wall_deadline = time.time() + budget_seconds
        try:
            futures = [
                self._executor.submit(
                    _scanner_scan,
                    pattern.pattern,
                    pattern.flags,
                    prefix,
                    tokens[i : i + size],
                    _MATCH_TIMEOUT_SECONDS,
                    wall_deadline,
                )
                for i in range(0, len(tokens), size)
            ]
        except Exception:  # noqa: BLE001 - broken or shut-down executor
            self._failed = True
            return None
        started = time.perf_counter()
        deadline = started + budget_seconds
        admitted = []
        overrun = False
        for future in futures:
            remaining = deadline - time.perf_counter()
            try:
                part = future.result(timeout=max(remaining, 0.0) + 0.05)
            except Exception as exc:  # noqa: BLE001 - timeout or worker failure: bounded path
                if _DEBUG:
                    import sys

                    print(f"[structured] POOL job error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                if type(exc).__name__ == "BrokenProcessPool":
                    self._failed = True
                overrun = True
                continue
            if part is None:
                overrun = True
                continue
            admitted.extend(part)
        elapsed = time.perf_counter() - started
        # Calibrate the per-token rate only on scans large enough that the
        # fixed dispatch overhead does not dominate.
        if len(tokens) >= _POOL_CALIBRATION_MIN:
            observed = max(elapsed - _POOL_FIXED_SECONDS, 0.0) / len(tokens)
            if not overrun:
                self.seconds_per_token = 0.5 * self.seconds_per_token + 0.5 * observed
            else:
                self.seconds_per_token = max(self.seconds_per_token, observed)
        return None if overrun else admitted

    def close(self):
        executor = self._executor
        self._executor = None
        if executor is not None:
            processes = list((getattr(executor, "_processes", None) or {}).values())
            # Do not wait: a worker that died mid-task must not block exit.
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                try:
                    process.terminate()
                except Exception:  # noqa: BLE001 - already gone
                    pass


_SCANNER_POOLS = {}


def shutdown_scanner_pools():
    """Terminate every scanner pool; safe to call repeatedly."""
    for pool in list(_SCANNER_POOLS.values()):
        pool.close()
    _SCANNER_POOLS.clear()


import atexit as _atexit

_atexit.register(shutdown_scanner_pools)


def scanner_pool_for(tokenizer, pieces):
    """One pool per vocabulary; ``MLX2_STRUCTURED_WORKERS=0`` disables it."""
    import os

    workers = int(os.environ.get("MLX2_STRUCTURED_WORKERS", "8") or 0)
    if workers <= 0:
        return None
    key = id(pieces)
    pool = _SCANNER_POOLS.get(key)
    if pool is None:
        pool = _SCANNER_POOLS[key] = _ScannerPool(pieces, workers)
    return pool


# Deepest recursive-grammar nesting the automaton engine will track, and how
# many trailing positions keep their configuration for speculative rollback.
_MAX_PUSHDOWN_DEPTH = 256
_TRACK_HISTORY = 256


class ThinkingBudgetProcessor:
    """Force a declared thinking-close sequence from token history alone.

    Design references: mlx-vlm#2230, omlx#3510, ollama#17566.  The output mask
    is a pure function of the generated prefix: no mutable call count decides
    which marker token comes next, so MTP draft/verify and rollback replay the
    exact same decision.
    """

    # P5: the mask is a pure function of the token history.
    history_pure = True

    def __init__(self, prompt_length, budget, close_token_ids):
        self.prompt_length = int(prompt_length)
        self.budget = int(budget)
        self.close_token_ids = tuple(int(token) for token in close_token_ids)
        if self.prompt_length < 0 or self.budget < 0 or not self.close_token_ids:
            raise ValueError("thinking budget processor requires valid bounds and a close marker")
        self.fired = False

    def _close_position(self, generated):
        marker = self.close_token_ids
        limit = len(generated) - len(marker) + 1
        for position in range(max(0, limit)):
            if tuple(generated[position : position + len(marker)]) == marker:
                return position
        return None

    def _generated_values(self, tokens):
        """Copy only the generated suffix; the fixed prompt is never rescanned."""
        if tokens is None:
            return []
        generated = tokens[self.prompt_length :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        return [int(item) for item in generated]

    def _partial_at_boundary(self, generated):
        """Longest marker prefix ending exactly at the budget boundary."""
        marker = self.close_token_ids
        for width in range(min(len(marker) - 1, self.budget), 0, -1):
            if tuple(generated[self.budget - width : self.budget]) == marker[:width]:
                return width
        return 0

    def fired_for_tokens(self, tokens):
        """Whether committed ``tokens`` contain the budget-forced marker."""
        return self.fired_for_generated(self._generated_values(tokens))

    def fired_for_generated(self, generated):
        """Whether a committed generated-token suffix crossed the boundary."""
        position = self._close_position(generated)
        return position is not None and (
            position <= self.budget < position + len(self.close_token_ids)
        )

    def dormant(self, tokens):
        """P5: whether ``tokens`` leave this processor masking nothing.

        True before the budget boundary and once a close marker is present;
        side-effect free (``fired`` is not touched).
        """
        generated = self._generated_values(tokens)
        return (
            self._close_position(generated) is not None
            or len(generated) < self.budget
        )

    def probe(self, tokens, logits):
        """Evaluate a provisional token prefix without publishing ``fired``."""
        import copy

        return copy.deepcopy(self)(tokens, logits)

    def __call__(self, tokens, logits):
        import mlx.core as mx

        generated = self._generated_values(tokens)
        position = self._close_position(generated)
        if position is not None:
            # A marker crossing the boundary was completed under enforcement;
            # one ending at or before it was emitted naturally.
            self.fired = (
                position <= self.budget < position + len(self.close_token_ids)
            )
            return logits
        if len(generated) < self.budget:
            self.fired = False
            return logits
        partial = self._partial_at_boundary(generated)
        start = self.budget - partial
        offset = len(generated) - start
        prefix = tuple(generated[start:])
        if offset >= len(self.close_token_ids) or prefix != self.close_token_ids[:offset]:
            # A caller that installs the processor after generation began, or a
            # stale speculative branch, restarts the marker. Normal decoding
            # never reaches this branch because every preceding row was masked.
            offset = 0
        target = self.close_token_ids[offset]
        if not 0 <= target < logits.shape[-1]:
            raise ValueError("thinking-close token is outside the probability vocabulary")
        self.fired = True
        mask = mx.full(logits.shape[-1], -float("inf"), dtype=logits.dtype)
        mask = mx.put_along_axis(
            mask,
            mx.array([target]),
            mx.zeros((1,), dtype=logits.dtype),
            axis=-1,
        )
        return logits + mask

class StructuredOutputProcessor:
    # P5: every mask is re-derived from the token ids (the tracking state is a
    # cache of them), so a replay rebuilds this processor from history alone.
    history_pure = True

    def __init__(
        self,
        tokenizer,
        prompt_length,
        constraint,
        *,
        greedy=False,
        top_k=0,
        top_p=0.0,
        defer_until=None,
        envelope=None,
        block_eos_while_deferred=False,
        constraint_kind=None,
        capture_failure_context=False,
        generation_stop_token_ids=None,
    ):
        self.tokenizer = tokenizer
        # Grammar deferral: while the generated ids do not yet contain the
        # ``defer_until`` token sequence (the adapter's thinking-close marker)
        # logits pass through untouched; constraining starts, with an empty
        # constrained prefix, at the token after the marker's first occurrence.
        self._defer_until = tuple(int(t) for t in defer_until) if defer_until else None
        self.deferred = self._defer_until is not None
        self._envelope = (
            (tuple(int(t) for t in envelope[0]), tuple(int(t) for t in envelope[1]))
            if envelope else None
        )
        self.block_eos_while_deferred = bool(block_eos_while_deferred)
        self.constraining = not self.deferred
        self.deferred_tokens = 0
        self.prompt_length = prompt_length
        self.constraint = constraint
        self.constraint_kind = str(
            constraint_kind or getattr(constraint, "kind", "grammar")
        )[:_FAILURE_TEXT_CHARS]
        self.capture_failure_context = bool(capture_failure_context)
        # Sampling controls of the owning lane decide when the logit-ordered
        # walk may stop (see _allowed_by_logit_order).
        self.greedy = bool(greedy)
        self.top_k = int(top_k or 0)
        self.top_p = float(top_p or 0.0)
        # Largest unexamined-to-admitted mass ratio at which a top_p walk
        # stopped without an exact nucleus decision; the receipt reports it.
        # 0.0 means every mask this request produced was exact.
        self.tail_mass_bound = 0.0
        stop_ids = (
            tokenizer.eos_token_ids
            if generation_stop_token_ids is None
            else generation_stop_token_ids
        )
        self.eos_ids = frozenset(int(token) for token in stop_ids)
        # Special tokens other than the terminals add no text to the tracked
        # grammar prefix: the decode it mirrors skips them, and the byte table
        # has no bytes for them.  Admitting one by its text would let the
        # stream show markup the grammar never checked (a ``maxLength: 6``
        # string delivered as ``<pad><pad>``), so the mask never does.  Tool
        # and envelope markers are ordinary added tokens and stay admissible.
        try:
            special = {int(token) for token in getattr(tokenizer, "all_special_ids", ())}
        except (TypeError, ValueError):
            special = set()
        self._special_ids = frozenset(special) - self.eos_ids
        unmaskable = self.eos_ids | self._special_ids
        # The whole tokenizer id space, not the base vocabulary: added tokens
        # carry the tool-call markers every adapter grammar matches against
        # (see ``vocabulary_bound``).
        self.vocab_size = vocabulary_bound(tokenizer)
        pieces = getattr(tokenizer, _TOKEN_PIECES_ATTRIBUTE, None)
        if not isinstance(pieces, tuple) or len(pieces) != self.vocab_size:
            pieces = token_pieces(tokenizer, self.vocab_size)
            try:
                setattr(tokenizer, _TOKEN_PIECES_ATTRIBUTE, pieces)
            except (AttributeError, TypeError):
                pass
        self._pieces = pieces
        index = getattr(tokenizer, _TOKEN_INDEX_ATTRIBUTE, None)
        if not isinstance(index, tuple) or len(index) != 2 or len(index[0]) > len(pieces):
            index = _build_piece_index(pieces, unmaskable)
            try:
                setattr(tokenizer, _TOKEN_INDEX_ATTRIBUTE, index)
            except (AttributeError, TypeError):
                pass
        (self._sorted_pieces, self._sorted_tokens) = index
        self._allowed_cache = OrderedDict()
        self._partial_cache = OrderedDict()
        self._pool = scanner_pool_for(tokenizer, pieces)
        self.parallel_scans = 0
        # Engine selection: an exact token-level automaton when the pattern is
        # in the compilable subset, else the regex scanner above.  Refusal is
        # fail-safe: the scanner stays in charge, nothing is left unmasked.
        self.engine = "scanner"
        self.automaton_refusal = None
        # Masks this processor produced (either engine); deferred passthrough
        # steps and steps after a failure are not counted.
        self.constrained_steps = 0
        self._automaton = None
        self._trie = None
        self._track_ids = []
        self._track_configs = []
        self._track_lengths = []
        self._track_pending = []
        self._track_text = ""
        self._last_automaton_config = None
        # Byte-level view of the vocabulary: lets pieces that are only part of
        # a UTF-8 character be admitted.  None keeps them inadmissible.
        self._token_bytes = None
        self._fragments = None
        if __import__("os").environ.get("MLX2_STRUCTURED_AUTOMATON", "1") == "0":
            self.automaton_refusal = "disabled by MLX2_STRUCTURED_AUTOMATON=0"
        elif not isinstance(constraint, _Constraint):
            self.automaton_refusal = "constraint is not a compiled pattern"
        else:
            from .structured_automaton import AutomatonUnsupported, automaton_for, trie_for

            try:
                self._automaton = automaton_for(constraint.pattern)
            except AutomatonUnsupported as exc:
                self.automaton_refusal = str(exc)
            else:
                self._trie = trie_for(tokenizer, pieces, unmaskable)
                self._track_configs = [self._automaton.start]
                self._track_lengths = [0]
                self._track_pending = [b""]
                from .structured_automaton import fragments_for

                self._token_bytes, self._fragments = fragments_for(
                    tokenizer, pieces, unmaskable
                )
                self.engine = "automaton"
        # Set when the constraint cannot be honoured (dead end or budget
        # overrun).  The processor then stops masking and the serving layer
        # fails the request closed; raising from inside a batched forward
        # would take every other lane down with it.
        self.failure = None
        self.failure_context = None

    def __deepcopy__(self, memo):
        """Snapshot for a speculative round without copying shared machinery.

        Schedulers deep-copy lane state to roll a round back.  The automaton,
        vocabulary trie, byte index and scanner pool are shared, immutable for
        this purpose, and hold thread locks that cannot be copied; only the
        per-lane bookkeeping is duplicated.  (It is re-derived from the token
        ids on every call, so a restored snapshot is self-correcting.)
        """
        import copy

        clone = copy.copy(self)
        for name in ("_track_ids", "_track_configs", "_track_lengths", "_track_pending"):
            setattr(clone, name, list(getattr(self, name)))
        clone._allowed_cache = OrderedDict(self._allowed_cache)
        clone._partial_cache = OrderedDict(self._partial_cache)
        memo[id(self)] = clone
        return clone

    def dormant(self, tokens):
        """P5: whether the next row for ``tokens`` passes logits through.

        True while the grammar is deferred before its marker (unless EOS is
        blocked meanwhile) and after a latched failure.  Side-effect free:
        unlike ``__call__`` it does not update ``constraining``.
        """
        if self.failure is not None:
            return True
        marker = self._defer_until
        if marker is None or self.block_eos_while_deferred:
            return False
        generated = tokens[self.prompt_length :]
        if hasattr(generated, "tolist"):
            generated = generated.tolist()
        generated = [int(item) for item in generated]
        width = len(marker)
        return not any(
            tuple(generated[position : position + width]) == marker
            for position in range(len(generated) - width + 1)
        )

    def probe(self, tokens, logits):
        """Mask provisional draft logits without publishing processor state."""
        import copy

        return copy.deepcopy(self)(tokens, logits)

    def _admissible(self, text, deadline):
        """Whether ``text`` can still be extended into (or is) a full match.

        ``None`` marks a timed-out check.  Excluding the piece, and every piece
        extending it, is conservative: it can only narrow the accepted
        language (a brute-force scan would still have tested the longer pieces
        individually), while allowing it after an inconclusive match would
        violate fail-closed structured output.  Each call is capped by both
        the per-match timeout and the remaining per-token budget, so the
        budget is a hard bound on the generation thread.
        """
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            if _DEBUG:
                import sys

                print(f"[structured] BUDGET text={text[-60:]!r}", file=sys.stderr, flush=True)
            raise ValueError(
                "structured-output grammar exceeded its per-token match budget"
            )
        try:
            return (
                self.constraint.fullmatch(
                    text,
                    partial=True,
                    timeout=min(_MATCH_TIMEOUT_SECONDS, remaining),
                )
                is not None
            )
        except TimeoutError:
            return None

    def _allowed(self, prefix):
        cached = self._allowed_cache.get(prefix)
        if cached is not None:
            self._allowed_cache.move_to_end(prefix)
            return cached
        complete = self._complete(prefix)
        allowed = set(self.eos_ids if complete else ())
        # Partial-match admissibility is monotone in the text: if
        # ``prefix + q`` cannot begin a match, no extension of ``q`` can.  Walk
        # the sorted vocabulary as a prefix tree and prune whole subtrees, so a
        # 248K-piece vocabulary costs a few thousand regex calls per new prefix
        # instead of one per piece (which ran to tens of seconds per token on
        # the generation thread for a plain json_object grammar).
        pieces = self._sorted_pieces
        tokens = self._sorted_tokens
        deadline = time.perf_counter() + _ALLOWED_BUDGET_SECONDS
        scanned = 0
        stack = [(0, len(pieces), 0)]  # [start, end) of pieces sharing depth chars
        while stack:
            (start, end, depth) = stack.pop()
            index = start
            while index < end:
                piece = pieces[index]
                if len(piece) == depth:
                    # A duplicate decode of the head this range was expanded
                    # from; the parent already established its admissibility.
                    allowed.add(tokens[index])
                    index += 1
                    continue
                head = piece[: depth + 1]
                stop = index + 1
                while stop < end and pieces[stop].startswith(head):
                    stop += 1
                    scanned += 1
                    if scanned % _BUDGET_CHECK_INTERVAL == 0 and time.perf_counter() > deadline:
                        raise ValueError(
                            "structured-output grammar exceeded its per-token match budget"
                        )
                if self._admissible(prefix + head, deadline):
                    first = index
                    if piece == head:
                        allowed.add(tokens[index])
                        first += 1
                    if first < stop:
                        stack.append((first, stop, depth + 1))
                index = stop
        if not allowed:
            raise ValueError("structured-output grammar has no valid token continuation")
        result = tuple(sorted(allowed))
        self._allowed_cache[prefix] = result
        if len(self._allowed_cache) > _MAX_PREFIX_CACHE:
            self._allowed_cache.popitem(last=False)
        return result

    def _complete(self, prefix):
        try:
            return (
                self.constraint.fullmatch(prefix, timeout=_MATCH_TIMEOUT_SECONDS)
                is not None
            )
        except TimeoutError as exc:
            raise ValueError("structured-output grammar exceeded its match budget") from exc

    def _allowed_by_logit_order(self, prefix, logits_row):
        """Admissible tokens the sampler can actually reach, cheapest first.

        Inside a JSON string almost every vocabulary piece is admissible, so
        the prefix-tree walk cannot prune and a full pass costs seconds per
        token on the real vocabulary.  Examine pieces in descending logit
        order instead -- by (-logit, id), so equal logits have a fixed order
        and the frontier stays consistent as it grows; every unexamined token
        then has no higher probability than every admitted one, which makes
        these stopping rules exact for the lane's sampler (top_p, then min_p,
        then top_k):

        * greedy: the first admissible token is the argmax of the fully
          masked row;
        * top_k without top_p: top_k keeps the k most probable admissible
          tokens, and min_p compares against the maximum, so the first k
          admissible tokens are the whole support;
        * top_p in (0, 1): a tail token survives the nucleus only if the
          admissible tail mass exceeds (1 - top_p) times the admissible
          total, so once the unexamined mass R is below (1 - top_p) times the
          admitted mass S no tail token is kept.  The nucleus boundary among
          admitted tokens is exact when it agrees at both extremes of the
          unknown tail mass (0 and R).  Otherwise the walk still stops and
          records R / S as ``tail_mass_bound``: the boundary can move by at
          most that fraction of cumulative mass.  Served bfloat16 logits
          carry a flat floor of ~5% mass spread over the whole vocabulary,
          so driving R lower would mean a full scan (seconds per token);
          the route is qualified as numerically bounded and the receipt
          carries the bound actually incurred.

        Pure temperature sampling (no top_k, no top_p) stops when the
        unexamined mass is below _MASS_RESOLUTION of the admitted mass or
        the vocabulary is exhausted, recording the same bound.
        Decisions are memoized per prefix so verify positions and rollbacks
        reuse them.
        """
        import mlx.core as mx
        import numpy as np

        if isinstance(logits_row, mx.array):
            # Served logits are bfloat16, which numpy cannot read directly.
            logits_row = np.array(logits_row.astype(mx.float32))
        memo = self._partial_cache.get(prefix)
        if memo is None:
            memo = self._partial_cache[prefix] = {}
            if len(self._partial_cache) > _MAX_PREFIX_CACHE:
                self._partial_cache.popitem(last=False)
        else:
            self._partial_cache.move_to_end(prefix)
        complete = memo.get(_COMPLETE_KEY)
        if complete is None:
            complete = memo[_COMPLETE_KEY] = self._complete(prefix)
        deadline = time.perf_counter() + _ALLOWED_BUDGET_SECONDS
        row = np.asarray(logits_row, dtype=np.float32)
        finite = np.isfinite(row)
        if not finite.any():
            return ()
        shifted = np.where(finite, row - row[finite].max(), -np.inf)
        probs = np.exp(shifted)
        total = float(probs.sum())
        vocab = row.shape[0]
        # Ranking key for the walk.  Non-finite logits already carry zero mass
        # (``shifted`` sends them to -inf), so ranking them last costs nothing
        # and keeps NaN out of the tie-break comparisons below.
        rank = np.where(finite, row, -np.inf)
        allowed = []
        admissible_mass = 0.0
        examined_mass = 0.0
        chunk = 256
        start = 0
        order = None
        tail_fraction = None
        pool_tried = False
        top_p_active = 0.0 < self.top_p < 1.0

        def decide(token):
            decision = memo.get(token)
            if decision is None:
                if token in self.eos_ids:
                    decision = complete
                elif token in self._special_ids:
                    decision = False
                else:
                    piece = self._pieces[token] if token < len(self._pieces) else ""
                    if not piece or "\ufffd" in piece:
                        decision = False
                    else:
                        decision = bool(self._admissible(prefix + piece, deadline))
                memo[token] = decision
            return decision

        while start < vocab:
            if order is None or start >= order.shape[0]:
                # Grow the examined frontier geometrically; a full argsort of
                # the vocabulary is only paid for flat distributions.  ``start``
                # carries over from the previous, smaller array, so the walk
                # only ever reads ``order[start:]`` -- which is correct exactly
                # when the new prefix [0, start) is the set already examined.
                # Every rebuild therefore orders by (-logit, id), a total order
                # whose top-``want`` prefix is stable as ``want`` grows.  Under
                # ties this is not automatic: ``argpartition`` picks an
                # arbitrary representative set among equal values, so a naive
                # rebuild can bury never-examined ids in the skipped prefix and
                # drop them from the mask for good.
                want = min(vocab, max(chunk, (order.shape[0] if order is not None else 0) * 8))
                if want >= vocab:
                    # Stable argsort already breaks ties by ascending id.
                    order = np.argsort(-rank, kind="stable")
                else:
                    top = np.argpartition(-rank, want - 1)[:want]
                    # Re-derive the cut deterministically: everything strictly
                    # above it (all of which ``top`` holds, so at most want - 1
                    # ids), then the lowest-numbered ids sitting on it.
                    cut = float(rank[top].min())
                    above = np.flatnonzero(rank > cut)
                    tied = np.flatnonzero(rank == cut)
                    chosen = np.concatenate((above, tied[: want - above.shape[0]]))
                    order = chosen[np.argsort(-rank[chosen], kind="stable")]
            stop = min(order.shape[0], start + chunk)
            try:
                for token in order[start:stop].tolist():
                    mass = float(probs[token])
                    examined_mass += mass
                    if mass <= 0.0:
                        continue
                    if decide(token):
                        allowed.append(token)
                        admissible_mass += mass
            except ValueError:
                # Budget exhausted.  With admitted tokens in hand this is a
                # fidelity limit, not a failure: stop and record the bound
                # (the whole unexamined remainder if no estimate exists yet).
                if not allowed:
                    raise
                remaining = max(total - examined_mass, 0.0)
                share = 1.0 if tail_fraction is None else min(1.0, tail_fraction)
                self.tail_mass_bound = max(
                    self.tail_mass_bound,
                    min(1.0, share * remaining / admissible_mass),
                )
                break
            start = stop
            if not allowed:
                continue
            if self.greedy:
                break
            remaining = max(total - examined_mass, 0.0)
            if remaining <= admissible_mass * _MASS_RESOLUTION:
                break  # the tail cannot matter to any sampler
            if self.top_k and len(allowed) >= self.top_k and not top_p_active:
                break  # exact: top_k selects among admitted tokens only
            if top_p_active and remaining <= (1.0 - self.top_p) * admissible_mass and (
                self._nucleus_boundary_stable(
                    [float(probs[token]) for token in allowed], remaining
                )
            ):
                break  # exact: no tail token or boundary can change
            # Parallel heads: the in-process shortcut did not settle this
            # position, so finish the tail exactly on the scanner pool while
            # it is still cheap, instead of walking towards the bounded stop.
            if self._pool is not None and not pool_tried:
                pool_tried = True
                exact_tail = self._finish_exactly(
                    prefix, order if order.shape[0] >= vocab else np.argsort(-rank, kind="stable"),
                    start, probs, memo, deadline
                )
                if exact_tail is not None:
                    for token in exact_tail:
                        allowed.append(token)
                        admissible_mass += float(probs[token])
                    self.parallel_scans += 1
                    break
            # Bounded stop.  The unexamined tail is a flat floor of tiny
            # probabilities; its admissible share of mass is estimated once
            # from a probability-weighted sample, and every remaining test
            # uses that estimate in place of the raw tail mass.
            if tail_fraction is None:
                if order.shape[0] < vocab:
                    order = np.argsort(-rank, kind="stable")
                try:
                    tail_fraction = self._estimate_tail_fraction(
                        order[start:], probs, decide, prefix
                    )
                except ValueError:
                    self.tail_mass_bound = max(
                        self.tail_mass_bound,
                        min(1.0, remaining / admissible_mass),
                    )
                    break
            tail_admissible = min(1.0, tail_fraction) * remaining
            bound = min(1.0, tail_admissible / admissible_mass)
            # Work cap: past this share of the vocabulary the remaining tail
            # is the flat floor; stop with the bound rather than scan it all.
            capped = start >= vocab * _MAX_EXAMINED_FRACTION
            if top_p_active:
                if (
                    tail_admissible > (1.0 - self.top_p) * admissible_mass
                    and not capped
                    and tail_fraction > _TAIL_UNRESOLVED
                ):
                    continue  # an admissible tail token could enter the nucleus
                if len(allowed) >= max(self.top_k, 1) and self._nucleus_boundary_stable(
                    [float(probs[token]) for token in allowed], tail_admissible
                ):
                    bound = 0.0  # boundary stable for any tail within the estimate
                self.tail_mass_bound = max(self.tail_mass_bound, bound)
                break
            if (
                (self.top_k and len(allowed) < self.top_k)
                or bound <= _BOUND_STOP
                or capped
                or tail_fraction <= _TAIL_UNRESOLVED
            ):
                self.tail_mass_bound = max(self.tail_mass_bound, bound)
                break
        if _DEBUG:
            import sys

            print(
                f"[structured] prefix_len={len(prefix)} examined={start} admitted={len(allowed)} "
                f"S={admissible_mass:.4f} R={max(total - examined_mass, 0.0):.4f} "
                f"elapsed={time.perf_counter() - (deadline - _ALLOWED_BUDGET_SECONDS):.2f}s "
                f"greedy={self.greedy} top_k={self.top_k} top_p={self.top_p}",
                file=sys.stderr, flush=True,
            )
        if not allowed:
            raise ValueError("structured-output grammar has no valid token continuation")
        return tuple(allowed)

    def _estimate_tail_fraction(self, unexamined, probs, decide, prefix):
        """Upper-bound the admissible share of the unexamined tail's mass.

        Tail tokens are sampled in proportion to their probability (seeded by
        the prefix so the mask is deterministic), so the admissible count
        estimates the admissible *mass* fraction directly; the recorded bound
        uses the upper end of its binomial confidence interval.
        """
        import hashlib

        import numpy as np

        count = int(unexamined.shape[0])
        if count == 0:
            return 0.0
        weights = probs[unexamined].astype(np.float64)
        weight_total = float(weights.sum())
        if weight_total <= 0.0:
            return 0.0
        sample = min(_TAIL_SAMPLE, count)
        seed = int.from_bytes(hashlib.sha256(prefix.encode("utf-8", "replace")).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        if sample == count:
            picks = unexamined
            admissible_mass = sum(
                float(probs[token]) for token in picks.tolist() if decide(token)
            )
            return min(1.0, admissible_mass / weight_total)
        picks = rng.choice(unexamined, size=sample, replace=True, p=weights / weight_total)
        admissible = sum(1 for token in picks.tolist() if decide(token))
        fraction = admissible / sample
        sigma = (fraction * (1.0 - fraction) / sample) ** 0.5
        return min(1.0, fraction + 3.0 * sigma + 1.0 / sample)

    def _finish_exactly(self, prefix, order, start, probs, memo, deadline):
        """Scan the unexamined tail exactly on the pool if it fits the budget."""
        remaining_budget = deadline - time.perf_counter()
        if remaining_budget <= 0.05:
            return None
        unexamined = order[start:]
        for eos in self.eos_ids:
            memo.setdefault(eos, memo.get(_COMPLETE_KEY, False))
        limit = len(self._pieces)
        pending = []
        for token in unexamined.tolist():
            if token in memo:
                continue
            if token >= limit or token in self._special_ids:
                memo[token] = False  # padding beyond the vocabulary, or special
                continue
            pending.append(int(token))
        if len(pending) < _POOL_MIN_TOKENS:
            # Too small to be worth a dispatch; finish in-process.
            admitted_small = []
            for token in pending:
                decision = self._admissible(prefix + self._pieces[token], deadline)
                memo[token] = bool(decision)
                if decision:
                    admitted_small.append(token)
            result = []
            for token in unexamined.tolist():
                if memo.get(token) and float(probs[token]) > 0.0:
                    result.append(int(token))
            return result
        projected = self._pool.seconds_per_token * len(pending) * 1.5 + _POOL_FIXED_SECONDS
        if projected > remaining_budget:
            if _DEBUG:
                import sys

                print(f"[structured] POOL skip pending={len(pending)} projected={projected:.2f}s budget={remaining_budget:.2f}s", file=sys.stderr, flush=True)
            return None
        started = time.perf_counter()
        admitted = self._pool.scan(self.constraint.pattern, prefix, pending, remaining_budget)
        if _DEBUG:
            import sys

            print(f"[structured] POOL scan pending={len(pending)} took={time.perf_counter() - started:.2f}s budget={remaining_budget:.2f}s result={'overrun' if admitted is None else len(admitted)} rate={self._pool.seconds_per_token * 1e6:.1f}us", file=sys.stderr, flush=True)
        if admitted is None:
            return None
        admitted_set = set(admitted)
        for token in pending:
            memo[token] = token in admitted_set
        result = []
        for token in unexamined.tolist():
            if memo.get(token) and float(probs[token]) > 0.0:
                result.append(int(token))
        return result

    def _nucleus_boundary_stable(self, admitted_probs, tail_mass):
        """Whether top_p keeps the same admitted tokens for any tail mass.

        ``apply_top_p`` sorts ascending, accumulates, and keeps tokens whose
        cumulative mass exceeds ``(1 - top_p)`` of the total.  Admitted
        tokens all outrank the unexamined tail, so an admitted token's
        cumulative is its own ascending cumulative plus the admissible tail
        mass ``t`` in ``[0, tail_mass]``; it is kept iff
        ``c + t > (1 - top_p) * (S + t)``, which is monotone in ``t``.  The
        decision is therefore stable iff it agrees at ``t = 0`` and
        ``t = tail_mass``.
        """
        threshold = 1.0 - self.top_p
        total = sum(admitted_probs)
        cumulative = 0.0
        for probability in sorted(admitted_probs):
            cumulative += probability
            keep_low = cumulative > threshold * total
            keep_high = cumulative + tail_mass > threshold * (total + tail_mass)
            if keep_low != keep_high:
                return False
        return True

    def _constrained_ids(self, token_ids):
        """Generated ids after the deferral marker, or None while deferred."""
        marker = self._defer_until
        if marker is None:
            return token_ids
        # Rescan on every call: speculative verification can roll the ids
        # back, so a remembered activation point is not trustworthy.
        width = len(marker)
        position = 0
        while True:
            try:
                position = token_ids.index(marker[0], position)
            except ValueError:
                self.constraining = False
                self.deferred_tokens = len(token_ids) + 1
                return None
            if tuple(token_ids[position : position + width]) == marker:
                self.constraining = True
                self.deferred_tokens = position + width
                return token_ids[position + width :]
            position += 1

    def _automaton_config(self, token_ids):
        """Automaton configuration for ``token_ids``, advanced incrementally.

        The configuration is that of the decoded text, exactly the prefix the
        scanner would match (special tokens skipped by the decode contribute
        nothing).  Each step walks only the text the new tokens added; the
        whole output is re-walked only if the decode is not prefix-stable.
        """
        tracked = self._track_ids
        count = len(tracked)
        if len(token_ids) >= count and token_ids[:count] == tracked:
            common = count
        else:
            common = 0
            for ours, theirs in zip(tracked, token_ids):
                if ours != theirs:
                    break
                common += 1
        if common == len(token_ids) and self._track_configs[common] is not False:
            self._pending = self._track_pending[common]
            return self._track_configs[common]
        pending = b""
        if self._token_bytes is not None:
            from .structured_automaton import decode_token_bytes

            split = decode_token_bytes(self._token_bytes, token_ids)
            if split is None:  # bytes that can never be UTF-8: dead text
                split = ("", b"")
                dead = True
            else:
                dead = False
            text, pending = split
        else:
            dead = False
            text = self.tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        # ``False`` marks positions skipped by a multi-token step; position 0
        # is always known, and the early return above means base < len(ids).
        base = common
        while self._track_configs[base] is False:
            base -= 1
        length = self._track_lengths[base]
        config = self._track_configs[base]
        if dead:
            config = None
        elif text[:length] == self._track_text[:length]:
            if config is not None:  # a dead prefix stays dead
                config = self._automaton.advance(config, text[length:])
        else:
            config = self._automaton.advance(self._automaton.start, text)
        if config is not None and pending:
            from .structured_automaton import prefix_viable

            if not prefix_viable(self._automaton, config, pending):
                config = None
        if config is not None and len(config[1]) > _MAX_PUSHDOWN_DEPTH:
            # Every push copies the return stack and every position keeps its
            # configuration for rollback, so unbounded nesting is quadratic
            # memory on the single generation worker.  Fail this lane closed.
            raise ValueError(
                f"structured output nests deeper than {_MAX_PUSHDOWN_DEPTH} levels"
            )
        skipped = len(token_ids) - base - 1
        self._track_ids = list(token_ids)
        self._track_configs = self._track_configs[: base + 1] + [False] * skipped + [config]
        self._track_lengths = self._track_lengths[: base + 1] + [0] * skipped + [len(text)]
        self._track_pending = self._track_pending[: base + 1] + [b""] * skipped + [pending]
        self._pending = pending
        self._track_text = text
        # Rollbacks are bounded by the speculative verify window; older
        # positions only need to be re-derivable, not retained.
        stale = len(self._track_configs) - 1 - _TRACK_HISTORY
        first = min(getattr(self, "_track_thinned", 1), max(stale, 1))
        for index in range(first, stale + 1):
            self._track_configs[index] = False
            self._track_lengths[index] = 0
        self._track_thinned = max(first, stale + 1)
        return config

    def _automaton_allowed(self, token_ids):
        """Exact boolean admissibility per token id for the automaton engine."""
        from .structured_automaton import NO_CONTINUATION

        config = self._automaton_config(token_ids)
        self._last_automaton_config = config
        if config is None:
            raise ValueError(NO_CONTINUATION)
        pending = getattr(self, "_pending", b"")
        if pending:
            # Inside a character only its continuation can follow: every
            # whole-text piece would start a new one.
            import numpy as np

            allowed = np.zeros(self._trie.vocab_size, dtype=bool)
        else:
            allowed = self._trie.allowed(self._automaton, config)
        if self._fragments is not None:
            allowed[self._fragments.allowed_ids(self._automaton, config, pending)] = True
        if self.eos_ids and not pending and self._automaton.is_accepting(config):
            # Same EOS rule as the scanner: admissible iff the text is a
            # complete match.  EOS ids can sit past the tokenizer vocabulary.
            import numpy as np

            top = max(self.eos_ids) + 1
            if top > allowed.shape[0]:
                allowed = np.concatenate((allowed, np.zeros(top - allowed.shape[0], dtype=bool)))
            allowed[list(self.eos_ids)] = True
        if not allowed.any():
            raise ValueError(NO_CONTINUATION)
        return allowed

    def _terminal_only(self, token_ids, logits):
        """Keep only in-vocabulary generation stops, or latch a lane failure."""
        import mlx.core as mx

        width = logits.shape[-1]
        terminal_ids = [
            token for token in self.eos_ids if 0 <= token < width
        ]
        if not terminal_ids:
            self._latch_failure(
                ValueError(
                    "structured-output generation stop tokens are outside "
                    "the logits vocabulary"
                ),
                token_ids,
                logits,
            )
            return logits
        keep = mx.zeros(width, dtype=mx.bool_)
        keep = mx.put_along_axis(
            keep,
            mx.array(terminal_ids),
            mx.ones((len(terminal_ids),), dtype=mx.bool_),
            axis=-1,
        )
        return mx.where(
            keep, logits, mx.array(-float("inf"), dtype=logits.dtype)
        )

    def __call__(self, tokens, logits):
        import mlx.core as mx

        if self.failure is not None:
            return logits
        token_ids = [int(item) for item in tokens.tolist()][self.prompt_length :]
        self._generated_token_count = len(token_ids)
        self._recent_generated_ids = tuple(token_ids[-_FAILURE_HISTORY_TOKENS:])
        token_ids = self._constrained_ids(token_ids)
        if token_ids is None:
            if not self.block_eos_while_deferred:
                return logits  # deferred: no grammar is active yet
            mask = mx.zeros(logits.shape[-1], dtype=logits.dtype)
            eos = [token for token in self.eos_ids if 0 <= token < logits.shape[-1]]
            if eos:
                mask = mx.put_along_axis(
                    mask,
                    mx.array(eos),
                    mx.full((len(eos),), -float("inf"), dtype=logits.dtype),
                    axis=-1,
                )
            return logits + mask
        terminal_positions = [
            index for index, token in enumerate(token_ids) if token in self.eos_ids
        ]
        if terminal_positions:
            # GenerationBatch evaluates processors for the unused next-token
            # row before it applies StopSequenceMatcher to the current token.
            # A terminal sampled from an accepting state is therefore visible
            # here once. Treat it as framing, not grammar text; any terminal
            # at a non-accepting state (or followed by a nonterminal) fails
            # closed. This uses the exact same frozen ids as stop matching.
            first_terminal = terminal_positions[0]
            before_terminal = token_ids[:first_terminal]
            accepted = terminal_positions == list(
                range(first_terminal, len(token_ids))
            )
            if accepted and self._envelope is not None:
                opens, closes = self._envelope
                closed = (
                    bool(closes)
                    and len(before_terminal) >= len(closes)
                    and tuple(before_terminal[-len(closes) :]) == closes
                )
                if closed:
                    before_terminal = before_terminal[: -len(closes)]
                elif any(token in closes for token in before_terminal):
                    accepted = False
                before_terminal = [
                    token for token in before_terminal if token not in opens
                ]
            if accepted:
                accepted = self._is_complete(before_terminal)
            if not accepted:
                from .structured_automaton import NO_CONTINUATION

                self._latch_failure(
                    ValueError(NO_CONTINUATION), token_ids, logits
                )
                return logits
            # Speculative proposers may ask for more than one token after the
            # first terminal before the generation loop observes the stop.
            # Keep that provisional tail terminal-only; returning raw logits
            # would let invalid grammar text escape into the draft block.
            self.constrained_steps += 1
            return self._terminal_only(token_ids, logits)
        if self._envelope is None:
            return self._mask(token_ids, logits)
        # Channel envelope (e.g. <|START_TEXT|> ... <|END_TEXT|>): the model's
        # trained framing around an answer.  The markers carry no answer text,
        # so they are kept out of the grammar's view and admitted only where
        # the framing allows them: one opener before any answer text, a closer
        # once the answer is complete, and nothing but end-of-turn after it.
        import numpy as np

        opens, closes = self._envelope
        width = logits.shape[-1]
        keep = np.zeros(width, dtype=bool)
        if any(token in closes for token in token_ids):
            return self._terminal_only(token_ids, logits)
        text_ids = [token for token in token_ids if token not in opens]
        masked = self._mask(text_ids, logits)
        if self.failure is not None:
            return masked
        # The grammar sees envelope-stripped text, so on its own it would admit
        # a marker token wherever the marker's text is grammatical -- inside
        # any JSON string, since North's markers are ordinary added tokens.  A
        # closer sampled there leaves only end-of-turn, which the terminal
        # check then rejects.  Block each marker wherever the framing does not
        # allow it, whatever the grammar says about its text.
        block = np.zeros(width, dtype=bool)
        open_ids = [token for token in opens if token < width]
        close_ids = [token for token in closes if token < width]
        if not token_ids:
            keep[open_ids] = True
        else:
            block[open_ids] = True
        if self._is_complete(text_ids):
            keep[close_ids] = True
        else:
            block[close_ids] = True
        if keep.any():
            masked = mx.where(mx.array(keep), logits, masked)
        if block.any():
            masked = mx.where(
                mx.array(block), mx.array(-float("inf"), dtype=logits.dtype), masked
            )
        return masked

    def _is_complete(self, token_ids):
        """Whether the constrained text so far is a full match."""
        if self._automaton is not None:
            config = self._automaton_config(token_ids)
            return not getattr(self, "_pending", b"") and self._automaton.is_accepting(config)
        prefix = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        compiled = self.constraint.pattern if isinstance(self.constraint, _Constraint) else self.constraint
        try:
            return compiled.fullmatch(prefix, timeout=_MATCH_TIMEOUT_SECONDS) is not None
        except TimeoutError:
            return False

    def _mask(self, token_ids, logits):
        import mlx.core as mx
        if self._automaton is not None:
            import numpy as np

            try:
                allowed = self._automaton_allowed(token_ids)
            except Exception as exc:  # noqa: BLE001 - this lane fails closed, never the batch
                self._latch_failure(exc, token_ids, logits)
                return logits
            width = logits.shape[-1]
            if allowed.shape[0] < width:
                allowed = np.concatenate(
                    (allowed, np.zeros(width - allowed.shape[0], dtype=bool))
                )
            elif allowed.shape[0] > width:
                allowed = allowed[:width]
            self.constrained_steps += 1
            return mx.where(
                mx.array(allowed), logits, mx.array(-float("inf"), dtype=logits.dtype)
            )
        prefix = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Sliding window: the canonical prefix has the same grammar state, is
        # shorter for the regex, and is shared by every position inside the
        # same free string value, so their admissibility tables coincide.
        prefix = self.constraint.canonicalize(prefix)
        try:
            cached = self._allowed_cache.get(prefix)
            if cached is not None:
                allowed = cached
            elif logits.ndim == 1 or logits.shape[0] == 1:
                allowed = self._allowed_by_logit_order(prefix, logits.reshape(-1))
            else:
                # Several rows share one prefix: the mask must be exact for
                # every row, so take the full admissible set.
                allowed = self._allowed(prefix)
        except Exception as exc:  # noqa: BLE001 - this lane fails closed, never the batch
            self._latch_failure(exc, token_ids, logits, prefix=prefix)
            return logits
        # The piece table spans the tokenizer id space, which a model whose
        # head is narrower than its tokenizer would not cover; an id past the
        # row cannot be sampled and must not be written into the mask either.
        width = logits.shape[-1]
        allowed = tuple(token for token in allowed if 0 <= token < width)
        if not allowed:
            self._latch_failure(
                ValueError(
                    "structured-output grammar admits no token inside the "
                    "logits vocabulary"
                ),
                token_ids,
                logits,
                prefix=prefix,
            )
            return logits
        self.constrained_steps += 1
        mask = mx.full(width, -float("inf"), dtype=logits.dtype)
        indexes = mx.array(allowed)
        mask = mx.put_along_axis(mask, indexes, mx.zeros(indexes.shape, dtype=logits.dtype), axis=-1)
        return logits + mask

    def _bounded_token(self, token_id):
        """Return a small, JSON-safe description of one vocabulary token."""
        from .structured_automaton import token_bytes_value

        token_id = int(token_id)
        decoded = self._pieces[token_id] if 0 <= token_id < len(self._pieces) else ""
        convert = getattr(self.tokenizer, "convert_ids_to_tokens", None)
        name = None
        if callable(convert):
            try:
                name = convert(token_id)
                if isinstance(name, (list, tuple)):
                    name = name[0] if name else None
            except (TypeError, ValueError, IndexError):
                try:
                    names = convert([token_id])
                    name = names[0] if names else None
                except (TypeError, ValueError, IndexError):
                    name = None
        if not isinstance(name, str):
            name = decoded
        try:
            raw = token_bytes_value(self.tokenizer, token_id)
        except (AttributeError, TypeError, ValueError, IndexError):
            raw = decoded.encode("utf-8", errors="replace")
        raw = bytes(raw)
        return {
            "id": token_id,
            "tokenizer_piece": name[:_FAILURE_TEXT_CHARS],
            "decoded_piece": decoded[:_FAILURE_TEXT_CHARS],
            "bytes_hex": raw[:_FAILURE_BYTES].hex(),
            "bytes_truncated": len(raw) > _FAILURE_BYTES,
        }

    def _latch_failure(self, exc, token_ids, logits, *, prefix=None):
        """Latch a failure and optional qualification-only dead-end evidence."""
        from .structured_automaton import NO_CONTINUATION

        self.failure = str(exc)
        if not self.capture_failure_context or self.failure != NO_CONTINUATION:
            return
        try:
            self.failure_context = self._dead_end_context(
                token_ids, logits, prefix=prefix
            )
        except Exception as diagnostic_error:  # noqa: BLE001
            # Diagnostic extraction must never turn one lane failure into a
            # generation-worker failure for every lane in the batch.
            self.failure_context = {
                "schema": "mlx2.structured-output-dead-end.v1",
                "constraint_kind": self.constraint_kind,
                "engine": self.engine,
                "generated_tokens": int(
                    getattr(self, "_generated_token_count", len(token_ids))
                ),
                "recent_token_ids": [
                    int(item)
                    for item in getattr(
                        self,
                        "_recent_generated_ids",
                        tuple(token_ids)[-_FAILURE_HISTORY_TOKENS:],
                    )
                ],
                "diagnostic_error": str(diagnostic_error)[:_FAILURE_TEXT_CHARS],
            }

    def _dead_end_context(self, token_ids, logits, *, prefix=None):
        """Build bounded qualification evidence without changing grammar state."""
        import mlx.core as mx
        import numpy as np

        row = np.asarray(logits.astype(mx.float32)).reshape(-1, logits.shape[-1])[-1]
        top = np.argsort(-row, kind="stable")[:_FAILURE_TOP_LOGITS]
        top_logits = []
        for token_id in top.tolist():
            item = self._bounded_token(token_id)
            value = float(row[token_id])
            item["logit"] = value if np.isfinite(value) else None
            top_logits.append(item)
        state = None
        if self.engine == "automaton":
            config = self._last_automaton_config
            if config is None:
                state = {"state": None, "stack_depth": 0, "stack_tail": []}
            else:
                automaton_state, stack = config
                state = {
                    "state": int(automaton_state),
                    "stack_depth": len(stack),
                    "stack_tail": [int(item) for item in stack[-16:]],
                }
            pending = bytes(getattr(self, "_pending", b""))
            state["pending_bytes_hex"] = pending[:_FAILURE_BYTES].hex()
            state["pending_bytes_truncated"] = len(pending) > _FAILURE_BYTES
        elif prefix is not None:
            state = {
                "canonical_prefix_chars": len(prefix),
                "canonical_prefix_tail": prefix[-128:],
            }
        history = getattr(self, "_recent_generated_ids", tuple(token_ids)[-_FAILURE_HISTORY_TOKENS:])
        return {
            "schema": "mlx2.structured-output-dead-end.v1",
            "constraint_kind": self.constraint_kind,
            "engine": self.engine,
            "generated_tokens": int(getattr(self, "_generated_token_count", len(token_ids))),
            "recent_tokens": [self._bounded_token(token_id) for token_id in history],
            "automaton_state": state,
            "top_logits": top_logits,
        }


def make_structured_processor(
    tokenizer,
    prompt_length,
    *,
    response_format=None,
    grammar=None,
    greedy=False,
    top_k=0,
    top_p=0.0,
    defer_until=None,
    envelope=None,
    block_eos_while_deferred=False,
    constraint_kind=None,
    capture_failure_context=False,
    generation_stop_token_ids=None,
    server_grammar=None,
):
    if server_grammar is not None:
        # A server-composed pattern (item 12 tool grammars): not client input,
        # so the client grammar length cap does not apply; it is used verbatim.
        try:
            constraint = _Constraint(
                regex.compile(rf"(?:{server_grammar})"), kind="tool_grammar"
            )
        except regex.error as exc:
            raise ValueError(f"invalid server tool grammar: {exc}") from exc
    else:
        constraint = compile_constraint(
            response_format, grammar, leading_whitespace=bool(defer_until)
        )
    if constraint is None:
        return None
    return StructuredOutputProcessor(
        tokenizer,
        prompt_length,
        constraint,
        greedy=greedy,
        top_k=top_k,
        top_p=top_p,
        defer_until=defer_until,
        envelope=envelope,
        block_eos_while_deferred=block_eos_while_deferred,
        constraint_kind=constraint_kind,
        capture_failure_context=capture_failure_context,
        generation_stop_token_ids=generation_stop_token_ids,
    )


def structured_receipt(processor, completion_tokens=0):
    """Receipt fields describing how a request's constraint was enforced."""
    deferred = bool(getattr(processor, "deferred", False))
    if deferred and not getattr(processor, "constraining", True):
        # The marker never appeared: every generated token was unconstrained.
        deferred_tokens = int(completion_tokens)
    else:
        deferred_tokens = int(getattr(processor, "deferred_tokens", 0)) if deferred else 0
    return {
        "engine": getattr(processor, "engine", "scanner"),
        "tail_mass_bound": getattr(processor, "tail_mass_bound", 0.0),
        "parallel_scans": getattr(processor, "parallel_scans", 0),
        "deferred": deferred,
        "deferred_tokens": deferred_tokens,
    }
