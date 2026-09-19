"""Fail-closed regex and JSON-schema token constraints for batched decoding."""

from __future__ import annotations

import json
from collections import OrderedDict

import regex


_WS = r"[\x20\x09\x0a\x0d]*"
_STRING = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_MATCH_TIMEOUT_SECONDS = 0.01
_MAX_PREFIX_CACHE = 256
_TOKEN_PIECES_ATTRIBUTE = "_mlx2_structured_token_pieces_v1"


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


def _schema_pattern(schema, depth=0):
    if depth > 16 or not isinstance(schema, dict):
        raise ValueError("JSON schema must be an object with nesting depth at most 16")
    if "enum" in schema:
        _require_schema_keys(schema, {"enum"})
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum or len(enum) > 256:
            raise ValueError("JSON schema enum must contain 1 to 256 values")
        return "(?:" + "|".join(regex.escape(_json_literal(v)) for v in enum) + ")"
    if "const" in schema:
        _require_schema_keys(schema, {"const"})
        return regex.escape(_json_literal(schema["const"]))
    kind = schema.get("type")
    if isinstance(kind, list):
        _require_schema_keys(schema, {"type"})
        if not kind or len(kind) > 8:
            raise ValueError("JSON schema type union is invalid")
        return "(?:" + "|".join(_schema_pattern({**schema, "type": item}, depth + 1) for item in kind) + ")"
    if kind == "string":
        _require_schema_keys(schema, {"type"})
        return _STRING
    if kind in ("number", "integer"):
        _require_schema_keys(schema, {"type"})
        return _NUMBER if kind == "number" else r"-?(?:0|[1-9][0-9]*)"
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
        item = _schema_pattern(schema.get("items", {}), depth + 1)
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
            value = _schema_pattern(subschema, depth + 1)
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
        variants = []
        for mask in range(1 << len(optional_parts)):
            chosen = required_parts + [
                part for index, part in enumerate(optional_parts) if mask & (1 << index)
            ]
            variants.append(f"{_WS},{_WS}".join(chosen))
        body = "(?:" + "|".join(variants) + ")" if len(variants) > 1 else variants[0]
        return rf"\{{{_WS}{body}{_WS}\}}"
    raise ValueError(f"unsupported JSON schema type {kind!r}")


def compile_constraint(response_format=None, grammar=None):
    if grammar is not None:
        if not isinstance(grammar, str) or not grammar or len(grammar) > 4096:
            raise ValueError("grammar must be a nonempty regex of at most 4096 characters")
        try:
            return regex.compile(rf"(?:{grammar})")
        except regex.error as exc:
            raise ValueError(f"invalid grammar regex: {exc}") from exc
    if response_format is None or response_format == {"type": "text"}:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "json_object" and set(response_format) == {"type"}:
        # Recursive JSON value grammar, narrowed to an object at the root.
        pattern = rf"(?&object)(?(DEFINE)(?P<value>{_WS}(?:{_STRING}|{_NUMBER}|true|false|null|(?&object)|(?&array)){_WS})(?P<object>\{{{_WS}(?:{_STRING}{_WS}:{_WS}(?&value)(?:,{_WS}{_STRING}{_WS}:{_WS}(?&value))*)?{_WS}\}})(?P<array>\[{_WS}(?:(?&value)(?:,{_WS}(?&value))*)?{_WS}\]))"
        return regex.compile(pattern)
    if kind == "json_schema" and set(response_format) <= {"type", "json_schema"}:
        wrapper = response_format.get("json_schema")
        if not isinstance(wrapper, dict) or set(wrapper) - {"name", "description", "schema", "strict"}:
            raise ValueError("json_schema wrapper is invalid")
        if wrapper.get("strict", True) is not True or not isinstance(wrapper.get("schema"), dict):
            raise ValueError("json_schema requires strict:true and a schema object")
        return regex.compile(_schema_pattern(wrapper["schema"]))
    raise ValueError("response_format supports text, json_object, or strict json_schema")


class StructuredOutputProcessor:
    def __init__(self, tokenizer, prompt_length, constraint):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.constraint = constraint
        self.eos_ids = frozenset(tokenizer.eos_token_ids)
        self.vocab_size = int(tokenizer.vocab_size)
        pieces = getattr(tokenizer, _TOKEN_PIECES_ATTRIBUTE, None)
        if not isinstance(pieces, tuple) or len(pieces) != self.vocab_size:
            pieces = tuple(
                tokenizer.decode(
                    [token],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for token in range(self.vocab_size)
            )
            try:
                setattr(tokenizer, _TOKEN_PIECES_ATTRIBUTE, pieces)
            except (AttributeError, TypeError):
                pass
        self._pieces = pieces
        self._allowed_cache = OrderedDict()

    def _allowed(self, prefix):
        cached = self._allowed_cache.get(prefix)
        if cached is not None:
            self._allowed_cache.move_to_end(prefix)
            return cached
        try:
            complete = (
                self.constraint.fullmatch(prefix, timeout=_MATCH_TIMEOUT_SECONDS)
                is not None
            )
        except TimeoutError as exc:
            raise ValueError("structured-output grammar exceeded its match budget") from exc
        allowed = set(self.eos_ids if complete else ())
        for token, piece in enumerate(self._pieces):
            if token in self.eos_ids or not piece or "\ufffd" in piece:
                continue
            try:
                match = self.constraint.fullmatch(
                    prefix + piece,
                    partial=True,
                    timeout=_MATCH_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                # A single adversarial vocabulary piece must not terminate the
                # generation worker.  Excluding that piece is conservative:
                # it can only narrow the accepted language, while allowing it
                # after an inconclusive match would violate fail-closed
                # structured output.  The complete prefix check above remains
                # fatal because it determines whether EOS may be admitted.
                continue
            if match is not None:
                allowed.add(token)
        if not allowed:
            raise ValueError("structured-output grammar has no valid token continuation")
        result = tuple(sorted(allowed))
        self._allowed_cache[prefix] = result
        if len(self._allowed_cache) > _MAX_PREFIX_CACHE:
            self._allowed_cache.popitem(last=False)
        return result

    def __call__(self, tokens, logits):
        import mlx.core as mx

        token_ids = [int(item) for item in tokens.tolist()][self.prompt_length :]
        prefix = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        allowed = self._allowed(prefix)
        mask = mx.full(logits.shape[-1], -float("inf"), dtype=logits.dtype)
        indexes = mx.array(allowed)
        mask = mx.put_along_axis(mask, indexes, mx.zeros(indexes.shape, dtype=logits.dtype), axis=-1)
        return logits + mask


def make_structured_processor(tokenizer, prompt_length, *, response_format=None, grammar=None):
    constraint = compile_constraint(response_format, grammar)
    return None if constraint is None else StructuredOutputProcessor(tokenizer, prompt_length, constraint)
