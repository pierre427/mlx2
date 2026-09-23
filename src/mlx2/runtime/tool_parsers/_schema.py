# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; original notices in provenance/NOTICE.
"""Shared JSON-schema helpers for tool-call parsers.

Model tool schemas frequently express optional / union-typed parameters with
``anyOf`` / ``oneOf`` or a list-form ``type`` (e.g. ``["string", "null"]``).
A naive ``schema["type"]`` lookup misses these shapes, so parsers either default
optional params to "string" (deserializing structured values as raw text) or
fall through to an unguarded ``ast.literal_eval`` that can raise ``SyntaxError``.

``infer_type_from_json_schema`` resolves any of those shapes to a single
concrete, non-null type name, preferring the first non-"null" branch. It returns
``None`` when no concrete type can be determined; callers should then treat the
value as a raw string rather than raising.
"""

import copy
import json
import math
from typing import Any, Optional

_UNION_KEYS = ("anyOf", "oneOf", "allOf")
_MAX_REF_DEPTH = 16
_MAX_RESOLVED_NODES = 4096
_ANNOTATION_KEYS = {
    "$comment",
    "default",
    "deprecated",
    "description",
    "example",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
}
_SCHEMA_MAP_KEYS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
_SCHEMA_LIST_KEYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_VALUE_KEYS = {
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
}


class SchemaReferenceError(ValueError):
    """A JSON-schema reference cannot be resolved within mlx2's safe subset."""

    schema_reference_error = True


def _pointer_parts(reference: str) -> tuple[str, ...]:
    if not reference.startswith("#/"):
        raise SchemaReferenceError("JSON schema references must be local #/$defs or #/definitions pointers")
    encoded = reference[2:].split("/")
    if not encoded or encoded[0] not in {"$defs", "definitions"}:
        raise SchemaReferenceError("JSON schema references must target #/$defs or #/definitions")
    parts = []
    for item in encoded:
        output = []
        index = 0
        while index < len(item):
            if item[index] != "~":
                output.append(item[index])
                index += 1
                continue
            if index + 1 >= len(item) or item[index + 1] not in "01":
                raise SchemaReferenceError("JSON schema reference contains a malformed JSON pointer escape")
            output.append("~" if item[index + 1] == "0" else "/")
            index += 2
        parts.append("".join(output))
    return tuple(parts)


def resolve_local_refs(schema: Any) -> dict:
    """Inline bounded, non-recursive local ``$ref`` targets.

    Design reference: FreeToken#435.

    Only ``#/$defs/...`` and ``#/definitions/...`` JSON pointers are accepted.
    Remote references, constraining sibling keywords, malformed pointers,
    cycles and an expanded tree beyond the fixed depth/node budgets fail
    closed. Annotation siblings are ignored because they do not change the
    accepted instance language.

    Traversal is schema-position aware: ``properties`` and definition maps are
    containers whose keys are user names, not schema keywords. This preserves
    properties literally named ``$ref``, ``$defs`` or ``definitions``.
    """
    if not isinstance(schema, dict):
        raise SchemaReferenceError("JSON schema must be an object")
    root = schema
    nodes = 0

    def target(reference: str):
        current: Any = root
        for part in _pointer_parts(reference):
            if not isinstance(current, dict) or part not in current:
                raise SchemaReferenceError(f"JSON schema reference does not exist: {reference}")
            current = current[part]
        if not isinstance(current, dict):
            raise SchemaReferenceError("JSON schema references must resolve to schema objects")
        return current

    def too_large(message: str, *, through_ref: bool):
        if through_ref:
            raise SchemaReferenceError(message)
        raise ValueError(message)

    def visit(value: Any, depth: int, active: tuple[str, ...], ref_depth: int):
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_RESOLVED_NODES:
            too_large(
                f"resolved JSON schema exceeds {_MAX_RESOLVED_NODES} schema nodes",
                through_ref=bool(active),
            )
        if ref_depth > _MAX_REF_DEPTH:
            raise SchemaReferenceError(
                f"JSON schema reference expansion exceeds depth {_MAX_REF_DEPTH}"
            )
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        if "$ref" in value:
            siblings = set(value) - {"$ref", "$defs", "definitions"}
            constraining = siblings - _ANNOTATION_KEYS
            if constraining or not isinstance(value["$ref"], str):
                raise SchemaReferenceError(
                    "JSON schema $ref may only have annotation siblings and must contain a string"
                )
            reference = value["$ref"]
            if reference in active:
                raise SchemaReferenceError("recursive JSON schema references are not supported")
            resolved = visit(
                target(reference), depth, (*active, reference), ref_depth + 1
            )
            # Annotation siblings do not constrain instances. Drop them from
            # the executable schema so the bounded strict compiler need not
            # grow a second annotation vocabulary.
            return resolved

        result = {}
        for key, item in value.items():
            if key in {"$defs", "definitions"}:
                continue
            if key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
                result[key] = {
                    name: visit(child, depth + 1, active, ref_depth)
                    for name, child in item.items()
                }
            elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
                result[key] = [
                    visit(child, depth + 1, active, ref_depth) for child in item
                ]
            elif key in _SCHEMA_VALUE_KEYS and isinstance(item, dict):
                result[key] = visit(item, depth + 1, active, ref_depth)
            else:
                result[key] = copy.deepcopy(item)
        # Do not turn the pre-existing schema nesting bound into a reference
        # failure for schemas that contain no reference at all. Once a target
        # has been expanded, however, its schema nesting is bounded at the same
        # sixteen levels as the strict compiler.
        if depth > _MAX_REF_DEPTH and active:
            raise SchemaReferenceError(
                f"JSON schema reference expansion exceeds schema depth {_MAX_REF_DEPTH}"
            )
        return result

    resolved = visit(root, 0, (), 0)
    if not isinstance(resolved, dict):  # defensive: the root was checked above
        raise SchemaReferenceError("resolved JSON schema must be an object")
    return resolved


def strip_annotations(schema: Any) -> Any:
    """Copy ``schema`` without its annotation keywords.

    ``title``, ``description``, ``examples``, ``default`` and the rest of
    ``_ANNOTATION_KEYS`` never change which instances a schema accepts, but
    Pydantic (and so the OpenAI SDK's ``.parse()``) puts ``title`` on the root
    and on every property.  The bounded compilers reject every keyword they do
    not implement, so they are handed the executable schema only; constraining
    keywords they do not implement still fail closed.  The request keeps the
    annotated schema for prompt rendering.

    Traversal is schema-position aware, as in :func:`resolve_local_refs`: the
    keys of ``properties`` and definition maps are user names, so a property
    literally named ``title`` or ``description`` is kept, and ``enum``,
    ``const`` and ``required`` values are copied verbatim.
    """
    if not isinstance(schema, dict):
        return copy.deepcopy(schema)
    result = {}
    for key, item in schema.items():
        if key in _ANNOTATION_KEYS:
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
            result[key] = {name: strip_annotations(child) for name, child in item.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
            result[key] = [strip_annotations(child) for child in item]
        elif key in _SCHEMA_VALUE_KEYS and isinstance(item, dict):
            result[key] = strip_annotations(item)
        else:
            result[key] = copy.deepcopy(item)
    return result


def executable_schema(schema: Any) -> dict:
    """The strict compilers' schema: local references inlined, annotations dropped."""
    return strip_annotations(resolve_local_refs(schema))


def infer_type_from_json_schema(schema: Any) -> Optional[str]:
    """Resolve a JSON-schema fragment to one concrete, non-null type name.

    Handles ``anyOf`` / ``oneOf`` / ``allOf`` unions and list-form ``type``
    (e.g. ``{"type": ["string", "null"]}``), preferring the first non-"null"
    branch. Returns ``None`` when no concrete type can be determined.
    """
    if not isinstance(schema, dict):
        return None

    declared = schema.get("type")
    if isinstance(declared, str) and declared.strip().lower() != "null":
        return declared
    if isinstance(declared, (list, tuple)):
        for item in declared:
            if isinstance(item, str) and item.strip().lower() != "null":
                return item

    for key in _UNION_KEYS:
        branches = schema.get(key)
        if isinstance(branches, (list, tuple)):
            for branch in branches:
                resolved = infer_type_from_json_schema(branch)
                if resolved is not None:
                    return resolved

    return None


def is_string_type(schema: Any) -> bool:
    """True when a schema fragment resolves to a JSON ``string`` type.

    Matches the exact JSON-schema type name ``string`` (as the parsers did
    before), but first resolves ``anyOf`` / ``oneOf`` / list-form unions so that
    e.g. ``{"type": ["string", "null"]}`` is recognized.
    """
    resolved = infer_type_from_json_schema(schema)
    return isinstance(resolved, str) and resolved.strip().lower() == "string"


def string_length_bounds(schema: dict, *, wire_max: int = 4096) -> tuple[int, int]:
    """Return bounded raw-string lengths or reject unsupported constraints."""
    minimum = schema.get("minLength", 0)
    maximum = schema.get("maxLength", wire_max)
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 0
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < minimum
    ):
        raise ValueError("string minLength/maxLength must be ordered non-negative integers")
    if minimum > wire_max:
        raise ValueError(f"string minLength exceeds the {wire_max}-character wire bound")
    return minimum, min(maximum, wire_max)


def raw_string_pattern(schema: Any, *, forbidden: str = "<", wire_max: int = 4096):
    """Regex for schemas that can use an adapter's unquoted text field.

    ``None`` means the schema needs JSON spelling (for example a union of
    string and null). Unsupported string constraints raise so strict tool
    grammars fail closed instead of silently widening their language.
    """
    if not isinstance(schema, dict):
        return None
    import regex

    if "pattern" in schema:
        raise ValueError("string pattern is unsupported by raw tool parameters")
    if "enum" in schema:
        unknown = set(schema) - {"type", "enum"}
        values = schema.get("enum")
        if unknown or schema.get("type", "string") != "string" or not isinstance(values, list):
            return None
        if not values or any(not isinstance(value, str) for value in values):
            return None
        if any(len(value) > wire_max for value in values):
            raise ValueError(f"string enum exceeds the {wire_max}-character wire bound")
        if any(any(char in value for char in forbidden) for value in values):
            raise ValueError("string enum contains a tool-wire delimiter")
        return "(?:" + "|".join(regex.escape(value) for value in values) + ")"
    if "const" in schema:
        unknown = set(schema) - {"type", "const"}
        value = schema.get("const")
        if unknown or schema.get("type", "string") != "string" or not isinstance(value, str):
            return None
        if len(value) > wire_max:
            raise ValueError(f"string const exceeds the {wire_max}-character wire bound")
        if any(char in value for char in forbidden):
            raise ValueError("string const contains a tool-wire delimiter")
        return regex.escape(value)
    if schema.get("type") != "string":
        return None
    unknown = set(schema) - {"type", "minLength", "maxLength"}
    if unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"unsupported raw string schema keywords: {names}")
    minimum, maximum = string_length_bounds(schema, wire_max=wire_max)
    excluded = regex.escape(forbidden)
    return rf"[^{excluded}]{{{minimum},{maximum}}}"


def schema_value_matches(value: Any, schema: Any) -> bool:
    """Validate values against the bounded schema subset used by tool wires."""
    if not isinstance(schema, dict):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if "const" in schema and value != schema["const"]:
        return False
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = schema[key]
            if not isinstance(branches, list) or not branches:
                return False
            matches = sum(schema_value_matches(value, branch) for branch in branches)
            return matches >= 1 if key == "anyOf" else matches == 1
    declared = schema.get("type")
    if isinstance(declared, list):
        return any(
            schema_value_matches(value, {**schema, "type": item})
            for item in declared
        )
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in {int, float},
        "boolean": lambda item: type(item) is bool,
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "null": lambda item: item is None,
    }
    if declared in checks and not checks[declared](value):
        return False
    if declared == "string":
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or minimum < 0
            or maximum is not None
            and (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < minimum
            )
        ):
            return False
        if len(value) < minimum or maximum is not None and len(value) > maximum:
            return False
    if declared == "array" and isinstance(value, list) and "items" in schema:
        if not all(schema_value_matches(item, schema["items"]) for item in value):
            return False
    if (declared == "object" or "properties" in schema) and isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", [])):
            return False
        if schema.get("additionalProperties") is False and any(
            name not in properties for name in value
        ):
            return False
        if any(
            name in properties and not schema_value_matches(item, properties[name])
            for name, item in value.items()
        ):
            return False
    return True


def json_native(value: Any) -> bool:
    """Whether tool-call arguments carry ``value`` as the model wrote it.

    Parsers decode a free argument best-effort, as JSON and then as a Python
    literal.  The arguments are serialized with ``allow_nan=False``, which
    rejects a non-finite float, a set or bytes, and silently rewrites others:
    a tuple becomes an array, and a key such as ``1``, ``True`` or ``None``
    becomes ``"1"``, ``"true"`` or ``"null"``.  Only JSON's own types, nested
    in lists and string-keyed dicts, serialize to what decodes back to them.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is dict:
            if any(type(key) is not str for key in item):
                return False
            pending.extend(item.values())
        elif kind is list:
            pending.extend(item)
        elif kind is float:
            if not math.isfinite(item):
                return False
        elif item is not None and kind not in (str, int, bool):
            return False
    try:
        # An integer past the interpreter's digit limit cannot be written.
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def required_parameter_names(function: dict) -> list[str]:
    """Required argument names of a tool, declared-property order first.

    Non-strict tool grammars leave argument values free but still have to
    make every required argument appear; without that, greedy decoding can
    close the call with no arguments at all (sglang #40051). Reference
    resolution is best-effort, as for non-strict parsing.
    """
    schema = function.get("parameters")
    if not isinstance(schema, dict):
        return []
    try:
        schema = resolve_local_refs(schema)
    except ValueError:
        pass
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    properties = schema.get("properties")
    declared = list(properties) if isinstance(properties, dict) else []
    names = [name for name in declared if name in required]
    for name in required:
        if name not in names:
            names.append(name)
    return names
