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

from typing import Any, Optional

_UNION_KEYS = ("anyOf", "oneOf", "allOf")


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
