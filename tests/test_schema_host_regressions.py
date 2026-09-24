"""Strict tool schema checks that require no tensor runtime."""
import pytest

from mlx2.runtime.tool_parsers._schema import resolve_local_refs, schema_value_matches
from mlx2.structured_output import compile_constraint


@pytest.mark.parametrize("value, schema", [
    (True, {"enum": [1]}),
    (1, {"const": True}),
    ({"x": [False]}, {"const": {"x": [0]}}),
    ([True], {"enum": [[1]]}),
])
def test_json_literal_equality_does_not_equate_booleans_and_numbers(value, schema):
    assert not schema_value_matches(value, schema)
    assert schema_value_matches(1.0, {"enum": [1]})


@pytest.mark.parametrize("required", [[{}], [[]], [1], [None]])
def test_malformed_required_schema_raises_validation_error(required):
    with pytest.raises(ValueError, match="required"):
        compile_constraint({"type": "json_schema", "json_schema": {"schema": {
            "type": "object", "properties": {"x": {"type": "string"}}, "required": required,
        }}})


def test_excessive_schema_depth_fails_before_recursive_traversal():
    schema = {"type": "string"}
    for _ in range(2000):
        schema = {"type": "array", "items": schema}
    with pytest.raises(ValueError, match="depth"):
        resolve_local_refs(schema)


@pytest.mark.parametrize("declared", [None, {}, ["string", {}], "unknown"])
def test_invalid_declared_types_do_not_pass_literal_validation(declared):
    assert not schema_value_matches(1, {"type": declared})
    with pytest.raises(ValueError):
        compile_constraint({"type": "json_schema", "json_schema": {"schema": {
            "type": declared, "enum": [1],
        }}})
