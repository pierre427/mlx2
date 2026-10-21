"""Trace accounting/restoration only. Never GPU attribution evidence."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def tracer():
    path = Path(__file__).resolve().parents[1] / "scripts/profile_flash_next_tree.py"
    spec = importlib.util.spec_from_file_location("flash_stage_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StageTrace(SimpleNamespace(array=bytes), lambda _: None)


def test_nested_stages_not_double_counted_and_hooks_restored():
    owner = SimpleNamespace(inner=lambda x: x + 1)
    owner.outer = lambda x: owner.inner(x) * 2
    original = owner.inner, owner.outer
    trace = tracer()
    with trace.hooks([(owner, "inner", "child"), (owner, "outer", "parent")]):
        assert owner.outer(2) == 6
    assert (owner.inner, owner.outer) == original
    assert not trace.stack
    assert trace.rows["parent"]["exclusive_seconds"] <= trace.rows["parent"]["inclusive_seconds"]
    assert trace.rows["child"]["calls"] == 1


def test_exception_restores_inherited_call_without_leaving_a_class_override():
    class Parent:
        def __call__(self):
            raise RuntimeError("injected")
    class Child(Parent):
        pass
    trace = tracer()
    with pytest.raises(RuntimeError, match="injected"), trace.hooks([(Child, "__call__", "failed")]):
        Child()()
    assert "__call__" not in vars(Child)
    assert not trace.stack
