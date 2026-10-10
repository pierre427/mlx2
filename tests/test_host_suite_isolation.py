"""A newly added host import guard must not poison the shared CPU suite."""

import ast
from pathlib import Path

from conftest import _ISOLATED_HOST_MODULES
from test_direct_run_source_contracts_cpu import _is_direct_only


def _collection_calls(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    if isinstance(node, ast.Call):
        yield ast.unparse(node.func)
    for child in ast.iter_child_nodes(node):
        yield from _collection_calls(child)


def test_collection_import_guards_have_an_isolated_runner():
    unisolated = []
    for path in Path(__file__).parent.glob("test_*.py"):
        if "sys.meta_path.insert" not in _collection_calls(ast.parse(path.read_text())):
            continue
        if path.name not in _ISOLATED_HOST_MODULES and not _is_direct_only(path):
            unisolated.append(path.name)
    assert not unisolated, f"collection-time import guards require isolation: {unisolated}"
