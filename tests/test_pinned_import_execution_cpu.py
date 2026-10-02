"""Host-only cached-module pin checks; no external imports or model loading."""

import hashlib
import importlib
import sys

import pytest

from mlx2.adapters.pinned_imports import PinnedSourceFinder


@pytest.fixture
def source(tmp_path, monkeypatch):
    name = "_mlx2_pinned_execution_probe"
    path = tmp_path / (name + ".py")
    path.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def finder(name, path):
    return PinnedSourceFinder(
        (name,), {path: hashlib.sha256(path.read_bytes()).hexdigest()}
    )


@pytest.mark.parametrize("rewrite", [False, True])
def test_same_origin_unpinned_cached_module_refused(source, rewrite):
    name, path = source
    module = importlib.import_module(name)
    if rewrite:
        path.write_text("VALUE = 2\n")
    with (
        finder(name, path) as gate,
        pytest.raises(ImportError, match="pinned execution"),
    ):
        gate.validate_loaded()
    assert sys.modules[name] is module
    assert module.VALUE == 1


def test_pinned_cached_module_reusable_under_identical_pin(source):
    name, path = source
    with finder(name, path) as gate:
        module = importlib.import_module(name)
        gate.validate_loaded()
    with finder(name, path) as gate:
        gate.validate_loaded()
        assert importlib.import_module(name) is module
    assert module.VALUE == 1


def test_previously_pinned_module_refuses_new_source_revision(source):
    name, path = source
    with finder(name, path) as gate:
        module = importlib.import_module(name)
        gate.validate_loaded()
    path.write_text("VALUE = 2\n")
    with (
        finder(name, path) as gate,
        pytest.raises(ImportError, match="pinned execution"),
    ):
        gate.validate_loaded()
    assert sys.modules[name] is module and module.VALUE == 1


def test_loader_pin_mutation_does_not_change_compiled_receipt(source):
    name, path = source
    with finder(name, path) as gate:
        module = importlib.import_module(name)
        gate.validate_loaded()
    path.write_text("VALUE = 2\n")
    module.__spec__.loader.expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with (
        finder(name, path) as gate,
        pytest.raises(ImportError, match="pinned execution"),
    ):
        gate.validate_loaded()
    assert module.VALUE == 1
