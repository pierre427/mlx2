import pytest

from mlx2.source_dependencies import (
    UnresolvedDependency,
    digest_files,
    file_digest,
    source_closure,
)


def package(tmp_path):
    root = tmp_path / "sample"
    root.mkdir()
    (root / "__init__.py").write_text("")
    # Explicit source keeps this fixture independent of import-time execution.
    (root / "entry.py").write_text("def run():\n    from .cache import state\n")
    (root / "cache.py").write_text("state = 1\n")
    (root / "unused.py").write_text("value = 1\n")
    return root


def test_lazy_imports_and_parent_initializers_are_bound(tmp_path):
    root = package(tmp_path)
    files = source_closure(root, ["sample.entry"])
    assert set(files) == {"__init__.py", "entry.py", "cache.py"}
    old = digest_files(files)
    (root / "unused.py").write_text("value = 2\n")
    assert digest_files(source_closure(root, ["sample.entry"])) == old
    (root / "cache.py").write_text("state = 2\n")
    assert digest_files(source_closure(root, ["sample.entry"])) != old


def test_unknown_dynamic_import_fails_closed_and_review_is_byte_bound(tmp_path):
    root = package(tmp_path)
    path = root / "entry.py"
    path.write_text(
        "import importlib\ndef run(name):\n    return importlib.import_module(name)\n"
    )
    with pytest.raises(UnresolvedDependency, match="unresolved dynamic"):
        source_closure(root, ["sample.entry"])
    review = {
        "sample.entry": {"sha256": file_digest(path), "targets": ["sample.cache"]}
    }
    assert "cache.py" in source_closure(root, ["sample.entry"], reviewed_dynamic=review)
    path.write_text(path.read_text() + "# changed dispatcher\n")
    with pytest.raises(UnresolvedDependency, match="dispatcher changed"):
        source_closure(root, ["sample.entry"], reviewed_dynamic=review)


def test_literal_relative_dynamic_import_and_missing_target(tmp_path):
    root = package(tmp_path)
    path = root / "entry.py"
    path.write_text(
        "import importlib\nimportlib.import_module('.cache', __package__)\n"
    )
    assert "cache.py" in source_closure(root, ["sample.entry"])
    (root / "cache.py").unlink()
    with pytest.raises(UnresolvedDependency, match="missing local"):
        source_closure(root, ["sample.entry"])


@pytest.mark.parametrize(
    "code",
    [
        "from importlib.util import spec_from_file_location as load; load('hidden', '/tmp/hidden.py')",
        "from importlib import import_module as load; alias = load; alias('sample.cache')",
        "import importlib; alias = importlib.import_module; alias('sample.cache')",
        "import importlib; getattr(importlib, 'import_module')('sample.cache')",
        "from builtins import eval as execute; execute('1')",
        "import builtins; builtins.__import__('sample', fromlist=['cache'])",
    ],
)
def test_obscured_loaders_fail_closed(tmp_path, code):
    root = package(tmp_path)
    (root / "entry.py").write_text(code)
    with pytest.raises(UnresolvedDependency):
        source_closure(root, ["sample.entry"])


def test_literal_loader_alias_and_keyword_package_are_resolved(tmp_path):
    root = package(tmp_path)
    (root / "entry.py").write_text(
        "from importlib import import_module as load; load('.cache', package='sample')"
    )
    assert "cache.py" in source_closure(root, ["sample.entry"])
