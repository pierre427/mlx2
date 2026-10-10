import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.route_identity import (
    adapter_roots,
    bind_route_runtime,
    recompute_runtime_identity,
)
from mlx2.source_dependencies import UnresolvedDependency, file_digest


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "mlx2"
    for name, content in {
        "__init__.py": "",
        "server.py": "from .adapters import registry\n",
        "serving.py": "from .runtime.models import cache\n",
        "qualification.py": "",
        "route_identity.py": "",
        "adapters/__init__.py": "",
        "adapters/registry.py": "import importlib\ndef resolve(name):\n    return importlib.import_module(name)\n",
        "adapters/alpha.py": "def load():\n    from ..runtime.models.alpha import Model\n",
        "adapters/beta.py": "def load():\n    from ..runtime.models.beta import Model\n",
        "runtime/__init__.py": "",
        "runtime/models/__init__.py": "",
        "runtime/models/cache.py": "import importlib\ndef restore(name):\n    return importlib.import_module(name)\n",
        "runtime/models/alpha.py": "class Model: pass\n",
        "runtime/models/beta.py": "class Model: pass\n",
        "runtime/unrelated_research.py": "value = 1\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    reviews = {
        module: {"sha256": file_digest(root / path), "targets": []}
        for module, path in [
            ("mlx2.adapters.registry", "adapters/registry.py"),
            ("mlx2.runtime.models.cache", "runtime/models/cache.py"),
        ]
    }
    (root / "route_dispatch.json").write_text(
        json.dumps({"schema": 1, "dispatchers": reviews})
    )
    return root


def adapter(name):
    return type("Adapter", (), {"__module__": "mlx2.adapters." + name})()


def bind(root, name, build="build-1", route="ordinary"):
    return bind_route_runtime(
        {"source_sha256": build, "mlx": "test", "mlx_native_sha256": "native"},
        adapter(name),
        route,
        root=root,
    )


def edit(root, file):
    path = root / file
    path.write_text(path.read_text() + "# changed\n")


def test_unrelated_edit_keeps_route_and_persistent_namespace(source):
    from mlx2.serving import persistent_runtime_revision

    first, status = bind(source, "alpha")
    assert status["mode"] == "route_source"
    edit(source, "runtime/unrelated_research.py")
    second, _ = bind(source, "alpha", build="build-2")
    assert first == second
    assert persistent_runtime_revision(first) == persistent_runtime_revision(second)
    assert persistent_runtime_revision(first)[0] == "mlx2-route-runtime-v2"


def test_model_change_invalidates_only_routes_that_depend_on_it(source):
    alpha, _ = bind(source, "alpha")
    beta, _ = bind(source, "beta")
    edit(source, "runtime/models/alpha.py")
    assert bind(source, "alpha", build="build-2")[0] != alpha
    assert bind(source, "beta", build="build-2")[0] == beta
    assert bind(source, "alpha", route="prompt_lookup")[0] != alpha


@pytest.mark.parametrize("changed", ["serving.py", "runtime/models/cache.py"])
def test_shared_laws_invalidate_every_route(source, changed):
    before = {name: bind(source, name)[0] for name in ("alpha", "beta")}
    edit(source, changed)
    for name, previous in before.items():
        current, status = bind(source, name, build="build-2")
        assert current != previous
        if changed.endswith("cache.py"):
            assert status["mode"] == "full_source"
            assert "dispatcher changed" in status["reason"]


def test_new_unknown_dynamic_import_retains_full_source_identity(source):
    path = source / "adapters/alpha.py"
    path.write_text(
        path.read_text()
        + "import importlib\ndef extra(name):\n    return importlib.import_module(name)\n"
    )
    identity, status = bind(source, "alpha")
    assert identity["source_sha256"] == "build-1"
    assert "source_scope" not in identity
    assert status["mode"] == "full_source"


def test_qualification_recomputes_lazy_dependencies_and_rejects_deleted_source(source):
    identity, _ = bind(source, "alpha")
    build = {
        "source_sha256": "different-full-build",
        "mlx": "test",
        "mlx_native_sha256": "native",
    }
    assert recompute_runtime_identity(identity, build=build, root=source) == identity
    edit(source, "runtime/models/alpha.py")
    assert recompute_runtime_identity(identity, build=build, root=source) != identity
    (source / "runtime/models/alpha.py").unlink()
    with pytest.raises(UnresolvedDependency, match="missing local"):
        recompute_runtime_identity(identity, build=build, root=source)


def test_external_adapter_retains_full_source_identity(source):
    result, status = bind_route_runtime(
        {"source_sha256": "full"}, SimpleNamespace(), "ordinary", root=source
    )
    assert result == {"source_sha256": "full"}
    assert status["mode"] == "full_source"


def test_nested_packaged_policy_data_is_bound(source):
    directory = source / "runtime/data"
    directory.mkdir()
    item = directory / "policy.json"
    item.write_text('{"limit": 1}')
    first, _ = bind(source, "alpha")
    item.write_text('{"limit": 2}')
    assert bind(source, "alpha")[0] != first


def test_full_preflight_binding_independently_recomputes_active_route(
    source, monkeypatch
):
    from test_qualify_serving_receipts import qualify

    import mlx2.route_identity as module

    active, _ = bind(source, "alpha")
    original = module.recompute_runtime_identity
    monkeypatch.setattr(
        module,
        "recompute_runtime_identity",
        lambda runtime, *, build: original(runtime, build=build, root=source),
    )
    full = {
        "source_sha256": "whole-build",
        "mlx": "test",
        "mlx_native_sha256": "native",
    }
    assert qualify._active_runtime_matches_build(active, full)
    assert not qualify._active_runtime_matches_build(
        {**active, "source_sha256": "fabricated"}, full
    )
    edit(source, "runtime/models/alpha.py")
    assert not qualify._active_runtime_matches_build(active, full)


def test_repository_manifest_and_standard_decoder_ordinary_scope():
    package = Path(__file__).resolve().parents[1] / "src" / "mlx2"
    manifest = json.loads((package / "route_dispatch.json").read_text())
    reviews = manifest["dispatchers"]
    for module, review in reviews.items():
        source = package.joinpath(*module.split(".")[1:]).with_suffix(".py")
        assert review["sha256"] == file_digest(source), module

    # Build the real adapter/model module topology without constructing a model
    # or importing tensor code. The route identity must stay source-scoped.
    model_type = type(
        "Model", (), {"__module__": "mlx2.runtime.models.standard_decoder"}
    )
    standard_type = type(
        "StandardDecoderAdapter", (), {"__module__": "mlx2.adapters.standard_decoder"}
    )
    external_type = type(
        "ExternalDraftAdapter", (),
        {"__module__": "mlx2.adapters.external_draft_policy"},
    )
    model = object.__new__(model_type)
    standard = object.__new__(standard_type)
    standard.model = model
    adapter = object.__new__(external_type)
    adapter.model = standard

    expected = [
        "mlx2.adapters.external_draft_policy",
        "mlx2.adapters.standard_decoder",
        "mlx2.runtime.models.standard_decoder",
    ]
    assert adapter_roots(adapter) == expected
    runtime, status = bind_route_runtime(
        {"source_sha256": "test-build", "mlx": "test"},
        adapter,
        "ordinary",
        root=package,
    )
    assert status["mode"] == "route_source", status
    assert runtime["source_scope"]["roots"] == expected
