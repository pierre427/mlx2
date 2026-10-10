"""Content equivalence permits unrelated upstream changes without weakening pins."""

import json
import sys
from types import SimpleNamespace

import pytest

from mlx2.adapters import vlm_runtime as V
from mlx2.source_dependencies import source_closure


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "mlx_vlm"
    for name, text in {
        "__init__.py": "from .utils import load\n",
        "utils.py": "def load():\n    from .models.toy import Model\n",
        "models/__init__.py": "",
        "models/toy.py": "from .cache import Cache\nclass Model: pass\n",
        "models/cache.py": "class Cache: pass\n",
        "models/unrelated.py": "value = 1\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    roots = ["mlx_vlm", "mlx_vlm.models.toy"]
    files = source_closure(root, roots)
    manifest = {"schema": 1, "source_revision": "a" * 40, "files": files}
    contract = {"roots": roots, "reviewed_dynamic": {}, "files": sorted(files)}
    monkeypatch.setattr(V, "_contract", lambda family: (manifest, contract))
    monkeypatch.setattr(V, "_package_root", lambda: root)
    # A real mlx_vlm imported by an earlier test in this process is, for the
    # toy root, "another mlx-vlm package"; hide it for the toy binding.
    for name in [name for name in sys.modules if name == "mlx_vlm" or name.startswith("mlx_vlm.")]:
        monkeypatch.delitem(sys.modules, name)
    from mlx2.adapters import mlx_vlm_pin

    monkeypatch.setattr(
        mlx_vlm_pin,
        "mlx_vlm_runtime",
        lambda: {"revision": "different-commit", "version": "test"},
    )
    return root


def test_unrelated_model_edit_preserves_content_identity(source):
    first = V.bind_backend("toy")
    assert first.provenance["revision"] == "different-commit"
    (source / "models/unrelated.py").write_text("value = 2\n")
    assert V.bind_backend("toy").identity == first.identity
    assert first.identity["reference_revision"] == "a" * 40


@pytest.mark.parametrize("path", ["utils.py", "models/cache.py", "models/toy.py"])
def test_shared_loader_cache_and_selected_model_edits_fail_closed(source, path):
    item = source / path
    item.write_text(item.read_text() + "# changed\n")
    with pytest.raises(RuntimeError, match="dependency content differs"):
        V.bind_backend("toy")


def test_imported_package_from_another_checkout_is_refused(source, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mlx_vlm", SimpleNamespace(__file__="/other/mlx_vlm/__init__.py")
    )
    with pytest.raises(RuntimeError, match="already imported"):
        V.bind_backend("toy")


def test_owned_loader_refuses_dispatch_escape_and_preserves_strict_loading(
    source, tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    calls = []
    loaded = type("Model", (), {"__module__": "mlx_vlm.models.toy"})()
    package = SimpleNamespace(
        __file__=str(source / "__init__.py"),
        load=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or (loaded, "processor")
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    backend = V.bind_backend("toy")
    config.write_text(json.dumps({"model_type": "toy"}))
    assert backend.load(model, lazy=False) == (loaded, "processor")
    assert calls[0][1] == {"lazy": False, "strict": True, "trust_remote_code": False}
    for change in [
        {"model_type": "another"},
        {"model_file": "custom.py"},
        {"dflash_config": {}},
        {"architectures": ["DFlash2DraftModel"]},
    ]:
        config.write_text(json.dumps({"model_type": "toy", **change}))
        with pytest.raises(ValueError):
            backend.load(model)
    assert len(calls) == 1
    config.write_text(json.dumps({"model_type": "toy"}))
    with pytest.raises(ValueError, match="remote processor"):
        backend.load(model, trust_remote_code=True)
    with pytest.raises(ValueError, match="strict weight"):
        backend.load(model, strict=False)


def test_source_edit_during_load_is_detected_before_return(
    source, tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "toy"}))

    def load(*args, **kwargs):
        (source / "models/cache.py").write_text("class Cache: changed = True\n")
        return "model", "processor"

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(
            __file__=str(source / "__init__.py"),
            load=load,
        ),
    )
    backend = V.bind_backend("toy")
    with pytest.raises(RuntimeError, match="dependency content differs"):
        backend.load(model)


def test_checked_in_contracts_have_reference_bytes_for_every_dependency():
    from pathlib import Path

    for path in Path(V.__file__).with_name("vlm_contracts").glob("*.json"):
        manifest = json.loads(path.read_text())
        assert len(manifest["source_revision"]) == 40
        for contract in manifest["families"].values():
            assert contract["roots"] and contract["files"]
            assert all(len(manifest["files"][name]) == 64 for name in contract["files"])
            for module, entry in contract["reviewed_dynamic"].items():
                name = module.removeprefix("mlx_vlm.").replace(".", "/") + ".py"
                if name in contract["files"]:
                    assert manifest["files"][name] == entry["sha256"]
                assert entry["reason"]


def test_owned_loader_rejects_overrides_and_unbound_returned_model(
    source, tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "toy"}))
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        SimpleNamespace(
            __file__=str(source / "__init__.py"),
            load=lambda *a, **kw: (object(), None),
        ),
    )
    backend = V.bind_backend("toy")
    with pytest.raises(ValueError, match="loader overrides"):
        backend.load(model, model_config={"model_file": "escape.py"})
    with pytest.raises(RuntimeError, match="escaped"):
        backend.load(model)


def test_checkout_documentation_is_not_an_executable_dependency(source):
    first = V.bind_backend("toy").identity
    (source / "README.md").write_text("source-only documentation")
    assert V.bind_backend("toy").identity == first


# ---- the media qualification producers and readers bind the same identity ----
# 0abde0ba1 replaced ``adapter.mlx_vlm_runtime`` (``{version, source, editable,
# revision}``) with the dependency-content identity.  The three approved media
# producers still gated on ``["revision"]`` and so refused every adapter, and
# their evidence readers could not accept a report carrying the new identity.

import importlib.util
from pathlib import Path

REPO = Path(V.__file__).resolve().parents[3]
PRODUCERS = (
    ("qualify_media_serving", "smolvlm"),
    ("qualify_lfm25_media_serving", "lfm2_vl"),
    ("qualify_qwen25_media_serving", "qwen2_5_vl"),
)


def _producer(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_runtime(revision):
    return {"version": "0.7.3", "source": "index", "editable": False, "revision": revision}


@pytest.mark.parametrize("name,family", PRODUCERS)
def test_media_producers_gate_on_the_dependency_contract(source, monkeypatch, name, family):
    producer = _producer(name)
    assert producer.FAMILY == family
    identity = V.bind_backend(family).identity
    assert set(identity) == {
        "schema", "family", "source_sha256", "dependency_files", "reference_revision",
    }
    assert identity["family"] == family
    monkeypatch.setattr(producer, "SOURCE_REVISION", identity["reference_revision"])
    adapter = SimpleNamespace(mlx_vlm_runtime=identity)
    assert producer.source_contract(adapter) == identity
    for runtime in (
        _legacy_runtime(identity["reference_revision"]),
        {**identity, "reference_revision": "b" * 40},
        {**identity, "source_sha256": None},
        {**identity, "schema": "other"},
        {**identity, "family": "toy"},
        V.bind_backend("toy").identity,
        None,
    ):
        with pytest.raises(AssertionError, match="dependency contract"):
            producer.source_contract(SimpleNamespace(mlx_vlm_runtime=runtime))


@pytest.mark.parametrize("name,family", PRODUCERS)
def test_media_producers_refuse_an_installed_contract_on_another_source_line(name, family):
    """A live bind_backend identity from the installed pin is only accepted by
    a producer bound to that reference revision; the 8a5e704e producers refuse
    the 67599f2e families instead of silently qualifying on the wrong line."""
    pytest.importorskip("mlx_vlm")
    try:
        identity = V.bind_backend("gemma4").identity
    except RuntimeError as exc:
        pytest.skip(f"installed mlx-vlm is not the gemma4 contract: {exc}")
    producer = _producer(name)
    adapter = SimpleNamespace(mlx_vlm_runtime=identity)
    assert identity["family"] != family
    with pytest.raises(AssertionError, match="dependency contract"):
        producer.source_contract(adapter)


def _content_identity(revision, family="qwen2_5_vl", sha="f" * 64):
    return {"schema": "mlx2.vlm-dependencies.v1", "family": family,
            "source_sha256": sha, "dependency_files": 105,
            "reference_revision": revision}


def test_evidence_readers_accept_the_contract_identity_and_refuse_legacy():
    from mlx2.media_qualification import (
        arm_source_bound, parity_source_bound, source_contract_identity,
    )

    revision = "a" * 40
    identity = _content_identity(revision)
    assert source_contract_identity(identity, revision)
    assert source_contract_identity(identity, revision, family="qwen2_5_vl")
    assert not source_contract_identity(identity, revision, family="smolvlm")
    assert not source_contract_identity(_legacy_runtime(revision), revision)
    assert not source_contract_identity({**identity, "reference_revision": "b" * 40}, revision)
    assert not source_contract_identity({**identity, "source_sha256": "F" * 64}, revision)
    assert not source_contract_identity({**identity, "dependency_files": 0}, revision)
    assert not source_contract_identity(None, revision)
    parity = {"source_revision": revision, "source_sha256": identity["source_sha256"]}
    assert parity_source_bound(parity, identity, revision)
    assert not parity_source_bound({"source_revision": revision}, identity, revision)
    assert not parity_source_bound({**parity, "source_sha256": "e" * 64}, identity, revision)
    assert not parity_source_bound({**parity, "source_revision": "b" * 40}, identity, revision)
    assert not parity_source_bound(parity, _legacy_runtime(revision), revision)
    # An arm is evidence only for the identity the served settings carry, for
    # the family the report qualifies: a self-asserted identity with a
    # matching fabricated digest is not.
    arm = {"mlx_vlm_runtime": dict(identity), "parity": dict(parity)}
    settings = {"max_lanes": 1, "mlx_vlm": dict(identity)}
    bound = dict(settings=settings, family="qwen2_5_vl", revision=revision)
    assert arm_source_bound(arm, **bound)
    assert not arm_source_bound(arm, **{**bound, "family": "smolvlm"})
    assert not arm_source_bound(arm, **{**bound, "settings": {"max_lanes": 1}})
    assert not arm_source_bound(arm, **{**bound, "settings": None})
    other = _content_identity(revision, sha="e" * 64)
    assert not arm_source_bound(
        {"mlx_vlm_runtime": other, "parity": {**parity, "source_sha256": "e" * 64}}, **bound
    )
    assert not arm_source_bound({"parity": dict(parity)}, **bound)
    assert not arm_source_bound(None, **bound)


def test_tower_reuse_bench_gates_on_the_dependency_contract(source, monkeypatch):
    """scripts/bench_media_tower_reuse_m3.py kept the same ``["revision"]``
    read, so every benchmark mode aborted on a correctly bound adapter."""
    bench = _producer("bench_media_tower_reuse_m3")
    assert bench.FAMILIES == {"smol": "smolvlm", "qwen": "qwen2_5_vl"}
    for label, family in bench.FAMILIES.items():
        identity = V.bind_backend(family).identity
        monkeypatch.setattr(bench, "SOURCE_REV", identity["reference_revision"])
        adapter = SimpleNamespace(mlx_vlm_runtime=identity)
        assert bench.source_contract(adapter, label) == identity
        other = next(name for name in bench.FAMILIES if name != label)
        with pytest.raises(RuntimeError, match="dependency contract"):
            bench.source_contract(adapter, other)
        with pytest.raises(RuntimeError, match="dependency contract"):
            bench.source_contract(SimpleNamespace(
                mlx_vlm_runtime=_legacy_runtime(identity["reference_revision"])
            ), label)


def test_pre_contract_media_records_no_longer_match_the_served_identity():
    """Every checked-in VLM companion was recorded with the legacy
    ``settings.mlx_vlm`` shape.  The engine now publishes the dependency
    content identity there, and ``mlx_vlm`` is deliberately not provenance
    only, so those routes serve unqualified until the producers are re-run."""
    from mlx2 import qualification

    assert "mlx_vlm" not in qualification.PROVENANCE_ONLY_SETTINGS
    records = sorted(
        list((REPO / "docs/experiments").glob("*-M3-LIVE-MEDIA-QUALIFICATION-*.json"))
        + list((REPO / "qualification/runs").glob("*/*/media-recert-*/*.json"))
    )
    assert records
    legacy = []
    for path in records:
        record = json.loads(path.read_text())
        recorded = record["settings"]["mlx_vlm"]
        served = _content_identity(record["source_revision"], record["model_type"])
        if recorded.get("schema") == served["schema"]:
            continue
        legacy.append(path.name)
        rebound = {**record, "qualification_harness":
                   qualification.APPROVED_MEDIA_PRODUCERS[record["model_type"]][0]}
        with pytest.raises(ValueError, match="does not match serving settings"):
            qualification.validate_adapter_qualification(
                rebound, runtime=record["runtime"], artifact=record["artifact"],
                settings={**record["settings"], "mlx_vlm": served},
            )
    assert legacy, "no legacy record left: drop this test with the requalification"
