"""Adversarial source-delta binding over an unchanged full-suite baseline."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

h = load("qualify_serving")
delta = load("preflight_source_delta")


@pytest.fixture
def proof(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/mlx2").mkdir(parents=True)
    (tmp_path / "src/mlx2/foo.py").write_text("def value(): return 1\n")
    (tmp_path / "tests/test_foo.py").write_text("import foo\n")
    monkeypatch.setattr(h, "__file__", str(tmp_path / "scripts/qualify_serving.py"))
    tree = {"tests/test_foo.py": "testhash"}
    before = {"runtime": {"source_sha256": "before", "mlx_native_sha256": "native"},
              "interpreter": h.interpreter_identity(), "qualification_harness": {"sha256": next(iter(delta.BASE_HARNESSES))},
              "pytest_config_sha256": "config", "test_source_sha256": h.preflight_tree_sha256(tree)}
    after = copy.deepcopy(before)
    after["runtime"]["source_sha256"] = "after"
    after["qualification_harness"] = {"sha256": "new-approved-harness"}
    for guard in h.PREFLIGHT_IMPORT_GUARD_MODULES:
        (tmp_path / guard).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / guard).touch()
    commands = h.preflight_test_commands()
    base = {"schema": h.PREFLIGHT_SCHEMA, "passed": True, "identity": before, "identity_before": before,
            "tree": tree, "source_stable": True, "test_command": commands[0], "guard_test_command": commands[1],
            "returncode": 0, "guard_returncode": 0}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    monkeypatch.setattr(delta, "revision_source", lambda *a: ("before", {"src/mlx2/foo.py": "old"}))
    monkeypatch.setattr(delta, "runtime_source", lambda *a: ("after", {"src/mlx2/foo.py": "new"}))
    monkeypatch.setattr(delta, "approved_profile", lambda *a: {"base_source_revision": "a" * 40, "files": {"src/mlx2/foo.py": {"after_sha256": "new", "tests": ["tests/test_foo.py"]}}})
    return SimpleNamespace(root=tmp_path, base=base, path=path, after=after, tree=tree, revision="a" * 40,
                           identity_fn=lambda: after, tree_fn=lambda: tree)


def write(proof, **kwargs):
    path = proof.root / "delta.json"
    receipt = delta.write(h, path, proof.path, proof.revision, run=lambda *a, **kw: SimpleNamespace(returncode=0, stdout="1 passed", stderr=""), identity_fn=proof.identity_fn, tree_fn=proof.tree_fn, **kwargs)
    return path, receipt


def test_writer_and_validator_cover_source_change_without_reusing_source_verdict(proof):
    path, receipt = write(proof)
    assert "tests/test_foo.py" in receipt["impacted"]
    assert receipt["changes"]["modified"] == ["src/mlx2/foo.py"]
    assert receipt["execution_environment"]["MLX2_PREFLIGHT_CPU_ONLY"] == "1"
    result = delta.validate(h, path, receipt, proof.after, proof.tree_fn)
    assert result["passed"] is True
    with pytest.raises(AssertionError, match="does not match"):
        h.write_preflight_delta(proof.root / "incompatible-delta.json", proof.path, identity_fn=proof.identity_fn, tree_fn=proof.tree_fn)


@pytest.mark.parametrize("field,value,error", [
    ("native", "different", "native build"),
    ("pytest", "different", "pytest configuration"),
    ("producer", "unknown", "historical harness"),
    ("stable", False, "not stable"),
    ("tree", {}, "tree is absent or corrupt"),
])
def test_base_incompatibilities_are_refused(proof, field, value, error):
    if field == "native":
        proof.after["runtime"]["mlx_native_sha256"] = value
    elif field == "pytest":
        proof.after["pytest_config_sha256"] = value
    elif field == "producer":
        proof.base["identity"]["qualification_harness"]["sha256"] = value
    elif field == "stable":
        proof.base["source_stable"] = value
    else:
        proof.base["tree"] = value
    proof.path.write_text(json.dumps(proof.base))
    with pytest.raises(AssertionError, match=error):
        write(proof)


@pytest.mark.parametrize("which,error", [("base", "Git reconstruction"), ("active", "active runtime")])
def test_manifest_digest_must_match_each_runtime(proof, monkeypatch, which, error):
    target = "revision_source" if which == "base" else "runtime_source"
    monkeypatch.setattr(delta, target, lambda *a: ("incorrect", {}))
    with pytest.raises(AssertionError, match=error):
        write(proof)


@pytest.mark.parametrize("mutation", ["coverage", "source", "returncode", "command", "interpreter", "base_hash", "source_stable"])
def test_forged_or_incomplete_delta_is_refused(proof, mutation):
    path, receipt = write(proof)
    if mutation == "coverage": receipt["impacted"] = []
    elif mutation == "source": receipt["source_tree"] = {}
    elif mutation == "returncode": receipt["returncode"] = False
    elif mutation == "command": receipt["test_command"][0] = "true"
    elif mutation == "interpreter": receipt["identity"]["interpreter"]["executable"] = "/other/python"
    elif mutation == "base_hash": receipt["base"]["sha256"] = "invalid"
    else: receipt["source_stable"] = False
    with pytest.raises(AssertionError):
        delta.validate(h, path, receipt, proof.after, proof.tree_fn)


def test_writer_rejects_source_mutation_during_tests(proof, monkeypatch):
    values = iter([("after", {"src/mlx2/foo.py": "new"}), ("after", {"src/mlx2/foo.py": "changed"})])
    monkeypatch.setattr(delta, "runtime_source", lambda *a: next(values))
    with pytest.raises(AssertionError, match="source changed"):
        write(proof)


def test_runtime_digest_matches_existing_runtime_identity():
    from mlx2.serving import runtime_identity
    digest, manifest = delta.runtime_source(ROOT)
    assert digest == runtime_identity()["source_sha256"]
    assert manifest["src/mlx2/serving.py"] == hashlib.sha256((ROOT / "src/mlx2/serving.py").read_bytes()).hexdigest()


@pytest.mark.parametrize("revision", ["main", "HEAD", "a" * 39, "A" * 40, None])
def test_only_immutable_git_commit_names_are_allowed(revision):
    with pytest.raises(AssertionError, match="immutable"):
        delta.revision_source(ROOT, revision)


@pytest.mark.parametrize("fault", ["missing", "empty", "absent", "wrong_base"])
def test_reviewed_coverage_cannot_be_skipped(proof, monkeypatch, fault):
    profile = {"base_source_revision": proof.revision, "files": {"src/mlx2/foo.py": {"after_sha256": "new", "tests": ["tests/test_foo.py"]}}}
    if fault == "missing": profile["files"] = {}
    elif fault == "empty": profile["files"]["src/mlx2/foo.py"]["tests"] = []
    elif fault == "absent": profile["files"]["src/mlx2/foo.py"]["tests"] = ["tests/test_absent.py"]
    else: profile["base_source_revision"] = "b" * 40
    monkeypatch.setattr(delta, "approved_profile", lambda *a: profile)
    with pytest.raises(AssertionError):
        write(proof)


def test_reviewed_profile_bytes_are_pinned(tmp_path, monkeypatch):
    path = tmp_path / delta.PROFILE_PATH
    path.parent.mkdir(parents=True)
    path.write_text('{"files":{}}')
    monkeypatch.setattr(delta, "PROFILE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert delta.approved_profile(tmp_path) == {"files": {}}
    path.write_text('{"files":{"forged":{}}}')
    with pytest.raises(AssertionError, match="coverage profile"):
        delta.approved_profile(tmp_path)


def test_qualification_approval_exception_is_pin_only(tmp_path, monkeypatch):
    path = tmp_path / "src/mlx2/qualification.py"
    path.parent.mkdir(parents=True)
    old = "APPROVED_QUALIFICATION_HARNESS = {'sha256':'old'}\ndef check(): return True\n"
    monkeypatch.setattr(delta.subprocess, "check_output", lambda *a, **kw: old)
    path.write_text(old.replace("'old'", "'new'"))
    delta.verify_pin_only_change(tmp_path, "a" * 40)
    path.write_text(path.read_text().replace("return True", "return False"))
    with pytest.raises(AssertionError, match="only the qualification harness pin"):
        delta.verify_pin_only_change(tmp_path, "a" * 40)


def test_harness_binds_the_helper_code(monkeypatch):
    helper = ROOT / "scripts/preflight_source_delta.py"
    monkeypatch.setattr(h, "SOURCE_DELTA_HELPER_SHA256", hashlib.sha256(helper.read_bytes()).hexdigest())
    assert h.source_delta_helper().SCHEMA == delta.SCHEMA
    monkeypatch.setattr(h, "SOURCE_DELTA_HELPER_SHA256", "wrong")
    with pytest.raises(AssertionError, match="helper changed"):
        h.source_delta_helper()


def test_reviewed_profile_must_cover_lexically_reachable_tests(proof):
    path = proof.root / "tests/test_extra.py"
    path.write_text("import foo\n")
    proof.tree["tests/test_extra.py"] = "extra"
    # Keep the historical baseline tree and digest intact; the new current
    # test is outside the profile even though it reaches the changed source.
    with pytest.raises(AssertionError, match="reviewed coverage|omits impacted"):
        write(proof)


def test_uniform_profile_cannot_cover_unreachable_source(proof):
    (proof.root / "tests/test_foo.py").write_text("assert True\n")
    with pytest.raises(AssertionError, match="cannot reach any test module"):
        write(proof)


@pytest.mark.parametrize("path", ["scripts/qualify_serving.py", "scripts/preflight_source_delta.py", delta.PROFILE_PATH])
def test_source_delta_trust_chain_files_have_no_mutable_approval_exception(path):
    assert path not in delta.APPROVAL_PATHS
