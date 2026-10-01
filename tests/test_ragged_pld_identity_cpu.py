"""Host-only tests for the ragged PLD identity gates (scripts/qualify_ragged_pld.py).

No mlx, mlx_lm or mlx2 module may load: a meta-path blocker is installed
before the driver is imported, and every test checks that none is present.
Artifacts are synthetic directories of small metadata files and placeholder
shards whose permissions are removed, so any shard read would fail; only
their stat is used. Git is a host substitute (``Q._git``) except where a
test names the subprocess seam.

The native gate ORDER is executed through the driver's seams with host
fakes (``Q._import``, ``Q.build_identity``, ``Q.loaded_module_files``): no
MLX is imported, no adapter, model or array is built. That proves the
source order of the gates, not that a real MLX build or model passes them.
Recipes are cross-checked against the real inspectors by AST only.

  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
      -o addopts= -q tests/test_ragged_pld_identity_cpu.py
"""

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

NATIVE = frozenset({"mlx", "mlx_lm", "mlx2"})


def _native_loaded():
    return sorted(name for name in sys.modules if name.split(".")[0] in NATIVE)


class _NativeBlocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in NATIVE:
            raise ImportError(f"native import blocked in a host test: {name}")


assert not _native_loaded(), _native_loaded()
sys.meta_path.insert(0, _NativeBlocker())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import qualify_ragged_pld as Q

COMMIT = "a" * 40
UNAVAILABLE = {"status": "unavailable", "sha256": None, "reason": "no state returned"}


@pytest.fixture(autouse=True)
def host_only():
    assert not _native_loaded()
    yield
    assert not _native_loaded(), _native_loaded()


class FakeGit:
    """``Q._git`` substitute: clean and tracked unless a test says otherwise."""

    def __init__(self):
        self.head, self.status, self.tracked, self.fail = COMMIT, "", True, set()

    def __call__(self, *args):
        if args[0] in self.fail:
            return None
        if args[0] == "rev-parse":
            return self.head + "\n"
        if args[0] == "status":
            return self.status
        if args[0] == "ls-files":
            return "\n".join(args[3:]) if self.tracked else None
        raise AssertionError(f"unexpected git call {args}")


@pytest.fixture
def git(monkeypatch):
    fake = FakeGit()
    monkeypatch.setattr(Q, "_git", fake)
    return fake


# ---- synthetic artifacts (metadata + stat only) ----

CONFIGS = {
    "muse": {"model_type": "muse_glimmer", "num_hidden_layers": 4, "hidden_size": 16},
    "qwen38": {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 64, "hidden_size": 5120}},
    "qwen36": {"model_type": "qwen3_5_moe", "text_config": {"num_hidden_layers": 40, "hidden_size": 2048}},
    "north": {"model_type": "cohere2_moe", "num_hidden_layers": 49, "hidden_size": 2048},
}
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def make_artifact(path, family="muse", *, shards=SHARDS, index=True, config=None, sizes=(3, 5)):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config if config is not None else CONFIGS[family]))
    for name, text in (("tokenizer.json", '{"t": 1}'), ("tokenizer_config.json", '{"c": 2}'),
                       ("chat_template.jinja", "{{ m }}"), ("generation_config.json", '{"g": 3}')):
        (path / name).write_text(text)
    if index:
        weight_map = {f"w{i}": name for i, name in enumerate(shards)}
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name, size in zip(shards, sizes):
        item = path / name
        if isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts:
            item.parent.mkdir(parents=True, exist_ok=True)
            item.write_bytes(b"\0" * size)
            item.chmod(0)  # any read of a shard now fails; stat still works
    return path


def independent_fingerprint(path, metadata, shards):
    digest = hashlib.sha256()
    for name in metadata:
        if (path / name).is_file():
            digest.update(name.encode())
            digest.update((path / name).read_bytes())
    for name in sorted(set(shards)):
        stat = (path / name).stat()
        digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
    return digest.hexdigest()


def refused(path, match):
    with pytest.raises(ValueError, match=match):
        Q.artifact_manifest(path)


def test_shards_are_unreadable_so_no_test_reads_weights(tmp_path):
    path = make_artifact(tmp_path / "m")
    if os.geteuid() == 0:
        pytest.skip("permission removal does not bind root")
    with pytest.raises(PermissionError):
        (path / SHARDS[0]).read_bytes()


@pytest.mark.parametrize("family,metadata", [("muse", Q.MUSE_METADATA), ("qwen38", Q.QWEN_METADATA),
                                             ("qwen36", Q.QWEN_METADATA)])
def test_manifest_matches_an_independent_recompute_of_each_recipe(tmp_path, family, metadata):
    path = make_artifact(tmp_path / family, family)
    manifest = Q.artifact_manifest(path)
    assert manifest["family"] == family and manifest["adapter"] == Q.FAMILIES[family]["adapter"]
    assert manifest["fingerprint"] == independent_fingerprint(path, metadata, SHARDS)
    assert manifest["shards"] == [[n, (path / n).stat().st_size, (path / n).stat().st_mtime_ns] for n in SHARDS]
    assert "no shard or tensor bytes read" in manifest["fingerprint_scope"]
    assert "not tensor-content verification" in manifest["fingerprint_scope"]


def test_generation_config_binds_qwen_but_not_muse(tmp_path):
    muse, qwen = make_artifact(tmp_path / "muse", "muse"), make_artifact(tmp_path / "qwen", "qwen38")
    before = Q.artifact_manifest(muse)["fingerprint"], Q.artifact_manifest(qwen)["fingerprint"]
    for path in (muse, qwen):
        (path / "generation_config.json").write_text('{"g": 4}')
    after = Q.artifact_manifest(muse)["fingerprint"], Q.artifact_manifest(qwen)["fingerprint"]
    assert before[0] == after[0] and before[1] != after[1]
    assert "generation_config.json" not in Q.MUSE_METADATA and Q.QWEN_METADATA[-1] == "generation_config.json"


def test_absent_optional_metadata_is_skipped_like_the_inspectors(tmp_path):
    path = make_artifact(tmp_path / "m")
    (path / "chat_template.jinja").unlink()
    manifest = Q.artifact_manifest(path)
    assert "chat_template.jinja" not in manifest["metadata_sha256"]
    assert manifest["fingerprint"] == independent_fingerprint(path, Q.MUSE_METADATA, SHARDS)


def test_shard_mtime_and_size_drift_change_the_fingerprint(tmp_path):
    path = make_artifact(tmp_path / "m")
    first = Q.artifact_manifest(path)["fingerprint"]
    stat = (path / SHARDS[0]).stat()
    os.utime(path / SHARDS[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    second = Q.artifact_manifest(path)["fingerprint"]
    (path / SHARDS[1]).chmod(0o600)
    (path / SHARDS[1]).write_bytes(b"\0" * 9)
    (path / SHARDS[1]).chmod(0)
    third = Q.artifact_manifest(path)["fingerprint"]
    assert len({first, second, third}) == 3


def test_muse_single_shard_convention_and_qwen_index_requirement(tmp_path):
    muse = make_artifact(tmp_path / "muse", "muse", shards=("model.safetensors",), sizes=(4,), index=False)
    manifest = Q.artifact_manifest(muse)
    assert [s[0] for s in manifest["shards"]] == ["model.safetensors"]
    assert manifest["fingerprint"] == independent_fingerprint(muse, Q.MUSE_METADATA, ["model.safetensors"])
    refused(make_artifact(tmp_path / "qwen", "qwen38", shards=("model.safetensors",), sizes=(4,), index=False),
            "requires model.safetensors.index.json")
    refused(make_artifact(tmp_path / "bare", "muse", shards=(), index=False), "missing weight shard")


def test_shard_traversal_absolute_symlink_suffix_and_missing_are_refused(tmp_path):
    (tmp_path / "outside.safetensors").write_bytes(b"x")
    refused(make_artifact(tmp_path / "a", shards=("../outside.safetensors",)), "not a local .safetensors")
    refused(make_artifact(tmp_path / "b", shards=(str(tmp_path / "outside.safetensors"),)), "not a local")
    link = make_artifact(tmp_path / "c", shards=(SHARDS[0],))
    os.symlink(tmp_path / "outside.safetensors", link / "escape.safetensors")
    (link / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "escape.safetensors"}}))
    refused(link, "not a local")
    refused(make_artifact(tmp_path / "d", shards=("model.bin",)), "not a local .safetensors")
    missing = make_artifact(tmp_path / "e")
    (missing / SHARDS[1]).unlink()
    refused(missing, "missing weight shard")


@pytest.mark.parametrize("index,match", [
    ({"weight_map": {}}, "nonempty weight_map"),
    ({"weight_map": ["a"]}, "nonempty weight_map"),
    ({}, "nonempty weight_map"),
    ([], "nonempty weight_map"),
    ({"weight_map": {"w": 3}}, "non-string shard"),
])
def test_malformed_weight_index_is_refused(tmp_path, index, match):
    path = make_artifact(tmp_path / "m")
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    refused(path, match)


def test_unreadable_metadata_and_non_file_metadata_are_refused(tmp_path):
    bad = make_artifact(tmp_path / "a")
    (bad / "model.safetensors.index.json").write_text("{not json")
    refused(bad, "index.json unreadable")
    folder = make_artifact(tmp_path / "b")
    (folder / "tokenizer.json").unlink()
    (folder / "tokenizer.json").mkdir()
    refused(folder, "not a regular file")
    refused(tmp_path / "absent", "not a directory")
    noconfig = make_artifact(tmp_path / "c")
    (noconfig / "config.json").unlink()
    refused(noconfig, "config.json unreadable")


@pytest.mark.parametrize("config,match", [
    ([], "not a JSON object"),
    ({"model_type": "muse_glimmer", "dflash_config": {}}, "drafter"),
    ({"model_type": "qwen3_5", "architectures": ["Qwen3DraftModel"]}, "drafter"),
    (CONFIGS["north"], "header digests"),
    ({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 32, "hidden_size": 4096}}, "no identity recipe"),
    ({"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 64, "hidden_size": 5120, "num_experts": 8}},
     "no identity recipe"),
    ({"model_type": "qwen3_5_moe", "text_config": {"num_hidden_layers": 48, "hidden_size": 3072}},
     "no identity recipe"),
    ({"model_type": "llama"}, "no identity recipe"),
    ({"model_type": "qwen3_5", "text_config": []}, "text_config"),
])
def test_dispatch_refuses_what_it_cannot_bind(tmp_path, config, match):
    refused(make_artifact(tmp_path / "m", config=config), match)


# ---- recipes and dispatch cross-checked against the real sources (AST only) ----

def _source(rel):
    return ast.parse((ROOT / rel).read_text())


def _function_named(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _metadata_tuple(fn):
    for node in ast.walk(fn):
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple)
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)):
            return tuple(e.value for e in node.iter.elts)
    return None


@pytest.mark.parametrize("family", ["muse", "qwen38", "qwen36"])
def test_recipe_matches_the_real_inspector_source(family):
    adapter_file = Q.FAMILIES[family]["files"][0]
    fn = _function_named(_source(adapter_file), "inspect_artifact")
    text = ast.unparse(fn)
    assert _metadata_tuple(fn) == Q.FAMILIES[family]["metadata"]
    assert "record = (name, stat.st_size, stat.st_mtime_ns)" in text
    assert "digest.update(json.dumps(record).encode())" in text
    assert "header" not in text  # no shard-content digest in these recipes
    module, cls = Q.FAMILIES[family]["adapter"].rsplit(".", 1)
    assert module == "mlx2." + adapter_file.removeprefix("src/mlx2/").removesuffix(".py").replace("/", ".")
    assert any(isinstance(n, ast.ClassDef) and n.name == cls for n in _source(adapter_file).body)


def test_muse_single_shard_and_identity_shapes_match_the_adapters():
    muse = ast.unparse(_function_named(_source("src/mlx2/adapters/muse_glimmer.py"), "inspect_artifact"))
    assert "names = ['model.safetensors']" in muse
    assert "self.identity = inspect_artifact(path)" in (ROOT / "src/mlx2/adapters/muse_glimmer.py").read_text()
    for rel in ("src/mlx2/adapters/qwen36_35b.py", "src/mlx2/adapters/qwen38_27b.py"):
        assert 'self.identity = artifact["identity"]' in (ROOT / rel).read_text()


def test_north_is_refused_because_its_recipe_reads_shard_headers():
    fn = _function_named(_source("src/mlx2/adapters/north_mini_code.py"), "inspect_artifact")
    assert "_validate_weight_headers" in ast.unparse(fn) and "header_digests" in ast.unparse(fn)
    assert Q.artifact_family(CONFIGS["north"])[0] is None


def test_dispatch_mirrors_the_registry_resolvers():
    tree = _source("src/mlx2/adapters/registry.py")
    resolvers = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "_RESOLVERS")
    table = {k.value: v.id for k, v in zip(resolvers.value.keys, resolvers.value.values)}
    assert table["muse_glimmer"] == table["muse_glimmer_text"] == "_muse_glimmer"
    assert table["qwen3_5"] == "_qwen3_5_dense" and table["qwen3_5_moe"] == "_qwen36_35b"
    dense = ast.unparse(_function_named(tree, "_qwen3_5_dense"))
    assert "topology != (64, 5120)" in dense and "Qwen3827BAdapter" in dense
    moe = ast.unparse(_function_named(tree, "_qwen36_35b"))
    assert "== (48, 3072)" in moe and "Qwen3635BA3BAdapter" in moe
    assert "MuseGlimmerAdapter" in ast.unparse(_function_named(tree, "_muse_glimmer"))
    inspect_model = ast.unparse(_function_named(tree, "inspect_model"))
    assert "dflash_config" in inspect_model and "'Draft' in name" in inspect_model
    for family, model_type in (("muse", "muse_glimmer"), ("qwen38", "qwen3_5"), ("qwen36", "qwen3_5_moe")):
        assert Q.artifact_family({**CONFIGS[family], "model_type": model_type})[0] == family


def test_required_files_exist_and_extend_the_scheduler_six():
    six = ("scripts/qualify_ragged_pld.py", "scripts/paired_direct_ab.py", "src/mlx2/runtime/pld.py",
           "src/mlx2/runtime/generate.py", "src/mlx2/runtime/segmented_rotating_kv.py",
           "src/mlx2/runtime/models/cache.py")
    assert Q.IDENTITY_FILES[:6] == six
    assert {"src/mlx2/adapters/registry.py", "src/mlx2/serving.py", "src/mlx2/contracts.py"} <= set(Q.IDENTITY_FILES)
    for name in Q.IDENTITY_FILES + tuple(f for fam in Q.FAMILIES.values() for f in fam["files"]):
        assert (ROOT / name).is_file(), name


# ---- source identity: git failures are never clean ----

def _run_raising(error):
    def run(*args, **kwargs):
        raise error
    return run


@pytest.mark.parametrize("run", [
    _run_raising(OSError("no git")),
    _run_raising(subprocess.TimeoutExpired("git", 60)),
    lambda *a, **k: subprocess.CompletedProcess(a[0], 128, "", "fatal: not a git repository"),
])
def test_git_failure_or_nonzero_is_unknown_not_clean(monkeypatch, run):
    monkeypatch.setattr(Q.subprocess, "run", run)
    assert Q._git("status") is None
    identity = Q.source_identity(Q.IDENTITY_FILES)
    assert identity["commit"] is None and identity["status_known"] is False
    assert identity["dirty"] is None and identity["tracked"] is False
    problems = " | ".join(Q.source_identity_refusals(identity, Q.IDENTITY_FILES))
    assert "git failed" in problems and "unknown" in problems and "not all tracked" in problems


def test_clean_tracked_source_passes_and_dirty_or_untracked_refuses(git):
    clean = Q.source_identity(Q.IDENTITY_FILES)
    assert Q.source_identity_refusals(clean, Q.IDENTITY_FILES) == []
    git.status = " M src/mlx2/runtime/pld.py\n"
    assert any("not clean" in p for p in Q.source_identity_refusals(Q.source_identity(Q.IDENTITY_FILES),
                                                                     Q.IDENTITY_FILES))
    git.status, git.tracked = "", False
    assert any("not all tracked" in p for p in Q.source_identity_refusals(Q.source_identity(Q.IDENTITY_FILES),
                                                                           Q.IDENTITY_FILES))


def test_missing_required_file_has_no_hash_and_refuses(git):
    names = Q.IDENTITY_FILES + ("src/mlx2/runtime/does_not_exist.py",)
    identity = Q.source_identity(names)
    assert identity["files"]["src/mlx2/runtime/does_not_exist.py"] is None
    assert any("does_not_exist" in p for p in Q.source_identity_refusals(identity, names))


@pytest.mark.parametrize("mutate,match", [
    (lambda i: i.update(commit="HEAD"), "40-hex"),
    (lambda i: i.update(commit="A" * 40), "40-hex"),
    (lambda i: i.update(dirty=0), "unknown"),
    (lambda i: i.update(status_known=None), "unknown"),
    (lambda i: i["files"].update({Q.IDENTITY_FILES[0]: "F" * 64}), "no sha256"),
    (lambda i: i["files"].update({Q.IDENTITY_FILES[0]: "f" * 63}), "no sha256"),
    (lambda i: i["files"].update({Q.IDENTITY_FILES[0]: 7}), "no sha256"),
    (lambda i: i["files"].pop(Q.IDENTITY_FILES[0]), "exactly the required set"),
    (lambda i: i["files"].update({"extra.py": "f" * 64}), "exactly the required set"),
    (lambda i: i.update(files=[]), "exactly the required set"),
])
def test_malformed_source_identity_is_refused(git, mutate, match):
    identity = Q.source_identity(Q.IDENTITY_FILES)
    mutate(identity)
    assert any(match in p for p in Q.source_identity_refusals(identity, Q.IDENTITY_FILES))
    assert Q.source_identity_refusals(None, Q.IDENTITY_FILES) == ["source identity missing"]
    assert any("required set" in p for p in Q.source_identity_refusals(Q.source_identity([]), ()))


def test_identity_changes_name_every_moved_field(git):
    before = Q.source_identity(Q.IDENTITY_FILES)
    after = json.loads(json.dumps(before))
    after["commit"] = "b" * 40
    after["files"][Q.IDENTITY_FILES[2]] = "c" * 64
    changes = Q.identity_changes("source", before, after)
    assert "source commit changed during the run" in changes
    assert f"source file {Q.IDENTITY_FILES[2]} changed during the run" in changes
    assert Q.identity_changes("source", before, None) == ["source identity before or after the run is missing"]


# ---- build identity, module origin, adapter source and identity ----

def build(tmp_path, **override):
    return {"version": "0.30.0", "package": "0.30.0", "path": str(tmp_path),
            "metallib_sha256": "b" * 64, "device": "Device(gpu, 0)", **override}


@pytest.mark.parametrize("override,match", [
    ({"metallib_sha256": None}, "metallib"),
    ({"metallib_sha256": "B" * 64}, "metallib"),
    ({"package": None}, "no package"),
    ({"version": " "}, "no version"),
    ({"device": None}, "no device"),
    ({"device": "Device(cpu, 0)"}, "not the GPU"),
    ({"path": "relative/mlx"}, "absolute directory"),
    ({"path": "/nonexistent/mlx/core"}, "absolute directory"),
])
def test_build_identity_missing_or_invalid_is_refused(tmp_path, override, match):
    assert Q.build_identity_refusals(build(tmp_path)) == []
    assert any(match in p for p in Q.build_identity_refusals(build(tmp_path, **override)))
    assert Q.build_identity_refusals(None) == ["MLX build identity missing"]


def worktree_modules():
    return {"mlx2": str(ROOT / "src/mlx2/__init__.py"),
            "mlx2.adapters": str(ROOT / "src/mlx2/adapters/__init__.py"),
            "mlx2.adapters.registry": str(ROOT / "src/mlx2/adapters/registry.py"),
            "mlx2.adapters.muse_glimmer": str(ROOT / "src/mlx2/adapters/muse_glimmer.py"),
            "scripts.paired_direct_ab": str(ROOT / "scripts/paired_direct_ab.py")}


def test_module_origin_guard(tmp_path):
    foreign = tmp_path / "site-packages" / "mlx2" / "serving.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("")
    assert Q.module_path_refusals(worktree_modules()) == []
    cases = {"mlx2.serving": str(foreign), "mlx2.ns": None, "mlx2.rel": "src/mlx2/serving.py",
             "mlx2.pyc": str(ROOT / "src/mlx2/serving.pyc"),
             "mlx2.outside_package": str(ROOT / "scripts/paired_direct_ab.py"),
             "scripts.paired_direct_ab": str(ROOT / "src/mlx2/serving.py")}
    for name, file in cases.items():
        problems = Q.module_path_refusals({**worktree_modules(), name: file})
        assert problems and name in problems[0], (name, problems)
    assert Q.module_path_refusals({"scripts.paired_direct_ab": str(ROOT / "scripts/paired_direct_ab.py")}) \
        == ["no mlx2 module is loaded"]
    fake_modules = {"mlx2": SimpleNamespace(__file__="x"), "mlx2x": SimpleNamespace(__file__="y"),
                    "mlx2.gone": None, "scripts.other": SimpleNamespace(__file__="z")}
    assert Q.loaded_module_files(fake_modules) == {"mlx2": "x"}


def test_module_closure_binds_loaded_files_and_refuses_dirty_or_drifted(git):
    closure = Q.module_closure(worktree_modules())
    assert closure["required"] == sorted(str(Path(f).relative_to(ROOT)) for f in worktree_modules().values())
    assert Q.module_closure_refusals(closure) == []
    git.status = "?? src/mlx2/adapters/registry.py\n"
    assert any("not clean" in p for p in Q.module_closure_refusals(Q.module_closure(worktree_modules())))
    source = {"files": {"src/mlx2/adapters/registry.py": "0" * 64, "src/mlx2/serving.py": "1" * 64}}
    assert Q.closure_drift_refusals(closure, source) == [
        "loaded src/mlx2/adapters/registry.py differs from its preflight hash"]
    assert Q.module_closure_refusals(None) == ["loaded module closure missing"]


def test_adapter_source_must_be_the_family_class_file_and_hash(git, tmp_path):
    manifest = {"family": "muse"}
    source = Q.source_identity(Q.IDENTITY_FILES + Q.FAMILIES["muse"]["files"])
    good = str(ROOT / "src/mlx2/adapters/muse_glimmer.py")
    name = Q.FAMILIES["muse"]["adapter"]
    assert Q.adapter_source_refusals(name, good, manifest, source) == []
    assert "dispatch selected" in Q.adapter_source_refusals("mlx2.adapters.x.Other", good, manifest, source)[0]
    other = str(ROOT / "src/mlx2/adapters/qwen38_27b.py")
    assert "not this worktree's" in Q.adapter_source_refusals(name, other, manifest, source)[0]
    assert "not this worktree's" in Q.adapter_source_refusals(name, None, manifest, source)[0]
    stale = json.loads(json.dumps(source))
    stale["files"]["src/mlx2/adapters/muse_glimmer.py"] = "0" * 64
    assert Q.adapter_source_refusals(name, good, manifest, stale) == [
        "adapter source sha256 differs from the preflight hash"]


def test_adapter_identity_muse_top_level_and_qwen_nested(tmp_path):
    path = make_artifact(tmp_path / "m")
    manifest = Q.artifact_manifest(path)
    records = [tuple(s) for s in manifest["shards"]]
    muse = SimpleNamespace(identity={"path": manifest["path"], "fingerprint": manifest["fingerprint"],
                                     "files": records, "model_type": "muse_glimmer", "cache_layout": "x"},
                           draft_model=None)
    qwen = SimpleNamespace(identity={"path": manifest["path"], "fingerprint": manifest["fingerprint"],
                                     "files": records})
    assert Q.adapter_identity_refusals(muse, manifest) == [] == Q.adapter_identity_refusals(qwen, manifest)
    cases = [({"fingerprint": "0" * 64}, "fingerprint differs"), ({"fingerprint": None}, "fingerprint differs"),
             ({"path": "/elsewhere"}, "path differs"), ({"files": records[:1]}, "shard records differ"),
             ({"files": None}, "shard records differ"),
             ({"draft_fingerprint": "d" * 64}, "external drafter")]
    for override, match in cases:
        adapter = SimpleNamespace(identity={**qwen.identity, **override})
        assert any(match in p for p in Q.adapter_identity_refusals(adapter, manifest)), override
    drafted = SimpleNamespace(identity=dict(qwen.identity), draft_model=object())
    assert any("external drafter" in p for p in Q.adapter_identity_refusals(drafted, manifest))
    assert Q.adapter_identity_refusals(SimpleNamespace(), manifest) == ["adapter has no identity mapping"]


# ---- workload binding ----

def test_prompt_file_is_hashed_and_bounded(tmp_path, monkeypatch):
    item = tmp_path / "p.json"
    item.write_text('{"prompts": [[1, 2]], "max_tokens": [1]}')
    identity, raw, problems = Q.workload_identity(item)
    assert problems == [] and identity["sha256"] == hashlib.sha256(raw).hexdigest()
    assert Q.load_workload(tmp_path / "absent.json", raw=raw)[0] == [[1, 2]]  # the preflight bytes are used
    assert Q.workload_identity(tmp_path / "absent.json")[2]
    assert Q.workload_identity(tmp_path)[2]
    monkeypatch.setattr(Q, "MAX_WORKLOAD_BYTES", 4)
    assert Q.workload_identity(item)[2]


# ---- native gate order, executed through host seams ----

class World:
    """Host stand-ins for the native seams; records every gate event in order."""

    def __init__(self, tmp_path):
        self.events = []
        self.build = build(tmp_path)
        self.build_after = None
        self.build_error = None
        self.build_error_after = None
        self.modules = worktree_modules()
        self.modules_after = None
        self.loaded_error_after = None
        self.dispatch = Q.FAMILIES["muse"]["adapter"]
        self.dispatch_error = None
        self.identity_override = {}
        self.adapter = None
        world = self

        def __init__(adapter, model):
            world.events.append("construct")
            manifest = Q.artifact_manifest(model)
            adapter.identity = {"path": manifest["path"], "fingerprint": manifest["fingerprint"],
                                "files": [tuple(s) for s in manifest["shards"]], **world.identity_override}
            adapter.model, adapter.tokenizer, adapter.environment = object(), None, {"MLX_ENABLE_TF32": "0"}
            world.adapter = adapter

        module, qualname = self.dispatch.rsplit(".", 1)
        self.cls = type(qualname, (), {"__module__": module, "__init__": __init__})

    def import_(self, name):
        self.events.append(f"import {name}")
        if name == "mlx.core":
            return SimpleNamespace(name="host stand-in, not MLX")
        if name == "mlx2.adapters.registry":
            return SimpleNamespace(resolve_adapter=self.resolve)
        if name == "mlx2.serving":
            return SimpleNamespace(generation_stop_token_ids=lambda adapter: (9,))
        return SimpleNamespace()

    def resolve(self, model, *, mtp, qualification_mode):
        self.events.append("resolve")
        assert mtp is False and qualification_mode is True
        if self.dispatch_error:
            raise self.dispatch_error
        return self.cls

    def _after_arms(self):
        return any(event.startswith("arm ") for event in self.events)

    def build_identity(self, mx):
        self.events.append("build")
        error = self.build_error_after if self._after_arms() else self.build_error
        if error is not None:
            raise error
        return dict(self.build_after if self.build_after is not None and self._after_arms() else self.build)

    def loaded(self):
        if self.loaded_error_after is not None and self._after_arms():
            raise self.loaded_error_after
        return dict(self.modules_after if self.modules_after is not None and self._after_arms() else self.modules)


@pytest.fixture
def world(tmp_path, monkeypatch, git):
    world = World(tmp_path)
    monkeypatch.setattr(Q, "_import", world.import_)
    monkeypatch.setattr(Q, "build_identity", world.build_identity)
    monkeypatch.setattr(Q, "loaded_module_files", world.loaded)
    world.artifact = make_artifact(tmp_path / "muse")
    world.prompts = tmp_path / "prompts.json"
    world.prompts.write_text(json.dumps({"prompts": [[1, 2, 3], [4, 5]], "max_tokens": [2, 2]}))
    world.out = tmp_path / "out" / "receipt.json"
    world.git = git
    return world


def native_args(world, *extra):
    return Q.resolve_args(Q.build_parser(), ["--i-own-the-gpu", "--model", str(world.artifact),
                                             "--prompt-ids", str(world.prompts), "--out", str(world.out), *extra])


def test_gates_run_in_order_before_the_model_and_bind_the_adapter(world):
    args = native_args(world)
    gate = Q.preflight(args)
    assert gate["refusals"] == [] and world.events == []  # preflight imported nothing native
    driver = Q.Driver(args, gate)
    order = ["import mlx.core", "build", "import mlx2.adapters.registry", "resolve", "construct",
             "import mlx2.serving"]
    assert [world.events.index(e) for e in order] == sorted(world.events.index(e) for e in order)
    assert driver.native_identity["adapter"] == Q.FAMILIES["muse"]["adapter"]
    assert driver.native_identity["adapter_identity"]["fingerprint"] == gate["artifact"]["fingerprint"]
    assert driver.identity["adapter_sha256"] == gate["source"]["files"]["src/mlx2/adapters/muse_glimmer.py"]
    assert driver.prompts == [[1, 2, 3], [4, 5]] and driver.stops == (9,)
    assert set(Q.ROUTE_MODULES) <= {e.removeprefix("import ") for e in world.events}


def _refusal(world, args=None):
    args = args or native_args(world)
    with pytest.raises(Q.IdentityRefusal) as caught:
        Q.Driver(args, Q.preflight(args))
    return caught.value


def test_invalid_build_refuses_before_any_mlx2_import(world):
    world.build = build(Path("/"), metallib_sha256=None)
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_BUILD and world.events == ["import mlx.core", "build"]


@pytest.mark.parametrize("error", [AttributeError("no __version__"), OSError("metallib unreadable"),
                                   RuntimeError("no default device")])
def test_build_collector_failure_refuses_before_any_mlx2_import(world, error):
    world.build_error = error
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_BUILD and world.events == ["import mlx.core", "build"]
    assert refusal.identity == {"mlx": None}
    assert refusal.refusals == [f"MLX build identity could not be collected: {type(error).__name__}: {error}"]


def test_real_collector_on_an_incomplete_module_is_a_refusal(tmp_path):
    # The real paired_direct_ab.mlx_identity, over host objects that are not MLX.
    assert Q.build_identity is not None and "World" not in repr(Q.build_identity)
    build, problems = Q.collect_build_identity(SimpleNamespace())
    assert build is None and problems[0].startswith("MLX build identity could not be collected: AttributeError")
    core = SimpleNamespace(__version__="0", __file__=str(tmp_path / "core.py"))
    build, problems = Q.collect_build_identity(core)  # no default_device
    assert build is None and "AttributeError" in problems[0]


def test_foreign_module_refuses_before_construction(world, tmp_path):
    foreign = tmp_path / "elsewhere" / "mlx2" / "__init__.py"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("")
    world.modules["mlx2"] = str(foreign)
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_DISPATCH and "construct" not in world.events
    assert any("mlx2 imported from" in p for p in refusal.refusals)


def test_wrong_dispatch_or_dispatch_error_refuses_before_construction(world):
    world.cls = type("Qwen3827BAdapter", (), {"__module__": "mlx2.adapters.qwen38_27b"})
    world.modules["mlx2.adapters.qwen38_27b"] = str(ROOT / "src/mlx2/adapters/qwen38_27b.py")
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_DISPATCH and "construct" not in world.events
    assert any("dispatch selected mlx2.adapters.qwen38_27b.Qwen3827BAdapter" in p for p in refusal.refusals)
    world.events.clear()
    world.dispatch_error = ValueError("No mlx2 adapter")
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_DISPATCH and "construct" not in world.events


def test_source_committed_between_preflight_and_import_refuses(world):
    args = native_args(world)
    gate = Q.preflight(args)
    gate["source"]["files"]["src/mlx2/adapters/registry.py"] = "0" * 64
    with pytest.raises(Q.IdentityRefusal) as caught:
        Q.Driver(args, gate)
    assert caught.value.stage == Q.STAGE_DISPATCH and "construct" not in world.events
    assert any("registry.py differs from its preflight hash" in p for p in caught.value.refusals)


def test_adapter_identity_mismatch_refuses_after_load_before_any_arm(world):
    world.identity_override = {"fingerprint": "0" * 64}
    refusal = _refusal(world)
    assert refusal.stage == Q.STAGE_ADAPTER and "import mlx2.serving" not in world.events


def test_driver_refuses_without_a_clean_preflight_before_importing_mlx(world):
    args = native_args(world)
    for gate in (None, {"refusals": ["x"], "artifact": {}}, {"refusals": [], "artifact": None}):
        with pytest.raises(Q.IdentityRefusal) as caught:
            Q.Driver(args, gate)
        assert caught.value.stage == Q.STAGE_PREFLIGHT
    assert world.events == []


def test_source_order_of_the_native_gates():
    init = ast.parse(textwrap.dedent(inspect.getsource(Q.Driver.__init__))).body[0]
    assert not [n for n in ast.walk(init) if isinstance(n, (ast.Import, ast.ImportFrom))]  # only the seam
    calls = {}
    for node in ast.walk(init):
        if isinstance(node, ast.Call):
            calls.setdefault(ast.unparse(node), node.lineno)
    order = ["_import('mlx.core')", "collect_build_identity(mx)",
             "_import('mlx2.adapters.registry')",
             "registry.resolve_adapter(args.model, mtp=False, qualification_mode=True)",
             "module_closure_refusals(before_adapter)", "cls(args.model)",
             "adapter_identity_refusals(self.adapter, manifest)", "_import('mlx2.serving')",
             "module_closure_refusals(before_arms)"]
    lines = [calls[c] for c in order]
    assert lines == sorted(lines) and len(set(lines)) == len(lines)
    run_all = ast.parse(textwrap.dedent(inspect.getsource(Q.run_all))).body[0]
    first = {}
    for node in ast.walk(run_all):
        if isinstance(node, ast.Call):
            first.setdefault(ast.unparse(node.func), node.lineno)
    assert first["preflight"] < first["refused_receipt"] < first["Driver"] < first["driver.run"] \
        < first["checked_post_run_identity"]
    # Only identity is guarded: run_all's one try wraps Driver construction and
    # catches IdentityRefusal alone; nothing wraps the arms.
    tries = [n for n in ast.walk(run_all) if isinstance(n, ast.Try)]
    assert len(tries) == 1 and [ast.unparse(s) for s in tries[0].body] == ["driver = Driver(args, gate)"]
    assert [ast.unparse(h.type) for h in tries[0].handlers] == ["IdentityRefusal"]
    guard = ast.parse(textwrap.dedent(inspect.getsource(Q.checked_post_run_identity))).body[0]
    tries = [n for n in ast.walk(guard) if isinstance(n, ast.Try)]
    assert len(tries) == 1 and [ast.unparse(s) for s in tries[0].body] == [
        "return post_run_identity(args, gate, driver)"]
    assert [ast.unparse(h.type) for h in tries[0].handlers] == ["COLLECTION_ERRORS"]


# ---- run_all and main: refused receipts and post-run drift ----

ROUNDS = {"rounds": 0, "proposed": 0, "accepted": 0, "rejected_rounds": 0,
          "partial_accept_rounds": 0, "rollback_then_append": False}


@pytest.fixture
def arms(world, monkeypatch):
    """Host arms: real run_all bookkeeping, synthetic lane records, an optional mid-run action."""
    world.during = None

    def run(driver, arm, lanes, **kwargs):
        world.events.append(f"arm {arm}")
        if arm == "pld_batched" and world.during:
            world.during(world)
        tokens = [1, 2]
        out = {i: {"tokens": list(tokens), "finish_reason": "length", "covered_tokens": None,
                   "execution_widths": [len(lanes)] if arm.startswith("pld") else "not reported",
                   **Q.lane_row_evidence([], tokens, driver.args.logprob_rows),
                   "final_state": dict(UNAVAILABLE), "boundary": None, "rounds": dict(ROUNDS),
                   "_final": None} for i in lanes}
        return out, {}, [], None

    monkeypatch.setattr(Q.Driver, "run", run)
    monkeypatch.setattr(Q.Driver, "continuation",
                        lambda driver, lane, record: {"status": "unavailable", "reason": "host test"})
    return world


def test_clean_run_is_executed_and_keeps_its_own_verdict(arms):
    record = Q.run_all(native_args(arms))
    assert record["executed"] is True and record["arms_executed"] == list(Q.ARMS)
    assert record["identity_refusals"] == [] and record["refused_at"] is None
    assert record["verdict"] == "coverage_refused"  # synthetic lanes cover nothing; never 'pass'
    assert record["identity"]["after"]["artifact"]["fingerprint"] == record["identity"]["preflight"]["artifact"]["fingerprint"]
    assert "not acquired or verified" in record["gpu_ownership"]
    assert "workload_raw" not in record["identity"]["preflight"]
    json.dumps(record, default=sorted)


def _touch_shard(world):
    item = world.artifact / SHARDS[0]
    stat = item.stat()
    os.utime(item, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5000))


def _rewrite_prompts(world):
    world.prompts.write_text(json.dumps({"prompts": [[1, 2, 3], [4, 5]], "max_tokens": [2, 1]}))


def _move_adapter(world):
    world.adapter.identity["fingerprint"] = "0" * 64


@pytest.mark.parametrize("during,match", [
    (_touch_shard, "artifact fingerprint changed during the run"),
    (lambda w: setattr(w.git, "head", "b" * 40), "source commit changed during the run"),
    (lambda w: setattr(w.git, "status", " M scripts/qualify_ragged_pld.py\n"), "source dirty changed"),
    (_rewrite_prompts, "the --prompt-ids file changed during the run"),
    (lambda w: setattr(w, "build_after", {**w.build, "metallib_sha256": "c" * 64}),
     "MLX build identity changed during the run"),
    (lambda w: setattr(w, "modules_after", {**w.modules, "mlx2.runtime.pld": "/elsewhere/mlx2/runtime/pld.py"}),
     "mlx2.runtime.pld imported from"),
    (lambda w: setattr(w, "modules_after", {k: v for k, v in w.modules.items() if k != "mlx2.adapters.registry"}),
     "registry.py changed or unloaded"),
    (_move_adapter, "adapter artifact identity changed during the run"),
])
def test_drift_during_the_arms_refuses_and_keeps_the_real_results(arms, during, match):
    clean = Q.run_all(native_args(arms))
    arms.events.clear()
    arms.during = during
    record = Q.run_all(native_args(arms))
    assert record["verdict"] == "refused" and record["refused_at"] == Q.STAGE_POST_RUN
    assert record["executed"] is True and record["arms_executed"] == list(Q.ARMS)
    assert any(match in p for p in record["identity_refusals"]), record["identity_refusals"]
    assert any(p.startswith("identity: ") for p in record["refusals"])
    assert record["results"] == clean["results"] and record["comparisons"] == clean["comparisons"]


@pytest.mark.parametrize("during,match", [
    (lambda w: setattr(w, "build_error_after", OSError("metallib unreadable")),
     "after the arms: MLX build identity could not be collected: OSError: metallib unreadable"),
    (lambda w: setattr(w, "loaded_error_after", RuntimeError("dictionary changed size during iteration")),
     "post-run identity could not be collected: RuntimeError: dictionary changed size during iteration"),
])
def test_post_run_collection_failure_refuses_and_keeps_the_real_results(arms, during, match):
    clean = Q.run_all(native_args(arms))
    arms.events.clear()
    arms.during = during
    record = Q.run_all(native_args(arms))
    assert record["verdict"] == "refused" and record["refused_at"] == Q.STAGE_POST_RUN
    assert record["executed"] is True and record["arms_executed"] == list(Q.ARMS)
    assert match in record["identity_refusals"], record["identity_refusals"]
    assert record["results"] == clean["results"] and record["comparisons"] == clean["comparisons"]
    assert record["coverage"] == clean["coverage"]
    json.dumps(record, default=sorted)


def test_preflight_refusal_writes_an_unexecuted_receipt_without_native_import(world, monkeypatch, capsys):
    world.git.fail = {"rev-parse"}
    monkeypatch.setattr(Q.Driver, "__init__", lambda *a, **k: pytest.fail("Driver must not be constructed"))
    code = Q.main(["--i-own-the-gpu", "--model", str(world.artifact), "--prompt-ids", str(world.prompts),
                   "--out", str(world.out)])
    record = json.loads(world.out.read_text())
    assert code == 1 and world.events == []
    assert record["executed"] is False and record["arms_executed"] == [] and record["verdict"] == "refused"
    assert record["refused_at"] == Q.STAGE_PREFLIGHT and record["results"] == {} and record["comparisons"] == []
    assert record["coverage"] is None and record["ordinary_geometry"] == "not_run"
    assert record["protocol"]["stop_tokens"] == "not resolved (no adapter loaded)"
    assert any("git failed" in p for p in record["refusals"])
    assert record["identity"]["preflight"]["source"]["commit"] is None
    out = capsys.readouterr().out
    assert '"executed": false' in out and Q.STAGE_PREFLIGHT in out and "vs" not in out


@pytest.mark.parametrize("prepare,match", [
    (lambda w: (w.artifact / "config.json").write_text(json.dumps(CONFIGS["north"])), "header digests"),
    (lambda w: (w.artifact / SHARDS[1]).unlink(), "missing weight shard"),
    (lambda w: w.prompts.unlink(), "--prompt-ids"),
    (lambda w: setattr(w.git, "tracked", False), "not all tracked"),
])
def test_run_all_preflight_refusals_never_import_native(world, prepare, match):
    prepare(world)
    record = Q.run_all(native_args(world))
    assert record["executed"] is False and record["refused_at"] == Q.STAGE_PREFLIGHT
    assert any(match in p for p in record["refusals"]) and world.events == []


def test_driver_refusal_becomes_an_unexecuted_receipt(arms, capsys):
    arms.identity_override = {"path": "/elsewhere"}
    code = Q.main(["--i-own-the-gpu", "--model", str(arms.artifact), "--prompt-ids", str(arms.prompts),
                   "--out", str(arms.out)])
    record = json.loads(arms.out.read_text())
    assert code == 1 and record["executed"] is False and record["refused_at"] == Q.STAGE_ADAPTER
    assert not [e for e in arms.events if e.startswith("arm ")]
    assert record["identity"]["adapter_identity"]["path"] == "/elsewhere"
    assert Q.STAGE_ADAPTER in capsys.readouterr().out


def test_gpu_assertion_is_still_required_and_not_claimed_as_enforcement(capsys):
    with pytest.raises(SystemExit):
        Q.resolve_args(Q.build_parser(), ["--model", "/m", "--out", "/dev/null"])
    assert "--i-own-the-gpu" in capsys.readouterr().err
    assert "not acquired or verified" in Q.GPU_OWNERSHIP and "admission wrapper" in Q.GPU_OWNERSHIP


def test_tiny_preflight_records_without_enforcing(git):
    git.fail = {"rev-parse", "status", "ls-files"}
    args = Q.resolve_args(Q.build_parser(), ["--tiny", "--out", "/dev/null"])
    gate = Q.preflight(args)  # tiny itself is never executed here
    assert gate["enforced"] is False and gate["artifact"] is None and gate["refusals"]
    run_all = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(Q.run_all))).body[0])
    assert "if gate['enforced'] and gate['refusals']:" in run_all
    assert "if enforced and identity_refusals:" in run_all
