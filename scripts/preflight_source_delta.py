"""Verified source changes over a historical full preflight; CPU-only tests.

The historical source is reconstructed from immutable Git objects and must
exactly hash to the full receipt's runtime digest. Neither a caller-supplied
file list nor an unsigned pass flag substitutes for the impacted-test closure.
See provenance/source-preflight-delta-20261022.json.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

SCHEMA = "mlx2.qualification-preflight-source-delta.v1"
# Historical approved producer; its original full-suite verdict stays intact.
PROFILE_PATH = "qualification/source_preflight_profiles/sage-a15-20261022.json"
PROFILE_SHA256 = "0a903812f14a2b4da47aa22c1518ec214799148c584f9fce44b44604fd8a625f"
APPROVAL_PATHS = frozenset({"src/mlx2/qualification.py"})
BASE_HARNESSES = frozenset({"b7212bf33b2d06bc191cfc7e585ebdf208f33c0cba861b38aa3040b4ce6560a4"})


def runtime_source(root):
    from mlx2.source_dependencies import add_resources

    root = Path(root) / "src/mlx2"
    digest = hashlib.sha256()
    sources = sorted(root.rglob("*.py"))
    if not sources:
        raise AssertionError("source delta has no runtime sources")
    files = {}
    resources = {}
    for path in sources:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(name.encode()); digest.update(data)
        files["src/mlx2/" + name] = hashlib.sha256(data).hexdigest()
        resources[name] = ""
    add_resources(root, resources)
    data = {name: value for name, value in resources.items() if not name.endswith(".py")}
    digest.update(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    files.update({"src/mlx2/" + name: value for name, value in data.items()})
    return digest.hexdigest(), files


def revision_source(root, revision):
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise AssertionError("source delta needs an immutable 40-character Git commit")
    archive = subprocess.check_output(["git", "archive", "--format=tar", revision, "src/mlx2"], cwd=root)
    with tempfile.TemporaryDirectory(prefix="mlx2-preflight-base-") as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(directory, filter="data")
        return runtime_source(directory)


def approved_profile(root):
    path = Path(root) / PROFILE_PATH
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != PROFILE_SHA256:
        raise AssertionError("source delta reviewed coverage profile is absent or changed")
    return json.loads(path.read_text())


def verify_pin_only_change(root, revision):
    path = "src/mlx2/qualification.py"
    old = subprocess.check_output(["git", "show", revision + ":" + path], cwd=root, text=True)
    new = (Path(root) / path).read_text()
    trees = [ast.parse(text) for text in (old, new)]
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "APPROVED_QUALIFICATION_HARNESS" for t in node.targets):
                if not isinstance(node.value, ast.Dict):
                    raise AssertionError("qualification approval pin is not a literal mapping")
                for i, key in enumerate(node.value.keys):
                    if isinstance(key, ast.Constant) and key.value == "sha256":
                        node.value.values[i] = ast.Constant(value="<approval-pin>")
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise AssertionError("source delta profile permits only the qualification harness pin change")


def plan(h, base_path, revision, expected, tree_fn, root):
    base_path = Path(base_path).resolve()
    base = json.loads(base_path.read_text())
    identity = base.get("identity") or {}
    h._validate_full_preflight(base_path, identity.get("runtime"), identity)
    if identity.get("qualification_harness", {}).get("sha256") not in BASE_HARNESSES:
        raise AssertionError("source delta base producer is not an approved historical harness")
    if base.get("source_stable") is not True or base.get("identity_before") != identity:
        raise AssertionError("source delta base source was not stable during its full run")
    tree = base.get("tree")
    if not isinstance(tree, dict) or h.preflight_tree_sha256(tree) != identity.get("test_source_sha256"):
        raise AssertionError("source delta base per-file test tree is absent or corrupt")
    before_runtime, now_runtime = identity.get("runtime", {}), expected.get("runtime", {})
    if {k:v for k,v in before_runtime.items() if k != "source_sha256"} != {k:v for k,v in now_runtime.items() if k != "source_sha256"}:
        raise AssertionError("source delta cannot cross native build, dependencies or host runtime changes")
    if identity.get("pytest_config_sha256") != expected.get("pytest_config_sha256"):
        raise AssertionError("source delta cannot cross pytest configuration changes")
    digest, before_source = revision_source(root, revision)
    if digest != before_runtime.get("source_sha256"):
        raise AssertionError("Git reconstruction does not match the full preflight runtime digest")
    digest, current_source = runtime_source(root)
    if digest != now_runtime.get("source_sha256"):
        raise AssertionError("current source manifest does not match the active runtime digest")
    current = tree_fn()
    changes = h.preflight_tree_changes({**tree, **before_source}, {**current, **current_source})
    profile = approved_profile(root)
    if profile.get("base_source_revision") != revision:
        raise AssertionError("source delta reviewed coverage uses a different base")
    combined = {**current, **current_source}
    impacted = set(h.PREFLIGHT_IMPORT_GUARD_MODULES)
    for path in (item for group in changes.values() for item in group):
        entry = profile.get("files", {}).get(path)
        if not isinstance(entry, dict) or not entry.get("tests"):
            raise AssertionError("source delta has no reviewed coverage for " + path)
        if path not in APPROVAL_PATHS and combined.get(path) != entry.get("after_sha256"):
            raise AssertionError("source changed outside reviewed coverage: " + path)
        if path == "src/mlx2/qualification.py":
            verify_pin_only_change(root, revision)
        for test in entry["tests"]:
            if not h._is_test_module(test) or not (Path(root) / test).is_file():
                raise AssertionError("source delta profile names an absent test module")
            impacted.add(test)
        if h._is_test_module(path) and path in combined:
            impacted.add(path)
    lexical = set(h.preflight_impacted_tests(changes, combined, root=root))
    if not lexical <= impacted:
        raise AssertionError("source delta profile omits impacted tests: " + ", ".join(sorted(lexical - impacted)))
    impacted = sorted(impacted)
    return {"tree": current, "source_tree": current_source, "changes": changes, "impacted": impacted}


def write(h, path, base_path, revision, *, run=subprocess.run, identity_fn=None, tree_fn=None):
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    root = Path(h.__file__).resolve().parents[1]
    identity_fn = identity_fn or h.preflight_identity
    tree_fn = tree_fn or h.preflight_tree
    expected = identity_fn()
    before = plan(h, base_path, revision, expected, tree_fn, root)
    ordinary, guarded = h._delta_test_commands(before["impacted"])
    results = []
    for name, command in (("ordinary", ordinary), ("guards", guarded)):
        started = time.monotonic()
        result = run(command, cwd=root, env={**os.environ, **h.PREFLIGHT_CPU_ENV}, capture_output=True, text=True) if command else None
        text = "" if result is None else result.stdout + result.stderr
        log = Path(path).with_suffix("." + name + ".log")
        with log.open("x") as handle:
            handle.write(text)
        results.append({"command": command, "returncode": None if result is None else result.returncode, "log": str(log.resolve()), "sha256": hashlib.sha256(text.encode()).hexdigest(), "seconds": time.monotonic()-started})
    after_identity = identity_fn()
    after = plan(h, base_path, revision, after_identity, tree_fn, root)
    passed = before == after and expected == after_identity and all(h._lane_passed(x["command"], x["returncode"]) for x in results)
    receipt = {"schema": SCHEMA, "passed": passed, "timestamp": time.time(), "identity": expected,
               "base": {"path": str(Path(base_path).resolve()), "sha256": hashlib.sha256(Path(base_path).read_bytes()).hexdigest(), "source_revision": revision},
               **before, "test_command": ordinary, "guard_test_command": guarded, "returncode": results[0]["returncode"], "guard_returncode": results[1]["returncode"], "lanes": results, "source_stable": before == after and expected == after_identity, "execution_environment": dict(h.PREFLIGHT_CPU_ENV)}
    h._write_receipt_exclusive(path, receipt)
    if not passed:
        raise AssertionError("focused source preflight failed or source changed during testing: " + str(path))
    return receipt


def validate(h, path, receipt, expected, tree_fn):
    if receipt.get("passed") is not True or receipt.get("source_stable") is not True:
        raise AssertionError("source delta did not pass with stable source")
    base = receipt.get("base") or {}
    base_path = Path(base.get("path", ""))
    if not base_path.is_file() or hashlib.sha256(base_path.read_bytes()).hexdigest() != base.get("sha256"):
        raise AssertionError("source delta base receipt is absent or changed")
    interpreter = h.interpreter_identity()
    if h.preflight_binding(receipt.get("identity") or {}) != h.preflight_binding(expected) or not h._same_local_interpreter(receipt.get("identity", {}).get("interpreter"), interpreter):
        raise AssertionError("source delta does not match the active source, harness or interpreter")
    root = Path(h.__file__).resolve().parents[1]
    current = plan(h, base_path, base.get("source_revision"), expected, tree_fn, root)
    if any(receipt.get(key) != value for key,value in current.items()):
        raise AssertionError("source delta omitted or changed source/test coverage")
    ordinary, guarded = h._delta_test_commands(current["impacted"])
    for command, field, code in ((ordinary, "test_command", "returncode"), (guarded, "guard_test_command", "guard_returncode")):
        if h._command_args(receipt.get(field)) != h._command_args(command) or not h._lane_passed(command, receipt.get(code)) or command is not None and not h._ran_under(receipt.get(field), interpreter):
            raise AssertionError("source delta commands, interpreter or return codes do not match impacted tests")
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "identity": expected, "test_command": ordinary, "guard_test_command": guarded, "passed": True, "source_delta": {"base": base, "changes": current["changes"], "impacted": current["impacted"]}}
