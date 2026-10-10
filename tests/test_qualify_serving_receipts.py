import ast
import importlib.util
import hashlib
import math
import json
import copy
import os
import shutil
from types import SimpleNamespace
from pathlib import Path

import platform
import sys

import pytest

from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qualify_serving", ROOT / "scripts" / "qualify_serving.py")
qualify = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qualify)


def test_qualification_harness_receipt_binds_portable_script_identity():
    receipt = qualify.qualification_harness_identity()
    assert receipt == {
        "schema": "mlx2.qualification-harness.v1",
        "name": "scripts/qualify_serving.py",
        "sha256": hashlib.sha256(
            (ROOT / "scripts" / "qualify_serving.py").read_bytes()
        ).hexdigest(),
    }
    assert not receipt["name"].startswith("/")
    assert receipt == APPROVED_QUALIFICATION_HARNESS


def test_preflight_receipt_is_bound_to_git_runtime_tests_and_harness(tmp_path):
    identity = {
        "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
        "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
        "interpreter": qualify.interpreter_identity(),
    }
    path = tmp_path / "preflight.json"
    receipt = qualify.write_preflight_receipt(
        path,
        run=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="42 passed", stderr=""),
        identity_fn=lambda: identity,
    )
    assert receipt["passed"] and receipt["schema"] == qualify.PREFLIGHT_SCHEMA
    assert receipt["returncode"] == receipt["guard_returncode"] == 0
    assert receipt["test_command"] != receipt["guard_test_command"]
    evidence = qualify.validate_preflight_receipt(
        path, identity["runtime"], identity_fn=lambda: identity
    )
    assert evidence["passed"] and evidence["identity"] == identity
    assert evidence["guard_test_command"] == receipt["guard_test_command"]


def test_preflight_wires_available_owned_tensorfold_sources(tmp_path, monkeypatch):
    tensorfold = tmp_path / "tensorfold"
    mlx_lm = tmp_path / "mlx-lm"
    tensorfold.mkdir()
    mlx_lm.mkdir()
    monkeypatch.setattr(qualify, "PREFLIGHT_TENSORFOLD_OWNED_SOURCE", tensorfold)
    monkeypatch.setattr(
        qualify, "PREFLIGHT_TENSORFOLD_OWNED_MLX_LM_SOURCE", mlx_lm
    )
    monkeypatch.delenv("MLX2_TEST_TENSORFOLD_OWNED_SOURCE", raising=False)
    monkeypatch.delenv(
        "MLX2_TEST_TENSORFOLD_OWNED_MLX_LM_SOURCE", raising=False
    )
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="passed", stderr="")

    qualify.write_preflight_receipt(
        tmp_path / "preflight.json",
        run=run,
        identity_fn=lambda: {"runtime": {}},
    )
    assert len(calls) == 2
    assert all(
        env["MLX2_TEST_TENSORFOLD_OWNED_SOURCE"] == str(tensorfold)
        and env["MLX2_TEST_TENSORFOLD_OWNED_MLX_LM_SOURCE"] == str(mlx_lm)
        for env in calls
    )


def test_preflight_identity_binds_pytest_configuration():
    identity = qualify.preflight_identity(runtime_identity_fn=lambda: {"source_sha256": "test"})
    assert identity["pytest_config_sha256"] == hashlib.sha256(
        (ROOT / "pyproject.toml").read_bytes()
    ).hexdigest()


def test_preflight_identity_test_source_digest_covers_the_whole_tree(monkeypatch):
    # Review round 1: the cross-host (v3) receipt binds the identity only, and
    # its test_source_sha256 covered tests/**/*.py alone, so a scripts fixture
    # or provenance change did not move it.  The key keeps its name for the
    # v3 schema; its value is the digest of the whole preflight tree.
    tree = qualify.preflight_tree()
    moved = {**tree, "scripts/fixtures/qwen4_ple_adaptive_no_warm_policy.json": "0" * 64}
    seen = []
    for current in (tree, moved):
        monkeypatch.setattr(qualify, "preflight_tree", lambda root=None, current=current: current)
        identity = qualify.preflight_identity(runtime_identity_fn=lambda: {"source_sha256": "t"})
        assert identity["test_source_sha256"] == qualify.preflight_tree_sha256(current)
        seen.append(identity["test_source_sha256"])
    assert seen[0] != seen[1]


def test_unimplemented_generic_http_gates_are_machine_visible():
    coverage = qualify.QUALIFICATION_COVERAGE
    assert coverage["strict_json_schema"] is True
    assert coverage["response_format_json_object"] is False
    assert coverage["physical_n2"] is False
    assert coverage["overload_429_retry_after"] is False
    assert coverage["latency_ttft_itl_percentiles"] is False
    assert coverage["tenant_jain_fairness"] is False
    assert coverage["progress_events"] is False


@pytest.mark.parametrize("mutation", ["failed", "identity", "active_runtime",
                                      "returncode_false", "guard_returncode_false",
                                      "empty_executable", "guard_empty_executable"])
def test_preflight_receipt_rejects_failed_or_mismatched_evidence(tmp_path, mutation):
    identity = {
        "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
        "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
        "interpreter": qualify.interpreter_identity(),
    }
    ordinary, guarded = qualify.preflight_test_commands()
    receipt = {"schema": qualify.PREFLIGHT_SCHEMA, "passed": True, "identity": identity,
               "test_command": ordinary, "guard_test_command": guarded,
               "returncode": 0, "guard_returncode": 0}
    if mutation == "failed":
        receipt["passed"] = False
    elif mutation == "returncode_false":
        receipt["returncode"] = False  # review round 3: JSON false == 0 in Python
    elif mutation == "guard_returncode_false":
        receipt["guard_returncode"] = False
    elif mutation == "empty_executable":
        receipt["test_command"] = ["", *ordinary[1:]]  # review round 4
    elif mutation == "guard_empty_executable":
        receipt["guard_test_command"] = ["", *guarded[1:]]
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(receipt))
    current = identity if mutation != "identity" else {**identity, "test_source_sha256": "changed"}
    active = identity["runtime"] if mutation != "active_runtime" else {"source_sha256": "other"}
    with pytest.raises(AssertionError, match="preflight receipt"):
        qualify.validate_preflight_receipt(path, active, identity_fn=lambda: current)


@pytest.mark.parametrize(
    "pytest_args",
    [
        ["--collect-only"],
        ["--co", "-q"],
        ["-k", "sdk_smoke"],
        ["tests/test_sdk_smoke.py"],
        ["--lf"],
        ["--deselect", "tests/test_serving_contract.py"],
        ["--ignore=tests/test_peer_pr_regressions.py"],
    ],
)
def test_preflight_receipt_must_have_run_the_full_unit_suite(tmp_path, pytest_args):
    identity = {
        "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
        "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
        "interpreter": qualify.interpreter_identity(),
    }
    # A collect-only or scoped run exits 0 and writes passed=True: the
    # receipt looks exactly like a full-suite pass unless its command is read.
    ran = []

    def run(command, **kwargs):
        ran.append(command)
        return SimpleNamespace(returncode=0, stdout="no tests ran", stderr="")

    scoped = tmp_path / "scoped.json"
    qualify.write_preflight_receipt(
        scoped, pytest_args=pytest_args, run=run, identity_fn=lambda: identity
    )
    with pytest.raises(AssertionError, match="full unit suite"):
        qualify.validate_preflight_receipt(
            scoped, identity["runtime"], identity_fn=lambda: identity
        )
    full = tmp_path / "full.json"
    qualify.write_preflight_receipt(full, run=run, identity_fn=lambda: identity)
    evidence = qualify.validate_preflight_receipt(
        full, identity["runtime"], identity_fn=lambda: identity
    )
    # The serving receipt's unit_tests evidence names the command it trusts.
    assert evidence["passed"] and evidence["test_command"] == ran[-2]
    ordinary, guarded = qualify.preflight_test_commands()
    assert ran[-2:] == [ordinary, guarded]
    assert evidence["guard_test_command"] == guarded


# --------------------------------------------------------------------------
# Scoped preflight deltas (2026-10-09): a tests/scripts change reruns only the
# test modules it can reach; the binding (src/, mlx build, harness, pytest
# configuration) still demands the full suite.
# --------------------------------------------------------------------------

_BOUND = {
    "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
    "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
    "pytest_config_sha256": "config", "interpreter": qualify.interpreter_identity(),
}


def _passing_run(ran=None):
    def run(command, **kwargs):
        if ran is not None:
            ran.append(command)
        return SimpleNamespace(returncode=0, stdout="passed", stderr="")
    return run


def _full_base(tmp_path, tree):
    path = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(
        path, run=_passing_run(), identity_fn=lambda: _BOUND, tree_fn=lambda: tree
    )
    return path


def _edited(tree, name):
    return {**tree, name: "0" * 64}


def test_full_preflight_survives_a_commit_that_changes_no_bound_file(tmp_path):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    moved = {**_BOUND, "git": {"revision": "def"}}  # docs-only commit
    evidence = qualify.validate_preflight_receipt(
        base, _BOUND["runtime"], identity_fn=lambda: moved, tree_fn=lambda: tree
    )
    assert evidence["passed"] and "delta" not in evidence


def test_full_preflight_refuses_a_changed_test_tree_and_names_the_delta(tmp_path):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    with pytest.raises(AssertionError, match="--preflight-delta"):
        qualify.validate_preflight_receipt(
            base, _BOUND["runtime"], identity_fn=lambda: _BOUND,
            tree_fn=lambda: _edited(tree, "scripts/sdk_smoke.py"),
        )


def test_preflight_delta_reruns_only_impacted_modules_and_validates(tmp_path):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    current = _edited(tree, "scripts/sdk_smoke.py")
    ran = []
    delta_path = tmp_path / "delta.json"
    delta = qualify.write_preflight_delta(
        delta_path, base, run=_passing_run(ran), identity_fn=lambda: _BOUND,
        tree_fn=lambda: current,
    )
    assert delta["changes"]["modified"] == ["scripts/sdk_smoke.py"]
    assert "tests/test_sdk_smoke.py" in delta["impacted"]
    # The closure over-selects (the harness names the script, and many tests
    # name the harness) but stays well short of the suite.
    suite = [p for p in tree if qualify._is_test_module(p)]
    assert len(suite) > 100 and len(delta["impacted"]) < len(suite) // 4
    # The closure reaches import-guard modules too (through tests/mlx_blocker.py),
    # which run under the guard command in their own interpreters.
    guards = set(qualify.PREFLIGHT_IMPORT_GUARD_MODULES)
    assert ran == [delta["test_command"], delta["guard_test_command"]]
    assert ran[0][1:3] == ["-m", "pytest"]
    assert ran[0][3:] == [m for m in delta["impacted"] if m not in guards]
    assert set(ran[1]) & guards == set(delta["impacted"]) & guards != set()
    evidence = qualify.validate_preflight_receipt(
        delta_path, _BOUND["runtime"], identity_fn=lambda: _BOUND, tree_fn=lambda: current
    )
    assert evidence["passed"] and evidence["delta"]["impacted"] == delta["impacted"]
    # The full-suite commands it stands on are the base's.
    assert evidence["test_command"] == json.loads(base.read_text())["test_command"]


@pytest.mark.parametrize("mutation", ["binding", "conftest"])
def test_preflight_delta_refuses_what_needs_the_full_suite(tmp_path, mutation):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    identity, current = _BOUND, _edited(tree, "scripts/sdk_smoke.py")
    if mutation == "binding":
        identity = {**_BOUND, "runtime": {"source_sha256": "src-changed"}}
    else:
        current = _edited(tree, "tests/conftest.py")
    with pytest.raises(AssertionError, match="full preflight"):
        qualify.write_preflight_delta(
            tmp_path / "delta.json", base, run=_passing_run(),
            identity_fn=lambda: identity, tree_fn=lambda: current,
        )


@pytest.mark.parametrize("mutation", ["trimmed", "stale", "base_edited", "failed",
                                      "returncode_missing", "guard_returncode_missing",
                                      "returncode_false", "guard_returncode_false",
                                      "executable_not_str"])
def test_preflight_delta_validation_fails_closed(tmp_path, mutation):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    current = _edited(tree, "scripts/sdk_smoke.py")
    delta_path = tmp_path / "delta.json"
    qualify.write_preflight_delta(
        delta_path, base, run=_passing_run(), identity_fn=lambda: _BOUND,
        tree_fn=lambda: current,
    )
    receipt = json.loads(delta_path.read_text())
    check_tree = current
    if mutation == "trimmed":
        receipt["impacted"] = receipt["impacted"][:1]
        receipt["test_command"] = receipt["test_command"][:3] + receipt["impacted"]
    elif mutation == "stale":
        check_tree = _edited(current, "tests/test_sdk_smoke.py")
    elif mutation == "base_edited":
        base.write_text(base.read_text().replace('"passed": true', '"passed": true ', 1))
    elif mutation == "returncode_missing":
        # Review round 2: a hand-written delta with the exact commands,
        # passed: true and no return code carried no evidence the run happened.
        assert receipt["test_command"]
        receipt["returncode"] = None
    elif mutation == "guard_returncode_missing":
        assert receipt["guard_test_command"]
        del receipt["guard_returncode"]
    elif mutation == "returncode_false":
        receipt["returncode"] = False  # JSON false == 0 in Python
    elif mutation == "guard_returncode_false":
        receipt["guard_returncode"] = False
    elif mutation == "executable_not_str":
        receipt["test_command"][0] = 0
    else:
        receipt["passed"] = False
    delta_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="preflight delta"):
        qualify.validate_preflight_receipt(
            delta_path, _BOUND["runtime"], identity_fn=lambda: _BOUND,
            tree_fn=lambda: check_tree,
        )


@pytest.mark.parametrize("code", [None, False])
def test_preflight_receipt_writer_fails_a_lane_without_a_zero_code(tmp_path, code):
    # Review round 4: the full-receipt writer compared with ==, so a lane
    # returning False (or None == None) was stamped passed: True.
    def run(command, **kwargs):
        return SimpleNamespace(returncode=code, stdout="", stderr="")

    path = tmp_path / "preflight.json"
    with pytest.raises(AssertionError, match="import guards failed"):
        qualify.write_preflight_receipt(path, run=run, identity_fn=lambda: _BOUND)
    assert json.loads(path.read_text())["passed"] is False


@pytest.mark.parametrize("code", [None, False])
def test_preflight_delta_writer_fails_a_commanded_lane_without_a_zero_code(tmp_path, code):
    # Review round 3: the writer stamped passed: True for a lane whose run
    # returned no code (or JSON false), which the validator then refused.
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    current = _edited(tree, "scripts/sdk_smoke.py")

    def run(command, **kwargs):
        return SimpleNamespace(returncode=code, stdout="", stderr="")

    delta_path = tmp_path / "delta.json"
    with pytest.raises(AssertionError, match="impacted tests failed"):
        qualify.write_preflight_delta(
            delta_path, base, run=run, identity_fn=lambda: _BOUND, tree_fn=lambda: current
        )
    assert json.loads(delta_path.read_text())["passed"] is False


def test_impacted_tests_select_by_name_and_include_changed_tests(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts").mkdir()
    files = {
        "tests/test_a.py": "spec = 'scripts/tool_x.py'",
        "tests/test_b.py": "from helper_y import thing",
        "tests/test_c.py": "data = open('fixture_z.json')",
        "tests/test_d.py": "unrelated = 'tool_xy'",
        "tests/helper_y.py": "thing = 1",
        "tests/fixture_z.json": "{}",
        "scripts/tool_x.py": "pass",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    tree = {name: "h" for name in files}

    def impacted(*changed, kind="modified"):
        changes = {"added": [], "modified": [], "removed": []}
        changes[kind] = list(changed)
        return qualify.preflight_impacted_tests(changes, tree, root=tmp_path)

    assert impacted("scripts/tool_x.py") == ["tests/test_a.py"]  # not tool_xy
    assert impacted("tests/helper_y.py") == ["tests/test_b.py"]
    assert impacted("tests/fixture_z.json") == ["tests/test_c.py"]
    assert impacted("tests/test_d.py") == ["tests/test_d.py"]
    assert impacted("scripts/tool_x.py", kind="removed") == ["tests/test_a.py"]


def _tmp_tree(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return {name: "h" for name in files}


def _impacted_in(tmp_path, tree, *changed, kind="modified"):
    changes = {"added": [], "modified": [], "removed": []}
    changes[kind] = list(changed)
    return qualify.preflight_impacted_tests(changes, tree, root=tmp_path)


def test_impacted_selection_is_the_transitive_reverse_closure(tmp_path):
    # Sweep 2026-10-09 (delta-underselect): the scan used to be one hop and
    # left changed test modules out of the consumer scan, so the 30 modules
    # that import from tests/test_batched_mtp.py were not rerun when it
    # changed, a script imported by another script selected nothing, and a
    # helper->helper->test chain selected nothing.
    tree = _tmp_tree(tmp_path, {
        "tests/test_shared.py": "def tiny():\n    return 1\n",
        "tests/test_user.py": "from test_shared import tiny\n",
        "tests/test_user_user.py": "from test_user import tiny\n",
        "tests/helper_a.py": "A = 1\n",
        "tests/helper_b.py": "from helper_a import A\n",
        "tests/test_c.py": "from helper_b import A\n",
        "scripts/tool.py": "import inner\n",
        "scripts/inner.py": "pass\n",
        "tests/route_helper.py": "spec = 'scripts/tool.py'\n",
        "tests/test_route.py": "from route_helper import spec\n",
        "tests/test_unrelated.py": "x = 1\n",
    })
    # A changed (or re-added) test module reruns itself and its importers.
    for kind in ("modified", "added"):
        assert _impacted_in(tmp_path, tree, "tests/test_shared.py", kind=kind) == [
            "tests/test_shared.py", "tests/test_user.py", "tests/test_user_user.py",
        ]
    # helper -> helper -> test, and script -> script -> helper -> test.
    assert _impacted_in(tmp_path, tree, "tests/helper_a.py") == ["tests/test_c.py"]
    assert _impacted_in(tmp_path, tree, "scripts/inner.py") == ["tests/test_route.py"]
    # A removed test module that others still import reruns them.
    removed = {k: v for k, v in tree.items() if k != "tests/test_shared.py"}
    assert _impacted_in(tmp_path, removed, "tests/test_shared.py", kind="removed") == [
        "tests/test_user.py", "tests/test_user_user.py",
    ]


def test_impacted_refuses_a_touched_file_no_test_reaches(tmp_path):
    # A delta over a change nothing names used to run zero tests and pass.
    tree = _tmp_tree(tmp_path, {
        "tests/test_a.py": "x = 1\n",
        "tests/conftest.py": "import mapping\n",
        "tests/mapping.py": "M = {}\n",
        "tests/test_uses_conftest.py": "# see conftest\n",
        "scripts/orphan.py": "pass\n",
        "scripts/fixtures/orphan.json": "{}\n",
    })
    for path in ("scripts/orphan.py", "scripts/fixtures/orphan.json"):
        for kind in ("modified", "added", "removed"):
            with pytest.raises(AssertionError, match="full preflight") as info:
                _impacted_in(tmp_path, tree, path, kind=kind)
            assert path in str(info.value) and "preflight delta" in str(info.value)
    # conftest.py is scanned like any helper (a changed one is refused by
    # the delta writer): the consumers of its stem are selected.
    assert _impacted_in(tmp_path, tree, "tests/mapping.py") == ["tests/test_uses_conftest.py"]
    # A removed test module nothing imports reaches nothing: that is exact.
    gone = {k: v for k, v in tree.items() if k != "tests/test_a.py"}
    assert _impacted_in(tmp_path, gone, "tests/test_a.py", kind="removed") == []


def test_impacted_refuses_a_chain_that_reaches_no_test_module(tmp_path):
    # Review round 1: a touched helper with a consumer that is itself an
    # orphan (helper_a <- helper_b <- nothing) has consumers but reaches no
    # test module; it used to yield an empty, passing delta.
    tree = _tmp_tree(tmp_path, {
        "tests/test_a.py": "x = 1\n",
        "tests/helper_a.py": "A = 1\n",
        "tests/helper_b.py": "from helper_a import A\n",
        "tests/helper_c.py": "C = 1\n",
        "tests/test_c.py": "from helper_c import C\n",
    })
    with pytest.raises(AssertionError, match="helper_a.*full preflight"):
        _impacted_in(tmp_path, tree, "tests/helper_a.py")
    # Even beside a change that does reach a test.
    with pytest.raises(AssertionError, match="helper_a.*full preflight"):
        _impacted_in(tmp_path, tree, "tests/helper_a.py", "tests/helper_c.py")
    assert _impacted_in(tmp_path, tree, "tests/helper_c.py") == ["tests/test_c.py"]
    # A changed conftest.py reaches every test: refused by the shared
    # selector, so the validator refuses it as well as the writer.
    with pytest.raises(AssertionError, match="conftest.py.*full preflight"):
        _impacted_in(tmp_path, {**tree, "tests/conftest.py": "h"}, "tests/conftest.py")


def test_impacted_selects_directory_walking_consumers(tmp_path):
    # Review round 3: a module that walks a bound directory (glob/rglob/
    # iterdir/walk/listdir/scandir on scripts/, tests/, ...) reads every file
    # under it without naming one, so it is a consumer of each touched file
    # there.  tests/test_no_hardcoded_home_paths.py rglobs scripts/**/*.py.
    # The walk calls are spliced so that this module's own text does not
    # read as a walker of scripts/ or provenance/ on the real tree.
    def walker_text(top, call, pattern=None):
        arg = "" if pattern is None else repr(pattern)
        return f"for p in (ROOT / {top!r})." + call + "(" + arg + "):\n    pass\n"

    tree = _tmp_tree(tmp_path, {
        "scripts/tool.py": "pass\n",
        "scripts/fixtures/f.json": "{}\n",
        "tests/test_tool.py": "spec = 'scripts/tool.py'\n",
        "tests/test_meta.py": walker_text("scripts", "rg" "lob", "*.py"),
        "tests/test_prov_walk.py": walker_text("provenance", "gl" "ob", "*.json"),
        "tests/test_any_walk.py": walker_text("scripts", "iter" "dir"),
        "tests/test_names_dir_only.py": "name = 'scripts'\n",
        "provenance/p.json": "{}\n",
    })
    assert _impacted_in(tmp_path, tree, "scripts/tool.py") == [
        "tests/test_any_walk.py", "tests/test_meta.py", "tests/test_tool.py",
    ]
    # A walked directory places an otherwise unnamed file (no refusal), but
    # a literal glob pattern limits the walker to the names it matches.
    assert _impacted_in(tmp_path, tree, "provenance/p.json") == ["tests/test_prov_walk.py"]
    assert _impacted_in(tmp_path, tree, "scripts/fixtures/f.json") == ["tests/test_any_walk.py"]
    real = qualify.preflight_tree()
    changes = {"added": [], "modified": ["scripts/agent_client_conformance.py"], "removed": []}
    impacted = qualify.preflight_impacted_tests(changes, real)
    assert {"tests/test_no_hardcoded_home_paths.py",
            "tests/test_server_subprocess_pythonpath.py"} <= set(impacted)


def test_delta_validator_refuses_a_changed_conftest(tmp_path):
    # Review round 1: the writer refused a conftest.py change but the
    # validator recomputed the lexical closure and accepted a hand-written
    # (or foreign-harness) delta that reran only that closure.
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    current = _edited(tree, "tests/conftest.py")
    changes = qualify.preflight_tree_changes(tree, current)
    try:
        impacted = qualify.preflight_impacted_tests(changes, current)
    except AssertionError:
        impacted = []  # the selector already refuses; the receipt must still be refused
    ordinary, guarded = qualify._delta_test_commands(impacted)
    receipt = {
        "schema": qualify.PREFLIGHT_DELTA_SCHEMA, "passed": True, "identity": _BOUND,
        "base": {"path": str(base), "sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
        "tree": current, "changes": changes, "impacted": impacted,
        "test_command": ordinary, "returncode": 0 if ordinary else None,
        "guard_test_command": guarded, "guard_returncode": 0 if guarded else None,
    }
    delta_path = tmp_path / "delta.json"
    delta_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="conftest.py"):
        qualify.validate_preflight_receipt(
            delta_path, _BOUND["runtime"], identity_fn=lambda: _BOUND, tree_fn=lambda: current
        )


def test_write_preflight_delta_refuses_a_change_no_test_reaches(tmp_path):
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    # Built at run time so that this module's own text does not name it.
    orphan = "scripts/fixtures/" + "_".join(["nothing", "names", "this"]) + ".json"
    current = _edited(tree, orphan)
    with pytest.raises(AssertionError, match="full preflight"):
        qualify.write_preflight_delta(
            tmp_path / "delta.json", base, run=_passing_run(),
            identity_fn=lambda: _BOUND, tree_fn=lambda: current,
        )


def test_real_tree_delta_reruns_importers_of_a_changed_test_module(tmp_path):
    changed = "tests/test_batched_mtp.py"
    importers = sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
        if "from test_batched_mtp import" in p.read_text(errors="replace")
    )
    assert "tests/test_qwen4_external_taps_cpu.py" in importers
    tree = qualify.preflight_tree()
    base = _full_base(tmp_path, tree)
    current = _edited(tree, changed)
    ran = []
    delta_path = tmp_path / "delta.json"
    delta = qualify.write_preflight_delta(
        delta_path, base, run=_passing_run(ran), identity_fn=lambda: _BOUND,
        tree_fn=lambda: current,
    )
    assert delta["changes"]["modified"] == [changed]
    assert not set(importers) - set(delta["impacted"])
    assert not set(importers) - set(ran[0])
    evidence = qualify.validate_preflight_receipt(
        delta_path, _BOUND["runtime"], identity_fn=lambda: _BOUND, tree_fn=lambda: current
    )
    assert evidence["passed"] and evidence["delta"]["impacted"] == delta["impacted"]
    # A receipt written by the one-hop selector (the changed module alone)
    # is refused: it did not rerun the importers.
    narrow = json.loads(delta_path.read_text())
    narrow["impacted"] = [changed]
    narrow["test_command"] = narrow["test_command"][:3] + [changed]
    delta_path.write_text(json.dumps(narrow))
    with pytest.raises(AssertionError, match="every impacted test module"):
        qualify.validate_preflight_receipt(
            delta_path, _BOUND["runtime"], identity_fn=lambda: _BOUND, tree_fn=lambda: current
        )


def test_preflight_tree_binds_script_fixtures_and_provenance(tmp_path):
    # Sweep 2026-10-09 (receipt-binding-omits-inputs): tests assert on
    # scripts/fixtures/*.json (through the PLE smoke script) and read
    # provenance/*.json, but the tree bound only tests/** and scripts/**/*.py.
    files = {
        "tests/test_a.py": "x = 1\n",
        "scripts/a.py": "pass\n",
        "scripts/fixtures/policy.json": "{}\n",
        "scripts/research/m5/notes.md": "n\n",
        "provenance/x.json": "{}\n",
        "provenance/nested/y.json": "{}\n",
        "provenance/NOTICE": "n\n",
        "docs/Q.md": "d\n",
        "scripts/__pycache__/a.cpython-312.pyc": "",
        # Review round 1: tests also read native sources, qualification plans,
        # policies and top-level records, experiment companions and
        # docs/PROVENANCE.md.  Campaign evidence under qualification/runs/ is
        # left out by design (a campaign must not invalidate the receipt).
        "native/paged_kv/arena.cpp": "// c\n",
        "qualification/plans/p.json": "{}\n",
        "qualification/policies/q.json": "{}\n",
        "qualification/top.json": "{}\n",
        "qualification/decision_reference.py": "pass\n",
        "qualification/runs/r/receipt.json": "{}\n",
        "qualification/runs/r/run_campaign.py": "pass\n",
        "docs/experiments/e.json": "{}\n",
        "docs/experiments/E.md": "e\n",
        "docs/PROVENANCE.md": "p\n",
        # Review round 2: everything under qualification/ except runs/ (a
        # test round-trips the committed capsule artifact, weights included)
        # and every provenance file (a qualifier hashes a .NOTICE).
        "qualification/artifacts/capsule/manifest.json": "{}\n",
        "qualification/artifacts/capsule/weights.npz": "\x00bin",
        "qualification/corpora/c.jsonl": "{}\n",
        "qualification/experiments/x/sample.npz": "\x00bin",
        "qualification/experiments/x/probe.py": "pass\n",
        "qualification/history/h.json": "{}\n",
        "qualification/receipts/r.json": "{}\n",
        "provenance/tensor-fa-research.NOTICE": "n\n",
        "provenance/LICENSE.unified-MIT": "l\n",
    }
    _tmp_tree(tmp_path, files)
    tree = qualify.preflight_tree(tmp_path)
    assert set(tree) == {
        "tests/test_a.py", "scripts/a.py", "scripts/fixtures/policy.json",
        "scripts/research/m5/notes.md", "provenance/x.json", "provenance/nested/y.json",
        "provenance/NOTICE", "provenance/tensor-fa-research.NOTICE",
        "provenance/LICENSE.unified-MIT",
        "native/paged_kv/arena.cpp", "qualification/plans/p.json",
        "qualification/policies/q.json", "qualification/top.json",
        "qualification/decision_reference.py",
        "qualification/artifacts/capsule/manifest.json",
        "qualification/artifacts/capsule/weights.npz", "qualification/corpora/c.jsonl",
        "qualification/experiments/x/sample.npz", "qualification/experiments/x/probe.py",
        "qualification/history/h.json", "qualification/receipts/r.json",
        "docs/experiments/e.json", "docs/experiments/E.md", "docs/PROVENANCE.md",
    }
    assert not any(p.startswith("qualification/runs/") for p in tree)
    (tmp_path / "scripts/fixtures/policy.json").write_text('{"a": 1}\n')
    assert qualify.preflight_tree_changes(tree, qualify.preflight_tree(tmp_path)) == {
        "added": [], "modified": ["scripts/fixtures/policy.json"], "removed": [],
    }
    real = qualify.preflight_tree()
    fixture = "scripts/fixtures/qwen4_ple_adaptive_no_warm_policy.json"
    provenance = "provenance/omlx3958-wide-verify-sdpa.json"
    assert fixture in real and provenance in real
    assert "provenance/tensor-fa-research.NOTICE" in real
    assert any(p.startswith("qualification/artifacts/") and p.endswith("weights.npz") for p in real)
    assert not any(p.startswith("qualification/runs/") for p in real)
    # And a change to either reaches the test that asserts on it (the fixture
    # through the script that loads it).
    changes = {"added": [], "modified": [fixture, provenance], "removed": []}
    impacted = qualify.preflight_impacted_tests(changes, real)
    assert {"tests/test_smoke_qwen4_ple_incremental_composition.py",
            "tests/test_omlx3958_wide_sdpa.py"} <= set(impacted)


def test_preflight_guard_manifest_partitions_the_test_tree():
    ordinary, guarded = qualify.preflight_test_commands()
    guards = qualify.PREFLIGHT_IMPORT_GUARD_MODULES
    assert guards and len(guards) == len(set(guards))
    assert ordinary[1:3] == ["-m", "pytest"]
    assert ordinary[3:] == [f"--ignore={module}" for module in guards]
    assert guarded[1:] == ["scripts/qualify_serving.py", "--run-import-guards", *guards]
    assert all((ROOT / module).is_file() for module in guards)
    assert "tests/test_qualify_serving_receipts.py" not in guards
    discovered = set()
    for path in (ROOT / "tests").glob("test*.py"):
        tree = ast.parse(path.read_text())
        for index, node in enumerate(tree.body):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            function = node.value.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr == "insert"
                and isinstance(function.value, ast.Attribute)
                and function.value.attr == "meta_path"
            ):
                # A few source-only test programs raise SkipTest before their
                # blocker when pytest imports them; they cannot poison the
                # shared interpreter and intentionally run only as scripts.
                if any(
                    isinstance(earlier, ast.Raise)
                    for previous in tree.body[:index]
                    for earlier in ast.walk(previous)
                ):
                    continue
                discovered.add(str(path.relative_to(ROOT)))
    assert discovered <= set(guards)


@pytest.mark.parametrize("mutation", ["ordinary_failed", "guard_failed", "guard_missing",
                                      "guard_scoped", "ordinary_scoped"])
def test_preflight_rejects_incomplete_or_failed_partition(tmp_path, mutation):
    identity = {
        "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
        "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
        "interpreter": qualify.interpreter_identity(),
    }
    path = tmp_path / "preflight.json"
    ordinary, guarded = qualify.preflight_test_commands()
    receipt = {
        "schema": qualify.PREFLIGHT_SCHEMA, "passed": True, "identity": identity,
        "test_command": ordinary, "guard_test_command": guarded,
        "returncode": 0, "guard_returncode": 0,
    }
    if mutation == "ordinary_failed":
        receipt["returncode"] = 1
    elif mutation == "guard_failed":
        receipt["guard_returncode"] = 1
    elif mutation == "guard_missing":
        receipt.pop("guard_test_command")
    elif mutation == "guard_scoped":
        receipt["guard_test_command"] = guarded[:-1]
    else:
        receipt["test_command"] = ordinary[:-1]
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="full unit suite"):
        qualify.validate_preflight_receipt(
            path, identity["runtime"], identity_fn=lambda: identity
        )


def test_preflight_records_guard_failure_even_when_ordinary_passes(tmp_path):
    identity = {
        "git": {"revision": "abc"}, "runtime": {"source_sha256": "runtime"},
        "qualification_harness": {"sha256": "harness"}, "test_source_sha256": "tests",
        "interpreter": qualify.interpreter_identity(),
    }
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 1,
                               stdout="ordinary pass" if len(calls) == 1 else "guard fail",
                               stderr="")

    path = tmp_path / "failed.json"
    with pytest.raises(AssertionError, match="import guards failed"):
        qualify.write_preflight_receipt(path, run=run, identity_fn=lambda: identity)
    receipt = json.loads(path.read_text())
    assert receipt["passed"] is False and receipt["guard_returncode"] == 1
    assert all(call[1]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1" for call in calls)


def _cross_host_case(tmp_path):
    ordinary, guard = qualify.preflight_test_commands()
    source_runtime = {"source_sha256": "source", "mlx_native_sha256": "m5"}
    target_runtime = {"source_sha256": "source", "mlx_native_sha256": "m3"}
    identity = {
        "git": {"revision": "commit"},
        "runtime": target_runtime,
        "qualification_harness": qualify.qualification_harness_identity(),
        "test_source_sha256": "test-tree",
        "pytest_config_sha256": "config",
        "interpreter": qualify.interpreter_identity(),
    }
    artifact_node = "tests/test_a.py::test_local_artifact"
    guard_node = "tests/test_b.py::test_import_guard"
    # The source proof's heads are the same interpreter path (both hosts ran
    # the same venv layout); the caller vouches for it as the source identity.
    target_command = [sys.executable, "-m", "pytest", "--noconftest", guard_node]
    source = {
        "host_id": "m5",
        "identity": {**copy.deepcopy(identity), "runtime": copy.deepcopy(source_runtime)},
        "suites": {
            "ordinary": {"command": ordinary, "returncode": 0, "results": [
                {"nodeid": artifact_node, "status": "passed"}]},
            "guard": {"command": guard, "returncode": 0, "results": [
                {"nodeid": guard_node, "status": "passed"}]},
        },
        "artifact_evidence": {artifact_node: "a" * 64},
    }
    target = {
        "host_id": "m3", "identity": copy.deepcopy(identity),
        "artifact": "target-fingerprint", "settings": {"mtp": False},
        "command": list(target_command), "returncode": 0,
        "results": [{"nodeid": guard_node, "status": "passed"}],
    }
    bundle = {"schema": qualify.CROSS_HOST_PREFLIGHT_SCHEMA, "passed": True,
              "source": source, "target": target}
    arguments = {
        "active_runtime": copy.deepcopy(target_runtime),
        "active_artifact": "target-fingerprint",
        "active_settings": {"mtp": False},
        "active_host_id": "m3",
        "expected_source_nodes": {"ordinary": {artifact_node}, "guard": {guard_node}},
        "required_m3_nodes": {guard_node},
        "target_command": list(target_command),
        "artifact_dependent_nodes": {artifact_node},
        "identity_fn": lambda: copy.deepcopy(identity),
        "source_interpreter": dict(identity["interpreter"]),
    }
    path = tmp_path / "cross-host.json"
    return path, bundle, arguments


def _write_cross_host(path, bundle):
    bundle["source_proof_sha256"] = qualify._proof_sha256(bundle["source"])
    bundle["target_proof_sha256"] = qualify._proof_sha256(bundle["target"])
    path.write_text(json.dumps(bundle))


def test_cross_host_preflight_requires_two_exact_proofs(tmp_path):
    path, bundle, arguments = _cross_host_case(tmp_path)
    _write_cross_host(path, bundle)
    result = qualify.validate_cross_host_preflight_receipt(path, **arguments)
    assert result["passed"] and result["source_nodes"] == 2
    assert result["target_nodes"] == 1
    assert result["runtime"] == arguments["active_runtime"]


@pytest.mark.parametrize("mutation", [
    "source_commit", "source_runtime_source", "target_runtime", "target_artifact",
    "target_settings", "same_host", "scoped_source", "scoped_target",
    "missing_source_node", "duplicate_source_node", "extra_source_node",
    "artifact_skip", "artifact_evidence_missing", "missing_target_node",
    "skipped_target_node", "failed_source", "failed_target", "tampered_digest",
    "empty_target_executable",
])
def test_cross_host_preflight_fails_closed(tmp_path, mutation):
    path, bundle, arguments = _cross_host_case(tmp_path)
    source, target = bundle["source"], bundle["target"]
    source_rows = source["suites"]["ordinary"]["results"]
    if mutation == "source_commit":
        source["identity"]["git"]["revision"] = "other"
    elif mutation == "source_runtime_source":
        source["identity"]["runtime"]["source_sha256"] = "other"
    elif mutation == "target_runtime":
        target["identity"]["runtime"]["mlx_native_sha256"] = "other"
    elif mutation == "target_artifact":
        target["artifact"] = "other"
    elif mutation == "target_settings":
        target["settings"]["mtp"] = True
    elif mutation == "same_host":
        target["host_id"] = "m5"
    elif mutation == "scoped_source":
        source["suites"]["ordinary"]["command"].append("-k")
    elif mutation == "scoped_target":
        target["command"].append("-k")
    elif mutation == "missing_source_node":
        source_rows.clear()
    elif mutation == "duplicate_source_node":
        source_rows.append(copy.deepcopy(source_rows[0]))
    elif mutation == "extra_source_node":
        source_rows.append({"nodeid": "tests/test_extra.py::test_new", "status": "passed"})
    elif mutation == "artifact_skip":
        source_rows[0] = {"nodeid": source_rows[0]["nodeid"],
                          "status": "skipped", "reason": "local artifact absent"}
        arguments["approved_source_skips"] = {source_rows[0]["nodeid"]: "local artifact absent"}
    elif mutation == "artifact_evidence_missing":
        source["artifact_evidence"].clear()
    elif mutation == "missing_target_node":
        target["results"].clear()
    elif mutation == "skipped_target_node":
        target["results"][0]["status"] = "skipped"
        target["results"][0]["reason"] = "M3 artifact absent"
    elif mutation == "failed_source":
        source["suites"]["ordinary"]["returncode"] = 1
    elif mutation == "failed_target":
        target["returncode"] = 1
    elif mutation == "empty_target_executable":
        # Review round 4: the target command and the caller's expectation
        # agree, but neither names an interpreter.
        arguments["target_command"] = ["", *list(arguments["target_command"])[1:]]
        target["command"] = list(arguments["target_command"])
    _write_cross_host(path, bundle)
    if mutation == "tampered_digest":
        bundle["source"]["host_id"] = "changed"
        path.write_text(json.dumps(bundle))
    with pytest.raises(AssertionError, match="cross-host preflight"):
        qualify.validate_cross_host_preflight_receipt(path, **arguments)


def test_source_text_http_witness_requires_exact_ids_prompt_and_route():
    tokens = list(range(16))
    sampling = {"temperature": 0, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0}
    case = {"prompt": "Reply with exactly MLX2_READY", "prompt_tokens": 18,
            "generated_token_ids": tokens, "max_tokens": 16, "sampling": sampling}
    companion = {"model_type": "smolvlm", "text_source": {
        "cases": {"cold_text": case}}}
    assert qualify.source_text_cases(companion)["cold_text"] is case
    assert qualify.source_text_cases({"model_type": "qwen2_5_vl"}) is None
    with pytest.raises(ValueError, match="approved Smol producer"):
        qualify.source_text_cases({"model_type": "qwen2_5_vl", "text_source": {
            "cases": {"cold_text": case}}})
    response = {
        "choices": [{"message": {"content": "some deterministic text"},
                     "finish_reason": "length",
                     "logprobs": {"content": [{"id": token} for token in tokens]}}],
        "usage": {"prompt_tokens": 18, "completion_tokens": 16},
        "mlx2": {"route": "ordinary", "qualification": "candidate",
                 "cache": "apcv2", "request_controls": {
                     "max_tokens": 16, "min_tokens": 0, "sampling": sampling}},
    }
    assert qualify.source_token_response_matches(
        response, case, prompt_text=case["prompt"])
    for changed in (
        {**response, "usage": {"prompt_tokens": 9, "completion_tokens": 16}},
        {**response, "mlx2": {**response["mlx2"], "route": "mtp"}},
        {**response, "choices": [{**response["choices"][0], "logprobs": {
            "content": [{"id": 99}, *[{"id": token} for token in tokens[1:]]]}}]},
    ):
        assert not qualify.source_token_response_matches(
            changed, case, prompt_text=case["prompt"])
    assert not qualify.source_token_response_matches(
        response, case, prompt_text="Reply with exactly HERMES_READY")
    near_case = {**case, "prompt": "near-context compiler prompt",
                 "prompt_tokens": 3869, "generated_token_ids": list(range(64)),
                 "max_tokens": 64, "min_tokens": 64}
    near_response = {**response,
                     "choices": [{**response["choices"][0], "logprobs": {
                         "content": [{"id": token} for token in range(64)]}}],
                     "usage": {"prompt_tokens": 3869, "completion_tokens": 64},
                     "mlx2": {**response["mlx2"], "request_controls": {
                         "max_tokens": 64, "min_tokens": 64, "sampling": sampling}}}
    assert qualify.source_token_response_matches(
        near_response, near_case, prompt_text=near_case["prompt"])
    near_response["mlx2"]["request_controls"]["min_tokens"] = 0
    assert not qualify.source_token_response_matches(
        near_response, near_case, prompt_text=near_case["prompt"])


def test_source_text_stop_witness_is_interior_and_unique():
    output = "To solve the problem, we need to find the value of the variable x"
    witness = qualify.interior_stop_witness(output)
    assert witness is not None
    stop, expected = witness
    assert stop and expected and output.count(stop) == 1
    assert output.split(stop, 1)[0] == expected
    assert qualify.interior_stop_witness("short") is None


def test_long_context_probe_uses_one_consistent_safe_headroom():
    cap = 1024
    text = qualify.long_context_prompt(cap)
    assert qualify.LONG_CONTEXT_HEADROOM == 256
    assert text.endswith(qualify.LONG_CONTEXT_INSTRUCTION)
    filler = text[: -len(qualify.LONG_CONTEXT_INSTRUCTION)].split(" ")
    assert len(filler) == cap - qualify.LONG_CONTEXT_HEADROOM
    assert set(filler) <= set(qualify.LONG_CONTEXT_WORDS)
    # Varied, not one repeated token; identical bytes on every call.
    assert len(set(filler)) > 40
    assert text == qualify.long_context_prompt(cap)
    assert qualify.long_context_filler(2048).startswith(qualify.long_context_filler(512))
    with pytest.raises(ValueError, match="exceed near-limit headroom"):
        qualify.long_context_prompt(qualify.LONG_CONTEXT_HEADROOM)


def test_cancellation_probe_budget_is_long_but_context_bounded():
    assert qualify.cancellation_probe_budget(1024) == 768
    assert qualify.cancellation_probe_budget(16384) == 8192
    request = {
        "max_tokens": qualify.cancellation_probe_budget(1024),
        "reasoning_effort": "none",
        "think": False,
    }
    assert qualify.with_thinking_budget(request, True)["max_tokens"] == 768
    with pytest.raises(ValueError, match="too small"):
        qualify.cancellation_probe_budget(qualify.LONG_CONTEXT_HEADROOM)


def test_long_context_delegation_requires_generated_thermal_near_limit_matrix(tmp_path):
    prompt = tmp_path / "prompt.json"
    prompt.write_text("{}")
    prompt_sha256 = hashlib.sha256(prompt.read_bytes()).hexdigest()
    manifest = {
        "schema": "mlx2.qualification-matrix.v1",
        "thermal": {"consecutive_samples": 3, "max_wait_seconds": 1800},
        "models": [{
            "name": "qwen", "arms": [{"name": "ordinary"}],
            "contexts": [{"tokens": 262016, "prompt_path": str(prompt),
                          "prompt_sha256": prompt_sha256}],
            "context": {"runs_per_cell": 3, "max_tokens": 64},
        }],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(manifest))
    evidence = qualify.validate_context_matrix_delegation(path, 262144)
    assert evidence["delegated_checks"] == list(qualify.LONG_CONTEXT_CHECKS)
    assert evidence["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    manifest["models"][0]["context"]["runs_per_cell"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="no generated, repeated near-limit"):
        qualify.validate_context_matrix_delegation(path, 262144)


def _reasoning_response(*, reasoning="work", content="221", finish_reason="stop"):
    return {"choices": [{
        "finish_reason": finish_reason,
        "message": {"reasoning_content": reasoning, "content": content},
    }]}


def test_reasoning_probe_retries_cap_interrupted_channel_transition():
    responses = iter([
        _reasoning_response(content="", finish_reason="length"),
        _reasoning_response(content="The answer is 221."),
    ])
    requests = []

    def post(body):
        requests.append(body)
        return next(responses)

    final, attempts = qualify.run_reasoning_probe(post)
    assert qualify.reasoning_response_passes(final)
    assert len(attempts) == 2
    assert [request["max_tokens"] for request in requests] == [512, 1024]
    assert all(request["enable_thinking"] is True for request in requests)
    assert all(request["messages"] == [
        {"role": "system", "content": qualify.REASONING_PROBE_SYSTEM},
        {"role": "user", "content": "Calculate 13 times 17. Reply with the integer."},
    ] for request in requests)


def test_reasoning_probe_bounds_declared_state_aware_thinking():
    requests = []

    def post(body):
        requests.append(body)
        return _reasoning_response(content="The answer is 221.")

    final, attempts = qualify.run_reasoning_probe(
        post, thinking_budget=qualify.REASONING_PROBE_THINKING_BUDGET
    )

    assert qualify.reasoning_response_passes(final)
    assert len(attempts) == 1
    assert requests[0]["max_tokens"] == 512
    assert requests[0]["thinking_budget"] == 128


def test_structured_reasoning_probe_bounds_declared_state_aware_thinking():
    requests = []

    def post(body):
        requests.append(body)
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "reasoning_content": "work",
                    "content": '{"answer": 221}',
                },
            }],
            "mlx2": {
                "request_controls": {
                    "structured_output": {"deferred": True, "engine": "automaton"}
                }
            },
        }

    passed, _evidence = qualify.run_structured_thinking_probe(
        post, {"structured_output": {"thinking_deferral": True}}
    )

    assert passed
    assert requests[0]["thinking_budget"] == 128


def test_reasoning_probe_uses_third_bounded_attempt_when_transition_stays_capped():
    responses = iter([
        _reasoning_response(content="", finish_reason="length"),
        _reasoning_response(content="", finish_reason="length"),
        _reasoning_response(content="221"),
    ])
    requests = []

    def post(body):
        requests.append(body)
        return next(responses)

    final, attempts = qualify.run_reasoning_probe(post)
    assert qualify.reasoning_response_passes(final)
    assert len(attempts) == 3
    assert [request["max_tokens"] for request in requests] == [512, 1024, 2048]


@pytest.mark.parametrize(
    "response",
    [
        _reasoning_response(reasoning=""),
        _reasoning_response(content=""),
        _reasoning_response(content="The answer is 2221."),
        {"choices": []},
    ],
)
def test_reasoning_oracle_requires_both_channels_and_exact_numeric_answer(response):
    assert not qualify.reasoning_response_passes(response)


def test_reasoning_probe_does_not_retry_non_truncation_failure():
    requests = []

    def post(body):
        requests.append(body)
        return _reasoning_response(content="wrong", finish_reason="stop")

    final, attempts = qualify.run_reasoning_probe(post)
    assert not qualify.reasoning_response_passes(final)
    assert len(attempts) == len(requests) == 1


@pytest.mark.parametrize(
    ("family", "context_cap", "prompt_tokens"),
    [
        ("flash-next", 262144, 261923),
        ("qwen38-27b", 262144, 261923),
        ("muse-glimmer", 131072, 130898),
        ("north-mini-code", 500000, 499883),
    ],
)
def test_near_limit_headroom_fits_full_completion_across_templates(
    family, context_cap, prompt_tokens
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": qualify.LONG_CONTEXT_COMPLETION_TOKENS,
    }
    assert family
    assert qualify.near_limit_usage_passes(usage, context_cap)
    assert (
        prompt_tokens - 1
        > qualify.near_limit_prompt_floor(context_cap)
        - qualify.LONG_CONTEXT_CACHE_TOLERANCE
    )


def test_near_limit_gate_rejects_overflow_short_completion_and_distant_prompt():
    cap = 131072
    assert not qualify.near_limit_usage_passes(
        {"prompt_tokens": 131026, "completion_tokens": 64}, cap
    )
    assert not qualify.near_limit_usage_passes(
        {"prompt_tokens": 130898, "completion_tokens": 63}, cap
    )
    assert not qualify.near_limit_usage_passes(
        {
            "prompt_tokens": qualify.near_limit_prompt_floor(cap),
            "completion_tokens": 64,
        },
        cap,
    )


def test_shared_qsa_probe_respects_auto_policy_context_budget_crossover():
    settings = {
        "mtp": True,
        "environment": {
            "MLX_LM_SHARED_QSA_SUFFIX": "auto",
            "MLX_LM_SHARED_QSA_SUFFIX_MAX_REMAINING": "64",
        }
    }
    assert qualify.shared_qsa_completion_budget(settings, 32768) == 32
    assert qualify.shared_qsa_completion_budget(settings, 65536) == 64
    assert qualify.shared_qsa_completion_budget(
        {"mtp": True, "environment": {"MLX_LM_SHARED_QSA_SUFFIX": "1"}}, 32768
    ) == qualify.LONG_CONTEXT_COMPLETION_TOKENS
    assert qualify.shared_qsa_completion_budget({}, 32768) == 64
    assert qualify.near_limit_usage_passes(
        {"prompt_tokens": 32547, "completion_tokens": 32}, 32768, 32
    )


def test_route_mechanism_checks_require_apcv2_and_full_mtp_invariants():
    before = {
        "settings": {"mtp": True},
        "apcv2": {"hits": 1, "stores": 2},
        "execution": {"segmented_mtp": {
            "segmented_attention_calls": 0, "transaction_branches": 0,
            "committed_cycles": 0, "true_batched_engaged": 0,
            "batched_target_forwards": 0, "full_prefix_materializations": 0,
            "physical_b2_formations": 0, "failures": 0,
        }},
    }
    after = {
        "settings": {"mtp": True},
        "apcv2": {"hits": 5, "stores": 7},
        "execution": {"segmented_mtp": {
            "segmented_attention_calls": 8, "transaction_branches": 9,
            "committed_cycles": 6, "true_batched_engaged": 2,
            "batched_target_forwards": 3, "full_prefix_materializations": 0,
            "physical_b2_formations": 0, "failures": 0,
        }},
    }
    checks = qualify.route_mechanism_checks(before, after)
    assert checks and all(row["passed"] for row in checks.values())
    after["execution"]["segmented_mtp"]["physical_b2_formations"] = 1
    assert not qualify.route_mechanism_checks(before, after)["mtp_zero_physical_b2"]["passed"]


def test_route_mechanism_checks_fail_closed_on_missing_counters():
    checks = qualify.route_mechanism_checks(
        {"settings": {"mtp": True}, "apcv2": {}},
        {"settings": {"mtp": True}, "apcv2": {}, "execution": {"segmented_mtp": {}}},
    )
    assert checks and not any(row["passed"] for row in checks.values())

def test_actual_compute_widths_cover_all_serving_routes():
    assert qualify.observed_compute_widths({"mtp": {"observed_compute_widths": [1, 4]}, "ordinary_compute_width": None}) == [1, 4]
    assert qualify.observed_compute_widths({"mtp": None, "speculation": {"target_width": 3}, "ordinary_compute_width": None}) == [3]
    assert qualify.observed_compute_widths({"mtp": None, "speculation": None, "ordinary_compute_width": 2}) == [2]
    assert qualify.observed_compute_widths({"mtp": None, "speculation": None, "ordinary_compute_width": None}) == []

def test_dflash_width_wins_over_null_ordinary_width():
    receipts = [{"mtp": None, "speculation": {"target_width": width}, "ordinary_compute_width": None} for width in (1, 4, 4, 2)]
    widths = sorted({width for receipt in receipts for width in qualify.observed_compute_widths(receipt)})
    assert widths == [1, 2, 4]
    assert max(widths, default=0) >= 2


def test_final_quiescence_waits_for_response_cleanup_and_records_samples():
    statuses = iter([
        {"inflight": 0, "queue_depth": 0,
         "apcv2": {"cow": {"active_leases": 1}}},
        {"inflight": 0, "queue_depth": 0,
         "apcv2": {"cow": {"active_leases": 0}}},
    ])
    clock = iter([10.0, 10.0, 10.1])
    final, evidence = qualify.wait_for_quiescence(
        lambda: next(statuses),
        timeout_seconds=1,
        poll_interval_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    assert evidence["passed"] and not evidence["timed_out"]
    assert evidence["attempts"] == 2
    assert [row["active_cow_leases"] for row in evidence["samples"]] == [1, 0]
    assert final["apcv2"]["cow"]["active_leases"] == 0


def test_final_quiescence_times_out_fail_closed_with_evidence():
    status = {"inflight": 0, "queue_depth": 0,
              "apcv2": {"cow": {"active_leases": 1}}}
    clock = iter([20.0, 20.0, 20.5, 21.0])
    final, evidence = qualify.wait_for_quiescence(
        lambda: status,
        timeout_seconds=1,
        poll_interval_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    assert not evidence["passed"] and evidence["timed_out"]
    assert evidence["attempts"] == 3
    assert evidence["samples"][-1]["active_cow_leases"] == 1
    assert final is status


def test_final_quiescence_rejects_missing_or_malformed_counters():
    status = {"inflight": False, "queue_depth": 0, "apcv2": {"cow": {}}}
    clock = iter([30.0, 30.0, 31.0])
    _, evidence = qualify.wait_for_quiescence(
        lambda: status,
        timeout_seconds=1,
        poll_interval_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    assert evidence["timed_out"] and not evidence["passed"]
    assert evidence["samples"][-1]["inflight"] is None
    assert evidence["samples"][-1]["active_cow_leases"] is None


@pytest.mark.parametrize(
    ("timeout", "interval"),
    [(-1, 0.05), (math.inf, 0.05), (math.nan, 0.05),
     (1, -0.01), (1, math.inf), (1, math.nan)],
)
def test_final_quiescence_requires_finite_bounded_timing(timeout, interval):
    with pytest.raises(ValueError, match="positive and bounded"):
        qualify.wait_for_quiescence(
            lambda: {},
            timeout_seconds=timeout,
            poll_interval_seconds=interval,
        )


def test_long_context_answer_accepts_marker_or_topic_only():
    assert qualify.long_context_answer_passes("LONG_READY\n1. Inlining")
    assert qualify.long_context_answer_passes("The user wants a guide to compiler optimization.")
    # North's warm B2 answer in Arabic quotes the marker from the far-end
    # instruction without naming the English topic.
    assert qualify.long_context_answer_passes(
        "\u0628\u062f\u0627\u064a\u0629\u064b\u060c LONG_READY \u062b\u0645 \u0627\u0643\u062a\u0628"
    )
    assert not qualify.long_context_answer_passes("")
    assert not qualify.long_context_answer_passes("river stone garden market")


@pytest.mark.parametrize("artifact", [
    "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp",
    "Qwen3.8-27B-oQ4e-mtp", "Qwen3.8-Flash-Next-MLX-4bit-MTP",
    "Muse-Glimmer-30B-mlx-4bit", "North-Mini-Code-1.0-mlx-4bit",
    "Xing4.0-29B-A4B-mlx-6bit",
])
def test_long_context_words_are_one_token_in_supported_tokenizers(artifact, monkeypatch):
    import os

    path = os.path.expanduser("~/mlx-models/" + artifact)
    if not os.path.isdir(path):
        pytest.skip("tokenizer artifact is not present")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    if artifact.startswith("Xing"):
        # Custom slow reference class; serving loads the stamped fast tokenizer.
        from mlx2.adapters.xing_tokenizer import load_tokenizer

        tokenizer, _ = load_tokenizer(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert len(tokenizer.encode(qualify.long_context_filler(4000), add_special_tokens=False)) == 4000


def test_thinking_default_servers_get_a_reasoning_allowance():
    extra = qualify.THINKING_BUDGET_TOKENS
    body = {"messages": [], "max_tokens": 64}
    assert qualify.with_thinking_budget(body, True)["max_tokens"] == 64 + extra
    assert body["max_tokens"] == 64  # the caller's request is not mutated
    # A server that does not think by default keeps every budget as written.
    assert qualify.with_thinking_budget(body, False) is body
    assert qualify.with_thinking_budget({**body, "enable_thinking": True}, False)["max_tokens"] == 64
    # Turning thinking off keeps the exact budget (near-limit checks rely on it).
    for off in ({"enable_thinking": False}, {"think": False}, {"reasoning_effort": "none"}, {"reasoning_effort": "NONE"}):
        assert qualify.with_thinking_budget({**body, **off}, True)["max_tokens"] == 64
    assert qualify.with_thinking_budget({**body, "reasoning_effort": "low"}, True)["max_tokens"] == 64 + extra
    assert "max_tokens" not in qualify.with_thinking_budget({"messages": []}, True)
    # The allowance never crosses the API ceiling, and never lowers a budget.
    assert qualify.with_thinking_budget({"max_tokens": 2_097_152}, True)["max_tokens"] == 2_097_152
    assert qualify.with_thinking_budget({"max_tokens": 2_096_000}, True)["max_tokens"] == 2_097_152


def test_thinking_allowance_follows_the_declared_model_value():
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
    assert qualify.with_thinking_budget(body, True, 4096)["max_tokens"] == 64 + 4096
    assert qualify.with_thinking_budget(body, True, 9000)["max_tokens"] == 64 + 9000
    near = {**body, "max_tokens": qualify.SERVER_MAX_TOKENS - 10}
    assert qualify.with_thinking_budget(near, True, 9000)["max_tokens"] == qualify.SERVER_MAX_TOKENS
    assert qualify.with_thinking_budget(body, False, 4096) is body


def test_xing_declares_a_larger_thinking_allowance():
    from mlx2.adapters.xing import XingAdapter

    assert XingAdapter.thinking_allowance_tokens == 4096 > qualify.THINKING_BUDGET_TOKENS


@pytest.mark.parametrize(
    ("sequential", "concurrent", "passes"),
    [
        (3.73, 2.31, True),     # qwen35-9b receipt
        (59.22, 45.07, True),   # Xing receipt, the case the 30 s clock failed
        (2.0, 5.0, False),      # slower than the pair run one after the other
        (2.0, 29.0, False),
        (20.66, 22.0, False),
        (2.0, 2.0, False),      # serialized: no faster than sequential
    ],
)
def test_mixed_warm_concurrent_pair_must_beat_its_sequential_reference(
    sequential, concurrent, passes
):
    assert qualify.mixed_warm_timing_passes(concurrent, sequential) is passes


def test_fused_gdn_decode_is_not_observed_while_decode_is_refused_on_geometry():
    # 2026-09-23: a Flash-Next ordinary receipt held 10,224 fused decode calls
    # beside 24,012 "rollback geometry not describable" refusals and passed on
    # fused_calls > 0.  Engagement now also requires no geometry refusal.
    healthy = {"fused_calls": 120, "decode_fallback_reasons": {"batch of 2 rows": 40}}
    assert qualify.fused_gdn_decode_observation(healthy) == 120
    for reason in qualify.FUSED_GDN_DECODE_GEOMETRY_REFUSALS:
        refused = {"fused_calls": 10224, "decode_fallback_reasons": {reason: 1}}
        assert qualify.fused_gdn_decode_observation(refused) == 0
    assert qualify.fused_gdn_decode_observation({}) == 0
    observed = qualify.feature_observations(
        {"execution": {"fused_gdn": {
            "fused_calls": 10224,
            "decode_fallback_reasons": {"rollback geometry not describable": 24012},
        }}}
    )
    assert observed["fused_gdn_decode"] == 0


def test_a_refused_request_becomes_check_evidence_not_an_exception():
    import io
    from urllib.error import HTTPError

    body = b'{"error": {"message": "declared batch cohort could not atomically admit every member"}}'
    error = HTTPError("http://x/v1/chat/completions", 429, "Too Many Requests", {}, io.BytesIO(body))
    assert qualify.http_refusal(error) == {
        "status": 429,
        "error": {"message": "declared batch cohort could not atomically admit every member"},
    }
    garbled = HTTPError("http://x", 503, "Unavailable", {}, io.BytesIO(b"not json"))
    assert qualify.http_refusal(garbled) == {"status": 503, "error": {}}
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    # The shared-cohort pair is posted through the refusal-tolerant helper.
    assert "shared = list(pool.map(post_or_refusal, shared_pair))" in source


def _lane(first, last, tokens=64):
    """Arrival instants of ``tokens`` evenly spaced deltas from first to last."""
    step = (last - first) / (tokens - 1)
    return [first + index * step for index in range(tokens)]


def test_mixed_warm_per_lane_route_is_judged_by_overlap_not_speedup():
    # Qwen3.6 prompt lookup keeps the per-lane driver (receipts report width
    # 1) and measured 1.675 s concurrent against 1.641 s sequential in the
    # sweep's GPU smoke: overlapped, not faster, as a per-lane route must be.
    # Overlap is judged from each lane's token arrivals on the pair's clock.
    per_lane = [
        {"speculation": {"target_width": 1}, "ttft_seconds": 0.12, "elapsed_seconds": 1.66},
        {"speculation": {"target_width": 1}, "ttft_seconds": 0.15, "elapsed_seconds": 1.67},
    ]
    interleaved = [_lane(0.12, 1.665), _lane(0.15, 1.675)]

    def passes(concurrent, sequential, receipts, lane_token_seconds=interleaved):
        return qualify.mixed_warm_timing_passes(
            concurrent, sequential, receipts, lane_token_seconds=lane_token_seconds
        )

    assert passes(1.675, 1.641, per_lane) is True
    assert qualify.mixed_warm_lane_overlap_shares(interleaved) == [
        pytest.approx(1.0, abs=0.05), pytest.approx(1.0, abs=0.05)
    ]
    # Attached together, then run one after the other: lane B's first token
    # comes after lane A finished.
    serial = [
        {"speculation": {"target_width": 1}, "ttft_seconds": 0.1, "elapsed_seconds": 1.0},
        {"speculation": {"target_width": 1}, "ttft_seconds": 0.1, "elapsed_seconds": 1.9},
    ]
    serialized = [_lane(0.1, 1.0), _lane(1.05, 1.9)]
    assert qualify.mixed_warm_lane_overlap_shares(serialized) == [0.0, 0.0]
    assert passes(1.9, 2.0, serial, serialized) is False
    # Lane B gets one early token, stalls while lane A runs to completion,
    # then runs alone.  Receipt times imply continuous service from first
    # token to finish (0.1-1.0 s and 0.1-1.9 s), so the gate built on them
    # passed this serialized service; B's token arrivals show the stall.
    stalled = [_lane(0.1, 1.0), [0.1] + _lane(1.0 + 0.9 / 63, 1.9, tokens=63)]
    shares = qualify.mixed_warm_lane_overlap_shares(stalled)
    assert shares[0] == 1.0 and shares[1] < 0.05
    assert passes(1.9, 2.0, serial, stalled) is False
    # The same receipts with the lanes' tokens genuinely interleaved pass.
    assert passes(1.9, 2.0, serial, [_lane(0.1, 1.85), _lane(0.12, 1.9)]) is True
    # A lane that joins partway still overlaps once a quarter of its tokens
    # arrive while the other lane is producing, and not before.
    joined = [_lane(0.1, 1.0), _lane(0.6, 1.9)]
    assert passes(1.9, 2.0, serial, joined) is True
    late = [_lane(0.1, 1.0), _lane(0.95, 1.9)]
    assert passes(1.9, 2.0, serial, late) is False
    # Without token arrivals for every lane there is no evidence of overlap.
    for missing in (None, [], [interleaved[0]], [interleaved[0], []],
                    [interleaved[0], [math.nan] * 64], [interleaved[0], [True] * 64]):
        assert passes(1.675, 1.641, per_lane, missing) is False
    # Overlapped but clearly slower than sequential still fails.
    assert passes(2.0, 1.641, per_lane) is False
    # A route that batched (width 2) must still beat its sequential pair.
    batched = [{"speculation": {"target_width": 2}, "elapsed_seconds": 1.66}] * 2
    assert qualify.mixed_warm_timing_passes(1.675, 1.641, batched) is False
    assert qualify.mixed_warm_timing_passes(1.2, 1.641, batched) is True


def test_streamed_chat_is_rebuilt_with_its_receipt_and_token_arrivals():
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"choices": [{"index": 0, "delta": {}}]},  # a logprob-only chunk
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 2},
            "mlx2": {"cached_tokens": 12},
        },
    ]
    wire = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    wire = [line for chunk in wire for line in (chunk, b"\n")] + [b"data: [DONE]\n", b"\n"]
    ticks = iter([0.5, 0.75, 0.8, 0.9])
    response, arrivals = qualify.read_streamed_chat(wire, lambda: next(ticks))
    assert response["choices"][0]["message"]["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"] == {"completion_tokens": 2}
    assert response["mlx2"] == {"cached_tokens": 12}
    assert arrivals == [0.5, 0.75]
    # A stream cut off before its receipt chunk is not a response.
    with pytest.raises(AssertionError):
        qualify.read_streamed_chat(wire[:4], lambda: 0.0)


def test_stream_and_nonstream_content_use_symmetric_outer_whitespace_normalization():
    assert qualify.normalize_chat_content("\n\nMLX2_READY") == "MLX2_READY"
    assert qualify.normalize_chat_content("MLX2_READY") == "MLX2_READY"
    assert qualify.normalize_chat_content("MLX2_READY extra") != "MLX2_READY"


def test_mixed_warm_pair_is_streamed_and_judged_on_token_arrivals():
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    block = source[source.index("def post_streamed(item, clock_start):"):]
    block = block[: block.index('"mixed_warm"')]
    assert "post(item, stream=True)" in block
    assert "read_streamed_chat(" in block
    assert "post_streamed(item, sequential_start)" in block
    assert "post_streamed(item, start)" in block
    assert "lane_token_seconds=lane_token_seconds" in source


def test_capability_scope_rejects_conflicting_status_fields():
    status = {
        "selected_capabilities": ["text", "vision"],
        "implemented_capabilities": ["text", "vision", "video"],
        "capabilities": ["text", "vision"],
    }
    assert qualify.selected_capability_scope(status) == ({"text", "vision"}, True)
    assert qualify.selected_capability_scope({**status, "capabilities": ["text"]})[1] is False
    assert qualify.selected_capability_scope({**status, "implemented_capabilities": ["text"]})[1] is False
    assert qualify.selected_capability_scope({**status, "selected_capabilities": ["text", "text"]})[1] is False


def test_live_media_companion_binds_the_generic_harness_route(tmp_path):
    from mlx2.qualification import APPROVED_MEDIA_PRODUCERS

    report = json.loads(
        (ROOT / "docs/experiments/SMOLVLM2-M3-LIVE-MEDIA-QUALIFICATION-2026-09-26.json").read_text()
    )
    # The recorded run names the producer revision that ran it; the producer
    # has since tightened its media-reuse predicate and bound its traces to
    # the mlx-vlm dependency-content identity, which invalidates the receipt
    # until it is re-run.  Re-bind the real traces to the current pin and
    # identity to exercise the binding (the evaluator still recomputes every
    # check from the recorded traces).
    from test_qualification import _rebind_to_current_producer

    _rebind_to_current_producer(report)
    assert report["qualification_harness"] == APPROVED_MEDIA_PRODUCERS[report["model_type"]][0]
    path = tmp_path / "companion.json"
    path.write_text(json.dumps(report))
    status = {key: report[key] for key in ("runtime", "artifact", "settings")}
    assert qualify.read_adapter_qualification(path, status)["passed"] is True
    with pytest.raises(ValueError, match="does not match"):
        qualify.read_adapter_qualification(path, {**status, "artifact": "other"})


def test_near_limit_requests_hold_eos_until_the_full_budget():
    # near_limit_usage_passes demands completion_tokens == budget; a model
    # that ends its answer early must not decide whether the server passed.
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    builder = source[source.index("def long_context_request("):]
    builder = builder[: builder.index("if not context_delegation:")]
    assert "min_tokens=completion_budget" in builder
    assert "max_tokens=completion_budget" in builder


def test_mixed_warm_pairs_do_identical_work():
    # The timing comparison is only meaningful when both pairs generate the
    # same tokens: thinking off and EOS held to the budget.
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    block = source[source.index("mixed_prompts = ["):]
    block = block[: block.index("]\n")]
    assert block.count("min_tokens=64") == 2
    assert block.count('reasoning_effort="none"') == 2
    assert 'r["usage"]["completion_tokens"] == 64 for r in mixed' in source


def test_stop_check_runs_with_thinking_off():
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    call = source[source.index('cold_text_prompt, stop=stop_text'):]
    call = call[: call.index("))")]
    assert 'reasoning_effort="none"' in call and "think=False" in call


def test_reply_evidence_carries_text_usage_and_receipt():
    reply = {
        "choices": [{
            "message": {"content": "  " + "x" * 2500 + "  ", "reasoning_content": "why " * 10},
            "finish_reason": "length",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 64},
        "mlx2": {"cached_tokens": 5, "request_controls": {}},
    }
    evidence = qualify.reply_evidence(reply)
    assert evidence["content"].startswith("x" * qualify.REPLY_EVIDENCE_TEXT_CHARS)
    assert "truncated 500 chars" in evidence["content"]
    assert len(evidence["content"]) < 2100
    assert evidence["reasoning_content"] == "why " * 10
    assert evidence["finish_reason"] == "length"
    assert evidence["usage"] == reply["usage"]
    assert evidence["mlx2"] == reply["mlx2"]

    short = qualify.reply_evidence({
        "choices": [{"message": {"content": "LONG_READY"}}],
        "usage": {}, "mlx2": {},
    })
    assert short["content"] == "LONG_READY"
    assert "reasoning_content" not in short

    refusal = {"refused": {"status": 429, "body": "busy"}}
    assert qualify.reply_evidence(refusal) is refusal
    assert qualify.reply_evidence({"error": "bad"})["content"] == ""


def test_long_context_checks_record_reply_evidence():
    source = (ROOT / "scripts" / "qualify_serving.py").read_text()
    assert "[reply_evidence(r) for r in shared]" in source
    assert "reply_evidence(primed)" in source
    assert '{"cold": reply_evidence(long), "warm": reply_evidence(repeated)}' in source
    assert '[r.get("mlx2", r) for r in shared]' not in source


def test_import_guards_run_one_module_per_interpreter(monkeypatch):
    # Guard modules poison imports at module level, some only mlx and some
    # every mlx2 import; sharing one interpreter made them fail each other
    # (16 collection errors on 2026-10-01).  The guard command runs each alone.
    ordinary, guarded = qualify.preflight_test_commands()
    assert guarded[1:3] == ["scripts/qualify_serving.py", "--run-import-guards"]
    assert guarded[3:] == list(qualify.PREFLIGHT_IMPORT_GUARD_MODULES)
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return SimpleNamespace(returncode=1 if cmd[-1].endswith("b.py") else 0)

    monkeypatch.setattr(qualify, "PREFLIGHT_IMPORT_GUARD_MODULES", ("tests/a.py", "tests/b.py"))
    monkeypatch.setattr(qualify.subprocess, "run", fake_run)
    assert qualify.run_import_guards(["tests/a.py", "tests/b.py"]) == 1
    assert [c[-1] for c in calls] == ["tests/a.py", "tests/b.py"]
    assert all("--noconftest" in c and c.count("tests/a.py") + c.count("tests/b.py") == 1 for c in calls)
    calls.clear()
    assert qualify.run_import_guards(["tests/a.py"]) == 0
    assert qualify.run_import_guards(["tests/unlisted.py"]) == 2


# --------------------------------------------------------------------------
# Interpreter binding (2026-10-09, integration review item 3): a receipt must
# prove that this Python ran pytest.  Before the fix the validators discarded
# the command head, so `true` with pytest's arguments and exit 0 was a pass.
# --------------------------------------------------------------------------


def _true():
    exe = shutil.which("true")
    assert exe, "no `true` executable on this host"
    return exe


def _bound_identity():
    """The real preflight identity over an injected runtime: it carries the
    interpreter binding the validators compare."""
    return qualify.preflight_identity(runtime_identity_fn=lambda: {"source_sha256": "runtime"})


def test_interpreter_identity_is_the_canonical_executable_and_version():
    identity = qualify.interpreter_identity()
    assert set(identity) == {"executable", "implementation", "version"}
    assert identity["executable"] == sys.executable  # as invoked; compared canonically
    assert identity["implementation"] == platform.python_implementation()
    assert identity["version"] == "%d.%d.%d" % sys.version_info[:3]
    assert _bound_identity()["interpreter"] == identity


@pytest.mark.parametrize("lane", ["test_command", "guard_test_command"])
@pytest.mark.parametrize("head", ["true", "missing"])
def test_full_preflight_refuses_a_lane_this_interpreter_did_not_run(tmp_path, lane, head):
    identity = _bound_identity()
    path = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(path, run=_passing_run(), identity_fn=lambda: identity)
    receipt = json.loads(path.read_text())
    executable = _true() if head == "true" else str(tmp_path / "no-such-python")
    receipt[lane] = [executable, *receipt[lane][1:]]  # pytest's arguments, exit 0, no Python
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="interpreter"):
        qualify.validate_preflight_receipt(path, identity["runtime"], identity_fn=lambda: identity)


@pytest.mark.parametrize("field", ["executable", "implementation", "version", "absent"])
def test_full_preflight_refuses_another_interpreter_identity(tmp_path, field):
    identity = _bound_identity()
    path = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(path, run=_passing_run(), identity_fn=lambda: identity)
    receipt = json.loads(path.read_text())
    if field == "absent":
        del receipt["identity"]["interpreter"]
    else:
        receipt["identity"]["interpreter"][field] = "other"
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="interpreter"):
        qualify.validate_preflight_receipt(path, identity["runtime"], identity_fn=lambda: identity)


def test_full_preflight_accepts_another_path_to_the_same_interpreter(tmp_path):
    # Symlinks are resolved on both sides: a venv's python, python3 and the
    # binary they point at are one interpreter.
    identity = _bound_identity()
    path = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(path, run=_passing_run(), identity_fn=lambda: identity)
    receipt = json.loads(path.read_text())
    link = tmp_path / "python-alias"
    link.symlink_to(sys.executable)
    receipt["test_command"] = [str(link), *receipt["test_command"][1:]]
    receipt["guard_test_command"] = [os.path.realpath(sys.executable),
                                     *receipt["guard_test_command"][1:]]
    path.write_text(json.dumps(receipt))
    evidence = qualify.validate_preflight_receipt(
        path, identity["runtime"], identity_fn=lambda: identity
    )
    assert evidence["passed"]


@pytest.mark.parametrize("lane", ["test_command", "guard_test_command"])
def test_delta_preflight_refuses_a_lane_this_interpreter_did_not_run(tmp_path, lane):
    identity = _bound_identity()
    tree = qualify.preflight_tree()
    base = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(
        base, run=_passing_run(), identity_fn=lambda: identity, tree_fn=lambda: tree
    )
    current = _edited(tree, "scripts/sdk_smoke.py")
    delta_path = tmp_path / "delta.json"
    qualify.write_preflight_delta(
        delta_path, base, run=_passing_run(), identity_fn=lambda: identity,
        tree_fn=lambda: current,
    )
    receipt = json.loads(delta_path.read_text())
    assert receipt[lane], "the delta runs both lanes"
    receipt[lane] = [_true(), *receipt[lane][1:]]
    delta_path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="preflight delta.*interpreter"):
        qualify.validate_preflight_receipt(
            delta_path, identity["runtime"], identity_fn=lambda: identity,
            tree_fn=lambda: current,
        )


def test_delta_preflight_refuses_a_base_this_interpreter_did_not_run(tmp_path):
    identity = _bound_identity()
    tree = qualify.preflight_tree()
    base = tmp_path / "preflight.json"
    qualify.write_preflight_receipt(
        base, run=_passing_run(), identity_fn=lambda: identity, tree_fn=lambda: tree
    )
    receipt = json.loads(base.read_text())
    receipt["test_command"] = [_true(), *receipt["test_command"][1:]]
    base.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="interpreter"):
        qualify.write_preflight_delta(
            tmp_path / "delta.json", base, run=_passing_run(),
            identity_fn=lambda: identity, tree_fn=lambda: _edited(tree, "scripts/sdk_smoke.py"),
        )


def _cross_host_interpreter_case(tmp_path):
    """The v3 case over the real local interpreter on the target and an
    independently trusted, different interpreter on the source host."""
    path, bundle, arguments = _cross_host_case(tmp_path)
    source, target = bundle["source"], bundle["target"]
    local = qualify.interpreter_identity()
    remote = {"executable": "/Volumes/m5/mlx2/.venv/bin/python3.12",
              "implementation": local["implementation"], "version": local["version"]}
    assert remote != local
    target["identity"]["interpreter"] = dict(local)
    target["command"][0] = sys.executable
    arguments["target_command"] = list(target["command"])
    arguments["identity_fn"] = (lambda identity=copy.deepcopy(target["identity"]):
                                copy.deepcopy(identity))
    source["identity"]["interpreter"] = dict(remote)
    for suite in source["suites"].values():
        suite["command"] = [remote["executable"], *suite["command"][1:]]
    arguments["source_interpreter"] = dict(remote)
    return path, bundle, arguments


def test_cross_host_preflight_accepts_a_trusted_source_interpreter(tmp_path):
    path, bundle, arguments = _cross_host_interpreter_case(tmp_path)
    _write_cross_host(path, bundle)
    result = qualify.validate_cross_host_preflight_receipt(path, **arguments)
    assert result["passed"] and result["source_nodes"] == 2


@pytest.mark.parametrize("mutation", [
    "true_source_ordinary", "true_source_guard", "true_target",
    "source_identity_version", "source_identity_executable", "source_identity_absent",
    "trusted_source_absent", "trusted_source_malformed", "trusted_source_other",
    "target_identity_version", "target_identity_absent",
])
def test_cross_host_preflight_binds_both_interpreters(tmp_path, mutation):
    path, bundle, arguments = _cross_host_interpreter_case(tmp_path)
    source, target = bundle["source"], bundle["target"]
    if mutation == "true_source_ordinary":
        # The trusted source interpreter is right; the proof's command is not it.
        source["suites"]["ordinary"]["command"][0] = _true()
    elif mutation == "true_source_guard":
        source["suites"]["guard"]["command"][0] = _true()
    elif mutation == "true_target":
        target["command"][0] = _true()
        arguments["target_command"] = list(target["command"])
    elif mutation == "source_identity_version":
        source["identity"]["interpreter"]["version"] = "0.0.0"
    elif mutation == "source_identity_executable":
        source["identity"]["interpreter"]["executable"] = _true()
    elif mutation == "source_identity_absent":
        del source["identity"]["interpreter"]
    elif mutation == "trusted_source_absent":
        del arguments["source_interpreter"]
    elif mutation == "trusted_source_malformed":
        arguments["source_interpreter"] = {"executable": arguments["source_interpreter"]["executable"]}
    elif mutation == "trusted_source_other":
        # The proof is self-consistent, but the operator vouched for a different one.
        arguments["source_interpreter"]["version"] = "0.0.0"
    elif mutation == "target_identity_version":
        target["identity"]["interpreter"]["version"] = "0.0.0"
        arguments["identity_fn"] = (lambda identity=copy.deepcopy(target["identity"]):
                                    copy.deepcopy(identity))
    elif mutation == "target_identity_absent":
        del target["identity"]["interpreter"]
        arguments["identity_fn"] = (lambda identity=copy.deepcopy(target["identity"]):
                                    copy.deepcopy(identity))
    _write_cross_host(path, bundle)
    with pytest.raises(AssertionError, match="cross-host preflight"):
        qualify.validate_cross_host_preflight_receipt(path, **arguments)


@pytest.mark.parametrize("mutate_identity", [False, True])
def test_full_preflight_refuses_source_changes_during_passing_tests(tmp_path, mutate_identity):
    identity = {"runtime": {"source_sha256": "old"}}
    tree = {"tests/test_sample.py": "old"}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            if mutate_identity:
                identity["runtime"]["source_sha256"] = "new"
            else:
                tree["tests/test_sample.py"] = "new"
        return __import__("subprocess").CompletedProcess(command, 0, "passed", "")

    output = tmp_path / "preflight.json"
    with pytest.raises(AssertionError, match="source changed"):
        qualify.write_preflight_receipt(
            output, run=run, identity_fn=lambda: identity, tree_fn=lambda: tree
        )
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is False
    assert receipt["source_stable"] is False
    assert receipt["returncode"] == receipt["guard_returncode"] == 0
