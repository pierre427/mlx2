"""Host-only native-identity association controls for the segmented MoE numerical qualifier.

No MLX: a meta-path blocker refuses every ``mlx``/``mlx.*`` import (``mlx2`` is allowed) BEFORE the
qualifier and candidate are imported at collection time and during every test; cached MLX modules
are hidden, then restored. No device is queried or set. The original ``NativeBackend`` captured at
import has its constructor and every method forbidden, as do the candidate's native backend class
and research entry points, so nothing is constructed or dispatched. The fake is the retained
``FakeBackend`` of tests/test_segmented_moe_prefill_native_qualifier.py, extracted as SOURCE TEXT
(ast) so that module and its tests are never imported or collected here.

Run alone, without the repository conftest or plugin autoload:
  PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -p no:cacheprovider \
      -o addopts= -q tests/test_segmented_moe_native_identity_cpu.py

Positive association controls use an UNINITIALIZED instance of the original class and are labelled
host association checks: they isolate the association logic and are never native execution.
Evidence-association hygiene only, not protection against code that edits closures, class
attributes or private objects. Nothing here is native execution, qualification or a timing claim.
"""

import ast
import contextlib
import importlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_segmented_moe_prefill_research.py"
PROVENANCE = ROOT / "provenance/segmented-moe-prefill-native-qualifier.json"
RETAINED_FAKE_SOURCE = ROOT / "tests/test_segmented_moe_prefill_native_qualifier.py"
THIS_TEST = "tests/test_segmented_moe_native_identity_cpu.py"
MODULE_NAME = "mlx2_segmented_moe_native_identity_qualifier"


def _is_mlx(name):
    return name == "mlx" or name.startswith("mlx.")


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _is_mlx(name):
            raise ImportError(f"host-only identity control: MLX import blocked ({name})")
        return None


@contextlib.contextmanager
def _mlx_blocked():
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_mlx(k)}
    blocker = _BlockMLX()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        leaked = [k for k in sys.modules if _is_mlx(k)]
        for k in leaked:
            del sys.modules[k]
        sys.modules.update(hidden)
    assert not leaked, leaked


def _load():
    """The harness module (an in-memory variant may be pre-installed under MODULE_NAME)."""
    mod = sys.modules.get(MODULE_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


with _mlx_blocked():                       # collection time: blocked before the harness/candidate import
    q = _load()
    cand = q._candidate()
    MLX_AFTER_IMPORT = sorted(k for k in sys.modules if _is_mlx(k))

# The originals as the module bound them at import, before any test rebinds a module name.
ORIGINAL = types.SimpleNamespace(native_cls=q.NativeBackend, live_cls=q._LiveNativeRun, witness=q._WITNESS,
                                 evaluator=q.evaluate, run_native=q._run_native, native_verdict=q.native_verdict)


def _retained_fake():
    """FakeBackend and its helpers from the retained host-only test module, as source text only."""
    tree = ast.parse(RETAINED_FAKE_SOURCE.read_text())
    keep = {"_pattern", "_eng", "tiny_tables", "FakeBackend"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in keep]
    assert {n.name for n in nodes} == keep
    ns = {"np": np, "q": q, "cand": cand}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RETAINED_FAKE_SOURCE), "exec"), ns)
    return ns


_FAKE = _retained_fake()
FakeBackend, tiny_tables = _FAKE["FakeBackend"], _FAKE["tiny_tables"]

# ------------------------------------------------------------------ nothing native may run

_NATIVE_METHODS = ("__init__", "identity", "_linear", "load_tables", "drop_tables", "release", "_bits", "_routes",
                   "_call", "run_cell")
_CAND_METHODS = ("__init__", "require_capability", "upload_u32", "_kernel", "_tiles", "mapped_gate_up_swiglu",
                 "segmented_down")
_CAND_ENTRIES = ("research_mapped_gate_up_swiglu", "research_segmented_down")


class NativeForbidden(AssertionError):
    pass


def _forbid(label):
    def fn(*_a, **_k):
        raise NativeForbidden(f"host-only identity control: {label} must never run")
    return fn


@contextlib.contextmanager
def _nothing_native():
    """Forbid the captured original NativeBackend (constructor and methods), the candidate's captured
    native class and its research entries. Class/module objects keep their identity (the harness's
    candidate_refusal compares them), only their callables are replaced, then restored."""
    saved = []
    for owner, names in ((ORIGINAL.native_cls, _NATIVE_METHODS), (cand._MLXBackend, _CAND_METHODS),
                         (cand, _CAND_ENTRIES)):
        for name in names:
            saved.append((owner, name, owner.__dict__[name]))
            setattr(owner, name, _forbid(f"{getattr(owner, '__name__', owner)}.{name}"))
    before = dict(cand.ENGAGEMENT.snapshot())
    try:
        with _mlx_blocked():
            yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)
    assert dict(cand.ENGAGEMENT.snapshot()) == before, "candidate engagement counters moved"


@pytest.fixture(autouse=True)
def nothing_native():
    with _nothing_native():
        yield


# ------------------------------------------------------------------ the root control (full catalogue)

def _per_kind(cells):
    per_kind = {k: 0 for k in q.KINDS}
    for c in cells:
        if c["expect"] == "pass":
            for e in q.expectations(c, cand).values():
                if e.get("kind") in q.KINDS:
                    per_kind[e["kind"]] += 1
    return per_kind


@pytest.fixture(scope="module")
def full():
    """Root reproduction inputs: FakeBackend(prefix='native', native=True) over the full mandatory
    catalogue through the actual run_gate and evaluate, with matching fresh native counters."""
    with _nothing_native():
        cells, full_requested = q.select_cells(None)
        fake = FakeBackend(prefix="native", native=True)
        report = q.run_gate(fake, cells, tables=tiny_tables)
        evaluation = ORIGINAL.evaluator(report, cells, full_requested=full_requested)
    per_kind = _per_kind(cells)
    fresh = {"calls": sum(per_kind.values()), **{f"native.successful_chains.{k}": v for k, v in per_kind.items()}}
    return types.SimpleNamespace(cells=cells, full_requested=full_requested, fake=fake, report=report,
                                 evaluation=evaluation, per_kind=per_kind, fresh=fresh)


def _live(full, backend, *, witness=None, post=(), fresh=None, cells=None, full_requested=True,
          live_cls=None):
    return (live_cls or ORIGINAL.live_cls)(ORIGINAL.witness if witness is None else witness, backend, {},
                                           full.report, list(post), full.cells if cells is None else cells,
                                           full_requested, full.fresh if fresh is None else fresh)


def _uninitialized_original():
    """HOST ASSOCIATION CHECK ONLY: an uninitialized instance of the captured original class. It isolates
    the association logic; it never ran MLX, a device or a kernel and is not native execution."""
    return object.__new__(ORIGINAL.native_cls)


def test_collection_import_was_blocked_and_no_mlx_is_loaded():
    assert MLX_AFTER_IMPORT == []
    assert not [k for k in sys.modules if _is_mlx(k)]
    with pytest.raises(ImportError, match="blocked"):
        importlib.import_module("mlx.core")
    assert Path(cand.__file__).resolve() == ROOT / q.CANDIDATE_FILE
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1", "run without plugin autoload"
    assert not [m for m in sys.modules if m == "conftest" or m.endswith(".conftest")], "run with --noconftest"


def test_full_catalogue_root_inputs_reproduce_the_root_control(full):
    assert full.full_requested is True and len(full.cells) == len(q.CATALOGUE) == 21
    assert full.evaluation["verdict"] == "bit_identity_evidence_pass", full.evaluation["refusals"][:5]
    assert full.per_kind == {"gate_up_mapped_swiglu": 26, "down_segmented": 40}
    assert full.evaluation["chains"] == {"native": 66, "substituted": 0} and full.evaluation["expected_chains"] == 66
    assert full.evaluation["native_synthetic_gate"] is False
    assert full.report["producer"] == "FakeBackend"


def test_root_control_supplied_class_is_now_a_type_error_and_plain_call_refuses(full):
    live = _live(full, full.fake)
    with pytest.raises(TypeError):
        q.native_verdict(live, FakeBackend)                     # the root reproduction call
    v = q.native_verdict(live)
    assert v["native_synthetic_gate"] is False and v["reasons"] == ["backend is not the native backend"]
    assert v["evaluation"]["verdict"] == "bit_identity_evidence_pass"
    assert all(v[k] is False for k in ("qualified", "selected", "observed_used", "model_gain"))


def test_signatures_accept_only_live_and_run_arguments():
    assert list(inspect.signature(q.native_verdict).parameters) == ["live"]
    assert list(inspect.signature(q._run_native).parameters) == ["admission", "cells", "full_requested"]
    for fn in (q.native_verdict, q._run_native):
        assert fn.__defaults__ is None and fn.__kwdefaults__ is None
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in inspect.signature(fn).parameters.values())


_OVERRIDES = {"_cls": FakeBackend, "_live": type("_LiveNativeRun", (), {}), "_witness": object(),
              "_evaluate": lambda r, c, full_requested: {"verdict": "bit_identity_evidence_pass"},
              "native_cls": FakeBackend, "live_cls": object, "witness": object(), "evaluator": len}


@pytest.mark.parametrize("name", sorted(_OVERRIDES))
def test_keyword_identity_overrides_raise_type_error(full, name):
    live = _live(full, full.fake)
    with pytest.raises(TypeError):
        q.native_verdict(live, **{name: _OVERRIDES[name]})
    with pytest.raises(TypeError):
        q._run_native({}, full.cells, True, **{name: _OVERRIDES[name]})


@pytest.mark.parametrize("extra", [(FakeBackend,), (FakeBackend, ORIGINAL.live_cls),
                                   (FakeBackend, ORIGINAL.live_cls, ORIGINAL.witness),
                                   (FakeBackend, ORIGINAL.live_cls, ORIGINAL.witness, ORIGINAL.evaluator)])
def test_positional_identity_overrides_raise_type_error(full, extra):
    with pytest.raises(TypeError):
        q.native_verdict(_live(full, full.fake), *extra)
    with pytest.raises(TypeError):                              # bound before the captured constructor runs
        q._run_native({}, full.cells, True, *extra)


def test_closures_capture_the_import_time_originals():
    verdict = inspect.getclosurevars(q.native_verdict).nonlocals
    assert verdict == {"native_cls": ORIGINAL.native_cls, "live_cls": ORIGINAL.live_cls,
                       "witness": ORIGINAL.witness, "evaluator": ORIGINAL.evaluator}
    run = inspect.getclosurevars(q._run_native).nonlocals
    assert run == {"native_cls": ORIGINAL.native_cls, "live_cls": ORIGINAL.live_cls, "witness": ORIGINAL.witness}
    assert ORIGINAL.native_cls.__name__ == "NativeBackend" and ORIGINAL.native_cls.__module__ == MODULE_NAME
    assert ORIGINAL.live_cls.__name__ == "_LiveNativeRun" and ORIGINAL.evaluator.__name__ == "evaluate"
    assert ORIGINAL.evaluator.__module__ == MODULE_NAME and type(ORIGINAL.witness) is object
    for fn, name in ((q.native_verdict, "native_verdict"), (q._run_native, "_run_native")):
        assert fn.__qualname__ == name and fn.__module__ == MODULE_NAME
    assert q._bind_native_identity.__module__ == MODULE_NAME


def test_module_name_rebinding_cannot_change_construction(monkeypatch, full):
    built = []

    class Rebound(FakeBackend):
        def __init__(self, admission):
            built.append(admission)
            super().__init__(prefix="native", native=True)
    monkeypatch.setattr(q, "NativeBackend", Rebound)
    monkeypatch.setattr(q, "_LiveNativeRun", type("_LiveNativeRun", (), {"__init__": lambda self, *a: None}))
    monkeypatch.setattr(q, "_WITNESS", object())
    monkeypatch.setattr(q, "evaluate", _forbid("rebound evaluate"))
    with pytest.raises(NativeForbidden, match="NativeBackend.__init__"):   # the captured original, not Rebound
        q._run_native({}, full.cells, True)
    assert built == []


def test_cli_constructs_the_captured_original_and_writes_no_receipt(monkeypatch, capsys, tmp_path):
    built = []
    monkeypatch.setattr(sys, "path", list(sys.path))             # main() prepends src; restore afterwards
    monkeypatch.setattr(q, "native_admission", lambda args, environ: {"commit": "x"})
    monkeypatch.setattr(q, "NativeBackend", lambda admission: built.append(admission) or FakeBackend())
    out = tmp_path / "receipt.json"
    code = q.main(["--run-native", "--i-own-the-gpu", "--source-root", str(ROOT), "--source-commit", "0" * 40,
                   "--out", str(out)], environ={})
    printed = json.loads(capsys.readouterr().out)
    assert code == 1 and printed["verdict"] == "refused" and printed["native_synthetic_gate"] is False
    assert "NativeForbidden" in printed["refusals"][0] and "NativeBackend.__init__" in printed["refusals"][0]
    assert built == [] and not out.exists()


def test_module_name_rebinding_cannot_make_the_full_fake_native(monkeypatch, full):
    consulted = []
    monkeypatch.setattr(q, "NativeBackend", FakeBackend)
    monkeypatch.setattr(q, "_LiveNativeRun", type("_LiveNativeRun", (), {}))
    monkeypatch.setattr(q, "_WITNESS", object())
    monkeypatch.setattr(q, "evaluate",
                        lambda *a, **k: consulted.append(a) or {"verdict": "bit_identity_evidence_pass"})
    v = q.native_verdict(_live(full, full.fake))
    assert v["native_synthetic_gate"] is False and v["reasons"] == ["backend is not the native backend"]
    assert consulted == [] and v["evaluation"]["chains"]["native"] == 66


def test_fake_subclass_lookalike_and_wrong_witness_are_refused(full):
    sub = type("NativeBackend", (ORIGINAL.native_cls,), {})
    assert q.native_verdict(_live(full, object.__new__(sub)))["reasons"] == ["backend is not the native backend"]
    for live in (_live(full, _uninitialized_original(), witness=object()),
                 _live(full, _uninitialized_original(), live_cls=type("_LiveNativeRun", (ORIGINAL.live_cls,), {}))):
        v = q.native_verdict(live)
        assert v["native_synthetic_gate"] is False
        assert v["reasons"][0].startswith("no live native orchestration")
    look = types.SimpleNamespace(report=full.report, cells=full.cells, full_requested=True, _witness=ORIGINAL.witness,
                                 backend=_uninitialized_original(), post_run=[], fresh=full.fresh)
    assert q.native_verdict(look)["native_synthetic_gate"] is False


def test_evidence_alone_never_stamps(full):
    rt = json.loads(json.dumps(full.report, default=str))
    for obj in (full.report, rt, {"report": rt, "cells": full.cells, "_witness": None}, full.evaluation, None):
        v = q.native_verdict(obj)
        assert v["native_synthetic_gate"] is False
        assert any("no live native orchestration" in r for r in v["reasons"])


def test_host_association_check_original_class_full_scope_is_admissible(full):
    """HOST ASSOCIATION CHECK, not native execution: with an uninitialized original-class instance,
    the captured witness/run type and the actual evaluator, only counters and post-run gate."""
    v = q.native_verdict(_live(full, _uninitialized_original()))
    assert v["native_synthetic_gate"] is True and v["reasons"] == []
    assert all(v[k] is False for k in ("qualified", "selected", "observed_used", "model_gain"))


def test_host_association_check_post_run_refusal_is_honestly_rejected(full):
    post = ["git HEAD differs from the admitted commit", "module mlx2.x loaded from outside this checkout (/x)"]
    v = q.native_verdict(_live(full, _uninitialized_original(), post=post))
    assert v["native_synthetic_gate"] is False and v["reasons"] == post


@pytest.mark.parametrize("change", ["empty", "short_down", "stray", "substituted", "extra_calls", "fewer_calls"])
def test_host_association_check_fresh_counters_must_match(full, change):
    f = dict(full.fresh)
    key = "native.successful_chains.down_segmented"
    f = {"empty": {}, "short_down": dict(f, **{key: f[key] - 1}), "stray": dict(f, backend_raised=1),
         "substituted": dict(f, **{"substituted.successful_chains.down_segmented": 1}),
         "extra_calls": dict(f, calls=f["calls"] + 1), "fewer_calls": dict(f, calls=f["calls"] - 1)}[change]
    v = q.native_verdict(_live(full, _uninitialized_original(), fresh=f))
    assert v["native_synthetic_gate"] is False and v["reasons"], change


def test_host_association_check_partial_scope_uses_the_actual_evaluator(monkeypatch, full):
    consulted = []
    monkeypatch.setattr(q, "evaluate", lambda *a, **k: consulted.append(a) or {
        "verdict": "bit_identity_evidence_pass", "chains": {"native": 66, "substituted": 0}, "expected_chains": 66})
    for cells, flag in ((full.cells[:3], True), (full.cells[:3], False), (full.cells, False)):
        v = q.native_verdict(_live(full, _uninitialized_original(), cells=cells, full_requested=flag))
        assert v["native_synthetic_gate"] is False
        assert v["evaluation"]["verdict"] != "bit_identity_evidence_pass"
        assert "bit-identity evidence did not pass the full mandatory catalogue" in v["reasons"]
    assert consulted == []


def test_new_test_is_bound_in_harness_files_and_provenance():
    assert q.HARNESS_FILES == ("scripts/qualify_segmented_moe_prefill_research.py",
                               "tests/test_segmented_moe_prefill_native_qualifier.py",
                               "provenance/segmented-moe-prefill-native-qualifier.json", THIS_TEST)
    assert THIS_TEST in q.BOUND_FILES and THIS_TEST not in q.FROZEN
    prov = json.loads(PROVENANCE.read_text())
    assert prov["destination_paths"] == list(q.HARNESS_FILES)
    assert all(prov[k] is False for k in ("native_run", "qualified", "selected", "observed_used", "model_gain"))
    repair = prov["native_identity_repair_2026_10_01"]
    assert all(repair[k] is False for k in ("native_run", "qualified", "selected", "observed_used", "model_gain"))


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, *args):
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", *args)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A temporary git checkout holding the bound files (expected bytes come from its own HEAD blobs)."""
    repo = (tmp_path / "mlx2").resolve()
    for rel in q.BOUND_FILES:
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    (repo / ".gitignore").write_text("__pycache__/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _commit(repo, "-m", "fixture")
    monkeypatch.setattr(sys, "path", [str(repo / "src")] + sys.path)
    monkeypatch.delitem(sys.modules, "mlx2", raising=False)
    monkeypatch.setattr(q, "mlx_base", _forbid("mlx_base"))
    (tmp_path / "exists.json").write_text("{}")

    def admit():
        head = _git(repo, "rev-parse", "HEAD")
        args = q.build_parser().parse_args(["--run-native", "--i-own-the-gpu", "--source-root", str(repo),
                                            "--source-commit", head, "--out", str(tmp_path / "exists.json")])
        with pytest.raises(q.Refused) as info:
            q.native_admission(args, {"MLX2_INTAKE_SOURCE_COMMIT": head}, root=repo)
        return str(info.value)
    return types.SimpleNamespace(repo=repo, admit=admit)


def test_committed_head_guard_binds_the_new_test(checkout):
    assert "--out exists" in checkout.admit()                  # every source check passed; refused at --out
    (checkout.repo / THIS_TEST).write_text("# edited\n")
    assert THIS_TEST in checkout.admit()                       # dirty bytes against HEAD refuse
    _commit(checkout.repo, "-am", "edit")
    assert "--out exists" in checkout.admit()                  # no self-pinned hash: the new HEAD blob is expected
    _git(checkout.repo, "rm", "-q", THIS_TEST)
    _commit(checkout.repo, "-m", "drop")
    assert "bound files not committed at HEAD" in checkout.admit() and THIS_TEST in checkout.admit()


_PRELUDE = """
import importlib.abc, runpy, sys
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, n, p=None, t=None):
        if n == "mlx" or n.startswith("mlx."):
            raise ImportError("blocked " + n)
sys.meta_path.insert(0, B())
script, argv = sys.argv[1], sys.argv[2:]
sys.argv = [script] + argv
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as e:
    code = e.code or 0
assert not [m for m in sys.modules if m == "mlx" or m.startswith("mlx.")]
print("HOST-ONLY-EXIT", code)
"""


@pytest.mark.parametrize("argv,code,text", [(["--catalogue"], 0, '"mandatory"'),
                                            (["--run-native", "--source-root", str(ROOT)], 1, "acknowledgement"),
                                            (["--timing", "--run-native", "--i-own-the-gpu"], 1, "timing")])
def test_fresh_process_cli_stays_host_only(argv, code, text):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.pop("MLX2_INTAKE_SOURCE_COMMIT", None)
    proc = subprocess.run([sys.executable, "-c", _PRELUDE, str(SCRIPT), *argv], capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=120)
    assert f"HOST-ONLY-EXIT {code}" in proc.stdout and text in proc.stdout, proc.stderr[-2000:]
