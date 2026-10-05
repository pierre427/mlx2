"""Host-only native-identity association controls for the tensor attention numerical qualifier.

No MLX: a meta-path blocker refuses every ``mlx``/``mlx.*`` import (``mlx2`` is allowed) while the
qualifier and timing scripts are imported and during every test, and any cached MLX modules are
hidden, then restored. No device is queried or set. The original ``NativeBackend`` class is
captured at import and its ``__init__`` is forbidden in every test, so it is never constructed.
The fake backend is the retained ``FakeBackend`` of tests/test_tensor_fa_research_qualifier.py,
extracted as SOURCE TEXT (ast): that module imports MLX and is never imported here. Its outputs
come from the qualifier's independent float64 host reference, never the candidate host mirror.

Run without the repository conftest (it imports MLX and sets the default device):
  PYTHONPATH=src:. python -m pytest --noconftest -p no:cacheprovider tests/test_tensor_fa_native_identity_cpu.py

Evidence-association hygiene only, not protection against code that edits closures, class
attributes or private objects. Nothing here is native execution, qualification or a timing claim.
"""

import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
QUALIFIER = ROOT / "scripts" / "qualify_tensor_fa_research.py"
RETAINED_FAKE_SOURCE = ROOT / "tests" / "test_tensor_fa_research_qualifier.py"
THIS_TEST = "tests/test_tensor_fa_native_identity_cpu.py"
MX_FILE = "/nonexistent-mlx-identity-control/core.so"


def _is_mlx(name):
    return name == "mlx" or name.startswith("mlx.")


class _BlockMLX:
    """Meta-path finder: any mlx import fails loudly."""

    def find_spec(self, name, path=None, target=None):
        if _is_mlx(name):
            raise ImportError(f"{name} import is blocked in host-only identity controls")
        return None


_BLOCKER = _BlockMLX()


def _hide_mlx():
    hidden = {n: sys.modules.pop(n) for n in list(sys.modules) if _is_mlx(n)}
    sys.meta_path.insert(0, _BLOCKER)
    return hidden


def _restore_mlx(hidden):
    while _BLOCKER in sys.meta_path:
        sys.meta_path.remove(_BLOCKER)
    leaked = sorted(n for n in sys.modules if _is_mlx(n))
    for n in leaked:
        del sys.modules[n]
    sys.modules.update(hidden)
    return leaked


_hidden = _hide_mlx()
try:
    for _p in (str(ROOT / "src"), str(ROOT)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from scripts import qualify_tensor_fa_research as Q  # noqa: E402
    from scripts import benchmark_tensor_fa_research as T  # noqa: E402
finally:
    _LEAKED_AT_IMPORT = _restore_mlx(_hidden)

ORIGINAL_NATIVE = Q.NativeBackend          # the class only; its __init__ is forbidden in every test
ORIGINAL_LIVE, ORIGINAL_WITNESS, ORIGINAL_EVALUATE = Q._LiveNativeRun, Q._WITNESS, Q.evaluate


def _forbid(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run in host-only controls")
    return fail


def _retained_fake_class(module):
    """FakeBackend (and its helper) from the retained test module, compiled from source text only."""
    tree = ast.parse(RETAINED_FAKE_SOURCE.read_text())
    keep = [n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {"FakeBackend", "_stale_max_output"}]
    assert sorted(n.name for n in keep) == ["FakeBackend", "_stale_max_output"]
    names = {x.id for n in keep for x in ast.walk(n) if isinstance(x, ast.Name)}
    assert "mx" not in names and "mlx" not in names
    ns = {"Q": module, "np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(RETAINED_FAKE_SOURCE), "exec"), ns)
    return ns["FakeBackend"]


FakeBackend = _retained_fake_class(Q)


class GateFake(FakeBackend):
    """The retained fake shaped like a gate backend factory product (admission arg, module paths)."""

    def __init__(self, admission=None):
        super().__init__()
        self.cand = types.SimpleNamespace(__file__=str(Q.ROOT / Q.CANDIDATE_MODULE))
        self.mx = types.SimpleNamespace(__file__=MX_FILE)


@pytest.fixture(autouse=True)
def host_only(monkeypatch):
    hidden = _hide_mlx()
    monkeypatch.setattr(ORIGINAL_NATIVE, "__init__", _forbid("original NativeBackend construction"))
    monkeypatch.setattr(T.NativeTimingOps, "dispatch", _forbid("NativeTimingOps.dispatch"))
    try:
        yield
    finally:
        leaked = _restore_mlx(hidden)
    assert leaked == []


@pytest.fixture(scope="module")
def full_run():
    hidden = _hide_mlx()
    try:
        cases, full = Q.select_cases()
        report = Q.run_gate(FakeBackend(), cases, source=lambda: {"f": "a" * 64})
    finally:
        assert _restore_mlx(hidden) == []
    return cases, full, report


def _shell(cls=None):
    """An instance of the class WITHOUT __init__ (no MLX, no device). Private construction is not a
    security boundary; it isolates the association checks."""
    return object.__new__(cls or ORIGINAL_NATIVE)


def _failing(report):
    bad = copy.deepcopy(report)
    for record in bad["cases"]:
        if record["id"] == "c03-d128-f16-g6-w31-mid":
            record["evidence"]["candidate"] = (record["evidence"]["candidate"] * 1.5).astype(np.float32)
    return bad


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- import, CLI, admission binding

def test_fresh_import_help_and_refusals_import_no_mlx():
    assert _LEAKED_AT_IMPORT == []
    code = (
        "import sys, io, contextlib\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'mlx' or name.startswith('mlx.'):\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, B())\n"
        "import scripts.qualify_tensor_fa_research as Q\n"
        "import scripts.benchmark_tensor_fa_research as T\n"
        "rc = []\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    rc.append(Q.main(['--catalogue']))\n"
        "    for main in (Q.main, T.main):\n"
        "        try: main(['--help'])\n"
        "        except SystemExit as e: rc.append(e.code)\n"
        "    rc.append(Q.main(['--out', '/nonexistent-identity-control/never.json'], environ={}))\n"
        "    rc.append(Q.main(['--timing'], environ={}))\n"
        "print(rc, [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.')],\n"
        "      'mlx2.adapters.tensor_fa_research' in sys.modules)\n")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                           env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip().splitlines()[-1] == "[0, 0, 0, 1, 1] [] False"


def test_this_control_module_is_bound_and_must_be_committed_at_head(monkeypatch, tmp_path):
    assert THIS_TEST in Q.SOURCE_FILES and len(set(Q.SOURCE_FILES)) == len(Q.SOURCE_FILES)
    assert _sha(QUALIFIER) not in QUALIFIER.read_text()           # no self-pinned hash
    missing = [rel for rel in Q.SOURCE_FILES if not (Q.ROOT / rel).is_file()]
    if missing:
        pytest.skip(f"bound source absent: {missing[0]} (private provenance is not "
                    "exported to the public mirror)")
    args = types.SimpleNamespace(run_native=True, i_own_the_gpu=True, source_root=str(Q.ROOT),
                                 source_commit="a" * 40, out=str(tmp_path / "never.json"))
    env = {"MLX2_INTAKE_SOURCE_COMMIT": "a" * 40}

    def git(missing):
        def run(*a):
            if a[0] == "rev-parse":
                return b"a" * 40 + b"\n"
            if a[0] == "status":
                return b""
            path = a[1].split(":", 1)[1]
            return None if path == missing else (Q.ROOT / path).read_bytes()
        return run

    monkeypatch.setattr(Q, "mlx_files", _forbid("MLX file hashing"))
    monkeypatch.setattr(Q, "_git", git(THIS_TEST))
    with pytest.raises(Q.Refused, match=f"{THIS_TEST} is not committed at HEAD"):
        Q.native_admission(args, env)
    monkeypatch.setattr(Q, "_git", git(None))                      # every bound file equals its HEAD blob
    with pytest.raises(AssertionError, match="MLX file hashing"):  # source admission passed, still no MLX
        Q.native_admission(args, env)


# ---------------------------------------------------------------- root repro and class identity

def test_root_repro_full_fake_report_with_rebound_native_class_is_refused(monkeypatch, full_run):
    cases, full, report = full_run
    before = _sha(QUALIFIER)
    evaluation = Q.evaluate(report, cases, full_requested=full)
    assert full is True and len(cases) == 12 and evaluation["verdict"] == "numeric_evidence_pass"
    monkeypatch.setattr(Q, "NativeBackend", FakeBackend)
    native = Q.native_verdict(Q._LiveNativeRun(Q._WITNESS, FakeBackend(), {}, report, [], cases, full))
    assert native["native_synthetic_gate"] is False
    assert native["reasons"] == ["backend is not the native backend"]
    assert native["evaluation"]["verdict"] == "numeric_evidence_pass"
    assert (native["qualified"], native["selected"], native["model_gain"]) == (False, False, False)
    assert _sha(QUALIFIER) == before and Q.evaluate is ORIGINAL_EVALUATE


def test_subclasses_and_lookalike_classes_are_not_the_native_backend(monkeypatch, full_run):
    cases, full, report = full_run

    class Sub(ORIGINAL_NATIVE):
        pass

    lookalike = type("NativeBackend", (), {"__module__": Q.__name__, "__qualname__": "NativeBackend",
                                           **{k: v for k, v in vars(ORIGINAL_NATIVE).items()
                                              if k not in ("__dict__", "__weakref__")}})
    for cls in (Sub, lookalike, GateFake):
        monkeypatch.setattr(Q, "NativeBackend", cls)
        native = Q.native_verdict(Q._LiveNativeRun(Q._WITNESS, _shell(cls), {}, report, [], cases, full))
        assert native["reasons"] == ["backend is not the native backend"], cls


def test_run_native_constructs_the_captured_original_not_the_rebound_name(monkeypatch, full_run):
    cases, full, _ = full_run
    made = []

    class Rebound(GateFake):
        def __init__(self, admission=None):
            made.append(admission)
            raise RuntimeError("rebound fake constructed")

    monkeypatch.setattr(Q, "NativeBackend", Rebound)
    monkeypatch.setattr(Q, "run_gate", _forbid("gate dispatch"))
    with pytest.raises(AssertionError, match="original NativeBackend construction"):
        Q._run_native({"commit": "a" * 40}, cases, full)
    assert made == []


def test_no_caller_argument_can_supply_a_class_live_type_or_evaluator(full_run):
    cases, full, report = full_run
    assert list(inspect.signature(Q.native_verdict).parameters) == ["live"]
    assert list(inspect.signature(Q._run_native).parameters) == ["admission", "cases", "full_requested"]
    live = ORIGINAL_LIVE(ORIGINAL_WITNESS, FakeBackend(), {}, report, [], cases, full)
    for call in (lambda: Q.native_verdict(live, FakeBackend), lambda: Q.native_verdict(live, native_cls=FakeBackend),
                 lambda: Q.native_verdict(live, evaluator=lambda *a, **k: {"verdict": "numeric_evidence_pass"})):
        with pytest.raises(TypeError):
            call()


# ---------------------------------------------------------------- live type / witness / evaluator rebinding

def test_rebinding_live_type_witness_and_evaluator_names_does_not_forge_nativeness(monkeypatch, full_run):
    cases, full, report = full_run
    calls = []

    class LookalikeLive:
        __slots__ = ORIGINAL_LIVE.__slots__
        __init__ = ORIGINAL_LIVE.__init__

    forged_witness = object()
    monkeypatch.setattr(Q, "_LiveNativeRun", LookalikeLive)
    monkeypatch.setattr(Q, "_WITNESS", forged_witness)
    monkeypatch.setattr(Q, "evaluate", lambda *a, **k: calls.append(a) or {"verdict": "numeric_evidence_pass"})
    for live in (Q._LiveNativeRun(Q._WITNESS, _shell(), {}, report, [], cases, full),
                 ORIGINAL_LIVE(Q._WITNESS, _shell(), {}, report, [], cases, full)):
        native = Q.native_verdict(live)
        assert native["native_synthetic_gate"] is False
        assert native["reasons"][0].startswith("no live native orchestration")
    # The captured evaluator judges the run's own report: a rebound always-pass evaluator is never consulted.
    native = Q.native_verdict(ORIGINAL_LIVE(ORIGINAL_WITNESS, _shell(), {}, _failing(report), [], cases, full))
    assert native["native_synthetic_gate"] is False and native["evaluation"]["verdict"] == "refused"
    assert any("c03-d128-f16-g6-w31-mid: candidate outside" in r for r in native["evaluation"]["refusals"])
    assert calls == []


@pytest.mark.parametrize("make", [
    lambda r, c, f: r,
    lambda r, c, f: json.loads(json.dumps({"report": Q._jsonable(r), "native_execution": {"native_synthetic_gate": True}})),
    lambda r, c, f: {"native_synthetic_gate": True, "report": r, "cases": c, "full_requested": f},
    lambda r, c, f: types.SimpleNamespace(_witness=ORIGINAL_WITNESS, backend=_shell(), post_run=[], report=r,
                                          cases=c, full_requested=f),
], ids=["report-dict", "receipt-json", "verdict-dict", "namespace-with-witness"])
def test_plain_receipts_and_json_stay_evidence_only(full_run, make):
    cases, full, report = full_run
    native = Q.native_verdict(make(report, cases, full))
    assert native["native_synthetic_gate"] is False
    assert native["reasons"][0].startswith("no live native orchestration")


def test_partial_or_mismatched_cases_on_an_associated_run_stay_evidence_only(full_run):
    cases, full, report = full_run
    subset = cases[:3]
    partial = {**report, "cases": [r for r in report["cases"] if r["id"] in {c["id"] for c in subset}]}
    for rep, cs, fl, verdict in ((partial, subset, False, "partial_numeric_evidence"),
                                 (partial, subset, True, "refused"),
                                 (report, cases[:2], True, "refused"),
                                 (report, cases, False, "partial_numeric_evidence")):
        native = Q.native_verdict(ORIGINAL_LIVE(ORIGINAL_WITNESS, _shell(), {}, rep, [], cs, fl))
        assert native["evaluation"]["verdict"] == verdict
        assert native["reasons"] == ["numeric evidence did not pass the full mandatory catalogue"]


def test_a_valid_scope_on_the_captured_class_is_admissible_and_still_checks_post_run(full_run):
    cases, full, report = full_run
    ok = Q.native_verdict(ORIGINAL_LIVE(ORIGINAL_WITNESS, _shell(), {}, report, [], cases, full))
    assert ok["native_synthetic_gate"] is True and ok["reasons"] == []
    assert (ok["qualified"], ok["selected"], ok["model_gain"]) == (False, False, False)
    post = ["git HEAD differs from the admitted commit"]
    native = Q.native_verdict(ORIGINAL_LIVE(ORIGINAL_WITNESS, _shell(), {}, report, post, cases, full))
    assert native["native_synthetic_gate"] is False and native["reasons"] == post


# ---------------------------------------------------------------- timing's combined numeric gate

def _quiet_guards(monkeypatch):
    monkeypatch.setattr(T, "timing_guard", lambda admission, paths=None: None)
    monkeypatch.setattr(Q, "source_guard", lambda admission: None)
    return {"commit": "a" * 40, "mlx_files_before": {MX_FILE: "0" * 64}}


@pytest.mark.parametrize("rebind", [False, True], ids=["supplied-factory", "rebound-module-name"])
def test_timing_numeric_gate_with_a_fake_cannot_reach_timing_ops(monkeypatch, full_run, rebind):
    cases, full, _ = full_run
    admission = _quiet_guards(monkeypatch)
    if rebind:
        monkeypatch.setattr(Q, "NativeBackend", GateFake)
    live = T._run_numeric_gate(admission, cases, full, backend_factory=None if rebind else GateFake)
    assert type(live) is ORIGINAL_LIVE and live.post_run == [] and type(live.backend) is GateFake
    native = Q.native_verdict(live)
    assert native["evaluation"]["verdict"] == "numeric_evidence_pass"
    assert native["reasons"] == ["backend is not the native backend"]
    with pytest.raises(Q.Refused, match="live numeric gate did not pass"):
        T.NativeTimingOps(live)


def test_timing_run_with_a_rebound_fake_never_builds_ops_or_times(monkeypatch, full_run):
    admission = _quiet_guards(monkeypatch)
    monkeypatch.setattr(Q, "NativeBackend", GateFake)
    monkeypatch.setattr(T, "NativeTimingOps", _forbid("NativeTimingOps"))
    monkeypatch.setattr(T, "run_timing", _forbid("run_timing"))
    with pytest.raises(Q.Refused, match="live numeric gate did not pass; no timed dispatch: backend is not the native"):
        T._run_native_timing(admission)


def test_timing_default_factory_without_rebinding_targets_the_original_class(monkeypatch, full_run):
    cases, full, _ = full_run
    admission = _quiet_guards(monkeypatch)
    monkeypatch.setattr(Q, "run_gate", _forbid("gate dispatch"))
    with pytest.raises(AssertionError, match="original NativeBackend construction"):
        T._run_numeric_gate(admission, cases, full)


# ---------------------------------------------------------------- in-memory mutations of this repair only

MUTATIONS = {
    "class-check-by-module-name": ("elif type(live.backend) is not native_cls:",
                                   "elif type(live.backend) is not NativeBackend:"),
    "constructor-by-module-name": ("backend = native_cls(admission)", "backend = NativeBackend(admission)"),
    "evaluator-by-module-name": ("evaluation = evaluator(report, list(cases), full_requested=full)",
                                 "evaluation = evaluate(report, list(cases), full_requested=full)"),
    "live-type-and-witness-by-module-name": (
        'if type(live) is not live_cls or getattr(live, "_witness", None) is not witness:',
        'if type(live) is not _LiveNativeRun or getattr(live, "_witness", None) is not _WITNESS:'),
    "overridable-default-argument": ("    def native_verdict(live):", "    def native_verdict(live, native_cls=native_cls):"),
    "post-run-dropped": ("            reasons += live.post_run\n", "            pass\n"),
}


def _load(source, name):
    module = types.ModuleType(name)
    module.__file__ = str(QUALIFIER)
    saved = list(sys.path)
    try:
        exec(compile(source, str(QUALIFIER), "exec"), module.__dict__)
    finally:
        sys.path[:] = saved
    return module


def _forgeries(M, report, cases, full):
    """Which association controls a module FAILS (True = a CPU fake or bad run was promoted)."""
    fake_cls = _retained_fake_class(M)
    original, live_cls, witness, evaluate = M.NativeBackend, M._LiveNativeRun, M._WITNESS, M.evaluate
    original_init = vars(original)["__init__"]
    out = {}
    try:
        M.NativeBackend = fake_cls
        out["class"] = M.native_verdict(live_cls(witness, fake_cls(), {}, report, [], cases, full))["native_synthetic_gate"]

        def rebound(admission):
            raise RuntimeError("rebound fake constructed")
        original.__init__ = lambda self, admission: (_ for _ in ()).throw(AssertionError("original constructed"))
        M.NativeBackend = rebound
        try:
            M._run_native({}, cases, full)
        except AssertionError:
            out["constructor"] = False
        except RuntimeError:
            out["constructor"] = True
        M.NativeBackend = original

        M.evaluate = lambda *a, **k: {"verdict": "numeric_evidence_pass"}
        out["evaluator"] = M.native_verdict(
            live_cls(witness, _shell(original), {}, _failing(report), [], cases, full))["native_synthetic_gate"]
        M.evaluate = evaluate

        class LookalikeLive:
            __slots__ = live_cls.__slots__
            __init__ = live_cls.__init__
        M._LiveNativeRun, M._WITNESS = LookalikeLive, object()
        out["live_witness"] = M.native_verdict(
            LookalikeLive(M._WITNESS, _shell(original), {}, report, [], cases, full))["native_synthetic_gate"]
        M._LiveNativeRun, M._WITNESS = live_cls, witness

        try:
            out["argument"] = M.native_verdict(live_cls(witness, fake_cls(), {}, report, [], cases, full),
                                               fake_cls)["native_synthetic_gate"]
        except TypeError:
            out["argument"] = False

        out["post_run"] = M.native_verdict(
            live_cls(witness, _shell(original), {}, report, ["post-run refusal"], cases, full))["native_synthetic_gate"]
    finally:
        M.NativeBackend, M._LiveNativeRun, M._WITNESS, M.evaluate = original, live_cls, witness, evaluate
        original.__init__ = original_init
    return out


EXPECTED_CAUGHT = {"class-check-by-module-name": "class", "constructor-by-module-name": "constructor",
                   "evaluator-by-module-name": "evaluator", "live-type-and-witness-by-module-name": "live_witness",
                   "overridable-default-argument": "argument", "post-run-dropped": "post_run"}


def test_the_repaired_source_fails_none_of_the_association_controls(full_run):
    cases, full, report = full_run
    M = _load(QUALIFIER.read_text(), "q_identity_unmutated")
    assert _forgeries(M, report, cases, full) == {k: False for k in EXPECTED_CAUGHT.values()}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_in_memory_mutation_of_the_repair_is_caught(full_run, name):
    cases, full, report = full_run
    old, new = MUTATIONS[name]
    source = QUALIFIER.read_text()
    assert source.count(old) == 1, name
    forged = _forgeries(_load(source.replace(old, new), f"q_identity_mutant_{len(name)}"), report, cases, full)
    assert forged[EXPECTED_CAUGHT[name]] is True, (name, forged)
    assert _sha(QUALIFIER) == hashlib.sha256(source.encode()).hexdigest()       # mutation never written
