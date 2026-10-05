"""CPU tests for scripts/qualify_tensor_fa_research.py (the native path is never run).

The CPU device is set before any tensor; mx.fast.metal_kernel,
mx.metal.is_available, mx.device_info and NativeBackend are forbidden. The
fake backend computes its "candidate" with the independent float64 formula
(never the candidate's host mirror); its evidence is CPU-diagnostic. Evidence
evaluation can never establish native execution; only a live run object built
by the native orchestration can.
"""

import copy
import json
import subprocess
import sys
import types

import mlx.core as mx

mx.set_default_device(mx.cpu)  # before any tensor

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scripts import qualify_tensor_fa_research as Q  # noqa: E402

REAL_NATIVE_BACKEND = Q.NativeBackend  # captured before the autouse fixture forbids it (never constructed)


def _forbid(name):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not run in CPU tests")
    return fail


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(mx.fast, "metal_kernel", _forbid("mx.fast.metal_kernel"))
    monkeypatch.setattr(mx.metal, "is_available", _forbid("mx.metal.is_available"))
    monkeypatch.setattr(mx, "device_info", _forbid("mx.device_info"))
    monkeypatch.setattr(Q, "NativeBackend", _forbid("NativeBackend"))
    assert mx.default_device() == mx.cpu


class FakeBackend:
    """CPU stub: independent float64 formula, declared (fake) engagement. Never native."""

    law = "exact"

    def __init__(self):
        self.calls, self.releases = [], 0

    def run_case(self, case, inputs):
        self.calls.append(case["id"])
        ref = Q.host_reference(case, inputs)
        if self.law == "no_causal_mask":
            ref = Q.host_reference(dict(case, causal=False), inputs)
        candidate = (ref * (1 + 1e-5)).astype(np.float32)
        if self.law == "no_later_rescale" and case["law"] == "rescale_mixed":
            candidate = _stale_max_output(case, inputs)
        return {"candidate": candidate, "baseline_f16": ref.astype(np.float16),
                "baseline_meta": {**Q.BASELINE, "kv_heads": case["hkv"], "q_heads": case["hq"]},
                "baseline_f32_scope": "omitted",
                "plan": {"threadgroups": case["expected_threadgroups"], "kv_end": list(case["kv_end"])},
                "engagement": {"expected_threadgroups": case["expected_threadgroups"],
                               "observed_threadgroups": case["expected_threadgroups"], "confirmed": True,
                               "issued_delta": 1, "confirmed_delta": 1}}

    def identity(self):
        return {"label": "cpu fake backend"}

    def release(self):
        self.releases += 1


def _stale_max_output(case, inputs):
    """What a kernel that skipped the mid-loop rescale would produce (half P with a stale max)."""
    q, k, v = (inputs[n].astype(np.float64) for n in ("q", "k", "v"))
    group = case["hq"] // case["hkv"]
    out = np.zeros(q.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        for h in range(case["hq"]):
            s = q[0, h] @ k[0, h // group].T * case["scale"]
            stale = s[:, :Q.C_KEYS].max(axis=1, keepdims=True)
            p = np.exp(s - stale).astype(np.float16).astype(np.float64)
            out[0, h] = (p @ v[0, h // group]) / p.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


@pytest.fixture(scope="module")
def full_cpu_run():
    cases, full = Q.select_cases()
    report = Q.run_gate(FakeBackend(), cases, source=lambda: {"f": "a" * 64})
    return cases, full, report


# ---------------------------------------------------------------- import / CLI / admission (no MLX)

def test_import_help_and_catalogue_do_not_import_mlx():
    code = ("import sys, contextlib, io\nimport scripts.qualify_tensor_fa_research as Q\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n    Q.main(['--catalogue'])\n"
            "    try: Q.main(['--help'])\n    except SystemExit: pass\n"
            "    Q.main(['--out', '/tmp/never.json'], environ={})\n"
            "Q.mlx_files()\nprint('mlx.core' in sys.modules, 'mlx2.adapters.tensor_fa_research' in sys.modules)\n")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.split()[-2:] == ["False", "False"], probe.stderr


def _args(**kw):
    base = dict(run_native=True, i_own_the_gpu=True, source_root=str(Q.ROOT), source_commit="a" * 40,
                out="/tmp/opus55-tfa-never-written.json")
    return types.SimpleNamespace(**{**base, **kw})


def _git_ok(head="a" * 40, status=b"", blobs=True):
    def git(*args):
        if args[0] == "rev-parse":
            return head.encode() + b"\n"
        if args[0] == "status":
            return status
        if args[0] == "show":
            path = args[1].split(":", 1)[1]
            return (Q.ROOT / path).read_bytes() if blobs and (Q.ROOT / path).exists() else None
        return None
    return git


ENV = {"MLX2_INTAKE_SOURCE_COMMIT": "a" * 40}


def _require_sources():
    """Admission past the blob check needs every Q.SOURCE_FILES file, two of
    which are private provenance records not exported to the public mirror."""
    missing = [rel for rel in Q.SOURCE_FILES if not (Q.ROOT / rel).is_file()]
    if missing:
        pytest.skip(f"bound source absent: {missing[0]} (private provenance)")


@pytest.mark.parametrize("args,env,git,reason", [
    (_args(run_native=False), ENV, _git_ok(), "no CPU fallback"),
    (_args(i_own_the_gpu=False), ENV, _git_ok(), "acknowledgement"),
    (_args(source_root="/tmp"), ENV, _git_ok(), "--source-root"),
    (_args(source_commit="b" * 40), ENV, _git_ok(), "MLX2_INTAKE_SOURCE_COMMIT"),
    (_args(source_commit="abc"), {"MLX2_INTAKE_SOURCE_COMMIT": "abc"}, _git_ok(), "full sha"),
    (_args(), ENV, _git_ok(head="c" * 40), "git HEAD"),
    (_args(), ENV, _git_ok(status=b"?? scripts/new.py\n"), "not clean"),
    (_args(), ENV, _git_ok(status=None), "not clean or status unavailable"),
    (_args(), ENV, _git_ok(blobs=False), "not committed at HEAD"),
    (_args(out=None), ENV, _git_ok(), "--out is required"),
    (_args(out=__file__), ENV, _git_ok(), "refusing to overwrite"),
])
def test_native_admission_refuses_before_any_mlx_or_device(monkeypatch, args, env, git, reason):
    monkeypatch.setattr(Q, "_git", git)
    monkeypatch.setattr(Q, "mlx_files", _forbid("mlx_files before git/source checks"))
    if reason in ("--out is required", "refusing to overwrite"):
        _require_sources()
        monkeypatch.setattr(Q, "mlx_files", lambda: {"f": "0" * 64})
    with pytest.raises(Q.Refused, match=reason):
        Q.native_admission(args, env)


def test_working_source_must_equal_head_blobs(monkeypatch):
    _require_sources()
    git = _git_ok()

    def tampered(*args):
        out = git(*args)
        return b"different" if args[0] == "show" and args[1].endswith("tensor_fa_research.py") else out

    monkeypatch.setattr(Q, "_git", tampered)
    with pytest.raises(Q.Refused, match="differs from HEAD blobs"):
        Q.native_admission(_args(), ENV)


def test_main_refusal_writes_nothing(tmp_path):
    out = tmp_path / "r.json"
    assert Q.main(["--out", str(out)], environ={}) == 1 and not out.exists()


# ---------------------------------------------------------------- catalogue

def test_catalogue_covers_the_required_space_and_is_memory_bounded():
    table = Q.catalogue_by_id()
    numeric = [c for c in Q.CATALOGUE if c["expect"] == "pass" and not c["optional"]]
    assert {c["dim"] for c in numeric} == {128, 256} and {c["q_dtype"] for c in numeric} == {"float16", "float32"}
    assert {c["hq"] // c["hkv"] for c in numeric} == {1, 2, 6}
    assert {1, 17, 31, 32, 33, 65} <= {c["length"] for c in numeric}
    assert {True, False} == {c["causal"] for c in numeric}
    assert any(c["causal"] and c["q_start"] == 0 for c in numeric)                       # early absolute start
    assert any(c["causal"] and c["q_start"] + c["length"] == c["kv_len"] for c in numeric)  # final position
    assert {"large_logits", "rescale_mixed"} <= {c["law"] for c in numeric}
    assert {c["law"] for c in Q.CATALOGUE if c["expect"] == "refuse"} == {"half_overflow", "nonfinite_v"}
    assert all(c["kv_len"] <= 1024 for c in Q.CATALOGUE if not c["optional"])
    assert all(c["kv_len"] <= 8192 for c in Q.CATALOGUE) and Q.TIMING_CELLS == ("t01-d256-f16-g6-w32-n8192-timing",)
    assert max(c["expected_memory_bytes"]["total"] for c in Q.CATALOGUE) < 64 * 2**20
    assert len(table) == len(Q.CATALOGUE)


def test_catalogue_law_matches_the_candidate_plan():
    from mlx2.adapters import tensor_fa_research as F

    for case in Q.CATALOGUE:
        inputs = Q.host_inputs(case)
        if case["expect"] == "refuse":
            continue
        plan = F.plan_tensor_fa(*(mx.array(inputs[n]) for n in ("q", "k", "v")), scale=case["scale"],
                                causal=case["causal"], q_start=case["q_start"])
        assert plan.threadgroups == case["expected_threadgroups"] > 0 and list(plan.kv_end) == case["kv_end"]
        mask = Q.visible_mask(case)
        assert all(list(plan.visible_keys(r)) == list(np.flatnonzero(mask[r])) for r in range(case["length"]))


def test_case_selection_rejects_unknown_duplicates_and_labels_partial():
    with pytest.raises(Q.Refused):
        Q.select_cases(["nope"])
    with pytest.raises(Q.Refused):
        Q.select_cases([Q.MANDATORY[0], Q.MANDATORY[0]])
    cases, full = Q.select_cases(list(Q.MANDATORY))
    assert full is False                                          # explicit lists are partial runs
    assert Q.select_cases()[1] is True


def test_rescale_witnesses_and_deletion_sensitivity():
    for case in Q.CATALOGUE:
        if case["law"] != "rescale_mixed":
            continue
        inputs = Q.host_inputs(case)
        w = Q.rescale_witness(case, inputs)
        assert w["grow_rows"] >= 1 and w["stay_rows"] >= 1
        assert not np.isfinite(_stale_max_output(case, inputs)).all()   # deleting the later rescale overflows half P
    # the initial rescale is always taken: deleting it is exp(s + FLT_MAX/2) = inf for every row
    with np.errstate(over="ignore"):
        assert np.isinf(np.exp(np.float32(1.0) + np.float32(np.finfo(np.float32).max / 2)))


# ---------------------------------------------------------------- evaluation of a full fake run

def test_full_fake_run_is_numeric_evidence_never_native(full_cpu_run):
    cases, full, report = full_cpu_run
    verdict = Q.evaluate(report, cases, full_requested=full)
    assert verdict["refusals"] == [] and verdict["verdict"] == "numeric_evidence_pass"
    assert verdict["native_synthetic_gate"] is False and verdict["qualified"] is False and verdict["selected"] is False
    assert verdict["model_gain"] is False and "does not establish native execution" in verdict["evidence_label"]
    assert report["producer"] == "FakeBackend" and "backend_kind" not in report
    for cid in ("r01-d128-f32-half-scaled-overflow", "r02-d128-f16-nonfinite-v"):
        ev = next(r for r in report["cases"] if r["id"] == cid)["evidence"]
        assert "refused" in ev and "candidate" not in ev                  # refused before any backend call
    assert Q.native_verdict(report)["native_synthetic_gate"] is False


def _re(full_cpu_run, edit, *, full=None):
    cases, f, report = full_cpu_run
    report = copy.deepcopy(report)
    edit(report)
    return Q.evaluate(report, cases, full_requested=f if full is None else full)


def _ev(report, cid="c03-d128-f16-g6-w31-mid"):
    return next(r for r in report["cases"] if r["id"] == cid)["evidence"]


@pytest.mark.parametrize("edit,reason", [
    (lambda r: _ev(r)["engagement"].update(confirmed=False), "engagement not confirmed"),
    (lambda r: _ev(r)["engagement"].update(observed_threadgroups=0), "engagement not confirmed"),
    (lambda r: _ev(r)["engagement"].update(confirmed_delta=0), "engagement not confirmed"),
    (lambda r: _ev(r).update(candidate=_ev(r)["candidate"][:, :, :-1]), "candidate: dtype/shape"),
    (lambda r: _ev(r).update(candidate=_ev(r)["candidate"].astype(np.float16)), "candidate: dtype/shape"),
    (lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])), "duplicate case"),
    (lambda r: r["cases"].pop(2), "missing case"),
    (lambda r: _ev(r)["input_hashes"].update(q="0" * 64), "input identity hashes"),
    (lambda r: _ev(r).update(source_after={"f": "b" * 64}), "source hashes missing or changed"),
    (lambda r: _ev(r)["plan"].update(threadgroups=1), "plan threadgroups"),
    (lambda r: _ev(r).update(baseline_meta={**Q.BASELINE, "kv_heads": 12, "q_heads": 12}), "baseline is not ordinary"),
    (lambda r: _ev(r).update(qualified=True), "verdict fields"),
    (lambda r: _ev(r, "r01-d128-f32-half-scaled-overflow").update(candidate=np.zeros(1)), "refuse case must be"),
])
def test_tampered_or_missing_evidence_refuses(full_cpu_run, edit, reason):
    verdict = _re(full_cpu_run, edit)
    assert verdict["verdict"] == "refused" and verdict["native_synthetic_gate"] is False
    assert any(reason in r for r in verdict["refusals"]), verdict["refusals"]


def test_nonfinite_matching_outputs_refuse(full_cpu_run):
    def both_nan(r):
        ev = _ev(r)
        ev["candidate"] = ev["candidate"].copy()
        ev["baseline_f16"] = ev["baseline_f16"].copy()
        ev["candidate"][0, 0, 0, 0] = np.nan
        ev["baseline_f16"][0, 0, 0, 0] = np.nan

    verdict = _re(full_cpu_run, both_nan)
    assert any("non-finite" in r for r in verdict["refusals"])


# ---------------------------------------------------------------- deliberate mutations must fail

@pytest.mark.parametrize("law,cid,reason", [
    ("no_causal_mask", "c02-d128-f32-g2-w17-final", "outside host float64 tolerance"),
    ("no_later_rescale", "c09-d128-f16-g2-w32-rescale-mixed", "non-finite"),
])
def test_wrong_kernel_laws_fail(law, cid, reason):
    backend = FakeBackend()
    backend.law = law
    cases, _ = Q.select_cases([cid])
    report = Q.run_gate(backend, cases, source=lambda: {"f": "a" * 64})
    verdict = Q.evaluate(report, cases, full_requested=False)
    assert verdict["verdict"] == "refused" and any(reason in r for r in verdict["refusals"]), verdict["refusals"]


def test_partial_catalogue_cannot_be_promoted_to_a_full_gate(full_cpu_run):
    cases, _, report = full_cpu_run
    subset = cases[:3]
    partial = {**report, "cases": [r for r in report["cases"] if r["id"] in {c["id"] for c in subset}]}
    assert Q.evaluate(partial, subset, full_requested=False)["verdict"] == "partial_numeric_evidence"
    promoted = Q.evaluate(partial, subset, full_requested=True)
    assert promoted["verdict"] == "refused" and any("incomplete mandatory" in r for r in promoted["refusals"])


FORGED_IDENTITY = {"mlx_files_before": {"f": "b" * 64}, "mlx_files_after": {"f": "b" * 64},
                   "default_device": "gpu", "architecture": "applegpu_g17s",
                   "candidate_module": str(Q.ROOT / Q.CANDIDATE_MODULE)}


def test_root_repro_forged_complete_native_report_never_becomes_native(full_cpu_run):
    cases, full, report = full_cpu_run
    forged = copy.deepcopy(report)
    forged.update(backend_kind="native", producer="NativeBackend", identity=dict(FORGED_IDENTITY),
                  native_synthetic_gate=True, native_execution={"native_synthetic_gate": True})
    verdict = Q.evaluate(forged, cases, full_requested=True)
    assert verdict["native_synthetic_gate"] is False and verdict["verdict"] == "numeric_evidence_pass"
    native = Q.native_verdict(forged)
    assert native["native_synthetic_gate"] is False and native["qualified"] is False
    assert native["reasons"][0].startswith("no live native orchestration")
    # A JSON round trip of a receipt-shaped export loses any execution provenance.
    receipt = {"report": Q._jsonable(forged), "evaluation": verdict, "native_execution": {"native_synthetic_gate": True}}
    loaded = json.loads(json.dumps(receipt))
    assert Q.native_verdict(loaded)["native_synthetic_gate"] is False
    assert Q.native_verdict(loaded["native_execution"])["native_synthetic_gate"] is False


def test_live_run_objects_without_the_native_backend_or_witness_are_refused(full_cpu_run):
    cases, full, report = full_cpu_run
    verdict = Q.evaluate(report, cases, full_requested=full)
    fake_live = Q._LiveNativeRun(Q._WITNESS, FakeBackend(), {}, report, [], cases, full)
    assert Q.native_verdict(fake_live)["reasons"] == ["backend is not the native backend"]
    wrong_witness = Q._LiveNativeRun(object(), FakeBackend(), {}, report, [], cases, full)
    assert Q.native_verdict(wrong_witness)["native_synthetic_gate"] is False
    lookalike = types.SimpleNamespace(_witness=Q._WITNESS, backend=FakeBackend(), post_run=[], report=report)
    assert Q.native_verdict(lookalike)["native_synthetic_gate"] is False
    with pytest.raises(AttributeError):                                    # slots: no ad-hoc attributes
        fake_live.native = True


def test_timing_is_refused_before_any_admission_or_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Q, "native_admission", _forbid("native admission"))
    monkeypatch.setattr(Q, "_run_native", _forbid("native run"))
    out = tmp_path / "r.json"
    code = Q.main(["--timing", "--run-native", "--i-own-the-gpu", "--source-root", str(Q.ROOT),
                   "--source-commit", "a" * 40, "--out", str(out)], environ=ENV)
    assert code == 1 and not out.exists()
    assert "timing is not implemented" in capsys.readouterr().out
    assert not hasattr(Q.NativeBackend, "time_case") and not hasattr(Q, "evaluate_timing")


def test_timing_evidence_in_a_report_is_refused(full_cpu_run):
    verdict = _re(full_cpu_run, lambda r: r["cases"][0].update(timing={"order": "ABBA"}))
    assert any("timing evidence is not accepted" in r for r in verdict["refusals"])


def test_optional_catalogue_cell_runs_only_as_partial_numeric_evidence():
    cases, full = Q.select_cases([Q.TIMING_CELLS[0]])
    report = Q.run_gate(FakeBackend(), cases, source=lambda: {"f": "a" * 64})
    verdict = Q.evaluate(report, cases, full_requested=full)
    assert full is False and verdict["verdict"] == "partial_numeric_evidence"


# ---------------------------------------------------------------- per-dispatch guards and cleanup

def test_guard_stops_before_the_next_dispatch_and_releases():
    calls = {"n": 0}

    def guard():
        calls["n"] += 1
        if calls["n"] == 2:                                                 # after the first case
            raise Q.Refused("bound source files changed since admission")

    backend = FakeBackend()
    cases, _ = Q.select_cases(list(Q.MANDATORY[:3]))
    with pytest.raises(Q.Refused, match="changed since admission"):
        Q.run_gate(backend, cases, source=lambda: {"f": "a" * 64}, guard=guard)
    assert backend.calls == [cases[0]["id"]] and backend.releases == 1


def test_guard_refusing_before_the_first_dispatch_never_calls_the_backend():
    backend = FakeBackend()
    cases, _ = Q.select_cases(list(Q.MANDATORY[:2]))
    with pytest.raises(Q.Refused):
        Q.run_gate(backend, cases, guard=_forbid_refused)
    assert backend.calls == [] and backend.releases == 1


def _forbid_refused():
    raise Q.Refused("stale")


def test_release_runs_when_a_case_raises():
    class Exploding(FakeBackend):
        def run_case(self, case, inputs):
            raise RuntimeError("device error")

    backend = Exploding()
    cases, _ = Q.select_cases([Q.MANDATORY[0]])
    with pytest.raises(RuntimeError, match="device error"):
        Q.run_gate(backend, cases)
    assert backend.releases == 1


def test_source_guard_compares_the_admitted_snapshot(monkeypatch):
    admitted = {"commit": "a" * 40, "source_expected": {"a": "1" * 64}, "mlx_files_before": {"m": "2" * 64}}
    monkeypatch.setattr(Q, "_git", _git_ok())
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "1" * 64})
    monkeypatch.setattr(Q, "mlx_files", lambda: {"m": "2" * 64})
    Q.source_guard(admitted)
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "3" * 64})
    with pytest.raises(Q.Refused, match="bound source"):
        Q.source_guard(admitted)
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "1" * 64})
    monkeypatch.setattr(Q, "mlx_files", lambda: {"m": "2" * 64, "extra": "4" * 64})
    with pytest.raises(Q.Refused, match="MLX files"):
        Q.source_guard(admitted)


@pytest.mark.parametrize("observed,expected,ok", [
    ({"a": "1" * 64}, {"a": "1" * 64}, True),
    ({}, {"a": "1" * 64}, False),
    ({"a": "1" * 64, "b": "2" * 64}, {"a": "1" * 64}, False),
    ({"a": "x"}, {"a": "x"}, False),
    ({}, {}, False),
    (None, {"a": "1" * 64}, False),
    ({"a": "1" * 64}, ["a"], False),
])
def test_snapshots_must_match_exactly(observed, expected, ok):
    assert Q.same_snapshot(observed, expected) is ok


def test_import_identity_is_checked_against_the_actual_extension(tmp_path):
    core = tmp_path / "core.so"
    core.write_bytes(b"extension")
    digest = __import__("hashlib").sha256(b"extension").hexdigest()
    admission = {"mlx_files_before": {str(core.resolve()): digest}, "mlx_version": "1.0"}
    assert Q.import_identity_refusal(admission, str(core), "1.0") is None
    assert "not the admitted file" in Q.import_identity_refusal(admission, str(tmp_path / "other.so"), "1.0")
    assert "version" in Q.import_identity_refusal(admission, str(core), "2.0")
    core.write_bytes(b"swapped")
    assert "bytes differ" in Q.import_identity_refusal(admission, str(core), "1.0")
    src = __import__("inspect").getsource(REAL_NATIVE_BACKEND.__init__)
    assert src.index("import_identity_refusal") < src.index("mx.default_device()") < src.index("mx.device_info()")


def test_post_run_rechecks_head_status_hashes_and_paths(monkeypatch):
    admission = {"commit": "a" * 40, "source_expected": {"a": "1" * 64}, "mlx_files_before": {"/m/core.so": "2" * 64}}
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "1" * 64})
    monkeypatch.setattr(Q, "mlx_files", lambda: {"/m/core.so": "2" * 64})
    monkeypatch.setattr(Q, "_git", _git_ok())
    cand = str(Q.ROOT / Q.CANDIDATE_MODULE)
    assert Q.post_run_refusals(admission, cand, "/m/core.so") == []
    monkeypatch.setattr(Q, "_git", _git_ok(head="c" * 40, status=b" M src/x.py\n"))
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "9" * 64})
    found = Q.post_run_refusals(admission, "/tmp/other/tensor_fa_research.py", "/other/core.so")
    for needle in ("HEAD differs", "candidate module path", "mlx.core path"):
        assert any(needle in r for r in found), found


def test_a_rescale_cell_without_growing_rows_is_refused(monkeypatch):
    """If the input law stops producing a >8 mid-loop growth, the cell no longer witnesses the rescale."""
    real = Q.host_inputs

    def flat(case):
        return real(dict(case, law="random")) if case["law"] == "rescale_mixed" else real(case)

    monkeypatch.setattr(Q, "host_inputs", flat)
    cases, _ = Q.select_cases(["c09-d128-f16-g2-w32-rescale-mixed"])
    report = Q.run_gate(FakeBackend(), cases, source=lambda: {"f": "a" * 64})
    verdict = Q.evaluate(report, cases, full_requested=False)
    assert verdict["verdict"] == "refused"
    assert any("rescale witness needs both" in r for r in verdict["refusals"]), verdict["refusals"]


def test_the_witness_alone_refuses_a_native_typed_backend_without_a_live_run(full_cpu_run, monkeypatch):
    """Isolates the witness: a NativeBackend-typed shell (object.__new__, no __init__, so no MLX or
    device call) paired with a wrong witness or a look-alike object is still refused. Code that
    reaches into this module's private _WITNESS is out of scope (no security claim)."""
    cases, full, report = full_cpu_run
    verdict = Q.evaluate(report, cases, full_requested=full)
    monkeypatch.setattr(Q, "NativeBackend", REAL_NATIVE_BACKEND)          # the class only; never constructed
    shell = object.__new__(REAL_NATIVE_BACKEND)
    for live in (Q._LiveNativeRun(object(), shell, {}, report, [], cases, full),
                 types.SimpleNamespace(_witness=Q._WITNESS, backend=shell, post_run=[], report=report,
                                       cases=cases, full_requested=full)):
        native = Q.native_verdict(live)
        assert native["native_synthetic_gate"] is False
        assert native["reasons"][0].startswith("no live native orchestration")



def _with_f32(value):
    def edit(report):
        ev = _ev(report)
        ev.pop("baseline_f32_scope", None)
        ev["baseline_f32"] = value(ev)
    return edit


@pytest.mark.parametrize("edit,reason", [
    (_with_f32(lambda ev: np.full(ev["candidate"].shape, np.nan, dtype=np.float32)), "baseline_f32 diagnostic: non-finite"),
    (_with_f32(lambda ev: ev["candidate"].astype(np.float16)), "baseline_f32 diagnostic: dtype/shape"),
    (_with_f32(lambda ev: ev["candidate"][:, :, :-1].copy()), "baseline_f32 diagnostic: dtype/shape"),
    (_with_f32(lambda ev: "not an array"), "baseline_f32 diagnostic: missing array"),
    (lambda r: _ev(r).pop("baseline_f32_scope"), "absent without an explicit 'omitted' scope"),
])
def test_float32_diagnostic_is_admitted_before_use(full_cpu_run, edit, reason):
    verdict = _re(full_cpu_run, edit)
    assert verdict["verdict"] == "refused" and any(reason in r for r in verdict["refusals"]), verdict["refusals"]


def test_a_valid_float32_diagnostic_is_reported_without_an_acceptance_threshold(full_cpu_run):
    verdict = _re(full_cpu_run, _with_f32(lambda ev: ev["candidate"].copy()))
    assert verdict["refusals"] == []
    assert "vs_native_f32_diagnostic" in verdict["per_case"]["c03-d128-f16-g6-w31-mid"]["metrics"]


def test_native_command_template_is_explicit():
    doc = Q.__doc__
    for needle in ("MLX2_INTAKE_SOURCE_ROOT=", "MLX2_INTAKE_SOURCE_COMMIT=", "MLX2_INTAKE_CPG_WORKFLOW=",
                   "MLX2_INTAKE_CPG_TASK=", "~/Desktop/mlx2/.venv/bin/python"):
        assert needle in doc
    assert ".venv/bin/python scripts" not in doc.replace("~/Desktop/mlx2/.venv/bin/python scripts", "")



def test_native_verdict_takes_no_caller_evaluation_and_binds_its_own(full_cpu_run):
    cases, full, report = full_cpu_run
    with pytest.raises(TypeError):
        Q.native_verdict(types.SimpleNamespace(report=report, cases=cases, full_requested=True),
                         {"verdict": "numeric_evidence_pass"})                 # no evaluation parameter exists
    # Evaluation comes from the live object's own report + cases + scope.
    partial = types.SimpleNamespace(report=report, cases=cases, full_requested=False)
    native = Q.native_verdict(partial)
    assert native["evaluation"]["verdict"] == "partial_numeric_evidence" and native["native_synthetic_gate"] is False
    mismatched = types.SimpleNamespace(report=report, cases=cases[:2], full_requested=True)
    native = Q.native_verdict(mismatched)
    assert native["evaluation"]["verdict"] == "refused"                           # report/cases not associated
    assert any("unexpected case" in r for r in native["evaluation"]["refusals"])
    empty = types.SimpleNamespace(report=report, cases=[], full_requested=True)
    assert Q.native_verdict(empty)["evaluation"]["verdict"] == "refused"


@pytest.mark.parametrize("git,reason", [
    (_git_ok(head="c" * 40), "HEAD differs"),
    (_git_ok(status=b" M src/mlx2/serving.py\n"), "not clean"),
    (_git_ok(status=b"?? tests/new_test.py\n"), "not clean"),
    (_git_ok(status=None), "status unavailable"),
])
def test_source_guard_refuses_head_or_tree_changes(monkeypatch, git, reason):
    admitted = {"commit": "a" * 40, "source_expected": {"a": "1" * 64}, "mlx_files_before": {"m": "2" * 64}}
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "1" * 64})
    monkeypatch.setattr(Q, "mlx_files", lambda: {"m": "2" * 64})
    monkeypatch.setattr(Q, "_git", git)
    with pytest.raises(Q.Refused, match=reason):
        Q.source_guard(admitted)


def test_a_head_change_after_the_first_case_stops_the_next_dispatch(monkeypatch):
    admitted = {"commit": "a" * 40, "source_expected": {"a": "1" * 64}, "mlx_files_before": {"m": "2" * 64}}
    monkeypatch.setattr(Q, "source_hashes", lambda: {"a": "1" * 64})
    monkeypatch.setattr(Q, "mlx_files", lambda: {"m": "2" * 64})
    state = {"head": "a" * 40}
    good = _git_ok()

    def git(*args):
        return state["head"].encode() + b"\n" if args[0] == "rev-parse" else good(*args)

    monkeypatch.setattr(Q, "_git", git)
    backend = FakeBackend()
    original = backend.run_case

    def run_then_move_head(case, inputs):
        out = original(case, inputs)
        state["head"] = "c" * 40                                               # a commit lands mid-run
        return out

    backend.run_case = run_then_move_head
    cases, _ = Q.select_cases(list(Q.MANDATORY[:3]))
    with pytest.raises(Q.Refused, match="HEAD differs"):
        Q.run_gate(backend, cases, source=lambda: {"a": "1" * 64}, guard=lambda: Q.source_guard(admitted))
    assert backend.calls == [cases[0]["id"]] and backend.releases == 1
