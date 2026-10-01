"""CPU tests for scripts/qualify_sparse_tree_gdn.py (the native gate is never run).

The CPU device is set before any tensor and every Metal construction,
dispatch and capability query is forbidden. A fake backend builds evidence
from the reviewed host executors; it exercises the evaluator and the
orchestration only and proves nothing about Metal bits.
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

from mlx2.adapters import qwen38_tree_gdn as T  # noqa: E402
from scripts import qualify_sparse_tree_gdn as Q  # noqa: E402

COMMIT = "81bf357ee9353451d1c59ab8fd610a91f8817d2e"
REAL_GIT_IDENTITY = Q.git_identity  # captured before the autouse stub


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
    monkeypatch.setattr(Q, "git_identity", lambda: {"available": False})
    assert mx.default_device() == mx.cpu


class FakeBackend:
    """Host-executor evidence with fake (declared) engagement; CPU only."""

    def __init__(self):
        self.events, self._confirmed, self._result = [], 0, None

    def identity(self):
        return {"default_device": "gpu", "device_info": {"architecture": "fake-cpu-backend"},
                "mlx": {"version": "fake", "metallib_sha256": "0" * 64},
                "reference": "gated_delta._gated_delta_kernel_impl(allow_packed=False)",
                "modules": {name: str(Q.ROOT / rel) for name, rel in Q.MODULE_PATHS.items()}}

    def arrays(self, case, host):
        self.events.append(("arrays", case["case"]))
        dtype = mx.bfloat16 if case["dtype"] == "bfloat16" else mx.float32
        out = {k: mx.array(host[k]).astype(dtype) for k in ("q", "k", "v")}
        out.update({k: mx.array(host[k]) for k in ("g", "beta", "state")})
        out["_parents"] = case["parents"]
        return out

    @staticmethod
    def _evidence(x):
        name = "bfloat16" if x.dtype == mx.bfloat16 else "float32"
        view = mx.uint16 if name == "bfloat16" else mx.uint32
        return {"dtype": name, "shape": list(x.shape), "bits": np.array(x.view(view))}

    def _args(self, a):
        return a["q"], a["k"], a["v"], a["g"], a["beta"], a["state"]

    def confirmed(self):
        return self._confirmed

    def forward(self, a, parents):
        self._result = T.sparse_tree_forward(*self._args(a), list(parents))
        self._oracle_y, _ = T.diagnostic_full_node_reference(*self._args(a), list(parents))
        self._confirmed += 1
        return self._evidence(self._result.y), True

    def row(self, y, node):
        return {"dtype": y["dtype"], "shape": [1, 1, *y["shape"][2:]], "bits": y["bits"][:, node:node + 1].copy()}

    def reference(self, a, path):
        state = T.ordinary_path_reference(*self._args(a), list(path))
        return self._evidence(self._oracle_y[:, path[-1]:path[-1] + 1]), self._evidence(state)

    def replay(self, a, parents, path):
        self._confirmed += 1
        return self._evidence(T.replay_accepted_path(self._result, list(path))), True

    def release(self, arrays):
        self.events.append(("release", None))
        arrays.clear()


SMALL = dict(dtypes=["float32", "bfloat16"], geometries=["ratio_1x2"])


def _args(**extra):
    return types.SimpleNamespace(i_own_the_gpu=True, seed=1, time_limit_s=600.0, **extra)


@pytest.fixture(scope="module")
def clean():
    cases = Q.catalogue(**SMALL)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Q, "git_identity", lambda: {"available": False})
        report, verdict = Q.run_gate(_args(), FakeBackend(), cases, commit=COMMIT)
    return cases, report, verdict


def _re(clean, edit):
    cases, report, _ = clean
    report = copy.deepcopy(report)
    edit(report)
    return Q.evaluate(report, cases)


def _case(report, key):
    return next(r for r in report["cases"] if r["case"] == key)


BRANCH = "branch/float32/ratio_1x2"
BRANCH16 = "branch/bfloat16/ratio_1x2"


# ---------------------------------------------------------------- catalogue and admission

def test_import_makes_no_mlx_or_reference_import():
    probe = subprocess.run([sys.executable, "-c", "import sys, scripts.qualify_sparse_tree_gdn; "
                            "print('mlx.core' in sys.modules, 'mlx2.runtime.models.gated_delta' in sys.modules)"],
                           capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.split() == ["False", "False"], probe.stderr


def test_catalogue_covers_the_required_cases():
    cases = Q.catalogue()
    assert len(cases) == len(Q.TOPOLOGIES) * 2 * 3
    assert {(c["hk"], c["hv"]) for c in cases} == {(16, 48), (2, 4), (1, 2)}
    assert {c["dtype"] for c in cases} == {"bfloat16", "float32"}
    for i, case in enumerate(cases):
        T.plan_tree_restarts(case["parents"])  # every topology is a valid plan
        if case["topology"] == "short_after_wide":
            assert cases[i - 1]["case"] == Q.case_key("width32", case["dtype"], case["geometry"])
    assert len(Q.WIDTH32) == 32 and Q.WIDTH32.count(-1) == 2  # includes a forest root
    assert sum(1 + len(c["parents"]) for c in cases) == Q.evaluate({"cases": []}, cases)["expected_confirmed_launches"]
    with pytest.raises(Q.Refused, match="width32"):
        Q.catalogue(["short_after_wide"])


@pytest.mark.parametrize("argv,environ,reason", [
    ([], {"MLX2_INTAKE_SOURCE_COMMIT": COMMIT}, "--i-own-the-gpu"),
    (["--i-own-the-gpu"], {}, "MLX2_INTAKE_SOURCE_COMMIT"),
    (["--i-own-the-gpu"], {"MLX2_INTAKE_SOURCE_COMMIT": "HEAD"}, "MLX2_INTAKE_SOURCE_COMMIT"),
    (["--i-own-the-gpu", "--time-limit-s", "nan"], {"MLX2_INTAKE_SOURCE_COMMIT": COMMIT}, "finite"),
    (["--i-own-the-gpu", "--topology", "short_after_wide"], {"MLX2_INTAKE_SOURCE_COMMIT": COMMIT}, "width32"),
])
def test_admission_refuses_before_any_backend(tmp_path, argv, environ, reason):
    out = tmp_path / "r.json"
    assert Q.main([*argv, "--out", str(out)], environ=environ, backend_factory=_forbid("backend")) == 1
    data = json.loads(out.read_text())
    assert data["verdict"] == "refused" and reason in data["refusals"][0]
    assert data["scope"].startswith("native-only") and data["upstream"]["head"].startswith("eea96c87")


def test_cli_refusal_never_imports_mlx():
    code = ("import sys, scripts.qualify_sparse_tree_gdn as Q\n"
            "Q.main(['--out', '/dev/null'], environ={})\n"
            "print('mlx.core' in sys.modules, 'mlx2.runtime.models.gated_delta' in sys.modules)\n")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.split()[-2:] == ["False", "False"], probe.stderr


# ---------------------------------------------------------------- clean run

def test_clean_fake_run_passes_one_case_at_a_time(clean):
    cases, report, verdict = clean
    assert verdict["verdict"] == "pass", verdict["refusals"][:5] + verdict["differences"][:5]
    rows = sum(len(c["parents"]) for c in cases)
    assert verdict["comparisons"] == 2 * rows
    assert verdict["confirmed_launches"] == verdict["expected_confirmed_launches"] == len(cases) + rows
    assert report["identity"]["ownership_note"].startswith("operator acknowledgement only")
    for record in report["cases"]:
        assert [n["path"] for n in record["nodes"]] == [Q.path_to(record["parents"], r) for r in range(len(record["parents"]))]


def test_backend_releases_each_case_before_the_next():
    backend = FakeBackend()
    cases = Q.catalogue(["singleton", "width2", "chain"], ["float32"], ["ratio_1x2"])
    Q.run_gate(_args(), backend, cases, commit=COMMIT)
    assert backend.events == [e for c in cases for e in (("arrays", c["case"]), ("release", None))]


def test_summary_is_json_safe_and_carries_no_raw_bits(clean):
    summary = Q.summarize(clean[1])
    text = json.dumps(summary)
    assert "bits" not in text and "sha256" in text


def test_geometry_27b_fixture_runs_on_the_fake():
    cases = Q.catalogue(["singleton", "width2"], ["bfloat16"], ["qwen38_27b"])
    report, verdict = Q.run_gate(_args(), FakeBackend(), cases, commit=COMMIT)
    assert verdict["verdict"] == "pass" and report["cases"][0]["hv"] == 48


# ---------------------------------------------------------------- bit drift, nonfinite, signed zero

def _flip(entry_key, what):
    def edit(report):
        record = _case(report, BRANCH)
        node = record["nodes"][3]
        if what == "readout":
            node["candidate_y"]["bits"][0, 0, 0, 5] ^= 1
            record["forward"]["y"]["bits"][0, 3, 0, 5] ^= 1  # keep the row consistent
        else:
            node[entry_key]["bits"][0, 0, 7, 9] ^= 1
    return edit


@pytest.mark.parametrize("edit,difference", [
    (_flip("candidate_y", "readout"), f"{BRANCH} row 3 readout: 1 raw-bit mismatches (0 signed zero)"),
    (_flip("replay_state", "state"), f"{BRANCH} row 3 state: 1 raw-bit mismatches (0 signed zero)"),
    (_flip("reference_state", "state"), f"{BRANCH} row 3 state: 1 raw-bit mismatches (0 signed zero)"),
])
def test_bit_drift_is_a_counterexample(clean, edit, difference):
    verdict = _re(clean, edit)
    assert verdict["verdict"] == "counterexample" and verdict["differences"] == [difference]


def test_signed_zero_is_a_difference(clean):
    def edit(report):
        node = _case(report, BRANCH)["nodes"][2]
        node["reference_state"]["bits"][0, 0, 0, 0] = np.uint32(0)
        node["replay_state"]["bits"][0, 0, 0, 0] = np.uint32(0x80000000)
    verdict = _re(clean, edit)
    assert verdict["verdict"] == "counterexample"
    assert verdict["differences"] == [f"{BRANCH} row 2 state: 1 raw-bit mismatches (1 signed zero)"]


def _set_bits(key, value, case=BRANCH16):
    def edit(report):
        node = _case(report, case)["nodes"][1]
        bits = node[key]["bits"]
        bits.flat[0] = bits.dtype.type(value)
        if key == "candidate_y":
            _case(report, case)["forward"]["y"]["bits"][0, 1].flat[0] = bits.dtype.type(value)
    return edit


@pytest.mark.parametrize("edit,reason", [
    (_set_bits("reference_y", 0x7FC0), f"{BRANCH16} row 1 readout: reference 1 non-finite values"),   # bf16 NaN
    (_set_bits("candidate_y", 0xFF80), f"{BRANCH16}: forward y 1 non-finite values"),                  # bf16 -inf
    (_set_bits("replay_state", 0x7F800000), f"{BRANCH16} row 1 state: candidate 1 non-finite values"),  # f32 +inf
])
def test_nonfinite_values_refuse(clean, edit, reason):
    verdict = _re(clean, edit)
    assert verdict["verdict"] == "refused" and reason in verdict["refusals"], verdict["refusals"]


# ---------------------------------------------------------------- missing, partial, duplicate, engagement

def _node(report, i=1, case=BRANCH):
    return _case(report, case)["nodes"][i]


@pytest.mark.parametrize("edit,reason", [
    (lambda r: _case(r, BRANCH)["forward"].update(engaged=False), f"{BRANCH}: forward engagement not confirmed"),
    (lambda r: _node(r).update(replay_engaged=None), f"{BRANCH} row 1: replay engagement not confirmed"),
    (lambda r: _case(r, BRANCH).update(confirmed_launches=7), f"{BRANCH}: confirmed launches 7 != 8"),
    (lambda r: _case(r, BRANCH)["nodes"].pop(4), f"{BRANCH}: missing row 4"),
    (lambda r: _case(r, BRANCH)["nodes"].append(copy.deepcopy(_node(r, 2))), f"{BRANCH}: duplicate row 2"),
    (lambda r: _node(r, 4).update(path=[0, 4]), f"{BRANCH} row 4: path is not the root-to-node path"),
    (lambda r: _node(r).pop("reference_state"), f"{BRANCH} row 1 state: reference evidence missing"),
    (lambda r: _node(r)["replay_state"].update(bits=None), f"{BRANCH} row 1 state: candidate raw bits unavailable"),
    (lambda r: _node(r)["reference_y"].update(dtype="bfloat16"), f"{BRANCH} row 1 readout: reference dtype bfloat16"),
    (lambda r: _node(r)["replay_state"].update(shape=[1, 2, 128, 64]), f"{BRANCH} row 1 state: candidate shape"),
    (lambda r: _node(r)["candidate_y"]["bits"].__setitem__((0, 0, 0, 0), 7), f"{BRANCH} row 1: candidate readout is not"),
    (lambda r: r["cases"].remove(_case(r, BRANCH)), f"missing case {BRANCH}"),
    (lambda r: r["cases"].append(copy.deepcopy(_case(r, BRANCH))), f"duplicate case {BRANCH}"),
    (lambda r: r["cases"].append(dict(_case(r, BRANCH), case="branch/float32/qwen38_27b")),
     "unexpected case branch/float32/qwen38_27b"),
    (lambda r: _case(r, BRANCH).update(hv=4), f"{BRANCH}: geometry or topology differs"),
    (lambda r: _case(r, BRANCH).update(parents=[-1, 0, 0, 1, 1, 2, 4]), f"{BRANCH}: geometry or topology differs"),
    (lambda r: r.update(cases=[]), "no case evidence"),
])
def test_missing_partial_or_inconsistent_evidence_refuses(clean, edit, reason):
    verdict = _re(clean, edit)
    assert verdict["verdict"] == "refused"
    assert any(x.startswith(reason) for x in verdict["refusals"]), verdict["refusals"]


def test_matching_nonfinite_and_vacuous_zero_evidence_refuse(clean):
    def matching_nan(report):
        node = _node(report, 2)
        node["reference_state"]["bits"].flat[3] = np.uint32(0x7FC00000)
        node["replay_state"]["bits"].flat[3] = np.uint32(0x7FC00000)

    verdict = _re(clean, matching_nan)
    assert verdict["verdict"] == "refused"
    assert f"{BRANCH} row 2 state: candidate 1 non-finite values" in verdict["refusals"]

    def zeros(report):
        node = _node(report, 2)
        for key in ("reference_state", "replay_state"):
            node[key]["bits"][...] = 0

    verdict = _re(clean, zeros)
    assert verdict["verdict"] == "refused"
    assert f"{BRANCH} row 2 state: reference is all zero (vacuous evidence)" in verdict["refusals"]


def test_source_label_status_is_recorded(clean):
    identity = clean[1]["identity"]
    assert identity["source_commit_status"] == "explicit_label_unverified"   # git unavailable in tests
    assert "paired with the parent wrapper's ownership receipt" in identity["ownership_note"]
    verdict = _re(clean, lambda r: r["identity"].pop("source_commit_status"))
    assert "source commit status missing" in verdict["refusals"]


def test_short_after_wide_must_follow_width32(clean):
    def edit(report):
        keys = [r["case"] for r in report["cases"]]
        i = keys.index("short_after_wide/float32/ratio_1x2")
        report["cases"][i - 1], report["cases"][i - 2] = report["cases"][i - 2], report["cases"][i - 1]
    verdict = _re(clean, edit)
    assert "short_after_wide/float32/ratio_1x2: did not run directly after width32" in verdict["refusals"]


# ---------------------------------------------------------------- identity and source binding

@pytest.mark.parametrize("edit,reason", [
    (lambda i: i.update(ownership_acknowledged=False), "no --i-own-the-gpu acknowledgement"),
    (lambda i: i.update(source_commit=None), "no explicit MLX2_INTAKE_SOURCE_COMMIT"),
    (lambda i: i.update(default_device="cpu"), "default device is not the GPU"),
    (lambda i: i.update(device_info={}), "device identity missing"),
    (lambda i: i["mlx"].update(metallib_sha256=None), "MLX build identity missing"),
    (lambda i: i["mlx"].update(metallib_sha256="x"), "MLX build identity missing"),               # root 2882
    (lambda i: i["sources_before"].update({"scripts/paired_direct_ab.py": "x"}), "source hashes missing"),
    (lambda i: i["sources_after"].update({"scripts/paired_direct_ab.py": "X" * 64}),
     "source hashes during the run missing"),
    (lambda i: i["modules"].update(candidate="/tmp/sister/mlx2/src/mlx2/adapters/qwen38_tree_gdn.py"),
     "candidate module is not src/mlx2/adapters/qwen38_tree_gdn.py of this checkout"),
    (lambda i: i["modules"].pop("reference"), "reference module is not"),
    (lambda i: i.pop("modules"), "candidate module is not"),
    (lambda i: i["sources_after_import"].update({"src/mlx2/runtime/models/gated_delta.py": "1" * 64}),
     "source changed at import"),
    (lambda i: i.update(reference="gated_delta.gated_delta_kernel"), "ordinary reference is not the unpacked"),
    (lambda i: i["sources_after"].update({"src/mlx2/adapters/qwen38_tree_gdn.py": "0" * 64}),
     "source changed during the run: ['src/mlx2/adapters/qwen38_tree_gdn.py']"),
    (lambda i: i["sources_before"].update({"scripts/qualify_sparse_tree_gdn.py": None}), "source hashes missing"),
    (lambda i: i.update(git={"available": True, "head": "deadbeef", "dirty": []}), "git HEAD does not match"),
    (lambda i: i.update(git={"available": True, "head": COMMIT, "status_ok": True,
                             "dirty": ["scripts/qualify_sparse_tree_gdn.py"]}), "bound source files are uncommitted"),
    (lambda i: i.update(git={"available": True, "head": COMMIT, "status_ok": False, "dirty": None}),
     "git status unavailable"),
    (lambda i: i.update(git={"available": True, "head": COMMIT, "dirty": []}), "git status unavailable"),
    (lambda i: i.update(git=None), "git identity missing"),
])
def test_identity_and_source_binding_refuse(clean, edit, reason):
    verdict = _re(clean, lambda r: edit(r["identity"]))
    assert verdict["verdict"] == "refused" and any(x.startswith(reason) for x in verdict["refusals"]), verdict["refusals"]


def test_source_changed_at_import_refuses_before_any_case(monkeypatch):
    """Root 2882: hashes are taken before the backend imports; a change at
    import must stop the run before any native call."""
    before = {n: "a" * 64 for n in Q.SOURCE_FILES}
    monkeypatch.setattr(Q, "source_hashes", lambda: {**before, "src/mlx2/adapters/qwen38_tree_gdn.py": "b" * 64})
    backend = FakeBackend()
    cases = Q.catalogue(["singleton"], ["float32"], ["ratio_1x2"])
    report, verdict = Q.run_gate(_args(), backend, cases, commit=COMMIT, sources_before=before,
                                 git={"available": False})
    assert backend.events == [] and report["cases"] == []
    assert any(r.startswith("source changed at import") for r in verdict["refusals"])


def test_missing_identity_refuses_before_any_case():
    class NoDevice(FakeBackend):
        def identity(self):
            return {k: v for k, v in super().identity().items() if k != "device_info"}

    backend = NoDevice()
    report, verdict = Q.run_gate(_args(), backend, Q.catalogue(["singleton"], ["float32"], ["ratio_1x2"]),
                                 commit=COMMIT)
    assert backend.events == [] and "device identity missing" in verdict["refusals"]


def test_main_hashes_sources_before_the_backend_is_built(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(Q, "source_hashes", lambda: order.append("hash") or {n: "a" * 64 for n in Q.SOURCE_FILES})

    def factory():
        order.append("backend")
        return FakeBackend()

    Q.main(["--i-own-the-gpu", "--topology", "singleton", "--dtype", "float32", "--geometry", "ratio_1x2",
            "--out", str(tmp_path / "r.json")], environ={"MLX2_INTAKE_SOURCE_COMMIT": COMMIT}, backend_factory=factory)
    assert order[:2] == ["hash", "backend"] and order.count("hash") == 3   # before, after import, at the end


def test_a_source_changed_mid_run_refuses(monkeypatch):
    calls = iter([{n: "a" * 64 for n in Q.SOURCE_FILES}] * 2 + [{n: "b" * 64 for n in Q.SOURCE_FILES}])
    monkeypatch.setattr(Q, "source_hashes", lambda: next(calls))
    cases = Q.catalogue(["singleton"], ["float32"], ["ratio_1x2"])
    _, verdict = Q.run_gate(_args(), FakeBackend(), cases, commit=COMMIT)
    assert verdict["verdict"] == "refused" and verdict["refusals"][0].startswith("source changed during the run")


def test_time_limit_leaves_missing_cases_refused():
    ticks = iter([0.0, 0.0, 1e9, 1e9, 1e9])
    cases = Q.catalogue(["singleton", "width2"], ["float32"], ["ratio_1x2"])
    _, verdict = Q.run_gate(_args(), FakeBackend(), cases, commit=COMMIT, clock=lambda: next(ticks))
    assert verdict["verdict"] == "refused" and "missing case width2/float32/ratio_1x2" in verdict["refusals"]


def test_zero_engagement_backend_refuses():
    class Unengaged(FakeBackend):
        def forward(self, a, parents):
            y, _ = super().forward(a, parents)
            self._confirmed -= 1
            return y, False

    cases = Q.catalogue(["width2"], ["float32"], ["ratio_1x2"])
    _, verdict = Q.run_gate(_args(), Unengaged(), cases, commit=COMMIT)
    assert verdict["verdict"] == "refused"
    assert "width2/float32/ratio_1x2: forward engagement not confirmed" in verdict["refusals"]
    assert "confirmed launches 2 != expected 3" in verdict["refusals"]


# ---------------------------------------------------------------- root 2885: typed receipts, pre-admission

@pytest.fixture(scope="module")
def width2():
    cases = Q.catalogue(["width2"], ["float32"], ["ratio_1x2"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Q, "git_identity", lambda: {"available": False})
        report, verdict = Q.run_gate(_args(), FakeBackend(), cases, commit=COMMIT)
    assert verdict["verdict"] == "pass"
    return cases, report, verdict


W2 = "width2/float32/ratio_1x2"


@pytest.mark.parametrize("edit,reason", [
    (lambda r: r["cases"].append(None), "case entry 1 is not a dict"),                          # root repro
    (lambda r: r["cases"].append("width2/float32/ratio_1x2"), "case entry 1 is not a dict"),
    (lambda r: r["cases"][0]["nodes"][1].update(node=True), f"{W2}: row id True is not a plain int"),  # root repro
    (lambda r: r["cases"][0]["nodes"][1].update(node=1.0), f"{W2}: row id 1.0 is not a plain int"),
    (lambda r: r["cases"][0]["nodes"][1].update(node=np.int64(1)), f"{W2}: row id np.int64(1) is not a plain int"),
    (lambda r: r["cases"][0]["nodes"].append(None), f"{W2}: row entry 2 is not a dict"),
    (lambda r: r["cases"][0].update(nodes={"0": {}}), f"{W2}: rows are not a list"),
    (lambda r: r["cases"][0]["nodes"][1].update(path=[0, True]), f"{W2} row 1: path is not the root-to-node path"),
    (lambda r: r["cases"][0].update(parents=[-1, False]), f"{W2}: geometry or topology differs"),
    (lambda r: r["cases"][0].update(hk=True), f"{W2}: geometry or topology differs"),
    (lambda r: r["cases"][0].update(hv=2.0), f"{W2}: geometry or topology differs"),
    (lambda r: r["cases"][0].update(confirmed_launches=3.0), f"{W2}: confirmed launches 3.0 != 3"),
])
def test_untyped_or_non_dict_evidence_refuses(width2, edit, reason):
    cases, report, _ = width2
    report = copy.deepcopy(report)
    edit(report)
    verdict = Q.evaluate(report, cases)
    assert verdict["verdict"] == "refused"
    assert any(r.startswith(reason) for r in verdict["refusals"]), verdict["refusals"]


@pytest.mark.parametrize("hashes,git,reason", [
    ({}, {"available": False}, "source hashes missing"),
    ("bad", {"available": False}, "source hashes missing"),
    (None, {"available": True, "head": "deadbeef" * 5, "status_ok": True, "dirty": []}, "git HEAD does not match"),
    (None, {"available": True, "head": COMMIT, "status_ok": False, "dirty": None}, "git status unavailable"),
    (None, {"available": True, "head": COMMIT, "status_ok": True, "dirty": ["src/mlx2/adapters/qwen38_tree_gdn.py"]},
     "bound source files are uncommitted"),
    (None, None, "git identity missing"),
])
def test_main_refuses_bound_source_problems_before_building_the_backend(tmp_path, monkeypatch, hashes, git, reason):
    good = {n: "a" * 64 for n in Q.SOURCE_FILES}
    monkeypatch.setattr(Q, "source_hashes", lambda: good if hashes is None else hashes)
    monkeypatch.setattr(Q, "git_identity", lambda: git)
    out = tmp_path / "r.json"
    assert Q.main(["--i-own-the-gpu", "--out", str(out)], environ={"MLX2_INTAKE_SOURCE_COMMIT": COMMIT},
                  backend_factory=_forbid("backend factory")) == 1
    data = json.loads(out.read_text())
    assert data["verdict"] == "refused" and reason in data["refusals"][0]


def test_frozen_export_without_git_is_an_unverified_label_but_still_hash_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "source_hashes", lambda: {n: "a" * 64 for n in Q.SOURCE_FILES})
    monkeypatch.setattr(Q, "git_identity", lambda: {"available": False})
    out = tmp_path / "r.json"
    Q.main(["--i-own-the-gpu", "--topology", "singleton", "--dtype", "float32", "--geometry", "ratio_1x2",
            "--out", str(out)], environ={"MLX2_INTAKE_SOURCE_COMMIT": COMMIT}, backend_factory=FakeBackend)
    data = json.loads(out.read_text())
    assert data["report"]["identity"]["source_commit_status"] == "explicit_label_unverified"
    assert data["verdict"] == "pass"


def test_git_status_failure_is_never_clean(monkeypatch):
    real = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return types.SimpleNamespace(stdout=COMMIT + "\n")
        if cmd[:2] == ["git", "status"]:
            raise subprocess.CalledProcessError(128, cmd)
        return real(cmd, **kwargs)

    monkeypatch.setattr(Q, "git_identity", REAL_GIT_IDENTITY)  # Metal stays forbidden
    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    git = Q.git_identity()
    assert git == {"available": True, "head": COMMIT, "status_ok": False, "dirty": None}
    assert Q.source_refusals(COMMIT, {n: "a" * 64 for n in Q.SOURCE_FILES}, git) == [
        "git status unavailable for the bound source files"]
