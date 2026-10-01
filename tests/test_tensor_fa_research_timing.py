"""CPU tests for scripts/benchmark_tensor_fa_research.py (no MLX import, no device, no native path).

Fake ops/clock/guard drive the real orchestration and evaluator. The fake
"candidate" is the independent float64 formula (never the candidate's host
mirror). Fake evidence can test orchestration but can never establish native
timing or any gain; forged gates, reports and identities cannot unlock the
native runner. Run without the repository conftest (it imports mlx.core):

  PYTHONPATH=src:. python -m pytest --noconftest -p no:cacheprovider tests/test_tensor_fa_research_timing.py
"""

import copy
import inspect
import itertools
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_tensor_fa_research as T
from scripts import qualify_tensor_fa_research as Q

CAND, BASE = T.CAND, T.BASE


# ================================================================ fakes

class FakeArr:
    def __init__(self, value):
        self.value, self.evaluated = value, False


class FakeOps:
    """CPU stand-in for NativeTimingOps. Index k counts dispatches per (cell, arm): 0 = witness."""

    def __init__(self, log, deferred=None):
        self.log, self.deferred = log, deferred
        self.serial = itertools.count(1)
        self.issued = self.confirmed = 0
        self.index = {}
        self.releases = self.cell_releases = 0
        self.candidate_fn = lambda cell, ref, k: (ref * (1 + 1e-5)).astype(np.float32)
        self.baseline_fn = lambda cell, ref, k: ref.astype(np.float16)
        self.observed_fn = lambda cell, k: cell["expected_threadgroups"]
        self.confirm_fn = lambda cell, k: True
        self.extra_issue_fn = lambda cell, k: 0
        self.conversion_fn = lambda name, k, v: (k.copy(), v.copy())
        self.serial_fn = None

    def identity(self):
        return {"label": "cpu fake ops"}

    def prepare(self, cell, inputs):
        self.log.append("prepare")
        return {"cell": cell, "ref": Q.host_reference(cell, inputs)}

    def describe(self, prep):
        c = prep["cell"]
        return {"q": ["float16", [1, c["hq"], c["length"], c["dim"]]],
                "k": ["float16", [1, c["hkv"], c["kv_len"], c["dim"]]],
                "v": ["float16", [1, c["hkv"], c["kv_len"], c["dim"]]]}

    def prepare_conversion(self, cell, inputs):
        self.log.append("prepare_conversion")
        return {"k": inputs["k"], "v": inputs["v"]}

    def materialize_deferred(self):
        return self.deferred

    def dispatch(self, arm, prep):
        self.log.append(f"dispatch:{arm}")
        serial = next(self.serial)
        if self.serial_fn is not None:
            serial = self.serial_fn(serial)
        if arm in (T.CONVERT, T.MATERIALIZE):
            return T.Dispatch(arm, tuple(FakeArr(x) for x in self.conversion_fn(arm, prep["k"], prep["v"])),
                              None, None, serial)
        cell = prep["cell"]
        k = self.index[(cell["id"], arm)] = self.index.get((cell["id"], arm), -1) + 1
        if arm == BASE:
            return T.Dispatch(arm, (FakeArr(self.baseline_fn(cell, prep["ref"], k)),), None, None, serial)
        self.issued += 1 + self.extra_issue_fn(cell, k)
        plan = SimpleNamespace(threadgroups=cell["expected_threadgroups"], kv_end=tuple(cell["kv_end"]))
        out = FakeArr(self.candidate_fn(cell, prep["ref"], k))
        engaged = FakeArr((self.observed_fn(cell, k), self.confirm_fn(cell, k)))
        return T.Dispatch(arm, (out,), engaged, plan, serial)

    def complete(self, d):
        self.log.append("complete")
        for a in (*d.outputs, *([d.engaged] if d.engaged is not None else [])):
            assert not a.evaluated, "a previously evaluated output was reused"
            a.evaluated = True

    def engagement(self, d):
        self.log.append("engagement")
        assert d.engaged.evaluated
        observed, confirmed = d.engaged.value
        self.confirmed += int(confirmed)
        return observed, confirmed

    def counters(self):
        self.log.append("counters")
        return {"issued": self.issued, "confirmed": self.confirmed}

    def to_host(self, d):
        self.log.append("to_host")
        assert all(a.evaluated for a in d.outputs)
        hosts = tuple(a.value for a in d.outputs)
        d.outputs = d.engaged = None
        return hosts

    def memory_reset(self):
        self.log.append("memory")
        return {"active_bytes_before": 1}

    def memory_peak(self):
        self.log.append("memory")
        return {"peak_bytes": 2, "scope": "fake"}

    def release_cell(self, prep):
        self.log.append("release_cell")
        self.cell_releases += 1

    def release(self):
        self.log.append("release")
        self.releases += 1


class Clock:
    def __init__(self, log):
        self.log, self.t = log, 0

    def __call__(self):
        self.log.append("clock")
        self.t += 1000
        return self.t


class Guard:
    def __init__(self, log, fail_at=None):
        self.log, self.calls, self.fail_at = log, 0, fail_at

    def __call__(self):
        self.calls += 1
        self.log.append("guard")
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise Q.Refused("source changed (fake guard)")


def run(cells=T.CELLS, ops=None, guard=None, **kw):
    log = []
    ops = ops or FakeOps(log, **kw)
    ops.log = log
    guard = guard or Guard(log)
    guard.log = log
    report = T.run_timing(ops, cells, guard=guard, clock=Clock(log))
    return report, ops, guard, log


def small(**kw):
    """The cheapest cell (d128 L32 N1024, with conversion) alone; the evaluator still demands all cells."""
    return run(cells=T.CELLS[:1], **kw)


@pytest.fixture(scope="module")
def full():
    report, ops, guard, log = run()
    return report, ops, guard, log


def _refused(report, needle):
    ev = T.evaluate_timing(report)
    assert ev["verdict"] == "refused" and ev["native_timing_execution"] is False
    assert any(needle in r for r in ev["refusals"]), ev["refusals"][:5]


# ================================================================ no MLX, CLI

def test_import_help_catalogue_and_refusal_never_import_mlx():
    code = ("import sys, io, contextlib\nfrom scripts import benchmark_tensor_fa_research as T\n"
            "with contextlib.redirect_stdout(io.StringIO()):\n"
            "    assert T.main(['--catalogue']) == 0\n    assert T.main([]) == 1\n"
            "print(sorted(m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') or 'tensor_fa_research' in m))")
    probe = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": "src:."},
                           check=False)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "['scripts.benchmark_tensor_fa_research', 'scripts.qualify_tensor_fa_research']"
    helped = subprocess.run([sys.executable, "scripts/benchmark_tensor_fa_research.py", "--help"],
                            capture_output=True, text=True, env={"PYTHONPATH": "src:."}, check=False)
    assert helped.returncode == 0 and "--run-native" in helped.stdout
    cat = subprocess.run([sys.executable, "scripts/benchmark_tensor_fa_research.py", "--catalogue"],
                         capture_output=True, text=True, env={"PYTHONPATH": "src:."}, check=False)
    assert [c["id"] for c in json.loads(cat.stdout)["cells"]] == [c["id"] for c in T.CELLS]


def test_cli_has_no_external_gate_input_and_refusal_writes_nothing(tmp_path, capsys):
    for flag in ("--gate", "--gate-json", "--numeric-pass", "--evidence"):
        with pytest.raises(SystemExit):
            T.main([flag, "x"])
    out = tmp_path / "r.json"
    assert T.main(["--run-native", "--out", str(out)], environ={}) == 1
    assert not out.exists() and "refused" in capsys.readouterr().out


# ================================================================ catalogue and schedule

def test_catalogue_is_the_predeclared_fair_space():
    assert {(c["dim"], c["length"], c["kv_len"]) for c in T.CELLS} == {
        (d, length, n) for d in (128, 256) for length in (32, 65) for n in (1024, 8192)}
    for c in T.CELLS:
        assert (c["q_dtype"], c["hq"], c["hkv"], c["causal"], c["law"]) == ("float16", 12, 2, True, "random")
        assert c["q_start"] == c["kv_len"] - c["length"] and not c["optional"]
        assert c["expected_threadgroups"] == -(-c["length"] // Q.Q_ROWS) * 12 and c["kv_end"][-1] == c["kv_len"]
        assert c["conversion"] == (c["length"] == 32)
        assert max(c["device_bytes"].values()) <= 80 << 20            # memory-safe single cell
    assert len({c["id"] for c in T.CELLS}) == len(T.CELLS)


def test_schedule_is_alternating_and_balanced():
    orders = [T.pair_order(i) for i in range(T.PAIRS)]
    assert T.PAIRS % 2 == 0 and orders.count((CAND, BASE)) == orders.count((BASE, CAND)) == T.PAIRS // 2
    assert all(orders[i] != orders[i + 1] for i in range(T.PAIRS - 1))


# ================================================================ fake run: evidence only

def test_full_fake_run_is_complete_evidence_never_native(full):
    report, ops, _guard, _log = full
    ev = T.evaluate_timing(report)
    assert ev["verdict"] == "timing_evidence_complete", ev["refusals"][:5]
    assert (ev["native_timing_execution"], ev["qualified"], ev["selected"], ev["model_gain"]) == (False,) * 4
    for cell in T.CELLS:
        s = ev["summary"][cell["id"]]
        assert len(s["candidate_ns"]) == len(s["baseline_ns"]) == len(s["pair_ratio"]) == T.PAIRS
        assert s["median_pair_ratio"] == 1.0 and "no model or end-to-end meaning" in s["ratio_definition"]
        assert ("conversion" in s) == cell["conversion"]
    text = json.dumps(ev["summary"])
    assert "speedup" not in text and "end_to_end_ns" not in text and "total_ns" not in text
    verdict = T.timing_verdict(report)                           # a dict is never a live run
    assert verdict["native_timing_execution"] is False and "no live native timing" in verdict["reasons"][0]
    assert T.timing_verdict(json.loads(json.dumps(report)))["native_timing_execution"] is False
    assert ops.releases == 1 and ops.cell_releases == len(T.CELLS) + sum(c["conversion"] for c in T.CELLS)


def test_json_round_trip_keeps_evidence_and_forged_native_labels_are_ignored(full):
    report = json.loads(json.dumps(full[0]))
    report.update(producer="NativeTimingOps", native_timing_execution=True, qualified=True,
                  identity={"architecture": "applegpu_g17s", "commit": "a" * 40})
    ev = T.evaluate_timing(report)
    assert ev["verdict"] == "timing_evidence_complete" and ev["native_timing_execution"] is False
    assert ev["qualified"] is False and ev["model_gain"] is False


# ================================================================ timer boundary, guards, freshness

def test_only_dispatch_eval_and_sync_sit_inside_the_timer(full):
    log = full[3]
    clocks = [i for i, e in enumerate(log) if e == "clock"]
    assert len(clocks) % 2 == 0 and clocks
    for start, end in zip(clocks[0::2], clocks[1::2]):
        inside = log[start + 1:end]
        assert len(inside) == 2 and inside[0].startswith("dispatch:") and inside[1] == "complete", inside
    timed = sum(1 for s, e in zip(clocks[0::2], clocks[1::2]))
    conv = sum(c["conversion"] for c in T.CELLS) * (T.CONV_WARMUP + T.CONV_REPEATS) * 2
    assert timed == len(T.CELLS) * 2 * (T.WARMUP_PAIRS + T.PAIRS) + conv   # witnesses are untimed
    # engagement readback, counters, host copy, guards and memory probes never fall inside a window
    inside_any = {i for s, e in zip(clocks[0::2], clocks[1::2]) for i in range(s + 1, e)}
    assert not any(log[i] in ("engagement", "counters", "to_host", "guard", "memory") for i in inside_any)


def test_guard_runs_before_every_dispatch_and_after_every_repeat(full):
    log, guard = full[3], full[2]
    last_guard = last_host = -1
    for i, e in enumerate(log):
        if e == "guard":
            last_guard = i
        elif e == "to_host":
            last_host = i
        elif e.startswith("dispatch:") or e in ("prepare", "prepare_conversion"):
            assert last_guard > last_host and last_guard >= 0, (i, e)
    per_cell = 1 + 4 + 3 * (T.WARMUP_PAIRS + T.PAIRS) + 1 + 1
    conv = 1 + 3 * (T.CONV_WARMUP + T.CONV_REPEATS)
    assert guard.calls == len(T.CELLS) * per_cell + sum(c["conversion"] for c in T.CELLS) * conv + 1


def test_every_repeat_is_a_fresh_invocation(full):
    report, ops = full[0], full[1]
    for rec in report["cells"]:
        serials = [s["serial"] for s in rec["warmup"] + rec["samples"]]
        assert len(set(serials)) == len(serials)
        assert ops.index[(rec["id"], CAND)] == ops.index[(rec["id"], BASE)] == T.WARMUP_PAIRS + T.PAIRS


def test_reused_serial_fails_closed_inline():
    ops = FakeOps([])
    ops.serial_fn = lambda s: 7 if s > 5 else s
    with pytest.raises(Q.Refused, match="not fresh"):
        small(ops=ops)
    assert ops.releases == 1


# ================================================================ per-repeat numerical and engagement binding

def _dispatches(ops, cell, arm):
    return ops.index.get((cell["id"], arm), -1) + 1


@pytest.mark.parametrize("where", ["all", "timed_only", "one_middle"])
def test_stable_or_single_wrong_candidate_outputs_fail(where):
    ops = FakeOps([])
    wrong = {"all": lambda k: True, "timed_only": lambda k: k >= 1, "one_middle": lambda k: k == 20}[where]
    ops.candidate_fn = lambda cell, ref, k: ((ref * 1.05 if wrong(k) else ref * (1 + 1e-5)).astype(np.float32))
    with pytest.raises(Q.Refused, match="vs_host_f64 outside tolerance"):
        small(ops=ops)
    stop = {"all": 0, "timed_only": 1, "one_middle": 20}[where]
    assert _dispatches(ops, T.CELLS[0], CAND) == stop + 1          # fail-stop: no further candidate dispatch
    assert ops.releases == 1 and ops.cell_releases == 1


def test_wrong_baseline_outputs_fail():
    ops = FakeOps([])
    ops.baseline_fn = lambda cell, ref, k: (ref * (1.05 if k >= 5 else 1.0)).astype(np.float16)
    with pytest.raises(Q.Refused, match="baseline: vs_host_f64 outside"):
        small(ops=ops)


def _nudged(sign, at):
    """Inside host tolerance alone; opposite-signed arms disagree beyond the native tolerance."""
    def fn(cell, ref, k):
        if k != at:
            return (ref * (1 + 1e-5)).astype(np.float32 if sign > 0 else np.float16)
        noise = np.random.default_rng(0).choice([-1.0, 1.0], size=ref.shape)
        out = ref + sign * 0.0018 * np.sqrt((ref ** 2).mean()) * noise
        return out.astype(np.float32 if sign > 0 else np.float16)
    return fn


@pytest.mark.parametrize("at, needle", [(9, r"pair 5 candidate: vs_pair_baseline outside"),
                                        (0, r"witness candidate: vs_baseline_witness outside")])
def test_candidate_is_bound_to_the_same_pair_or_witness_baseline(at, needle):
    ops = FakeOps([])
    ops.candidate_fn, ops.baseline_fn = _nudged(+1, at), _nudged(-1, at)
    with pytest.raises(Q.Refused, match=needle):
        small(ops=ops)


@pytest.mark.parametrize("mutate, needle", [
    ("observed_fn", "engagement not confirmed"),
    ("confirm_fn", "engagement not confirmed"),
    ("extra_issue_fn", "engagement not confirmed"),
    ("dtype", "dtype/shape"),
    ("shape", "dtype/shape"),
    ("nan", "non-finite"),
])
def test_engagement_dtype_shape_and_nonfinite_fail(mutate, needle):
    ops = FakeOps([])
    def bad(k):
        return k == 6                                               # a timed repeat
    if mutate == "observed_fn":
        ops.observed_fn = lambda cell, k: 0 if bad(k) else cell["expected_threadgroups"]
    elif mutate == "confirm_fn":
        ops.confirm_fn = lambda cell, k: not bad(k)
    elif mutate == "extra_issue_fn":
        ops.extra_issue_fn = lambda cell, k: int(bad(k))
    else:
        def cand(cell, ref, k):
            out = (ref * (1 + 1e-5)).astype(np.float32)
            if bad(k) and mutate == "dtype":
                return out.astype(np.float16)
            if bad(k) and mutate == "shape":
                return out[:, :, :-1]
            if bad(k) and mutate == "nan":
                out[0, 0, 0, 0] = np.nan
            return out
        ops.candidate_fn = cand
    with pytest.raises(Q.Refused, match=needle):
        small(ops=ops)
    assert _dispatches(ops, T.CELLS[0], CAND) == 7 and ops.releases == 1


def test_zero_engagement_on_the_witness_stops_before_any_timed_dispatch():
    ops = FakeOps([])
    ops.observed_fn = lambda cell, k: 0
    report_log = []
    with pytest.raises(Q.Refused, match="witness candidate"):
        run(cells=T.CELLS[:1], ops=ops, guard=Guard(report_log))
    assert "clock" not in ops.log


# ================================================================ conversion experiment

def test_conversion_scope_is_labelled_and_never_combined(full):
    rec = next(r for r in full[0]["cells"] if r["cell"]["conversion"])
    conv = rec["conversion"]
    assert conv["scope"] == T.CONVERSION_SCOPE
    assert set(conv["experiments"]) == {T.CONVERT, T.MATERIALIZE}
    assert all(len(e["samples"]) == T.CONV_REPEATS for e in conv["experiments"].values())
    kv = 2 * rec["cell"]["hkv"] * rec["cell"]["kv_len"] * rec["cell"]["dim"]
    assert conv["bytes"][T.CONVERT] == {"read_float32": kv * 4, "write_float16": kv * 2}
    for label in ("quantized", "served route", "QSA", "end-to-end"):
        assert any(label in x for x in T.CONVERSION_SCOPE["not"])
    assert "never summed" in T.CONVERSION_SCOPE["combination"]
    assert all("conversion" not in r for r in full[0]["cells"] if not r["cell"]["conversion"])


def test_wrong_conversion_output_fails_closed():
    ops = FakeOps([])
    ops.conversion_fn = lambda name, k, v: (k.copy(), (v + np.float16(1)) if name == T.MATERIALIZE else v.copy())
    with pytest.raises(Q.Refused, match="materialize_strided_f16 repeat 0"):
        small(ops=ops)
    assert ops.releases == 1 and ops.cell_releases == 2


def test_materialization_may_only_be_deferred_explicitly():
    report, _ops, _g, _l = small(deferred="mx.contiguous unavailable (fake)")
    conv = report["cells"][0]["conversion"]
    assert conv["experiments"][T.MATERIALIZE] == {"status": "deferred", "reason": "mx.contiguous unavailable (fake)"}
    assert not T._conversion_problems(T.CELLS[0], Q.host_inputs(T.CELLS[0]), conv)
    conv["experiments"][T.CONVERT] = {"status": "deferred", "reason": "no"}
    assert T._conversion_problems(T.CELLS[0], Q.host_inputs(T.CELLS[0]), conv)


# ================================================================ evaluator tamper controls

TAMPER = {
    "drop_cell": (lambda c, r: c.pop(), "full catalogue is mandatory"),
    "dup_cell": (lambda c, r: c.append(copy.deepcopy(c[0])), "full catalogue is mandatory"),
    "reorder_cells": (lambda c, r: c.reverse(), "full catalogue is mandatory"),
    "drop_sample": (lambda c, r: c[0]["samples"].pop(5), "alternating order"),
    "dup_sample": (lambda c, r: c[0]["samples"].insert(3, copy.deepcopy(c[0]["samples"][3])), "alternating order"),
    "swap_order": (lambda c, r: c[0]["samples"].__setitem__(slice(0, 2), c[0]["samples"][1::-1]),
                   "alternating order"),
    "drop_warmup": (lambda c, r: c[0]["warmup"].pop(), "alternating order"),
    "zero_ns": (lambda c, r: c[1]["samples"][4].__setitem__("ns", 0), "no positive integer sample"),
    "float_ns": (lambda c, r: c[1]["samples"][4].__setitem__("ns", 1.5), "no positive integer sample"),
    "reuse_serial": (lambda c, r: c[2]["samples"][6].__setitem__("serial", c[2]["samples"][4]["serial"]),
                     "not fresh"),
    "forged_metric": (lambda c, r: c[0]["samples"][7]["metrics"]["vs_host_f64"].__setitem__("normalized_rms", 0.5),
                      "vs_host_f64 outside"),
    "missing_witness_metric": (lambda c, r: c[0]["samples"][9]["metrics"].pop("vs_witness"), "vs_witness metrics"),
    "missing_pair_metric": (lambda c, r: next(s for s in c[0]["samples"] if s["arm"] == CAND)["metrics"]
                            .pop("vs_pair_baseline"), "vs_pair_baseline metrics"),
    "zero_engagement": (lambda c, r: next(s for s in c[3]["samples"] if s["arm"] == CAND)["engagement"]
                        .__setitem__("observed_threadgroups", 0), "engagement not confirmed"),
    "issued_twice": (lambda c, r: next(s for s in c[3]["samples"] if s["arm"] == CAND)["engagement"]
                     .__setitem__("issued_delta", 2), "engagement not confirmed"),
    "wrong_dtype": (lambda c, r: c[0]["samples"][0]["output"].__setitem__("dtype", "float16"), "dtype/shape"),
    "no_hash": (lambda c, r: c[0]["samples"][0]["output"].pop("sha256"), "dtype/shape"),
    "nonfinite": (lambda c, r: c[0]["samples"][0].__setitem__("array_problem", "non-finite values"), "non-finite"),
    "float32_query": (lambda c, r: c[0]["cell"].__setitem__("q_dtype", "float32"), "predeclared catalogue"),
    "tiled_kv_baseline": (lambda c, r: c[0]["baseline_meta"].__setitem__("kv_heads", 12), "native-GQA"),
    "fp32_kv_device": (lambda c, r: c[0]["device_inputs"]["k"].__setitem__(0, "float32"), "un-tiled float16 KV"),
    "input_hash": (lambda c, r: c[0]["input_hashes"].__setitem__("q", "0" * 64), "input identity"),
    "witness_unbound": (lambda c, r: c[0]["witness"][CAND]["metrics"].pop("vs_baseline_witness"),
                        "vs_baseline_witness"),
    "fairness": (lambda c, r: r["fairness"].__setitem__("timed_region", "whatever"), "fairness declaration"),
    "conv_hash": (lambda c, r: c[0]["conversion"]["experiments"][T.CONVERT]["samples"][3]["hashes"]
                  .__setitem__("k", "0" * 64), "not the cell's float16 KV"),
    "conv_missing_repeat": (lambda c, r: c[0]["conversion"]["experiments"][T.MATERIALIZE]["samples"].pop(),
                            "out of order"),
    "conv_scope_claim": (lambda c, r: c[0]["conversion"]["scope"].__setitem__("applies_to", "quantized cache"),
                         "scope labels"),
    "conv_dropped": (lambda c, r: c[0].pop("conversion"), "conversion record missing"),
    "conv_extra": (lambda c, r: c[1].__setitem__("conversion", copy.deepcopy(c[0]["conversion"])),
                   "without a declared conversion"),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_tampered_reports_are_refused(full, name):
    fn, needle = TAMPER[name]
    report = copy.deepcopy(full[0])
    fn(report["cells"], report)
    _refused(report, needle)


def test_missing_report_is_refused():
    for bad in (None, {}, {"cells": "x"}, {"cells": []}):
        assert T.evaluate_timing(bad)["verdict"] == "refused"


# ================================================================ gate: forged or fake can never unlock native timing

class NumericFake:
    """Independent float64 formula for the qualifier's mandatory catalogue (fake, never native)."""

    def run_case(self, case, inputs):
        ref = Q.host_reference(case, inputs)
        return {"candidate": (ref * (1 + 1e-5)).astype(np.float32), "baseline_f16": ref.astype(np.float16),
                "baseline_meta": {**Q.BASELINE, "kv_heads": case["hkv"], "q_heads": case["hq"]},
                "baseline_f32_scope": "omitted",
                "plan": {"threadgroups": case["expected_threadgroups"], "kv_end": list(case["kv_end"])},
                "engagement": {"expected_threadgroups": case["expected_threadgroups"],
                               "observed_threadgroups": case["expected_threadgroups"], "confirmed": True,
                               "issued_delta": 1, "confirmed_delta": 1}}

    def identity(self):
        return {"label": "fake numeric backend", "architecture": "applegpu_g17s"}

    def release(self):
        pass


@pytest.fixture(scope="module")
def fake_gate_run():
    cases, full_requested = Q.select_cases(None)
    report = Q.run_gate(NumericFake(), cases, source=lambda: {"a": "1" * 64})
    assert Q.evaluate(report, cases, full_requested=True)["verdict"] == "numeric_evidence_pass"
    return Q._LiveNativeRun(Q._WITNESS, NumericFake(), {"commit": "a" * 40}, report, [], cases, full_requested)


def _forbid(name):
    def fail(*a, **k):
        raise AssertionError(f"{name} must not run")
    return fail


def test_full_passing_fake_numeric_gate_cannot_build_native_ops(fake_gate_run):
    assert Q.native_verdict(fake_gate_run)["native_synthetic_gate"] is False
    for forged in (fake_gate_run, {"native_synthetic_gate": True}, SimpleNamespace(_witness=Q._WITNESS),
                   json.loads(json.dumps({"native_synthetic_gate": True, "verdict": "numeric_evidence_pass"}))):
        with pytest.raises(Q.Refused):
            T.NativeTimingOps(forged)


def test_native_runner_refuses_before_any_timed_dispatch_when_the_live_gate_fails(fake_gate_run, monkeypatch):
    monkeypatch.setattr(T, "_run_numeric_gate", lambda admission, cases, full: fake_gate_run)
    monkeypatch.setattr(T, "run_timing", _forbid("run_timing"))
    monkeypatch.setattr(T, "NativeTimingOps", _forbid("NativeTimingOps"))
    with pytest.raises(Q.Refused, match="live numeric gate did not pass; no timed dispatch"):
        T._run_native_timing({"commit": "a" * 40})


def test_native_runner_always_runs_the_full_mandatory_numeric_gate(monkeypatch):
    seen = {}

    def gate(admission, cases, full):
        seen.update(ids=[c["id"] for c in cases], full=full)
        raise Q.Refused("stop here")
    monkeypatch.setattr(T, "_run_numeric_gate", gate)
    with pytest.raises(Q.Refused, match="stop here"):
        T._run_native_timing({})
    assert seen == {"ids": list(Q.MANDATORY), "full": True}


class GateFake(NumericFake):
    """Fake gate backend with module paths; records numeric dispatches."""

    def __init__(self, admission, on_case=None):
        self.cand = SimpleNamespace(__file__=str(T.ROOT / Q.CANDIDATE_MODULE))
        self.mx = SimpleNamespace(__file__="/m/core.so")
        self.calls, self.releases, self.on_case = [], 0, on_case

    def run_case(self, case, inputs):
        self.calls.append(case["id"])
        if self.on_case:
            self.on_case(len(self.calls))
        return super().run_case(case, inputs)

    def release(self):
        self.releases += 1


def _gate_env(monkeypatch):
    state = {"changed": False}
    monkeypatch.setattr(Q, "source_guard", lambda admission: None)
    monkeypatch.setattr(T, "timing_hashes", lambda: {p: ("4" if state["changed"] and p == T.THIS_SCRIPT else "3") * 64
                                                     for p in T.TIMING_FILES})
    return state


def test_numeric_gate_runs_under_the_combined_guard_and_a_fake_backend_stays_non_native(monkeypatch):
    _gate_env(monkeypatch)
    made = []
    cases, full = Q.select_cases(None)
    live = T._run_numeric_gate(_admitted(), cases, full, backend_factory=lambda a: made.append(GateFake(a)) or made[0])
    assert made[0].calls == [c["id"] for c in cases if c["expect"] == "pass"] and live.post_run == []
    verdict = Q.native_verdict(live)
    assert verdict["evaluation"]["verdict"] == "numeric_evidence_pass"
    assert verdict["native_synthetic_gate"] is False and "backend is not the native backend" in verdict["reasons"]
    with pytest.raises(Q.Refused, match="did not pass"):
        T.NativeTimingOps(live)


def test_timing_file_change_before_the_gate_backend_stops_before_mlx(monkeypatch):
    state = _gate_env(monkeypatch)
    state["changed"] = True
    cases, full = Q.select_cases(None)
    with pytest.raises(Q.Refused, match="timing files changed"):
        T._run_numeric_gate(_admitted(), cases, full, backend_factory=_forbid("gate backend construction"))


@pytest.mark.parametrize("change_after", [0, 1, 5])
def test_timing_file_change_stops_the_next_numeric_dispatch(monkeypatch, change_after):
    state = _gate_env(monkeypatch)
    made = []

    def factory(admission):
        def on_case(n):
            if n == change_after:
                state["changed"] = True
        made.append(GateFake(admission, on_case))
        if change_after == 0:
            state["changed"] = True                      # after construction, before the first dispatch
        return made[0]
    cases, full = Q.select_cases(None)
    with pytest.raises(Q.Refused, match="timing files changed"):
        T._run_numeric_gate(_admitted(), cases, full, backend_factory=factory)
    assert len(made[0].calls) == change_after
    assert made[0].releases == max(change_after, 1)        # q.run_gate releases in finally, even on a guard stop


def test_timing_verdict_is_only_stamped_for_a_live_native_run(full, fake_gate_run):
    report = full[0]
    live_fake_ops = T._LiveTimingRun(T._TIMING_WITNESS, FakeOps([]), {}, fake_gate_run, report, [])
    no_witness = T._LiveTimingRun(object(), FakeOps([]), {}, fake_gate_run, report, [])
    look_alike = SimpleNamespace(_witness=T._TIMING_WITNESS, ops=None, gate_run=fake_gate_run, report=report,
                                 post_run=[])
    forged_ops = object.__new__(T.NativeTimingOps)
    forged_ops.gate_run = fake_gate_run
    typed = T._LiveTimingRun(T._TIMING_WITNESS, forged_ops, {}, fake_gate_run, report, [])
    for live in (report, live_fake_ops, no_witness, look_alike, typed, None):
        verdict = T.timing_verdict(live)
        assert verdict["native_timing_execution"] is False and verdict["reasons"]
        assert (verdict["qualified"], verdict["selected"], verdict["model_gain"]) == (False, False, False)
    assert T.timing_verdict(no_witness)["reasons"][0].startswith("no live native timing orchestration")
    typed_no_witness = T._LiveTimingRun(object(), forged_ops, {}, fake_gate_run, report, [])
    assert T.timing_verdict(typed_no_witness)["reasons"][0].startswith("no live native timing orchestration")
    # the typed forgery passes every structural check except the live numeric gate itself
    assert T.timing_verdict(typed)["reasons"] == ["live numeric gate did not pass"]
    assert T.timing_verdict(typed)["evaluation"]["verdict"] == "timing_evidence_complete"


def test_timing_verdict_takes_no_caller_gate_or_evaluation():
    assert list(inspect.signature(T.timing_verdict).parameters) == ["live"]
    assert list(inspect.signature(T._run_native_timing).parameters) == ["admission"]


# ================================================================ admission and guards (no MLX)

def test_admission_reuses_the_qualifier_admission_first(monkeypatch):
    calls = []
    monkeypatch.setattr(Q, "native_admission", lambda a, e: calls.append("q") or (_ for _ in ()).throw(
        Q.Refused("no --run-native")))
    monkeypatch.setattr(Q, "_git", _forbid("git before the qualifier admission"))
    with pytest.raises(Q.Refused, match="no --run-native"):
        T.timing_admission(SimpleNamespace(), {})
    assert calls == ["q"]


def test_admission_requires_timing_files_committed_and_unchanged(monkeypatch):
    base = {"commit": "a" * 40, "source_expected": {}, "mlx_files_before": {}}
    monkeypatch.setattr(Q, "native_admission", lambda a, e: dict(base))
    monkeypatch.setattr(Q, "_git", lambda *a: None)
    with pytest.raises(Q.Refused, match="not committed at HEAD"):
        T.timing_admission(None, {})
    monkeypatch.setattr(Q, "_git", lambda *a: b"other bytes")
    with pytest.raises(Q.Refused, match="differ from HEAD blobs"):
        T.timing_admission(None, {})
    monkeypatch.setattr(Q, "_git", lambda *a: (T.ROOT / a[1].split(":", 1)[1]).read_bytes())
    admission = T.timing_admission(None, {})
    assert admission["timing_expected"] == T.timing_hashes() and set(admission["timing_expected"]) == set(
        T.TIMING_FILES)


def _admitted():
    return {"commit": "a" * 40, "source_expected": {"s": "1" * 64}, "mlx_files_before": {"/m/core.so": "2" * 64},
            "timing_expected": {p: "3" * 64 for p in T.TIMING_FILES}}


PATHS = {"candidate": str(T.ROOT / Q.CANDIDATE_MODULE), "mx": "/m/core.so"}


@pytest.mark.parametrize("change, needle", [
    ("source", "source changed"),
    ("timing", "timing files changed"),
    ("candidate", "candidate module path"),
    ("mx", "mlx.core path"),
])
def test_guard_checks_source_build_timing_files_and_module_paths(monkeypatch, change, needle):
    def source_guard(admission):
        if change == "source":
            raise Q.Refused("bound source changed (HEAD/tree/hash/MLX)")
    monkeypatch.setattr(Q, "source_guard", source_guard)
    hashes = {p: "3" * 64 for p in T.TIMING_FILES}
    if change == "timing":
        hashes[T.THIS_SCRIPT] = "4" * 64
    monkeypatch.setattr(T, "timing_hashes", lambda: hashes)
    paths = dict(PATHS)
    if change == "candidate":
        paths["candidate"] = "/elsewhere/tensor_fa_research.py"
    if change == "mx":
        paths["mx"] = "/other/core.so"
    with pytest.raises(Q.Refused, match=needle):
        T.timing_guard(_admitted(), paths)


def test_guard_passes_on_an_unchanged_admission(monkeypatch):
    monkeypatch.setattr(Q, "source_guard", lambda admission: None)
    monkeypatch.setattr(T, "timing_hashes", lambda: {p: "3" * 64 for p in T.TIMING_FILES})
    T.timing_guard(_admitted(), PATHS)


@pytest.mark.parametrize("fail_at", [1, 2, 7, 40])
def test_guard_refusal_stops_the_next_dispatch_and_cleans_up(fail_at):
    log = []
    ops = FakeOps(log)
    guard = Guard(log, fail_at=fail_at)
    with pytest.raises(Q.Refused, match="fake guard"):
        T.run_timing(ops, T.CELLS[:1], guard=guard, clock=Clock(log))
    last_guard = len(log) - 1 - log[::-1].index("guard")
    assert not any(e.startswith("dispatch:") for e in log[last_guard:])   # nothing dispatched after the refusal
    assert ops.releases == 1 and ops.cell_releases == (0 if fail_at == 1 else 1)


def test_dispatch_exception_still_cleans_up():
    ops = FakeOps([])

    def boom(cell, ref, k):
        raise RuntimeError("device error")
    ops.candidate_fn = boom
    with pytest.raises(RuntimeError):
        small(ops=ops)
    assert ops.releases == 1 and ops.cell_releases == 1


# ================================================================ native code shape (static, never executed)

def test_native_ops_use_the_ordinary_baseline_and_no_reference_expansion():
    src = inspect.getsource(T.NativeTimingOps)
    assert "reference_attention" not in src and "host_mirror" not in src and "repeat" not in src
    assert 'mask="causal"' in src and "allow_research_metal=True" in src and "float32" not in src.split(
        "def dispatch")[1].split("if arm == CONVERT")[0]
    one = inspect.getsource(T._one)
    window = one.split("t0 = clock()")[1].split("t1 = clock()")[0]
    assert [ln.strip() for ln in window.strip().splitlines()] == ["d = ops.dispatch(arm, prep)", "ops.complete(d)"]


def test_command_template_names_the_wrapper_and_acknowledgement():
    doc = T.__doc__
    for needle in ("/tmp/mlx2-intake/stage3_gpu.py tensor-fa-timing", "--run-native --i-own-the-gpu",
                   "MLX2_INTAKE_SOURCE_COMMIT=", "MLX2_INTAKE_CPG_TASK=",
                   "~/Desktop/mlx2/.venv/bin/python scripts/benchmark_tensor_fa_research.py"):
        assert needle in doc
