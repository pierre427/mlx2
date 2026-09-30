"""CPU checks for scripts/qualify_moe_tensorfold_gpu.py.  Nothing here uses the GPU."""

import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models import moe_tensorfold, switch_layers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qualify_moe_tensorfold_gpu.py"


def _load():
    spec = importlib.util.spec_from_file_location("qualify_moe_tensorfold_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


q = _load()
FMT = {fmt.label: fmt for fmt in q.FORMATS}
SMALL = {"dims": 64, "hidden": 32, "experts": 8, "seed": 11, "warmups": 1, "repeats": 2,
         "budget_bytes": 1 << 34}


@pytest.fixture(autouse=True)
def cpu_and_clean_stats():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    moe_tensorfold.ENABLED[0] = True
    moe_tensorfold.STATS.clear()
    try:
        yield
    finally:
        mx.set_default_device(previous)
        moe_tensorfold.ENABLED[0] = True
        moe_tensorfold.STATS.clear()


def _small_tail_window(monkeypatch, rows=64):
    monkeypatch.setattr(switch_layers, "_SORTED_GATHER_TAIL_ROWS", rows)


# ------------------------------------------------------------------ case plan

@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8, 10])
def test_case_plan_hits_every_routing_window(top_k):
    limit = switch_layers._SORTED_GATHER_TAIL_ROWS
    cases = {case.kind: case for case in q.build_cases(top_k)}
    assert set(cases) == {"unsorted", "sorted", "doubled_tail", "stock_tail"}
    for case in q.build_cases(top_k):
        assert case.sorted == (case.kind != "unsorted")
        assert case.expected_stats["calls"] == 1
        assert case.expected_stats["assignments"] == case.rows_seen
    doubled = cases["doubled_tail"]
    assert doubled.rows == doubled.rows_seen <= limit < 2 * doubled.rows
    assert (2 * doubled.rows) % 64 and doubled.tensorfold_pad
    assert doubled.expected_stats["tail_padded_calls"] == 1
    stock = cases["stock_tail"]
    assert stock.rows > limit and stock.rows % 64
    assert stock.rows_seen % 64 == 0 and stock.rows_seen > stock.rows
    assert not stock.tensorfold_pad and "tail_padded_calls" not in stock.expected_stats


def test_case_plan_refuses_top_k_without_an_unsorted_case():
    with pytest.raises(ValueError, match="no unsorted case"):
        q.build_cases(switch_layers._GATHER_SORT_MIN_ASSIGNMENTS)


def test_default_matrix_names_required_and_optional_formats():
    required = {fmt.label for fmt in q.FORMATS if fmt.required}
    optional = {fmt.label for fmt in q.FORMATS if not fmt.required}
    assert required == {"dense-bf16", "dense-f16", "dense-f32",
                        "affine-q4-g32", "affine-q4-g64", "affine-q8-g64"}
    assert {"mxfp4-g32", "nvfp4-g16", "nvfp4-g16-dense-tail"} <= optional
    tail = FMT["nvfp4-g16-dense-tail"]
    assert (2048 + tail.dims_offset) % 32 and (2048 + tail.dims_offset) % 16 == 0


# ----------------------------------------------------------------- comparison

def test_bitwise_compare_sees_signed_zero_and_accepts_identical_nan():
    for dtype in (mx.float32, mx.bfloat16, mx.float16):
        zero = mx.array([0.0, 1.0], dtype)
        negative_zero = mx.array([-0.0, 1.0], dtype)
        assert mx.array_equal(negative_zero, zero).item()
        result = q.bitwise_compare(negative_zero, zero)
        assert not result["bitwise_equal"] and result["mismatched_elements"] == 1
        nan = mx.array([float("nan"), 2.0], dtype)
        assert q.bitwise_compare(nan, nan)["bitwise_equal"]
    assert not q.bitwise_compare(mx.zeros((2,), mx.float16),
                                 mx.zeros((2,), mx.float32))["shape_dtype_match"]


# ------------------------------------------------------------ run_format (CPU)

@pytest.mark.parametrize("label", ["dense-f32", "affine-q4-g32", "mxfp4-g32"])
def test_run_format_passes_with_exact_counters(monkeypatch, label):
    _small_tail_window(monkeypatch)
    cases = q.build_cases(4, prefill_tokens=16)
    report = q.run_format(FMT[label], cases, **SMALL)
    assert report["status"] == "pass", json.dumps(report, default=str)[:2000]
    assert report["install"]["ok"] and not report["install"]["reference_installed"]
    assert report["post_uninstall"]["ok"]
    for entry, case in zip(report["cases"], cases):
        assert entry["parity"]["bitwise_equal"]
        assert entry["candidate_stats_delta"] == case.expected_stats
        assert entry["reference_stats_delta"] == {}
        assert entry["timed_candidate_calls"] == SMALL["warmups"] + SMALL["repeats"] + 2
    padded = next(e for e in report["cases"] if e["case"]["kind"] == "doubled_tail")
    assert padded["candidate_stats_delta"]["tail_padded_calls"] == 1


def test_run_format_at_the_real_tail_window():
    cases = [case for case in q.build_cases(8) if case.kind.endswith("tail")]
    report = q.run_format(FMT["dense-f32"], cases, dims=16, hidden=16, experts=8, seed=3,
                          warmups=0, repeats=1, budget_bytes=1 << 34)
    assert report["status"] == "pass", json.dumps(report, default=str)[:2000]
    stock = next(e for e in report["cases"] if e["case"]["kind"] == "stock_tail")
    assert stock["candidate_stats_delta"]["assignments"] % 64 == 0
    assert stock["candidate_stats_delta"]["assignments"] > switch_layers._SORTED_GATHER_TAIL_ROWS


def test_swapped_halves_fail_the_gate(monkeypatch):
    _small_tail_window(monkeypatch)
    original = moe_tensorfold._GateUpGroup.__call__

    def swapped(self, x, indices, *, sorted_indices):
        gate, up = original(self, x, indices, sorted_indices=sorted_indices)
        return up, gate

    monkeypatch.setattr(moe_tensorfold._GateUpGroup, "__call__", swapped)
    report = q.run_format(FMT["dense-f32"], q.build_cases(4, prefill_tokens=16), **SMALL)
    assert report["status"] == "fail"
    assert all(not entry["parity"]["bitwise_equal"] for entry in report["cases"])
    assert q.verdict([report]) == "fail"


def test_stock_fallback_with_zero_counters_fails_the_gate(monkeypatch):
    # Bitwise equal because the stock path ran: the gate must still refuse.
    _small_tail_window(monkeypatch)
    monkeypatch.setattr(moe_tensorfold._GateUpGroup, "admit", lambda self, module: False)
    report = q.run_format(FMT["dense-f32"], q.build_cases(4, prefill_tokens=16), **SMALL)
    assert report["status"] == "fail"
    for entry in report["cases"]:
        assert entry["parity"]["bitwise_equal"]
        assert not entry["mechanism_engaged"] and entry["candidate_stats_delta"] == {}


def test_unsupported_optional_format_is_recorded_not_failed():
    bogus = q.Format("bogus", "bfloat16", {"group_size": 32, "bits": 4, "mode": "bogus"},
                     required=False)
    report = q.run_format(bogus, q.build_cases(4, prefill_tokens=16), **SMALL)
    assert report["status"] == "unsupported" and report["probe"]["reason"]
    required = q.Format("bogus", "bfloat16", {"group_size": 32, "bits": 4, "mode": "bogus"})
    assert q.run_format(required, [], **SMALL)["status"] == "fail"


def test_memory_budget_skips_and_leaves_the_gate_incomplete(monkeypatch):
    _small_tail_window(monkeypatch)
    cases = q.build_cases(4, prefill_tokens=16)
    fmt = FMT["dense-f32"]
    pair = q.estimate_bytes(fmt, None, dims=64, hidden=32, experts=8)
    report = q.run_format(fmt, cases, **{**SMALL, "budget_bytes": pair + 1})
    assert report["status"] == "incomplete"
    assert {entry["status"] for entry in report["cases"]} == {"skipped_memory"}
    assert q.verdict([report]) == "incomplete"


def test_verdict_table():
    def entry(label, status, required=True):
        return {"format": {"label": label, "required": required}, "status": status}

    passing = [entry("dense-bf16", "pass"), entry("mxfp4-g32", "unsupported", False)]
    assert q.verdict(passing) == "pass"
    assert q.verdict(passing + [entry("nvfp4-g16", "fail", False)]) == "fail"
    assert q.verdict(passing + [entry("affine-q8-g64", "error")]) == "fail"
    assert q.verdict(passing + [entry("affine-q4-g32", "skipped_memory")]) == "incomplete"
    assert q.verdict([entry("mxfp4-g32", "pass", False)]) == "incomplete"
    assert q.verdict(passing, aborted="LeaseChanged: x") == "aborted"


def test_reference_arm_is_uninstalled_not_disabled():
    # The A/B must never lean on the process switch: no set_enabled() call and
    # no assignment into ENABLED other than forcing it on.
    tree = ast.parse(SCRIPT.read_text())
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "set_enabled" not in calls and {"install", "uninstall"} <= calls
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any("ENABLED" in ast.unparse(target) for target in node.targets)]
    assert stores and all(ast.unparse(node.value) == "True" for node in stores)


# ---------------------------------------------------------------------- locks

def _receipts(tmp_path, first, second=None):
    paths = []
    for name, owner in (("shared", first), ("tmp", first if second is None else second)):
        lock = tmp_path / name / "gpu.lock"
        lock.mkdir(parents=True)
        if owner is not None:
            (lock / "owner.json").write_text(json.dumps(owner))
        paths.append(lock / "owner.json")
    return paths


def test_locks_bind_the_holder_pid_or_an_explicit_lease(tmp_path):
    owner = {"lease_id": "codex-tensorfold-coalescing-20260930-moe", "pid": os.getppid()}
    paths = _receipts(tmp_path, owner)
    bound = q.require_locks(paths)
    assert bound["owner"] == owner and bound["holder_pid_is_ancestor"]
    stranger = {**owner, "pid": 999_999_999}
    paths = _receipts(tmp_path / "b", stranger)
    with pytest.raises(q.LockError, match="not an ancestor"):
        q.require_locks(paths)
    assert not q.require_locks(paths, lease_id=owner["lease_id"])["holder_pid_is_ancestor"]
    with pytest.raises(q.LockError, match="is not the expected"):
        q.require_locks(paths, lease_id="someone-else")


@pytest.mark.parametrize("first,second,match", [
    ({"lease_id": "a", "pid": 1}, {"lease_id": "b", "pid": 1}, "disagree"),
    ({"lease_id": "a", "pid": 1}, {"lease_id": "a", "pid": 2}, "disagree"),
    ({"pid": 1}, None, "no lease_id"),
    (["not", "an", "object"], None, "not an object"),
])
def test_locks_refuse_mismatched_or_malformed_receipts(tmp_path, first, second, match):
    with pytest.raises(q.LockError, match=match):
        q.require_locks(_receipts(tmp_path, first, second), lease_id="a")


def test_locks_refuse_a_missing_receipt(tmp_path):
    paths = _receipts(tmp_path, {"lease_id": "a", "pid": os.getppid()})
    paths[1].unlink()
    with pytest.raises(q.LockError, match="missing"):
        q.require_locks(paths)


def test_main_refuses_the_gpu_without_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "LOCK_RECEIPTS", (tmp_path / "a/owner.json", tmp_path / "b/owner.json"))
    output = tmp_path / "receipt.json"
    with pytest.raises(SystemExit, match="refusing to use the GPU"):
        q.main(["--output", str(output), "--allow-dirty", "--dims", "64", "--hidden", "32",
                "--experts", "8"])
    assert not output.exists()
    assert mx.default_device() == mx.cpu


def test_plan_mode_binds_identities_without_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "LOCK_RECEIPTS", (tmp_path / "a/owner.json", tmp_path / "b/owner.json"))
    output = tmp_path / "plan.json"
    assert q.main(["--output", str(output), "--plan"]) == 0
    report = json.loads(output.read_text())
    assert report["mode"] == "plan" and report["verdict"] == "not_run" and "locks" not in report
    assert report["model_serving_qualified"] is False and report["route_selected"] is False
    assert report["law_id"] == moe_tensorfold.LAW_ID
    assert report["source"]["all_modules_from_this_tree"]
    assert set(report["source"]["files"]) == set(q.BOUND_SOURCES)
    assert report["mlx"]["version"] and report["mlx"]["binaries"]
    assert "device_info" not in report["host"]
    assert {case["kind"] for case in report["cases"]} == {
        "unsorted", "sorted", "doubled_tail", "stock_tail"}
    assert mx.default_device() == mx.cpu
