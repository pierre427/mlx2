"""Value-sweep parity canaries reach every value they claim, gates build once, and
the tree/ordinary token comparison never calls partial evidence identical."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_fused_gdn_prefill as prefill
from mlx2.runtime.models import qwen4_fused_gdn_verify as verify
from mlx2.runtime.models import qwen38_tree_gdn as tree
from mlx2.runtime.models import served_exp
from mlx2.runtime.models.served_exp import covering_rows
from mlx2.runtime.tensorfold_qwen38 import lane_glue

ROOT = Path(__file__).resolve().parents[1]


def _finite_bf16():
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    finite = np.isfinite((bits.astype(np.uint32) << 16).view(np.float32))
    return bits[finite]


def _distinct(values) -> set:
    mx.eval(values)
    return set(np.array(values.view(mx.uint16)).reshape(-1).tolist())


def _permute_then_truncate(grid, count, seed):
    # The construction the canaries used before: shuffle a longer tile, then cut it.
    reps = -(-count // int(grid.size))
    order = mx.random.permutation(reps * int(grid.size), key=mx.random.key(seed))
    return mx.tile(grid, (reps,))[order][:count]


def test_permute_then_truncate_drops_values():
    grid = mx.array(_finite_bf16()).view(mx.bfloat16)
    assert len(_distinct(_permute_then_truncate(grid, 16 * 4096, 10170))) < int(grid.size)


@pytest.mark.parametrize("rows,width", [(16, 4096), (11, 48 * 128), (1, 1 << 16), (3, 70000)])
def test_covering_rows_holds_every_value(rows, width):
    grid = mx.array(_finite_bf16()).view(mx.bfloat16)
    out = covering_rows(grid, rows, width, 5)
    assert out.shape == (rows, width)
    if width >= int(grid.size):
        for r in range(rows):
            assert _distinct(out[r]) == set(_finite_bf16().tolist())
    else:
        assert _distinct(out) == set(_finite_bf16().tolist())


def test_covering_rows_refuses_a_block_too_small_to_cover():
    grid = mx.array(_finite_bf16()).view(mx.bfloat16)
    with pytest.raises(ValueError, match="cannot cover"):
        covering_rows(grid, 2, 100, 0)


@pytest.mark.parametrize("name", ["mlp_act", "gdn_post"])
def test_lane_glue_parity_case_reaches_every_finite_bf16(name):
    args, _ = lane_glue.parity_case(name)
    swept = args[0] if name == "mlp_act" else args[1]
    assert _distinct(swept) == set(_finite_bf16().tolist())


def _window_grid():
    return set(np.array(verify._decode_parity_grid().view(mx.uint16)).tolist())


def test_gdn_canaries_cover_the_magnitude_window_on_every_row():
    want = _window_grid()
    qkv = tree._parity_inputs(2, 16, 48, 128, 128, 4)[0]
    for r in range(qkv.shape[1]):
        assert _distinct(qkv[0, r]) == want
    swept = prefill._sweep(3, 10240, 0)
    for r in range(swept.shape[1]):
        assert _distinct(swept[0, r]) == want
    qkv = verify._decode_parity_inputs(2, 2, mx.float32)[0]
    for lane in range(qkv.shape[0]):
        for r in range(qkv.shape[1]):
            assert _distinct(qkv[lane, r]) == want


def test_lane_glue_gate_is_built_once_under_concurrent_first_use(monkeypatch):
    built = []

    class SlowGate:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)
            built.append(self)

    monkeypatch.setattr(served_exp, "ServedExpGate", SlowGate)
    monkeypatch.setattr(lane_glue, "_SERVED", {})
    results = []
    barrier = threading.Barrier(8)

    def first_use():
        barrier.wait()
        results.append(lane_glue._gate("mlp_act")[0])

    threads = [threading.Thread(target=first_use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1
    assert all(gate is built[0] for gate in results)


def _parity_script():
    spec = importlib.util.spec_from_file_location(
        "check_qwen38_tree_ordinary_parity", ROOT / "scripts" / "check_qwen38_tree_ordinary_parity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(path, prompts, passed=True):
    path.write_text(json.dumps({"passed": passed,
                                "prompts": {k: {"token_ids": v} for k, v in prompts.items()}}))
    return path


def test_parity_compare_identical_only_on_complete_passing_evidence(tmp_path, capsys):
    compare = _parity_script().compare
    ref = _receipt(tmp_path / "ref.json", {"a": [1, 2], "b": [3]})
    same = _receipt(tmp_path / "same.json", {"a": [1, 2], "b": [3]})
    assert compare([ref, same]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True

    partial = _receipt(tmp_path / "partial.json", {"a": [1, 2]})
    empty = _receipt(tmp_path / "empty.json", {})
    failed = _receipt(tmp_path / "failed.json", {"a": [1, 2], "b": [3]}, passed=False)
    for arm in (partial, empty, failed):
        assert compare([ref, arm]) == 1
        report = json.loads(capsys.readouterr().out)
        assert report["identical"] is False and report["arms"][0]["identical"] is False
    assert compare([empty, partial]) == 1
    assert json.loads(capsys.readouterr().out)["arms"][0]["prompts"]["a"]["missing_from"] == "reference"


def test_parity_route_observation_separates_request_and_scheduler_rounds():
    observe = _parity_script()._route_observation
    prompts = {
        "a": {"route_receipt": {"route": "external_draft", "speculation": {
            "external_rounds": 7, "ordinary_fallback": False,
            "target_tree": {"provider": "qwen38_tensorfold_v1",
                            "lane_width_histogram": {"1": 8}}}}},
        "b": {"route_receipt": {"route": "external_draft", "speculation": {
            "external_rounds": 5, "ordinary_fallback": False,
            "target_tree": {"provider": "qwen38_tensorfold_v1",
                            "lane_width_histogram": {"1": 5}}}}},
    }
    row = observe("tree", prompts, {
        "scheduler.external_rounds": 10,
        "scheduler.external_auto_tree_rounds": 11,
        "scheduler.external_tree_rounds": 10,
        "scheduler.external_target_tree_target_rounds": 11,
    })
    assert row == {
        "selected_for_requests": True,
        "observed_used": True,
        "request_count": 2,
        "external_draft_request_count": 2,
        "tensorfold_provider_request_count": 2,
        "ordinary_fallback_request_count": 0,
        "request_receipt_external_rounds_sum": 12,
        "request_receipt_target_dispatches_sum": 13,
        "scheduler_external_rounds": 10,
        "scheduler_external_auto_tree_rounds": 11,
        "scheduler_external_tree_rounds": 10,
        "scheduler_target_dispatches": 11,
        "scheduler_snapshot_state": "periodic_snapshot_lag",
    }


def test_parity_route_observation_rejects_fallback_or_wrong_provider():
    observe = _parity_script()._route_observation
    prompts = {"a": {"route_receipt": {"route": "external_draft", "speculation": {
        "external_rounds": 1, "ordinary_fallback": True,
        "target_tree": {"provider": "other"}}}}}
    row = observe("tree", prompts, {})
    assert row["selected_for_requests"] is True
    assert row["observed_used"] is False
    assert row["ordinary_fallback_request_count"] == 1


def test_parity_source_identity_uses_registry_artifact_identity(tmp_path, monkeypatch):
    module = _parity_script()
    root = tmp_path / "checkout"
    src = root / "src"
    src.mkdir(parents=True)
    model = tmp_path / "Qwen3.8-27B-oQ4e-mtp"
    model.mkdir()

    def output(command, **_kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        assert command[1] == "-c"
        return json.dumps({
            "artifact_name": model.name,
            "artifact_fingerprint": "b" * 64,
        }) + "\n"

    monkeypatch.setattr(module.subprocess, "check_output", output)
    identity = module._source_identity(src, model)
    assert identity == {
        "mlx2_commit": "a" * 40,
        "src": str(src.resolve()),
        "tracked_source_dirty": False,
        "tracked_source_status": "",
        "artifact_name": model.name,
        "artifact_fingerprint": "b" * 64,
        "checkpoint": f"{model.name}@{'b' * 64}",
    }
