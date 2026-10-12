"""CPU/static contracts for the real-model wide-row diagnostic producer."""

import importlib.util
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from test_qwen38_dflash2_cpu import tiny_target


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "qualification/runs/speculative-composition-qualification-20261012"
    / "diagnose_qwen_wide_row_law.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("qwen_wide_row_diag", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_planner_imports_no_mlx_and_claims_no_execution():
    code = f"""
import json, runpy, sys
sys.argv = [{str(SCRIPT)!r}, '--profile', 'qwen38-m5']
try:
    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert not any(name == 'mlx' or name.startswith('mlx.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "planned"
    assert not plan["gpu_executed"]
    assert not plan["qualification_claim"]
    assert not plan["preflight_run"]
    assert plan["one_launch_contract"]["wide"] == [[1, 4]]
    assert plan["downstream_continuation"] == {
        "teacher_forced_tokens": 32,
        "teacher_policy": "reference greedy token is submitted to both states",
        "free_running_greedy": False,
        "free_running_max_tokens": 32,
    }


def test_distribution_receipt_names_first_argmax_difference_and_margins():
    module = _module()
    reference = np.array([[[4.0, 2.0, 1.0], [1.0, 4.0, 2.0]]])
    wide = np.array([[[3.5, 2.0, 1.0], [1.0, 3.0, 3.2]]])
    receipt = module._distribution_rows(wide, reference, [7, 8])
    assert not receipt["argmax_equal_all_rows"]
    assert receipt["first_argmax_differing_row_token"] == {
        "row": 1,
        "input_token": 8,
        "wide_token": 2,
        "reference_token": 1,
    }
    assert receipt["first_numerically_differing_row_token"]["row"] == 0
    assert all(row["wide_top_two"]["margin_nats"] >= 0 for row in receipt["rows"])
    assert all(
        row["reference_top_two"]["margin_nats"] >= 0 for row in receipt["rows"]
    )


def test_distribution_ties_follow_lowest_token_argmax_and_keep_zero_margin():
    module = _module()
    candidate = np.array([[[4.0, 4.0, 1.0, 0.0]]])
    reference = np.array([[[4.0, 4.0, 0.5, -1.0]]])
    receipt = module._distribution_rows(candidate, reference, [9])
    row = receipt["rows"][0]
    assert row["wide_top_two"]["token_ids"] == [0, 1]
    assert row["reference_top_two"]["token_ids"] == [0, 1]
    assert row["wide_top_two"]["margin_nats"] == 0.0
    assert row["reference_top_two"]["margin_nats"] == 0.0
    assert row["argmax_equal"]
    assert receipt["argmax_equal_all_rows"]
    assert receipt["first_argmax_differing_row_token"] is None


def test_target_forward_tracker_distinguishes_wide_and_serial_invocations():
    module = _module()
    mx.set_default_device(mx.cpu)
    model = tiny_target()
    tokens = mx.array([[1, 2, 3, 4]])
    with module._target_forward_tracker(model) as calls:
        logits, taps = model.forward_with_taps(tokens, model.make_cache(), [1, 6])
        mx.eval(logits, taps)
    assert [row["shape"] for row in calls] == [[1, 4]]

    cache = model.make_cache()
    with module._target_forward_tracker(model) as calls:
        for index in range(4):
            logits, taps = model.forward_with_taps(
                tokens[:, index : index + 1], cache, [1, 6]
            )
            mx.eval(logits, taps)
    assert [row["shape"] for row in calls] == [[1, 1]] * 4


def test_both_model_plans_bind_expected_capture_layers():
    module = _module()
    common = dict(
        execute=False,
        prefix_tokens=128,
        continuation_tokens=32,
        free_running_greedy=False,
        free_running_max_tokens=32,
        prompt="bounded",
        output_dir=None,
        i_own_the_gpu=False,
    )
    q38 = module._plan(SimpleNamespace(profile="qwen38-m5", **common))
    q36 = module._plan(SimpleNamespace(profile="qwen36-m5", **common))
    assert q38["capture_layers"] == [5, 19, 33, 47, 61]
    assert q36["capture_layers"] == [1, 6, 11, 16, 22, 27, 32, 37]
    assert q38["target_rows"] == q36["target_rows"] == 4
    assert q38["downstream_continuation"] == {
        "teacher_forced_tokens": 32,
        "teacher_policy": "reference greedy token is submitted to both states",
        "free_running_greedy": False,
        "free_running_max_tokens": 32,
    }


def test_cache_receipt_detects_exact_and_advanced_hybrid_state():
    module = _module()
    mx.set_default_device(mx.cpu)
    model = tiny_target()
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=cache))
    same = copy.deepcopy(cache)
    exact = module._cache_delta(mx, cache, same)
    assert exact["structural_equal"] and exact["all_bit_equal"]

    advanced = copy.deepcopy(cache)
    mx.eval(model(mx.array([[4]]), cache=advanced))
    changed = module._cache_delta(mx, advanced, cache)
    assert not changed["structural_equal"] or not changed["all_bit_equal"]


def test_teacher_forced_continuation_uses_reference_tokens_and_finishes_exact():
    module = _module()
    mx.set_default_device(mx.cpu)
    model = tiny_target()
    cache = model.make_cache()
    initial = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(initial)
    receipt = module._teacher_forced_continuation_scan(
        mx, model, cache, cache, initial, 4
    )
    assert receipt["mode"] == "reference_greedy_teacher_forced"
    assert receipt["steps_completed"] == 4
    assert receipt["argmax_equal_all_steps"]
    assert receipt["first_argmax_difference"] is None
    assert len(receipt["rows"]) == 4
    assert all(
        row["teacher_input_token"] == row["input_token"]
        for row in receipt["rows"]
    )
    assert receipt["final_state_delta"]["all_bit_equal"]


def test_free_running_scan_records_first_divergence_and_both_margins():
    module = _module()
    mx.set_default_device(mx.cpu)
    model = tiny_target()
    cache = model.make_cache()
    candidate = mx.array([[[4.0, 2.0, 1.0]]])
    reference = mx.array([[[1.0, 4.0, 2.0]]])
    receipt = module._free_running_greedy_scan(
        mx, model, cache, cache, candidate, reference, 8
    )
    assert receipt["diverged"]
    assert receipt["common_token_count"] == 0
    divergence = receipt["first_token_divergence"]
    assert divergence["step"] == 0
    assert divergence["candidate_token"] == 0
    assert divergence["reference_token"] == 1
    assert divergence["candidate_margin_nats"] > 0
    assert divergence["reference_margin_nats"] > 0
    assert receipt["final_state_delta"]["all_bit_equal"]
