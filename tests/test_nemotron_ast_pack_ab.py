from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AB = load("nemotron_ast_pack_ab", "scripts/run_nemotron_ast_pack_ab.py")
EXACT = load("thermal_ladder_exact_contexts", "scripts/thermal_ladder_exact_contexts.py")
SESSION = "session-a"
LABEL = "label-a"


def test_exact_context_parser_is_strict():
    assert EXACT.parse_contexts("1024,16384") == (1024, 16384)
    for value in ("", "16384,1024", "1024,1024", "512,1024", "one,1024"):
        with pytest.raises(ValueError):
            EXACT.parse_contexts(value)


def test_exact_wrapper_asserts_loaded_module_and_restores_dict(tmp_path, monkeypatch):
    module = tmp_path / "thermal.py"
    module.write_text(
        "class Stream:\n"
        "    pass\n"
        "LENGTHS = (1024,)\n"
        "def main(): return 0\n"
    )
    wrapper = tmp_path / "b4.py"
    wrapper.write_text(
        "import thermal_ladder\n"
        "def declared_batch(): pass\n"
        "class FakeDict: pass\n"
        "thermal_ladder.Stream.batch = declared_batch\n"
        "thermal_ladder.dict = FakeDict\n"
    )
    series = tmp_path / "benchmark_results/2026-09-29/scripts/series"
    series.mkdir(parents=True)
    monkeypatch.setenv("MLX2_CAMPAIGN_ROOT", str(tmp_path))
    # configure() installs the staged module process-wide, as its CLI needs;
    # keep this fake from answering later tests' ``import thermal_ladder``.
    monkeypatch.setitem(sys.modules, "thermal_ladder", None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    loaded, assertions = EXACT.configure(wrapper, module, (1024, 16384))
    assert loaded.LENGTHS == (1024, 16384)
    assert loaded.EXPLICIT_WIDTHS == (4,)
    assert assertions["declared_batch_installed"] is True
    assert assertions["builtin_dict_restored"] is True


def response(mode: str) -> dict:
    frontier = None
    observed = "eager_all_branches"
    if mode == "ordered_live_frontier":
        observed = mode
        frontier = {"finished": True, "sessions_retired": 4, "max_service_gap": 1}
    return {
        "done": True,
        "prompt_tokens": 1024,
        "completion_tokens": 32,
        "receipt": {
            "route": "external_draft",
            "speculation": {
                "progressive_verification": {
                    "target_row_executor": {
                        "launches": 5,
                        "adapter_owned_launches": 5,
                        "legacy_launches": 0,
                        "row_exact_launches": 5,
                        "nonexact_launches": 0,
                    },
                    "ordered_alternatives": {
                        "cross_session_ast_packing": {
                            "configured_pack_rows": 8,
                            "observed_used": True,
                            "mode": observed,
                            "live_frontier": frontier,
                        }
                    },
                }
            },
        },
    }


def run_row(mode: str, *, warm_equals_cold: int = 4) -> dict:
    owner = {
        "session": SESSION,
        "label": LABEL,
        "lease_id": "lease-a",
        "pid": 10,
    }
    return {
        "width": 4,
        "calibrated_tokens": [1024] * 4,
        "contaminated": False,
        "contamination": [],
        "thermal_pre": {"stable": True},
        "thermal_post": {"breached": False},
        "gpu_lock_owner": {"before": owner, "after": dict(owner)},
        "summary": {
            "streams_done": True,
            "needle_correct": 8,
            "needle_total": 8,
            "warm_apc_hits": 4,
            "warm_equals_cold": warm_equals_cold,
        },
        "requests": {
            arm: {
                "rows": [response(mode) for _ in range(4)],
                "batch_cohort": {"id": f"cohort-{arm}", "size": 4},
                "aggregate_completion_tokens_per_second": 12.5,
            }
            for arm in ("cold", "warm")
        },
    }


def ladder(mode: str, *, warm_equals_cold: int = 4) -> dict:
    cells = []
    for context in AB.CONTEXTS:
        warmup = run_row(mode)
        measured = run_row(mode, warm_equals_cold=warm_equals_cold)
        passed = warm_equals_cold == 4
        cells.append(
            {
                "requested_tokens": context,
                "width": 4,
                "error": None,
                "passed": passed,
                "warmup": warmup,
                "runs": [measured],
            }
        )
    scheduler = {}
    if mode == "ordered_live_frontier":
        scheduler = {
            "external_progressive_ast_frontier_sessions_retired": 32,
            "external_progressive_ast_frontier_max_service_gap": 1,
        }
    return {
        "passed": warm_equals_cold == 4,
        "runs_per_cell": 1,
        "cells": cells,
        "final": {"scheduler": scheduler},
    }


def validate(value: dict, mode: str, *, rc: int = 0):
    return AB.validate_ladder(
        value,
        mode,
        session=SESSION,
        label=LABEL,
        main_rc=rc,
        route_validation={"passed": True},
    )


def test_validator_accepts_real_b4_shapes_for_both_arms():
    eager = validate(ladder("eager"), "eager")
    live = validate(ladder("ordered_live_frontier"), "ordered_live_frontier")
    assert eager["status"] == "passed"
    assert eager["queue_retirement_receipts"] == 0
    assert live["status"] == "passed"
    assert live["queue_retirement_receipts"] == 32
    assert live["primary_metric"]["values_by_context"] == {
        "1024": 12.5,
        "16384": 12.5,
    }


def test_validator_fails_closed_on_owner_geometry_and_route_gate():
    value = ladder("ordered_live_frontier")
    row = value["cells"][0]["runs"][0]
    row["gpu_lock_owner"]["after"]["lease_id"] = "changed"
    row["requests"]["cold"]["rows"].pop()
    result = AB.validate_ladder(
        value,
        "ordered_live_frontier",
        session=SESSION,
        label=LABEL,
        main_rc=0,
        route_validation={"passed": False},
    )
    assert result["status"] == "failed"
    assert any("owner changed" in item for item in result["errors"])
    assert any("exactly four" in item for item in result["errors"])
    assert any("route-receipt" in item for item in result["errors"])


def test_validator_fails_closed_on_executor_contamination_and_retirement():
    value = ladder("ordered_live_frontier")
    row = value["cells"][0]["runs"][0]
    row["contaminated"] = True
    progressive = row["requests"]["cold"]["rows"][0]["receipt"]["speculation"][
        "progressive_verification"
    ]
    progressive["target_row_executor"]["nonexact_launches"] = 1
    progressive["ordered_alternatives"]["cross_session_ast_packing"][
        "live_frontier"
    ] = None
    result = validate(value, "ordered_live_frontier")
    assert result["status"] == "failed"
    assert any("contaminated" in item for item in result["errors"])
    assert any("row-exact" in item for item in result["errors"])
    assert any("queue-retirement" in item for item in result["errors"])


def test_near_tie_mismatch_is_pending_not_correctness_failure():
    result = validate(
        ladder("ordered_live_frontier", warm_equals_cold=3),
        "ordered_live_frontier",
        rc=1,
    )
    assert result["status"] == "pending_bounded_first_divergence"
    assert result["passed"] is False
    assert result["correctness_failed"] is False
    assert result["performance_evidence_usable"] is False
    assert result["pending_classification"]


def test_plan_is_balanced_and_binds_untracked_content(tmp_path, monkeypatch):
    files = {}
    for name in ("main", "b4", "cpg-python", "cpg-job", "model", "head"):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text("python")
    monkeypatch.setattr(
        AB,
        "source_binding",
        lambda *_args: {
            "head": "h",
            "branch": "b",
            "diff_sha256": "d",
            "worktree_content": {"sha256": "content", "files": []},
        },
    )
    args = SimpleNamespace(
        venv=venv, model=files["model"], head=files["head"],
        main_harness=files["main"], b4_wrapper=files["b4"],
        cpg_python=files["cpg-python"], cpg_job=files["cpg-job"],
        cpg_task="task", timeout=7200, port_base=19000,
        out=tmp_path / "evidence", expected_head="h", expected_branch="b",
        expected_diff_sha256="d",
    )
    plan = AB.make_plan(args)
    assert [unit["mode"] for unit in plan["units"]] == list(AB.ARM_ORDER)
    assert plan["design"]["primary_metric"] == (
        "cold aggregate completion tokens per second"
    )
    assert plan["source"]["worktree_content"]["sha256"] == "content"
    for unit in plan["units"]:
        command = unit["command"]
        assert Path(unit["output"]).is_dir()
        owned_index = command.index(str(ROOT / "scripts/run_qualification_owned.py"))
        assert command[owned_index - 1] == str(files["cpg-python"])
        assert str(ROOT / "scripts/run_with_gpu_locks.py") in command
        assert "--expected-worktree-content-sha256" in command


def test_plan_refuses_nonempty_evidence_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        AB,
        "source_binding",
        lambda *_args: {"worktree_content": {"sha256": "content"}},
    )
    out = tmp_path / "evidence"
    out.mkdir()
    (out / "prior.json").write_text(json.dumps({"status": "prior"}))
    args = SimpleNamespace(
        venv=tmp_path, model=tmp_path, head=tmp_path, main_harness=tmp_path,
        b4_wrapper=tmp_path, cpg_python=tmp_path, cpg_job=tmp_path,
        cpg_task="task", timeout=1, port_base=1, out=out,
        expected_head="h", expected_branch="b", expected_diff_sha256="d",
    )
    with pytest.raises(RuntimeError, match="nonempty evidence"):
        AB.make_plan(args)


def test_main_preserves_cpg_virtualenv_entrypoint_symlink(tmp_path, monkeypatch):
    real_python = tmp_path / "base-python"
    real_python.write_text("base")
    venv_python = tmp_path / "cpg-venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    captured = {}

    def fake_make_plan(args):
        captured["cpg_python"] = args.cpg_python
        return {}

    monkeypatch.setattr(AB, "make_plan", fake_make_plan)
    monkeypatch.setattr(AB, "atomic_json", lambda *_args: None)
    common = tmp_path / "placeholder"
    common.write_text("x")
    out = tmp_path / "out"
    rc = AB.main([
        "plan", "--main-harness", str(common), "--b4-wrapper", str(common),
        "--model", str(common), "--head", str(common), "--venv", str(tmp_path),
        "--out", str(out), "--cpg-python", str(venv_python),
        "--cpg-job", str(common), "--cpg-task", "task",
        "--expected-head", "head", "--expected-branch", "branch",
        "--expected-diff-sha256", "diff",
    ])
    assert rc == 0
    assert captured["cpg_python"] == venv_python.absolute()
    assert captured["cpg_python"] != venv_python.resolve()


def test_harness_shadow_freezes_b4_and_uses_explicit_width_hook(tmp_path, monkeypatch):
    main = tmp_path / "legacy" / "run_main.py"
    b4 = tmp_path / "legacy" / "thermal_ladder_ast_b4.py"
    main.parent.mkdir()
    main.write_text(
        "legacy-main\n"
        '        "layerwise_short_exact", "auto", "auto",\n'
    )
    b4.write_text("legacy-b4")
    canonical = tmp_path / "canonical.py"
    canonical.write_text("for width in dict.fromkeys((1, args.wide)):\n    pass\n")
    monkeypatch.setattr(AB, "THERMAL_LADDER", canonical)
    out = tmp_path / "unit" / "run"
    out.parent.mkdir()
    staged = AB.stage_unit_harness(main, b4, out)
    assert "layerwise_short_exact_routed_batch" in staged["paths"]["main"].read_text()
    assert staged["paths"]["base_b4"].read_text() == "legacy-b4"
    transformed = staged["paths"]["thermal_module"].read_text()
    assert "for width in EXPLICIT_WIDTHS:" in transformed
    assert "dict.fromkeys" not in transformed
    assert all(
        Path(record["path"]).is_file()
        for key, record in staged["manifest"].items()
        if key != "transformation"
    )


def test_stop_process_group_records_already_stopped(monkeypatch):
    lifecycle = {"cleanup_actions": []}
    monkeypatch.setattr(AB, "process_group_alive", lambda _pgid: False)
    AB.stop_process_group(123, lifecycle)
    assert lifecycle["cleanup_actions"] == [
        {"pgid": 123, "term_sent": False, "kill_sent": False, "already_stopped": True}
    ]


def test_interrupt_handlers_cover_term_int_and_hup(monkeypatch):
    installed = {}
    monkeypatch.setattr(AB.signal, "getsignal", lambda signum: f"old-{signum}")
    monkeypatch.setattr(
        AB.signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler)
    )
    lifecycle = {"signals": []}
    old = AB.install_interrupt_handlers(lifecycle)
    assert set(installed) == {AB.signal.SIGTERM, AB.signal.SIGINT, AB.signal.SIGHUP}
    assert set(old) == set(installed)
    with pytest.raises(AB.UnitInterrupted):
        installed[AB.signal.SIGTERM](AB.signal.SIGTERM, None)
    assert lifecycle["signals"][0]["signal"] == AB.signal.SIGTERM


def test_process_group_probe_uses_process_table_and_ignores_zombies(monkeypatch):
    monkeypatch.setattr(
        AB.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (
            "101 500 S python server.py\n"
            "102 500 Z defunct\n"
            "103 700 S other\n"
        ),
    )
    assert AB.process_group_members(500) == [
        {"pid": 101, "state": "S", "command": "python server.py"}
    ]
    assert AB.process_group_alive(500) is True
    assert AB.process_group_alive(600) is False
