"""Static CPU contracts for the opt-in D3 artifact diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_d3_artifact.py"


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("probe_d3_artifact", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli(tmp_path, output, *extra):
    return [
        str(SCRIPT),
        "--artifact",
        str(tmp_path / "artifact"),
        "--output",
        str(output),
        "--cpg-task",
        "task",
        "--cpg-owner-lock",
        str(tmp_path / "owner"),
        "--cpg-generation",
        "1",
        *extra,
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--context-tokens", "511"],
        ["--context-tokens", "8193"],
        ["--timeout-seconds", "0"],
        ["--timeout-seconds", "901"],
        ["--cpg-generation", "0"],
    ],
)
def test_cli_rejects_unbounded_controls_before_ownership(
    probe, monkeypatch, tmp_path, extra
):
    output = tmp_path / "fresh.json"
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output, *extra))
    monkeypatch.setattr(
        probe,
        "_ownership",
        lambda args: pytest.fail("ownership touched after invalid CLI"),
    )
    with pytest.raises(SystemExit, match="2"):
        probe.main()
    assert not output.exists()


def test_cli_refuses_to_overwrite_evidence_before_ownership(
    probe, monkeypatch, tmp_path
):
    output = tmp_path / "existing.json"
    output.write_text("prior evidence\n")
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output))
    monkeypatch.setattr(
        probe,
        "_ownership",
        lambda args: pytest.fail("ownership touched before output refusal"),
    )
    with pytest.raises(SystemExit, match="2"):
        probe.main()
    assert output.read_text() == "prior evidence\n"


def test_diagnostic_requires_unqualified_route(probe):
    probe._assert_unqualified({"qualification": "unqualified", "qualified": False})
    with pytest.raises(AssertionError, match="must not promote"):
        probe._assert_unqualified({"qualification": "qualified", "qualified": True})


def test_pinned_artifact_checks_complete_inventory_size_and_hash(
    probe, monkeypatch, tmp_path
):
    model = b"model bytes"
    path = tmp_path / "model.safetensors"
    path.write_bytes(model)
    monkeypatch.setattr(
        probe,
        "EXPECTED_ARTIFACT_FILES",
        {"model.safetensors": (len(model), hashlib.sha256(model).hexdigest())},
    )
    evidence = probe._verify_artifact_files(tmp_path)
    assert evidence["model.safetensors"]["bytes"] == len(model)

    (tmp_path / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        probe._verify_artifact_files(tmp_path)
    (tmp_path / "unexpected.json").unlink()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned D3 artifact file changed"):
        probe._verify_artifact_files(tmp_path)


def test_private_publication_is_exclusive_and_mode_0600(probe, tmp_path):
    output = tmp_path / "receipt.json"
    probe._publish_private(output, {"synthetic": "evidence"})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text()) == {"synthetic": "evidence"}
    with pytest.raises(FileExistsError):
        probe._publish_private(output, {"replacement": True})
    assert json.loads(output.read_text()) == {"synthetic": "evidence"}
    assert not list(tmp_path.glob(".receipt.json.tmp-*"))


def test_ownership_requires_matching_live_and_paired_cpg_owner(
    probe, monkeypatch, tmp_path
):
    owner = {
        "agent_id": "agent",
        "cpg_session": "session",
        "cpg_task": "task",
        "cpg_generation": 3,
        "pid": 10,
    }
    paired = {
        "cpg_owner": owner,
        "paired_gpu_owners": {"shared": {"pid": 11}, "temporary": {"pid": 11}},
    }
    monkeypatch.setitem(
        sys.modules,
        "qualification_gpu_ownership",
        SimpleNamespace(require_qualification_lease=lambda **kwargs: paired),
    )
    live_module = SimpleNamespace(
        observe=lambda *args: (owner, {"status": "in_progress"})
    )
    monkeypatch.setitem(sys.modules, "run_qualification_owned", live_module)
    args = SimpleNamespace(
        cpg_task="task",
        cpg_owner_lock=tmp_path,
        cpg_generation=3,
        mcp_url="http://radio.invalid/mcp",
    )
    assert probe._ownership(args)["validated"] == paired

    live_module.observe = lambda *args: (
        dict(owner, cpg_generation=4),
        {"status": "in_progress"},
    )
    with pytest.raises(RuntimeError, match="receipts differ"):
        probe._ownership(args)


@pytest.mark.parametrize(
    ("active_pid", "expected_contaminated"), [("123", True), ("-", False)]
)
def test_launchd_inventory_distinguishes_active_and_inactive_hyphenated_services(
    probe, monkeypatch, active_pid, expected_contaminated
):
    monkeypatch.setitem(
        sys.modules,
        "qualification_gpu_ownership",
        SimpleNamespace(_ancestor_pids=lambda pid: {pid}),
    )
    monkeypatch.setitem(
        sys.modules,
        "run_with_gpu_locks",
        SimpleNamespace(foreign_model_processes=lambda: []),
    )

    def check_output(command, **_kwargs):
        if command[:2] == ["ps", "-axo"]:
            return f"{probe.os.getpid()} 1 python probe_d3_artifact.py\n"
        assert command == ["launchctl", "list"]
        return (
            f"{active_pid}\t0\tcom.user.mlx2-image-mcp\n"
            "-\t0\tcom.user.mlx2-wiki\n"
            "-\t0\tcom.user.mlx2-qwen38-27b\n"
            "-\t0\tcom.user.rapid-mlx-server\n"
        )

    monkeypatch.setattr(probe.subprocess, "check_output", check_output)
    inventory = probe._contamination_inventory()
    assert len(inventory["loaded_gpu_services"]) == 4
    assert len(inventory["active_gpu_services"]) == (1 if active_pid == "123" else 0)
    assert inventory["contaminated"] is expected_contaminated


def test_forward_boundary_marks_partial_execution_truthfully(probe):
    calls = []
    engine = SimpleNamespace(_begin_execution=lambda: calls.append("forward"))
    report = {"gpu_executed": False, "observed_used": False}
    probe._track_execution(engine, report)
    engine._begin_execution()
    assert calls == ["forward"]
    assert report == {"gpu_executed": True, "observed_used": True}


def test_alarm_handler_raises_catchable_timeout(probe):
    with pytest.raises(TimeoutError, match="deadline reached"):
        probe._alarm_timeout(None, None)


def test_partial_forward_failure_publishes_private_receipt(
    probe, monkeypatch, tmp_path
):
    output = tmp_path / "failure.json"
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output))
    owner = {
        "agent_id": "agent",
        "cpg_session": "session",
        "cpg_task": "task",
        "cpg_generation": 1,
        "pid": 10,
    }
    ownership = {
        "validated": {
            "cpg_owner": owner,
            "paired_gpu_owners": {
                "shared": {"pid": 11},
                "temporary": {"pid": 11},
            },
        },
        "live": {"status": "in_progress"},
    }
    monkeypatch.setattr(probe, "_ownership", lambda args: ownership)
    monkeypatch.setattr(probe, "_source_hashes", lambda root: {"source.py": "abc"})
    monkeypatch.setattr(
        probe,
        "_verify_artifact_files",
        lambda root: {"model.safetensors": {"bytes": 1, "sha256": "def"}},
    )
    monkeypatch.setattr(
        probe,
        "_environment",
        lambda root: {
            "thermal": {"thermal_state": 0},
            "swap": "synthetic",
            "contamination": {"contaminated": False},
        },
    )

    def fail_after_forward(_args, report):
        report["gpu_executed"] = True
        report["observed_used"] = True
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(probe, "_execute", fail_after_forward)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        probe.main()
    receipt = json.loads(output.read_text())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert receipt["passed"] is False
    assert receipt["gpu_executed"] is receipt["observed_used"] is True
    assert receipt["error"] == "RuntimeError: synthetic forward failure"
    assert receipt["source_unchanged"] is True
    assert receipt["ownership_unchanged"] is True


def test_initial_ownership_failure_still_collects_and_publishes_receipt(
    probe, monkeypatch, tmp_path
):
    output = tmp_path / "precondition-failure.json"
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output))
    calls = {"ownership": 0, "source": 0, "artifact": 0, "environment": 0}

    def ownership(_args):
        calls["ownership"] += 1
        raise RuntimeError("no live lease")

    def source(_root):
        calls["source"] += 1
        return {"source.py": "abc"}

    def artifact(_root):
        calls["artifact"] += 1
        return {"model": {"bytes": 1, "sha256": "def"}}

    def environment(_root):
        calls["environment"] += 1
        return {
            "thermal": {"thermal_state": 0},
            "swap": "synthetic",
            "contamination": {"contaminated": False},
        }

    monkeypatch.setattr(probe, "_ownership", ownership)
    monkeypatch.setattr(probe, "_source_hashes", source)
    monkeypatch.setattr(probe, "_verify_artifact_files", artifact)
    monkeypatch.setattr(probe, "_environment", environment)
    monkeypatch.setattr(
        probe, "_execute", lambda *_: pytest.fail("execution after failed ownership")
    )
    with pytest.raises(RuntimeError, match="no live lease"):
        probe.main()
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is False
    assert receipt["gpu_executed"] is False
    assert receipt["error"] == "RuntimeError: no live lease"
    assert calls == {"ownership": 2, "source": 2, "artifact": 2, "environment": 2}
    assert "ownership_after" in receipt["postcondition_errors"][0]


def test_all_postconditions_are_attempted_and_ownership_is_last(
    probe, monkeypatch, tmp_path
):
    output = tmp_path / "postcondition-failure.json"
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output))
    order = []
    owner = {
        "agent_id": "agent",
        "cpg_session": "session",
        "cpg_task": "task",
        "cpg_generation": 1,
        "pid": 10,
    }
    ownership = {
        "validated": {
            "cpg_owner": owner,
            "paired_gpu_owners": {"shared": {"pid": 11}, "temporary": {"pid": 11}},
        },
        "live": {"status": "in_progress"},
    }
    counts = {"source": 0, "artifact": 0, "environment": 0}

    def fail_second(name, value):
        counts[name] += 1
        order.append(name)
        if counts[name] == 2:
            raise RuntimeError(f"{name} post failed")
        return value

    def get_owner(_args):
        order.append("ownership")
        return ownership

    environment = {
        "thermal": {"thermal_state": 0},
        "swap": "synthetic",
        "contamination": {"contaminated": False},
    }
    monkeypatch.setattr(probe, "_ownership", get_owner)
    monkeypatch.setattr(
        probe, "_source_hashes", lambda root: fail_second("source", {"s": "a"})
    )
    monkeypatch.setattr(
        probe,
        "_verify_artifact_files",
        lambda root: fail_second("artifact", {"a": {"bytes": 1, "sha256": "b"}}),
    )
    monkeypatch.setattr(
        probe, "_environment", lambda root: fail_second("environment", environment)
    )
    monkeypatch.setattr(probe, "_watch_thermal", lambda *args: None)
    monkeypatch.setattr(probe, "_execute", lambda *args: None)
    with pytest.raises(RuntimeError, match="source post failed"):
        probe.main()
    receipt = json.loads(output.read_text())
    assert order[-1] == "ownership"
    assert receipt["ownership_after"] == ownership
    assert len(receipt["postcondition_errors"]) == 3
    assert {item.split(":", 1)[0] for item in receipt["postcondition_errors"]} == {
        "source_sha256_end",
        "artifact_files_end",
        "environment_after",
    }


@pytest.mark.parametrize("mode", ["probe_failure", "excursion"])
def test_thermal_watcher_failure_or_excursion_fails_receipt(
    probe, monkeypatch, tmp_path, mode
):
    output = tmp_path / "thermal-failure.json"
    monkeypatch.setattr(sys, "argv", cli(tmp_path, output))
    owner = {
        "agent_id": "agent",
        "cpg_session": "session",
        "cpg_task": "task",
        "cpg_generation": 1,
        "pid": 10,
    }
    ownership = {
        "validated": {
            "cpg_owner": owner,
            "paired_gpu_owners": {"shared": {"pid": 11}, "temporary": {"pid": 11}},
        },
        "live": {"status": "in_progress"},
    }
    environment = {
        "thermal": {"thermal_state": 0},
        "swap": "synthetic",
        "contamination": {"contaminated": False},
    }
    monkeypatch.setattr(probe, "_ownership", lambda args: ownership)
    monkeypatch.setattr(probe, "_source_hashes", lambda root: {"s": "a"})
    monkeypatch.setattr(probe, "_verify_artifact_files", lambda root: {"a": "b"})
    monkeypatch.setattr(probe, "_environment", lambda root: environment)
    monkeypatch.setattr(probe, "_execute", lambda *args: None)

    def fail_watcher(_root, _stop, samples, errors):
        if mode == "probe_failure":
            errors.append("RuntimeError: synthetic thermal probe failure")
        else:
            samples.append({"monotonic_seconds": 2.0, "thermal_state": 1})

    monkeypatch.setattr(probe, "_watch_thermal", fail_watcher)
    with pytest.raises(RuntimeError, match="thermal watcher failed"):
        probe.main()
    receipt = json.loads(output.read_text())
    assert receipt["passed"] is False
    if mode == "probe_failure":
        assert receipt["thermal_monitor"]["errors"] == [
            "RuntimeError: synthetic thermal probe failure"
        ]
    else:
        assert any(
            item["thermal_state"] == 1 for item in receipt["thermal_monitor"]["samples"]
        )
