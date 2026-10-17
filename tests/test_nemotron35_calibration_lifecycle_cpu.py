import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return (importlib.import_module("run_nemotron35_super_quick_probe"),
            importlib.import_module("collect_nemotron35_calibration"))


def test_calibration_requires_same_live_owner_source_and_stopped_server(modules, monkeypatch):
    _, collector = modules
    owner = {"cpg_owner": {"cpg_task": "task", "cpg_generation": 7},
             "paired_gpu_owners": {"shared": {"lease_id": "pair"}}}
    receipt = {"schema": "mlx2.nemotron35-calibration-server-unit.v1", "artifact": "target", "source_sha256": "source",
               "ordinary_reference_passed": True, "ordinary_server_passed": True, "server_group_stopped": True,
               "max_lanes": 1, "max_inflight": 1, "server_pgid": 1234,
               "dependencies": {"mlx": "fixture"},
               "cpg_owner": owner["cpg_owner"], "paired_owner": owner["paired_gpu_owners"]["shared"]}
    monkeypatch.setattr(collector.common, "group_running", lambda _: [])
    monkeypatch.setattr(collector.common, "runtime_dependencies", lambda: {"mlx": "fixture"})
    collector.validate_unit_receipt(receipt, owner=owner, artifact="target", source_sha="source")
    for key, value in (("max_lanes", 4), ("max_inflight", 4), ("artifact", "another"),
                       ("source_sha256", "another"), ("ordinary_reference_passed", False),
                       ("dependencies", {"mlx": "another"}),
                       ("paired_owner", {"lease_id": "other"}), ("server_group_stopped", False)):
        with pytest.raises(ValueError):
            collector.validate_unit_receipt({**receipt, key: value}, owner=owner, artifact="target", source_sha="source")
    monkeypatch.setattr(collector.common, "group_running", lambda _: [1234])
    with pytest.raises(ValueError, match="still|live"):
        collector.validate_unit_receipt(receipt, owner=owner, artifact="target", source_sha="source")


@pytest.mark.parametrize("collector_fails", [False, True])
@pytest.mark.parametrize("component", ["calibration", "kernels"])
def test_outer_unit_stops_server_before_capture_and_retains_failure_receipts(modules, monkeypatch, tmp_path, collector_fails, component):
    probe, _ = modules
    from mlx2.adapters import nemotron35_super
    from mlx2.conversion import kernel_probe

    model, out = tmp_path / "target", tmp_path / "evidence"
    model.mkdir()
    artifact = {"identity": {"fingerprint": "target"}, "conversion": {"tensor_bytes": 1000}}
    owner = {"cpg_owner": {"cpg_task": "task", "cpg_generation": 7},
             "paired_gpu_owners": {"shared": {"lease_id": "pair", "pid": 17}}}
    host = {"foreign_model_processes": [], "foreign_live_services": [], "waiter_files": [],
            "memory": {"available_bytes": 100 * probe.common.GIB, "swapouts": 0}, "swap": {"used_bytes": 0},
            "memory_pressure_level": 1, "thermal": {"thermal_state": 0, "pmset_no_thermal_warning": True,
             "pmset_no_performance_warning": True, "pmset_no_cpu_power_warning": True,
             "battery_temperature_c": 30, "virtual_temperature_c": 35}}
    initial = {"artifact": "target", "runtime": "runtime", "profile": "ordinary", "qualification": "candidate",
               "settings": {"route": "ordinary", "max_lanes": 1, "max_inflight": 1}}
    events, live = [], set()
    monkeypatch.setattr(nemotron35_super, "inspect_artifact", lambda _: artifact)
    monkeypatch.setattr(probe.common, "require_owner", lambda _: owner)
    monkeypatch.setattr(probe.common, "source_identity", lambda: {"sha256": "source"})
    monkeypatch.setattr(probe.common, "runtime_dependencies", lambda: {"mlx": "fixture"})
    monkeypatch.setattr(probe.common, "host_snapshot", lambda: host)
    monkeypatch.setattr(probe.common, "group_running", lambda pgid: [pgid] if pgid in live else [])
    monkeypatch.setattr(probe, "wait_health", lambda *_: initial)
    monkeypatch.setattr(kernel_probe, "verify", lambda _: {"fingerprint": "kernel-inputs"})
    inputs = tmp_path / "kernel-inputs"
    inputs.mkdir()

    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            self.pid, self.returncode = 100000 + len(events), 0
            if "--reference-child" in command:
                events.append("ordinary_reference")
                (out / "ordinary-reference.json").write_text(json.dumps({"status": "passed"}))
            elif "mlx2.server" in command:
                assert command[command.index("--max-lanes") + 1] == "1"
                assert command[command.index("--max-inflight") + 1] == "1"
                events.append("server_start")
                live.add(self.pid)
                self.returncode = None
            else:
                script = "collect_nemotron35_calibration.py" if component == "calibration" else "probe_nemotron35_nvfp4_kernels.py"
                assert command[2].endswith(script)
                assert not live
                events.append("component_start")
                (out / component).mkdir()
                receipt = "calibration/unit.json" if component == "calibration" else "kernels/kernels.json"
                (out / receipt).write_text(json.dumps({"state": "failed" if collector_fails else "completed",
                                                       "eligible_for_merge": not collector_fails, "finalization_errors": []}))
                self.returncode = 1 if collector_fails else 0

        def wait(self, **kwargs):
            return self.returncode

        def poll(self):
            return self.returncode

    def stop(process):
        if process is not None:
            if process.pid in live:
                live.remove(process.pid)
                events.append("server_stop")
            process.returncode = process.returncode or 0

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, body):
            events.append("server_request")
            text = 'Return only JSON' in json.dumps(body)
            vision = 'image_url' in json.dumps(body)
            content = "red" if vision else '{"code":"NEMOTRON_K7","count":9}' if text else "323"
            return {"choices": [{"message": {"content": content}}],
                    "mlx2": {"profile": initial["profile"], "route": "ordinary"}}

        def get(self, path):
            return initial

    monkeypatch.setattr(probe.subprocess, "Popen", Process)
    monkeypatch.setattr(probe.common, "stop_group", stop)
    monkeypatch.setattr(probe.matrix, "Client", Client)
    component_args = (["--calibration-corpus", str(tmp_path / "corpus"), "--calibration-count", "2", "--calibration-seconds", "30"]
                      if component == "calibration" else ["--kernel-inputs", str(inputs)])
    monkeypatch.setattr(sys, "argv", [str(probe.__file__), "--model", str(model), "--out-dir", str(out),
        "--cpg-owner-lock", str(tmp_path / "owner"), "--port", "0", *component_args])
    if collector_fails:
        with pytest.raises(RuntimeError, match="Offline calibration failed|Kernel component probe failed"):
            probe.main()
    else:
        probe.main()
    assert events.index("ordinary_reference") < events.index("server_start") < events.index("server_request")
    assert events.index("server_request") < events.index("server_stop") < events.index("component_start")
    assert not live
    unit = json.loads((out / ("calibration-server-unit.json" if component == "calibration" else "kernel-server-unit.json")).read_text())
    assert unit["server_group_stopped"] and unit["max_lanes"] == unit["max_inflight"] == 1
    campaign = json.loads((out / "campaign.json").read_text())
    assert campaign["status"] == ("failed" if collector_fails else "completed")
    if component == "calibration":
        assert all(row["performance_reportable"] is False for row in campaign["cases"])
