import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_qwen36_overnight.py"
SPEC = importlib.util.spec_from_file_location("qwen36_overnight", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def identity():
    value = {"sha256": "test-identity", "artifacts": [], "settings": {},
             "git_head": "test", "git_status": "", "runtime_source_sha256": "test"}
    return value


def campaign(tmp_path, steps, **kwargs):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    return module.Campaign(
        module.Config(root=root, run_dir=tmp_path / "run", heartbeat_seconds=0.02,
                      terminate_grace=0.2, **kwargs), steps, identity()
    )


def command(name, code, timeout=2):
    return module.Command(name, (sys.executable, "-c", code), timeout)


def test_fail_soft_blocks_only_dependents_and_always_reports(tmp_path):
    marker = tmp_path / "survived"
    steps = [
        module.Step("bad", "test", "bad", (command("bad", "raise SystemExit(7)"),)),
        module.Step("blocked", "test", "blocked", dependencies=("bad",)),
        module.Step("independent", "test", "independent",
                    (command("write", f"open({str(marker)!r}, 'w').write('yes')"),)),
        module.Step("report", "report", "report", always_run=True),
    ]
    run = campaign(tmp_path, steps)
    run.run(resume=False)
    state = json.loads(run.state_path.read_text())
    assert state["steps"]["bad"]["status"] == "failed"
    assert state["steps"]["blocked"]["status"] == "blocked"
    assert state["steps"]["independent"]["status"] == "passed"
    assert state["steps"]["report"]["status"] == "passed"
    assert marker.read_text() == "yes"
    assert (run.config.run_dir / "summary.json").exists()


def test_resume_does_not_repeat_a_receipted_pass(tmp_path):
    counter = tmp_path / "counter"
    code = f"p={str(counter)!r}; open(p,'a').write('x')"
    steps = [module.Step("once", "test", "once", (command("once", code),))]
    first = campaign(tmp_path, steps)
    first.run(resume=False)
    second = campaign(tmp_path, steps)
    second.run(resume=True)
    assert counter.read_text() == "x"
    state = json.loads(second.state_path.read_text())
    assert len(state["steps"]["once"]["attempts"]) == 1


def test_resume_reruns_when_pass_receipt_was_tampered(tmp_path):
    counter = tmp_path / "counter"
    steps = [module.Step("once", "test", "once",
                         (command("once", f"open({str(counter)!r},'a').write('x')"),))]
    first = campaign(tmp_path, steps); first.run(resume=False)
    state = json.loads(first.state_path.read_text())
    Path(state["steps"]["once"]["attempts"][-1]["receipt"]).write_text("{}")
    second = campaign(tmp_path, steps); second.run(resume=True)
    assert counter.read_text() == "xx"
    assert len(json.loads(second.state_path.read_text())["steps"]["once"]["attempts"]) == 2


def test_timeout_kills_the_command_process_group(tmp_path):
    pid_file = tmp_path / "pid"
    code = f"import os,time; open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)"
    steps = [module.Step("slow", "test", "slow", (command("slow", code, timeout=0.15),)),
             module.Step("report", "report", "report", always_run=True)]
    run = campaign(tmp_path, steps)
    run.run(resume=False)
    state = json.loads(run.state_path.read_text())
    assert state["steps"]["slow"]["status"] == "timed_out"
    pid = int(pid_file.read_text())
    with __import__("pytest").raises(ProcessLookupError):
        os.kill(pid, 0)
    assert any(json.loads(line)["event"] == "health_snapshot"
               for line in run.events_path.read_text().splitlines())


def test_server_is_stopped_after_success(tmp_path):
    port_file = tmp_path / "port"
    server_code = """
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body=json.dumps({'healthy': True, 'inflight': 0, 'queue_depth': 0,
                         'memory_waiting': 0, 'apcv2': {'cow': {'active_leases': 0}},
                         'runtime': 'r', 'artifact': 'a', 'settings': {}, 'profile': 'p'}).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)
s=HTTPServer(('127.0.0.1',0),H)
open(os.environ['PORT_FILE'],'w').write(str(s.server_port))
s.serve_forever()
"""
    # Use a fixed free port because Server.url is immutable.
    import socket
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    server_code = server_code.replace("('127.0.0.1',0)", f"('127.0.0.1',{port})")
    wrapper = tmp_path / "server.py"; wrapper.write_text(server_code)
    steps = [module.Step("server", "test", "server", (command("client", "print('ok')"),),
                         server=module.Server((sys.executable, str(wrapper)), f"http://127.0.0.1:{port}", ready_timeout=2, drain_timeout=0.2))]
    old = os.environ.get("PORT_FILE"); os.environ["PORT_FILE"] = str(port_file)
    try:
        run = campaign(tmp_path, steps); run.run(resume=False)
    finally:
        if old is None: os.environ.pop("PORT_FILE", None)
        else: os.environ["PORT_FILE"] = old
    receipt = json.loads((run.receipts_dir / "server.json").read_text())
    cleanup = next(row for row in receipt["commands"] if row["name"] == "server_cleanup")
    assert cleanup["stopped"] is True
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_prepare_writes_exact_policies_and_refuses_drift(tmp_path):
    run = campaign(tmp_path, [])
    run.prepare()
    policies = run.config.root / "qualification/policies"
    assert json.loads((policies / "qwen36-ordinary.json").read_text()) == {}
    assert json.loads((policies / "qwen36-mtp-artifact-ordinary.json").read_text()) == {}
    assert json.loads((policies / "qwen36-mtp2.json").read_text()) == {"num_draft": 2}
    (policies / "qwen36-mtp2.json").write_text('{"num_draft": 3}')
    other = module.Campaign(run.config, [], identity())
    with __import__("pytest").raises(ValueError, match="refusing to overwrite"):
        other.prepare()


def test_static_inputs_are_created_before_identity_freeze(tmp_path):
    root = tmp_path / "fresh"
    (root / "src/mlx2").mkdir(parents=True)
    module.ensure_qwen36_policies(root)
    first = module.campaign_identity(root, [], {"policy": "frozen"})
    run = module.Campaign(module.Config(root=root, run_dir=tmp_path / "state"), [], first)
    run.prepare()
    module.ensure_qwen36_policies(root)
    second = module.campaign_identity(root, [], {"policy": "frozen"})
    assert first == second
    module.Campaign(module.Config(root=root, run_dir=tmp_path / "state"), [], second).load()


def test_campaign_identity_ignores_outputs_and_os_metadata_but_guards_inputs(tmp_path):
    root = tmp_path / "repo"
    (root / "src/mlx2").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "qualification/policies").mkdir(parents=True)
    (root / "src/mlx2/runtime.py").write_text("VALUE = 1\n")
    (root / "scripts/run_qwen36_overnight.py").write_text("print('runner')\n")
    (root / "tests/test_runtime.py").write_text("def test_runtime(): pass\n")
    (root / "qualification/policies/qwen36-ordinary.json").write_text("{}\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "qwen"}\n')
    (model / "weights.safetensors").write_bytes(b"weights")
    settings = {"manifest_sha256": "manifest-v1"}
    baseline = module.campaign_identity(root, [model], settings)["sha256"]

    # Campaign state and macOS metadata are observations/outputs, not inputs.
    (root / "qualification/runs/campaign").mkdir(parents=True)
    (root / "qualification/runs/.DS_Store").write_bytes(b"finder")
    (root / "qualification/runs/campaign/state.json").write_text("{}\n")
    (root / "src/mlx2/.DS_Store").write_bytes(b"finder")
    (model / ".DS_Store").write_bytes(b"finder")
    assert module.campaign_identity(root, [model], settings)["sha256"] == baseline

    guarded = (
        root / "src/mlx2/runtime.py",
        root / "scripts/run_qwen36_overnight.py",
        root / "tests/test_runtime.py",
        root / "qualification/policies/qwen36-ordinary.json",
    )
    for path in guarded:
        original = path.read_bytes()
        path.write_bytes(original + b"# changed\n")
        assert module.campaign_identity(root, [model], settings)["sha256"] != baseline
        path.write_bytes(original)

    assert module.campaign_identity(
        root, [model], {"manifest_sha256": "manifest-v2"}
    )["sha256"] != baseline
    (model / "weights.safetensors").write_bytes(b"changed weights")
    assert module.campaign_identity(root, [model], settings)["sha256"] != baseline


def test_two_non_nominal_thermal_samples_latch_global_stop(tmp_path):
    steps = [module.Step("long", "test", "long",
                         (command("long", "import time; time.sleep(5)", timeout=6),)),
             module.Step("after", "test", "after", (command("no", "raise SystemExit(9)"),)),
             module.Step("report", "report", "report", always_run=True)]
    run = campaign(tmp_path, steps)
    run.sample_thermal = lambda: {"thermal_state": 1, "available": True}
    run.run(resume=False)
    state = json.loads(run.state_path.read_text())
    assert state["steps"]["long"]["status"] == "cancelled"
    assert state["steps"]["after"]["status"] == "cancelled"
    assert state["steps"]["report"]["status"] == "passed"
    events = [json.loads(line) for line in run.events_path.read_text().splitlines()]
    assert any(row["event"] == "global_stop_latched" for row in events)


def test_every_gpu_step_is_transitively_gated_by_complete_preflight(tmp_path):
    root = Path(__file__).parents[1]
    steps = module.build_plan(root, tmp_path / "run", Path("/ordinary"), Path("/mtp"),
                              ".venv/bin/python", root / "qualification/missing.json")
    by_id = {step.step_id: step for step in steps}
    required = {"preflight-full", "preflight-focused", "preflight-diff", "preflight-build",
                "preflight-json", "preflight-ports"}
    assert set(by_id["preflight-ready"].dependencies) == required

    def ancestors(step_id):
        found = set()
        for dependency in by_id[step_id].dependencies:
            found.add(dependency); found.update(ancestors(dependency))
        return found

    for step in steps:
        if step.gpu:
            assert "preflight-ready" in ancestors(step.step_id)
    preflight = by_id["preflight-full"].commands[0].argv
    assert "--preflight-only" in preflight
    for step_id in ("ordinary-serving", "mtp-artifact-ordinary-serving", "mtp2-serving"):
        assert "--preflight-receipt" in by_id[step_id].commands[0].argv
        qualifier = by_id[step_id].commands[0].argv
        assert qualifier[qualifier.index("--defer-long-context-to-matrix") + 1] == str(
            root / "qualification/missing.json"
        )
        assert by_id[step_id].commands[1].name == "http-qualification"
        assert "--resume" in by_id[step_id].commands[1].argv
    for step_id in ("context-ladder", "batch-20x20"):
        assert "--resume" in by_id[step_id].commands[0].argv
        assert "--continue-on-error" in by_id[step_id].commands[0].argv
    for batch_id, context_id in (
        ("ordinary-b20-serving", "ordinary-serving"),
        ("mtp-artifact-ordinary-b20-serving", "mtp-artifact-ordinary-serving"),
        ("mtp2-b20-serving", "mtp2-serving"),
    ):
        assert context_id in by_id[batch_id].dependencies
        argv = by_id[batch_id].server.argv
        assert argv[argv.index("--max-lanes") + 1] == "20"
        assert argv[argv.index("--max-inflight") + 1] == "40"
    order = [step.step_id for step in steps]
    assert order.index("mtp2-serving") < order.index("context-ladder")
    assert order.index("context-ladder") < order.index("ordinary-b20-serving")
    assert order.index("mtp2-b20-serving") < order.index("batch-20x20")


def test_generated_manifest_template_separates_context_and_batch_identities(tmp_path):
    root = Path(__file__).parents[1]
    value = module.qwen36_manifest_template(root, tmp_path / "run", Path("/ordinary"),
                                            Path("/mtp"), ".venv/bin/python")
    model = value["models"][0]
    assert [row["tokens"] for row in model["contexts"]] == [32768, 65536, 131072, 262016]
    assert model["context"]["runs_per_cell"] == 3
    assert model["batch_stress"]["rounds"] == 20
    assert model["batch_stress"]["width"] == 20
    for arm in model["arms"]:
        context_command = arm["activate_command"]
        batch = arm["suites"]["batch_stress"]
        batch_command = batch["activate_command"]
        assert context_command[context_command.index("--max-lanes") + 1] == "4"
        assert batch_command[batch_command.index("--max-lanes") + 1] == "20"
        assert context_command[context_command.index("--max-inflight") + 1] == "8"
        assert batch_command[batch_command.index("--max-inflight") + 1] == "40"
        assert arm["qualification_receipt"]["path"] != batch["qualification_receipt"]["path"]
        assert "long_context_delegation" in arm["qualification_receipt"]["required_checks"]
        assert "context" not in arm["qualification_receipt"]["required_checks"]
        assert {row["path"] for row in batch["status_requirements"]} >= {
            "max_lanes", "settings.max_inflight"
        }


def test_matrix_filters_to_serving_steps_that_fully_passed(tmp_path):
    root = Path(__file__).parents[1]
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(module.qwen36_manifest_template(
        root, tmp_path / "run", Path("/ordinary"), Path("/mtp"), ".venv/bin/python")))
    steps = module.build_plan(root, tmp_path / "run", Path("/ordinary"), Path("/mtp"),
                              ".venv/bin/python", manifest)
    run = module.Campaign(module.Config(root=root, run_dir=tmp_path / "run"), steps, identity())
    run.prepare()
    run.state["steps"]["ordinary-serving"]["status"] = "passed"
    run.state["steps"]["mtp-artifact-ordinary-serving"]["status"] = "failed"
    run.state["steps"]["mtp2-serving"]["status"] = "passed"
    step = next(item for item in steps if item.step_id == "context-ladder")
    effective = run.effective_command(step, step.commands[0])
    filtered = json.loads(Path(effective.argv[effective.argv.index("--manifest") + 1]).read_text())
    assert [arm["name"] for arm in filtered["models"][0]["arms"]] == ["ordinary", "mtp2"]


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_explicit_stop_halts_controller_then_resume_continues(tmp_path):
    first = tmp_path / "first"
    allow = tmp_path / "allow"
    second = tmp_path / "second"
    third = tmp_path / "third"
    wait_code = (
        "import pathlib,time; p=pathlib.Path(" + repr(str(allow)) + "); "
        "[(time.sleep(.05)) for _ in range(400) if not p.exists()]; "
        "pathlib.Path(" + repr(str(second)) + ").write_text('done')"
    )
    steps = [
        module.Step("first", "t", "first", (command("first", f"open({str(first)!r},'a').write('x')"),)),
        module.Step("middle", "t", "middle", (command("middle", wait_code, timeout=30),)),
        module.Step("last", "t", "last", (command("last", f"open({str(third)!r},'w').write('done')"),)),
    ]
    run = campaign(tmp_path, steps)
    child = os.fork()
    if child == 0:
        try:
            run.run(resume=False)
        finally:
            os._exit(0)
    try:
        wait_for(lambda: run.state_path.exists() and
                 json.loads(run.state_path.read_text())["steps"]["middle"]["status"] == "running")
        stopper = campaign(tmp_path, steps)
        result = stopper.explicit_stop()
        assert result["stopped"] is True
        os.waitpid(child, 0)
        assert first.read_text() == "x"
        assert not third.exists(), "stop request allowed a later independent step to start"
        allow.write_text("go")
        resumed = campaign(tmp_path, steps)
        resumed.run(resume=True)
        state = json.loads(resumed.state_path.read_text())
        assert first.read_text() == "x", "passed step reran"
        assert second.read_text() == "done" and third.read_text() == "done"
        assert len(state["steps"]["middle"]["attempts"]) == 2
        assert state["steps"]["middle"]["attempts"][0]["status"] == "interrupted"
    finally:
        try: os.kill(child, signal.SIGKILL)
        except ProcessLookupError: pass
        try: os.waitpid(child, os.WNOHANG)
        except ChildProcessError: pass


def test_explicit_stop_marks_pre_registration_race_interrupted(tmp_path, monkeypatch):
    steps = [module.Step("work", "t", "work", (command("work", "pass"),))]
    run = campaign(tmp_path, steps)
    run.prepare()
    row = run.state["steps"]["work"]
    row.update({"status": "running", "started_at": time.time()})
    row["attempts"].append({"attempt": 1, "started_at": time.time(), "log_dir": "race"})
    controller = {
        "pid": 424242,
        "pgid": 424242,
        "process_start_identity": "start",
        "process_command": "controller",
        "registered_at": time.time(),
    }
    run.state["controller"] = controller
    run.save()

    stopper = campaign(tmp_path, steps)
    identities = iter([
        {"pid": 424242, "pgid": 424242, "start_identity": "start",
         "command": "controller"},
        None,
    ])
    monkeypatch.setattr(stopper, "process_identity", lambda _pid: next(identities, None))

    def controller_cancels_before_exit(pid, sig):
        assert (pid, sig) == (424242, signal.SIGTERM)
        state = json.loads(stopper.state_path.read_text())
        state["steps"]["work"].update({"status": "cancelled", "reason": "command cancelled"})
        state["steps"]["work"]["attempts"][-1].update(
            {"status": "cancelled", "reason": "command cancelled"}
        )
        stopper.state_path.write_text(json.dumps(state))

    monkeypatch.setattr(os, "kill", controller_cancels_before_exit)
    result = stopper.explicit_stop()

    assert result["stopped"] is True
    state = json.loads(stopper.state_path.read_text())
    assert state["steps"]["work"]["status"] == "interrupted"
    assert state["steps"]["work"]["attempts"][0]["status"] == "interrupted"


def test_resume_reconciles_exact_process_left_by_crashed_controller(tmp_path):
    marker = tmp_path / "recovered"
    steps = [module.Step("work", "t", "work",
                         (command("work", f"open({str(marker)!r},'w').write('ok')"),))]
    run = campaign(tmp_path, steps); run.prepare()
    row = run.state["steps"]["work"]
    row.update({"status": "running", "started_at": time.time()})
    row["attempts"].append({"attempt": 1, "started_at": time.time(), "log_dir": "crashed"})
    process = __import__("subprocess").Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                             start_new_session=True)
    run.register_process(steps[0], 1, process, (sys.executable, "-c", "import time; time.sleep(30)"),
                         "command:work")
    run.save()
    resumed = campaign(tmp_path, steps); resumed.run(resume=True)
    process.wait(timeout=2)
    assert process.returncode != 0
    state = json.loads(resumed.state_path.read_text())
    assert marker.read_text() == "ok"
    assert len(state["steps"]["work"]["attempts"]) == 2
    assert state["steps"]["work"]["attempts"][0]["status"] == "interrupted"


def test_pid_reuse_or_identity_mismatch_is_never_killed(tmp_path):
    steps = [module.Step("work", "t", "work")]
    run = campaign(tmp_path, steps); run.prepare()
    row = run.state["steps"]["work"]
    row.update({"status": "running", "started_at": time.time()})
    row["attempts"].append({"attempt": 1, "started_at": time.time(), "log_dir": "crashed"})
    process = __import__("subprocess").Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                             start_new_session=True)
    run.register_process(steps[0], 1, process, (sys.executable, "-c", "import time; time.sleep(30)"),
                         "command:work")
    record = next(iter(run.state["active_processes"].values()))
    record["process_start_identity"] = "definitely-not-the-same-process"
    run.save()
    try:
        result = campaign(tmp_path, steps).explicit_stop()
        assert result["stopped"] is False
        assert result["refused_steps"] == ["work"]
        os.kill(process.pid, 0)  # still alive: controller refused the ambiguous kill
        state = json.loads(run.state_path.read_text())
        assert state["steps"]["work"]["status"] == "blocked"
    finally:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def test_stop_cleans_proven_process_even_when_source_identity_drifted(tmp_path):
    steps = [module.Step("work", "t", "work")]
    run = campaign(tmp_path, steps); run.prepare()
    run.state["steps"]["work"].update({"status": "running", "attempts": [
        {"attempt": 1, "started_at": time.time(), "log_dir": "crashed"}
    ]})
    process = __import__("subprocess").Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                             start_new_session=True)
    run.register_process(steps[0], 1, process, (sys.executable, "-c", "import time; time.sleep(30)"),
                         "command:work")
    run.save()
    drifted = identity(); drifted["sha256"] = "different-source"
    stopper = module.Campaign(run.config, steps, drifted)
    result = stopper.explicit_stop()
    process.wait(timeout=2)
    assert result["stopped"] is True
    assert result["identity_drift"] is True
    assert process.returncode != 0
