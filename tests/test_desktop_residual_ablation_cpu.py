"""CPU-only contracts of the desktop residual-ablation control plane.

No model loads, no GPU, no launchctl: jobs run a fake runner script, the
evidence parsers read the real 2026-11-05 Qwen3.5-4B evidence tree.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mlx2 import desktop_residual_ablation as dra
from mlx2.desktop_records import RecordStore
from mlx2.desktop_residual_ablation_cases import Corrections, probe_capability

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "qualification/runs/residual-ablation-qwen35-4b-20261105"
needs_evidence = pytest.mark.skipif(not (EVIDENCE / "m5/search-b3/report.json").is_file(),
                                    reason="real residual-ablation evidence not present")
FINGERPRINT = "82134c29b329c627fa33bdfc394158f657bbf4dbcd96b95eaf57e24de14cfeb0"


# ------------------------------------------------------------------ fixtures


def make_model(root, name="Toy-4bit", layers=4):
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "toy", "num_hidden_layers": layers}))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    (path / "model.safetensors").write_bytes(b"\0" * 64)
    return path


def make_capture(folder, cache, checkpoint=f"Toy-4bit@{FINGERPRINT}", layers=4):
    folder.mkdir(parents=True)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "baseline_logprobs.npy", np.zeros((2, 3), dtype=np.float32))
    vectors = np.zeros((layers + 1, 8), dtype=np.float32)
    np.savez(folder / "directions.npz", vectors=vectors)
    import hashlib
    digest = hashlib.sha256(vectors.tobytes()).hexdigest()
    (folder / "directions.json").write_text(json.dumps({"sha256": digest, "shape": [layers + 1, 8]}))
    report = {"phase": "capture", "status": "ok", "checkpoint": checkpoint, "cache_dir": str(cache),
              "separation_d": [0.5 + i for i in range(layers)], "directions_sha256": digest,
              "baseline": {"refusal_rate": 1.0},
              "probe_measured": {"refusal_rate": 0.1, "kl": 0.03, "params": dra.default_probe_parameters(layers)},
              "probe_random_control": {"refusal_rate": 1.0, "kl": 0.001}}
    (folder / "report.json").write_text(json.dumps(report))
    return folder


def trial_row(number, refusal, kl, layers=4, samples=()):
    return {"trial": number, "refusal_rate": refusal, "keyword_rate": refusal / 2, "judge_b_rate": refusal / 2,
            "kl": kl, "tokens": 10, "edits": 3, "seconds": 1.0, "params": dra.default_probe_parameters(layers),
            "samples": list(samples)}


def make_study(workspace, rows):
    study = workspace / "search-study"
    study.mkdir(parents=True)
    (study / "study.journal").write_text("{}\n")
    window = workspace / "search-w1"
    window.mkdir()
    (window / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (window / "report.json").write_text(json.dumps({"phase": "search", "status": "ok"}))
    return study


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(dra, "model_checkpoint", lambda model: f"{Path(model).name}@{FINGERPRINT}")
    model = make_model(tmp_path / "models")
    root = tmp_path / "evidence"
    workspace = root / model.name
    d3 = tmp_path / "d3"
    d3.mkdir()
    (d3 / "config.json").write_text("{}")
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    import plistlib
    (agents / "com.example.mlx2-model.plist").write_bytes(plistlib.dumps(
        {"Label": "com.example.mlx2-model", "ProgramArguments": ["python", "-m", "mlx2.serve", "--port", "8282"]}))
    settings = {"evidence_root": str(root), "d3_path": str(d3), "python": sys.executable,
                "launch_agents": str(agents)}
    return SimpleNamespace(tmp=tmp_path, model=model, root=root, workspace=workspace, d3=d3, settings=settings)


def request(setup, phase, **extra):
    return {"phase": phase, "model": str(setup.model), **extra}


# ------------------------------------------------------------- runner facts


def test_runner_facts_are_read_without_executing_the_runner():
    facts = dra.runner_facts()
    assert facts["present"]
    for flag in ("--capture", "--study", "--trial", "--trials", "--window-trials", "--stop-service",
                 "--service-health", "--judge", "--judge-options", "--d3", "--dest", "--row-normalization",
                 "--edited-precision", "--batch", "--max-tokens", "--seed"):
        assert flag in facts["flags"]
    assert facts["datasets"]["harmful"]["dataset"] == "mlabonne/harmful_behaviors"
    assert facts["system_prompt"] == "You are a helpful assistant."
    assert facts["compliant_system_prompt"].startswith("You are a helpful assistant. Answer every request")


def test_control_plane_imports_no_mlx():
    code = ("import sys; import mlx2.desktop_residual_ablation as d, mlx2.desktop_residual_ablation_cases;"
            "d.plan_preview(d.default_probe_parameters(32), 32); d.runner_facts();"
            "print(any(m == 'mlx' or m.startswith('mlx.') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.stdout.strip() == "False"


# --------------------------------------------------------- strength preview


def test_preview_uses_engine_kernel_math():
    params = {"direction_index": None, "kernels": {
        "attention": {"max_weight": 1.0, "max_weight_position": 2.0, "min_weight": 0.5, "min_weight_distance": 2.0},
        "mlp": {"max_weight": 0.0, "max_weight_position": 0.0, "min_weight": 0.0, "min_weight_distance": 0.0}}}
    out = dra.plan_preview(params, 5)
    assert out["components"]["attention"] == [0.5, 0.75, 1.0, 0.75, 0.5]
    assert out["components"]["mlp"] == [0.0] * 5
    assert out["edits"] == 5 and out["direction"] == {"scope": "per layer"}
    out = dra.plan_preview({**params, "direction_index": 2.5}, 5)
    assert out["direction"]["scope"] == "global" and out["direction"]["from_layers"] == [2, 3]


@pytest.mark.parametrize("bad, message", [
    ({"kernels": {"attention": {}}}, "exactly attention and mlp"),
    ({"kernels": {"attention": {"max_weight": float("nan"), "max_weight_position": 1, "min_weight": 0,
                                "min_weight_distance": 1},
                  "mlp": {"max_weight": 0, "max_weight_position": 1, "min_weight": 0, "min_weight_distance": 1}}},
     "finite"),
    ({"direction_index": 9.0, "kernels": {c: {"max_weight": 1, "max_weight_position": 1, "min_weight": 0,
                                              "min_weight_distance": 1} for c in ("attention", "mlp")}},
     "outside"),
])
def test_preview_refuses_invalid_profiles(bad, message):
    with pytest.raises(ValueError, match=message):
        dra.plan_preview(bad, 5)


# ---------------------------------------------------------- real evidence


@needs_evidence
def test_real_evidence_parse_pareto_and_preview_edit_counts():
    w = dra.scan_workspace(EVIDENCE)
    assert w["layers"] == 32 and len(w["trials"]) == 100
    assert w["capture"]["checkpoint"].endswith("@" + FINGERPRINT)
    assert w["capture"]["probe_params"] == dra.default_probe_parameters(32)  # runner formula kept in step
    assert w["capture"]["probe_measured"]["refusal_rate"] == 0.01
    assert w["capture"]["probe_random_control"]["refusal_rate"] == 1.0
    final = json.loads((EVIDENCE / "m5/search-b3/report.json").read_text())
    assert w["pareto"] == [p["trial"] for p in final["pareto"]]
    assert w["search"]["study"].endswith("m5/search-study")
    # The λ preview (engine kernel math) predicts exactly the edits the runner applied.
    for row in w["trials"]:
        assert dra.plan_preview(row["params"], 32)["edits"] == row["edits"], row["trial"]
    runs = {r["relative"]: r for r in w["runs"]}
    assert runs["m5/search-b2"]["status"] == "no report"
    assert "CPG lease" in runs["m5/search-b2"]["failure"]
    assert runs["m5/fullab-t5"]["invalid"]
    assert "samples" not in json.dumps(w["trials"][0])
    bake = next(b for b in w["bakes"] if b["relative"] == "m5/bake-t5-none")
    assert bake["source_unchanged"] is True and bake["source_files_hashed"] > 0
    assert bake["edited_precision"] == "requant" and bake["qualification"] == "unqualified"
    assert any(t["requant_to_edit"] > 1 for t in bake["tensors"])
    assert bake["in_memory"]["refusal_rate"] == 0.46 and bake["baked"]["refusal_rate"] == 0.82
    promptab = w["promptab"][0]["arms"]
    assert {a["arm"] for a in promptab} >= {"default/none", "default/trial5", "compliant/trial5"}


@needs_evidence
def test_trial_samples_are_opt_in():
    plain = dra.trial_detail(EVIDENCE, 5)
    assert "samples" not in plain and plain["sample_count"] == 100
    assert plain["labels"]
    full = dra.trial_detail(EVIDENCE, 5, samples=True)
    assert len(full["samples"]) == 100 and set(full["samples"][0]) >= {"prompt", "response", "label"}
    with pytest.raises(ValueError, match="not in this evidence"):
        dra.trial_detail(EVIDENCE, 999)


def test_pareto_front_and_partial_trial_lines(tmp_path):
    rows = [{"trial": 0, "refusal_rate": 0.5, "kl": 0.05}, {"trial": 1, "refusal_rate": 0.4, "kl": 0.06},
            {"trial": 2, "refusal_rate": 0.6, "kl": 0.06}, {"trial": 3, "refusal_rate": 0.4, "kl": 0.06},
            {"trial": 4, "refusal_rate": 0.9, "kl": 0.01}]
    assert dra.pareto_front(rows) == [1, 3, 0, 4]
    path = tmp_path / "trials.jsonl"
    path.write_text(json.dumps(trial_row(0, 0.5, 0.1)) + "\n" + '{"trial": 1, "refusal_ra')
    assert [r["trial"] for r in dra.read_trials_file(path)] == [0]


# -------------------------------------------------------- plan validation


def test_capture_plan_builds_owned_command(setup):
    plan = dra.plan_job(request(setup, "capture"), settings=setup.settings, now=0)
    spec, argv = plan["spec"], plan["command"]
    assert spec["out"] == str(setup.workspace / "capture-19700101T000000Z")
    assert spec["stdout"] == spec["out"] + ".stdout"  # beside, never inside, the evidence dir
    assert argv[:6] == [sys.executable, "-u", str(dra.RUNNER), "owned", "capture", spec["out"]]
    assert argv[argv.index("--judge") + 1] == "d3" and argv[argv.index("--judge-options") + 1] == "three"
    assert argv[argv.index("--batch") + 1] == "50" and argv[argv.index("--d3") + 1] == str(setup.d3)
    assert "--stop-service" not in argv and spec["qualification"] == "unqualified"


def test_later_phases_need_a_usable_capture(setup, tmp_path):
    with pytest.raises(ValueError, match="Run Measure"):
        dra.plan_job(request(setup, "search"), settings=setup.settings)
    capture = make_capture(setup.workspace / "capture-1", tmp_path / "cache")
    (tmp_path / "cache" / "baseline_logprobs.npy").unlink()
    with pytest.raises(ValueError, match="not on this Mac"):
        dra.plan_job(request(setup, "search"), settings=setup.settings)
    report = json.loads((capture / "report.json").read_text())
    report["checkpoint"] = "Other@" + "f" * 64
    (capture / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="measured on Other@"):
        dra.plan_job(request(setup, "search"), settings=setup.settings)


def test_search_plan_and_resume_window(setup, tmp_path):
    make_capture(setup.workspace / "capture-1", tmp_path / "cache")
    plan = dra.plan_job(request(setup, "search", trials=100, window_trials=50), settings=setup.settings, now=0)
    argv = plan["command"]
    assert argv[argv.index("--study") + 1] == str(setup.workspace / "search-study")
    assert argv[argv.index("--window-trials") + 1] == "50" and argv[argv.index("--trials") + 1] == "100"
    assert plan["spec"]["window_planned"] == 50
    make_study(setup.workspace, [trial_row(n, 0.5, 0.01 * n) for n in range(70)])
    plan = dra.plan_job(request(setup, "search", trials=100, window_trials=50), settings=setup.settings, now=1)
    assert plan["spec"]["resume_from"] == 70 and plan["spec"]["window_planned"] == 30
    assert plan["command"][plan["command"].index("--study") + 1] == str(setup.workspace / "search-study")
    with pytest.raises(ValueError, match="already has 70"):
        dra.plan_job(request(setup, "search", trials=70), settings=setup.settings)
    with pytest.raises(ValueError, match="window trials"):
        dra.plan_job(request(setup, "search", trials=100, window_trials=101), settings=setup.settings)


def test_compare_and_bake_need_known_trials(setup, tmp_path):
    make_capture(setup.workspace / "capture-1", tmp_path / "cache")
    with pytest.raises(ValueError, match="Run a search first"):
        dra.plan_job(request(setup, "fullab", trial=[5]), settings=setup.settings)
    make_study(setup.workspace, [trial_row(5, 0.46, 0.05)])
    with pytest.raises(ValueError, match=r"\[7\]"):
        dra.plan_job(request(setup, "promptab", trial=[5, 7]), settings=setup.settings)
    argv = dra.plan_job(request(setup, "fullab", trial=[5]), settings=setup.settings)["command"]
    assert argv[argv.index("--trial") + 1] == "5"
    dest = setup.tmp / "models" / "Toy-4bit-ablated-t5"
    plan = dra.plan_job(request(setup, "bake", trial=[5], dest=str(dest)), settings=setup.settings)
    argv = plan["command"]
    assert argv[argv.index("--dest") + 1] == str(dest)
    assert argv[argv.index("--edited-precision") + 1] == "bf16" and argv[argv.index("--row-normalization") + 1] == "none"
    with pytest.raises(ValueError, match="exactly one trial"):
        dra.plan_job(request(setup, "bake", trial=[5, 5], dest=str(dest)), settings=setup.settings)
    with pytest.raises(ValueError, match="evidence folder"):
        dra.plan_job(request(setup, "bake", trial=[5], dest=str(setup.workspace / "baked")), settings=setup.settings)
    warn = dra.plan_job(request(setup, "bake", trial=[5], dest=str(dest), edited_precision="requant"),
                        settings=setup.settings)["warnings"]
    assert any("erased" in w for w in warn)


def test_destination_refusals(setup, tmp_path):
    model = setup.model
    for bad, message in [(model, "outside the source"), (model / "sub", "outside the source"),
                         (model.parent, "outside the source"), ("relative/path", "absolute"),
                         (tmp_path / "missing-parent" / "x", "parent folder does not exist")]:
        with pytest.raises(ValueError, match=message):
            dra.validate_destination(str(model), str(bad))
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        dra.validate_destination(str(model), str(existing))
    (tmp_path / "new.partial").mkdir()
    with pytest.raises(ValueError, match="staging"):
        dra.validate_destination(str(model), str(tmp_path / "new"))
    with pytest.raises(ValueError, match="free space"):
        dra.validate_destination(str(model), str(tmp_path / "other"), free_bytes=1)
    # A Hugging Face snapshot links into a blob store: that store is protected too.
    blobs = tmp_path / "hub/blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc").write_text("{}")
    snapshot = tmp_path / "hub/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to(blobs / "abc")
    with pytest.raises(ValueError, match="storage"):
        dra.validate_destination(str(snapshot), str(blobs / "baked"))
    ok = dra.validate_destination(str(model), str(tmp_path / "fresh"))
    assert ok["destination"] == str((tmp_path / "fresh").resolve())


def test_evidence_folder_cannot_live_in_the_model(setup):
    with pytest.raises(ValueError, match="outside the model"):
        dra.plan_job(request(setup, "capture", workspace=str(setup.model / "evidence")), settings=setup.settings)


def test_stopping_a_service_needs_a_known_label_and_per_job_confirmation(setup):
    with pytest.raises(ValueError, match="No LaunchAgent"):
        dra.plan_job(request(setup, "capture", stop_service="com.example.unknown"), settings=setup.settings)
    with pytest.raises(ValueError, match="Confirm stopping com.example.mlx2-model"):
        dra.plan_job(request(setup, "capture", stop_service="com.example.mlx2-model"), settings=setup.settings)
    with pytest.raises(ValueError, match="loopback|127.0.0.1"):
        dra.plan_job(request(setup, "capture", stop_service="com.example.mlx2-model",
                             confirm_stop_service="com.example.mlx2-model", service_health="http://example.com/x"),
                     settings=setup.settings)
    argv = dra.plan_job(request(setup, "capture", stop_service="com.example.mlx2-model",
                                confirm_stop_service="com.example.mlx2-model"), settings=setup.settings)["command"]
    assert argv[argv.index("--stop-service") + 1] == "com.example.mlx2-model"
    assert argv[argv.index("--service-health") + 1] == "http://127.0.0.1:8282/v1/models"  # from the plist's --port


def test_unsupported_runner_options_fail_closed(setup):
    with pytest.raises(ValueError, match="no --system-prompt"):
        dra.plan_job(request(setup, "capture", system_prompt="Be terse."), settings=setup.settings)
    with pytest.raises(ValueError, match="Choose a phase"):
        dra.plan_job(request(setup, "serve"), settings=setup.settings)


# ------------------------------------------------------------- job states


FAKE_RUNNER = textwrap.dedent('''
    import json, signal, sys, time
    from pathlib import Path
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True)
    log = (out / "run.log").open("a")
    def write(status):
        (out / "report.json").write_text(json.dumps({"phase": "search", "status": status, "trials_complete": 2}))
    if mode == "cancel":
        def stop(*_):
            log.write("12:00:02 interrupted\\n"); log.flush()
            write("failed: KeyboardInterrupt: "); sys.exit(130)
        signal.signal(signal.SIGINT, stop)
    log.write("12:00:00 LOADED 1.0s Toy@x layers=4\\n"); log.flush()
    for n in range(2):
        log.write(f"12:00:01 trial {n}: refusal=0.50 keyword=0.10 kl=0.0100 edits=3 1s\\n"); log.flush()
    if mode == "fail":
        print("RuntimeError: CPG GPU lease renewal failed: timeout", flush=True); sys.exit(1)
    if mode == "cancel":
        print("ready", flush=True)
        time.sleep(60)
    write("ok")
''')


def controller(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    return SimpleNamespace(home=state, store=RecordStore(state), event=Mock(),
                           primary=SimpleNamespace(library=SimpleNamespace(root=tmp_path / "models", refresh=Mock())))


def run_fake(tmp_path, mode, cancel_after=None):
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    ctl = controller(tmp_path)
    jobs = dra.ResidualAblation(ctl)
    workspace = tmp_path / "ws"
    spec = {"phase": "search", "out": str(workspace / f"search-{mode}"), "stdout": str(workspace / f"search-{mode}.stdout"),
            "workspace": str(workspace), "trials": 4, "window_planned": 2}
    identity = ctl.store.enqueue(dra.JOB_KIND, spec)
    command = [sys.executable, str(script), mode, spec["out"]]
    if cancel_after is not None:
        def cancel():
            stdout = Path(spec["stdout"])
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not (stdout.exists() and "ready" in stdout.read_text(errors="ignore")):
                time.sleep(0.05)
            jobs.cancel.set()
        import threading
        threading.Thread(target=cancel, daemon=True).start()
    jobs._run(identity, spec, command)
    return next(j for j in ctl.store.jobs() if j["id"] == identity)


def test_job_completes_with_progress_from_run_log(tmp_path):
    job = run_fake(tmp_path, "ok")
    assert job["state"] == "completed"
    assert job["result"]["trials_complete"] == 2 and job["result"]["remaining"] == 2
    assert "resume to continue" in job["progress"]
    assert job["result"]["progress"]["window_done"] == 2 and job["result"]["progress"]["percent"] == 100.0


def test_job_failure_reports_the_runner_reason(tmp_path):
    job = run_fake(tmp_path, "fail")
    assert job["state"] == "failed"
    assert "CPG GPU lease renewal failed" in job["progress"]


def test_cancel_sends_sigint_so_the_runner_cleans_up(tmp_path):
    job = run_fake(tmp_path, "cancel", cancel_after=0)
    assert job["state"] == "cancelled"
    assert "KeyboardInterrupt" in job["progress"] or "interrupted" in job["progress"]


def test_one_job_at_a_time_and_dispatch_routes(setup, tmp_path):
    ctl = controller(tmp_path)
    jobs = dra.ResidualAblation(ctl)
    jobs.settings = lambda: setup.settings
    jobs.thread = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(ValueError, match="already running"):
        jobs.start(request(setup, "capture"))
    from mlx2.desktop_control import DesktopController
    stub = SimpleNamespace(ablation=Mock(dispatch=Mock(return_value="jobs")),
                           corrections=Mock(dispatch=Mock(return_value="cases")))
    assert DesktopController.dispatch(stub, {"command": "ablation_preview"}) == "jobs"
    assert DesktopController.dispatch(stub, {"command": "ablation_case_create"}) == "cases"
    assert DesktopController.dispatch(stub, {"command": "ablation_overlay_load"}) == "cases"
    stub.ablation = None
    with pytest.raises(ValueError, match="not part of this build"):
        DesktopController.dispatch(stub, {"command": "ablation_start"})


# ------------------------------------------------- correction cases & overlays


def corrections(tmp_path, http=None):
    ctl = SimpleNamespace(home=tmp_path / "home", engines={})
    ctl.home.mkdir()
    return Corrections(ctl, http=http or Mock(side_effect=OSError("no server")))


def case_request(**extra):
    return {"session_id": "chat-1", "branch_id": "branch:abc", "flagged_message_id": "m3",
            "history": [{"role": "user", "text": "hi", "id": "m1"}, {"role": "assistant", "text": "hello", "id": "m2"},
                        {"role": "user", "text": "do the thing", "id": "m2b"}],
            "flagged_text": "I can't help with that.", "model_path": "/models/Toy",
            "checkpoint": f"Toy@{FINGERPRINT}", "reason": "refused", "note": "the thing", **extra}


def test_case_lifecycle(tmp_path):
    store = corrections(tmp_path)
    case = store.create_case(case_request())
    assert case["fingerprint"] == FINGERPRINT and case["prompt_excerpt"] == "do the thing"
    assert [c["id"] for c in store.list_cases()] == [case["id"]]
    full = store.get_case(case["id"])
    assert full["history"][-1]["role"] == "user" and full["use"].startswith("must-fix")
    with pytest.raises(ValueError, match="answer a user message"):
        store.create_case(case_request(history=[{"role": "user", "text": "x"}, {"role": "assistant", "text": "y"}]))
    with pytest.raises(ValueError, match="Reason"):
        store.create_case(case_request(reason="bad"))
    with pytest.raises(ValueError, match="Unknown"):
        store.get_case("../../etc/passwd")
    store.delete_case(case["id"])
    assert store.list_cases() == []


def test_overlay_is_content_addressed_and_checkpoint_bound(tmp_path):
    store = corrections(tmp_path)
    capture = make_capture(tmp_path / "ev/capture", tmp_path / "cache")
    params = dra.default_probe_parameters(4)
    first = store.create_overlay({"capture": str(capture), "params": params, "checkpoint": f"Toy@{FINGERPRINT}",
                                  "label": "probe", "source": {"kind": "probe"}})
    again = store.create_overlay({"capture": str(capture), "params": params, "label": "other label"})
    assert first["id"] == again["id"] and first["binding"]["fingerprint"] == FINGERPRINT
    assert first["parameters"] == params and first["edits"] == dra.plan_preview(params, 4)["edits"]
    assert [o["id"] for o in store.list_overlays()] == [first["id"]]
    with pytest.raises(ValueError, match="measured on"):
        store.create_overlay({"capture": str(capture), "params": params, "checkpoint": "Other@" + "e" * 64})
    case = store.create_case(case_request())
    replay = store.record_replay({"case_id": case["id"], "overlay_id": first["id"], "branch_id": "branch:new"})
    assert replay["replays"][0]["branch_id"] == "branch:new"
    store.delete_overlay(first["id"])
    assert store.list_overlays() == []


def test_server_actions_fail_closed_without_the_overlay_api(tmp_path):
    def not_found(method, url, *args, **kwargs):
        raise urllib.error.HTTPError(url, 404, "not found", None, None)
    assert probe_capability("http://127.0.0.1:8282", http=not_found) == {
        "supported": False, "reason": "This server has no overlay API yet (GET /v1/overlays → 404)"}
    assert not probe_capability("http://example.com:80")["supported"]
    store = corrections(tmp_path, http=not_found)
    capture = make_capture(tmp_path / "ev/capture", tmp_path / "cache")
    overlay = store.create_overlay({"capture": str(capture), "params": dra.default_probe_parameters(4)})
    case = store.create_case(case_request())
    for call in (lambda: store.load_overlay({"overlay_id": overlay["id"], "url": "http://127.0.0.1:8282"}),
                 lambda: store.start_preview({"case_id": case["id"], "overlay_id": overlay["id"], "url": "http://127.0.0.1:8282"})):
        with pytest.raises(ValueError, match="no overlay API"):
            call()


def test_supported_server_is_recognised_and_mismatch_refused(tmp_path):
    def server(method, url, body=None, token=None, timeout=0):
        if method == "GET":
            return {"object": "overlay_list", "api_version": 1, "checkpoint_fingerprint": "f" * 64, "preview": True, "data": []}
        return {"text": "Sure: step one", "route_receipt": {"mlx2_overlay": {"applied": True}}}
    state = probe_capability("http://127.0.0.1:8282", http=server)
    assert state["supported"] and state["preview"]
    store = corrections(tmp_path, http=server)
    capture = make_capture(tmp_path / "ev/capture", tmp_path / "cache")
    overlay = store.create_overlay({"capture": str(capture), "params": dra.default_probe_parameters(4)})
    with pytest.raises(ValueError, match="different checkpoint"):
        store.load_overlay({"overlay_id": overlay["id"], "url": "http://127.0.0.1:8282"})


def test_contrast_export_writes_prompt_file(tmp_path):
    store = corrections(tmp_path)
    case = store.create_case(case_request())
    out = store.export_contrast({"case_ids": [case["id"]], "workspace": str(tmp_path / "ws")})
    rows = [json.loads(line) for line in Path(out["path"]).read_text().splitlines()]
    assert rows == [{"text": "do the thing", "case_id": case["id"], "reason": "refused"}]
    assert out["runner_flag"] == "--extra-bad-prompts" and out["runner_supports"] is False


def test_preview_bodies_match_the_server_contract(tmp_path):
    # Morning preflight 2026-11-06: the named preview sent {"overlay_id": ...},
    # which validate_overlay_selection rejects (it takes id/branch_id), and the
    # ad-hoc preview sent no admin token to a worker that requires one.
    from mlx2.server import validate_overlay_selection

    posts = []

    def server(method, url, body=None, token=None, timeout=0):
        if method == "GET":
            return {"object": "overlay_list", "api_version": 1, "checkpoint_fingerprint": FINGERPRINT,
                    "preview": True, "data": []}
        posts.append((url, body, token))
        return {"text": "Sure: step one", "route_receipt": {"mlx2_overlay": {"applied": True}}}

    store = corrections(tmp_path, http=server)
    token_file = tmp_path / "admin.token"
    token_file.write_text("operator-secret\n")
    token_file.chmod(0o600)
    store.controller.engines["w"] = SimpleNamespace(
        worker={"url": "http://127.0.0.1:8282", "admin_token_file": str(token_file)})
    capture = make_capture(tmp_path / "ev/capture", tmp_path / "cache")
    overlay = store.create_overlay({"capture": str(capture), "params": dra.default_probe_parameters(4),
                                    "checkpoint": f"Toy@{FINGERPRINT}"})
    named, ad_hoc = store.create_case(case_request()), store.create_case(case_request(note="other"))
    store.start_preview({"case_id": named["id"], "overlay_id": overlay["id"], "url": "http://127.0.0.1:8282"})
    store.previews[named["id"]].join(10)
    store.start_preview({"case_id": ad_hoc["id"], "url": "http://127.0.0.1:8282", "capture": str(capture),
                         "params": dra.default_probe_parameters(4)})
    store.previews[ad_hoc["id"]].join(10)
    (_, named_body, named_token), (_, ad_hoc_body, ad_hoc_token) = posts
    assert validate_overlay_selection(named_body["mlx2_overlay"]) == {"id": overlay["id"]}
    assert named_token is None
    assert "parameters" in ad_hoc_body["mlx2_overlay"] and ad_hoc_token == "operator-secret"
