"""Bounded desktop campaign subprocess, run only after explicit UI enqueue.

The owning process registers its own real CPG identity. It renews that lease
and watches a paired-lock child, and aborts immediately on renewal failure.
The artifact child validates both ancestor-owned receipts before importing MLX.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .desktop_campaign_process import spawn

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "qualification/runs/qualify-1010-correctness"


def atomic(path, value):
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def structured(result):
    if not isinstance(result, dict) or result.get("isError"):
        raise RuntimeError("CPG operation failed")
    value = result.get("structuredContent")
    if value is None:
        value = json.loads("".join(item.get("text", "") for item in result.get("content", []) if item.get("type") == "text"))
    if not isinstance(value, dict):
        raise TypeError("CPG response is not an object")
    return value


class Radio:
    def __init__(self, url, client=None):
        if client is None:
            from .tool_backend import _HTTPMCPClient
            client = _HTTPMCPClient(url, {}, timeout=15)
        self.client = client
        self.token = None

    def call(self, name, arguments):
        if self.token:
            arguments = {**arguments, "binding_token": self.token}
        value = structured(self.client.call_tool(name, arguments))
        if value.get("binding_token"):
            self.token = value["binding_token"]
        return value


def verify_claim(claim, *, task, agent=None, generation=None, renewed=False, now=None):
    now = time.time() if now is None else now
    if claim.get("renewed" if renewed else "claimed") is not True or claim.get("task_id") != task:
        raise RuntimeError("CPG GPU lease claim/renewal failed")
    actual_generation = claim.get("lease_generation")
    if type(actual_generation) is not int or actual_generation < 1 or (generation is not None and actual_generation != generation):
        raise RuntimeError("CPG GPU lease generation changed")
    if agent is not None and claim.get("owner_agent_id") != agent:
        raise RuntimeError("CPG GPU lease owner changed")
    expiry = claim.get("lease_expires_at")
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry <= now + 15:
        raise RuntimeError("CPG GPU lease expires too soon")
    return actual_generation


def stop(process, timeout=30):
    if process is None:
        return
    from .desktop_campaign_process import OwnedProcess
    if not isinstance(process, OwnedProcess):
        # Compatibility for CPU callers which have not reaped their leader.
        process = OwnedProcess(process)
    return process.stop(timeout=timeout)


def interrupts(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def owned(spec_path):
    spec = json.loads(spec_path.read_text())
    out = Path(spec["out"])
    radio = Radio(spec["mcp_url"])
    agent = "mlx2-desktop-" + spec["job_id"]
    owner = {"agent": "mlx2_desktop_campaign", "agent_id": agent, "pid": os.getpid(),
             "source_revision": spec["source_revision"], "job_id": spec["job_id"]}
    child = None
    task = worker = workflow = generation = None
    code = 1
    try:
        plan = radio.call("plan_session", {"goal": "One bounded desktop model qualification window",
            "tasks": [{"title": "GPU: " + spec["checkpoint"], "description": "Own one artifact; reference, features and controlled measurements; release before next artifact"}],
            "agent_id": agent, "archive_previous": False})
        task = plan["task_ids"][0]
        session = plan["session_id"]
        workflow = radio.call("start_workflow", {"name": "Desktop qualification " + spec["job_id"], "task_ids": [task]})["workflow_id"]
        worker = radio.call("register_worker", {"workflow_id": workflow, "label": "desktop-model", "role": "executor"})["worker_id"]
        claim = radio.call("claim_task", {"task_id": task, "lease_seconds": 180})
        generation = verify_claim(claim, task=task, agent=agent)
        owner.update(cpg_task=task, cpg_session=session, cpg_worker=worker, cpg_generation=generation,
                     claim=claim, renewed_at=time.time(), expires=claim["lease_expires_at"])
        atomic(out / "desktop-lease.json", owner)
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", session, "--label", spec["job_id"],
                   "--desktop-cpg-lease", str(out / "desktop-lease.json"), "--mcp-url", spec["mcp_url"],
                   "--receipt", str(out / "paired-locks.json"), "--", sys.executable,
                   "-u", "-m", "mlx2.desktop_campaign_worker", "artifact", str(spec_path)]
        child = spawn(command, start_new_session=True)
        deadline = time.monotonic() + 28800
        next_renewal = time.monotonic() + 30
        while child.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded model ownership window expired")
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= next_renewal:
                renewed = radio.call("renew_lease", {"task_id": task, "lease_generation": generation, "lease_seconds": 180})
                verify_claim(renewed, task=task, generation=generation, renewed=True)
                radio.call("worker_heartbeat", {"worker_id": worker})
                owner.update(renewal=renewed, expires=renewed["lease_expires_at"], renewed_at=time.time())
                atomic(out / "desktop-lease.json", owner)
                next_renewal = time.monotonic() + 30
        code = child.returncode
    except BaseException as error:  # noqa: BLE001 -- persist cancellation and guarantee full child-group cleanup
        print(f"Ownership failure: {type(error).__name__}: {error}", flush=True)
        owner["error"] = str(error)
    finally:
        stop(child)
        # Write actual replies, including cleanup failures, never claim release
        # merely because a subprocess exited.
        for name, arguments in (
            ("release_task", {"task_id": task, "lease_generation": generation, "status": "done" if code == 0 else "failed"}),
            ("complete_worker", {"worker_id": worker, "status": "completed" if code == 0 else "failed"}),
        ):
            if (name == "release_task" and generation is None) or (name == "complete_worker" and worker is None):
                continue
            try:
                owner[name] = radio.call(name, arguments)
            except Exception as error:  # noqa: BLE001 -- preserve failed release for operator inspection
                owner[name] = {"error": str(error)}
        owner["finished_at"] = time.time()
        atomic(out / "desktop-lease.json", owner)
    return code


def check_ownership(spec):
    # Parent-owned real desktop lease, no impersonation of the cpg_job producer.
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import _ancestor_pids
    ancestors = _ancestor_pids(os.getpid())
    owner = json.loads((Path(spec["out"]) / "desktop-lease.json").read_text())
    if owner.get("agent") != "mlx2_desktop_campaign" or owner.get("pid") not in ancestors:
        raise RuntimeError("Desktop CPG lease owner is not this process's live ancestor")
    if owner.get("job_id") != spec["job_id"] or owner.get("release_task") or owner.get("expires", 0) <= time.time() + 15:
        raise RuntimeError("Desktop GPU lease is absent, expired or closed")
    pair = []
    for path in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        pair.append(json.loads((path / "owner.json").read_text()))
    if pair[0] != pair[1] or pair[0].get("pid") not in ancestors or pair[0].get("session") != owner["cpg_session"] or pair[0].get("label") != spec["job_id"]:
        raise RuntimeError("Paired GPU locks do not match this campaign and ancestor")
    # An independent live read confirms stored claims still describe CPG state.
    radio = Radio(spec["mcp_url"])
    radio.call("join_session", {"session_id": owner["cpg_session"], "agent_id": owner["agent_id"]})
    result = radio.call("get_nodes", {"node_ids": [owner["cpg_task"]]})
    nodes = result.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 1 or result.get("missing"):
        raise RuntimeError("Live CPG task is absent")
    node = nodes[0]
    payload = node.get("payload", {})
    if node.get("id") != owner["cpg_task"] or node.get("session_id") != owner["cpg_session"] or payload.get("status") != "in_progress" or payload.get("owner_agent_id") != owner["agent_id"] or payload.get("lease_generation") != owner["cpg_generation"] or payload.get("lease_expires_at", 0) <= time.time() + 15:
        raise RuntimeError("Live CPG task no longer belongs to this desktop campaign")
    return {"desktop_lease": owner, "paired_lock_owner": pair[0], "live_task": node}


def load_helper(name):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("desktop_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_run(command, out, name, timeout=7200):
    print("Running " + name, flush=True)
    with (out / (name + ".log")).open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {name}.log")


def http_json(base, path, body=None):
    from urllib.request import Request, urlopen
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=600) as response:
        return json.load(response)


def persistence_seed(base, model, job_id):
    """Publish one real session to disk before a restart-bound feature run."""
    session = "desktop-persistence-" + job_id
    response = http_json(base, "/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Reply with APC_SESSION_READY"}],
        "session_id": session, "max_tokens": 32, "temperature": 0,
        "reasoning_effort": "none", "enable_thinking": False})
    state = http_json(base, f"/v1/apc/sessions/{session}/park", {"ttl_seconds": 3600})
    deadline = time.monotonic() + 30
    while state.get("state") != "disk" and time.monotonic() < deadline:
        time.sleep(0.25)
        state = http_json(base, f"/v1/apc/sessions/{session}")
    if state.get("state") != "disk":
        raise ValueError("APCv2 session did not reach persisted disk state")
    return {"session_id": session, "response": response, "parked": state,
            "status": http_json(base, "/v1/status")}


def performance_samples(ladder):
    helper = load_helper("performance_assessment")
    cells = helper._cells(ladder)
    if ladder.get("runs_per_cell") != 3 or not helper._coverage_valid(ladder, cells):
        raise ValueError("Performance plan has incomplete three-repetition cell coverage")
    samples = []
    for (context, width), cell in sorted(cells.items()):
        if not helper._runs_clean(cell):
            raise ValueError(f"Performance cell {context}/{width} failed thermal or contamination gates")
        metrics = {}
        for short, field in (("prefill_tps", "prefill_tokens_per_second"), ("decode_tps", "decode_tokens_per_second")):
            numbers = [run.get("summary", {}).get(field) for run in cell["runs"]]
            if any(type(number) not in (int, float) or not math.isfinite(number) or number <= 0 for number in numbers):
                raise ValueError("Performance cell lacks positive finite throughput")
            metrics[short] = statistics.median(numbers)
        samples.append({"context_tokens": context, "width": width, "max_tokens": ladder.get("max_tokens"),
                        "repetitions": 3, **metrics})
    return samples


def artifact(spec):
    out = Path(spec["out"])
    profile = spec["profile"]
    result = {"schema": "mlx2.desktop-campaign.v1", "records": [], "failed_levels": [],
              "review_pending": False, "evidence_path": str(out), "qualification_claim": False}
    server = None
    server_log = None
    code = 1
    try:
        ownership = check_ownership(spec)
        atomic(out / "ownership-at-start.json", ownership)
        from .desktop_runner import digest as source_digest
        from .desktop_runner import source_manifest
        if source_digest(source_manifest(ROOT)) != spec["source_sha256"]:
            raise RuntimeError("Frozen campaign source digest changed before model execution")
        # The existing safe probe checks nominal/fair thermal admission before
        # loading reference weights. Throughput has stricter per-cell controls.
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_single_model_smoke_ladder as existing
        initial_thermal = existing.thermal_sample()
        if initial_thermal["thermal_state"] not in {0, 1}:
            raise RuntimeError("Reference smoke deferred until nominal/fair thermal admission")
        atomic(out / "thermal-at-start.json", initial_thermal)
        from .adapters.registry import inspect_model
        resolution = inspect_model(spec["path"])
        checkpoint = f"{Path(spec['path']).name}@{resolution.artifact_fingerprint}"
        if checkpoint != spec["checkpoint"]:
            raise RuntimeError("Artifact fingerprint changed after enqueue")
        kind = resolution.descriptor.metadata.get("execution_kind", "autoregressive")
        if kind != "autoregressive":
            raise ValueError("This profile requires a dedicated non-autoregressive reference")
        from smoke_local_model import _ordinary_generate

        from .serving import runtime_identity
        started = time.time()
        adapter = resolution.adapter_type(spec["path"])
        try:
            output = _ordinary_generate(adapter, "Say hello in one sentence.", 16)
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            del adapter
        # Reference model memory must be reclaimed before one serving process.
        import gc

        import mlx.core as mx
        gc.collect()
        mx.clear_cache()
        atomic(out / "ordinary-reference.json", {"checkpoint": checkpoint,
            "source_revision": spec["source_revision"], "runtime": runtime_identity(),
            "output": output, "started_at": started, "finished_at": time.time(),
            "qualification_claim": False})
        check_ownership(spec)
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", profile["port"])) == 0:
                raise ValueError("Qualification loopback port is already occupied")
        base = f"http://127.0.0.1:{profile['port']}"
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", spec["path"],
                   "--host", "127.0.0.1", "--fabric", "off", "--port", str(profile["port"]), "--qualification-mode",
                   "--max-context", str(profile["context"]), "--max-lanes", str(profile["lanes"]),
                   "--max-inflight", str(profile["inflight"]), "--cache-bytes", str(profile["cache_gib"] << 30),
                   "--apc-persist-dir", str(out / "apcv2-cache"), "--apc-persist-on-shutdown", "--prefill-step", str(profile["prefill_step"])]
        if profile["route"] != "auto":
            command.append("--" + profile["route"].replace("_", "-"))
        server_log = (out / "server.log").open("w")
        server = spawn(command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        status = existing.wait_health(base, server, 900)
        if status.get("artifact") != resolution.artifact_fingerprint or status.get("qualification") != "candidate":
            raise RuntimeError("Server did not report the exact artifact and candidate route")
        if 2 in spec["levels"]:
            # Persistence is restart-bound. Establish a real disk entry, then
            # run the qualifier on the rescanned server, before other feature
            # clients can consume per-run mechanism engagement.
            try:
                seeded = persistence_seed(base, Path(spec["path"]).name, spec["job_id"])
                atomic(out / "persistence-seed.json", seeded)
                original = status
                stop(server)
                server_log.close()
                server_log = (out / "server-after-rescan.log").open("w")
                check_ownership(spec)
                server = spawn(command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                status = existing.wait_health(base, server, 900)
                if any(status.get(key) != original.get(key) for key in ("artifact", "runtime", "settings")):
                    raise ValueError("Restart changed the bound runtime, artifact or serving shape")
                rescan = (status.get("apcv2") or {}).get("persistence", {}).get("rescan", {})
                if rescan.get("registered", 0) < 1:
                    raise ValueError("APCv2 restart did not rescan the persisted entry")
                atomic(out / "persistence-rescan.json", status)
            except Exception as error:  # noqa: BLE001 -- preserve the missing restart gate, continue other useful evidence
                result["persistence_error"] = str(error)
        settings = status["settings"]
        route = settings["route"]
        binding = {"checkpoint": checkpoint, "model_path": spec["path"],
                   "source_revision": spec["source_revision"], "source_sha256": spec["source_sha256"],
                   "runtime": status["runtime"], "host": {"hostname": platform.node(), "platform": platform.platform()},
                   "profile": {"route": route, "settings": settings, "max_tokens": profile["max_tokens"]}}
        result.update(binding=binding, ordinary_output=output, review_pending=1 in spec["levels"])
        atomic(out / "server-startup.json", {"status": status, "pid": server.pid, "command": command})
        if 2 in spec["levels"]:
            try:
                receipt = spec.get("preflight_receipt")
                if not receipt:
                    raise ValueError("A source-bound CPU preflight must finish before GPU ownership")
                qualify = [sys.executable, str(ROOT / "scripts/qualify_serving.py"), "--url", base,
                           "--output", str(out / "features.json"), "--preflight-receipt", receipt]
                for feature in profile["features"]:
                    qualify.extend(["--require-feature", feature])
                command_run(qualify, out, "features")
                if json.loads((out / "features.json").read_text()).get("passed") is not True:
                    raise ValueError("Approved feature qualifier did not pass")
                command_run([sys.executable, str(ROOT / "scripts/sdk_smoke.py"),
                    "--sdk-python", sys.executable, "--url", base], out, "official-sdks")
            except Exception as error:  # noqa: BLE001 -- retain feature failures but still collect ladder/performance
                result["feature_error"] = str(error)
        if set(spec["levels"]) & {2, 3}:
            try:
                ladder = [sys.executable, str(HERE / "thermal_ladder.py"), "--url", base,
                    "--output", str(out / "ladder.json"), "--model", checkpoint, "--model-id", Path(spec["path"]).name,
                    "--route", route, "--artifact-identity", resolution.artifact_fingerprint,
                    "--max-context", str(settings["max_context"]), "--cache-bytes", str(settings["cache_bytes"]),
                    "--max-lanes", str(settings["max_lanes"]), "--max-inflight", str(settings["max_inflight"]),
                    "--prefill-step", str(settings["prefill_step"]), "--apc-persistence", "on" if settings.get("apc_persistence") else "off",
                    "--apc-persist-on-shutdown", "on" if settings.get("apc_persist_on_shutdown") else "off",
                    "--mtp-policy", json.dumps(settings.get("mtp")),
                    "--draft-loop-policy", json.dumps({k: settings.get(k) for k in ("draft_loop", "draft_loop_threshold", "draft_loop_widths")}),
                    "--prefill-policy", json.dumps({k: settings.get(k) for k in ("prefill_depth_budget", "prefill_step_autoscale")}),
                    "--server-pid", str(server.pid), "--runs", "3", "--wide", str(profile["lanes"]),
                    "--wide-max-context", str(profile["context"]), "--max-length", str(profile["context"]),
                    "--max-tokens", str(profile["max_tokens"])]
                if profile["lanes"] == 1:
                    ladder.extend(["--only-width", "1"])
                if 3 in spec["levels"]:
                    ladder.append("--performance-mode")
                command_run(ladder, out, "ladder", timeout=21600)
            except Exception as error:  # noqa: BLE001 -- partial ladder is evidence, not success
                result["ladder_error"] = str(error)
            report_path = out / "ladder.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if 2 in spec["levels"]:
                verdict = load_helper("qualification_verdict").verdict(report)
                atomic(out / "correctness-verdict.json", verdict)
                passed = (verdict.get("qualified") is True and not result.get("feature_error")
                          and not result.get("ladder_error") and not result.get("persistence_error"))
                record = {"binding": binding, "level": 2, "passed": passed,
                          "evidence_path": str(out), "verdict": verdict,
                          "feature_error": result.get("feature_error"),
                          "persistence_error": result.get("persistence_error"),
                          "coverage": {"route": route,
                              "required_routes": list(dict.fromkeys(["ordinary", resolution.default_route])),
                              "tested_routes": [route], "scope": "this exact serving profile"},
                          "qualification_claim": False}
                result["records"].append(record)
                if not passed:
                    result["failed_levels"].append(2)
            if 3 in spec["levels"]:
                try:
                    if result.get("ladder_error"):
                        raise ValueError(result["ladder_error"])
                    samples = performance_samples(report)
                    record = {"binding": binding, "level": 3, "passed": True,
                              "samples": samples, "evidence_path": str(out), "qualification_claim": False}
                except Exception as error:  # noqa: BLE001 -- never promote noisy/incomplete performance
                    record = {"binding": binding, "level": 3, "passed": False,
                              "error": str(error), "evidence_path": str(out), "qualification_claim": False}
                    result["failed_levels"].append(3)
                result["records"].append(record)
        check_ownership(spec)
        if source_digest(source_manifest(ROOT)) != spec["source_sha256"]:
            raise RuntimeError("Frozen source changed during campaign execution")
        result["message"] = "Review ordinary output for coherence" if result["review_pending"] else "Requested levels completed"
        code = 1 if result["failed_levels"] else 0
    except BaseException as error:  # noqa: BLE001 -- persist cancellation and guarantee full child-group cleanup
        for record in result["records"]:
            record["passed"] = False
            record["aborted"] = True
        result["review_pending"] = False
        result["error"] = f"{type(error).__name__}: {error}"
        result["message"] = result["error"]
        print(result["error"], flush=True)
    finally:
        stop(server)
        if server_log:
            server_log.close()
        result["finished_at"] = time.time()
        atomic(out / "result.json", result)
    return code


def main():
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupts)
    if len(sys.argv) != 3 or sys.argv[1] not in {"owned", "artifact"}:
        raise SystemExit("Usage: desktop_campaign_worker {owned|artifact} SPEC.json")
    path = Path(sys.argv[2]).resolve(strict=True)
    return owned(path) if sys.argv[1] == "owned" else artifact(json.loads(path.read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
