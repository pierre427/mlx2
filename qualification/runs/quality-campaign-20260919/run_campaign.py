#!/usr/bin/env python3
"""Sequential 2026-09-19 GPU quality campaign (build/dry-run is CPU-only).

One owned server runs at a time on port 8297.  Every server cycle and check is
recorded in status.json, failures are stage-local, named stages are rerunnable,
and a production launchd service is restored only when it was loaded on entry.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from campaign_config import (
    BASE_URL,
    MODELS,
    PORT,
    PYTHON,
    ROOT,
    RUN,
    SDK_PYTHON,
    Model,
    Route,
    requires_mlx_vlm,
    server_args,
    stage_pythonpath,
    validate_cpu_preflight,
    validate_server_arguments,
    verify_mlx_vlm_runtime,
)
from feature_smoke import CHECKS_BY_GROUP

SERVICE = "com.example.fn-uncensored-mlx-serve"
PLIST = Path.home() / "Library/LaunchAgents" / f"{SERVICE}.plist"
STATUS = RUN / "status.json"
PREFLIGHT = RUN / "preflight.json"
ENV = {
    **os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def stage_environment(model: Model) -> dict:
    return {**ENV, "PYTHONPATH": stage_pythonpath(model)}


def command_text(command):
    return shlex.join(str(item) for item in command)


def common_server_command(
    model: Model,
    route: Route,
    phase: str,
    stage_dir: Path,
    *,
    opt_in=False,
    persistence=False,
):
    command = [str(PYTHON), "-u", "-m", "mlx2.server", *server_args(model, route, phase, opt_in=opt_in)]
    cache_dir = stage_dir / "cache"
    command += [
        "--cache-dir", str(cache_dir),
        "--admin-token-file", str(RUN / ".runtime" / "admin-token"),
        "--reasoning-signing-key-file", str(RUN / ".runtime" / "reasoning-key"),
    ]
    if persistence:
        command += [
            "--apc-persist-dir", str(cache_dir),
            "--apc-persist-on-shutdown",
        ]
    return command


def feature_command(model, route, stage_dir, group):
    command = [
        str(PYTHON), str(RUN / "feature_smoke.py"), "--url", BASE_URL,
        "--output", str(stage_dir / f"feature-{group}.json"),
        "--raw-dir", str(stage_dir / "raw" / group), "--model", model.name,
        "--route", route.name, "--group", group,
        "--model-id", Path(model.path).name,
        "--ordinary-baseline", str(RUN / "baselines" / f"{model.name}-ordinary.json"),
        "--admin-token-file", str(RUN / ".runtime" / "admin-token"),
    ]
    for capability in sorted(model.capabilities):
        command += ["--capability", capability]
    if route.apc_interior:
        command += ["--capability", "apc-interior"]
    if route.speculative:
        command.append("--speculative")
    if route.name in {"mtp1", "mtp2", "dflash2"}:
        command.append("--fly")
    return command


def qualifier_command(name, stage_dir):
    return [
        str(PYTHON), "scripts/qualify_serving.py", "--url", BASE_URL,
        "--output", str(stage_dir / "qualification.json"),
        "--preflight-receipt", str(PREFLIGHT), "--timeout", "1800",
    ]


def sdk_command():
    return [str(PYTHON), "scripts/sdk_smoke.py", "--sdk-python", str(SDK_PYTHON), "--url", BASE_URL]


def prefetch_command(model, stage_dir, *, dry_run=False):
    command = [
        str(PYTHON), "scripts/gpu_check_apc_prefetch.py", "--i-own-the-gpu",
        "--model-path", model.path, "--dir", str(stage_dir / "prefetch"),
        "--output", str(stage_dir / "apc-prefetch.json"), "--timeout-seconds", "1800",
        "--startup-timeout-seconds", "600",
    ]
    if dry_run:
        command.insert(3, "--dry-run")
    return command


def sanity_command(stage_dir):
    return [str(PYTHON), str(RUN / "sanity_20x20.py"), BASE_URL, str(stage_dir / "sanity.json")]


def ladder_command(model, route, stage_dir):
    return [
        str(PYTHON), str(RUN / "ladder.py"), "--url", BASE_URL,
        "--output", str(stage_dir / "ladder.json"), "--model", model.name,
        "--model-id", Path(model.path).name, "--route", route.name,
        "--max-context", str(model.max_context),
        "--timeout", "7200",
    ]


def stage_specs(phase):
    stages = []
    for model in MODELS:
        routes = model.routes if phase != "ladder" else tuple(route for route in model.routes if route.name == model.default_route)
        for route in routes:
            name = f"{phase}-{model.name}-{route.name}"
            stage_dir = RUN / "results" / name
            if phase == "smoke":
                cycles = [
                    {"name": "base", "server": common_server_command(model, route, phase, stage_dir),
                     "steps": [("qualifier", qualifier_command(name, stage_dir), 4000),
                               ("feature-core", feature_command(model, route, stage_dir, "core"), 14400),
                               ("official-sdks", sdk_command(), 7200)]},
                    {"name": "opt-in", "server": common_server_command(model, route, phase, stage_dir, opt_in=True),
                     "steps": [("feature-opt-in", feature_command(model, route, stage_dir, "opt-in"), 7200)]},
                    {"name": "persist-seed", "server": common_server_command(
                        model, route, phase, stage_dir, persistence=True
                    ),
                     "steps": [("persist-seed", feature_command(model, route, stage_dir, "persist-seed"), 3600)]},
                    {"name": "persist-rescan", "server": common_server_command(
                        model, route, phase, stage_dir, persistence=True
                    ),
                     "steps": [("persist-rescan", feature_command(model, route, stage_dir, "persist-rescan"), 3600)]},
                ]
                standalone = [("apc-prefetch", prefetch_command(model, stage_dir), 7200)]
            elif phase == "sanity":
                cycles = [{"name": "sanity", "server": common_server_command(model, route, phase, stage_dir),
                           "steps": [("sanity-20x20", sanity_command(stage_dir), 7200)]}]
                standalone = []
            else:
                cycles = [{"name": "ladder", "server": common_server_command(model, route, phase, stage_dir),
                           "steps": [("context-ladder", ladder_command(model, route, stage_dir), 172800)]}]
                standalone = []
            stages.append({"name": name, "phase": phase, "model": model, "route": route,
                           "dir": stage_dir, "cycles": cycles, "standalone": standalone})
    return stages


class Campaign:
    def __init__(self, args, stages):
        self.args, self.stages = args, stages
        previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        self.state = {
            "schema": "mlx2.quality-campaign-status.v1", "started": previous.get("started", time.strftime("%F %T")),
            "updated": time.strftime("%F %T"), "phase": "starting",
            "order": [stage["name"] for stage in stages], "stages": previous.get("stages", {}),
            "history": previous.get("history", []),
        }

    def save(self):
        self.state["updated"] = time.strftime("%F %T")
        STATUS.write_text(json.dumps(self.state, indent=2) + "\n")

    @staticmethod
    def health_state():
        payload = None
        http_status = None
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=3) as response:
                http_status = response.status
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            http_status = error.code
            try:
                payload = json.loads(error.read())
            except (ValueError, OSError):
                payload = None
        except (OSError, ValueError):
            return {
                "ready": False,
                "reachable": False,
                "status": None,
                "error": None,
                "engine_state": None,
                "http_status": None,
            }
        status = payload.get("status") if isinstance(payload, dict) else None
        explicit_error = payload.get("error") if isinstance(payload, dict) else None
        engine_status = None
        if status == "unavailable" and not explicit_error:
            try:
                with urllib.request.urlopen(BASE_URL + "/v1/status", timeout=3) as response:
                    engine_status = json.load(response)
            except urllib.error.HTTPError as error:
                try:
                    engine_status = json.loads(error.read())
                except (ValueError, OSError):
                    engine_status = None
            except (OSError, ValueError):
                engine_status = None
        engine_error = (
            engine_status.get("error") if isinstance(engine_status, dict) else None
        )
        engine_state = (
            engine_status.get("state") if isinstance(engine_status, dict) else None
        )
        health_error = explicit_error or engine_error
        if (
            not health_error
            and status == "unavailable"
            and engine_state == "ready"
            and engine_status.get("healthy") is False
        ):
            health_error = "ServingEngine became unhealthy after reaching ready state"
        return {
            "ready": http_status == 200 and status == "ok",
            "reachable": True,
            "status": status,
            "error": health_error,
            "engine_state": engine_state,
            "http_status": http_status,
        }

    @classmethod
    def healthy(cls):
        return cls.health_state()["ready"]

    def wait_for_server(self, server, *, timeout=2400, poll_seconds=3):
        """Wait through normal loading, but stop on process or engine failure."""
        deadline = time.monotonic() + timeout
        health = {
            "ready": False,
            "reachable": False,
            "status": None,
            "error": None,
            "engine_state": None,
            "http_status": None,
        }
        while True:
            server_code = server.poll()
            if server_code is not None:
                return {
                    "ready": False,
                    "reason": f"server exited {server_code}",
                    "health": health,
                }
            health = self.health_state()
            if health["error"]:
                return {
                    "ready": False,
                    "reason": f"server health error: {health['error']}",
                    "health": health,
                }
            if health["ready"]:
                return {"ready": True, "reason": None, "health": health}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {
                    "ready": False,
                    "reason": f"server did not become healthy within {timeout}s",
                    "health": health,
                }
            time.sleep(min(poll_seconds, remaining))

    @staticmethod
    def stop(server):
        if server.poll() is None:
            try: os.killpg(server.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError): pass
            try: server.wait(120)
            except subprocess.TimeoutExpired: pass
        try: os.killpg(server.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError): pass
        time.sleep(3)

    @staticmethod
    def stop_step(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()

    def run_step(self, entry, label, command, limit, log_path, *, env=None, server=None):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        abort_reason = None
        with log_path.open("w") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env or ENV,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = started + limit
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    abort_reason = f"step timeout after {limit}s"
                    break
                if server is not None:
                    server_code = server.poll()
                    if server_code is not None:
                        abort_reason = f"server exited {server_code} during {label}"
                        break
                    health = self.health_state()
                    if health["error"]:
                        abort_reason = f"server health error during {label}: {health['error']}"
                        break
                time.sleep(0.25)
            if abort_reason is not None:
                self.stop_step(process)
                code = "timeout" if abort_reason.startswith("step timeout") else "server_failure"
            else:
                code = process.returncode
        lines = log_path.read_text(errors="replace").splitlines()
        entry["steps"][label] = {
            "command": command_text(command), "returncode": code,
            "seconds": round(time.monotonic() - started, 2),
            "failed_lines": [line[:500] for line in lines if line.startswith("FAIL") or ": FAIL" in line][:20],
            "last": lines[-1][:500] if lines else "", "log": str(log_path),
        }
        if abort_reason is not None:
            entry["steps"][label]["reason"] = abort_reason
            entry["steps"][label]["server_failure"] = code == "server_failure"
        self.save()
        return code == 0

    def run_cycle(self, stage, entry, cycle, *, env):
        stage["dir"].mkdir(parents=True, exist_ok=True)
        command = cycle["server"]
        log_path = stage["dir"] / f"server-{cycle['name']}.log"
        cycle_entry = entry["cycles"].setdefault(cycle["name"], {})
        cycle_entry.update({"state": "loading", "command": command_text(command), "started": time.strftime("%T")})
        self.save()
        with log_path.open("w") as log:
            server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                startup = self.wait_for_server(server, timeout=2400)
                health = startup["health"]
                if not startup["ready"]:
                    cycle_entry["state"] = "failed"
                    cycle_entry["reason"] = startup["reason"]
                    cycle_entry["health"] = health
                    cycle_entry["log_tail"] = log_path.read_text(errors="replace")[-2000:]
                    return False
                cycle_entry["state"] = "running"; self.save()
                ok = True
                for label, step, limit in cycle["steps"]:
                    step_label = f"{cycle['name']}:{label}"
                    ok = self.run_step(
                        entry,
                        step_label,
                        step,
                        limit,
                        stage["dir"] / f"{cycle['name']}-{label}.log",
                        env=env,
                        server=server,
                    ) and ok
                    if entry["steps"][step_label].get("server_failure"):
                        break
                final_health = self.health_state()
                if server.poll() is not None or final_health["error"]:
                    ok = False
                cycle_entry["state"] = "passed" if ok else "failed"
                cycle_entry["server_alive_after_steps"] = final_health["ready"]
                cycle_entry["health_after_steps"] = final_health
                return ok
            finally:
                self.stop(server)
                cycle_entry["finished"] = time.strftime("%T"); self.save()

    def run_stage(self, stage):
        old = self.state["stages"].get(stage["name"])
        if old:
            self.state["history"].append({stage["name"]: old})
        entry = self.state["stages"][stage["name"]] = {
            "state": "running", "phase": stage["phase"], "model": stage["model"].label,
            "route": stage["route"].name, "started": time.strftime("%F %T"), "steps": {}, "cycles": {},
        }
        env = stage_environment(stage["model"])
        if requires_mlx_vlm(stage["model"]):
            try:
                runtime = verify_mlx_vlm_runtime(env=env)
            except Exception as error:  # noqa: BLE001 - fail only this stage
                entry["state"] = "failed"
                entry["reason"] = f"multimodal runtime validation failed: {error}"
                entry["finished"] = time.strftime("%F %T")
                self.save()
                return
            entry["runtime"] = {"mlx_vlm": runtime}
            stage["dir"].mkdir(parents=True, exist_ok=True)
            (stage["dir"] / "runtime.json").write_text(
                json.dumps(entry["runtime"], indent=2) + "\n"
            )
        self.save(); ok = True
        for cycle in stage["cycles"]:
            try:
                ok = self.run_cycle(stage, entry, cycle, env=env) and ok
            except Exception as error:  # noqa: BLE001 - a failed cycle must not stop the campaign
                ok = False; entry["cycles"].setdefault(cycle["name"], {})["error"] = repr(error); self.save()
        for label, command, limit in stage["standalone"]:
            try:
                ok = self.run_step(
                    entry,
                    label,
                    command,
                    limit,
                    stage["dir"] / f"{label}.log",
                    env=env,
                ) and ok
            except Exception as error:  # noqa: BLE001 - a failed step must not stop the campaign
                ok = False; entry["steps"][label] = {"returncode": "exception", "reason": repr(error)}; self.save()
        entry["state"] = "passed" if ok else "failed"; entry["finished"] = time.strftime("%F %T"); self.save()


def validate_static_paths():
    required = [PYTHON, SDK_PYTHON, ROOT / "scripts/qualify_serving.py", ROOT / "scripts/sdk_smoke.py",
                ROOT / "scripts/gpu_check_apc_prefetch.py", RUN / "feature_smoke.py", RUN / "ladder.py", RUN / "sanity_20x20.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("missing campaign paths: " + ", ".join(missing))


def generate_preflight_receipt(campaign):
    """Replace any prior receipt with one bound to the served source root."""
    replaced_existing = PREFLIGHT.exists()
    result = subprocess.run(
        [
            str(PYTHON),
            "scripts/qualify_serving.py",
            "--preflight-only",
            "--output",
            str(PREFLIGHT),
        ],
        cwd=ROOT,
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    campaign.state["preflight"] = {
        "returncode": result.returncode,
        "root": str(ROOT),
        "replaced_existing": replaced_existing,
        "tail": (result.stdout + result.stderr)[-1000:],
    }
    campaign.save()
    if result.returncode:
        raise RuntimeError(
            "campaign preflight generation failed against served source root: "
            + campaign.state["preflight"]["tail"]
        )


def dry_run(stages):
    validate_static_paths()
    preflight = validate_cpu_preflight()
    report = {
        "schema": "mlx2.quality-campaign-dry-run.v1", "cpu_only": True,
        "root": str(ROOT), "port": PORT, "preflight": preflight, "stages": [],
    }
    from mlx2.server import build_parser
    parser = build_parser()
    for stage in stages:
        runtime = (
            {"mlx_vlm": verify_mlx_vlm_runtime(env=stage_environment(stage["model"]))}
            if requires_mlx_vlm(stage["model"])
            else None
        )
        row = {"name": stage["name"], "phase": stage["phase"], "cycles": [], "standalone": []}
        if runtime is not None:
            row["runtime"] = runtime
        for cycle in stage["cycles"]:
            validate_server_arguments(cycle["server"][4:], parser=parser)
            row["cycles"].append({
                "name": cycle["name"], "server": command_text(cycle["server"]),
                "checks": [{
                    "name": label, "command": command_text(command), "timeout": timeout,
                    "named_checks": list(CHECKS_BY_GROUP.get(command[command.index("--group") + 1], ())) if "--group" in command else [],
                } for label, command, timeout in cycle["steps"]],
            })
        for label, command, timeout in stage["standalone"]:
            printed = prefetch_command(stage["model"], stage["dir"], dry_run=True) if label == "apc-prefetch" else command
            row["standalone"].append({"name": label, "command": command_text(command), "validation_command": command_text(printed), "timeout": timeout})
        report["stages"].append(row)
        print(f"STAGE {stage['name']}")
        for cycle in row["cycles"]:
            print(f"  SERVER[{cycle['name']}] {cycle['server']}")
            for check in cycle["checks"]:
                print(f"    CHECK {check['name']} timeout={check['timeout']} {check['command']}")
                for named in check["named_checks"]: print(f"      NAMED {named}")
        for check in row["standalone"]: print(f"  CHECK {check['name']} timeout={check['timeout']} {check['command']}")
    output = RUN / "dry-run.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    counts = {phase: sum(row["phase"] == phase for row in report["stages"]) for phase in ("smoke", "sanity", "ladder")}
    print(f"DRY-RUN PASS cpu_only=true models={len(preflight['models'])} policies={len(preflight['policies'])} smoke={counts['smoke']} sanity={counts['sanity']} ladder={counts['ladder']} output={output}")


def launchd_loaded():
    return subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{SERVICE}"],
        capture_output=True, check=False,
    ).returncode == 0


def restore_production(was_loaded):
    if not was_loaded:
        return {"attempted": False, "ready": None}
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(PLIST)],
        capture_output=True, check=False,
    )
    ready = False
    for _ in range(120):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8282/v1/models", timeout=3) as response:
                if b'"state":"ready"' in response.read(): ready = True; break
        except (OSError, ValueError):
            pass
        time.sleep(3)
    return {"attempted": True, "ready": ready}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "sanity", "ladder", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("stages", nargs="*", help="optional exact stage names to rerun")
    args = parser.parse_args(argv)
    phases = ("smoke", "sanity", "ladder") if args.phase == "all" else (args.phase,)
    stages = [stage for phase in phases for stage in stage_specs(phase)]
    known = {stage["name"] for stage in stages}
    unknown = sorted(set(args.stages) - known)
    if unknown: parser.error("unknown stage(s) for selected phase: " + ", ".join(unknown))
    if args.stages: stages = [stage for stage in stages if stage["name"] in set(args.stages)]
    if args.dry_run:
        dry_run(stages); return 0
    validate_static_paths()
    runtime = RUN / ".runtime"; runtime.mkdir(mode=0o700, exist_ok=True)
    for name in ("admin-token", "reasoning-key"):
        path = runtime / name
        path.write_text(secrets.token_urlsafe(48) + "\n"); path.chmod(0o600)
    campaign = Campaign(args, stages)
    lock = Path("/tmp/gpu.lock")
    if lock.exists():
        raise RuntimeError(f"GPU lock already exists; coordinator must serialize this campaign: {lock.read_text().strip()}")
    lock.write_text(f"mlx2 quality-campaign-20260919 pid {os.getpid()}\n")
    was_loaded = launchd_loaded()
    if was_loaded:
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE}"],
            capture_output=True, check=False,
        )
        time.sleep(8)
    try:
        campaign.state["phase"] = "preflight"; campaign.save()
        generate_preflight_receipt(campaign)
        campaign.state["phase"] = "stages"; campaign.save()
        for stage in stages:
            try: campaign.run_stage(stage)
            except Exception as error:  # noqa: BLE001 - record and continue to later stages
                campaign.state["stages"].setdefault(stage["name"], {})["state"] = "failed"
                campaign.state["stages"][stage["name"]]["reason"] = repr(error); campaign.save()
    finally:
        if lock.exists() and f"pid {os.getpid()}" in lock.read_text(): lock.unlink()
        campaign.state["production_service_was_loaded"] = was_loaded
        campaign.state["production_restore"] = restore_production(was_loaded)
        campaign.state["phase"] = "done"; campaign.save()
    return 0 if all(campaign.state["stages"].get(stage["name"], {}).get("state") == "passed" for stage in stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
