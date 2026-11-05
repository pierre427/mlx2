#!/usr/bin/env python3
"""One bounded Kimi LoRA child under CPG and paired locks, restoring launchd.

No model imports occur here. A spec's command is an argv list, never a shell
fragment. --check-only performs read-only identity/admission checks. Execution
refuses nonempty receipt directories. The existing plist is never rewritten.

Mechanisms reused from this repository at 91379787e: DesktopCPGLease and
run_with_gpu_locks (ownership, child cleanup, sentinel checks). This supervisor
adds an exact-identity, reversible managed-service pause; it does not qualify a
model, adapter, training route, or performance result.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
LABEL = "com.example.mlx2-kimi-linear-48b-q8"
DEFAULT_SECONDS = 1080


class AdmissionBusy(RuntimeError):
    """A competing GPU owner/waiter must finish before service restoration."""


def atomic(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_guards():
    path = ROOT / "scripts/run_with_gpu_locks.py"
    spec = importlib.util.spec_from_file_location("kimi_lora_gpu_locks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_spec(raw):
    if not isinstance(raw, dict):
        raise ValueError("spec must be an object")
    spec = dict(raw)
    for key in ("model", "checkpoint", "source_revision", "out", "job_id"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ValueError(f"missing nonempty {key}")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", spec["job_id"]):
        raise ValueError("job_id must contain only letters, digits, underscore or hyphen")
    if not re.fullmatch(r"[0-9a-f]{40}", spec["source_revision"]):
        raise ValueError("source_revision must be an exact 40-character git commit")
    if not re.fullmatch(r"[^@]+@[0-9a-f]{64}", spec["checkpoint"]):
        raise ValueError("checkpoint must be artifact-name@local-64-character-fingerprint")
    command = spec.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v or "\0" in v for v in command):
        raise ValueError("command must be a nonempty argv list")
    for key, default, limit in (("timeout_seconds", DEFAULT_SECONDS, 1200), ("restore_timeout_seconds", 300, 600)):
        value = spec.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= limit:
            raise ValueError(f"{key} must be between 1 and {limit}")
        spec[key] = float(value)
    model = Path(spec["model"]).expanduser().resolve()
    out = Path(spec["out"]).expanduser().resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("model directory/config.json is missing")
    if out == model or model in out.parents:
        raise ValueError("receipt output must not be inside the immutable model")
    spec.update(model=str(model), out=str(out))
    service = spec.get("service")
    if not isinstance(service, dict) or service.get("label") != LABEL or service.get("port") != 8282:
        raise ValueError("explicit service must name the authorized Kimi launchd label and port 8282")
    plist = Path(service.get("plist", "")).expanduser()
    expected = Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
    if plist != expected or plist.is_symlink() or not plist.is_file():
        raise ValueError("service plist must be the exact regular authorized LaunchAgent file")
    spec["service"] = {**service, "plist": str(plist), "host": "127.0.0.1"}
    spec.setdefault("mcp_url", "http://127.0.0.1:8766/mcp")
    return spec


class Host:
    """Small injectable host boundary; tests never call launchctl or the server."""
    def __init__(self, guards=None):
        self.guards = guards or load_guards()
        self._transition_files = None

    @staticmethod
    def run(argv, **kwargs):
        return subprocess.run(argv, capture_output=True, text=True, timeout=15, **kwargs)

    def launchd(self, service):
        target = f"gui/{os.getuid()}/{service['label']}"
        result = self.run(["/bin/launchctl", "print", target])
        if result.returncode:
            # A failed query is not automatically proof that the job is absent.
            if "Could not find service" in result.stderr:
                return None
            raise RuntimeError("launchctl service status is unverifiable: " + result.stderr.strip())
        text = result.stdout
        pid = re.search(r"^\s*pid = (\d+)\s*$", text, re.M)
        path = re.search(r"^\s*path = (.+)$", text, re.M)
        args = re.search(r"^\s*arguments = \{\n(.*?)^\s*\}", text, re.M | re.S)
        state = re.search(r"^\s*state = (\w+)\s*$", text, re.M)
        return {"pid": int(pid[1]) if pid else None, "path": path[1].strip() if path else None,
                "argv": [v.strip() for v in args[1].splitlines()] if args else None,
                "state": state[1] if state else None, "raw": text}

    @staticmethod
    def process(pid):
        import psutil
        try:
            proc = psutil.Process(pid)
            return {"pid": pid, "argv": proc.cmdline(), "created": proc.create_time(),
                    "uid": proc.uids().real, "cwd": proc.cwd()}
        except psutil.NoSuchProcess:
            return None

    def listener_pids(self, port):
        result = self.run(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"])
        if result.returncode not in (0, 1):
            raise RuntimeError("listener PID query failed")
        return sorted({int(row[1:]) for row in result.stdout.splitlines() if re.fullmatch(r"p\d+", row)})

    @staticmethod
    def request(service, method="GET", path="/v1/status", body=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(f"http://127.0.0.1:{service['port']}{path}", data=data, method=method,
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError("service returned a non-object response")
        return result

    def action(self, action, service):
        target = f"gui/{os.getuid()}"
        argv = (["/bin/launchctl", "bootout", target + "/" + service["label"]]
                if action == "bootout" else ["/bin/launchctl", "bootstrap", target, service["plist"]])
        result = self.run(argv)
        if result.returncode:
            raise RuntimeError(f"launchctl {action} failed: {result.stderr.strip()}")

    @staticmethod
    def sleep(seconds):
        time.sleep(seconds)

    @staticmethod
    def now():
        return time.monotonic()

    def live_waiters(self):
        found = []
        if not self.guards.WAITERS.exists():
            return found
        for path in sorted(self.guards.WAITERS.iterdir()):
            if not path.is_file():
                continue
            fields = path.name.rsplit(".", 3)
            if len(fields) != 4:
                continue
            try:
                pid = int(fields[-2])
                modified = path.stat().st_mtime
                os.kill(pid, 0)
            except (ValueError, ProcessLookupError, FileNotFoundError):
                continue
            except PermissionError:
                raise RuntimeError("cannot verify ownership of a GPU waiter")
            if time.time() - modified <= 300:
                found.append(path.name)
        return found

    def contenders(self, allowed_pid=None):
        waiters = self.live_waiters()
        if waiters:
            raise AdmissionBusy("live GPU waiters prevent service transition: " + ", ".join(waiters))
        foreign = self.guards.foreign_model_processes()
        unexpected = [row for row in foreign if int(row.split(None, 1)[0]) != allowed_pid]
        if unexpected:
            raise AdmissionBusy("foreign unmanaged model process present: " + "; ".join(unexpected[:5]))
        return waiters

    @contextlib.contextmanager
    def transition_guard(self, allowed_pid=None):
        """Hold both idle sentinel flocks across a managed-service transition.

        This is not a training ownership window. It closes the admission race
        while draining or bootstrapping the known service. The normal paired
        directory/owner protocol is still acquired by the GPU child wrapper.
        """
        if self._transition_files is not None:
            raise RuntimeError("nested service transition guard")
        snapshots = []
        with contextlib.ExitStack() as stack:
            opened = []
            for path in self.guards.LOCKS:
                if path.is_symlink():
                    raise RuntimeError("GPU sentinel is a symlink")
                row = self.guards.snapshot(path)
                snapshots.append({"path": str(path), **row})
                if row["kind"] != "file" or row["size"] != 0:
                    raise AdmissionBusy("GPU lock is not idle: " + str(path))
                stream = stack.enter_context(path.open("r+"))
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise AdmissionBusy("GPU sentinel is flock-held") from error
                if os.fstat(stream.fileno()).st_ino != path.stat().st_ino:
                    raise RuntimeError("GPU sentinel changed during admission")
                opened.append((path, stream))
            waiters = self.contenders(allowed_pid)
            self._transition_files = opened
            try:
                yield {"locks": snapshots, "live_waiters": waiters, "allowed_managed_pid": allowed_pid,
                       "transition_protection": "both_empty_sentinel_flocks"}
            finally:
                self._transition_files = None

    def idle(self, allowed_pid=None):
        """Check regular sentinel inode locks as well as directory/owner receipts."""
        if self._transition_files is not None:
            for path, stream in self._transition_files:
                if (not path.is_file() or path.is_symlink() or path.stat().st_size != 0
                        or os.fstat(stream.fileno()).st_ino != path.stat().st_ino):
                    raise RuntimeError("GPU sentinel changed during protected transition")
            return {"live_waiters": self.contenders(allowed_pid), "transition_guard_held": True}
        with self.transition_guard(allowed_pid) as receipt:
            return receipt

    def telemetry(self):
        result = {}
        for key, argv in (("swap", ["/usr/sbin/sysctl", "vm.swapusage"]),
                          ("thermal", ["/usr/bin/pmset", "-g", "therm"]),
                          ("vm", ["/usr/bin/vm_stat"])):
            try:
                row = self.run(argv)
                result[key] = {"returncode": row.returncode, "stdout": row.stdout, "stderr": row.stderr}
            except (OSError, subprocess.SubprocessError) as error:
                result[key] = {"error": str(error)}
        return result


def identity(status):
    execution = status.get("execution") or {}
    required = ("model", "artifact", "base_artifact", "profile", "settings", "runtime")
    if any(key not in status for key in required) or not execution.get("policy"):
        raise RuntimeError("service identity/policy status is incomplete")
    return {**{key: status[key] for key in ("model", "artifact", "base_artifact", "profile", "settings")},
            "runtime": status["runtime"],
            "execution": {key: execution.get(key) for key in ("route", "speculative_route", "policy", "artifact")}}


def require_idle_status(status, state="serving", *, idle=True):
    if status.get("healthy") is not True or status.get("state") != "ready":
        raise RuntimeError("managed service is not healthy and ready")
    if (status.get("quiesce") or {}).get("state") != state:
        raise RuntimeError("managed service lifecycle is not " + state)
    if idle:
        for key in ("inflight", "queue_depth", "accepted_lifecycles"):
            if status.get(key) != 0:
                raise RuntimeError("managed service is busy or missing " + key)
        if (status.get("quiesce") or {}).get("terminal_cleanup_pending") != 0:
            raise RuntimeError("managed service terminal cleanup is pending or unverifiable")
    if ((status.get("lora") or {}).get("enabled") or (status.get("multi_lora") or {}).get("enabled")
            or status.get("weight_program")):
        raise RuntimeError("dynamic adapters cannot be restored from this launchd plist")


class ManagedService:
    def __init__(self, spec, host, receipt=None):
        self.spec, self.service, self.host = spec, spec["service"], host
        self.receipt = Path(receipt) if receipt else None
        self.record = {"schema": "mlx2.kimi-lora-service-pause.v1", "events": []}
        self.before = None
        self.pause_requested = False
        self.bootout_attempted = False

    def save(self, event, **values):
        self.record.update(values)
        self.record["events"].append({"event": event, "at": time.time()})
        if self.receipt:
            atomic(self.receipt, self.record)

    def snapshot(self):
        plist_path = Path(self.service["plist"])
        if plist_path.is_symlink() or plist_path.stat().st_uid != os.getuid():
            raise RuntimeError("service plist is not an owned regular file")
        data = plist_path.read_bytes()
        plist = plistlib.loads(data)
        if plist.get("Label") != self.service["label"]:
            raise RuntimeError("plist label changed")
        args = plist.get("ProgramArguments")
        if not isinstance(args, list) or any(not isinstance(v, str) for v in args):
            raise RuntimeError("plist has no explicit argv")
        for flag, expected in (("--model", self.spec["model"]), ("--host", "127.0.0.1"), ("--port", "8282")):
            if args.count(flag) != 1 or args.index(flag) + 1 >= len(args) or args[args.index(flag) + 1] != expected:
                raise RuntimeError("plist differs from expected " + flag)
        launchd = self.host.launchd(self.service)
        if not launchd or launchd["state"] != "running" or not launchd["pid"]:
            raise RuntimeError("expected managed service is not running")
        if launchd["path"] != str(plist_path) or launchd["argv"] != args:
            raise RuntimeError("loaded launchd definition differs from exact plist")
        proc = self.host.process(launchd["pid"])
        if not proc or proc["uid"] != os.getuid() or proc["argv"] != args or proc["cwd"] != plist.get("WorkingDirectory"):
            raise RuntimeError("managed process identity does not match launchd/plist")
        if self.host.listener_pids(self.service["port"]) != [proc["pid"]]:
            raise RuntimeError("port listener is not exclusively the verified managed PID")
        status = self.host.request(self.service)
        bound = identity(status)
        if bound["model"] != Path(self.spec["model"]).name or bound["artifact"] != self.spec["checkpoint"].rsplit("@", 1)[1]:
            raise RuntimeError("managed service checkpoint differs from spec")
        expected_policy = self.service.get("expected_policy")
        if expected_policy is not None and bound["execution"]["policy"] != expected_policy:
            raise RuntimeError("managed service policy differs from spec")
        expected_hash = self.service.get("plist_sha256")
        digest = hashlib.sha256(data).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError("managed service plist hash differs from spec")
        return {"plist_sha256": digest, "plist_stat": {"uid": plist_path.stat().st_uid, "mode": plist_path.stat().st_mode},
                "plist": plist, "launchd": launchd, "process": proc, "status": status, "identity": bound}

    def preflight(self):
        snapshot = self.snapshot()
        require_idle_status(snapshot["status"])
        self.host.idle(snapshot["process"]["pid"])
        self.before = snapshot
        self.save("verified", before=snapshot)
        return snapshot

    def unchanged(self, *, require_process=True):
        path = Path(self.service["plist"])
        if self.before is None or path.is_symlink() or sha256(path) != self.before["plist_sha256"]:
            raise RuntimeError("service plist changed; refusing service mutation")
        if require_process:
            current = self.snapshot()
            if current["process"] != self.before["process"] or current["identity"] != self.before["identity"]:
                raise RuntimeError("managed service identity or policy changed")
            return current

    def pause(self, check_lease):
        with self.host.transition_guard(self.before["process"]["pid"]) as guard:
            self.save("pause_transition_guard", pause_transition_guard=guard)
            self._pause(check_lease)

    def _pause(self, check_lease):
        current = self.unchanged()
        require_idle_status(current["status"])
        check_lease()
        self.host.idle(self.before["process"]["pid"])
        self.pause_requested = True  # POST may succeed even if its response is lost.
        self.save("quiesce_requested")
        self.host.request(self.service, "POST", "/v1/admin/quiesce", {"drain_timeout_seconds": 120, "suspend": True})
        deadline = self.host.now() + 130
        while True:
            check_lease()
            self.host.idle(self.before["process"]["pid"])
            current = self.unchanged()
            if (current["status"].get("quiesce") or {}).get("state") == "suspended":
                require_idle_status(current["status"], "suspended")
                break
            if self.host.now() >= deadline:
                raise TimeoutError("managed service did not suspend within drain bound")
            self.host.sleep(0.5)
        self.save("suspended", suspended_status=current["status"])
        self.unchanged()
        self.host.idle(self.before["process"]["pid"])
        check_lease()
        self.bootout_attempted = True
        self.save("bootout_requested")
        self.host.action("bootout", self.service)
        deadline = self.host.now() + 30
        while (self.host.launchd(self.service) is not None or self.host.listener_pids(self.service["port"])
               or self.host.process(self.before["process"]["pid"]) == self.before["process"]):
            if self.host.now() >= deadline:
                raise TimeoutError("verified managed service did not stop")
            check_lease()
            self.host.sleep(0.5)
        self.host.idle()
        self.save("stopped")

    def restore(self):
        if not self.pause_requested:
            return
        self.unchanged(require_process=False)
        loaded = self.host.launchd(self.service)
        allowed_pid = None
        if loaded is not None:
            current = self.snapshot()
            if current["identity"] != self.before["identity"]:
                raise RuntimeError("loaded service changed during restoration")
            allowed_pid = current["process"]["pid"]
        deadline = self.host.now() + self.spec["restore_timeout_seconds"]
        while True:
            self.unchanged(require_process=False)
            try:
                with self.host.transition_guard(allowed_pid) as guard:
                    self.save("restore_transition_guard", restore_transition_guard=guard)
                    self._restore()
                    return
            except AdmissionBusy as error:
                if self.host.now() >= deadline:
                    raise TimeoutError("service restoration admission stayed busy: " + str(error)) from error
                if not self.record.get("restore_waiting"):
                    self.save("restore_waiting", restore_waiting=str(error))
                self.host.sleep(1)

    def _restore(self):
        self.unchanged(require_process=False)
        loaded = self.host.launchd(self.service)
        if not self.bootout_attempted:
            # A partial quiesce belongs to the original PID only; never resume
            # an independently replaced listener or changed service.
            self.unchanged()
            self.host.idle(self.before["process"]["pid"])
            self.host.request(self.service, "POST", "/v1/admin/resume", {})
            self.save("resume_requested")
        elif loaded is None:
            self.host.idle()
            if self.host.listener_pids(self.service["port"]):
                raise RuntimeError("foreign listener prevents service restoration")
            self.unchanged(require_process=False)
            self.host.action("bootstrap", self.service)
            self.save("bootstrap_requested")
        else:
            # bootout may have failed before doing anything, or an operator
            # restored the identical job. Verify it before resuming admission.
            current = self.snapshot()
            if current["identity"] != self.before["identity"]:
                raise RuntimeError("loaded service changed during restoration")
            if (current["status"].get("quiesce") or {}).get("state") != "serving":
                self.host.idle(current["process"]["pid"])
                self.host.request(self.service, "POST", "/v1/admin/resume", {})
                self.save("resume_requested")
        deadline = self.host.now() + self.spec["restore_timeout_seconds"]
        last_error = None
        while self.host.now() < deadline:
            self.unchanged(require_process=False)
            try:
                current = self.snapshot()
                if current["identity"] != self.before["identity"]:
                    raise RuntimeError("restored checkpoint/runtime/policy/settings differ")
                # Reopened admission may immediately accept new legitimate
                # requests. Ready/healthy identity is sufficient to restore.
                require_idle_status(current["status"], idle=False)
                self.save("restored", after=current, restored=True)
                return
            except (OSError, RuntimeError, ValueError) as error:
                last_error = str(error)
            self.host.sleep(1)
        raise TimeoutError("managed service did not restore exact ready identity: " + str(last_error))


def stop_wrapper(child, guards, receipt):
    if child is None or child.poll() is not None:
        return
    # The lock wrapper owns the nested child's separate process group. Give
    # its SIGTERM handler time to stop that tree and release both locks.
    receipt["wrapper_stop_requested"] = True
    child.send_signal(signal.SIGTERM)
    try:
        child.wait(timeout=70)
    except subprocess.TimeoutExpired:
        guards.stop_child(child, receipt)
        receipt["wrapper_forced_cleanup"] = True
    if child.poll() is None:
        raise RuntimeError("owned lock wrapper survived cleanup")


def check_lock_release(path, guards, child_pid):
    if not Path(path).is_file():
        raise RuntimeError("paired-lock terminal receipt is missing")
    receipt = json.loads(Path(path).read_text())
    if receipt.get("pid") != child_pid or not receipt.get("finished_at"):
        raise RuntimeError("paired-lock terminal receipt is not bound to this wrapper")
    if receipt.get("release_error") or receipt.get("stop_child_error") or receipt.get("child_descendants_surviving"):
        raise RuntimeError("paired-lock child cleanup/release failed")
    for path in guards.LOCKS:
        if path.is_dir():
            raise RuntimeError("GPU lock is still owned; service restore refused")
    return receipt


def execute(spec, *, host=None, lease_factory=None, spawn=subprocess.Popen):
    host = host or Host()
    if lease_factory is None:
        from mlx2.desktop_campaign_lease import DesktopCPGLease
        lease_factory = DesktopCPGLease
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {"schema": "mlx2.kimi-lora-owned-round.v1", "spec": spec, "started_at": time.time(),
              "implemented": True, "qualified": False, "selected": False, "observed_used": False,
              "script_sha256": sha256(__file__), "lock_script_sha256": sha256(ROOT / "scripts/run_with_gpu_locks.py"),
              "status": "preflight"}
    path = out / "ownership.json"
    atomic(path, record)
    managed = ManagedService(spec, host, out / "service.json")
    child = None
    rc = 1
    try:
        managed.preflight()
        record["host_before"] = host.telemetry()
        atomic(path, record)
        with lease_factory(out, spec["job_id"], mcp_url=spec["mcp_url"], checkpoint=spec["checkpoint"],
                           source_revision=spec["source_revision"]) as lease:
            try:
                managed.pause(lease.check)
                lease.check()
                host.idle()
                command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                           "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
                           "--desktop-cpg-lease", str(out / "desktop-lease.json"),
                           "--mcp-url", spec["mcp_url"], "--receipt", str(out / "paired-locks.json"),
                           "--", *spec["command"]]
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
                with (out / "child.log").open("x") as log:
                    child = spawn(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    record.update(status="running", wrapper_pid=child.pid, command=command, child_started_at=time.time())
                    atomic(path, record)
                    deadline = host.now() + spec["timeout_seconds"]
                    while child.poll() is None:
                        lease.check()
                        if host.now() >= deadline:
                            raise TimeoutError("bounded LoRA round expired")
                        try:
                            child.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            pass
                    rc = child.returncode
            finally:
                cleanup_error = None
                try:
                    stop_wrapper(child, host.guards, record)
                    if child is not None:
                        record["paired_locks"] = check_lock_release(out / "paired-locks.json", host.guards, child.pid)
                except BaseException as error:
                    cleanup_error = error
                    record["cleanup_error"] = f"{type(error).__name__}: {error}"
                try:
                    # A failed cleanup never authorizes loading a second model.
                    if cleanup_error is None:
                        managed.restore()
                    else:
                        raise RuntimeError("restoration deferred because child/lock cleanup is unverified")
                except BaseException as error:
                    record["restore_error"] = f"{type(error).__name__}: {error}"
                    rc = 1
                if cleanup_error is not None:
                    rc = 1
                lease.success = rc == 0 and managed.record.get("restored") is True
                atomic(path, record)
        record.update(status="completed" if rc == 0 else "failed", command_returncode=child.returncode if child else None)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        rc = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        record.update(finished_at=time.time(), returncode=rc, service_restored=managed.record.get("restored", False))
        atomic(path, record)
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    spec = validate_spec(json.loads(args.spec.read_text()))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != spec["source_revision"]:
        raise ValueError("worktree HEAD differs from source_revision")
    if args.check_only:
        before = ManagedService(spec, Host()).preflight()
        print(json.dumps({"checked": True, "service_pid": before["process"]["pid"],
                          "plist_sha256": before["plist_sha256"], "identity": before["identity"]}, sort_keys=True))
        return 0
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received {signal.Signals(signum).name}")
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        return execute(spec)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
