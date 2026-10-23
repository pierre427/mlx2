"""Typed self-lifecycle actions, accepted receipts and one-use managed resume.

No arbitrary PID, signal, executable or command is accepted from a model.
The UI alone supplies its bundle/PID over the private bridge when restarting.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .desktop import artifact_row, atomic_json, source_identity


def checkpoint(path):
    from .adapters.registry import inspect_model
    resolution = inspect_model(str(path))
    return path.name + "@" + resolution.artifact_fingerprint


class Lifecycle:
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / "lifecycle"
        self.resume_path = controller.home / "restart-resume.json"
        self.lock = threading.RLock()
        self.pending = {}
        self.closed = False

    def _spec(self, identity, engine):
        worker = engine.worker or {}
        path = Path(worker["path"])
        row = artifact_row(path)
        engine.library.checked_path(str(path), row["identity"])
        exact = checkpoint(path)
        if worker.get("checkpoint") != exact:
            raise ValueError("Model artifact changed; restart refused")
        return {"profile_id": identity, "path": str(path), "identity": row["identity"], "checkpoint": exact,
                "settings": dict(engine.settings), "profile": self.controller.profile(identity) if identity != "default" else None}

    def _managed(self, target):
        if not isinstance(target, dict) or not target.get("url"):
            raise ValueError("Self lifecycle needs a freshly validated chat target")
        matches = [(identity, engine) for identity, engine in self.controller.engines.items()
                   if engine.worker and engine.worker.get("url") == target["url"] and engine.child is not None]
        if len(matches) != 1:
            raise ValueError("This engine is not owned by this UI; self lifecycle is unavailable")
        identity, engine = matches[0]
        if engine.child.poll() is not None:
            raise ValueError("This engine has already stopped")
        return identity, engine

    def _accepted(self, action, **details):
        self.home.mkdir(mode=0o700, exist_ok=True)
        operation = uuid.uuid4().hex
        receipt = {"operation_id": operation, "action": action, "state": "accepted", "created": time.time(), "source": source_identity(), **details}
        atomic_json(self.home / (operation + ".json"), receipt)
        return receipt

    def dispatch(self, request):
        command = request["command"]
        if command == "ui_close":
            return {"terminal_action": True, "native_action": "ui_close", "message": "Closing the main window. The menu bar app and models keep running."}
        if command == "ui_restart":
            return {"terminal_action": True, "native_action": "ui_restart", "requires_native_restart_prepare": True,
                    "message": "Restarting the UI. Managed engines will stop and reload; external services remain separately managed."}
        if command == "lifecycle_prepare_restart":
            return self.prepare_restart(request["app_path"], request["native_pid"])
        if command not in {"engine_unload_self", "engine_restart_self"}:
            raise ValueError("Unknown lifecycle action")
        with self.lock:
            if self.closed or self.pending:
                raise ValueError("A lifecycle action is already pending or the UI is closing")
            identity, engine = self._managed(request.get("_target"))
            spec = self._spec(identity, engine) if command == "engine_restart_self" else None
            original_child = engine.child
            receipt = self._accepted(command, profile_id=identity, checkpoint=(engine.worker or {}).get("checkpoint"))
            def perform():
                try:
                    if self.closed:
                        receipt["state"] = "cancelled"
                        return
                    # Ownership is bound to the captured child object, never a recycled PID.
                    if engine.child is not original_child:
                        raise ValueError("Engine changed before lifecycle action")
                    engine.stop()
                    if spec is not None:
                        self._restore(spec)
                    receipt["state"] = "completed" if spec is None else "reload_requested"
                except Exception as exc:  # noqa: BLE001 - lifecycle failures must leave durable receipts
                    receipt["state"] = "failed"
                    receipt["error"] = type(exc).__name__ + ": lifecycle action failed"
                finally:
                    receipt["finished"] = time.time()
                    atomic_json(self.home / (receipt["operation_id"] + ".json"), receipt)
                    with self.lock:
                        self.pending.pop(receipt["operation_id"], None)
            timer = threading.Timer(0.3, perform)
            timer.daemon = True
            self.pending[receipt["operation_id"]] = timer
            timer.start()
            return {"terminal_action": True, "operation_id": receipt["operation_id"], "state": "accepted",
                    "message": "Self-unload accepted. This turn ends before the engine stops." if spec is None else "Self-restart accepted. This turn ends before the engine reloads."}

    def _restore(self, spec):
        if self.closed or getattr(self.controller, "stopping", threading.Event()).is_set():
            raise ValueError("UI is stopping; reload cancelled")
        identity = spec["profile_id"]
        engine = self.controller.engines.get(identity)
        if identity != "default" and self.controller.profile(identity) != spec["profile"]:
            raise ValueError("Serving profile changed; automatic restore refused")
        library = self.controller.primary.library
        path = library.checked_path(spec["path"], spec["identity"])
        if checkpoint(path) != spec["checkpoint"]:
            raise ValueError("Model checkpoint changed; automatic restore refused")
        if identity == "default":
            if self.controller.primary.settings != spec["settings"]:
                raise ValueError("Serving settings changed; automatic restore refused")
            self.controller.primary.begin_start(spec["path"], spec["identity"])
        elif engine is None or engine.child is None:
            self.controller.start_profile(identity)
        else:
            engine.begin_start(spec["path"], spec["identity"])

    def prepare_restart(self, app_path, native_pid):
        import psutil
        if type(native_pid) is not int or native_pid != os.getppid():
            raise ValueError("Restart requires the native UI parent process")
        app = Path(app_path).resolve(strict=True)
        expected = app / "Contents/MacOS/MLX2"
        process = psutil.Process(native_pid)
        if app.suffix != ".app" or Path(process.exe()).resolve() != expected.resolve(strict=True):
            raise ValueError("Restart bundle does not match the running UI")
        with self.lock:
            if self.closed or self.pending:
                raise ValueError("Finish the pending lifecycle action first")
            specs = [self._spec(identity, engine) for identity, engine in self.controller.engines.items()
                     if engine.child is not None and engine.child.poll() is None]
            plan = {"id": uuid.uuid4().hex, "created": time.time(), "source": source_identity(), "engines": specs,
                    "scope": "managed engines only; external services and jobs are not restarted"}
            atomic_json(self.resume_path, plan)
            source = Path(__file__).resolve().parents[1]
            environment = dict(os.environ, PYTHONPATH=str(source))
            command = [sys.executable, "-m", "mlx2.desktop_lifecycle", "--wait-pid", str(native_pid),
                       "--creation-time", str(process.create_time()), "--app-path", str(app)]
            try:
                subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True, env=environment)
            except OSError:
                self.resume_path.unlink(missing_ok=True)
                raise
            receipt = self._accepted("ui_restart", resume_id=plan["id"], engine_count=len(specs))
            return {"terminal_action": True, "native_action": "ui_restart", "restart_prepared": True,
                    "operation_id": receipt["operation_id"], "message": "Restart prepared. Owned engines stop and reload after the UI returns; running jobs are not resumed."}

    def startup_resume(self):
        """Consume exactly once before normal saved startup profiles are considered."""
        if not self.resume_path.exists():
            return False
        consumed = self.resume_path.with_name("restart-resume-consumed-" + uuid.uuid4().hex + ".json")
        os.replace(self.resume_path, consumed)
        try:
            if consumed.stat().st_size > 1_000_000:
                raise ValueError("Resume plan exceeds limit")
            plan = json.loads(consumed.read_text())
            if not 0 <= time.time() - plan["created"] <= 300 or not isinstance(plan["engines"], list) or len(plan["engines"]) > 32:
                raise ValueError("Restart plan expired or invalid")
            for spec in plan["engines"]:
                if getattr(self.controller, "stopping", threading.Event()).is_set():
                    break
                self._restore(spec)
                engine = self.controller.engines.get(spec["profile_id"])
                deadline = time.monotonic() + 660
                while engine is not None and time.monotonic() < deadline:
                    if getattr(self.controller, "stopping", threading.Event()).wait(0.2):
                        break
                    if not engine.preparing() and (engine.worker or {}).get("state") not in {"Preparing", "Loading"}:
                        break
        except Exception as exc:  # noqa: BLE001 - never turn stale resume plans into model loads
            self.controller.event("restart_resume_failed", error=type(exc).__name__)
        return True

    def close(self):
        with self.lock:
            self.closed = True
            timers = list(self.pending.values())
            for operation, timer in list(self.pending.items()):
                timer.cancel()
                path = self.home / (operation + ".json")
                if path.exists():
                    receipt = json.loads(path.read_text())
                    if receipt.get("state") == "accepted":
                        receipt["state"] = "cancelled_by_shutdown"
                        receipt["finished"] = time.time()
                        atomic_json(path, receipt)
            self.pending.clear()
        for timer in timers:
            if timer is not threading.current_thread():
                timer.join(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--creation-time", type=float, required=True)
    parser.add_argument("--app-path", required=True)
    args = parser.parse_args(argv)
    import psutil
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            process = psutil.Process(args.wait_pid)
            if process.create_time() != args.creation_time or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.2)
    else:
        return 1
    subprocess.run(["/usr/bin/open", "-a", str(Path(args.app_path).resolve(strict=True)), "--args", "--details"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
