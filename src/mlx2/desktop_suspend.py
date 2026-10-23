"""Managed-worker suspension orchestration; no model or accelerator imports."""
from __future__ import annotations

import copy
import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class Suspension:
    def __init__(self, supervisor):
        self.owner = supervisor
        self.cancel = threading.Event()
        self.thread = None
        self.record = None
        pointer = supervisor.home / "desktop-suspension.json"
        if pointer.is_file():
            from .desktop import read_config
            record = read_config(pointer)
            if record.get("state") == "Saved":
                self.record = record
                # Export was durable but the companion may have exited between
                # retiring the process and writing its final Suspended marker.
                # Never adopt or signal a still-live PID from an old companion.
                import psutil
                try:
                    process = psutil.Process(record["worker"]["resume_previous_pid"])
                    live = process.status() != psutil.STATUS_ZOMBIE
                except psutil.NoSuchProcess:
                    live = False
                if not live:
                    record["state"] = "Suspended"
                    self.record = record
                    self.save()
                else:
                    supervisor.worker = {**record["worker"], "state": "SuspensionRecoveryRequired",
                        "suspension_message": "The previous worker still exists; retire it through its owner before restoring this saved suspension"}
            if record.get("state") == "Suspended":
                self.record = record
                supervisor.worker = {**record["worker"], "state": "Suspended", "suspension_id": record["id"],
                    "suspension_path": str(self.path(record["id"])), "suspend_summary": record.get("apc_receipt", {}), "apc_branches": record.get("apc_receipt", {}).get("branches", {})}

    def path(self, identity):
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid suspension identity")
        return self.owner.home / "suspensions" / identity / "suspension.json"

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def save(self):
        from .desktop import atomic_json
        path = self.path(self.record["id"])
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_json(path, self.record)
        atomic_json(self.owner.home / "desktop-suspension.json", self.record)

    def _capture_operation(self):
        self.operation_child = self.owner.child

    def _assert_operation(self):
        child = getattr(self, "operation_child", None)
        if child is None or self.owner.child is not child:
            raise ValueError("Managed worker ownership changed during suspension")
        poll = getattr(child, "poll", None)
        if callable(poll) and poll() is not None:
            raise ValueError("Managed worker exited during suspension")

    def request(self, method, route, body=None):
        from .http_security import load_secret_file
        child = self.owner.child
        if child is None or (callable(getattr(child, "poll", None)) and child.poll() is not None):
            raise ValueError("Managed metadata endpoint no longer has its owned process")
        worker = dict(self.owner.worker)
        token = load_secret_file(worker["api_key_file"], flag="managed admin key")
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(worker["url"] + route, method=method, data=data,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
                payload = response.read((4 << 20) + 1)
                if len(payload) > 4 << 20:
                    raise ValueError("Managed suspension response exceeds its bound")
                value = json.loads(payload)
        except urllib.error.HTTPError as error:
            raise ValueError("Managed suspension request refused (HTTP " + str(error.code) + "); the current worker may need restarting with suspension support") from None
        if self.owner.child is not child or any(self.owner.worker.get(key) != worker.get(key) for key in ("url", "api_key_file")):
            raise ValueError("Managed metadata endpoint changed while receiving its receipt")
        if callable(getattr(child, "poll", None)) and child.poll() is not None:
            raise ValueError("Managed metadata process exited before receipt validation")
        if not isinstance(value, dict):
            raise TypeError("Managed server returned an invalid suspension receipt")
        return value

    def begin_suspend(self):
        owner = self.owner
        with owner.lifecycle_lock:
            owner.reap()
            if self.busy() or owner.preparing():
                raise ValueError("Wait for the current model lifecycle operation")
            if owner.child is None or not owner.worker:
                raise ValueError("Suspend requires this app's live managed model; external workers are not controlled")
            if owner.worker.get("state") not in {"Ready", "Busy"}:
                raise ValueError("Wait until the model is ready before suspending")
            if owner.worker.get("execution_kind") != "autoregressive" or not owner.worker.get("apc_persist_dir"):
                raise ValueError("Restart this managed autoregressive worker with APCv2 suspension support")
            from .desktop import artifact_row
            row = artifact_row(Path(owner.worker["path"]))
            worker = copy.deepcopy(owner.worker)
            worker.update(resume_previous_pid=owner.child.pid, resume_previous_url=worker["url"], resume_previous_checkpoint=worker["checkpoint"], identity=row["identity"])
            identity = uuid.uuid4().hex
            self.record = {"schema": "mlx2.desktop.suspension.v1", "id": identity, "state": "Suspending", "created_at": time.time(),
                "settings": dict(owner.settings), "worker": worker}
            self.cancel.clear()
            self._capture_operation()
            owner.worker_error = None
            owner.worker.update(state="Suspending", suspension_id=identity, suspension_message="Draining active work and saving APCv2")
            self.save()
            self.thread = threading.Thread(target=self.suspend, daemon=True, name="mlx2-model-suspend")
            self.thread.start()
            return dict(owner.worker)

    def suspend(self):
        owner = self.owner
        try:
            self._assert_operation()
            live_status = self.request("GET", "/v1/status")
            expected_artifact = self.record["worker"]["checkpoint"].split("@", 1)[1]
            if live_status.get("artifact") != expected_artifact or not isinstance(live_status.get("runtime"), dict) or not live_status["runtime"]:
                raise ValueError("Live worker does not match its managed artifact/runtime identity")
            self.record["runtime"] = live_status["runtime"]
            self.request("POST", "/v1/admin/quiesce", {"drain_timeout_seconds": 600, "suspend": True})
            deadline = time.monotonic() + 620
            while True:
                if self.cancel.is_set():
                    raise InterruptedError("Suspension cancelled before process unload")
                status = self.request("GET", "/v1/admin/state")
                self.record["service_state"] = status
                if self.cancel.is_set():
                    raise InterruptedError("Suspension cancelled while waiting for drain")
                self._assert_operation()
                if status.get("process_id") != owner.child.pid:
                    raise ValueError("Managed drain receipt belongs to another process")
                if status.get("state") == "suspended":
                    break
                if status.get("state") in {"failed", "error"} or time.monotonic() >= deadline:
                    raise ValueError("Model could not finish draining and saving APCv2; process remains owned")
                owner.worker["suspension_message"] = "Draining active work · " + str(status.get("state", "unknown"))
                self.cancel.wait(.25)
            suspend = (status.get("last_transition", {}).get("result", {}).get("suspend") or {})
            if suspend.get("failures") != 0 or suspend.get("resident_entries_after") != 0:
                raise ValueError("Runtime suspend did not confirm all resident APCv2 entries safely persisted")
            receipt = self.request("POST", "/v1/admin/apc/suspend-export", {"suspension_id": self.record["id"]})
            if receipt.get("suspension_id") != self.record["id"] or receipt.get("persistence_dir") != owner.worker["apc_persist_dir"] or type(receipt.get("frozen_at")) not in (int, float) or not math.isfinite(receipt["frozen_at"]):
                raise ValueError("Runtime did not confirm expiry-frozen APCv2 persistence; process retained")
            if receipt.get("process_id") != owner.child.pid:
                raise ValueError("Saved APCv2 receipt belongs to another process")
            if receipt.get("artifact") != expected_artifact or receipt.get("runtime") != self.record["runtime"] or type(receipt.get("entries")) is not int or receipt["entries"] < 0:
                raise ValueError("Saved APCv2 receipt does not match the exact artifact/runtime identity")
            inventory = receipt.get("inventory_sha256")
            if not isinstance(inventory, str) or len(inventory) != 64 or any(c not in "0123456789abcdef" for c in inventory):
                raise ValueError("Saved APCv2 receipt lacks an exact retained-entry inventory")
            loras = receipt.get("loras", [])
            if not isinstance(loras, list) or len(loras) > 64:
                raise ValueError("Saved LoRA registry exceeds its bound")
            from .runtime.apc_suspension import verify_lora_receipt
            for adapter in loras:
                verify_lora_receipt(adapter, self.record["settings"].get("lora_root"))
            self.record["apc_receipt"] = receipt
            if self.cancel.is_set():
                raise InterruptedError("Suspension cancelled before process unload")
            self.record["state"] = "Saved"
            self.save()  # crash-safe proof before retiring the process
            with owner.lifecycle_lock:
                self._assert_operation()
                owner.stop_worker()
                self.record.update(state="Suspended", suspended_at=time.time())
                self.save()
                owner.worker = {**self.record["worker"], "state": "Suspended", "suspension_id": self.record["id"],
                    "suspension_path": str(self.path(self.record["id"])), "suspend_summary": receipt, "apc_branches": receipt.get("branches", {}),
                    "suspension_message": "APCv2 saved; process unloaded and GPU ownership released"}
        except Exception as error:  # noqa: BLE001 -- asynchronous lifecycle result remains inspectable
            self.record.update(state="Cancelled" if isinstance(error, InterruptedError) else "Error", error=str(error))
            self.save()
            same_owner = owner.child is getattr(self, "operation_child", None)
            if same_owner:
                owner.worker_error = str(error)
                if owner.worker:
                    owner.worker.update(state="SuspendError", suspension_message=str(error))
            # A cancelled/failed quiesce must reopen admission when possible;
            # ownership is retained until an explicit Stop if recovery fails.
            if same_owner and owner.child is not None:
                try:
                    self.request("POST", "/v1/admin/resume", {})
                except Exception:  # noqa: BLE001 -- original persistence failure remains primary
                    owner.worker["suspension_message"] += "; server admission recovery failed, use Stop"

    def begin_resume(self, identity=None):
        owner = self.owner
        with owner.lifecycle_lock:
            owner.reap()
            if self.busy() or owner.child is not None or owner.preparing():
                raise ValueError("Stop the current model or wait for its lifecycle operation before resuming")
            if not self.record or self.record.get("state") != "Suspended" or (identity is not None and identity != self.record["id"]):
                raise ValueError("No matching completed suspension is available")
            record = copy.deepcopy(self.record)
            worker = record["worker"]
            from .runtime.apc_suspension import verify_lora_receipt
            for adapter in record["apc_receipt"].get("loras", []):
                verify_lora_receipt(adapter, record["settings"].get("lora_root"))
            if owner.settings != record["settings"]:
                raise ValueError("Launch settings changed since suspension; restore them before resuming")
            owner.library.checked_path(worker["path"], worker["identity"])
            from .desktop_runner import digest, source_manifest
            runner = worker["runner"]
            if digest(source_manifest(Path(runner["source"]))) != runner["binding"].get("source_sha256") or not Path(runner["python"]).is_file():
                raise ValueError("The suspended model's frozen runner changed; resume refused")
            from .adapters.registry import inspect_model
            resolution = inspect_model(Path(worker["path"]))
            if worker["checkpoint"] != Path(worker["path"]).name + "@" + resolution.artifact_fingerprint:
                raise ValueError("Model checkpoint changed since suspension; resume refused")
            context = owner.context_resolver(Path(worker["path"]), owner.settings["context"]) if owner.context_resolver else owner.settings["context"]
            if context != worker.get("effective_context"):
                raise ValueError("Effective context configuration changed since suspension; resume refused")
            self.cancel.clear()
            owner.preparation_cancel.clear()
            owner.worker = {**worker, "state": "Resuming", "suspension_id": record["id"], "suspension_message": "Restoring the same artifact, runtime and frozen APCv2 state"}
            def resume():
                try:
                    with owner.lifecycle_lock:
                        if self.cancel.is_set():
                            raise InterruptedError("Resume cancelled")
                        owner.launch(Path(worker["path"]), resolution, worker["execution_kind"], runner, resume_worker=worker)
                        self._capture_operation()
                        owner.worker.update(state="Resuming", resume_lineage_id="pending:" + uuid.uuid4().hex,
                            suspension_id=record["id"], suspension_message="Waiting for exact APCv2 restore confirmation")
                        for field in ("resume_previous_pid", "resume_previous_url", "resume_previous_checkpoint"):
                            owner.worker.pop(field, None)
                        self.record.update(state="Resuming", resume_started_at=time.time())
                        self.save()
                    deadline = time.monotonic() + 600
                    while True:
                        if self.cancel.is_set():
                            raise InterruptedError("Resume cancelled; use Stop to retire the worker")
                        owner.reap()
                        if owner.child is None:
                            raise ValueError("Resumed worker exited before cache restoration was confirmed")
                        try:
                            status = self.request("GET", "/v1/status")
                        except (OSError, urllib.error.URLError):
                            status = {}
                        if status.get("state") == "ready":
                            self._assert_operation()
                            live_state = self.request("GET", "/v1/admin/state")
                            if live_state.get("process_id") != owner.child.pid:
                                raise ValueError("Resume metadata belongs to another process")
                            receipt = record["apc_receipt"]
                            rescan = status.get("apcv2", {}).get("persistence", {}).get("rescan", {})
                            registered = rescan.get("registered")
                            if (status.get("artifact") != receipt["artifact"] or status.get("runtime") != receipt["runtime"]
                                    or type(registered) is not int or registered != receipt["entries"]
                                    or rescan.get("inventory_sha256") != receipt["inventory_sha256"]
                                    or rescan.get("suspension_epochs", {}).get(record["id"], 0) != receipt["entries"]):
                                owner.worker["cache_restore"] = {"confirmed": False, "expected_entries": receipt["entries"], "registered_entries": registered}
                                raise ValueError("APCv2 restore was partial or incompatible; chat resume lineage was not restored")
                            adapters = receipt.get("loras", [])
                            for adapter in adapters:
                                verify_lora_receipt(adapter, record["settings"].get("lora_root"))
                                loaded = self.request("POST", "/v1/load_lora_adapter", {"lora_name": adapter["name"], "lora_path": adapter["path"]})
                                if loaded.get("id") != adapter["name"] or loaded.get("fingerprint") != adapter["fingerprint"]:
                                    raise ValueError("Restored LoRA adapter did not match its suspended fingerprint")
                            if adapters:
                                status = self.request("GET", "/v1/status")
                                restored_names = (status.get("multi_lora") or {}).get("registered", [])
                                if (not (status.get("multi_lora") or {}).get("enabled")
                                        or set(restored_names) != {adapter["name"] for adapter in adapters}
                                        or status.get("runtime") != receipt["runtime"] or status.get("artifact") != receipt["artifact"]):
                                    raise ValueError("Restored LoRA registry was not confirmed; chat lineage remains blocked")
                            self._assert_operation()
                            owner.worker.update(state="Ready", resume_lineage_id=worker["resume_lineage_id"],
                                cache_restore={"confirmed": True, "expected_entries": receipt["entries"], "registered_entries": registered},
                                suspension_message="Exact artifact/runtime and retained APCv2 entries restored",
                                **{key: worker[key] for key in ("resume_previous_pid", "resume_previous_url", "resume_previous_checkpoint")})
                            self.record.update(state="Resumed", resumed_at=time.time(), restore=owner.worker["cache_restore"])
                            self.save()
                            break
                        if time.monotonic() >= deadline:
                            raise ValueError("Resumed model did not become ready within ten minutes")
                        self.cancel.wait(.25)
                except Exception as error:  # noqa: BLE001 -- durable resumable failure
                    owner.worker_error = str(error)
                    owner.worker = {**(owner.worker or worker), "state": "ResumeError" if owner.child is not None else "Suspended", "suspension_id": record["id"], "suspension_message": str(error)}
                    if self.record.get("state") == "Resuming":
                        self.record.update(state="ResumeError", error=str(error))
                        self.save()
            self.thread = threading.Thread(target=resume, daemon=True, name="mlx2-model-resume")
            self.thread.start()
            return dict(owner.worker)
