"""Bounded desktop controls for the app's exact, currently owned engine.

Long admin operations run off the JSON command thread. Receipts are polled by
operation ID; neither transport timeouts nor pending trials are called success.
This module imports no model or accelerator framework.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _child_exited(child):
    # Only Supervisor.stop_group may reap a managed process-group leader.
    from .desktop import child_exited
    return child_exited(child)


class RuntimeRefusal(ValueError):
    def __init__(self, status, receipt):
        self.status = status
        self.receipt = receipt
        error = receipt.get("error", {})
        super().__init__(error.get("message", str(error)) if isinstance(error, dict) else str(error))


class DesktopRuntime:
    ROUTES = {
        "load_lora": "/v1/load_lora_adapter",
        "unload_lora": "/v1/unload_lora_adapter",
        "reload_kernels": "/v1/admin/kernels/reload",
        "weight_space": "/v1/admin/weight-space/call",
    }

    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.operations = {}
        self.threads = {}
        self.closed = False

    def _owner(self, request):
        engine = self.controller.engines.get(request.get("profile_id", "default"))
        if engine is None:
            raise ValueError("Choose an app-managed engine in Runtime")
        with engine.lifecycle_lock:
            child, worker = engine.child, dict(engine.worker or {})
            if child is None or _child_exited(child) or worker.get("state") not in {"Ready", "Busy"}:
                raise ValueError("Wait for this managed engine to be ready")
            if worker.get("execution_kind") != "autoregressive":
                raise ValueError("These runtime controls require an autoregressive engine")
            if type(request.get("pid")) is not int or request["pid"] != child.pid or request.get("url") != worker.get("url"):
                raise ValueError("The engine changed; refresh Runtime before trying again")
            # Re-discover the listener before sending the private admin credential.
            engine.chat_target(worker["url"], child.pid)
            return engine, child, worker

    @staticmethod
    def _unchanged(owner):
        engine, child, worker = owner
        if engine.child is not child or _child_exited(child) or any(
            (engine.worker or {}).get(key) != worker.get(key)
            for key in ("url", "checkpoint", "admin_token_file")
        ):
            raise ValueError("The engine changed during this operation; its outcome is unknown")

    def _request(self, owner, method, route, body=None):
        from .http_security import load_secret_file
        self._unchanged(owner)
        engine, child, worker = owner
        engine.chat_target(worker["url"], child.pid)
        token = load_secret_file(Path(worker["admin_token_file"]), flag="managed admin key")
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = urllib.request.Request(worker["url"] + route, data=data, method=method,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        def read(response):
            payload = response.read((4 << 20) + 1)
            if len(payload) > 4 << 20:
                raise ValueError("Runtime response exceeds 4 MiB")
            result = json.loads(payload)
            if not isinstance(result, dict):
                raise ValueError("Runtime response must be an object")
            return result

        # Never send the local admin token through an inherited HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=5 if method == "GET" else 30) as response:
                result = read(response)
        except urllib.error.HTTPError as error:
            try:
                result = read(error)
            except (ValueError, UnicodeError):
                result = {"error": {"message": "Runtime request refused (HTTP " + str(error.code) + ")"}}
            self._unchanged(owner)
            raise RuntimeRefusal(error.code, result) from None
        self._unchanged(owner)
        return result

    def status(self, request):
        owner = self._owner(request)
        status = self._request(owner, "GET", "/v1/status")
        kernels = self._request(owner, "GET", "/v1/admin/kernels")
        try:
            weights = self._request(owner, "GET", "/v1/admin/weight-space/capabilities")
        except RuntimeRefusal as error:
            if error.status != 404:
                raise
            weights = {"enabled": False, "message": str(error)}
        return {"status": status, "kernels": kernels, "weight_space": weights,
                "pid": owner[1].pid, "url": owner[2]["url"],
                "checkpoint": owner[2].get("checkpoint")}

    def begin(self, request):
        action, body = request.get("action"), request.get("body")
        if action not in self.ROUTES or not isinstance(body, dict):
            raise ValueError("Choose a supported Runtime action with an object request")
        payload = json.dumps(body, allow_nan=False).encode()
        if len(payload) > (65536 if action == "reload_kernels" else 1 << 20):
            raise ValueError("Runtime request exceeds its size limit")
        owner = self._owner(request)
        profile = request.get("profile_id", "default")
        with self.lock:
            if self.closed:
                raise ValueError("Runtime controls have closed")
            if any(value["pending"] and value["profile_id"] == profile for value in self.operations.values()):
                raise ValueError("A Runtime operation is already pending for this engine")
            # Bound history while retaining pending outcomes.
            for key in list(self.operations):
                if len(self.operations) < 64:
                    break
                if not self.operations[key]["pending"]:
                    self.operations.pop(key)
                    self.threads.pop(key, None)
            if len(self.operations) >= 64:
                raise ValueError("Too many pending Runtime operations")
            identity = uuid.uuid4().hex
            record = {"id": identity, "profile_id": profile, "pid": owner[1].pid,
                "url": owner[2]["url"], "checkpoint": owner[2].get("checkpoint"),
                "action": action, "pending": True, "state": "running", "started_at": time.time()}
            self.operations[identity] = record

            def run():
                try:
                    receipt = self._request(owner, "POST", self.ROUTES[action], json.loads(payload))
                    # A submitted repair trial is still pending inside the engine.
                    result = {"state": "submitted" if receipt.get("pending") else "completed", "receipt": receipt}
                except RuntimeRefusal as error:
                    result = {"state": "failed", "error": str(error), "receipt": error.receipt, "http_status": error.status,
                        "guidance": "Inspect the server receipt before retrying; a server error can follow a partially applied change."}
                except Exception as error:  # Preserve unknown outcomes; never replay a mutation.
                    result = {"state": "unknown", "error": str(error),
                        "guidance": "Refresh engine status or poll the repair attempt before retrying. The request may still be running."}
                with self.lock:
                    record.update(result, pending=False, finished_at=time.time())

            thread = threading.Thread(target=run, daemon=True, name="mlx2-runtime-" + identity[:8])
            self.threads[identity] = thread
            thread.start()
            return copy.deepcopy(record)

    def dispatch(self, request):
        if request["command"] == "runtime_status":
            return self.status(request)
        if request["command"] == "runtime_action":
            return self.begin(request)
        if request["command"] == "runtime_operation":
            with self.lock:
                record = self.operations.get(request.get("operation_id"))
                if record is None:
                    raise ValueError("Runtime operation receipt is no longer available; refresh engine status")
                return copy.deepcopy(record)
        raise ValueError("Unknown Runtime command")

    def close(self):
        with self.lock:
            self.closed = True
            threads = list(self.threads.values())
        deadline = time.monotonic() + 35
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
