"""Explicit, bounded agent launches on exact app-managed model instances.

Construction never loads a model. Launch receipts bind the selected artifact,
listener, user profile and optional LoRA adapter; parent histories stay in memory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

from .desktop import atomic_json, source_identity

ACTIVE = {"queued", "loading", "dispatching", "running"}
PRIVATE = {"cancel", "thread", "history", "child", "engine", "preparation", "created_profile", "principal"}


class AgentLaunch:
    def __init__(self, controller, *, inspector=None, clock=time.monotonic, poll_seconds=0.25):
        self.controller, self.inspector, self.clock = controller, inspector, clock
        self.poll_seconds = poll_seconds
        self.home = controller.home / "agent-launches"
        self.home.mkdir(mode=0o700, exist_ok=True)
        if self.home.is_symlink():
            raise ValueError("Agent launch receipts cannot be a symbolic link")
        self.lock = threading.RLock()
        self.closed = False
        self.jobs = {}
        for path in sorted(self.home.glob("*.json"))[-100:]:
            try:
                if path.is_symlink() or path.stat().st_size > 2_000_000:
                    continue
                row = json.loads(path.read_text())
                if row.get("id") != path.stem:
                    continue
                if row.get("state") in ACTIVE:
                    row["state"] = "interrupted"
                    atomic_json(path, row)
                row["cancel"] = threading.Event()
                # Principal is deliberately retained privately for model-scope reads.
                row["principal"] = row.pop("owner", None)
                self.jobs[row["id"]] = row
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _profile(self):
        return getattr(getattr(self.controller, "user_profiles", None), "active", "")

    @staticmethod
    def _public(job):
        return copy.deepcopy({key: value for key, value in dict(job).items() if key not in PRIVATE})

    def _save(self, job):
        atomic_json(self.home / (job["id"] + ".json"), {**self._public(job), "owner": job.get("principal")})

    def _visible(self, job, principal):
        return job.get("user_profile", "") == self._profile() and (principal is None or job.get("principal") == principal)

    def snapshot(self, *, principal=None):
        with self.lock:
            return {"jobs": [self._public(row) for row in self.jobs.values() if self._visible(row, principal)]}

    def dispatch(self, request, *, principal=None):
        if not isinstance(request, dict):
            raise TypeError("Agent launch request must be an object")
        if principal is not None and (not isinstance(principal, str) or not principal or len(principal) > 256):
            raise ValueError("Invalid agent launch principal")
        command = request.get("command")
        if command in {"agent_launch_status", "agent_launch_cancel"}:
            identity = request.get("job_id")
            with self.lock:
                job = self.jobs.get(identity)
                if job is None or not self._visible(job, principal):
                    raise ValueError("Unknown agent launch")
                if command == "agent_launch_cancel" and job["state"] in ACTIVE:
                    job["cancel"].set()
                return self._public(job)
        if command != "agent_launch":
            raise ValueError("Unknown agent launch command")
        allowed = {"command", "path", "identity", "goal", "instance", "profile_id", "lanes", "context",
                   "max_tokens", "lora_path", "lora_name", "max_lora_rank", "tools", "deadline_seconds", "_history", "parent_id"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("Unknown agent launch fields")
        if not self.controller.preferences.get("agent_delegation_enabled", True):
            raise ValueError("Agent delegation is disabled")
        goal = request.get("goal")
        if not isinstance(goal, str) or not goal.strip() or len(goal.encode()) > 16384:
            raise ValueError("Supply one goal of at most 16 KiB")
        instance = request.get("instance", "new")
        if instance not in ("new", "reuse"):
            raise ValueError("Instance must be new or reuse")
        selected_profile = request.get("profile_id")
        if selected_profile is not None and (not isinstance(selected_profile, str) or not selected_profile or len(selected_profile) > 64):
            raise ValueError("Invalid serving profile identifier")
        if selected_profile is not None and instance != "reuse":
            raise ValueError("profile_id is only used when reusing an instance")
        for key, low, high, default in (("lanes", 1, 4, 1), ("context", 0, 1048576, 0),
                                         ("max_tokens", 128, 8192, 2048), ("deadline_seconds", 1, 1800, 900),
                                         ("max_lora_rank", 1, 1024, 16)):
            value = request.get(key, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} must be an integer in {low}..{high}")
        parent = request.get("parent_id", "ui")
        if not isinstance(parent, str) or not parent or len(parent) > 1024:
            raise ValueError("Invalid parent task identity")
        raw, identity = request.get("path"), request.get("identity")
        if not isinstance(raw, str) or not isinstance(identity, str):
            raise TypeError("Select a current model library entry")
        path = self.controller.primary.library.checked_path(raw, identity)
        resolution = self._inspect(path)
        if resolution.descriptor.metadata.get("execution_kind", "autoregressive") != "autoregressive":
            raise ValueError("Agent launch requires a supported autoregressive chat model")
        lora = request.get("lora_path")
        if lora is not None:
            if not isinstance(lora, str) or not lora:
                raise ValueError("LoRA path must be a directory")
            candidate = Path(lora).expanduser()
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError("LoRA path must be a real adapter directory")
            lora = str(candidate.resolve(strict=True))
            name = request.get("lora_name")
            if not isinstance(name, str) or not 1 <= len(name) <= 128 or not all(c.isalnum() or c in "-_" for c in name):
                raise ValueError("LoRA needs a bounded adapter name")
        elif request.get("lora_name") is not None:
            raise ValueError("A LoRA name requires an explicit adapter path")
        from .desktop_agents import Agents
        Agents._validate_tools(request.get("tools", []))
        history = request.get("_history")
        if history is not None:
            from .desktop_agents import Agents
            history = Agents._validate_seed(history)
        spec = {key: request[key] for key in ("profile_id", "lanes", "context", "max_tokens", "max_lora_rank", "tools") if key in request}
        spec.update(path=str(path), identity=identity, instance=instance, lora_path=lora, lora_name=request.get("lora_name"))
        with self.lock:
            if self.closed or sum(job["state"] in ACTIVE for job in self.jobs.values()) >= 4:
                raise ValueError("At most four independent agent launches may run; wait or cancel one")
            job_id = uuid.uuid4().hex
            row = {"id": job_id, "state": "queued", "created": time.time(), "user_profile": self._profile(),
                   "principal": principal, "parent_id": parent, "goal": goal, "spec": spec,
                   "checkpoint": path.name + "@" + resolution.artifact_fingerprint, "source": source_identity(),
                   "cancel": threading.Event(), "history": history, "seed_kind": "parent_history" if history is not None else "goal_only", "deadline": self.clock() + request.get("deadline_seconds", 900),
                   "child": None, "engine": None, "created_profile": None, "preparation": None}
            if history is not None:
                row["seed_receipt"] = {"messages": len(history), "sha256": hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()}
            self.jobs[job_id] = row
            self._save(row)
            row["thread"] = threading.Thread(target=self._run, args=(row,), daemon=True, name="mlx2-agent-launch")
            row["thread"].start()
            return {"job_id": job_id, "state": "queued", "checkpoint": row["checkpoint"]}

    def _inspect(self, path):
        if self.inspector is not None:
            return self.inspector(path)
        from .adapters.registry import inspect_model
        return inspect_model(path)

    def _guard(self, job):
        if self.closed or job["cancel"].is_set() or job["user_profile"] != self._profile():
            raise InterruptedError("Agent launch cancelled or user profile changed")
        if self.clock() >= job["deadline"]:
            raise TimeoutError("Agent launch exceeded its bounded wait")
        if getattr(self.controller, "stopping", threading.Event()).is_set():
            raise InterruptedError("Application is stopping")

    def _current_artifact(self, job):
        spec = job["spec"]
        path = self.controller.primary.library.checked_path(spec["path"], spec["identity"])
        if path.name + "@" + self._inspect(path).artifact_fingerprint != job["checkpoint"]:
            raise ValueError("Model artifact changed during agent launch")

    def _select(self, job):
        spec = job["spec"]
        if spec["instance"] == "reuse":
            matches = []
            for identity, engine in self.controller.engines.items():
                if spec.get("profile_id") is not None and identity != spec["profile_id"]:
                    continue
                worker = engine.worker or {}
                if (worker.get("checkpoint") == job["checkpoint"] and worker.get("state") in {"Ready", "Busy"}
                        and engine.child is not None and engine.child.poll() is None):
                    if any(key in spec and engine.settings.get(key) != spec[key] for key in ("lanes", "context")):
                        continue
                    matches.append((identity, engine))
            if len(matches) != 1:
                raise ValueError("Choose exactly one ready matching app-managed instance to reuse")
            return matches[0]
        if spec.get("profile_id") is not None:
            raise ValueError("A new instance creates its own distinct profile; profile_id is for reuse")
        identity = "agent-" + job["id"]
        profile = {"id": identity, "path": spec["path"], "port": 8285, "context": spec.get("context", 0),
                   "lanes": spec.get("lanes", 1), "route": "ordinary" if spec["lora_path"] else "auto",
                   "autostart": False, "share": False, "sticky_port": False}
        if spec["lora_path"]:
            profile.update(lora_root=spec["lora_path"], max_loras=1, max_lora_rank=spec.get("max_lora_rank", 16))
        with self.controller.lock:
            self._guard(job)
            self.controller.configure_preferences({"profiles": [*copy.deepcopy(self.controller.preferences["profiles"]), profile]})
            job["created_profile"] = profile
            job["profile_id"] = identity
            self._guard(job)
            self.controller.start_profile(identity)
            engine = self.controller.engines[identity]
            job["engine"], job["preparation"] = engine, getattr(engine, "preparation_thread", None)
        return identity, engine

    def _cleanup(self, job):
        engine = job.get("engine")
        if job.get("created_profile") is None:
            return
        if self.controller.profile(job["profile_id"]) != job["created_profile"]:
            return
        if engine is not None and job.get("child") is not None:
            if engine.child is not job["child"]:
                return
        elif engine is not None and getattr(engine, "preparation_thread", None) is not job.get("preparation"):
            return
        if engine is not None:
            engine.stop()
        with self.controller.lock:
            if self.controller.profile(job["profile_id"]) == job["created_profile"]:
                self.controller.configure_preferences({"profiles": [profile for profile in self.controller.preferences["profiles"]
                                                                    if profile["id"] != job["profile_id"]]})
                if engine is not None and self.controller.engines.get(job["profile_id"]) is engine:
                    closer = getattr(engine, "close", None)
                    if closer is not None:
                        closer()
                    self.controller.engines.pop(job["profile_id"])

    def _run(self, job):
        try:
            self._guard(job)
            self._current_artifact(job)
            job["state"] = "loading"
            identity, engine = self._select(job)
            job["profile_id"], job["engine"] = identity, engine
            self._save(job)
            while True:
                self._guard(job)
                engine.snapshot()
                worker = engine.worker or {}
                if job["child"] is None and engine.child is not None:
                    job["child"] = engine.child
                if job["child"] is not None and engine.child is not job["child"]:
                    raise ValueError("Managed model instance changed during launch")
                if worker.get("state") in {"Ready", "Busy"} and engine.child is not None:
                    if engine.child.poll() is not None or worker.get("checkpoint") != job["checkpoint"]:
                        raise ValueError("Ready worker does not match the selected checkpoint")
                    break
                if worker.get("state") in {"Error", "Stopped", "Suspended", "Suspending", "Resuming"}:
                    raise ValueError("Selected model instance did not become ready")
                job["cancel"].wait(self.poll_seconds)
            self._current_artifact(job)
            self._guard(job)
            target = {"url": worker["url"], "pid": engine.child.pid}
            spec = job["spec"]
            before = engine.suspension.request("GET", "/v1/status")
            fingerprint = before.get("artifact")
            if isinstance(fingerprint, dict):
                fingerprint = fingerprint.get("fingerprint")
            if fingerprint != job["checkpoint"].split("@", 1)[1] or not isinstance(before.get("runtime"), dict) or not before["runtime"]:
                raise ValueError("Listener did not prove the exact artifact and runtime")
            job["runtime"] = before["runtime"]
            if spec["lora_path"]:
                if engine.settings.get("route") != "ordinary" or engine.settings.get("lora_root") != spec["lora_path"]:
                    raise ValueError("LoRA requires its exact allowlisted directory on the ordinary route")
                if not (before.get("multi_lora") or {}).get("enabled"):
                    raise ValueError("The selected runtime does not support concurrent LoRA")
                self._guard(job)
                if spec["lora_name"] in (before.get("multi_lora") or {}).get("registered", []):
                    from .runtime.apc_suspension import lora_file_receipt
                    source = lora_file_receipt(spec["lora_path"])
                    registered = (before.get("multi_lora") or {}).get("registered_fingerprints", {}).get(spec["lora_name"])
                    if registered != source["fingerprint"]:
                        raise ValueError("Already registered LoRA does not prove the requested adapter fingerprint")
                    receipt = {"id": spec["lora_name"], "fingerprint": registered, "status": "reused", "mode": "concurrent"}
                else:
                    receipt = engine.suspension.request("POST", "/v1/load_lora_adapter",
                        {"lora_name": spec["lora_name"], "lora_path": spec["lora_path"]})
                self._guard(job)
                after = engine.suspension.request("GET", "/v1/status")
                if (after.get("artifact") != before.get("artifact") or after.get("runtime") != job["runtime"]
                        or receipt.get("id") != spec["lora_name"] or not receipt.get("fingerprint")
                        or spec["lora_name"] not in (after.get("multi_lora") or {}).get("registered", [])
                        or ((after.get("multi_lora") or {}).get("registered_fingerprints", {}).get(spec["lora_name"], receipt["fingerprint"]) != receipt["fingerprint"])):
                    raise ValueError("LoRA runtime did not confirm the exact adapter load")
                job["lora_receipt"] = receipt
                target["model"] = spec["lora_name"]
            self._guard(job)
            if engine.child is not job["child"]:
                raise ValueError("Listener ownership changed before dispatch")
            job["listener"] = dict(target)
            job["state"] = "dispatching"
            request = {"command": "agent_fork", "target": target, "parent_id": job["parent_id"],
                       "max_tokens": spec.get("max_tokens", 2048), "goal": job["goal"],
                       "tools": spec.get("tools", []), "_history": job["history"] if job["history"] is not None else [{"role": "user", "content": job["goal"]}]}
            result = self.controller.agents.dispatch(request)
            job["agent"] = result
            job["state"] = "running"
            self._save(job)
            while True:
                self._guard(job)
                result = self.controller.agents.dispatch({"command": "agent_result", "job_id": job["agent"]["job_id"]})
                job["agent_state"] = result.get("status")
                if result.get("status") not in {"queued", "running", "synthesizing"}:
                    job["agent_result"] = self._bounded_agent_result(result)
                    job["state"] = "completed" if result.get("status") == "completed" else "failed"
                    break
                job["cancel"].wait(self.poll_seconds)
        except Exception as error:  # noqa: BLE001 -- durable asynchronous failure receipt
            job["state"] = "cancelled" if isinstance(error, InterruptedError) else "failed"
            job["error"] = str(error)
            if job.get("agent"):
                try:
                    self.controller.agents.dispatch({"command": "agent_cancel", "job_id": job["agent"]["job_id"]})
                except Exception as cancel_error:  # noqa: BLE001 -- cleanup must continue
                    job["agent_cancel_error"] = type(cancel_error).__name__
            try:
                self._cleanup(job)
            except Exception as cleanup_error:  # noqa: BLE001 -- retain ownership failure for the operator
                job["cleanup_error"] = type(cleanup_error).__name__
        finally:
            try:
                self._cleanup(job)
            except Exception as cleanup_error:
                job["cleanup_error"] = type(cleanup_error).__name__
            job.pop("history", None)
            job["finished"] = time.time()
            self._save(job)
            self.controller.event("agent_launch_finished", job_id=job["id"], state=job["state"])

    @staticmethod
    def _bounded_agent_result(result):
        # Agent results are already public receipts, but never duplicate a
        # transport's private context if a provider adds one in the future.
        clean = {key: value for key, value in result.items()
                 if key not in {"_history", "history", "target", "responses", "thread", "cancel"}}
        encoded = json.dumps(clean, ensure_ascii=False, allow_nan=False).encode()
        if len(encoded) <= 256 * 1024:
            return copy.deepcopy(clean)
        return {"id": clean.get("id"), "status": clean.get("status"), "truncated": True,
                "total_bytes": len(encoded), "preview": encoded[:64 * 1024].decode("utf8", "ignore"),
                "notice": "Full child result remains in its agent task receipt; this launch receipt retains a bounded preview."}

    def close(self):
        with self.lock:
            self.closed = True
            jobs = list(self.jobs.values())
        for job in jobs:
            job["cancel"].set()
        for job in jobs:
            thread = job.get("thread")
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=45)
