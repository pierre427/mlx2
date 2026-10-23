"""Desktop client for Studio's shared, durable media services.

No media runtime is imported or model loaded by this module. Existing MCP job
owners keep their accelerator/lifecycle contracts, including YuE's mlx2 handoff.
"""
from __future__ import annotations

import json
import os
import plistlib
import re
import subprocess
import threading
from pathlib import Path

from .tool_backend import _allowlisted_server_url, _HTTPMCPClient

TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted", "succeeded", "refused"}
FIELDS = {"id", "label", "capability", "endpoint", "generate_tool", "status_tool", "cancel_tool", "models_tool", "list_tool", "launch_agents", "state_file", "header_ref", "drain_tool", "resume_tool", "model_parameter", "managed", "edit_tool", "transport", "artifacts_tool"}


def validate_services(rows):
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("Media services must be a list of at most 32 profiles")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) - FIELDS:
            raise ValueError("Unknown media service fields")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", row.get("id", "")) or row["id"] in seen:
            raise ValueError("Media services need unique names")
        seen.add(row["id"])
        if row.get("capability") not in {"image", "video", "music", "voice", "voice_training"} or not _allowlisted_server_url(row.get("endpoint")):
            raise ValueError("Media needs a capability and HTTPS or loopback HTTP MCP endpoint")
        for key in ("generate_tool", "status_tool"):
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", row.get(key, "")):
                raise ValueError("Media services need generation and status tool names")
        if row.get("transport", "mcp") not in {"mcp", "mlx2-media"}:
            raise ValueError("Unknown media transport")
        for key in ("artifacts_tool", "cancel_tool", "models_tool", "list_tool", "drain_tool", "resume_tool", "edit_tool"):
            if row.get(key) and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", row[key]):
                raise ValueError("Invalid media tool name")
        if "managed" in row and type(row["managed"]) is not bool:
            raise ValueError("Managed media state must be boolean")
        if row.get("model_parameter") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", row["model_parameter"]):
            raise ValueError("Invalid media model parameter name")
        agents = row.get("launch_agents", [])
        if not isinstance(agents, list) or len(agents) > 8 or any(not isinstance(a, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", a) for a in agents):
            raise ValueError("LaunchAgents must be explicit local labels")
        if row.get("header_ref") and not re.fullmatch(r"[a-f0-9]{32}", row["header_ref"]):
            raise ValueError("Invalid media credential reference")
        state = row.get("state_file", "")
        if not isinstance(state, str) or (state and not Path(state).expanduser().is_absolute()):
            raise ValueError("Media job state must be an absolute file path")


def known_services():
    home = Path.home()
    from .desktop_media_canonical import canonical_profile
    return [
        {"id": "mlx2-image", "label": "MLX2 / Studio images", "capability": "image",
         "endpoint": "http://127.0.0.1:8776/mcp", "generate_tool": "generate_image", "edit_tool": "edit_image", "status_tool": "get_image_job",
         "cancel_tool": "cancel_image_job", "models_tool": "image_models", "list_tool": "list_image_jobs",
         "launch_agents": ["com.example.mlx2-image-mcp"], "state_file": "", "header_ref": ""},
        {"id": "mlx2-video", "label": "MLX2 / Studio LTX video", "capability": "video",
         "endpoint": "http://127.0.0.1:8778/mcp", "generate_tool": "generate_video", "status_tool": "get_video_job",
         "cancel_tool": "cancel_video_job", "models_tool": "video_models", "list_tool": "list_video_jobs",
         "launch_agents": [], "state_file": "", "header_ref": ""},
        {"id": "yue2", "label": "YuE2 · shared with Studio", "capability": "music",
         "endpoint": "http://127.0.0.1:8774/mcp", "generate_tool": "generate_music", "status_tool": "get_music_job",
         "cancel_tool": "cancel_music_job", "models_tool": "", "list_tool": "",
         "launch_agents": ["com.example.yue2-api", "com.example.yue2-mcp"],
         "state_file": str(home / "Library/Application Support/mlxuag-yue2/runtime/state/yue2_jobs.json"), "header_ref": ""},
        canonical_profile("image"), canonical_profile("video"), canonical_profile("voice"), canonical_profile("voice_training"),
    ]


def unwrap(result):
    if not isinstance(result, dict) or result.get("isError"):
        raise ValueError("Media service refused the operation; inspect its logs")
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                value = json.loads(block["text"])
            except (ValueError, KeyError):
                continue
            if isinstance(value, dict):
                if value.get("error"):
                    raise ValueError("Media service reported an error; inspect its logs")
                return value
    raise ValueError("Media service returned no structured job receipt")


class MediaController:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.cache = {}
        self.clients = {}
        self.stop = threading.Event()
        from .desktop_conversion import Conversions
        self.conversions = Conversions(controller)
        from .desktop_media_direct import DirectServices
        self.direct = DirectServices(controller)
        from .desktop_media_canonical_service import CanonicalService
        self.canonical = CanonicalService(controller)
        # RecordStore interrupts app-owned jobs after a crash. Media jobs are
        # service-owned and can continue while this app is closed.
        for job in controller.store.jobs(1000):
            if job["kind"] == "media" and job["state"] == "interrupted" and (job.get("result") or {}).get("remote_job_id"):
                controller.store.update(job["id"], "running", "Reconnecting to the shared service", job["result"])
        self.thread = threading.Thread(target=self._monitor, daemon=True, name="mlx2-media-monitor")
        self.thread.start()

    def profiles(self):
        return self.controller.preferences.get("media_services", [])

    def profile(self, identity):
        profile = next((row for row in self.profiles() if row["id"] == identity), None)
        if not profile:
            raise ValueError("Configure this media service before using it")
        return profile

    def client(self, profile):
        headers = {}
        if profile.get("header_ref"):
            from .http_security import load_secret_file
            path = self.controller.home / "mcp-secrets" / (profile["header_ref"] + ".json")
            headers = json.loads(load_secret_file(path, flag="Media MCP credentials"))
        key = json.dumps([profile["endpoint"], headers, profile.get("transport", "mcp"), profile["capability"]], sort_keys=True)
        with self.lock:
            if key not in self.clients:
                if profile.get("transport") == "mlx2-media":
                    from .desktop_media_canonical import CanonicalMediaClient
                    self.clients[key] = CanonicalMediaClient(profile, headers)
                else:
                    self.clients[key] = _HTTPMCPClient(profile["endpoint"], headers, timeout=15)
            return self.clients[key]

    def call(self, profile, tool, arguments):
        return unwrap(self.client(profile).call_tool(tool, arguments, cancel_event=self.stop))

    def _monitor(self):
        while not self.stop.wait(3):
            try:
                self.refresh()
            except Exception as error:  # noqa: BLE001 -- retain a sanitized monitor failure
                self.controller.event("media_monitor_failed", error=type(error).__name__)

    def refresh(self):
        for profile in list(self.profiles()):
            if self.stop.is_set():
                return {}
            try:
                tools = self.client(profile).list_tools(cancel_event=self.stop)
                if self.stop.is_set():
                    return {}
                names = {tool["name"] for tool in tools}
                required = {profile["generate_tool"], profile["status_tool"]}
                value = {"reachable": True, "ready": required <= names, "tools": sorted(names)}
                if profile.get("models_tool"):
                    value["models"] = self.call(profile, profile["models_tool"], {})
                if profile.get("list_tool"):
                    value["jobs"] = self.call(profile, profile["list_tool"], {"limit": 100}).get("jobs", [])
                with self.lock:
                    self.cache[profile["id"]] = value
            except Exception as error:  # noqa: BLE001 -- sanitized service status
                with self.lock:
                    self.cache[profile["id"]] = {"reachable": False, "ready": False, "error": type(error).__name__}
        if self.stop.is_set():
            return {}
        for job in self.controller.store.jobs(500):
            if job["kind"] != "media" or job["state"] in TERMINAL or job["state"] == "queued":
                continue
            try:
                profile = self.profile(job["spec"]["provider_id"])
                remote_id = (job.get("result") or {}).get("remote_job_id")
                if not remote_id:
                    continue
                value = self.call(profile, profile["status_tool"], {"job_id": remote_id})
                state = value.get("status", "running")
                if state in {"completed", "partial"} and profile.get("artifacts_tool"):
                    value["artifacts"] = self.call(profile, profile["artifacts_tool"], {"job_id": remote_id}).get("artifacts", [])
                self.controller.store.update(job["id"], "completed" if state == "succeeded" else state,
                    str(value.get("stage") or value.get("progress") or state), {"remote_job_id": remote_id, "receipt": value})
            except Exception as error:  # noqa: BLE001 -- retain submitted job; never resubmit
                self.controller.event("media_poll_failed", job_id=job["id"], error=type(error).__name__)
        return self.snapshot()

    def snapshot(self):
        configured = {p["id"]: p for p in self.profiles()}
        with self.lock:
            services = [{**p, **self.cache.get(p["id"], {}), "configured": p["id"] in configured}
                        for p in list(configured.values()) + [p for p in known_services() if p["id"] not in configured]]
        # Credential references are internal metadata; actual headers never leave client().
        try:
            from .image_mcp.catalog import ImageBackendCatalog
        except ModuleNotFoundError as error:
            if error.name not in {"mlx2.image_mcp", "mlx2.image_mcp.catalog"}:
                raise
            images = []
            image_available = False
        else:
            images = ImageBackendCatalog(self.controller.primary.library.root).inventory()
            image_available = True
        providers = [{"id": "mlx2-image", "capability": "image", "models": images,
            "available": image_available,
            "unavailable_reason": "" if image_available else "Built-in image service is not included in this distribution",
            "route_capabilities": {"text_to_image": image_available, "image_to_image": image_available,
                "lora": {"adapter_implemented": True, "service_exposed": False, "qualified": False,
                    "reason": "ImageService does not yet pass a LoRA artifact to its bounded runner"},
                "voice_training": {"implemented": False, "reason": "Image generation is not voice training"}}}]
        providers.extend({"id": p["id"], "capability": p["capability"], "models": self.cache.get(p["id"], {}).get("models", {})} for p in services if p["id"] != "mlx2-image")
        jobs = [j for j in self.controller.store.jobs(100) if j["kind"] == "media"]
        return {"providers": providers, "services": services, "jobs": jobs, "conversions": self.conversions.snapshot(), "direct": self.direct.snapshot(), "canonical": self.canonical.snapshot(),
                "studio_url": self.controller.preferences.get("studio_url", ""),
                "contract": "Shared service jobs and artifact receipts; opening either app does not create a second model owner."}

    def generate(self, request):
        if self.stop.is_set():
            raise ValueError("Media controller is closing; nothing was submitted")
        profile = self.profile(request["provider_id"])
        parameters = request.get("parameters")
        if not isinstance(parameters, dict) or len(json.dumps(parameters)) > 65536:
            raise ValueError("Media parameters must be a bounded object")
        parameters = dict(parameters)
        operation = parameters.pop("operation", "generate")
        tool = profile["generate_tool"]
        if operation == "edit":
            tool = profile.get("edit_tool") or ("edit_image" if profile["id"] == "mlx2-image" else "")
            if profile["capability"] != "image" or not tool:
                raise ValueError("This media service does not expose image editing")
        elif operation != "generate":
            raise ValueError("Unknown media operation")
        # The service remains the source of truth for model-specific validation.
        job_id = self.controller.store.enqueue("media", {"provider_id": profile["id"], "capability": profile["capability"]})
        self.controller.store.update(job_id, "submitting", "Submitting once to the shared media service")
        try:
            result = self.call(profile, tool, parameters)
            remote_id = result.get("job_id") or result.get("id")
            if not isinstance(remote_id, str) or not remote_id:
                raise ValueError("Service returned no durable job identity")
            self.controller.store.update(job_id, "running", "Submitted", {"remote_job_id": remote_id, "receipt": result})
            self.controller.event("media_submitted", provider_id=profile["id"], job_id=job_id)
        except Exception:
            # Transport errors can occur after acceptance. Retain uncertainty and
            # require operator reconciliation rather than duplicating a render.
            self.controller.store.update(job_id, "submission_unknown", "Submission outcome unknown; inspect the shared service before retrying")
            raise
        return {"job_id": job_id, "remote_job_id": remote_id}

    def _require_idle(self, profile):
        active = [j for j in self.controller.store.jobs(1000) if j["kind"] == "media" and j["spec"].get("provider_id") == profile["id"] and j["state"] not in TERMINAL]
        if active:
            raise ValueError("Service has active or unresolved desktop jobs")
        if profile.get("list_tool"):
            jobs = self.call(profile, profile["list_tool"], {"limit": 100}).get("jobs")
            if not isinstance(jobs, list):
                raise ValueError("Service did not prove its job queue idle")
        elif profile.get("state_file"):
            path = Path(profile["state_file"]).expanduser()
            if not path.is_file() or path.stat().st_size > 16 << 20:
                raise ValueError("Cannot prove the shared service idle: no readable job store")
            saved = json.loads(path.read_text())
            jobs = saved.get("jobs")
            jobs = list(jobs.values()) if isinstance(jobs, dict) else jobs
            if not isinstance(jobs, list):
                raise ValueError("Invalid shared job state")
        else:
            raise ValueError("Configure a job-list tool or durable job store before stopping this shared service")
        if any(j.get("status") not in TERMINAL for j in jobs):
            raise ValueError("Studio or another client has an active job; wait for completion")

    def lifecycle(self, identity, start, *, force_shared=False):
        profile = self.profile(identity)
        if profile.get("transport") == "mlx2-media":
            return self.canonical.start() if start else self.canonical.stop(force_shared=force_shared)
        if start and (profile.get("managed") or (identity == "mlx2-video" and not profile.get("launch_agents"))):
            return self.direct.start(profile["capability"])
        if not start and profile.get("managed"):
            self._require_idle(profile)
            drained = self.call(profile, profile["drain_tool"], {})
            if drained.get("admission_closed") is not True or drained.get("active_jobs") != 0:
                self.call(profile, profile["resume_tool"], {})
                raise ValueError("Shared service did not atomically drain; Stop refused")
            return self.direct.stop_service(profile)
        agents = profile.get("launch_agents", [])
        if not agents:
            raise ValueError("No LaunchAgents explicitly configured for this service")
        paths = []
        for label in agents:
            path = Path.home() / "Library/LaunchAgents" / (label + ".plist")
            if not path.is_file() or path.is_symlink():
                raise ValueError("Configured LaunchAgent is absent or linked")
            value = plistlib.loads(path.read_bytes())
            if value.get("Label") != label:
                raise ValueError("LaunchAgent identity differs")
            paths.append((label, path))
        if not start:
            self._require_idle(profile)
            if not profile.get("drain_tool") or not profile.get("resume_tool"):
                if force_shared is not True:
                    raise ValueError("This shared service has no atomic admission-drain API. Explicitly confirm a manual shared Stop after checking Studio clients.")
                self.controller.event("service_stop_manual_risk_accepted", service_id=identity)
            else:
                drained = self.call(profile, profile["drain_tool"], {})
                if drained.get("admission_closed") is not True or drained.get("active_jobs") != 0:
                    self.call(profile, profile["resume_tool"], {})
                    raise ValueError("Service did not prove admission closed and all work drained; Stop refused")
        domain = f"gui/{os.getuid()}"
        for label, path in (paths if start else reversed(paths)):
            exists = subprocess.run(["/bin/launchctl", "print", domain + "/" + label], capture_output=True, timeout=5, check=False).returncode == 0
            if start and exists:
                continue  # never kickstart/restart somebody else's resident model
            args = ["/bin/launchctl", "bootstrap", domain, str(path)] if start else ["/bin/launchctl", "bootout", domain + "/" + label]
            if start or exists:
                completed = subprocess.run(args, capture_output=True, timeout=15, check=False)
                if completed.returncode:
                    raise ValueError("LaunchAgent operation failed; inspect the service logs")
        return {"service_id": identity, "state": "starting" if start else "stopped"}

    def dispatch(self, request):
        command = request["command"]
        if command.startswith("conversion_"):
            return self.conversions.dispatch(request)
        if command == "media_canonical_configure":
            return self.canonical.configure(request["settings"])
        if command == "media_canonical_start":
            return self.canonical.start()
        if command == "media_canonical_cancel":
            self.canonical.cancel.set()
            return {"cancelling": True}
        if command == "media_prepare_voice_reference":
            from .desktop_media_canonical import prepare_reference
            return prepare_reference(self.controller, request)
        if command == "media_direct_configure":
            return self.direct.configure(request["settings"])
        if command == "media_direct_start":
            capability = request["capability"]
            from .desktop_media_canonical import canonical_profile
            canonical = canonical_profile(capability)
            try:
                names = {t["name"] for t in self.client(canonical).list_tools()}
                if canonical["generate_tool"] in names:
                    self.controller.configure_preferences({"media_services": [p for p in self.profiles() if p["id"] != canonical["id"]] + [canonical]})
                    return {"service_id": canonical["id"], "state": "reused", "endpoint": canonical["endpoint"]}
            except Exception:  # noqa: BLE001 -- absent canonical service has no state mutated
                self.controller.event("canonical_media_unavailable", capability=capability)
            if capability in {"voice", "voice_training"}:
                return self.canonical.start()
            if capability == "image":
                known = next(p for p in known_services() if p["id"] == "mlx2-image")
                if all((Path.home() / "Library/LaunchAgents" / (label + ".plist")).is_file() for label in known["launch_agents"]):
                    if not any(p["id"] == known["id"] for p in self.profiles()):
                        self.controller.configure_preferences({"media_services": self.profiles() + [known]})
                    return self.lifecycle(known["id"], True)
            return self.direct.start(capability)
        if command == "media_generate":
            return self.generate(request)
        if command == "media_refresh":
            return self.refresh()
        if command == "media_use_known":
            profile = next((p for p in known_services() if p["id"] == request["service_id"]), None)
            if not profile:
                raise ValueError("Unknown built-in media service")
            self.controller.configure_preferences({"media_services": [p for p in self.profiles() if p["id"] != profile["id"]] + [profile]})
            return {"configured": True}
        if command == "media_cancel":
            job = next((j for j in self.controller.store.jobs(1000) if j["id"] == request["job_id"] and j["kind"] == "media"), None)
            if not job or not (job.get("result") or {}).get("remote_job_id"):
                raise ValueError("No remote job identity; reconcile this submission in the shared service")
            profile = self.profile(job["spec"]["provider_id"])
            if not profile.get("cancel_tool"):
                raise ValueError("This service does not advertise cancellation")
            return self.call(profile, profile["cancel_tool"], {"job_id": job["result"]["remote_job_id"]})
        if command in {"service_start", "service_stop"}:
            return self.lifecycle(request["service_id"], command == "service_start", force_shared=request.get("force_shared") is True)
        raise ValueError("Unknown media operation")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=35)
        self.conversions.close()
        self.canonical.close()
