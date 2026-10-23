"""Shared lightweight loopback media MCP; models run only in bounded workers.

Image jobs delegate to the existing ImageService. Video jobs wrap the existing
LTX adapter; this service is a durable owner shared by desktop and Studio.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .desktop import atomic_json

TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}


def video_catalog(settings):
    from .adapters.generative_media import (
        LTX_CONVERTER_SHA256,
        LTX_REVISION,
        LTX_RUNTIME_REVISION,
    )
    source, model, runtime = (Path(settings[k]).expanduser() for k in ("video_source", "video_model", "video_runtime"))
    reasons = []
    try:
        manifest = json.loads((source / ".hf-download-manifest.json").read_text())
        complete = json.loads((source / ".hf-download-complete.json").read_text())
        if manifest.get("revision") != LTX_REVISION or complete.get("revision") != LTX_REVISION:
            reasons.append("Original LTX-2.5 source is not at the pinned revision")
    except (OSError, ValueError):
        reasons.append("Select the original complete LTX-2.5 source snapshot")
    try:
        conversion = json.loads((model / ".mlx2-cpu-conversion.json").read_text())
        if conversion.get("runtime_revision") != LTX_RUNTIME_REVISION or conversion.get("converter_sha256") != LTX_CONVERTER_SHA256:
            reasons.append("LTX MLX conversion receipt differs from the integrated adapter")
        if not (model / "transformer-distilled.safetensors").is_file():
            reasons.append("The converted distilled transformer is absent")
    except (OSError, ValueError):
        reasons.append("Select a complete receipt-bound LTX-2.5 MLX conversion")
    if not (runtime / ".venv/bin/python").is_file():
        reasons.append("Select the pinned ltx-2-mlx runtime with its prepared environment")
    fingerprint = hashlib.sha256(json.dumps(conversion if "conversion" in locals() else {}, sort_keys=True).encode()).hexdigest()
    return {"id": "ltx-2.5-distilled", "label": "LTX-2.5 distilled", "path": str(model), "available": not reasons,
            "implemented": True, "qualified": False, "selected": True, "observed_used": False,
            "checkpoint": model.name + "@" + fingerprint, "source_revision": LTX_REVISION,
            "runtime_revision": LTX_RUNTIME_REVISION, "unavailable_reasons": reasons,
            "validation": "Lightweight prerequisite inventory; full source/checksum/runtime verification occurs before generation",
            "operations": ["generate_video"], "lora": {"load": True, "training": False, "qualified": False,
                "note": "Bound media LoRA loading is supported by the LTX adapter. Training requires an explicit latent training route and is not exposed here."},
            "voice_training": {"implemented": False, "reason": "LTX video generation is not a voice-training route"}}


class VideoJobs:
    def __init__(self, home, settings):
        self.home, self.settings = Path(home), settings
        self.root = self.home / "video/jobs"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.cancel = {}
        self.stop = threading.Event()
        self.child = None
        for path in self.root.glob("*.json"):
            value = json.loads(path.read_text())
            if value.get("status") not in TERMINAL:
                value.update(status="interrupted", detail="Media service restarted; submission was not replayed")
                atomic_json(path, value)
        self.worker = threading.Thread(target=self._work, daemon=True, name="mlx2-video-jobs")
        self.worker.start()

    def path(self, identity):
        if not isinstance(identity, str) or not identity.startswith("video_") or len(identity) != 38 or any(c not in "0123456789abcdef" for c in identity[6:]):
            raise ValueError("Invalid video job ID")
        return self.root / (identity + ".json")

    def read(self, identity):
        return json.loads(self.path(identity).read_text())

    def public(self, value):
        return {k: v for k, v in value.items() if k != "parameters"}

    def jobs(self):
        return sorted((json.loads(p.read_text()) for p in self.root.glob("*.json")), key=lambda x: x["created_at"], reverse=True)

    def submit(self, parameters):
        allowed = {"prompt", "model", "width", "height", "frames", "frame_rate", "seed", "timeout_seconds", "lora_path"}
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            raise ValueError("Unsupported video parameters")
        prompt = parameters.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 8192:
            raise ValueError("Video needs a nonempty prompt up to 8192 bytes")
        if parameters.get("model", "ltx-2.5-distilled") != "ltx-2.5-distilled":
            raise ValueError("Video model is not in this service's capability catalog")
        defaults = {"width": 704, "height": 480, "frames": 97, "frame_rate": 24, "seed": 42, "timeout_seconds": 7200}
        values = {**defaults, **parameters}
        for key in defaults:
            if type(values[key]) is not int:
                raise ValueError("Video dimensions, timing and seed must be integers")
        if any(not 256 <= values[k] <= 1920 or values[k] % 32 for k in ("width", "height")):
            raise ValueError("Video dimensions must be multiples of 32 between 256 and 1920")
        if not 9 <= values["frames"] <= 257 or (values["frames"] - 1) % 8 or not 1 <= values["frame_rate"] <= 120 or not 0 <= values["seed"] < 2**63 or not 1 <= values["timeout_seconds"] <= 14400:
            raise ValueError("Video timing/seed is outside the bounded adapter contract")
        if values.get("lora_path"):
            lora = Path(values["lora_path"]).expanduser().resolve(strict=True)
            allowed_roots = [Path(self.settings["model_root"]).resolve(), self.home / "loras"]
            if not lora.is_dir() or not any(lora.is_relative_to(root) for root in allowed_roots):
                raise ValueError("Choose a LoRA directory inside the model library or shared-media/loras")
            values["lora_path"] = str(lora)
        model = video_catalog(self.settings)
        if not model["available"]:
            raise ValueError("; ".join(model["unavailable_reasons"]))
        identity = "video_" + uuid.uuid4().hex
        job = {"id": identity, "job_id": identity, "created_at": time.time(), "status": "queued", "detail": "Queued for bounded GPU ownership",
               "parameters": values, "model": model, "qualification": "unqualified", "results": []}
        with self.lock:
            if self.stop.is_set():
                raise ValueError("Video service is closing")
            atomic_json(self.path(identity), job)
            self.cancel[identity] = threading.Event()
            self.queue.put(identity)
        return self.public(job)

    def _work(self):
        while not self.stop.is_set():
            try:
                identity = self.queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                self.run(identity)
            finally:
                self.queue.task_done()

    def run(self, identity):
        from .desktop import source_identity, stop_group
        from .desktop_campaign_lease import DesktopCPGLease
        job = self.read(identity)
        event = self.cancel[identity]
        if event.is_set():
            job.update(status="cancelled", detail="Cancelled before GPU ownership")
            atomic_json(self.path(identity), job)
            self.cancel.pop(identity, None)
            return
        folder = self.home / "video/runs" / identity
        folder.mkdir(parents=True, mode=0o700)
        child = None
        try:
            spec = {"out": str(folder), "job_id": identity, "mcp_url": self.settings["cpg_mcp_url"],
                    "parameters": job["parameters"], "settings": self.settings, "source": source_identity(), "checkpoint": job["model"]["checkpoint"]}
            atomic_json(folder / "request.json", spec)
            job.update(status="waiting", detail="Waiting for CPG/radio and paired GPU ownership")
            atomic_json(self.path(identity), job)
            with DesktopCPGLease(folder, identity, mcp_url=spec["mcp_url"], checkpoint=spec["checkpoint"], source_revision=spec["source"]["source_revision"]) as lease:
                try:
                    if event.is_set() or self.stop.is_set():
                        raise InterruptedError("Video job cancelled before worker startup")
                    source = Path(__file__).resolve().parents[2]
                    args = [sys.executable, str(source / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"], "--label", identity,
                            "--desktop-cpg-lease", str(lease.path), "--mcp-url", spec["mcp_url"],
                            "--receipt", str(folder / "paired-locks.json"), "--", sys.executable, "-m", "mlx2.desktop_media_direct", "--worker", str(folder / "request.json")]
                    with (folder / "worker.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        self.child = child
                        while child.poll() is None:
                            lease.check()
                            if event.wait(.5) or self.stop.is_set():
                                stop_group(child)
                                raise InterruptedError("Video job cancelled")
                            progress = folder / "progress.json"
                            if progress.exists():
                                detail = json.loads(progress.read_text())
                                job.update(status="in_progress", detail=detail["detail"])
                                atomic_json(self.path(identity), job)
                        if child.returncode:
                            raise ValueError("Video worker failed; inspect " + str(folder / "worker.log"))
                    if event.is_set() or self.stop.is_set():
                        raise InterruptedError("Video job cancelled before completion")
                    result = json.loads((folder / "result.json").read_text())
                    job.update(status="completed", detail="Video complete · unqualified", results=[result], result=result, completed_at=time.time())
                    atomic_json(self.path(identity), job)
                    lease.success = True
                finally:
                    if child is not None and child.poll() is None:
                        stop_group(child)
        except Exception as error:  # noqa: BLE001 -- durable job outcome
            job.update(status="cancelled" if isinstance(error, InterruptedError) else "failed", detail=str(error) if isinstance(error, (ValueError, InterruptedError)) else type(error).__name__)
            atomic_json(self.path(identity), job)
        finally:
            self.child = None
            self.cancel.pop(identity, None)

    def cancel_job(self, identity):
        with self.lock:
            event = self.cancel.get(identity)
            if event:
                event.set()
            job = self.read(identity)
            if job["status"] == "queued":
                job.update(status="cancelled", detail="Cancelled before GPU ownership")
                atomic_json(self.path(identity), job)
            return self.public(job)

    def close(self):
        with self.lock:
            self.stop.set()
            for identity, event in list(self.cancel.items()):
                event.set()
                job = self.read(identity)
                if job["status"] == "queued":
                    job.update(status="cancelled", detail="Service closed before GPU ownership")
                    atomic_json(self.path(identity), job)
        self.worker.join(timeout=30)


def video_worker(path):
    from .desktop_campaign_worker import check_ownership
    spec = json.loads(Path(path).read_text())
    check_ownership(spec)
    stop = threading.Event()
    def watch():
        while not stop.wait(5):
            try:
                check_ownership(spec)
            except Exception:  # noqa: BLE001 -- fail closed if parent/lease disappears during native render
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    watcher = threading.Thread(target=watch, daemon=True, name="mlx2-video-ownership")
    watcher.start()
    folder = Path(spec["out"])
    try:
        atomic_json(folder / "progress.json", {"detail": "Verifying source, converted weights and pinned LTX runtime"})
        from .adapters.generative_media import LTX25Adapter
        settings, parameters = spec["settings"], spec["parameters"]
        adapter = LTX25Adapter(source=settings["video_source"], mlx_model=settings["video_model"], runtime_root=settings["video_runtime"])
        if parameters.get("lora_path"):
            adapter.load_lora(parameters["lora_path"])
        atomic_json(folder / "progress.json", {"detail": "Generating video using the direct LTX-2.5 adapter"})
        output = folder / "video.mp4"
        started = time.monotonic()
        result = adapter.generate_video(parameters["prompt"], output=output, **{k: parameters[k] for k in ("width", "height", "frames", "frame_rate", "seed", "timeout_seconds")})
        check_ownership(spec)
        receipt = {"artifact_path": str(result.path), "path": str(result.path), "media_type": result.mime_type,
                   "artifact_bytes": output.stat().st_size, "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                   "artifact_fingerprint": result.artifact_fingerprint, "lora_fingerprint": result.lora_fingerprint, "state_epoch": result.state_epoch, "checkpoint": spec["checkpoint"],
                   "mlx2_source": spec["source"], "runtime_revision": job_runtime_revision(), "generation_seconds": time.monotonic() - started,
                   "parameters": {k: v for k, v in parameters.items() if k != "prompt"}, "qualification": "unqualified"}
        atomic_json(folder / "result.json", receipt)
    finally:
        stop.set()
        watcher.join(timeout=10)


def job_runtime_revision():
    from .adapters.generative_media import LTX_RUNTIME_REVISION
    return LTX_RUNTIME_REVISION


class SharedMediaService:
    def __init__(self, home, settings, capability):
        self.home, self.settings, self.capability = Path(home), settings, capability
        self.admission = threading.RLock()
        self.admitting = True
        self.loop = None
        if capability == "image":
            try:
                from .image_mcp.service import ImageService, ImageSettings
            except ModuleNotFoundError as error:
                if error.name not in {"mlx2.image_mcp", "mlx2.image_mcp.service"}:
                    raise
                raise ValueError("The built-in image service is unavailable in this distribution") from error
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="mlx2-image-loop")
            self.loop_thread.start()
            self.images = ImageService(ImageSettings(model_root=Path(settings["model_root"]),
                state_root=self.home / "images/state", output_root=self.home / "images/artifacts",
                runner_python=Path(settings["python"]), source_root=Path(__file__).resolve().parents[2],
                allowed_input_roots=(*ImageSettings().allowed_input_roots, Path(settings["input_root"]), Path(settings["app_home"]) / "attachments")))
        else:
            self.video = VideoJobs(home, settings)

    def wait(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout=30)

    def jobs(self):
        if self.capability == "video":
            return self.video.jobs()
        return [json.loads(p.read_text()) for p in self.images.jobs_root.glob("image_*.json")]

    def tools(self):
        if self.capability == "video":
            definitions = [("video_models", {}), ("generate_video", {"prompt": {"type": "string"}}),
                ("get_video_job", {"job_id": {"type": "string"}}), ("list_video_jobs", {}), ("cancel_video_job", {"job_id": {"type": "string"}})]
        else:
            definitions = [("image_models", {}), ("generate_image", {"prompt": {"type": "string"}}), ("edit_image", {"prompt": {"type": "string"}, "reference_images": {"type": "array", "items": {"type": "string"}}}),
                ("get_image_job", {"job_id": {"type": "string"}}), ("list_image_jobs", {}), ("cancel_image_job", {"job_id": {"type": "string"}}), ("get_image_artifacts", {"job_id": {"type": "string"}})]
        definitions += [("service_drain", {}), ("service_resume", {})]
        image_properties = {"prompt": {"type": "string"}, "backend": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"}, "steps": {"type": "integer"}, "seed": {"type": "integer"}}
        video_properties = {"prompt": {"type": "string"}, "model": {"type": "string"}, **{k: {"type": "integer"} for k in ("width", "height", "frames", "frame_rate", "seed", "timeout_seconds")}, "lora_path": {"type": "string"}}
        definitions = [(name, {**(video_properties if name == "generate_video" else image_properties if name in {"generate_image", "edit_image"} else {}), **properties}) for name, properties in definitions]
        return [{"name": name, "description": "Shared durable " + name, "inputSchema": {"type": "object", "properties": properties}} for name, properties in definitions]

    def call(self, name, arguments):
        with self.admission:
            if name == "service_drain":
                self.admitting = False
                return {"admission_closed": True, "active_jobs": sum(j.get("status") not in TERMINAL for j in self.jobs())}
            if name == "service_resume":
                self.admitting = True
                return {"admission_closed": False}
            if (name.startswith("generate_") or name == "edit_image") and not self.admitting:
                raise ValueError("Service admission is closed")
            if self.capability == "video":
                if name == "video_models":
                    return {"models": [video_catalog(self.settings)]}
                if name == "generate_video":
                    return self.video.submit(arguments)
                if name == "get_video_job":
                    return self.video.public(self.video.read(arguments["job_id"]))
                if name == "list_video_jobs":
                    return {"jobs": [self.video.public(j) for j in self.video.jobs()[:100]]}
                if name == "cancel_video_job":
                    return self.video.cancel_job(arguments["job_id"])
            else:
                if name == "image_models":
                    return self.wait(self.images.capabilities())
                if name in {"generate_image", "edit_image"}:
                    allowed = {"prompt", "backend", "width", "height", "steps", "seed", "reference_images"}
                    if set(arguments) - allowed:
                        raise ValueError("Unknown image generation parameters")
                    return self.wait(self.images.submit(operation="edit" if name == "edit_image" else "generate", prompt=arguments.get("prompt", ""), backend=arguments.get("backend", "selected"),
                        width=arguments.get("width", 1024), height=arguments.get("height", 1024), steps=arguments.get("steps", 30), seed=arguments.get("seed", 42), reference_images=arguments.get("reference_images")))
                methods = {"get_image_job": self.images.get_job, "cancel_image_job": self.images.cancel_job, "get_image_artifacts": self.images.artifacts}
                if name in methods:
                    return self.wait(methods[name](arguments["job_id"]))
                if name == "list_image_jobs":
                    return self.wait(self.images.list_jobs(min(100, arguments.get("limit", 100))))
            raise ValueError("Unknown media tool")

    def close(self):
        if self.capability == "video":
            self.video.close()
        elif self.loop:
            async def retire():
                tasks = list(self.images._tasks.values())
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            # Preserve the event loop until runner process groups and locks
            # have actually retired; stopping it early abandons cleanup.
            asyncio.run_coroutine_threadsafe(retire(), self.loop).result()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join(timeout=20)


def serve(config):
    settings = json.loads(Path(config).read_text())
    from .http_security import load_secret_file
    token = load_secret_file(settings["token_file"], flag="Shared media token")
    if not token:
        raise ValueError("Media service requires a nonempty private bearer token")
    service = None
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *_args):
            return
        def reply(self, value, code=200):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)
        def do_GET(self):
            if not self.authorized():
                self.reply({"error": "Unauthorized"}, 401)
            elif self.path == "/health":
                self.reply({"status": "ok", "capability": settings["capability"], "admitting": service.admitting})
            else:
                self.reply({"error": "Unknown route"}, 404)
        def do_POST(self):
            if not self.authorized():
                self.reply({"error": "Unauthorized"}, 401)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 128 << 10 or self.path != "/mcp":
                    raise ValueError("Invalid bounded MCP request")
                message = json.loads(self.rfile.read(size))
                method, identity = message.get("method"), message.get("id")
                if method.startswith("notifications/"):
                    self.send_response(202); self.send_header("Content-Length", "0"); self.end_headers(); return
                if method == "initialize":
                    result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "mlx2-shared-media", "version": "1"}}
                elif method == "tools/list":
                    result = {"tools": service.tools()}
                elif method == "tools/call":
                    params = message["params"]
                    value = service.call(params["name"], params.get("arguments", {}))
                    result = {"content": [{"type": "text", "text": json.dumps(value)}], "structuredContent": value}
                else:
                    raise ValueError("Unsupported MCP method")
                self.reply({"jsonrpc": "2.0", "id": identity, "result": result})
            except Exception as error:  # noqa: BLE001 -- bounded RPC failure without credentials
                self.reply({"jsonrpc": "2.0", "id": locals().get("identity"), "error": {"code": -32602, "message": str(error) if isinstance(error, ValueError) else type(error).__name__}})
    # Bind first. A second controller cannot recover or mutate job state behind
    # an already occupied canonical endpoint.
    server = ThreadingHTTPServer(("127.0.0.1", settings["port"]), Handler)
    try:
        service = SharedMediaService(settings["home"], settings, settings["capability"])
        def shutdown(*_args):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        if service:
            service.close()



class DirectServices:
    """Source-bound service supervisor. Closing the UI leaves Studio jobs owned
    by the shared service; explicit Stop drains that service before exiting it.
    """
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / "shared-media"
        self.config_path = controller.home / "media-direct-settings.json"
        self.lock = threading.RLock()

    def settings(self):
        root = self.controller.primary.library.root
        defaults = {"model_root": str(root), "python": sys.executable,
            "video_source": str(root / "LTX-2.5"), "video_model": str(root / "LTX-2.5-MLX"),
            "video_runtime": str(root / "_runtimes/ltx-2-mlx-fbc4b05"),
            "app_home": str(self.controller.home), "input_root": "/Users/Shared/mlx2-desktop-inputs-" + str(os.getuid()),
            "cpg_mcp_url": self.controller.preferences.get("cpg_mcp_url", "http://127.0.0.1:8766/mcp")}
        if self.config_path.exists():
            defaults.update(json.loads(self.config_path.read_text()))
        return defaults

    def configure(self, values):
        allowed = {"python", "video_source", "video_model", "video_runtime"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("Unknown direct media runtime setting")
        for value in values.values():
            if not isinstance(value, str) or not value or not Path(value).expanduser().is_absolute():
                raise ValueError("Media runtime settings must be absolute paths")
        atomic_json(self.config_path, {k: v for k, v in {**self.settings(), **values}.items() if k in allowed})
        return {"saved": True, "settings": self.settings()}

    def snapshot(self):
        settings = self.settings()
        return {"settings": settings, "video_models": [video_catalog(settings)],
                "contract": "Lightweight shared services. Starting a service does not load a model; generation requests own bounded accelerator windows."}

    def source(self):
        import shutil

        from .desktop_runner import source_manifest
        root = Path(__file__).resolve().parents[2]
        files = source_manifest(root)
        fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        target = self.home / "sources" / fingerprint
        if target.exists():
            if source_manifest(target) != files:
                raise ValueError("Saved media source snapshot changed")
            return target, fingerprint
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".preparing-" + uuid.uuid4().hex)
        staging.mkdir(mode=0o700)
        for name in files:
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        if source_manifest(staging) != files:
            raise ValueError("Media source changed during preparation; retry")
        from .desktop import source_identity
        atomic_json(staging / "desktop-source.json", {**source_identity(), "source_sha256": fingerprint})
        staging.rename(target)
        return target, fingerprint

    def start(self, capability):
        if capability not in {"image", "video"}:
            raise ValueError("Local managed media supports images and LTX video; YuE keeps its canonical service")
        from .desktop_media import known_services
        identity = "mlx2-image" if capability == "image" else "mlx2-video"
        profile = next((p for p in self.controller.preferences.get("media_services", []) if p["id"] == identity),
            next((p for p in known_services() if p["id"] == identity), None))
        if profile:
            try:
                tools = {t["name"] for t in self.controller.media.client(profile).list_tools()}
                if {profile["generate_tool"], profile["status_tool"]} <= tools:
                    if not any(p["id"] == identity for p in self.controller.preferences.get("media_services", [])):
                        self.controller.configure_preferences({"media_services": self.controller.preferences.get("media_services", []) + [profile]})
                    return {"service_id": identity, "state": "reused", "endpoint": profile["endpoint"]}
            except Exception:  # noqa: BLE001 -- an absent endpoint can be started; socket bind remains exclusive
                profile = None
        with self.lock:
            settings = self.settings()
            python = Path(settings["python"]).expanduser()
            if not python.is_file():
                raise ValueError("Select a Python executable for the shared media runtime")
            self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
            source, fingerprint = self.source()
            token_file = self.home / (capability + "-token")
            if not token_file.exists():
                descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w") as stream:
                    stream.write(uuid.uuid4().hex + uuid.uuid4().hex)
            from .http_security import load_secret_file
            token = load_secret_file(token_file, flag="Shared media token")
            from .desktop_tools import tool_command
            reference = tool_command(self.controller, {"command": "mcp_save_secret", "headers": {"Authorization": "Bearer " + token}})["header_ref"]
            port = 8776 if capability == "image" else 8778
            config = {**settings, "capability": capability, "port": port, "home": str(self.home), "token_file": str(token_file)}
            path = self.home / (capability + "-service.json")
            atomic_json(path, config)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(source / "src")
            with (self.home / (capability + ".log")).open("ab", buffering=0) as log:
                child = subprocess.Popen([str(python), "-m", "mlx2.desktop_media_direct", "--serve", str(path)],
                    cwd=source, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            import psutil
            receipt = {"pid": child.pid, "created_at": psutil.Process(child.pid).create_time(), "source": str(source), "source_sha256": fingerprint,
                       "command": [str(python), "-m", "mlx2.desktop_media_direct", "--serve", str(path)]}
            profile = {"id": identity, "label": "MLX2 / Studio " + capability, "capability": capability,
                "endpoint": f"http://127.0.0.1:{port}/mcp", "generate_tool": "generate_" + capability,
                "status_tool": "get_" + capability + "_job", "cancel_tool": "cancel_" + capability + "_job",
                "models_tool": capability + "_models", "list_tool": "list_" + capability + "_jobs",
                "drain_tool": "service_drain", "resume_tool": "service_resume", "edit_tool": "edit_image" if capability == "image" else "", "launch_agents": [], "state_file": "",
                "header_ref": reference, "managed": True, "model_parameter": "backend" if capability == "image" else "model"}
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise ValueError("Media service failed to start; inspect " + str(self.home / (capability + ".log")))
                try:
                    tools = {t["name"] for t in self.controller.media.client(profile).list_tools()}
                    if profile["generate_tool"] in tools:
                        atomic_json(self.home / (capability + "-process.json"), receipt)
                        self.controller.configure_preferences({"media_services": [p for p in self.controller.preferences.get("media_services", []) if p["id"] != identity] + [profile]})
                        return {"service_id": identity, "state": "ready", "endpoint": profile["endpoint"], "source_sha256": fingerprint}
                except Exception:  # noqa: BLE001 -- wait only for lightweight HTTP readiness
                    time.sleep(.2)
            from .desktop import stop_group
            stop_group(child)
            raise ValueError("Media service did not become ready")

    def stop_service(self, profile):
        capability = profile["capability"]
        if capability not in {"image", "video"} or profile["id"] != "mlx2-" + capability:
            raise ValueError("This is not an owned media service")
        import psutil
        receipt = json.loads((self.home / (capability + "-process.json")).read_text())
        try:
            process = psutil.Process(receipt["pid"])
        except psutil.NoSuchProcess:
            return {"service_id": profile["id"], "state": "stopped"}
        if process.create_time() != receipt["created_at"] or process.cmdline() != receipt["command"] or os.getpgid(process.pid) != process.pid:
            raise ValueError("Media process identity changed; refusing to signal it")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except psutil.TimeoutExpired:
            raise ValueError("Shared media service did not stop after draining") from None
        return {"service_id": profile["id"], "state": "stopped"}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in {"--serve", "--worker"}:
        raise SystemExit("Use --serve CONFIG or --worker REQUEST")
    (serve if sys.argv[1] == "--serve" else video_worker)(sys.argv[2])
