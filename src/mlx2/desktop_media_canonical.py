"""Original bounded client of Studio's canonical mlx2-media job HTTP API.

Only catalog/status/job operations are exposed; the server owns accelerator,
source binding, reference snapshots, job cancellation and artifact receipts.
Reference conditioning is never labelled model training.
"""
from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

TERMINAL = {"completed", "failed", "cancelled", "interrupted", "refused"}


def canonical_profile(capability):
    if capability not in {"image", "video", "voice", "voice_training"}:
        raise ValueError("Unknown canonical media capability")
    return {"id": "studio-" + capability, "label": "MLX2 / Studio " + capability.replace("_", " "),
        "capability": capability, "endpoint": "http://127.0.0.1:8777", "transport": "mlx2-media",
        "generate_tool": "generate_" + capability, "status_tool": "get_" + capability + "_job",
        "cancel_tool": "cancel_" + capability + "_job", "models_tool": capability + "_models",
        "list_tool": "list_" + capability + "_jobs", "artifacts_tool": "get_" + capability + "_artifacts",
        "launch_agents": [], "state_file": "", "header_ref": "", "model_parameter": "model",
        "edit_tool": "edit_image" if capability == "image" else ""}


class CanonicalMediaClient:
    def __init__(self, profile, headers=None):
        self.profile, self.headers = profile, headers or {}
        url = urllib.parse.urlsplit(profile["endpoint"])
        if url.scheme != "http" or url.hostname != "127.0.0.1" or url.port is None or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
            raise ValueError("Canonical Studio media requires an explicit loopback HTTP port")
        self.base = profile["endpoint"].rstrip("/")
        self.routes = {}

    def request(self, method, path, body=None):
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
            headers={**self.headers, "Accept": "application/json", "Content-Type": "application/json"})
        # No redirects: a local service cannot forward credentials or reference
        # paths to a third-party host.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_args, **_kwargs):
                return None
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=10) as response:
                payload = response.read((4 << 20) + 1)
                if len(payload) > 4 << 20:
                    raise ValueError("Canonical media response exceeds 4 MiB")
                value = json.loads(payload)
        except urllib.error.HTTPError as error:
            raise ValueError("Canonical media refused the request (HTTP " + str(error.code) + "); inspect its route receipt") from None
        if not isinstance(value, dict):
            raise TypeError("Canonical media returned an invalid object")
        return value

    def models(self):
        status = self.request("GET", "/v1/media/status")
        if status.get("service") != "mlx2-media":
            raise ValueError("Endpoint is not the canonical mlx2-media service")
        catalog = self.request("GET", "/v1/media/routes")
        rows = catalog.get("data", [])
        if not isinstance(rows, list):
            raise TypeError("Canonical media has no typed route catalog")
        capability = self.profile["capability"]
        self.routes = {}
        models = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("route_id"), str):
                continue
            operation = row.get("operation")
            suitable = ((capability == "voice" and operation == "tts") or
                (capability == "voice_training" and operation in {"voice_training", "train_voice"}) or
                (capability == "image" and str(row.get("capability", "")).startswith("image.")) or
                (capability == "video" and str(row.get("capability", "")).startswith("video.")))
            if not suitable:
                continue
            self.routes[row["route_id"]] = row
            models.append({**row, "id": row["route_id"], "name": row.get("label") or row["route_id"], "available": row.get("ready") is True,
                "qualification": row.get("qualification", {}), "reference_conditioning": any(i.get("role") == "speaker_reference" for i in row.get("inputs", [])),
                "training": capability == "voice_training", "state": "service-advertised; exact source/artifact checks remain with the server worker"})
        return {"models": models, "service": "mlx2-media", "source_contract": "mlx2.media.api/1", "live_status": status}

    def list_tools(self, *, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Media operation cancelled")
        models = self.models()["models"]
        names = [self.profile[k] for k in ("models_tool", "status_tool", "list_tool", "cancel_tool", "artifacts_tool") if self.profile.get(k)]
        if any(row["available"] for row in models):
            names.append(self.profile["generate_tool"])
            if self.profile.get("edit_tool"):
                names.append(self.profile["edit_tool"])
        return [{"name": name, "inputSchema": {"type": "object"}} for name in names]

    def call_tool(self, name, arguments, *, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Media operation cancelled")
        profile = self.profile
        if name == profile["models_tool"]:
            result = self.models()
        elif name == profile["list_tool"]:
            result = self.request("GET", "/v1/media/jobs?limit=100")
            rows = result.get("data")
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Canonical media returned an invalid job catalog")
            result = {**result, "jobs": rows}
        elif name in {profile["status_tool"], profile["cancel_tool"], profile.get("artifacts_tool")}:
            identity = urllib.parse.quote(arguments["job_id"], safe="")
            suffix = "/cancel" if name == profile["cancel_tool"] else "/artifacts" if name == profile.get("artifacts_tool") else ""
            result = self.request("POST" if suffix == "/cancel" else "GET", "/v1/media/jobs/" + identity + suffix, {} if suffix == "/cancel" else None)
            result = result.get("job", result)
        elif name in {profile["generate_tool"], profile.get("edit_tool")}:
            catalog = self.models()["models"]
            values = dict(arguments)
            editing = name == profile.get("edit_tool")
            route_id = values.pop("model", None) or next((row["id"] for row in catalog if row["available"] and (not editing or row.get("operation") == "edit")), None)
            route = self.routes.get(route_id)
            if not route or route.get("ready") is not True:
                raise ValueError("Selected media route or its pinned runtime/artifact prerequisites are unavailable")
            inputs, loras = values.pop("inputs", []), values.pop("loras", [])
            if editing and route.get("operation") != "edit":
                raise ValueError("Select an image-edit route for reference conditioning")
            references = values.pop("reference_images", [])
            if not isinstance(references, list):
                raise ValueError("Image references must be a path list")
            for reference in references:
                inputs.append({"role": "reference", "path": reference})
            for role in ("speaker_reference", "emotion_reference"):
                if values.get(role):
                    inputs.append({"role": role, "path": values.pop(role)})
            prompt = values.pop("prompt", None)
            if prompt is not None:
                values["text" if profile["capability"].startswith("voice") else "prompt"] = prompt
            consent, intent = values.pop("consent", None), values.pop("intent", None)
            if not isinstance(inputs, list) or not isinstance(loras, list):
                raise ValueError("Inputs and LoRAs must be typed lists")
            body = {"route": route_id, "operation": route["operation"], "params": values,
                    "inputs": inputs, "loras": loras, "idempotency_key": uuid.uuid4().hex,
                    "client": {"module": "mlx2-desktop", "reference_conditioning_is_training": False}}
            if consent is not None:
                body["consent"] = consent
            if intent is not None:
                body["intent"] = intent
            result = self.request("POST", "/v1/media/jobs", body)
            result = result.get("job", result)
        else:
            raise ValueError("Unknown canonical media operation")
        return {"structuredContent": result, "content": [{"type": "text", "text": json.dumps(result)}]}


def prepare_reference(controller, request):
    """Normalize one explicitly selected audio/video sample with a fixed ffmpeg
    command. This is sample preparation, not speaker training.
    """
    import shutil
    source = Path(request["path"]).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".mp4", ".mov", ".webm"}:
        raise ValueError("Choose a local audio or video reference sample")
    if source.stat().st_size > 512 << 20:
        raise ValueError("Reference sample must be at most 512 MiB")
    executable = shutil.which("ffmpeg")
    if not executable:
        raise ValueError("ffmpeg is required to extract/normalize a voice reference")
    import hashlib
    import os
    import wave

    from .desktop import atomic_json
    directory = Path("/Users/Shared/mlx2-desktop-inputs-" + str(os.getuid()))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(directory).free < (2 << 20):
        raise ValueError("Insufficient disk space for the bounded voice reference")
    output = directory / ("voice-" + uuid.uuid4().hex + ".wav")
    def digest(path):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                checksum.update(chunk)
        return checksum.hexdigest()
    source_hash = digest(source)
    try:
        subprocess.run([executable, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-n", "-i", str(source), "-t", "15", "-vn", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(output)],
            check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        with wave.open(str(output), "rb") as stream:
            if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 22050 or not 0 < stream.getnframes() <= 15 * 22050:
                raise ValueError("Normalized voice reference failed its bounded PCM contract")
            duration = stream.getnframes() / stream.getframerate()
        if digest(source) != source_hash:
            raise ValueError("Reference source changed during normalization; select it again")
        receipt = {"path": str(output), "duration_seconds": duration, "reference_conditioning": True, "training": False,
            "source": str(source), "source_sha256": source_hash, "sha256": digest(output),
            "transform": "first-15-seconds-mono-22050-pcm16", "executable": executable}
        atomic_json(output.with_suffix(".receipt.json"), receipt)
        return receipt
    except BaseException:
        output.unlink(missing_ok=True)
        raise
