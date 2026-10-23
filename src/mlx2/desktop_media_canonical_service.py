"""Prepare and own a frozen canonical mlx2-media control plane for Studio.

The source is operator-selected and never edited. Only its media entrypoint is
launched; this is not a text serving fallback. Models stay in its existing
capability-owned worker environments and are not loaded during preparation.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from .desktop import atomic_json
from .desktop_media_canonical import CanonicalMediaClient, canonical_profile


class CanonicalService:
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / "canonical-media"
        self.config = controller.home / "canonical-media-settings.json"
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.closed = False

    def settings(self):
        primary = Path(__file__).resolve().parents[2]
        source = primary if (primary / "src/mlx2/media/server.py").is_file() else Path.home() / "Desktop/mlx2-wt-media-m8-voice"
        values = {"source": str(source)}
        if self.config.exists():
            values.update(json.loads(self.config.read_text()))
        return values

    def configure(self, values):
        if not isinstance(values, dict) or set(values) != {"source"} or not isinstance(values["source"], str):
            raise ValueError("Canonical media settings require one source checkout path")
        source = Path(values["source"]).expanduser().resolve(strict=True)
        if not source.is_dir() or not (source / "src/mlx2/media/server.py").is_file() or not (source / "pyproject.toml").is_file():
            raise ValueError("Select a canonical mlx2 source checkout containing the media server and its dependency contract")
        if self.thread and self.thread.is_alive():
            raise ValueError("Wait for or cancel canonical media preparation before changing source")
        atomic_json(self.config, {"source": str(source)})
        return {"saved": True, "settings": self.settings()}

    def snapshot(self):
        return {"settings": self.settings(), "preparing": bool(self.thread and self.thread.is_alive()),
            "endpoint": "http://127.0.0.1:8777", "model_load_on_start": False,
            "jobs": [j for j in self.controller.store.jobs(50) if j["kind"] == "canonical_media_prepare"],
            "runtimes": {"audio": str(Path.home() / ".local/share/mlx2/envs/audio/.venv/bin/python"),
                         "indextts": str(self.controller.primary.library.root / "_runtimes/mlx-indextts2/.venv/bin/python")}}

    def start(self):
        with self.lock:
            if self.closed:
                raise ValueError("Canonical media controller is closing")
            try:
                client = CanonicalMediaClient(canonical_profile("voice"))
                status = client.request("GET", "/v1/media/status")
                if status.get("service") == "mlx2-media":
                    self.connect_profiles()
                    return {"state": "reused", "endpoint": client.base}
            except Exception:  # noqa: BLE001 -- preparation is authorized when the canonical endpoint is absent
                self.controller.event("canonical_media_prepare_requested")
            if self.thread and self.thread.is_alive():
                raise ValueError("Canonical media preparation is already running")
            source = Path(self.settings()["source"]).expanduser().resolve(strict=True)
            if not (source / "src/mlx2/media/server.py").is_file():
                raise ValueError("Configure a source checkout containing the canonical media service")
            identity = self.controller.store.enqueue("canonical_media_prepare", {"source": str(source), "entrypoint": "mlx2.media.server"})
            self.cancel.clear()
            self.thread = threading.Thread(target=self.run, args=(identity, source), daemon=True, name="mlx2-canonical-media-prepare")
            self.thread.start()
            return {"job_id": identity, "state": "preparing"}

    def connect_profiles(self):
        profiles = self.controller.preferences.get("media_services", [])
        defaults = [canonical_profile(cap) for cap in ("image", "video", "voice", "voice_training")]
        self.controller.configure_preferences({"media_services": [p for p in profiles if p["id"] not in {d["id"] for d in defaults}] + defaults})

    def run(self, identity, source):
        from .desktop import stop_group
        from .desktop_runner import PreparationCancelled, RunnerManager
        child = None
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_path = self.home / (identity + ".log")
        try:
            def progress(message):
                self.controller.store.update(identity, "running", message)
            runner = RunnerManager(self.controller.home, source=source).ensure(source, "canonical-media-control-plane",
                multimodal=False, cancel=self.cancel, progress=progress, log_path=log_path,
                purpose="canonical-media", extras_override="media,observability")
            if self.cancel.is_set():
                raise PreparationCancelled("Canonical media preparation cancelled")
            progress("Starting the frozen lightweight canonical media service; no model loads")
            environment = os.environ.copy()
            environment.update(PYTHONPATH=str(Path(runner["source"]) / "src"), MLX2_MEDIA_SOURCE_ROOT=runner["source"])
            # The canonical service owns its existing durable store, output root,
            # reference contracts and source-pinned per-route environments.
            args = [runner["python"], "-m", "mlx2.media.server", "--host", "127.0.0.1", "--port", "8777"]
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(args, cwd=runner["source"], env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            import psutil
            receipt = {"pid": child.pid, "created_at": psutil.Process(child.pid).create_time(), "command": args,
                       "runner": runner, "endpoint": "http://127.0.0.1:8777", "job_id": identity}
            client = CanonicalMediaClient(canonical_profile("voice"))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if self.cancel.is_set():
                    raise PreparationCancelled("Canonical media preparation cancelled")
                if child.poll() is not None:
                    raise ValueError("Canonical media control plane failed; inspect " + str(log_path))
                try:
                    if client.request("GET", "/v1/media/status").get("service") == "mlx2-media":
                        if self.cancel.is_set():
                            raise PreparationCancelled("Canonical media preparation cancelled")
                        atomic_json(self.home / "process.json", receipt)
                        self.connect_profiles()
                        self.controller.store.update(identity, "completed", "Canonical media ready; model routes retain their own prerequisite and qualification labels", receipt)
                        child = None  # shared service remains alive when the UI closes
                        return
                except PreparationCancelled:
                    raise
                except Exception:  # noqa: BLE001 -- bounded control-plane readiness polling only
                    self.cancel.wait(.3)
            raise ValueError("Canonical media did not become ready; inspect " + str(log_path))
        except Exception as error:  # noqa: BLE001 -- durable preparation outcome, no credentials in logs
            self.controller.store.update(identity, "cancelled" if isinstance(error, PreparationCancelled) else "failed",
                str(error) if isinstance(error, (ValueError, PreparationCancelled)) else type(error).__name__)
        finally:
            if child is not None and child.poll() is None:
                stop_group(child)

    def stop(self, *, force_shared=False):
        if force_shared is not True:
            raise ValueError("Confirm stopping the shared canonical service; its current API cannot atomically close admission")
        client = CanonicalMediaClient(canonical_profile("voice"))
        status = client.request("GET", "/v1/media/status")
        queue = status.get("queue")
        if status.get("service") != "mlx2-media" or not isinstance(queue, dict) or not isinstance(queue.get("jobs"), list) or queue["jobs"]:
            raise ValueError("Canonical service has active or unverified work; Stop refused")
        import psutil
        receipt = json.loads((self.home / "process.json").read_text())
        process = psutil.Process(receipt["pid"])
        if process.create_time() != receipt["created_at"] or process.cmdline() != receipt["command"] or os.getpgid(process.pid) != process.pid:
            raise ValueError("Canonical service process identity changed; refusing to signal it")
        self.controller.event("canonical_media_stop_manual_risk_accepted")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except psutil.TimeoutExpired:
            raise ValueError("Canonical service is still retiring its shared workers") from None
        return {"state": "stopped"}

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel.set()
        if self.thread:
            self.thread.join(timeout=30)
