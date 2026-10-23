"""Private newline-JSON companion for the native macOS app.

No HTTP control surface, weight loading, or Metal imports in this process.
Only an explicit start command spawns an inference worker. EOF stops owned work.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .desktop_runner import PreparationCancelled, RunnerManager
from .http_security import load_secret_file
from .inference_monitor import InferenceCollector
from .system_monitor import HostCollector

APP_HOME = Path.home() / "Library/Application Support/MLX2"
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
WAITERS = Path("/Users/Shared/mlxuag/gpu.lock.waiters")


def serving_port(settings):
    """Check the requested binding without changing or stopping its owner."""
    port = settings["port"] if settings.get("sticky_port", True) else 0
    with socket.socket() as probe:
        # Match the serving socket: closed connections in TIME_WAIT must not
        # prevent a restart, while an active listener still owns its binding.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            raise ValueError(
                f"Cannot start model: port {port} on 127.0.0.1 is already in use. "
                "Select the existing running model to chat, choose another serving port, "
                "or turn off sticky port assignment."
            ) from None
        return probe.getsockname()[1]


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".desktop-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_safe(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_identity():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src"],
            cwd=root,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        snapshot = root / "desktop-source.json"
        revision = (
            json.loads(snapshot.read_text()).get("source_revision", "unavailable")
            if snapshot.exists()
            else "unavailable"
        )
        diff = b""
    return {
        "source_revision": revision,
        "working_src_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "desktop_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def read_config(path: Path) -> dict:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("config.json exceeds the desktop metadata limit")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("config.json must be an object")
    return value


def folder_usage(path: Path, cancelled=lambda: False) -> tuple[int, int, list[str]]:
    """Never follow directory or weight symlinks; count each inode once."""
    logical = allocated = 0
    seen = set()
    errors = []

    def onerror(error):
        if len(errors) < 8:
            errors.append(str(error))

    for root, directories, files in os.walk(path, followlinks=False, onerror=onerror):
        if cancelled():
            raise InterruptedError("library scan superseded")
        directories[:] = [d for d in directories if not (Path(root) / d).is_symlink()]
        for name in files:
            try:
                info = (Path(root) / name).lstat()
                identity = (info.st_dev, info.st_ino)
                if stat.S_ISREG(info.st_mode) and identity not in seen:
                    seen.add(identity)
                    logical += info.st_size
                    allocated += info.st_blocks * 512
            except OSError as error:
                onerror(error)
    return logical, allocated, errors


def artifact_row(path: Path, *, cancelled=lambda: False) -> dict:
    info = path.lstat()  # Capture folder timestamps before enumeration.
    token = f"{info.st_dev}:{info.st_ino}:{info.st_ctime_ns}"
    row = {
        "id": str(path),
        "name": path.name,
        "path": str(path),
        "identity": token,
        "atime": info.st_atime,
        "mtime": info.st_mtime,
        "ctime": info.st_ctime,
        "created": getattr(info, "st_birthtime", None),
        "kind": "Other",
        "can_start": False,
        "reason": "No supported serving metadata",
        "errors": [],
    }
    if path.is_symlink():
        return {
            **row,
            "kind": "Link",
            "reason": "Linked folders are read-only",
            "logical_bytes": None,
            "allocated_bytes": None,
        }
    logical, allocated, errors = folder_usage(path, cancelled)
    row.update(logical_bytes=logical, allocated_bytes=allocated, errors=errors)
    try:
        config = read_config(path / "config.json")
        architectures = config.get("architectures") or []
        if "dflash_config" in config or any("Draft" in str(x) for x in architectures):
            row.update(
                kind="Draft", reason="Auxiliary artifact; not a standalone target"
            )
        elif config.get("model_type"):
            from .adapters.registry import _RESOLVERS

            # A candidate, never a qualification claim. The registry owns the
            # full artifact and execution-kind check immediately before start.
            if config["model_type"] in _RESOLVERS:
                row.update(
                    kind="Serving candidate",
                    can_start=True,
                    reason="Adapter and artifact checked on start; unqualified",
                )
                # Artifact inspection is GPU-free and lives on the library's
                # background scan, never on the telemetry request thread.
                from .adapters.registry import inspect_model
                try:
                    resolution = inspect_model(path)
                    row["checkpoint"] = f"{path.name}@{resolution.artifact_fingerprint}"
                    row["default_route"] = resolution.default_route
                except Exception as error:  # noqa: BLE001 -- inspection failure cannot discard the rest of the library
                    row["artifact_inspection_error"] = str(error)
            else:
                row.update(
                    kind="Other model",
                    reason="No desktop serving adapter; use the artifact's dedicated runtime",
                )
        else:
            row.update(
                kind="Media / other", reason="Use this artifact's dedicated runtime"
            )
    except FileNotFoundError:
        if (path / "model_index.json").exists():
            row.update(kind="Media", reason="Use this artifact's dedicated runtime")
    except (OSError, ValueError, TypeError) as error:
        row["reason"] = str(error)
    return row


class Library:
    """Scan off the request thread; opening the menu never traverses weights."""

    def __init__(self, root: Path):
        self.root = root
        self.rows = []
        self.error = None
        self.scanning = False
        self.generation = 0
        self.guard = threading.Lock()
        self.refresh()

    def refresh(self, root: Path | None = None):
        with self.guard:
            if root is None and self.scanning:
                return
            self.generation += 1
            generation = self.generation
            if root is not None:
                self.root = root
                self.rows = []
            target = self.root
            self.scanning = True
            self.error = None

        def scan():
            rows = []
            error = None
            try:
                for path in sorted(target.iterdir(), key=lambda p: p.name.casefold()):
                    if generation != self.generation:
                        return
                    if path.name.startswith(".") or not path.is_dir():
                        continue
                    try:
                        rows.append(
                            artifact_row(
                                path, cancelled=lambda: generation != self.generation
                            )
                        )
                    except (OSError, ValueError) as exc:
                        error = str(exc)
                    with self.guard:
                        if generation == self.generation:
                            self.rows = list(rows)
            except OSError as exc:
                error = str(exc)
            finally:
                with self.guard:
                    if generation == self.generation:
                        self.rows = rows
                        self.error = error
                        self.scanning = False

        threading.Thread(target=scan, name="mlx2-library", daemon=True).start()

    def snapshot(self):
        with self.guard:
            return {
                "root": str(self.root),
                "rows": list(self.rows),
                "scanning": self.scanning,
                "error": self.error,
            }

    def checked_path(self, raw: str, identity: str) -> Path:
        path = Path(raw)
        if path.parent != self.root or path.name.startswith(".") or path.is_symlink():
            raise ValueError(
                "Only real, immediate model folders in the selected library can be changed"
            )
        if not path.is_dir() or path.resolve().parent != self.root:
            raise ValueError("Artifact left the selected library")
        info = path.lstat()
        if identity != f"{info.st_dev}:{info.st_ino}:{info.st_ctime_ns}":
            raise ValueError(
                "Folder changed since it was listed; refresh before continuing"
            )
        return path


class GPUReservation:
    """Interop with the project's paired directory locks and idle sentinels.

    Interactive serving owns the reservation until Stop. It is not a campaign,
    CPG lease, qualification result, or performance measurement.
    """

    def __init__(self, locks=LOCKS, waiters=WAITERS):
        self.locks = locks
        self.waiters = waiters
        self.owned = []
        self.owner = {
            "lease_id": f"mlx2-desktop-{uuid.uuid4().hex}",
            "pid": os.getpid(),
            "since": time.time(),
            "cpg_used": False,
            "purpose": "interactive-serving",
        }

    def acquire(self):
        descriptors = []
        try:
            for waiter in self.waiters.glob("*"):
                fields = waiter.name.rsplit(".", 3)
                if len(fields) != 4 or time.time() - waiter.stat().st_mtime > 300:
                    continue
                try:
                    os.kill(int(fields[-2]), 0)
                except (ValueError, ProcessLookupError):
                    continue
                raise RuntimeError("GPU work is queued; start after the queue clears")
            for path in self.locks:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.is_symlink():
                    raise RuntimeError(f"Refusing linked GPU lock: {path}")
                if path.exists():
                    if not path.is_file() or path.stat().st_size:
                        raise RuntimeError(f"GPU already reserved: {path}")
                    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                    descriptors.append(fd)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    info = os.fstat(fd)
                    current = path.stat()
                    if (info.st_ino, info.st_dev) != (current.st_ino, current.st_dev):
                        raise RuntimeError("GPU sentinel changed while acquiring")
            for path in self.locks:
                backup = None
                if path.exists():
                    backup = path.with_name(f"{path.name}.desktop-{uuid.uuid4().hex}")
                    path.rename(backup)
                try:
                    path.mkdir()
                except BaseException:
                    if backup is not None and not path.exists():
                        backup.rename(path)
                    raise
                self.owned.append((path, backup))
                atomic_json(path / "owner.json", self.owner)
            if any(
                json.loads((p / "owner.json").read_text()) != self.owner
                for p in self.locks
            ):
                raise RuntimeError("Paired GPU ownership receipts differ")
        except BaseException:
            self.release()
            raise
        finally:
            for fd in descriptors:
                os.close(fd)

    def release(self):
        errors = []
        for path, backup in reversed(self.owned):
            try:
                receipt = path / "owner.json"
                if receipt.exists() and json.loads(receipt.read_text()) != self.owner:
                    raise RuntimeError("GPU owner changed; refusing removal")
                receipt.unlink(missing_ok=True)
                path.rmdir()
                if backup is not None:
                    backup.rename(path)
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(str(error))
        self.owned = []
        if errors:
            raise RuntimeError("GPU reservation cleanup: " + "; ".join(errors))


def model_processes() -> list[dict]:
    """Use psutil argv arrays so paths with spaces are never split by regex."""
    import psutil

    found = []
    modules = {
        "mlx2.server",
        "mlx2.decisions.server",
        "mlx2.image_mcp.server",
        "mlx_lm.server",
        "mlx_vlm.server",
    }
    executables = {
        "mlx2-serve",
        "mlx2-image-mcp",
        "mlx2-decisions",
        "llama-server",
        "rapid-mlx",
    }
    for process in psutil.process_iter(["pid", "cmdline", "name"]):
        args = process.info["cmdline"] or []
        if not args:
            continue
        is_model = any(Path(a).name in executables for a in args[:2]) or any(
            a == "-m" and i + 1 < len(args) and args[i + 1] in modules
            for i, a in enumerate(args)
        )
        if not is_model:
            continue
        model = None
        for i, arg in enumerate(args):
            if arg == "--model" and i + 1 < len(args):
                model = args[i + 1]
            elif arg.startswith("--model="):
                model = arg.partition("=")[2]
        found.append({"pid": process.info["pid"], "model": model})
    return found


def child_exited(child: subprocess.Popen) -> bool:
    """Observe exit without reaping: its PID must not be reused before cleanup."""
    import psutil

    try:
        return psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def stop_group(child: subprocess.Popen) -> int:
    """Retire descendants before reaping the group leader and releasing locks."""
    import psutil

    def live_members():
        live = []
        for process in psutil.process_iter(["pid"]):
            try:
                if (
                    process.pid != os.getpid()
                    and os.getpgid(process.pid) == child.pid
                    and process.status() != psutil.STATUS_ZOMBIE
                ):
                    live.append(process)
            except (ProcessLookupError, psutil.NoSuchProcess):
                pass
        return live

    def send(signum):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin returns EPERM for a group containing only zombies.
            if live_members():
                raise

    send(signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and live_members():
        time.sleep(0.05)
    # A group may still contain a late-spawned child. SIGKILL is harmless for
    # a zombie leader and closes that race while its PID is still reserved.
    send(signal.SIGKILL)
    deadline = time.monotonic() + 5
    while live_members():
        if time.monotonic() >= deadline:
            raise RuntimeError("Worker group is still alive; retaining GPU ownership")
        time.sleep(0.05)
    return child.wait(timeout=10)


class Supervisor:
    def __init__(self, home: Path, trash_helper: Path, *, reservation_factory=None, owned_pids=None, context_resolver=None):
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.home = home
        self.trash_helper = trash_helper
        self.lock_file = (home / "desktop.lock").open("a")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another MLX2 desktop supervisor is already running"
            ) from None
        self.settings = {
            "root": str(Path.home() / "mlx-models"),
            "port": 8285,
            "context": 0,
            "lanes": 1,
            "route": "auto",
            "sticky_port": True,
        }
        settings_path = home / "desktop-settings.json"
        if settings_path.exists():
            self.settings.update(read_config(settings_path))
        if self.settings["lanes"] == 0:
            self.settings["lanes"] = 1
        self.settings = self.valid_settings(self.settings, allow_missing_root=True)
        self.history = {}
        history_path = home / "desktop-history.json"
        if history_path.exists():
            self.history = read_config(history_path)
        self.library = Library(Path(self.settings["root"]))
        self.host = HostCollector(interval_seconds=2, powermetrics=False)
        self.external = InferenceCollector(discover=True, timeout_seconds=0.5)
        self.collector = None
        self.child = None
        self.reservation = None
        self.worker = None
        self.log_path = None
        self.worker_error = None
        self.started = None
        self.recorded_load = False
        self.health_misses = 0
        self.health_miss_since = None
        self.runner_manager = RunnerManager(home)
        self.preparation_thread = None
        self.preparation_cancel = threading.Event()
        self.lifecycle_lock = threading.RLock()
        self.reservation_factory = reservation_factory
        self.owned_pids = owned_pids or (lambda: set())
        self.context_resolver = context_resolver
        from .desktop_suspend import Suspension
        self.suspension = Suspension(self)

    @staticmethod
    def valid_settings(settings, *, allow_missing_root=False):
        expected = {"root", "port", "context", "lanes", "route"}
        if set(settings) - (expected | {"sticky_port", "lora_root", "max_loras", "max_lora_rank"}) or not expected <= set(settings):
            raise ValueError("Unknown or missing desktop settings")
        settings = {"sticky_port": True, **settings}
        if type(settings["sticky_port"]) is not bool:
            raise ValueError("Sticky port must be boolean")
        root = (
            Path(settings["root"]).expanduser().resolve(strict=not allow_missing_root)
        )
        if not root.is_dir() and not (allow_missing_root and not root.exists()):
            raise ValueError("Choose a model directory")
        for key, low, high in (
            ("port", 1024, 65535),
            ("context", 0, 1048576),
            ("lanes", 0, 32),
        ):
            if type(settings[key]) is not int or not low <= settings[key] <= high:
                raise ValueError(f"Invalid {key}: expected {low}–{high}")
        if settings["route"] not in {"auto", "ordinary"}:
            raise ValueError("Choose automatic or ordinary decode")
        if any(key in settings for key in ("lora_root", "max_loras", "max_lora_rank")):
            lora_root = settings.get("lora_root")
            slots, rank = settings.get("max_loras", 1 if lora_root else 0), settings.get("max_lora_rank", 16)
            if type(slots) is not int or not 0 <= slots <= 64 or type(rank) is not int or not 1 <= rank <= 1024:
                raise ValueError("LoRA slots must be 0–64 and rank 1–1024")
            if lora_root is not None:
                if not isinstance(lora_root, str) or not lora_root:
                    raise ValueError("LoRA root must be a real adapter directory")
                adapter_root = Path(lora_root).expanduser()
                if adapter_root.is_symlink() or not adapter_root.is_dir():
                    raise ValueError("LoRA root must be a real adapter directory")
                lora_root = str(adapter_root.resolve(strict=True))
                if settings["route"] != "ordinary":
                    raise ValueError("Desktop LoRA requires the ordinary route")
            elif slots:
                raise ValueError("LoRA slots require an explicit adapter directory")
            settings = {**settings, "lora_root": lora_root, "max_loras": slots, "max_lora_rank": rank}
        return {**settings, "root": str(root)}

    def configure(self, settings):
        self.reap()
        if self.child is not None or self.preparing() or self.suspension.busy() or (self.worker or {}).get("state") in {"Suspended", "SuspensionRecoveryRequired"}:
            raise ValueError("Stop the managed or suspended model before changing launch settings")
        settings = self.valid_settings(settings)
        atomic_json(self.home / "desktop-settings.json", settings)
        self.settings = settings
        self.library.refresh(Path(settings["root"]))
        return settings

    def reap(self):
        with self.lifecycle_lock:
            if self.child is not None and child_exited(self.child):
                code = stop_group(self.child)
                self.child = None
                self.worker_error = f"Worker exited ({code}). See Logs." if code else None
                self.retire()

    def retire(self):
        if self.collector is not None:
            self.collector.close()
            self.collector = None
        if self.reservation is not None:
            self.reservation.release()
            self.reservation = None
        if self.worker:
            self.worker["state"] = "Error" if self.worker_error else "Stopped"

    def preparing(self):
        return (
            self.preparation_thread is not None and self.preparation_thread.is_alive()
        )

    def begin_start(self, raw: str, identity: str):
        with self.lifecycle_lock:
            self.reap()
            if self.child is not None or self.preparing() or self.suspension.busy() or (self.worker or {}).get("state") in {"Suspended", "SuspensionRecoveryRequired"}:
                raise ValueError("Stop the current model or discard its suspension before starting another")
            path = self.library.checked_path(raw, identity)
            self.preparation_cancel.clear()
            logs = self.home / "preparations"
            logs.mkdir(exist_ok=True, mode=0o700)
            self.log_path = logs / f"{uuid.uuid4().hex}.log"
            self.log_path.touch(mode=0o600)
            self.worker_error = None
            self.worker = {
                "path": str(path),
                "name": path.name,
                "state": "Preparing",
                "qualification": "unqualified",
                "log_path": str(self.log_path),
                "preparation_message": "Checking the model's runner",
            }

            def prepare_and_start():
                try:
                    self.start(raw, identity)
                except Exception as error:  # noqa: BLE001 -- surface async preparation errors
                    with self.lifecycle_lock:
                        if self.preparation_cancel.is_set():
                            self.worker_error = None
                            self.worker["state"] = "Stopped"
                        else:
                            self.worker_error = str(error)
                            self.worker["state"] = "Error"

            self.preparation_thread = threading.Thread(
                target=prepare_and_start, name="mlx2-runner-prepare", daemon=True
            )
            self.preparation_thread.start()
            return dict(self.worker)

    def start(self, raw: str, identity: str):
        self.reap()
        if self.child is not None or self.suspension.busy() or (self.worker or {}).get("state") in {"Suspended", "SuspensionRecoveryRequired"}:
            raise ValueError("Stop the current model or discard its suspension before starting another")
        path = self.library.checked_path(raw, identity)
        from .adapters.registry import inspect_model

        resolution = inspect_model(path)
        execution = resolution.descriptor.metadata.get(
            "execution_kind", "autoregressive"
        )
        if execution not in {"autoregressive", "embeddings", "rerank"}:
            raise ValueError(
                f"{execution} needs its dedicated runtime; desktop serving is not applicable"
            )
        if self.log_path is None:
            logs = self.home / "preparations"
            logs.mkdir(exist_ok=True, mode=0o700)
            self.log_path = logs / f"{uuid.uuid4().hex}.log"
            self.log_path.touch(mode=0o600)

        def progress(message):
            if self.worker:
                self.worker["preparation_message"] = message

        capabilities = {
            str(getattr(capability, "value", capability))
            for capability in resolution.descriptor.capabilities
        }
        runner = self.runner_manager.ensure(
            path,
            resolution.artifact_fingerprint,
            multimodal=bool(capabilities & {"vision", "video", "audio"}),
            cancel=self.preparation_cancel,
            progress=progress,
            log_path=self.log_path,
        )
        # Environment preparation is CPU-only and may run beside an existing
        # server. Only the actual worker launch needs exclusive GPU ownership.
        with self.lifecycle_lock:
            if self.preparation_cancel.is_set():
                raise PreparationCancelled("Runner preparation cancelled")
            return self.launch(path, resolution, execution, runner)

    def launch(self, path, resolution, execution, runner, *, resume_worker=None):
        if self.settings.get("lora_root") and (execution != "autoregressive" or self.settings["route"] != "ordinary"):
            raise ValueError("Desktop LoRA requires an autoregressive ordinary route")
        if any(p["pid"] not in self.owned_pids() for p in model_processes()):
            raise RuntimeError(
                "An external model process is running. Stop it through its owner before starting a managed model."
            )
        port = serving_port(self.settings)
        reservation = self.reservation_factory() if self.reservation_factory else GPUReservation()
        reservation.acquire()
        self.reservation = reservation
        try:
            if self.preparation_cancel.is_set():
                raise PreparationCancelled("Runner preparation cancelled before model launch")
            if any(p["pid"] not in self.owned_pids() for p in model_processes()):
                raise RuntimeError("An external model appeared during GPU reservation")
            run = self.home / "runs" / uuid.uuid4().hex
            run.mkdir(parents=True, mode=0o700)
            key = uuid.uuid4().hex + uuid.uuid4().hex
            key_path = run / "api-key"
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(key)
            self.log_path = run / "server.log"
            args = [
                runner["python"],
                "-u",
                "-m",
                "mlx2.server",
                "--model",
                str(path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--api-key-file",
                str(key_path),
                "--drain-on-sigterm",
                "15",
                "--fabric",
                "off",
            ]
            context = self.context_resolver(path, self.settings["context"]) if self.context_resolver else self.settings["context"]
            if context:
                args.extend(["--max-context", str(context)])
            if self.settings["lanes"]:
                args.extend(
                    [
                        "--max-lanes",
                        str(self.settings["lanes"]),
                        "--max-inflight",
                        str(max(4, self.settings["lanes"])),
                    ]
                )
            args.extend(["--drafter-root", str(self.library.root)])
            if self.settings["route"] == "ordinary" and execution == "autoregressive":
                args.append("--ordinary")
            if self.settings.get("lora_root"):
                args.extend(["--lora-dir", self.settings["lora_root"],
                             "--max-loras", str(self.settings.get("max_loras", 1)),
                             "--max-lora-rank", str(self.settings.get("max_lora_rank", 16))])
            lineage = resume_worker["resume_lineage_id"] if resume_worker else uuid.uuid4().hex
            binding = hashlib.sha256(json.dumps({"checkpoint": resolution.artifact_fingerprint, "runner": runner.get("binding", {})}, sort_keys=True).encode()).hexdigest()
            persist_dir = Path(resume_worker["apc_persist_dir"]) if resume_worker else self.home / "apc-state" / binding / lineage
            if execution == "autoregressive":
                persist_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                args.extend(["--apc-persist-dir", str(persist_dir), "--admin-token-file", str(key_path)])
            self.worker = {
                "path": str(path),
                "name": path.name,
                "state": "Loading",
                "url": f"http://127.0.0.1:{port}",
                "checkpoint": f"{path.name}@{resolution.artifact_fingerprint}",
                "qualification": "unqualified",
                "log_path": str(self.log_path),
                "api_key_file": str(key_path),
                "execution_kind": execution,
                "runner": runner,
                "resume_lineage_id": lineage,
                "effective_context": context,
                "launch_settings": dict(self.settings),
                "apc_persist_dir": str(persist_dir) if execution == "autoregressive" else None,
                "admin_token_file": str(key_path) if execution == "autoregressive" else None,
                **({key: resume_worker[key] for key in ("resume_previous_pid", "resume_previous_url", "resume_previous_checkpoint") if key in resume_worker} if resume_worker else {}),
            }
            atomic_json(
                run / "launch.json",
                {
                    "arguments": args,
                    "worker": self.worker,
                    "ownership": reservation.owner,
                    "source": source_identity(),
                    "runner": runner,
                },
            )
            with self.log_path.open("ab") as log:
                self.child = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    cwd=runner["source"],
                    env={
                        **os.environ,
                        "PYTHONPATH": str(Path(runner["source"]) / "src"),
                    },
                )
            self.started = time.monotonic()
            self.recorded_load = False
            self.health_misses = 0
            self.health_miss_since = None
            self.worker_error = None
            self.collector = InferenceCollector(
                urls=(self.worker["url"],),
                discover=False,
                api_key=key,
                timeout_seconds=0.5,
            )
            return self.worker
        except BaseException:
            # Do not call Stop here: that would mark a failed async launch as
            # a user cancellation and erase the useful error.
            self.stop_worker()
            raise

    def begin_suspend(self):
        return self.suspension.begin_suspend()

    def begin_resume(self, suspension_id=None):
        return self.suspension.begin_resume(suspension_id)

    def cancel_lifecycle(self):
        self.suspension.cancel.set()
        self.preparation_cancel.set()
        return {"cancel_requested": True}

    def stop(self, *, preserve_suspension=False):
        self.preparation_cancel.set()
        self.suspension.cancel.set()
        if self.suspension.busy() and self.suspension.thread is not threading.current_thread():
            self.suspension.thread.join(timeout=40)
            if self.suspension.busy():
                raise RuntimeError("Model suspension operation is still stopping; retry Stop")
        if preserve_suspension and self.child is None and (self.worker or {}).get("state") in {"Suspended", "SuspensionRecoveryRequired"}:
            return {"stopped": True, "suspended": True}
        if (
            self.preparing()
            and self.preparation_thread is not threading.current_thread()
        ):
            self.preparation_thread.join(timeout=40)
            if self.preparation_thread.is_alive():
                raise RuntimeError("Runner preparation is still stopping; retry Stop")
        with self.lifecycle_lock:
            result = self.stop_worker()
            if self.suspension.record:
                self.suspension.record["state"] = "Stopped"
                self.suspension.save()
            return result

    def stop_worker(self):
        child = self.child
        if child is not None:
            if self.worker:
                self.worker["state"] = "Stopping"
            stop_group(child)
        self.child = None
        self.worker_error = None
        self.retire()
        return {"stopped": True}

    def snapshot(self):
        self.reap()
        readings = list(self.collector.sample()) if self.collector is not None else []
        if self.child is not None and self.worker:
            readings = [
                dataclasses.replace(
                    reading,
                    endpoint=dataclasses.replace(
                        reading.endpoint,
                        pid=self.child.pid,
                        model_path=self.worker["path"],
                    ),
                )
                for reading in readings
            ]
        if (readings and self.worker and not self.suspension.busy()
                and self.worker.get("state") not in {"ResumeError", "SuspendError", "SuspensionRecoveryRequired"}):
            reading = readings[0]
            # Fresh /metrics readiness is authoritative for liveness. A slow
            # optional detail-status request must not invalidate a healthy engine.
            if reading.reachable and reading.ready is True:
                self.health_misses = 0
                self.health_miss_since = None
                self.worker.pop("health_check", None)
                self.worker["state"] = "Busy" if reading.running else "Ready"
                self.worker["qualification"] = reading.qualification or "unqualified"
                if not self.recorded_load:
                    self.history[self.worker["path"]] = time.time()
                    atomic_json(self.home / "desktop-history.json", self.history)
                    self.recorded_load = True
            elif (
                not self.recorded_load
                and self.started
                and time.monotonic() - self.started > 600
            ):
                # Do not leave an indefinitely loading worker holding ownership.
                self.stop()
                self.worker_error = (
                    "Model did not become ready within 10 minutes. See Logs."
                )
                self.worker["state"] = "Error"
            elif self.recorded_load:
                now = time.monotonic()
                self.health_misses += 1
                if self.health_miss_since is None:
                    self.health_miss_since = now
                elapsed = max(0, now - self.health_miss_since)
                self.worker["health_check"] = {"state": "delayed", "consecutive_misses": self.health_misses,
                                                "seconds_since_first_failure": elapsed}
                # A real process exit is handled immediately by reap(). Poll
                # timeouts alone need sustained evidence before a reconnect state.
                if (reading.reachable and reading.ready is False) or (self.health_misses >= 3 and elapsed >= 10):
                    self.worker["state"] = "Unreachable"
        external = [
            r
            for r in self.external.sample()
            if self.worker is None
            or r.endpoint.url != self.worker.get("url")
            or self.child is None
        ]
        return json_safe(
            {
                "settings": self.settings,
                "library": self.library.snapshot(),
                "history": self.history,
                "worker": self.worker,
                "error": self.worker_error,
                "host": self.host.sample(),
                "servers": readings + external,
                "sampled_at": time.time(),
            }
        )

    def trash(self, raw: str, identity: str):
        path = self.library.checked_path(raw, identity)
        self.reap()
        if (
            (self.child is not None or self.preparing())
            and self.worker
            and Path(self.worker["path"]) == path
        ):
            raise ValueError("Stop this model before moving it to Trash")
        # Include loading processes, which are invisible to listener discovery.
        for process in model_processes():
            if process["model"] is None:
                raise ValueError(
                    "An external model's artifact is unknown; stop it before deleting models"
                )
            other = Path(process["model"]).expanduser().resolve()
            if other == path or path in other.parents or other in path.parents:
                raise ValueError("An external model process is using this artifact")
        opened = subprocess.run(
            ["/usr/sbin/lsof", "-nP", "+D", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if opened.returncode != 1 or opened.stdout.strip() or opened.stderr.strip():
            raise ValueError(
                "Folder has open files, or its usage could not be verified; close users and try again"
            )
        self.library.checked_path(raw, identity)
        result = subprocess.run(
            [str(self.trash_helper), "--trash", str(path), identity],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or "macOS could not move the folder to Trash"
            )
        self.library.refresh()
        return {"trashed": raw}

    def logs(self):
        if self.log_path is None:
            return {"text": "No managed model has been started."}
        if not self.log_path.exists():
            return {"text": "The selected log file is not available yet."}
        with self.log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 64000))
            return {"text": stream.read().decode("utf-8", "replace")}

    def chat_target(self, url, pid):
        """Resolve credentials only for a freshly discovered local server.

        Secrets travel over the private pipe for this request only; they never
        enter snapshots, receipts, logs or saved chat history.
        """
        import psutil

        from .inference_monitor import discover_servers

        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not parsed.port
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Chat requires a discovered local MLX2 server")
        endpoint = next(
            (e for e in discover_servers() if e.url == url and e.pid == pid), None
        )
        if endpoint is None:
            raise ValueError("This server changed or stopped; select it again")
        process = psutil.Process(pid)
        args = process.cmdline()
        key = None
        for index, argument in enumerate(args):
            flag, _, inline = argument.partition("=")
            if flag not in {"--api-key-file", "--api-key-env"}:
                continue
            value = inline or (args[index + 1] if index + 1 < len(args) else "")
            if flag == "--api-key-file":
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = Path(process.cwd()) / path
                key = load_secret_file(path, flag="--api-key-file")
            else:
                key = process.environ().get(value)
                if not key:
                    raise ValueError("The server API key is unavailable")
        return {"url": url, "api_key": key}

    def dispatch(self, request):
        command = request.get("command")
        if command == "snapshot":
            return self.snapshot()
        if command == "refresh":
            self.library.refresh()
            return {"refreshing": True}
        if command == "configure":
            return self.configure(request["settings"])
        if command == "start":
            return self.begin_start(request["path"], request["identity"])
        if command == "suspend":
            return self.begin_suspend()
        if command == "resume":
            return self.begin_resume(request.get("suspension_id"))
        if command == "cancel_lifecycle":
            return self.cancel_lifecycle()
        if command == "stop":
            if request.get("path") is not None and (
                self.worker is None or request["path"] != self.worker["path"]
            ):
                raise ValueError(
                    "This model is no longer the managed worker; refresh its status"
                )
            return self.stop()
        if command == "trash":
            return self.trash(request["path"], request["identity"])
        if command == "logs":
            return self.logs()
        if command == "chat_target":
            return self.chat_target(request["url"], request["pid"])
        raise ValueError("Unknown desktop command")

    def close(self):
        try:
            self.stop(preserve_suspension=True)
        finally:
            self.host.close()
            self.external.close()
            self.lock_file.close()


@contextlib.contextmanager
def _protocol_output():
    """Reserve one private wire FD; every ordinary stdout write is diagnostic.

    redirect_stdout changes process-global Python state and cannot isolate
    background workers or C/subprocess writes. Redirect FD 1 permanently before
    creating the controller, retaining only a non-inheritable duplicate for JSON.
    """
    original = sys.stdout
    original.flush()
    with os.fdopen(os.dup(original.fileno()), "w", encoding="utf-8", buffering=1) as wire:
        os.dup2(sys.stderr.fileno(), original.fileno())
        sys.stdout = sys.stderr
        yield wire


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=APP_HOME)
    parser.add_argument("--trash-helper", type=Path, required=True)
    args = parser.parse_args(argv)
    with _protocol_output() as wire:
        _serve_companion(args, wire)


def _serve_companion(args, wire):
    from .desktop_control import DesktopController

    supervisor = DesktopController(args.home, args.trash_helper)

    def terminate(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("Expected a JSON object")
                result = supervisor.dispatch(request)
                reply = {
                    "id": request.get("id"),
                    "ok": True,
                    "result": json_safe(result),
                }
            except Exception as error:  # noqa: BLE001 -- report command errors without abandoning ownership
                reply = {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": str(error),
                }
            wire.write(json.dumps(reply, allow_nan=False) + "\n")
            wire.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        supervisor.close()


if __name__ == "__main__":
    main()
