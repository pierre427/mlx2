"""Serial, source-preserving quantization jobs behind the desktop control plane.

The optional oMLX dependency is confined to a conversion subprocess. Its serving
backend is never imported into mlx2's inference process.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .desktop import atomic_json, source_identity

OMLX_REVISION = "d76292753cd61faab831cc5677d1262764d1bed3"
OMLX_OQ_SHA256 = "ceacbddd9ad83e2f4339c825ebe86f862f501b860780dcd075ec5e59937f1c92"
RECIPES = {**{f"q{bits}": {"label": f"Uniform affine Q{bits}", "bits": bits, "mixed": False} for bits in (2, 4, 8)},
           **{f"oq{bits}": {"label": f"oMLX calibrated oQ{bits}", "bits": bits, "mixed": True} for bits in (2, 3, 4, 5, 6, 8)}}


def inspect_source(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("Conversion needs a model directory containing config.json")
    config_raw = (path / "config.json").read_bytes()
    config = json.loads(config_raw)
    if not isinstance(config, dict):
        raise ValueError("Source config.json must contain a model configuration object")  # noqa: TRY004 -- malformed file content, not caller argument type
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Conversion requires original BF16/FP16 weights, not an already quantized model")
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise ValueError("No safetensors source weights found")
    tensors, total = 0, 0
    identity = hashlib.sha256(config_raw)
    for shard in shards:
        if shard.is_symlink():
            raise ValueError("Source weights must be local regular files")
        with shard.open("rb") as stream:
            length = stream.read(8)
            if len(length) != 8:
                raise ValueError("Truncated safetensors header")
            size = struct.unpack("<Q", length)[0]
            if not 2 <= size <= 64 << 20:
                raise ValueError("Invalid safetensors metadata size")
            raw = stream.read(size)
            header = json.loads(raw)
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if tensor.get("dtype") not in {"BF16", "F16", "F32", "I32", "I64", "U32", "BOOL"}:
                raise ValueError("Source includes compressed or unsupported tensor storage")
            if name.endswith((".scales", ".biases")):
                raise ValueError("Source contains already quantized tensors")
            offsets = tensor.get("data_offsets", [])
            if len(offsets) != 2 or not 0 <= offsets[0] <= offsets[1] <= shard.stat().st_size - size - 8:
                raise ValueError("Safetensors tensor extends outside its file")
            tensors += 1
        total += shard.stat().st_size
        identity.update(shard.name.encode() + raw + str((shard.stat().st_size, shard.stat().st_mtime_ns, shard.stat().st_ctime_ns)).encode())
    if not tensors:
        raise ValueError("No source tensors found")
    return {"path": str(path), "weight_bytes": total, "tensors": tensors,
            "manifest_fingerprint": identity.hexdigest(), "shards": len(shards)}


def verify_omlx(path):
    path = Path(path).expanduser().resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"], text=True, timeout=5)
    oq = path / "omlx/oq.py"
    if head != OMLX_REVISION or dirty or hashlib.sha256(oq.read_bytes()).hexdigest() != OMLX_OQ_SHA256:
        raise ValueError("oMLX converter must be the clean pinned source revision " + OMLX_REVISION)
    return path


class Conversions:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.process = None
        self.preparing = False
        self.closed = False

    def snapshot(self):
        prefs = self.controller.preferences
        source = prefs.get("conversion_omlx_source", "")
        return {"recipes": [{"id": key, **value} for key, value in RECIPES.items()],
                "jobs": [j for j in self.controller.store.jobs(100) if j["kind"] in {"conversion", "conversion_prepare"}],
                "running": bool(self.thread and self.thread.is_alive()),
                "python": prefs.get("conversion_python", ""), "omlx_source": source,
                "omlx_revision": OMLX_REVISION, "omlx_configured": bool(source),
                "reserve_bytes": prefs.get("conversion_reserve_bytes", 10 << 30),
                "note": "Outputs remain unqualified. oQ uses the pinned converter's sensitivity calibration; uniform Q uses bounded CPU affine conversion. No serving fallback."}

    def plan(self, request):
        recipe = RECIPES.get(request.get("recipe"))
        if not recipe:
            raise ValueError("Choose Q2/Q4/Q8 or calibrated oQ2/oQ3/oQ4/oQ5/oQ6/oQ8")
        if not isinstance(request.get("path"), str) or not request["path"].strip():
            raise ValueError("Choose a source model before checking or converting")
        root = self.controller.primary.library.root.resolve()
        source = Path(request["path"]).expanduser().resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Choose a source from the model library")
        info = inspect_source(source)
        destination = root / (source.name + "-" + request["recipe"])
        if destination.exists():
            raise ValueError("Output already exists; conversion will never overwrite a model")
        # Calibrated conversion may create a temporary proxy and an imatrix.
        # Reserve conservatively rather than promising ideal compressed size.
        reserve = self.controller.preferences.get("conversion_reserve_bytes", 10 << 30)
        required = 2 * info["weight_bytes"] + reserve
        free = shutil.disk_usage(root).free
        if free < required:
            raise ValueError(f"Insufficient free space: need {required:,} bytes including temporary weights and reserve; {free:,} available")
        return {**info, "recipe": request["recipe"], "destination": str(destination),
                "required_free_bytes": required, "free_bytes": free, "reserve_bytes": reserve,
                "estimated_output_bytes": int(info["weight_bytes"] * min(1, (recipe["bits"] + (1 if recipe["mixed"] else .5)) / 16)),
                "checkpoint": source.name + "@" + info["manifest_fingerprint"], "qualification": "unqualified"}

    def start(self, request):
        with self.lock:
            if self.closed:
                raise ValueError("Conversion controller is closing")
            if self.thread and self.thread.is_alive():
                raise ValueError("A conversion or converter preparation is already running")
            plan = self.plan(request)  # refusal before queueing or creating any output
            python = Path(self.controller.preferences.get("conversion_python") or sys.executable).expanduser()
            if not python.is_file():
                raise ValueError("Set the conversion Python executable in Settings")
            source = self.controller.preferences.get("conversion_omlx_source", "")
            if RECIPES[plan["recipe"]]["mixed"]:
                verify_omlx(source)
                if (Path(plan["path"]) / "oq_sensitivity_map.json").exists():
                    raise ValueError("Source has an external sensitivity map; use a source without an unverified calibration override")
            plan.update(python=str(python), omlx_source=source, omlx_revision=OMLX_REVISION if RECIPES[plan["recipe"]]["mixed"] else None,
                        source_identity=source_identity())
            disk = getattr(self.controller, "disk", None)
            reservation_id = "conversion-" + uuid.uuid4().hex
            if disk is not None:
                disk.reserve(reservation_id, Path(plan["destination"]).parent,
                             plan["required_free_bytes"] - plan["reserve_bytes"], reserve=plan["reserve_bytes"])
                plan["disk_reservation_id"] = reservation_id
            try:
                identity = self.controller.store.enqueue("conversion", plan)
                self.cancel.clear()
                self.thread = threading.Thread(target=self._run, args=(identity, plan), daemon=True, name="mlx2-conversion")
                self.thread.start()
                return {"job_id": identity, "plan": plan}
            except BaseException:
                if disk is not None:
                    disk.release(reservation_id)
                raise

    def _run(self, identity, plan):
        store = self.controller.store
        folder = self.controller.home / "conversion-jobs" / identity
        child = None
        try:
            folder.mkdir(parents=True, mode=0o700)
            # A distinct staging directory preserves partial failures and never
            # exposes an incomplete artifact as a library model.
            unit = Path(plan["destination"]).parent / ".conversion-state" / identity
            unit.mkdir(parents=True, mode=0o700)
            staging = unit / "model"
            request = {**plan, "staging": str(staging), "progress_path": str(folder / "progress.json"), "out": str(folder), "job_id": identity, "mcp_url": self.controller.preferences.get("cpg_mcp_url", "http://127.0.0.1:8766/mcp")}
            atomic_json(folder / "request.json", request)
            store.update(identity, "waiting", "Waiting for exclusive CPG/radio and paired GPU ownership" if RECIPES[plan["recipe"]]["mixed"] else "Preparing bounded CPU conversion · no GPU ownership")
            from .desktop_campaign_lease import DesktopCPGLease
            mixed = RECIPES[plan["recipe"]]["mixed"]
            mcp_url = self.controller.preferences.get("cpg_mcp_url", "http://127.0.0.1:8766/mcp")
            lease = DesktopCPGLease(folder, identity, mcp_url=mcp_url, checkpoint=plan["checkpoint"],
                                    source_revision=plan["source_identity"]["source_revision"]) if mixed else contextlib.nullcontext()
            with lease as ownership:
                try:
                    if self.cancel.is_set():
                        raise InterruptedError("Conversion cancelled")
                    # Recheck disk immediately before execution; other applications
                    # may have consumed space since the plan was shown.
                    if shutil.disk_usage(Path(plan["destination"]).parent).free < plan["required_free_bytes"]:
                        raise ValueError("Free space fell below the conversion reservation")
                    environment = os.environ.copy()
                    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
                    environment["HF_HUB_OFFLINE"] = "1"
                    environment["TRANSFORMERS_OFFLINE"] = "1"
                    wrapper = Path(__file__).resolve().parents[2] / "scripts/run_with_gpu_locks.py"
                    worker = [plan["python"], "-m", "mlx2.desktop_conversion", "--run", str(folder / "request.json")]
                    args = [sys.executable, str(wrapper), "--session", ownership.owner["cpg_session"], "--label", identity,
                            "--desktop-cpg-lease", str(ownership.path), "--mcp-url", mcp_url,
                            "--receipt", str(folder / "paired-locks.json"), "--", *worker] if mixed else worker
                    with (folder / "runner.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=environment)
                        self.process = child
                        while child.poll() is None:
                            if getattr(self.controller, "disk", None) is not None and plan.get("disk_reservation_id"):
                                used = sum(path.stat().st_size for path in unit.rglob("*") if path.is_file() and not path.is_symlink())
                                self.controller.disk.used(plan["disk_reservation_id"], used)
                                self.controller.disk.check(plan["disk_reservation_id"])
                            if ownership is not None:
                                ownership.check()
                            if self.cancel.wait(.5):
                                from .desktop import stop_group
                                stop_group(child)
                                raise InterruptedError("Conversion cancelled; partial output retained outside the library")
                            progress = folder / "progress.json"
                            if progress.is_file():
                                value = json.loads(progress.read_text())
                                store.update(identity, "running", value.get("detail", "Converting"), value)
                        if child.returncode:
                            raise ValueError("Converter failed; inspect " + str(folder / "runner.log"))
                    if self.cancel.is_set():
                        raise InterruptedError("Conversion cancelled before model publication")
                    if not (staging / "config.json").is_file() or not list(staging.glob("*.safetensors")):
                        raise ValueError("Converter exited without a complete model artifact")
                    if Path(plan["destination"]).exists():
                        raise ValueError("Output path appeared during conversion; refusing overwrite")
                    receipt = {**plan, "job_id": identity, "completed_at": time.time(), "qualification": "unqualified"}
                    atomic_json(staging / "mlx2-conversion.json", receipt)
                    from .desktop_downloads import Downloads
                    if self.cancel.is_set():
                        raise InterruptedError("Conversion cancelled before model publication")
                    Downloads._publish(staging, Path(plan["destination"]))
                    store.update(identity, "completed", "Conversion complete · unqualified", receipt)
                    try:
                        self.controller.primary.library.refresh()
                    except Exception:  # noqa: BLE001 -- published artifact remains complete if UI refresh fails
                        self.controller.event("conversion_library_refresh_failed", job_id=identity)
                    if ownership is not None:
                        ownership.success = True
                finally:
                    if child is not None and child.poll() is None:
                        from .desktop import stop_group
                        stop_group(child)
        except Exception as error:  # noqa: BLE001 -- durable failure; no secrets in exception text
            store.update(identity, "cancelled" if isinstance(error, InterruptedError) else "failed", str(error) if isinstance(error, (ValueError, InterruptedError)) else type(error).__name__)
        finally:
            if child is not None and child.poll() is None:
                from .desktop import stop_group
                stop_group(child)
            self.process = None
            if getattr(self.controller, "disk", None) is not None and plan.get("disk_reservation_id"):
                self.controller.disk.release(plan["disk_reservation_id"])

    def prepare(self):
        """Fetch only pinned conversion source; never install its server dependencies."""
        with self.lock:
            if self.closed:
                raise ValueError("Conversion controller is closing")
            if self.thread and self.thread.is_alive():
                raise ValueError("Wait for the active conversion operation")
            folder = self.controller.home / "conversion-tooling" / OMLX_REVISION
            identity = self.controller.store.enqueue("conversion_prepare", {"revision": OMLX_REVISION, "destination": str(folder)})
            self.cancel.clear()
            def work():
                try:
                    self.controller.store.update(identity, "running", "Fetching the pinned Apache-2.0 oMLX converter source")
                    folder.parent.mkdir(parents=True, exist_ok=True)
                    if not folder.exists():
                        staging = folder.parent / ("prepare-" + identity)
                        self._prepare_command(["git", "clone", "--no-checkout", "--filter=blob:none", "https://github.com/jundot/omlx.git", str(staging)], identity)
                        self._prepare_command(["git", "-C", str(staging), "checkout", "--detach", OMLX_REVISION], identity)
                        verify_omlx(staging)
                        if folder.exists():
                            raise ValueError("Converter destination appeared during preparation; preserving both checkouts")
                        staging.rename(folder)
                    verify_omlx(folder)
                    if self.cancel.is_set():
                        raise InterruptedError("Converter preparation cancelled")
                    # No dependency installer: oMLX's dependency metadata contains
                    # upstream git dependencies disallowed by this repository.
                    self.controller.configure_preferences({"conversion_omlx_source": str(folder)})
                    self.controller.store.update(identity, "completed", "Pinned converter prepared. Select a compatible isolated Python environment in Settings; no dependencies were installed.")
                except Exception as error:  # noqa: BLE001 -- retain preparation failure without stderr credentials
                    self.controller.store.update(identity, "cancelled" if isinstance(error, InterruptedError) else "failed", str(error) if isinstance(error, (ValueError, InterruptedError)) else type(error).__name__)
            self.thread = threading.Thread(target=work, daemon=True, name="mlx2-conversion-prepare")
            self.thread.start()
            return {"job_id": identity}

    def _prepare_command(self, args, identity):
        """Own and retire each fixed git subprocess before releasing the job."""
        from .desktop import stop_group
        if self.cancel.is_set():
            raise InterruptedError("Converter preparation cancelled")
        log = self.controller.home / ("conversion-prepare-" + identity + ".log")
        child = None
        try:
            with log.open("ab") as stream:
                child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.process = child
                deadline = time.monotonic() + 300
                while child.poll() is None:
                    if self.cancel.wait(.1):
                        raise InterruptedError("Converter preparation cancelled")
                    if time.monotonic() >= deadline:
                        raise ValueError("Converter source fetch timed out; inspect " + str(log))
                if self.cancel.is_set():
                    raise InterruptedError("Converter preparation cancelled")
                if child.returncode:
                    raise ValueError("Converter source fetch failed; inspect " + str(log))
        finally:
            if child is not None and child.poll() is None:
                stop_group(child)
            self.process = None

    def dispatch(self, request):
        if request["command"] == "conversion_plan":
            return self.plan(request)
        if request["command"] == "conversion_start":
            return self.start(request)
        if request["command"] == "conversion_prepare":
            return self.prepare()
        if request["command"] == "conversion_cancel":
            self.cancel.set()
            return {"cancel_requested": True}
        raise ValueError("Unknown conversion operation")

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel.set()
        if self.thread:
            self.thread.join(timeout=25)


def run_conversion(request_path):
    request = json.loads(Path(request_path).read_text())
    progress = Path(request["progress_path"])
    destination = Path(request["staging"])
    if destination.exists():
        raise ValueError("Conversion staging directory already exists")
    if RECIPES[request["recipe"]]["mixed"]:
        from .desktop_campaign_worker import check_ownership
        check_ownership(request)  # before any converter or MLX import
    source = Path(request["path"])
    if inspect_source(source)["manifest_fingerprint"] != request["manifest_fingerprint"]:
        raise ValueError("Source model changed after conversion was queued")
    recipe = RECIPES[request["recipe"]]
    def update(phase, percent, detail="", meta=None):
        if shutil.disk_usage(destination.parent).free < request["reserve_bytes"]:
            raise ValueError("Conversion stopped to preserve the free-space reserve")
        atomic_json(progress, {"phase": phase, "percent": percent, "detail": detail or phase, "meta": meta or {}})
    update("preparing", 0, "Preparing the isolated conversion runtime")
    if recipe["mixed"]:
        tooling = verify_omlx(request["omlx_source"])
        sys.path.insert(0, str(tooling))
        from omlx.oq import quantize_oq_streaming
        quantize_oq_streaming(str(source), str(destination), recipe["bits"], progress_callback=update,
                              trust_remote_code=False, enhanced=False)
    else:
        # CPU-only original affine conversion. The output stays a candidate;
        # architecture compatibility is established by subsequent model tests.
        from .desktop_conversion_uniform import convert
        convert(source, destination, recipe["bits"], update)
    update("completed", 100, "Conversion complete · qualification pending")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--run":
        raise SystemExit("Use the desktop conversion queue")
    run_conversion(sys.argv[2])
