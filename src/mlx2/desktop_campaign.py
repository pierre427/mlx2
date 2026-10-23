"""Persistent, serial qualification queue for the native desktop control plane.

Construction and status calls never initialize MLX or reserve accelerator time.
Only an explicit enqueue starts work; saved jobs are not replayed at startup.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

from .desktop_campaign_process import spawn
from .desktop_records import digest

_HARNESS = "qualification/runs/qualify-1010-correctness"
QUALIFICATION_SOURCE_FILES = tuple(f"{_HARNESS}/{name}" for name in (
    "thermal_ladder.py", "thermal-policy.json", "ladder.py", "needle_match.py",
    "qualification_verdict.py", "performance_assessment.py",
))
DEFAULT_PROFILE = {"route": "auto", "context": 32768, "lanes": 4, "inflight": 4,
                   "cache_gib": 4, "prefill_step": 2048, "port": 18421,
                   "max_tokens": 512, "features": []}
ACTIVE = {"Preparing", "Loading", "Ready", "Busy", "Stopping", "Unreachable"}


def validate_profile(value):
    if not isinstance(value, dict) or set(value) - DEFAULT_PROFILE.keys():
        raise ValueError("Unknown qualification profile fields")
    value = {**DEFAULT_PROFILE, **value}
    if not isinstance(value["route"], str) or value["route"] not in {"auto", "ordinary", "native_mtp", "prompt_lookup"}:
        raise ValueError("Unknown qualification route")
    bounds = {"context": (1024, 1048576), "lanes": (1, 32), "inflight": (1, 128),
              "cache_gib": (1, 512), "prefill_step": (128, 32768), "port": (1024, 65535),
              "max_tokens": (16, 4096)}
    for key, (low, high) in bounds.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"{key} must be an integer in {low}..{high}")
    if value["inflight"] < value["lanes"]:
        raise ValueError("Inflight capacity must cover the selected lane width")
    features = value["features"]
    # These are typed harness requirements, never command fragments. The harness
    # itself checks its complete choices list and actual per-run engagement.
    if not isinstance(features, list) or len(features) > 64 or any(
        not isinstance(item, str) or not item or not item.replace("_", "").isalnum()
        for item in features
    ):
        raise ValueError("Features must be a list of harness feature identifiers")
    value["features"] = list(dict.fromkeys(features))
    return value


def comparison_rows(records):
    """Cross-model comparisons group matched host/runtime/shape, never model ID."""
    latest = {}
    for record in sorted(records, key=lambda r: r.get("created", 0), reverse=True):
        if record.get("level") != 3:
            continue
        binding = record["binding"]
        key = digest(binding)
        if key in latest:
            continue
        latest[key] = record
    rows = []
    for record in latest.values():
        if not record.get("passed"):
            continue
        binding = record["binding"]
        shape = {key: binding[key] for key in ("source_revision", "source_sha256", "host", "profile")}
        # The same build has different route-scoped source closures for different
        # adapters. Full frozen-source binding is already in source_sha256;
        # compare runtime machinery/version fields without the adapter closure.
        shape["runtime"] = {key: value for key, value in binding["runtime"].items()
                            if key not in {"source_sha256", "source_scope", "source_files", "source_manifest"}}
        from .qualification import PROVENANCE_ONLY_SETTINGS
        profile = dict(shape["profile"])
        if isinstance(profile.get("settings"), dict):
            profile["settings"] = {key: value for key, value in profile["settings"].items()
                                   if key not in PROVENANCE_ONLY_SETTINGS}
        shape["profile"] = profile
        # Model-specific automatic route choices must not compare across routes.
        for sample in record.get("samples", []):
            cell_shape = {**shape, "context_tokens": sample.get("context_tokens"),
                          "width": sample.get("width"), "max_tokens": sample.get("max_tokens")}
            rows.append({"checkpoint": binding["checkpoint"], "group": digest(cell_shape),
                         "shape": cell_shape, "prefill_tps": sample["prefill_tps"],
                         "decode_tps": sample["decode_tps"], "record_id": record.get("id"),
                         "evidence_path": record.get("evidence_path")})
    return rows


class CampaignQueue:
    def __init__(self, control):
        self.control = control
        self.home = control.home / "campaigns"
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings_path = control.home / "campaign-settings.json"
        self.settings = {"mcp_url": "http://127.0.0.1:8766/mcp", "profile": dict(DEFAULT_PROFILE)}
        if self.settings_path.exists():
            self.settings.update(json.loads(self.settings_path.read_text()))
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.child = None
        self.current = None
        self.log_path = None
        self.closed = False
        self.preflight_cache = {}
        for job in control.store.jobs(10000):
            if job["kind"] == "qualification" and job["state"] == "queued":
                control.store.update(job["id"], "interrupted", "Not resumed after application restart")

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def status(self):
        return {"running": self.running, "current_job": self.current,
                "settings": self.settings, "comparisons": comparison_rows(self.control.store.records()),
                "levels": {"1": "Load and ordinary output, with coherence review",
                           "2": "Approved feature qualifier and replicated context/cache ladder",
                           "3": "Controlled prefill and decode measurements"}}

    def logs(self):
        if self.log_path is None or not self.log_path.exists():
            return {"text": "No qualification campaign has run in this session."}
        with self.log_path.open("rb") as stream:
            stream.seek(max(0, self.log_path.stat().st_size - 128000))
            return {"text": stream.read(128000).decode("utf-8", "replace")}

    def dispatch(self, request):
        command = request.get("command")
        if command == "campaign_configure":
            changes = request.get("settings", {})
            if not isinstance(changes, dict) or set(changes) - self.settings.keys():
                raise ValueError("Unknown campaign settings")
            settings = {**self.settings, **changes}
            from urllib.parse import urlsplit
            if not isinstance(settings["mcp_url"], str):
                raise ValueError("CPG radio URL must be text")
            url = urlsplit(settings["mcp_url"])
            if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username or url.password or not url.port:
                raise ValueError("CPG radio must be a local 127.0.0.1 HTTP endpoint with a port")
            settings["profile"] = validate_profile(settings["profile"])
            from .desktop import atomic_json
            with self.lock:
                if self.running:
                    raise ValueError("Wait for the campaign before changing its settings")
                atomic_json(self.settings_path, settings)
                self.settings = settings
            return self.status()
        if command == "campaign_enqueue":
            return self.enqueue(request)
        if command == "campaign_cancel":
            self.cancel.set()
            return {"cancelling": self.running}
        if command == "campaign_review":
            return self.review(request)
        if command == "campaign_compare":
            rows = comparison_rows(self.control.store.records())
            metric = request.get("sort", "decode_tps")
            if metric not in {"decode_tps", "prefill_tps"}:
                raise ValueError("Sort by prefill_tps or decode_tps")
            return {"rows": sorted(rows, key=lambda row: (row["group"], -row[metric]))}
        raise ValueError("Unknown campaign command")

    def enqueue(self, request):
        paths = request.get("paths")
        levels = request.get("levels", [1, 2, 3])
        if not isinstance(paths, list) or not paths or len(paths) > 256 or any(not isinstance(p, str) for p in paths):
            raise ValueError("Select one or more model paths")
        if not isinstance(levels, list) or not levels or any(type(v) is not int or v not in (1, 2, 3) for v in levels):
            raise ValueError("Select qualification levels 1, 2 and/or 3")
        levels = sorted(set(levels))
        if 2 in levels and request.get("approve_preflight") is not True and not request.get("preflight_receipt"):
            raise ValueError("Level 2 requires explicit full preflight approval or a matching saved receipt")
        profile = validate_profile(request.get("profile", self.settings["profile"]))
        with self.lock:
            if self.closed or self.running:
                raise ValueError("Wait for or cancel the current campaign")
            if any((getattr(engine, "worker", None) or {}).get("state") in ACTIVE
                   or getattr(engine, "child", None) is not None
                   or (getattr(engine, "preparing", lambda: False)())
                   for engine in self.control.engines.values()):
                raise ValueError("Stop managed model servers before starting the qualification queue")
            rows = {row["path"]: row for row in self.control.primary.library.rows}
            selected = []
            seen = set()
            for path in paths:
                row = rows.get(path)
                if row is None:
                    raise ValueError("Refresh the model library before queuing this model")
                checked = self.control.primary.library.checked_path(path, row["identity"])
                # Exact paths/inodes are deduplicated; auxiliary artifacts remain separate.
                identity = (checked.stat().st_dev, checked.stat().st_ino)
                if identity not in seen:
                    seen.add(identity)
                    selected.append(row)
            selected.sort(key=lambda row: (row.get("allocated_bytes") or row.get("logical_bytes") or math.inf, row["path"]))
            ids = []
            for row in selected:
                spec = {"path": row["path"], "identity": row["identity"], "levels": levels,
                        "profile": profile, "mcp_url": self.settings["mcp_url"],
                        "approve_preflight": request.get("approve_preflight") is True,
                        "preflight_receipt": request.get("preflight_receipt")}
                ids.append(self.control.store.enqueue("qualification", spec))
            self.cancel.clear()
            self.thread = threading.Thread(target=self._run, args=(ids,), daemon=True, name="mlx2-qualification")
            self.thread.start()
        return {"job_ids": ids, "order": [row["path"] for row in selected]}

    def _run(self, ids):
        for identity in ids:
            if self.cancel.is_set():
                self.control.store.update(identity, "cancelled", "Cancelled before GPU acquisition")
                continue
            self.current = identity
            try:
                self._unit(identity)
            except Exception as error:  # noqa: BLE001 -- persistent per-model failure, continue serial queue
                state = "cancelled" if self.cancel.is_set() else "failed"
                self.control.store.update(identity, state, str(error))
                self.control.event("qualification_job_finished", job_id=identity, state=state)
        self.current = None

    def _unit(self, identity):
        job = next(j for j in self.control.store.jobs(10000) if j["id"] == identity)
        spec = job["spec"]
        path = self.control.primary.library.checked_path(spec["path"], spec["identity"])
        out = self.home / identity
        out.mkdir(mode=0o700)  # no reuse, even after a failed or cancelled attempt
        self.log_path = out / "campaign.log"
        self.log_path.touch(mode=0o600)
        self.control.store.update(identity, "running", "Inspecting exact artifact and preparing frozen runner")
        from .adapters.registry import inspect_model
        resolution = inspect_model(path)
        kind = resolution.descriptor.metadata.get("execution_kind", "autoregressive")
        caps = {str(getattr(c, "value", c)) for c in resolution.descriptor.capabilities}
        if kind != "autoregressive" or caps & {"vision", "video", "audio"}:
            raise ValueError(f"{kind} / {sorted(caps)} requires a dedicated reference and media qualification profile; text gates are not applicable")
        runner = self.control.primary.runner_manager.ensure(path, resolution.artifact_fingerprint,
            multimodal=False, cancel=self.cancel,
            progress=lambda message: self.control.store.update(identity, "running", message),
            log_path=self.log_path, purpose="qualification")
        source = Path(runner["source"])
        from .desktop import atomic_json
        manifest = json.loads((source / "desktop-source.json").read_text())
        # Git discovery must remain directory-local: exporting GIT_DIR would
        # redirect the unit suite's temporary-repository tests into our metadata.
        # This housekeeping directory is excluded from source_manifest; no
        # checkout/reset modifies any frozen source byte.
        git_dir = source / ".git"
        subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
        with (source / "desktop-source-commit.bin").open("rb") as commit:
            actual = subprocess.check_output(["git", "--git-dir", str(git_dir), "hash-object", "-t", "commit", "-w", "--stdin"], stdin=commit, text=True).strip()
        if actual != manifest["source_revision"]:
            raise ValueError("Frozen source commit object does not match recorded revision")
        subprocess.run(["git", "--git-dir", str(git_dir), "update-ref", "HEAD", manifest["source_revision"]], check=True, capture_output=True)
        spec.update(checkpoint=f"{path.name}@{resolution.artifact_fingerprint}",
                    source_revision=manifest["source_revision"], source_sha256=manifest["source_sha256"],
                    job_id=identity, out=str(out), git_dir=str(git_dir))
        atomic_json(out / "spec.json", spec)
        command = [runner["python"], "-u", "-m", "mlx2.desktop_campaign_worker", "owned", str(out / "spec.json")]
        environment = {**os.environ, "PYTHONPATH": str(source / "src"), "MLX2_CAMPAIGN_ROOT": str(source),
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        environment.pop("GIT_DIR", None)
        environment.pop("GIT_WORK_TREE", None)
        # Full CPU preflight precedes any CPG claim/GPU ownership window.
        preflight_key = (runner["python"], spec["source_sha256"])
        if 2 in spec["levels"] and not spec.get("preflight_receipt") and preflight_key in self.preflight_cache:
            spec["preflight_receipt"] = self.preflight_cache[preflight_key]
            atomic_json(out / "spec.json", spec)
        if 2 in spec["levels"] and not spec.get("preflight_receipt"):
            if not spec.get("approve_preflight"):
                raise ValueError("Full preflight requires explicit approval")
            preflight = out / "preflight.json"
            preflight_command = [runner["python"], str(source / "scripts/qualify_serving.py"),
                                 "--preflight-only", "--output", str(preflight)]
            self.control.store.update(identity, "running", "Running explicitly approved CPU preflight before GPU ownership")
            with (out / "preflight.log").open("w") as log:
                self.child = spawn(preflight_command, cwd=source, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic() + 14400
                    while self.child.poll() is None:
                        if self.cancel.wait(0.25):
                            raise RuntimeError("Preflight cancelled")
                        if time.monotonic() > deadline:
                            raise TimeoutError("Preflight deadline exceeded")
                    if self.child.returncode:
                        raise RuntimeError("Approved preflight failed; inspect preflight.log")
                finally:
                    self._stop()
            spec["preflight_receipt"] = str(preflight)
            self.preflight_cache[preflight_key] = str(preflight)
            atomic_json(out / "spec.json", spec)
        self.control.store.update(identity, "running", "Waiting for CPG and paired GPU ownership")
        with self.log_path.open("a") as log:
            self.child = spawn(command, cwd=source, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 28800
                while self.child.poll() is None:
                    if self.cancel.wait(0.25):
                        raise RuntimeError("Qualification cancelled; releasing model and ownership")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Qualification exceeded eight-hour bounded window")
                code = self.child.returncode
            finally:
                self._stop()
        result_path = out / "result.json"
        if not result_path.exists():
            raise RuntimeError("Campaign worker produced no result; inspect campaign logs")
        result = json.loads(result_path.read_text())
        for record in result.get("records", []):
            self.control.store.evidence(record)
        state = "completed" if code == 0 else "failed"
        if result.get("review_pending") and code == 0:
            state = "awaiting_review"
        self.control.store.update(identity, state, result.get("message", state), result)
        self.control.event("qualification_job_finished", job_id=identity, state=state)

    def review(self, request):
        identity = request.get("job_id")
        passed = request.get("coherent")
        if type(passed) is not bool:
            raise ValueError("Coherence review needs an explicit boolean verdict")
        job = next((j for j in self.control.store.jobs(10000) if j["id"] == identity), None)
        if job is None or not job.get("result") or not job["result"].get("review_pending"):
            raise ValueError("This job has no unreviewed ordinary output")
        result = job["result"]
        record = {"binding": result["binding"], "level": 1, "passed": passed,
                  "evidence_path": result["evidence_path"], "review": {"coherent": passed, "reviewed_at": time.time()},
                  "output": result["ordinary_output"], "qualification_claim": False}
        self.control.store.evidence(record)
        result["review_pending"] = False
        result["coherence_review"] = record["review"]
        self.control.store.update(identity, "completed" if passed and not result.get("failed_levels") else "failed",
                                  "Coherence accepted" if passed else "Output rejected as incoherent", result)
        return {"record": record}

    def _stop(self):
        child = self.child
        if child is None:
            return
        from .desktop_campaign_worker import stop
        stop(child, timeout=45)
        self.child = None

    def close(self):
        self.closed = True
        self.cancel.set()
        if self.thread:
            self.thread.join(timeout=55)
