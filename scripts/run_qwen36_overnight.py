#!/usr/bin/env python3
"""Resumable, fail-soft Qwen3.6 overnight qualification orchestrator.

This is a campaign control plane, not a qualification installer.  It preserves
every attempt and continues independent work after a failure.  Live GPU steps
remain gated on an externally acquired CPG lease receipt and a matching local
lock.  The script never promotes a receipt or changes a serving default.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import re
from typing import Any, Callable
from urllib.request import urlopen


SCHEMA = "mlx2.qwen36-overnight-campaign.v1"
EVENT_SCHEMA = "mlx2.qwen36-overnight-event.v1"
RECEIPT_SCHEMA = "mlx2.qwen36-overnight-step-receipt.v1"
TERMINAL = {"passed", "failed", "blocked", "skipped", "timed_out", "cancelled", "interrupted"}
PASS = {"passed"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


VOLATILE_INPUT_NAMES = {".DS_Store"}
VOLATILE_INPUT_PARTS = {"__pycache__", ".pytest_cache"}


def input_tree(root: Path, path: Path) -> dict[str, dict[str, Any]]:
    """Return a content identity for a guarded file or tree.

    Campaign outputs are deliberately not passed to this helper. Finder metadata
    and interpreter/test caches are also excluded so merely observing a campaign
    cannot change its identity.
    """
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists() and not path.is_symlink():
        return rows
    candidates = [path] if not path.is_dir() else sorted(path.rglob("*"))
    for candidate in candidates:
        relative_parts = candidate.relative_to(root).parts
        if (candidate.name in VOLATILE_INPUT_NAMES
                or any(part in VOLATILE_INPUT_PARTS for part in relative_parts)):
            continue
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            rows[relative] = {"kind": "symlink", "target": os.readlink(candidate)}
        elif candidate.is_file():
            rows[relative] = {
                "kind": "file",
                "mode": candidate.stat().st_mode & 0o111,
                "sha256": sha256_file(candidate),
            }
    return rows


def command_output(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable ({result.returncode})"


def campaign_identity(root: Path, artifacts: list[Path], settings: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for path in artifacts:
        resolved = path.expanduser().resolve()
        stat = resolved.stat() if resolved.exists() else None
        metadata = {}
        payload_files = {}
        if resolved.is_dir():
            for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                         "tokenizer_config.json", "generation_config.json"):
                candidate = resolved / name
                if candidate.exists():
                    metadata[name] = sha256_file(candidate)
            for pattern in ("*.safetensors", "*.npz", "*.gguf"):
                for candidate in sorted(resolved.glob(pattern)):
                    payload_stat = candidate.stat()
                    payload_files[candidate.name] = {
                        "size": payload_stat.st_size,
                        "mtime_ns": payload_stat.st_mtime_ns,
                    }
        rows.append({"path": str(resolved), "exists": stat is not None,
                     "device": stat.st_dev if stat else None, "inode": stat.st_ino if stat else None,
                     "mtime_ns": stat.st_mtime_ns if stat and not resolved.is_dir() else None,
                     "metadata_sha256": metadata,
                     "payload_files": payload_files,
                     "fingerprint": sha256_bytes(canonical({
                         "metadata": metadata, "payload_files": payload_files,
                     })) if metadata or payload_files else None})
    harness_names = ("run_qwen36_overnight.py", "qualify_serving.py",
                     "qwen36_http_qualification.py",
                     "run_qualification_matrix.py", "benchmark_serving.py",
                     "activate_qualification_arm.py", "qwen36_fused_gdn_probe.py",
                     "thermal_probe.swift")
    harness = {name: sha256_file(root / "scripts" / name) for name in harness_names
               if (root / "scripts" / name).exists()}
    guarded_inputs = {
        "runtime": input_tree(root, root / "src" / "mlx2"),
        "harness": {
            key: value
            for name in harness_names
            for key, value in input_tree(root, root / "scripts" / name).items()
        },
        "policies": {
            key: value
            for policy in sorted((root / "qualification" / "policies").glob("qwen36*.json"))
            for key, value in input_tree(root, policy).items()
        },
        "tests": input_tree(root, root / "tests"),
    }
    identity = {
        # These fields aid diagnosis, but are intentionally outside the guarded
        # digest. An unrelated commit or a new campaign receipt must not stop a
        # run whose relevant inputs are byte-for-byte unchanged.
        "git_head": command_output(["git", "rev-parse", "HEAD"], root),
        "git_status": command_output(["git", "status", "--short"], root),
        "runtime_source_sha256": source_hash(root),
        "harness_sha256": harness,
        "guarded_inputs": guarded_inputs,
        "artifacts": rows,
        "settings": settings,
    }
    identity["sha256"] = sha256_bytes(canonical({
        "guarded_inputs": guarded_inputs,
        "artifacts": rows,
        "settings": settings,
    }))
    return identity


def ensure_qwen36_policies(root: Path) -> None:
    policies = {
        "qualification/policies/qwen36-ordinary.json": {},
        "qualification/policies/qwen36-mtp-artifact-ordinary.json": {},
        "qualification/policies/qwen36-mtp2.json": {"num_draft": 2},
    }
    for relative, expected in policies.items():
        path = root / relative
        if path.exists():
            actual = json.loads(path.read_text())
            if actual != expected:
                raise ValueError(
                    f"refusing to overwrite mismatched reviewed policy {path}: "
                    f"expected {expected}, found {actual}"
                )
        else:
            atomic_json(path, expected)


def qwen36_manifest_template(root: Path, run_dir: Path, ordinary: Path,
                             mtp: Path, python: str) -> dict[str, Any]:
    py = str(root / python) if not Path(python).is_absolute() else python
    artifacts = run_dir / "artifacts"
    activation_state = run_dir / "activation/server.json"
    policies = {
        "ordinary": root / "qualification/policies/qwen36-ordinary.json",
        "mtp-artifact-ordinary": root / "qualification/policies/qwen36-mtp-artifact-ordinary.json",
        "mtp2": root / "qualification/policies/qwen36-mtp2.json",
    }
    models = {"ordinary": ordinary, "mtp-artifact-ordinary": mtp, "mtp2": mtp}
    profiles = {"ordinary": "qwen36-35b-a3b-apcv2-ordinary",
                "mtp-artifact-ordinary": "qwen36-35b-a3b-apcv2-ordinary",
                "mtp2": "qwen36-35b-a3b-apcv2-mtp2"}
    context_steps = {"ordinary": "ordinary-serving",
                     "mtp-artifact-ordinary": "mtp-artifact-ordinary-serving",
                     "mtp2": "mtp2-serving"}
    batch_steps = {"ordinary": "ordinary-b20-serving",
                   "mtp-artifact-ordinary": "mtp-artifact-ordinary-b20-serving",
                   "mtp2": "mtp2-b20-serving"}

    def activation(arm: str, suite: str, lanes: int, inflight: int) -> list[str]:
        argv = [py, str(root / "scripts/activate_qualification_arm.py"),
                "--state", str(activation_state),
                "--log", str(run_dir / "logs/matrix" / f"{arm}-{suite}.log"),
                # Absolute, not "src": a relative entry resolves against
                # --cwd, so a run queued from a worktree would import mlx2
                # from whichever checkout that happened to be.
                "--cwd", str(root), "--", "/usr/bin/env", f"PYTHONPATH={root / 'src'}", py, "-u",
                "-m", "mlx2.server", "--model", str(models[arm]), "--host", "127.0.0.1",
                "--port", "8296", "--max-context", "262144", "--max-lanes", str(lanes),
                "--max-inflight", str(inflight), "--cache-bytes", "12884901888",
                "--cache-dir", str(run_dir / "cache/matrix" / arm / suite),
                "--execution-policy", str(policies[arm]), "--qualification-mode"]
        if arm != "mtp2":
            argv.append("--ordinary")
        return argv

    core_checks = ["unit_tests", "cold_text", "warm_prefix", "batch",
                   "cache_leases", "runtime_stable", "long_context_delegation"]
    arms = []
    for arm in ("ordinary", "mtp-artifact-ordinary", "mtp2"):
        mtp_arm = arm == "mtp2"
        receipt_requirements = [
            {"path": "cache", "equals": "apcv2"},
            {"path": "profile", "equals": profiles[arm]},
        ]
        if mtp_arm:
            receipt_requirements += [
                {"path": "mtp.route", "equals": "segmented_self_mtp"},
                {"path": "mtp.num_draft", "equals": 2},
                {"path": "mtp.stats.draft_proposed", "gt": 0},
            ]
        else:
            receipt_requirements += [{"path": "mtp", "equals": None},
                                     {"path": "ordinary_compute_width", "gte": 1}]
        counters = [{"path": "apcv2.hits", "delta_gte": 1},
                    {"path": "apcv2.stores", "delta_gte": 1}]
        after = [{"path": "apcv2.cow.active_leases", "equals": 0}]
        checks = list(core_checks)
        if mtp_arm:
            counters += [{"path": "execution.segmented_mtp.engaged", "delta_gte": 1},
                         {"path": "execution.segmented_mtp.committed_cycles", "delta_gte": 1}]
            after += [{"path": "execution.segmented_mtp.failures", "equals": 0},
                      {"path": "execution.segmented_mtp.full_prefix_materializations", "equals": 0}]
            checks += ["feature_segmented_transaction", "feature_segmented_rollback"]
        base = {
            "name": arm, "url": "http://127.0.0.1:8296",
            "activation_key": f"qwen36-{arm}-context-max4",
            "activate_command": activation(arm, "context", 4, 8),
            "status_requirements": [
                {"path": "profile", "equals": profiles[arm]},
                {"path": "max_lanes", "equals": 4},
                {"path": "settings.speculation", "equals": "self_mtp" if mtp_arm else "ordinary"},
            ],
            "receipt_requirements": receipt_requirements,
            "counter_requirements": counters,
            "after_status_requirements": after,
            "qualification_receipt": {
                "path": str(artifacts / f"{context_steps[arm]}-qualification.json"),
                "required_checks": checks,
            },
        }
        base["suites"] = {"batch_stress": {
            "activation_key": f"qwen36-{arm}-batch-max20",
            "activate_command": activation(arm, "batch_stress", 20, 40),
            "status_requirements": [
                {"path": "profile", "equals": profiles[arm]},
                {"path": "max_lanes", "equals": 20},
                {"path": "settings.max_inflight", "equals": 40},
                {"path": "settings.speculation", "equals": "self_mtp" if mtp_arm else "ordinary"},
            ],
            "qualification_receipt": {
                "path": str(artifacts / f"{batch_steps[arm]}-qualification.json"),
                "required_checks": checks,
            },
        }}
        arms.append(base)
    return {
        "schema": "mlx2.qualification-matrix.v1",
        "thermal": {"required_thermal_state": 0, "max_battery_temperature_c": 40.0,
                    "max_virtual_temperature_c": 45.0, "max_temperature_delta_c": 0.5,
                    "consecutive_samples": 3, "sample_interval_seconds": 15,
                    "post_sample_interval_seconds": 15, "max_wait_seconds": 1800,
                    "max_swap_growth_bytes": 2 * 1024**3, "require_temperatures": True},
        "models": [{
            "name": "qwen36-35b-a3b", "tokenizer_renderer": "qwen_direct",
            "tokenizer_path": str(ordinary), "arm_order_claim": "alternating",
            "contexts": [{"tokens": value} for value in (32768, 65536, 131072, 262016)],
            "context": {"runs_per_cell": 3, "max_tokens": 64,
                        "prompt_token_tolerance": 0, "min_cached_tokens": 1,
                        "min_prime_to_warm_cache_gain_tokens": 1,
                        "timeout_seconds": 14400},
            "batch_stress": {"rounds": 20, "width": 20, "max_tokens": 64,
                             "timeout_seconds": 7200,
                             "prompts": [f"Explain compiler optimization {index:02d} in detail."
                                         for index in range(1, 21)]},
            "arms": arms,
        }],
    }


def ensure_qwen36_manifest(root: Path, run_dir: Path, output: Path, ordinary: Path,
                           mtp: Path, python: str) -> None:
    """Build exact tokenizer-frozen prompts, preserving an identical manifest."""
    template = run_dir / "inputs/qwen36-overnight-experiments.template.json"
    atomic_json(template, qwen36_manifest_template(root, run_dir, ordinary, mtp, python))
    candidate = run_dir / "inputs/qwen36-overnight-experiments.generated.json"
    prompt_dir = run_dir / "inputs/prompts"
    py = str(root / python) if not Path(python).is_absolute() else python
    completed = subprocess.run(
        [py, str(root / "scripts/build_context_prompts.py"), "--manifest", str(template),
         "--output-manifest", str(candidate), "--prompt-dir", str(prompt_dir)],
        cwd=root, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"context prompt generation failed: {completed.stderr.strip()}")
    if output.exists():
        if json.loads(output.read_text()) != json.loads(candidate.read_text()):
            raise ValueError(f"refusing to overwrite drifted frozen experiment manifest {output}")
    else:
        atomic_json(output, json.loads(candidate.read_text()))


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]
    timeout: float


@dataclass(frozen=True)
class Server:
    argv: tuple[str, ...]
    url: str
    ready_timeout: float = 1800
    drain_timeout: float = 180


@dataclass(frozen=True)
class Step:
    step_id: str
    phase: str
    label: str
    commands: tuple[Command, ...] = ()
    dependencies: tuple[str, ...] = ()
    required_paths: tuple[str, ...] = ()
    audit: str | None = None
    server: Server | None = None
    gpu: bool = False
    always_run: bool = False
    optional: bool = False

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value["commands"] = [asdict(item) for item in self.commands]
        return value


@dataclass
class Config:
    root: Path
    run_dir: Path
    lease_receipt: Path | None = None
    gpu_lock: Path = Path("/tmp/gpu.lock")
    hard_deadline: float | None = None
    heartbeat_seconds: float = 900
    terminate_grace: float = 20
    cleanup_reserve: float = 300
    dry_run: bool = False
    event_hook: Callable[[dict[str, Any]], None] | None = None


class Campaign:
    def __init__(self, config: Config, steps: list[Step], identity: dict[str, Any]):
        self.config, self.steps, self.identity = config, steps, identity
        self.step_map = {step.step_id: step for step in steps}
        if len(self.step_map) != len(steps):
            raise ValueError("step ids must be unique")
        self.state_path = config.run_dir / "state.json"
        self.events_path = config.run_dir / "events.jsonl"
        self.receipts_dir = config.run_dir / "receipts"
        self.logs_dir = config.run_dir / "logs"
        self.cancel = threading.Event()
        self.owned: set[subprocess.Popen[Any]] = set()
        self.state: dict[str, Any] = {}
        self._event_lock = threading.Lock()
        self._safety_lock = threading.Lock()
        self._thermal_failures = 0
        self._swap_baseline = self.sample_swap_bytes()
        self._server_status: dict[int, dict[str, Any]] = {}

    def event(self, event: str, **data: Any) -> None:
        row = {"schema": EVENT_SCHEMA, "timestamp": time.time(), "event": event, **data}
        with self._event_lock:
            append_jsonl(self.events_path, row)
        if self.config.event_hook:
            self.config.event_hook(row)

    def prepare(self, replace: bool = False) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_static_inputs()
        if self.state_path.exists() and not replace:
            self.load()
            return
        now = time.time()
        self.state = {
            "schema": SCHEMA, "created_at": now, "updated_at": now,
            "identity": self.identity,
            "plan_sha256": sha256_bytes(canonical([step.public() for step in self.steps])),
            "steps": {step.step_id: {"status": "planned", "attempts": []} for step in self.steps},
            "active_processes": {},
        }
        self.save()
        self.event("campaign_prepared", step_count=len(self.steps), identity_sha256=self.identity["sha256"])
        self.write_summary()

    def ensure_static_inputs(self) -> None:
        """Create the three adapter-valid policies, refusing silent drift."""
        ensure_qwen36_policies(self.config.root)

    def load(self, validate_identity: bool = True) -> None:
        self.state = json.loads(self.state_path.read_text())
        if self.state.get("schema") != SCHEMA:
            raise ValueError("campaign state schema mismatch")
        if validate_identity and self.state.get("identity", {}).get("sha256") != self.identity.get("sha256"):
            raise ValueError("campaign source/artifact/settings identity changed; create a new run directory")
        expected = sha256_bytes(canonical([step.public() for step in self.steps]))
        if self.state.get("plan_sha256") != expected:
            if validate_identity:
                raise ValueError("campaign plan changed; create a new run directory")
        self.state.setdefault("active_processes", {})

    def save(self) -> None:
        self.state["updated_at"] = time.time()
        atomic_json(self.state_path, self.state)

    def validate(self) -> list[dict[str, Any]]:
        results = []
        for step in self.steps:
            missing = [str((self.config.root / path).resolve()) for path in step.required_paths
                       if not (self.config.root / path).exists()]
            audit_error = self.audit(step.audit) if step.audit else None
            results.append({"step_id": step.step_id, "missing_paths": missing,
                            "audit_error": audit_error,
                            "ready": not missing and audit_error is None,
                            "next_command": self.next_command(step, missing, audit_error)})
        atomic_json(self.config.run_dir / "validation.json", {
            "schema": "mlx2.qwen36-overnight-validation.v1", "identity": self.identity,
            "validated_at": time.time(), "steps": results,
            "loads_models": False,
        })
        return results

    def next_command(self, step: Step, missing: list[str], audit_error: str | None) -> str | None:
        if any(path.endswith("qwen36-overnight-experiments.json") for path in missing):
            return ("Create qualification/qwen36-overnight-experiments.template.json, then run: "
                    ".venv/bin/python scripts/build_context_prompts.py "
                    "--manifest qualification/qwen36-overnight-experiments.template.json "
                    "--output-manifest qualification/qwen36-overnight-experiments.json "
                    "--prompt-dir qualification/prompts/qwen36-overnight")
        if audit_error:
            return f"Resolve prerequisite: {audit_error}"
        return None

    def audit(self, name: str | None) -> str | None:
        if name == "gpu_lease":
            try:
                self.validate_gpu_lease()
            except Exception as exc:
                return str(exc)
        elif name == "qualifier_preflight_receipt":
            text = (self.config.root / "scripts/qualify_serving.py").read_text()
            if "--preflight-only" not in text or "--preflight-receipt" not in text:
                return "qualify_serving.py still launches pytest after model load; add an identity-bound preflight receipt"
        elif name == "qualifier_http_coverage":
            companion = self.config.root / "scripts/qwen36_http_qualification.py"
            if not companion.exists():
                return "missing scripts/qwen36_http_qualification.py companion probe"
            tree = ast.parse(companion.read_text())
            coverage = None
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "COVERAGE_DEFINITION"
                    for target in node.targets
                ):
                    coverage = ast.literal_eval(node.value)
                    break
            required = ("response_format_json_object", "strict_json_schema", "explicit_grammar",
                        "physical_n2", "overload_429_retry_after",
                        "latency_ttft_itl_percentiles", "tenant_jain_fairness",
                        "progress_events", "full_length_completions")
            gates = (coverage or {}).get("gates", {})
            missing = [item for item in required
                       if gates.get(item, {}).get("required") is not True]
            if missing:
                return f"serving qualifier does not yet cover required HTTP rows: {missing}"
        elif name == "matrix_fail_soft":
            text = (self.config.root / "scripts/run_qualification_matrix.py").read_text()
            if "except BaseException as exc:" in text and "atomic_json(args.output, report)\n            raise" in text:
                return "matrix runner aborts on the first failed cell instead of preserving and continuing independent cells"
        elif name == "matrix_schedule":
            text = (self.config.root / "scripts/run_qualification_matrix.py").read_text()
            if "for arm_index, arm in enumerate(arms):" in text and "for context in traversal:" in text:
                return "context matrix is arm-grouped, not arm-alternating after each valid measurement"
        elif name == "activation_cleanup":
            text = (self.config.root / "scripts/activate_qualification_arm.py").read_text()
            if "start_new_session=True" not in text or "os.killpg" not in text:
                return "activation helper does not stop its owned server process group"
        elif name == "pld_route":
            sources = [self.config.root / "src/mlx2/server.py",
                       self.config.root / "src/mlx2/generation.py"]
            text = "\n".join(path.read_text() for path in sources if path.exists())
            required = ("prompt_lookup", "proposed_tokens", "accepted_tokens")
            missing = [token for token in required if token not in text]
            if missing:
                return f"PLD target-forward serving route/receipts are absent: {missing}"
        elif name == "json_inputs":
            qualification = self.config.root / "qualification"
            candidates = list(qualification.glob("*.json"))
            candidates += list((qualification / "policies").glob("qwen36-*.json"))
            qwen_runs = qualification / "runs/qwen36-35b-a3b"
            if qwen_runs.exists():
                candidates += list(qwen_runs.rglob("*.json"))
            for path in sorted(set(candidates)):
                try:
                    json.loads(path.read_text())
                except Exception as exc:
                    # Some preserved HTTP transcripts use one JSON object per
                    # line despite their historical .json suffix.
                    try:
                        rows = [line for line in path.read_text().splitlines() if line.strip()]
                        if not rows:
                            raise ValueError("empty JSON/JSONL file")
                        for line in rows:
                            json.loads(line)
                    except Exception:
                        return f"invalid JSON {path}: {exc}"
        elif name == "ports_free":
            busy = []
            for port in (8296, 8297, 8298):
                with socket.socket() as sock:
                    sock.settimeout(0.1)
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        busy.append(port)
            if busy:
                return f"candidate ports already have listeners: {busy}"
        elif name == "closeout":
            if self.owned:
                return f"owned processes remain: {[process.pid for process in self.owned]}"
            return self.audit("ports_free")
        elif name == "report":
            artifacts = []
            artifact_dir = self.config.run_dir / "artifacts"
            if artifact_dir.exists():
                for path in sorted(artifact_dir.glob("*.json")):
                    try:
                        payload = json.loads(path.read_text())
                        artifacts.append({"path": str(path), "sha256": sha256_file(path),
                                          "schema": payload.get("schema"),
                                          "passed": payload.get("passed"),
                                          "summary": payload.get("summary")})
                    except Exception as exc:
                        artifacts.append({"path": str(path), "error": str(exc)})
            atomic_json(self.config.run_dir / "matched-report.json", {
                "schema": "mlx2.qwen36-overnight-matched-report.v1",
                "generated_at": time.time(), "identity": self.identity,
                "steps": {key: value.get("status") for key, value in self.state.get("steps", {}).items()},
                "artifacts": artifacts,
                "note": "Only passed, identity-bound source receipts are eligible for performance claims.",
            })
            return None
        elif name:
            return f"unknown audit {name}"
        return None

    @staticmethod
    def _read_lock(path: Path) -> dict[str, Any]:
        if path.is_dir():
            for name in ("lease.json", "owner.json", "metadata.json"):
                candidate = path / name
                if candidate.exists():
                    return json.loads(candidate.read_text())
            raise ValueError(f"GPU lock directory {path} lacks lease.json, owner.json or metadata.json")
        return json.loads(path.read_text())

    def validate_gpu_lease(self) -> None:
        if not self.config.lease_receipt:
            raise ValueError("GPU execution requires --gpu-lease-receipt from an externally acquired CPG lease")
        receipt = json.loads(self.config.lease_receipt.read_text())
        lock = self._read_lock(self.config.gpu_lock)
        required = ("campaign", "lease_id", "generation", "owner", "purpose")
        absent = [key for key in required if receipt.get(key) in (None, "")]
        if absent:
            raise ValueError(f"GPU lease receipt is missing {absent}")
        mismatched = [key for key in required if lock.get(key) != receipt.get(key)]
        if mismatched:
            raise ValueError(f"GPU lock does not match lease receipt for {mismatched}")

    def run(self, resume: bool) -> None:
        if self.state_path.exists():
            self.load()
        else:
            self.prepare()
        reconcile_refused = self.reconcile_active_processes("resume" if resume else "run")
        if resume:
            self.state.pop("stop_requested", None)
            self.cancel.clear()
        elif self.external_stop_requested():
            raise RuntimeError("campaign has a pending stop request; use resume to continue")
        self.register_controller()
        previous_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal)
        self.event("campaign_started", resume=resume, dry_run=self.config.dry_run)
        try:
            for step in self.steps:
                if self.external_stop_requested():
                    self.cancel.set()
                current = self.state["steps"][step.step_id]["status"]
                if step.step_id in reconcile_refused:
                    self.event("step_blocked_unproven_process", step_id=step.step_id)
                    continue
                if resume and current == "passed":
                    if self.passed_receipt_valid(step):
                        self.event("step_resumed_as_complete", step_id=step.step_id)
                        continue
                    self.event("step_receipt_invalid", step_id=step.step_id)
                if self.cancel.is_set() and not step.always_run:
                    self.finish_without_run(step, "cancelled", "campaign cancellation requested")
                    continue
                if self.deadline_expired(include_cleanup_reserve=True) and not step.always_run:
                    self.finish_without_run(step, "skipped", "hard campaign deadline or cleanup reserve reached")
                    continue
                failed_deps = [dep for dep in step.dependencies
                               if self.state["steps"][dep]["status"] not in PASS]
                if failed_deps and not step.always_run:
                    self.finish_without_run(step, "blocked", f"dependencies did not pass: {failed_deps}")
                    continue
                missing = [path for path in step.required_paths if not (self.config.root / path).exists()]
                if missing:
                    self.finish_without_run(step, "blocked", f"missing prerequisites: {missing}")
                    continue
                audit_error = self.audit(step.audit) if step.audit else None
                if audit_error:
                    self.finish_without_run(step, "blocked", audit_error)
                    continue
                if self.config.dry_run:
                    self.finish_without_run(step, "skipped", "dry-run: command intentionally not executed")
                    continue
                self.run_step(step)
        finally:
            self.stop_all()
            self.unregister_controller()
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            self.state["completed_at"] = time.time()
            self.save()
            self.write_summary()
            self.event("campaign_finished", counts=self.counts())

    def register_controller(self) -> None:
        identity = self.process_identity(os.getpid())
        if not identity or "pid" not in identity:
            raise RuntimeError(f"could not record campaign controller identity: {identity}")
        self.state["controller"] = {
            "pid": os.getpid(), "pgid": identity["pgid"],
            "process_start_identity": identity["start_identity"],
            "process_command": identity["command"], "registered_at": time.time(),
        }
        self.save()

    def unregister_controller(self) -> None:
        controller = self.state.get("controller")
        if controller and controller.get("pid") == os.getpid():
            self.state.pop("controller", None)
            self.save()

    def external_stop_requested(self) -> bool:
        if not self.state_path.exists():
            return False
        try:
            return bool(json.loads(self.state_path.read_text()).get("stop_requested"))
        except Exception:
            return False

    @staticmethod
    def process_identity(pid: int) -> dict[str, Any] | None:
        try:
            result = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "pid=", "-o", "pgid=",
                 "-o", "state=", "-o", "lstart=", "-o", "command="],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode or not result.stdout.strip():
                return None
            line = result.stdout.strip()
            # pid, pgid, state, and fixed-width lstart precede command.
            match = re.match(r"\s*(\d+)\s+(\d+)\s+(\S+)\s+(.{24})\s+(.*)$", line)
            if not match:
                return {"raw": line}
            if match.group(3).startswith("Z"):
                return None
            return {"pid": int(match.group(1)), "pgid": int(match.group(2)),
                    "state": match.group(3), "start_identity": match.group(4).strip(),
                    "command": match.group(5)}
        except Exception as exc:
            return {"error": str(exc)}

    def register_process(self, step: Step, attempt: int, process: subprocess.Popen[Any],
                         argv: tuple[str, ...] | list[str], role: str,
                         server_url: str | None = None) -> None:
        identity = self.process_identity(process.pid)
        if (not identity or "pid" not in identity) and process.poll() is not None:
            # A short command can exit (a zombie ps reports as gone) before
            # its identity is read, most often on a loaded host.  There is
            # nothing left to own or reconcile; its exit code decides the
            # step, not a registration failure.
            self.event("process_exited_before_registration", step_id=step.step_id,
                       attempt=attempt, role=role, pid=process.pid,
                       returncode=process.returncode)
            return
        if not identity or "pid" not in identity:
            self.stop_process(process, self.config.terminate_grace, unregister=False)
            raise RuntimeError(f"could not capture process start identity for PID {process.pid}: {identity}")
        key = f"{step.step_id}:{attempt}:{role}:{process.pid}"
        self.state["active_processes"][key] = {
            "key": key, "step_id": step.step_id, "attempt": attempt, "role": role,
            "pid": process.pid, "pgid": identity["pgid"], "argv": list(argv),
            "process_start_identity": identity["start_identity"],
            "process_command": identity["command"], "server_url": server_url,
            "registered_at": time.time(),
        }
        self.save()
        self.event("process_registered", **self.state["active_processes"][key])

    def unregister_process(self, pid: int, outcome: str = "exited") -> None:
        keys = [key for key, row in self.state.get("active_processes", {}).items()
                if int(row.get("pid", -1)) == pid]
        for key in keys:
            row = self.state["active_processes"].pop(key)
            self.event("process_unregistered", process_key=key, pid=pid, outcome=outcome,
                       step_id=row.get("step_id"))
        if keys:
            self.save()

    def process_record_matches(self, record: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        current = self.process_identity(int(record["pid"]))
        if current is None:
            return True, None
        passed = (current.get("pid") == record.get("pid")
                  and current.get("pgid") == record.get("pgid")
                  and current.get("start_identity") == record.get("process_start_identity")
                  and current.get("command") == record.get("process_command"))
        return passed, current

    def stop_record(self, record: dict[str, Any], reason: str) -> dict[str, Any]:
        matches, current = self.process_record_matches(record)
        if current is None:
            self.unregister_process(int(record["pid"]), "already_exited")
            return {"stopped": True, "already_exited": True, "record": record}
        if not matches:
            return {"stopped": False, "ownership_refused": True, "record": record,
                    "current": current, "reason": "PID/PGID/start/command identity mismatch"}
        if record.get("server_url"):
            self.wait_recovered_server_drain(record["server_url"], self.config.terminate_grace)
        try:
            os.killpg(int(record["pgid"]), signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + self.config.terminate_grace
        while self.process_identity(int(record["pid"])) is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        forced = False
        matches_after, current_after = self.process_record_matches(record)
        if current_after is not None and matches_after:
            forced = True
            try: os.killpg(int(record["pgid"]), signal.SIGKILL)
            except ProcessLookupError: pass
            kill_deadline = time.monotonic() + max(1.0, self.config.terminate_grace)
            while time.monotonic() < kill_deadline:
                matches_after, current_after = self.process_record_matches(record)
                if current_after is None or not matches_after:
                    break
                time.sleep(0.05)
        matches_final, current_final = self.process_record_matches(record)
        exited = current_final is None or not matches_final
        listener_gone = True
        if exited and record.get("server_url"):
            host, port = record["server_url"].split("://", 1)[-1].split(":", 1)
            listener_gone = False
            for _ in range(40):
                with socket.socket() as sock:
                    sock.settimeout(0.1)
                    if sock.connect_ex((host, int(port))) != 0:
                        listener_gone = True
                        break
                time.sleep(0.05)
        if exited:
            self.unregister_process(int(record["pid"]), reason)
        return {"stopped": bool(exited and listener_gone), "forced_kill": forced,
                "listener_gone": listener_gone, "record": record,
                "current_after": current_final}

    @staticmethod
    def wait_recovered_server_drain(url: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urlopen(url.rstrip("/") + "/v1/status", timeout=2) as response:
                    status = json.load(response)
                cow = ((status.get("apcv2") or {}).get("cow") or {})
                if (not status.get("inflight") and not status.get("queue_depth", 0)
                        and not status.get("memory_waiting", 0)
                        and cow.get("active_leases") == 0):
                    return
            except Exception:
                return
            time.sleep(0.1)

    def reconcile_active_processes(self, reason: str) -> set[str]:
        for step_id in ("context-ladder", "batch-20x20"):
            row = self.state.get("steps", {}).get(step_id, {})
            if row.get("status") == "running" and row.get("attempts"):
                step = self.step_map.get(step_id)
                if step:
                    self.adopt_activation_server(step, int(row["attempts"][-1]["attempt"]))
        refused: set[str] = set()
        affected: set[str] = set()
        for record in list(self.state.get("active_processes", {}).values()):
            step_id = record.get("step_id")
            affected.add(step_id)
            result = self.stop_record(record, reason)
            self.event("process_reconciled", step_id=step_id, result=result)
            if result.get("ownership_refused") or not result.get("stopped"):
                refused.add(step_id)
        for step_id in affected:
            row = self.state["steps"].get(step_id)
            if not row or row.get("status") != "running":
                continue
            attempt = row.get("attempts", [])[-1] if row.get("attempts") else None
            if step_id in refused:
                reason_text = "refused to stop process because exact ownership could not be proven"
                row.update({"status": "blocked", "reason": reason_text, "finished_at": time.time()})
                if attempt: attempt.update({"status": "blocked", "reason": reason_text,
                                            "finished_at": time.time()})
            else:
                reason_text = f"attempt interrupted during {reason}; safe to resume in a new attempt"
                row.update({"status": "interrupted", "reason": reason_text, "finished_at": time.time()})
                if attempt: attempt.update({"status": "interrupted", "reason": reason_text,
                                            "finished_at": time.time()})
        for step_id, row in self.state.get("steps", {}).items():
            if row.get("status") == "running" and step_id not in affected:
                reason_text = f"controller stopped before process ownership was registered during {reason}"
                row.update({"status": "interrupted", "reason": reason_text,
                            "finished_at": time.time()})
                if row.get("attempts"):
                    row["attempts"][-1].update({"status": "interrupted", "reason": reason_text,
                                                "finished_at": time.time()})
        self.save()
        return refused

    def explicit_stop(self) -> dict[str, Any]:
        self.load(validate_identity=False)
        for step_id in ("context-ladder", "batch-20x20"):
            row = self.state.get("steps", {}).get(step_id, {})
            if row.get("status") == "running" and row.get("attempts") and self.step_map.get(step_id):
                self.adopt_activation_server(self.step_map[step_id], int(row["attempts"][-1]["attempt"]))
        recorded_identity = self.state.get("identity", {}).get("sha256")
        identity_drift = recorded_identity != self.identity.get("sha256")
        # A step is persisted as running before its child process ownership record
        # can be registered.  Include those rows in the stop snapshot so a
        # controller racing through cancellation cannot leave the attempt as
        # cancelled/failed instead of resumably interrupted.
        active_step_ids = {
            row.get("step_id") for row in self.state.get("active_processes", {}).values()
        } | {
            step_id for step_id, row in self.state.get("steps", {}).items()
            if row.get("status") == "running"
        }
        self.state["stop_requested"] = {"requested_at": time.time(),
                                        "requester_pid": os.getpid()}
        controller = self.state.get("controller")
        self.save()
        controller_result: dict[str, Any] = {"present": False}
        if controller and int(controller.get("pid", -1)) != os.getpid():
            current = self.process_identity(int(controller["pid"]))
            matches = bool(current and current.get("pid") == controller.get("pid")
                           and current.get("pgid") == controller.get("pgid")
                           and current.get("start_identity") == controller.get("process_start_identity")
                           and current.get("command") == controller.get("process_command"))
            controller_result = {"present": current is not None, "ownership_matches": matches,
                                 "record": controller, "current": current}
            if matches:
                os.kill(int(controller["pid"]), signal.SIGTERM)
                deadline = time.monotonic() + max(5.0, self.config.terminate_grace)
                while self.process_identity(int(controller["pid"])) is not None and time.monotonic() < deadline:
                    time.sleep(0.05)
                remaining = self.process_identity(int(controller["pid"]))
                if remaining is not None:
                    still_matches = (remaining.get("pgid") == controller.get("pgid")
                                     and remaining.get("start_identity") == controller.get("process_start_identity")
                                     and remaining.get("command") == controller.get("process_command"))
                    if still_matches:
                        os.kill(int(controller["pid"]), signal.SIGKILL)
                        controller_result["forced_kill"] = True
                        kill_deadline = time.monotonic() + 5
                        while self.process_identity(int(controller["pid"])) is not None and time.monotonic() < kill_deadline:
                            time.sleep(0.05)
                    else:
                        controller_result["ownership_refused_after_term"] = True
            elif current is not None:
                controller_result["ownership_refused"] = True
        # The controller may have persisted cleanup after receiving SIGTERM.
        self.load(validate_identity=False)
        refused = self.reconcile_active_processes("explicit_stop")
        for step_id in active_step_ids - refused:
            row = self.state.get("steps", {}).get(step_id)
            if not row or row.get("status") == "passed":
                continue
            reason_text = "attempt interrupted by explicit stop; safe to resume in a new attempt"
            row.update({"status": "interrupted", "reason": reason_text,
                        "finished_at": time.time()})
            if row.get("attempts"):
                row["attempts"][-1].update({"status": "interrupted", "reason": reason_text,
                                            "finished_at": time.time()})
        self.state["stopped_at"] = time.time()
        self.state["stop_status"] = "blocked" if refused else "stopped"
        self.save(); self.write_summary()
        self.event("campaign_stopped", refused_steps=sorted(refused))
        return {"stopped": not refused and not controller_result.get("ownership_refused"),
                "identity_drift": identity_drift,
                "recorded_identity_sha256": recorded_identity,
                "current_identity_sha256": self.identity.get("sha256"),
                "controller": controller_result, "refused_steps": sorted(refused),
                "active_processes": self.state.get("active_processes", {})}

    def status(self) -> dict[str, Any]:
        self.load(validate_identity=False)
        active = []
        for record in self.state.get("active_processes", {}).values():
            matches, current = self.process_record_matches(record)
            active.append({"record": record, "ownership_matches": matches,
                           "current": current})
        return {"schema": self.state.get("schema"), "counts": self.counts(),
                "identity_drift": self.state.get("identity", {}).get("sha256") != self.identity.get("sha256"),
                "active_processes": active, "updated_at": self.state.get("updated_at")}

    def deadline_expired(self, include_cleanup_reserve: bool = False) -> bool:
        reserve = self.config.cleanup_reserve if include_cleanup_reserve else 0
        return self.config.hard_deadline is not None and time.time() + reserve >= self.config.hard_deadline

    def _signal(self, signum: int, _frame: Any) -> None:
        self.event("signal_received", signal=signum)
        self.cancel.set()
        self.state["stop_requested"] = {"requested_at": time.time(),
                                        "reason": f"signal {signum}",
                                        "requester_pid": os.getpid()}
        self.save()
        self.stop_all()

    def finish_without_run(self, step: Step, status: str, reason: str) -> None:
        row = self.state["steps"][step.step_id]
        row.update({"status": status, "reason": reason, "finished_at": time.time()})
        self.save()
        self.event("step_finished", step_id=step.step_id, status=status, reason=reason)
        self.write_step_receipt(step, status, reason, [])

    def passed_receipt_valid(self, step: Step) -> bool:
        row = self.state["steps"][step.step_id]
        attempts = row.get("attempts", [])
        if not attempts:
            return False
        receipt_path = attempts[-1].get("receipt")
        receipt_hash = attempts[-1].get("receipt_sha256")
        if not receipt_path or not receipt_hash:
            return False
        path = Path(receipt_path)
        if not path.exists() or sha256_file(path) != receipt_hash:
            return False
        receipt = json.loads(path.read_text())
        return (receipt.get("status") == "passed"
                and receipt.get("identity", {}).get("sha256") == self.identity["sha256"]
                and sha256_bytes(canonical(receipt.get("step"))) == sha256_bytes(canonical(step.public())))

    def run_step(self, step: Step) -> None:
        row = self.state["steps"][step.step_id]
        attempt = len(row["attempts"]) + 1
        attempt_dir = self.logs_dir / step.step_id / f"attempt-{attempt:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        started = time.time()
        attempt_row = {"attempt": attempt, "started_at": started, "log_dir": str(attempt_dir)}
        row.update({"status": "running", "started_at": started})
        row["attempts"].append(attempt_row)
        self.save()
        self.event("step_started", step_id=step.step_id, attempt=attempt, phase=step.phase)
        server_process = None
        results: list[dict[str, Any]] = []
        status, reason = "passed", ""
        try:
            if step.gpu:
                self.validate_gpu_lease()
            if step.server:
                server_process = self.start_server(step, attempt, attempt_dir)
            for command in step.commands:
                command = self.effective_command(step, command)
                result = self.run_command(step, attempt, command, attempt_dir, server_process)
                results.append(result)
                if result["status"] != "passed":
                    status = result["status"]
                    reason = f"command {command.name} {status}"
                    break
        except Exception as exc:
            status, reason = "failed", f"{type(exc).__name__}: {exc}"
            self.event("step_exception", step_id=step.step_id, error=reason)
        finally:
            cleanup = (self.stop_server(server_process, step.server)
                       if server_process is not None and step.server else
                       self.stop_process(server_process, self.config.terminate_grace))
            if server_process is not None and not cleanup["stopped"] and status == "passed":
                status, reason = "failed", "owned server could not be stopped"
            if server_process is not None:
                results.append({"name": "server_cleanup", **cleanup})
            external_cleanup = self.stop_external_for_step(step.step_id)
            if external_cleanup:
                results.append({"name": "matrix_server_cleanup", "results": external_cleanup,
                                "stopped": all(row.get("stopped") for row in external_cleanup)})
                if not all(row.get("stopped") for row in external_cleanup) and status == "passed":
                    status, reason = "failed", "matrix activation server cleanup failed"
        finished = time.time()
        attempt_row.update({"finished_at": finished, "status": status, "reason": reason})
        row.update({"status": status, "reason": reason, "finished_at": finished})
        self.save()
        receipt_path = self.write_step_receipt(step, status, reason, results, attempt)
        attempt_row.update({"receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path)})
        self.save()
        self.event("step_finished", step_id=step.step_id, attempt=attempt, status=status, reason=reason)

    def effective_command(self, step: Step, command: Command) -> Command:
        if step.step_id not in {"context-ladder", "batch-20x20"}:
            return command
        suite = "context" if step.step_id == "context-ladder" else "batch_stress"
        status_step = {
            "ordinary": "ordinary-serving" if suite == "context" else "ordinary-b20-serving",
            "mtp-artifact-ordinary": ("mtp-artifact-ordinary-serving" if suite == "context"
                                      else "mtp-artifact-ordinary-b20-serving"),
            "mtp2": "mtp2-serving" if suite == "context" else "mtp2-b20-serving",
            "pld": "pld-serving" if suite == "context" else "pld-b20-serving",
        }
        argv = list(command.argv)
        manifest_index = argv.index("--manifest") + 1
        source = Path(argv[manifest_index])
        payload = json.loads(source.read_text())
        selected_count = 0
        for model in payload["models"]:
            selected = [arm for arm in model["arms"]
                        if self.state["steps"].get(status_step.get(arm["name"], ""), {}).get("status") == "passed"]
            model["arms"] = selected
            selected_count += len(selected)
            if len(selected) == 1:
                model["arm_order_claim"] = "none"
        payload["models"] = [model for model in payload["models"] if model["arms"]]
        if not selected_count:
            raise RuntimeError(f"no {suite} arm has a passing serving and HTTP qualification step")
        eligible = self.config.run_dir / "inputs" / f"qwen36-{suite}-eligible.json"
        atomic_json(eligible, payload)
        argv[manifest_index] = str(eligible)
        self.event("matrix_eligible_arms", suite=suite,
                   arms=[arm["name"] for model in payload["models"] for arm in model["arms"]],
                   manifest=str(eligible), manifest_sha256=sha256_file(eligible))
        return Command(command.name, tuple(argv), command.timeout)

    def start_server(self, step: Step, attempt: int, attempt_dir: Path) -> subprocess.Popen[Any]:
        assert step.server
        host, port = step.server.url.split("://", 1)[-1].split(":", 1)
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex((host, int(port))) == 0:
                raise RuntimeError(f"refusing launch: candidate port {host}:{port} is already in use")
        stdout = (attempt_dir / "server.stdout.log").open("wb")
        stderr = (attempt_dir / "server.stderr.log").open("wb")
        process = subprocess.Popen(step.server.argv, cwd=self.config.root, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        stdout.close(); stderr.close()
        self.owned.add(process)
        self.register_process(step, attempt, process, step.server.argv, "server", step.server.url)
        self.event("server_started", step_id=step.step_id, pid=process.pid, url=step.server.url)
        try:
            deadline = time.monotonic() + step.server.ready_timeout
            last_error = "not ready"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode} before readiness")
                try:
                    with urlopen(step.server.url.rstrip("/") + "/v1/status", timeout=5) as response:
                        status = json.load(response)
                    if status.get("healthy") and not status.get("inflight"):
                        for key in ("runtime", "artifact", "settings", "profile"):
                            if key not in status:
                                raise RuntimeError(f"server readiness lacks identity field {key}")
                        self._server_status[process.pid] = status
                        return process
                    last_error = f"unhealthy status: {status}"
                except Exception as exc:
                    last_error = str(exc)
                time.sleep(1)
            raise TimeoutError(f"server readiness timed out: {last_error}")
        except BaseException:
            self.stop_process(process, self.config.terminate_grace)
            raise

    def run_command(self, step: Step, attempt: int, command: Command, attempt_dir: Path,
                    server: subprocess.Popen[Any] | None) -> dict[str, Any]:
        stdout_path, stderr_path = attempt_dir / f"{command.name}.stdout.log", attempt_dir / f"{command.name}.stderr.log"
        started = time.time()
        timeout = min(command.timeout, max(0.0, self.config.hard_deadline - time.time())) \
            if self.config.hard_deadline else command.timeout
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command.argv, cwd=self.config.root, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            self.owned.add(process)
            self.register_process(step, attempt, process, command.argv, f"command:{command.name}")
            monitor = threading.Thread(target=self.monitor, args=(step, process, server), daemon=True)
            monitor.start()
            status = "passed"
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                self.adopt_activation_server(step, attempt)
                if self.external_stop_requested():
                    self.cancel.set()
                if self.cancel.is_set():
                    status = "cancelled"; self.stop_process(process, self.config.terminate_grace); break
                if time.monotonic() >= deadline:
                    status = "timed_out"; self.stop_process(process, self.config.terminate_grace); break
                time.sleep(min(0.2, max(0.01, deadline - time.monotonic())))
            monitor.join(timeout=1)
            self.owned.discard(process)
            self.unregister_process(process.pid, status)
            if status == "passed" and process.returncode:
                status = "failed"
        return {"name": command.name, "argv": list(command.argv), "status": status,
                "returncode": process.returncode, "started_at": started, "finished_at": time.time(),
                "stdout": str(stdout_path), "stdout_sha256": sha256_file(stdout_path),
                "stderr": str(stderr_path), "stderr_sha256": sha256_file(stderr_path)}

    def monitor(self, step: Step, process: subprocess.Popen[Any], server: subprocess.Popen[Any] | None) -> None:
        # Emit immediately, then every configured interval.  The immediate row
        # makes short steps observable and lets tests validate the same path.
        while process.poll() is None:
            snapshot: dict[str, Any] = {
                "step_id": step.step_id, "command_pid": process.pid,
                "server_pid": server.pid if server else None,
                "disk_free_bytes": shutil.disk_usage(self.config.run_dir).free,
            }
            try:
                snapshot["process"] = subprocess.run(
                    ["/bin/ps", "-p", str(process.pid), "-o", "pid=,rss=,etime=,command="],
                    text=True, capture_output=True, timeout=5).stdout.strip()
            except Exception as exc:
                snapshot["process_error"] = str(exc)
            if step.server:
                try:
                    with urlopen(step.server.url.rstrip("/") + "/v1/status", timeout=5) as response:
                        snapshot["server_status"] = json.load(response)
                except Exception as exc:
                    snapshot["server_error"] = str(exc)
            swap = self.sample_swap_bytes()
            snapshot["swap_bytes"] = swap
            snapshot["swap_growth_bytes"] = (swap - self._swap_baseline
                                               if swap is not None and self._swap_baseline is not None else None)
            thermal = self.sample_thermal()
            snapshot["thermal"] = thermal
            self.event("health_snapshot", **snapshot)
            with self._safety_lock:
                if snapshot["swap_growth_bytes"] is not None and snapshot["swap_growth_bytes"] > 2 << 30:
                    self.event("global_stop_latched", reason="swap growth exceeded 2 GiB", step_id=step.step_id)
                    self.cancel.set()
                if step.phase != "5" and thermal.get("thermal_state") is not None:
                    self._thermal_failures = self._thermal_failures + 1 if int(thermal["thermal_state"]) > 0 else 0
                    if self._thermal_failures >= 2:
                        self.event("global_stop_latched", reason="two consecutive non-nominal thermal samples", step_id=step.step_id)
                        self.cancel.set()
            try:
                process.wait(timeout=self.config.heartbeat_seconds)
                return
            except subprocess.TimeoutExpired:
                pass

    @staticmethod
    def sample_swap_bytes() -> int | None:
        try:
            result = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], capture_output=True,
                                    text=True, timeout=5)
            match = re.search(r"used\s*=\s*([0-9.]+)([MG])", result.stdout)
            if not match:
                return None
            scale = 1 << (30 if match.group(2) == "G" else 20)
            return int(float(match.group(1)) * scale)
        except Exception:
            return None

    def sample_thermal(self) -> dict[str, Any]:
        source = self.config.root / "scripts/thermal_probe.swift"
        if not source.exists() or not Path("/usr/bin/swift").exists():
            return {"available": False}
        try:
            result = subprocess.run(["/usr/bin/swift", str(source)], capture_output=True,
                                    text=True, timeout=30)
            value = json.loads(result.stdout)
            value["available"] = True
            return value
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def stop_server(self, process: subprocess.Popen[Any], server: Server) -> dict[str, Any]:
        status_before = None
        drain_error = None
        drained = False
        deadline = time.monotonic() + server.drain_timeout
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(server.url.rstrip("/") + "/v1/status", timeout=5) as response:
                    status_before = json.load(response)
                baseline = self._server_status.get(process.pid)
                if baseline and any(status_before.get(key) != baseline.get(key)
                                    for key in ("runtime", "artifact", "settings", "profile")):
                    drain_error = "server identity changed before cleanup"
                    break
                apcv2 = status_before.get("apcv2") or {}
                cow = apcv2.get("cow") or {}
                active_leases = cow.get("active_leases")
                memory_waiting = status_before.get("memory_waiting", 0)
                if (status_before.get("healthy") and not status_before.get("inflight")
                        and not status_before.get("queue_depth", 0)
                        and active_leases == 0 and not memory_waiting):
                    drained = True
                    break
            except Exception as exc:
                drain_error = str(exc)
            time.sleep(0.25)
        result = self.stop_process(process, self.config.terminate_grace)
        host, port = server.url.split("://", 1)[-1].split(":", 1)
        listener_gone = False
        for _ in range(40):
            with socket.socket() as sock:
                sock.settimeout(0.1)
                if sock.connect_ex((host, int(port))) != 0:
                    listener_gone = True
                    break
            time.sleep(0.05)
        self._server_status.pop(process.pid, None)
        return {**result, "drain_status": status_before, "drain_error": drain_error,
                "drained": drained, "listener_gone": listener_gone,
                "stopped": bool(result["stopped"] and listener_gone and drained)}

    def stop_process(self, process: subprocess.Popen[Any] | None, grace: float,
                     unregister: bool = True) -> dict[str, Any]:
        if process is None:
            return {"stopped": True, "pid": None}
        self.owned.discard(process)
        if process.poll() is not None:
            if unregister:
                self.unregister_process(process.pid)
            return {"stopped": True, "pid": process.pid, "returncode": process.returncode}
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if unregister:
                self.unregister_process(process.pid)
            return {"stopped": True, "pid": process.pid}
        deadline = time.monotonic() + grace
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        killed = False
        if process.poll() is None:
            killed = True
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
        if unregister:
            self.unregister_process(process.pid, "stopped")
        return {"stopped": process.poll() is not None, "pid": process.pid,
                "returncode": process.returncode, "forced_kill": killed}

    def stop_all(self) -> None:
        for process in list(self.owned):
            self.stop_process(process, self.config.terminate_grace)
        for record in list(self.state.get("active_processes", {}).values()):
            if record.get("role") == "activation-server":
                self.stop_record(record, "controller_cleanup")

    def adopt_activation_server(self, step: Step, attempt: int) -> None:
        if step.step_id not in {"context-ladder", "batch-20x20"}:
            return
        state_path = self.config.run_dir / "activation/server.json"
        if not state_path.exists():
            return
        try:
            state = json.loads(state_path.read_text())
            pid = int(state["pid"])
            if any(int(row.get("pid", -1)) == pid
                   for row in self.state.get("active_processes", {}).values()):
                return
            current = self.process_identity(pid)
            allowed_commands = {state.get("command")}
            if state.get("exec_command"):
                allowed_commands.add(state["exec_command"])
            elif isinstance(state.get("argv"), list):
                from activate_qualification_arm import exec_transition_command
                transitioned = exec_transition_command(
                    state["argv"], cwd=Path(state["cwd"]) if state.get("cwd") else None
                )
                if transitioned:
                    allowed_commands.add(transitioned)
            if (not current or current.get("command") not in allowed_commands
                    or current.get("pgid") != int(state.get("pgid", pid))
                    or os.getsid(pid) != int(state.get("sid", pid))):
                self.event("activation_server_adoption_refused", step_id=step.step_id,
                           activation_state=state, current=current)
                self.cancel.set()
                return
            argv = list(state.get("argv", []))
            port = argv[argv.index("--port") + 1] if "--port" in argv else "8296"
            key = f"{step.step_id}:{attempt}:activation-server:{pid}"
            self.state["active_processes"][key] = {
                "key": key, "step_id": step.step_id, "attempt": attempt,
                "role": "activation-server", "pid": pid, "pgid": current["pgid"],
                "argv": argv, "process_start_identity": current["start_identity"],
                "process_command": current["command"],
                "server_url": f"http://127.0.0.1:{port}",
                "activation_state_path": str(state_path), "registered_at": time.time(),
            }
            self.save()
            self.event("activation_server_adopted", process_key=key, pid=pid,
                       step_id=step.step_id)
        except Exception as exc:
            self.event("activation_server_adoption_error", step_id=step.step_id,
                       error=f"{type(exc).__name__}: {exc}")
            self.cancel.set()

    def stop_external_for_step(self, step_id: str) -> list[dict[str, Any]]:
        results = []
        for record in list(self.state.get("active_processes", {}).values()):
            if record.get("step_id") == step_id and record.get("role") == "activation-server":
                result = self.stop_record(record, "step_cleanup")
                results.append(result)
                if result.get("stopped") and record.get("activation_state_path"):
                    path = Path(record["activation_state_path"])
                    if path.exists():
                        try:
                            current = json.loads(path.read_text())
                            if int(current.get("pid", -1)) == int(record["pid"]):
                                path.unlink()
                        except Exception:
                            pass
        return results

    def write_step_receipt(self, step: Step, status: str, reason: str,
                           commands: list[dict[str, Any]], attempt: int | None = None) -> Path:
        value = {"schema": RECEIPT_SCHEMA, "step": step.public(), "status": status,
                 "reason": reason, "attempt": attempt, "identity": self.identity,
                 "commands": commands, "written_at": time.time(), "qualified": False,
                 "installed": False}
        target_dir = self.receipts_dir / step.step_id
        target_dir.mkdir(parents=True, exist_ok=True)
        name = f"attempt-{attempt:04d}.json" if attempt is not None else f"decision-{int(time.time() * 1_000_000)}.json"
        path = target_dir / name
        atomic_json(path, value)
        atomic_json(self.receipts_dir / f"{step.step_id}.json", value)
        return path

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.state.get("steps", {}).values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return counts

    def write_summary(self) -> None:
        counts = self.counts()
        summary = {"schema": "mlx2.qwen36-overnight-summary.v1", "identity": self.identity,
                   "counts": counts, "steps": self.state.get("steps", {}),
                   "generated_at": time.time(), "qualification_installed": False}
        atomic_json(self.config.run_dir / "summary.json", summary)
        lines = ["# Qwen3.6 overnight campaign summary", "", f"Identity: `{self.identity['sha256']}`", "",
                 "This campaign does not install qualification receipts or change serving defaults.", "",
                 "| Phase | Step | Status | Reason |", "|---|---|---|---|"]
        for step in self.steps:
            row = self.state.get("steps", {}).get(step.step_id, {})
            reason = str(row.get("reason", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {step.phase} | {step.label} | {row.get('status', 'planned')} | {reason} |")
        lines += ["", "## Counts", "", "```json", json.dumps(counts, sort_keys=True), "```", ""]
        (self.config.run_dir / "summary.md").write_text("\n".join(lines))


def build_plan(root: Path, run_dir: Path, ordinary: Path, mtp: Path, python: str,
               manifest: Path) -> list[Step]:
    py = str(root / python) if not Path(python).is_absolute() else python
    runs = run_dir / "artifacts"
    policy_o = "qualification/policies/qwen36-ordinary.json"
    policy_mo = "qualification/policies/qwen36-mtp-artifact-ordinary.json"
    policy_m = "qualification/policies/qwen36-mtp2.json"
    policy_p = "qualification/policies/qwen36-pld.json"
    q = lambda name, *args, timeout=3600: Command(name, (py, *map(str, args)), timeout)
    qualifier_source = (root / "scripts/qualify_serving.py").read_text()
    preflight_artifact = runs / "preflight.json"
    qualifier_preflight_args: list[str | Path] = []
    if "--preflight-receipt" in qualifier_source:
        qualifier_preflight_args = ["--preflight-receipt", preflight_artifact]
    elif "--unit-test-receipt" in qualifier_source:
        qualifier_preflight_args = ["--unit-test-receipt", run_dir / "receipts/preflight-full.json"]
    steps: list[Step] = [
        Step("preflight-full", "0", "Full CPU/Metal-aware suite with bound preflight receipt",
             (q("preflight", root / "scripts/qualify_serving.py", "--preflight-only",
                "--output", preflight_artifact, timeout=7200),)),
        Step("preflight-focused", "0", "Focused Qwen/APCv2/MTP/PLD/serving tests", (q("pytest-focused", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_qwen36_35b_port.py", "tests/test_prompt_lookup_core.py", "tests/test_batched_mtp.py", "tests/test_segmented_mtp.py", "tests/test_apcv2_lifecycle.py", "tests/test_admission_progress.py", "tests/test_serving_contract.py", "tests/test_serving_batch_features.py", "tests/test_structured_output.py", timeout=3600),)),
        Step("preflight-diff", "0", "Git diff whitespace check", (Command("git-diff-check", ("git", "diff", "--check"), 120),)),
        Step("preflight-build", "0", "Build distributions", (Command("uv-build", ("uv", "build"), 900),)),
        Step("preflight-json", "0", "Validate qualification JSON inputs", audit="json_inputs"),
        Step("preflight-ports", "0", "Candidate ports 8296-8298 are free", audit="ports_free"),
        Step("preflight-ready", "0", "All Phase 0 gates passed",
             dependencies=("preflight-full", "preflight-focused", "preflight-diff",
                           "preflight-build", "preflight-json", "preflight-ports")),
        Step("policies", "0.5", "Immutable Qwen3.6 execution policies", required_paths=(policy_o, policy_mo, policy_m)),
        Step("qualifier-contract", "0.5", "Identity-bound qualifier preflight receipt contract", audit="qualifier_preflight_receipt"),
        Step("qualifier-http-coverage", "0.5", "Tracker-required HTTP qualifier coverage", audit="qualifier_http_coverage"),
        Step("matrix-fail-soft", "0.5", "Matrix per-cell fail-soft continuation", audit="matrix_fail_soft"),
        Step("matrix-schedule", "0.5", "Context arm alternation contract", audit="matrix_schedule"),
        Step("activation-cleanup", "0.5", "Activation helper process-group cleanup contract", audit="activation_cleanup"),
        Step("matrix-manifest", "0.5", "Qwen3.6 experiment manifest", (q("validate-context", root / "scripts/run_qualification_matrix.py", "--manifest", manifest, "--output", runs / "validation-unused.json", "--suite", "context", "--validate-only", timeout=120), q("validate-batch", root / "scripts/run_qualification_matrix.py", "--manifest", manifest, "--output", runs / "validation-unused.json", "--suite", "batch_stress", "--validate-only", timeout=120)), required_paths=(str(manifest),)),
        Step("gpu-lease", "safety", "External CPG GPU lease and matching gpu.lock", audit="gpu_lease"),
    ]
    for label, artifact in (("ordinary", ordinary), ("mtp-artifact", mtp)):
        for round_index in range(1, 6):
            steps.append(Step(f"fused-{label}-r{round_index}", "1", f"Fused GDN {label} round {round_index}/5",
                (q("fused-gdn", root / "scripts/qwen36_fused_gdn_probe.py", "--model", artifact, "--steps", "512", "--output", runs / f"fused-{label}-r{round_index}.json", timeout=14400),),
                dependencies=("preflight-ready", "gpu-lease"), gpu=True))

    def serving(step_id: str, phase: str, label: str, artifact: Path, policy: str, port: int,
                ordinary_flag: bool, features: tuple[str, ...] = (),
                max_lanes: int = 4, max_inflight: int = 8) -> Step:
        url = f"http://127.0.0.1:{port}"
        server_argv = [py, "-m", "mlx2.server", "--model", str(artifact), "--host", "127.0.0.1", "--port", str(port), "--execution-policy", policy, "--max-context", "262144", "--max-lanes", str(max_lanes), "--max-inflight", str(max_inflight), "--cache-bytes", "12884901888", "--cache-dir", str(run_dir / "cache" / step_id), "--qualification-mode"]
        if ordinary_flag: server_argv.append("--ordinary")
        qualification = [root / "scripts/qualify_serving.py", "--url", url, "--timeout", "14400", "--quiescence-timeout", "180", *qualifier_preflight_args, "--defer-long-context-to-matrix", manifest, "--output", runs / f"{step_id}-qualification.json"]
        for feature in features: qualification.extend(("--require-feature", feature))
        http_probe = q("http-qualification", root / "scripts/qwen36_http_qualification.py",
                       "--url", url, "--timeout", "7200", "--resume",
                       "--output", runs / f"{step_id}-http-qualification.json", timeout=18000)
        return Step(step_id, phase, label, (q("qualification", *qualification, timeout=18000), http_probe, q("benchmark-b1-b2-b4", root / "scripts/benchmark_serving.py", "--url", url, "--rounds", "5", "--widths", "1", "2", "4", "--max-tokens", "256", "--timeout", "7200", "--output", runs / f"{step_id}-benchmark.json", timeout=18000)), dependencies=("preflight-ready", "policies", "qualifier-contract", "qualifier-http-coverage", "gpu-lease"), required_paths=(policy, "scripts/qwen36_http_qualification.py"), server=Server(tuple(server_argv), url), gpu=True)

    steps += [
        serving("ordinary-serving", "2", "Product ordinary serving qualification", ordinary, policy_o, 8296, True),
        serving("mtp-artifact-ordinary-serving", "2.5", "M artifact forced ordinary serving qualification", mtp, policy_mo, 8297, True),
        serving("mtp2-serving", "3", "Native MTP2 serving qualification", mtp, policy_m, 8297, False, ("segmented_transaction", "segmented_rollback")),
        Step("pld-prerequisite", "3.5", "PLD serving slice prerequisite", dependencies=("preflight-focused",), required_paths=(policy_p,), audit="pld_route"),
        serving("pld-serving", "3.5", "Conditional PLD serving qualification", mtp, policy_p, 8298, False),
        # The matrix itself gates each arm on its exact serving receipt; the
        # eligible-arm manifest is generated immediately before execution.
        Step("context-ladder", "4", "Alternating thermally controlled context ladder", (q("context", root / "scripts/run_qualification_matrix.py", "--manifest", manifest, "--suite", "context", "--output", runs / "context-ladder.json", "--resume", "--continue-on-error", timeout=43200),), dependencies=("preflight-ready", "matrix-manifest", "matrix-fail-soft", "matrix-schedule", "activation-cleanup", "gpu-lease"), gpu=True),
        serving("ordinary-b20-serving", "5", "Product ordinary max20 serving qualification",
                ordinary, policy_o, 8296, True, max_lanes=20, max_inflight=40),
        serving("mtp-artifact-ordinary-b20-serving", "5",
                "M artifact forced ordinary max20 serving qualification",
                mtp, policy_mo, 8297, True, max_lanes=20, max_inflight=40),
        serving("mtp2-b20-serving", "5", "Native MTP2 max20 serving qualification",
                mtp, policy_m, 8297, False,
                ("segmented_transaction", "segmented_rollback"), 20, 40),
        serving("pld-b20-serving", "5", "Conditional PLD max20 serving qualification",
                mtp, policy_p, 8298, False, max_lanes=20, max_inflight=40),
        Step("batch-20x20", "5", "Independent 20 rounds x actual B20", (q("batch", root / "scripts/run_qualification_matrix.py", "--manifest", manifest, "--suite", "batch_stress", "--output", runs / "batch-20x20.json", "--resume", "--continue-on-error", timeout=43200),), dependencies=("preflight-ready", "matrix-manifest", "matrix-fail-soft", "activation-cleanup", "gpu-lease"), gpu=True),
        Step("matched-report", "6", "Matched performance report", audit="report", always_run=True),
        Step("closeout", "cleanup", "Unconditional owned-process cleanup and handoff", audit="closeout", always_run=True),
    ]
    # The PLD arm is conditional on its own prerequisite without making core arms depend on it.
    for pld_step_id in ("pld-serving", "pld-b20-serving"):
        pld_index = next(i for i, step in enumerate(steps) if step.step_id == pld_step_id)
        pld = steps[pld_index]
        steps[pld_index] = Step(**{**pld.__dict__,
                                  "dependencies": pld.dependencies + ("pld-prerequisite",)})
    for batch_id, context_id in (
        ("ordinary-b20-serving", "ordinary-serving"),
        ("mtp-artifact-ordinary-b20-serving", "mtp-artifact-ordinary-serving"),
        ("mtp2-b20-serving", "mtp2-serving"),
        ("pld-b20-serving", "pld-serving"),
    ):
        index = next(i for i, step in enumerate(steps) if step.step_id == batch_id)
        step = steps[index]
        steps[index] = Step(**{**step.__dict__,
                              "dependencies": step.dependencies + (context_id,)})
    return steps


def parse_deadline(text: str | None) -> float | None:
    if text is None: return None
    try: return float(text)
    except ValueError:
        from datetime import datetime
        return datetime.fromisoformat(text).timestamp()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "run", "resume", "stop", "status", "report"))
    parser.add_argument("--run-dir", type=Path, default=root / "qualification/runs/qwen36-35b-a3b/overnight-campaign")
    parser.add_argument("--manifest", type=Path, default=root / "qualification/qwen36-overnight-experiments.json")
    parser.add_argument("--ordinary-model", type=Path, default=Path.home() / "mlx-models/Qwen3.6-35B-A3B-Abliterated-Heretic-MLX-4bit")
    parser.add_argument("--mtp-model", type=Path, default=Path.home() / "mlx-models/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--gpu-lease-receipt", type=Path)
    parser.add_argument("--gpu-lock", type=Path, default=Path("/tmp/gpu.lock"))
    parser.add_argument("--hard-deadline", help="Unix timestamp or local ISO-8601 time")
    parser.add_argument("--heartbeat-seconds", type=float, default=900)
    parser.add_argument("--terminate-grace", type=float, default=20)
    parser.add_argument("--cleanup-reserve", type=float, default=300,
                        help="seconds reserved before the hard deadline for cleanup/reporting")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    ensure_qwen36_policies(root)
    if args.action in {"prepare", "validate", "run", "resume"}:
        ensure_qwen36_manifest(root, run_dir, args.manifest.resolve(),
                               args.ordinary_model, args.mtp_model, args.python)
    settings = {"manifest": str(args.manifest.resolve()), "ordinary_model": str(args.ordinary_model.resolve()),
                "mtp_model": str(args.mtp_model.resolve()), "python": args.python,
                "hard_deadline": args.hard_deadline,
                "policy_sha256": {
                    name: sha256_file(root / "qualification/policies" / name)
                    for name in ("qwen36-ordinary.json", "qwen36-mtp-artifact-ordinary.json",
                                 "qwen36-mtp2.json")
                },
                "manifest_sha256": sha256_file(args.manifest) if args.manifest.exists() else None}
    identity = campaign_identity(root, [args.ordinary_model, args.mtp_model], settings)
    plan = build_plan(root, run_dir, args.ordinary_model, args.mtp_model, args.python, args.manifest.resolve())
    campaign = Campaign(Config(root=root, run_dir=run_dir, lease_receipt=args.gpu_lease_receipt,
                               gpu_lock=args.gpu_lock, hard_deadline=parse_deadline(args.hard_deadline),
                               heartbeat_seconds=args.heartbeat_seconds,
                               terminate_grace=args.terminate_grace,
                               cleanup_reserve=args.cleanup_reserve,
                               dry_run=args.action == "validate"), plan, identity)
    if args.action == "prepare":
        campaign.prepare(); print(campaign.state_path)
    elif args.action == "validate":
        campaign.prepare(); results = campaign.validate(); campaign.write_summary()
        print(json.dumps({"ready": sum(row["ready"] for row in results), "total": len(results),
                          "validation": str(run_dir / "validation.json")}, indent=2))
    elif args.action in {"run", "resume"}:
        campaign.run(resume=args.action == "resume")
        print(run_dir / "summary.md")
    elif args.action == "stop":
        result = campaign.explicit_stop(); print(json.dumps(result, indent=2))
        if not result["stopped"]:
            raise SystemExit(2)
    elif args.action == "status":
        print(json.dumps(campaign.status(), indent=2))
    else:
        campaign.load(); campaign.write_summary(); print(run_dir / "summary.md")


if __name__ == "__main__":
    main()
