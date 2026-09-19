#!/usr/bin/env python3
"""Run resumable, receipt-gated context and batching experiments.

The runner is deliberately model-agnostic.  Model families and route arms live
in a reviewed manifest; the serving runtime remains free of experiment names.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.request import Request, urlopen


SCHEMA = "mlx2.qualification-matrix.v1"
REPORT_SCHEMA = "mlx2.qualification-matrix-report.v1"
DEFAULT_SWAP_GROWTH_LIMIT_BYTES = 2 * 1024**3


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def verify_frozen_prompt(cell: dict[str, Any], text: str) -> str:
    """Bind a context cell to the exact prompt produced by calibration."""
    expected = cell.get("prompt_sha256")
    calibrated = cell.get("calibrated_prompt_tokens")
    if not expected or calibrated is None:
        raise ValueError("context suite requires a generated manifest with frozen prompt metadata")
    if int(calibrated) != int(cell["context_tokens"]):
        raise ValueError(
            f"calibrated prompt tokens {calibrated} do not match cell target {cell['context_tokens']}"
        )
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual != expected:
        raise ValueError(f"frozen prompt hash mismatch: expected {expected}, got {actual}")
    return actual


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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def nested(value: Any, path: str) -> Any:
    for part in path.split(".") if path else ():
        if isinstance(value, list):
            value = value[int(part)]
        else:
            value = value[part]
    return value


def requirement_passes(value: Any, requirement: dict[str, Any]) -> tuple[bool, Any]:
    try:
        actual = nested(value, requirement["path"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False, None
    operators = [name for name in ("equals", "gt", "gte", "lt", "lte", "contains") if name in requirement]
    if len(operators) != 1:
        raise ValueError(f"requirement needs exactly one operator: {requirement}")
    operator = operators[0]
    expected = requirement[operator]
    checks = {
        "equals": lambda: actual == expected,
        "gt": lambda: actual > expected,
        "gte": lambda: actual >= expected,
        "lt": lambda: actual < expected,
        "lte": lambda: actual <= expected,
        "contains": lambda: expected in actual,
    }
    return bool(checks[operator]()), actual


def validate_requirements(value: Any, requirements: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    evidence = []
    for requirement in requirements:
        passed, actual = requirement_passes(value, requirement)
        row = {"requirement": requirement, "actual": actual, "passed": passed}
        evidence.append(row)
        if not passed:
            raise AssertionError(f"{label} requirement failed: {row}")
    return evidence


def validate_bound_qualification_receipt(
    manifest_path: Path, arm: dict[str, Any], status: dict[str, Any]
) -> dict[str, Any] | None:
    """Require aggregate route evidence for mechanisms a single cell cannot engage."""
    configured = arm.get("qualification_receipt")
    if not configured:
        return None
    path = (manifest_path.parent / configured["path"]).resolve()
    report = json.loads(path.read_text())
    if report.get("schema") != "mlx2.serving-qualification.v1" or report.get("passed") is not True:
        raise AssertionError(f"bound qualification receipt is absent or failed: {path}")
    for name in ("runtime", "artifact", "settings"):
        if report.get(name) != status.get(name):
            raise AssertionError(f"bound qualification receipt {name} does not match active server")
    checks = report.get("checks", {})
    required = configured.get("required_checks", [])
    missing = [name for name in required if checks.get(name, {}).get("passed") is not True]
    if missing:
        raise AssertionError(f"bound qualification receipt lacks passed checks: {missing}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "required_checks": required,
        "passed": True,
    }


def counter_delta(before: Any, after: Any, path: str) -> float:
    return float(nested(after, path)) - float(nested(before, path))


def validate_counter_deltas(before: Any, after: Any, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for requirement in requirements:
        paths = requirement.get("paths") or [requirement["path"]]
        try:
            before_values = [nested(before, path) for path in paths]
            after_values = [nested(after, path) for path in paths]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            row = {"paths": paths, "before": None, "after": None,
                   "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            evidence.append(row)
            raise AssertionError(f"mechanism counter unavailable: {row}") from exc
        delta = sum(float(value) for value in after_values) - sum(float(value) for value in before_values)
        minimum = float(requirement.get("delta_gte", 1))
        maximum = float(requirement.get("delta_lte", float("inf")))
        row = {"paths": paths, "before": before_values, "after": after_values,
               "delta": delta, "delta_gte": minimum, "delta_lte": maximum,
               "passed": minimum <= delta <= maximum}
        evidence.append(row)
        if not row["passed"]:
            raise AssertionError(f"mechanism counter did not advance: {row}")
    return evidence


def wait_for_quiescence(
    client: Any,
    *,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.05,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait until terminal request cleanup is visible in server telemetry.

    A completion response and the periodically published APCv2 snapshot cross
    different threads.  Seeing the response therefore does not prove that the
    final status sample includes the branch release.  Poll all three lifecycle
    counters and preserve the samples so a genuine leak remains a hard failure.
    """
    timeout_seconds = float(timeout_seconds)
    poll_interval_seconds = float(poll_interval_seconds)
    if (
        not math.isfinite(timeout_seconds)
        or not math.isfinite(poll_interval_seconds)
        or timeout_seconds <= 0
        or poll_interval_seconds < 0
    ):
        raise ValueError("quiescence timing must be positive and bounded")

    started = monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        status = client.get("/v1/status")

        def counter(value: Any) -> int | None:
            return value if type(value) is int and value >= 0 else None

        sample = {
            "elapsed_seconds": monotonic() - started,
            "inflight": counter(status.get("inflight")),
            "queue_depth": counter(status.get("queue_depth")),
            "active_cow_leases": counter(
                status.get("apcv2", {}).get("cow", {}).get("active_leases")
            ),
        }
        samples.append(sample)
        if all(
            sample[name] == 0
            for name in ("inflight", "queue_depth", "active_cow_leases")
        ):
            return status, {
                "passed": True,
                "timed_out": False,
                "timeout_seconds": timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "attempts": len(samples),
                "elapsed_seconds": sample["elapsed_seconds"],
                "samples": samples,
            }
        remaining = timeout_seconds - sample["elapsed_seconds"]
        if remaining <= 0:
            raise AssertionError(
                "server did not reach observable request/cache quiescence: "
                f"{samples[-1]}"
            )
        sleep(min(poll_interval_seconds, remaining))


def applicable_requirements(requirements: list[dict[str, Any]], cell: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for requirement in requirements:
        if requirement.get("suite") not in (None, cell["suite"]):
            continue
        context = cell.get("context_tokens")
        if context is not None and context < requirement.get("min_context_tokens", 0):
            continue
        if context is not None and context > requirement.get("max_context_tokens", float("inf")):
            continue
        selected.append({key: value for key, value in requirement.items()
                         if key not in {"suite", "min_context_tokens", "max_context_tokens"}})
    return selected


def parse_pmset_therm(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {"raw": text.strip()}
    names = {
        "CPU Power notify": "cpu_power_notify",
        "GPU Power notify": "gpu_power_notify",
        "CPU Speed Limit": "cpu_speed_limit",
        "Scheduler limit": "scheduler_limit",
        "Available CPUs": "available_cpus",
    }
    for source, target in names.items():
        match = re.search(rf"^{re.escape(source)}\s*=\s*(-?\d+)", text, re.MULTILINE)
        if match:
            metrics[target] = int(match.group(1))
    return metrics


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise RuntimeError(f"thermal command failed ({completed.returncode}): {completed.stderr.strip()}")
    return completed


def _thermal_probe_binary() -> Path:
    source = Path(__file__).with_name("thermal_probe.swift")
    if not source.exists():
        raise RuntimeError(f"missing thermal probe source: {source}")
    target = Path(tempfile.gettempdir()) / f"mlx2-thermal-probe-{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}"
    if not target.exists():
        completed = subprocess.run(["/usr/bin/swiftc", str(source), "-o", str(target)],
                                   capture_output=True, text=True, timeout=120)
        if completed.returncode:
            raise RuntimeError(f"could not compile thermal probe: {completed.stderr.strip()}")
    return target


def _battery_temperatures() -> tuple[dict[str, float], dict[str, Any]]:
    completed = _run(["/usr/sbin/ioreg", "-r", "-n", "AppleSmartBattery", "-a"])
    values: dict[str, float] = {}
    raw: dict[str, Any] = {}
    if completed.stdout.strip():
        payload = plistlib.loads(completed.stdout.encode())
        row = payload[0] if payload else {}
        for source, target in (("Temperature", "battery_temperature_c"),
                               ("VirtualTemperature", "virtual_temperature_c")):
            value = row.get(source)
            if isinstance(value, (int, float)):
                raw[source] = value
                values[target] = float(value) / 100.0
    return values, raw


def sample_thermal(command: list[str] | None = None) -> dict[str, Any]:
    if command:
        completed = _run(command)
        try:
            sample = json.loads(completed.stdout)
        except json.JSONDecodeError:
            sample = parse_pmset_therm(completed.stdout)
        sample.update({"timestamp": time.time(), "command": command, "raw": completed.stdout.strip()})
        return sample
    state_result = _run([str(_thermal_probe_binary())])
    try:
        state = json.loads(state_result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"thermal probe returned invalid JSON: {state_result.stdout!r}") from exc
    pmset_result = _run(["/usr/bin/pmset", "-g", "therm"])
    pmset = parse_pmset_therm(pmset_result.stdout)
    temperatures, ioreg_raw = _battery_temperatures()
    sample = {**state, **pmset, **temperatures, "timestamp": time.time(),
              "command": [str(_thermal_probe_binary())],
              "pmset_no_thermal_warning": "No thermal warning" in pmset_result.stdout,
              "pmset_no_performance_warning": "No performance warning" in pmset_result.stdout,
              "pmset_no_cpu_power_warning": ("No CPU power" in pmset_result.stdout
                                               or "CPU Power notify = 0" in pmset_result.stdout),
              "raw_evidence": {"thermal_probe": state_result.stdout.strip(),
                               "pmset": pmset_result.stdout.strip(), "ioreg_temperature_fields": ioreg_raw}}
    return sample


def thermally_stable(sample: dict[str, Any], policy: dict[str, Any]) -> bool:
    if "thermal_state" not in sample:
        return False
    if int(sample["thermal_state"]) != int(policy.get("required_thermal_state", 0)):
        return False
    for key in ("pmset_no_thermal_warning", "pmset_no_performance_warning", "pmset_no_cpu_power_warning"):
        if sample.get(key) is not True:
            return False
    if policy.get("require_temperatures", True):
        if "battery_temperature_c" not in sample or "virtual_temperature_c" not in sample:
            return False
    return (sample.get("battery_temperature_c", float("-inf")) <= policy.get("max_battery_temperature_c", 40.0)
            and sample.get("virtual_temperature_c", float("-inf")) <= policy.get("max_virtual_temperature_c", 45.0))


def stabilize_thermal(policy: dict[str, Any], sleep=time.sleep) -> list[dict[str, Any]]:
    command = policy.get("command")
    consecutive = int(policy.get("consecutive_samples", 3))
    interval = float(policy.get("sample_interval_seconds", 15))
    deadline = time.monotonic() + float(policy.get("max_wait_seconds", 900))
    samples: list[dict[str, Any]] = []
    stable_samples: list[dict[str, Any]] = []
    while time.monotonic() <= deadline:
        sample = sample_thermal(command)
        samples.append(sample)
        stable_samples = stable_samples + [sample] if thermally_stable(sample, policy) else []
        stable_samples = stable_samples[-consecutive:]
        temperatures = [row.get("virtual_temperature_c") for row in stable_samples
                        if row.get("virtual_temperature_c") is not None]
        settled = (len(temperatures) == consecutive
                   and max(temperatures) - min(temperatures)
                   <= float(policy.get("max_temperature_delta_c", 0.5)))
        if len(stable_samples) >= consecutive and (settled or not policy.get("require_temperatures", True)):
            return samples
        sleep(interval)
    raise TimeoutError(f"thermal stabilization timed out after {len(samples)} samples")


def post_thermal_samples(policy: dict[str, Any], sleep=time.sleep) -> list[dict[str, Any]]:
    """Apply the overnight rule: invalidate only two consecutive bad samples."""
    command = policy.get("command")
    samples = [sample_thermal(command)]
    if not thermally_stable(samples[0], policy):
        sleep(float(policy.get("post_sample_interval_seconds",
                               policy.get("sample_interval_seconds", 15))))
        samples.append(sample_thermal(command))
    if len(samples) == 2 and all(not thermally_stable(row, policy) for row in samples):
        raise AssertionError(f"thermal state breached twice after measured cell: {samples}")
    return samples


def validate_post_thermal(sample: dict[str, Any], policy: dict[str, Any]) -> None:
    """Compatibility helper for callers that already collected one sample."""
    if not thermally_stable(sample, policy):
        raise AssertionError(f"thermal state breached immediately after measured cell: {sample}")


def sample_swap(command: list[str] | None = None) -> dict[str, Any]:
    """Return byte-accurate swap use with the raw source preserved."""
    command = command or ["/usr/sbin/sysctl", "-n", "vm.swapusage"]
    completed = _run(command)
    raw = completed.stdout.strip()
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", raw, re.IGNORECASE)
    if not match:
        # Test probes and alternate tools may return a small JSON object.
        try:
            value = json.loads(raw)
            used = int(value["used_bytes"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not parse swap usage: {raw!r}") from exc
    else:
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3,
                 "T": 1024**4, "P": 1024**5}[match.group(2).upper()]
        used = int(float(match.group(1)) * scale)
    return {"used_bytes": used, "timestamp": time.time(), "command": command, "raw": raw}


def diagnostic_thermal(policy: dict[str, Any]) -> dict[str, Any]:
    """Collect batch-suite telemetry without turning it into admission control."""
    try:
        return {"sample": sample_thermal(policy.get("command")), "available": True}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def swap_evidence(baseline: dict[str, Any], before: dict[str, Any], after: dict[str, Any],
                  limit_bytes: int = DEFAULT_SWAP_GROWTH_LIMIT_BYTES,
                  *, enforce: bool = True) -> dict[str, Any]:
    growth = int(after["used_bytes"]) - int(baseline["used_bytes"])
    evidence = {"baseline": baseline, "before": before, "after": after,
                "growth_from_baseline_bytes": growth, "limit_bytes": int(limit_bytes),
                "enforced": enforce, "passed": growth <= int(limit_bytes)}
    if enforce and not evidence["passed"]:
        raise AssertionError(f"swap grew {growth} bytes from suite baseline; limit is {limit_bytes}")
    return evidence


def alternating_cells(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for model in models:
        runs = int(model.get("context", {}).get("runs_per_cell", 3))
        contexts = model.get("contexts", [])
        for context in contexts:
            for run in range(runs):
                arms = model["arms"]
                for arm in arms:
                    if "prompt_path" not in context:
                        raise ValueError(
                            "context suite requires a generated manifest; run "
                            "scripts/build_context_prompts.py first"
                        )
                    cell = {"suite": "context", "model": model["name"], "arm": arm["name"],
                            "run": run, "context_tokens": int(context["tokens"]),
                            "arm_order_claim": model.get("arm_order_claim", "alternating"),
                            "prompt_path": context["prompt_path"],
                            "prompt_sha256": context.get("prompt_sha256"),
                            "calibrated_prompt_tokens": context.get("calibrated_prompt_tokens")}
                    cell["cell_id"] = digest(cell)
                    cells.append(cell)
    return cells


def batch_stress_cells(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for model in models:
        config = model.get("batch_stress", {})
        rounds, width = int(config.get("rounds", 20)), int(config.get("width", 20))
        for arm in model["arms"]:
            for run in range(rounds):
                cell = {"suite": "batch_stress", "model": model["name"], "arm": arm["name"],
                        "run": run, "requested_width": width,
                        "arm_order_claim": model.get("arm_order_claim", "alternating"),
                        "semantics": f"{rounds} rounds x {width} simultaneous requests per arm"}
                cell["cell_id"] = digest(cell)
                cells.append(cell)
    return cells


def host_identity() -> dict[str, Any]:
    def command(*argv: str) -> str:
        result = subprocess.run(argv, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unavailable"
    script_dir = Path(__file__).resolve().parent
    harness_files = [script_dir / name for name in (
        "run_qualification_matrix.py", "run_spomin_20x20.py",
        "build_context_prompts.py", "activate_qualification_arm.py", "thermal_probe.swift"
    ) if (script_dir / name).exists()]
    harness_source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in harness_files}
    return {"platform": platform.platform(), "macos": platform.mac_ver()[0],
            "sw_vers": command("/usr/bin/sw_vers"), "uname": command("/usr/bin/uname", "-a"),
            "python": sys.version, "mlx": mlx_version,
            "git_revision": command("/usr/bin/git", "rev-parse", "HEAD"),
            "git_status": command("/usr/bin/git", "status", "--short"),
            "harness_source": harness_source, "harness_source_sha256": digest(harness_source)}


def stable_host_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in ("platform", "macos", "sw_vers", "uname", "python", "mlx",
                                            "git_revision", "harness_source_sha256")}


class Client:
    def __init__(self, url: str, timeout: float):
        self.url, self.timeout = url.rstrip("/"), timeout

    def get(self, path: str) -> Any:
        with urlopen(self.url + path, timeout=min(self.timeout, 30)) as response:
            return json.load(response)

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        with urlopen(Request(self.url + "/v1/chat/completions", data=canonical(body),
                             headers={"Content-Type": "application/json"}), timeout=self.timeout) as response:
            result = json.load(response)
        result["_client_wall_seconds"] = time.monotonic() - started
        return result


def activate(arm: dict[str, Any], timeout: float) -> Client:
    global _ACTIVE_ARM
    activation_key = arm.get("activation_key", digest({name: arm.get(name) for name in ("name", "url", "activate_command")}))
    command = arm.get("activate_command")
    if command and _ACTIVE_ARM != activation_key:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"activation failed for {arm['name']}: {completed.stderr.strip()}")
    client = Client(arm["url"], timeout)
    deadline = time.monotonic() + float(arm.get("ready_timeout_seconds", 900))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = client.get("/v1/status")
            if status.get("healthy") and status.get("inflight") == 0:
                _ACTIVE_ARM = activation_key
                return client
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"arm {arm['name']} did not become ready: {last_error}")


def identity(status: dict[str, Any]) -> dict[str, Any]:
    return {name: status[name] for name in ("runtime", "artifact", "settings", "profile")}


def observed_width(receipt: dict[str, Any]) -> int:
    mtp = receipt.get("mtp") or {}
    if mtp.get("observed_compute_widths"):
        return max(int(item) for item in mtp["observed_compute_widths"])
    speculation = receipt.get("speculation") or {}
    if speculation.get("target_width") is not None:
        return int(speculation["target_width"])
    return int(receipt.get("ordinary_compute_width") or 1)


def request_evidence(result: dict[str, Any]) -> dict[str, Any]:
    content = result["choices"][0]["message"].get("content", "")
    return {"usage": result["usage"], "finish_reason": result["choices"][0]["finish_reason"],
            "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "client_wall_seconds": result["_client_wall_seconds"], "receipt": result["mlx2"]}


def prompt_body(text: str, max_tokens: int) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}], "temperature": 0,
            "max_tokens": max_tokens, "enable_thinking": False}


def arm_for(model: dict[str, Any], name: str) -> dict[str, Any]:
    return next(arm for arm in model["arms"] if arm["name"] == name)


def effective_arm(arm: dict[str, Any], suite: str) -> dict[str, Any]:
    """Resolve settings-bound activation and receipts for one experiment suite."""
    suites = arm.get("suites", {})
    override = suites.get(suite, {})
    if not isinstance(override, dict):
        raise ValueError(f"{arm.get('name')} suite override {suite} must be an object")
    resolved = {key: value for key, value in arm.items() if key != "suites"}
    resolved.update(override)
    return resolved


def isolated_context_arm(arm: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Bind one context measurement to a freshly cleared cache namespace."""
    command = arm.get("activate_command")
    if not command:
        raise ValueError(f"context arm {arm.get('name')} requires an activation command for cold-prime isolation")
    command = list(command)
    try:
        separator = command.index("--")
        cache_flag = command.index("--cache-dir", separator + 1)
        base_cache = Path(command[cache_flag + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"context arm {arm.get('name')} activation must launch through the helper with --cache-dir"
        ) from exc
    cache_dir = base_cache / f"cell-{cell['cell_id']}"
    command[cache_flag + 1] = str(cache_dir)
    command[separator:separator] = ["--fresh-cache-dir", str(cache_dir)]
    isolated = dict(arm)
    isolated["activate_command"] = command
    isolated["activation_key"] = f"{arm.get('activation_key', arm.get('name'))}:context:{cell['cell_id']}"
    isolated["context_cache_dir"] = str(cache_dir)
    return isolated


def context_cell(manifest_path: Path, model: dict[str, Any], arm: dict[str, Any], cell: dict[str, Any],
                 thermal: dict[str, Any], swap_baseline: dict[str, Any]) -> dict[str, Any]:
    prompt_path = (manifest_path.parent / cell["prompt_path"]).resolve()
    text = prompt_path.read_text()
    prompt_sha256 = verify_frozen_prompt(cell, text)
    config = model.get("context", {})
    body = prompt_body(text, int(config.get("max_tokens", 64)))
    arm = isolated_context_arm(arm, cell)
    client = activate(arm, float(config.get("timeout_seconds", 3600)))
    swap_before = sample_swap(thermal.get("swap_command"))
    before = client.get("/v1/status")
    validate_requirements(before, arm.get("status_requirements", []), "status")
    aggregate_receipt = validate_bound_qualification_receipt(manifest_path, arm, before)
    # Prime outside the measured interval, then cool to the configured baseline.
    prime = client.post(body)
    prime_evidence = request_evidence(prime)
    thermal_samples = stabilize_thermal(thermal)
    measured = client.post(body)
    thermal_after = post_thermal_samples(thermal)
    swap_after = sample_swap(thermal.get("swap_command"))
    swap = swap_evidence(swap_baseline, swap_before, swap_after,
                         int(thermal.get("max_swap_growth_bytes", DEFAULT_SWAP_GROWTH_LIMIT_BYTES)))
    after, quiescence = wait_for_quiescence(
        client,
        timeout_seconds=float(config.get("quiescence_timeout_seconds", 5.0)),
        poll_interval_seconds=float(config.get("quiescence_poll_interval_seconds", 0.05)),
    )
    if identity(before) != identity(after):
        raise AssertionError("server identity changed during context cell")
    receipt = measured["mlx2"]
    receipt_checks = validate_requirements(
        receipt,
        applicable_requirements(arm.get("receipt_requirements", []), cell),
        "receipt",
    )
    prime_cached = int(prime["mlx2"].get("cached_tokens", 0))
    measured_cached = int(receipt.get("cached_tokens", 0))
    if measured_cached < int(config.get("min_cached_tokens", 1)):
        raise AssertionError("measured context request did not reuse APCv2")
    if measured_cached - prime_cached < int(config.get("min_prime_to_warm_cache_gain_tokens", 1)):
        raise AssertionError(
            f"context prime was not demonstrably cold: prime cached {prime_cached}, "
            f"warm cached {measured_cached}"
        )
    cold_ttft = prime["mlx2"].get("ttft_seconds")
    warm_ttft = receipt.get("ttft_seconds")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
               for value in (cold_ttft, warm_ttft)):
        raise AssertionError("context cell lacks finite cold/warm TTFT evidence")
    actual = int(measured["usage"]["prompt_tokens"])
    tolerance = int(config.get("prompt_token_tolerance", 8))
    if abs(actual - cell["context_tokens"]) > tolerance:
        raise AssertionError(f"prompt tokens {actual} outside target {cell['context_tokens']} +/- {tolerance}")
    counters = validate_counter_deltas(before, after,
                                       applicable_requirements(arm.get("counter_requirements", []), cell))
    after_checks = validate_requirements(after, arm.get("after_status_requirements", []), "final status")
    return {"cell_manifest": cell, "cell_manifest_sha256": digest(cell),
            "prompt": {"path": str(prompt_path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "frozen_sha256": prompt_sha256,
                       "target_tokens": cell["context_tokens"], "actual_tokens": actual},
            "server_identity": identity(before), "status_before": before, "status_after": after,
            "context_cache_dir": arm["context_cache_dir"],
            "bound_qualification_receipt": aggregate_receipt,
            "thermal_control": {"enabled": True, "pre_samples": thermal_samples,
                                "post_samples": thermal_after, "bracket_passed": True},
            "quiescence": quiescence,
            "swap": swap,
            "prime": prime_evidence, "measured": request_evidence(measured),
            "latency": {"cold_ttft_seconds": cold_ttft, "warm_ttft_seconds": warm_ttft},
            "receipt_checks": receipt_checks, "counter_checks": counters,
            "after_status_checks": after_checks, "passed": True}


_ACTIVE_ARM: str | None = None


def declared_batch_bodies(
    bodies: list[dict[str, Any]], *, cell_id: str, width: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(bodies) != width:
        raise ValueError("declared batch cohort must match its request count")
    cohort = {"id": cell_id, "size": width}
    return cohort, [{**body, "batch_cohort": cohort} for body in bodies]


def batch_cell(manifest_path: Path, model: dict[str, Any], arm: dict[str, Any], cell: dict[str, Any],
               thermal: dict[str, Any], swap_baseline: dict[str, Any]) -> dict[str, Any]:
    config = model.get("batch_stress", {})
    timeout = float(config.get("timeout_seconds", 3600))
    width = int(cell["requested_width"])
    prompts = config.get("prompts")
    if not prompts or len(prompts) < width:
        raise ValueError(f"{model['name']} batch.prompts needs at least {width} entries")
    bodies = [prompt_body(text, int(config.get("max_tokens", 64))) for text in prompts[:width]]
    client = activate(arm, timeout)
    diagnostic_before = diagnostic_thermal(thermal)
    swap_before = sample_swap(thermal.get("swap_command"))
    before = client.get("/v1/status")
    if int(before.get("max_lanes", 0)) < width:
        raise AssertionError(f"server max_lanes={before.get('max_lanes')} cannot run requested width {width}")
    validate_requirements(before, arm.get("status_requirements", []), "status")
    aggregate_receipt = validate_bound_qualification_receipt(manifest_path, arm, before)
    # Every measured request must be a warm APCv2 request.
    warmup = [request_evidence(client.post(body)) for body in bodies]
    # Independent HTTP handlers cannot infer how many peers are still being
    # published. Declare the exact measured cohort so ServingEngine releases
    # all members atomically to the generation queue; this preserves the
    # strict physical-width oracle without imposing a wider timer on B1-B4.
    batch_cohort, measured_bodies = declared_batch_bodies(
        bodies, cell_id=cell["cell_id"], width=width
    )
    started = time.monotonic()
    barrier = threading.Barrier(width + 1)
    request_starts: list[float] = []
    start_lock = threading.Lock()
    def synchronized_post(body):
        barrier.wait(timeout=30)
        request_started = time.monotonic()
        with start_lock:
            request_starts.append(request_started)
        return client.post(body)
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(synchronized_post, body) for body in measured_bodies]
        barrier.wait(timeout=30)
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    after, quiescence = wait_for_quiescence(
        client,
        timeout_seconds=float(config.get("quiescence_timeout_seconds", 5.0)),
        poll_interval_seconds=float(config.get("quiescence_poll_interval_seconds", 0.05)),
    )
    diagnostic_after = diagnostic_thermal(thermal)
    swap_after = sample_swap(thermal.get("swap_command"))
    swap = swap_evidence(swap_baseline, swap_before, swap_after,
                         int(thermal.get("max_swap_growth_bytes", DEFAULT_SWAP_GROWTH_LIMIT_BYTES)))
    if identity(before) != identity(after):
        raise AssertionError("server identity changed during batch cell")
    rows, checks = [], []
    for result in results:
        receipt = result["mlx2"]
        if receipt.get("cache") != "apcv2" or int(receipt.get("cached_tokens", 0)) < 1:
            raise AssertionError("batch request missed APCv2 warm state")
        if (receipt.get("request_controls") or {}).get("batch_cohort") != batch_cohort:
            raise AssertionError("batch request did not preserve its declared cohort")
        checks.append(validate_requirements(receipt, arm.get("receipt_requirements", []), "receipt"))
        rows.append(request_evidence(result))
    widths = [observed_width(row["receipt"]) for row in rows]
    if max(widths, default=0) != width:
        raise AssertionError(f"requested actual compute width {width}, observed {sorted(set(widths))}")
    start_spread = max(request_starts) - min(request_starts)
    if start_spread > float(config.get("max_request_start_spread_seconds", 0.25)):
        raise AssertionError(f"B{width} client start spread {start_spread:.6f}s exceeded bound")
    counters = validate_counter_deltas(before, after,
                                       applicable_requirements(arm.get("counter_requirements", []), cell))
    after_checks = validate_requirements(after, arm.get("after_status_requirements", []), "final status")
    return {"cell_manifest": cell, "cell_manifest_sha256": digest(cell),
            "server_identity": identity(before), "status_before": before, "status_after": after,
            "bound_qualification_receipt": aggregate_receipt,
            "thermal_control": {"enabled": False, "reason": "separate no-thermal batch suite",
                                "diagnostic_only": True, "before": diagnostic_before,
                                "after": diagnostic_after},
            "quiescence": quiescence,
            "swap": swap,
            "warmup": warmup, "requests": rows, "receipt_checks": checks,
            "batch_cohort": batch_cohort,
            "counter_checks": counters, "after_status_checks": after_checks,
            "observed_compute_widths": sorted(set(widths)),
            "elapsed_seconds": elapsed,
            "request_start_spread_seconds": start_spread,
            "aggregate_tokens_per_second": sum(row["usage"]["completion_tokens"] for row in rows) / elapsed,
            "passed": True}


def validate_manifest(
    manifest: dict[str, Any], *, allow_exploratory_single_run: bool = False
) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    if not manifest.get("models"):
        raise ValueError("manifest needs models")
    for service in manifest.get("host_services", []):
        if not isinstance(service.get("label"), str) or not isinstance(service.get("plist"), str):
            raise ValueError("host_services entries need a launchd label and the plist that restores it")
    for model in manifest["models"]:
        if model.get("tokenizer_renderer") not in {"qwen_direct", "muse_direct", "north_direct"}:
            raise ValueError(f"{model.get('name')} needs an explicit supported tokenizer_renderer")
        arms = model.get("arms", [])
        order_claim = model.get("arm_order_claim", "alternating")
        if order_claim == "none":
            if len(arms) != 1:
                raise ValueError(f"{model.get('name')} arm_order_claim none requires exactly one route arm")
        elif order_claim == "alternating":
            if len(arms) < 2:
                raise ValueError(f"{model.get('name')} comparative qualification needs at least two route arms")
        else:
            raise ValueError(f"{model.get('name')} has unsupported arm_order_claim {order_claim!r}")
        if len({arm["name"] for arm in arms}) != len(arms):
            raise ValueError(f"{model['name']} arm names must be unique")
        selected = model.get("selected_arm")
        if selected is not None and selected not in {arm["name"] for arm in arms}:
            raise ValueError(f"{model['name']} selected_arm {selected!r} is not one of its arms")
        for arm in arms:
            unknown = set(arm.get("suites", {})) - {"context", "batch_stress"}
            if unknown:
                raise ValueError(f"{model['name']}/{arm['name']} has unknown suite overrides {sorted(unknown)}")
            for suite in ("context", "batch_stress"):
                resolved = effective_arm(arm, suite)
                if not resolved.get("receipt_requirements"):
                    raise ValueError(f"{model['name']}/{arm['name']}/{suite} must fail closed on receipt requirements")
                aggregate = resolved.get("qualification_receipt")
                if not aggregate:
                    raise ValueError(
                        f"{model['name']}/{arm['name']}/{suite} must bind a canonical qualification receipt"
                    )
                if (
                    not isinstance(aggregate.get("path"), str)
                    or not aggregate["path"]
                    or not aggregate.get("required_checks")
                ):
                    raise ValueError(f"{model['name']}/{arm['name']}/{suite} has an incomplete aggregate qualification receipt gate")
        runs_per_cell = int(model.get("context", {}).get("runs_per_cell", 3))
        allowed_runs = {1, 3} if allow_exploratory_single_run else {3}
        if runs_per_cell not in allowed_runs:
            qualifier = " or one explicitly exploratory run" if allow_exploratory_single_run else ""
            raise ValueError(
                "thermally controlled qualification requires exactly three runs per cell"
                + qualifier
            )
        batch = model.get("batch_stress", {})
        if int(batch.get("rounds", 20)) != 20 or int(batch.get("width", 20)) != 20:
            raise ValueError("batch_stress requires exactly 20 rounds at simultaneous request width 20")


def launchd_loaded(label: str) -> bool:
    domain = f"gui/{os.getuid()}"
    return subprocess.run(["/bin/launchctl", "print", f"{domain}/{label}"], capture_output=True).returncode == 0


def quiesce_host_services(services: list[dict[str, Any]], *, allowed: bool) -> list[dict[str, Any]]:
    """Boot out resident GPU services for the run and restore them at exit.

    A resident model server shares unified memory with every arm: on this host
    it left 24.7 GiB of headroom against a 20 GiB reserve, which turned a whole
    probe run into 429s and looked like a capacity limit.  Taking a production
    service down is the operator's decision, so without ``--quiesce-host-services``
    a loaded service stops the run instead.
    """
    resident = [service for service in services if launchd_loaded(service["label"])]
    if resident and not allowed:
        raise SystemExit(
            "host service(s) " + ", ".join(service["label"] for service in resident)
            + " are loaded and would share the GPU with every arm; rerun with "
            "--quiesce-host-services to boot them out for this run and restore them afterwards"
        )
    domain = f"gui/{os.getuid()}"
    outcome = []

    def restore() -> None:
        for service, row in zip(resident, outcome):
            plist = str(Path(service["plist"]).expanduser())
            subprocess.run(["/bin/launchctl", "bootstrap", domain, plist], capture_output=True)
            row["restored"] = launchd_loaded(service["label"])
            print(f"[host-services] restored {service['label']}: {row['restored']}", flush=True)

    for service in resident:
        subprocess.run(["/bin/launchctl", "bootout", f"{domain}/{service['label']}"], capture_output=True)
        outcome.append({"label": service["label"], "quiesced": not launchd_loaded(service["label"]), "restored": None})
        print(f"[host-services] quiesced {service['label']}: {outcome[-1]['quiesced']}", flush=True)
    if resident:
        import atexit
        import signal

        atexit.register(restore)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so atexit still restores
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("context", "batch_stress"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Persist failed cells and continue independent cells")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--quiesce-host-services", action="store_true",
        help=("Boot out the manifest's resident host services for this run and restore them at "
              "exit. Without it, a loaded service stops the run before any arm starts."),
    )
    parser.add_argument(
        "--allow-exploratory-single-run",
        action="store_true",
        help=("Permit a one-run-per-cell context matrix. This is exploratory "
              "A/B evidence, not thermally replicated qualification."),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(
        manifest,
        allow_exploratory_single_run=args.allow_exploratory_single_run,
    )
    manifest_hash = digest(manifest)
    cells = alternating_cells(manifest["models"]) if args.suite == "context" else batch_stress_cells(manifest["models"])
    if args.validate_only:
        print(json.dumps({"manifest_sha256": manifest_hash, "suite": args.suite,
                          "cells": cells, "cell_count": len(cells)}, indent=2))
        return
    host_services = quiesce_host_services(
        manifest.get("host_services", []), allowed=args.quiesce_host_services
    )
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report.get("manifest_sha256") != manifest_hash or report.get("suite") != args.suite:
            raise ValueError("resume report does not match manifest and suite")
        if stable_host_identity(report["host"]) != stable_host_identity(host_identity()):
            raise ValueError("resume host/OS/runtime/source identity mismatch")
        if "swap_baseline" not in report:
            report["swap_baseline"] = sample_swap(manifest["thermal"].get("swap_command"))
    else:
        report = {"schema": REPORT_SCHEMA, "manifest_sha256": manifest_hash,
                  "manifest": manifest, "suite": args.suite, "host": host_identity(),
                  "semantics": (("one exploratory run per model/context/arm; comparative models cycle every qualified arm after each measurement; thermal stabilization precedes every measured cell; results are not thermally replicated qualification"
                                 if args.allow_exploratory_single_run else
                                 "three runs per model/context/arm; comparative models cycle every qualified arm after each measurement while explicit ordinary-only models make no arm-order claim; thermal stabilization precedes every measured cell")
                                if args.suite == "context" else
                                "20 rounds x actual compute width 20 for each model/arm; thermal gating intentionally disabled"),
                  "started_at": time.time(), "cells": {}, "passed": False}
        report["swap_baseline"] = sample_swap(manifest["thermal"].get("swap_command"))
        atomic_json(args.output, report)
    report["host_services"] = host_services  # rows are updated in place when restored at exit
    completed = {key for key, row in report["cells"].items() if row.get("passed")}
    arm_identities = {}
    for row in report["cells"].values():
        if row.get("passed") and row.get("server_identity"):
            key = (row["cell_manifest"]["model"], row["cell_manifest"]["arm"])
            previous = arm_identities.setdefault(key, row["server_identity"])
            if previous != row["server_identity"]:
                raise ValueError(f"resume report mixes server identities for {key}")
    model_map = {model["name"]: model for model in manifest["models"]}
    for index, cell in enumerate(cells, 1):
        if cell["cell_id"] in completed:
            continue
        model = model_map[cell["model"]]
        arm = effective_arm(arm_for(model, cell["arm"]), args.suite)
        print(f"[{index}/{len(cells)}] {cell}", flush=True)
        try:
            result = (context_cell(args.manifest, model, arm, cell, manifest["thermal"], report["swap_baseline"])
                      if args.suite == "context" else batch_cell(args.manifest, model, arm, cell,
                                                                  manifest["thermal"], report["swap_baseline"]))
            key = (cell["model"], cell["arm"])
            if key in arm_identities and arm_identities[key] != result["server_identity"]:
                raise ValueError(f"server identity changed across resumed cells for {key}")
            arm_identities[key] = result["server_identity"]
            previous = report["cells"].get(cell["cell_id"])
            if previous:
                result["attempts"] = [*previous.get("attempts", []),
                                      {name: value for name, value in previous.items()
                                       if name != "attempts"}]
        except Exception as exc:
            previous = report["cells"].get(cell["cell_id"])
            attempts = list(previous.get("attempts", [])) if previous else []
            if previous:
                attempts.append({name: value for name, value in previous.items()
                                 if name != "attempts"})
            report["cells"][cell["cell_id"]] = {"cell_manifest": cell, "passed": False,
                                                   "error": f"{type(exc).__name__}: {exc}",
                                                   "failed_at": time.time(), "attempts": attempts}
            report["last_progress_at"] = time.time()
            atomic_json(args.output, report)
            if not args.continue_on_error:
                raise
            continue
        report["cells"][cell["cell_id"]] = result
        report["last_progress_at"] = time.time()
        atomic_json(args.output, report)
    report["passed"] = len(report["cells"]) == len(cells) and all(row.get("passed") for row in report["cells"].values())
    report["completed_at"] = time.time()
    atomic_json(args.output, report)
    if not report["passed"]:
        raise SystemExit("matrix incomplete")


if __name__ == "__main__":
    main()
