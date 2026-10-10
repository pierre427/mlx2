#!/usr/bin/env python3
"""Run one VLM media companion and its matching generic serving profile.

This coordinator is intended to run inside run_qualification_owned.py and
run_with_gpu_locks.py. It starts no model unless the reviewed media producer
first emits a passing report for the exact artifact and lease.
"""

from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import math
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from scripts.qualify_vlm_routes import FAMILIES, artifact_sha256
except ModuleNotFoundError:
    from qualify_vlm_routes import FAMILIES, artifact_sha256


def _safe_profile_value(node):
    """Evaluate only literal dict/list/scalars and integer shift expressions."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            _safe_profile_value(key): _safe_profile_value(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_safe_profile_value(value) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_profile_value(value) for value in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _safe_profile_value(node.operand)
        if type(value) in (int, float):
            return -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
        left, right = _safe_profile_value(node.left), _safe_profile_value(node.right)
        if type(left) is int and type(right) is int and 0 <= right <= 40:
            return left << right
    raise ValueError(f"unsupported producer profile expression: {ast.dump(node)}")


def producer_profile(root, family):
    if family not in FAMILIES:
        raise ValueError(f"unsupported VLM family: {family}")
    path = Path(root) / "scripts" / FAMILIES[family]["producer"]
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SETUP"
            for target in node.targets
        ):
            profile = _safe_profile_value(node.value)
            break
    else:
        raise ValueError(f"producer has no literal SETUP profile: {path}")
    if not isinstance(profile, dict):
        raise TypeError("producer SETUP must be a dict")
    for key in (
        "qualification_mode",
        "mtp",
        "max_lanes",
        "max_inflight",
        "max_context",
        "prefill_step",
        "cache_bytes",
    ):
        if key not in profile:
            raise ValueError(f"producer SETUP omits {key}")
    if profile["qualification_mode"] is not True or profile["mtp"] is not False:
        raise ValueError(
            "VLM qualification profile must select qualification mode and ordinary decode"
        )
    for key in (
        "max_lanes",
        "max_inflight",
        "max_context",
        "prefill_step",
        "cache_bytes",
    ):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"producer SETUP {key} must be a positive integer")
    if profile.get("execution_policy") is not None and not isinstance(
        profile["execution_policy"], dict
    ):
        raise ValueError("producer execution_policy must be a literal object or null")
    return profile


def server_argv(*, python, root, artifact, profile, host, port, policy_path):
    command = [
        str(python),
        "-m",
        "mlx2.server",
        "--model",
        str(Path(artifact).resolve()),
        "--host",
        host,
        "--port",
        str(port),
        "--qualification-mode",
        "--max-lanes",
        str(profile["max_lanes"]),
        "--max-inflight",
        str(profile["max_inflight"]),
        "--max-context",
        str(profile["max_context"]),
        "--prefill-step",
        str(profile["prefill_step"]),
        "--cache-bytes",
        str(profile["cache_bytes"]),
    ]
    if profile.get("route_selection_source") == "explicit_flag":
        command.append("--ordinary")
    policy = profile.get("execution_policy")
    if policy is not None:
        with policy_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(policy, sort_keys=True) + "\n")
        command.extend(("--execution-policy", str(policy_path)))
    return command


def _write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _unused_loopback_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_owned(command, *, cwd, env, timeout):
    """Run a child and reap it before propagating cancellation."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _validate_run_args(args):
    if type(args.generation) is not int or args.generation <= 0:
        raise ValueError("generation must be a positive integer")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError as exc:
        raise ValueError("qualification server host must be a loopback IP") from exc
    if not loopback:
        raise ValueError("qualification server host must be a loopback IP")
    for name in ("media_timeout", "server_start_timeout", "generic_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value > 14400:
            raise ValueError(f"{name} must be finite and between 0 and 14400 seconds")


def _raise_on_termination(_signum, _frame):
    raise KeyboardInterrupt("owned VLM qualification unit interrupted")


def _wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"VLM server exited before readiness ({process.returncode})"
            )
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"VLM server readiness timed out: {last_error}")


def run_unit(args):
    _validate_run_args(args)
    root = Path(args.root).resolve()
    artifact = Path(args.artifact).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not artifact.is_dir() or not (artifact / "config.json").is_file():
        raise FileNotFoundError(f"artifact is unavailable: {artifact}")
    if not Path(args.preflight_receipt).is_file():
        raise FileNotFoundError("full frozen-source preflight receipt is required")
    output.mkdir(parents=True, exist_ok=True)
    media_path = output / "media-companion.json"
    generic_path = output / "generic-serving.json"
    server_log_path = output / "server.log"
    final_path = output / "unit-report.json"
    for path in (media_path, generic_path, server_log_path, final_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite qualification output: {path}")

    profile = producer_profile(root, args.family)
    artifact_config = json.loads((artifact / "config.json").read_text())
    if artifact_config.get("model_type") != args.family:
        raise ValueError("artifact config model_type does not match family")
    python = Path(args.python).expanduser().resolve()
    dispatcher = root / "scripts" / "qualify_vlm_routes.py"
    media_command = [
        str(python),
        str(dispatcher),
        "--root",
        str(root),
        "--family",
        args.family,
        "--artifact",
        str(artifact),
        "--generation",
        str(args.generation),
        "--cpg-owner-lock",
        args.cpg_owner_lock,
        "--cpg-task",
        args.cpg_task,
        "--legacy-source-root",
        args.legacy_source_root,
    ]
    media_env = os.environ.copy()
    source_paths = [str(root / "src")]
    if FAMILIES[args.family]["revision"].startswith("8a5e704e"):
        source_paths.insert(0, str(Path(args.legacy_source_root).resolve()))
    media_env["PYTHONPATH"] = os.pathsep.join(source_paths)
    media_env["MLX2_VLM_8A5E_SOURCE_ROOT"] = str(
        Path(args.legacy_source_root).resolve()
    )
    media = _run_owned(
        media_command,
        cwd=root,
        env=media_env,
        timeout=args.media_timeout,
    )
    _write_once(media_path, media.stdout.encode())
    if media.returncode != 0:
        raise RuntimeError(
            f"media companion failed ({media.returncode}): {media.stderr[-3000:]}"
        )
    media_report = json.loads(media.stdout)
    if not isinstance(media_report, dict) or media_report.get("passed") is not True:
        raise RuntimeError("media companion did not emit a passing source-bound report")

    port = _unused_loopback_port()
    base_url = f"http://{args.host}:{port}"
    policy_path = output / "execution-policy.json"
    server_command = server_argv(
        python=python,
        root=root,
        artifact=artifact,
        profile=profile,
        host=args.host,
        port=port,
        policy_path=policy_path,
    )
    with server_log_path.open("xb") as server_log:
        server = subprocess.Popen(
            server_command,
            cwd=root,
            env=media_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_ready(server, base_url, args.server_start_timeout)
            generic_command = [
                str(python),
                str(root / "scripts" / "qualify_serving.py"),
                "--url",
                base_url,
                "--output",
                str(generic_path),
                "--timeout",
                str(args.generic_timeout),
                "--adapter-qualification",
                str(media_path),
                "--preflight-receipt",
                str(Path(args.preflight_receipt).resolve()),
            ]
            generic = _run_owned(
                generic_command,
                cwd=root,
                env=media_env,
                timeout=args.generic_timeout + 60,
            )
            if generic.returncode != 0:
                raise RuntimeError(
                    f"generic serving qualification failed ({generic.returncode}): "
                    f"{generic.stderr[-3000:]}"
                )
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
    generic_report = json.loads(generic_path.read_text())
    report = {
        "schema": "mlx2.vlm-qualification-unit.v1",
        "family": args.family,
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256(artifact),
        "producer_profile": profile,
        "media_report": str(media_path),
        "generic_report": str(generic_path),
        "preflight_receipt": str(Path(args.preflight_receipt).resolve()),
        "media_passed": media_report.get("passed") is True,
        "generic_passed": generic_report.get("passed") is True,
        "passed": media_report.get("passed") is True
        and generic_report.get("passed") is True,
    }
    _write_once(final_path, (json.dumps(report, indent=2) + "\n").encode())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--legacy-source-root", required=True)
    parser.add_argument("--cpg-owner-lock", required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--media-timeout", type=float, default=3600)
    parser.add_argument("--server-start-timeout", type=float, default=240)
    parser.add_argument("--generic-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    for handled in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(handled, _raise_on_termination)
    try:
        report = run_unit(args)
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
        KeyboardInterrupt,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema": "mlx2.vlm-qualification-unit.v1",
                    "family": args.family,
                    "passed": False,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
