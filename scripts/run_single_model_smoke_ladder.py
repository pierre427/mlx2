#!/usr/bin/env python3
"""Run one ordinary smoke and one thermally controlled context ladder.

The caller must hold both shared GPU locks and the CPG lease.  This runner
loads only one model at a time, always terminates its loopback server, and
refuses to overwrite evidence. ``--runs 1`` is intentionally exploratory.
The explicit ``--qualification-ladder`` mode instead requires three measured
runs per cell; it provides ladder evidence but does not by itself issue a
qualification receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
LADDER = ROOT / "benchmark_results/2026-09-29/scripts/series/thermal_ladder.py"


def read_owner(path: Path) -> dict:
    owner = path / "owner.json"
    if not path.is_dir() or not owner.is_file():
        raise RuntimeError(f"GPU lock is not owned: {path}")
    return json.loads(owner.read_text())


def validate_ownership(session: str, label: str) -> dict:
    owners = [read_owner(path) for path in LOCKS]
    if owners[0] != owners[1]:
        raise RuntimeError("shared GPU lock owner receipts differ")
    owner = owners[0]
    if owner.get("session") != session or owner.get("label") != label:
        raise RuntimeError(
            f"GPU owner mismatch: expected {session}/{label}, got "
            f"{owner.get('session')}/{owner.get('label')}"
        )
    return owner


def wait_health(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with rc={process.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as response:
                health = json.load(response)
            if health.get("status") == "ok":
                with urllib.request.urlopen(base + "/v1/status", timeout=10) as response:
                    return json.load(response)
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout:g}s")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_startup_status(
    status: dict,
    *,
    expected_artifact: str | None,
    expected_profile: str | None,
    expected_cache_layout: str | None,
) -> None:
    """Bind a campaign to the requested ordinary artifact/profile/layout."""
    expected = {
        "artifact": (status.get("artifact"), expected_artifact),
        "profile": (status.get("profile"), expected_profile),
        "cache layout": ((status.get("execution") or {}).get("layout"), expected_cache_layout),
    }
    for label, (observed, wanted) in expected.items():
        if wanted is not None and observed != wanted:
            raise RuntimeError(f"startup {label} mismatch: expected {wanted!r}, got {observed!r}")
    settings = status.get("settings") or {}
    if settings.get("route") != "ordinary" or settings.get("mtp") is not False:
        raise RuntimeError(f"startup did not select ordinary non-MTP route: {settings!r}")
    if status.get("qualification") != "candidate":
        raise RuntimeError(
            "qualification-mode startup must publish qualification='candidate'"
        )


def ladder_semantics(*, qualification_ladder: bool, runs: int) -> str:
    if qualification_ladder:
        if runs != 3:
            raise ValueError("qualification ladder requires exactly three runs per cell")
        return (
            "three thermally admitted runs per cell; qualification-ladder evidence "
            "requiring a separate qualification verdict"
        )
    if runs != 1:
        raise ValueError("exploratory ladder requires exactly one run per cell")
    return "one exploratory run per cell; not thermally replicated qualification"


def server_capacity(*, wide: int, max_inflight: int | None) -> tuple[int, int]:
    lanes = max(1, wide)
    inflight = lanes if max_inflight is None else max_inflight
    if inflight < lanes:
        raise ValueError("--max-inflight must be at least --wide")
    return lanes, inflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--smoke-max-tokens", type=int, default=16)
    parser.add_argument("--wide", type=int, default=4)
    parser.add_argument("--wide-max-context", type=int, default=32768)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--cache-gib", type=int, default=4)
    parser.add_argument(
        "--max-inflight",
        type=int,
        help="server request capacity; defaults to --wide",
    )
    parser.add_argument(
        "--memory-only-cache",
        action="store_true",
        help="omit --cache-dir so the isolated APCv2 cache remains memory-only",
    )
    parser.add_argument("--expected-artifact")
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-cache-layout")
    parser.add_argument(
        "--qualification-ladder",
        action="store_true",
        help="require three measured runs per cell and label the evidence accordingly",
    )
    args = parser.parse_args(argv)
    try:
        semantics = ladder_semantics(
            qualification_ladder=args.qualification_ladder,
            runs=args.runs,
        )
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.max_tokens <= 128:
        parser.error("--max-tokens must be 1..128")
    if not 1 <= args.smoke_max_tokens <= 16:
        parser.error("--smoke-max-tokens must be 1..16")
    if args.max_context < 1024 or not 1024 <= args.max_length <= args.max_context:
        parser.error("context bounds must admit at least the 1K rung")
    if args.cache_gib <= 0:
        parser.error("--cache-gib must be positive")
    try:
        max_lanes, max_inflight = server_capacity(
            wide=args.wide,
            max_inflight=args.max_inflight,
        )
    except ValueError as error:
        parser.error(str(error))

    model = args.model.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    owner = validate_ownership(args.session, args.label)
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is already occupied")

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip())
    campaign = {
        "schema": "mlx2.single-model-smoke-ladder.v1",
        "status": "running",
        "semantics": semantics,
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model),
        "model_name": args.model_name,
        "max_context": args.max_context,
        "max_length": args.max_length,
        "runs_per_cell": args.runs,
        "qualification_ladder": args.qualification_ladder,
        "smoke_max_tokens": args.smoke_max_tokens,
        "cache_bytes": args.cache_gib << 30,
        "max_lanes": max_lanes,
        "max_inflight": max_inflight,
        "memory_only_cache": args.memory_only_cache,
        "owner": owner,
        "started_at": time.time(),
    }
    atomic_json(out / "campaign.json", campaign)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    smoke_command = [
        sys.executable, str(ROOT / "scripts/smoke_local_model.py"),
        "--model", str(model), "--out", str(out / "ordinary-smoke.json"),
        "--max-tokens", str(args.smoke_max_tokens), "--i-own-gpu",
        "--cpg-lease", f"cpg:{args.session}:{args.label}",
    ]
    server_command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", str(model),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--ordinary", "--qualification-mode", "--max-context", str(args.max_context),
        "--max-lanes", str(max_lanes), "--max-inflight", str(max_inflight),
        "--cache-bytes", str(args.cache_gib << 30),
    ]
    if not args.memory_only_cache:
        server_command += ["--cache-dir", str(out / "apcv2-cache")]
    base = f"http://127.0.0.1:{args.port}"
    server = None
    rc = 1
    try:
        with (out / "ordinary-smoke.log").open("w") as log:
            subprocess.run(smoke_command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=args.load_timeout)
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(server_command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            status = wait_health(base, server, args.load_timeout)
            validate_startup_status(
                status,
                expected_artifact=args.expected_artifact,
                expected_profile=args.expected_profile,
                expected_cache_layout=args.expected_cache_layout,
            )
            atomic_json(out / "server-startup.json", {
                "command": server_command,
                "pid": server.pid,
                "status": status,
            })
            ladder_command = [
                sys.executable, str(LADDER), "--url", base,
                "--output", str(out / "ladder.json"), "--model", args.model_name,
                "--model-id", model.name, "--route", "ordinary",
                "--max-context", str(args.max_context), "--max-length", str(args.max_length),
                "--server-pid", str(server.pid), "--runs", str(args.runs),
                "--max-tokens", str(args.max_tokens), "--wide", str(args.wide),
                "--wide-max-context", str(args.wide_max_context),
            ]
            with (out / "ladder.log").open("w") as ladder_log:
                completed = subprocess.run(
                    ladder_command,
                    cwd=ROOT,
                    env=env,
                    stdout=ladder_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            rc = completed.returncode
            campaign["server_command"] = server_command
            campaign["ladder_command"] = ladder_command
            campaign["ladder_returncode"] = rc
            campaign["status"] = "passed" if rc == 0 else "failed"
    except BaseException as error:  # preserve evidence and cleanup on every exit
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_server(server)
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = (
            validate_ownership(args.session, args.label) == owner
        )
        atomic_json(out / "campaign.json", campaign)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
