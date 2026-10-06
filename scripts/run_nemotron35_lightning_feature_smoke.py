#!/usr/bin/env python3
"""Run bounded ordinary-route feature smoke for Nemotron 3.5 Lightning q8.

The caller must hold the CPG lease and both filesystem locks.  This checks a
single fresh ordinary server and records exact route bindings plus APCv2,
width parity, structured output, tools, thinking, cancellation, recovery, and
quiescence.  It is a component smoke, not qualification or performance
evidence.  Native MTP is deliberately a separate ownership/server arm.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import run_north_feature_smoke as common

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARTIFACT = "54d9ce39f91180c7ad8ebf5e417ea4abaa5aa9cc115a6d223b7063e073253504"
EXPECTED_PROFILE = "nemotron35-lightning-q8-apcv2-ordinary"
EXPECTED_LAYOUT = "nemotron35-lightning-hybrid-mamba-kv-v1"


def startup_binding_errors(status: dict) -> list[str]:
    errors = []
    if status.get("artifact") != EXPECTED_ARTIFACT:
        errors.append("artifact")
    if status.get("profile") != EXPECTED_PROFILE:
        errors.append("profile")
    if (status.get("apcv2") or {}).get("layout_name") != EXPECTED_LAYOUT:
        errors.append("cache_layout")
    if status.get("qualification") != "candidate":
        errors.append("qualification")
    if "mtp" not in (status.get("implemented_capabilities") or []):
        errors.append("mtp_implementation")
    if "mtp" in (status.get("capabilities") or []):
        errors.append("mtp_selected_on_ordinary_server")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18954)
    parser.add_argument("--load-timeout", type=float, default=900)
    args = parser.parse_args(argv)

    model = args.model.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    owner = common.validate_ownership(args.session, args.label)
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
        "schema": "mlx2.nemotron35-lightning-feature-campaign.v1",
        "status": "running",
        "semantics": "bounded component smoke; not qualification or performance evidence",
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model),
        "owner": owner,
        "route": "ordinary",
        "started_at": time.time(),
    }
    common.atomic_json(out / "campaign.json", campaign)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    smoke_command = [
        sys.executable,
        str(ROOT / "scripts/smoke_local_model.py"),
        "--model", str(model),
        "--out", str(out / "ordinary-smoke.json"),
        "--max-tokens", "16",
        "--i-own-gpu",
        "--cpg-lease", f"cpg:{args.session}:{args.label}",
    ]
    server_command = [
        sys.executable, "-u", "-m", "mlx2.server",
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ordinary",
        "--qualification-mode",
        "--max-context", "16384",
        "--max-lanes", "4",
        "--max-inflight", "4",
        "--cache-bytes", str(4 * 1024**3),
        "--cache-dir", str(out / "apcv2-cache"),
    ]
    server = None
    report = None
    try:
        with (out / "ordinary-smoke.log").open("w") as log:
            subprocess.run(
                smoke_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.load_timeout,
            )
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base = f"http://127.0.0.1:{args.port}"
            initial = common.wait_health(base, server, args.load_timeout)
            binding_errors = startup_binding_errors(initial)
            common.atomic_json(out / "server-startup.json", {
                "command": server_command,
                "pid": server.pid,
                "status": initial,
                "binding_errors": binding_errors,
            })
            if binding_errors:
                raise RuntimeError(
                    "Nemotron Lightning startup binding failed: "
                    + ", ".join(binding_errors)
                )
            report = common.run_features(
                base,
                initial,
                expert_gather_sort="auto",
                batch_row_exact_q4=False,
                schema="mlx2.nemotron35-lightning-feature-smoke.v1",
                marker="NEMOTRON_LIGHTNING_APC_READY",
                nonce_prefix="nemotron-lightning-parity",
            )
            report["startup_binding_errors"] = binding_errors
            report["route"] = "ordinary"
            common.atomic_json(out / "feature-smoke.json", report)
            campaign["status"] = "passed" if report["passed"] else "failed"
    except BaseException as error:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        common.stop_server(server)
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = (
            common.validate_ownership(args.session, args.label) == owner
        )
        common.atomic_json(out / "campaign.json", campaign)
    return 0 if report is not None and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
