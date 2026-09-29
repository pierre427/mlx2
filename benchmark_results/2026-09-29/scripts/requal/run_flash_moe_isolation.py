#!/usr/bin/env python3
"""Isolate Flash-Next MoE load pressure with source-bound, owned server arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SERIES = ROOT / "qualification/runs/series-20260924"
sys.path.insert(0, str(SERIES))
import campaign_config as config  # noqa: E402
import experimental_job as experimental  # noqa: E402
import extra_common as xc  # noqa: E402
import run as owned  # noqa: E402
from queue_smoke import swapouts  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flash-next", "flash-next-uncensored"), required=True)
    parser.add_argument("--host-label", required=True)
    args = parser.parse_args()
    model = next(m for m in config.MODELS if m.name == args.model)
    out = HERE / args.host_label / model.name / "moe-isolation"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "qualification.json"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {
        "schema": "mlx2.requal.flash-moe-isolation.v1",
        "host": args.host_label, "model": model.name,
        "source_head": head,
        "source_tree": subprocess.check_output(
            ["git", "rev-parse", f"{head}:src"], cwd=ROOT, text=True).strip(),
        "artifact_config_sha256": digest(Path(model.path) / "config.json"),
        "harness_sha256": {"isolation": digest(Path(__file__)),
                           "experimental_job": digest(SERIES / "experimental_job.py")},
        "started_at": time.time(), "status": "running", "phases": [],
    }

    def save() -> None:
        receipt["updated_at"] = time.time()
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")

    base = experimental.FLASH_NEXT_UNFUSED
    arms = (
        ("gate_up_only", {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1"}, None),
        ("expert_only", {"MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto"},
         ("execution", "moe", "dispatches")),
        ("gate_up_and_expert", {"MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
                                "MLX_QWEN4_FUSED_EXPERT_KERNEL": "auto"},
         ("execution", "moe", "dispatches")),
    )
    before = swapouts()
    save()
    try:
        with ExitStack() as stack:
            receipt["locks"] = owned.lock_host(stack)
            try:
                owned.get_json(f"http://127.0.0.1:{config.PORT}/health", timeout=1)
            except (OSError, urllib.error.URLError):
                pass
            else:
                raise RuntimeError(f"port {config.PORT} already serves /health")
            xc.ensure_runtime_files()
            prior_kernels = experimental.KERNELS["flash-next"]
            original_start, original_stop = xc.Server.start, xc.Server.stop

            def start(server):
                count = swapouts()
                try:
                    return original_start(server)
                finally:
                    end = swapouts()
                    receipt["phases"].append({"arm": server.name, "phase": "load",
                                              "before": count, "after": end,
                                              "delta": end - count, "at": time.time()})
                    server._swap_at_ready = end
                    save()

            def stop(server):
                count = swapouts()
                status = None
                if server.alive():
                    try:
                        status = owned.get_json(f"http://127.0.0.1:{config.PORT}/v1/status", timeout=2)
                    except (OSError, urllib.error.URLError):
                        pass
                receipt["phases"].append({
                    "arm": server.name, "phase": "active",
                    "before": getattr(server, "_swap_at_ready", None), "after": count,
                    "delta": count - server._swap_at_ready if hasattr(server, "_swap_at_ready") else None,
                    "metal_active_bytes": (status or {}).get("metal_active_bytes"),
                    "metal_peak_bytes": (status or {}).get("metal_peak_bytes"),
                    "physical_footprint_bytes": (status or {}).get("process_physical_footprint_bytes"),
                    "host_memory_available_bytes": (status or {}).get("host_memory_available_bytes"),
                    "at": time.time(),
                })
                save()
                return original_stop(server)

            experimental.KERNELS["flash-next"] = (base, arms)
            xc.Server.start, xc.Server.stop = start, stop
            try:
                facts = xc.probe([model], {model.name: experimental.kernel_env_request(model)})["models"][model.name]
                rows, detail = experimental.run_kernels(model, out, facts)
            finally:
                xc.Server.start, xc.Server.stop = original_start, original_stop
                experimental.KERNELS["flash-next"] = prior_kernels
            receipt["rows"] = rows
            receipt["detail"] = detail
            receipt["status"] = "pass" if all(row["ok"] for row in rows) else "partial"
            if any(phase.get("delta", 0) > 0 for phase in receipt["phases"]
                   if phase["phase"] == "active"):
                receipt["status"] = "contaminated"
    except BaseException as error:
        receipt["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        receipt["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, KeyboardInterrupt):
            raise
    finally:
        after = swapouts()
        receipt["swapouts"] = {"before": before, "after": after,
                               "delta": after - before if after is not None and before is not None else None}
        receipt["finished_at"] = time.time()
        save()
    print(f"{model.name} MoE isolation {receipt['status']} {target}", flush=True)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
