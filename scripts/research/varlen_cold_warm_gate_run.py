"""One owned, 180-second cold63/warm64 calibration and loopback HTTP gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from varlen_http_cold_warm_price_gate import preflight
from varlen_pack_price_bench import _gpuq_owner

ROOT = Path(__file__).resolve().parents[2]
HARD_SECONDS = 180
MAX_BYTES = 24 * (1 << 30)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(artifact: Path, wheel: Path, kernel: Path, output: Path) -> dict:
    """Write separate exact cold and warm manifests from one clean source."""
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from varlen_pack_price_bench import verify_artifact_manifest

    if output.resolve().is_relative_to(ROOT):
        raise ValueError("frozen manifests must be outside the source tree")
    model = verify_artifact_manifest(artifact)
    paths = {"artifact": str(artifact.resolve()),
             "mlx_wheel": str(wheel.resolve()),
             "kernel": str(kernel.resolve())}
    identity = compute_live_price_identity(
        paths["artifact"], paths["mlx_wheel"], paths["kernel"],
        adapter_artifact_root=model["root"])
    common = {"identity": identity, "paths": paths,
              "max_resident_bytes": MAX_BYTES}
    cold = {**common, "profile_id": "qwen3-06b-fp16-cold63-" +
            identity["source_commit"][:8], "context_tokens": [63]}
    warm = {**common, "profile_id": "qwen3-06b-fp16-warm64-" +
            identity["source_commit"][:8], "context_tokens": [64],
            "request_shape": {"prompt_tokens": 64, "cached_tokens": 63,
                              "suffix_rows": 1, "output_tokens": 2,
                              "decode_rows_after_prefill": 1},
            "hard_seconds": HARD_SECONDS}
    output.mkdir(parents=True, exist_ok=False)
    (output / "cold-manifest.json").write_text(json.dumps(cold, indent=2) + "\n")
    (output / "warm-manifest.json").write_text(json.dumps(warm, indent=2) + "\n")
    return {"status": "frozen", "identity": identity,
            "cold_manifest_sha256": _sha(output / "cold-manifest.json"),
            "warm_manifest_sha256": _sha(output / "warm-manifest.json")}


def _child_rss_bytes(pid: int) -> int | None:
    """Sample the direct stage process RSS on macOS (ps reports KiB)."""
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=1)
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9]+", value):
        return None
    return int(value) * 1024


def _run_stage(label: str, command: list[str], output: Path, deadline: float) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("combined cold/warm HTTP gate exceeded 180 seconds")
    with (output / f"{label}.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True)
        sampled_peak = 0
        try:
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"{label} exceeded remaining combined time")
                rss = _child_rss_bytes(child.pid)
                if rss is None and child.poll() is None:
                    raise RuntimeError(f"{label} live RSS sample unavailable")
                if rss is not None:
                    sampled_peak = max(sampled_peak, rss)
                    if rss > MAX_BYTES:
                        raise MemoryError(
                            f"{label} sampled RSS {rss} exceeds {MAX_BYTES} bytes")
                try:
                    child.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = child.returncode
        except (TimeoutError, MemoryError, RuntimeError) as error:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait()
            failure = output / f"{label}.json"
            if not failure.exists():
                failure.write_text(json.dumps({"status": "failed", "stage": label,
                                               "error_type": type(error).__name__,
                                               "error": str(error),
                                               "sampled_peak_rss_bytes": sampled_peak},
                                              indent=2) + "\n")
            raise
    if code:
        raise RuntimeError(f"{label} failed with rc {code}; see preserved log")
    receipt = output / f"{label}.json"
    if not receipt.is_file():
        raise RuntimeError(f"{label} did not write a receipt")
    return {"label": label, "receipt_sha256": _sha(receipt),
            "log_sha256": _sha(output / f"{label}.log"),
            "sampled_peak_rss_bytes": sampled_peak}


def run(cold_manifest: Path, warm_manifest: Path, output: Path) -> dict:
    cold = json.loads(cold_manifest.read_text())
    warm = json.loads(warm_manifest.read_text())
    frozen = preflight(cold, warm)
    if output.resolve().is_relative_to(ROOT):
        raise ValueError("gate receipts must be outside the source tree")
    owner = _gpuq_owner()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "mlx2.varlen-cold-warm-http-combined.v1",
            "status": "running", "gpuq_owner": owner,
            "identity": frozen["identity"], "hard_seconds": HARD_SECONDS,
            "cold_manifest_sha256": _sha(cold_manifest),
            "warm_manifest_sha256": _sha(warm_manifest), "stages": []}
    (output / "run.json").write_text(json.dumps(plan, indent=2) + "\n")
    deadline = time.monotonic() + HARD_SECONDS
    env_python = sys.executable
    commands = (
        ("cold", [env_python, "scripts/research/varlen_live_request_price.py",
                   "--manifest", str(cold_manifest),
                   "--driver", "varlen_live_qwen3_request_driver:create",
                   "--execute-gpu", "--receipt", str(output / "cold.json")]),
        ("warm", [env_python, "scripts/research/varlen_warm_request_price.py",
                   "--manifest", str(warm_manifest),
                   "--execute-gpu", "--receipt", str(output / "warm.json")]),
        ("http", [env_python, "scripts/research/varlen_http_cold_warm_price_gate.py",
                   "--cold-manifest", str(cold_manifest),
                   "--warm-manifest", str(warm_manifest),
                   "--cold-price", str(output / "cold.json"),
                   "--warm-price", str(output / "warm.json"),
                   "--execute-gpu", "--receipt", str(output / "http.json")]),
    )
    try:
        for label, command in commands:
            plan["current_stage"] = label
            (output / "run.json").write_text(json.dumps(plan, indent=2) + "\n")
            plan["stages"].append(_run_stage(label, command, output, deadline))
        plan["status"] = "passed"
        plan["current_stage"] = None
        return plan
    except BaseException as error:
        plan.update(status="failed", error_type=type(error).__name__,
                    error=str(error))
        raise
    finally:
        (output / "run.json").write_text(json.dumps(plan, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-manifest", type=Path)
    parser.add_argument("--warm-manifest", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--kernel", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        if args.execute_gpu or not all((args.artifact, args.wheel,
                                        args.kernel, args.output)):
            parser.error("freeze requires artifact, wheel, kernel and output")
        result = freeze(args.artifact, args.wheel, args.kernel, args.output)
    elif args.execute_gpu:
        if not all((args.cold_manifest, args.warm_manifest)):
            parser.error("GPU gate requires both frozen manifests")
        if args.output is None:
            parser.error("GPU execution requires an outside-tree output directory")
        result = run(args.cold_manifest, args.warm_manifest, args.output)
    else:
        if not all((args.cold_manifest, args.warm_manifest)):
            parser.error("preflight requires both frozen manifests")
        result = preflight(json.loads(args.cold_manifest.read_text()),
                           json.loads(args.warm_manifest.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
