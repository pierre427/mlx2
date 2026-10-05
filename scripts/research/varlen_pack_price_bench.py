"""Dry-by-default, bounded complete-forward price cell for paged packs.

The GPU driver is supplied by a future native Qwen3 integration. It must run
the whole model forward, synchronize terminal work, and report native paged
read engagement. A read-kernel microbenchmark cannot supply this price.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mlx2.paged-pack-price.v1"
MAX_SECONDS = 180
MAX_RESIDENT_BYTES = 24 * 1024**3
REPEATS = 3
CONTEXT_TOKENS = (63, 65, 129)
# q=1/3/17 are mandatory, including decode and verify. The same skewed
# contexts recur with bounded prefill choices; no cost is extrapolated.
CASES = (
    (("decode", 1),), (("verify", 3),), (("verify", 17),),
    (("decode", 1), ("verify", 3)),
)
PREFILL_ROWS = (0, 1, 3, 17)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifact_manifest(path: Path) -> dict:
    """Bind each model file byte to the reviewed artifact manifest."""
    data = json.loads(path.read_text())
    root = Path(data["root"]).resolve()
    files = data["files"]
    if not root.is_dir() or not isinstance(files, dict) or not files:
        raise RuntimeError("artifact manifest needs a model root and file hashes")
    for name, digest in files.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in "0123456789abcdef" for c in digest)):
            raise RuntimeError("malformed artifact file hash")
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or sha256(file) != digest:
            raise RuntimeError(f"artifact file bytes differ: {name}")
    return data


def source_tree_sha256() -> str:
    """Hash the tracked source snapshot; dirty tracked or untracked code fails."""
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    h = hashlib.sha256()
    for raw in sorted(filter(None, files)):
        path = ROOT / os.fsdecode(raw)
        if path.is_file():
            h.update(raw + b"\0")
            h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def dry_run() -> dict:
    return {"schema": SCHEMA, "status": "planned", "gpu_executed": False,
            "measurement_scope": "complete_model_forward",
            "context_tokens": list(CONTEXT_TOKENS), "mandatory_shapes": [list(s) for s in CASES],
            "prefill_rows": list(PREFILL_ROWS), "repeats": REPEATS,
            "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES,
            "planned_forwards": len(CASES) * len(PREFILL_ROWS) * REPEATS,
            "requires": ["one short gpuq lease with both locks", "exact identity manifest",
                         "complete model-forward driver with native engagement counters"],
            "not_proven": ["GPU execution", "price", "model serving", "qualification"]}


def _gpuq_owner() -> dict:
    session = os.environ.get("GPUQ_SESSION")
    lease = os.environ.get("GPUQ_LEASE")
    if not session or not lease:
        raise RuntimeError("GPUQ_SESSION and GPUQ_LEASE are required")
    owners = []
    for lock in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        if not lock.is_dir() or lock.is_symlink():
            raise RuntimeError(f"GPU lock is not an owned directory: {lock}")
        owner = json.loads((lock / "owner.json").read_text())
        if (owner.get("session") != session or owner.get("lease_id") != lease or
                type(owner.get("pid")) is not int or owner["pid"] < 1):
            raise RuntimeError(f"GPU lock ownership differs: {lock}")
        os.kill(owner["pid"], 0)
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("GPU lock owner receipts disagree")
    return owners[0]


def _preflight(manifest: dict) -> tuple[dict, dict]:
    from mlx2.runtime.paged_pack_price import _identity

    identity = _identity(manifest.get("identity"))
    if identity["host"] != platform.node():
        raise RuntimeError("host identity differs")
    hardware = subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "-json"], text=True,
        timeout=10)
    hardware_items = json.loads(hardware).get("SPHardwareDataType", [])
    chips = [item.get("chip_type") for item in hardware_items]
    if chips != [identity["hardware"]]:
        raise RuntimeError("hardware identity differs")
    if identity["source_commit"] != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip():
        raise RuntimeError("source commit differs")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("dirty source tree cannot be priced")
    if identity["source_tree_sha256"] != source_tree_sha256():
        raise RuntimeError("source tree digest differs")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"artifact", "mlx_wheel", "kernel"}:
        raise RuntimeError("exact artifact, wheel, and kernel paths required")
    for label, field in (("artifact", "artifact_sha256"), ("mlx_wheel", "mlx_wheel_sha256"),
                         ("kernel", "kernel_sha256")):
        if sha256(Path(paths[label])) != identity[field]:
            raise RuntimeError(f"{label} bytes differ")
    verify_artifact_manifest(Path(paths["artifact"]))
    if manifest.get("context_tokens") != list(CONTEXT_TOKENS):
        raise RuntimeError("context lengths differ")
    if (type(manifest.get("max_resident_bytes")) is not int or
            manifest["max_resident_bytes"] < 1 or
            manifest["max_resident_bytes"] > MAX_RESIDENT_BYTES):
        raise RuntimeError("resident memory estimate exceeds cell cap")
    return identity, _gpuq_owner()


def _timeout(_signum, _frame):
    raise TimeoutError(f"complete-forward cell exceeded {MAX_SECONDS}s")


def execute(manifest: dict, driver_spec: str) -> dict:
    """Driver factory returns run(reserved, prefill_rows, contexts) result.

    Each result must contain synchronized=True, peak_resident_bytes and exact
    native read/terminal counters. No fake or estimated timing is accepted.
    """
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAX_SECONDS)
    driver = None
    try:
        identity, owner = _preflight(manifest)
        if ":" not in driver_spec:
            raise ValueError("driver must be module:factory")
        module, function = driver_spec.rsplit(":", 1)
        import mlx.core as mx
        if mx.__version__ != identity["mlx_wheel_version"]:
            raise RuntimeError("imported MLX version differs from pinned wheel")
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("explicit GPU device required")
        factory = getattr(importlib.import_module(module), function)
        driver = factory(manifest)
        cases = []
        for reserved in CASES:
            for prefill in PREFILL_ROWS:
                raw = []
                reads = terminals = 0
                peak = 0
                for _ in range(REPEATS):
                    start = time.perf_counter_ns()
                    proof = driver.run_forward(reserved, prefill, CONTEXT_TOKENS)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if (not isinstance(proof, dict) or proof.get("synchronized") is not True or
                            type(proof.get("paged_read_calls")) is not int or
                            proof["paged_read_calls"] < 1 or
                            type(proof.get("terminal_successes")) is not int or
                            proof["terminal_successes"] < 1 or
                            type(proof.get("peak_resident_bytes")) is not int or
                            proof["peak_resident_bytes"] > MAX_RESIDENT_BYTES or
                            proof["peak_resident_bytes"] < 1):
                        raise RuntimeError("complete-forward terminal or memory proof missing")
                    raw.append(elapsed)
                    reads += proof["paged_read_calls"]
                    terminals += proof["terminal_successes"]
                    peak = max(peak, proof["peak_resident_bytes"])
                cases.append({"reserved": [list(x) for x in reserved],
                              "prefill_rows": prefill, "forward_ms": raw,
                              "peak_resident_bytes": peak,
                              "kernel_engagement": {"paged_read_calls": reads,
                                                    "terminal_successes": terminals}})
        return {"schema": SCHEMA, "status": "measured", "gpu_executed": True,
                "measurement_scope": "complete_model_forward",
                "profile_id": manifest["profile_id"], "identity": identity,
                "context_tokens": list(CONTEXT_TOKENS), "gpuq_owner": owner,
                "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES,
                "cases": cases, "qualified": False, "selected": False}
    finally:
        if driver is not None:
            driver.close()
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--driver")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.execute_gpu:
        if not args.manifest or not args.driver or not args.receipt:
            parser.error("GPU mode requires --manifest, --driver, and --receipt")
        try:
            result = execute(json.loads(args.manifest.read_text()), args.driver)
        except Exception as exc:
            result = {"schema": SCHEMA, "status": "failed", "gpu_executed": None,
                      "measurement_scope": "complete_model_forward",
                      "error_type": type(exc).__name__, "error": str(exc),
                      "price_usable": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    else:
        result = dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
