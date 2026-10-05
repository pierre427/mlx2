"""Default-off, source-bound Qwen3 complete-request price cell.

The ordinary and paged arms use the same loaded model.  Each timed invocation
must create and retire its request state, perform admission, all 28 layers,
sampling/output, and terminal cleanup.  This is distinct from the forward-only
research cell in varlen_pack_price_bench.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import signal
import subprocess
import time
from pathlib import Path

from varlen_pack_price_bench import (
    CASES, CONTEXT_TOKENS, MAX_RESIDENT_BYTES, MAX_SECONDS, PREFILL_ROWS,
    REPEATS, ROOT, SCHEMA, _preflight, sha256, source_tree_sha256,
    verify_artifact_manifest,
)

SCOPE = "offline_request_probe"


def dry_run() -> dict:
    return {
        "schema": SCHEMA, "status": "planned", "gpu_executed": False,
        "measurement_scope": SCOPE, "context_tokens": list(CONTEXT_TOKENS),
        "mandatory_shapes": [list(shape) for shape in CASES],
        "prefill_rows": list(PREFILL_ROWS), "repeats": REPEATS,
        "planned_paired_requests": len(CASES) * len(PREFILL_ROWS) * REPEATS,
        "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES,
        "requires": ["exact source, artifact, wheel and kernel identity",
                     "owned GPU lease and both locks", "reviewed Qwen3 request driver",
                     "per-layer native terminal proof and paired ordinary timings"],
        "not_proven": ["GPU execution", "live sampler/cache handoff",
                       "MeasuredPackPrice", "qualification", "selection"],
    }


def _positive_ms(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def preflight_cpu(manifest: dict) -> dict:
    """Verify all source and artifact bytes without touching a GPU or lease."""
    from mlx2.runtime.paged_pack_price import _identity

    identity = _identity(manifest.get("identity"))
    if identity["host"] != platform.node():
        raise RuntimeError("host identity differs")
    hardware = json.loads(subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "-json"], text=True,
        timeout=10)).get("SPHardwareDataType", [])
    if [item.get("chip_type") for item in hardware] != [identity["hardware"]]:
        raise RuntimeError("hardware identity differs")
    if (identity["source_commit"] != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip() or
            identity["source_tree_sha256"] != source_tree_sha256()):
        raise RuntimeError("mlx2 source commit, cleanliness, or bytes differ")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"artifact", "mlx_wheel", "kernel"}:
        raise RuntimeError("exact artifact, wheel, and kernel paths required")
    for label, field in (("artifact", "artifact_sha256"),
                         ("mlx_wheel", "mlx_wheel_sha256"),
                         ("kernel", "kernel_sha256")):
        if sha256(Path(paths[label])) != identity[field]:
            raise RuntimeError(f"{label} bytes differ")
    artifact = verify_artifact_manifest(Path(paths["artifact"]))
    if not {"config.json", "model.safetensors", "tokenizer.json"} <= set(artifact["files"]):
        raise RuntimeError("Qwen3 artifact manifest lacks required model files")
    config = json.loads((Path(artifact["root"]) / "config.json").read_text())
    geometry = {"model_type": "qwen3", "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "tie_word_embeddings": True}
    if (any(config.get(key) != value for key, value in geometry.items()) or
            config.get("rope_scaling") is not None or config.get("num_experts", 0)):
        raise RuntimeError("artifact is not the reviewed dense Qwen3 geometry")
    source = Path(manifest["mlx_lm_source"]).resolve()
    revision, digest = manifest["mlx_lm_revision"], manifest["qwen3_source_sha256"]
    if (not source.is_dir() or type(revision) is not str or len(revision) != 40 or
            type(digest) is not str or len(digest) != 64 or
            subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=source, text=True).strip() != revision or
            subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=source).strip() or
            sha256(source / "mlx_lm/models/qwen3.py") != digest):
        raise RuntimeError("external Qwen3 model source differs")
    if (manifest.get("context_tokens") != list(CONTEXT_TOKENS) or
            type(manifest.get("max_resident_bytes")) is not int or
            not 0 < manifest["max_resident_bytes"] <= MAX_RESIDENT_BYTES or
            not isinstance(manifest.get("profile_id"), str) or not manifest["profile_id"]):
        raise RuntimeError("request context, memory, or profile identity differs")
    return {"schema": SCHEMA, "status": "preflight_passed", "gpu_executed": False,
            "measurement_scope": SCOPE, "identity": identity,
            "context_tokens": list(CONTEXT_TOKENS), "price_usable": False}


def validate_proof(proof: object, *, arm: str, expected_layers: int) -> dict:
    """Fail closed on simulated, partial, or unretired request work."""
    if type(proof) is not dict or proof.get("arm") != arm:
        raise RuntimeError(f"{arm} request proof missing")
    required = ("admitted", "cache_transaction_published", "sampled_output",
                "synchronized", "request_state_released", "queue_state_released")
    if any(proof.get(field) is not True for field in required):
        raise RuntimeError(f"{arm} request boundary proof missing")
    if (type(proof.get("model_layers")) is not int or
            proof["model_layers"] != expected_layers):
        raise RuntimeError(f"{arm} model depth differs")
    if (type(proof.get("peak_resident_bytes")) is not int or
            not 0 < proof["peak_resident_bytes"] <= MAX_RESIDENT_BYTES):
        raise RuntimeError(f"{arm} resident peak missing or above cap")
    if (type(proof.get("output_token_ids")) is not list or
            not proof["output_token_ids"] or
            any(type(token) is not int or token < 0 for token in proof["output_token_ids"])):
        raise RuntimeError(f"{arm} sampled output missing")
    if arm == "paged":
        if (type(proof.get("paged_read_calls")) is not int or
                type(proof.get("terminal_successes")) is not int or
                proof["paged_read_calls"] < expected_layers or
                proof["terminal_successes"] != proof["paged_read_calls"] or
                type(proof.get("pending_native_epochs")) is not int or
                proof["pending_native_epochs"] != 0 or
                type(proof.get("retained_pages")) is not int or
                proof["retained_pages"] != 0):
            raise RuntimeError("native per-layer read/terminal/cleanup proof missing")
    return proof


def _timeout(_signum, _frame):
    raise TimeoutError(f"complete-request cell exceeded {MAX_SECONDS}s")


def execute(manifest: dict, driver_spec: str) -> dict:
    """Measure paired complete requests; preflight precedes every GPU import."""
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAX_SECONDS)
    driver = None
    try:
        preflight_cpu(manifest)
        identity, owner = _preflight(manifest)
        if not isinstance(driver_spec, str) or ":" not in driver_spec:
            raise ValueError("driver must be module:factory")
        module, function = driver_spec.rsplit(":", 1)
        import mlx.core as mx
        if mx.__version__ != identity["mlx_wheel_version"]:
            raise RuntimeError("imported MLX version differs from pinned wheel")
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("explicit GPU device required")
        driver = getattr(importlib.import_module(module), function)(manifest)
        if getattr(driver, "model_layers", None) != 28:
            raise RuntimeError("reviewed dense Qwen3 depth required")
        cases = []
        for reserved in CASES:
            for prefill in PREFILL_ROWS:
                paged_ms, ordinary_ms = [], []
                reads = terminals = peak = 0
                for _ in range(REPEATS):
                    paired = []
                    for arm in ("ordinary", "paged"):
                        started = time.perf_counter_ns()
                        proof = validate_proof(driver.run_request(
                            arm, reserved, prefill, CONTEXT_TOKENS),
                            arm=arm, expected_layers=driver.model_layers)
                        elapsed = (time.perf_counter_ns() - started) / 1e6
                        if not _positive_ms(elapsed):
                            raise RuntimeError("invalid raw request timing")
                        paired.append((elapsed, proof))
                    if paired[0][1]["output_token_ids"] != paired[1][1]["output_token_ids"]:
                        raise RuntimeError("paired ordinary/paged sampled output differs")
                    ordinary_ms.append(paired[0][0])
                    paged_ms.append(paired[1][0])
                    reads += paired[1][1]["paged_read_calls"]
                    terminals += paired[1][1]["terminal_successes"]
                    peak = max(peak, *(result["peak_resident_bytes"] for _, result in paired))
                cases.append({
                    "reserved": [list(row) for row in reserved], "prefill_rows": prefill,
                    # Existing MeasuredPackPrice consumes this key as a conservative
                    # raw maximum; the scope explicitly says complete_request.
                    "forward_ms": paged_ms, "ordinary_request_ms": ordinary_ms,
                    "peak_resident_bytes": peak,
                    "kernel_engagement": {"paged_read_calls": reads,
                                          "terminal_successes": terminals},
                })
        evidence = {"schema": SCHEMA, "status": "measured", "gpu_executed": True,
                    "measurement_scope": SCOPE, "profile_id": manifest["profile_id"],
                    "identity": identity, "context_tokens": list(CONTEXT_TOKENS),
                    "gpuq_owner": owner, "hard_seconds": MAX_SECONDS,
                    "max_resident_bytes": MAX_RESIDENT_BYTES, "cases": cases,
                    "qualified": False, "selected": False}
        encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        evidence["offline_request_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
        return evidence
    finally:
        if driver is not None:
            driver.close()
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--preflight-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--driver", default="varlen_qwen3_request_driver:make_driver")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.execute_gpu and args.preflight_manifest:
        parser.error("choose GPU execution or CPU preflight")
    if args.execute_gpu:
        if not args.manifest or not args.receipt:
            parser.error("GPU mode requires --manifest and --receipt")
        try:
            result = execute(json.loads(args.manifest.read_text()), args.driver)
        except Exception as exc:
            result = {"schema": SCHEMA, "status": "failed", "gpu_executed": None,
                      "measurement_scope": SCOPE, "price_usable": False,
                      "error_type": type(exc).__name__, "error": str(exc)}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    elif args.preflight_manifest:
        result = preflight_cpu(json.loads(args.preflight_manifest.read_text()))
    else:
        result = dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
