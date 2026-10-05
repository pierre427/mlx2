"""Default-off source-bound research calibration for live Qwen3 q=1 requests.

Only context (63,) and one decode row are admitted by this first route.  The
driver must enter BatchGenerator's native cache/sampler/response lifecycle;
offline or model-forward-only timings are refused. A cold request total and
its second q=1 step are separate spans; neither is a scheduler pack price.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from varlen_pack_price_bench import (
    MAX_RESIDENT_BYTES, MAX_SECONDS, ROOT, SCHEMA, _gpuq_owner, sha256,
    source_tree_sha256, verify_artifact_manifest,
)

CONTEXT_TOKENS = (63,)
REPEATS = 3
SCOPE = "research_live_request_calibration"


def dry_run() -> dict:
    return {"schema": SCHEMA, "status": "planned", "gpu_executed": False,
            "measurement_scope": SCOPE, "context_tokens": [63],
            "request_shape": {"prompt_tokens": 63, "output_tokens": 2,
                              "decode_rows_after_prefill": 1},
            "paired_repeats": REPEATS, "hard_seconds": MAX_SECONDS,
            "max_resident_bytes": MAX_RESIDENT_BYTES,
            "requires": ["observed native BatchGenerator route and response",
                         "ordinary paired request", "one source-bound owned GPU lease"],
            "not_proven": ["GPU execution", "measured price", "qualification", "selection"]}


def preflight_cpu(manifest: dict) -> dict:
    """Check exact source/model/wheel/kernel before touching GPU ownership."""
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
        raise RuntimeError("Qwen3 model files missing from artifact manifest")
    config = json.loads((Path(artifact["root"]) / "config.json").read_text())
    expected = {"model_type": "qwen3", "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "tie_word_embeddings": True}
    if (any(config.get(key) != value for key, value in expected.items()) or
            config.get("rope_scaling") is not None or config.get("num_experts", 0)):
        raise RuntimeError("unsupported Qwen3 model geometry")
    if (manifest.get("context_tokens") != [63] or
            type(manifest.get("max_resident_bytes")) is not int or
            not 0 < manifest["max_resident_bytes"] <= MAX_RESIDENT_BYTES or
            not isinstance(manifest.get("profile_id"), str) or not manifest["profile_id"]):
        raise RuntimeError("live request context, memory, or profile differs")
    return {"schema": SCHEMA, "status": "preflight_passed", "gpu_executed": False,
            "measurement_scope": SCOPE, "identity": identity,
            "context_tokens": [63], "price_usable": False}


def validate_proof(proof: object, arm: str) -> dict:
    if type(proof) is not dict or proof.get("arm") != arm:
        raise RuntimeError(f"{arm} request proof missing")
    boundaries = ("request_inserted", "admitted", "cache_transaction_published",
                  "response_emitted", "sampled_output", "synchronized",
                  "request_removed", "request_state_released")
    if any(proof.get(key) is not True for key in boundaries):
        raise RuntimeError(f"{arm} request lifecycle proof missing")
    if (type(proof.get("output_token_ids")) is not list or
            len(proof["output_token_ids"]) != 2 or
            any(type(token) is not int or token < 0 for token in proof["output_token_ids"]) or
            type(proof.get("output_token_id")) is not int or
            proof["output_token_id"] < 0 or
            proof["output_token_id"] != proof["output_token_ids"][-1] or
            type(proof.get("model_layers")) is not int or
            proof["model_layers"] != 28 or
            type(proof.get("peak_resident_bytes")) is not int or
            not 0 < proof["peak_resident_bytes"] <= MAX_RESIDENT_BYTES):
        raise RuntimeError(f"{arm} model/output/memory proof missing")
    if (type(proof.get("q1_step_ms")) not in (int, float) or
            not math.isfinite(proof["q1_step_ms"]) or proof["q1_step_ms"] <= 0):
        raise RuntimeError(f"{arm} q=1 scheduler/response span missing")
    route = proof.get("route_receipt")
    if arm == "paged":
        first = proof.get("first_response_receipt")
        second = proof.get("second_response_receipt")
        if (type(route) is not dict or route.get("route") != "native_qwen3_paged" or
                route.get("research_executed") is not True or
                route.get("serving_selected") is not False or
                route.get("selected") is not False or
                route.get("observed_used") is not False or
                type(proof.get("ordinary_model_forward_calls")) is not int or
                proof["ordinary_model_forward_calls"] != 0 or
                type(proof.get("paged_read_calls")) is not int or
                proof["paged_read_calls"] < 56 or
                type(first) is not dict or type(second) is not dict or
                first.get("research_executed") is not True or
                second.get("research_executed") is not True or
                type(first.get("native_read_calls")) is not int or
                first["native_read_calls"] < 28 or
                second.get("native_read_calls") != proof["paged_read_calls"] or
                second.get("terminal_successes") != proof["terminal_successes"] or
                proof.get("prefill_read_calls") != first["native_read_calls"] or
                proof["paged_read_calls"] - first["native_read_calls"] !=
                    proof.get("decode_read_calls") or
                type(proof.get("decode_read_calls")) is not int or
                proof["decode_read_calls"] < 28 or
                type(proof.get("terminal_successes")) is not int or
                proof["terminal_successes"] != proof["paged_read_calls"] or
                type(proof.get("pending_native_epochs")) is not int or
                proof["pending_native_epochs"] != 0 or
                type(proof.get("retained_pages")) is not int or
                proof["retained_pages"] != 0):
            raise RuntimeError("live native route/engagement evidence missing")
    elif (type(proof.get("ordinary_model_forward_calls")) is not int or
          proof["ordinary_model_forward_calls"] < 1 or
          type(route) is not dict or route.get("route") != "ordinary"):
        raise RuntimeError("ordinary live reference proof missing")
    return proof


def _timeout(_signum, _frame):
    raise TimeoutError(f"live request price exceeded {MAX_SECONDS}s")


def execute(manifest: dict, driver_spec: str) -> dict:
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MAX_SECONDS)
    driver = None
    try:
        preflight_cpu(manifest)
        if "q1_tile_arm" in manifest and (
                type(manifest["q1_tile_arm"]) is not bool or
                (os.environ.get("MLX2_PAGED_Q1_SIMD_TILE") == "1") !=
                manifest["q1_tile_arm"]):
            raise RuntimeError("Q1 SIMD tile environment differs from manifest")
        owner = _gpuq_owner()
        identity = manifest["identity"]
        if not isinstance(driver_spec, str) or ":" not in driver_spec:
            raise ValueError("driver must be module:factory")
        import mlx.core as mx
        if mx.__version__ != identity["mlx_wheel_version"]:
            raise RuntimeError("imported MLX version differs")
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("explicit GPU device required")
        import _paged_kv_native
        from mlx2.runtime.paged_price_identity import compute_live_price_identity
        artifact = verify_artifact_manifest(Path(manifest["paths"]["artifact"]))
        live_identity = compute_live_price_identity(
            Path(manifest["paths"]["artifact"]),
            Path(manifest["paths"]["mlx_wheel"]),
            Path(_paged_kv_native.__file__).resolve(),
            adapter_artifact_root=Path(artifact["root"]),
        )
        if (Path(_paged_kv_native.__file__).resolve() !=
                Path(manifest["paths"]["kernel"]).resolve() or
                live_identity != identity):
            raise RuntimeError("loaded native/model/wheel/source identity differs")
        module, function = driver_spec.rsplit(":", 1)
        driver = getattr(importlib.import_module(module), function)(manifest)
        paged_ms, ordinary_ms, paged_q1_ms, ordinary_q1_ms, request_proofs = [], [], [], [], []
        reads = terminals = peak = 0
        for _ in range(REPEATS):
            pair = []
            for arm in ("ordinary", "paged"):
                begin = time.perf_counter_ns()
                proof = validate_proof(driver.run_request(arm, CONTEXT_TOKENS), arm)
                if arm == "paged" and "q1_tile_arm" in manifest:
                    expected_tiles = 28 if manifest["q1_tile_arm"] else 0
                    if proof.get("q1_tile_dispatches") != expected_tiles:
                        raise RuntimeError("Q1 tile dispatch proof differs")
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise RuntimeError("invalid raw live request timing")
                pair.append((elapsed, proof))
            if pair[0][1]["output_token_ids"] != pair[1][1]["output_token_ids"]:
                raise RuntimeError("paired live request outputs differ")
            ordinary_ms.append(pair[0][0])
            paged_ms.append(pair[1][0])
            ordinary_q1_ms.append(pair[0][1]["q1_step_ms"])
            paged_q1_ms.append(pair[1][1]["q1_step_ms"])
            request_proofs.append({"ordinary": pair[0][1], "paged": pair[1][1]})
            reads += pair[1][1]["paged_read_calls"]
            terminals += pair[1][1]["terminal_successes"]
            peak = max(peak, *(p["peak_resident_bytes"] for _, p in pair))
        evidence = {"schema": SCHEMA, "status": "calibrated", "gpu_executed": True,
                    "measurement_scope": SCOPE, "profile_id": manifest["profile_id"],
                    "measured_route": "native_qwen3_paged",
                    "identity": identity, "context_tokens": [63], "gpuq_owner": owner,
                    "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES,
                    "request_shape": {"prompt_tokens": 63, "output_tokens": 2,
                                      "decode_rows_after_prefill": 1},
                    "cases": [{"cold_native_request_ms": paged_ms,
                               "cold_ordinary_request_ms": ordinary_ms,
                               "native_q1_step_ms": paged_q1_ms,
                               "ordinary_q1_step_ms": ordinary_q1_ms,
                               "request_proofs": request_proofs,
                               "peak_resident_bytes": peak,
                               "kernel_engagement": {"paged_read_calls": reads,
                                                     "terminal_successes": terminals}}],
                    "qualified": False, "selected": False, "price_usable": False,
                    "serving_selected": False, "research_executed": True}
        if "q1_tile_arm" in manifest:
            evidence["q1_tile_arm"] = manifest["q1_tile_arm"]
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        evidence["research_request_evidence_sha256"] = hashlib.sha256(raw).hexdigest()
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
    parser.add_argument("--driver")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.execute_gpu and args.preflight_manifest:
        parser.error("choose GPU execution or CPU preflight")
    if args.execute_gpu:
        if not args.manifest or not args.driver or not args.receipt:
            parser.error("GPU mode requires --manifest, --driver, and --receipt")
        manifest_data = json.loads(args.manifest.read_text())
        try:
            result = execute(manifest_data, args.driver)
        except Exception as exc:
            result = {"schema": SCHEMA, "status": "failed", "gpu_executed": None,
                      "measurement_scope": SCOPE, "price_usable": False,
                      "identity": manifest_data.get("identity"),
                      "profile_id": manifest_data.get("profile_id"),
                      "context_tokens": [63], "hard_seconds": MAX_SECONDS,
                      "max_resident_bytes": MAX_RESIDENT_BYTES,
                      "driver": args.driver,
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
