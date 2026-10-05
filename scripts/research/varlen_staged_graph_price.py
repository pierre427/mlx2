"""Source-bound paired live B2 research screen; price admission stays false."""

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
    MAX_RESIDENT_BYTES,
    MAX_SECONDS,
    ROOT,
    SCHEMA,
    _gpuq_owner,
    sha256,
    source_tree_sha256,
    verify_artifact_manifest,
)

CONTEXTS = (63, 65)
ALIGNED_CONTROL = (63, 63)
REPEATS = 3
SCOPE = "research_live_scheduler_b2_step"
STEP = {"scope": "live_scheduler_pack_step",
        "reserved": [["decode", 1], ["decode", 1]], "prefill_rows": 0,
        "context_tokens": [63, 65], "active_lane_count": 2,
        "generator_step": True, "sampled_response": True, "synchronized": True}


def contexts_for(manifest: dict) -> tuple[int, int]:
    contexts = manifest.get("context_tokens")
    if contexts not in (list(CONTEXTS), list(ALIGNED_CONTROL)):
        raise RuntimeError("B2 contexts must be 63/65 or aligned ready-step 63/63 control")
    return tuple(contexts)


def step_for(contexts: tuple[int, int]) -> dict:
    return {**STEP, "context_tokens": list(contexts)}
SAMPLER = {"mode": "default_argmax", "custom_samplers": False,
           "logits_processors": False}


class PairParityFailure(RuntimeError):
    def __init__(self, pair: dict, case_index: int):
        super().__init__("paired B2 output tokens differ")
        self.pair = pair
        self.case_index = case_index


def dry_run() -> dict:
    return {"schema": SCHEMA, "status": "planned", "gpu_executed": False,
            "measurement_scope": SCOPE, "context_tokens": list(CONTEXTS),
            "ready_step": STEP, "paired_repeats": REPEATS,
            "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES,
            "requires": ["one owned GPU lease", "exact source/model/wheel/native manifest",
                         "two real generator responses from one two-span native read per layer"],
            "price_usable": False, "qualified": False, "selected": False}


def preflight_cpu(manifest: dict, *, context_validator=contexts_for) -> dict:
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
        raise RuntimeError("source commit, cleanliness, or bytes differ")
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
    contexts = context_validator(manifest)
    if (
            type(manifest.get("diagnostic_mode", False)) is not bool or
            type(manifest.get("max_resident_bytes")) is not int or
            not 0 < manifest["max_resident_bytes"] <= MAX_RESIDENT_BYTES or
            not isinstance(manifest.get("profile_id"), str) or not manifest["profile_id"]):
        raise RuntimeError("B2 context, memory, or profile differs")
    return {"schema": SCHEMA, "status": "preflight_passed",
            "gpu_executed": False, "measurement_scope": SCOPE,
            "identity": identity, "context_tokens": list(contexts),
            "price_usable": False}


def validate_proof(proof: object, arm: str, contexts: tuple[int, int]) -> dict:
    if type(proof) is not dict or proof.get("arm") != arm:
        raise RuntimeError(f"{arm} B2 request proof missing")
    required = ("request_inserted", "admitted", "cache_transaction_published",
                "response_emitted", "sampled_output", "synchronized",
                "request_removed", "request_state_released")
    if any(proof.get(field) is not True for field in required):
        raise RuntimeError(f"{arm} complete-request lifecycle proof missing")
    timing = proof.get("ready_pack_step_ms")
    outputs = proof.get("output_token_ids")
    if (type(timing) not in (int, float) or not math.isfinite(timing) or timing <= 0 or
            proof.get("scheduler_step") != step_for(contexts) or
            proof.get("sampler_config") != SAMPLER or
            type(outputs) is not list or len(outputs) != 2 or
            any(type(row) is not list or len(row) != 2 or
                any(type(token) is not int or token < 0 for token in row)
                for row in outputs) or
            proof.get("ordered_context_tokens") != list(contexts) or
            proof.get("model_layers") != 28 or
            type(proof.get("peak_resident_bytes")) is not int or
            not 0 < proof["peak_resident_bytes"] <= MAX_RESIDENT_BYTES):
        raise RuntimeError(f"{arm} exact ready-step/output proof missing")
    route = proof.get("route_receipt")
    if arm == "paged":
        receipts = proof.get("response_receipts")
        if (type(route) is not dict or
                route.get("route") != "native_qwen3_paged_graph_research" or
                route.get("research_executed") is not True or
                route.get("selected") is not False or
                route.get("observed_used") is not False or
                route.get("serving_selected") is not False or
                type(receipts) is not list or len(receipts) != 2 or
                proof.get("response_execution_widths") != [[1, 2], [1, 2]] or
                any(type(pair) is not list or len(pair) != 2 or
                    pair[1].get("packed_lanes") != 2 or
                    pair[1].get("native_span_counts") != [2] * 28 or
                    pair[1].get("published_layer_offsets") !=
                    [contexts[index] + 1] * 28 or
                    pair[1].get("native_read_delta") != 28 or
                    pair[1].get("terminal_success_delta") != 28
                    for index, pair in enumerate(receipts)) or
                proof.get("ordinary_model_forward_calls") != 0 or
                proof.get("decode_read_calls") != 28 or
                proof.get("paged_read_calls") != 84 or
                proof.get("terminal_successes") != 84 or
                proof.get("pending_native_epochs") != 0 or
                proof.get("retained_pages") != 0):
            raise RuntimeError("physical two-span native B2 or retirement proof missing")
    elif (type(route) is not dict or route.get("route") != "ordinary" or
          type(proof.get("ordinary_model_forward_calls")) is not int or
          proof["ordinary_model_forward_calls"] < 1):
        raise RuntimeError("ordinary B2 model/route proof missing")
    return proof


def execute(manifest: dict, driver_spec: str) -> dict:
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError(f"B2 cell exceeded {MAX_SECONDS}s")))
    signal.alarm(MAX_SECONDS)
    driver = None
    try:
        preflight_cpu(manifest)
        contexts = contexts_for(manifest)
        if "q1_tile_arm" in manifest and (
                type(manifest["q1_tile_arm"]) is not bool or
                (os.environ.get("MLX2_PAGED_Q1_SIMD_TILE") == "1") !=
                manifest["q1_tile_arm"]):
            raise RuntimeError("Q1 SIMD tile environment differs from manifest")
        if "grouped_q1_write_arm" in manifest and (
                type(manifest["grouped_q1_write_arm"]) is not bool or
                (os.environ.get("MLX2_PAGED_GROUPED_Q1_WRITE") == "1") !=
                manifest["grouped_q1_write_arm"]):
            raise RuntimeError("grouped Q1 write environment differs from manifest")
        if "private_tail_reuse_arm" in manifest and (
                type(manifest["private_tail_reuse_arm"]) is not bool or
                (os.environ.get("MLX2_PAGED_PRIVATE_TAIL_REUSE") == "1") !=
                manifest["private_tail_reuse_arm"]):
            raise RuntimeError("private tail reuse environment differs from manifest")
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
        live = compute_live_price_identity(
            Path(manifest["paths"]["artifact"]),
            Path(manifest["paths"]["mlx_wheel"]),
            Path(_paged_kv_native.__file__).resolve(),
            adapter_artifact_root=Path(artifact["root"]))
        if (Path(_paged_kv_native.__file__).resolve() !=
                Path(manifest["paths"]["kernel"]).resolve() or live != identity):
            raise RuntimeError("loaded source/artifact/native identity differs")
        module, function = driver_spec.rsplit(":", 1)
        driver = getattr(importlib.import_module(module), function)(manifest)
        cases = []
        diagnostic_mode = manifest.get("diagnostic_mode") is True
        warm_screen = manifest.get("warm_screen") is True
        if "warm_screen" in manifest and type(manifest["warm_screen"]) is not bool:
            raise RuntimeError("warm screen selector must be a bool")
        orders = (("ordinary", "paged"),) if diagnostic_mode and not warm_screen else (
            ("ordinary", "paged"), ("paged", "ordinary"),
            ("ordinary", "paged"))
        for case_index, order in enumerate(orders):
            pair = {}
            for arm in order:
                begin = time.perf_counter_ns()
                proof = validate_proof(driver.run_request(arm, contexts), arm, contexts)
                if arm == "paged" and "q1_tile_arm" in manifest:
                    profile = proof.get("ready_native_host_profile")
                    expected_tiles = 28 if manifest["q1_tile_arm"] else 0
                    if (type(profile) is not dict or
                            profile.get("q1_tile_dispatches") != expected_tiles):
                        raise RuntimeError("Q1 tile dispatch proof differs")
                if arm == "paged" and "grouped_q1_write_arm" in manifest:
                    profile = proof.get("ready_native_host_profile")
                    grouped = manifest["grouped_q1_write_arm"]
                    # The completion watermark also covers packed-read leases
                    # and delayed prefill completions; native dispatch counters
                    # are the exact physical write engagement proof.
                    if (type(profile) is not dict or
                            profile.get("grouped_q1_writes") != (28 if grouped else 0) or
                            profile.get("native_write_dispatches") != (0 if grouped else 448)):
                        observed = (None if not isinstance(profile, dict) else {
                            key: profile.get(key) for key in (
                                "grouped_q1_writes", "native_write_dispatches",
                                "write_epoch_delta")})
                        raise RuntimeError(
                            f"grouped Q1 physical write dispatch proof differs: {observed}")
                complete_ms = (time.perf_counter_ns() - begin) / 1e6
                if complete_ms < proof["ready_pack_step_ms"]:
                    raise RuntimeError("ready step exceeds complete request")
                pair[arm] = {"complete_request_ms": complete_ms, "proof": proof}
            if pair["ordinary"]["proof"]["output_token_ids"] != pair["paged"]["proof"]["output_token_ids"]:
                raise PairParityFailure({"order": list(order), **pair}, case_index)
            cases.append({"order": list(order), **pair})
        evidence = {"schema": SCHEMA,
                    "status": "diagnostic_only" if diagnostic_mode else "screened",
                    "gpu_executed": True,
                    "measurement_scope": SCOPE, "profile_id": manifest["profile_id"],
                    "identity": identity, "context_tokens": list(contexts),
                    "scheduler_step": step_for(contexts), "gpuq_owner": owner,
                    "hard_seconds": MAX_SECONDS,
                    "max_resident_bytes": MAX_RESIDENT_BYTES,
                    "paired_cases": cases, "qualified": False,
                    "selected": False, "serving_selected": False,
                    "research_executed": True, "price_usable": False,
                    "research_price_candidate": not diagnostic_mode,
                    "diagnostic_mode": diagnostic_mode}
        evidence["warm_screen"] = warm_screen
        if "q1_tile_arm" in manifest:
            evidence["q1_tile_arm"] = manifest["q1_tile_arm"]
        if "grouped_q1_write_arm" in manifest:
            evidence["grouped_q1_write_arm"] = manifest["grouped_q1_write_arm"]
        if "private_tail_reuse_arm" in manifest:
            evidence["private_tail_reuse_arm"] = manifest["private_tail_reuse_arm"]
        raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
        evidence["research_request_evidence_sha256"] = hashlib.sha256(raw).hexdigest()
        return evidence
    finally:
        if driver is not None:
            driver.close()
        signal.alarm(0)


def failure_receipt(exc: Exception, manifest: dict, driver_spec: str) -> dict:
    parity = isinstance(exc, PairParityFailure)
    result = {"schema": SCHEMA, "status": "parity_failed" if parity else "failed",
              "gpu_executed": True if parity else None,
              "measurement_scope": SCOPE, "price_usable": False,
              "profile_id": manifest.get("profile_id"),
              "identity": manifest.get("identity"),
              "context_tokens": manifest.get("context_tokens"),
              "hard_seconds": MAX_SECONDS,
              "max_resident_bytes": MAX_RESIDENT_BYTES,
              "driver": driver_spec,
              "diagnostic_mode": manifest.get("diagnostic_mode") is True,
              "error_type": type(exc).__name__, "error": str(exc)}
    if parity:
        result["failed_case_index"] = exc.case_index
        result["failed_pair"] = exc.pair
    physical = getattr(exc, "physical_counters", None)
    if isinstance(physical, dict):
        result["failure_physical_counters"] = physical
    return result


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
            parser.error("GPU mode requires --manifest, --driver and --receipt")
        manifest = json.loads(args.manifest.read_text())
        try:
            result = execute(manifest, args.driver)
        except Exception as exc:
            result = failure_receipt(exc, manifest, args.driver)
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
