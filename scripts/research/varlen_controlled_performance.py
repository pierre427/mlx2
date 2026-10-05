"""Controlled, default-off live-request performance screen for Qwen3 paged KV.

The native B2 arm has two live request lanes but presently executes two
request-private width-one continuations. It is *not* a fused B2 pack price.
Run only under an owned shared gpuq lease; no result qualifies a route.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import resource
import signal
import statistics
import subprocess
import time
from pathlib import Path
from threading import RLock

from varlen_pack_price_bench import (
    MAX_RESIDENT_BYTES,
    ROOT,
    _gpuq_owner,
    sha256,
    source_tree_sha256,
    verify_artifact_manifest,
)
from varlen_performance_phases import (
    PhaseTotals,
    derived_host_phases,
    instrument_native_backend,
    subtract_phase_snapshots,
)

SCHEMA = "mlx2.varlen-controlled-performance.v1"
CONTEXTS = (63, 65, 129)
BATCHES = (1, 2)
REPEATS = 4
HARD_SECONDS = 120


def arm_order(repeat: int) -> tuple[str, str]:
    if type(repeat) is not int or repeat < 0:
        raise ValueError("repeat index must be nonnegative")
    return ("ordinary", "paged") if repeat % 2 == 0 else ("paged", "ordinary")


def paired_token_parity(ordinary: dict, paged: dict) -> bool:
    """Compare ordered per-lane trajectories, independent of unrelated UIDs."""
    def trajectories(case):
        output = case.get("outputs")
        if not isinstance(output, dict) or not output:
            raise ValueError("complete request outputs required")
        values = list(output.values())
        if any(type(row) is not list or len(row) != 2 or
               any(type(token) is not int or token < 0 for token in row)
               for row in values):
            raise ValueError("two valid output tokens per lane required")
        return values

    return trajectories(ordinary) == trajectories(paged)


def summarize_eligible_pairs(pairs: list[dict]) -> dict:
    if not pairs or not all(p.get("ratio_eligible") is True for p in pairs):
        return {"ratio_eligible": False, "native_over_ordinary": None,
                "reason": "token parity missing or differing"}
    result = {}
    for key, field in (("complete_request", "total_request_wall_ms"),
                       ("response_ready", "response_ready_wall_ms")):
        ratios = [p["arms"]["paged"][field] / p["arms"]["ordinary"][field]
                  for p in pairs]
        result[key] = {"paired_native_over_ordinary": ratios,
                       "median_native_over_ordinary": statistics.median(ratios)}
    return {"ratio_eligible": True, "native_over_ordinary": result}


def _swap_used_bytes() -> int:
    output = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=4)
    found = re.search(r"\bused\s*=\s*([0-9.]+)([MGT])", output)
    if not found:
        raise RuntimeError("swap usage unavailable")
    return int(float(found.group(1)) * {"M": 2**20, "G": 2**30, "T": 2**40}[found.group(2)])


def _thermal_state() -> int:
    # ProcessInfo exposes 0 nominal, 1 fair, 2 serious, 3 critical.
    output = subprocess.check_output(
        ["swift", "-e", "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)"],
        text=True, timeout=12).strip()
    if output not in ("0", "1", "2", "3"):
        raise RuntimeError("thermal state unavailable")
    return int(output)


def preflight(manifest: dict, *, contexts: tuple[int, ...], batches: tuple[int, ...],
              repeats: int) -> dict:
    from mlx2.runtime.paged_pack_price import _identity

    identity = _identity(manifest.get("identity"))
    if (identity["host"] != platform.node() or
            identity["source_commit"] != subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip() or
            identity["source_tree_sha256"] != source_tree_sha256()):
        raise RuntimeError("host, source commit, source bytes, or cleanliness differ")
    hardware = json.loads(subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "-json"], text=True,
        timeout=10))["SPHardwareDataType"]
    if [item.get("chip_type") for item in hardware] != [identity["hardware"]]:
        raise RuntimeError("hardware identity differs")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"artifact", "mlx_wheel", "kernel"}:
        raise RuntimeError("artifact, MLX wheel and native kernel paths required")
    for name, field in (("artifact", "artifact_sha256"),
                        ("mlx_wheel", "mlx_wheel_sha256"),
                        ("kernel", "kernel_sha256")):
        if sha256(Path(paths[name])) != identity[field]:
            raise RuntimeError(f"{name} bytes differ")
    artifact = verify_artifact_manifest(Path(paths["artifact"]))
    config = json.loads((Path(artifact["root"]) / "config.json").read_text())
    geometry = {"model_type": "qwen3", "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "tie_word_embeddings": True}
    if any(config.get(k) != v for k, v in geometry.items()) or config.get("rope_scaling"):
        raise RuntimeError("unsupported pinned model geometry")
    if (not contexts or any(type(c) is not int or c not in CONTEXTS for c in contexts) or
            len(set(contexts)) != len(contexts) or
            not batches or any(type(b) is not int or b not in BATCHES for b in batches) or
            len(set(batches)) != len(batches) or
            type(repeats) is not int or not 2 <= repeats <= REPEATS or
            manifest.get("max_resident_bytes") not in range(1, MAX_RESIDENT_BYTES + 1)):
        raise RuntimeError("unsupported bounded performance matrix")
    return {"schema": SCHEMA, "status": "preflight_passed", "gpu_executed": False,
            "identity": identity, "contexts": contexts, "batches": batches,
            "repeats": repeats, "qualified": False, "selected": False}


def _timeout(_signal, _frame):
    raise TimeoutError("controlled performance exceeded 120 seconds")


def _run_case(driver, *, arm: str, context: int, batch_size: int,
              resident_limit: int) -> dict:
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_native_batch_lifecycle import (
        prepare_queued_native_first_response,
        run_research_native_queued_qwen3,
    )
    from mlx2.runtime.paged_request_transaction import CandidateRequest

    mx = driver.mx
    prompt = [1000] * context
    lock = RLock()
    generator = BatchGenerator(driver.model, max_tokens=2,
                               prefill_batch_size=batch_size,
                               completion_batch_size=batch_size,
                               prefill_step_size=context, stop_tokens=[])
    owners = []
    candidates = []
    uids = []
    totals = PhaseTotals()
    responses: dict[int, list[int]] = {}
    route_receipts: dict[int, list[dict]] = {}
    began = time.perf_counter_ns()
    try:
        with instrument_native_backend(totals):
            with lock:
                uids = generator.insert([prompt[:] for _ in range(batch_size)],
                                        max_tokens=[2] * batch_size)
            responses = {uid: [] for uid in uids}
            route_receipts = {uid: [] for uid in uids}
            if arm == "paged":
                for uid in uids:
                    owner, candidate = driver.adapter.create_native_paged_qwen3_request(
                        revision=driver.revision, prompt_tokens=context,
                        max_tokens=2, permit_candidate=True)
                    owners.append(owner)
                    candidates.append(candidate)
                    probe = run_research_native_queued_qwen3(
                        generator, lock, CandidateRequest(uid, driver.revision,
                                                          context, ("kv",)),
                        owner, candidate, research_permit=True)
                    if probe.reason != "research_executed" or not probe.research_executed:
                        raise RuntimeError(f"native prompt refused: {probe.reason}")
                    prepared = prepare_queued_native_first_response(generator, lock,
                                                                     owner, probe)
                    with lock:
                        installed = generator.install_native_queued(
                            prepared, owner, candidate, lock,
                            permit_native=True, research_only=True)
                    if installed.get("reason") != "native_installed":
                        raise RuntimeError(f"native handoff refused: {installed}")
            setup_ms = (time.perf_counter_ns() - began) / 1e6
            steps = []
            for _ in range(6):
                start = time.perf_counter_ns()
                before = totals.snapshot()
                before_counts = {uid: len(tokens) for uid, tokens in responses.items()}
                with lock:
                    _, produced = generator.next()
                failures = generator.take_lane_failures()
                if failures:
                    raise RuntimeError(f"live lane failure: {failures}")
                for response in produced:
                    if response.uid not in responses:
                        raise RuntimeError("response UID drifted")
                    mx.eval(response.logprobs)
                    responses[response.uid].append(int(response.token))
                    route_receipts[response.uid].append(dict(response.mtp_receipt or {}))
                mx.synchronize()
                after = totals.snapshot()
                phase_delta = subtract_phase_snapshots(before, after)
                steps.append({"wall_ms": (time.perf_counter_ns() - start) / 1e6,
                              "emitted": len(produced),
                              "second_token_emissions": sum(
                                  before_counts[uid] < 2 and len(tokens) == 2
                                  for uid, tokens in responses.items()),
                              "native_phases": phase_delta,
                              "derived_host_phases": derived_host_phases(phase_delta)})
                if all(len(tokens) == 2 for tokens in responses.values()):
                    break
            if not all(len(tokens) == 2 for tokens in responses.values()):
                raise RuntimeError("two complete outputs per request required")
            response_ready_ms = (time.perf_counter_ns() - began) / 1e6
            for uid in uids:
                receipts = route_receipts[uid]
                if arm == "paged" and (
                    len(receipts) != 2 or
                    any(r.get("route") != "native_qwen3_paged" or
                        r.get("research_executed") is not True or
                        r.get("selected") is not False or
                        r.get("qualified") is not False for r in receipts) or
                    receipts[-1].get("native_read_calls") != 56 or
                    receipts[-1].get("terminal_successes") != 56
                ):
                    raise RuntimeError("native route or physical reads missing")
            with lock:
                generator.remove(uids)
            mx.synchronize()
            total_ms = (time.perf_counter_ns() - began) / 1e6
            retirement = []
            for owner, candidate in zip(owners, candidates):
                writer = candidate.backend.writer
                retirement.append({"fully_retired": owner.fully_retired,
                                   "pending_epochs": len(writer.pending_epochs),
                                   "retained_pages": writer.pool.allocated_count})
            if any(not r["fully_retired"] or r["pending_epochs"] or
                   r["retained_pages"] for r in retirement):
                raise RuntimeError("native request did not retire")
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if peak > resident_limit:
                raise RuntimeError("resident memory cap exceeded")
            raw_phases = totals.snapshot()
            return {"arm": arm, "context": context, "active_request_count": batch_size,
                    "native_execution_width": 1 if arm == "paged" else None,
                    "total_request_wall_ms": total_ms,
                    "response_ready_wall_ms": response_ready_ms,
                    "setup_wall_ms": setup_ms,
                    "steps": steps, "outputs": {str(k): v for k, v in responses.items()},
                    "route_receipts": {str(k): v for k, v in route_receipts.items()},
                    "native_phases": raw_phases,
                    "derived_host_phases": derived_host_phases(raw_phases),
                    "retirement": retirement, "peak_resident_bytes": peak,
                    "synchronized": True}
    finally:
        if uids:
            with lock:
                generator.remove(uids)
        generator.close()
        for owner in owners:
            if not owner.fully_retired:
                owner.close()
                owner.reap_retired()
                owner.reap_quarantine()


def execute(manifest: dict, *, contexts: tuple[int, ...], batches: tuple[int, ...],
            repeats: int) -> dict:
    preflight(manifest, contexts=contexts, batches=batches, repeats=repeats)
    owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from varlen_live_qwen3_request_driver import Qwen3RequestDriver

    from mlx2.runtime.paged_price_identity import compute_live_price_identity

    identity = manifest["identity"]
    if (mx.__version__ != identity["mlx_wheel_version"] or
            mx.default_device() != mx.gpu or not mx.metal.is_available() or
            Path(_paged_kv_native.__file__).resolve() !=
                Path(manifest["paths"]["kernel"]).resolve()):
        raise RuntimeError("loaded MLX device, wheel, or native binary differs")
    if compute_live_price_identity(
            manifest["paths"]["artifact"], manifest["paths"]["mlx_wheel"],
            manifest["paths"]["kernel"],
            adapter_artifact_root=verify_artifact_manifest(
                Path(manifest["paths"]["artifact"]))["root"]) != identity:
        raise RuntimeError("live runtime identity differs")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(HARD_SECONDS)
    driver = None
    try:
        swap_before, thermal_before = _swap_used_bytes(), _thermal_state()
        if thermal_before >= 2:
            raise RuntimeError("host thermal state is serious or critical")
        driver = Qwen3RequestDriver(manifest)
        cells = []
        for context in contexts:
            for batch_size in batches:
                pairs = []
                for repeat in range(repeats):
                    if _thermal_state() >= 2:
                        raise RuntimeError("thermal state reached serious before pair")
                    order = arm_order(repeat)
                    by_arm = {}
                    for arm in order:
                        before = _swap_used_bytes()
                        if before > swap_before:
                            raise RuntimeError("swap use grew before performance arm")
                        by_arm[arm] = _run_case(driver, arm=arm, context=context,
                                                batch_size=batch_size,
                                                resident_limit=manifest["max_resident_bytes"])
                        after = _swap_used_bytes()
                        if after > swap_before:
                            raise RuntimeError("swap use grew during performance arm")
                        if _thermal_state() >= 2:
                            raise RuntimeError("thermal state reached serious during pair")
                    parity = paired_token_parity(by_arm["ordinary"], by_arm["paged"])
                    pairs.append({"order": list(order), "arms": by_arm,
                                  "token_parity": parity,
                                  "ratio_eligible": parity})
                cells.append({"context": context, "active_request_count": batch_size,
                              "native_execution_width": 1, "pairs": pairs,
                              "all_token_parity": all(p["token_parity"] for p in pairs),
                              "ratio_eligible": all(p["ratio_eligible"] for p in pairs),
                              "summary": summarize_eligible_pairs(pairs)})
        return {"schema": SCHEMA, "status": "measured", "gpu_executed": True,
                "measurement_scope": "complete_live_batchgenerator_request",
                "identity": identity, "gpuq_owner": owner,
                "contexts": list(contexts), "batches": list(batches),
                "repeats": repeats, "cells": cells,
                "swap_used_bytes_before": swap_before,
                "swap_used_bytes_after": _swap_used_bytes(),
                "thermal_state_before": thermal_before,
                "thermal_state_after": _thermal_state(),
                "device_kernel_timing_available": False,
                "qualified": False, "selected": False,
                "native_default_enabled": False}
    finally:
        if driver is not None:
            driver.close()
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[63])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.preflight and args.execute_gpu:
        parser.error("choose preflight or GPU execution")
    if not args.manifest:
        result = {"schema": SCHEMA, "status": "planned", "gpu_executed": False,
                  "supported_contexts": CONTEXTS, "supported_batches": BATCHES,
                  "max_repeats": REPEATS, "qualified": False, "selected": False}
    else:
        manifest = json.loads(args.manifest.read_text())
        kwargs = {"contexts": tuple(args.contexts), "batches": tuple(args.batches),
                  "repeats": args.repeats}
        if args.execute_gpu:
            try:
                result = execute(manifest, **kwargs)
            except Exception as exc:
                result = {"schema": SCHEMA, "status": "failed", "gpu_executed": None,
                          "error_type": type(exc).__name__, "error": str(exc),
                          "qualified": False, "selected": False}
                if args.receipt:
                    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
                raise
        else:
            result = preflight(manifest, **kwargs)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
