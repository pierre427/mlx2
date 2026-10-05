"""Short source-bound packed-vs-serial native B2 prompt and Q1 oracle."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import time
from pathlib import Path

from varlen_b2_http_gate import _contexts, _prompt
from varlen_live_qwen3_request_driver import Qwen3RequestDriver, retired_native_state
from varlen_pack_price_bench import MAX_RESIDENT_BYTES, _gpuq_owner
from varlen_staged_graph_price import preflight_cpu

SCHEMA = "mlx2.varlen-b2-packed-prefill-oracle.v1"


def _timeout(_signum, _frame):
    raise TimeoutError("packed B2 prompt oracle exceeded 100 seconds")


def _profile_delta(after, before):
    return {key: after["host_ns"][key] - before["host_ns"].get(key, 0)
            for key in after["host_ns"]}


def execute(manifest):
    contexts = _contexts(manifest)
    preflight_cpu(manifest, context_validator=_contexts)
    gpuq_owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from mlx2.adapters.qwen3_paged_candidate import PackedLane
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from mlx2.runtime.qwen3_paged_graph_factory import create_shared_qwen3_graph_pack

    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native binary differs")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("GPU execution required")
    if any(os.environ.get(key) != "1" for key in (
        "MLX2_PAGED_Q1_SIMD_TILE", "MLX2_PAGED_GROUPED_Q1_WRITE",
        "MLX2_PAGED_PRIVATE_TAIL_REUSE")):
        raise RuntimeError("oracle needs live grouped/tile/private-tail flags")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(100)
    driver = None
    try:
        driver = Qwen3RequestDriver(manifest)
        live = compute_live_price_identity(
            manifest["paths"]["artifact"], manifest["paths"]["mlx_wheel"],
            kernel, adapter_artifact_root=driver.adapter.identity["path"])
        if live != manifest["identity"]:
            raise RuntimeError("loaded Qwen3 source/model/wheel/native identity differs")
        prompts = (_prompt(driver.adapter, 1000, contexts[0])[1],
                   _prompt(driver.adapter, 1001, contexts[1])[1])
        revision = driver.adapter.identity["fingerprint"]
        depth = len(driver.model.layers)

        def arm(name):
            factory_start = time.perf_counter_ns()
            owners, candidate = create_shared_qwen3_graph_pack(
                driver.adapter,
                tuple((revision, len(tokens), 2) for tokens in prompts),
                permit_candidate=True, profile_host=True)
            factory_ms = (time.perf_counter_ns() - factory_start) / 1e6
            backend = candidate.backend
            branches = []
            ready_branches = []
            try:
                branches = [owner.begin(CandidateRequest(index + 1, revision,
                            len(tokens), ("kv",)))
                            for index, (owner, tokens) in enumerate(zip(owners, prompts))]
                before_reads = backend.read_submissions
                before_terminals = backend.terminal_successes
                before_spans = len(backend.staged_read_spans)
                before_host = backend.profile_snapshot()
                prefill_start = time.perf_counter_ns()
                if name == "packed":
                    logits, receipt = candidate.forward_staged(
                        tuple(PackedLane(tokens, branch.layers)
                              for tokens, branch in zip(prompts, branches)),
                        tuple(branches), permit_candidate=True)
                    if (receipt.get("packed_lanes") != 2 or
                            tuple(logits.shape[:1]) != (sum(contexts),)):
                        raise RuntimeError("packed prefill row geometry differs")
                    first_logits = (logits[contexts[0] - 1], logits[-1])
                    expected_reads = depth
                    if tuple(backend.staged_read_spans[before_spans:]) != (2,) * depth:
                        raise RuntimeError("packed prefill lacks two-span layer reads")
                else:
                    first_logits = []
                    for tokens, branch in zip(prompts, branches):
                        logits, _ = candidate.forward(
                            (PackedLane(tokens, branch.layers),),
                            permit_candidate=True, atomic_branch=branch)
                        mx.eval(logits)
                        first_logits.append(logits[-1])
                    expected_reads = 2 * depth
                mx.eval(first_logits)
                prefill_ms = (time.perf_counter_ns() - prefill_start) / 1e6
                if (backend.read_submissions - before_reads != expected_reads or
                        backend.terminal_successes - before_terminals != expected_reads):
                    raise RuntimeError("prefill native read/terminal count differs")
                prefill_host_ns = _profile_delta(backend.profile_snapshot(), before_host)
                for branch, tokens in zip(branches, prompts):
                    branch.prepare(len(tokens)).publish()
                first_ids = tuple(int(mx.argmax(logits).item()) for logits in first_logits)
                first_values = tuple(mx.array(logits) for logits in first_logits)
                with owners[0].snapshot() as a, owners[1].snapshot() as b:
                    if (a.offset, b.offset) != contexts or (a.generation, b.generation) != (1, 1):
                        raise RuntimeError("prefill owner publication differs")
                for index, owner in enumerate(owners):
                    ready_branches.append(owner.begin(CandidateRequest(
                        index + 1, revision, 1, ("kv",))))
                ready_branches = tuple(ready_branches)
                before_reads = backend.read_submissions
                before_terminals = backend.terminal_successes
                before_spans = len(backend.staged_read_spans)
                before_host = backend.profile_snapshot()
                ready_start = time.perf_counter_ns()
                ready_logits, receipt = candidate.forward_staged(
                    tuple(PackedLane((token,), branch.layers)
                          for token, branch in zip(first_ids, ready_branches)),
                    ready_branches, permit_candidate=True)
                ready_ms = (time.perf_counter_ns() - ready_start) / 1e6
                if (receipt.get("packed_lanes") != 2 or
                        tuple(ready_logits.shape[:1]) != (2,) or
                        backend.read_submissions - before_reads != depth or
                        backend.terminal_successes - before_terminals != depth or
                        tuple(backend.staged_read_spans[before_spans:]) != (2,) * depth):
                    raise RuntimeError("ready B2 native terminal/span proof differs")
                ready_host_ns = _profile_delta(backend.profile_snapshot(), before_host)
                after_ready = backend.profile_snapshot()
                ready_physical = {
                    key: after_ready[key] - before_host[key]
                    for key in ("grouped_q1_writes", "native_write_dispatches",
                                "q1_tile_dispatches")
                }
                if ready_physical != {"grouped_q1_writes": depth,
                                      "native_write_dispatches": 0,
                                      "q1_tile_dispatches": depth}:
                    raise RuntimeError("ready B2 grouped/tile physical proof differs")
                for branch in ready_branches:
                    branch.prepare(1).publish()
                second_ids = tuple(int(mx.argmax(logits).item()) for logits in ready_logits)
                second_values = tuple(mx.array(logits) for logits in ready_logits)
                mx.eval(first_values, second_values)
                with owners[0].snapshot() as a, owners[1].snapshot() as b:
                    if ((a.offset, b.offset) != (contexts[0] + 1, contexts[1] + 1) or
                            (a.generation, b.generation) != (2, 2)):
                        raise RuntimeError("ready owner publication differs")
                return ({"arm": name, "factory_ms": factory_ms,
                         "prefill_ms": prefill_ms, "ready_ms": ready_ms,
                         "prefill_reads_and_terminals": expected_reads,
                         "ready_reads_and_terminals": depth,
                         "prefill_host_ns": prefill_host_ns,
                         "ready_host_ns": ready_host_ns,
                         "ready_physical": ready_physical,
                         "first_ids": list(first_ids), "second_ids": list(second_ids)},
                        first_values, second_values)
            finally:
                for branch in (*branches, *ready_branches):
                    branch.rollback()
                for owner in owners:
                    owner.close()
                    owner.reap_retired()
                pending, retained, released = retired_native_state(backend.writer)
                if not released:
                    raise RuntimeError(f"{name} native owner retirement differs: {pending}, {retained}")
                backend.writer.backend.close_after_terminal()

        serial, serial_first, serial_second = arm("serial")
        packed, packed_first, packed_second = arm("packed")
        if (serial["first_ids"] != packed["first_ids"] or
                serial["second_ids"] != packed["second_ids"]):
            raise RuntimeError("packed prefill changes greedy first or second tokens")
        differences = []
        for left, right in zip(serial_first + serial_second,
                               packed_first + packed_second):
            differences.append(float(mx.max(mx.abs(left.astype(mx.float32) -
                                               right.astype(mx.float32))).item()))
        if any(value != 0 for value in differences):
            raise RuntimeError(f"packed prefill logits are not bitwise exact: {differences}")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RESIDENT_BYTES:
            raise RuntimeError("packed prefill oracle exceeded memory cap")
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": live, "gpuq_owner": gpuq_owner, "contexts": list(contexts),
                "serial": serial, "packed": packed, "last_logit_max_abs_diff": differences,
                "owners_fully_retired": True, "pending_native_epochs": 0,
                "retained_pages": 0, "qualified": False, "price_usable": False,
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    finally:
        signal.alarm(0)
        if driver is not None:
            driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    _contexts(manifest)
    if args.preflight:
        result = preflight_cpu(manifest, context_validator=_contexts)
    else:
        if args.receipt is None:
            parser.error("GPU oracle requires --receipt")
        try:
            result = execute(manifest)
        except Exception as error:
            result = {"schema": SCHEMA, "status": "failed",
                      "identity": manifest.get("identity"),
                      "error_type": type(error).__name__, "error": str(error),
                      "qualified": False, "price_usable": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
