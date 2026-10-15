#!/usr/bin/env python3
"""Bounded full-artifact Qwen3.8 exact-checkpoint diagnostic on an owned GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

from probe_fold_lifecycle import atomic_json, thermal

PHASES = ("greedy", "sampled", "corruption", "cleanup")
MAX_STARTUP_TIMEOUT_SECONDS = 900.0
MAX_PHASE_TIMEOUT_SECONDS = 1800.0


def executor_step_bound(context_tokens, prefill_step, output_tokens):
    return math.ceil(context_tokens / prefill_step) + output_tokens + 16


def checkpoint_due(emitted, row_count, checkpoints, captured):
    return bool(
        emitted
        and not emitted[-1].finish_reason
        and row_count in checkpoints
        and row_count not in captured
    )


def validate_timeouts(startup_timeout, phase_timeout) -> None:
    """Keep every probe phase inside one finite owned accelerator window."""
    for name, value, maximum in (
        ("startup-timeout", startup_timeout, MAX_STARTUP_TIMEOUT_SECONDS),
        ("timeout", phase_timeout, MAX_PHASE_TIMEOUT_SECONDS),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise ValueError(f"{name} must be finite and in (0, {maximum:g}]")


def _array_digest(value) -> dict:
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            np.asarray(value.view(mx.uint8)).tobytes()
        ).hexdigest(),
    }


def run_phase(args, ownership):
    # Do not import MLX until the live CPG lease and both ancestor locks have
    # been validated by main().
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir / args.phase
    output.mkdir(parents=True)
    source_names = (
        "src/mlx2/runtime/decode_checkpoint.py",
        "src/mlx2/runtime/generate.py",
        "src/mlx2/runtime/models/cache.py",
        "src/mlx2/runtime/models/qwen38_27b.py",
        "src/mlx2/runtime/sample_utils.py",
        "src/mlx2/adapters/qwen38_27b.py",
        "src/mlx2/serving.py",
        "scripts/probe_qwen38_checkpoint_pressure.py",
        "scripts/probe_fold_lifecycle.py",
        "scripts/qualification_gpu_ownership.py",
        "scripts/run_qualification_owned.py",
        "scripts/run_with_gpu_locks.py",
        "scripts/thermal_probe.swift",
    )

    def source_hashes():
        return {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_names
        }

    source_before = {}
    report = {
        "schema": "mlx2.qwen38-exact-checkpoint-pressure.v1",
        "phase": args.phase,
        "ownership": ownership,
        "passed": False,
        "qualified": False,
        "performance_qualified": False,
        "device": mx.device_info(),
        "mlx_version": mx.__version__,
        "thermal": [],
        "memory": [],
        "source_sha256": {},
    }
    engine = None
    monitor_stop = threading.Event()
    monitor_errors = []
    monitor = None
    try:
        source_before = source_hashes()
        report["source_sha256"] = source_before
        report["thermal"].append(thermal(root))
        if report["thermal"][0]["thermal_state"] != 0:
            raise RuntimeError("phase must start at nominal thermal state")
        from mlx2.serving import ServingEngine

        context = args.context if args.phase in {"greedy", "sampled"} else args.failure_context
        engine = ServingEngine(
            str(args.model),
            mtp=False,
            max_lanes=1,
            max_inflight=1,
            max_context=context + 64,
            prefill_step=args.prefill_step,
            cache_bytes=512 << 20,
            qualification_mode=True,
            lane_matmul="off",
            thinking_auto_calibration=False,
        )
        deadline = time.monotonic() + args.startup_timeout
        while not engine.ready.wait(0.1):
            if engine.error or time.monotonic() >= deadline:
                raise RuntimeError(engine.error or "adapter startup deadline")
        if engine.error:
            raise RuntimeError(engine.error)
        phase_deadline = time.monotonic() + args.timeout

        def require_phase_time():
            if time.monotonic() >= phase_deadline:
                raise TimeoutError("checkpoint diagnostic phase deadline")

        # The adapter pins import-time model flags during construction.  Model
        # runtime helpers must therefore be imported only after readiness.
        from mlx2.adapters.qwen38_27b import inspect_artifact
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime.os_memory import physical_footprint_bytes
        from mlx2.runtime.sample_utils import LaneRNG, draw_key

        adapter = engine.adapter
        if getattr(adapter, "draft_model", None) is not None:
            raise RuntimeError("checkpoint diagnostic requires the ordinary target only")
        report["artifact_identity"] = json.loads(json.dumps(adapter.identity))

        def memory(label, *, staging=0):
            row = {
                "label": label,
                "active_bytes": int(mx.get_active_memory()),
                "cached_bytes": int(mx.get_cache_memory()),
                "footprint_bytes": physical_footprint_bytes(),
                "raw_headroom_bytes": int(engine.admission_headroom_fn()()),
                "staging_bytes": int(staging),
            }
            report["memory"].append(row)
            return row

        memory("adapter_ready")

        def watch():
            try:
                while not monitor_stop.wait(10):
                    report["thermal"].append(thermal(root))
            except Exception as error:  # noqa: BLE001 - any daemon exit must fail evidence
                monitor_errors.append(f"{type(error).__name__}: {error}")
                monitor_stop.set()

        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()

        def greedy_sampler(values):
            return mx.argmax(values, axis=-1)

        unit = "Record 000000: gauge 17; seal intact; valve closed.\n"
        repeats = max(1, context // 10)
        while True:
            prompt = adapter.prompt_tokens({"prompt": unit * repeats})
            if len(prompt) >= context:
                prompt = prompt[:context]
                break
            repeats *= 2
        report.update(
            context_tokens=len(prompt),
            prompt_sha256=hashlib.sha256(
                b"".join(int(token).to_bytes(4, "little") for token in prompt)
            ).hexdigest(),
        )

        def execute(*, sampled, checkpoints):
            rng = LaneRNG(91)
            if sampled:
                sampler = lambda values: mx.random.categorical(
                    values / 0.8, key=draw_key(rng)
                )
            else:
                sampler = greedy_sampler
            batch = BatchGenerator(
                adapter.model,
                completion_batch_size=1,
                prefill_batch_size=1,
                prefill_step_size=args.prefill_step,
            )
            (uid,) = batch.insert(
                [prompt],
                max_tokens=[args.tokens],
                samplers=[sampler],
                lane_rngs=[rng if sampled else None],
            )
            rows = []
            owners = []
            captured = set()
            # One ``next`` call advances at most one prefill chunk.  Leave an
            # explicit decode and finish guard after every required chunk so a
            # context exactly divisible by the prefill step cannot exhaust the
            # diagnostic bound before its first emitted token.
            max_steps = executor_step_bound(
                len(prompt), args.prefill_step, args.tokens
            )
            try:
                for _ in range(max_steps):
                    require_phase_time()
                    _, emitted = batch.next()
                    require_phase_time()
                    rows.extend(emitted)
                    if checkpoint_due(emitted, len(rows), checkpoints, captured):
                        staging = batch.ordinary_checkpoint_staging_bytes()
                        before = memory(
                            f"checkpoint_{len(rows)}_before", staging=staging
                        )
                        if staging > before["raw_headroom_bytes"]:
                            raise MemoryError(
                                "checkpoint staging exceeds measured raw headroom"
                            )
                        binding = (adapter.identity["fingerprint"], args.phase, len(rows))
                        owner = batch.suspend_ordinary_checkpoint(uid, binding)
                        owners.append(owner)
                        require_phase_time()
                        batch.resume_ordinary_checkpoint(owner, binding)
                        require_phase_time()
                        if not owner.closed or not owner.consumed:
                            raise RuntimeError(
                                "successful checkpoint resume retained its storage owner"
                            )
                        captured.add(len(rows))
                        memory(f"checkpoint_{len(rows)}_restored", staging=staging)
                    if emitted and emitted[-1].finish_reason:
                        break
                else:
                    raise RuntimeError(
                        f"bounded executor did not finish after {max_steps} steps"
                    )
                if captured != set(checkpoints):
                    raise RuntimeError(
                        "checkpoint exercise mismatch: "
                        f"expected={sorted(checkpoints)} observed={sorted(captured)}"
                    )
                return {
                    "tokens": [int(row.token) for row in rows],
                    "logprobs": [_array_digest(row.logprobs) for row in rows],
                    "rng_draws": rng.draws,
                    "finish_reason": rows[-1].finish_reason,
                    "checkpoint_captures": sorted(captured),
                }
            finally:
                for owner in owners:
                    owner.close()
                batch.close()

        if args.phase in {"greedy", "sampled"}:
            sampled = args.phase == "sampled"
            reference = execute(sampled=sampled, checkpoints=())
            actual = execute(sampled=sampled, checkpoints=(1, 3, 5))
            report["checkpoint_captures"] = actual.pop("checkpoint_captures")
            reference_captures = reference.pop("checkpoint_captures")
            if reference_captures:
                raise RuntimeError("reference unexpectedly exercised checkpoints")
            report["reference"] = reference
            report["checkpointed"] = actual
            report["exact"] = actual == reference
            if not report["exact"]:
                raise RuntimeError("checkpointed continuation changed exact output state")
        else:
            batch = BatchGenerator(
                adapter.model,
                completion_batch_size=1,
                prefill_batch_size=1,
                prefill_step_size=args.prefill_step,
            )
            owner = None
            try:
                (uid,) = batch.insert(
                    [prompt], max_tokens=[4], samplers=[greedy_sampler]
                )
                max_steps = executor_step_bound(len(prompt), args.prefill_step, 4)
                for _ in range(max_steps):
                    require_phase_time()
                    _, emitted = batch.next()
                    require_phase_time()
                    if emitted:
                        break
                else:
                    raise RuntimeError("bounded failure arm did not reach decode")
                if emitted[-1].finish_reason:
                    raise RuntimeError(
                        "failure arm reached a terminal row before checkpoint capture"
                    )
                staging = batch.ordinary_checkpoint_staging_bytes()
                before = memory("failure_checkpoint_before", staging=staging)
                if staging > before["raw_headroom_bytes"]:
                    raise MemoryError(
                        "failure-arm checkpoint staging exceeds measured raw headroom"
                    )
                binding = (adapter.identity["fingerprint"], args.phase)
                owner = batch.suspend_ordinary_checkpoint(uid, binding)
                if args.phase == "corruption":
                    handle = owner.file.handle
                    handle.seek(0)
                    first = handle.read(1)
                    handle.seek(0)
                    handle.write(bytes([first[0] ^ 0xFF]))
                    handle.flush()
                    try:
                        batch.resume_ordinary_checkpoint(owner, binding)
                    except ValueError as error:
                        report["corruption_refused"] = "digest mismatch" in str(error)
                    else:
                        report["corruption_refused"] = False
                    if not report["corruption_refused"]:
                        raise RuntimeError("corrupt checkpoint was not refused")
                owner.close()
                report["checkpoint_owner_cleaned"] = bool(
                    owner.closed
                    and owner.control is None
                    and owner.file is None
                    and owner.owner_executor is None
                )
                if not report["checkpoint_owner_cleaned"]:
                    raise RuntimeError("checkpoint owner retained state after close")
                if args.phase == "cleanup":
                    report["owner_close_cleanup"] = True
                owner = None
                (fresh_uid,) = batch.insert(
                    [[1, 2, 3]], max_tokens=[1], samplers=[greedy_sampler]
                )
                fresh = []
                for _ in range(32):
                    require_phase_time()
                    _, emitted = batch.next()
                    require_phase_time()
                    fresh.extend(row for row in emitted if row.uid == fresh_uid)
                    if fresh and fresh[-1].finish_reason:
                        break
                report["fresh_after_cleanup"] = bool(fresh and fresh[-1].finish_reason)
                if not report["fresh_after_cleanup"]:
                    raise RuntimeError("executor did not accept work after checkpoint cleanup")
            finally:
                if owner is not None:
                    owner.close()
                batch.close()
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=35)
            if monitor.is_alive():
                raise RuntimeError("thermal watcher did not stop within its subprocess bound")
        if monitor_errors:
            raise RuntimeError(f"thermal watcher failed: {monitor_errors[0]}")
        report["artifact_identity_end"] = json.loads(
            json.dumps(inspect_artifact(args.model)["identity"])
        )
        report["artifact_identity_unchanged"] = (
            report["artifact_identity_end"] == report["artifact_identity"]
        )
        if not report["artifact_identity_unchanged"]:
            raise RuntimeError("artifact identity changed during checkpoint phase")
        report["source_sha256_end"] = source_hashes()
        report["source_unchanged"] = report["source_sha256_end"] == source_before
        if not report["source_unchanged"]:
            raise RuntimeError("checkpoint diagnostic source changed during phase")
        report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=35)
        if engine is not None:
            engine.close()
        if "source_sha256_end" not in report:
            try:
                report["source_sha256_end"] = source_hashes()
                report["source_unchanged"] = bool(
                    source_before and report["source_sha256_end"] == source_before
                )
            except Exception as error:  # noqa: BLE001 - preserve the primary failure receipt
                report["source_hash_end_error"] = f"{type(error).__name__}: {error}"
        if monitor_errors:
            report["thermal_monitor_error"] = monitor_errors[0]
        report["swap_after"] = subprocess.check_output(
            ["/usr/sbin/sysctl", "vm.swapusage"], text=True
        ).strip()
        atomic_json(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--failure-context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--phase", choices=("all",) + PHASES, default="all")
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    args = parser.parse_args()
    if not 8192 <= args.context <= 32768 or not 512 <= args.failure_context <= 8192:
        parser.error("context bound exceeded")
    if not 6 <= args.tokens <= 16 or not 1 <= args.prefill_step <= 256:
        parser.error("token/prefill bound exceeded")
    try:
        validate_timeouts(args.startup_timeout, args.timeout)
    except ValueError as error:
        parser.error(str(error))
    target = args.output_dir if args.phase == "all" else args.output_dir / args.phase
    if target.exists():
        parser.error("refusing to overwrite evidence")
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(
        task_id=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        generation=args.cpg_generation,
    )
    if args.phase != "all":
        return 0 if run_phase(args, ownership)["passed"] else 1
    args.output_dir.mkdir(parents=True)
    summary = {
        "ownership": ownership,
        "phases": [],
        "passed": False,
        "qualified": False,
        "performance_qualified": False,
    }
    try:
        for name in PHASES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--phase",
                name,
            ]
            try:
                child = subprocess.run(
                    command,
                    timeout=args.startup_timeout + args.timeout + 120,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                summary["phases"].append({"phase": name, "timed_out": True})
                return 1
            summary["phases"].append({"phase": name, "returncode": child.returncode})
            if child.returncode:
                return 1
        summary["passed"] = True
        return 0
    finally:
        atomic_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
