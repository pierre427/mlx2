#!/usr/bin/env python3
"""Owned M3 Core ML projection diagnostic; never claim ANE use from a plan.

Run under an existing CPG/radio + paired-lock ownership window. The supervisor
bounds conversion, compilation, prediction and plan inspection together. It
never acquires/releases a lease itself. No full preflight or qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "mlx2.coreml-projection-probe.v1"


def _atomic(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _stop(process):
    """Retain the supervisor/lease if an owned process cannot yet be reaped."""
    for sig, seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=seconds)
            break
        except subprocess.TimeoutExpired:
            continue
    # Native uninterruptible OS work has no safe artificial ownership expiry.
    # Remain the live owner until the child actually exits in that rare case.
    if process.poll() is None:
        print("Owned worker did not drain after SIGKILL; retaining supervisor ownership", file=sys.stderr, flush=True)
        process.wait()
    # Remove any descendants still in this job's process group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(command, *, timeout, output):
    """Run on the main thread; retain ownership until the worker is reaped."""
    process = None
    started = time.monotonic()
    previous = {}
    interrupted_signal = None

    def interrupt(signum, _frame):
        nonlocal interrupted_signal
        # Do not raise across Popen or native cleanup: the process must first
        # be registered and then drained before ownership can be released.
        interrupted_signal = signum

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        with (output / "worker.log").open("x") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                while True:
                    if interrupted_signal is not None:
                        reason = "interrupted"
                        break
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        reason = "deadline"
                        break
                    try:
                        code = process.wait(timeout=min(remaining, .1))
                        reason = "completed" if code == 0 else "worker_failed"
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                _stop(process)
    finally:
        if process is not None and process.poll() is None:
            _stop(process)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return {"status": reason, "returncode": process.returncode,
            "interrupted_signal": interrupted_signal,
            "worker_exit_confirmed": process.poll() is not None,
            "elapsed_seconds": time.monotonic() - started}


def fixture():
    import numpy as np

    weights = ((np.arange(32).reshape(4, 8) % 7 - 3) / 8).astype(np.float16)
    x = ((np.arange(16).reshape(2, 8) - 8) / 8).astype(np.float16)
    inputs = [x, -x, np.zeros_like(x)]
    identity = {"kind": "synthetic-linear-no-bias", "shape": [2, 8, 4], "dtype": "float16",
                "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return weights, inputs, identity, digest


def worker(args, ownership):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Core ML probe requires Apple Silicon macOS")
    chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5).strip()
    if not chip.startswith("Apple M3"):
        raise RuntimeError(f"this probe window requires M3; found {chip}")
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    from mlx2.runtime.ane_projection import (
        CoreMLProjectionExecutor,
        FixedProjection,
        compiled_model_digest,
        verify_projection_output,
    )

    weights, inputs, recipe, artifact_digest = fixture()

    @mb.program(input_specs=[mb.TensorSpec(shape=(2, 8), dtype=types.fp16)],
                opset_version=ct.target.iOS16)
    def projection(x):
        return mb.linear(x=x, weight=weights, name="projection")

    # Conversion does not load a model. Explicit compute policy is repeated
    # when the compiled model and compute plan are loaded by the executor.
    model = ct.convert(projection, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                       skip_model_load=True)
    package = args.output_dir / "projection.mlpackage"
    model.save(str(package))
    compiled = ct.models.utils.compile_model(str(package),
                                             destination_path=str(args.output_dir / "projection.mlmodelc"))
    digest = compiled_model_digest(compiled)
    spec = FixedProjection("mlx2.synthetic_projection_probe", artifact_digest, 8, 4, 2)
    predictions = []
    with CoreMLProjectionExecutor(spec, compiled, declared_sha256=digest) as executor:
        for repetition in range(args.repetitions):
            for index, x in enumerate(inputs):
                reference = x.astype(np.float32) @ weights.astype(np.float32).T
                output, receipt = executor.predict(x)
                parity = verify_projection_output(spec, output, reference,
                                                   absolute_tolerance=.002, relative_tolerance=.002)
                predictions.append({"repetition": repetition, "case": index,
                                    "prediction": receipt, "parity": parity})
        # A plan is optional diagnostic metadata; unsupported APIs do not erase
        # successful prediction evidence, and never imply device observation.
        try:
            plan = {"available": True, **executor.compute_plan()}
        except (AttributeError, RuntimeError) as exc:
            plan = {"available": False, "error": f"{type(exc).__name__}: {exc}",
                    "observed_used": False, "execution_device_observed": False}
    result = {"schema": SCHEMA, "host": chip, "ownership": ownership, "recipe": recipe,
              "artifact_sha256": artifact_digest, "compiled_model_sha256": digest,
              "coremltools_version": ct.__version__, "predictions": predictions, "compute_plan": plan,
              "prediction_completed": True, "observed_used": False, "execution_device_observed": False,
              "qualified": False, "performance_claim": False,
              "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path(__file__).resolve().parents[1] / "src/mlx2/runtime/ane_projection.py")}}
    _atomic(args.output_dir / "prediction.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 300 or not 1 <= args.repetitions <= 5:
        parser.error("timeout must be 1..300 seconds and repetitions 1..5")
    args.output_dir = args.output_dir.expanduser().absolute()
    if not args.worker and args.output_dir.exists():
        parser.error("output directory already exists; use a fresh evidence directory")
    from qualification_gpu_ownership import require_qualification_lease

    ownership = require_qualification_lease(task_id=args.cpg_task, cpg_owner_lock=args.cpg_owner_lock,
                                           generation=args.cpg_generation)
    if args.worker:
        worker(args, ownership)
        return 0
    args.output_dir.mkdir(mode=0o700, parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir", str(args.output_dir),
               "--timeout", str(args.timeout), "--repetitions", str(args.repetitions),
               "--cpg-task", args.cpg_task, "--cpg-owner-lock", str(args.cpg_owner_lock),
               "--cpg-generation", str(args.cpg_generation)]
    result = supervise(command, timeout=args.timeout, output=args.output_dir)
    result.update(schema=SCHEMA, qualified=False, observed_used=False, ownership=ownership)
    _atomic(args.output_dir / "supervisor.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
