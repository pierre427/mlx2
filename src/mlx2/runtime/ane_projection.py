"""Adapter-owned fixed-shape Core ML projection contract.

Core ML may partition work between CPU and the Neural Engine even when the GPU
is excluded.  Therefore this contract reports ``cpu_and_ne`` as the allowed
compute set and never upgrades that to an observed Neural Engine claim without
an external per-operation execution report.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "mlx2.ane-fixed-projection.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            total += len(chunk)
            if total > 64 << 20:
                raise ValueError("compiled model file exceeds the byte limit")
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FixedProjection:
    adapter: str
    artifact_sha256: str
    input_width: int
    output_width: int
    batch_rows: int
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter or len(self.adapter) > 256:
            raise ValueError("adapter must be nonempty bounded text")
        if (
            not isinstance(self.artifact_sha256, str)
            or len(self.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.artifact_sha256)
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        for name in ("input_width", "output_width", "batch_rows"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 1_000_000:
                raise ValueError(f"{name} must be an integer in [1, 1000000]")
        if self.dtype != "float16":
            raise ValueError("the initial ANE projection contract is float16 only")

    @property
    def input_bytes(self) -> int:
        return self.batch_rows * self.input_width * 2

    @property
    def output_bytes(self) -> int:
        return self.batch_rows * self.output_width * 2

    @property
    def weight_bytes(self) -> int:
        return self.input_width * self.output_width * 2

    def receipt(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "adapter": self.adapter,
            "artifact_sha256": self.artifact_sha256,
            "shape": [self.batch_rows, self.input_width, self.output_width],
            "dtype": self.dtype,
            "allowed_compute_units": "cpu_and_ne",
            "bytes": {
                "input": self.input_bytes,
                "output": self.output_bytes,
                "weights": self.weight_bytes,
                "minimum_live": self.input_bytes + self.output_bytes + self.weight_bytes,
            },
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }


def compiled_model_digest(model_path: str | Path, *, max_bytes: int = 64 << 20) -> str:
    """Hash a bounded regular-file tree; callers must retain the reviewed bytes."""
    path = Path(model_path).expanduser().absolute()
    if not path.exists() or path.is_symlink():
        raise ValueError("Core ML model must be a present non-symlink path")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 64 << 20:
        raise ValueError("compiled model byte limit must be in [1, 64 MiB]")
    if path.is_file():
        if path.stat().st_size > max_bytes:
            raise ValueError("compiled model exceeds the byte limit")
        actual = _sha256(path)
    else:
        digest = hashlib.sha256()
        entries = []
        for item in path.rglob("*"):
            if len(entries) >= 4096:
                raise ValueError("compiled model tree exceeds the entry limit")
            entries.append(item)
        entries.sort()
        if any(item.is_symlink() or not (item.is_file() or item.is_dir()) for item in entries):
            raise ValueError("compiled model tree may contain only regular files and directories")
        files = [item for item in entries if item.is_file()]
        if not files:
            raise ValueError("compiled model directory must contain regular files only")
        if sum(item.stat().st_size for item in files) > max_bytes:
            raise ValueError("compiled model exceeds the byte limit")
        for item in files:
            relative = item.relative_to(path).as_posix()
            digest.update(relative.encode())
            digest.update(bytes.fromhex(_sha256(item)))
        actual = digest.hexdigest()
    return actual


def bind_compiled_model(
    spec: FixedProjection,
    model_path: str | Path,
    *,
    declared_sha256: str,
) -> dict[str, object]:
    """Describe reviewed bytes. The executor separately retains a private copy."""
    path = Path(model_path).expanduser().absolute()
    if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
        raise ValueError("compiled model digest is invalid")
    actual = compiled_model_digest(path)
    if actual != declared_sha256:
        raise ValueError("compiled model bytes differ from the reviewed identity")
    return {
        **spec.receipt(),
        "compiled_model_path": str(path),
        "compiled_model_sha256": actual,
    }


def verify_projection_output(
    spec: FixedProjection,
    values: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    """Check full fixed-shape output; no sampled-row parity is accepted."""

    for name, value in (("absolute_tolerance", absolute_tolerance),
                        ("relative_tolerance", relative_tolerance)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if len(values) != spec.batch_rows or len(reference) != spec.batch_rows:
        raise ValueError("projection output row count differs from the fixed contract")
    max_abs = 0.0
    max_rel = 0.0
    for actual_row, reference_row in zip(values, reference):
        if len(actual_row) != spec.output_width or len(reference_row) != spec.output_width:
            raise ValueError("projection output width differs from the fixed contract")
        for actual, expected in zip(actual_row, reference_row):
            actual, expected = float(actual), float(expected)
            if not math.isfinite(actual) or not math.isfinite(expected):
                raise ValueError("projection outputs must be finite")
            absolute = abs(actual - expected)
            relative = absolute / max(abs(expected), 1e-12)
            max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
            if absolute > absolute_tolerance + relative_tolerance * abs(expected):
                raise RuntimeError("Core ML projection differs from the ordinary reference")
    return {
        "schema": SCHEMA,
        "parity": True,
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "execution_device_observed": False,
        "note": "CPU_AND_NE excludes GPU but does not by itself prove per-operation Neural Engine placement",
    }


def record_compute_plan(
    receipt: dict[str, object], *, operations: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Record anticipated placement without claiming actual device execution.

    ``MLComputePlan`` describes the device Core ML expects to use. It is not a
    post-prediction trace, so even an all-ANE plan cannot prove observed use.
    """

    if receipt.get("schema") != SCHEMA:
        raise ValueError("foreign ANE projection receipt")
    if not operations:
        raise ValueError("a compute plan must contain operations")
    normalized = []
    for operation in operations:
        if not isinstance(operation, dict):
            raise TypeError("compute plan operation must be an object")
        name = operation.get("name")
        preferred = operation.get("anticipated_device")
        if not isinstance(name, str) or not name:
            raise ValueError("compute plan operation name is invalid")
        if preferred not in {"neural_engine", "cpu", "unknown"}:
            raise ValueError("CPU_AND_NE plan contains an invalid anticipated device")
        normalized.append({"name": name, "anticipated_device": preferred})
    return {
        **receipt,
        "observed_used": False,
        "execution_device_observed": False,
        "compute_plan": normalized,
        "note": "MLComputePlan placement is anticipated, not observed execution",
    }


def _coreml_runtime():
    # Imports are deliberately inside the owned execution path. Merely
    # importing this module must not initialize Core ML, Torch or MLX.
    try:
        import coremltools
    except ImportError as exc:
        raise RuntimeError("Core ML execution requires coremltools in a macOS runtime") from exc
    return coremltools


def _copy_compiled_tree(source: Path, destination: Path) -> None:
    """Copy regular files with a total byte ceiling, then verify at the caller."""
    destination.mkdir()
    remaining = 64 << 20
    for index, item in enumerate(source.rglob("*")):
        if index >= 4096:
            raise ValueError("compiled model tree exceeds the entry limit")
        info = item.lstat()
        target = destination / item.relative_to(source)
        if stat.S_ISDIR(info.st_mode):
            target.mkdir(exist_ok=True)
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("compiled model tree may contain only regular files and directories")
        descriptor = os.open(item, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as reader:
            if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                raise ValueError("compiled model input is no longer a regular file")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as writer:
                while chunk := reader.read(min(1 << 20, remaining + 1)):
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise ValueError("compiled model exceeds the byte limit while copying")
                    writer.write(chunk)
            target.chmod(0o400)


class CoreMLProjectionExecutor:
    """Synchronous fixed-fp16 executor; the caller owns accelerator admission.

    Predictions and close are serialized. Close waits for an in-flight native
    call instead of deleting its model. Native calls have no cancellation API:
    callers needing a deadline must use a supervised process, as the probe does.
    This executor is not selected by any serving adapter automatically.
    """

    def __init__(self, spec: FixedProjection, model_path: str | Path, *,
                 declared_sha256: str, input_name: str = "x", output_name: str = "projection"):
        for name in (input_name, output_name):
            if not isinstance(name, str) or not name or len(name) > 256:
                raise ValueError("projection feature names must be bounded nonempty text")
        if spec.input_bytes + spec.output_bytes + spec.weight_bytes > 64 << 20:
            raise ValueError("projection exceeds the 64 MiB declared tensor budget")
        source = Path(model_path).expanduser().absolute()
        if source.suffix != ".mlmodelc" or not source.is_dir():
            raise ValueError("executor requires a compiled .mlmodelc directory")
        self.spec, self.input_name, self.output_name = spec, input_name, output_name
        self._lock = threading.RLock()
        self._closed = False
        self._predictions = 0
        self._model = None
        self._storage = tempfile.TemporaryDirectory(prefix="mlx2-coreml-")
        self._path = Path(self._storage.name) / "projection.mlmodelc"
        try:
            bind_compiled_model(spec, source, declared_sha256=declared_sha256)
            _copy_compiled_tree(source, self._path)
            self._binding = bind_compiled_model(spec, self._path, declared_sha256=declared_sha256)
            self._runtime = _coreml_runtime()
            self._model = self._runtime.models.CompiledMLModel(
                str(self._path), compute_units=self._runtime.ComputeUnit.CPU_AND_NE,
            )
        except BaseException:
            self._model = None
            self._storage.cleanup()
            raise

    def _require_open(self):
        if self._closed:
            raise RuntimeError("Core ML projection executor is closed")

    def predict(self, values):
        """Return a private fp16 output and a prediction-completion receipt."""
        import numpy as np

        with self._lock:
            self._require_open()
            if not isinstance(values, np.ndarray):
                raise TypeError("projection input must be a float16 ndarray")
            array = values
            if array.dtype != np.float16 or array.shape != (self.spec.batch_rows, self.spec.input_width):
                raise ValueError("projection input must match the fixed float16 shape")
            if not np.isfinite(array).all():
                raise ValueError("projection input must be finite")
            owned = np.array(array, dtype=np.float16, order="C", copy=True)
            input_digest = hashlib.sha256(owned.tobytes()).hexdigest()
            result = self._model.predict({self.input_name: owned})
            if not isinstance(result, dict) or set(result) != {self.output_name}:
                raise ValueError("Core ML prediction returned unexpected output features")
            output = np.asarray(result[self.output_name])
            native_output_dtype = str(output.dtype)
            # Core ML's Python bridge may widen an fp16 model feature to
            # float32. Accept only that lossless transport widening, then
            # return the contract's owned fp16 representation.
            if output.dtype not in (np.float16, np.float32) or output.shape != (
                self.spec.batch_rows,
                self.spec.output_width,
            ):
                raise ValueError("projection output must match the fixed floating shape")
            if not np.isfinite(output).all():
                raise ValueError("projection output must be finite")
            output = np.array(output, dtype=np.float16, order="C", copy=True)
            if not np.isfinite(output).all():
                raise ValueError("projection output overflows the fixed float16 contract")
            output.setflags(write=False)
            # Count only a validated, publishable projection.  A native call
            # that returned malformed data is not a completed prediction for
            # route-status or compute-plan receipts.
            self._predictions += 1
            return output, {
                **self.spec.receipt(),
                "compiled_model_sha256": self._binding["compiled_model_sha256"],
                "coremltools_version": str(self._runtime.__version__),
                "selected": True,
                "prediction_completed": True,
                "prediction_count": self._predictions,
                "native_output_dtype": native_output_dtype,
                "returned_output_dtype": "float16",
                "input_sha256": input_digest,
                "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "execution_device_observed": False,
                "note": "Prediction completed with CPU_AND_NE permission; actual device use was not observed",
            }

    def compute_plan(self) -> dict[str, object]:
        """Inspect anticipated placement of the exact retained compiled tree."""
        with self._lock:
            self._require_open()
            plan = self._runtime.models.MLComputePlan.load_from_path(
                str(self._path), compute_units=self._runtime.ComputeUnit.CPU_AND_NE,
            )
            program = plan.model_structure.program
            if program is None:
                raise ValueError("projection compute plan is not an ML Program")
            operations = []

            def visit(block, prefix, depth=0):
                if depth > 16:
                    raise ValueError("projection compute plan exceeds nesting limit")
                for index, operation in enumerate(block.operations):
                    if len(operations) >= 4096:
                        raise ValueError("projection compute plan exceeds operation limit")
                    name = f"{prefix}/{index}:{operation.operator_name}"
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                    device = None if usage is None else usage.preferred_compute_device
                    device_name = type(device).__name__
                    if device_name == "MLGPUComputeDevice":
                        raise ValueError("CPU_AND_NE compute plan unexpectedly prefers GPU")
                    anticipated = {
                        "MLCPUComputeDevice": "cpu", "MLNeuralEngineComputeDevice": "neural_engine",
                    }.get(device_name, "unknown")
                    operations.append({"name": name, "anticipated_device": anticipated})
                    for child_index, child in enumerate(operation.blocks):
                        visit(child, f"{name}/block{child_index}", depth + 1)

            for name, function in program.functions.items():
                visit(function.block, name)
            return record_compute_plan({
                **self.spec.receipt(),
                "compiled_model_sha256": self._binding["compiled_model_sha256"],
                "selected": True,
                "prediction_completed": self._predictions > 0,
                "prediction_count": self._predictions,
            }, operations=operations)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._model = None
            self._storage.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class AdapterProjection:
    """Adapter-owned Core ML projection with an ordinary reference gate.

    The adapter owns both implementations.  Serving may invoke this binding,
    but it cannot install one by model name or silently use the ordinary arm
    after Core ML fails.  A result is publishable only after the complete
    output agrees with the ordinary reference under the declared tolerance.
    """

    def __init__(
        self,
        executor: CoreMLProjectionExecutor,
        ordinary_reference: Callable[[object], object],
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> None:
        if not isinstance(executor, CoreMLProjectionExecutor):
            raise TypeError("adapter projection requires a CoreMLProjectionExecutor")
        if not callable(ordinary_reference):
            raise TypeError("adapter projection requires an ordinary reference")
        for name, value in (
            ("absolute_tolerance", absolute_tolerance),
            ("relative_tolerance", relative_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        self.executor = executor
        self.spec = executor.spec
        self.ordinary_reference = ordinary_reference
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self._completed = 0
        self._lock = threading.RLock()

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                **self.spec.receipt(),
                "route": "coreml_projection",
                "selected": False,
                "prediction_completed": self._completed > 0,
                "prediction_count": self._completed,
                "ordinary_reference": True,
                "execution_device_observed": False,
            }

    def predict(self, values):
        """Execute Core ML, then gate publication on the ordinary reference."""
        import numpy as np

        with self._lock:
            if not isinstance(values, np.ndarray):
                raise TypeError("projection input must be a float16 ndarray")
            # Retain two independent immutable inputs before either implementation
            # runs.  A native bridge or an adapter reference cannot mutate the
            # other arm's input and manufacture parity.
            native_input = np.array(values, dtype=values.dtype, order="C", copy=True)
            reference_input = np.array(values, dtype=values.dtype, order="C", copy=True)
            native_input.setflags(write=False)
            reference_input.setflags(write=False)
            output, execution = self.executor.predict(native_input)
            reference = np.asarray(self.ordinary_reference(reference_input))
            parity = verify_projection_output(
                self.spec,
                output,
                reference,
                absolute_tolerance=self.absolute_tolerance,
                relative_tolerance=self.relative_tolerance,
            )
            self._completed += 1
            return output, {
                **execution,
                **parity,
                "route": "coreml_projection",
                "adapter_owned": True,
                "selected": True,
                "prediction_completed": True,
                "prediction_count": self._completed,
                "ordinary_reference": True,
                # This field refers specifically to actual Neural Engine use.
                "observed_used": False,
            }
