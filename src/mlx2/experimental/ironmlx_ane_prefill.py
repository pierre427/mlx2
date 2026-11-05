# SPDX-License-Identifier: Apache-2.0
"""CPU-executable dense-prefill channel split, opt-in and unqualified.

The trailing FFN channels use signed block-Hadamard W8A8 projections; the
leading channels preserve the ordinary FP32 reference. Two owned slots let a
caller submit the next half-chunk before collecting the previous one. A single
CPU worker models the serial offload engine while leading work runs on the
caller. See provenance/ironmlx-ane-prefill-20261104.json.

This is an integer-quantization reference and lifecycle implementation. It does
not emulate ANE FP16 rounding, compile MIL, import MLX/Core ML, or execute on a
GPU/ANE. Native backends are deliberately refused until separately implemented.
No state is published to APCv2 or a production model adapter.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Protocol, Self

import numpy as np

SCHEMA = "mlx2.ironmlx-ane-prefill-cpu.v1"
INPUT_SEED = 0x5EED0001
MID_SEED = 0x5EED0002


def _integer(name: str, value: int, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _digest(name: str, value: str, width: int) -> None:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{width}}}", value) is None:
        raise ValueError(f"{name} must be a lowercase {width}-character digest")


@dataclass(frozen=True)
class SplitBinding:
    adapter: str
    checkpoint: str
    source_revision: str
    weights_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, str) or not self.adapter or len(self.adapter) > 256:
            raise ValueError("adapter must be bounded nonempty text")
        if not isinstance(self.checkpoint, str) or "@" not in self.checkpoint:
            raise ValueError("checkpoint must be artifact@fingerprint")
        artifact, fingerprint = self.checkpoint.rsplit("@", 1)
        if not artifact or len(artifact) > 512:
            raise ValueError("checkpoint artifact must be bounded nonempty text")
        _digest("checkpoint fingerprint", fingerprint, 64)
        _digest("source_revision", self.source_revision, 40)
        _digest("weights_sha256", self.weights_sha256, 64)


@dataclass(frozen=True)
class SplitShape:
    hidden: int
    intermediate: int
    offload_channels: int
    half_rows: int
    input_block: int = 1024
    mid_block: int = 256
    max_position: int = 65536

    def __post_init__(self) -> None:
        for name, maximum in (("hidden", 16384), ("intermediate", 262144),
                              ("offload_channels", 10240), ("half_rows", 1024),
                              ("max_position", 1 << 24)):
            _integer(name, getattr(self, name), maximum)
        for name in ("input_block", "mid_block"):
            value = getattr(self, name)
            _integer(name, value, 1024)
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")
        if (self.offload_channels >= self.intermediate
                or self.hidden % self.input_block
                or self.offload_channels % self.mid_block):
            raise ValueError("split channels and rotation blocks do not fit the FFN")

    @property
    def leading_channels(self) -> int:
        return self.intermediate - self.offload_channels

    @property
    def accounted_bytes(self) -> int:
        """Conservative CPU array admission envelope, not allocator/RSS telemetry.

        Includes private FP32 weights, two immutable inputs/results/leading
        outputs and two full reference workspaces. int64 integer products and
        float32 rotation temporaries are included. Python/BLAS allocator caches
        and thread stacks are outside this array-byte estimate.
        """
        h, i, a, r = self.hidden, self.intermediate, self.offload_channels, self.half_rows
        weights = 3 * h * i * 4
        per_slot = r * (h * 48 + i * 32 + a * 32) + 3 * h * a * 32
        return weights + 2 * per_slot


def _matrix(value: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    # Do not accept arbitrary array protocols: a device array conversion could
    # synchronize/dispatch before this CPU-only contract can refuse it.
    if type(value) is not np.ndarray or value.dtype != np.float32:
        raise ValueError("CPU input must be a NumPy float32 ndarray")
    if value.ndim != 2 or (shape is not None and value.shape != shape):
        raise ValueError("CPU array shape differs from the split contract")
    if not np.isfinite(value).all():
        raise ValueError("CPU array must be finite")
    return value


def weight_digest(gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, array in (("gate", gate), ("up", up), ("down", down)):
        array = _matrix(array)
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        # Bounded row materialization also supports strided source weights;
        # hashing must not allocate a second matrix-sized byte string.
        for row in array:
            digest.update(np.ascontiguousarray(row, dtype="<f4").view(np.uint8))
    return digest.hexdigest()


def _owned(array: np.ndarray) -> np.ndarray:
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FfnWeights:
    gate: np.ndarray
    up: np.ndarray
    down: np.ndarray

    @classmethod
    def capture(cls, shape: SplitShape, binding: SplitBinding, gate: np.ndarray,
                up: np.ndarray, down: np.ndarray) -> FfnWeights:
        _matrix(gate, (shape.intermediate, shape.hidden))
        _matrix(up, (shape.intermediate, shape.hidden))
        _matrix(down, (shape.hidden, shape.intermediate))
        # Hash the private copies, so caller mutation cannot change a job after
        # binding. The copied arrays remain held until every worker completes.
        weights = cls(*map(_owned, (gate, up, down)))
        if weight_digest(weights.gate, weights.up, weights.down) != binding.weights_sha256:
            raise ValueError("weights differ from the bound content identity")
        return weights


def _signs(width: int, seed: int) -> np.ndarray:
    flips = []
    for index in range(width):
        value = index ^ seed
        value ^= value >> 16
        value = (value * 0x85EBCA6B) & 0xFFFFFFFF
        value ^= value >> 13
        value = (value * 0xC2B2AE35) & 0xFFFFFFFF
        value ^= value >> 16
        flips.append(-1 if value & 1 else 1)
    return np.asarray(flips, dtype=np.float32)


def hadamard_rows(values: np.ndarray, block: int, seed: int) -> np.ndarray:
    """Apply the same orthogonal sign/rotation to each row without a dense H."""
    values = _matrix(values)
    _integer("block", block, 1024)
    if block & (block - 1) or values.shape[1] % block:
        raise ValueError("rotation block must be a dividing power of two")
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("rotation seed must fit uint32")
    rotated = (values * _signs(values.shape[1], seed)).reshape(-1, block)
    stride = 1
    while stride < block:
        groups = rotated.reshape(-1, block // (2 * stride), 2, stride)
        left, right = groups[:, :, 0, :].copy(), groups[:, :, 1, :].copy()
        groups[:, :, 0, :] = left + right
        groups[:, :, 1, :] = left - right
        stride *= 2
    rotated *= np.float32(1 / math.sqrt(block))
    if not np.isfinite(rotated).all():
        raise ValueError("rotation overflow")
    return rotated.reshape(values.shape)


@dataclass(frozen=True)
class QuantizedRows:
    codes: np.ndarray
    scales: np.ndarray


def quantize_rotated_rows(values: np.ndarray, block: int, seed: int,
                          *, peak_floor: float = 0.0) -> QuantizedRows:
    if (isinstance(peak_floor, bool) or not isinstance(peak_floor, (int, float))
            or not math.isfinite(peak_floor) or peak_floor < 0):
        raise ValueError("peak_floor must be finite and nonnegative")
    rotated = hadamard_rows(values, block, seed)
    peak = np.maximum(np.max(np.abs(rotated), axis=1), np.float32(peak_floor))
    scales = peak / np.float32(127)
    divisor = np.where(scales > 0, scales, np.float32(1))
    codes = np.clip(np.rint(rotated / divisor[:, None]), -127, 127).astype(np.int8)
    return QuantizedRows(codes, scales)


def _quantized_dot(left: QuantizedRows, right: QuantizedRows) -> np.ndarray:
    # int64 prevents accidental overflow in the independent CPU oracle. Native
    # FP16 ANE accumulation/scaling still requires a separate numerical oracle.
    products = left.codes.astype(np.int64) @ right.codes.astype(np.int64).T
    return products.astype(np.float32) * left.scales[:, None] * right.scales[None, :]


def _silu(values: np.ndarray) -> np.ndarray:
    exp = np.exp(-np.abs(values))
    sigmoid = np.where(values >= 0, 1 / (1 + exp), exp / (1 + exp))
    return values * sigmoid


def ordinary_ffn(inputs: np.ndarray, weights: FfnWeights) -> np.ndarray:
    inputs = _matrix(inputs)
    return (_silu(inputs @ weights.gate.T) * (inputs @ weights.up.T)) @ weights.down.T


class SplitBackend(Protocol):
    """Offload-half backend. Native implementations are not admitted yet."""

    name: str
    execution_device: str

    def execute(self, shape: SplitShape, inputs: np.ndarray,
                weights: FfnWeights, cancelled: threading.Event) -> np.ndarray: ...


class CpuW8A8Backend:
    name = "numpy-w8a8-reference"
    execution_device = "cpu"

    def execute(self, shape: SplitShape, inputs: np.ndarray,
                weights: FfnWeights, cancelled: threading.Event) -> np.ndarray:
        if cancelled.is_set():
            raise CancelledError()
        start = shape.leading_channels
        x = quantize_rotated_rows(inputs, shape.input_block, INPUT_SEED)
        gate = quantize_rotated_rows(weights.gate[start:], shape.input_block, INPUT_SEED)
        up = quantize_rotated_rows(weights.up[start:], shape.input_block, INPUT_SEED)
        hidden = _silu(_quantized_dot(x, gate)) * _quantized_dot(x, up)
        if cancelled.is_set():
            raise CancelledError()
        mid = quantize_rotated_rows(hidden, shape.mid_block, MID_SEED, peak_floor=1 / 512)
        down = quantize_rotated_rows(weights.down[:, start:], shape.mid_block, MID_SEED)
        return _quantized_dot(mid, down)


@dataclass(frozen=True)
class PendingSplit:
    owner: str
    slot: int
    generation: int


@dataclass
class _Job:
    handle: PendingSplit
    inputs: np.ndarray
    cancelled: threading.Event = field(default_factory=threading.Event)
    prepared: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    leading: np.ndarray | None = None
    output: np.ndarray | None = None
    error: BaseException | None = None


class SplitPrefillExecutor:
    """Explicitly selected CPU experiment, bounded to two uncollected jobs.

    Cancellation does not make a queued/running slot reusable. Its worker must first
    stop; late results are discarded. Timeout or backend failure retires this
    executor. Closing waits for completion by default; ``wait=False`` keeps
    owned inputs alive in the worker rather than reusing their memory.
    """

    def __init__(self, shape: SplitShape, binding: SplitBinding, gate: np.ndarray,
                 up: np.ndarray, down: np.ndarray, *, enabled: bool = False,
                 array_budget_bytes: int, backend: SplitBackend | None = None):
        if type(enabled) is not bool or not enabled:
            raise ValueError("experimental split requires explicit enabled=True")
        _integer("array_budget_bytes", array_budget_bytes, 1 << 50)
        if shape.accounted_bytes > array_budget_bytes:
            raise MemoryError("split array admission envelope exceeds budget")
        backend = CpuW8A8Backend() if backend is None else backend
        if getattr(backend, "execution_device", None) != "cpu":
            raise ValueError("native GPU/ANE backend is not implemented or admitted")
        self.shape, self.binding, self.backend = shape, binding, backend
        self._weights = FfnWeights.capture(shape, binding, gate, up, down)
        self._owner = uuid.uuid4().hex
        self._generations = [0, 0]
        self._slots: list[_Job | None] = [None, None]
        self._lock = threading.RLock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx2-cpu-split")
        self._closed = False
        self._retired = False
        self._completed = 0

    def receipt(self) -> dict:
        with self._lock:
            return {
                "schema": SCHEMA, "adapter": self.binding.adapter,
                "checkpoint": self.binding.checkpoint,
                "source_revision": self.binding.source_revision,
                "weights_sha256": self.binding.weights_sha256,
                "implemented": True, "implementation_scope": "cpu_reference_and_lifecycle",
                "qualified": False, "selected": True, "observed_used": False,
                "execution_device_observed": False, "backend": self.backend.name,
                "backend_execution_device": "cpu", "ane_backend_implemented": False,
                "approximate": True, "apcv2_state_publication": False,
                "accounted_array_bytes": self.shape.accounted_bytes,
                "array_accounting_scope": "arrays only; excludes BLAS caches and thread stacks",
                "active_slots": sum(job is not None for job in self._slots),
                "completed_cpu_jobs": self._completed,
                "retired": self._retired, "closed": self._closed,
            }

    def _job(self, handle: PendingSplit) -> _Job:
        if (not isinstance(handle, PendingSplit) or handle.owner != self._owner
                or type(handle.slot) is not int or not 0 <= handle.slot < 2):
            raise ValueError("foreign pending split")
        job = self._slots[handle.slot]
        if job is None or job.handle != handle:
            raise ValueError("stale pending split generation")
        return job

    def _run(self, job: _Job) -> None:
        try:
            with self._lock:
                if job.cancelled.is_set() or self._closed or self._retired:
                    return
            output = self.backend.execute(self.shape, job.inputs, self._weights, job.cancelled)
            output = _matrix(output, job.inputs.shape)
            with self._lock:
                if not job.cancelled.is_set() and not self._closed and not self._retired:
                    job.output = _owned(output)
        except CancelledError:
            pass
        except BaseException as error:  # noqa: BLE001 -- every failed worker must retire its owned slots.
            with self._lock:
                self._retired = True
                job.error = error

    def _release_cancelled(self, job: _Job) -> None:
        with self._lock:
            if job.cancelled.is_set() and self._slots[job.handle.slot] is job:
                job.output = None
                job.leading = None
                job.future = None
                self._slots[job.handle.slot] = None

    def submit(self, inputs: np.ndarray, *, binding: SplitBinding,
               start_position: int = 0) -> PendingSplit:
        if binding != self.binding:
            raise ValueError("split identity changed")
        inputs = _matrix(inputs)
        if (inputs.shape[1] != self.shape.hidden
                or not 1 <= inputs.shape[0] <= self.shape.half_rows):
            raise ValueError("input exceeds fixed half-chunk shape")
        if (type(start_position) is not int or start_position < 0
                or start_position + inputs.shape[0] > self.shape.max_position):
            raise ValueError("split position lies outside the admitted prefill range")
        with self._lock:
            if self._closed or self._retired:
                raise RuntimeError("split executor is closed or retired")
            slot = next((i for i, job in enumerate(self._slots) if job is None), None)
            if slot is None:
                raise RuntimeError("both split slots are still owned")
            self._generations[slot] += 1
            handle = PendingSplit(self._owner, slot, self._generations[slot])
            job = _Job(handle, _owned(inputs))
            self._slots[slot] = job
            job.future = self._worker.submit(self._run, job)
            job.future.add_done_callback(lambda _: self._release_cancelled(job))
        # Start the offload side before computing the leading channels. These
        # computations are CPU operations in this experiment; no device alias.
        try:
            n = self.shape.leading_channels
            hidden = _silu(job.inputs @ self._weights.gate[:n].T)
            hidden *= job.inputs @ self._weights.up[:n].T
            job.leading = hidden @ self._weights.down[:, :n].T
        except BaseException:
            with self._lock:
                if self._slots[slot] is job:
                    self.cancel(handle)
            raise
        finally:
            job.prepared.set()
        with self._lock:
            if self._closed:
                raise RuntimeError("split executor closed during leading preparation")
        return handle

    def cancel(self, handle: PendingSplit) -> None:
        with self._lock:
            job = self._job(handle)
            job.cancelled.set()
            job.output = None
            future = job.future
            assert future is not None
            # Do not cancel a queued Future: ThreadPoolExecutor retains the
            # cancelled work item's input until dequeue. Reusing that slot
            # could then enqueue unbounded retained arrays behind a slow job.
            # Let the worker consume its cancellation event and release it.
            if future.done():
                self._release_cancelled(job)

    def finish(self, handle: PendingSplit, *, timeout: float | None = None) -> np.ndarray:
        if timeout is not None and (isinstance(timeout, bool)
                                    or not isinstance(timeout, (float, int))
                                    or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        with self._lock:
            job = self._job(handle)
            future = job.future
        assert future is not None
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            with self._lock:
                self._retired = True
            self.cancel(handle)
            raise
        with self._lock:
            if job.error is not None:
                error = job.error
                self.cancel(handle)
                raise RuntimeError("split backend failed; executor retired") from error
            if self._retired or self._closed or job.cancelled.is_set():
                raise RuntimeError("split result cannot publish after cancellation or retirement")
            if self._slots[handle.slot] is not job or job.output is None or job.leading is None:
                raise RuntimeError("split result is missing or already collected")
            output = job.leading + job.output
            if not np.isfinite(output).all():
                self._retired = True
                self.cancel(handle)
                raise ValueError("split output is nonfinite")
            self._slots[handle.slot] = None
            job.output = None
            job.leading = None
            job.future = None
            self._completed += 1
            return output

    def close(self, *, wait: bool = True) -> None:
        if type(wait) is not bool:
            raise ValueError("wait must be boolean")
        with self._lock:
            self._closed = True
            jobs = [job for job in self._slots if job is not None]
            for job in jobs:
                if job is not None:
                    job.cancelled.set()
                    job.output = None
                    if job.future is not None:
                        job.future.cancel()
            self._slots = [None, None]
        self._worker.shutdown(wait=wait, cancel_futures=True)
        if wait:
            # CPU leading work is issued by submit's caller, not the offload
            # worker. A draining close must wait for both sides to stop.
            for job in jobs:
                job.prepared.wait()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
