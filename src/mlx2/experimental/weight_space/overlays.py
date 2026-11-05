"""In-memory, request-bound quantization experiments on selected weight tensors.

Q1..Q15 mean this module's uniform affine block format, not GGUF or hardware
formats. FP4 is separately named E2M1 with float32 block scales (not NVFP4 or
MXFP4). Inputs are the source provider's effective float32 weights, so an FP8
source is never silently decoded as some other FP8 encoding. No GPU imports.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from mlx2.runtime.weight_space_metrics import (
    clear_weight_space_gauge,
    register_overlay_metrics,
    set_weight_space_gauge,
    track_weight_space,
)

from .programs import _identity

MAX_ELEMENTS = 16 * 1024 * 1024
FP4_VALUES = np.frombuffer(np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32).tobytes(), dtype=np.float32)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _array(value, *, dimensions=None):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "fiu" or raw.size == 0 or raw.size > MAX_ELEMENTS
            or (dimensions is not None and raw.ndim != dimensions)):
        raise ValueError("invalid or oversized numeric tensor")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(raw, dtype="<f4")
    if not np.isfinite(result).all():
        raise ValueError("tensor must contain finite float32 values")
    return np.frombuffer(result.tobytes(), dtype="<f4").reshape(result.shape)


def pack_codes(codes, bits):
    """Little-endian bit packing, including odd widths and partial final bytes."""
    if type(bits) is not int or not 1 <= bits <= 15:
        raise ValueError("quantization bits must be Q1 through Q15")
    values = np.asarray(codes)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or len(values) > MAX_ELEMENTS
            or np.any(values < 0) or np.any(values >= 1 << bits)):
        raise ValueError("code values exceed their bit width")
    chunks = []
    for start in range(0, len(values), 4096):
        block = values[start:start + 4096].astype(np.uint16)
        planes = (block[:, None] >> np.arange(bits, dtype=np.uint16)) & 1
        chunks.append(np.packbits(planes.reshape(-1).astype(np.uint8), bitorder="little").tobytes())
    return b"".join(chunks)


def unpack_codes(packed, bits, count):
    if type(bits) is not int or not 1 <= bits <= 15 or type(count) is not int or not 0 <= count <= MAX_ELEMENTS:
        raise ValueError("invalid packed geometry")
    if not isinstance(packed, bytes) or len(packed) != (count * bits + 7) // 8:
        raise ValueError("packed byte count differs from geometry")
    if count * bits % 8 and packed and packed[-1] >> (count * bits % 8):
        raise ValueError("nonzero padding in packed final byte")
    result = np.empty(count, dtype=np.uint16)
    offset = 0
    for start in range(0, count, 4096):
        size = min(4096, count - start)
        nbytes = (size * bits + 7) // 8
        planes = np.unpackbits(np.frombuffer(packed[offset:offset + nbytes], dtype=np.uint8),
                               bitorder="little")[:size * bits].reshape(size, bits)
        result[start:start + size] = (planes.astype(np.uint16) << np.arange(bits, dtype=np.uint16)).sum(axis=1)
        offset += nbytes
    return result


@dataclass(frozen=True)
class TensorTarget:
    layer: int
    expert: int | None
    projection: str

    def __post_init__(self):
        if type(self.layer) is not int or not 0 <= self.layer <= 10000:
            raise ValueError("invalid layer index")
        if self.expert is not None and (type(self.expert) is not int or not 0 <= self.expert <= 100000):
            raise ValueError("invalid expert index")
        if (not isinstance(self.projection, str) or not self.projection
                or len(self.projection) > 256 or any(c.isspace() for c in self.projection)):
            raise ValueError("explicit projection identity required")

    def to_dict(self):
        return {"layer": self.layer, "expert": self.expert, "projection": self.projection}


@dataclass(frozen=True)
class QuantizationSpec:
    bits: int
    group_size: int = 64
    format: str = "affine_uniform"

    def __post_init__(self):
        if type(self.bits) is not int or not 1 <= self.bits <= 15:
            raise ValueError("quantization bits must be Q1 through Q15")
        if type(self.group_size) is not int or not 1 <= self.group_size <= 1024:
            raise ValueError("group_size must be 1..1024")
        if self.format not in {"affine_uniform", "fp4_e2m1_f32_scale"}:
            raise ValueError("unsupported explicit quantization format")
        if self.format == "fp4_e2m1_f32_scale" and self.bits != 4:
            raise ValueError("E2M1 FP4 requires exactly 4 bits")


@dataclass(frozen=True)
class QuantizedTensor:
    shape: tuple
    spec: QuantizationSpec
    packed: bytes
    scales: np.ndarray
    offsets: np.ndarray
    source_sha256: str

    def __post_init__(self):
        shape = tuple(self.shape)
        if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
                or math.prod(shape) > MAX_ELEMENTS or not isinstance(self.spec, QuantizationSpec)):
            raise ValueError("invalid quantized tensor geometry")
        if (not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_sha256)):
            raise ValueError("source SHA-256 required")
        unpack_codes(self.packed, self.spec.bits, math.prod(shape))
        groups = (math.prod(shape) + self.spec.group_size - 1) // self.spec.group_size
        for key in ("scales", "offsets"):
            value = _array(getattr(self, key), dimensions=1)
            if len(value) != groups or (key == "scales" and np.any(value < 0)):
                raise ValueError("quantization metadata geometry or scale is invalid")
            object.__setattr__(self, key, value)
        object.__setattr__(self, "shape", shape)

    @property
    def nbytes(self):
        return len(self.packed) + self.scales.nbytes + self.offsets.nbytes

    @property
    def fingerprint(self):
        return _digest({"shape": self.shape, "spec": vars(self.spec),
                        "source_sha256": self.source_sha256,
                        "packed": hashlib.sha256(self.packed).hexdigest(),
                        "scales": hashlib.sha256(self.scales.tobytes()).hexdigest(),
                        "offsets": hashlib.sha256(self.offsets.tobytes()).hexdigest()})

    def dequantize(self):
        count = math.prod(self.shape)
        codes = unpack_codes(self.packed, self.spec.bits, count)
        groups = np.arange(count) // self.spec.group_size
        if self.spec.format == "affine_uniform":
            values = codes.astype(np.float64) * self.scales[groups] + self.offsets[groups]
        else:
            values = FP4_VALUES[codes & 7] * np.where(codes & 8, -1.0, 1.0) * self.scales[groups]
        return _array(values.reshape(self.shape))


def quantize_tensor(weights, spec):
    if not isinstance(spec, QuantizationSpec):
        raise TypeError("QuantizationSpec required")
    weights = _array(weights, dimensions=2)
    flat = weights.reshape(-1).astype(np.float64)
    group_size = spec.group_size
    groups = (len(flat) + group_size - 1) // group_size
    scales, offsets = np.empty(groups, dtype=np.float32), np.zeros(groups, dtype=np.float32)
    codes = np.empty(len(flat), dtype=np.uint16)
    levels = (1 << spec.bits) - 1
    for i in range(groups):
        start, stop = i * group_size, min((i + 1) * group_size, len(flat))
        values = flat[start:stop]
        if spec.format == "affine_uniform":
            low, high = float(values.min()), float(values.max())
            scale = (high - low) / levels
            with np.errstate(over="ignore"):
                scales[i], offsets[i] = scale, low
            if not np.isfinite(scales[i]) or not np.isfinite(offsets[i]):
                raise ValueError("quantization metadata overflows float32")
            codes[start:stop] = 0 if scales[i] == 0 else np.clip(
                np.rint((values - float(offsets[i])) / float(scales[i])), 0, levels).astype(np.uint16)
        else:
            scales[i] = float(np.max(np.abs(values))) / 6
            normalized = np.zeros_like(values) if scales[i] == 0 else np.abs(values) / scales[i]
            nearest = np.argmin(np.abs(normalized[:, None] - FP4_VALUES[None, :]), axis=1)
            codes[start:stop] = nearest | (np.signbit(values).astype(np.uint16) << 3)
    result = QuantizedTensor(tuple(weights.shape), spec, pack_codes(codes, spec.bits),
        _array(scales), _array(offsets), hashlib.sha256(weights.tobytes()).hexdigest())
    result.dequantize()  # Validate all rounded metadata before admitting a candidate.
    return result


@dataclass(frozen=True)
class WeightOverride:
    target: TensorTarget
    tensor: QuantizedTensor
    source_format: str
    source_tensor_fingerprint: str
    effective_weight_sha256: str

    @property
    def fingerprint(self):
        return _digest({"target": self.target.to_dict(), "tensor": self.tensor.fingerprint,
                        "source_format": self.source_format,
                        "source_tensor_fingerprint": self.source_tensor_fingerprint,
                        "effective_weight_sha256": self.effective_weight_sha256})


@dataclass(frozen=True, init=False, eq=False)
class OverlayView:
    """Pinned immutable request selection; close when the request terminates."""

    def __init__(self, bank, name, overrides):
        values = {"checkpoint": bank.checkpoint, "source_revision": bank.source_revision,
                  "targets": MappingProxyType({item.target: item for item in overrides}),
                  "fingerprint": bank._fingerprint(overrides), "name": name, "_bank": bank, "_closed": False}
        for key, value in values.items():
            object.__setattr__(self, key, value)

    @property
    def nbytes(self):
        return sum(item.tensor.nbytes for item in self.targets.values())

    def close(self):
        with self._bank._lock:
            if not self._closed:
                self._bank._pins[self.name] -= 1
                object.__setattr__(self, "_closed", True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class InMemoryOverlayBank:
    """Memory-only candidates; no checkpoint writes and no automatic selection.

    The adapter owns source validation. Supplying an external source fingerprint
    requires that validator; otherwise fingerprints hash the effective weights.
    The runtime separately verifies those weights against its loaded tensor.
    """

    def __init__(self, *, checkpoint, source_revision, budget_bytes, source_validator=None,
                 construction_budget_bytes=256 * 1024**2):
        _identity(checkpoint, source_revision)
        if type(budget_bytes) is not int or not 1 <= budget_bytes <= 64 * 1024**3:
            raise ValueError("explicit overlay memory budget required")
        if type(construction_budget_bytes) is not int or not 1 <= construction_budget_bytes <= 64 * 1024**3:
            raise ValueError("explicit overlay construction budget required")
        self.checkpoint, self.source_revision = checkpoint, source_revision
        self.budget_bytes = budget_bytes
        self.construction_budget_bytes = construction_budget_bytes
        self.source_validator = source_validator
        self._entries, self._pins = {}, {}
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()
        self._payload_reserved_bytes = self._construction_reserved_bytes = 0
        self._pending_names = set()
        self._counts = {"created": 0, "selected": 0, "deleted": 0, "refused": 0}
        register_overlay_metrics(self)

    def _fingerprint(self, overrides):
        return _digest({"checkpoint": self.checkpoint, "source_revision": self.source_revision,
                        "overrides": [item.fingerprint for item in overrides]})

    def _validate_name(self, name):
        if (not isinstance(name, str) or not name or len(name) > 128
                or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in name)):
            raise ValueError("overlay name must be a bounded identifier")

    def _resident_bytes(self, extra=()):
        unique = {id(item.tensor): item.tensor.nbytes
                  for items in (*self._entries.values(), extra) for item in items}
        return sum(unique.values())

    def check_capacity(self, name, shape, spec):
        """Cheap admission before source materialization; create rechecks/reserves.

        The construction estimate conservatively charges NumPy input/float64
        scratch/codes/dequantization temporaries plus fixed packing workspace.
        It is an admission model, not a measured process-RSS upper bound.
        """
        self._validate_name(name)
        if (not isinstance(spec, QuantizationSpec) or not isinstance(shape, (tuple, list))
                or len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
                or math.prod(shape) > MAX_ELEMENTS):
            raise ValueError("invalid or oversized quantization source geometry")
        count = math.prod(shape)
        groups = (count + spec.group_size - 1) // spec.group_size
        payload = (count * spec.bits + 7) // 8 + groups * 8
        construction = count * 96 + groups * 32 + 128 * 1024
        with self._lock:
            if name in self._entries or name in self._pending_names or len(self._entries) >= 128:
                raise ValueError("overlay exists, is pending, or registry is full")
            if self._resident_bytes() + self._payload_reserved_bytes + payload > self.budget_bytes:
                self._counts["refused"] += 1
                raise ValueError("overlay exceeds memory budget including quantization metadata")
            if construction > self.construction_budget_bytes:
                self._counts["refused"] += 1
                raise ValueError("overlay exceeds conservative construction workspace budget")
        return {"payload_bytes": payload, "construction_bytes": construction}

    @track_weight_space("quantization_overlay")
    def create(self, name, target, weights, *, source_format, spec, source_tensor_fingerprint=None):
        self._validate_name(name)
        if not isinstance(target, TensorTarget):
            raise TypeError("TensorTarget required")
        if not isinstance(source_format, str) or not source_format or len(source_format) > 128:
            raise ValueError("source provider must declare its actual storage format")
        if source_tensor_fingerprint is not None and self.source_validator is None:
            raise ValueError("external source fingerprint requires an adapter validator")
        # Serialize construction while allowing /metrics and other lightweight
        # registry operations to observe progress without waiting for NumPy.
        with self._build_lock:
            raw = np.asarray(weights)
            with self._lock:
                estimate = self.check_capacity(name, raw.shape, spec)
                self._pending_names.add(name)
                self._payload_reserved_bytes = estimate["payload_bytes"]
                self._construction_reserved_bytes = estimate["construction_bytes"]
            try:
                tensor = quantize_tensor(raw, spec)
                source_fingerprint = source_tensor_fingerprint or tensor.source_sha256
                if not isinstance(source_fingerprint, str) or not source_fingerprint:
                    raise ValueError("invalid source tensor fingerprint")
                if self.source_validator is not None and self.source_validator(target, source_fingerprint) is not True:
                    raise ValueError("source tensor revision changed")
                override = WeightOverride(target, tensor, source_format, source_fingerprint, tensor.source_sha256)
                with self._lock:
                    self._entries[name] = (override,)
                    self._pins[name] = 0
                    self._counts["created"] += 1
            finally:
                with self._lock:
                    self._pending_names.discard(name)
                    self._payload_reserved_bytes = self._construction_reserved_bytes = 0
        return self.inspect(name)

    @track_weight_space("quantization_overlay")
    def compose(self, name, names):
        self._validate_name(name)
        if not isinstance(names, (tuple, list)) or not 1 <= len(names) <= 128:
            raise ValueError("composition requires bounded registered overlay names")
        with self._lock:
            if name in self._entries or name in self._pending_names or len(self._entries) + len(self._pending_names) >= 128:
                raise ValueError("overlay exists or registry is full")
            items = tuple(item for source in names for item in self._entries[source])
            if len({item.target for item in items}) != len(items):
                raise ValueError("multiple overrides for the same target are ambiguous")
            self._entries[name], self._pins[name] = items, 0
        return self.inspect(name)

    @track_weight_space("quantization_overlay")
    def acquire(self, name):
        with self._lock:
            if name not in self._entries:
                raise ValueError("unknown overlay")
            items = self._entries[name]
            if self.source_validator is not None and any(
                    self.source_validator(item.target, item.source_tensor_fingerprint) is not True for item in items):
                raise ValueError("source tensor revision changed")
            # Build the view before pinning: a failed construction (for
            # example MemoryError while fingerprinting) leaves no handle
            # through which a caller could release the pin.
            view = OverlayView(self, name, items)
            self._pins[name] += 1
            self._counts["selected"] += 1
            return view

    @track_weight_space("quantization_overlay")
    def delete(self, name):
        with self._lock:
            if name not in self._entries:
                raise ValueError("unknown overlay")
            if self._pins[name]:
                raise ValueError("overlay is pinned by an active request")
            del self._entries[name], self._pins[name]
            self._counts["deleted"] += 1

    def inspect(self, name):
        with self._lock:
            if name not in self._entries:
                raise ValueError("unknown overlay")
            items = self._entries[name]
            return {"name": name, "checkpoint": self.checkpoint, "source_revision": self.source_revision,
                    "fingerprint": self._fingerprint(items), "in_memory_only": True,
                    "qualified": False, "selected_requests": self._pins[name],
                    "targets": [{**item.target.to_dict(), "source_format": item.source_format,
                                 "source_tensor_fingerprint": item.source_tensor_fingerprint,
                                 "effective_weight_sha256": item.effective_weight_sha256,
                                 "format": item.tensor.spec.format, "bits": item.tensor.spec.bits,
                                 "group_size": item.tensor.spec.group_size,
                                 "packed_bytes": len(item.tensor.packed), "resident_bytes": item.tensor.nbytes}
                                for item in items]}

    def snapshot(self):
        with self._lock:
            return {"checkpoint": self.checkpoint, "source_revision": self.source_revision,
                    "resident_bytes": self._resident_bytes(), "budget_bytes": self.budget_bytes,
                    "construction_reserved_bytes": self._construction_reserved_bytes,
                    "construction_budget_bytes": self.construction_budget_bytes,
                    "payload_reserved_bytes": self._payload_reserved_bytes,
                    "tensors": len({id(item.tensor) for items in self._entries.values() for item in items}),
                    "candidates": len(self._entries), "active_requests": sum(self._pins.values()),
                    "counts": dict(self._counts)}


@track_weight_space("quality_evaluate")
def reconstruction_metrics(reference, candidate):
    reference, candidate = _array(reference), _array(candidate)
    if reference.shape != candidate.shape:
        raise ValueError("reconstruction geometry differs")
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64)))
    mse = float(np.mean(delta**2))
    set_weight_space_gauge("quantization_error_mse", mse)
    return {"mse": mse, "maximum_absolute_error": float(np.max(np.abs(delta))),
            "relative_frobenius_error": float(np.linalg.norm(delta) / norm) if norm else float(np.linalg.norm(delta)),
            "behavioral_quality_measured": False}


@track_weight_space("perplexity_evaluate")
def teacher_forced_quality(logits, next_token_ids, *, tokenizer_fingerprint, input_digest, mask=None):
    """Evaluate aligned next-token logits. No implicit shift or tokenization."""
    logits = _array(logits, dimensions=2).astype(np.float64)
    labels = np.asarray(next_token_ids)
    if (labels.ndim != 1 or len(labels) != len(logits) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= logits.shape[1])):
        raise ValueError("next-token labels must align exactly with supplied logits")
    if any(not isinstance(value, str) or not value for value in (tokenizer_fingerprint, input_digest)):
        raise ValueError("tokenizer and input identities are required")
    keep = np.ones(len(labels), dtype=bool) if mask is None else np.asarray(mask)
    if keep.dtype != np.bool_ or keep.shape != labels.shape or not keep.any():
        raise ValueError("quality mask must select at least one aligned token")
    # Stay in max-shifted space: adding the max back and subtracting the label
    # logit cancels catastrophically for large logits (NLL of [1e16, 1e16] is ln 2).
    shifted = logits - logits.max(axis=1)[:, None]
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    mean = float(nll[keep].mean())
    return {"tokenizer_fingerprint": tokenizer_fingerprint, "input_digest": input_digest,
            "label_digest": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
            "mask_digest": hashlib.sha256(keep.tobytes()).hexdigest(), "tokens": int(keep.sum()),
            "mean_nll_nats": mean, "perplexity": math.exp(mean) if mean < 700 else None,
            "perplexity_overflow": mean >= 700, "alignment": "explicit_next_token_rows",
            "evidence_scope": "supplied_logits_only", "model_execution_verified": False}


@track_weight_space("quality_evaluate")
def compare_quality(reference, candidate):
    for key in ("tokenizer_fingerprint", "input_digest", "label_digest", "mask_digest", "tokens", "alignment"):
        if reference.get(key) != candidate.get(key):
            raise ValueError("quality comparison changed tokenization, inputs, labels or mask")
    result = {"tokens": reference["tokens"],
            "delta_nll_nats": candidate["mean_nll_nats"] - reference["mean_nll_nats"],
            "delta_perplexity": (candidate["perplexity"] - reference["perplexity"])
                if candidate["perplexity"] is not None and reference["perplexity"] is not None else None,
            "reference": reference, "candidate": candidate, "qualified": False}
    set_weight_space_gauge("quality_score_delta", result["delta_nll_nats"])
    if result["delta_perplexity"] is not None:
        set_weight_space_gauge("perplexity_delta", result["delta_perplexity"])
    else:
        clear_weight_space_gauge("perplexity_delta")
    return result
