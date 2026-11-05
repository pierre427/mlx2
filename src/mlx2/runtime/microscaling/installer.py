# SPDX-License-Identifier: MIT
"""Instance-scoped, opt-in SC-MXFP4 W4A8 prefill installation.

No runtime requantization: this installer borrows standard MXFP4 weight words
and losslessly re-encodes their E8M0 scale metadata. The activation quantizer
is approximate relative to ordinary W4A16, so installed routes require their
own APCv2 semantic identity. Decode and verification retain original calls.
See provenance/integer-exact-microscaling.json for the mined MIT mechanism.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import json
import math
import threading
from typing import Any
import weakref

from .format import SC_FORMAT, metadata_from_mxfp4_scales_numpy

SCHEMA = "mlx2.microscaling-prefill.v1"
SCOPES = ("experts", "mlp", "all")
DEFAULT_ROW_THRESHOLD = 512
_STATE_ATTR = "_mlx2_microscaling_prefill"
_ROOT_ATTR = "_mlx2_microscaling_root"


class MicroscalingError(RuntimeError):
    """An explicitly requested SC-MXFP4 route cannot be honored."""


@dataclass(frozen=True)
class MicroscalingPolicy:
    # Keep opt-in: separate quantize/unpack calls did not improve warmed
    # Qwen3-0.6B SC-MLP TTFT at 512/2048 in the bounded 2026-11-03 probe.
    # Qwen3-30B-A3B expert A8 was also slower at prefill steps 512/2048;
    # the positive dense 1.7B lead does not generalize to these grouped cells.
    # Integer matrix throughput alone does not establish a model speedup;
    # see docs/research/sc-mxfp4-implementation-20261103/README.md.
    enabled: bool = False
    scope: str = "experts"
    row_threshold: int = DEFAULT_ROW_THRESHOLD

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("microscaling enabled must be a boolean")
        if self.scope not in SCOPES:
            raise ValueError(f"microscaling scope must be one of {SCOPES}")
        if type(self.row_threshold) is not int or self.row_threshold < 2:
            raise ValueError("microscaling row_threshold must be an integer >= 2")

    @classmethod
    def from_value(cls, value):
        if value is None or value is False or value == "off":
            return cls()
        if isinstance(value, cls):
            return value
        if value is True:
            return cls(enabled=True)
        if isinstance(value, str):
            if value not in SCOPES:
                raise ValueError(f"microscaling must be 'off' or one of {SCOPES}")
            return cls(enabled=True, scope=value)
        if isinstance(value, Mapping):
            unknown = set(value) - {"enabled", "scope", "row_threshold"}
            if unknown:
                raise ValueError(f"unknown microscaling policy keys: {sorted(unknown)}")
            return cls(**dict(value))
        raise ValueError("microscaling policy must be off, a scope, or a mapping")

    def as_dict(self):
        return {"enabled": self.enabled, "scope": self.scope,
                "row_threshold": self.row_threshold}

    @property
    def fidelity(self):
        from ...contracts import Fidelity
        return Fidelity.APPROXIMATE if self.enabled else Fidelity.EXACT

    @property
    def revision(self):
        from . import kernels
        descriptor = {"schema": SCHEMA, **self.as_dict(),
            "format": repr(SC_FORMAT), "kernels": kernels.KERNEL_REVISION,
            "activation_law": "per-row-symmetric-int8-rint-amax127-v1",
            "phase_law": "model-prefill-only-no-verify-or-stepped-s1-cache-v2"}
        return _digest(descriptor)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def validate_decode_row_bound(policy, max_decode_rows: int):
    """Reject thresholds that could admit a decode/verify forward.

    For switched projections the caller supplies the conservative expanded
    assignment bound (target rows times maximum expert top-k), not just lanes.
    The separate phase gate remains required even when this check passes.
    """
    policy = MicroscalingPolicy.from_value(policy)
    if type(max_decode_rows) is not int or max_decode_rows < 0:
        raise ValueError("max_decode_rows must be a nonnegative integer")
    if policy.enabled and policy.row_threshold <= max_decode_rows:
        raise MicroscalingError(f"microscaling row_threshold {policy.row_threshold} "
            f"does not exceed decode/verify assignment bound {max_decode_rows}")


def require_supported_device():
    from ..int8_prefill import require_supported_device as require
    return require()


def _module_classes():
    from mlx import nn
    from ..models.switch_layers import QuantizedSwitchLinear
    return nn.QuantizedLinear, QuantizedSwitchLinear


def default_select(scope: str):
    if scope not in SCOPES:
        raise ValueError(f"unknown microscaling scope {scope!r}")
    from ..int8_prefill import EXCLUDED_SEGMENTS, MLP_SEGMENTS
    # The affine installer excludes routed experts because it has no grouped
    # primitive. This installer explicitly owns that operation.
    excluded = EXCLUDED_SEGMENTS - {"switch_mlp", "experts"}
    def select(path, module):
        segments = path.split(".")
        if any(s in excluded for s in segments):
            return False
        if segments[-1] in {"gate", "router", "shared_expert_gate"}:
            return False
        if scope == "experts":
            _, switch_cls = _module_classes()
            return isinstance(module, switch_cls)
        return scope == "all" or any(s in MLP_SEGMENTS for s in segments)
    return select


@dataclass(frozen=True)
class _Spec:
    path: str
    kind: str
    n: int
    k: int
    experts: int = 1


def module_spec(path, module):
    """Pure geometry/dtype eligibility, without evaluation or device queries."""
    import mlx.core as mx
    dense_cls, switch_cls = _module_classes()
    if isinstance(module, switch_cls):
        kind, rank = "grouped", 3
    elif isinstance(module, dense_cls):
        kind, rank = "dense", 2
    else:
        return None, "unsupported module type"
    if getattr(module, "mode", "affine") != "mxfp4":
        return None, "requires standard mxfp4 weights; runtime conversion is forbidden"
    if int(module.bits) != 4 or int(module.group_size) != 32:
        return None, "requires 4-bit/32-element MXFP4 blocks"
    weight, scales = module["weight"], module["scales"]
    if weight.ndim != rank or scales.ndim != rank:
        return None, "packed weight/scale rank mismatch"
    if weight.dtype != mx.uint32 or scales.dtype != mx.uint8:
        return None, "requires uint32 codes and uint8 E8M0 scales"
    if module.get("biases") is not None:
        return None, "MXFP4 cannot carry affine quantization biases"
    n, k = int(weight.shape[-2]), int(weight.shape[-1]) * 8
    experts = int(weight.shape[0]) if rank == 3 else 1
    if n <= 0 or n % 64 or k <= 0 or k % 128:
        return None, "requires positive N multiple64 and K multiple128"
    if k > SC_FORMAT.max_reduction:
        return None, "reduction exceeds SC-MXFP4 INT32 bound"
    if kind == "grouped" and not 1 <= experts <= 1024:
        return None, "grouped expert count must be in 1..1024"
    if tuple(scales.shape) != (*weight.shape[:-1], k // 32):
        return None, "packed scale geometry mismatch"
    return _Spec(path, kind, n, k, experts), ""


@dataclass
class _Bound:
    module: Any
    base: type
    spec: _Spec
    weight: Any
    source_scales: Any
    bcodes: Any
    row_scales: Any
    metadata_bytes: int
    metadata_fingerprint: str


@lru_cache(maxsize=None)
def _projection_subclass(base, grouped):
    base_call = base.__call__
    if grouped:
        def __call__(self, x, indices, sorted_indices=False):
            reference = self.__dict__.get(_STATE_ATTR)
            handle = reference() if reference is not None else None
            if handle is None:
                return base_call(self, x, indices, sorted_indices=sorted_indices)
            return handle._call(self, x, base_call, indices, sorted_indices)
    else:
        def __call__(self, x):
            reference = self.__dict__.get(_STATE_ATTR)
            handle = reference() if reference is not None else None
            if handle is None:
                return base_call(self, x)
            return handle._call(self, x, base_call)
    return type(f"Microscaling{base.__name__}", (base,),
        {"__call__": __call__, "_mlx2_microscaling_base": base})


@lru_cache(maxsize=None)
def _model_subclass(base):
    base_call = base.__call__
    parameters = list(inspect.signature(base_call).parameters.values())[1:]
    cache_position = next((i for i, p in enumerate(parameters)
        if p.name == "cache" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)), None)
    def __call__(self, *args, **kwargs):
        reference = self.__dict__.get(_ROOT_ATTR)
        handle = reference() if reference is not None else None
        if handle is None or not handle.active or handle._prefill.get() is not None:
            return base_call(self, *args, **kwargs)
        inputs = args[0] if args else kwargs.get("inputs", kwargs.get("input_ids", kwargs.get("x")))
        cache = kwargs.get("cache")
        if cache is None and cache_position is not None and len(args) > cache_position:
            cache = args[cache_position]
        enabled = _prefill_reason(inputs, cache) is None
        with handle.prefill_scope(enabled):
            return base_call(self, *args, **kwargs)
    return type(f"MicroscalingModel{base.__name__}", (base,),
        {"__call__": __call__, "_mlx2_microscaling_model_base": base})


def _prefill_reason(inputs, cache):
    from ..models.invariant_prefill import prefill_decline_reason
    reason = prefill_decline_reason(inputs, cache)
    if reason is not None:
        return reason
    if cache is not None:
        # Segmented caches prepare both long prefills and padded decode.
        # Host lengths distinguish the latter without reading a device array.
        # Decline mixed S1/long rows too: the S1 lane may be ordinary decode.
        for entry in cache:
            lengths = getattr(entry, "_step_lengths", None)
            if lengths is None:
                continue
            if (not isinstance(lengths, (list, tuple))
                or len(lengths) != int(inputs.shape[0])
                or any(type(n) is not int or n < 0 or n > int(inputs.shape[1]) for n in lengths)
                or max(lengths, default=0) != int(inputs.shape[1])):
                return "unverified_stepped_cache"
            if any(n == 1 for n in lengths) or not any(n > 1 for n in lengths):
                return "stepped_decode_cache"
    return None


class MicroscalingHandle:
    def __init__(self, policy, *, artifact_fingerprint="", device=""):
        self.policy = MicroscalingPolicy.from_value(policy)
        self.artifact_fingerprint = artifact_fingerprint
        self.device = device
        self.active = False
        self._bound = {}
        self._skipped = {}
        self._root = None
        self._root_base = None
        self._scope_bound = []
        self._workspace_max_prefill_rows = None
        self._workspace_expert_multiplier = 1
        self._lock = threading.RLock()
        self._prefill = ContextVar(f"mlx2_microscaling_prefill_{id(self)}", default=None)
        self.counts = {key: 0 for key in ("engaged_calls", "engaged_rows", "dense_calls",
            "grouped_calls", "fallback_phase", "fallback_rows", "fallback_dtype",
            "fallback_geometry", "fallback_inactive")}

    @contextmanager
    def prefill_scope(self, enabled=True):
        """Explicit component boundary; serving normally uses the model wrapper."""
        token = self._prefill.set(bool(enabled))
        try:
            yield
        finally:
            self._prefill.reset(token)

    def _count(self, key, amount=1):
        with self._lock:
            self.counts[key] += amount

    def _call(self, module, x, base_call, indices=None, sorted_indices=False):
        def ordinary():
            return base_call(module, x) if indices is None else base_call(module, x,
                indices, sorted_indices=sorted_indices)
        bound = self._bound.get(id(module))
        if not self.active or bound is None:
            self._count("fallback_inactive")
            return ordinary()
        if module["weight"] is not bound.weight or module["scales"] is not bound.source_scales:
            raise MicroscalingError(f"{bound.spec.path}: revision-bound weight/scale tensors changed")
        if self._prefill.get() is not True:
            self._count("fallback_phase")
            return ordinary()
        rows = int(indices.size) if indices is not None else math.prod(x.shape[:-1])
        if self._workspace_max_prefill_rows is not None:
            bound_rows = self._workspace_max_prefill_rows
            if indices is not None:
                bound_rows *= self._workspace_expert_multiplier
            if rows > bound_rows:
                raise MicroscalingError(f"{bound.spec.path}: activation rows {rows} exceed "
                    f"the admitted workspace bound {bound_rows}")
        if rows < self.policy.row_threshold:
            self._count("fallback_rows")
            return ordinary()
        import mlx.core as mx
        if x.dtype not in (mx.bfloat16, mx.float16):
            self._count("fallback_dtype")
            return ordinary()
        from . import kernels
        try:
            if indices is None:
                kernels.validate_dense_geometry(x.shape, bound.weight.shape,
                    bound.bcodes.shape, bound.row_scales.shape)
            else:
                kernels.validate_grouped_geometry(x.shape, indices.shape, bound.weight.shape,
                    bound.bcodes.shape, bound.row_scales.shape)
        except ValueError:
            self._count("fallback_geometry")
            return ordinary()
        if indices is None:
            y = kernels.dense(x, bound.weight, bound.bcodes, bound.row_scales)
            if module.get("bias") is not None:
                y = y + module["bias"]
            self._count("dense_calls")
        else:
            y = kernels.grouped(x, indices, bound.weight, bound.bcodes,
                bound.row_scales, sorted_indices=sorted_indices)
            if module.get("bias") is not None:
                y = y + mx.expand_dims(module["bias"][indices], -2)
            self._count("grouped_calls")
        self._count("engaged_calls")
        self._count("engaged_rows", rows)
        return y

    @property
    def revision(self):
        return _digest({"policy": self.policy.revision,
            "artifact": self.artifact_fingerprint,
            "projections": [(b.spec.path, b.spec.kind, b.spec.experts, b.spec.n,
                b.spec.k, b.metadata_fingerprint) for b in self._bound.values()]})

    def metadata_bytes(self):
        return sum(b.metadata_bytes for b in self._bound.values())

    def transient_weight_bytes_max(self):
        return max((b.spec.n * b.spec.k for b in self._bound.values()
            if b.spec.kind == "dense"), default=0)

    def weight_copy_bytes(self):
        """Dense per-call scratch only; grouped weights are never copied to INT8."""
        return self.transient_weight_bytes_max()

    def configure_workspace_bound(self, max_prefill_rows, expert_multiplier=1):
        """Bind the serving shape before admission checks or first dispatch.

        Public component installs may omit this bound. Serving supplies its
        physical prefill step times lane count and the adapter's expert top-k.
        Calls exceeding the admitted shape fail closed rather than allocating
        unaccounted temporary buffers or changing their numerical route.
        """
        if not self.active:
            raise MicroscalingError("workspace bounds require an active install")
        if type(max_prefill_rows) is not int or max_prefill_rows < 1:
            raise ValueError("max_prefill_rows must be a positive integer")
        if type(expert_multiplier) is not int or expert_multiplier < 1:
            raise ValueError("expert_multiplier must be a positive integer")
        self._workspace_max_prefill_rows = max_prefill_rows
        self._workspace_expert_multiplier = expert_multiplier

    def workspace_reserve_bytes(self):
        """Conservative maximum live scratch for one bound projection call.

        Includes padding, dynamic activation codes/scales, dense INT8 unpack,
        grouped broadcast/sort/concatenation copies and inverse-gather output.
        Ordinary activation storage is owned by the model's forward budget.
        """
        rows = self._workspace_max_prefill_rows or self.policy.row_threshold
        largest = 0
        for bound in self._bound.values():
            n, k = bound.spec.n, bound.spec.k
            if bound.spec.kind == "dense":
                padded = (rows + 127) // 128 * 128
                size = n * k + padded * k + padded * 4 + padded * n * 2
            else:
                assignments = rows * self._workspace_expert_multiplier
                size = (3 * (assignments + 64) * k + 64 * assignments
                    + 4 * assignments * n + 12 * (bound.spec.experts + 1))
            largest = max(largest, size)
        return largest

    def unfilled_reserve_bytes(self):
        # Count metadata conservatively even when its lazy MLX arrays have
        # already materialized; this never subtracts unproven allocator reuse.
        return self.metadata_bytes() + self.workspace_reserve_bytes() if self.active else 0

    def status(self):
        from . import kernels
        with self._lock:
            counts = dict(self.counts)
        kinds = {kind: sum(b.spec.kind == kind for b in self._bound.values())
            for kind in ("dense", "grouped")}
        return {"schema": SCHEMA, **self.policy.as_dict(), "active": self.active,
            "fidelity": self.policy.fidelity.value, "qualified": False,
            "revision": self.revision if self.policy.enabled else None,
            "kernel_revision": kernels.KERNEL_REVISION, "device": self.device,
            "artifact_fingerprint": self.artifact_fingerprint,
            "modules": len(self._bound), "module_kinds": kinds,
            "projection_paths": [b.spec.path for b in self._bound.values()],
            "skipped": dict(self._skipped), "counts": counts,
            "metadata_bytes": self.metadata_bytes(), "resident_int8_weight_bytes": 0,
            "transient_weight_bytes_max": self.transient_weight_bytes_max(),
            "weight_copy_bytes": self.weight_copy_bytes(),
            "workspace_max_prefill_rows": self._workspace_max_prefill_rows,
            "workspace_expert_multiplier": self._workspace_expert_multiplier,
            "workspace_reserve_bytes": self.workspace_reserve_bytes(),
            "unfilled_reserve_bytes": self.unfilled_reserve_bytes(),
            "engaged": counts["engaged_calls"] > 0}

    def receipt(self):
        if not self.policy.enabled:
            return None
        status = self.status()
        return {k: status[k] for k in ("schema", "enabled", "scope", "row_threshold",
            "active", "fidelity", "qualified", "revision", "kernel_revision", "modules",
            "module_kinds", "counts", "engaged", "metadata_bytes", "weight_copy_bytes")}


def apply(model, policy, *, artifact_fingerprint="", select: Callable | None = None,
          max_decode_rows=0):
    """Install on eligible model instances atomically, borrowing weight words."""
    policy = MicroscalingPolicy.from_value(policy)
    handle = MicroscalingHandle(policy, artifact_fingerprint=artifact_fingerprint)
    if not policy.enabled:
        return handle
    if not isinstance(artifact_fingerprint, str) or not artifact_fingerprint.strip():
        raise MicroscalingError("microscaling requires the target artifact fingerprint")
    validate_decode_row_bound(policy, max_decode_rows)
    named = list(model.named_modules())
    if _ROOT_ATTR in model.__dict__:
        raise MicroscalingError("model already carries a microscaling install")
    for path, module in named:
        if _ROOT_ATTR in module.__dict__:
            raise MicroscalingError(f"{path}: nested model already carries a microscaling install")
        if _STATE_ATTR in module.__dict__ or hasattr(type(module), "_mlx2_microscaling_base"):
            raise MicroscalingError(f"{path}: model already carries a microscaling install")
    chooser = select or default_select(policy.scope)
    aliases = {}
    for path, module in named:
        _, selected, excluded = aliases.setdefault(id(module), (module, [], []))
        (selected if chooser(path, module) else excluded).append(path)
    candidates = []
    for module, selected, excluded in aliases.values():
        if not selected:
            continue
        path = selected[0]
        spec, reason = module_spec(path, module)
        if spec is None:
            if reason != "unsupported module type":
                handle._skipped[path] = reason
            continue
        if excluded:
            raise MicroscalingError(f"{path}: alias {excluded[0]} lies outside selected scope")
        if hasattr(type(module), "_mlx2_int8_prefill_base"):
            raise MicroscalingError(f"{path}: conflicts with installed affine int8 prefill")
        candidates.append((module, spec))
    if not candidates:
        raise MicroscalingError(f"microscaling scope {policy.scope!r} selected no eligible MXFP4 projections")
    handle.device = require_supported_device()
    import numpy as np
    import mlx.core as mx
    prepared = []
    for module, spec in candidates:
        scales = np.asarray(module["scales"])
        try:
            bc, rs = metadata_from_mxfp4_scales_numpy(scales)
        except ValueError as error:
            raise MicroscalingError(f"{spec.path}: not losslessly SC-MXFP4 embeddable: {error}") from error
        # Array metadata is copied once; checkpoint codes stay borrowed.
        bcodes, row_scales = mx.array(bc), mx.array(rs)
        metadata_fingerprint = hashlib.sha256(bc.tobytes() + rs.tobytes()).hexdigest()
        prepared.append(_Bound(module, type(module), spec, module["weight"],
            module["scales"], bcodes, row_scales, int(bc.nbytes + rs.nbytes), metadata_fingerprint))
    reference = weakref.ref(handle)
    try:
        for bound in prepared:
            handle._bound[id(bound.module)] = bound
            bound.module.__dict__[_STATE_ATTR] = reference
            bound.module.__class__ = _projection_subclass(bound.base, bound.spec.kind == "grouped")
        handle._root, handle._root_base = model, type(model)
        # Body/tap forwards can bypass the outer model.__call__. Wrap actual
        # input-id trunks too, recognized structurally among projection
        # ancestors rather than by model-family names.
        wrapped = set()
        for path, boundary in [("", model), *named]:
            if id(boundary) in wrapped:
                continue
            ancestor = boundary is model or any(b.spec.path.startswith(path + ".")
                for b in prepared)
            if not ancestor:
                continue
            parameters = list(inspect.signature(type(boundary).__call__).parameters.values())[1:]
            first = next((p for p in parameters if p.kind in
                (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)), None)
            if boundary is not model and (first is None or first.name not in {"inputs", "input_ids"}):
                continue
            base = type(boundary)
            handle._scope_bound.append((boundary, base))
            boundary.__dict__[_ROOT_ATTR] = reference
            boundary.__class__ = _model_subclass(base)
            wrapped.add(id(boundary))
        handle.active = True
    except BaseException:
        handle.active = True
        remove(handle)
        raise
    return handle


def remove(handle):
    """Restore exact prior classes and release borrowed metadata references."""
    if handle is None or not handle.active:
        return False
    handle.active = False
    for bound in handle._bound.values():
        reference = bound.module.__dict__.get(_STATE_ATTR)
        if reference is not None and reference() is handle:
            bound.module.__dict__.pop(_STATE_ATTR, None)
            if getattr(type(bound.module), "_mlx2_microscaling_base", None) is bound.base:
                bound.module.__class__ = bound.base
    for boundary, base in reversed(handle._scope_bound):
        reference = boundary.__dict__.get(_ROOT_ATTR)
        if reference is not None and reference() is handle:
            boundary.__dict__.pop(_ROOT_ATTR, None)
            if getattr(type(boundary), "_mlx2_microscaling_model_base", None) is base:
                boundary.__class__ = base
    handle._scope_bound.clear()
    handle._bound.clear()
    handle._root = handle._root_base = None
    return True


def adapter_scopes(adapter):
    declaration = getattr(adapter, "microscaling_supported", None)
    scopes = declaration() if callable(declaration) else declaration
    if isinstance(scopes, str):
        scopes = (scopes,)
    result = frozenset(scopes or ())
    if not result.issubset(SCOPES):
        raise ValueError(f"unknown declared microscaling scopes: {sorted(result)}")
    return result


def apply_for_adapter(adapter, policy, *, max_decode_rows=0):
    policy = MicroscalingPolicy.from_value(policy)
    if not policy.enabled:
        return MicroscalingHandle(policy)
    if policy.scope not in adapter_scopes(adapter):
        raise MicroscalingError(f"model adapter does not declare microscaling scope {policy.scope!r}")
    selector = getattr(adapter, "microscaling_select", None)
    identity = getattr(adapter, "identity", {})
    fingerprint = identity.get("fingerprint", "") if isinstance(identity, dict) else ""
    return apply(adapter.model, policy, artifact_fingerprint=fingerprint,
        select=selector(policy.scope) if callable(selector) else None,
        max_decode_rows=max_decode_rows)


def apc_semantic_fingerprint(base, handle):
    """An installed projection mask, artifact and activation law bind APCv2."""
    if handle is None:
        return base
    if not isinstance(handle, MicroscalingHandle):
        raise MicroscalingError("APCv2 microscaling identity requires an installed handle")
    if not handle.policy.enabled:
        return base
    if not handle.active:
        raise MicroscalingError("APCv2 microscaling identity requires an active installed handle")
    return (base, "sc-mxfp4-w4a8-prefill", handle.revision)
