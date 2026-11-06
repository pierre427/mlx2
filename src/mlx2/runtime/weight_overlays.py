"""Explicit adapter-owned, request-private weight replacement experiments.

Packed overlays remain host artifacts. This reference execution dequantizes a
selected matrix to float32; it is not a packed GPU kernel or a memory saving
claim. Shared base weights and unselected request outputs remain unchanged.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np

_ATTRIBUTE = "_mlx2_weight_overlays"
ARITHMETIC = "fp32_replacement_round_once_v1"


@dataclass(frozen=True)
class WeightOverlayBinding:
    """An adapter explicitly identifies a projection owner and attribute."""

    parent: object
    attribute: str


def weight_overlay_manager(model):
    return getattr(model, _ATTRIBUTE, None)


def bind_weight_overlay_rows(model, uids):
    managers = tuple(manager for manager in (weight_overlay_manager(model),
                     getattr(model, "_mlx2_layer_excision", None),
                     getattr(model, "_mlx2_expert_masks", None),
                     getattr(model, "_mlx2_ablation_overlays", None)) if manager is not None)
    for manager in managers:
        manager.bind_rows(uids)
    return managers or None


def clear_weight_overlay_rows(managers):
    for manager in managers or ():
        manager.clear_rows()


def _source_format(module):
    from mlx import nn

    from .models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    if type(module) in (nn.Linear, SwitchLinear):
        return f"mlx_dense_{module.weight.dtype}"
    if type(module) in (nn.QuantizedLinear, QuantizedSwitchLinear):
        return f"mlx_{module.mode}_q{module.bits}_g{module.group_size}"
    raise ValueError("weight overlay target is not an explicit supported linear module")


def _source_shape(module):
    output_dims = int(module.weight.shape[-2])
    input_dims = (int(module.scales.shape[-1]) * module.group_size if hasattr(module, "scales")
                  else int(module.weight.shape[-1]))
    return output_dims, input_dims


def _source(module, expert, *, maximum_bytes):
    import mlx.core as mx

    output_dims, input_dims = _source_shape(module)
    if output_dims * input_dims * 4 > maximum_bytes:
        raise ValueError("weight overlay source capture exceeds materialization budget")
    value = module.weight
    if expert is not None:
        value = value[expert]
    if hasattr(module, "scales"):
        scales = module.scales if expert is None else module.scales[expert]
        bias = getattr(module, "biases", None)
        if bias is not None and expert is not None:
            bias = bias[expert]
        value = mx.dequantize(value, scales, bias, group_size=module.group_size,
                              bits=module.bits, mode=module.mode)
    value = np.asarray(value.astype(mx.float32), dtype="<f4")
    if value.ndim != 2 or value.size > 16 * 1024 * 1024 or not np.isfinite(value).all():
        raise ValueError("weight overlay source must be a bounded finite matrix")
    return np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype="<f4").reshape(value.shape)


def _wrapper_class():
    import mlx.core as mx
    from mlx import nn

    class OverlayLinear(nn.Module):
        def __init__(self, base, manager, targets):
            super().__init__()
            self.base = base
            self._manager = manager
            self._targets = tuple(targets)

        def compiled_trace_dependencies(self):
            return (self._manager.active_fingerprint,)

        def __call__(self, x, *args, **kwargs):
            manager = self._manager
            selected = [(target, manager.active_weights[target]) for target in self._targets
                        if target in manager.active_weights]
            if not selected:
                return self.base(x, *args, **kwargs)
            if self._targets[0].expert is None:
                if args or kwargs or len(selected) != 1:
                    raise RuntimeError("dense overlay projection call contract changed")
                target, weights = selected[0]
                result = mx.matmul(x.astype(mx.float32), weights.T)
                bias = getattr(self.base, "bias", None)
                if bias is not None:
                    result = result + bias.astype(mx.float32)
                manager.record_application(target)
                return result.astype(x.dtype)
            # The reference expert route retains the base computation and
            # replaces only selected expert rows. No shared bank is rebuilt.
            indices = args[0] if args else kwargs.get("indices")
            if indices is None:
                raise RuntimeError("expert overlay needs explicit expert indices")
            result = self.base(x, *args, **kwargs)
            for target, weights in selected:
                replacement = mx.matmul(x.astype(mx.float32), weights.T)
                bias = getattr(self.base, "bias", None)
                if bias is not None:
                    replacement = replacement + bias[target.expert].astype(mx.float32)
                mask = indices == target.expert
                result = mx.where(mask[..., None, None], replacement.astype(result.dtype), result)
                # Defer the synchronization to the terminal route receipt.
                manager.record_application(target, mx.any(mask).astype(mx.int32))
            return result

    return OverlayLinear


class WeightOverlayManager:
    """A frozen overlay view is bound to one UID for its complete lifetime."""

    def __init__(self, model, bindings, *, checkpoint, bank=None, max_materialized_bytes=64 * 1024**2):
        if not bindings:
            raise ValueError("adapter has no weight overlay target bindings")
        if getattr(model, _ATTRIBUTE, None) is not None:
            raise ValueError("weight overlays are already installed")
        self.model, self.checkpoint, self.bank = model, checkpoint, bank
        if type(max_materialized_bytes) is not int or max_materialized_bytes < 1:
            raise ValueError("explicit positive overlay materialization budget required")
        self.max_materialized_bytes = max_materialized_bytes
        self.bindings = dict(bindings)
        self.lock = threading.RLock()
        self.originals, self.sources, self.views, self.uid_views = {}, {}, {}, {}
        self.applications, self.completed = {}, Counter()
        self.counts = Counter()
        self.active_weights, self.active_fingerprint, self.active_uid = {}, None, None
        groups = {}
        for target, binding in self.bindings.items():
            key = (id(binding.parent), binding.attribute)
            module = getattr(binding.parent, binding.attribute)
            _source_format(module)
            if target.expert is None and module.weight.ndim != 2:
                raise ValueError("dense overlay target needs a rank-two weight")
            if target.expert is not None and (module.weight.ndim != 3 or
                    not 0 <= target.expert < module.weight.shape[0]):
                raise ValueError("expert overlay index does not identify a stacked weight")
            groups.setdefault(key, (binding, module, []))[2].append(target)
        wrapper = _wrapper_class()
        for key, (binding, module, targets) in groups.items():
            self.originals[key] = (binding, module)
            setattr(binding.parent, binding.attribute, wrapper(module, self, targets))
        object.__setattr__(model, _ATTRIBUTE, self)

    def source(self, target):
        """Capture this loaded module's effective weights, never reinterpret bytes."""
        # The immutable binding table requires no manager lock. The bank can
        # invoke this validator while holding its lock without a lock cycle.
        if target not in self.bindings:
            self.counts["provider_errors"] += 1
            raise ValueError("overlay target is not declared by this adapter")
        binding = self.bindings[target]
        _, module = self.originals[(id(binding.parent), binding.attribute)]
        weights = _source(module, target.expert, maximum_bytes=self.max_materialized_bytes)
        digest = hashlib.sha256(weights.tobytes()).hexdigest()
        return {"weights": weights, "source_format": _source_format(module),
                "source_tensor_fingerprint": digest,
                "effective_weight_sha256": digest}

    def source_validator(self, target, fingerprint):
        return self.source(target)["source_tensor_fingerprint"] == fingerprint

    def source_shape(self, target):
        if target not in self.bindings:
            raise ValueError("overlay target is not declared by this adapter")
        binding = self.bindings[target]
        _, module = self.originals[(id(binding.parent), binding.attribute)]
        return _source_shape(module)

    def select(self, name):
        import mlx.core as mx

        with self.lock:
            view = None
            try:
                if self.bank is None:
                    raise ValueError("weight overlay bank is not configured")
                view = self.bank.acquire(name)
                if view.checkpoint != self.checkpoint:
                    raise ValueError("weight overlay checkpoint differs from loaded model")
                needed = sum(int(np.prod(item.tensor.shape)) * 4 for item in view.targets.values())
                resident = sum(int(value.nbytes) for _, weights in self.views.values()
                               for value in weights.values())
                if resident + needed > self.max_materialized_bytes:
                    raise ValueError("weight overlay materialized float32 budget exceeded")
                weights = {}
                for target, override in view.targets.items():
                    source = self.source(target)
                    if (override.source_tensor_fingerprint != source["source_tensor_fingerprint"]
                            or override.effective_weight_sha256 != source["effective_weight_sha256"]
                            or override.source_format != source["source_format"]):
                        raise ValueError("weight overlay source binding differs from loaded tensor")
                    value = override.tensor.dequantize()
                    if value.shape != source["weights"].shape:
                        raise ValueError("weight overlay matrix shape differs from source")
                    weights[target] = mx.array(value, dtype=mx.float32)
                self.views[id(view)] = (view, weights)
                self.counts["selected"] += 1
                return view
            except BaseException:
                # Any failure (MemoryError from dequantize included) must drop
                # the bank pin, or the overlay can never be deleted.
                self.counts["rejected"] += 1
                if view is not None:
                    self.views.pop(id(view), None)
                    view.close()
                raise

    def bind_uid(self, uid, view):
        with self.lock:
            if id(view) not in self.views or uid in self.uid_views:
                raise ValueError("overlay view is not admitted or UID already bound")
            self.uid_views[uid] = id(view)
            self.applications[uid] = 0

    def bind_rows(self, uids):
        self.clear_rows()
        selected = [uid for uid in uids if uid in self.uid_views]
        if selected and len(uids) != 1:
            self.counts["rejected"] += 1
            raise RuntimeError("weight overlays require physical width one")
        if selected:
            uid = selected[0]
            view, weights = self.views[self.uid_views[uid]]
            self.active_uid, self.active_weights = uid, weights
            self.active_fingerprint = view.fingerprint

    def clear_rows(self):
        self.active_uid, self.active_weights, self.active_fingerprint = None, {}, None

    def record_application(self, target, count=1):
        del target
        uid = self.active_uid
        if uid is None:
            raise RuntimeError("weight overlay projection has no bound request")
        self.applications[uid] = self.applications.get(uid, 0) + count

    def receipt(self, view, uid=None):
        value = self.applications.get(uid, 0)
        applications = int(value.item() if hasattr(value, "item") else value)
        previous = self.completed.get(uid, 0)
        self.counts["applied"] += max(0, applications - previous)
        self.completed[uid] = applications
        return {"schema": "mlx2.weight-overlay-runtime.v1", "checkpoint": self.checkpoint,
                "fingerprint": view.fingerprint, "arithmetic": ARITHMETIC,
                "implemented": True, "qualified": False, "selected": True,
                "observed_used": applications > 0, "applications": applications,
                "execution": "dequantized_float32_reference", "persistent": False,
                "apc_publication": "suppressed_unqualified_approximate", "apc_reuse": "private_cold_replay",
                "targets": [{**target.to_dict(), "format": override.tensor.spec.format,
                             "bits": override.tensor.spec.bits}
                            for target, override in view.targets.items()]}

    def release(self, view, uid=None):
        with self.lock:
            if uid is not None:
                self.uid_views.pop(uid, None)
                self.applications.pop(uid, None)
                self.completed.pop(uid, None)
            if self.views.pop(id(view), None) is not None:
                view.close()

    def status(self):
        with self.lock:
            return {"implemented": True, "qualified": False, "selected": bool(self.views),
                    "observed_used": self.counts["applied"] > 0,
                    "counts": dict(self.counts), "bound_requests": len(self.uid_views),
                    "dequantized_bytes": sum(int(value.nbytes) for _, weights in self.views.values()
                                             for value in weights.values()),
                    "materialized_budget_bytes": self.max_materialized_bytes,
                    "execution": "dequantized_float32_reference"}

    def restore(self):
        with self.lock:
            if self.views:
                raise RuntimeError("weight overlay views are still pinned")
            for binding, module in self.originals.values():
                setattr(binding.parent, binding.attribute, module)
            object.__delattr__(self.model, _ATTRIBUTE)
