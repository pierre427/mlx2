"""Request-private block bypass with adapter-owned cache advancement.

Original layers and cache slots remain resident for exact rollback. Selected
blocks perform no attention/MLP work; their reserved KV slots contain zeros and
are never consumed by the selected model. Selection has its own APC namespace.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerExcisionBinding:
    layers: list
    hidden_size: int
    advance_cache: object


@dataclass(frozen=True)
class LayerExcisionView:
    plan: object
    name: str

    @property
    def fingerprint(self):
        return self.plan.fingerprint


def _wrapper_class():
    import mlx.core as mx
    from mlx import nn

    class ExcisionBlock(nn.Module):
        def __init__(self, base, manager, index):
            super().__init__()
            self.base = base
            self._manager, self._index = manager, index

        def compiled_trace_dependencies(self):
            view = self._manager.active_view
            return (view.fingerprint if view is not None else None,)

        def __call__(self, x, mask=None, cache=None, *, row_exact=False):
            manager, index = self._manager, self._index
            view = manager.active_view
            if view is None:
                return self.base(x, mask, cache, row_exact=row_exact)
            if row_exact:
                raise RuntimeError("layer excision does not support row-exact execution")
            if index in view.plan.removed_layers:
                manager.binding.advance_cache(index, x, cache)
                manager.record("skipped_blocks")
                return x
            hidden = self.base(x, mask, cache, row_exact=False)
            healing = view.plan.healing
            if healing is not None and index == healing.boundary_layer:
                a, b = manager.healing[id(view)]
                hidden = (hidden.astype(mx.float32) + (hidden.astype(mx.float32) @ a) @ b).astype(hidden.dtype)
                manager.record("healing_applications")
            return hidden

    return ExcisionBlock


class LayerExcisionManager:
    def __init__(self, model, binding, *, checkpoint, max_healing_bytes=64 * 1024**2):
        if not isinstance(binding, LayerExcisionBinding):
            raise TypeError("adapter must provide an explicit LayerExcisionBinding")
        if getattr(model, "_mlx2_layer_excision", None) is not None:
            raise ValueError("layer excision already installed")
        self.model, self.binding, self.checkpoint = model, binding, checkpoint
        if type(max_healing_bytes) is not int or max_healing_bytes < 1:
            raise ValueError("explicit positive healing budget required")
        self.max_healing_bytes = max_healing_bytes
        self.lock = threading.RLock()
        self.registry, self.views, self.uids, self.healing = {}, {}, {}, {}
        self.applications, self.counts = {}, Counter()
        self.originals = tuple(binding.layers)
        self.active_uid, self.active_view = None, None
        wrapper = _wrapper_class()
        for index, layer in enumerate(self.originals):
            binding.layers[index] = wrapper(layer, self, index)
        object.__setattr__(model, "_mlx2_layer_excision", self)

    def register(self, name, plan):
        from ..experimental.weight_space.excision import LayerExcisionPlan

        if not isinstance(plan, LayerExcisionPlan):
            raise TypeError("LayerExcisionPlan required")
        if (not isinstance(name, str) or not name or len(name) > 128
                or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in name)):
            raise ValueError("layer excision name must be a bounded identifier")
        if (plan.checkpoint != self.checkpoint or plan.layer_count != len(self.originals)
                or plan.hidden_size != self.binding.hidden_size):
            raise ValueError("layer excision plan differs from loaded checkpoint/geometry")
        if plan.healing is not None and plan.healing.nbytes > self.max_healing_bytes:
            raise ValueError("layer excision healing exceeds materialization budget")
        with self.lock:
            if name in self.registry or len(self.registry) >= 128:
                raise ValueError("layer excision name exists or registry full")
            self.registry[name] = plan
        return plan.receipt()

    def select(self, name):
        import mlx.core as mx

        with self.lock:
            if name not in self.registry:
                self.counts["rejected"] += 1
                raise ValueError("unknown layer excision plan")
            plan = self.registry[name]
            resident = sum(a.nbytes + b.nbytes for a, b in self.healing.values())
            if plan.healing is not None and resident + plan.healing.nbytes > self.max_healing_bytes:
                self.counts["rejected"] += 1
                raise ValueError("layer excision healing materialization budget exceeded")
            view = LayerExcisionView(plan, name)
            self.views[id(view)] = view
            if plan.healing is not None:
                self.healing[id(view)] = (mx.array(plan.healing.a), mx.array(plan.healing.b))
            self.counts["selected"] += 1
            return view

    def bind_uid(self, uid, view):
        if id(view) not in self.views or uid in self.uids:
            raise ValueError("layer excision view not admitted or UID already bound")
        self.uids[uid] = view
        self.applications[uid] = Counter()

    def bind_rows(self, uids):
        self.clear_rows()
        selected = [uid for uid in uids if uid in self.uids]
        if selected and len(uids) != 1:
            self.counts["rejected"] += 1
            raise RuntimeError("layer excision requires physical width one")
        if selected:
            self.active_uid = selected[0]
            self.active_view = self.uids[self.active_uid]

    def clear_rows(self):
        self.active_uid, self.active_view = None, None

    def record(self, operation):
        self.counts[operation] += 1
        self.applications[self.active_uid][operation] += 1

    def receipt(self, view, uid=None):
        applied = dict(self.applications.get(uid, {}))
        return {**view.plan.receipt(), "implemented": True, "selected": True,
                "observed_used": applied.get("skipped_blocks", 0) > 0,
                "applications": applied, "execution": "block_bypass_zero_reserved_kv",
                "apc_publication": "suppressed_unqualified_approximate", "apc_reuse": "private_cold_replay",
                "base_weights_retained": True, "cache_slots_retained": True}

    def release(self, view, uid=None):
        with self.lock:
            self.views.pop(id(view), None)
            self.healing.pop(id(view), None)
            self.uids.pop(uid, None)
            self.applications.pop(uid, None)

    def status(self):
        return {"implemented": True, "qualified": False, "selected": bool(self.views),
                "observed_used": self.counts["skipped_blocks"] > 0,
                "counts": dict(self.counts), "bound_requests": len(self.uids),
                "healing_bytes": sum(a.nbytes + b.nbytes for a, b in self.healing.values()),
                "base_weights_retained": True, "cache_slots_retained": True}

    def restore(self):
        if self.views:
            raise RuntimeError("layer excision requests are still pinned")
        self.binding.layers[:] = self.originals
        object.__delattr__(self.model, "_mlx2_layer_excision")
