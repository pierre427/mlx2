"""Request-private logical expert masks over adapter-owned native routers.

This controller owns selection only. Expert payloads, asynchronous reads,
leases, eviction and combination stay in the adapter's existing native store.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertMaskBinding:
    source_fingerprint: str
    quantization_profile: str
    execution_profile: str
    layers: tuple
    install: object


@dataclass(frozen=True)
class ExpertMaskView:
    name: str
    plan: object

    @property
    def fingerprint(self):
        return self.plan.plan_id


class ExpertMaskManager:
    def __init__(self, model, binding, *, checkpoint):
        if not isinstance(binding, ExpertMaskBinding):
            raise TypeError("adapter must declare ExpertMaskBinding")
        if getattr(model, "_mlx2_expert_masks", None) is not None:
            raise ValueError("expert masks already installed")
        self.model, self.binding, self.checkpoint = model, binding, checkpoint
        self.lock = threading.RLock()
        self.registry, self.views, self.uids, self.masks = {}, {}, {}, {}
        self.applications, self.counts = {}, Counter()
        self.active_uid, self.active_view = None, None
        self._restore = binding.install(self)
        object.__setattr__(model, "_mlx2_expert_masks", self)

    def describe(self):
        return {"checkpoint": self.checkpoint, "source_fingerprint": self.binding.source_fingerprint,
                "quantization_profile": self.binding.quantization_profile,
                "execution_profile": self.binding.execution_profile,
                "tie_policy": "native_topk_after_mask",
                "layers": [{"layer": layer.layer, "num_experts": layer.num_experts,
                            "top_k": layer.top_k, "normalize": layer.normalize,
                            "score_scale": layer.score_scale} for layer in self.binding.layers],
                "registered_names": sorted(self.registry), "native_expert_store_unchanged": True,
                "implemented": True, "qualified": False}

    def register(self, name, plan):
        from ..experimental.weight_space.experts import ExpertMaskPlan

        if not isinstance(plan, ExpertMaskPlan):
            raise TypeError("ExpertMaskPlan required")
        if (not isinstance(name, str) or not name or len(name) > 128
                or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in name)):
            raise ValueError("expert mask name must be a bounded identifier")
        binding = self.binding
        if (plan.checkpoint != self.checkpoint or plan.source_fingerprint != binding.source_fingerprint
                or plan.quantization_profile != binding.quantization_profile
                or plan.execution_profile != binding.execution_profile
                or plan.overlay_fingerprint != "none" or plan.tie_policy != "native_topk_after_mask"):
            raise ValueError("expert mask plan differs from the exact native source route")
        expected = {layer.layer: layer for layer in binding.layers}
        if {layer.layer for layer in plan.layers} != set(expected):
            raise ValueError("expert mask must cover every declared routed layer")
        for layer in plan.layers:
            base = expected[layer.layer]
            if (layer.num_experts, layer.top_k, layer.normalize, layer.score_scale) != (
                    base.num_experts, base.top_k, base.normalize, base.score_scale):
                raise ValueError("expert mask changes native router geometry or score law")
        with self.lock:
            if name in self.registry or len(self.registry) >= 128:
                raise ValueError("expert mask name exists or registry full")
            self.registry[name] = plan
        return {"name": name, "plan_id": plan.plan_id, "registered": True, "selected": False}

    def select(self, name):
        import mlx.core as mx

        with self.lock:
            if name not in self.registry:
                self.counts["rejected"] += 1
                raise ValueError("unknown expert mask")
            view = ExpertMaskView(name, self.registry[name])
            self.views[id(view)] = view
            self.masks[id(view)] = {layer.layer: mx.array([
                expert in layer.enabled for expert in range(layer.num_experts)])
                for layer in view.plan.layers if len(layer.enabled) != layer.num_experts}
            self.counts["selected"] += 1
            return view

    def bind_uid(self, uid, view):
        if id(view) not in self.views or uid in self.uids:
            raise ValueError("expert mask view not admitted or UID already bound")
        self.uids[uid] = view
        self.applications[uid] = Counter()

    def bind_rows(self, uids):
        self.clear_rows()
        selected = [uid for uid in uids if uid in self.uids]
        if selected and len(uids) != 1:
            self.counts["rejected"] += 1
            raise RuntimeError("expert masks require physical width one")
        if selected:
            self.active_uid = selected[0]
            self.active_view = self.uids[self.active_uid]

    def clear_rows(self):
        self.active_uid, self.active_view = None, None

    def selection(self, layer):
        if self.active_view is None:
            return None
        mask = self.masks[id(self.active_view)].get(layer)
        operation = "masked_router_calls" if mask is not None else "all_enabled_native_calls"
        self.counts[operation] += 1
        self.applications[self.active_uid][operation] += 1
        return mask

    def receipt(self, view, uid=None):
        applied = dict(self.applications.get(uid, {}))
        return {"schema": "mlx2.expert-mask-runtime.v1", "checkpoint": self.checkpoint,
                "plan_id": view.plan.plan_id, "source_revision": view.plan.source_revision,
                "approximate_routing": view.plan.approximate, "implemented": True,
                "qualified": False, "selected": True,
                "observed_used": applied.get("masked_router_calls", 0) > 0,
                "applications": applied, "tie_policy": "native_topk_after_mask",
                "native_expert_store_unchanged": True, "original_expert_ids_preserved": True,
                "apc_publication": "suppressed_unqualified_approximate" if view.plan.approximate else "exact_plan_scope",
                "apc_reuse": "private_cold_replay" if view.plan.approximate else "exact_plan_scope"}

    def release(self, view, uid=None):
        with self.lock:
            self.views.pop(id(view), None)
            self.masks.pop(id(view), None)
            self.uids.pop(uid, None)
            self.applications.pop(uid, None)

    def status(self):
        return {"implemented": True, "qualified": False, "selected": bool(self.views),
                "observed_used": self.counts["masked_router_calls"] > 0,
                "counts": dict(self.counts), "bound_requests": len(self.uids),
                "native_expert_store_unchanged": True}

    def restore(self):
        if self.views:
            raise RuntimeError("expert mask requests are still pinned")
        self._restore()
        object.__delattr__(self.model, "_mlx2_expert_masks")
