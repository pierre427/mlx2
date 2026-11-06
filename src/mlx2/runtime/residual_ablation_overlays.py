"""Request-private residual-ablation overlays in serving (experimental, unqualified).

An overlay (``weight_space.residual_ablation.Overlay``) is a per-layer plan of
output-space projections ``y - lam * v (v . y)`` on the adapter-declared
residual write ports.  This manager wraps those output projections once, the
way ``weight_overlays`` and ``multi_lora`` wrap linears, and applies the
projection per batch row: each row carries its own overlay (or none), so a
batch can mix requests with different overlays.  Rows without an overlay
return the base output bit for bit.

Lifetimes: a request pins its overlay from admission (``acquire``) until its
release, so an overlay cannot be unloaded under an admitted request.
Temporary (preview) overlays are reference counted and removed when the last
pin is released, and never shadow an overlay that was loaded persistently.

Cache identity: a request's overlay enters its APCv2 scope by content digest;
writes are suppressed like other unqualified weight-space operations.  Routes
that bypass the wrapped projections are refused by the engine at install.
A row-geometry mismatch (packed/compacted forward) raises and stops the
worker instead of serving an unprojected row; those forwards are declined
while the manager is installed, so this is a fail-closed guard, not a path.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

_ATTRIBUTE = "_mlx2_ablation_overlays"
SCHEMA = "mlx2.residual-ablation-overlay-runtime.v1"


class OverlayNotLoaded(ValueError):
    """A request named an overlay that is not loaded (HTTP 409 overlay_not_loaded)."""

    http_status = 409
    code = "overlay_not_loaded"


class OverlayInUse(RuntimeError):
    """Unload refused while a request holds the overlay (HTTP 409)."""


def ablation_overlay_manager(model):
    return getattr(model, _ATTRIBUTE, None)


def _wrapper_class():
    import mlx.core as mx
    from mlx import nn

    class AblationProjectionLinear(nn.Module):
        def __init__(self, base, manager, layer, component):
            super().__init__()
            self.base = base
            self._manager = manager
            self._layer = int(layer)
            self._component = int(component)

        def compiled_trace_dependencies(self):
            rows = self._manager.rows
            return () if rows is None else (rows["lam"], rows["vectors"])

        def __call__(self, x, *args, **kwargs):
            out = self.base(x, *args, **kwargs)
            manager = self._manager
            rows = manager.rows
            if rows is None:
                return out
            lam = rows["lam"][:, self._layer, self._component]
            if out.ndim < 2 or out.shape[0] != lam.shape[0]:
                # Packed/compacted geometry: the row identity is lost.
                manager.counts["row_geometry_refusals"] += 1
                raise RuntimeError("ablation overlay needs one leading batch row per bound request")
            vectors = rows["vectors"][:, self._layer, self._component]
            shape = (lam.shape[0],) + (1,) * (out.ndim - 2)
            y = out.astype(mx.float32)
            coefficient = mx.sum(y * vectors.reshape(shape + (-1,)), axis=-1, keepdims=True)
            projected = (y - lam.reshape(shape + (1,)) * coefficient * vectors.reshape(shape + (-1,)))
            active = (lam != 0).reshape(shape + (1,))
            # Evidence that the projection executed for these requests.
            applications = manager.applications
            for uid in rows["active_uids"]:
                if uid in applications:
                    applications[uid] += 1
            return mx.where(active, projected.astype(out.dtype), out)

    return AblationProjectionLinear


class AblationOverlayManager:
    """Owns loaded overlays, pins, request bindings and per-forward row tables."""

    def __init__(self, model, declaration, *, checkpoint, max_overlays=16):
        from mlx2.experimental.weight_space.residual_ablation import (
            COMPONENTS, AblationDeclaration, resolve_linears,
        )

        if getattr(model, _ATTRIBUTE, None) is not None:
            raise ValueError("ablation overlays are already installed")
        if any(getattr(model, name, None) is not None for name in ("_varlen_dense_mlp", "_varlen_sparse_moe")):
            raise ValueError("ablation overlays are incompatible with varlen MLP compaction")
        declared = declaration if isinstance(declaration, AblationDeclaration) else AblationDeclaration(**declaration)
        self.model, self.checkpoint = model, checkpoint
        self.components = COMPONENTS
        self.lock = threading.RLock()
        self.max_overlays = int(max_overlays)
        self.overlays, self.tables, self.uid_overlays = {}, {}, {}
        self.loaded_at, self.uid_width, self.uid_branch = {}, {}, {}
        self.pins, self.transient = Counter(), {}  # transient: id -> outstanding temporary loads
        self.applications, self.counts = Counter(), Counter()
        self.rows, self._row_cache = None, {}
        linears = resolve_linears(model, declared)
        # Validate everything before wrapping anything: no partial install.
        self.hidden = None
        for _layer, _component, _path, _parent, _attr, module in linears:
            weight = module["weight"]
            if weight.ndim != 2:
                raise ValueError("ablation overlays support rank-two output projections only")
            out_dims = int(weight.shape[0])
            if self.hidden is None:
                self.hidden = out_dims
            elif self.hidden != out_dims:
                raise ValueError("declared output projections disagree on the residual width")
        self.layer_count = 1 + max(layer for layer, *_ in linears)
        wrapper = _wrapper_class()
        self.originals = []
        for layer, component, _path, parent, attr, module in linears:
            self.originals.append((parent, attr, module))
            parent[attr] = wrapper(module, self, layer, COMPONENTS.index(component))
        object.__setattr__(model, _ATTRIBUTE, self)

    # -- registry ---------------------------------------------------------

    @property
    def fingerprint(self):
        return self.checkpoint.rsplit("@", 1)[1] if "@" in self.checkpoint else self.checkpoint

    def load_document(self, document, *, transient=False):
        """Validate and register an overlay document (OVERLAY-API.md).

        Idempotent for the same id.  ``OverlayBindingError`` (409) when bound
        to another checkpoint; ``ValueError`` (422) for any other defect.
        A ``transient`` load is counted and removed by ``drop_transient`` (or
        when its last pin is released); it never demotes a persistent load,
        and a persistent load of the same id clears the transient state.
        """
        import mlx.core as mx

        from mlx2.experimental.weight_space.residual_ablation import overlay_from_document

        overlay = overlay_from_document(document, fingerprint=self.fingerprint,
                                        layer_count=self.layer_count, checkpoint=self.checkpoint)
        if overlay.vectors.shape[-1] != self.hidden:
            raise ValueError("overlay width differs from the model's residual width")
        with self.lock:
            identity = overlay.overlay_id
            if identity not in self.overlays:
                if len(self.overlays) >= self.max_overlays:
                    raise ValueError("ablation overlay registry is full")
                self.overlays[identity] = overlay
                self.tables[identity] = (mx.array(overlay.lam), mx.array(overlay.vectors))
                self.loaded_at[identity] = time.time()
                self._row_cache.clear()
                self.counts["loaded"] += 1
                if transient:
                    self.transient[identity] = 1
            elif transient:
                if identity in self.transient:
                    self.transient[identity] += 1
                # else: already loaded persistently; the preview just uses it.
            else:
                self.transient.pop(identity, None)  # promoted to persistent
        return self.describe(identity)

    def load(self, path):
        """Load ``<path>/overlay.json``."""
        document = json.loads((Path(path) / "overlay.json").read_text())
        return self.load_document(document)

    def _remove_locked(self, overlay_id):
        self.overlays.pop(overlay_id)
        self.tables.pop(overlay_id)
        self.loaded_at.pop(overlay_id, None)
        self.transient.pop(overlay_id, None)
        self._row_cache.clear()
        self.counts["unloaded"] += 1

    def unload(self, overlay_id):
        with self.lock:
            if not isinstance(overlay_id, str) or overlay_id not in self.overlays:
                raise OverlayNotLoaded(f"overlay {overlay_id!r} is not loaded on this server")
            if self.pins[overlay_id] > 0:
                raise OverlayInUse("overlay is held by an admitted request")
            self._remove_locked(overlay_id)

    def drop_transient(self, overlay_id):
        """Release one temporary load; remove the overlay once nothing holds it."""
        with self.lock:
            if overlay_id not in self.transient:
                return
            self.transient[overlay_id] -= 1
            if self.transient[overlay_id] <= 0:
                self.transient[overlay_id] = 0
                if self.pins[overlay_id] == 0:
                    self._remove_locked(overlay_id)

    def describe(self, overlay_id):
        overlay = self.overlays[overlay_id]
        return {"id": overlay_id, "label": overlay.label, "edits": overlay.edits,
                "fingerprint": overlay.fingerprint, "loaded": True,
                "loaded_at": self.loaded_at.get(overlay_id), "transient": overlay_id in self.transient}

    def list(self):
        with self.lock:
            return [self.describe(i) for i in self.overlays if i not in self.transient]

    # -- request lifetime ---------------------------------------------------

    def acquire(self, selection):
        """Resolve ``id`` or ``{"id", "branch_id"}`` and pin it for an admitted request."""
        overlay_id = selection.get("id") if isinstance(selection, dict) else selection
        with self.lock:
            if not isinstance(overlay_id, str) or overlay_id not in self.overlays:
                raise OverlayNotLoaded(f"overlay {overlay_id!r} is not loaded on this server")
            self.pins[overlay_id] += 1
            return overlay_id

    def resolve(self, selection):
        """Like ``acquire`` without pinning (read-only lookups and tests)."""
        overlay_id = selection.get("id") if isinstance(selection, dict) else selection
        with self.lock:
            if isinstance(overlay_id, str) and overlay_id in self.overlays:
                return overlay_id
        raise OverlayNotLoaded(f"overlay {overlay_id!r} is not loaded on this server")

    def bind_uid(self, uid, overlay_id, branch_id=None):
        with self.lock:
            if overlay_id not in self.overlays or self.pins[overlay_id] <= 0:
                raise ValueError("overlay must be acquired before binding")
            if uid in self.uid_overlays:
                raise ValueError("UID already bound to an overlay")
            self.uid_overlays[uid] = overlay_id
            self.applications[uid] = 0
            self.uid_width[uid] = 0
            if branch_id is not None:
                self.uid_branch[uid] = str(branch_id)[:256]
            self._row_cache.clear()

    def release(self, uid, overlay_id):
        """End a request's hold: unbind ``uid`` (if bound) and drop its pin."""
        with self.lock:
            if uid is not None and self.uid_overlays.get(uid) == overlay_id:
                self.uid_overlays.pop(uid, None)
                self.applications.pop(uid, None)
                self.uid_width.pop(uid, None)
                self.uid_branch.pop(uid, None)
                self._row_cache.clear()
            if self.pins[overlay_id] > 0:
                self.pins[overlay_id] -= 1
            if (self.pins[overlay_id] == 0 and overlay_id in self.transient
                    and self.transient[overlay_id] == 0):
                self._remove_locked(overlay_id)

    def bind_rows(self, uids):
        import mlx.core as mx

        self.clear_rows()
        with self.lock:
            ids = tuple(self.uid_overlays.get(uid) for uid in uids)
            if not any(ids):
                return
            key = (tuple(uids), ids)
            cached = self._row_cache.get(key)
            if cached is None:
                zeros_lam = mx.zeros((self.layer_count, len(self.components)), mx.float32)
                zeros_vec = mx.zeros((self.layer_count, len(self.components), self.hidden), mx.float32)
                cached = {
                    "lam": mx.stack([self.tables[i][0] if i else zeros_lam for i in ids]),
                    "vectors": mx.stack([self.tables[i][1] if i else zeros_vec for i in ids]),
                    "active_uids": tuple(uid for uid, i in zip(uids, ids) if i),
                }
                if len(self._row_cache) > 64:
                    self._row_cache.clear()
                self._row_cache[key] = cached
            for uid in cached["active_uids"]:
                self.uid_width[uid] = max(self.uid_width.get(uid, 0), len(uids))
            self.rows = cached

    def clear_rows(self):
        self.rows = None

    # -- receipts -----------------------------------------------------------

    def receipt(self, uid):
        """Spec receipt (OVERLAY-API.md section 5); ``applied`` counts wrapper executions."""
        with self.lock:
            overlay_id = self.uid_overlays.get(uid)
            if overlay_id is None:
                return None
            overlay = self.overlays[overlay_id]
            executions = int(self.applications.get(uid, 0))
            return {"id": overlay_id, "label": overlay.label, "edits": overlay.edits,
                    "fingerprint": overlay.fingerprint, "row_normalization": "none",
                    "applied": executions > 0, "qualified": False,
                    "lane_width": int(self.uid_width.get(uid, 0)), "per_row": True,
                    "projection_executions": executions, "ad_hoc": overlay_id in self.transient,
                    **({"branch_id": self.uid_branch[uid]} if uid in self.uid_branch else {}),
                    "apc_publication": "suppressed_unqualified", "schema": SCHEMA}

    def status(self):
        with self.lock:
            return {"implemented": True, "qualified": False, "schema": SCHEMA,
                    "checkpoint": self.checkpoint, "overlays": self.list(),
                    "bound_requests": len(self.uid_overlays), "pinned": sum(self.pins.values()),
                    "counts": dict(self.counts), "per_row": True}

    def restore(self):
        with self.lock:
            if self.uid_overlays or any(self.pins.values()):
                raise RuntimeError("ablation overlays are held by admitted requests")
            for parent, attr, module in self.originals:
                parent[attr] = module
            object.__delattr__(self.model, _ATTRIBUTE)
