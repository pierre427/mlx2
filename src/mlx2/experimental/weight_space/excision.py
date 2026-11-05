"""Revision-bound decoder block bypass and heldout boundary healing.

The adapter owns execution and cache advancement. These CPU contracts never
delete checkpoint tensors. The first slice removes one contiguous interior
range and fits a residual correction after its next retained block. Calibration
reconstructs supplied activations; it does not establish task quality.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

import numpy as np

from mlx2.runtime.weight_space_metrics import set_weight_space_gauge, track_weight_space

from .memory import CalibrationSplit, _fit, _score
from .overlays import _array
from .programs import _identity


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("expected lowercase SHA-256 identity")
    return value


@dataclass(frozen=True)
class BoundaryHealing:
    boundary_layer: int
    a: np.ndarray
    b: np.ndarray
    capture_digest: str
    calibration_digest: str
    heldout_digest: str

    def __post_init__(self):
        if type(self.boundary_layer) is not int or self.boundary_layer < 1:
            raise ValueError("invalid healing boundary")
        a, b = _array(self.a, dimensions=2), _array(self.b, dimensions=2)
        if a.shape[0] != b.shape[1] or a.shape[1] != b.shape[0] or a.shape[1] > 128:
            raise ValueError("healing must be a bounded square residual map")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        for key in ("capture_digest", "calibration_digest", "heldout_digest"):
            _sha(getattr(self, key))

    @property
    def nbytes(self):
        return self.a.nbytes + self.b.nbytes

    @property
    def fingerprint(self):
        return _digest({"boundary_layer": self.boundary_layer, "shape": [self.a.shape, self.b.shape],
                        "a": hashlib.sha256(self.a.tobytes()).hexdigest(),
                        "b": hashlib.sha256(self.b.tobytes()).hexdigest(),
                        "capture_digest": self.capture_digest, "calibration_digest": self.calibration_digest,
                        "heldout_digest": self.heldout_digest})

    def apply_numpy(self, hidden):
        hidden = _array(hidden)
        if hidden.shape[-1] != self.a.shape[0]:
            raise ValueError("hidden width differs from healing factors")
        return _array(hidden + (hidden @ self.a) @ self.b)


@dataclass(frozen=True)
class LayerExcisionPlan:
    checkpoint: str
    source_revision: str
    layer_count: int
    removed_layers: tuple[int, ...]
    hidden_size: int
    healing: BoundaryHealing | None = None

    def __post_init__(self):
        _identity(self.checkpoint, self.source_revision)
        if type(self.layer_count) is not int or not 3 <= self.layer_count <= 10000:
            raise ValueError("at least three decoder layers are required")
        if type(self.hidden_size) is not int or not 1 <= self.hidden_size <= 65536:
            raise ValueError("invalid hidden width")
        removed = tuple(self.removed_layers)
        if (not removed or len(removed) > self.layer_count
                or any(type(i) is not int or not 1 <= i < self.layer_count - 1 for i in removed)
                or removed != tuple(range(removed[0], removed[-1] + 1))
                or removed[0] < 1 or removed[-1] >= self.layer_count - 1):
            raise ValueError("remove one contiguous interior range, preserving first and final layers")
        object.__setattr__(self, "removed_layers", removed)
        if self.healing is not None and (not isinstance(self.healing, BoundaryHealing)
                or self.healing.boundary_layer != removed[-1] + 1
                or self.healing.a.shape[0] != self.hidden_size):
            raise ValueError("healing must target the next retained boundary with matching hidden width")

    @property
    def fingerprint(self):
        return _digest({"schema": "mlx2.layer-excision.v1", "checkpoint": self.checkpoint,
                        "source_revision": self.source_revision, "layer_count": self.layer_count,
                        "removed_layers": self.removed_layers, "hidden_size": self.hidden_size,
                        "healing": self.healing.fingerprint if self.healing else None,
                        "cache_layout": "original_indices_inactive_slots_advanced"})

    def receipt(self):
        return {"schema": "mlx2.layer-excision.v1", "checkpoint": self.checkpoint,
                "source_revision": self.source_revision, "fingerprint": self.fingerprint,
                "layer_count": self.layer_count, "removed_layers": list(self.removed_layers),
                "hidden_size": self.hidden_size,
                "healing_window": [self.removed_layers[0] - 1, self.removed_layers[-1] + 1],
                "healing_boundary": self.healing.boundary_layer if self.healing else None,
                "healing_rank": self.healing.a.shape[1] if self.healing else 0,
                "healing_bytes": self.healing.nbytes if self.healing else 0,
                "checkpoint_weights_deleted": False, "cache_slots_reclaimed": False,
                "qualified": False, "default_selected": False, "observed_used": False}


@dataclass(frozen=True)
class ExcisionCapture:
    plan_fingerprint: str
    sample_ids: tuple[str, ...]
    excised_outputs: np.ndarray
    reference_outputs: np.ndarray
    input_digest: str

    def __post_init__(self):
        _sha(self.plan_fingerprint)
        _sha(self.input_digest)
        ids = tuple(self.sample_ids)
        if (not 2 <= len(ids) <= 512 or len(set(ids)) != len(ids)
                or any(not isinstance(i, str) or not i or len(i) > 128 for i in ids)):
            raise ValueError("capture requires unique bounded sample identifiers")
        object.__setattr__(self, "sample_ids", ids)
        for key in ("excised_outputs", "reference_outputs"):
            object.__setattr__(self, key, _array(getattr(self, key), dimensions=2))
        if (self.excised_outputs.shape != self.reference_outputs.shape
                or len(self.excised_outputs) != len(ids)):
            raise ValueError("paired boundary captures must match samples and geometry")

    def digest_rows(self, ids):
        lookup = {name: index for index, name in enumerate(self.sample_ids)}
        rows = [lookup[name] for name in ids]
        return _digest({"plan": self.plan_fingerprint, "inputs": self.input_digest,
                        "ids": ids, "shape": self.excised_outputs[rows].shape,
                        "excised": hashlib.sha256(self.excised_outputs[rows].tobytes()).hexdigest(),
                        "reference": hashlib.sha256(self.reference_outputs[rows].tobytes()).hexdigest()})

    @property
    def fingerprint(self):
        return self.digest_rows(self.sample_ids)


@track_weight_space("layer_excision")
def fit_boundary_healing(plan, capture, *, split, rank=8, ridge=1e-4,
                         workspace_budget_bytes=512 * 1024**2):
    """Fit only calibration rows; report untouched heldout boundary error.

    For removing layer 21 the capture compares the reference and excised
    residual streams after layer 22, conditioned on the same input at layer 20.
    The correction is a separate low-rank residual after 22, not a claim to
    have retrained layers 20 and 22 or reproduced the missing layer exactly.
    """
    if not isinstance(plan, LayerExcisionPlan) or not isinstance(capture, ExcisionCapture):
        raise TypeError("explicit excision plan and paired boundary capture required")
    if plan.healing is not None or capture.plan_fingerprint != plan.fingerprint:
        raise ValueError("capture must bind the exact unhealed excision plan")
    if capture.excised_outputs.shape[1] != plan.hidden_size:
        raise ValueError("capture hidden width differs from plan")
    if not isinstance(split, CalibrationSplit) or set(capture.sample_ids) != set(split.calibration_ids + split.heldout_ids):
        raise ValueError("capture must cover the exact disjoint calibration and heldout split")
    if type(rank) is not int or not 1 <= rank <= 128:
        raise ValueError("healing rank must be 1..128")
    if isinstance(ridge, bool) or not isinstance(ridge, (float, int)) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    # Thin SVD/QR workspaces can dwarf retained float32 factors. Admit a
    # conservative estimate before allocating residuals or entering LAPACK.
    # Native allocator overhead and process RSS are not claimed by this bound.
    rows, width = capture.excised_outputs.shape
    workspace = capture.excised_outputs.size * 96 + min(rows, width)**2 * 128
    if (type(workspace_budget_bytes) is not int or workspace_budget_bytes < 1
            or workspace > workspace_budget_bytes):
        raise ValueError("healing exceeds conservative CPU workspace budget")
    lookup = {name: index for index, name in enumerate(capture.sample_ids)}
    train = [lookup[name] for name in split.calibration_ids]
    heldout = [lookup[name] for name in split.heldout_ids]
    x = capture.excised_outputs
    residual = capture.reference_outputs.astype(np.float64) - x.astype(np.float64)
    a, b = _fit(x[train], residual[train], rank, ridge)
    healing = BoundaryHealing(plan.removed_layers[-1] + 1, a, b, capture.fingerprint,
                              capture.digest_rows(split.calibration_ids), capture.digest_rows(split.heldout_ids))
    fitted = replace(plan, healing=healing)
    report = {**fitted.receipt(), "construction": "fitted_boundary_residual",
              "calibration": _score(x[train], residual[train], a, b),
              "heldout": _score(x[heldout], residual[heldout], a, b),
              "calibration_digest": healing.calibration_digest, "heldout_digest": healing.heldout_digest,
              "selection_split": "calibration_only", "ridge": float(ridge),
              "evidence_scope": "supplied_boundary_activation_reconstruction_only",
              "task_quality_evaluated": False}
    set_weight_space_gauge("layer_excision_layers", len(plan.removed_layers))
    return fitted, report


@track_weight_space("layer_excision")
def execute_numpy(layers, hidden, plan):
    """CPU functional reference; each retained callable preserves hidden width."""
    if not isinstance(plan, LayerExcisionPlan) or len(layers) != plan.layer_count:
        raise ValueError("layer inventory differs from plan")
    hidden = _array(hidden)
    if hidden.shape[-1] != plan.hidden_size:
        raise ValueError("hidden width differs from plan")
    original_shape = hidden.shape
    for index, layer in enumerate(layers):
        if index not in plan.removed_layers:
            hidden = _array(layer(hidden))
            if hidden.shape != original_shape:
                raise ValueError("retained layer changed hidden geometry")
            if plan.healing is not None and index == plan.healing.boundary_layer:
                hidden = plan.healing.apply_numpy(hidden)
    set_weight_space_gauge("layer_excision_layers", len(plan.removed_layers))
    return hidden
