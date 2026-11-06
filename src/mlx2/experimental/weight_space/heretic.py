"""CPU directional editing and retention metrics inspired by inspected Heretic.

Original NumPy mathematics, without importing/copying Heretic's AGPL code or
its optimizer. See provenance/heretic-weight-space-20261102.json. Captures are
caller-supplied; no model executes and no behavior is automatically selected.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from mlx2.runtime.weight_space_metrics import set_weight_space_gauge, track_weight_space

from .programs import MAX_RANK, _array, create_program

HERETIC_REVISION = "0db5c31e10ecc426440b6fb0ce4bf85352faf1a7"
MAX_CAPTURE_ROWS = 512
MAX_FULL_ELEMENTS = 1024 * 1024
COMPONENTS = {"attention_output", "mlp_output"}


def _finite(value, name, *, limit=16):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f"{name} must be finite and bounded by {limit}")
    return float(value)


def _digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def _unit(value):
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("direction is degenerate or numerically zero")
    return value / norm


@dataclass(frozen=True)
class DirectionEstimate:
    direction: np.ndarray
    report: dict


@track_weight_space("direction_estimate")
def estimate_direction(reference, contrast, *, orthogonalize_reference=False, winsorization_quantile=1.0):
    """Unit contrast-minus-reference mean, optionally orthogonal to reference mean.

    Winsorization clips each captured row symmetrically by its absolute-value
    quantile over components, matching the inspected source's orientation.
    """
    reference, contrast = _array(reference), _array(contrast)
    if reference.shape[1] != contrast.shape[1] or max(len(reference), len(contrast)) > MAX_CAPTURE_ROWS:
        raise ValueError("captures must have common width and at most 512 rows per group")
    if type(orthogonalize_reference) is not bool:
        raise ValueError("orthogonalize_reference must be boolean")
    quantile = _finite(winsorization_quantile, "winsorization_quantile", limit=1)
    if quantile < 0:
        raise ValueError("winsorization_quantile must be in [0,1]")

    def mean(capture):
        values = capture.astype(np.float64)
        if quantile < 1:
            bound = np.quantile(np.abs(values), quantile, axis=1, keepdims=True)
            values = np.clip(values, -bound, bound)
        return values.mean(axis=0)

    reference_mean, contrast_mean = mean(reference), mean(contrast)
    direction = _unit(contrast_mean - reference_mean)
    if orthogonalize_reference:
        preserved = _unit(reference_mean)
        direction = _unit(direction - np.dot(direction, preserved) * preserved)
    direction = _array(direction, ndim=1)
    report = {
        "method": "contrast_minus_reference_mean",
        "reference_rows": len(reference), "contrast_rows": len(contrast),
        "reference_sha256": _digest(reference), "contrast_sha256": _digest(contrast),
        "direction_sha256": _digest(direction),
        "orthogonalize_reference": orthogonalize_reference,
        "winsorization_quantile": quantile,
        "reference_mean_projection": float(np.dot(direction.astype(np.float64), reference_mean)),
        "contrast_mean_projection": float(np.dot(direction.astype(np.float64), contrast_mean)),
        "behavioral_interpretation_verified": False,
    }
    return DirectionEstimate(direction, report)


@dataclass(frozen=True)
class LayerProfile:
    peak_layer: float
    radius: float
    peak_strength: float = 1.0
    edge_strength: float = 0.0

    def __post_init__(self):
        _finite(self.peak_layer, "peak_layer", limit=10000)
        _finite(self.radius, "radius", limit=10000)
        _finite(self.peak_strength, "peak_strength")
        _finite(self.edge_strength, "edge_strength")
        if self.peak_layer < 0 or self.radius < 0:
            raise ValueError("layer profile positions and radius must be nonnegative")

    def strength(self, layer):
        if type(layer) is not int or not 0 <= layer <= 10000:
            raise ValueError("layer must be an explicit nonnegative integer")
        distance = abs(layer - self.peak_layer)
        if distance > self.radius:
            return 0.0
        if self.radius == 0:
            return float(self.peak_strength)
        fraction = distance / self.radius
        return float((1 - fraction) * self.peak_strength + fraction * self.edge_strength)


@track_weight_space("abliteration_compile")
def compile_abliteration(
    weights, directions, targets, *, profiles, checkpoint, source_revision,
    provenance, row_normalization="none", full_rank=None,
):
    """Build reversible explicit output-projection edits with layer/component gates.

    ``targets`` maps module keys to ``{layer: int, component: str}``; components
    are declared by the caller, not inferred from model-name strings. Weights
    are output×input. No embedding, input projection or scheduler policy runs.
    """
    if not isinstance(weights, dict) or not weights or len(weights) > 128 or set(weights) != set(directions) or set(weights) != set(targets):
        raise ValueError("weights, directions and targets must cover the same bounded modules")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("nonempty provenance is required")
    if not isinstance(profiles, dict) or not profiles or set(profiles) - COMPONENTS or any(not isinstance(p, LayerProfile) for p in profiles.values()):
        raise ValueError("profiles require explicit output components and LayerProfile values")
    if row_normalization not in {"none", "pre", "full"}:
        raise ValueError("row_normalization must be none, pre, or full")
    if row_normalization == "full":
        if type(full_rank) is not int or not 1 <= full_rank <= MAX_RANK:
            raise ValueError("full normalization requires an explicit rank in 1..1024")
    elif full_rank is not None:
        raise ValueError("full_rank is only meaningful with full normalization")
    factors, receipts = {}, {}
    for key, value in weights.items():
        spec = targets[key]
        if not isinstance(spec, dict) or set(spec) != {"layer", "component"} or spec["component"] not in profiles:
            raise ValueError("target requires a layer and enabled output-component profile")
        strength = profiles[spec["component"]].strength(spec["layer"])
        if strength == 0:
            receipts[key] = {**spec, "strength": 0, "selected": False, "reason": "layer_component_gate"}
            continue
        w = _array(value).astype(np.float64)
        estimated = directions[key]
        direction = estimated.direction if isinstance(estimated, DirectionEstimate) else estimated
        u = _array(direction, ndim=1).astype(np.float64)
        if len(u) != w.shape[0] or not np.isclose(np.linalg.norm(u), 1, rtol=1e-5, atol=1e-6):
            raise ValueError("direction must be unit length and match output width")
        row_norms = np.linalg.norm(w, axis=1)
        normalized = np.divide(w, row_norms[:, None], out=np.zeros_like(w), where=row_norms[:, None] > 0)
        projected = w if row_normalization == "none" else normalized
        a = (u @ projected)[:, None]
        b = (-strength * u)[None, :]
        discarded = 0.0
        if row_normalization == "pre":
            b = b * row_norms[None, :]
        elif row_normalization == "full":
            if w.size > MAX_FULL_ELEMENTS or min(w.shape) > 1024:
                raise ValueError("full normalization exceeds bounded CPU SVD geometry")
            changed = normalized + (a @ b).T
            changed_norms = np.linalg.norm(changed, axis=1)
            if np.any((changed_norms <= 1e-12) & (row_norms > 0)):
                raise ValueError("full normalization cannot preserve a collapsed nonzero row")
            desired = np.divide(changed, changed_norms[:, None], out=np.zeros_like(changed), where=changed_norms[:, None] > 0) * row_norms[:, None]
            left, singular, right = np.linalg.svd((desired - w).T, full_matrices=False)
            kept = min(full_rank, len(singular))
            roots = np.sqrt(singular[:kept])
            a, b = left[:, :kept] * roots, roots[:, None] * right[:kept]
            discarded = float(np.linalg.norm(singular[kept:]))
        a, b = _array(a), _array(b)
        factors[key] = a, b
        delta = (a.astype(np.float64) @ b.astype(np.float64)).T
        effective_norms = np.linalg.norm(w + delta, axis=1)
        receipts[key] = {
            **spec, "strength": strength, "selected": True,
            "weight_sha256": _digest(_array(value)),
            "direction": estimated.report if isinstance(estimated, DirectionEstimate) else {"direction_sha256": _digest(_array(direction, ndim=1))},
            "base_frobenius_norm": float(np.linalg.norm(w)),
            "delta_frobenius_norm": float(np.linalg.norm(delta)),
            "maximum_row_norm_drift": float(np.max(np.abs(effective_norms - row_norms))),
            "discarded_svd_frobenius_norm": discarded,
        }
    if not factors:
        raise ValueError("all output targets are disabled by layer/component gates")
    return create_program(
        factors, checkpoint=checkpoint, source_revision=source_revision,
        construction="contrastive_directional_projection", approximate=row_normalization == "full",
        provenance={
            "source": provenance, "conceptual_source": "https://github.com/p-e-w/heretic",
            "conceptual_source_revision": HERETIC_REVISION,
            "method": "original_numpy_projection", "row_normalization": row_normalization,
            "profiles": {name: vars(profile) for name, profile in profiles.items()},
            "targets": receipts, "behavioral_quality_evaluated": False,
        },
    )


@track_weight_space("retention_evaluate")
def retention_metrics(reference, contrast, new_reference, new_contrast, *, neighbors=1, reference_mse_ceiling=None):
    """ARA-inspired preservation/steering measurements, without its optimizer.

    An optional MSE gate concerns these arrays only, never task-level retention.
    """
    arrays = [_array(value).astype(np.float64) for value in (reference, contrast, new_reference, new_contrast)]
    reference, contrast, new_reference, new_contrast = arrays
    if reference.shape != new_reference.shape or contrast.shape != new_contrast.shape or reference.shape[1] != contrast.shape[1]:
        raise ValueError("original/new output groups must have matching geometry")
    if max(len(reference), len(contrast)) > MAX_CAPTURE_ROWS or type(neighbors) is not int or not 1 <= neighbors <= min(len(reference), len(contrast)):
        raise ValueError("retention rows or neighbor count exceed bounds")

    def distance(first, second):
        squared = np.sum(first * first, axis=1)[:, None] + np.sum(second * second, axis=1)[None, :] - 2 * (first @ second.T)
        distances = np.sqrt(np.maximum(0, squared))
        return float(np.partition(distances, neighbors - 1, axis=1)[:, :neighbors].mean())

    mse = float(np.mean((new_reference - reference) ** 2))
    gate = None
    if reference_mse_ceiling is not None:
        ceiling = _finite(reference_mse_ceiling, "reference_mse_ceiling", limit=1e20)
        if ceiling < 0:
            raise ValueError("reference_mse_ceiling must be nonnegative")
        gate = mse <= ceiling
    set_weight_space_gauge("retention_error_mse", mse)
    return {
        "reference_mse": mse,
        "contrast_to_reference_knn_distance": distance(new_contrast, reference),
        "contrast_to_original_contrast_knn_distance": distance(new_contrast, contrast),
        "neighbors": neighbors,
        "reference_array_gate_passed": gate,
        "evidence_scope": "supplied_output_arrays_only",
        "task_retention_verified": False,
        "ara_optimizer_executed": False,
    }
