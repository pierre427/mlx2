"""CPU research oracles for Xing MLA shared-prefix and shifted reuse.

These NumPy functions are deliberately not imported by the serving adapter.
The hybrid calculation is an exact attention rearrangement (up to rounding);
the shifted-cache calculation is an *approximate* candidate that can only be
examined offline against recomputed target state. Neither is a qualified route.

The hybrid formulation is inspired by TyphoonMLA (arXiv:2509.21081), and the
delta rotation experiment by Irminsul (arXiv:2605.05696). This is original
NumPy reference code, not a port of either implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _positions(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative integer positions of shape {shape}")
    return array


@dataclass(frozen=True)
class HybridMLAResult:
    output: np.ndarray
    shared_prefix_tiles: int
    private_suffix_tiles: int


def hybrid_shared_prefix_attention(
    *,
    query_nope: np.ndarray,
    query_rope: np.ndarray,
    shared_latent: np.ndarray,
    shared_rope: np.ndarray,
    suffix_latent: np.ndarray,
    suffix_rope: np.ndarray,
    embed_weight: np.ndarray,
    unembed_weight: np.ndarray,
    query_positions: np.ndarray,
    shared_positions: np.ndarray,
    suffix_positions: np.ndarray,
    scale: float,
    tile_size: int = 128,
) -> HybridMLAResult:
    """Causal hybrid MLA with one shared prefix and a private suffix per lane.

    All query RoPE keys and cached RoPE keys are already rotated. The shared
    prefix is expanded into per-head K/V once; private suffix scores use the
    absorbed latent form. Tile softmax sums are merged without materializing
    the full ``[B,H,L,S]`` score tensor. Output is ``[B,H,L,Dv]``.

    This is a numerical CPU oracle, not a Metal implementation or an APCv2
    cache operation. It has no claim of speedup on its own.
    """
    q = _array(query_nope, "query_nope", 4)
    qr = _array(query_rope, "query_rope", 4)
    shared = _array(shared_latent, "shared_latent", 2)
    shared_r = _array(shared_rope, "shared_rope", 2)
    suffix = _array(suffix_latent, "suffix_latent", 3)
    suffix_r = _array(suffix_rope, "suffix_rope", 3)
    ew = _array(embed_weight, "embed_weight", 3)
    uw = _array(unembed_weight, "unembed_weight", 3)
    b, h, l, dn = q.shape
    r = shared.shape[1]
    dr = qr.shape[-1]
    if (
        qr.shape != (b, h, l, dr)
        or shared_r.shape != (shared.shape[0], dr)
        or suffix.shape != (b, suffix.shape[1], r)
        or suffix_r.shape != (b, suffix.shape[1], dr)
        or ew.shape != (h, r, dn)
        or uw.shape[0] != h
        or uw.shape[2] != r
        or l == 0
        or shared.shape[0] + suffix.shape[1] == 0
    ):
        raise ValueError("inconsistent hybrid MLA geometry")
    qp = _positions(query_positions, "query_positions", (b, l))
    sp = _positions(shared_positions, "shared_positions", (shared.shape[0],))
    tp = _positions(suffix_positions, "suffix_positions", (b, suffix.shape[1]))
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    prefix_keys = np.einsum("sr,hrd->hsd", shared, ew)
    prefix_values = np.einsum("sr,hvr->hsv", shared, uw)
    absorbed_queries = np.einsum("bhld,hrd->bhlr", q, ew)
    running_max = np.full((b, h, l), -np.inf)
    denominator = np.zeros((b, h, l), dtype=np.float64)
    numerator = np.zeros((b, h, l, uw.shape[1]), dtype=np.float64)

    def merge(scores: np.ndarray, values: np.ndarray, visible: np.ndarray, *, latent: bool) -> None:
        nonlocal running_max, denominator, numerator
        scores = np.where(visible[:, None, :, :], scores, -np.inf)
        tile_max = np.max(scores, axis=-1)
        new_max = np.maximum(running_max, tile_max)
        with np.errstate(invalid="ignore", over="ignore"):
            old_factor = np.where(np.isfinite(running_max), np.exp(running_max - new_max), 0.0)
            tile_factor = np.where(np.isfinite(tile_max), np.exp(tile_max - new_max), 0.0)
            probabilities = np.where(
                np.isfinite(scores), np.exp(scores - tile_max[..., None]), 0.0
            )
        partial = np.einsum("bhlt,btr->bhlr", probabilities, values) if latent else np.einsum(
            "bhlt,htv->bhlv", probabilities, values
        )
        if latent:
            partial = np.einsum("bhlr,hvr->bhlv", partial, uw)
        numerator = numerator * old_factor[..., None] + partial * tile_factor[..., None]
        denominator = denominator * old_factor + probabilities.sum(axis=-1) * tile_factor
        running_max = new_max

    prefix_tiles = 0
    for start in range(0, shared.shape[0], tile_size):
        end = min(start + tile_size, shared.shape[0])
        scores = np.einsum("bhld,htd->bhlt", q, prefix_keys[:, start:end])
        scores += np.einsum("bhld,td->bhlt", qr, shared_r[start:end])
        visible = sp[None, None, start:end] <= qp[:, :, None]
        # Broadcast prefix values without copying them across lanes.
        merge(scores * scale, prefix_values[:, start:end], visible, latent=False)
        prefix_tiles += 1

    suffix_tiles = 0
    for start in range(0, suffix.shape[1], tile_size):
        end = min(start + tile_size, suffix.shape[1])
        scores = np.einsum("bhlr,btr->bhlt", absorbed_queries, suffix[:, start:end])
        scores += np.einsum("bhld,btd->bhlt", qr, suffix_r[:, start:end])
        visible = tp[:, None, start:end] <= qp[:, :, None]
        merge(scores * scale, suffix[:, start:end], visible, latent=True)
        suffix_tiles += 1

    output = np.divide(
        numerator,
        denominator[..., None],
        out=np.zeros_like(numerator),
        where=denominator[..., None] > 0,
    )
    return HybridMLAResult(output, prefix_tiles, suffix_tiles)


@dataclass(frozen=True)
class ReuseResearchContract:
    """Explicit offline boundary; this does not authorize a serving route."""

    experiment_id: str
    model_revision: str
    source_context_hash: str
    target_context_hash: str
    enabled: bool = False

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("shifted latent reuse requires explicit offline opt-in")
        for name in ("experiment_id", "model_revision", "source_context_hash", "target_context_hash"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class ShiftedReuseCandidate:
    latent: np.ndarray
    rotated_rope: np.ndarray
    source_positions: np.ndarray
    target_positions: np.ndarray
    contract: ReuseResearchContract
    fidelity: str = "approximate_candidate_only"


def rotate_rope_delta(
    rotated_keys: np.ndarray,
    delta: np.ndarray,
    *,
    theta: float | None = None,
    frequency_denominators: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a position delta to adjacent-pair RoPE keys.

    ``frequency_denominators`` is required for scaled RoPE such as Xing's
    YaRN, where the base-theta formula would rotate the cached key wrongly.
    Pass the live layer's RoPE ``_freqs`` values when probing that model.
    """
    keys = _array(rotated_keys, "rotated_keys", 2)
    if keys.shape[1] < 2 or keys.shape[1] % 2:
        raise ValueError("RoPE width must be positive and even")
    offsets = np.asarray(delta)
    if offsets.shape != (keys.shape[0],) or offsets.dtype.kind not in "iu":
        raise ValueError("delta must be one integer offset per key")
    width = keys.shape[1]
    if frequency_denominators is None:
        if theta is None or not np.isfinite(theta) or theta <= 0:
            raise ValueError("theta must be finite and positive")
        denominators = theta ** (2 * np.arange(width // 2) / width)
    else:
        if theta is not None:
            raise ValueError("provide either theta or frequency_denominators")
        denominators = _array(frequency_denominators, "frequency_denominators", 1)
        if denominators.shape != (width // 2,) or np.any(denominators <= 0):
            raise ValueError("RoPE frequency denominators have the wrong geometry")
    angle = offsets[:, None] / denominators[None, :]
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.empty_like(keys)
    result[:, 0::2] = keys[:, 0::2] * cosine - keys[:, 1::2] * sine
    result[:, 1::2] = keys[:, 0::2] * sine + keys[:, 1::2] * cosine
    return result


def propose_shifted_reuse(
    *,
    source_latent: np.ndarray,
    source_rotated_rope: np.ndarray,
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    source_token_ids: np.ndarray,
    target_token_ids: np.ndarray,
    theta: float | None = None,
    frequency_denominators: np.ndarray | None = None,
    contract: ReuseResearchContract,
) -> ShiftedReuseCandidate:
    """Propose shifted cache state for *offline* approximation measurement.

    Matching token IDs are necessary but not sufficient for equivalent latent
    state: prior context can change every layer's latent. No APCv2 lookup or
    serving publication uses this function.
    """
    if not isinstance(contract, ReuseResearchContract):
        raise TypeError("an enabled offline research contract is required")
    latent = _array(source_latent, "source_latent", 2)
    rope = _array(source_rotated_rope, "source_rotated_rope", 2)
    n = latent.shape[0]
    if rope.shape[0] != n:
        raise ValueError("source latent and RoPE key lengths differ")
    old = _positions(source_positions, "source_positions", (n,))
    new = _positions(target_positions, "target_positions", (n,))
    old_tokens = np.asarray(source_token_ids)
    new_tokens = np.asarray(target_token_ids)
    if (
        old_tokens.shape != (n,)
        or new_tokens.shape != (n,)
        or old_tokens.dtype.kind not in "iu"
        or new_tokens.dtype.kind not in "iu"
        or not np.array_equal(old_tokens, new_tokens)
    ):
        raise ValueError("shifted reuse requires identical token ID sequences")
    candidate_latent = latent.copy()
    candidate_rope = rotate_rope_delta(
        rope,
        new.astype(np.int64) - old.astype(np.int64),
        theta=theta,
        frequency_denominators=frequency_denominators,
    )
    candidate_latent.flags.writeable = False
    candidate_rope.flags.writeable = False
    return ShiftedReuseCandidate(candidate_latent, candidate_rope, old.copy(), new.copy(), contract)


@dataclass(frozen=True)
class ReuseComparison:
    latent_max_abs_error: float
    rope_max_abs_error: float
    within_tolerance: bool
    fidelity: str = "approximate_candidate_only"


def compare_reuse_to_recomputed(
    candidate: ShiftedReuseCandidate,
    *,
    recomputed_target_latent: np.ndarray,
    recomputed_target_rope: np.ndarray,
    atol: float,
) -> ReuseComparison:
    """Measure one candidate against target context, without qualifying it."""
    if not isinstance(candidate, ShiftedReuseCandidate):
        raise TypeError("candidate must come from the offline reuse API")
    target_latent = _array(recomputed_target_latent, "recomputed_target_latent", 2)
    target_rope = _array(recomputed_target_rope, "recomputed_target_rope", 2)
    if target_latent.shape != candidate.latent.shape or target_rope.shape != candidate.rotated_rope.shape:
        raise ValueError("recomputed target state shape differs")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    latent_error = float(np.max(np.abs(candidate.latent - target_latent), initial=0.0))
    rope_error = float(np.max(np.abs(candidate.rotated_rope - target_rope), initial=0.0))
    return ReuseComparison(latent_error, rope_error, latent_error <= atol and rope_error <= atol)
