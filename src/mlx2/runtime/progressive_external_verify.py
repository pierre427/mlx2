"""Exact tile verifier used by default-off external-draft experiments.

This module owns only the proposal acceptance law.  Target execution, cache
transactions, and publication remain with the external executor so a caller
can stage every tile on request-private state before atomically promoting it.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .speculative_sampling import (
    RequestRNG,
    VerifiedBlock,
    probability,
    verify_greedy_proposals,
    verify_proposals,
)

PROGRESSIVE_EXTERNAL_VERIFY_VERSION = "exact-linear-b1-private-promotion-v1"
PROGRESSIVE_TARGET_PROTOCOL_VERSION = "row-exact-b1s1-cache-state-v1"
PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION = (
    "near-tie-bounded-b1s1-cache-state-v1"
)
PROGRESSIVE_FIDELITY_EXACT = "exact"
PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED = "near_tie_bounded"
PROGRESSIVE_FIDELITIES = frozenset(
    {PROGRESSIVE_FIDELITY_EXACT, PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED}
)
PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA = (
    "mlx2.qwen38-approx-progressive-qualification-gate.v1"
)
# Identity keys keep their repository spelling, but the bytes are read from
# the imported package so a wheel/site-packages install (no ``src/`` tree)
# hashes the same files as an editable checkout.
_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"
_ROOT = Path(__file__).resolve().parents[3]
_ALGORITHM_SOURCE_PATHS = (
    "src/mlx2/runtime/progressive_external_verify.py",
    "src/mlx2/runtime/external_speculative.py",
    "src/mlx2/adapters/external_draft_policy.py",
    "src/mlx2/runtime/ast_planner.py",
    "src/mlx2/adapters/ast_execution_profile.py",
    "src/mlx2/runtime/proposal_providers.py",
    "src/mlx2/adapters/proposal_path_sources.py",
    "src/mlx2/runtime/drafters/dflash2.py",
    "src/mlx2/runtime/row_exact_verify.py",
    "src/mlx2/runtime/models/row_exact_components.py",
    "src/mlx2/runtime/models/qwen_hybrid_row_exact.py",
    "src/mlx2/runtime/models/qwen4_exp.py",
    "src/mlx2/runtime/models/muse_glimmer.py",
    "src/mlx2/runtime/drafters/native_mtp.py",
    "src/mlx2/adapters/muse_glimmer.py",
)


def _source_path(name):
    if name.startswith(_PACKAGE_PREFIX):
        return _PACKAGE / name[len(_PACKAGE_PREFIX):]
    return _ROOT / name


def _source_hashes(paths):
    result = {}
    for name in paths:
        path = _source_path(name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def progressive_target_protocol_binding(model, fidelity):
    """Resolve one explicitly selected target fidelity and bind its sources.

    Exact admission retains the original singular model declaration.  The
    approximate candidate uses a separate declaration map, so its existence
    cannot weaken or alias the row-exact gate.
    """

    if fidelity not in PROGRESSIVE_FIDELITIES:
        raise ValueError("unsupported progressive verification fidelity")
    target_sources = ()
    if fidelity == PROGRESSIVE_FIDELITY_EXACT:
        protocol = getattr(model, "progressive_external_verify_protocol", None)
        if protocol != PROGRESSIVE_TARGET_PROTOCOL_VERSION:
            raise ValueError(
                "progressive verification requires a row-exact progressive "
                "target protocol; the target adapter must implement the "
                "row-exact B1 state-promotion protocol"
            )
    else:
        declarations = getattr(
            model, "progressive_external_approximate_protocols", None
        )
        if not isinstance(declarations, dict):
            raise ValueError(
                "near-tie-bounded progressive verification requires an "
                "explicit target protocol declaration"
            )
        declaration = declarations.get(
            PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
        )
        if not isinstance(declaration, dict):
            raise ValueError(
                "near-tie-bounded progressive verification requires an "
                "explicit target protocol declaration"
            )
        if declaration.get("fidelity") != fidelity:
            raise ValueError("progressive target protocol fidelity mismatch")
        target_sources = declaration.get("source_paths")
        if (
            not isinstance(target_sources, (list, tuple))
            or not target_sources
            or any(not isinstance(path, str) or not path for path in target_sources)
        ):
            raise ValueError("progressive target protocol has no source binding")
        protocol = PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION
    source_paths = tuple(dict.fromkeys((*_ALGORITHM_SOURCE_PATHS, *target_sources)))
    return {
        "schema": "mlx2.progressive-external-target-binding.v1",
        "algorithm": PROGRESSIVE_EXTERNAL_VERIFY_VERSION,
        "target_protocol": protocol,
        "fidelity": fidelity,
        "source_hashes": _source_hashes(source_paths),
        "qualification_gate_schema": (
            PROGRESSIVE_APPROXIMATE_QUALIFICATION_GATE_SCHEMA
            if fidelity == PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
            else None
        ),
    }


@dataclass(frozen=True)
class ProgressiveTargetRows:
    """Result from an adapter-owned progressive target-row executor.

    This is a capability seam, not an exactness claim.  A model may expose
    ``progressive_external_target_rows`` only after its adapter owns every
    width-sensitive target stage.  The executor must report the bound
    protocol and whether this particular launch stayed on its row-exact
    path; exact verification fails closed on a fallback.
    """

    logits: object
    features: object
    protocol: str
    row_exact: bool
    detail: dict


def _bounded_target_executor_detail(detail):
    """Keep adapter evidence finite and free of request-private structures."""

    if not isinstance(detail, dict):
        raise TypeError("progressive target-row executor detail must be a mapping")
    if len(detail) > 16:
        raise ValueError("progressive target-row executor detail is too large")
    result = {}
    for key, value in detail.items():
        if not isinstance(key, str) or not key or len(key) > 64:
            raise ValueError("progressive target-row executor detail key is invalid")
        if value is not None and type(value) not in (bool, int, float, str):
            raise TypeError(
                "progressive target-row executor detail values must be scalars"
            )
        if type(value) is int and not -(1 << 63) <= value < (1 << 63):
            raise ValueError("progressive target-row executor integer is too large")
        if type(value) is float and not np.isfinite(value):
            raise ValueError("progressive target-row executor float must be finite")
        if isinstance(value, str) and len(value) > 256:
            raise ValueError("progressive target-row executor detail value is too large")
        result[key] = value
    return result


def execute_progressive_target_rows(
    model,
    tokens,
    caches,
    capture_layers,
    *,
    protocol,
    fidelity,
    input_lengths=None,
):
    """Run an optional model-owned target-row executor, with a safe fallback.

    Existing protocol owners keep their established ``forward_with_taps``
    path.  A future Qwen3.6 executor can use this seam without adding a model
    name branch to the scheduler.  Merely defining the hook is insufficient:
    exact fidelity also requires a launch-local row-exact attestation.
    """

    if fidelity not in PROGRESSIVE_FIDELITIES:
        raise ValueError("unsupported progressive verification fidelity")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("progressive target-row executor requires a protocol")
    if input_lengths is not None and (
        len(input_lengths) != int(tokens.shape[0])
        or any(type(n) is not int or not 1 <= n <= int(tokens.shape[1]) for n in input_lengths)
    ):
        raise ValueError("progressive target lengths disagree with the physical slab")
    executor = getattr(model, "progressive_external_target_rows", None)
    if executor is None:
        logits, features = model.forward_with_taps(tokens, caches, capture_layers)
        return logits, features, {
            "execution": "legacy_forward_with_taps",
            "protocol": protocol,
            "row_exact": None,
            "adapter_detail": {},
            "physical_width": int(tokens.shape[0]),
            "token_width": int(tokens.shape[1]),
        }
    if not callable(executor):
        raise TypeError("progressive target-row executor must be callable")
    from .row_exact_verify import valid_rows
    compact = (input_lengths is not None and fidelity == PROGRESSIVE_FIDELITY_EXACT
               and getattr(model, "progressive_external_compact_exact_rows", False))
    scope = valid_rows(input_lengths, int(tokens.shape[1])) if compact else nullcontext()
    with scope:
        result = executor(tokens, caches, capture_layers)
    if not isinstance(result, ProgressiveTargetRows):
        raise TypeError(
            "progressive target-row executor must return ProgressiveTargetRows"
        )
    if result.protocol != protocol:
        raise RuntimeError("progressive target-row executor protocol mismatch")
    if type(result.row_exact) is not bool:
        raise TypeError("progressive target-row executor row_exact must be boolean")
    if fidelity == PROGRESSIVE_FIDELITY_EXACT and not result.row_exact:
        reason = result.detail.get("reason") if isinstance(result.detail, dict) else None
        suffix = f": {reason}" if isinstance(reason, str) and reason else ""
        raise RuntimeError(
            "progressive target-row executor fell off its exact path" + suffix
        )
    return result.logits, result.features, {
        "execution": "adapter_owned",
        "protocol": result.protocol,
        "row_exact": result.row_exact,
        "adapter_detail": _bounded_target_executor_detail(result.detail),
        "physical_width": int(tokens.shape[0]),
        "token_width": int(tokens.shape[1]),
    }


def verify_proposal_tile(
    tokens: Sequence[int],
    proposal_laws: Sequence[np.ndarray],
    target_laws: Sequence[np.ndarray],
    rng: RequestRNG,
    *,
    final: bool,
) -> VerifiedBlock:
    """Verify one proposal slice without an intermediate bonus draw.

    A non-final, fully accepted tile returns only its proposals.  It consumes
    the same ordered acceptance uniforms as ordinary token-wise verification,
    leaving the next proposal row and the sole bonus draw to a later tile.
    """

    tokens = tuple(tokens)
    q = tuple(probability(law) for law in proposal_laws)
    p = tuple(probability(law) for law in target_laws)
    if final:
        return verify_proposals(tokens, q, p, rng)
    if not tokens or len(q) != len(tokens) or len(p) != len(tokens):
        raise ValueError("non-final tile needs one target law per proposal token")
    if any(law.shape != p[0].shape for law in (*q, *p)):
        raise ValueError("tile vocabulary mismatch")
    for token, law in zip(tokens, q):
        if not 0 <= token < len(law) or law[token] <= 0:
            raise ValueError("proposed token has zero proposal probability")

    emitted: list[int] = []
    target_rows: list[np.ndarray] = []
    for index, token in enumerate(tokens):
        if rng.uniform() < min(1.0, p[index][token] / q[index][token]):
            emitted.append(token)
            target_rows.append(p[index])
            continue
        residual = np.maximum(p[index] - q[index], 0)
        correction = rng.sample(residual if residual.sum() > 0 else p[index])
        emitted.append(correction)
        target_rows.append(p[index])
        return VerifiedBlock(
            index,
            tuple(emitted),
            tuple(target_rows),
            True,
        )
    return VerifiedBlock(
        len(tokens),
        tuple(emitted),
        tuple(target_rows),
        False,
    )


def verify_greedy_proposal_tile(
    tokens: Sequence[int],
    selected: Sequence[int],
    rng: RequestRNG,
    *,
    final: bool,
) -> VerifiedBlock:
    """Greedy target-token form of ``verify_proposal_tile``.

    ``selected`` contains one target argmax per proposal for a non-final
    tile, and the additional bonus argmax for a final tile.  Keep the dense
    one-hot verifier's RNG schedule exactly: one acceptance draw per visited
    proposal plus one categorical draw for a correction or final bonus.
    """

    tokens = tuple(int(token) for token in tokens)
    selected = tuple(int(token) for token in selected)
    if final:
        return verify_greedy_proposals(tokens, selected, rng)
    if not tokens or len(selected) != len(tokens):
        raise ValueError(
            "non-final greedy tile needs one target token per proposal token"
        )
    emitted = []
    for index, (token, target) in enumerate(zip(tokens, selected, strict=True)):
        rng.uniform()
        if token == target:
            emitted.append(token)
            continue
        rng.uniform()
        emitted.append(target)
        return VerifiedBlock(index, tuple(emitted), (), True)
    return VerifiedBlock(len(tokens), tuple(emitted), (), False)
