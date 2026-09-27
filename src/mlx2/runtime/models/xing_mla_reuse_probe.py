"""Bounded, offline Xing model-path probe of approximate shifted MLA reuse.

This module has no serving hook. It deliberately recomputes the target prompt,
then compares each proposed latent/RoPE row and one continuation's hidden and
logit vectors. Running it requires a GPU lease and an already loaded model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .xing_mla_research import (
    ReuseResearchContract,
    compare_reuse_to_recomputed,
    propose_shifted_reuse,
)


@dataclass(frozen=True)
class ShiftedReuseProbeReport:
    model_revision: str
    layer_count: int
    source_prefix_tokens: int
    target_prefix_tokens: int
    shared_chunk_tokens: int
    continuation_tokens: int
    max_latent_abs_error: float
    max_rope_abs_error: float
    max_hidden_abs_error: float
    max_logits_abs_error: float
    logits_rms_error: float
    greedy_token_agreement: bool
    fidelity: str = "approximate_candidate_only"


def _ids(value, name: str, *, maximum: int, allow_empty: bool = False) -> np.ndarray:
    ids = np.asarray(value)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or (not allow_empty and len(ids) == 0)
        or len(ids) > maximum
        or np.any(ids < 0)
    ):
        raise ValueError(f"{name} must be 1..{maximum} nonnegative token IDs")
    return ids.astype(np.int32, copy=True)


def _hash_context(ids: np.ndarray) -> str:
    return hashlib.sha256(ids.astype("<i4").tobytes()).hexdigest()


def probe_shifted_reuse_on_model(
    model,
    *,
    source_prefix_ids,
    target_prefix_ids,
    shared_chunk_ids,
    continuation_ids,
    model_revision: str,
) -> ShiftedReuseProbeReport:
    """Compare shifted cache rows with recomputed target rows and continuation.

    The probe is bounded to at most 256 prefix, 32 shared, and four continuation
    tokens. It accepts batch one only and never mutates the supplied model or
    an existing serving cache. ``model`` is a loaded Xing ``Model``.
    """
    source_prefix = _ids(source_prefix_ids, "source_prefix_ids", maximum=256, allow_empty=True)
    target_prefix = _ids(target_prefix_ids, "target_prefix_ids", maximum=256, allow_empty=True)
    chunk = _ids(shared_chunk_ids, "shared_chunk_ids", maximum=32)
    continuation = _ids(continuation_ids, "continuation_ids", maximum=4)
    if not model_revision:
        raise ValueError("model_revision is required")
    if len(source_prefix) + len(target_prefix) + len(chunk) + len(continuation) > 320:
        raise ValueError("probe token budget exceeded")

    import mlx.core as mx

    def host_float32(value):
        # NumPy has no native PEP 3118 format for MLX BF16 GPU buffers.
        return np.asarray(value.astype(mx.float32))

    if not hasattr(model, "make_cache") or not hasattr(model, "model"):
        raise ValueError("probe requires a loaded Xing model")
    source_cache = model.make_cache()
    target_cache = model.make_cache()
    candidate_cache = model.make_cache()

    def forward(ids: np.ndarray, cache):
        if len(ids):
            mx.eval(model(mx.array(ids[None, :]), cache=cache))

    forward(np.concatenate((source_prefix, chunk)), source_cache)
    forward(np.concatenate((target_prefix, chunk)), target_cache)
    forward(target_prefix, candidate_cache)
    if len(source_cache) != len(target_cache) or len(source_cache) != len(candidate_cache):
        raise ValueError("cache layer counts differ")

    contract = ReuseResearchContract(
        experiment_id="shifted-mla-model-probe",
        model_revision=model_revision,
        source_context_hash=_hash_context(np.concatenate((source_prefix, chunk))),
        target_context_hash=_hash_context(np.concatenate((target_prefix, chunk))),
        enabled=True,
    )
    source_positions = np.arange(len(source_prefix), len(source_prefix) + len(chunk))
    target_positions = np.arange(len(target_prefix), len(target_prefix) + len(chunk))
    latent_error = rope_error = 0.0
    for index, (source, target, destination) in enumerate(zip(source_cache, target_cache, candidate_cache)):
        source_latent, source_rope = source.keys_and_values()
        target_latent, target_rope = target.keys_and_values()
        layer = model.model.layers[index].self_attn
        rope_module = layer.rope
        if not getattr(rope_module, "traditional", False):
            raise ValueError("probe supports adjacent-pair traditional RoPE only")
        denominators = getattr(rope_module, "_freqs", None)
        if denominators is None:
            raise ValueError("probe needs the layer's actual RoPE frequency denominators")
        source_slice = slice(len(source_prefix), len(source_prefix) + len(chunk))
        target_slice = slice(len(target_prefix), len(target_prefix) + len(chunk))
        candidate = propose_shifted_reuse(
            source_latent=host_float32(source_latent)[0, 0, source_slice],
            source_rotated_rope=host_float32(source_rope)[0, 0, source_slice],
            source_positions=source_positions,
            target_positions=target_positions,
            source_token_ids=chunk,
            target_token_ids=chunk,
            frequency_denominators=host_float32(denominators),
            contract=contract,
        )
        comparison = compare_reuse_to_recomputed(
            candidate,
            recomputed_target_latent=host_float32(target_latent)[0, 0, target_slice],
            recomputed_target_rope=host_float32(target_rope)[0, 0, target_slice],
            atol=0.0,
        )
        latent_error = max(latent_error, comparison.latent_max_abs_error)
        rope_error = max(rope_error, comparison.rope_max_abs_error)
        destination.update_and_fetch(
            mx.array(candidate.latent[None, None], dtype=source_latent.dtype),
            mx.array(candidate.rotated_rope[None, None], dtype=source_rope.dtype),
        )

    probe_ids = mx.array(continuation[None, :])
    exact_hidden = model.model(probe_ids, cache=target_cache)
    candidate_hidden = model.model(probe_ids, cache=candidate_cache)
    exact_logits = model.lm_head(exact_hidden)
    candidate_logits = model.lm_head(candidate_hidden)
    mx.eval(exact_hidden, candidate_hidden, exact_logits, candidate_logits)
    exact_logits_host = host_float32(exact_logits)
    candidate_logits_host = host_float32(candidate_logits)
    hidden_delta = host_float32(exact_hidden).astype(np.float64) - host_float32(candidate_hidden).astype(np.float64)
    logit_delta = exact_logits_host.astype(np.float64) - candidate_logits_host.astype(np.float64)
    return ShiftedReuseProbeReport(
        model_revision=model_revision,
        layer_count=len(source_cache),
        source_prefix_tokens=len(source_prefix),
        target_prefix_tokens=len(target_prefix),
        shared_chunk_tokens=len(chunk),
        continuation_tokens=len(continuation),
        max_latent_abs_error=latent_error,
        max_rope_abs_error=rope_error,
        max_hidden_abs_error=float(np.max(np.abs(hidden_delta))),
        max_logits_abs_error=float(np.max(np.abs(logit_delta))),
        logits_rms_error=float(np.sqrt(np.mean(logit_delta**2))),
        greedy_token_agreement=bool(np.array_equal(
            np.argmax(exact_logits_host, axis=-1),
            np.argmax(candidate_logits_host, axis=-1),
        )),
    )
