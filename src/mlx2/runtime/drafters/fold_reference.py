"""CPU equation reference for the selected MLX FOLD draft read.

Serving attention is implemented separately in ``fold.py``. These source-bound
equations and component accounting do not qualify the complete live route.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FOLDPolicy:
    window: int
    alpha: float = 2.0
    sliding_layers: str = "window"

    def __post_init__(self):
        if type(self.window) is not int or self.window < 1:
            raise ValueError("FOLD window must be a positive integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or self.alpha < 1.0
        ):
            raise ValueError("FOLD alpha must be finite and at least one")
        if self.sliding_layers not in ("window", "fold"):
            raise ValueError("FOLD sliding_layers must be 'window' or 'fold'")

    @classmethod
    def from_value(cls, value) -> FOLDPolicy | None:
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("FOLD policy must be false, null, or a mapping")
        unknown = set(value) - {"window", "alpha", "sliding_layers"}
        if unknown:
            raise ValueError(f"unknown FOLD policy keys: {sorted(unknown)}")
        return cls(**dict(value))

    def as_dict(self) -> dict:
        return {
            "window": self.window,
            "alpha": float(self.alpha),
            "sliding_layers": self.sliding_layers,
        }


def side_cache_bytes(
    policy: FOLDPolicy,
    *,
    max_context: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> int:
    """Unrotated K+V side-cache reservation for full-context FOLD layers."""
    policy = FOLDPolicy.from_value(policy)
    values = (max_context, layers, kv_heads, head_dim, dtype_bytes)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("FOLD side-cache geometry must be positive integers")
    return 2 * max_context * layers * kv_heads * head_dim * dtype_bytes


def _rotate_neox(x, positions, inv_freq):
    x = np.asarray(x, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    inv_freq = np.asarray(inv_freq, dtype=np.float64)
    width = 2 * inv_freq.size
    if x.shape[-1] < width or x.shape[-1] % 2:
        raise ValueError("FOLD requires an even head dimension covering RoPE")
    angles = positions[:, None] * inv_freq[None, :]
    cosine = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1)[:, None]
    sine = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1)[:, None]
    first, second = np.split(x[..., :width], 2, axis=-1)
    rotated = x[..., :width] * cosine + np.concatenate((-second, first), axis=-1) * sine
    return np.concatenate((rotated, x[..., width:]), axis=-1)


def _softmax(value):
    value = np.asarray(value, dtype=np.float64)
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _logsumexp(value):
    maximum = np.max(value, axis=-1)
    return maximum + np.log(np.exp(value - maximum[..., None]).sum(axis=-1))


def fold_block(q, k_context, v_context, k_block, v_block, *, policy, inv_freq, scale):
    """Dense NumPy form of FOLD equations 1--4 for one proposal block.

    Inputs use ``[positions, heads, head_dim]`` and must already have GQA heads
    expanded. Queries and keys are unrotated. The target model and verifier do
    not call this function.
    """
    policy = FOLDPolicy.from_value(policy)
    if policy is None:
        raise ValueError("FOLD reference requires an enabled policy")
    arrays = [np.asarray(value, dtype=np.float64) for value in (q, k_context, v_context, k_block, v_block)]
    q, k_context, v_context, k_block, v_block = arrays
    if any(value.ndim != 3 for value in arrays):
        raise ValueError("FOLD tensors must be [positions, heads, head_dim]")
    block, context = q.shape[0], k_context.shape[0]
    if block < 1 or q.shape != k_block.shape or q.shape != v_block.shape:
        raise ValueError("FOLD proposal Q/K/V geometry mismatch")
    if k_context.shape != v_context.shape or k_context.shape[1:] != q.shape[1:]:
        raise ValueError("FOLD context K/V geometry mismatch")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("FOLD attention scale must be finite and positive")

    distance = policy.window + block
    query_positions = np.arange(context, context + block)
    key_positions = np.arange(context + block)
    keys = np.concatenate((k_context, k_block), axis=0)
    values = np.concatenate((v_context, v_block), axis=0)
    rotated_keys = _rotate_neox(keys, key_positions, inv_freq)
    output = np.empty_like(q)
    for row in range(block):
        position = context + row
        near = position - key_positions < distance
        far = ~near
        near_query = _rotate_neox(q[row : row + 1], query_positions[row : row + 1], inv_freq)[0]
        logits = np.einsum("hd,nhd->hn", near_query, rotated_keys) * float(scale)
        near_logits = logits[:, near]
        near_output = np.einsum("hn,nhd->hd", _softmax(near_logits), values[near])
        if not np.any(far):
            output[row] = near_output
            continue
        far_query = _rotate_neox(
            q[row : row + 1], np.asarray([distance + row]), inv_freq
        )[0]
        far_logits = np.einsum("hd,nhd->hn", far_query, keys[far]) * float(scale)
        concentrated = policy.alpha * far_logits
        far_output = np.einsum("hn,nhd->hd", _softmax(concentrated), values[far])
        masses = np.stack(
            (_logsumexp(near_logits), _logsumexp(concentrated) / policy.alpha),
            axis=0,
        )
        mixture = _softmax(masses.T).T
        output[row] = mixture[0, :, None] * near_output + mixture[1, :, None] * far_output
    return output


PROBE_CONTEXTS = (32768, 65536, 131072)
