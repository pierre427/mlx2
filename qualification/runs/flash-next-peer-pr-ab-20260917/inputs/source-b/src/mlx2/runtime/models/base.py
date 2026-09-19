# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import inspect
import math
import os
from dataclasses import dataclass
from typing import Optional
import mlx.core as mx
from mlx.utils import tree_map

_HADAMARD_BASES = (1, 12, 20, 28)


def hadamard_size_ok(n: int) -> bool:
    """True if ``mx.hadamard_transform`` supports a last-dim of size ``n``
    (i.e. ``n = m * 2**k`` for ``m`` in {1, 12, 20, 28})."""
    for m in _HADAMARD_BASES:
        if n % m == 0 and n // m & n // m - 1 == 0:
            return True
    return False


def rotate_last(x: mx.array) -> mx.array:
    """Orthonormal Walsh–Hadamard transform along the last axis.

    Normalized by ``1/sqrt(D)`` so the transform is orthonormal (``R^T R = I``).
    Applying it to *both* queries and keys leaves every ``q·k`` inner product —
    and hence the attention scores — unchanged in exact arithmetic, while
    spreading per-channel outliers into a near-Gaussian marginal that low-bit
    affine quantization handles far more gracefully. Only orthonormality is
    relied on here; the transform is additionally self-inverse only for
    power-of-two ``D``."""
    return mx.hadamard_transform(x, scale=1.0 / math.sqrt(x.shape[-1]))


def _expand_kv_scale(scale: mx.array, n_q_heads: int) -> mx.array:
    """Broadcast a per-kv-head channel scale ``(B, n_kv_heads, 1, D)`` up to the
    query heads. Under grouped-query attention several query heads share one kv
    head, so each kv-head scale is repeated ``n_q_heads // n_kv_heads`` times
    along the head axis to line up with a ``(B, n_q_heads, L, D)`` tensor."""
    n_repeats = n_q_heads // scale.shape[1]
    if n_repeats > 1:
        scale = mx.repeat(scale, n_repeats, axis=1)
    return scale


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for (k, v) in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims(offset + N - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


_QSDPA_ENV = os.environ.get("MLX_LM_QSDPA_FLASH_MIN_L")
if _QSDPA_ENV is not None and int(_QSDPA_ENV) <= 0:
    _QUANT_SDPA_FLASH_MIN_L_GQA = _QUANT_SDPA_FLASH_MIN_L_MHA = float("inf")
elif _QSDPA_ENV is not None:
    _QUANT_SDPA_FLASH_MIN_L_GQA = _QUANT_SDPA_FLASH_MIN_L_MHA = int(_QSDPA_ENV)
else:
    _QUANT_SDPA_FLASH_MIN_L_GQA = 128
    _QUANT_SDPA_FLASH_MIN_L_MHA = 192


def _contiguous_quant(q):
    """Return quantized cache views directly on mlx 0.32.2 or newer.

    mlx 0.32.2 includes ml-explore/mlx#4381, which fixes the strided
    ``mx.dequantize`` corruption tracked as mlx#4370. Keeping this helper as
    the common call-site seam makes the compatibility boundary explicit while
    avoiding the materializing copies required by mlx 0.32.1.
    """
    return q


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
    key_bits: Optional[int] = None,
    value_bits: Optional[int] = None,
) -> mx.array:
    (B, n_q_heads, L, D) = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    flash_min_l = (
        _QUANT_SDPA_FLASH_MIN_L_MHA if n_repeats == 1 else _QUANT_SDPA_FLASH_MIN_L_GQA
    )
    key_bits = bits if key_bits is None else key_bits
    value_bits = bits if value_bits is None else value_bits
    if L >= flash_min_l:
        keys = mx.dequantize(
            *_contiguous_quant(q_keys), group_size=group_size, bits=key_bits
        )
        values = mx.dequantize(
            *_contiguous_quant(q_values), group_size=group_size, bits=value_bits
        )
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
    queries *= scale
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)
    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=key_bits
    )
    if mask is not None:
        if isinstance(mask, str):
            (qL, kL) = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if n_repeats > 1 and mask.ndim == scores.ndim - 1:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=value_bits
    )
    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    bucketed_attention = getattr(cache, "bucketed_attention", None)
    if bucketed_attention is not None:
        output = bucketed_attention(queries, scale, mask, sinks=sinks)
        if output is not None:
            return output
    if hasattr(cache, "bits"):
        if sinks is not None:
            raise ValueError("Quantized SDPA does not support attention sinks.")
        if getattr(cache, "rotate", False) and hadamard_size_ok(queries.shape[-1]):
            queries = rotate_last(queries)
        normalize = getattr(cache, "normalize", False)
        key_scale = getattr(cache, "key_scale", None)
        value_scale = getattr(cache, "value_scale", None)
        if normalize and key_scale is not None:
            queries = queries * _expand_kv_scale(key_scale, queries.shape[1])
        legacy_bits = cache.bits
        out = quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            key_bits=getattr(cache, "key_bits", legacy_bits),
            value_bits=getattr(cache, "value_bits", legacy_bits),
        )
        if normalize and value_scale is not None:
            out = out * _expand_kv_scale(value_scale, out.shape[1])
        return out
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks
        )
