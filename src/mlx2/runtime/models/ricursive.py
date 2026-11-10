# SPDX-License-Identifier: Apache-2.0
# Adapted from the pinned Ricursive model; see provenance/ripple-65b.json.
"""Ripple tensor math; ordinary BF16 activations over source NVFP4 MLPs.

The source's W4A4 activation quantizer is deliberately not inferred from its
weight storage. The serving adapter requires explicit W4A16 selection.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx import nn

from ..prefill_checkpoint_rows import select_checkpoint_rows
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .modelopt_nvfp4 import ModelOptNVFP4Linear, shared_fp32_input

# A layer computes its own mask when the forward hands it none (sentinel:
# ``create_attention_mask`` legitimately returns ``None`` for one-token
# global attention).
_OWN_MASK = object()

def window_blocked_attention(q, k, v, *, scale, mask, window, block=256):
    """Sliding-window prefill SDPA over each row block's visible keys only.

    For a chunk of ``L`` query rows against temporally ordered keys (a plain
    RotatingKVCache returns them in order after a multi-row update), rows
    ``[r0, r1)`` can see only keys ``[N - L + r0 - window + 1, N - L + r1)``;
    the rest of their tiles is fully masked. Each block runs SDPA on its own
    key slice and mask slice: the same keys and mask per row as stock, but
    a different reduction tiling (not bit-identical). Returns ``None`` when
    the mask is not an explicit ``[..., L, N]`` array.
    """
    length, n = q.shape[2], k.shape[2]
    if not isinstance(mask, mx.array) or mask.shape[-2:] != (length, n) or length <= block:
        return None
    outs = []
    for r0 in range(0, length, block):
        r1 = min(length, r0 + block)
        k0 = max(0, n - length + r0 - window + 1)
        k1 = n - length + r1
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, r0:r1], k[:, :, k0:k1], v[:, :, k0:k1], scale=scale, mask=mask[..., r0:r1, k0:k1]))
    return mx.concatenate(outs, axis=2)


def gqa_folded_attention(q, k, v, *, scale, mask):
    """GQA attention with each KV head's query heads folded into rows.

    ``q`` ``[B, Hq, L, D]`` becomes ``[B, Hkv, G*L, D]`` (row = g*L + l), so
    the kernel reads each KV head once for all ``G`` query heads and gets a
    tile ``G`` times taller. The row mask is tiled ``G`` times; ``"causal"``
    becomes the explicit ``[L, N]`` mask at offset ``N - L``. The same keys
    and values per row as stock SDPA, but a different reduction schedule, so
    not bit-identical (near-tie validation, not exact parity).
    Returns ``None`` when the mask has per-head structure (caller falls back).
    """
    from .base import create_causal_mask

    b, hq, length, d = q.shape
    hkv, n = k.shape[1], k.shape[2]
    g = hq // hkv
    if isinstance(mask, str):
        if mask != "causal":
            return None
        mask = create_causal_mask(length, offset=n - length)
    if mask is not None:
        if mask.ndim > 4 or (mask.ndim >= 3 and mask.shape[-3] != 1):
            return None
        mask = mx.tile(mask, (*(1,) * (mask.ndim - 2), g, 1))
    rows = q.reshape(b, hkv, g * length, d)
    out = mx.fast.scaled_dot_product_attention(rows, k, v, scale=scale, mask=mask)
    return out.reshape(b, hq, length, d)


# ---------------------------------------------------------------------------
# Fused elementwise regions (opt-in: ``Model.set_fused_elementwise``).
#
# Each region is a pure function of arrays holding exactly the eager
# expressions below it (same ops, same dtypes, same cast points); matmuls,
# cache updates and SDPA stay eager. ``mx.compile`` fuses each region's
# elementwise chain into a few kernels instead of one kernel per op, which is
# most of Ripple's ~105 kernels per decode layer. Exactness against the eager
# path is a gate, not an assumption: tests/test_ripple_fused_cpu.py checks it
# bit for bit on CPU and scripts/research/ripple_fused_gate.py on Metal,
# region by region, because Metal JIT kernels may round transcendentals
# differently from the precompiled ones.
# ---------------------------------------------------------------------------
_FUSED = {}


def _compiled(key, factory):
    fn = _FUSED.get(key)
    if fn is None:
        fn = _FUSED[key] = mx.compile(factory())
    return fn


def _rms(x, w, eps):
    y = x.astype(mx.float32)
    y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + eps)
    return y.astype(x.dtype) * w


# Transcendentals never run inside a fused region: a JIT kernel's rsqrt or
# exp need not round like the precompiled one (on CPU, compiled FP32 rsqrt
# differs by one ULP at some row shapes). The row statistic (square, mean,
# +eps, rsqrt) therefore runs eagerly on the [rows, 1] tensor, with only the
# square fused, and sigmoid goes through precise_ops.sigmoid, which on Metal
# is a custom kernel compile cannot fuse and elsewhere is mx.sigmoid.

def _square32(x):
    y = x.astype(mx.float32)
    return y * y


def _inverse_rms(x, eps):
    squares = _compiled(("square",), lambda: _square32)(x)
    return mx.rsqrt(mx.mean(squares, axis=-1, keepdims=True) + eps)


def _scale(x, inv, w):
    return (x.astype(mx.float32) * inv).astype(x.dtype) * w


def fused_norm(x, w, eps):
    return _compiled(("scale",), lambda: _scale)(x, _inverse_rms(x, eps), w)


def fused_residual_norm(residual, x, w, eps):
    """``residual + rms(x) * w`` (a block's post-norm and residual add)."""
    return _compiled(("residual_scale",), lambda: lambda r, x, inv, w: r + _scale(x, inv, w))(
        residual, x, _inverse_rms(x, eps), w)


def fused_qk(raw, w, eps, cos=None, sin=None):
    """Per-head RMSNorm, head-major transpose and (local layers) RoPE."""
    inv = _inverse_rms(raw, eps)
    if cos is None:
        return _compiled(("qk",), lambda: lambda a, i, w: _scale(a, i, w).transpose(0, 2, 1, 3))(raw, inv, w)
    return _compiled(("qk_rope",), lambda: lambda a, i, w, c, s: apply_rotary(
        _scale(a, i, w).transpose(0, 2, 1, 3), c, s))(raw, inv, w, cos, sin)


def _blend(seed32, x, blend):
    return (seed32 + blend * (x.astype(mx.float32) - seed32)).astype(x.dtype)


def fused_blend(seed32, x, blend):
    return _compiled(("blend",), lambda: _blend)(seed32, x, blend)


def _gate(out, g, full_precision):
    gate = g.astype(mx.float32) if full_precision else g
    gate = mx.sigmoid(gate).astype(out.dtype)
    b, h, length, d = out.shape
    return (out.transpose(0, 2, 1, 3) * gate.reshape(b, length, h, d)).reshape(b, length, -1)


def _gate_product(out, sig, dtype):
    b, h, length, d = out.shape
    return (out.transpose(0, 2, 1, 3) * sig.astype(dtype).reshape(b, length, h, d)).reshape(b, length, -1)


def fused_gate(out, g, full_precision):
    from .precise_ops import sigmoid

    sig = sigmoid(g.astype(mx.float32) if full_precision else g)
    return _compiled(("gate_product", out.dtype), lambda: lambda o, s: _gate_product(o, s, out.dtype))(out, sig)


def _swiglu(gate, up, clamp):
    if clamp:
        gate = mx.minimum(gate, clamp)
        up = mx.clip(up, -clamp, clamp)
    gate32 = gate.astype(mx.float32)
    return (gate32 * mx.sigmoid(gate32)).astype(gate.dtype) * up


def _swiglu_inputs(gate, up, clamp):
    if clamp:
        gate = mx.minimum(gate, clamp)
        up = mx.clip(up, -clamp, clamp)
    return gate.astype(mx.float32), up


def _swiglu_product(gate32, sig, up, dtype):
    return (gate32 * sig).astype(dtype) * up


def fused_swiglu(gate, up, clamp):
    from .precise_ops import sigmoid

    gate32, up = _compiled(("swiglu_inputs", clamp), lambda: lambda g, u: _swiglu_inputs(g, u, clamp))(gate, up)
    return _compiled(("swiglu_product", gate.dtype), lambda: lambda g, s, u: _swiglu_product(g, s, u, gate.dtype))(
        gate32, sigmoid(gate32), up)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "ricursive"
    hidden_size: int = 5120
    intermediate_size: int = 40960
    num_hidden_layers: int = 80
    num_attention_heads: int = 80
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 100352
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sliding_window: int = 1024
    layer_types: list[str] | None = None
    embed_scale: float | None = 71.55417527999327
    ffn_clamp_threshold: float | None = 20.0
    use_gated_attention: bool = True
    gate_full_precision: bool = True
    use_valres: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 5 == 4 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or any(
            t not in {"full_attention", "sliding_attention"} for t in self.layer_types
        ):
            raise ValueError("invalid Ripple layer types")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("invalid Ripple attention geometry")
        if self.sliding_window < 1 or self.tie_word_embeddings:
            raise ValueError("invalid Ripple window or tied embeddings")


class RMSNorm(nn.Module):
    """Source rounding: normalize FP32, cast, then multiply the norm weight."""

    def __init__(self, dims, eps):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    fused = False
    metal = False

    def __call__(self, x):
        if self.metal:
            from .ricursive_rmsnorm import rms_norm

            out = rms_norm(x, self.weight, self.eps)
            if out is not None:
                return out
        if self.fused:
            return fused_norm(x, self.weight, self.eps)
        return _rms(x, self.weight, self.eps)

    def residual(self, residual, x):
        """``residual + self(x)``; one launch under the Metal kernel."""
        if self.metal:
            from .ricursive_rmsnorm import rms_norm

            out = rms_norm(x, self.weight, self.eps, residual=residual)
            if out is not None:
                return out
        if self.fused:
            return fused_residual_norm(residual, x, self.weight, self.eps)
        return residual + self(x)


def scale_embeddings(h, scale):
    """Source ``bf16_embeds * python_float``: torch multiplies in FP32 opmath by
    the full scalar, then rounds once. ``h * scale`` in MLX would first round
    the scalar to the activation dtype (71.554... -> 71.5 in BF16)."""
    return (h.astype(mx.float32) * scale).astype(h.dtype)


_INVERSE_FREQUENCIES = {}


def inverse_frequencies(dims, theta):
    """Correctly rounded FP32 ``1 / theta ** (2i / dims)``, as torch computes.

    MLX's FP32 ``pow`` is a few ULP off for some exponents, which moves about a
    quarter of the phases by one FP32 step and a small share of the
    BF16-rounded cos/sin values.

    The table is a constant of ``(dims, theta)``: it is built on the host and
    uploaded once, not once per projection per layer.
    """
    key = (int(dims), float(theta))
    table = _INVERSE_FREQUENCIES.get(key)
    if table is None:
        exponent = np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims)
        power = np.power(np.float64(np.float32(theta)), exponent.astype(np.float64)).astype(np.float32)
        table = mx.array(np.float32(1.0) / power)
        _INVERSE_FREQUENCIES[key] = table
    return table


def rotary_tables(length, offset, dims, theta, dtype, *, positions=None):
    """Source cos/sin for ``length`` positions from ``offset``: FP32 trig,
    rounded through the activation ``dtype`` (``mx.fast.rope`` would skip that
    BF16 rounding).  The source doubles the half phase before the trig; the
    doubled table is kept so every value is computed over exactly the array
    the source computes it over, and the rotation reads its first half.

    Every sliding layer of one forward shares the same positions, so the
    model builds this once per forward rather than twice per layer.

    ``positions`` (FP32 ``[length]``) replaces the ``offset + row`` law with
    one explicit position per row: an attention-tree tile's rows sit at
    ``offset + depth``, not at ``offset + row index``.  Integer positions
    below 2**24 are exact in FP32, so the table equals the one serial decode
    builds at that position.
    """
    inv = inverse_frequencies(dims, theta)
    if positions is not None:
        if isinstance(offset, mx.array) and offset.ndim:
            raise ValueError("explicit rotary positions take one scalar offset")
        if positions.ndim != 1 or int(positions.shape[0]) != length:
            raise ValueError("explicit rotary positions must give one position per row")
        phase = (positions.astype(mx.float32) + offset)[:, None] * inv
    elif isinstance(offset, mx.array) and offset.ndim:
        positions = mx.arange(length, dtype=mx.float32)[None, :] + offset[:, None]
        phase = positions[:, None, :, None] * inv
    else:
        positions = mx.arange(length, dtype=mx.float32) + offset
        phase = positions[:, None] * inv
    phase = mx.concatenate((phase, phase), axis=-1)
    cos = mx.cos(phase).astype(dtype).astype(mx.float32)
    sin = mx.sin(phase).astype(dtype).astype(mx.float32)
    half = dims // 2
    return cos[..., :half], sin[..., :half]


def apply_rotary(x, cos, sin):
    """Split-half rotation ``y * cos + cat(-y2, y1) * sin`` computed per half.

    ``y1 * c + (-y2) * s`` is bit-identical to ``y1 * c - y2 * s`` (negation
    is exact and ``a + (-b)`` is ``a - b`` in IEEE arithmetic), so neither the
    negated half nor the doubled cos/sin tables are materialized, and the
    halves are rounded to ``x.dtype`` before the one concatenation.
    """
    half = x.shape[-1] // 2
    y = x.astype(mx.float32)
    first, second = y[..., :half], y[..., half:]
    return mx.concatenate(
        ((first * cos - second * sin).astype(x.dtype),
         (second * cos + first * sin).astype(x.dtype)),
        axis=-1,
    )


def rotary(x, offset, theta):
    return apply_rotary(x, *rotary_tables(x.shape[-2], offset, x.shape[-1], theta, x.dtype))


class Attention(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.args = args
        self.local = args.layer_types[index] == "sliding_attention"
        qdim = args.num_attention_heads * args.head_dim
        kvdim = args.num_key_value_heads * args.head_dim
        self.q_proj = nn.Linear(args.hidden_size, qdim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, kvdim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, kvdim, bias=False)
        self.o_proj = nn.Linear(qdim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        if args.use_gated_attention:
            self.g_proj = nn.Linear(args.hidden_size, qdim, bias=False)
        if args.use_valres:
            self.value_res_lambda = mx.zeros((1,))

    def value_blend(self, dtype):
        """``sigmoid(lambda)`` rounded to the activation dtype, then FP32.

        Recomputed per call (three scalar ops per layer). An identity-keyed
        cache went stale under in-place parameter updates such as
        ``value_res_lambda[0] = v``, which keep the array object.
        """
        return mx.sigmoid(self.value_res_lambda[0]).astype(dtype).astype(mx.float32)

    def __call__(self, x, seed, cache=None, *, mask=_OWN_MASK, rope=None):
        """``mask``/``rope``: the forward's per-layer-kind mask and cos/sin
        tables (``Backbone`` builds each once).  Absent, the layer builds its
        own: the same values, once per layer."""
        if self.fused:
            return self._fused_call(x, seed, cache, mask=mask, rope=rope)
        args = self.args
        b, length, _ = x.shape
        if self.metal_glue:
            out = self._glue_call(x, seed, cache, mask=mask, rope=rope)
            if out is not None:
                return out
        q = self.q_norm(self.q_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, length, args.num_key_value_heads, args.head_dim))
        value_input = x
        if args.use_valres:
            # torch.lerp computes pointwise in opmath FP32 for BF16 inputs.
            # ``seed`` arrives in FP32 from the forward (one cast for all 80
            # layers); the cast is the identity then.
            value_input = _blend(seed.astype(mx.float32), x, self.value_blend(x.dtype))
        v = self.v_proj(value_input).reshape(b, length, args.num_key_value_heads, args.head_dim)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        if mask is _OWN_MASK:
            mask = create_attention_mask(x, cache, window_size=args.sliding_window if self.local else None)
        if self.local:
            if rope is None:
                offset = cache.offset if cache is not None else 0
                rope = rotary_tables(length, offset, args.head_dim, args.rope_theta, x.dtype)
            q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        return self._attend(x, q, k, v, cache, mask)

    fused = False
    metal_glue = False
    window_blocked_prefill = False
    window_blocked_calls = 0

    def _glue_call(self, x, seed, cache, *, mask, rope):
        """Eager forward with QK-norm+RoPE and the value blend as Metal kernels.

        ``None`` (nothing computed) when a kernel cannot take this shape, so
        the caller runs the eager law.
        """
        from .ricursive_glue import blend, qk_norm_rope

        args = self.args
        b, length, _ = x.shape
        if mask is _OWN_MASK:
            mask = create_attention_mask(x, cache, window_size=args.sliding_window if self.local else None)
        if self.local and rope is None:
            offset = cache.offset if cache is not None else 0
            rope = rotary_tables(length, offset, args.head_dim, args.rope_theta, x.dtype)
        tables = rope if self.local else (None, None)
        q = qk_norm_rope(self.q_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim),
                         self.q_norm.weight, args.rms_norm_eps, *tables)
        if q is None:
            return None
        k = qk_norm_rope(self.k_proj(x).reshape(b, length, args.num_key_value_heads, args.head_dim),
                         self.k_norm.weight, args.rms_norm_eps, *tables)
        if k is None:
            return None
        value_input = x
        if args.use_valres:
            value_input = blend(seed.astype(mx.float32), x, self.value_blend(x.dtype))
            if value_input is None:
                return None
        v = self.v_proj(value_input).reshape(
            b, length, args.num_key_value_heads, args.head_dim).transpose(0, 2, 1, 3)
        return self._attend(x, q, k, v, cache, mask)

    # Rows (decode and verify) up to which attention runs GQA-folded; 0 = off.
    gqa_fold_rows = 0
    gqa_fold_calls = 0

    def _fused_call(self, x, seed, cache, *, mask, rope):
        """The eager forward with its elementwise chains in fused regions."""
        args = self.args
        b, length, _ = x.shape
        if mask is _OWN_MASK:
            mask = create_attention_mask(x, cache, window_size=args.sliding_window if self.local else None)
        if self.local and rope is None:
            offset = cache.offset if cache is not None else 0
            rope = rotary_tables(length, offset, args.head_dim, args.rope_theta, x.dtype)
        tables = rope if self.local else (None, None)
        eps = args.rms_norm_eps
        q = fused_qk(self.q_proj(x).reshape(b, length, args.num_attention_heads, args.head_dim),
                     self.q_norm.weight, eps, *tables)
        k = fused_qk(self.k_proj(x).reshape(b, length, args.num_key_value_heads, args.head_dim),
                     self.k_norm.weight, eps, *tables)
        value_input = x
        if args.use_valres:
            value_input = fused_blend(seed.astype(mx.float32), x, self.value_blend(x.dtype))
        v = self.v_proj(value_input).reshape(
            b, length, args.num_key_value_heads, args.head_dim).transpose(0, 2, 1, 3)
        return self._attend(x, q, k, v, cache, mask)

    @staticmethod
    def _folded_rows(q, cache, mask, scale):
        outputs = []
        for index, valid, keys, values, row_mask in cache.row_views(mask):
            if not valid:
                outputs.append(mx.zeros(q[index:index + 1].shape, dtype=q.dtype))
                continue
            value = gqa_folded_attention(q[index:index + 1, :, :valid], keys, values,
                                         scale=scale, mask=row_mask)
            if value is None:
                return None  # per-head mask: the caller's stock path
            if valid < q.shape[2]:
                value = mx.pad(value, [(0, 0), (0, 0), (0, q.shape[2] - valid), (0, 0)])
            outputs.append(value)
        cache.note_attention()
        return mx.concatenate(outputs, axis=0)

    def _attend(self, x, q, k, v, cache, mask):
        args = self.args
        b, length, _ = x.shape
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = None
        if 1 <= length <= self.gqa_fold_rows and not hasattr(cache, "bits"):
            scale = args.head_dim ** -0.5
            if hasattr(cache, "row_views"):
                # Speculative verify transactions: the cache's public per-lane
                # seam (each lane's own history and mask), the contract its
                # bucketed_attention implements with stock SDPA.
                out = self._folded_rows(q, cache, mask, scale)
            elif getattr(cache, "bucketed_attention", None) is None:
                out = gqa_folded_attention(q, k, v, scale=scale, mask=mask)
            if out is not None:
                self.gqa_fold_calls += 1
        if (
            out is None
            and self.window_blocked_prefill
            and self.local
            and type(cache) is RotatingKVCache
            and length > self.gqa_fold_rows
        ):
            out = window_blocked_attention(q, k, v, scale=args.head_dim ** -0.5, mask=mask,
                                           window=args.sliding_window)
            if out is not None:
                self.window_blocked_calls += 1
        if out is None:
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=args.head_dim ** -0.5, mask=mask)
        return self._output(x, out)

    def _output(self, x, out):
        """Attention output ``out`` [B, H, L, D] -> gated, projected [B, L, hidden]."""
        args = self.args
        b, length, _ = x.shape
        if args.use_gated_attention and self.fused:
            return self.o_proj(fused_gate(out, self.g_proj(x), args.gate_full_precision))
        if args.use_gated_attention:
            gate = self.g_proj(x)
            if args.gate_full_precision:
                gate = gate.astype(mx.float32)
            gate = mx.sigmoid(gate).astype(out.dtype)
            # Multiply in [B, L, H, D]: the transposed attention output is a
            # view and the gate reshapes for free, the product lands
            # contiguous, so the final reshape is a view instead of the copy
            # a transposed reshape materializes before the multiply.
            out = out.transpose(0, 2, 1, 3) * gate.reshape(b, length, args.num_attention_heads, args.head_dim)
        else:
            out = out.transpose(0, 2, 1, 3)
        return self.o_proj(out.reshape(b, length, -1))


class MLP(nn.Module):
    def __init__(self, args, *, nvfp4):
        super().__init__()
        linear = ModelOptNVFP4Linear if nvfp4 else nn.Linear
        self.gate_proj = linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = linear(args.intermediate_size, args.hidden_size, bias=False)
        self.clamp = args.ffn_clamp_threshold

    def __call__(self, x):
        shared = shared_fp32_input((self.gate_proj, self.up_proj), x)
        if shared is None:
            gate, up = self.gate_proj(x), self.up_proj(x)
        else:
            # Native NVFP4 projections cast their input to FP32 and round
            # their FP32 product to the input dtype; given one FP32 copy, the
            # same product is rounded here, once per projection as before.
            gate, up = self.gate_proj(shared).astype(x.dtype), self.up_proj(shared).astype(x.dtype)
        # Match source's separately rounded SiLU and multiplication (one FP32
        # copy of the clamped gate feeds both factors).
        if self.metal_glue:
            from .ricursive_glue import swiglu

            product = swiglu(gate, up, self.clamp)
            if product is not None:
                return self.down_proj(product)
        if self.fused:
            return self.down_proj(fused_swiglu(gate, up, self.clamp))
        return self.down_proj(_swiglu(gate, up, self.clamp))

    fused = False
    metal_glue = False


class DecoderLayer(nn.Module):
    def __init__(self, args, index, *, nvfp4):
        super().__init__()
        self.self_attn = Attention(args, index)
        self.mlp = MLP(args, nvfp4=nvfp4)
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    fused = False

    def __call__(self, x, seed, cache, *, mask=_OWN_MASK, rope=None):
        attn = self.self_attn(self.input_layernorm(x), seed, cache, mask=mask, rope=rope)
        if self.post_attention_layernorm.metal:
            x = self.post_attention_layernorm.residual(x, attn)
            mlp = self.mlp(self.pre_feedforward_layernorm(x))
            return self.post_feedforward_layernorm.residual(x, mlp)
        if self.fused:
            eps = self.post_attention_layernorm.eps
            x = fused_residual_norm(x, attn, self.post_attention_layernorm.weight, eps)
            mlp = self.mlp(self.pre_feedforward_layernorm(x))
            return fused_residual_norm(x, mlp, self.post_feedforward_layernorm.weight, eps)
        x = x + self.post_attention_layernorm(attn)
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class Backbone(nn.Module):
    def __init__(self, args, *, nvfp4):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i, nvfp4=nvfp4) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)

    def __call__(self, tokens, cache=None, input_embeddings=None, capture=None):
        """``capture``: ``{layer index: None}`` filled with post-block,
        pre-final-norm states (the DFlash2 draft feature law)."""
        h = self.embed_tokens(tokens) if input_embeddings is None else input_embeddings
        if self.args.embed_scale is not None:
            h = scale_embeddings(h, self.args.embed_scale)
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Ripple cache must contain exactly one entry per layer")
        # One FP32 seed for every layer's value blend (each layer cast it
        # twice), and one mask per layer kind plus one RoPE table for the
        # whole forward: the caches of one kind advance in lockstep, so each
        # layer's own mask and table would be the same values (the pattern
        # the other hybrid-window models here use).
        seed = h.astype(mx.float32)
        shared = self.shared_layer_inputs(h, cache)
        for index, (layer, entry, kind) in enumerate(zip(self.layers, cache, self.args.layer_types, strict=True)):
            mask, rope = shared[kind]
            h = layer(h, seed, entry, mask=mask, rope=rope)
            if capture is not None and index in capture:
                capture[index] = h
        return self.norm(h)

    # Attention-tree tile rows (``Model.attention_tree_position_scope``):
    # int32 ``[rows]`` depth of each row below the committed boundary, or
    # None for the ordinary ``offset + row`` law.
    _tree_depths = None

    def shared_layer_inputs(self, h, cache):
        """``{layer kind: (mask, rope)}`` built from the first cache of each kind.

        The masks come from each kind's first cache (``make_mask``); a tree
        tile installs its prefix-plus-ancestor masks there.  The sliding RoPE
        table places every row at ``offset + depth`` under a tree scope, and
        the global layers stay NoPE, so a tile row sees exactly the positions
        serial decoding of its root path would.
        """
        args = self.args
        depths = self._tree_depths
        if depths is not None and (h.shape[0] != 1 or int(h.shape[1]) != int(depths.shape[0])):
            raise ValueError("Ripple attention-tree rows must be one lane of one row per depth")
        shared = {}
        for index, kind in enumerate(args.layer_types):
            if kind in shared:
                continue
            local = kind == "sliding_attention"
            entry = cache[index]
            mask = create_attention_mask(h, entry, window_size=args.sliding_window if local else None)
            rope = None
            if local:
                offset = entry.offset if entry is not None else 0
                rope = rotary_tables(h.shape[1], offset, args.head_dim, args.rope_theta, h.dtype,
                                     positions=depths)
            shared[kind] = (mask, rope)
        return shared


class Model(nn.Module):
    def __init__(self, args, *, nvfp4=True):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.forward_calls = 0
        self.model = Backbone(args, nvfp4=nvfp4)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    fused_elementwise = False

    def set_fused_elementwise(self, enabled):
        """Route the per-layer elementwise chains through fused regions."""
        enabled = bool(enabled)
        for _, module in self.named_modules():
            if isinstance(module, (RMSNorm, Attention, MLP, DecoderLayer)):
                module.fused = enabled
        self.fused_elementwise = enabled

    def set_window_blocked_prefill(self, enabled):
        for layer in self.layers:
            layer.self_attn.window_blocked_prefill = bool(enabled)

    def window_blocked_calls(self):
        return sum(layer.self_attn.window_blocked_calls for layer in self.layers)

    def set_metal_glue(self, enabled):
        """QK-norm+RoPE, value blend and SwiGLU as one-launch Metal kernels."""
        for _, module in self.named_modules():
            if isinstance(module, (Attention, MLP)):
                module.metal_glue = bool(enabled)

    def set_metal_rmsnorm(self, enabled):
        """One-launch RMSNorm (block, per-head Q/K, final) on Metal."""
        for _, module in self.named_modules():
            if isinstance(module, RMSNorm):
                module.metal = bool(enabled)

    def set_gqa_folded_attention(self, max_rows):
        """Fold GQA heads into rows for calls of up to ``max_rows`` rows (0 = off)."""
        if type(max_rows) is not int or not 0 <= max_rows <= 64:
            raise ValueError("gqa folded attention rows must be an integer in 0..64")
        for layer in self.layers:
            layer.self_attn.gqa_fold_rows = max_rows

    def gqa_fold_calls(self):
        return sum(layer.self_attn.gqa_fold_calls for layer in self.layers)

    def __call__(self, tokens, cache=None, input_embeddings=None):
        self.forward_calls += 1
        h = self.model(tokens, cache, input_embeddings)
        return self.lm_head(select_checkpoint_rows(h))

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        """Target features for an external drafter: ascending post-block taps
        before the final norm (the vendor SGLang capture), concatenated on the
        hidden axis, plus every row's logits unless ``body_only``."""
        capture_layers = tuple(int(i) for i in capture_layers)
        if (
            not capture_layers
            or tuple(sorted(set(capture_layers))) != capture_layers
            or capture_layers[0] < 0
            or capture_layers[-1] >= len(self.layers)
        ):
            raise ValueError("Invalid target capture layers")
        self.forward_calls += 1
        taps = dict.fromkeys(capture_layers)
        hidden = self.model(inputs, cache, capture=taps)
        features = mx.concatenate([taps[i] for i in capture_layers], axis=-1)
        if body_only:
            return None, features
        return self.lm_head(hidden), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    @contextmanager
    def attention_tree_position_scope(self, depths):
        """Place the rows of the next forward at ``offset + depths[row]``.

        The attention-tree executor's seam for a model whose RoPE tables are
        built once per forward rather than by a per-layer ``rope`` module.
        Only the sliding layers rotate; the global layers are NoPE.  Masks
        are the executor's, installed on the caches.  Nested scopes are
        refused: a tile forward never runs inside another.
        """
        if self.model._tree_depths is not None:
            raise RuntimeError("Ripple attention-tree position scope is not reentrant")
        if not isinstance(depths, mx.array) or depths.ndim != 1 or depths.dtype != mx.int32:
            raise ValueError("attention-tree depths must be an int32 vector")
        # A plain instance attribute: ``nn.Module.__setattr__`` would register
        # the array as a parameter.
        object.__setattr__(self.model, "_tree_depths", depths)
        try:
            yield self
        finally:
            object.__setattr__(self.model, "_tree_depths", None)

    def make_cache(self):
        return [RotatingKVCache(self.args.sliding_window, keep=0)
                if kind == "sliding_attention" else KVCache()
                for kind in self.args.layer_types]
