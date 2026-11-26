# SPDX-License-Identifier: MIT
# The pure-MLX recurrence and gate laws were adapted from mlx-lm-unified
# (moved here verbatim from runtime/models/gated_delta.py); see
# docs/PROVENANCE.md and provenance/flashnext.json.
"""Linear recurrence: the gated delta rule behind GDN and KDA mixers.

One reference for every Qwen GatedDeltaNet and every Kimi-style KDA layer
(Kimi Linear, Ling/Bailing, Raon RDA, Kimi K3.1, GLM-5.3):

* :class:`LinearRecurrenceSpec` carries what differs between models: head
  geometry, gate kind, decay and beta laws, q/k normalization law, conv
  layout and precision, state class and the chunk-scan candidates.
  ``from_config`` derives the geometry and bounded-gate value from the
  checkpoint config; legacy numerics fields are set by each model.
* the pure-MLX laws: :func:`gates` (decay + beta), :func:`normalize_qk`,
  :func:`short_conv` / :func:`conv_window`, :func:`gated_delta_ops` (the
  sequential recurrence), :func:`cohort_tree_recurrence` (parent-indexed
  depth cohorts over the one-token law) and :func:`select_rollback_rows`
  (verify snapshot selection).
* one dispatcher, :func:`recurrence`, that owns the gradient guard, the
  reference fallback and the registered fast paths (``chunk_wy``,
  ``chunk32``, ``prefill_scan``, ``core_scan``, ``step_kernel``) and returns
  a :class:`~mlx2.runtime.technologies.dispatch.Receipt` on request.

The Metal kernels themselves stay in ``runtime/models/gated_delta.py`` (and
the fused Qwen family in ``qwen4_fused_gdn*.py``); the dispatcher looks them
up there at call time.  See docs/technologies/linear_recurrence.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..models import differentiable as _differentiable
from ..models import gdn_state as _gdn_state
from ..models.precise_ops import gate_sigmoid
from .dispatch import FastPath, KernelRegistry, Receipt, spec_from_config

# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

GATE_KINDS = ("scalar", "vector")
#: decay laws; the ``A`` operand each one reads is in DECAY_A_OPERAND.
DECAY_LAWS = (
    "softplus",                 # exp(-exp(A_log) * softplus(a + dt_bias))   (Qwen3.5 GDN)
    "softplus_float32_inputs",  # same, a and dt_bias widened first          (Flash-Next)
    "lower_bound_sigmoid",      # exp(lb * sigmoid(exp(A_log) * (a + dt)))   (Ling, K3.1)
    "log_softplus",             # -exp(A_log) * stable softplus, log domain  (Kimi Linear)
    "stored_a_sigmoid",         # exp(lb * gate_sigmoid((a + dt) * A))       (GLM-5.3 GGUF)
    "rda",                      # Raon RDA: lower-bounded log alpha + value write ratio
)
DECAY_A_OPERAND = {
    "softplus": "A_log",
    "softplus_float32_inputs": "A_log",
    "lower_bound_sigmoid": "A_log",
    "log_softplus": "A_log",
    "stored_a_sigmoid": "stored_A",
    "rda": "exp_A",
}
#: laws whose beta is computed inside the decay's compiled function.
JOINT_BETA_LAWS = frozenset({"log_softplus", "rda"})
BETA_LAWS = (
    "gate_sigmoid_float32",  # gate_sigmoid(b.astype(float32))
    "gate_sigmoid_input",    # gate_sigmoid(b) in the projection dtype
    "sigmoid_float32",       # mx.sigmoid(b.astype(float32))
)
QK_NORMS = (
    "rms_folded",  # inv**2 * rms_norm(q, eps=1e-6/Dk), inv * rms_norm(k): FLA l2norm
    "rms_mean",    # inv**2 * rms_norm(q, eps=1e-6), inv * rms_norm(k, eps=1e-6)  (Ling)
    "l2_rsqrt",    # q * rsqrt(sum(q^2) + 1e-6) * inv, k * rsqrt(...)
    "l2_clamp",    # q / max(||q||, 1e-6) * inv, k / max(||k||, 1e-6)
)
CONV_LAYOUTS = ("packed_qkv", "separate_qkv")
CONV_PRECISIONS = ("activation", "float32")
EXP_SPELLINGS = ("fast", "precise", "probed")
CHUNK_LOG_DECAYS = ("finite", "log_of_decay")


@dataclass(frozen=True)
class LinearRecurrenceSpec:
    """Every per-model parameter of a gated-delta mixer.

    ``exp_spelling`` documents what the fused kernels of this layer use (the
    pure-MLX reference always uses ``mx.exp``); ``chunk_log_decay`` records
    how a model hands the log decay to the opt-in WY chunk scan.
    """

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    gate_kind: str = "scalar"
    decay_law: str = "softplus"
    beta_law: str = "gate_sigmoid_float32"
    lower_bound: Optional[float] = None
    qk_norm: str = "rms_folded"
    qk_norm_dtype: str = "input"
    qk_norm_round: bool = False
    qk_eps: float = 1e-6
    conv_layout: str = "packed_qkv"
    conv_precision: str = "activation"
    state_dtype: str = "float32"
    exp_spelling: str = "fast"
    chunk_size: int = 0
    chunk_log_decay: str = "finite"
    core_scan: bool = True

    def __post_init__(self):
        checks = (
            ("gate_kind", GATE_KINDS), ("decay_law", DECAY_LAWS),
            ("beta_law", BETA_LAWS), ("qk_norm", QK_NORMS),
            ("qk_norm_dtype", ("input", "float32")),
            ("conv_layout", CONV_LAYOUTS), ("conv_precision", CONV_PRECISIONS),
            ("state_dtype", ("float32", "float16")),
            ("exp_spelling", EXP_SPELLINGS), ("chunk_log_decay", CHUNK_LOG_DECAYS),
        )
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name}={getattr(self, name)!r}; expected one of {allowed}")
        if self.num_k_heads <= 0 or self.num_v_heads % self.num_k_heads:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be a multiple of "
                f"num_k_heads ({self.num_k_heads})"
            )
        if self.decay_law in ("lower_bound_sigmoid", "stored_a_sigmoid", "rda") and (
            self.lower_bound is None
        ):
            raise ValueError(f"decay_law {self.decay_law!r} needs lower_bound")
        if self.decay_law in JOINT_BETA_LAWS and self.beta_law != "sigmoid_float32":
            raise ValueError(f"decay_law {self.decay_law!r} computes beta as sigmoid_float32")
        if self.chunk_size not in (0, 8, 16, 32):
            raise ValueError("chunk_size must be 0, 8, 16 or 32")

    @property
    def decay_a_operand(self) -> str:
        return DECAY_A_OPERAND[self.decay_law]

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "LinearRecurrenceSpec":
        """Derive geometry and the bounded gate from a checkpoint config.

        Understands the Qwen3.5-family ``linear_*`` fields, Kimi's nested
        ``linear_attn_config`` and the Ling/Bailing/Raon ``num_attention_heads``
        + ``head_dim`` + ``short_conv_kernel_size`` + ``kda_lower_bound``
        fields.  A KDA config (any of the latter two shapes) defaults to a
        vector gate over separate q/k/v convolutions; the decay/beta/qk laws a
        model's numerics depend on are passed as overrides.
        """
        def get(key):
            if isinstance(config, dict):
                return config.get(key)
            return getattr(config, key, None)

        linear = get("linear_attn_config")
        linear = linear if isinstance(linear, dict) else None
        kda = linear is not None or any(
            get(key) is not None
            for key in ("short_conv_kernel_size", "kda_lower_bound", "kda_heads")
        )

        def heads(_):
            if linear is not None:
                return linear.get("num_heads")
            return get("kda_heads") or (get("num_attention_heads") if kda else None)

        def head_dim(_):
            if linear is not None:
                return linear.get("head_dim")
            return get("kda_head_dim") or (get("head_dim") if kda else None)

        def conv_kernel(_):
            if linear is not None and linear.get("short_conv_kernel_size") is not None:
                return linear["short_conv_kernel_size"]
            return (get("linear_conv_kernel_dim") or get("short_conv_kernel_size")
                    or get("conv_kernel"))

        mapping = {
            "num_k_heads": lambda c: get("linear_num_key_heads") or heads(c),
            "num_v_heads": lambda c: get("linear_num_value_heads") or heads(c),
            "head_k_dim": lambda c: get("linear_key_head_dim") or head_dim(c),
            "head_v_dim": lambda c: get("linear_value_head_dim") or head_dim(c),
            "conv_kernel": conv_kernel,
            "lower_bound": ("kda_lower_bound", "gate_lower_bound"),
        }
        if kda:
            derived = {"gate_kind": "vector", "conv_layout": "separate_qkv"}
            bound = get("kda_lower_bound")
            if bound is None:
                bound = get("gate_lower_bound")
            # Kimi's nested config carries FLA's log-domain softplus gate; a
            # Ling-style config without a bound runs the softplus GDN law over
            # the vector gate.
            derived["decay_law"] = (
                "lower_bound_sigmoid" if bound is not None
                else "log_softplus" if linear is not None else "softplus"
            )
            if derived["decay_law"] == "log_softplus":
                derived["beta_law"] = "sigmoid_float32"
            overrides = {**derived, **overrides}
        return spec_from_config(cls, config, mapping, **overrides)


#: Qwen3.5 GDN defaults (packed q/k/v conv, scalar softplus gate).
QWEN35_GDN = LinearRecurrenceSpec()

# ---------------------------------------------------------------------------
# gate laws
# ---------------------------------------------------------------------------


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_g_float32_inputs(A_log, a, dt_bias):
    """Qwen4-Exp decay: widen before the dynamic addition and softplus."""
    a = a.astype(mx.float32)
    dt_bias = dt_bias.astype(mx.float32)
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


@partial(mx.compile, shapeless=True)
def compute_lower_bound_g(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound
        * mx.sigmoid(
            mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias)
        )
    )


@mx.compile
def kda_gates(a, beta, a_log, dt_bias):
    """Kimi Linear's FLA gate: (finite fp32 log decay, fp32 sigmoid beta)."""
    shifted = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    # Stable softplus; checkpoint FLA gate returns log-decay in fp32.
    softplus = mx.maximum(shifted, 0) + mx.log1p(mx.exp(-mx.abs(shifted)))
    # Log decay stays finite where exp() of it underflows to zero.
    return (-mx.exp(a_log.astype(mx.float32)) * softplus,
            mx.sigmoid(beta.astype(mx.float32)))


@partial(mx.compile, shapeless=True)
def rda_gates(decay_raw, dt_bias, a_exp, beta_raw, write_raw, lower_bound):
    """fp32 RDA gates: (decay = exp(log_alpha), beta, value ratio)."""
    scaled = a_exp * (decay_raw.astype(mx.float32) + dt_bias.astype(mx.float32))
    log_alpha = lower_bound * mx.sigmoid(scaled)
    b32 = beta_raw.astype(mx.float32)
    beta = mx.sigmoid(b32)
    # sigmoid(b + w) / sigmoid(b) in log space, as the reference.
    ratio = mx.exp(nn.log_sigmoid(b32 + write_raw.astype(mx.float32)) - nn.log_sigmoid(b32))
    return mx.exp(log_alpha), beta, ratio


class Gates(NamedTuple):
    decay: mx.array
    beta: mx.array
    log_decay: Optional[mx.array] = None
    value_ratio: Optional[mx.array] = None


def decay(law: str, A, a, dt_bias, lower_bound=None):
    """The decay ``g`` of one law (not a joint-beta law)."""
    if law == "softplus":
        return compute_g(A, a, dt_bias)
    if law == "softplus_float32_inputs":
        return compute_g_float32_inputs(A, a, dt_bias)
    if law == "lower_bound_sigmoid":
        return compute_lower_bound_g(A, a, dt_bias, lower_bound)
    if law == "stored_a_sigmoid":
        a = a + dt_bias
        return mx.exp(lower_bound * gate_sigmoid(a * A))
    raise ValueError(f"decay law {law!r} is computed jointly by gates()")


def gdn_update_flags(spec: LinearRecurrenceSpec) -> dict:
    """``gated_delta_update`` keyword flags for a scalar-gate spec's laws.

    Only non-default laws produce a flag, so a Qwen3.5-law layer passes none.
    """
    flags = {}
    if spec.beta_law == "gate_sigmoid_input":
        flags["beta_input_dtype"] = True
    elif spec.beta_law != "gate_sigmoid_float32":
        raise ValueError(f"gated_delta_update has no beta law {spec.beta_law!r}")
    if spec.decay_law == "softplus_float32_inputs":
        flags["decay_input_float32"] = True
    elif spec.decay_law == "lower_bound_sigmoid":
        flags["lower_bound"] = spec.lower_bound
    elif spec.decay_law != "softplus":
        raise ValueError(f"gated_delta_update has no decay law {spec.decay_law!r}")
    return flags


def beta(law: str, b):
    if law == "gate_sigmoid_float32":
        return gate_sigmoid(b.astype(mx.float32))
    if law == "gate_sigmoid_input":
        return gate_sigmoid(b)
    if law == "sigmoid_float32":
        return mx.sigmoid(b.astype(mx.float32))
    raise ValueError(f"unknown beta law {law!r}")


def gates(spec: LinearRecurrenceSpec, a, b, A, dt_bias, *, write=None,
          lower_bound=None) -> Gates:
    """Decay and beta of ``spec``'s laws.

    ``A`` is the operand named by ``spec.decay_a_operand`` (``A_log`` for the
    softplus/lower-bound/log-softplus laws, the stored ``A`` for GLM, the
    precomputed ``exp(A_log)`` for RDA). ``lower_bound`` overrides the spec's
    value with an array when a model keeps it resident (Raon).
    """
    bound = spec.lower_bound if lower_bound is None else lower_bound
    law = spec.decay_law
    if law == "log_softplus":
        log_decay, gate_beta = kda_gates(a, b, A, dt_bias)
        return Gates(mx.exp(log_decay), gate_beta, log_decay=log_decay)
    if law == "rda":
        if write is None:
            raise ValueError("rda gates need the write projection")
        gate_decay, gate_beta, ratio = rda_gates(a, dt_bias, A, b, write, bound)
        return Gates(gate_decay, gate_beta, value_ratio=ratio)
    if law == "stored_a_sigmoid":
        # Two-step GLM order: decay first, then beta.
        g = decay(law, A, a, dt_bias, bound)
        return Gates(g, beta(spec.beta_law, b))
    gate_beta = beta(spec.beta_law, b)
    return Gates(decay(law, A, a, dt_bias, bound), gate_beta)


# ---------------------------------------------------------------------------
# q/k normalization
# ---------------------------------------------------------------------------


def normalize_gdn_qk(q: mx.array, k: mx.array) -> Tuple[mx.array, mx.array]:
    """L2-normalize GDN q/k and fold in the delta-rule query scale.

    Matches the reference FLA l2norm, ``x * rsqrt(sum(x^2) + 1e-6)``, followed
    by the delta rule's ``scale = head_dim ** -0.5`` applied to the query only.

    ``mx.fast.rms_norm(x, None, eps)`` computes ``x / sqrt(mean(x^2) + eps)``,
    i.e. it adds eps to the MEAN of squares, so the equivalent epsilon is
    ``1e-6 / head_dim``. Passing ``1e-6`` straight through inflates the
    effective epsilon on the sum by head_dim (128x at head_dim=128), which
    systematically shrinks q/k whenever their norm is small.
    """
    inv_scale = k.shape[-1] ** (-0.5)
    eps = 1e-06 * inv_scale**2
    q = inv_scale**2 * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    return (q, k)


def normalize_qk(spec: LinearRecurrenceSpec, q: mx.array, k: mx.array):
    """``spec.qk_norm`` over [..., D] q/k, query scale folded into q."""
    law = spec.qk_norm
    if law == "rms_folded":
        return normalize_gdn_qk(q, k)
    dtype = q.dtype
    if spec.qk_norm_dtype == "float32":
        q, k = q.astype(mx.float32), k.astype(mx.float32)
    inv = k.shape[-1] ** (-0.5)
    eps = spec.qk_eps
    if law == "rms_mean":
        return (inv**2 * mx.fast.rms_norm(q, None, eps), inv * mx.fast.rms_norm(k, None, eps))
    if law == "l2_rsqrt":
        q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + eps)
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + eps)
    else:  # l2_clamp: torch.nn.functional.normalize, the epsilon floors the norm
        q = q / mx.maximum(mx.sqrt((q * q).sum(-1, keepdims=True)), eps)
        k = k / mx.maximum(mx.sqrt((k * k).sum(-1, keepdims=True)), eps)
    if spec.qk_norm_round:
        q, k = q.astype(dtype), k.astype(dtype)
    return (q * inv, k)


# ---------------------------------------------------------------------------
# short causal convolution
# ---------------------------------------------------------------------------


def conv_tail(conv_input, lengths, keep, steps):
    """The persistent K-1 conv history after ``steps`` new rows.

    With ``lengths`` each row is cut at its own valid end.
    """
    if keep == 0:
        return conv_input[:, :0, :]
    if lengths is not None:
        ends = mx.clip(lengths, 0, steps)
        positions = (ends[:, None] + mx.arange(keep))[..., None]
        return mx.take_along_axis(conv_input, positions, axis=1)
    return mx.contiguous(conv_input[:, -keep:, :])


def conv_window(x, state, mask, kernel_size, *, dtype=None):
    """``[history, masked x]``: the input of the causal depthwise conv."""
    if mask is not None:
        x = mx.where(mask[..., None], x, 0)
    if state is None:
        state = mx.zeros(
            (x.shape[0], kernel_size - 1, x.shape[-1]),
            dtype=x.dtype if dtype is None else dtype,
        )
    return mx.concatenate([state, x], axis=1)


def short_conv(x, state, mask, lengths, conv, *, kernel_size, precision="activation",
               weight=None):
    """SiLU(causal depthwise conv) over one separate q/k/v stream, new history.

    ``conv`` is the layer's ``nn.Conv1d`` (spec ``conv_precision``
    ``"activation"``); ``"float32"`` convolves widened inputs with ``weight``
    and rounds once (Raon).
    """
    kernel = kernel_size
    conv_input = conv_window(x, state, mask, kernel)
    if precision == "float32":
        out = mx.conv1d(conv_input.astype(mx.float32), weight.astype(mx.float32),
                        groups=x.shape[-1])
        output = nn.silu(out).astype(x.dtype)
    else:
        output = nn.silu(conv(conv_input))
    return output, conv_tail(conv_input, lengths, kernel - 1, x.shape[1])


# ---------------------------------------------------------------------------
# the sequential recurrence (pure-MLX reference)
# ---------------------------------------------------------------------------


def _readout_needs_widening(input_type, state_type) -> bool:
    """Whether the fp32 state's range can overflow the activation dtype.

    A dtype with the state's exponent width holds its maximum to within a
    factor of two (bfloat16: 3.39e38 vs float32 3.40e38); a narrower one is
    smaller by many orders of magnitude (float16: 65504). So this comparison
    is the exponent-width test.
    """
    if input_type == state_type:
        return False
    return mx.finfo(input_type).max < mx.finfo(state_type).max / 2


_SATURATING_CASTS = {}


def _saturating_cast(dtype):
    """Compiled clamp-then-cast, so the narrowing is one elementwise pass."""
    fn = _SATURATING_CASTS.get(dtype)
    if fn is None:
        info = mx.finfo(dtype)
        fn = mx.compile(
            lambda y, lo=info.min, hi=info.max: mx.clip(y, lo, hi).astype(dtype),
            shapeless=True,
        )
        _SATURATING_CASTS[dtype] = fn
    return fn


def _cast_readout(y, input_type, state_type):
    """Narrow the readout to the activation dtype, saturating on overflow.

    Every GDN caller feeds the readout straight into a scale-invariant
    RMSNorm, so one infinite component gives inf/inf = NaN and destroys the
    layer. Clamping keeps the vector finite and signed. NaN still propagates.
    """
    if not _readout_needs_widening(input_type, state_type):
        return y.astype(input_type)
    return _saturating_cast(input_type)(y)


@mx.compile
def _gated_delta_step_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for a single recurrent step.

    Shapes:
      - q, k: [B, H, Dk]
      - v: [B, H, Dv]
      - g: [B, H] or [B, H, Dk]
      - beta: [B, H]
      - state: [B, H, Dv, Dk]
    Returns:
      - y: [B, H, Dv]
      - new_state: [B, H, Dv, Dk]
    """
    old_state = state
    if g.ndim == 2:
        decay = g[..., None, None]
    elif g.ndim == 3:
        decay = g[..., None, :]
    else:
        raise ValueError(f"Unsupported gating shape {g.shape}")
    state = state * decay
    kv_mem = (state * k[..., None, :]).sum(axis=-1)
    delta = (v - kv_mem) * beta[..., None]
    state = state + k[..., None, :] * delta[..., None]
    y = (state * q[..., None, :]).sum(axis=-1)
    if mask is not None:
        mask = mx.expand_dims(mask, axis=(1, 2, 3))
        state = mx.where(mask, state, old_state)
        # A masked (padding) step contributes nothing: the Metal kernels write
        # a zero readout there, and the reference must agree so CPU oracles
        # and CPU/GPU parity hold on left-padded batches.
        y = mx.where(mask[..., 0], y, mx.zeros_like(y))
    return (_cast_readout(y, q.dtype, old_state.dtype), state)


def gated_delta_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """
    Ops-based reference implementation for prompt prefill (sequential loop).
    Supports both scalar and vectorized gating.

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar) or [B, T, Hv, Dk] (vectorized)
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    (B, T, Hk, Dk) = q.shape
    (Hv, Dv) = v.shape[-2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (repeat_factor := (Hv // Hk)) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)
    # fp16 storage class: each step loads fp16, computes in fp32 and stores
    # fp16 (the kernels' per-token rounding).
    stored = state.dtype if state.dtype == mx.float16 else None
    if stored is not None:
        _gdn_state.STATS["ops_calls"] += 1
    ys = []
    for t in range(T):
        (y, state) = _gated_delta_step_ops(
            q[:, t],
            k[:, t],
            v[:, t],
            g[:, t],
            beta[:, t],
            state if stored is None else state.astype(mx.float32),
            None if mask is None else mask[:, t],
        )
        if stored is not None:
            state = state.astype(stored)
        ys.append(y)
    y = mx.stack(ys, axis=1)
    return (y, state)


# ---------------------------------------------------------------------------
# dispatcher: gradient guard, reference fallback, registered fast paths
# ---------------------------------------------------------------------------

#: Differentiable-reference site of the recurrence (runtime/models/differentiable.py).
GUARD_SITE = "gated_delta"


def kernel_refused(site: str) -> bool:
    """The one gradient guard: True inside ``differentiable.reference_path``.

    Every Metal selection site of this technology asks it (the recurrence
    under ``gated_delta``, the fused Qwen GDN bodies under ``fused_gdn``, the
    fused KDA bodies, gates and norms under ``kda_fused``).
    """
    return _differentiable.routed(site)


def _gd():
    from ..models import gated_delta

    return gated_delta


def _admit_chunk(name):
    def admits(spec, *, chunk, mask, q, k, v, g, beta, state, log_decay=None, **_):
        if chunk != name:
            return "not_requested"
        if state is None:
            return "state_required"
        if name == "wy":
            if log_decay is None:
                return "log_decay_required"
            from ..models.kda_chunk_wy import refusal

            return refusal(q, k, v, log_decay, beta, state, mask)
        from ..models.gdn_chunk32 import refusal

        return refusal(q, k, v, g, beta, state, mask)
    return admits


def step_kernel_key_supported(key_dim):
    return key_dim >= 32 and key_dim % 32 == 0


def _admit_step_kernel(spec, *, kernels, k, **_):
    if kernels is not None:
        return kernels
    return None if step_kernel_key_supported(k.shape[-1]) else "incomplete_simd_strip"


def _admit_prefill_scan(spec, *, kernels, prefill_chunk_size, q, k, v, g, beta, state,
                        mask, **_):
    if kernels is not None:
        return kernels
    if not prefill_chunk_size:
        return "not_requested"
    return _gd()._prefill_scan_refusal(q, k, v, g, beta, state, mask, prefill_chunk_size)


def _admit_core(spec, *, kernels, allow_core, q, k, v, g, state, mask, **_):
    if kernels is not None:
        return kernels
    if not allow_core:
        return "not_allowed"
    if not _gd()._can_use_core_gated_delta(q, k, v, g, state, mask):
        return "core_refused"
    return None


def _run_chunk_wy(op):
    from ..models.kda_chunk_wy import chunk_wy

    return chunk_wy(op["q"], op["k"], op["v"], op["log_decay"], op["beta"], op["state"])


def _run_chunk32(op):
    from ..models.gdn_chunk32 import chunk32

    return chunk32(op["q"], op["k"], op["v"], op["g"], op["beta"], op["state"])


def _run_prefill_scan(op):
    return _gd()._prefill_scan_run(
        op["q"], op["k"], op["v"], op["g"], op["beta"], op["state"],
        op["prefill_chunk_size"], op["prefill_stats"], op["prefill_segment_rows"],
    )


def _run_core(op):
    gd = _gd()
    gd._count_gdn_core_call()
    chunking = {"chunk_size": gd._CORE_GDN_CHUNK_SIZE} if gd._CORE_ACCEPTS_CHUNK_SIZE else {}
    return gd._core_gated_delta_update(
        op["q"], op["k"], op["v"], op["g"], op["beta"],
        initial_state=op["state"], stream=mx.gpu, **chunking,
    )


def _run_step_kernel(op):
    return _gd().gated_delta_kernel(
        op["q"], op["k"], op["v"], op["g"], op["beta"], op["state"], op["mask"]
    )


def _run_reference(op):
    return gated_delta_ops(
        op["q"], op["k"], op["v"], op["g"], op["beta"], op["state"], op["mask"]
    )


RECURRENCE = KernelRegistry("linear_recurrence.recurrence", reference=_run_reference)
# Order is the selection order.  The two chunk candidates are explicit
# per-layer opt-ins (KDA ``chunk_wy`` / ``chunk32`` switches) the caller only
# requests for an unmasked non-speculative prompt of at least 17 rows; they
# are pure MLX, so the gradient guard does not refuse them.
RECURRENCE.register(FastPath("chunk_wy", _admit_chunk("wy"), _run_chunk_wy))
RECURRENCE.register(FastPath("chunk32", _admit_chunk("chunk32"), _run_chunk32))
RECURRENCE.register(FastPath("prefill_scan", _admit_prefill_scan, _run_prefill_scan))
RECURRENCE.register(FastPath("core_scan", _admit_core, _run_core))
RECURRENCE.register(FastPath("step_kernel", _admit_step_kernel, _run_step_kernel))

_UNCOUNTED_PREFILL = frozenset({"not_requested", "kernels_disallowed"})


def recurrence(
    q, k, v, g, beta, state=None, mask=None, *,
    use_kernel: bool = True,
    allow_core: bool = True,
    prefill_chunk_size: int = 0,
    prefill_segment_rows: int = 256,
    prefill_stats=None,
    chunk: Optional[str] = None,
    log_decay=None,
    spec: Optional[LinearRecurrenceSpec] = None,
    policy: Optional[str] = None,
    with_receipt: bool = False,
):
    """Run the gated delta recurrence on the first admitting path.

    Kernels are disallowed (reference runs) when ``use_kernel`` is false,
    inside ``differentiable.reference_path``, off the GPU, or without Metal.
    ``state`` may be ``None`` for the reference only (it creates fp32 zeros);
    pass a state to any kernel route.  Returns ``(y, state)`` or, with
    ``with_receipt``, ``(y, state, receipt)``.
    """
    if chunk not in (None, "wy", "chunk32"):
        raise ValueError("chunk must be None, 'wy' or 'chunk32'")
    if chunk == "wy" and log_decay is None:
        raise ValueError("the WY chunk scan reads the log decay")
    # Short-circuit order is load-bearing: the guard counts a reached site
    # only when the caller asked for a kernel.
    kernels = None
    if (not use_kernel or kernel_refused(GUARD_SITE) or mx.default_device() != mx.gpu
            or (not mx.metal.is_available())):
        kernels = "kernels_disallowed"
    elif state is None:
        kernels = "state_required"
    if prefill_chunk_size and kernels is None:
        if type(prefill_chunk_size) is not int or prefill_chunk_size not in (8, 16, 32):
            raise ValueError("prefill_chunk_size must be 0, 8, 16 or 32")
    shape = dict(
        q=q, k=k, v=v, g=g, beta=beta, state=state, mask=mask, kernels=kernels,
        chunk=chunk, log_decay=log_decay, allow_core=allow_core, prefill_chunk_size=prefill_chunk_size,
    )
    run, receipt = RECURRENCE.select(spec, policy=policy, **shape)
    reason = receipt.refusals.get("prefill_scan")
    if reason is not None and reason not in _UNCOUNTED_PREFILL and prefill_stats is not None:
        prefill_stats["fallback_" + reason] += 1
    out, new_state = run(dict(
        shape, prefill_stats=prefill_stats,
        prefill_segment_rows=prefill_segment_rows,
    ))
    if with_receipt:
        return out, new_state, receipt
    return out, new_state


# ---------------------------------------------------------------------------
# parent-indexed tree cohorts and verify rollback selection
# ---------------------------------------------------------------------------


def cohort_tree_recurrence(parents: Sequence[int], levels: Sequence[Sequence[int]],
                           initial, operands, step: Callable):
    """Run each tree node's one-token recurrence from its own parent state.

    ``operands`` are ``[nodes, 1, ...]`` arrays (q, k, v, decay, beta, ...);
    nodes of one depth run together (``step(*taken operands, parent_states)``
    returns ``(output, state)``), siblings never see each other.  Returns the
    per-node outputs and states (lists indexed by node).
    """
    width = len(parents)
    outputs, states = [None] * width, [None] * width
    for level in levels:
        indices = mx.array(level, dtype=mx.int32)
        parent_states = mx.concatenate([
            initial if parents[row] < 0 else states[parents[row]]
            for row in level
        ])
        output, state = step(
            *(mx.take(value, indices, axis=0) for value in operands), parent_states,
        )
        for index, row in enumerate(level):
            outputs[row], states[row] = output[index:index + 1], state[index:index + 1]
    return outputs, states


def select_rollback_rows(lengths, counts, snapshots, before, final):
    """Each lane's state after its first ``lengths[r]`` accepted verify rows.

    ``snapshots[:, j]`` is the state after row ``j + 1`` (positions before
    the lane's final valid row only), ``before`` the checkpoint and
    ``final`` the state after all ``counts[r]`` rows.  Zero and full
    acceptance select ``before``/``final`` explicitly so neither reads an
    unwritten padded snapshot.  Pure selection on the device: ``lengths``
    may be a device array (never read back) or host ints.
    """
    rows = len(counts)
    if not isinstance(lengths, mx.array):
        values = list(lengths)
        if len(values) not in (1, rows):
            raise ValueError("batched GDN rollback length count changed")
        lengths = mx.array([int(value) for value in values], mx.int32)
    steps = int(snapshots.shape[1]) + 1
    limit = mx.array(counts, mx.int32)
    ends = mx.broadcast_to(lengths.astype(mx.int32).reshape(-1), (rows,))
    ends = mx.minimum(mx.maximum(ends, 0), limit)
    zero = (ends == 0).reshape((rows,) + (1,) * (before.ndim - 1))
    if steps == 1:
        # A single verify step has no intermediate snapshots to gather.
        return mx.contiguous(mx.where(zero, before, final))
    idx = mx.minimum(mx.maximum(ends - 1, 0), steps - 2)
    index = idx.reshape((rows, 1) + (1,) * (snapshots.ndim - 2))
    selected = mx.take_along_axis(snapshots, index, axis=1).squeeze(1)
    full = (ends == limit).reshape((rows,) + (1,) * (before.ndim - 1))
    return mx.contiguous(mx.where(zero, before, mx.where(full, final, selected)))


__all__ = [
    "BETA_LAWS", "CONV_LAYOUTS", "DECAY_LAWS", "Gates", "LinearRecurrenceSpec",
    "QK_NORMS", "QWEN35_GDN", "RECURRENCE", "beta", "cohort_tree_recurrence",
    "compute_g", "compute_g_float32_inputs", "compute_lower_bound_g", "conv_tail",
    "conv_window", "decay", "gated_delta_ops", "gates", "kda_gates", "kernel_refused",
    "normalize_gdn_qk", "normalize_qk", "rda_gates", "recurrence", "select_rollback_rows",
    "replace", "short_conv",
]
