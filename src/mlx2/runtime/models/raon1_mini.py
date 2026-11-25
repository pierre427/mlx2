# SPDX-License-Identifier: MIT
"""KRAFTON Raon-1-mini (``raon1_mini``) on mlx2's hybrid linear/MLA runtime.

Architecture (KRAFTON/Raon-1-mini@6691a076, Apache-2.0; provenance/raon1-mini.json):
25 decoder layers on a 4-stream "Raon_mHC" residual.  18 layers use Raon
Delta Attention (RDA): a gated delta rule with per-channel decay
``lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))``, short causal
convolutions and a write gate that scales the value by
``sigmoid(b + w) / sigmoid(b)``.  7 layers are NoPE multi-head latent attention
(Kimi Linear's shared 64-channel key slice, no rotation) with a low-rank query
and a sigmoid output gate.  Layer 0 is a dense SiTU-GLU MLP; the others are a
128-expert top-6 sigmoid MoE (expert bias for selection only) plus one shared
expert.  Five MoE layers add a reasoning-effort steering vector.

Every equation follows the vendor's reference modeling file
(``modeling_raon1_mini.py``), written here in mlx2's idiom.  The RDA and MLA
modules subclass Kimi Linear's (same recurrence, cache, rollback and latent
attention laws); the mHC residual, SiTU activation and router are new.  No
vendor code is copied (the vendor's optimized MLX file is not used; see the
provenance entry).

Rounding points match the reference: mHC coefficients, the stream mix,
router, SiTU, RDA gates and the recurrent state are fp32; each mix and the
stream aggregate round once to the activation dtype.  One exception: the
layer, final and MLA latent RMSNorms default to ``mx.fast.rms_norm``, which
rounds the normalized row to the activation dtype before the weight product
(two roundings); the reference multiplies the weight in fp32 and rounds once.
``rms_law: "fp32"`` (opt-in, default off) is the reference's single rounding
(:class:`RaonRMSNorm`; docs/research/2026-11-25-raon1-mini-pr-sweep.md).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import mlx.core as mx
from mlx import nn

from . import bailing_hybrid as ling
from . import kimi_linear as kimi
from .base import BaseModelArgs
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .switch_layers import SwitchGLU

MODEL_TYPE = "raon1_mini"


@dataclass
class ModelArgs(ling.ModelArgs):
    """Raon config mapped onto the Ling/Kimi argument contract."""

    raon_mhc_streams: int = 4
    raon_mhc_post_factor: float = 2.0
    raon_mhc_selector_eps: float = 1e-5
    mla_latent_norm_eps: float = 1e-6
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    rda_layers: list = field(default_factory=list)
    mla_layers: list = field(default_factory=list)
    rda_decay_rank: int = 128
    rda_full_rank_gate: bool = True
    linear_attn_config: dict | None = None
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    moe_layer_freq: int = 1
    q_lora_rank: int | None = 576
    reasoning_effort_steering: dict | None = None

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        if config.get("model_type") != MODEL_TYPE:
            raise ValueError(f"not a raon1_mini config: {config.get('model_type')!r}")
        rda = dict(config.get("rda_config") or {})
        if not rda.get("full_rank_gate", True):
            raise ValueError("Raon RDA low-rank output gate is not implemented")
        if not config.get("mla_use_nope", True) or not config.get("mla_use_output_gate", True):
            raise ValueError("Raon MLA must be NoPE with the output gate")
        if config.get("moe_router_activation_func", "sigmoid") != "sigmoid":
            raise ValueError("Raon uses the sigmoid router")
        if config.get("use_grouped_topk"):
            raise ValueError("Raon grouped top-k routing is not implemented")
        heads = int(rda.get("num_heads", config["num_attention_heads"]))
        head_dim = int(rda.get("head_dim", config.get("head_dim", 128)))
        return super().from_dict({
            **config,
            "num_key_value_heads": config["num_attention_heads"],
            "num_experts_per_tok": config["num_experts_per_token"],
            "n_group": 1,
            "topk_group": 1,
            "layer_group_size": 4,
            "rope_theta": 0.0,
            "norm_topk_prob": bool(config.get("moe_renormalize", True)),
            "score_function": "sigmoid",
            "short_conv_kernel_size": int(rda.get("conv_kernel_size", 4)),
            "kda_lower_bound": rda.get("gate_lower_bound", -5.0),
            "rda_layers": list(rda.get("rda_layers", [])),
            "mla_layers": list(rda.get("mla_layers", [])),
            "rda_full_rank_gate": bool(rda.get("full_rank_gate", True)),
            "linear_attn_config": {"num_heads": heads, "head_dim": head_dim},
            "moe_shared_expert_intermediate_size": config["moe_intermediate_size"],
        })

    def is_rda_layer(self, index: int) -> bool:
        # The config lists 1-based layer numbers (reference ``is_rda_layer``).
        return index + 1 in self.rda_layers

    def is_moe_layer(self, index: int) -> bool:
        return (bool(self.num_experts) and index >= self.first_k_dense_replace
                and index % self.moe_layer_freq == 0)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def helmert_basis(n: int) -> mx.array:
    """Orthonormal ``Q`` [n, n-1] with ``Q^T Q = I`` and ``Q^T 1 = 0`` (fp32)."""
    rows = []
    for i in range(n):
        row = []
        for k in range(1, n):
            norm = math.sqrt(k * (k + 1.0))
            row.append((1.0 if i < k else (-float(k) if i == k else 0.0)) / norm)
        rows.append(row)
    return mx.array(rows, dtype=mx.float32)


@partial(mx.compile, shapeless=True)
def situ_glu(up, gate, gate_beta, value_beta):
    """SiTU-GLU in fp32, one rounding: ``gb*tanh(g/gb)*sigmoid(g) * vb*tanh(v/vb)``."""
    g = gate.astype(mx.float32)
    v = up.astype(mx.float32)
    out = (gate_beta * mx.tanh(g / gate_beta) * mx.sigmoid(g)) * (value_beta * mx.tanh(v / value_beta))
    return out.astype(up.dtype)


class SiTU(nn.Module):
    """SwitchGLU activation ``act(x_up, x_gate)`` with the SiTU law."""

    def __init__(self, gate_beta: float, value_beta: float):
        super().__init__()
        self._betas = (mx.array(gate_beta, mx.float32), mx.array(value_beta, mx.float32))

    def __call__(self, x, gate):
        return situ_glu(x, gate, *self._betas)


RMS_LAWS = ("mlx", "fp32")


class RaonRMSNorm(nn.RMSNorm):
    """RMSNorm with a selectable rounding law (``law`` set by ``configure``).

    ``"mlx"`` (default): ``mx.fast.rms_norm(x, w)``, which normalizes in fp32,
    rounds to the activation dtype, then multiplies the weight in that dtype.
    ``"fp32"``: the reference ``Raon1MiniRMSNorm``: the weight product in
    fp32, one rounding.  Same parameters and checkpoint names either way.
    """

    law = "mlx"

    def __call__(self, x):
        if self.law == "fp32":
            return mx.fast.rms_norm(x.astype(mx.float32), self.weight.astype(mx.float32),
                                    self.eps).astype(x.dtype)
        return super().__call__(x)


MERGED_ROWS = 8


class MergedProjection:
    """Several linears that read one input, run as one matmul for <= 8 rows.

    Opt-in decode lever (``merged_projections``).  Parts must share one
    storage kind: plain bf16 ``nn.Linear`` (one concatenated matmul) or
    ``GlobalScaledNVFP4Linear`` (one concatenated NVFP4 qmm, then each
    column's own global scale with the parts' rounding:
    bf16(fp32(bf16(qmm)) * s)).  Each output column is the same dot product
    as in its own part; only the kernel's N changes.  The merged planes are
    a resident copy (the parts keep theirs for prompt rows).
    """

    def __init__(self, parts):
        from ..nvfp4_linear import GlobalScaledNVFP4Linear

        kinds = {type(p) for p in parts}
        if kinds == {nn.Linear} and not any("bias" in p for p in parts):
            self.kind = "dense"
            self.weight = mx.concatenate([p.weight for p in parts])
        elif (kinds == {nn.QuantizedLinear} and not any("bias" in p for p in parts)
              and len({(p.mode, p.bits, p.group_size) for p in parts}) == 1
              and parts[0].mode == "affine"):
            self.kind = "affine"
            self.bits, self.group_size = parts[0].bits, parts[0].group_size
            self.weight = mx.concatenate([p["weight"] for p in parts])
            self.scales = mx.concatenate([p["scales"] for p in parts])
            self.biases = mx.concatenate([p["biases"] for p in parts])
        elif kinds == {GlobalScaledNVFP4Linear} and not any("bias" in p for p in parts):
            self.kind = "nvfp4"
            self.weight = mx.concatenate([p["weight"] for p in parts])
            self.scales = mx.concatenate([p["scales"] for p in parts])
            self.column_scale = mx.concatenate([
                mx.broadcast_to(p["global_scale"].astype(mx.float32), (p["weight"].shape[0],)) for p in parts])
        else:
            raise ValueError(f"merged projection parts differ in storage: {sorted(k.__name__ for k in kinds)}")
        sizes = [p.weight.shape[0] for p in parts]
        self.splits = [sum(sizes[:i + 1]) for i in range(len(sizes) - 1)]
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)])

    def __call__(self, x):
        if self.kind == "dense":
            y = x @ self.weight.T
        elif self.kind == "affine":
            y = mx.quantized_matmul(x, self.weight, scales=self.scales, biases=self.biases, transpose=True,
                                    group_size=self.group_size, bits=self.bits, mode="affine")
        else:
            y = mx.quantized_matmul(x, self.weight, scales=self.scales, biases=None, transpose=True,
                                    group_size=16, bits=4, mode="nvfp4")
            y = (y.astype(mx.float32) * self.column_scale).astype(y.dtype)
        return mx.split(y, self.splits, axis=-1)


def _rows(x) -> int:
    return x.size // x.shape[-1]


class MLP(nn.Module):
    """Dense / shared-expert SiTU-GLU MLP."""

    def __init__(self, args: ModelArgs, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, hidden, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, args.hidden_size, bias=False)
        object.__setattr__(self, "_act", SiTU(args.activation_situ_beta, args.activation_situ_linear_beta))

    def merge(self, enabled: bool):
        object.__setattr__(self, "_merged", MergedProjection([self.up_proj, self.gate_proj]) if enabled else None)

    def __call__(self, x):
        merged = self.__dict__.get("_merged")
        if merged is not None and _rows(x) <= MERGED_ROWS:
            up, gate = merged(x)
            return self.down_proj(self._act(up, gate))
        return self.down_proj(self._act(self.up_proj(x), self.gate_proj(x)))


# ---------------------------------------------------------------------------
# Raon_mHC 4-stream residual
# ---------------------------------------------------------------------------

class StreamMix(nn.Module):
    """``res.*`` parameters of one mHC site (stream-to-stream chart)."""

    def __init__(self, n: int, hidden: int):
        super().__init__()
        m = (n - 1) ** 2
        self.phi_z = mx.zeros((m, n * hidden))
        self.alpha_z = mx.ones(())
        self.b_z = mx.zeros((m,))
        self.eta = mx.zeros((), dtype=mx.float32)


@mx.compile
def _mhc_coefficients(z, inv, alpha_pre, b_pre, alpha_post, b_post, alpha_z, b_z, eta,
                      helmert, eye, chart, post_factor):
    """fp32 selector output -> (h_pre [N,n], h_post [N,n], h_res [N,n,n])."""
    n = helmert.shape[0]
    m = n - 1
    z = z.astype(mx.float32) * inv
    t_pre = alpha_pre.astype(mx.float32) * z[..., :n] + b_pre.astype(mx.float32)
    t_post = alpha_post.astype(mx.float32) * z[..., n:2 * n] + b_post.astype(mx.float32)
    t_z = alpha_z.astype(mx.float32) * z[..., 2 * n:] + b_z.astype(mx.float32)
    h_pre = mx.sigmoid(t_pre)
    h_post = post_factor * mx.sigmoid(t_post)
    zz = t_z.reshape(*t_z.shape[:-1], m, m)
    e = helmert @ zz @ helmert.T
    rho = mx.maximum((-e / chart).max(axis=(-2, -1)), 0.0)
    phi = chart + e / (1.0 + rho)[..., None, None]
    gamma = mx.sigmoid(eta.astype(mx.float32))
    h_res = (1.0 - gamma) * eye + gamma * phi
    return h_pre, h_post, h_res


@partial(mx.compile, shapeless=True)
def _mhc_aggregate(stream32, h_pre):
    """[..., n, C] fp32 streams x [..., n] -> [..., C] fp32."""
    return (h_pre[..., None, :] @ stream32)[..., 0, :]


@partial(mx.compile, shapeless=True)
def _mhc_mix(stream32, branch, h_post, h_res):
    """Residual chart plus branch write-back, fp32 (rounded once by the caller)."""
    return h_res @ stream32 + h_post[..., None] * branch.astype(mx.float32)[..., None, :]


class RaonMHC(nn.Module):
    """One mHC site: selects the branch input from the streams and writes back."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        n, c = args.raon_mhc_streams, args.hidden_size
        self.phi_pre = mx.zeros((n, n * c))
        self.phi_post = mx.zeros((n, n * c))
        self.alpha_pre = mx.ones(())
        self.alpha_post = mx.ones(())
        self.b_pre = mx.zeros((n,))
        self.b_post = mx.zeros((n,))
        self.res = StreamMix(n, c)
        eye = mx.eye(n, dtype=mx.float32)
        object.__setattr__(self, "_consts", (
            helmert_basis(n), eye, 0.5 * (eye + mx.full((n, n), 1.0 / n, dtype=mx.float32)),
            mx.array(args.raon_mhc_post_factor, mx.float32)))
        object.__setattr__(self, "_eps", float(args.raon_mhc_selector_eps))
        object.__setattr__(self, "_joined", None)

    def joined(self, dtype):
        cached = self.__dict__.get("_joined")
        key = (id(self.phi_pre), id(self.phi_post), id(self.res.phi_z), str(dtype))
        if cached is None or cached[0] != key:
            weight = mx.concatenate([self.phi_pre, self.phi_post, self.res.phi_z], 0).astype(dtype)
            cached = (key, weight)
            object.__setattr__(self, "_joined", cached)
        return cached[1]

    def route(self, stream):
        """``stream`` [B, T, n, C] -> fp32 (h_pre, h_post, h_res) per row."""
        b, t, n, c = stream.shape
        flat = stream.reshape(b, t, n * c)
        # RMS scale applied after the model-dtype selector GEMM (reference).
        inv = mx.rsqrt(mx.mean(mx.square(flat.astype(mx.float32)), -1, keepdims=True) + self._eps)
        z = flat @ self.joined(flat.dtype).T
        return _mhc_coefficients(z, inv, self.alpha_pre, self.b_pre, self.alpha_post, self.b_post,
                                 self.res.alpha_z, self.res.b_z, self.res.eta, *self._consts)


# ---------------------------------------------------------------------------
# RDA (Raon Delta Attention)
# ---------------------------------------------------------------------------

class ShortConv(ling.ShortConv1d):
    """Depthwise causal conv with fp32 accumulation and SiLU, one rounding."""

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        if state is None:
            state = mx.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        out = mx.conv1d(conv_input.astype(mx.float32), self.conv.weight.astype(mx.float32),
                        groups=x.shape[-1])
        output = nn.silu(out).astype(x.dtype)
        keep = self.kernel_size - 1
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            new_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_state = mx.contiguous(conv_input[:, -keep:, :])
        return output, new_state


@partial(mx.compile, shapeless=True)
def _rda_gates(decay_raw, dt_bias, a_exp, beta_raw, write_raw, lower_bound):
    """fp32 RDA gates: (decay = exp(log_alpha), beta, value ratio)."""
    scaled = a_exp * (decay_raw.astype(mx.float32) + dt_bias.astype(mx.float32))
    log_alpha = lower_bound * mx.sigmoid(scaled)
    b32 = beta_raw.astype(mx.float32)
    beta = mx.sigmoid(b32)
    # sigmoid(b + w) / sigmoid(b) in log space, as the reference.
    ratio = mx.exp(nn.log_sigmoid(b32 + write_raw.astype(mx.float32)) - nn.log_sigmoid(b32))
    return mx.exp(log_alpha), beta, ratio


def _l2_normalize(x):
    # torch.nn.functional.normalize: x / max(||x||, 1e-6), in fp32.
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.sqrt(mx.sum(x * x, -1, keepdims=True)), 1e-6)


class RDA(kimi.KimiDeltaAttention):
    """Raon Delta Attention on Kimi's KDA cache/rollback/kernels."""

    def __init__(self, args: ModelArgs):
        super().__init__(replace(args, no_kda_lora=False))
        h, d = self.num_heads, self.head_dim
        hidden = args.hidden_size
        for name in ("g_a_proj", "g_b_proj"):
            self.pop(name)
        self.f_a_proj = nn.Linear(hidden, args.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(args.rda_decay_rank, h * d, bias=False)
        self.g_proj = nn.Linear(hidden, h * d, bias=False)
        self.rda_w_proj = nn.Linear(hidden, h, bias=False)
        self.q_conv1d = ShortConv(h * d, self.conv_kernel)
        self.k_conv1d = ShortConv(h * d, self.conv_kernel)
        self.v_conv1d = ShortConv(h * d, self.conv_kernel)
        self.A_log = mx.zeros((h,), dtype=mx.float32)
        self.dt_bias = mx.zeros((h * d,), dtype=mx.float32)
        self.o_norm = nn.RMSNorm(d, eps=args.rms_norm_eps)
        object.__setattr__(self, "_lower_bound", mx.array(float(args.kda_lower_bound), mx.float32))

    _PROJECTIONS = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "f_b_proj",
                    "g_proj", "rda_w_proj")

    def prepare_projections(self):
        raise NotImplementedError("Raon RDA has no joined-projection law yet")

    def merge(self, enabled: bool):
        parts = [self.q_proj, self.k_proj, self.v_proj, self.b_proj, self.f_a_proj, self.g_proj,
                 self.rda_w_proj]
        object.__setattr__(self, "_merged", MergedProjection(parts) if enabled else None)

    def project(self, x):
        """Row-wise pre-convolution projections: q, k, v, beta, decay, gate, write."""
        merged = self.__dict__.get("_merged")
        if merged is not None and _rows(x) <= MERGED_ROWS:
            q, k, v, b, fa, g, w = merged(x)
            self.calls["merged_projections"] += 1
            return q, k, v, b, self.f_b_proj(fa), g, w
        return (self.q_proj(x), self.k_proj(x), self.v_proj(x), self.b_proj(x),
                self.f_b_proj(self.f_a_proj(x)), self.g_proj(x), self.rda_w_proj(x))

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if tree is not None:
            raise ValueError("Raon RDA has no tree contract")
        b, t, _ = x.shape
        dtype = x.dtype
        kind = "multirow" if t > 1 else "single_row"
        self.calls[kind + "_calls"] += 1
        self.calls[kind + "_tokens"] += b * t
        h, d = self.num_heads, self.head_dim
        p = h * d
        if mask is not None:
            # Left padding: padded rows enter neither the conv window nor the
            # recurrence (reference token_mask).
            x = mx.where(mask[..., None], x, 0)
        q, k, v, beta_raw, decay_raw, gate, write = self.project(x)
        states = [None] * 4 if cache is None else list(cache)
        lengths = None if cache is None else cache.lengths
        spans = None
        if cache is not None and getattr(cache, "speculating", False):
            describe = getattr(cache, "rollback_spans", None)
            spans = describe(t, mask) if describe is not None else None
            if spans is None:
                self.calls["rollback_unstaged"] += 1
        snapshot = list(states) if spans is not None else None
        raw = (q, k, v) if spans is not None else None
        q, states[0] = self.q_conv1d(q, states[0], mask, lengths)
        k, states[1] = self.k_conv1d(k, states[1], mask, lengths)
        v, states[2] = self.v_conv1d(v, states[2], mask, lengths)
        q, k, v = (z.reshape(b, t, h, d) for z in (q, k, v))
        # Reference: l2 normalize in fp32, round, then scale q by d**-0.5.
        q = _l2_normalize(q).astype(dtype) * (d ** -0.5)
        k = _l2_normalize(k).astype(dtype)
        decay, beta, ratio = _rda_gates(
            decay_raw.reshape(b, t, h, d), self.dt_bias.reshape(h, d),
            mx.exp(self.A_log.astype(mx.float32)).reshape(h, 1), beta_raw, write,
            self._lower_bound)
        beta = beta.astype(dtype).astype(mx.float32)
        v = (v.astype(mx.float32) * ratio[..., None]).astype(dtype)
        q, k, v = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        state = states[3]
        if state is None:
            state = mx.zeros((b, h, d, d), dtype=mx.float32)
        if spans is not None:
            self._record_rollback(cache, t, spans, snapshot, raw, mask, (q, k, v, decay, beta), state, b)
        chunked = t >= 17 and mask is None and not getattr(cache, "speculating", False)
        if self.chunk_wy and chunked:
            from .kda_chunk_wy import chunk_wy
            out, state = chunk_wy(q, k, v, mx.log(decay), beta, state)
            self.calls["chunk_wy"] += 1
        elif self.chunk32 and chunked:
            from .gdn_chunk32 import chunk32
            out, state = chunk32(q, k, v, decay, beta, state)
            self.calls["chunk32"] += 1
        elif not self.reference and mx.default_device() == mx.gpu:
            out, state = gated_delta_kernel(q, k, v, decay, beta, state, mask)
            self.calls["vector_kernel"] += 1
        else:
            out, state = gated_delta_ops(q, k, v, decay, beta, state, mask)
            self.calls["reference"] += 1
        if cache is not None:
            for i in range(3):
                cache[i] = states[i]
            cache[3] = state
            cache.advance(t)
        return self.o_proj(self.norm_gate(out, gate, dtype).reshape(b, t, p))

    def _record_rollback(self, cache, t, spans, snapshot, raw, mask, scan, state, b):
        # Kimi's closed-form conv rollback replays the raw (pre-conv) rows;
        # Raon zeroes padded rows before projection, so the raw rows are
        # already masked and the law is unchanged.
        return super()._record_rollback(cache, t, spans, snapshot, raw, mask, scan, state, b)


# ---------------------------------------------------------------------------
# NoPE gated MLA
# ---------------------------------------------------------------------------

_EXPANDED_FLASH_OK: dict = {}


def expanded_flash_available(heads: int, dtype) -> bool:
    """Kimi's flash192 (NAX) prompt-chunk kernel is head-generic (template
    ``H``, head grid axis); only its admission and probe are pinned to 32
    heads.  This memoized probe checks it at ``heads`` against its FP32 ops
    law (same 2e-2 relative bound as Kimi's) on an M5 GPU."""
    key = (int(heads), str(dtype))
    if key not in _EXPANDED_FLASH_OK:
        from . import kimi_linear_mla_flash as fl

        ok = False
        if mx.default_device() == mx.gpu and fl._device_is_m5():
            try:
                k1, k2, k3 = mx.random.split(mx.random.key(20261125), 3)
                rows, width, n, r, v = 40, 100, fl.NOPE, fl.SHARED, fl.VDIM
                q = (mx.random.normal((1, rows, heads * (n + r)), key=k1) * 0.5).astype(dtype)
                kv = mx.random.normal((1, width, heads * (n + v)), key=k2).astype(dtype)
                sh = mx.random.normal((1, 1, width, r), key=k3).astype(dtype)
                scale = (n + r) ** -0.5
                got = fl.expanded(q, kv, sh, "causal", scale=scale, heads=heads)
                ref = fl.expanded_reference(q, kv, sh, "causal", scale=scale, heads=heads)
                rel, finite = fl._rel(ref, got)
                ok = finite and rel < 2e-2
            except Exception:  # noqa: BLE001 - compile failure / refusal declines cleanly
                ok = False
        _EXPANDED_FLASH_OK[key] = ok
    return _EXPANDED_FLASH_OK[key]


class MLA(kimi.MultiLatentAttention):
    """Kimi's NoPE latent attention with a low-rank query and a sigmoid output gate."""

    def _flash_admits(self, kind, q_rows, mask, starts=None):
        """Raon admits only the expanded (flash192 / key-window) prompt-chunk
        kernel, at its own head count; the absorbed decode kernel's grid
        covers exactly 32 heads and stays declined (counted)."""
        if self.flash == "off":
            return False
        a = self.args
        ok = (kind == "expanded" and self.flash == "expanded"
              and (mask is None or (isinstance(mask, str) and mask == "causal"))
              and q_rows.dtype in (mx.bfloat16, mx.float16)
              and (a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim, a.kv_lora_rank) == (128, 64, 128, 512)
              and expanded_flash_available(a.num_attention_heads, q_rows.dtype))
        if not ok:
            self.calls["flash_declined"] += 1
        return ok

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        h, n, r, v = (args.num_attention_heads, args.qk_nope_head_dim, args.qk_rope_head_dim,
                      args.v_head_dim)
        self.pop("q_proj")
        self.q_a_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_a_layernorm = RaonRMSNorm(args.q_lora_rank, eps=args.mla_latent_norm_eps)
        self.q_b_proj = nn.Linear(args.q_lora_rank, h * (n + r), bias=False)
        self.kv_a_layernorm = RaonRMSNorm(args.kv_lora_rank, eps=args.mla_latent_norm_eps)
        self.g_proj = nn.Linear(args.hidden_size, h * v, bias=False)
        # Kimi's forwards call ``self.q_proj`` / ``self.o_proj``; these
        # instance attributes shadow them (the o_proj Linear stays the module
        # child, so loading and quantization see the checkpoint names).
        object.__setattr__(self, "q_proj", self._query)
        object.__setattr__(self, "o_proj", self._gated_output)
        object.__setattr__(self, "_gate", None)

    def _query(self, x):
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    def _gated_output(self, context):
        gate = self.__dict__["_gate"]
        if gate is None or gate.shape[:2] != context.shape[:2]:
            raise RuntimeError("Raon MLA output gate was not staged for this forward")
        # The reference evaluates the gate in the activation dtype.
        return self["o_proj"](context * gate)

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if tree is not None:
            raise ValueError("Raon MLA has no tree contract")
        object.__setattr__(self, "_gate", mx.sigmoid(self.g_proj(x)))
        try:
            return super().__call__(x, mask, cache)
        finally:
            object.__setattr__(self, "_gate", None)

    def tail(self, x, mask=None, cache=None):
        object.__setattr__(self, "_gate", mx.sigmoid(self.g_proj(x[:, -1:])))
        try:
            return super().tail(x, mask, cache)
        finally:
            object.__setattr__(self, "_gate", None)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """fp32 sigmoid router; ``e_score_correction_bias`` for selection only."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,), dtype=mx.float32)
        object.__setattr__(self, "_top_k", int(args.num_experts_per_tok))
        object.__setattr__(self, "_renorm", bool(args.norm_topk_prob))
        object.__setattr__(self, "_scale", float(args.routed_scaling_factor))
        object.__setattr__(self, "_w32", None)

    def weight32(self):
        cached = self.__dict__.get("_w32")
        if cached is None or cached[0] is not self.weight:
            cached = (self.weight, self.weight.astype(mx.float32))
            object.__setattr__(self, "_w32", cached)
        return cached[1]

    def __call__(self, x):
        scores = mx.sigmoid(x.astype(mx.float32) @ self.weight32().T)
        choice = scores + self.e_score_correction_bias.astype(mx.float32)
        k = self._top_k
        ids = mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, ids, -1)
        if self._renorm:
            weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
        return ids, weights * self._scale if self._scale != 1.0 else weights


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, steering: bool):
        super().__init__()
        self.gate = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts,
            activation=SiTU(args.activation_situ_beta, args.activation_situ_linear_beta))
        self.shared_experts = MLP(args, args.moe_intermediate_size * args.num_shared_experts)
        if steering:
            self.reasoning_effort_steering = mx.zeros((args.hidden_size,), dtype=mx.float32)
        object.__setattr__(self, "calls", Counter())

    def __call__(self, x):
        self.calls["router_tokens"] += x.size // x.shape[-1]
        ids, weights = self.gate(x)
        expert = self.switch_mlp(x, ids)
        out = (expert.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
        return out + self.shared_experts(x)


# ---------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, index: int):
        super().__init__()
        self.index = index
        self.is_global = not args.is_rda_layer(index)
        self.is_moe = args.is_moe_layer(index)
        self.input_layernorm = RaonRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RaonRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = MLA(args) if self.is_global else RDA(args)
        self.self_attn_raon_mhc = RaonMHC(args)
        self.mlp_raon_mhc = RaonMHC(args)
        steering_layers = (args.reasoning_effort_steering or {}).get("layers", ())
        if self.is_moe:
            self.moe = MoE(args, steering=index in steering_layers)
        else:
            self.mlp = MLP(args, args.intermediate_size)
        object.__setattr__(self, "effort_alpha", 0.0)

    @property
    def attention(self):
        """Ling/Kimi seam name for the token mixer."""
        return self.self_attn

    @property
    def steered(self) -> bool:
        return self.is_moe and "reasoning_effort_steering" in self.moe

    def __call__(self, stream, mask=None, cache=None):
        if self.__dict__.get("_mhc_fused"):
            from . import raon1_mini_fused as F

            if F.admits(stream):
                return self._fused(stream, mask, cache, F)
        b, t, n, c = stream.shape
        dtype = stream.dtype
        h_pre, h_post, h_res = self.self_attn_raon_mhc.route(stream)
        s32 = stream.astype(mx.float32)
        branch_in = self.input_layernorm(_mhc_aggregate(s32, h_pre).astype(dtype))
        out = self.self_attn(branch_in, mask=mask, cache=cache)
        stream = _mhc_mix(s32, out, h_post, h_res).astype(dtype)

        h_pre, h_post, h_res = self.mlp_raon_mhc.route(stream)
        s32 = stream.astype(mx.float32)
        branch_in = self.post_attention_layernorm(_mhc_aggregate(s32, h_pre).astype(dtype))
        if self.is_moe:
            out = self.moe(branch_in)
            alpha = self.effort_alpha
            if alpha and self.steered:
                out = (out.astype(mx.float32) + alpha * self.moe.reasoning_effort_steering).astype(dtype)
        else:
            out = self.mlp(branch_in)
        return _mhc_mix(s32, out, h_post, h_res).astype(dtype)

    def _branch_mlp(self, branch_in, dtype):
        if not self.is_moe:
            return self.mlp(branch_in)
        out = self.moe(branch_in)
        alpha = self.effort_alpha
        if alpha and self.steered:
            out = (out.astype(mx.float32) + alpha * self.moe.reasoning_effort_steering).astype(dtype)
        return out

    def _fused(self, stream, mask, cache, F):
        """Two-launch mHC sites (raon1_mini_fused; opt-in ``mhc: fused``)."""
        dtype = stream.dtype
        _, h_post, h_res, agg = F.route_aggregate(self.self_attn_raon_mhc, stream)
        out = self.self_attn(self.input_layernorm(agg), mask=mask, cache=cache)
        stream = F.mix(stream, out, h_post, h_res)
        _, h_post, h_res, agg = F.route_aggregate(self.mlp_raon_mhc, stream)
        out = self._branch_mlp(self.post_attention_layernorm(agg), dtype)
        return F.mix(stream, out, h_post, h_res)


EFFORT_NONE = "none"
NO_THINKING_ALPHA = 0.1


class Model(nn.Module):
    model_type = MODEL_TYPE
    # RDA layers record exact conv/recurrent rollback (Kimi's law); MLA
    # layers are plain KV caches that trim.
    supports_speculative_rollback = True
    apc_v2_layout = "raon1-mini-mhc4-rda-fp32-nope-latent-mla-v1"

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = _Trunk(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        object.__setattr__(self, "reasoning_effort", None)
        self.set_reasoning_effort(None)

    # -- effort ----------------------------------------------------------
    def effort_alpha(self, effort: str | None) -> float:
        """Steering alpha for an effort (None: the config's default effort).

        ``"none"`` is thinking off: the config's ``none`` alpha when above 0,
        else 0.1 (the reference, vLLM, llama.cpp and the vendor MLX agree).
        """
        steering = self.args.reasoning_effort_steering or {}
        if not steering:
            return 0.0
        alphas = {k: float(v) for k, v in (steering.get("alpha") or {}).items()}
        effort = steering.get("default") if effort is None else effort
        if effort == EFFORT_NONE:
            return alphas.get(EFFORT_NONE, 0.0) or NO_THINKING_ALPHA
        if effort not in alphas:
            raise ValueError(f"unknown Raon reasoning effort {effort!r}")
        return alphas[effort]

    def set_reasoning_effort(self, effort: str | None) -> float:
        alpha = self.effort_alpha(effort)
        for layer in self.model.layers:
            object.__setattr__(layer, "effort_alpha", alpha)
        object.__setattr__(self, "reasoning_effort", effort)
        return alpha

    # -- seams -----------------------------------------------------------
    @property
    def layers(self):
        return self.model.layers

    @property
    def speculative_args(self):
        return self.args

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def tap_streams(self) -> int:
        """Each tapped layer exposes its full n-stream residual (DFlash2 taps)."""
        return int(self.args.raon_mhc_streams)

    def logits(self, hidden):
        return self.lm_head(hidden)

    def __call__(self, inputs, cache=None):
        from ..prefill_checkpoint_rows import select_checkpoint_rows

        return self.lm_head(select_checkpoint_rows(self.model(inputs, cache)))

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False,
                          last_logits_only=False):
        """Logits and the concatenated post-layer n-stream residual taps.

        A tap is the [B, T, n*C] stream after layer ``i`` (HF
        ``hidden_states[i + 1]`` before the stream mean), streams in order.
        """
        capture_layers = tuple(int(v) for v in capture_layers)
        if (not capture_layers or tuple(sorted(set(capture_layers))) != capture_layers
                or capture_layers[0] < 0 or capture_layers[-1] >= len(self.layers)):
            raise ValueError("invalid Raon target capture layers")
        if body_only and last_logits_only:
            raise ValueError("body_only and last_logits_only are mutually exclusive")
        hidden, taps = self.model.run(inputs, cache, capture=frozenset(capture_layers))
        features = mx.concatenate(taps, axis=-1)
        if body_only:
            return None, features
        return self.lm_head(hidden[:, -1:] if last_logits_only else hidden), features

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def make_cache(self) -> list[Any]:
        return [KVCache() if layer.is_global else ArraysCache(size=4) for layer in self.layers]

    # -- weights ---------------------------------------------------------
    def sanitize(self, weights: dict) -> dict:
        """Vendor names -> module names; per-expert MXFP8 tables are stacked
        by the conversion, never here (the served artifacts are pre-stacked)."""
        out = {}
        convs = {}
        for key, value in weights.items():
            if key.startswith("model.layers.") and ".moe.experts." in key:
                raise ValueError("Raon per-expert tensors need the mlx2 conversion "
                                 "(mlx2.conversion.raon1_mini); load a converted artifact")
            if key.endswith(".self_attn.qkv_conv1d.weight"):
                convs[key[:-len("qkv_conv1d.weight")]] = value
                continue
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                if key.endswith(f".self_attn.{name}.weight"):
                    key = key[:-len("weight")] + "conv.weight"
                    value = _conv_layout(value)
            if key.endswith(".self_attn.A_log") or key.endswith(".self_attn.dt_bias"):
                value = value.reshape(-1).astype(mx.float32)
            if key.endswith(".gate.e_score_correction_bias") or key.endswith(".res.eta"):
                value = value.astype(mx.float32)
            if key.endswith(".reasoning_effort_steering"):
                value = value.astype(mx.float32)
            out[key] = value
        for prefix, value in convs.items():
            # One [3P, K] (or MLX [3P, K, 1]) depthwise bank -> q, k, v.
            value = _conv_layout(value)
            for name, part in zip(("q_conv1d", "k_conv1d", "v_conv1d"), mx.split(value, 3, axis=0)):
                out[f"{prefix}{name}.conv.weight"] = part
        return out

    def quant_predicate(self):
        def predicate(path: str, _module):
            # Router, mHC selectors, norms and convolutions stay unquantized.
            if path.endswith(".moe.gate") or "raon_mhc" in path:
                return False
            return True
        return predicate

    # -- status ----------------------------------------------------------
    def route_status(self):
        counters = Counter()
        for layer in self.layers:
            counters.update(layer.self_attn.calls)
        return dict(counters)

    def mechanism_status(self):
        rda, mla = Counter(), Counter()
        for layer in self.layers:
            (mla if layer.is_global else rda).update(layer.self_attn.calls)
        return {"rda": dict(rda), "mla": dict(mla), "effort": self.reasoning_effort,
                "effort_alpha": self.layers[1].effort_alpha if len(self.layers) > 1 else 0.0}

    def configure(self, policy: dict):
        law = policy.get("rms_law", "mlx")
        if law not in RMS_LAWS:
            raise ValueError(f"rms_law must be one of {RMS_LAWS}")
        for _, module in self.named_modules():
            if isinstance(module, RaonRMSNorm):
                module.law = law
        for layer in self.layers:
            object.__setattr__(layer, "_mhc_fused", policy.get("mhc", "ops") == "fused")
            merge = bool(policy.get("merged_projections", False))
            if not layer.is_global:
                layer.self_attn.merge(merge)
            (layer.moe.shared_experts if layer.is_moe else layer.mlp).merge(merge)
            attn = layer.self_attn
            if layer.is_global:
                attn.policy = policy.get("mla", "auto")
                attn.flash = policy.get("mla_flash", "off")
                attn.expanded_bytes = int(policy.get("mla_expanded_gib", 4)) << 30
            else:
                kda = policy.get("rda", "vector")
                attn.reference = kda == "reference"
                attn.chunk32 = kda == "chunk32"
                attn.chunk_wy = kda == "chunk_wy"
                attn.fused_norm = bool(policy.get("fused_norm", False))


def _conv_layout(value):
    """HF depthwise [C, K] or [C, 1, K] -> MLX conv1d weight [C, K, 1]."""
    if value.ndim == 2:
        return value[..., None]
    if value.ndim == 3 and value.shape[1] == 1:
        return value.moveaxis(2, 1)
    return value


class _Trunk(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RaonRMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(i for i, layer in enumerate(self.layers) if layer.is_global)
        self._rda_idx = next(i for i, layer in enumerate(self.layers) if not layer.is_global)

    @property
    def word_embeddings(self):
        return self.embed_tokens

    def run(self, inputs, cache=None, *, capture=frozenset()):
        from . import moe_nax_gather

        # Prompt chunks admit the NAX sorted gather when the adapter claimed
        # it (moe_nax "gather", affine artifact); a no-op under stock claims.
        with moe_nax_gather.forward_scope(inputs, cache):
            return self._run(inputs, cache, capture=capture)

    def _run(self, inputs, cache=None, *, capture=frozenset()):
        hidden = self.embed_tokens(inputs)
        n = self.args.raon_mhc_streams
        b, t, c = hidden.shape
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = kimi.LanguageModel.attention_mask(hidden, cache[self._attention_idx])
        rda_mask = ling.create_ssm_mask(hidden, cache[self._rda_idx])
        stream = mx.broadcast_to(hidden[:, :, None, :], (b, t, n, c))
        taps = []
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            stream = layer(stream, mask=attention_mask if layer.is_global else rda_mask,
                           cache=layer_cache)
            if index in capture:
                taps.append(stream.reshape(b, t, n * c))
        reduced = stream.astype(mx.float32).mean(-2).astype(hidden.dtype)
        return self.norm(reduced), taps

    def __call__(self, inputs, cache=None):
        return self.run(inputs, cache)[0]


# ---------------------------------------------------------------------------
# quantized artifact installers
# ---------------------------------------------------------------------------

class NVFP4GlobalSwitchLinear(nn.Module):
    """Routed NVFP4 experts with MLX's in-kernel per-expert ``global_scale``.

    The vendor pack stores the routed experts' scale in MLX's own nvfp4
    encoding (the kernel multiplies its fp32 accumulator by
    ``global_scale / (448 * 6)`` before the single bf16 rounding), unlike the
    dense linears' direct output-side multiplier.  Same call signature as
    ``QuantizedSwitchLinear``.
    """

    group_size, bits, mode = 16, 4, "nvfp4"

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        if input_dims % 64:
            raise ValueError("NVFP4 switch input width must be a multiple of 64")
        self.weight = mx.zeros((num_experts, output_dims, input_dims // 8), dtype=mx.uint32)
        self.scales = mx.zeros((num_experts, output_dims, input_dims // 16), dtype=mx.uint8)
        self.global_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.freeze()

    @property
    def num_experts(self):
        return self.weight.shape[0]

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def input_dims(self):
        return self.scales.shape[2] * 16

    def __call__(self, x, indices, sorted_indices=False):
        return mx.gather_qmm(
            x, self["weight"], self["scales"], None, rhs_indices=indices, transpose=True,
            group_size=16, bits=4, mode="nvfp4", global_scale=self["global_scale"],
            sorted_indices=sorted_indices)


def _module(root, path):
    node = root
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else node[part]
    return node


def _replace(root, path, module):
    parent, _, name = path.rpartition(".")
    owner = _module(root, parent) if parent else root
    if name.isdigit():
        owner[int(name)] = module
    else:
        owner[name] = module


VENDOR_NVFP4_FLAGS = {
    "raon_fp32_export_head": False,
    "raon_merged_gate_up": True,
    "latent_moe_use_norm": False,
    "routed_expert_hidden_size": None,
    "attn_res_block_size": None,
    "num_nextn_predict_layers": 0,
}


def install_vendor_nvfp4(model: Model, config: dict, weight_names) -> dict:
    """Swap modules to the KRAFTON Raon-1-mini-mlx NVFP4 pack's storage.

    * ``raon_global_scaled_linears``: weight-only NVFP4 g16 with a direct
      fp32 output multiplier (``runtime.nvfp4_linear.GlobalScaledNVFP4Linear``:
      bf16(fp32(bf16(qmm)) * s), the vendor's two roundings);
    * ``moe.switch_mlp.*``: NVFP4 g16 experts with the in-kernel per-expert
      scale (:class:`NVFP4GlobalSwitchLinear`); the checkpoint keeps gate and
      up separate (the vendor merges them at load, an exact reassociation we
      do not need for correctness);
    * ``lm_head``: MXFP8 g32.
    Everything else (embedding, kv_b_proj, router, norms, mHC, convs) stays
    in its stored dtype.  Returns an install receipt.
    """
    from ..nvfp4_linear import GlobalScaledNVFP4Linear

    names = set(weight_names)
    for key, expected in VENDOR_NVFP4_FLAGS.items():
        if config.get(key, expected) != expected:
            raise ValueError(f"Raon NVFP4 pack flag {key}={config.get(key)!r} is not supported")
    quant = config.get("quantization") or {}
    if (quant.get("mode"), quant.get("bits"), quant.get("group_size")) != ("nvfp4", 4, 16):
        raise ValueError("Raon vendor pack must declare nvfp4 4-bit group 16")
    dense = list(config.get("raon_global_scaled_linears") or ())
    if not dense:
        raise ValueError("Raon vendor pack lists no global-scaled linears")
    for path in dense:
        linear = _module(model, path)
        if type(linear) is not nn.Linear or f"{path}.global_scale" not in names:
            raise ValueError(f"Raon global-scaled linear {path} is not a plain Linear with a scale")
        out_dims, in_dims = linear.weight.shape
        _replace(model, path, GlobalScaledNVFP4Linear(in_dims, out_dims, bias=False))
    switch = 0
    for index, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        glu = layer.moe.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            path = f"model.layers.{index}.moe.switch_mlp.{name}"
            spec = quant.get(path) or {}
            if spec != {"group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": True}:
                raise ValueError(f"Raon expert bank {path} is not nvfp4 with a global scale")
            old = glu[name]
            experts, out_dims, in_dims = old.weight.shape
            glu[name] = NVFP4GlobalSwitchLinear(in_dims, out_dims, experts)
            switch += 1
    head = quant.get("lm_head") or {}
    if (head.get("mode"), head.get("bits"), head.get("group_size")) != ("mxfp8", 8, 32):
        raise ValueError("Raon vendor head must be mxfp8 8-bit group 32")
    vocab, hidden = model.lm_head.weight.shape
    model.lm_head = nn.QuantizedLinear(hidden, vocab, bias=False, group_size=32, bits=8, mode="mxfp8")
    stray = sorted(n[:-len(".scales")] for n in names if n.endswith(".scales")
                   and n[:-len(".scales")] not in set(dense) | {"lm_head"}
                   and ".moe.switch_mlp." not in n)
    if stray:
        raise ValueError(f"Raon pack quantized tensors outside the declared set: {stray[:4]}")
    return {"format": "krafton-nvfp4-g16-mxfp8-head", "dense_global_scaled": len(dense),
            "expert_banks": switch, "head": "mxfp8-g32",
            "dense_scale_law": "output_side_bf16_twice", "expert_scale_law": "in_kernel_div_2688"}


def install_affine(model: Model, config: dict, weight_names) -> dict:
    """mlx2 affine conversion: every module the artifact stores ``.scales`` for
    becomes an MLX affine quantized module (``QuantizedLinear`` /
    mlx2 ``QuantizedSwitchLinear``) at its declared bits/group; the rest keeps
    its stored dtype.  The stock switch module keeps mlx2's routed-expert
    levers available (rhs row pad, NAX gather, fused gate/up)."""
    names = set(weight_names)
    quant = config.get("quantization") or {}
    default = {"group_size": int(quant["group_size"]), "bits": int(quant["bits"])}
    counts = {"linear": 0, "switch": 0}

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or f"{path}.scales" not in names:
            return False
        counts["switch" if ".switch_mlp." in path else "linear"] += 1
        spec = quant.get(path)
        return {**default, **spec, "mode": "affine"} if isinstance(spec, dict) else {**default, "mode": "affine"}

    nn.quantize(model, class_predicate=predicate)
    expected = {n[: -len(".scales")] for n in names if n.endswith(".scales")}
    if counts["linear"] + counts["switch"] != len(expected):
        raise ValueError("Raon affine artifact quantizes modules the model does not expose")
    return {"format": "mlx2-affine-a4g64-h8g64", "quantized_linears": counts["linear"],
            "expert_banks": counts["switch"]}


def install_mxfp8_experts(model: Model) -> dict:
    """mlx2 conversion: routed experts as MLX ``mxfp8`` (the source's own E4M3
    codes and E8M0 group-32 scales, byte for byte); everything else bf16."""
    from .switch_layers import QuantizedSwitchLinear

    banks = 0
    for layer in model.layers:
        if not layer.is_moe:
            continue
        glu = layer.moe.switch_mlp
        for name in ("gate_proj", "up_proj", "down_proj"):
            experts, out_dims, in_dims = glu[name].weight.shape
            glu[name] = QuantizedSwitchLinear(in_dims, out_dims, experts, bias=False,
                                              group_size=32, bits=8, mode="mxfp8")
            banks += 1
    return {"format": "mlx2-mxfp8-experts-bf16-rest", "expert_banks": banks}


def install_for_layout(model: Model, layout: str, config: dict, weight_names) -> dict:
    """The storage installer for an inspected artifact layout (adapter and tools)."""
    if layout == "krafton_nvfp4":
        return install_vendor_nvfp4(model, config, weight_names)
    if layout == "mlx2_affine":
        return install_affine(model, config, weight_names)
    if layout == "mlx2_mxfp8e":
        return install_mxfp8_experts(model)
    raise ValueError(f"unknown Raon artifact layout {layout!r}")
