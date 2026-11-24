# SPDX-License-Identifier: MIT
"""Nemotron 3.6 Academia (``Nemotron3_6AcademiaForCausalLM``), native mlx2 model.

No checkpoint of this family has been released (2026-11-23).  The semantics
are ported from the llama.cpp integration in ggml-org/llama.cpp#32766 (head
20af26f2b) and its rectangular gated-delta-net operator, #32765 (head
f677133bc); see provenance/nemotron36-academia.json.  Everything here is
``implemented`` only: nothing has run against real weights.

Architecture (physical layers ``input + recurrent + output``, each one of
``M`` Gated Delta Product, ``*`` dense NoPE attention, ``-`` relu^2 MLP):

* the ``recurrent`` block runs ``n_recurrent_loops`` times with shared weights
  (virtual layers ``input + recurrent * loops + output``);
* a sinusoidal loop condition (progress, budget) -> MLP -> per-layer FiLM
  (``1 + gamma``, ``beta``), residual gate (``1 + gate``) and an optional
  low-rank gated adapter, on every recurrent and output layer;
* a loop seam: entry RMSNorm + FiLM and optional input mixing from the second
  loop on, a learned state update ``state += sigmoid(W x) * softplus(alpha) *
  (cur - state)`` and an exit gate (``1 + gate`` on the first loop);
* Gated Delta Product layers: ``m`` Householder sub-steps per token on a
  rectangular ``Dk x Dv`` delta-rule state (64 x 32 in the announced model).
  The decay applies on the first sub-step only, the query reads the state
  after the last sub-step.  mlx2 runs this exactly on the existing GDN
  recurrence by flattening the sub-steps into time: sub-steps after the
  first get ``a = -inf`` (decay 1), and the readout keeps every m-th row.

Attention KV cache modes (``recurrent_kv_cache_mode``): ``per_loop`` keeps one
KV plane per virtual attention layer; ``last_loop`` keeps one plane per
physical recurrent attention layer, which every loop overwrites for the
current rows, so past rows hold the final loop's K/V.  ``last_loop`` output
therefore depends on how a prompt is split into forwards (llama.cpp
``check_academia_cache`` asserts exactly that); conv and GDP states are always
per loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..cache_operations import KV_OPERATIONS
from .base import BaseModelArgs, create_attention_mask, create_ssm_mask
from .base import scaled_dot_product_attention
from .cache import ArraysCache, BatchKVCache, KVCache
from .gated_delta import gated_delta_update, normalize_gdn_qk

MODEL_TYPE = "nemotron3_6_academia"
ARCHITECTURE = "Nemotron3_6AcademiaForCausalLM"
LAYER_KINDS = {"linear_attention": "M", "full_attention": "*", "mlp": "-"}
CACHE_MODES = ("per_loop", "last_loop")
MAX_HOUSEHOLDER = 16  # llama.cpp load_arch_hparams bound


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = MODEL_TYPE
    hidden_size: int = 0
    vocab_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: Optional[int] = None
    intermediate_size: int = 0
    rms_norm_eps: Optional[float] = None
    layer_norm_epsilon: Optional[float] = None
    norm_eps: Optional[float] = None
    hidden_act: str = "relu2"
    tie_word_embeddings: bool = False
    n_input_layers: int = 0
    n_recurrent_layers: int = 0
    n_output_layers: int = 0
    n_recurrent_loops: int = 0
    layer_types: List[str] = field(default_factory=list)
    recurrent_kv_cache_mode: str = "per_loop"
    loop_conditioning_hidden_size: int = 0
    loop_conditioning_max_loops: int = 0
    loop_adapter_rank: int = 0
    loop_input_mixing: bool = False
    linear_conv_kernel_dim: int = 4
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_householder: int = 1
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self):
        if self.head_dim is None and self.num_attention_heads:
            self.head_dim = self.hidden_size // self.num_attention_heads
        validate_args(self)

    @property
    def eps(self) -> float:
        for value in (self.rms_norm_eps, self.layer_norm_epsilon, self.norm_eps):
            if value is not None:
                return float(value)
        raise ValueError("Nemotron 3.6 Academia config has no RMSNorm epsilon")

    @property
    def pattern(self) -> str:
        return "".join(LAYER_KINDS[kind] for kind in self.layer_types)

    @property
    def n_physical(self) -> int:
        return self.n_input_layers + self.n_recurrent_layers + self.n_output_layers

    def physical_index(self) -> List[int]:
        """Virtual execution layer -> physical weight layer (llama.cpp ``physical``)."""
        n_in, n_rec = self.n_input_layers, self.n_recurrent_layers
        return (
            list(range(n_in))
            + list(range(n_in, n_in + n_rec)) * self.n_recurrent_loops
            + list(range(n_in + n_rec, self.n_physical))
        )


def validate_args(args: ModelArgs) -> None:
    """The llama.cpp ``load_arch_hparams`` gates, failing closed on a bad config."""
    if args.hidden_act != "relu2":
        raise ValueError("Nemotron 3.6 Academia requires relu2 activation")
    unknown = set(args.layer_types) - set(LAYER_KINDS)
    if unknown:
        raise ValueError(f"unknown Nemotron 3.6 Academia layer types: {sorted(unknown)}")
    if args.recurrent_kv_cache_mode not in CACHE_MODES:
        raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
    n_cond = args.loop_conditioning_hidden_size
    if (
        not args.n_recurrent_layers
        or not args.n_recurrent_loops
        or not args.loop_conditioning_max_loops
        or len(args.layer_types) != args.n_physical
        or not n_cond
        or n_cond % 2
        or n_cond > args.hidden_size
        or args.loop_adapter_rank < 0
        or args.loop_adapter_rank > args.hidden_size
    ):
        raise ValueError("invalid Nemotron 3.6 Academia recurrence metadata")
    nh, nk = args.num_attention_heads, args.num_key_value_heads
    hk, hv = args.linear_num_key_heads, args.linear_num_value_heads
    m = args.linear_num_householder
    if (
        not nh or not nk or nh % nk or not args.intermediate_size or not args.head_dim
        or not hk or not hv or hv % hk
        or not args.linear_key_head_dim or not args.linear_value_head_dim
        or not 1 <= m <= MAX_HOUSEHOLDER or args.linear_conv_kernel_dim < 2
    ):
        raise ValueError("invalid Nemotron 3.6 Academia attention dimensions")
    if not (math.isfinite(args.eps) and args.eps > 0):
        raise ValueError("Nemotron 3.6 Academia RMSNorm epsilon must be positive")
    if args.attention_bias or args.mlp_bias:
        # llama.cpp loads no projection or output bias for this family.
        raise ValueError("Nemotron 3.6 Academia attention/MLP bias is not implemented")


# Keys the llama.cpp converter reads with find_hparam (no default): a config
# missing one is refused instead of taking this module's dataclass default.
REQUIRED_CONFIG_KEYS = (
    "hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "hidden_act", "n_input_layers", "n_recurrent_layers",
    "n_output_layers", "n_recurrent_loops", "layer_types",
    "loop_conditioning_hidden_size", "loop_conditioning_max_loops",
    "loop_adapter_rank", "loop_input_mixing", "linear_conv_kernel_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_num_householder",
)


def check_required_config(config: dict) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"Nemotron 3.6 Academia config lacks {missing}")


class Linear(nn.Linear):
    """Biased projection (all loop-conditioning heads carry a bias)."""

    def __init__(self, din: int, dout: int):
        super().__init__(din, dout, bias=True)


def loop_condition_features(n_cond: int, progress: float, budget: float) -> mx.array:
    """``build_condition``'s sinusoidal input: ``[cos, sin](100 v f)`` for both values."""
    half = n_cond // 2
    freq = mx.exp(mx.arange(half, dtype=mx.float32) * (-math.log(100.0) / half))

    def embed(value):
        args = freq * (100.0 * value)
        return mx.concatenate([mx.cos(args), mx.sin(args)])

    return mx.concatenate([embed(progress), embed(budget)])


class LoopConditioner(nn.Module):
    def __init__(self, n_cond: int):
        super().__init__()
        self.up = Linear(2 * n_cond, n_cond)
        self.down = Linear(n_cond, n_cond)

    def __call__(self, features):
        return self.down(nn.silu(self.up(features)))


class LoopSeam(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n_cond, d = args.loop_conditioning_hidden_size, args.hidden_size
        self.entry_norm = nn.RMSNorm(d, eps=args.eps)
        self.entry_gamma = Linear(n_cond, d)
        self.entry_beta = Linear(n_cond, d)
        self.exit_gate = Linear(n_cond, d)
        if args.loop_input_mixing:
            self.input_mix = Linear(n_cond, d)


class StateUpdate(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.proj = Linear(d, d)
        self.base_alpha_raw = mx.zeros((d,))


class LayerConditioning(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n_cond, d, rank = (
            args.loop_conditioning_hidden_size, args.hidden_size, args.loop_adapter_rank
        )
        self.gamma_head = Linear(n_cond, d)
        self.beta_head = Linear(n_cond, d)
        self.gate_head = Linear(n_cond, d)
        self.rank = rank
        if rank:
            self.adapter_gate = Linear(n_cond, rank)
            self.up_proj = nn.Linear(rank, d, bias=False)
            self.down_proj = nn.Linear(d, rank, bias=False)

    def modulate(self, x, condition):
        """FiLM + optional gated adapter; returns the input and the residual gate."""
        x = x * (1.0 + self.gamma_head(condition)) + self.beta_head(condition)
        gate = 1.0 + self.gate_head(condition)
        if self.rank:
            x = x + self.up_proj(self.down_proj(x) * self.adapter_gate(condition))
        return x, gate


class Attention(nn.Module):
    """Dense causal GQA attention without position encoding (llama.cpp: NoPE)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        d, self.n_heads, self.n_kv = (
            args.hidden_size, args.num_attention_heads, args.num_key_value_heads
        )
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        bias = args.attention_bias
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=bias)

    def __call__(self, x, mask, cache, *, rewrite: bool = False):
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_kv, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_kv, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            if rewrite:
                k, v = rewrite_last_rows(cache, k, v)
            else:
                k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, S, -1))


LAST_LOOP_KV_OPERATIONS = replace(
    KV_OPERATIONS,
    state_revision="nemotron36-academia.last-loop-kv.v1",
    numerics="stored_dtype;prefill-partition-dependent",
    exact_prefix=False,
)


class LastLoopKVCache(KVCache):
    """The shared ``last_loop`` plane: an ordinary KV plane that is not an exact prefix.

    Past rows hold the final loop's K/V of whichever forward wrote them, so a
    prefix computed in one partition and continued in another gives different
    numbers.  Declaring ``exact_prefix=False`` makes APCv2 refuse to store any
    prompt cache that contains this plane.  Merge and extract keep the class,
    so a lane taken out of a decode batch is still refused.
    """

    cache_operations_contract = LAST_LOOP_KV_OPERATIONS

    def committed_descriptor_clone(self):
        clone = super().committed_descriptor_clone()
        clone.__class__ = LastLoopKVCache
        return clone

    @classmethod
    def merge(cls, caches):
        batch = BatchKVCache.merge(caches)
        batch.__class__ = LastLoopBatchKVCache
        return batch


class LastLoopBatchKVCache(BatchKVCache):
    """Batched ``last_loop`` plane; BatchKVCache declares no contract, nor does this."""

    def extract(self, idx):
        cache = super().extract(idx)
        cache.__class__ = LastLoopKVCache
        return cache


def rewrite_last_rows(cache, keys, values):
    """``last_loop``: overwrite this forward's rows in a shared KV plane.

    The first loop appended them; a later loop replaces them so the plane ends
    the forward holding the final loop's K/V (llama.cpp writes every loop into
    the same cache layer).  Only the contiguous planes (``KVCache`` and
    ``BatchKVCache`` with their ``last_loop`` subclasses) are supported; a
    quantized or rotating plane fails closed.  Bucketed batch-attention groups
    hold copies of the rows, so they are dropped and rebuilt on the next read.
    """
    if type(cache) in (KVCache, LastLoopKVCache):
        end = cache.offset
    elif type(cache) in (BatchKVCache, LastLoopBatchKVCache):
        cache._invalidate_attention_groups()
        end = cache._idx
    else:
        raise NotImplementedError(
            f"last_loop KV sharing does not support {type(cache).__name__}"
        )
    rows = keys.shape[2]
    if cache.keys is None or end < rows:
        raise RuntimeError("last_loop rewrite before the first loop appended its rows")
    cache.keys[..., end - rows : end, :] = keys
    cache.values[..., end - rows : end, :] = values
    return cache.keys_and_values()


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        return self.down_proj(nn.relu2(self.up_proj(x)))


class GatedDeltaProduct(nn.Module):
    """GDP mixer: ``m`` Householder delta-rule sub-steps per token, rectangular state.

    Recurrent state layout is mlx2's GDN layout ``[B, Hv, Dv, Dk]`` (the
    transpose of ggml's ``[S_k, S_v, H_v]`` column-major tensor; same
    numbers).  Conv channels are ``[v (m*Dv*Hv) | k (m*Dk*Hk) | q (Dk*Hk)]``,
    the order llama.cpp feeds ``ssm_conv`` and the converter keeps unchanged.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        d = args.hidden_size
        self.m = args.linear_num_householder
        self.num_k_heads, self.num_v_heads = args.linear_num_key_heads, args.linear_num_value_heads
        self.head_k_dim, self.head_v_dim = args.linear_key_head_dim, args.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.m * self.value_dim + (self.m + 1) * self.key_dim
        self.eps = args.eps
        self.q_proj = nn.Linear(d, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d, self.m * self.key_dim, bias=False)
        self.v_proj = nn.Linear(d, self.m * self.value_dim, bias=False)
        self.z_proj = nn.Linear(d, self.value_dim, bias=False)
        self.a_proj = nn.Linear(d, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(d, self.m * self.num_v_heads, bias=False)
        self.A_log = mx.zeros((self.num_v_heads,))
        self.dt_bias = mx.zeros((self.num_v_heads,))
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.conv_kernel_size,
            groups=self.conv_dim, padding=0, bias=False,
        )
        self.norm = nn.RMSNorm(self.value_dim)  # weight only; grouped norm below
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    def __call__(self, x, mask, cache):
        B, S, _ = x.shape
        m, Hk, Hv, Dk, Dv = self.m, self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim
        if cache is not None and getattr(cache, "speculating", False):
            raise NotImplementedError("Nemotron 3.6 Academia GDP has no speculative rollback")
        mixed = mx.concatenate([self.v_proj(x), self.k_proj(x), self.q_proj(x)], axis=-1)
        if mask is not None:
            mixed = mx.where(mask[..., None], mixed, 0)
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=mixed.dtype)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])
        conv = nn.silu(self.conv1d(conv_input))
        v_end, k_end = m * self.value_dim, m * self.value_dim + m * self.key_dim
        v = conv[..., :v_end].reshape(B, S * m, Hv, Dv)
        k = conv[..., v_end:k_end].reshape(B, S * m, Hk, Dk)
        q = conv[..., k_end:].reshape(B, S, 1, Hk, Dk)
        q = mx.broadcast_to(q, (B, S, m, Hk, Dk)).reshape(B, S * m, Hk, Dk)
        q, k = normalize_gdn_qk(q, k)
        # Decay on the first sub-step only: a = -inf gives exp(-exp(A) * 0) == 1.
        a = self.a_proj(x)[:, :, None, :]
        if m > 1:
            pad = mx.full((B, S, m - 1, Hv), -mx.inf, dtype=a.dtype)
            a = mx.concatenate([a, pad], axis=2)
        a = a.reshape(B, S * m, Hv)
        b = self.b_proj(x).reshape(B, S * m, Hv)
        sub_mask = None if mask is None else mx.repeat(mask, m, axis=1)
        state = cache[1] if cache is not None else None
        y, state = gated_delta_update(
            q, k, v, a, b, self.A_log, self.dt_bias, state, sub_mask,
            use_kernel=not self.training, allow_core=False,
        )
        if cache is not None:
            cache[1] = state
        y = y.reshape(B, S, m, Hv, Dv)[:, :, -1]
        z = self.z_proj(x).reshape(B, S, Hv, Dv)
        # Gate then RMS-normalize per key-head group (Hv/Hk value heads each).
        y = y.astype(mx.float32) * nn.silu(z.astype(mx.float32))
        y = mx.fast.rms_norm(y.reshape(B, S, Hk, -1), None, self.eps).reshape(B, S, -1)
        y = (y * self.norm.weight.astype(mx.float32)).astype(x.dtype)
        return self.out_proj(y)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, physical: int):
        super().__init__()
        self.kind = args.pattern[physical]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.eps)
        self.mixer = {"M": GatedDeltaProduct, "*": Attention, "-": MLP}[self.kind](args)
        if physical >= args.n_input_layers:
            self.loop_conditioning = LayerConditioning(args)

    def __call__(self, x, condition, *, attn_mask=None, ssm_mask=None, cache=None, rewrite=False):
        h = self.norm(x)
        gate = None
        if condition is not None:
            h, gate = self.loop_conditioning.modulate(h, condition)
        if self.kind == "-":
            h = self.mixer(h)
        elif self.kind == "*":
            h = self.mixer(h, attn_mask, cache, rewrite=rewrite)
        else:
            h = self.mixer(h, ssm_mask, cache)
        if gate is not None:
            h = h * gate
        return x + h


class AcademiaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, p) for p in range(args.n_physical)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.eps)
        self.loop_conditioner = LoopConditioner(args.loop_conditioning_hidden_size)
        self.loop_seam = LoopSeam(args)
        self.state_update = StateUpdate(args.hidden_size)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = AcademiaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.physical = args.physical_index()
        self.cache_slots, self.n_cache = self._cache_slots()

    # ----- topology -----
    def _loop_of(self, virtual: int) -> Optional[int]:
        """Loop index of a virtual layer inside the recurrent block, else None."""
        a = self.args
        start = a.n_input_layers
        if start <= virtual < start + a.n_recurrent_layers * a.n_recurrent_loops:
            return (virtual - start) // a.n_recurrent_layers
        return None

    def _cache_slots(self):
        """Virtual layer -> cache list index (``None`` for MLP layers).

        Mirrors llama.cpp ``cache_layer``: under ``last_loop`` every loop's
        visit of a recurrent attention layer maps to the first loop's plane.
        """
        a = self.args
        slots, owner = [], {}
        count = 0
        for virtual, p in enumerate(self.physical):
            kind = a.pattern[p]
            if kind == "-":
                slots.append(None)
                continue
            shared = (
                kind == "*"
                and a.recurrent_kv_cache_mode == "last_loop"
                and self._loop_of(virtual) is not None
            )
            if shared and p in owner:
                slots.append(owner[p])
                continue
            slots.append(count)
            if shared:
                owner[p] = count
            count += 1
        return slots, count

    @property
    def layers(self):
        """Physical layers (weights); the virtual schedule is ``physical``."""
        return self.model.layers

    def layer_receipt(self) -> dict:
        a = self.args
        return {
            "pattern": a.pattern,
            "physical_layers": a.n_physical,
            "virtual_layers": len(self.physical),
            "loops": a.n_recurrent_loops,
            "max_loops": a.loop_conditioning_max_loops,
            "recurrent_kv_cache_mode": a.recurrent_kv_cache_mode,
            "householder": a.linear_num_householder,
            "gdp_state": [a.linear_num_value_heads, a.linear_value_head_dim, a.linear_key_head_dim],
            "cache_planes": self.n_cache,
        }

    def make_cache(self):
        caches = [None] * self.n_cache
        for virtual, slot in enumerate(self.cache_slots):
            if slot is None or caches[slot] is not None:
                continue
            kind = self.args.pattern[self.physical[virtual]]
            shared = (kind == "*" and self.args.recurrent_kv_cache_mode == "last_loop"
                      and self._loop_of(virtual) is not None)
            if kind == "M":
                cache = ArraysCache(size=2)
            else:
                cache = LastLoopKVCache() if shared else KVCache()
            if kind == "M":
                cache.apcv2_layer_index = virtual
            caches[slot] = cache
        return caches

    # ----- forward -----
    def conditions(self, dtype):
        a = self.args
        loops, max_loops = a.n_recurrent_loops, a.loop_conditioning_max_loops
        budget = min(loops, max_loops) / max_loops
        features = mx.stack([
            loop_condition_features(a.loop_conditioning_hidden_size, (k + 1) / loops, budget)
            for k in range(loops)
        ])
        return self.model.loop_conditioner(features.astype(dtype))

    def __call__(self, inputs: mx.array, cache: Optional[List[Any]] = None, input_embeddings=None):
        a, inner = self.args, self.model
        h = input_embeddings if input_embeddings is not None else inner.embed_tokens(inputs)
        if cache is None:
            cache = [None] * self.n_cache
        first_attn = next((cache[s] for v, s in enumerate(self.cache_slots)
                           if s is not None and a.pattern[self.physical[v]] == "*"), None)
        first_ssm = next((cache[s] for v, s in enumerate(self.cache_slots)
                          if s is not None and a.pattern[self.physical[v]] == "M"), None)
        attn_mask = create_attention_mask(h, first_attn)
        ssm_mask = create_ssm_mask(h, first_ssm)
        conds = self.conditions(h.dtype)

        def run(virtual, x, condition):
            slot = self.cache_slots[virtual]
            layer_cache = None if slot is None else cache[slot]
            loop = self._loop_of(virtual)
            rewrite = (
                a.recurrent_kv_cache_mode == "last_loop"
                and loop is not None and loop > 0
                and a.pattern[self.physical[virtual]] == "*"
                and layer_cache is not None
            )
            return inner.layers[self.physical[virtual]](
                x, condition, attn_mask=attn_mask, ssm_mask=ssm_mask,
                cache=layer_cache, rewrite=rewrite,
            )

        virtual = 0
        for _ in range(a.n_input_layers):
            h = run(virtual, h, None)
            virtual += 1
        seam = inner.loop_seam
        alpha = nn.softplus(inner.state_update.base_alpha_raw)
        input_h = state = h
        for k in range(a.n_recurrent_loops):
            c = conds[k]
            loop_input = state + input_h
            cur = loop_input
            if k > 0:
                cur = seam.entry_norm(cur) * (1.0 + seam.entry_gamma(c)) + seam.entry_beta(c)
                if a.loop_input_mixing:
                    cur = cur + input_h * seam.input_mix(c)
            for _ in range(a.n_recurrent_layers):
                cur = run(virtual, cur, c)
                virtual += 1
            beta = mx.sigmoid(inner.state_update.proj(loop_input)) * alpha
            new_state = state + beta * (cur - state)
            gate = seam.exit_gate(c) + (1.0 if k == 0 else 0.0)
            state = state + (new_state - state) * gate
        h = state
        for _ in range(a.n_output_layers):
            h = run(virtual, h, conds[-1])
            virtual += 1
        h = inner.norm(h)
        if a.tie_word_embeddings:
            return inner.embed_tokens.as_linear(h)
        return self.lm_head(h)

    # ----- checkpoint loading -----
    def sanitize(self, weights: dict) -> dict:
        """Hugging Face names -> module names (the llama.cpp converter's mapping)."""
        out = {}
        for name, value in weights.items():
            if "rotary_emb.inv_freq" in name:
                continue
            if self.args.tie_word_embeddings and name == "lm_head.weight":
                continue
            name = name.replace("model.loop_conditioner.mlp.0.", "model.loop_conditioner.up.")
            name = name.replace("model.loop_conditioner.mlp.2.", "model.loop_conditioner.down.")
            if name == "model.state_update.base_alpha_raw.weight":
                name = "model.state_update.base_alpha_raw"
            if ".input_layernorm." in name and name.startswith("model.layers."):
                name = name.replace(".input_layernorm.", ".norm.")
            if name.endswith(".mixer.conv1d.weight") and value.ndim == 3 and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            out[name] = value
        return out


__all__ = ["ARCHITECTURE", "MODEL_TYPE", "Model", "ModelArgs", "loop_condition_features"]
