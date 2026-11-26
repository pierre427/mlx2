# Attribution (per section, matching the source files' own headers):
# * Gated MLP: ``silu_mul`` from mlx2's dflash_base / muse_dflash (Apache-2.0)
#   / laguna_dflash (MIT) / eagle3 (MIT) / cohere_eagle (MIT) copies;
#   ``silu_mul_fp32`` from inkling2_dspark (Apache-2.0, adapted from
#   Blaizzy/mlx-vlm and vLLM; provenance/inkling2-dspark.json).
# * Tap injection, block attention and the block layer loop: muse_dflash
#   (Apache-2.0; bstnxbt/dflash-mlx and transformers,
#   provenance/muse-dflash-assistant.json) and laguna_dflash (MIT;
#   Blaizzy/mlx-vlm, provenance/laguna-dflash.json).
# * Chain attention and the chain row loop: eagle3 and cohere_eagle (MIT
#   files whose architecture follows vLLM, Apache-2.0;
#   provenance/eagle3-drafter.json, provenance/north-cohere-eagle.json).
# * MTP fusion: the model files listed in docs/technologies/drafter_blocks.md,
#   each keeping its own provenance.
# No Eagle 4 Source-Available code is in this module: the EAGLE-4 commit
# window stays in runtime/drafters/eagle4.py.  See docs/PROVENANCE.md.
"""Drafter network blocks and MTP head fusion: one reference per block.

The speculative drafters (DFlash-family block drafters, EAGLE chain drafters)
and the native multi-token-prediction (MTP) heads of the target models are
built from a handful of small blocks that used to be copied file by file:

* a gated MLP ``down(act(gate(x)) * up(x))`` (:class:`GatedMLP`);
* the target-tap injection that turns concatenated target hidden states into
  the drafter's context rows (:func:`inject_target_taps`);
* a context-KV attention whose keys are committed target context plus an
  ephemeral proposal block or a round-local chain (:class:`DraftAttention`);
* the pre-norm block layer loop (:func:`prenorm_block_layers`);
* the MTP fusion ``proj([norm(embed); norm(hidden)])`` and its stream-sum
  variant (:func:`fuse_mtp`).

Every difference between the copies is a spec field named for what it does.
Parameter attribute names are the checkpoint tensor names and are themselves
spec fields where they differ (``qkv_layout``, ``MTPFusionSpec.embed_norm``
...), so existing checkpoints keep loading without any key remapping.

Weights are always per model.  Only the math is shared, and only where the
shared reference is bit-identical to the copy it replaced (CPU parity tests
in ``tests/test_technologies_drafter_blocks*.py``).

No capability is qualified by importing or constructing this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import mlx.core as mx
import mlx.nn as nn

from .dispatch import spec_from_config

# ---------------------------------------------------------------------------
# Gated MLP
# ---------------------------------------------------------------------------

#: ``silu_mul``: ``silu(gate) * up`` in the projection dtype (uncompiled, the
#: drafter law; not ``models.activations.swiglu`` which is ``mx.compile``-d).
#: ``silu_mul_fp32``: both projections cast to float32, the product cast back
#: to the input dtype before ``down_proj`` (vLLM's fp32 activation law).
GATE_LAWS = ("silu_mul", "silu_mul_fp32")


@dataclass(frozen=True)
class GatedMLPSpec:
    bias: bool = False
    gate_law: str = "silu_mul"

    def __post_init__(self):
        if self.gate_law not in GATE_LAWS:
            raise ValueError(f"unknown gate law {self.gate_law!r}")


def gated_mlp(spec: GatedMLPSpec, gate_proj, up_proj, down_proj, x: mx.array) -> mx.array:
    """Reference gated MLP."""
    if spec.gate_law == "silu_mul":
        return down_proj(nn.silu(gate_proj(x)) * up_proj(x))
    gate = gate_proj(x).astype(mx.float32)
    up = up_proj(x).astype(mx.float32)
    return down_proj((nn.silu(gate) * up).astype(x.dtype))


class GatedMLP(nn.Module):
    """``gate_proj`` / ``up_proj`` / ``down_proj`` (checkpoint names)."""

    def __init__(self, dim: int, hidden: int, spec: GatedMLPSpec = GatedMLPSpec()):
        super().__init__()
        self.mlp_spec = spec
        self.gate_proj = nn.Linear(dim, hidden, bias=spec.bias)
        self.up_proj = nn.Linear(dim, hidden, bias=spec.bias)
        self.down_proj = nn.Linear(hidden, dim, bias=spec.bias)

    def __call__(self, x: mx.array) -> mx.array:
        return gated_mlp(self.mlp_spec, self.gate_proj, self.up_proj, self.down_proj, x)


# ---------------------------------------------------------------------------
# Target-tap injection (drafter context rows)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TapInjectionSpec:
    """``hidden_norm(fc(cat(tap_norm_i(tap_i))))``.

    ``per_tap_norm``: each target tap has its own RMSNorm before ``fc``
    (Laguna's ``aux_hidden_norms``); otherwise the raw concatenation feeds
    ``fc``.
    """

    per_tap_norm: bool = False


def inject_target_taps(spec: TapInjectionSpec, hidden: mx.array, *, fc, hidden_norm,
                       tap_norms: Sequence | None = None) -> mx.array:
    if spec.per_tap_norm:
        parts = mx.split(hidden, len(tap_norms), axis=-1)
        normed = [norm(part) for norm, part in zip(tap_norms, parts)]
        hidden = mx.concatenate(normed, axis=-1)
    return hidden_norm(fc(hidden))


# ---------------------------------------------------------------------------
# Context-KV draft attention
# ---------------------------------------------------------------------------

#: Proposal/chain visibility laws.
#: ``bidirectional_window``: a block over a resident context ring; context
#:   column ``c`` is visible when ``c > row + resident - window``; the block is
#:   fully visible; no mask at all with no resident context.
#: ``causal_block``: all resident context visible, block lower-triangular.
#: ``causal``: chain attention, key position <= query position.
#: ``causal_window``: chain attention, ``0 <= query - key <= window``.
MASK_LAWS = ("bidirectional_window", "causal_block", "causal", "causal_window")
QKV_LAYOUTS = ("separate", "packed")
HEAD_GATES = ("none", "softplus_fp32")


@dataclass(frozen=True)
class DraftAttentionSpec:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    #: MLX's ``traditional`` (GPT-J interleaved pairing); False = NeoX halves.
    rope_traditional: bool = False
    #: q/k/v read ``input_width_factor * hidden_size`` (EAGLE-3 reads 2x).
    input_width_factor: int = 1
    #: ``separate`` = q_proj/k_proj/v_proj; ``packed`` = one qkv_proj.
    qkv_layout: str = "separate"
    qk_norm: bool = True
    o_bias: bool = False
    #: ``softplus_fp32``: per-head ``softplus(g_proj(x))`` (float32, cast to
    #: the attention dtype) scales each head's output before ``o_proj``.
    head_gate: str = "none"
    mask_law: str = "bidirectional_window"
    sliding_window: int | None = None

    def __post_init__(self):
        if self.qkv_layout not in QKV_LAYOUTS:
            raise ValueError(f"unknown qkv layout {self.qkv_layout!r}")
        if self.head_gate not in HEAD_GATES:
            raise ValueError(f"unknown head gate {self.head_gate!r}")
        if self.mask_law not in MASK_LAWS:
            raise ValueError(f"unknown mask law {self.mask_law!r}")
        if self.mask_law in ("bidirectional_window", "causal_window") and self.sliding_window is None:
            raise ValueError(f"mask law {self.mask_law!r} needs sliding_window")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")

    @property
    def scale(self) -> float:
        return self.head_dim ** -0.5

    @classmethod
    def from_config(cls, config: Any, **overrides) -> "DraftAttentionSpec":
        return spec_from_config(
            cls, config,
            {
                "hidden_size": "hidden_size",
                "num_attention_heads": "num_attention_heads",
                "num_key_value_heads": "num_key_value_heads",
                "head_dim": "head_dim",
                "rms_norm_eps": "rms_norm_eps",
                "rope_theta": "rope_theta",
                "sliding_window": "sliding_window",
            },
            **overrides,
        )


def block_mask(spec: DraftAttentionSpec, length: int, resident: int):
    """Mask for a proposal block of ``length`` rows over ``resident`` keys."""
    if spec.mask_law == "bidirectional_window":
        if not resident:
            return None
        # Context column c holds absolute position offset - resident + c.
        rows = mx.arange(length)[:, None]
        columns = mx.arange(resident)[None, :]
        context = columns > rows + resident - spec.sliding_window
        block = mx.ones((length, length), dtype=mx.bool_)
        return mx.concatenate([context, block], axis=-1)
    if spec.mask_law == "causal_block":
        return mx.concatenate(
            [mx.ones((length, resident), dtype=mx.bool_),
             mx.tril(mx.ones((length, length), dtype=mx.bool_))], axis=-1
        )
    raise ValueError(f"mask law {spec.mask_law!r} is a chain law, not a block law")


def chain_mask(spec: DraftAttentionSpec, *, prior_start: int, position: int, length: int):
    """Mask for chain queries ``position..+length`` over keys from ``prior_start``."""
    key_pos = mx.arange(prior_start, position + length)
    query_pos = mx.arange(position, position + length)
    if spec.mask_law == "causal":
        return key_pos[None, :] <= query_pos[:, None]
    if spec.mask_law == "causal_window":
        delta = query_pos[:, None] - key_pos[None, :]
        return (delta >= 0) & (delta <= spec.sliding_window)
    raise ValueError(f"mask law {spec.mask_law!r} is a block law, not a chain law")


class DraftAttention(nn.Module):
    """Context-KV attention for drafters.

    Two call shapes share the projections:

    * block drafters: :meth:`context_kv` commits target context K/V into a
      window cache (``append`` / ``keys_and_values`` / ``offset``);
      :meth:`block` attends one row's proposal block over it without storing;
    * chain drafters: :meth:`project` then :meth:`attend` over explicit prior
      keys (committed plus round-local chain).
    """

    def __init__(self, spec: DraftAttentionSpec):
        super().__init__()
        self.attn_spec = spec
        dim = spec.hidden_size
        width = spec.input_width_factor * dim
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.scale = spec.scale
        if spec.mask_law in ("bidirectional_window", "causal_window"):
            self.window = spec.sliding_window
        if spec.qkv_layout == "packed":
            self.qkv_proj = nn.Linear(
                width, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(width, self.n_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(width, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=spec.o_bias)
        if spec.head_gate == "softplus_fp32":
            self.g_proj = nn.Linear(dim, self.n_heads, bias=False)
        if spec.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=spec.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=spec.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=spec.rope_traditional, base=spec.rope_theta)

    # -- projections -------------------------------------------------------
    def _raw_qkv(self, x):
        batch, length, _ = x.shape
        if self.attn_spec.qkv_layout == "packed":
            qkv = self.qkv_proj(x)
            q_size = self.n_heads * self.head_dim
            kv_size = self.n_kv_heads * self.head_dim
            q = qkv[..., :q_size]
            k = qkv[..., q_size:q_size + kv_size]
            v = qkv[..., q_size + kv_size:]
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return (
            q.reshape(batch, length, self.n_heads, self.head_dim),
            k.reshape(batch, length, self.n_kv_heads, self.head_dim),
            v.reshape(batch, length, self.n_kv_heads, self.head_dim),
        )

    def project(self, x):
        """``(q, k, v)`` as ``[B, heads, L, D]``, q/k RMS-normed per head when
        the spec has ``qk_norm``; no RoPE."""
        q, k, v = self._raw_qkv(x)
        if self.attn_spec.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        return q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def _kv(self, x):
        if self.attn_spec.qkv_layout == "packed":
            _, k, v = self.project(x)
            return k, v
        batch, length, _ = x.shape
        k = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        if self.attn_spec.qk_norm:
            k = self.k_norm(k)
        return k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)

    def _gate_heads(self, x, out):
        """``out``: ``[B, L, heads * D]``."""
        if self.attn_spec.head_gate == "none":
            return out
        batch, length, _ = out.shape
        gate = nn.softplus(self.g_proj(x).astype(mx.float32)).astype(out.dtype)
        return (out.reshape(batch, length, self.n_heads, self.head_dim) * gate[..., None]).reshape(
            batch, length, -1
        )

    # -- block drafters ------------------------------------------------------
    def context_kv(self, context, cache):
        """Commit one row's context K/V (RoPE at absolute positions)."""
        k, v = self._kv(context)
        cache.append(self.rope(k, offset=cache.offset), v)

    def block(self, x, cache):
        """One row's proposal block over committed context (not stored)."""
        batch, length, _ = x.shape
        q, k, v = self.project(x)
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        ctx_k, ctx_v = cache.keys_and_values()
        resident = int(ctx_k.shape[2])
        keys = mx.concatenate([ctx_k.astype(k.dtype), k], axis=2) if resident else k
        values = mx.concatenate([ctx_v.astype(v.dtype), v], axis=2) if resident else v
        mask = block_mask(self.attn_spec, length, resident)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=self.scale, mask=mask)
        out = self._gate_heads(x, out.transpose(0, 2, 1, 3).reshape(batch, length, -1))
        return self.o_proj(out)

    # -- chain drafters ------------------------------------------------------
    def attend(self, q, k, v, *, position, prior_k, prior_v, prior_start=None):
        """One row.  ``q/k/v``: [1, H, L, D] for positions ``position..+L-1``;
        ``prior_*`` are earlier keys starting at absolute ``prior_start``
        (default: contiguous, ``position - prior_k.shape[2]``).  Returns the
        attention output and the rotated new keys."""
        length = q.shape[2]
        q = self.rope(q, offset=position)
        k = self.rope(k, offset=position)
        keys = mx.concatenate([prior_k, k.astype(prior_k.dtype)], axis=2) if prior_k.shape[2] else k
        values = mx.concatenate([prior_v, v.astype(prior_v.dtype)], axis=2) if prior_v.shape[2] else v
        if prior_start is None:
            prior_start = position - prior_k.shape[2]
        mask = chain_mask(self.attn_spec, prior_start=prior_start, position=position, length=length)
        out = mx.fast.scaled_dot_product_attention(
            q, keys.astype(q.dtype), values.astype(q.dtype), scale=self.scale, mask=mask
        )
        return out, k


def chain_attend_rows(attn: DraftAttention, q, k, v, caches, locals_, *, commit: bool):
    """Attend B chain rows, each over its committed cache plus round-local KV.

    ``caches``: one committed window cache per row.  ``locals_``: per-row
    round-local ``(k, v)`` (``(None, None)`` when empty), ignored when
    ``commit``.  A committed call appends the rotated keys to the row cache;
    otherwise the round-local KV is extended.  Returns ``(out [B, H, L, D],
    new_locals)``; ``new_locals`` is ``None`` when ``commit``.

    A cache is anything with ``keys_and_values()``, ``offset`` and
    ``append(k, v)``; a drafter with its own visibility rule over committed
    rows passes a view that applies it.
    """
    outputs = []
    new_locals = None if commit else []
    for row, cache in enumerate(caches):
        prior_k, prior_v = cache.keys_and_values()
        local = None if commit else locals_[row]
        if local is not None and local[0] is not None:
            prior_k = mx.concatenate([prior_k, local[0]], axis=2) if prior_k.shape[2] else local[0]
            prior_v = mx.concatenate([prior_v, local[1]], axis=2) if prior_v.shape[2] else local[1]
        position = cache.offset + (0 if local is None or local[0] is None else local[0].shape[2])
        out, rotated = attn.attend(
            q[row:row + 1], k[row:row + 1], v[row:row + 1],
            position=position, prior_k=prior_k, prior_v=prior_v,
            prior_start=position - prior_k.shape[2],
        )
        outputs.append(out)
        if commit:
            cache.append(rotated, v[row:row + 1])
        elif local[0] is None:
            new_locals.append((rotated, v[row:row + 1]))
        else:
            new_locals.append((
                mx.concatenate([local[0], rotated], axis=2),
                mx.concatenate([local[1], v[row:row + 1]], axis=2),
            ))
    return mx.concatenate(outputs, axis=0), new_locals


# ---------------------------------------------------------------------------
# Block layer loop
# ---------------------------------------------------------------------------

def prenorm_block_layers(layers, x: mx.array, rows) -> mx.array:
    """Sequential pre-norm layers over a proposal block, one cache row each.

    ``rows``: per batch row, the per-layer caches.  Attention runs one row at
    a time (each row has its own committed context); the MLP runs batched.
    """
    for index, layer in enumerate(layers):
        h = layer.input_layernorm(x)
        attn = mx.concatenate(
            [layer.self_attn.block(h[row:row + 1], caches[index]) for row, caches in enumerate(rows)],
            axis=0,
        )
        x = x + attn
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x


# ---------------------------------------------------------------------------
# MTP head fusion
# ---------------------------------------------------------------------------

#: ``concat``: ``projection(cat(parts in concat_order))`` with
#:   ``parts = {embed: embed_norm(e), hidden: hidden_norm(h)}``.
#: ``stream_sum``: ``embed_projection(embed_norm(e))`` broadcast-added to
#:   ``projection(hidden_norm(h))`` per residual stream (hyper-connection
#:   targets), flattened back to ``streams * hidden``.
MTP_LAYOUTS = ("concat", "stream_sum")


@dataclass(frozen=True)
class MTPFusionSpec:
    layout: str = "concat"
    concat_order: tuple[str, str] = ("embed", "hidden")
    # Module attribute names = checkpoint tensor names.
    embed_norm: str = "enorm"
    hidden_norm: str = "hnorm"
    projection: str = "eh_proj"
    embed_projection: str | None = None
    #: ``embed_dtype``: cast the hidden to the normed embedding's dtype before
    #: the hidden norm.
    hidden_cast: str = "none"
    #: ``stream_sum`` only: ``per_stream`` normalizes each stream (reshape
    #: first); ``packed`` normalizes the packed row with a grouped norm module
    #: that reshapes itself, then splits streams.
    hidden_norm_scope: str = "per_stream"

    def __post_init__(self):
        if self.layout not in MTP_LAYOUTS:
            raise ValueError(f"unknown MTP layout {self.layout!r}")
        if sorted(self.concat_order) != ["embed", "hidden"]:
            raise ValueError("concat_order must order 'embed' and 'hidden'")
        if self.hidden_cast not in ("none", "embed_dtype"):
            raise ValueError(f"unknown hidden cast {self.hidden_cast!r}")
        if self.hidden_norm_scope not in ("per_stream", "packed"):
            raise ValueError(f"unknown hidden norm scope {self.hidden_norm_scope!r}")
        if self.layout == "stream_sum" and self.embed_projection is None:
            raise ValueError("stream_sum fusion needs an embed_projection")


#: DeepSeek NEXTN tensor layout: ``enorm`` / ``hnorm`` / ``eh_proj``.
NEXTN_FUSION = MTPFusionSpec()
#: Qwen3.5-family layout: ``pre_fc_norm_embedding`` / ``pre_fc_norm_hidden`` / ``fc``.
PRE_FC_NORM_FUSION = MTPFusionSpec(
    embed_norm="pre_fc_norm_embedding", hidden_norm="pre_fc_norm_hidden", projection="fc"
)
#: Stream-sum layout over per-stream norms: ``enorm`` / ``hnorm`` / ``e_proj`` / ``h_proj``.
STREAM_SUM_FUSION = MTPFusionSpec(layout="stream_sum", projection="h_proj", embed_projection="e_proj")
#: Stream-sum layout over a packed grouped norm: ``pre_fc_norm_*`` / ``fc_embedding`` / ``fc_hidden``.
STREAM_SUM_PRE_FC_FUSION = MTPFusionSpec(
    layout="stream_sum", embed_norm="pre_fc_norm_embedding", hidden_norm="pre_fc_norm_hidden",
    projection="fc_hidden", embed_projection="fc_embedding", hidden_norm_scope="packed",
)
#: Hidden-first layout: ``hidden_norm`` / ``embed_norm`` / ``input_proj``.
HIDDEN_FIRST_FUSION = MTPFusionSpec(
    concat_order=("hidden", "embed"), embed_norm="embed_norm", hidden_norm="hidden_norm",
    projection="input_proj",
)


def fuse_mtp(spec: MTPFusionSpec, embeds: mx.array, hidden: mx.array, *,
             embed_norm: Callable, hidden_norm: Callable, projection: Callable,
             embed_projection: Callable | None = None) -> mx.array:
    """Reference MTP fusion over explicit callables."""
    e = embed_norm(embeds)
    if spec.layout == "concat":
        if spec.hidden_cast == "embed_dtype":
            hidden = hidden.astype(e.dtype)
        h = hidden_norm(hidden)
        parts = [e, h] if spec.concat_order == ("embed", "hidden") else [h, e]
        return projection(mx.concatenate(parts, axis=-1))
    # Streams = packed hidden width / embedding width (``hc_count`` /
    # ``num_residual_streams``), so one spec serves every stream count.
    width = embeds.shape[-1]
    n = hidden.shape[-1] // width
    e = embed_projection(e)
    if spec.hidden_norm_scope == "per_stream":
        b, s, nc = hidden.shape
        h = projection(hidden_norm(hidden.reshape(b, s, n, nc // n)))
        return (e[:, :, None, :] + h).reshape(b, s, nc)
    h = hidden_norm(hidden).reshape(*hidden.shape[:-1], n, width)
    h = projection(h)
    return (e[..., None, :] + h).reshape(*h.shape[:-2], n * width)


def fuse_mtp_module(spec: MTPFusionSpec, module, embeds: mx.array, hidden: mx.array) -> mx.array:
    """:func:`fuse_mtp` with the norms/projections read from ``module`` by
    their checkpoint names."""
    return fuse_mtp(
        spec, embeds, hidden,
        embed_norm=getattr(module, spec.embed_norm),
        hidden_norm=getattr(module, spec.hidden_norm),
        projection=getattr(module, spec.projection),
        embed_projection=(
            None if spec.embed_projection is None else getattr(module, spec.embed_projection)
        ),
    )


__all__ = [
    "DraftAttention", "DraftAttentionSpec", "GATE_LAWS", "GatedMLP", "GatedMLPSpec",
    "HEAD_GATES", "HIDDEN_FIRST_FUSION", "MASK_LAWS", "MTPFusionSpec", "MTP_LAYOUTS",
    "NEXTN_FUSION", "PRE_FC_NORM_FUSION", "QKV_LAYOUTS", "STREAM_SUM_FUSION",
    "STREAM_SUM_PRE_FC_FUSION", "TapInjectionSpec", "block_mask", "chain_attend_rows",
    "chain_mask", "fuse_mtp", "fuse_mtp_module", "gated_mlp", "inject_target_taps",
    "prenorm_block_layers",
]
