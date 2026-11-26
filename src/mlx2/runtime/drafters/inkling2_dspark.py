# SPDX-License-Identifier: Apache-2.0
# Adapted from Blaizzy/mlx-vlm@111ac668 mlx_vlm/speculative/drafters/
# inkling2_dspark/{dspark,config}.py, itself adapted from vLLM
# (WoosukKwon/vllm@d664cd1e vllm/models/inkling2/{dspark,ops/dspark}.py,
# Apache-2.0, copyright contributors to the vLLM project).
# See provenance/inkling2-dspark.json.
"""Inkling 2's bundled BF16 DSpark drafter: DFlare fusion + RNN hidden head.

Contract (single host):

- ``hidden`` passed to :meth:`Inkling2DSparkDraftModel.draft_block` holds the
  target features of the NEWLY committed context rows only, shape
  ``[B, L, len(target_layer_ids) * hidden_size]`` with ``L >= 1``: for each
  target layer id the hidden state entering that layer visit (the last id is
  the final post-norm hidden).  Those rows enter the persistent draft KV at
  positions ``cache.offset .. cache.offset + L - 1``; the anchor (last bonus
  token) sits at the next position.  Proposals never enter draft KV, so a
  rejection needs no draft rollback: the next call simply carries the
  accepted rows (previous anchor included) as its ``hidden``.
- The whole trained block (anchor + ``block_size - 1`` mask rows) is always
  run because block attention is bidirectional; fewer proposals are cut
  after the backbone.
- Local layers keep only ``sliding_window_size`` context positions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..models.cache import KVCache, RotatingKVCache
from ..technologies.drafter_blocks import GatedMLP, GatedMLPSpec, gated_mlp

LAW_VERSION = "inkling2-dspark-dflare-rnn-hidden-v1"
TARGET_MODEL_TYPES = ("inkling2_text", "inkling2_loop_text")

_MTP_KEYS = (
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "markov_rank", "max_position_embeddings",
    "sliding_window_size", "rope_theta", "rms_norm_eps", "target_layer_ids",
    "local_layer_ids", "block_size",
)
_MTP_OPTIONAL = {"markov_head_type", "q_bias", "o_bias", "torch_dtype"}


def _loop_dict(loop):
    if loop is None:
        return None
    value = asdict(loop) if is_dataclass(loop) else dict(loop)
    value.setdefault("is_varloop", False)
    return value


@dataclass
class Inkling2DSparkConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    markov_rank: int
    max_position_embeddings: int
    sliding_window_size: int
    rope_theta: float
    rms_norm_eps: float
    target_layer_ids: tuple
    local_layer_ids: tuple
    block_size: int
    vocab_size: int
    unpadded_vocab_size: Optional[int]
    final_logit_softcapping: Optional[float]
    target_model_type: str
    target_num_hidden_layers: int
    target_loop_config: Optional[dict]
    target_visits: int

    @classmethod
    def from_hf(cls, config: dict) -> "Inkling2DSparkConfig":
        """Parse a bundled checkpoint ``config.json`` (``mtp_config`` + text)."""
        from ..models.inkling2.config import TextConfig

        if not isinstance(config, dict):
            raise ValueError("Inkling 2 DSpark config must be a JSON object")
        text = config.get("text_config", config)
        draft = config.get("mtp_config") or text.get("mtp_config")
        if not isinstance(draft, dict) or not draft:
            raise ValueError("Inkling 2 DSpark requires mtp_config")
        unknown = set(draft) - set(_MTP_KEYS) - _MTP_OPTIONAL
        missing = [name for name in _MTP_KEYS if name not in draft]
        if unknown or missing:
            raise ValueError(
                f"Unsupported Inkling 2 mtp_config keys (unknown={sorted(unknown)}, missing={missing})"
            )
        if text.get("model_type") not in TARGET_MODEL_TYPES:
            raise ValueError("Inkling 2 DSpark requires an Inkling 2 text config")
        target = TextConfig.from_dict(text)  # validates the target topology
        values = {name: draft[name] for name in _MTP_KEYS}
        values["target_layer_ids"] = tuple(draft["target_layer_ids"])
        values["local_layer_ids"] = tuple(draft["local_layer_ids"])
        vocab = text["vocab_size"]
        args = cls(
            **values,
            vocab_size=vocab,
            unpadded_vocab_size=text.get("unpadded_vocab_size", vocab),
            final_logit_softcapping=text.get("final_logit_softcapping", 30.0),
            target_model_type=text["model_type"],
            target_num_hidden_layers=text["num_hidden_layers"],
            target_loop_config=_loop_dict(target.loop_config),
            target_visits=len(target.layer_indices),
        )
        if (
            draft.get("markov_head_type") != "rnn_hidden"
            or draft.get("q_bias")
            or draft.get("o_bias")
            or draft.get("torch_dtype", "bfloat16") != "bfloat16"
            or args.hidden_size != text["hidden_size"]
            or any(
                type(getattr(args, name)) is not int or getattr(args, name) < 1
                for name in ("hidden_size", "intermediate_size", "num_hidden_layers",
                             "num_attention_heads", "num_key_value_heads", "head_dim",
                             "markov_rank", "sliding_window_size", "block_size")
            )
            or args.block_size < 2
            or args.num_attention_heads % args.num_key_value_heads
            or args.head_dim % 2
            or (args.unpadded_vocab_size is not None
                and not 0 < args.unpadded_vocab_size <= vocab)
            or any(type(i) is not int or not 0 <= i < args.num_hidden_layers
                   for i in args.local_layer_ids)
            or len(set(args.local_layer_ids)) != len(args.local_layer_ids)
        ):
            raise ValueError("Unsupported Inkling 2 mtp_config")
        ids = list(args.target_layer_ids)
        if (
            len(ids) < 2
            or any(type(i) is not int for i in ids)
            or ids != sorted(set(ids))
            or ids[0] < 0
            or ids[-1] != args.target_visits
        ):
            raise ValueError("Invalid Inkling 2 DSpark target_layer_ids")
        return args

    @property
    def feature_width(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size


def expected_weights(args: Inkling2DSparkConfig) -> dict:
    """Exact ``mtp.safetensors`` schema: name -> shape (all BF16, no biases
    except the Markov head's two Linear layers)."""
    hidden, rank = args.hidden_size, args.markov_rank
    q = args.num_attention_heads * args.head_dim
    kv = args.num_key_value_heads * args.head_dim
    shapes = {
        "fuse_weight": [args.num_hidden_layers + 1, len(args.target_layer_ids)],
        "mask_embed": [hidden],
        "norm.weight": [hidden],
        "markov_head.dflare_norm.weight": [hidden],
        "markov_head.hidden_out.weight": [hidden, rank],
        "markov_head.hidden_out.bias": [hidden],
        "markov_head.markov_w1.weight": [args.vocab_size, rank],
        "markov_head.rnn_proj.weight": [3 * rank, 2 * rank + 2 * hidden],
        "markov_head.rnn_proj.bias": [3 * rank],
    }
    for i in range(args.num_hidden_layers):
        p = f"layers.{i}."
        shapes.update({
            p + "dflare_norm.weight": [hidden],
            p + "input_layernorm.weight": [hidden],
            p + "post_attention_layernorm.weight": [hidden],
            p + "mlp.gate_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.up_proj.weight": [args.intermediate_size, hidden],
            p + "mlp.down_proj.weight": [hidden, args.intermediate_size],
            p + "self_attn.q_proj.weight": [q, hidden],
            p + "self_attn.k_proj.weight": [kv, hidden],
            p + "self_attn.v_proj.weight": [kv, hidden],
            p + "self_attn.k_proj_ctx.weight": [kv, hidden],
            p + "self_attn.v_proj_ctx.weight": [kv, hidden],
            p + "self_attn.o_proj.weight": [hidden, q],
            p + "self_attn.q_norm.weight": [args.head_dim],
            p + "self_attn.k_norm.weight": [args.head_dim],
            p + "self_attn.k_norm_ctx.weight": [args.head_dim],
        })
    return shapes


def _rowwise_linear(module, x):
    """Shared weights, each row with its own singleton-shaped reduction.

    mlx-vlm's GPU wide-GEMV kernel is not ported: every row runs the exact
    ``[1, L, H]`` product the singleton path runs, so batched == singleton.
    """
    return mx.concatenate(
        [module(mx.contiguous(x[row : row + 1])) for row in range(x.shape[0])]
    )


class DraftAttention(nn.Module):
    def __init__(self, config, local):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window_size if local else None
        for name, heads in (
            ("q_proj", self.heads),
            ("k_proj", self.kv_heads),
            ("v_proj", self.kv_heads),
            ("k_proj_ctx", self.kv_heads),
            ("v_proj_ctx", self.kv_heads),
        ):
            self[name] = nn.Linear(config.hidden_size, heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        for name in ("q_norm", "k_norm", "k_norm_ctx"):
            self[name] = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)
        # Opt-in (inkling2_dspark_kv_headroom); see _with_block.
        self.kv_headroom = False
        self.kv_headroom_stats = {"writes": 0, "grows": 0, "fallbacks": 0}

    def _heads(self, x, heads):
        return x.reshape(*x.shape[:2], heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x, context, cache):
        # Every query sees the whole block and the same window BEFORE its
        # anchor.  Only target context enters persistent KV; proposals never.
        if self.window and context.shape[1] > self.window:
            skip = context.shape[1] - self.window
            context = context[:, skip:]
            cache.offset += skip
        offset = cache.offset
        k = self.k_norm_ctx(self._heads(self.k_proj_ctx(context), self.kv_heads))
        v = self._heads(self.v_proj_ctx(context), self.kv_heads)
        k, v = cache.update_and_fetch(self.rope(k, offset=offset), v)
        if isinstance(cache, RotatingKVCache):
            k, v = cache._temporal_order(k), cache._temporal_order(v)
            k, v = k[..., -self.window :, :], v[..., -self.window :, :]
        offset += context.shape[1]
        q = self.q_norm(self._heads(self.q_proj(x), self.heads))
        block_k = self.k_norm(self._heads(self.k_proj(x), self.kv_heads))
        k, v = self._with_block(
            cache, k, v, self.rope(block_k, offset=offset), self._heads(self.v_proj(x), self.kv_heads)
        )
        out = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset=offset), k, v, scale=self.head_dim**-0.5, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


    # Spare tail slots kept past the fill level once a block write must grow.
    KV_HEADROOM = 16

    def _with_block(self, cache, k, v, block_k, block_v):
        """Context K/V (``k``, ``v``: the cache's first ``offset`` positions)
        followed by the ephemeral block's K/V.

        Default: ``mx.concatenate``, which copies the whole context every
        draft.  With ``kv_headroom`` (mechanism from g13-engine's
        DraftKVCache, see docs/PROVENANCE.md) a global layer's block is
        written into its KVCache's spare slots at ``offset ..`` and the
        attention reads ``keys[..., :offset + L, :]``: the same values in the
        same order.  ``offset`` does not move, so the block is never
        committed; it is stale data past the fill level that the next
        context append overwrites, which ``KVCache``'s append-only recovery
        contract (``_RECOVERY_APPEND_ONLY_FIELDS``) already allows.  Rotating
        (local) layers and any dtype mismatch keep the concatenation.
        """
        if not self.kv_headroom or type(cache) is not KVCache or cache.keys is None:
            return mx.concatenate([k, block_k], axis=2), mx.concatenate([v, block_v], axis=2)
        if block_k.dtype != cache.keys.dtype or block_v.dtype != cache.values.dtype:
            self.kv_headroom_stats["fallbacks"] += 1
            return mx.concatenate([k, block_k], axis=2), mx.concatenate([v, block_v], axis=2)
        start = cache.offset
        end = start + block_k.shape[2]
        if end > cache.keys.shape[2]:
            # One copy per growth (as KVCache.update_and_fetch grows), not
            # one per draft; capacity stays a multiple of KVCache.step.
            step = KVCache.step
            capacity = -(-(end + self.KV_HEADROOM) // step) * step
            cache.keys, cache.values = (
                mx.concatenate(
                    [buf[..., :start, :],
                     mx.zeros((*buf.shape[:2], capacity - start, buf.shape[3]), buf.dtype)],
                    axis=2,
                )
                for buf in (cache.keys, cache.values)
            )
            self.kv_headroom_stats["grows"] += 1
        cache.keys[..., start:end, :] = block_k
        cache.values[..., start:end, :] = block_v
        self.kv_headroom_stats["writes"] += 1
        return cache.keys[..., :end, :], cache.values[..., :end, :]

    # ----- split commit / cached draft (mlx2 external-draft engine) -----

    def commit(self, context, cache):
        """Write target context rows into persistent draft KV (no queries)."""
        if self.window and context.shape[1] > self.window:
            skip = context.shape[1] - self.window
            context = context[:, skip:]
            cache.offset += skip
        offset = cache.offset
        k = self.k_norm_ctx(self._heads(self.k_proj_ctx(context), self.kv_heads))
        v = self._heads(self.v_proj_ctx(context), self.kv_heads)
        cache.update_and_fetch(self.rope(k, offset=offset), v)

    def _cached_kv(self, cache):
        if isinstance(cache, RotatingKVCache):
            k, v = cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
            return k[..., -self.window :, :], v[..., -self.window :, :]
        return cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :]

    def attend_cached(self, x, cache):
        """The block's attention over already committed draft KV."""
        offset = cache.offset
        k, v = self._cached_kv(cache)
        q = self.q_norm(self._heads(self.q_proj(x), self.heads))
        block_k = self.k_norm(self._heads(self.k_proj(x), self.kv_heads))
        k, v = self._with_block(
            cache, k, v, self.rope(block_k, offset=offset), self._heads(self.v_proj(x), self.kv_heads)
        )
        out = mx.fast.scaled_dot_product_attention(
            self.rope(q, offset=offset), k, v, scale=self.head_dim**-0.5, mask=None
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(*x.shape[:2], -1))


class DraftMLP(GatedMLP):
    """vLLM's fp32 activation law; math: technologies.drafter_blocks."""

    def __init__(self, config):
        super().__init__(
            config.hidden_size, config.intermediate_size, GatedMLPSpec(gate_law="silu_mul_fp32")
        )


class DraftLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        for name in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            self[name] = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DraftAttention(config, index in config.local_layer_ids)
        self.mlp = DraftMLP(config)

    def __call__(self, x, residual, context, cache):
        # Fused add/RMSNorm normalizes the unrounded sum, stores BF16 residuals.
        summed = x if residual is None else x.astype(mx.float32) + residual.astype(mx.float32)
        residual = summed.astype(x.dtype)
        x = self.self_attn(self.input_layernorm(summed).astype(x.dtype), context, cache)
        summed = x.astype(mx.float32) + residual.astype(mx.float32)
        return self.mlp(self.post_attention_layernorm(summed).astype(x.dtype)), summed.astype(x.dtype)


class MarkovHead(nn.Module):
    """``rnn_hidden`` head: gated recurrence whose latent is folded into the
    target LM head once (FP64, 4096-row vocabulary tiles)."""

    FOLD_TILE = 4096

    def __init__(self, config):
        super().__init__()
        rank, hidden = config.markov_rank, config.hidden_size
        self.markov_w1 = nn.Embedding(config.vocab_size, rank)
        self.rnn_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank)
        self.hidden_out = nn.Linear(rank, hidden)
        self.dflare_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self._fold_weight = self._fold_bias = None
        # Opt-in (inkling2_dspark_fold_q8): the folded [V, rank] matrix as
        # MLX affine 8-bit, (weight, scales, biases, group_size); see set_fold_q8.
        self.fold_q8 = False
        self._fold_q = None

    def fold(self, lm_head):
        projection = np.array(self.hidden_out.weight.astype(mx.float32)).astype(np.float64)
        bias = np.array(self.hidden_out.bias.astype(mx.float32)).astype(np.float64)
        weights, biases = [], []
        dtype = self.hidden_out.weight.dtype
        for start in range(0, lm_head.weight.shape[0], self.FOLD_TILE):
            rows = slice(start, start + self.FOLD_TILE)
            weight = lm_head.weight[rows]
            if isinstance(lm_head, nn.QuantizedLinear):
                weight = mx.dequantize(
                    weight,
                    lm_head.scales[rows],
                    lm_head.biases[rows] if "biases" in lm_head else None,
                    group_size=lm_head.group_size,
                    bits=lm_head.bits,
                    mode=lm_head.mode,
                )
            head = np.array(weight.astype(mx.float32)).astype(np.float64)
            weights.append(mx.array((head @ projection).astype(np.float32)).astype(dtype))
            biases.append(mx.array((head @ bias).astype(np.float32)).astype(dtype))
        self._fold_weight = mx.concatenate(weights)
        self._fold_bias = mx.concatenate(biases)
        mx.eval(self._fold_weight, self._fold_bias)
        self._fold_q = None
        if self.fold_q8:
            self._quantize_fold()

    @staticmethod
    def fold_q8_group_size(rank):
        for group in (64, 32):
            if rank % group == 0:
                return group
        raise ValueError(f"Inkling 2 DSpark fold_q8 needs a markov_rank divisible by 32 (got {rank})")

    def _quantize_fold(self):
        group = self.fold_q8_group_size(self._fold_weight.shape[1])
        weight, scales, biases = mx.quantize(self._fold_weight, group_size=group, bits=8)
        mx.eval(weight, scales, biases)
        self._fold_q = (weight, scales, biases, group)

    def set_fold_q8(self, enabled):
        """Opt-in 8-bit copy of the folded ``lm_head @ hidden_out`` matrix
        (mechanism from g13-engine's ``attach_fold``, docs/PROVENANCE.md):
        each Markov step reads ~half the bytes of the BF16 fold.  It changes
        draft logits, so proposals may differ; the target still verifies
        every proposal.  The BF16 fold is kept, so switching back is exact."""
        if enabled:
            self.fold_q8_group_size(self.markov_w1.weight.shape[1])  # refuse before changing state
        self.fold_q8 = bool(enabled)
        if not self.fold_q8:
            self._fold_q = None
        elif self._fold_weight is not None and self._fold_q is None:
            self._quantize_fold()

    def context(self, hidden, f_rnn):
        rank = self.markov_w1.weight.shape[1]
        inputs = mx.concatenate([hidden, mx.broadcast_to(f_rnn, hidden.shape)], axis=-1)
        return inputs.astype(mx.float32) @ self.rnn_proj.weight[:, 2 * rank :].astype(
            mx.float32
        ).T + self.rnn_proj.bias.astype(mx.float32)

    def step(self, state, previous, context):
        rank = state.shape[-1]
        inputs = mx.concatenate([state, self.markov_w1(previous)], axis=-1)
        raw = (
            inputs.astype(mx.float32)
            @ self.rnn_proj.weight[:, : 2 * rank].astype(mx.float32).T
            + context
        ).astype(state.dtype)
        gate, candidate, output = mx.split(raw, 3, axis=-1)
        gate = mx.sigmoid(gate.astype(mx.float32)).astype(state.dtype)
        candidate = mx.tanh(candidate.astype(mx.float32)).astype(state.dtype)
        state = gate * state + (1 - gate) * candidate
        latent = mx.tanh(output.astype(mx.float32)).astype(state.dtype)
        if self._fold_q is not None:
            weight, scales, biases, group = self._fold_q
            bias = self._fold_bias + mx.quantized_matmul(
                latent, weight, scales, biases, transpose=True, group_size=group, bits=8
            )
        else:
            bias = mx.addmm(self._fold_bias, latent, self._fold_weight.T)
        return state, bias


class Inkling2DSparkDraftModel(nn.Module):
    receipt_kind = "external_inkling2_dspark"
    prefer_requested_block_size = True
    proposal_distribution = "deterministic_point_mass"
    supports_logits_processors = False

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.markov_head = MarkovHead(config)
        self.fuse_weight = mx.zeros((config.num_hidden_layers + 1, len(config.target_layer_ids)))
        self.mask_embed = mx.zeros((config.hidden_size,))
        self._target = None
        self._fold_source = None
        self.accept_lens, self.draft_lens = [], []

    # ----- binding -----

    @property
    def feature_width(self):
        return self.config.feature_width

    @property
    def receipt_settings(self):
        c = self.config
        options = {name: True for name, value in self.option_receipt.items() if value["selected"]}
        return {**options,
            "kind": "inkling2_dspark", "law_version": LAW_VERSION,
            "trained_proposals": c.block_size - 1, "verification_width": c.block_size,
            "block_size": c.block_size, "target_layer_ids": list(c.target_layer_ids),
            "local_layer_ids": list(c.local_layer_ids), "prefix_window": c.sliding_window_size,
            "markov_head_type": "rnn_hidden", "markov_rank": c.markov_rank,
            "head_fold": "fp64-numpy-4096-row-tiles"
            + ("+mlx-affine-q8" if self.markov_head.fold_q8 else ""), "draft_logits": "activation-dtype-softcapped",
            "full_trained_block": True, "sample_from_anchor": False,
            "proposal_distribution": self.proposal_distribution, "qualified": False,
        }

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "config", None)
        c = self.config
        if (
            args is None
            or getattr(args, "model_type", None) not in TARGET_MODEL_TYPES
            or args.model_type != c.target_model_type
            or args.hidden_size != c.hidden_size
            or args.vocab_size != c.vocab_size
            or args.num_hidden_layers != c.target_num_hidden_layers
            or _loop_dict(getattr(args, "loop_config", None)) != c.target_loop_config
            or len(args.layer_indices) != c.target_visits
            or getattr(args, "unpadded_vocab_size", c.vocab_size) != c.unpadded_vocab_size
            or getattr(args, "final_logit_softcapping", None) != c.final_logit_softcapping
        ):
            raise ValueError("The DSpark head requires its matching Inkling 2 target")
        head = getattr(target, "lm_head", None)
        model = getattr(target, "model", None)
        if (
            head is None
            or model is None
            or not callable(getattr(model, "embed", None))
            or "bias" in head
            or (not isinstance(head, nn.QuantizedLinear)
                and tuple(head.weight.shape) != (c.vocab_size, c.hidden_size))
        ):
            raise ValueError("The DSpark head requires the target's embed and bias-free LM head")
        return target

    def bind(self, target_model):
        target = self.validate_target_compatibility(target_model)
        weight = target.lm_head.weight
        if self._target is not target or self._fold_source is not weight:
            self.markov_head.fold(target.lm_head)
            self._target, self._fold_source = target, weight
        return self

    def configure_options(self, *, kv_headroom=False, fold_q8=False):
        """Opt-in drafter options; each changes draft-side work only, never
        the target or its verification.  ``kv_headroom``: block K/V written
        in place on global layers (bitwise the same attention inputs).
        ``fold_q8``: the Markov head reads an 8-bit fold (drafts may differ)."""
        if type(kv_headroom) is not bool or type(fold_q8) is not bool:
            raise TypeError("Inkling 2 DSpark options must be booleans")
        self.markov_head.set_fold_q8(fold_q8)
        for layer in self.layers:
            layer.self_attn.kv_headroom = kv_headroom
        return self

    @property
    def option_receipt(self):
        """Selected options and the counters that show them engaged."""
        attention = [layer.self_attn for layer in self.layers]
        receipt = {"kv_headroom": {
            "selected": any(a.kv_headroom for a in attention),
            "global_layers": sum(a.window is None for a in attention),
            **{name: sum(a.kv_headroom_stats[name] for a in attention)
               for name in ("writes", "grows", "fallbacks")},
        }}
        head = self.markov_head
        receipt["fold_q8"] = {
            "selected": head.fold_q8,
            "engaged": head._fold_q is not None,
            "group_size": None if head._fold_q is None else head._fold_q[3],
            "bytes": None if head._fold_q is None else sum(a.nbytes for a in head._fold_q[:3]),
            "bf16_bytes": None if head._fold_weight is None else head._fold_weight.nbytes,
        }
        return receipt

    def make_cache(self):
        return [
            KVCache() if layer.self_attn.window is None
            else RotatingKVCache(max_size=layer.self_attn.window, keep=0)
            for layer in self.layers
        ]

    def reset(self, target_model):
        self.bind(target_model)
        self.accept_lens, self.draft_lens = [], []
        return self.make_cache()

    def sanitize(self, weights):
        expected = expected_weights(self.config)
        if set(weights) != set(expected):
            missing = sorted(set(expected) - set(weights))[:8]
            extra = sorted(set(weights) - set(expected))[:8]
            raise ValueError(f"Inkling 2 DSpark weights mismatch (missing={missing}, extra={extra})")
        for name, shape in expected.items():
            if list(weights[name].shape) != shape:
                raise ValueError(f"Inkling 2 DSpark tensor shape mismatch: {name}")
        return dict(weights)

    # ----- drafting -----

    def mix(self, hidden, row, norm):
        features = hidden.reshape(*hidden.shape[:2], -1, self.config.hidden_size).astype(mx.float32)
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(hidden.dtype)[row]
        mixed = mx.zeros_like(features[:, :, 0])
        for i in range(features.shape[2]):
            mixed = mixed + features[:, :, i] * alpha[i].astype(mx.float32)
        return norm(mixed).astype(hidden.dtype)

    def _check_count(self, block_size):
        count = int(block_size) - 1
        if not 0 <= count < self.config.block_size:
            raise ValueError("Requested DSpark block exceeds the trained depth")
        return count

    def _check_inputs(self, hidden, cache, rows):
        if self._target is None:
            raise RuntimeError("Bind the target before drafting")
        if (
            hidden.ndim != 3
            or hidden.shape[0] != rows
            or hidden.shape[1] < 1
            or hidden.shape[2] != self.config.feature_width
        ):
            raise ValueError(
                "Inkling 2 DSpark hidden must be [B, L>=1, len(target_layer_ids)*hidden_size]"
            )
        if len(cache) != len(self.layers):
            raise ValueError("Inkling 2 DSpark cache needs one entry per draft layer")

    def _logits(self, base, bias, dtype):
        logits = base + bias
        cap = self.config.final_logit_softcapping
        if cap:
            logits = (cap * mx.tanh(logits.astype(mx.float32) / cap)).astype(dtype)
        return logits[:, : self.config.unpadded_vocab_size]

    def draft_block(self, last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
        count = self._check_count(block_size)
        anchor = (
            mx.array([last_bonus], token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1)
        )
        if count == 0:
            return mx.zeros((anchor.size, 0), token_dtype)
        self._check_inputs(hidden, cache, anchor.size)
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype),
            (anchor.size, self.config.block_size - 1, embed.shape[-1]),
        )
        # Bidirectional attention needs the full trained block even near EOS/budget.
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            x, residual = layer(x, residual, self.mix(hidden, i, layer.dflare_norm), entry)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(
            embed.dtype
        )[:, 1 : count + 1]
        # Draft logits are in the activation dtype (BF16), unlike the target's FP32 head.
        base = self._target.lm_head(x).astype(x.dtype)
        f_rnn = self.mix(hidden[:, -1:], -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)
        state = mx.zeros((anchor.size, self.config.markov_rank), x.dtype)
        previous, tokens = anchor, []
        for i in range(count):
            state, bias = self.markov_head.step(state, previous, context[:, i])
            previous = sampler(self._logits(base[:, i], bias, x.dtype)).reshape(-1)
            tokens.append(previous[:, None])
        return mx.concatenate(tokens, axis=1).astype(token_dtype)

    def draft_batch_greedy(self, anchors, hidden, caches, block_size, sampler, token_dtype=mx.int32):
        """Ragged rows, each with its own draft cache; one row runs ``draft_block``."""
        count = self._check_count(block_size)
        if not (len(anchors) == len(hidden) == len(caches)) or not hidden:
            raise ValueError("Inkling 2 DSpark batch needs one anchor, hidden and cache per row")
        if count == 0:
            return mx.zeros((len(hidden), 0), token_dtype)
        if len(hidden) == 1:
            return self.draft_block(anchors[0], hidden[0], caches[0], block_size, sampler, token_dtype)
        return self._draft_local_batch(anchors, hidden, caches, block_size, sampler, token_dtype)

    def _draft_local_batch(self, anchors, hidden, caches, block_size, sampler, token_dtype):
        """Batch MLP/head work, with independent attention and Markov state."""
        count = self._check_count(block_size)
        rows = []
        for value in anchors:
            value = mx.array(value, token_dtype).reshape(-1)
            if value.size != 1:
                raise ValueError("Inkling 2 DSpark batch rows take one anchor each")
            rows.append(value)
        anchor = mx.concatenate(rows)
        for h, c in zip(hidden, caches):
            self._check_inputs(h, c, 1)
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype),
            (anchor.size, self.config.block_size - 1, embed.shape[-1]),
        )
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for i, layer in enumerate(self.layers):
            summed = x if residual is None else x.astype(mx.float32) + residual.astype(mx.float32)
            residual = summed.astype(x.dtype)
            normalized = layer.input_layernorm(summed).astype(x.dtype)
            x = mx.concatenate([
                layer.self_attn(normalized[row : row + 1], self.mix(h, i, layer.dflare_norm), caches[row][i])
                for row, h in enumerate(hidden)
            ])
            summed = x.astype(mx.float32) + residual.astype(mx.float32)
            normalized = layer.post_attention_layernorm(summed).astype(x.dtype)
            # ``normalized`` already carries ``x.dtype``, the law's output cast.
            x = gated_mlp(
                layer.mlp.mlp_spec,
                lambda t: _rowwise_linear(layer.mlp.gate_proj, t),
                lambda t: _rowwise_linear(layer.mlp.up_proj, t),
                lambda t: _rowwise_linear(layer.mlp.down_proj, t),
                normalized,
            )
            residual = summed.astype(x.dtype)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(
            embed.dtype
        )[:, 1 : count + 1]
        base = _rowwise_linear(self._target.lm_head, x).astype(x.dtype)
        f_rnn = [self.mix(h[:, -1:], -1, self.markov_head.dflare_norm) for h in hidden]
        context = mx.concatenate([
            self.markov_head.context(mx.contiguous(x[row : row + 1]), f_rnn[row])
            for row in range(anchor.size)
        ])
        state = mx.zeros((anchor.size, self.config.markov_rank), x.dtype)
        previous, tokens = anchor, []
        for i in range(count):
            parts = [
                self.markov_head.step(state[row : row + 1], previous[row : row + 1], context[row : row + 1, i])
                for row in range(anchor.size)
            ]
            state = mx.concatenate([part[0] for part in parts])
            bias = mx.concatenate([part[1] for part in parts])
            previous = sampler(self._logits(base[:, i], bias, x.dtype)).reshape(-1)
            tokens.append(previous[:, None])
        return mx.concatenate(tokens, axis=1).astype(token_dtype)


    # ----- split protocol: commit context, then draft from cached KV -----

    def commit_context(self, hidden, cache):
        """Append ``hidden`` ([1, L, D] target features) to one lane's draft KV."""
        if hidden.shape[1] == 0:
            return
        self._check_inputs(hidden, cache, 1)
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            layer.self_attn.commit(self.mix(hidden, i, layer.dflare_norm), entry)

    def draft_cached(self, anchor, last_feature, cache, block_size, sampler, token_dtype=mx.int32):
        """``draft_block`` after ``commit_context``: the context is already in
        KV; ``last_feature`` ([1, 1, D]) is the last committed feature row.
        ``anchor`` may be a lazy one-token array (a pre-draft's target token)."""
        count = self._check_count(block_size)
        anchor = (
            anchor.reshape(1).astype(token_dtype) if isinstance(anchor, mx.array)
            else mx.array([int(anchor)], token_dtype)
        )
        if count == 0:
            return mx.zeros((1, 0), token_dtype)
        self._check_inputs(last_feature, cache, 1)
        embed = self._target.model.embed(anchor[:, None])
        masks = mx.broadcast_to(
            self.mask_embed.astype(embed.dtype), (1, self.config.block_size - 1, embed.shape[-1])
        )
        x, residual = mx.concatenate([embed, masks], axis=1), None
        for layer, entry in zip(self.layers, cache):
            summed = x if residual is None else x.astype(mx.float32) + residual.astype(mx.float32)
            residual = summed.astype(x.dtype)
            x = layer.self_attn.attend_cached(layer.input_layernorm(summed).astype(x.dtype), entry)
            summed = x.astype(mx.float32) + residual.astype(mx.float32)
            x, residual = layer.mlp(layer.post_attention_layernorm(summed).astype(x.dtype)), summed.astype(x.dtype)
        x = self.norm(x.astype(mx.float32) + residual.astype(mx.float32)).astype(embed.dtype)[:, 1 : count + 1]
        base = self._target.lm_head(x).astype(x.dtype)
        f_rnn = self.mix(last_feature, -1, self.markov_head.dflare_norm)
        context = self.markov_head.context(x, f_rnn)
        state = mx.zeros((1, self.config.markov_rank), x.dtype)
        previous, tokens = anchor, []
        for i in range(count):
            state, bias = self.markov_head.step(state, previous, context[:, i])
            previous = sampler(self._logits(base[:, i], bias, x.dtype)).reshape(-1)
            tokens.append(previous[:, None])
        return mx.concatenate(tokens, axis=1).astype(token_dtype)


Model = Inkling2DSparkDraftModel

__all__ = [
    "Inkling2DSparkConfig", "Inkling2DSparkDraftModel", "LAW_VERSION", "Model",
    "expected_weights",
]
