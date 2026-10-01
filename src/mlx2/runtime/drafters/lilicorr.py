# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# LiLiCorr head adapted from sgl-project/sglang PR #37462 (Apache-2.0).
# MLX implementation; provenance/lilicorr.json.
"""Learned global candidate-lattice correlator with an exact greedy proposal law."""

import mlx.core as mx
import numpy as np
from mlx import nn

from .attention_windows import configure_attention_windows, validate_attention_windows
from .dflash2 import DFlash2DecoderLayer
from .dflash_base import DFlashDraftModel
from .xpress import XPressDraftModel


class LiLiCorrMLP(nn.Module):
    """The exported head uses nongated SiLU, not the backbone's SwiGLU."""

    def __init__(self, inputs, intermediate, outputs):
        super().__init__()
        self.up_proj = nn.Linear(inputs, intermediate, bias=True)
        self.down_proj = nn.Linear(intermediate, outputs, bias=True)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.up_proj(x)))


class LiLiCorrLatticeAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # Preserve exported torch MultiheadAttention parameter names/layout.
        self.in_proj_weight = mx.zeros((3 * hidden_size, hidden_size))
        self.in_proj_bias = mx.zeros((3 * hidden_size,))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, hidden, attention_bias):
        batch, length, width = hidden.shape
        qkv = hidden @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = [
            part.reshape(batch, length, self.num_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
            for part in mx.split(qkv, 3, axis=-1)
        ]
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=attention_bias
        )
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, width))


class LiLiCorrLayer(nn.Module):
    def __init__(self, width, heads, ratio, eps):
        super().__init__()
        self.attn_norm = nn.RMSNorm(width, eps=eps)
        self.attn = LiLiCorrLatticeAttention(width, heads)
        self.mlp_norm = nn.RMSNorm(width, eps=eps)
        self.mlp = LiLiCorrMLP(width, int(width * ratio), width)

    def __call__(self, x, attention_bias):
        x = x + self.attn(self.attn_norm(x), attention_bias)
        return x + self.mlp(self.mlp_norm(x))


def normalized_vectors(x, eps):
    return x / mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), eps)


class LiLiCorrHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_h = config.hidden_size
        h = config.lilicorr_hidden_size or model_h
        self.hidden_size = h
        self.block_size = config.block_size
        self.topk = config.lilicorr_candidate_topk
        self.num_heads = config.lilicorr_num_heads
        self.vector_eps = config.lilicorr_vector_eps
        self.logit_scale = config.lilicorr_logit_scale
        self.token_proj = (
            nn.Identity() if h == model_h else nn.Linear(model_h, h, bias=True)
        )
        self.pass_hidden_proj = nn.Linear(model_h, h, bias=True)
        self.feature_norm = nn.LayerNorm(5, eps=1e-5)
        self.feature_mlp = LiLiCorrMLP(5, h, h)
        self.slot_embedding = mx.zeros((1, 1, self.block_size - 1, 1, h))
        self.rank_embedding = mx.zeros((1, 1, 1, self.topk, h))
        self.relative_slot_bias = mx.zeros((self.num_heads, 2 * self.block_size - 1))
        self.same_slot_bias = mx.zeros((self.num_heads,))
        self.context_proj = nn.Linear(model_h, h, bias=True)
        self.layers = [
            LiLiCorrLayer(
                h, self.num_heads, config.lilicorr_mlp_ratio, config.rms_norm_eps
            )
            for _ in range(config.lilicorr_num_layers)
        ]
        self.output_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.anchor_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.factor_input_proj = nn.Linear(3 * h, h, bias=True)
        self.out_head = nn.Linear(h, config.lilicorr_factor_dim, bias=True)
        self.in_head = nn.Linear(h, config.lilicorr_factor_dim, bias=True)
        self.anchor_out_head = nn.Linear(h, config.lilicorr_factor_dim, bias=True)

    def attention_bias(self, slots):
        if type(slots) is not int or not 1 <= slots < self.block_size:
            raise ValueError("LiLiCorr slots outside trained block")
        slot_ids = mx.repeat(mx.arange(slots), self.topk)
        relative = slot_ids[:, None] - slot_ids[None, :]
        # Offset stays trained block_size-1 even when scoring a shorter prefix.
        return (
            self.relative_slot_bias[:, relative + self.block_size - 1]
            + self.same_slot_bias[:, None, None] * (relative == 0)[None]
        )

    def candidate_features(self, log_probs):
        shape = log_probs.shape
        ranks = mx.arange(self.topk, dtype=mx.float32) / max(self.topk - 1, 1)
        top1 = (mx.arange(self.topk) == 0).astype(mx.float32)
        log_probs = log_probs.astype(mx.float32)
        return mx.stack(
            [
                log_probs,
                mx.exp(log_probs),
                log_probs - mx.max(log_probs, axis=-1, keepdims=True),
                mx.broadcast_to(ranks, shape),
                mx.broadcast_to(top1, shape),
            ],
            axis=-1,
        )

    def score(
        self,
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
    ):
        batch, slots, topk = candidate_log_probs.shape
        if topk != self.topk or not 1 <= slots < self.block_size:
            raise ValueError("LiLiCorr candidate lattice geometry mismatch")
        dtype = self.slot_embedding.dtype
        token_states = self.token_proj(token_embeddings.astype(dtype))
        pass_states = self.pass_hidden_proj(pass_hidden.astype(dtype))[:, :, None]
        x = token_states + pass_states
        features = self.candidate_features(candidate_log_probs).astype(dtype)
        x = x + self.feature_mlp(self.feature_norm(features))
        x = x + self.slot_embedding[:, 0, :slots] + self.rank_embedding[:, 0]
        x = x.reshape(batch, slots * topk, self.hidden_size)
        anchor = self.context_proj(anchor_hidden.astype(dtype)) * anchor_valid[
            :, None
        ].astype(dtype)
        bias = self.attention_bias(slots)[None].astype(dtype)
        for layer in self.layers:
            x = layer(x, bias)
        x = self.output_norm(x).reshape(batch, slots, topk, self.hidden_size)
        anchor = self.anchor_norm(anchor)
        anchor_rows = mx.broadcast_to(anchor[:, None, None], x.shape)
        factor = nn.silu(
            self.factor_input_proj(
                mx.concatenate([x, anchor_rows, x * anchor_rows], axis=-1)
            )
        )
        out_vectors = normalized_vectors(self.out_head(factor), self.vector_eps)
        in_vectors = normalized_vectors(self.in_head(factor), self.vector_eps)
        anchor_out = normalized_vectors(self.anchor_out_head(anchor), self.vector_eps)
        starts = mx.sum(anchor_out[:, None] * in_vectors[:, 0], axis=-1)
        pairs = out_vectors[:, :-1] @ in_vectors[:, 1:].swapaxes(-1, -2)
        return starts, pairs

    def __call__(
        self,
        token_embeddings,
        candidate_log_probs,
        pass_hidden,
        anchor_hidden,
        anchor_valid,
    ):
        start, pairs = self.score(
            token_embeddings,
            candidate_log_probs,
            pass_hidden,
            anchor_hidden,
            anchor_valid,
        )
        start = start.astype(mx.float32) * self.logit_scale
        pairs = pairs.astype(mx.float32) * self.logit_scale
        first = mx.broadcast_to(
            start[:, None, None], (start.shape[0], 1, self.topk, self.topk)
        )
        return mx.concatenate([first, pairs], axis=1)


class LiLiCorrDraftModel(XPressDraftModel):
    """DFlash + learned LiLiCorr head; no XPress or local selector parameters.

    The target draws may be sampled. Drafts themselves are deterministic, and
    their exact point masses are supplied to the existing rejection sampler.
    """

    receipt_kind = "external_lilicorr"
    requires_processor_histories = True
    requires_pending_context_at_prefill = True
    proposal_distribution = "deterministic_point_mass"

    def __init__(self, config, *, draft_attention_windows=None):
        windows = validate_attention_windows(
            draft_attention_windows, config.num_hidden_layers
        )
        if config.conv_kernel_size:
            self.layer_class = DFlash2DecoderLayer
        DFlashDraftModel.__init__(self, config)
        configure_attention_windows(self, windows)
        self.lilicorr = LiLiCorrHead(config)
        self.stats = {"backbone_blocks": 0, "lattice_blocks": 0}

    @property
    def receipt_settings(self):
        return {
            "trained_block_size": self.config.block_size,
            "candidate_topk": self.config.lilicorr_candidate_topk,
            "lattice_layers": self.config.lilicorr_num_layers,
            "conv_kernel_size": self.config.conv_kernel_size,
            "conv_group_size": self.config.conv_group_size,
            "proposal_distribution": self.proposal_distribution,
            "anchor_convention": "normalized_last_committed_target_taps_or_invalid_zero_context",
            "co_trained_head_required": True,
            "draft_attention_windows": None
            if self.draft_attention_windows is None
            else list(self.draft_attention_windows),
        }

    def sanitize(self, weights):
        result = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in result:
                raise ValueError(f"Duplicate LiLiCorr weight after sanitize: {key}")
            result[key] = value
        return result

    def draft_distributions(
        self,
        anchors,
        hidden,
        cache,
        proposal_length,
        rngs,
        temperatures,
        *,
        logits_processors=None,
        processor_histories=None,
    ):
        from ..processor_probe import probe_logits_processors

        if (
            type(proposal_length) is not int
            or not 0 <= proposal_length < self.config.block_size
        ):
            raise ValueError("LiLiCorr proposal length outside trained block")
        anchor_ids = [int(a) for a in anchors]
        batch = len(anchor_ids)
        histories = processor_histories or [[] for _ in range(batch)]
        processors = logits_processors or [[] for _ in range(batch)]
        if (
            not batch
            or hidden.shape[0] != batch
            or len(rngs) != batch
            or len(temperatures) != batch
            or len(histories) != batch
            or len(processors) != batch
            or len(cache) != len(self.layers)
        ):
            raise ValueError("LiLiCorr context/processor/cache batch mismatch")
        if any(not 0 <= a < self.config.vocab_size for a in anchor_ids):
            raise ValueError("LiLiCorr anchor outside target vocabulary")
        if proposal_length == 0:
            self.append_context(hidden, cache)
            self.adaptive_confidence_features = [[] for _ in range(batch)]
            return [[] for _ in range(batch)], [[] for _ in range(batch)]
        if hidden.shape[1]:
            anchor_hidden = self.hidden_norm(self.fc(hidden[:, -1]))
            anchor_valid = mx.ones((batch,), dtype=mx.bool_)
        else:
            first = cache[0]
            entries = first.rows if hasattr(first, "rows") else [first]
            if any(entry.offset > 0 for entry in entries):
                raise ValueError(
                    "LiLiCorr cached context is missing pending anchor features"
                )
            # Match upstream's invalid-context anchor. Learned output biases
            # still participate; no invented target feature or stale cache state.
            anchor_hidden = mx.zeros((batch, self.config.hidden_size))
            anchor_valid = mx.zeros((batch,), dtype=mx.bool_)
        inputs = mx.concatenate(
            [
                mx.array(anchor_ids, dtype=mx.int32)[:, None],
                mx.full(
                    (batch, self.config.block_size - 1),
                    self.config.mask_token_id,
                    dtype=mx.int32,
                ),
            ],
            axis=1,
        )
        features = self._hidden(inputs, hidden, cache)[:, 1:]
        logits = self._logits(features)
        candidates = mx.argsort(-logits, axis=-1)[
            ..., : self.config.lilicorr_candidate_topk
        ]
        log_probs = logits.astype(mx.float32) - mx.logsumexp(
            logits.astype(mx.float32), axis=-1, keepdims=True
        )
        candidate_log_probs = mx.take_along_axis(log_probs, candidates, axis=-1)
        scores = self.lilicorr(
            self.embed_tokens(candidates),
            candidate_log_probs,
            features,
            anchor_hidden,
            anchor_valid,
        )
        hosted = np.asarray(scores.astype(mx.float32))
        ids = np.asarray(candidates)
        if not np.isfinite(hosted).all():
            raise ValueError("LiLiCorr produced nonfinite correlation scores")
        dense = np.asarray(logits.astype(mx.float32)) if any(processors) else None
        self.stats["backbone_blocks"] += batch
        self.stats["lattice_blocks"] += batch
        tokens, laws = [[] for _ in range(batch)], [[] for _ in range(batch)]
        confidence_features = [[] for _ in range(batch)]
        for row in range(batch):
            predecessor_column = 0
            for position in range(proposal_length):
                # These are correlation logits only. Adding backbone unary
                # logits here would implement a different trained head law.
                value = hosted[row, position, predecessor_column].copy()
                probabilities = np.exp(value.astype(np.float64) - np.max(value))
                pmax = float(np.max(probabilities) / np.sum(probabilities))
                pmax = float(np.clip(pmax, 1e-12, 1.0 - 1e-12))
                confidence = float(np.log(pmax) - np.log1p(-pmax))
                if processors[row]:
                    prefix = mx.array(
                        list(histories[row]) + [anchor_ids[row]] + tokens[row],
                        dtype=mx.int32,
                    )
                    processed = np.asarray(
                        probe_logits_processors(
                            processors[row],
                            prefix,
                            mx.array(dense[row, position])[None],
                        )[0].astype(mx.float32)
                    )
                    if np.isnan(processed).any() or np.isposinf(processed).any():
                        raise ValueError("LiLiCorr processor produced invalid logits")
                    value[~np.isfinite(processed[ids[row, position]])] = -np.inf
                    if not np.isfinite(value).any():
                        break
                predecessor_column = int(np.argmax(value))
                token = int(ids[row, position, predecessor_column])
                q = np.zeros(self.config.vocab_size, dtype=np.float64)
                q[token] = 1
                tokens[row].append(token)
                laws[row].append(q)
                confidence_features[row].append(confidence)
        self.adaptive_confidence_features = confidence_features
        return tokens, laws


Model = LiLiCorrDraftModel
