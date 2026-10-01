# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted to MLX; provenance/xpress.json.
"""One DFlash backbone evaluation followed by deterministic Jacobi refinement.

The proposal distribution is a point mass at this algorithm's output, even
when the *target* is sampled. Finite-pass stochastic Jacobi refinement is not
implemented: final-pass logits would not establish its sampling law.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn

from .attention_windows import (
    compact_window_caches,
    configure_attention_windows,
    make_window_caches,
    validate_attention_windows,
)
from .dflash_base import DFlashDraftModel, append_context_kv


def fold_causal_mixer(raw):
    """Training's residual x + tril(L)x, folded exactly once into I + tril(L)."""
    if len(raw.shape) != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError("XPress mixer must have shape [rank, block, block]")
    return mx.tril(raw) + mx.eye(raw.shape[-1], dtype=raw.dtype)[None]


class XPressRefinerHead(nn.Module):
    def __init__(self, vocab_size, hidden_size, block_size, rank=256, mlp_hidden=512):
        super().__init__()
        self.block_size = block_size
        self.rank = rank
        self.w1 = nn.Embedding(vocab_size, rank)
        self.down_h = nn.Linear(hidden_size, rank, bias=False)
        self.down_g = nn.Linear(hidden_size, rank, bias=False)
        self.in_proj = nn.Linear(3 * rank, rank, bias=False)
        self.mix_L = mx.broadcast_to(
            mx.eye(block_size)[None], (rank, block_size, block_size)
        )
        self.mlp_gate = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(rank, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, rank, bias=False)
        self.w2 = nn.Linear(rank, vocab_size, bias=False)

    def hidden_cache(self, hidden):
        if hidden.shape[1] != self.block_size:
            raise ValueError("XPress refiner requires the full trained block")
        global_hidden = mx.broadcast_to(
            mx.mean(hidden, axis=1, keepdims=True), hidden.shape
        )
        return mx.concatenate(
            [self.down_h(hidden), self.down_g(global_hidden)], axis=-1
        )

    def refine_bias(self, predecessor_ids, hidden_cache):
        x = self.in_proj(
            mx.concatenate([hidden_cache, self.w1(predecessor_ids)], axis=-1)
        )
        # [r,B,B] @ [r,B,N] -> [N,B,r], one causal mixer per rank channel.
        x = (self.mix_L.astype(x.dtype) @ x.transpose(2, 1, 0)).transpose(2, 1, 0)
        x = x + self.mlp_down(nn.silu(self.mlp_gate(x)) * self.mlp_up(x))
        return self.w2(x)

    def jacobi_refine_greedy(
        self, base_logits, hidden, anchors, predecessor_ids, num_passes
    ):
        if type(num_passes) is not int or num_passes <= 0:
            raise ValueError("XPress requires positive Jacobi passes")
        if (
            base_logits.shape[:2] != hidden.shape[:2]
            or hidden.shape[1] != self.block_size
        ):
            raise ValueError("XPress backbone/refiner geometry mismatch")
        hcache = self.hidden_cache(hidden)
        block = mx.concatenate(
            [anchors[:, None], mx.argmax(base_logits[:, 1:], axis=-1)], axis=1
        )
        refined = base_logits
        for _ in range(num_passes):
            # tok_am1 at position zero is the actual committed token immediately
            # before the anchor. It contributes to later positions through L.
            previous = mx.concatenate([predecessor_ids[:, None], block[:, :-1]], axis=1)
            refined = base_logits + self.refine_bias(previous, hcache)
            block = mx.concatenate(
                [anchors[:, None], mx.argmax(refined[:, 1:], axis=-1)], axis=1
            )
        return block[:, 1:], refined[:, 1:]


class XPressDraftModel(DFlashDraftModel):
    receipt_kind = "external_xpress"
    requires_processor_histories = True
    proposal_distribution = "deterministic_point_mass"

    def __init__(self, config, *, draft_attention_windows=None):
        windows = validate_attention_windows(
            draft_attention_windows, config.num_hidden_layers
        )
        super().__init__(config)
        configure_attention_windows(self, windows)
        self.xpress_head = XPressRefinerHead(
            config.vocab_size,
            config.hidden_size,
            config.block_size,
            config.xpress_rank,
            config.xpress_mlp_hidden,
        )
        self.stats = {"backbone_blocks": 0, "jacobi_passes": 0}
        self.adaptive_confidence_features = []

    @property
    def receipt_settings(self):
        return {
            "trained_block_size": self.config.block_size,
            "xpress_num_passes": self.config.xpress_num_passes,
            "xpress_rank": self.config.xpress_rank,
            "proposal_distribution": self.proposal_distribution,
            "predecessor_convention": "committed_history_last_or_anchor_for_empty_history",
            "selector": False,
            "draft_attention_windows": None
            if self.draft_attention_windows is None
            else list(self.draft_attention_windows),
        }

    def sanitize(self, weights):
        mapping = {
            "mix.L": "mix_L",
            "mlp.gate_proj.weight": "mlp_gate.weight",
            "mlp.up_proj.weight": "mlp_up.weight",
            "mlp.down_proj.weight": "mlp_down.weight",
        }
        out = {}
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key.startswith("xpress_head."):
                sub = key[len("xpress_head.") :]
                if sub == "mix.L":
                    value = fold_causal_mixer(value)
                key = "xpress_head." + mapping.get(sub, sub)
            if key in out:
                raise ValueError(f"Duplicate XPress weight after sanitize: {key}")
            out[key] = value
        return out

    def make_cache(self):
        caches = make_window_caches(self)
        if caches is None:
            caches = super().make_cache()
        for entry in caches:
            entry.keys = mx.zeros(
                (1, self.config.num_key_value_heads, 0, self.config.head_dim)
            )
            entry.values = mx.zeros_like(entry.keys)
        return caches

    def batch_caches(self, rows):
        if not rows or any(len(row) != len(self.layers) for row in rows):
            raise ValueError("XPress cache row topology mismatch")
        return [
            SimpleNamespace(rows=[row[i] for row in rows])
            for i in range(len(self.layers))
        ]

    def _hidden(self, inputs, target_hidden, cache):
        hidden = super()._hidden(inputs, target_hidden, cache)
        compact_window_caches(cache)
        return hidden

    def append_context(self, hidden, cache):
        context = self.hidden_norm(self.fc(hidden))
        batch, length, _ = context.shape
        if length == 0:
            return
        for layer, entries in zip(self.layers, cache):
            attn = layer.self_attn
            keys = attn.k_norm(
                attn.k_proj(context).reshape(
                    batch, length, attn.n_kv_heads, attn.head_dim
                )
            ).transpose(0, 2, 1, 3)
            values = (
                attn.v_proj(context)
                .reshape(batch, length, attn.n_kv_heads, attn.head_dim)
                .transpose(0, 2, 1, 3)
            )
            if hasattr(entries, "rows"):
                for row, entry in enumerate(entries.rows):
                    append_context_kv(
                        entry,
                        self.rope(keys[row : row + 1], offset=entry.offset),
                        values[row : row + 1],
                    )
            else:
                append_context_kv(
                    entries, self.rope(keys, offset=entries.offset), values
                )
        compact_window_caches(cache)

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
        predecessor_ids=None,
    ):
        from ..processor_probe import probe_logits_processors

        if (
            type(proposal_length) is not int
            or not 0 <= proposal_length < self.config.block_size
        ):
            raise ValueError("XPress proposal length must fit the trained block")
        anchor_values = [int(a) for a in anchors]
        batch = len(anchor_values)
        if not batch or len(rngs) != batch or len(temperatures) != batch:
            raise ValueError("XPress draft rows must match batch")
        histories = processor_histories
        if predecessor_ids is None:
            if histories is None or len(histories) != batch:
                raise ValueError("XPress requires committed predecessor histories")
            predecessor_ids = [
                int(history[-1]) if history else anchor_values[row]
                for row, history in enumerate(histories)
            ]
        else:
            predecessor_ids = [int(a) for a in predecessor_ids]
        histories = histories or [[] for _ in range(batch)]
        processors = logits_processors or [[] for _ in range(batch)]
        if (
            len(predecessor_ids) != batch
            or len(histories) != batch
            or len(processors) != batch
        ):
            raise ValueError("XPress predecessor/processor rows must match batch")
        if any(
            not 0 <= token < self.config.vocab_size
            for token in anchor_values + predecessor_ids
        ):
            raise ValueError("XPress anchor/predecessor outside target vocabulary")
        if hidden.shape[0] != batch or len(cache) != len(self.layers):
            raise ValueError("XPress context/cache batch mismatch")
        if proposal_length == 0:
            self.adaptive_confidence_features = [[] for _ in range(batch)]
            self.append_context(hidden, cache)
            return [[] for _ in range(batch)], [[] for _ in range(batch)]
        anchor_array = mx.array(anchor_values, dtype=mx.int32)
        # Never shorten this input. Both noncausal backbone attention and global
        # mean/mixer features depend on the full checkpoint-trained block.
        inputs = mx.concatenate(
            [
                anchor_array[:, None],
                mx.full(
                    (batch, self.config.block_size - 1),
                    self.config.mask_token_id,
                    dtype=mx.int32,
                ),
            ],
            axis=1,
        )
        features = self._hidden(inputs, hidden, cache)
        base = self._logits(features)
        proposals, refined = self.xpress_head.jacobi_refine_greedy(
            base,
            features,
            anchor_array,
            mx.array(predecessor_ids, dtype=mx.int32),
            self.config.xpress_num_passes,
        )
        dense = np.asarray(refined.astype(mx.float32))
        if not np.isfinite(dense).all():
            raise ValueError("XPress produced nonfinite refined logits")
        # This is a calibrated confidence proxy, separate from the actual
        # point-mass proposal law. The block is deterministic before target
        # verification, so using its confidence to set depth does not select
        # on a random draft draw when the target uses sampling.
        from ..acceptance_estimator import proposal_feature

        scores = dense.astype(np.float64)
        exponent = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities = exponent / exponent.sum(axis=-1, keepdims=True)
        self.adaptive_confidence_features = [
            [proposal_feature(law) for law in row[:proposal_length]]
            for row in probabilities
        ]
        chosen = np.asarray(proposals)
        self.stats["backbone_blocks"] += batch
        self.stats["jacobi_passes"] += batch * self.config.xpress_num_passes
        tokens, laws = [[] for _ in range(batch)], [[] for _ in range(batch)]
        for row in range(batch):
            for position in range(proposal_length):
                token = int(chosen[row, position])
                if processors[row]:
                    prefix = mx.array(
                        list(histories[row]) + [anchor_values[row]] + tokens[row],
                        dtype=mx.int32,
                    )
                    value = probe_logits_processors(
                        processors[row], prefix, mx.array(dense[row, position])[None]
                    )[0]
                    hosted = np.asarray(value.astype(mx.float32))
                    if np.isnan(hosted).any() or np.isposinf(hosted).any():
                        raise ValueError("XPress processor produced invalid logits")
                    if not np.isfinite(hosted).any():
                        break
                    token = int(np.argmax(hosted))
                q = np.zeros(self.config.vocab_size, dtype=np.float64)
                q[token] = 1.0
                tokens[row].append(token)
                laws[row].append(q)
        return tokens, laws


Model = XPressDraftModel
