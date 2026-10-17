# SPDX-License-Identifier: Apache-2.0
"""Super VL DSpark: causal proposal block, Markov bias and trained confidence.

Source mechanisms: vllm-project/speculators; provenance/nemotron35-super-vl.json.
Existing mlx2 DFlash owns KV/context/batching. Drafts use an exact deterministic
point mass; the ordinary target still performs every rejection/commit decision.
"""
import mlx.core as mx
import numpy as np
from mlx import nn

from .attention_windows import configure_attention_windows, validate_attention_windows
from .dflash_base import (
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashDraftModel,
    append_context_kv,
)
from .xpress import XPressDraftModel


def proposal_mask(context_length, proposal_length):
    # Source window is anchored at block start. The prefix stays visible for
    # every query; only synthetic proposal keys use a causal triangular mask.
    return mx.concatenate([mx.ones((proposal_length, context_length), dtype=mx.bool_),
                           mx.tril(mx.ones((proposal_length, proposal_length), dtype=mx.bool_))], axis=1)


class DSparkAttention(DFlashAttention):
    def __call__(self, x, x_ctx, rope, cache):
        batch, length, _ = x.shape
        context_length = x_ctx.shape[1]
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        ck = self.k_norm(self.k_proj(x_ctx).reshape(batch, context_length, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        cv = self.v_proj(x_ctx).reshape(batch, context_length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        pk = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        pv = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        entries = cache.rows if hasattr(cache, "rows") else [cache]
        if len(entries) != batch:
            raise ValueError("DSpark attention cache row mismatch")
        outputs = []
        for row, entry in enumerate(entries):
            query = rope(q[row:row + 1], offset=entry.offset + context_length)
            proposal_keys = rope(pk[row:row + 1], offset=entry.offset + context_length)
            keys, values = append_context_kv(entry, rope(ck[row:row + 1], offset=entry.offset), cv[row:row + 1])
            window = self.sliding_window - 1
            if keys.shape[2] > window:
                keys, values = keys[:, :, -window:], values[:, :, -window:]
                entry.keys, entry.values, entry._idx = keys, values, keys.shape[2]
            mask = proposal_mask(keys.shape[2], length)
            outputs.append(mx.fast.scaled_dot_product_attention(query,
                mx.concatenate([keys, proposal_keys], axis=2),
                mx.concatenate([values, pv[row:row + 1]], axis=2), scale=self.scale, mask=mask))
        output = mx.concatenate(outputs, axis=0).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class DSparkLayer(DFlashDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = DSparkAttention(config, layer_idx)


class MarkovHead(nn.Module):
    def __init__(self, vocabulary, rank):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocabulary, rank)
        self.markov_w2 = nn.Linear(rank, vocabulary, bias=False)

    def __call__(self, previous):
        return self.markov_w2(self.markov_w1(previous))


class ConfidenceHead(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.proj = nn.Linear(width + rank, 1, bias=True)

    def __call__(self, hidden, previous_embedding):
        return self.proj(mx.concatenate([hidden, previous_embedding], axis=-1)).squeeze(-1).astype(mx.float32)


class DSparkDraftModel(XPressDraftModel):
    layer_class = DSparkLayer
    receipt_kind = "external_dspark"
    requires_processor_histories = True
    requires_pending_context_at_prefill = True
    proposal_distribution = "deterministic_point_mass"
    supports_ordered_alternatives = True

    def __init__(self, config):
        DFlashDraftModel.__init__(self, config)
        windows = validate_attention_windows([config.sliding_window] * config.num_hidden_layers, config.num_hidden_layers)
        configure_attention_windows(self, windows)
        self.markov_head = MarkovHead(config.vocab_size, config.markov_rank)
        self.confidence_head = ConfidenceHead(config.hidden_size, config.markov_rank)
        self.stats = {"backbone_blocks": 0, "markov_steps": 0}
        self.adaptive_confidence_features = []
        self._ordered_alternative_context = None

    @property
    def receipt_settings(self):
        return {"trained_proposals": self.config.block_size - 1, "verification_width": self.config.block_size,
                "sample_from_anchor": True, "proposal_attention": "causal",
                "prefix_window": self.config.sliding_window, "markov_rank": self.config.markov_rank,
                "confidence": "trained-logit-with-Markov-features", "proposal_distribution": self.proposal_distribution,
                "runtime_weight_quantization": getattr(self, "runtime_weight_quantization", None),
                "qualified": False}

    def sanitize(self, weights):
        return dict(weights)

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures,
                            *, logits_processors=None, processor_histories=None, **kwargs):
        from ..processor_probe import probe_logits_processors
        batch = len(anchors)
        if (type(proposal_length) is not int or not 0 <= proposal_length < self.config.block_size
                or not batch or len(rngs) != batch or len(temperatures) != batch
                or hidden.shape[0] != batch or len(cache) != len(self.layers)
                or any(type(a) is not int or not 0 <= a < self.config.vocab_size for a in anchors)):
            raise ValueError("DSpark proposal geometry differs")
        histories = processor_histories or [[] for _ in anchors]
        processors = logits_processors or [[] for _ in anchors]
        self._ordered_alternative_context = None
        if len(histories) != batch or len(processors) != batch:
            raise ValueError("DSpark processor histories differ")
        if proposal_length == 0:
            self.append_context(hidden, cache)
            self.adaptive_confidence_features = [[] for _ in anchors]
            return [[] for _ in anchors], [[] for _ in anchors]
        inputs = mx.concatenate([mx.array(anchors, mx.int32)[:, None],
                                  mx.full((batch, self.config.block_size - 2), self.config.mask_token_id, mx.int32)], axis=1)
        features = self._hidden(inputs, hidden, cache)
        base = self._logits(features)
        previous, tokens, laws = list(anchors), [[] for _ in anchors], [[] for _ in anchors]
        confidence = [[] for _ in anchors]
        for position in range(proposal_length):
            previous_array = mx.array(previous, mx.int32)
            embedding = self.markov_head.markov_w1(previous_array)
            logits = base[:, position] + self.markov_head.markov_w2(embedding)
            predicted_confidence = np.asarray(self.confidence_head(features[:, position], embedding))
            for row in range(batch):
                value = logits[row:row + 1]
                if processors[row]:
                    prefix = mx.array(list(histories[row]) + [anchors[row]] + tokens[row], mx.int32)
                    value = probe_logits_processors(processors[row], prefix, value)
                dense = np.asarray(value.astype(mx.float32))[0]
                if np.isnan(dense).any() or np.isposinf(dense).any() or not np.isfinite(dense).any():
                    raise ValueError("DSpark produced invalid proposal logits")
                token = int(np.argmax(dense))
                previous[row] = token
                tokens[row].append(token)
                law = np.zeros(self.config.vocab_size, np.float64)
                law[token] = 1.
                laws[row].append(law)
                if not np.isfinite(predicted_confidence[row]):
                    raise ValueError("DSpark confidence head produced nonfinite output")
                confidence[row].append(float(np.clip(predicted_confidence[row], -40., 40.)))
        self.adaptive_confidence_features = confidence
        self.stats["backbone_blocks"] += batch
        self.stats["markov_steps"] += batch * proposal_length
        self._ordered_alternative_context = (
            base,
            tuple(int(anchor) for anchor in anchors),
            bool(any(processors)),
        )
        return tokens, laws

    def ordered_alternative_proposals(
        self,
        primary_tokens,
        *,
        branch_count,
        strategy,
        shared_prefix_tokens,
    ):
        """Return ranked deterministic paths from DSpark's private block law."""

        if type(branch_count) is not int or not 1 <= branch_count <= 4:
            raise ValueError("DSpark alternative branch count must be 1 through 4")
        context = self._ordered_alternative_context
        if context is None:
            raise ValueError("DSpark alternative context is unavailable")
        base, anchors, had_processors = context
        if had_processors:
            raise ValueError(
                "DSpark alternative branches do not support logits processors"
            )
        alternatives = []
        for row, primary in enumerate(primary_tokens):
            primary = list(primary)
            row_prefix = (
                shared_prefix_tokens[row]
                if isinstance(shared_prefix_tokens, (list, tuple))
                else shared_prefix_tokens
            )
            if strategy == "distinct_root":
                divergence = 0
            elif strategy in {
                "shared_prefix",
                "paired_continuations",
                "confidence_tree",
            }:
                divergence = row_prefix
            else:
                raise ValueError("unsupported DSpark alternative branch strategy")
            if (
                type(divergence) is not int
                or not 0 <= divergence < len(primary)
            ):
                raise ValueError(
                    "DSpark alternatives require a divergence inside the proposal"
                )
            previous = anchors[row] if divergence == 0 else primary[divergence - 1]
            logits = base[row, divergence : divergence + 1] + self.markov_head(
                mx.array([previous], mx.int32)
            )
            ranked = np.argsort(
                -np.asarray(logits.astype(mx.float32))[0], kind="stable"
            )[
                : (
                    branch_count + 1
                    if strategy == "confidence_tree"
                    else branch_count
                )
            ]
            paths = [list(primary)] if strategy == "confidence_tree" else []
            parents = []
            for token in ranked[
                : (
                    min(2, branch_count)
                    if strategy == "paired_continuations"
                    else branch_count
                )
            ]:
                if (
                    strategy == "confidence_tree"
                    and int(token) == int(primary[divergence])
                ):
                    continue
                path = list(primary[:divergence]) + [int(token)]
                previous = int(token)
                for position in range(divergence + 1, len(primary)):
                    value = base[row, position : position + 1] + self.markov_head(
                        mx.array([previous], mx.int32)
                    )
                    previous = int(np.argmax(np.asarray(value.astype(mx.float32))[0]))
                    path.append(previous)
                paths.append(path)
                parents.append((int(token), path))
                if strategy == "confidence_tree" and len(paths) == branch_count:
                    break
            if strategy == "paired_continuations":
                if divergence + 1 >= len(primary):
                    raise ValueError(
                        "paired DSpark continuations require two suffix positions"
                    )
                for parent_token, parent_path in parents[
                    : min(2, max(0, branch_count - 2))
                ]:
                    value = base[row, divergence + 1 : divergence + 2] + self.markov_head(
                        mx.array([parent_token], mx.int32)
                    )
                    continuation = np.argsort(
                        -np.asarray(value.astype(mx.float32))[0], kind="stable"
                    )
                    previous = int(continuation[1])
                    path = list(parent_path[: divergence + 1]) + [previous]
                    for position in range(divergence + 2, len(primary)):
                        value = base[row, position : position + 1] + self.markov_head(
                            mx.array([previous], mx.int32)
                        )
                        previous = int(
                            np.argmax(np.asarray(value.astype(mx.float32))[0])
                        )
                        path.append(previous)
                    paths.append(path)
            paths = paths[:branch_count]
            if not paths or paths[0] != primary:
                raise RuntimeError("DSpark ordered B1 path differs from primary")
            alternatives.append(paths)
        return alternatives
