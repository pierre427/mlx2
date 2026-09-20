# SPDX-License-Identifier: MIT
# Mined closure; provenance/muse-dflash2.json.
from collections.abc import Callable, Mapping

import mlx.core as mx
import mlx.nn as nn


from .dflash_base import DFlashDecoderLayer, DFlashDraftModel, append_context_kv

from .dflash2_config import DFlash2Config


def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, base.shape[0], groups, 1)
    output = mx.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = (
            blocks
            if offset == 0
            else mx.concatenate(
                [mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]], axis=1
            )
        )
        kernel = base[offset].reshape(1, 1, groups, group_size).astype(hidden.dtype)
        output = output + (kernel + dynamic[:, :, offset]) * values
    return output.reshape(hidden.shape)


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.group_size = group_size
        groups = hidden_size // group_size
        self.base_kernel = mx.zeros((2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size, 2 * kernel_size * groups, bias=False
        )

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        prepared = _grouped_dynamic_convolve(
            hidden,
            dynamic[..., 0, :, :],
            self.base_kernel[0],
            self.group_size,
        )
        return prepared, dynamic[..., 1, :, :]

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size
        )


class DFlash2DecoderLayer(DFlashDecoderLayer):
    def __init__(self, config: DFlash2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def __call__(self, x, x_ctx, rope, cache):
        residual = x
        x, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = residual + self.attention_conv.finish(
            self.self_attn(x, x_ctx, rope, cache), kernel
        )
        residual = x
        x, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return residual + self.mlp_conv.finish(self.mlp(x), kernel)


class CandidateSelector(nn.Module):
    def __init__(self, config: DFlash2Config):
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(
            config.vocab_size, config.selector_rank
        )
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = nn.Linear(
            config.hidden_size, config.selector_rank, bias=False
        )

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        sampler: Callable[[mx.array], mx.array],
    ) -> mx.array:
        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k :]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        predecessor = anchor_ids.reshape(-1)
        path = []
        # Only the position-keyed draft sampler can score a candidate against
        # the slot the verifier will draw for.  A plain sampler has no notion
        # of position, so it gets the greedy pick rather than a draw that the
        # exact-match verifier would have to guess its way back to.
        sample_candidate = getattr(sampler, "sample_candidate", None)
        vocab_size = self.predecessor_codebook.weight.shape[0]
        for position in range(hidden.shape[1]):
            edges = mx.sum(
                self.predecessor_codebook(predecessor)[:, None]
                * hidden[:, position, None]
                * self.successor_codebook(candidates[:, position]),
                axis=-1,
            )
            scores = unary[:, position] + edges
            selected = (
                sample_candidate(scores, candidates[:, position], vocab_size)
                if callable(sample_candidate)
                else mx.argmax(scores, axis=-1)
            ).reshape(-1)
            predecessor = mx.take_along_axis(
                candidates[:, position], selected[:, None], axis=-1
            )[:, 0]
            path.append(predecessor)
        return mx.stack(path, axis=1)



class DFlash2DraftModel(DFlashDraftModel):
    layer_class = DFlash2DecoderLayer
    prefer_requested_block_size = False
    dflash_initial_block_size = 3
    dflash_min_block_size = 3

    def __init__(self, config: DFlash2Config):
        super().__init__(config)
        self.candidate_selector = CandidateSelector(config)

    def validate_target_compatibility(self, target_model) -> None:
        args = target_model.args
        for field in ("hidden_size", "vocab_size"):
            if getattr(args, field) != getattr(self.config, field):
                raise ValueError(f"DFlash2 target {field} mismatch")
        if args.num_hidden_layers != self.config.num_target_layers:
            raise ValueError("DFlash2 target layer count mismatch")

    def bind(self, target_model) -> "DFlash2DraftModel":
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        return self

    def _embed_input_tokens(self, inputs: mx.array) -> mx.array:
        return (
            self.embed_tokens(inputs)
            * self.embed_scale
            * self.config.input_embedding_scale
        )

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.lm_head(hidden) * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
    ) -> mx.array:
        proposal_length = int(block_size) - 1
        if proposal_length <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        masks = mx.full(
            (anchor.shape[0], proposal_length),
            int(self.config.mask_token_id),
            dtype=token_dtype,
        )
        draft_inputs = mx.concatenate([anchor[:, None], masks], axis=1)
        draft_hidden = self._hidden(draft_inputs, hidden, cache)[:, 1:]
        return self.candidate_selector.select(
            draft_hidden,
            self._logits(draft_hidden),
            anchor,
            sampler,
        ).astype(token_dtype)

    def sanitize(self, weights: Mapping[str, mx.array]) -> dict[str, mx.array]:
        normalized = {}
        codebooks = {
            "candidate_selector.predecessor_codebook",
            "candidate_selector.successor_codebook",
        }
        for key, value in weights.items():
            key = key.removeprefix("model.")
            if key in codebooks:
                key = f"{key}.weight"
            if key in normalized:
                raise ValueError(
                    f"Duplicate DFlash2 weight key after sanitization: {key}"
                )
            normalized[key] = value
        return normalized


Model = DFlash2DraftModel


__all__ = [
    "CandidateSelector",
    "DFlash2DecoderLayer",
    "DFlash2DraftModel",
    "GroupedDynamicCausalConv",
    "Model",
    "_grouped_dynamic_convolve",
]

# Original mlx2 distribution-aware extension. Selector scores define q.
def _draft_distributions(
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
    """Return tokens and the exact selector laws that produced them.

    The processor-aware path is intentionally opt-in at the call boundary so
    unconstrained requests retain the original selector and sampling path.
    DFlash2 predicts a whole block in one trunk pass, but a grammar law at
    position ``j`` depends on tokens selected at positions ``< j``.  Apply the
    processors and sample sequentially while retaining the one-pass features.
    """
    import numpy as np
    from ..processor_probe import probe_logits_processors
    from ..speculative_sampling import softmax
    anchor_values = [int(value) for value in anchors]
    anchors = mx.array(anchor_values, dtype=mx.int32).reshape(-1)
    batch = anchors.shape[0]
    logits_processors = logits_processors or [[] for _ in range(batch)]
    processor_histories = processor_histories or [[] for _ in range(batch)]
    if len(logits_processors) != batch or len(processor_histories) != batch:
        raise ValueError("DFlash2 processor rows must match the draft batch")
    inputs = mx.concatenate([anchors[:, None], mx.full((batch, proposal_length), self.config.mask_token_id, dtype=mx.int32)], axis=1)
    features = self._hidden(inputs, hidden, cache)[:, 1:]
    logits = self._logits(features)
    selector = self.candidate_selector
    count = min(selector.top_k, logits.shape[-1])
    candidates = mx.argpartition(logits, -count, axis=-1)[..., -count:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(features)
    predecessor = anchors
    predecessor_values = list(anchor_values)
    active = [True] * batch
    tokens = [[] for _ in range(batch)]; laws = [[] for _ in range(batch)]
    for position in range(proposal_length):
        edges = mx.sum(selector.predecessor_codebook(predecessor)[:, None] * projected[:, position, None] * selector.successor_codebook(candidates[:, position]), axis=-1)
        hosted_edges = np.asarray(edges.astype(mx.float32))
        scores = np.asarray((unary[:, position] + edges).astype(mx.float32))
        dense_scores = np.asarray(logits[:, position].astype(mx.float32))
        ids = np.asarray(candidates[:, position])
        selected = []
        for row in range(batch):
            if not active[row]:
                selected.append(predecessor_values[row])
                continue
            processors = logits_processors[row]
            if processors:
                # Give grammar/logit-order probes the dense draft row.  The
                # selector edge is then added only to its top-k candidates.
                value = mx.array(dense_scores[row])[None]
                prefix = mx.array(
                    processor_histories[row]
                    + [anchor_values[row]]
                    + tokens[row],
                    dtype=mx.int32,
                )
                value = probe_logits_processors(processors, prefix, value)
                candidate_scores = (
                    np.asarray(value[0].astype(mx.float32))[ids[row]]
                    + hosted_edges[row]
                )
                if not np.isfinite(candidate_scores).any():
                    # The target still has a legal dense continuation, but the
                    # selector cannot propose it. End this lane's draft here;
                    # this is a round-local miss, not a drafter outage.
                    active[row] = False
                    selected.append(predecessor_values[row])
                    continue
                try:
                    q = softmax(
                        candidate_scores,
                        temperatures[row],
                    )
                except ValueError as error:
                    raise RuntimeError("finite DFlash2 candidates produced no law") from error
                column = rngs[row].sample(q)
                token = int(ids[row, column])
                full_q = np.zeros(self.config.vocab_size, dtype=np.float64)
                full_q[ids[row]] = q
                q = full_q
            else:
                restricted = softmax(scores[row], temperatures[row])
                column = rngs[row].sample(restricted)
                token = int(ids[row, column])
                q = np.zeros(self.config.vocab_size, dtype=np.float64)
                q[ids[row]] = restricted
            selected.append(token)
            tokens[row].append(token); laws[row].append(q)
        predecessor_values = selected
        predecessor = mx.array(selected, dtype=mx.int32)
    return tokens, laws


def _append_context(self, hidden, cache):
    """Project only committed target context; no draft block or vocabulary head."""
    context = self.hidden_norm(self.fc(hidden))
    for layer, entry in zip(self.layers, cache):
        attn = layer.self_attn
        batch, length, _ = context.shape
        keys = attn.k_norm(attn.k_proj(context).reshape(batch, length, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
        values = attn.v_proj(context).reshape(batch, length, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=entry.offset)
        append_context_kv(entry, keys, values)
        if hasattr(entry, "max_size") and entry.keys.shape[2] > entry.max_size:
            entry.keys = entry.keys[:, :, -entry.max_size:]
            entry.values = entry.values[:, :, -entry.max_size:]
            entry._idx = entry.max_size

DFlash2DraftModel.draft_distributions = _draft_distributions
DFlash2DraftModel.append_context = _append_context


def _batch_caches(self, rows):
    from types import SimpleNamespace
    if not rows or any(len(row) != len(self.layers) for row in rows):
        raise ValueError("DFlash2 cache row topology mismatch")
    return [SimpleNamespace(rows=[row[layer] for row in rows]) for layer in range(len(self.layers))]

DFlash2DraftModel.batch_caches = _batch_caches


def _make_serializable_cache(self):
    caches = super(DFlash2DraftModel, self).make_cache()
    # Empty rotating planes must be serializable at a one-token prompt's
    # first committed boundary; None-backed rings have no .state shape.
    for cache in caches:
        cache.keys = mx.zeros((1,self.config.num_key_value_heads,0,self.config.head_dim))
        cache.values = mx.zeros_like(cache.keys)
    return caches

DFlash2DraftModel.make_cache = _make_serializable_cache


# Batched pairwise selection (Splash-derived design; see
# provenance/splash-02-dflash-pair-select.json).  The sequential path above
# reads edges back to the host once per position.  Every edge the walk can
# need is known after one trunk pass: position k's predecessor is one of
# position k-1's C candidates (the anchor at k=0), so the whole block has a
# [B, K, C, C] score table and the walk is K gathers over it.
def pairwise_score_table(self, anchors, features, candidates, unary):
    """Return ``[B, K, C, C]`` f32 scores: ``[b, k, j, i]`` is candidate ``i``
    at position ``k`` after predecessor candidate ``j`` (row-constant at k=0).

    Mirrors the sequential expression and dtype exactly, so each entry equals
    the value the host path would score for that predecessor.
    """
    selector = self.candidate_selector
    batch, _, count = candidates.shape
    predecessors = mx.concatenate(
        [
            mx.broadcast_to(anchors.reshape(batch, 1, 1), (batch, 1, count)),
            candidates[:, :-1],
        ],
        axis=1,
    )
    projected = selector.hidden_projection(features)
    edges = mx.sum(
        selector.predecessor_codebook(predecessors)[:, :, :, None]
        * projected[:, :, None, None]
        * selector.successor_codebook(candidates)[:, :, None],
        axis=-1,
    )
    return (unary[:, :, None, :] + edges).astype(mx.float32)


def pairwise_walk(candidates, scores, uniforms, temperatures):
    """Sample a whole block from a pair-score table with pre-drawn uniforms.

    Pure mx ops (the reference any fused kernel must match).  Per row and
    position this reproduces ``RequestRNG.sample(softmax(scores, t))``:
    inverse CDF with ``searchsorted(side="right")``; ``t == 0`` is the argmax
    one-hot law and still consumes its uniform.  Returns ``(tokens [B,K]
    int32, cand_q [B,K,C] f32, invalid [B] bool)``; ``invalid`` marks rows
    whose walked scores the sequential softmax would reject.
    """
    batch, length, count = candidates.shape
    temperatures = mx.array(temperatures, dtype=mx.float32).reshape(batch, 1)
    greedy = temperatures == 0
    safe = mx.where(greedy, mx.ones_like(temperatures), temperatures)
    # f32 rounding may lift u < 1 to exactly 1.0; keep it inside the CDF.
    uniforms = mx.minimum(
        mx.array(uniforms, dtype=mx.float32).reshape(batch, length),
        mx.array(1.0 - 2.0**-24, dtype=mx.float32),
    )
    columns = mx.arange(count, dtype=mx.int32)[None]
    selected = mx.zeros((batch,), dtype=mx.int32)
    tokens, laws = [], []
    invalid = mx.zeros((batch,), dtype=mx.bool_)
    for position in range(length):
        row = mx.take_along_axis(
            scores[:, position],
            mx.broadcast_to(selected[:, None, None], (batch, 1, count)),
            axis=1,
        )[:, 0]
        invalid = invalid | mx.any(mx.isnan(row), axis=-1) | ~mx.any(
            mx.isfinite(row), axis=-1
        )
        best = mx.argmax(row, axis=-1).astype(mx.int32)
        one_hot = (columns == best[:, None]).astype(mx.float32)
        x = row / safe
        weights = mx.exp(x - mx.max(x, axis=-1, keepdims=True))
        q = weights / mx.sum(weights, axis=-1, keepdims=True)
        cdf = mx.cumsum(q, axis=-1)
        cdf = cdf / cdf[:, -1:]
        drawn = mx.minimum(
            mx.sum(cdf <= uniforms[:, position, None], axis=-1).astype(mx.int32),
            count - 1,
        )
        selected = mx.where(greedy[:, 0], best, drawn)
        laws.append(mx.where(greedy, one_hot, q))
        tokens.append(
            mx.take_along_axis(candidates[:, position], selected[:, None], axis=-1)[
                :, 0
            ]
        )
    return mx.stack(tokens, axis=1), mx.stack(laws, axis=1), invalid


def _propose_block(self, anchors, hidden, cache, proposal_length, uniforms, temperatures):
    """P3: one trunk pass, one pair table, one walk, one host read.

    ``uniforms`` is ``[B][proposal_length]`` drawn in position order from each
    lane's RequestRNG, the same draws the sequential path consumes.  Rows with
    logits processors must use ``draft_distributions``.
    """
    from ..verify_sync import record_verify_sync
    from .draft_block import DraftBlock

    anchor_values = [int(value) for value in anchors]
    anchors = mx.array(anchor_values, dtype=mx.int32).reshape(-1)
    batch = anchors.shape[0]
    temperatures = [float(value) for value in temperatures]
    if len(temperatures) != batch or len(uniforms) != batch:
        raise ValueError("DFlash2 block rows must match the draft batch")
    if any(value < 0 for value in temperatures):
        raise ValueError("Temperature cannot be negative")
    if any(len(row) != proposal_length for row in uniforms):
        raise ValueError("DFlash2 block needs one uniform per proposed position")
    inputs = mx.concatenate([anchors[:, None], mx.full((batch, proposal_length), self.config.mask_token_id, dtype=mx.int32)], axis=1)
    features = self._hidden(inputs, hidden, cache)[:, 1:]
    logits = self._logits(features)
    count = min(self.candidate_selector.top_k, logits.shape[-1])
    candidates = mx.argpartition(logits, -count, axis=-1)[..., -count:]
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    scores = self.pairwise_score_table(anchors, features, candidates, unary)
    tokens, cand_q, invalid = pairwise_walk(
        candidates, scores, uniforms, temperatures
    )
    candidates = candidates.astype(mx.int32)
    mx.eval(tokens, candidates, cand_q, invalid)
    record_verify_sync("external.draft.block_eval")
    # The sequential softmax rejects the same rows (NaN or no finite score).
    if bool(invalid.any().item()):
        raise ValueError("Invalid selector scores")
    return DraftBlock(tokens, candidates, cand_q, (int(proposal_length),) * batch)


DFlash2DraftModel.pairwise_score_table = pairwise_score_table
DFlash2DraftModel.propose_block = _propose_block
