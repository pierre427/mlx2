# SPDX-License-Identifier: Apache-2.0
"""DSpark: one DFlash backbone block, a chained Markov bias and trained confidence.

Source mechanisms: vllm-project/speculators (provenance/nemotron35-super-vl.json)
and the llama.cpp ``draft-dspark`` block contract for DFlash-format Qwen3
heads (provenance/qwen38-dspark-drafter.json).  Existing mlx2 DFlash owns
KV/context/batching.  Drafts use an exact deterministic point mass; the
ordinary target still performs every rejection/commit decision.

Two proposal-attention variants share every head and proposal law:

* ``causal`` (speculators/RedHat Super VL): a sliding prefix window and a
  causal triangle over the proposal rows (:class:`DSparkDraftModel`).
* ``bidirectional`` (DFlash-format Qwen3 heads, e.g. DimInfer Qwen3.8-27B):
  stock DFlash full attention with a non-causal proposal block
  (:class:`BidirectionalDSparkDraftModel`).

Both sample from the anchor: block row ``i`` (row 0 is the anchor) proposes
token ``i + 1`` with logits ``base[i] + markov_w2(markov_w1(previous))``.
"""
import heapq

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


def best_first_pair_tree(candidates, unary, root_bias, pair, *, children=4, max_nodes=15):
    """Best-first tree over a per-depth candidate lattice with pairwise edges.

    ``candidates``/``unary`` are ``[depth, K]`` token ids and parent-free
    logits; ``root_bias[k]`` is the edge from the anchor to depth-0 candidate
    ``k`` and ``pair[d - 1][k, j]`` the edge from depth ``d - 1`` candidate
    ``j`` to depth ``d`` candidate ``k``.  A node's score is its cumulative
    log-softmax over the K children of its parent.  Returns ``(tokens,
    parents)`` with parents preceding children and ``-1`` for the anchor's
    children; non-finite (forbidden) children are never proposed.
    """

    depth_count = int(candidates.shape[0])

    def expand(scores):
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        top = scores[finite].max()
        shifted = np.where(finite, scores - top, -np.inf)
        logsum = np.log(np.exp(shifted[finite]).sum())
        ranked = np.argsort(-shifted, kind="stable")[: int(children)]
        return [(int(index), float(shifted[index] - logsum)) for index in ranked if finite[index]]

    heap = []
    for index, score in expand(unary[0] + root_bias):
        heapq.heappush(heap, (-score, len(heap), -1, index, 0))
    tokens, parents, order = [], [], len(heap)
    while heap and len(tokens) < int(max_nodes):
        negative, _, parent, index, depth = heapq.heappop(heap)
        tokens.append(int(candidates[depth][index]))
        parents.append(parent)
        me = len(tokens) - 1
        if depth + 1 >= depth_count:
            continue
        for child, score in expand(unary[depth + 1] + pair[depth][:, index]):
            order += 1
            heapq.heappush(heap, (negative - score, order, me, child, depth + 1))
    return tokens, parents


class DSparkDraftModel(XPressDraftModel):
    """DSpark with causal proposal attention and a sliding prefix window."""

    layer_class = DSparkLayer
    receipt_kind = "external_dspark"
    requires_processor_histories = True
    requires_pending_context_at_prefill = True
    proposal_distribution = "deterministic_point_mass"
    supports_ordered_alternatives = True
    # Candidate width per lattice depth and children per expansion of the
    # best-first tree; the chain's greedy token is always candidate zero.
    tree_candidates = 8
    tree_children = 4

    def __init__(self, config):
        DFlashDraftModel.__init__(self, config)
        configure_attention_windows(self, self._attention_windows(config))
        self.markov_head = MarkovHead(config.vocab_size, config.markov_rank)
        self.confidence_head = ConfidenceHead(config.hidden_size, config.markov_rank)
        self.stats = {"backbone_blocks": 0, "markov_steps": 0, "tree_lattices": 0}
        self.adaptive_confidence_features = []
        self._ordered_alternative_context = None

    @staticmethod
    def _attention_windows(config):
        return validate_attention_windows([config.sliding_window] * config.num_hidden_layers, config.num_hidden_layers)

    @property
    def proposal_rows(self):
        """Block rows (anchor first); every row proposes, so also the depth."""
        return self.config.block_size - 1

    @property
    def receipt_settings(self):
        return {"trained_proposals": self.proposal_rows, "verification_width": self.config.block_size,
                "sample_from_anchor": True, "proposal_attention": self.config.proposal_attention,
                "prefix_window": self.config.sliding_window, "markov_rank": self.config.markov_rank,
                "confidence": "trained-logit-with-Markov-features", "proposal_distribution": self.proposal_distribution,
                "runtime_weight_quantization": getattr(self, "runtime_weight_quantization", None),
                "qualified": False}

    def validate_target_compatibility(self, target_model):
        # Hybrid targets nest text geometry; they expose it here (mlx2).
        args = getattr(target_model, "speculative_args", None) or getattr(target_model, "args", None)
        if args is None:
            return
        for field in ("hidden_size", "vocab_size"):
            value = getattr(args, field, None)
            if value is not None and value != getattr(self.config, field):
                raise ValueError(f"DSpark target {field} mismatch")
        layers = getattr(args, "num_hidden_layers", None)
        if layers is not None and layers != self.config.num_target_layers:
            raise ValueError("DSpark target layer count mismatch")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        return super().bind(target_model)

    def sanitize(self, weights):
        return dict(weights)

    def _block(self, anchors):
        anchors = mx.array([int(anchor) for anchor in anchors], mx.int32)
        masks = mx.full((anchors.shape[0], self.proposal_rows - 1), self.config.mask_token_id, mx.int32)
        return anchors, mx.concatenate([anchors[:, None], masks], axis=1)

    def _markov_chain(self, base, anchors, length):
        """In-graph greedy Markov chain: tokens, biased logits, w1(previous)."""
        previous, tokens, logits, embeddings = anchors, [], [], []
        for position in range(length):
            embedding = self.markov_head.markov_w1(previous)
            value = base[:, position] + self.markov_head.markov_w2(embedding)
            previous = mx.argmax(value, axis=-1).astype(mx.int32)
            tokens.append(previous)
            logits.append(value)
            embeddings.append(embedding)
        return mx.stack(tokens, axis=1), mx.stack(logits, axis=1), mx.stack(embeddings, axis=1)

    def _validate_geometry(self, anchors, hidden, cache, proposal_length, rngs, temperatures):
        batch = len(anchors)
        if (type(proposal_length) is not int or not 0 <= proposal_length <= self.proposal_rows
                or not batch or len(rngs) != batch or len(temperatures) != batch
                or hidden.shape[0] != batch or len(cache) != len(self.layers)
                or any(type(a) is not int or not 0 <= a < self.config.vocab_size for a in anchors)):
            raise ValueError("DSpark proposal geometry differs")
        return batch

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures,
                            *, logits_processors=None, processor_histories=None, **kwargs):
        anchors = [int(a) for a in anchors]
        batch = self._validate_geometry(anchors, hidden, cache, proposal_length, rngs, temperatures)
        histories = processor_histories or [[] for _ in anchors]
        processors = logits_processors or [[] for _ in anchors]
        self._ordered_alternative_context = None
        if len(histories) != batch or len(processors) != batch:
            raise ValueError("DSpark processor histories differ")
        if proposal_length == 0:
            self.append_context(hidden, cache)
            self.adaptive_confidence_features = [[] for _ in anchors]
            return [[] for _ in anchors], [[] for _ in anchors]
        anchor_array, inputs = self._block(anchors)
        features = self._hidden(inputs, hidden, cache)
        base = self._logits(features)
        if any(processors):
            tokens, confidence = self._processed_chain(anchors, features, base, proposal_length, processors, histories)
        else:
            tokens, confidence = self._fast_chain(anchor_array, features, base, proposal_length)
        laws = []
        for row in tokens:
            row_laws = []
            for token in row:
                law = np.zeros(self.config.vocab_size, np.float64)
                law[token] = 1.
                row_laws.append(law)
            laws.append(row_laws)
        self.adaptive_confidence_features = confidence
        self.stats["backbone_blocks"] += batch
        self.stats["markov_steps"] += batch * proposal_length
        from ..speculative_topology import processors_branch_safe

        self._ordered_alternative_context = (
            base,
            tuple(anchors),
            # Branch-safe processors do not block alternatives (verification
            # applies them to the target); any other processor does.
            any(row and not processors_branch_safe(row) for row in processors),
        )
        return tokens, laws

    def _fast_chain(self, anchor_array, features, base, proposal_length):
        """Whole chain in one graph and one host read (no logits processors)."""
        tokens, logits, embeddings = self._markov_chain(base, anchor_array, proposal_length)
        confidence = self.confidence_head(features[:, :proposal_length], embeddings)
        invalid = mx.any(mx.isnan(logits) | (logits == float("inf")), axis=(1, 2)) | mx.any(
            ~mx.any(mx.isfinite(logits), axis=-1), axis=-1)
        mx.eval(tokens, confidence, invalid)
        if bool(np.asarray(invalid).any()):
            raise ValueError("DSpark produced invalid proposal logits")
        confidence = np.asarray(confidence)
        if not np.isfinite(confidence).all():
            raise ValueError("DSpark confidence head produced nonfinite output")
        return ([[int(t) for t in row] for row in np.asarray(tokens).tolist()],
                np.clip(confidence, -40., 40.).astype(float).tolist())

    def _processed_chain(self, anchors, features, base, proposal_length, processors, histories):
        from ..processor_probe import probe_logits_processors

        batch = len(anchors)
        previous, tokens = list(anchors), [[] for _ in anchors]
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
                if not np.isfinite(predicted_confidence[row]):
                    raise ValueError("DSpark confidence head produced nonfinite output")
                confidence[row].append(float(np.clip(predicted_confidence[row], -40., 40.)))
        return tokens, confidence

    # -- tree tiles ---------------------------------------------------------

    def start_tree(self, anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None,
                   topology=None, lattice_positions=None, codebooks=None, **_options):
        """Queue the candidate lattice (draft-context append included) on ``cache``.

        Depth ``d`` candidates are the top-K of the greedy chain's biased
        logits there, so the chain path is always present; edges are the
        Markov bias between adjacent candidates.  Nothing is evaluated here,
        so a caller may ``mx.async_eval(*state["pending"])``.
        """
        anchors = [int(a) for a in anchors]
        batch = len(anchors)
        if (not batch or hidden.shape[0] != batch or len(cache) != len(self.layers)
                or type(max_nodes) is not int or max_nodes < 1
                or any(not 0 <= a < self.config.vocab_size for a in anchors)):
            raise ValueError("DSpark tree geometry differs")
        depth = min(self.proposal_rows, max_nodes)
        if lattice_positions is not None:
            depth = min(depth, int(lattice_positions) - 1)
        if depth < 1:
            raise ValueError("DSpark tree lattice needs at least one draft position")
        self._ordered_alternative_context = None
        # Tree rows carry no per-position chain confidence.
        self.adaptive_confidence_features = None
        anchor_array, inputs = self._block(anchors)
        features = self._hidden(inputs, hidden, cache)
        base = self._logits(features[:, :depth])
        _, logits, _ = self._markov_chain(base, anchor_array, depth)
        count = min(int(self.tree_candidates), int(logits.shape[-1]))
        candidates = mx.argpartition(logits, -count, axis=-1)[..., -count:].astype(mx.int32)
        unary = mx.take_along_axis(base, candidates, axis=-1).astype(mx.float32)
        if forbidden_token_ids is not None:
            if len(forbidden_token_ids) != batch:
                raise ValueError("DSpark tree forbidden-token rows must match the batch")
            width = max((len(row) for row in forbidden_token_ids), default=0)
            if width:
                forbidden = mx.array([list(row) + [-1] * (width - len(row)) for row in forbidden_token_ids],
                                     dtype=mx.int32)
                masked = mx.any(candidates[:, :, :, None] == forbidden[:, None, None, :], axis=-1)
                unary = mx.where(masked, -float("inf"), unary)
        w1 = self.markov_head.markov_w1.weight
        w2 = self.markov_head.markov_w2.weight
        successors = w2[candidates].astype(mx.float32)                      # [B, D, K, R]
        root_bias = mx.sum(successors[:, 0] * w1[anchor_array][:, None].astype(mx.float32), axis=-1)
        if depth > 1:
            predecessors = w1[candidates[:, :-1]].astype(mx.float32)        # [B, D-1, K, R]
            pair = successors[:, 1:] @ predecessors.swapaxes(-1, -2)        # [B, D-1, K, K]
        else:
            pair = mx.zeros((batch, 0, count, count), mx.float32)
        return {"anchors": anchors, "max_nodes": int(max_nodes),
                "pending": [candidates, unary, root_bias, pair], "cache": cache}

    def finish_tree(self, state, *, mark=None):
        """Evaluate a started lattice and run the host best-first search."""
        mx.eval(*state["pending"])
        if mark is not None:
            mark("tree_draft_wait")
        candidates, unary, root_bias, pair = (np.asarray(value) for value in state["pending"])
        result = [best_first_pair_tree(candidates[row], unary[row], root_bias[row], pair[row],
                                       children=self.tree_children, max_nodes=state["max_nodes"])
                  for row in range(len(state["anchors"]))]
        self.stats["tree_lattices"] += len(result)
        if mark is not None:
            mark("tree_search")
        return result

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None,
                     topology=None, mark=None, **options):
        """Per-row best-first ``(tokens, parents)`` trees, parents first, -1 = anchor."""
        state = self.start_tree(anchors, hidden, cache, max_nodes,
                                forbidden_token_ids=forbidden_token_ids, topology=topology, **options)
        if mark is not None:
            mark("tree_draft_build")
        return self.finish_tree(state, mark=mark)

    # -- ordered alternatives -------------------------------------------------

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
        base, anchors, unsafe_processors = context
        if unsafe_processors:
            raise ValueError(
                "DSpark alternative branches need branch-safe logits processors"
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


class BidirectionalDSparkDraftModel(DSparkDraftModel):
    """DSpark on a stock DFlash backbone: full prefix, non-causal proposal block.

    The whole trained block (anchor plus masks) is always computed: its
    non-causal attention lets later mask rows shape earlier proposals.
    """

    layer_class = DFlashDecoderLayer

    @staticmethod
    def _attention_windows(config):
        return None

    @property
    def receipt_settings(self):
        return {**super().receipt_settings,
                "target_layer_ids": list(self.config.target_layer_ids),
                "tree": {"ranking": "best_first_cumulative_markov_log_law",
                         "candidates": self.tree_candidates, "children": self.tree_children}}


def dspark_draft_model(config):
    """Construct the DSpark variant the configuration's attention law names."""
    variants = {"causal": DSparkDraftModel, "bidirectional": BidirectionalDSparkDraftModel}
    try:
        return variants[config.proposal_attention](config)
    except KeyError:
        raise ValueError("Unsupported DSpark proposal attention") from None


__all__ = [
    "BidirectionalDSparkDraftModel",
    "ConfidenceHead",
    "DSparkAttention",
    "DSparkDraftModel",
    "DSparkLayer",
    "MarkovHead",
    "best_first_pair_tree",
    "dspark_draft_model",
    "proposal_mask",
]
