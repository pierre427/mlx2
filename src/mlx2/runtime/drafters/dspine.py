# SPDX-License-Identifier: Apache-2.0
"""DSpine: a causal DFlash block with a code bridge after every draft layer.

Mechanism from Tencent/DSpine (arXiv 2609.36173), reimplemented from the
reference ``dspine_mlx/dspine_mlx/{bridge,model,propose}.py`` and
``sglang_dspine/models/dspine.py``; provenance/qwen38-dspine-drafter.json.

Forward of one block ``[anchor, mask * (T-1)]`` against committed target
context (the stock DFlash context path: fc + hidden_norm, context-only KV):

* every layer is a stock Qwen3 decoder layer with a causal proposal triangle
  (and the per-query sliding window where declared);
* after layer ``l`` the bridge reads the row: ``f = out`` (bf16),
  ``rms = sqrt(mean f^2)``, ``code_t = sqrt(C) * normalize(W_read[l] f_t/rms_t)``
  with row 0 replaced by ``sqrt(C) * codebook[anchor]``; then
  ``[q;s;v] = [G_q; G_p; A] code`` and for ``t >= 1``
  ``out_t = f_t + rms_t * W_write[l] (sigmoid(q_t + s_{t-1} + b) * v_{t-1})``.
  Row 0 is never written.  The next layer's input norm (or the final norm) is
  applied to ``out`` in fp32 with one bf16 rounding, as the reference does.

Proposal ("serial_table" second pass, ``exact_sender``): per proposal row the
top-K target-head candidates of the final row; the last write is redone with
the *actual* predecessor token's codebook code instead of the predicted code,
scored without the final RMS normalization through two vocabulary tables
``TG = (head * norm_w) W_write[-1]`` [V,R] and ``TSV = sqrt(C) codebook [G_p;A]^T``
[V,2R], and the greedy chain walks the resulting ``[K x K]`` tables.  Drafts
are deterministic point masses; the ordinary target performs every
acceptance decision, so verified output does not depend on this law.

Tree (original mlx2, not in the reference): the same ``[K x K]`` tables,
rescaled to target-head logits of the refined row, read as a per-edge law
``log softmax_r logit[i, s, r]`` over row ``i``'s candidates given the
predecessor's candidate ``s``.  Forbidden ids are removed at the root only
(the next position); the verifier enforces the rest.  A best-first search
over cumulative edge log-law grows one tree of up to ``max_nodes`` nodes
(depth at most the trained proposal rows); the greedy chain is the tree's
leftmost path.  Proposal only: the target-tree verifier decides every token.
"""
from __future__ import annotations

import heapq
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import dflash2 as _dflash2
from .dflash_base import DFlashDecoderLayer, DFlashDraftModel

# torch.finfo(torch.float32).tiny: clamp floor of the bridge read RMS.
_F32_TINY = 1.1754943508222875e-38


class DSpineCodeBridge(nn.Module):
    """Parameter names/shapes match the checkpoint under ``selected_prefix_bridge.``."""

    def __init__(self, config):
        super().__init__()
        hidden, code, rank = config.hidden_size, config.code_size, config.bridge_rank
        layers = config.num_hidden_layers
        self.code_size, self.rank, self.eps = code, rank, float(config.bridge_eps)
        self.num_layers = layers
        self.prev = nn.Linear(code, rank, bias=False)
        self.gate = nn.Linear(code, rank, bias=True)
        self.predecessor_gate = nn.Linear(code, rank, bias=False)
        self.out = nn.Linear(rank, code, bias=False)  # terminal write: must be zero
        self.read_weight = mx.zeros((code, hidden), dtype=mx.bfloat16)
        self.early_read_weights = mx.zeros((layers - 1, code, hidden), dtype=mx.bfloat16)
        self.feature_write_weights = mx.zeros((layers, hidden, rank), dtype=mx.bfloat16)
        self.codebook = mx.zeros((config.vocab_size, code), dtype=mx.bfloat16)

    def layer_read_weight(self, layer):
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def packed_weight(self):
        """[3R, C] = [G_q; G_p; A] (reference ``prepare``)."""
        return mx.concatenate(
            [self.gate.weight, self.predecessor_gate.weight, self.prev.weight], axis=0
        )

    def anchor_codes(self, anchors):
        return (math.sqrt(self.code_size) * self.codebook[anchors].astype(mx.float32)).astype(mx.bfloat16)

    def __call__(self, hidden, input_ids, layer, next_norm, stash=None):
        """One write after ``layer``; returns ``(next_norm(out), out)``."""
        if hidden.dtype != mx.bfloat16:
            raise ValueError("The DSpine code bridge requires BF16 activations")
        sqrt_c = math.sqrt(self.code_size)
        raw = hidden
        f = raw.astype(mx.float32)
        rms = mx.sqrt(mx.maximum(mx.mean(f * f, axis=-1), _F32_TINY))  # [B,T]
        normalized = (f / (rms[..., None] + self.eps)).astype(mx.bfloat16)
        projected = mx.matmul(normalized, self.layer_read_weight(layer).T).astype(mx.float32)
        proj_norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
        codes = (sqrt_c * projected / (proj_norm + self.eps)).astype(mx.bfloat16)
        codes = mx.concatenate(
            [self.anchor_codes(input_ids[:, 0])[:, None, :], codes[:, 1:, :]], axis=1
        )
        packed = mx.matmul(codes, self.packed_weight().T)  # [B,T,3R] bf16
        if stash is not None:
            stash.update(raw=raw, rms=rms, packed=packed)
        q, s, v = mx.split(packed.astype(mx.float32), [self.rank, 2 * self.rank], axis=-1)
        gate = mx.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.astype(mx.float32))
        message = mx.concatenate([mx.zeros_like(q[:, :1]), gate * v[:, :-1]], axis=1).astype(mx.bfloat16)
        delta = mx.matmul(message, self.feature_write_weights[layer].T).astype(mx.float32)
        not_anchor = (mx.arange(raw.shape[1]) != 0).astype(mx.float32)[None, :, None]
        out = (f + rms[..., None] * delta * not_anchor).astype(mx.bfloat16)
        x = out.astype(mx.float32)
        y = (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + next_norm.eps)).astype(mx.bfloat16)
        normed = (y.astype(mx.float32) * next_norm.weight.astype(mx.float32)).astype(mx.bfloat16)
        return normed, out


class HeadRows:
    """Dense BF16 rows of the target LM head (plain or affine-quantized)."""

    def __init__(self, head):
        module = getattr(head, "__self__", head)  # Embedding.as_linear binds
        self.module = module
        self.quantized = hasattr(module, "scales")

    def _take(self, index):
        m = self.module
        if not self.quantized:
            return m.weight[index].astype(mx.bfloat16)
        biases = None if getattr(m, "biases", None) is None else m.biases[index]
        return mx.dequantize(
            m.weight[index], m.scales[index], biases, group_size=m.group_size,
            bits=m.bits, mode=getattr(m, "mode", "affine"),
        ).astype(mx.bfloat16)

    def rows(self, ids):
        flat = ids.reshape(-1)
        out = self._take(flat)
        return out.reshape(*ids.shape, out.shape[-1])

    def rows_range(self, start, end):
        return self._take(slice(start, end))


def build_serial_tables(head_rows, vocab_size, norm_w, write_w, pv_w, codebook, chunk=8192):
    """``TG [V,R]`` and ``TSV [V,2R]`` in BF16 (reference ``build_serial_tables``)."""
    nwf = norm_w.astype(mx.float32)
    sqrt_c = math.sqrt(int(codebook.shape[1]))
    tg, tsv = [], []
    for start in range(0, int(vocab_size), chunk):
        end = min(start + chunk, int(vocab_size))
        rows = head_rows(start, end)
        tg.append(mx.matmul((rows.astype(mx.float32) * nwf).astype(mx.bfloat16), write_w.astype(mx.bfloat16)))
        codes = (sqrt_c * codebook[start:end].astype(mx.float32)).astype(mx.bfloat16)
        tsv.append(mx.matmul(codes, pv_w.astype(mx.bfloat16).T))
        mx.eval(tg[-1], tsv[-1])
    return mx.concatenate(tg, axis=0), mx.concatenate(tsv, axis=0)


def topk_ids(logits, k):
    """Top-k ids per row, returned ascending by id ([..., V] -> [..., k])."""
    part = mx.argpartition(-logits.astype(mx.float32), k - 1, axis=-1)[..., :k]
    return mx.sort(part, axis=-1).astype(mx.int32)


def dspine_best_first_tree(candidates, edge_log_law, max_nodes, *, children=4,
                           forbidden_root=()):
    """Best-first tree over the exact-sender edge laws (host side).

    ``candidates [n, K]`` are row ``i``'s candidate ids; ``edge_log_law
    [n, K, K]`` holds ``log q(r | s)`` for row ``i``'s candidate ``r`` after
    row ``i-1``'s candidate ``s`` (row 0: every ``s`` is the anchor).  A
    node's priority is its root path's summed edge log-law; each popped
    node offers its ``children`` best successors.  Returns ``tokens,
    parents, info`` in pop order (parents precede children, ``-1`` is the
    anchor).  The order does not depend on ``max_nodes``, so a smaller
    budget is a prefix of a larger one.
    """
    candidates = np.asarray(candidates)
    law = np.asarray(edge_log_law, dtype=np.float64)
    depth_count, width = candidates.shape
    if law.shape != (depth_count, width, width):
        raise ValueError("DSpine tree edge laws must be [n, K, K] over the candidates")
    fan = max(1, min(int(children), width))
    blocked = {int(token) for token in forbidden_root}
    heap, serial = [], 0

    def push(row, base, parent, depth):
        nonlocal serial
        offered = 0
        for r in np.argsort(-row, kind="stable"):
            if offered == fan or not np.isfinite(row[r]):
                return
            if depth == 0 and int(candidates[0, r]) in blocked:
                continue
            heapq.heappush(heap, (-(base + float(row[r])), serial, parent, depth, int(r), float(row[r])))
            serial += 1
            offered += 1

    push(law[0, 0], 0.0, -1, 0)
    tokens, parents, depths, log_q, path = [], [], [], [], []
    while heap and len(tokens) < int(max_nodes):
        negative, _serial, parent, depth, r, edge = heapq.heappop(heap)
        tokens.append(int(candidates[depth, r]))
        parents.append(parent)
        depths.append(depth + 1)
        log_q.append(edge)
        path.append(-negative)
        if depth + 1 < depth_count:
            push(law[depth + 1, r], -negative, len(tokens) - 1, depth + 1)
    return tokens, parents, {"depths": depths, "log_q": log_q, "path_log_q": path}


class DSpineDraftModel(DFlashDraftModel):
    layer_class = DFlashDecoderLayer
    receipt_kind = "external_dspine"
    proposal_distribution = "deterministic_point_mass"
    requires_processor_histories = False
    supports_empty_target_context = True

    def __init__(self, config):
        super().__init__(config)
        self.selected_prefix_bridge = DSpineCodeBridge(config)
        self._tables = None
        self._head = None
        self.stats = {"backbone_blocks": 0, "proposed": 0}

    # -- binding ------------------------------------------------------------

    @property
    def proposal_rows(self):
        return self.config.block_size - 1

    @property
    def receipt_settings(self):
        return {
            "trained_block_size": self.config.block_size,
            "proposal_rows": self.proposal_rows,
            "target_layer_ids": list(self.config.target_layer_ids),
            "sliding_window": self.config.sliding_window,
            "code_bridge": {"code_size": self.config.code_size, "rank": self.config.bridge_rank},
            "second_pass": {"mode": "serial_table", "kind": "exact_sender", "head_topk": self.config.head_topk},
            "tree": {"search": "best_first_edge_log_law", "children": self.tree_children},
            "proposal_distribution": self.proposal_distribution,
            "runtime_quantization": self.__dict__.get("runtime_quantization"),
            "draft_kernel": self.__dict__.get("draft_kernel_receipt"),
            "qualified": False,
        }

    def validate_target_compatibility(self, target_model):
        args = getattr(target_model, "speculative_args", None) or getattr(target_model, "args", None)
        if args is None:
            return
        for name in ("hidden_size", "vocab_size"):
            value = getattr(args, name, None)
            if value is not None and value != getattr(self.config, name):
                raise ValueError(f"DSpine target {name} mismatch")
        layers = getattr(args, "num_hidden_layers", None)
        if layers is not None and layers != self.config.num_target_layers:
            raise ValueError("DSpine target layer count mismatch")

    def bind(self, target_model, *, build_tables=True):
        self.validate_target_compatibility(target_model)
        super().bind(target_model)
        self._head = HeadRows(self.lm_head)
        if build_tables:
            self.build_tables()
        return self

    def build_tables(self):
        bridge = self.selected_prefix_bridge
        rank = bridge.rank
        self._tables = build_serial_tables(
            self._head.rows_range, self.config.vocab_size, self.norm.weight,
            bridge.feature_write_weights[-1], bridge.packed_weight()[rank:], bridge.codebook,
        )
        return self._tables

    def sanitize(self, weights):
        return {k.removeprefix("model."): v for k, v in weights.items()}

    # -- caches (stock DFlash2 context-only KV) -----------------------------

    def make_cache(self):
        caches = DFlashDraftModel.make_cache(self)
        for cache in caches:
            cache.keys = mx.zeros((1, self.config.num_key_value_heads, 0, self.config.head_dim))
            cache.values = mx.zeros_like(cache.keys)
        return caches

    append_context = _dflash2._append_context
    batch_caches = _dflash2._batch_caches

    # -- forward ------------------------------------------------------------

    def block_forward(self, inputs, target_hidden, cache):
        """``(final normed rows [B,T,H], last-write stash)`` for one block."""
        x = self._embed_input_tokens(inputs).astype(mx.bfloat16)
        ctx = self.hidden_norm(self.fc(self.target_activation(target_hidden)))
        residual, x_in = x, self.layers[0].input_layernorm(x)
        last = len(self.layers) - 1
        stash = {}
        for i, (layer, entry) in enumerate(zip(self.layers, cache)):
            h = residual + layer.self_attn(x_in, ctx, self.rope, entry)
            out = (h + layer.mlp(layer.post_attention_layernorm(h))).astype(mx.bfloat16)
            next_norm = self.layers[i + 1].input_layernorm if i < last else self.norm
            x_in, residual = self.selected_prefix_bridge(
                out, inputs, i, next_norm, stash if i == last else None
            )
        return x_in, stash

    def score_tables(self, normed, stash, anchors, proposal_length, *, logit_scale=False):
        """Candidates ``[B,n,K]`` and greedy predecessor tables ``[B,n,K,K]``.

        ``idx[b,i,j]`` is the best candidate at row ``i`` when row ``i-1``
        chose candidate ``j`` (row 0: the anchor, every ``j`` equal).
        The reference score skips the final RMS normalization, so it is the
        target logit times the refined row's RMS, a positive per-predecessor
        factor the argmax ignores.  ``logit_scale`` divides it back out
        (the refined row is recomputed per predecessor) so the returned
        score is the target-head logit of the refined row, which the tree's
        edge law needs.
        """
        if self._tables is None:
            raise RuntimeError("DSpine serial tables were not built (bind first)")
        tg, tsv = self._tables
        bridge = self.selected_prefix_bridge
        R, K, n = bridge.rank, self.config.head_topk, proposal_length
        rows = normed[:, 1:1 + n]
        raw = stash["raw"][:, 1:1 + n].astype(mx.float32)
        rms = stash["rms"][:, 1:1 + n].astype(mx.float32)
        q = stash["packed"][:, 1:1 + n, :R].astype(mx.float32)
        cand = topk_ids(self._logits(rows), K)                                # [B,n,K]
        head = self._head.rows(cand).astype(mx.float32)                       # [B,n,K,H]
        norm_w = self.norm.weight.astype(mx.float32)
        base = mx.sum((raw[:, :, None, :] * norm_w) * head, axis=-1)          # [B,n,K]
        anchor = mx.broadcast_to(anchors.astype(mx.int32)[:, None, None], (cand.shape[0], 1, K))
        sid = mx.concatenate([anchor, cand[:, :-1]], axis=1)                  # [B,n,K]
        tsv_sid = tsv[sid].astype(mx.float32)
        m = mx.sigmoid(q[:, :, None, :] + tsv_sid[..., :R] + bridge.gate.bias.astype(mx.float32)) * tsv_sid[..., R:]
        s = mx.matmul(m, tg[cand].astype(mx.float32).swapaxes(-1, -2))       # [B,n,Ks,Kr]
        score = base[:, :, None, :] + rms[:, :, None, None] * s
        idx = mx.argmax(score, axis=-1).astype(mx.int32)
        if logit_scale:
            write = bridge.feature_write_weights[-1].astype(mx.float32)            # [H,R]
            out = raw[:, :, None, :] + rms[:, :, None, None] * mx.matmul(m, write.T)  # [B,n,Ks,H]
            out_rms = mx.sqrt(mx.mean(out * out, axis=-1) + self.norm.eps)       # [B,n,Ks]
            score = score / out_rms[..., None]
        return cand, idx, score

    @staticmethod
    def walk_chain(cand, idx):
        batch, n, _ = cand.shape
        j = idx[:, 0, 0]
        tokens = []
        for i in range(n):
            tokens.append(mx.take_along_axis(cand[:, i], j[:, None], axis=-1)[:, 0])
            if i + 1 < n:
                j = mx.take_along_axis(idx[:, i + 1], j[:, None], axis=-1)[:, 0]
        return mx.stack(tokens, axis=1)

    def propose_chain(self, anchors, hidden, cache, proposal_length=None):
        """Greedy refined chain ``[B, n]`` (graph only; nothing evaluated)."""
        n = self.proposal_rows if proposal_length is None else int(proposal_length)
        if not 1 <= n <= self.proposal_rows:
            raise ValueError("DSpine proposal length must fit the trained block")
        anchors = mx.array([int(a) for a in anchors], mx.int32) if not isinstance(anchors, mx.array) else anchors
        masks = mx.full((anchors.shape[0], n), self.config.mask_token_id, mx.int32)
        # Causal block: a shorter block computes exactly the same leading rows.
        inputs = mx.concatenate([anchors[:, None], masks], axis=1)
        normed, stash = self.block_forward(inputs, hidden, cache)
        cand, idx, _ = self.score_tables(normed, stash, anchors, n)
        return self.walk_chain(cand, idx)

    # -- tree (tree15 / AST tile route) ---------------------------------------

    # Fan-out per node; the K=16 exact-sender candidates bound it from above.
    # Opt-in, not a default (2026-11-09 GSM8K B1 on Qwen3.8-27B-oQ4e-mtp): with
    # q4 + lane draft kernel, accept len 7.83 vs DFlash2 tree15 7.64 but tok/s at
    # parity (115.7 vs 115.8); q8 is 5% slower with no acceptance gain
    # (docs/research/dspine-qwen38-27b-gap-20261108.md).
    tree_children = 4

    def start_tree(self, anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None,
                   lattice_positions=None, mark=None, **_options):
        """Queue one block pass and its edge laws (draft-context append included).

        Nothing is evaluated here; the caller may ``mx.async_eval`` the
        returned ``state["pending"]`` and ``finish_tree`` searches it.
        ``lattice_positions`` bounds the block (anchor plus masks), as for
        DFlash2.
        Options of other drafters (calibration, codebooks, topology) do not
        apply to this law and are ignored.
        """
        if type(max_nodes) is not int or not 1 <= max_nodes <= 31:
            raise ValueError("DSpine tree serves 1..31 non-root nodes")
        anchors_list = [int(a) for a in anchors]
        batch = len(anchors_list)
        if not batch or hidden.shape[0] != batch or len(cache) != len(self.layers):
            raise ValueError("DSpine tree geometry differs")
        if forbidden_token_ids is not None and len(forbidden_token_ids) != batch:
            raise ValueError("DSpine tree forbidden-token rows must match the batch")
        depth = min(max_nodes, self.proposal_rows)
        if lattice_positions is not None:
            if type(lattice_positions) is not int or lattice_positions < 2:
                raise ValueError("DSpine tree lattice needs at least one draft position")
            depth = min(depth, lattice_positions - 1)
        anchors_array = mx.array(anchors_list, mx.int32)
        inputs = mx.concatenate(
            [anchors_array[:, None], mx.full((batch, depth), self.config.mask_token_id, mx.int32)],
            axis=1,
        )
        normed, stash = self.block_forward(inputs, hidden, cache)
        cand, _idx, score = self.score_tables(normed, stash, anchors_array, depth, logit_scale=True)
        law = score - mx.logsumexp(score, axis=-1, keepdims=True)
        return {
            "pending": [cand, law],
            "max_nodes": max_nodes,
            "forbidden": [tuple(row) for row in forbidden_token_ids]
            if forbidden_token_ids is not None else [() for _ in anchors_list],
        }

    def finish_tree(self, state, *, mark=None):
        cand, law = state["pending"]
        if mark is not None:
            mark("tree_draft_build")
        mx.eval(cand, law)
        if mark is not None:
            mark("tree_draft_wait")
        cand, law = np.asarray(cand), np.asarray(law)
        result = [
            dspine_best_first_tree(
                cand[row], law[row], state["max_nodes"],
                children=self.tree_children, forbidden_root=state["forbidden"][row],
            )
            for row in range(cand.shape[0])
        ]
        self.stats["tree_lattices"] = self.stats.get("tree_lattices", 0) + len(result)
        self.stats["backbone_blocks"] += len(result)
        self.stats["proposed"] += sum(len(tree[0]) for tree in result)
        if mark is not None:
            mark("tree_search")
        return result

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, mark=None, **options):
        """Per-row ``(tokens, parents, info)`` trees from one block pass."""
        state = self.start_tree(anchors, hidden, cache, max_nodes, mark=mark, **options)
        return self.finish_tree(state, mark=mark)

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures,
                            *, logits_processors=None, processor_histories=None, **_kwargs):
        anchors = [int(a) for a in anchors]
        batch = len(anchors)
        if (type(proposal_length) is not int or not 0 <= proposal_length <= self.proposal_rows
                or not batch or hidden.shape[0] != batch or len(cache) != len(self.layers)
                or len(rngs) != batch or len(temperatures) != batch):
            raise ValueError("DSpine proposal geometry differs")
        if logits_processors and any(logits_processors):
            raise ValueError("DSpine chain does not apply logits processors")
        if proposal_length == 0:
            self.append_context(hidden, cache)
            return [[] for _ in anchors], [[] for _ in anchors]
        tokens = self.propose_chain(anchors, hidden, cache, proposal_length)
        mx.eval(tokens)
        tokens = np.asarray(tokens).tolist()
        laws = []
        for row in tokens:
            row_laws = []
            for token in row:
                law = np.zeros(self.config.vocab_size, np.float64)
                law[token] = 1.0
                row_laws.append(law)
            laws.append(row_laws)
        self.stats["backbone_blocks"] += batch
        self.stats["proposed"] += batch * proposal_length
        return tokens, laws


__all__ = [
    "DSpineCodeBridge", "DSpineDraftModel", "HeadRows", "build_serial_tables",
    "dspine_best_first_tree", "topk_ids",
]
