"""Hy4-flash's embedded DFly2 head as an external-draft (AST) proposal backend.

Original mlx2 code.  The math is ``HY4FlashDFly`` (runtime/models/
hy_v4_flash.py, from vLLM PR 68296).  DFly2 shares the target's KV and
indexer selections, so it declares ``requires_target_caches``: the external
engine then passes each lane's committed target cache with every drafting
call.  It keeps no KV of its own; its per-lane draft state is the target
feature (post-norm final hidden) of the last committed token.

Proposals: greedy rows are point masses; sampled rows draw each block step
from ``softmax(logits / T)`` and feed the drawn token to the next step's
hidden correction (vLLM ``draft_sample_method: probabilistic``), returning
those laws.

Trees (``propose_tree``) have conditioned tails: DFly2 is autoregressive
through its hidden correction (draft ``k + 1`` reads mask-slot output
``k + 1`` corrected with the embedding of draft ``k``), so each tree node is
scored from logits conditioned on its actual parent token.  The round keeps
one block forward per lane (it never reads the drafts); only the per-node
correction plus LM head is rerun, batched across nodes and lanes.  Best-first
trees are scored in at most ``depth`` batched rounds; ``BxD`` and
``chain`` patterns unroll each branch's own greedy chain (one batched
correction per depth).  ``MLX2_HY4_DFLY_TREE_TAILS=independent`` restores the
earlier builders over the greedy chain's per-position logits, for A/B only.
Output stays exact either way, because the target verifies every node; the
tail law only moves acceptance.  Provenance:
provenance/hy-v4-flash-dfly-tree-tails.json.
"""

from __future__ import annotations

import hashlib
import heapq
import os
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..models.cache import _BaseCache

VERSION = "hy4-flash-dfly2-external-draft-v1"
TREE_TAILS_VERSION = "hy4-flash-dfly2-conditioned-tree-tails-v1"
TREE_TAILS = ("conditioned", "independent")


class DFlyFeatureCache(_BaseCache):
    """One lane's last committed target feature (``[1, D]``) or None."""

    cache_operations_contract = CacheOperationsContract(
        family="hy4-flash-dfly2-feature",
        topology="append_only",
        state_revision="hy4-flash-dfly2-last-feature-v1",
        cache_layout="hy4-flash-dfly2-feature-row-v1",
        numerics="target-post-norm-final-hidden",
        planes=(CachePlaneDeclaration("draft_mtp", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback="kv_trim",
    )

    window = 64

    def __init__(self):
        self.features = None  # [1, <= window, D] latest committed target features
        self.offset = 0  # committed positions covered (the engine's pairing check)

    @property
    def feature(self):
        return None if self.features is None or self.features.shape[1] == 0 else self.features[:, -1]

    def append(self, hidden):
        rows = hidden if self.features is None else mx.concatenate([self.features, hidden], axis=1)
        self.features = rows[:, -self.window:]
        self.offset += int(hidden.shape[1])

    @property
    def state(self):
        return [] if self.features is None else [self.features]

    @state.setter
    def state(self, value):
        self.features = value[0] if value else None

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0]) if value else 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(int(n), self.offset)
        if n <= 0:
            return 0
        self.offset -= n
        if self.features is not None:
            keep = self.features.shape[1] - n
            self.features = self.features[:, :keep] if keep > 0 else None
        return n

    def empty(self):
        return self.offset == 0

    def size(self):
        return self.offset

    @property
    def nbytes(self):
        return 0 if self.features is None else self.features.nbytes


def _unavailable(reason):
    from ..external_speculative import DraftUnavailable

    return DraftUnavailable(reason)


def _lane_rows(caches):
    """A lane's per-layer row caches from plain or one-row batch caches."""
    out = []
    for cache in caches:
        rows = getattr(cache, "rows", None)
        if rows is not None:
            if len(rows) != 1:
                raise ValueError("DFly2 lane cache must hold exactly one row")
            cache = rows[0]
        out.append(cache)
    return out


class Hy4DFlyDraftModel:
    receipt_kind = "external_hy4_dfly2"
    receipt_name = "hy4_dfly2"
    label = "DFly2"
    supports_logits_processors = False
    proposal_distribution = "stochastic_exact_law"
    supports_empty_target_context = True
    tree_children = 4
    tree_topology_aware = True
    requires_target_caches = True

    def __init__(self, target, head, *, num_draft: int, revision: str):
        if type(num_draft) is not int or num_draft < 1:
            raise ValueError("DFly2 num_draft must be a positive integer")
        self.target, self.head = target, head
        self.revision = revision
        self.maximum_proposal_length = num_draft
        a = target.args
        self.config = SimpleNamespace(
            block_size=num_draft + 1, vocab_size=a.vocab_size, hidden_size=a.hidden_size,
            target_layer_ids=(a.num_hidden_layers - 1,), feature_width=a.hidden_size,
            num_hidden_layers=len(head.layers), num_key_value_heads=1,
            head_dim=a.kv_lora_rank + a.qk_rope_head_dim, mask_token_id=head.mask_token,
            intermediate_size=int(head.layers[0].mlp.gate_proj.weight.shape[0]),
            num_target_layers=a.num_hidden_layers)
        self.stats = {"block_drafts": 0, "tree_lattices": 0, "context_appends": 0,
                      "tree_block_forwards": 0, "tree_correction_calls": 0,
                      "tree_correction_rows": 0}
        self._target_caches = None
        mode = os.environ.get("MLX2_HY4_DFLY_TREE_TAILS", "conditioned")
        if mode not in TREE_TAILS:
            raise ValueError(f"MLX2_HY4_DFLY_TREE_TAILS must be one of {TREE_TAILS}")
        self.tree_tails = mode

    # -- receipts / geometry --------------------------------------------------
    @property
    def receipt_settings(self):
        return {
            "kind": self.receipt_name, "algorithm": VERSION, "target_revision": self.revision,
            "block_size": self.config.block_size,
            "target_index_layers": list(self.head.index_layers),
            "target_kv_layers": list(self.head.kv_layers),
            "kv_share": True, "index_share": True,
            "feature_source": "final_hidden", "qualified": False,
            "tree_tails": self.tree_tails,
            "tree_tails_algorithm": TREE_TAILS_VERSION if self.tree_tails == "conditioned" else None,
            # Present only off the bf16 default, so default receipts are unchanged.
            **({"runtime_quantization": {k: quant[k] for k in ("algorithm", "bits", "group_size")}}
               if (quant := getattr(self.head, "runtime_quantization", None)) else {}),
        }

    @property
    def ast_execution_geometry_owner(self):
        return self.target

    @property
    def ast_execution_modules(self):
        return (self.head, getattr(self.target, "lm_head", None))

    # -- cache contract -----------------------------------------------------
    def make_cache(self):
        return [DFlyFeatureCache()]

    def batch_caches(self, rows):
        rows = [list(row) for row in rows]
        if not rows or any(len(row) != 1 for row in rows):
            raise ValueError("DFly2 cache row topology mismatch")
        return [SimpleNamespace(rows=[row[0] for row in rows])]

    @staticmethod
    def _entries(cache):
        entry = cache[0]
        return entry.rows if isinstance(entry, SimpleNamespace) else [entry]

    def append_context(self, hidden, cache, *, context_tokens=None):
        if hidden.shape[1] == 0:
            return
        for row, entry in enumerate(self._entries(cache)):
            entry.append(hidden[row : row + 1])
        self.stats["context_appends"] += 1

    # -- math ---------------------------------------------------------------------
    def _features(self, hidden, cache):
        """Consume the committed feature tail (as DFlash drafters do) and
        return each lane's last committed target feature."""
        out = []
        for row, entry in enumerate(self._entries(cache)):
            if hidden.shape[1]:
                entry.append(hidden[row : row + 1])
            if entry.feature is None:
                # A one-token prompt has no committed feature yet: let the
                # engine serve this lane's round through ordinary decode.
                raise _unavailable("DFly2 has no committed target feature for this lane")
            out.append(entry.feature)
        return out

    def _lane_caches(self, row, target_caches):
        caches = _lane_rows(target_caches[row])
        position = caches[0].offset
        for index in set(self.head.index_layers):
            if position - 1 not in caches[index].selections:
                # Not recorded (e.g. a cache restored without Index Share
                # state): the round falls back to ordinary decode.
                raise _unavailable("DFly2 index selection of the last committed token is missing")
        return caches, position

    def _lane(self, row, anchor, feature, count, target_caches, choose=None):
        caches, position = self._lane_caches(row, target_caches)
        tokens, logits = self.head.propose(
            self.target, caches, mx.array([int(anchor)]), feature, anchor=position,
            num_draft=count, choose=choose)
        return tokens, logits

    def _block_logits(self, anchors, hidden, cache, count):
        if self._target_caches is None:
            raise ValueError("DFly2 drafting requires the lanes' target caches")
        count = int(count)
        if not 1 <= count <= self.maximum_proposal_length:
            raise ValueError("DFly2 proposal length exceeds its block")
        features = self._features(hidden, cache)
        rows = [self._lane(row, anchor, features[row], count, self._target_caches)[1]
                for row, anchor in enumerate(anchors)]
        return mx.concatenate(rows, axis=0)

    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures, *,
                            target_caches=None, **_unused):
        from ..proposal_providers import PointMassLaw
        from ..speculative_sampling import softmax

        if target_caches is None:
            raise ValueError("DFly2 drafting requires the lanes' target caches")
        count = int(proposal_length)
        if not 1 <= count <= self.maximum_proposal_length:
            raise ValueError("DFly2 proposal length exceeds its block")
        features = self._features(hidden, cache)
        tokens, laws = [], []
        for row, anchor in enumerate(anchors):
            temperature = float(temperatures[row])
            if temperature == 0:
                drafted, _ = self._lane(row, anchor, features[row], count, target_caches)
                row_tokens = [int(t) for t in np.asarray(drafted[0])]
                tokens.append(row_tokens)
                laws.append([PointMassLaw(t) for t in row_tokens])
                continue
            row_laws = []

            def choose(step, logits, *, rng=rngs[row], temperature=temperature, row_laws=row_laws):
                q = softmax(np.asarray(logits[0].astype(mx.float32)), temperature)
                row_laws.append(q)
                return mx.array([int(rng.sample(q))])

            drafted, _ = self._lane(row, anchor, features[row], count, target_caches, choose)
            tokens.append([int(t) for t in np.asarray(drafted[0])])
            laws.append(row_laws)
        self.stats["block_drafts"] += len(anchors)
        return tokens, laws

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None,
                     topology=None, mark=None, target_caches=None, **_options):
        """Per-row ``(tokens, parents)`` trees (``-1`` = anchor), parents first.

        ``topology`` follows ``ast_tree_proposals``: ``BxD`` is B
        distinct-root branches (fitted to ``max_nodes``), ``chain`` one
        greedy path, ``drafter`` (or None) best-first search with
        ``tree_children`` children per node scored by summed path
        log-probability.  Every node's law is conditioned on its parent.
        """
        if target_caches is None:
            raise ValueError("DFly2 drafting requires the lanes' target caches")
        if self.tree_tails == "independent":
            return self._independent_trees(anchors, hidden, cache, max_nodes, forbidden_token_ids,
                                           topology, mark, target_caches)
        if not 1 <= int(max_nodes) <= 31:
            raise ValueError(f"{self.label} AST serves 1..31 non-root nodes")
        max_nodes = int(max_nodes)
        anchors = [int(anchor) for anchor in anchors]
        forbidden = forbidden_token_ids or [() for _ in anchors]
        if topology not in (None, "drafter"):
            from ..ast_tree_proposals import pattern_for_budget

            pattern = [min(depth, self.maximum_proposal_length)
                       for depth in pattern_for_budget(topology, max_nodes)]
            if not pattern:
                return [([], []) for _ in anchors]
            blocks = self._tree_blocks(anchors, hidden, cache, max(pattern), target_caches)
            result = self._pattern_paths(blocks, anchors, pattern, max_nodes, forbidden, mark)
        else:
            depth = min(max_nodes, self.maximum_proposal_length)
            blocks = self._tree_blocks(anchors, hidden, cache, depth, target_caches)
            result = self._best_first(blocks, anchors, depth, max_nodes, forbidden, mark)
        self.stats["tree_lattices"] += len(result)
        self.stats["block_drafts"] += len(result)
        if mark is not None:
            mark("tree_search")
        return result

    def _independent_trees(self, anchors, hidden, cache, max_nodes, forbidden, topology, mark,
                           target_caches):
        """The earlier builders over the greedy chain's logits (A/B arm)."""
        from .plain_dflash import PlainDFlashDraftModel

        self._target_caches = target_caches
        try:
            if topology not in (None, "drafter"):
                return PlainDFlashDraftModel._pattern_trees(
                    self, anchors, hidden, cache, int(max_nodes), topology, forbidden, mark)
            return PlainDFlashDraftModel.propose_tree(
                self, anchors, hidden, cache, max_nodes, forbidden_token_ids=forbidden,
                topology=None, mark=mark)
        finally:
            self._target_caches = None

    # -- conditioned trees ------------------------------------------------------
    def _tree_blocks(self, anchors, hidden, cache, depth, target_caches):
        """One block forward per lane, flattened to ``[lanes * (depth + 1), D]``
        (lane ``l``'s slot ``s`` is row ``l * (depth + 1) + s``)."""
        features = self._features(hidden, cache)
        blocks = []
        for row, anchor in enumerate(anchors):
            caches, position = self._lane_caches(row, target_caches)
            h = self.head.block_hidden(self.target, caches, mx.array([anchor]), features[row],
                                       anchor=position, num_draft=depth)
            blocks.append(h[0])
        self.stats["tree_block_forwards"] += len(blocks)
        return mx.concatenate(blocks, axis=0)

    def _node_logits(self, blocks, width, lanes, slots, previous):
        """fp32 logits of tree nodes: lane ``lanes[i]``'s mask slot
        ``slots[i]`` (1-based) corrected with token ``previous[i]``."""
        index = mx.array([int(l) * width + int(s) for l, s in zip(lanes, slots)], dtype=mx.int32)
        if not isinstance(previous, mx.array):
            previous = mx.array([int(t) for t in previous], dtype=mx.int32)
        self.stats["tree_correction_calls"] += 1
        self.stats["tree_correction_rows"] += int(index.size)
        return self.head.step_logits(self.target, mx.take(blocks, index, axis=0), previous)

    def _best_first(self, blocks, anchors, depth, max_nodes, forbidden, mark, record=None):
        """Best-first trees over conditioned laws, scored in batched rounds.

        A node's score is its path's summed log-probability, each term read
        from logits conditioned on that node's parent (its token path is its
        identity: sibling tokens are distinct).  Each round replays
        best-first search per lane using only the paths already scored; every
        accepted node that still has room below it but no scored children
        joins one batched correction across lanes.  A replay that needs
        nothing new is exactly sequential best-first search over these laws,
        so that is the result.  Each round scores the whole frontier the
        replay reaches, so level ``k`` of the final tree is scored by round
        ``k + 1``: at most ``depth`` rounds, each one correction plus LM head
        call (rows scored can exceed what sequential search needs).
        Ties break by (parent node, child rank).  ``record`` (tests) maps
        ``(lane, node)`` to the fp32 logit row that scored that node's
        children.
        """
        width = depth + 1
        children = max(1, int(self.tree_children))
        lanes = len(anchors)
        blocked = [{int(t) for t in forbidden[row]} for row in range(lanes)]
        # scored[lane][path]: [(rank, token, log_prob)] children of ``path``.
        scored = [{} for _ in anchors]
        rows = [{} for _ in anchors]
        need = [(lane, ()) for lane in range(lanes)]
        first = True
        while True:
            if need:
                logits = self._node_logits(
                    blocks, width, [lane for lane, _ in need], [len(path) + 1 for _, path in need],
                    [path[-1] if path else anchors[lane] for lane, path in need])
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                # Only root rows (the first round) need room for forbidden tokens.
                extra = max((len(b) for b in blocked), default=0) if first else 0
                take = min(children + extra, int(logits.shape[-1]))
                candidates = mx.argpartition(-log_probs, take - 1, axis=-1)[:, :take]
                scores = mx.take_along_axis(log_probs, candidates, axis=-1)
                order = mx.argsort(-scores, axis=-1)
                candidates = mx.take_along_axis(candidates, order, axis=-1)
                scores = mx.take_along_axis(scores, order, axis=-1)
                if first and mark is not None:
                    mark("tree_draft_build")
                mx.eval(candidates, scores)
                if first and mark is not None:
                    mark("tree_draft_wait")
                first = False
                candidates, scores = np.asarray(candidates), np.asarray(scores)
                for i, (lane, path) in enumerate(need):
                    kids = []
                    for rank in range(take):
                        token = int(candidates[i, rank])
                        if not path and token in blocked[lane]:
                            continue
                        kids.append((rank, token, float(scores[i, rank])))
                        if len(kids) >= children:
                            break
                    scored[lane][path] = kids
                    if record is not None:
                        rows[lane][path] = logits[i]
            need, trees = [], []
            for lane in range(lanes):
                tokens, parents, paths, heap = [], [], [], []

                def push(node, path, base, lane=lane, heap=heap):
                    for rank, token, log_prob in scored[lane][path]:
                        heapq.heappush(heap, (-(base + log_prob), node, rank, token, path + (token,)))

                push(-1, (), 0.0)
                while heap and len(tokens) < max_nodes:
                    negative, parent, _rank, token, path = heapq.heappop(heap)
                    tokens.append(token)
                    parents.append(parent)
                    paths.append(path)
                    if len(path) >= depth or len(tokens) >= max_nodes:
                        continue  # no room below, or no budget left for children
                    if path in scored[lane]:
                        push(len(tokens) - 1, path, -negative)
                    else:
                        need.append((lane, path))
                trees.append((tokens, parents))
                if record is not None:
                    for node, path in enumerate(paths):
                        if path in rows[lane]:
                            record[(lane, node)] = rows[lane][path]
                    record[(lane, -1)] = rows[lane][()]
            if not need:
                return trees

    def _pattern_paths(self, blocks, anchors, pattern, max_nodes, forbidden, mark):
        """Distinct-root branches, each unrolled on its own greedy chain."""
        from ..ast_tree_proposals import paths_to_tree

        depth = max(pattern)
        width = depth + 1
        lanes = len(anchors)
        branches = len(pattern)
        root = self._node_logits(blocks, width, range(lanes), [1] * lanes, anchors)
        vocab = int(root.shape[-1])
        if any(forbidden[lane] for lane in range(lanes)):
            banned = np.zeros((lanes, vocab), dtype=bool)
            for lane in range(lanes):
                banned[lane, [int(t) for t in forbidden[lane] if 0 <= int(t) < vocab]] = True
            root = mx.where(mx.array(banned), -mx.inf, root)
        take = min(branches, vocab)
        roots = mx.argpartition(-root, take - 1, axis=-1)[:, :take]
        roots = mx.take_along_axis(
            roots, mx.argsort(-mx.take_along_axis(root, roots, axis=-1), axis=-1), axis=-1)
        # steps[b]: device token arrays [lanes] of branch b, root first.
        steps = [[roots[:, b]] for b in range(take)]
        for slot in range(2, depth + 1):
            active = [b for b in range(take) if pattern[b] >= slot]
            if not active:
                break
            previous = mx.concatenate([steps[b][-1] for b in active])
            logits = self._node_logits(blocks, width, list(range(lanes)) * len(active),
                                       [slot] * (lanes * len(active)), previous)
            drafted = mx.argmax(logits, axis=-1).astype(mx.int32)
            for i, b in enumerate(active):
                steps[b].append(drafted[i * lanes:(i + 1) * lanes])
        if mark is not None:
            mark("tree_draft_build")
        stacked = [mx.stack(chain, axis=1) for chain in steps]  # [lanes, length]
        mx.eval(*stacked)
        if mark is not None:
            mark("tree_draft_wait")
        stacked = [np.asarray(chain) for chain in stacked]
        return [paths_to_tree([[int(t) for t in chain[lane]] for chain in stacked], max_nodes,
                              forbidden_root=forbidden[lane])
                for lane in range(lanes)]

QUANTIZATION_VERSION = "hy4-flash-dfly2-load-time-affine-v1"
# What load-time ``draft_quantization`` leaves in the checkpoint dtype, and why.
QUANTIZATION_KEPT = {
    "norms": "RMSNorm scales (tiny; set the scale every projection reads)",
    "mla_absorbed_maps": "self_attn.embed_q / unembed_out, the folded kv_b_proj "
                         "(per-head MultiLinear inside latent attention scores and "
                         "values; no quantized kernel, ~4% of the head)",
    "learnable_sink_param": "attention sink logits, fp32 by sanitize",
    "mtp_fusion.gate": "per-channel fusion gate vector",
    "target_shared": "embed_tokens and the LM head belong to the target and are "
                     "never touched (the target's own quantization applies)",
}


def _parameter_bytes(module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(v.nbytes for _, v in tree_flatten(module.parameters())))


def quantize_head(head, quantization) -> dict:
    """Quantize the DFly2 head's ``nn.Linear`` projections in place (load time).

    ``quantization`` is the shared ``draft_quantization`` policy value
    (``{"bits": 4|8, "group_size": 32|64|128}``, MLX affine) or None.  Every
    ``nn.Linear`` of the head (layer q/kv/gate/o projections and MLPs, the
    slot-0 fusion ``eh_proj`` and the hidden-correction MLP) whose input
    width divides the group size is quantized; QUANTIZATION_KEPT lists what
    stays in the checkpoint dtype.  Verification stays on the target, so
    committed tokens are unchanged; only the draft law (acceptance) moves.
    Returns the receipt (stored on ``head.quantization_receipt``; also on
    ``head.runtime_quantization`` when selected, else that is None).
    """
    from mlx import nn

    before = _parameter_bytes(head)
    if quantization is None:
        receipt = {"selected": False, "bits": None, "group_size": None, "mode": None,
                   "bytes_before": before, "bytes_after": before}
        object.__setattr__(head, "runtime_quantization", None)
        object.__setattr__(head, "quantization_receipt", receipt)
        return receipt
    bits, group = int(quantization["bits"]), int(quantization["group_size"])
    modules, skipped = [], []

    def predicate(path, module):
        if not isinstance(module, nn.Linear):
            return False
        if int(module.weight.shape[-1]) % group:
            skipped.append(path)
            return False
        modules.append(path)
        return True

    nn.quantize(head, group_size=group, bits=bits, mode="affine", class_predicate=predicate)
    mx.eval(head.parameters())
    receipt = {
        "selected": True, "algorithm": QUANTIZATION_VERSION, "mode": "affine",
        "bits": bits, "group_size": group,
        "modules": sorted(modules), "module_count": len(modules),
        "skipped_group_width": sorted(skipped),
        "kept_checkpoint_dtype": sorted(QUANTIZATION_KEPT),
        "bytes_before": before, "bytes_after": _parameter_bytes(head),
        "source_artifact_unchanged": True,
    }
    # A plain attribute: an nn.Module would file a dict value as a child tree.
    object.__setattr__(head, "runtime_quantization", receipt)
    object.__setattr__(head, "quantization_receipt", receipt)
    return receipt


def dfly_record(num_draft: int, target_fingerprint: str) -> dict:
    """Inspection record for the embedded head (it ships in the target artifact)."""
    if type(num_draft) is not int or not 1 <= num_draft <= 15:
        raise ValueError("DFly2 num_draft must be an integer from 1 to 15")
    revision = hashlib.sha256(f"{VERSION}:{target_fingerprint}".encode()).hexdigest()
    return {"fingerprint": revision, "config": {"architectures": ["HY4FlashDFly"]},
            "args": SimpleNamespace(block_size=num_draft + 1, minimum_proposal_length=1),
            "draft_revision": revision, "target_revision": target_fingerprint,
            "num_draft": num_draft}


def loader_for(head):
    def load(record, target):
        return Hy4DFlyDraftModel(target, head, num_draft=int(record["num_draft"]),
                                 revision=record["target_revision"])
    return load
