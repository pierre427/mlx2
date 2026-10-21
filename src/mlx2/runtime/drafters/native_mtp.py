"""A target's own native MTP head as an external-draft proposal backend.

Original mlx2 implementation.  It lets the external draft/verify engine
(``ExternalDraftBatchGenerator``: exact progressive verification, AST
confidence-tree alternatives, cross-session packing) take its proposals from
the MTP head that ships inside the target artifact instead of a separate
drafter artifact.  Nothing here verifies anything: every proposal is a
deterministic point mass that the target's exact verifier accepts or rejects.

Contract with the target model (generic, never a model-name branch):

- ``make_mtp_cache()``, ``mtp_step(hidden, tokens, cache)`` (and optionally
  ``mtp_step_full_vocab``) with the usual native-MTP meaning: the head takes
  the target's MTP seed hidden at position ``t`` and the token at ``t + 1``
  and returns ``(logits, next_seed)``;
- ``external_feature_source == "mtp_seed"``: the target's external taps
  (``forward_with_taps``/``prefill_body``/progressive target rows) return
  the MTP seed hidden itself, so the engine's committed-feature tail is
  exactly the head's teacher-forcing input.  The adapter selects this; the
  drafter only checks it.

Feature/token pairing is the EAGLE contract the engine already has
(``requires_context_tokens``): every committed target feature arrives with
the token that follows it, the last one paired with the round's anchor.  The
head cache only ever receives verified context; draft chains run on a private
copy that is discarded.

The head is drafted greedily (argmax).  Proposals are point masses, so a
sampled request on a progressive pooled route is served by the engine's
ordinary fallback.  A greedy chain records the head's top alternatives at
each position for the continuation pool's confidence tree: the alternatives
come from logits the chain already computed, so they cost no extra head
step.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

NATIVE_MTP_DRAFT_VERSION = "native-mtp-head-external-draft-v1"
FEATURE_SOURCE = "mtp_seed"
ALTERNATIVES_PER_POSITION = 3
_STASH_LIMIT = 64


@dataclass(frozen=True)
class NativeMTPDraftConfig:
    block_size: int
    vocab_size: int
    target_layer_ids: tuple
    feature_width: int
    # Head geometry for the engine's memory reservation estimate.
    num_hidden_layers: int = 1
    hidden_size: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    mask_token_id: int | None = None
    minimum_proposal_length: int = 1
    # Read by adapters: a complete-path head the continuation pool may wrap.
    native_mtp_head: bool = True
    architectures: tuple = field(default=("NativeMTPHead",))


def _confidence(log_probability):
    probability = min(1 - 1e-12, max(1e-12, math.exp(log_probability)))
    return math.log(probability) - math.log1p(-probability)


def require_native_mtp_target(target):
    """Validate the resident head protocol; return its feature width."""

    for name in ("make_mtp_cache", "mtp_step", "forward_with_taps", "prefill_body"):
        if not callable(getattr(target, name, None)):
            raise ValueError(f"native MTP draft requires target.{name}")  # noqa: TRY004
    if getattr(target, "mtp", None) is None:
        raise ValueError("native MTP draft requires a loaded MTP head")
    if getattr(target, "external_feature_source", None) != FEATURE_SOURCE:
        raise ValueError(
            "native MTP draft requires the target to expose its MTP seed as "
            "the external feature (adapter-selected)"
        )
    width = getattr(target, "external_feature_width", None)
    if type(width) is not int or width < 1:
        raise ValueError("native MTP draft requires a declared feature width")
    return width


def mx_repeat(array, count):
    import mlx.core as mx

    return mx.repeat(array, count, axis=0)


class NativeMTPDraftModel:
    """External-draft backend over the target's resident MTP head."""
    continuation_source_reuses_backbone = True
    ast_sequential_draft = True
    # propose_tree folds ``topology`` paths and takes the context pairing.
    tree_topology_aware = True

    @property
    def ast_execution_modules(self):
        language = getattr(self.target, "language_model", self.target)
        return tuple(module for module in (self.target.mtp, getattr(language, "lm_head", None)) if module is not None)

    requires_context_tokens = True
    supports_logits_processors = True
    proposal_distribution = "deterministic_point_mass"
    continuation_protocol = "native_mtp_head"
    receipt_kind = "external_native_mtp"

    def __init__(self, target, *, num_draft, revision):
        width = require_native_mtp_target(target)
        if type(num_draft) is not int or num_draft < 1:
            raise ValueError("native MTP num_draft must be a positive integer")
        if not isinstance(revision, str) or not revision:
            raise ValueError("native MTP draft requires the target revision")
        args = getattr(target, "speculative_args", None) or target.args
        layers = getattr(target, "layers", None)
        if layers is None:
            layers = target.model.layers
        self.target = target
        self.revision = revision
        head_layers = len(getattr(target.mtp, "layers", ()) or ()) or 1
        experts = int(getattr(args, "num_experts_per_tok", 0) or 0)
        intermediate = max(
            int(getattr(args, "intermediate_size", 0) or 0),
            int(getattr(args, "moe_intermediate_size", 0) or 0) * (experts + 1),
        )
        self.config = NativeMTPDraftConfig(
            block_size=num_draft + 1,
            vocab_size=int(args.vocab_size),
            target_layer_ids=(len(layers) - 1,),
            feature_width=width,
            num_hidden_layers=head_layers,
            hidden_size=int(args.hidden_size),
            num_key_value_heads=int(args.num_key_value_heads),
            head_dim=int(getattr(args, "head_dim", None)
                         or args.hidden_size // args.num_attention_heads),
            intermediate_size=intermediate,
        )
        self.stats = {
            "committed_positions": 0,
            "draft_calls": 0,
            "chain_steps": 0,
            "complete_path_calls": 0,
            "complete_path_stash_misses": 0,
        }
        self._stash = {}

    # -- receipts ----------------------------------------------------------
    @property
    def receipt_settings(self):
        return {
            "kind": "native_mtp_head",
            "algorithm": NATIVE_MTP_DRAFT_VERSION,
            "target_revision": self.revision,
            "block_size": self.config.block_size,
            "feature_source": FEATURE_SOURCE,
            "selection": "greedy_argmax",
            "alternatives_per_position": ALTERNATIVES_PER_POSITION,
            "qsa_index_sharing": False,
            "qualified": False,
        }

    # -- cache contract ------------------------------------------------------
    def make_cache(self):
        return list(self.target.make_mtp_cache())

    def batch_caches(self, rows):
        rows = [list(row) for row in rows]
        if not rows or len({len(row) for row in rows}) != 1:
            raise ValueError("native MTP cache row topology mismatch")
        return [SimpleNamespace(rows=[row[layer] for row in rows])
                for layer in range(len(rows[0]))]

    @staticmethod
    def _rows(cache, batch):
        if cache and all(hasattr(layer, "rows") for layer in cache):
            rows = [[layer.rows[row] for layer in cache] for row in range(batch)]
        elif batch == 1:
            rows = [list(cache)]
        else:
            raise ValueError("native MTP draft requires individually owned cache rows")
        return rows

    # -- math ------------------------------------------------------------------
    def _step(self, hidden, tokens, cache, *, full_vocab):
        step = self.target.mtp_step
        if full_vocab and callable(getattr(self.target, "mtp_step_full_vocab", None)):
            step = self.target.mtp_step_full_vocab
        return step(hidden, tokens, cache)

    def append_context(self, hidden, cache, *, context_tokens=None):
        """Commit verified target seeds, each paired with its next token."""
        import mlx.core as mx

        if hidden.shape[1] == 0:
            return
        if context_tokens is None:
            raise ValueError("native MTP draft needs the token following each feature")
        rows = self._rows(cache, int(hidden.shape[0]))
        if len(context_tokens) != len(rows):
            raise ValueError("native MTP context tokens must pair with feature rows")
        for row, (caches, tokens) in enumerate(zip(rows, context_tokens)):
            if len(tokens) != hidden.shape[1]:
                raise ValueError("native MTP context tokens must pair with features")
            # Only the KV write is used; the unevaluated logits never run.
            self._step(
                hidden[row : row + 1],
                mx.array([list(tokens)], mx.uint32),
                caches,
                full_vocab=False,
            )
        self.stats["committed_positions"] += int(hidden.shape[1]) * len(rows)
        return

    def draft_distributions(
        self,
        anchors,
        hidden,
        cache,
        proposal_length,
        rngs,
        temperatures,
        *,
        context_tokens=None,
        logits_processors=None,
        processor_histories=None,
        _packed_capture=None,
        _packed_root_count=0,
        **_unused,
    ):
        """Greedy-draft ``proposal_length`` tokens per row; point-mass laws."""
        import mlx.core as mx

        from ..processor_probe import probe_logits_processors
        from ..proposal_providers import PointMassLaw

        anchors = [int(anchor) for anchor in anchors]
        batch = len(anchors)
        if hidden.shape[1] == 0:
            # No target seed yet (a one-token prompt): an ordinary round.
            return [[] for _ in range(batch)], [[] for _ in range(batch)]
        if context_tokens is None:
            if hidden.shape[1] != 1:
                raise ValueError("native MTP draft needs pairing for a multi-feature tail")
            context_tokens = [[anchor] for anchor in anchors]
        rows = self._rows(cache, batch)
        logits_processors = logits_processors or [[] for _ in range(batch)]
        histories = processor_histories or [None for _ in range(batch)]
        start = getattr(self.target, "mtp_start_cycle", None)
        end = getattr(self.target, "mtp_end_cycle", None)
        tokens_out, laws_out = [], []
        self.stats["draft_calls"] += 1
        for row in range(batch):
            processors = logits_processors[row]
            full_vocab = bool(processors)
            pairing = list(context_tokens[row])
            if len(pairing) != hidden.shape[1] or pairing[-1] != anchors[row]:
                raise ValueError("native MTP pending pairing must end with the anchor")
            if len(pairing) > 1:
                # Earlier pending seeds only extend the head's KV; their
                # (lazy) logits are never evaluated, so a long prefill tail
                # does not materialize a vocabulary row per position.
                self._step(
                    hidden[row : row + 1, :-1],
                    mx.array([pairing[:-1]], mx.uint32),
                    rows[row],
                    full_vocab=False,
                )
            logits, seed = self._step(
                hidden[row : row + 1, -1:],
                mx.array([pairing[-1:]], mx.uint32),
                rows[row],
                full_vocab=full_vocab,
            )
            self.stats["committed_positions"] += len(pairing)
            top = logits[:, -1]
            seed = seed[:, -1:]
            private = copy.deepcopy(rows[row])
            drafted, ranked = [], []
            if _packed_capture is not None:
                # A branch starts AFTER the verified anchor, but BEFORE any
                # proposed token. Never reuse the primary chain's final seed
                # or cache for a different root token.
                _packed_capture.append((seed, copy.deepcopy(private), ranked))
            if callable(start):
                start(private, share_index_state=False)
            try:
                for position in range(int(proposal_length)):
                    if position:
                        top, seed = self._step(
                            seed,
                            mx.array([[drafted[-1]]], mx.uint32),
                            private,
                            full_vocab=full_vocab,
                        )
                        top = top[:, -1]
                        self.stats["chain_steps"] += 1
                    # Rank on the device; only the top candidates reach the
                    # host (a full vocabulary row per step cost two host
                    # copies and a NumPy softmax on Flash-Next's 248K vocab).
                    values = top[0].astype(mx.float32)
                    if processors:
                        prefix = mx.array(
                            list(histories[row] or []) + [anchors[row]] + drafted,
                            dtype=mx.int32,
                        )
                        values = probe_logits_processors(
                            processors, prefix, values[None]
                        )[0].astype(mx.float32)
                    count = min(
                        max(ALTERNATIVES_PER_POSITION + 1,
                            _packed_root_count if position == 0 else 0),
                        int(values.shape[-1]),
                    )
                    candidates = mx.argpartition(-values, count - 1)[:count]
                    normalizer = mx.logsumexp(values)
                    chosen = values[candidates] - normalizer
                    mx.eval(candidates, chosen, normalizer)
                    if not math.isfinite(float(normalizer.item())):
                        break  # grammar dead end: a shorter chain, never a guess
                    pairs = sorted(
                        (
                            (int(t), float(v))
                            for t, v in zip(
                                np.asarray(candidates).tolist(),
                                np.asarray(chosen).tolist(),
                            )
                            if math.isfinite(v)
                        ),
                        key=lambda item: (-item[1], item[0]),
                    )
                    if not pairs:
                        break
                    drafted.append(pairs[0][0])
                    ranked.append(tuple(pairs))
            finally:
                if callable(end):
                    end(private)
            tokens_out.append(drafted)
            laws_out.append([PointMassLaw(token) for token in drafted])
            if histories[row] is not None:
                self._remember(histories[row], anchors[row], drafted, ranked)
        return tokens_out, laws_out

    def draft_packed_paths(
        self, anchors, hidden, cache, pattern, *, context_tokens=None,
        logits_processors=None, processor_histories=None, strict=True,
    ):
        """Make full-depth, distinct-root paths for one bounded AST tile.

        ``pattern`` counts proposal rows only (e.g. 3x5 or 5x3); the target
        owns the extra anchor row. This is a proposal capability, NOT a target
        TensorFold executor. Verified head context advances once. Each branch
        unrolls on an independent private cache; no target state is touched.
        All transient tensor captures are discarded before returning.

        ``strict=False`` (the AST tree tile) keeps every path it can make
        instead of raising: fewer legal roots or a branch's grammar dead end
        drops that branch, since the head context has already advanced.
        """
        import mlx.core as mx

        from ...adapters.proposal_path_sources import SourceContinuation
        from ..processor_probe import probe_logits_processors

        pattern = tuple(pattern)
        if (not pattern or len(pattern) > 15
            or any(type(depth) is not int or not 1 <= depth <= 15 for depth in pattern)
            or sum(pattern) > 15):
            raise ValueError("packed MTP requires 1..15 proposal rows plus one anchor")
        if max(pattern) >= self.config.block_size:
            raise ValueError("packed MTP depth exceeds the loaded proposal capability")
        anchors = list(anchors)
        if not anchors or int(hidden.shape[0]) != len(anchors) or hidden.shape[1] == 0:
            raise ValueError("packed MTP requires one verified seed tail per anchor")
        processors = logits_processors or [[] for _ in anchors]
        histories = processor_histories or [None for _ in anchors]
        if len(processors) != len(anchors) or len(histories) != len(anchors):
            raise ValueError("packed MTP context topology mismatch")
        captures = []
        try:
            primary, _ = self.draft_distributions(
                anchors, hidden, cache, max(pattern), [None] * len(anchors),
                [0.0] * len(anchors), context_tokens=context_tokens,
                logits_processors=processors, processor_histories=histories,
                _packed_capture=captures, _packed_root_count=len(pattern),
            )
            output = []
            for row, (seed, root_cache, ranked) in enumerate(captures):
                def rank(top, prefix, count):
                    values = top[0].astype(mx.float32)
                    if processors[row]:
                        history = mx.array(
                            [*list(histories[row] or []), int(anchors[row]), *prefix],
                            dtype=mx.int32,
                        )
                        values = probe_logits_processors(
                            processors[row], history, values[None]
                        )[0].astype(mx.float32)
                    count = min(count, int(values.shape[-1]))
                    indices = mx.argpartition(-values, count - 1)[:count]
                    normalizer = mx.logsumexp(values)
                    probabilities = values[indices] - normalizer
                    mx.eval(indices, probabilities, normalizer)
                    if not math.isfinite(float(normalizer.item())):
                        return []
                    return sorted(
                        ((int(token), float(value)) for token, value in zip(
                            np.asarray(indices).tolist(), np.asarray(probabilities).tolist(),
                        ) if math.isfinite(value)), key=lambda item: (-item[1], item[0]),
                    )

                roots = ranked[0][:len(pattern)] if ranked else []
                if not strict:
                    if not primary[row]:
                        output.append([])
                        continue
                elif len(roots) != len(pattern) or not primary[row]:
                    raise ValueError("packed MTP has fewer legal distinct roots than requested")
                paths = []
                for branch, ((token, log_probability), depth) in enumerate(zip(roots, pattern)):
                    if branch == 0:
                        if len(primary[row]) < depth:
                            if strict:
                                raise ValueError("packed MTP primary reached a grammar dead end")
                            depth = len(primary[row])
                        logps = [entry[0][1] for entry in ranked[:depth]]
                        paths.append(SourceContinuation(
                            tuple(primary[row][:depth]),
                            tuple(_confidence(value) for value in logps),
                            float(sum(logps)), "native_mtp_conditioned_full_path_log_proxy",
                        ))
                        continue
                    private = copy.deepcopy(root_cache)
                    branch_seed = seed
                    start = getattr(self.target, "mtp_start_cycle", None)
                    end = getattr(self.target, "mtp_end_cycle", None)
                    path, logps = [token], [log_probability]
                    try:
                        if callable(start):
                            start(private, share_index_state=False)
                        while len(path) < depth:
                            top, branch_seed = self._step(
                                branch_seed, mx.array([[path[-1]]], mx.uint32),
                                private, full_vocab=bool(processors[row]),
                            )
                            self.stats["chain_steps"] += 1
                            chosen = rank(top[:, -1], path, 1)
                            if not chosen:
                                if not strict:
                                    break
                                raise ValueError("packed MTP branch reached a grammar dead end")
                            path.append(chosen[0][0])
                            logps.append(chosen[0][1])
                        paths.append(SourceContinuation(
                            tuple(path), tuple(_confidence(value) for value in logps),
                            float(sum(logps)), "native_mtp_conditioned_full_path_log_proxy",
                        ))
                    finally:
                        if callable(end):
                            end(private)
                if paths[0].tokens != tuple(primary[row][:len(paths[0].tokens)]):
                    raise RuntimeError("packed MTP primary changed during private branching")
                output.append(paths)
            return output
        finally:
            captures.clear()

    def propose_tree(
        self, anchors, hidden, cache, max_nodes, *, context_tokens=None,
        forbidden_token_ids=None, topology=None, logits_processors=None,
        processor_histories=None, mark=None, **_unused,
    ):
        """One AST tree tile per row from the head's distinct-root branches.

        ``topology`` (default ``3x5``) is fitted to ``max_nodes`` proposal
        rows; branches come from ``draft_packed_paths`` (verified head
        context advances once, each branch unrolls privately) and fold into
        a parent-first tree.  A row whose head cannot supply that many
        distinct roots keeps the branches it can make.
        """
        from ..ast_tree_proposals import paths_to_tree, pattern_for_budget

        anchors = [int(anchor) for anchor in anchors]
        forbidden = forbidden_token_ids or [() for _ in anchors]
        pattern = tuple(
            min(depth, self.config.block_size - 1)
            for depth in pattern_for_budget(topology, int(max_nodes))
        )
        if not pattern or hidden.shape[1] == 0:
            # No verified seed yet (a one-token prompt): an ordinary round.
            return [([], []) for _ in anchors]
        if not any(logits_processors or ()):
            rows = self._lazy_tree_paths(anchors, hidden, cache, pattern, context_tokens)
        elif len(pattern) > 1:
            rows = self.draft_packed_paths(
                anchors, hidden, cache, pattern, context_tokens=context_tokens,
                logits_processors=logits_processors,
                processor_histories=processor_histories, strict=False,
            )
            rows = [[path.tokens for path in paths] for paths in rows]
            self.stats["tree_calls"] = self.stats.get("tree_calls", 0) + 1
        else:
            drafted, _laws = self.draft_distributions(
                anchors, hidden, cache, pattern[0],
                [None] * len(anchors), [0.0] * len(anchors),
                context_tokens=context_tokens, logits_processors=logits_processors,
                processor_histories=processor_histories,
            )
            rows = [[tuple(tokens)] for tokens in drafted]
        if mark is not None:
            mark("tree_draft_build")
        return [
            paths_to_tree(paths, int(max_nodes), forbidden_root=forbidden[row])
            for row, paths in enumerate(rows)
        ]

    @staticmethod
    def _branch_batch_cache(committed, branches):
        """One head cache holding ``branches`` copies of the committed prefix.

        Only for plain KV head caches (rows stay independent, so each row is
        that branch's ordinary unroll); ``None`` otherwise.
        """
        from ..models.cache import KVCache

        if not all(type(layer) is KVCache and layer.keys is not None for layer in committed):
            return None
        batched = []
        for layer in committed:
            clone = KVCache()
            clone.keys = mx_repeat(layer.keys[..., :layer.offset, :], branches)
            clone.values = mx_repeat(layer.values[..., :layer.offset, :], branches)
            clone.offset = layer.offset
            batched.append(clone)
        return batched

    def _lazy_tree_paths(self, anchors, hidden, cache, pattern, context_tokens):
        """Greedy distinct-root paths with no host sync until the end.

        Commits the verified pending seeds to each row's head cache exactly
        as ``draft_distributions`` does, takes the ``len(pattern)`` best
        first tokens on the device, unrolls each branch greedily on a
        private copy of the committed head cache, and reads every token
        back in one evaluation (the synced path costs one ``.item()`` round
        trip per head step).
        """
        import mlx.core as mx

        if context_tokens is None:
            if hidden.shape[1] != 1:
                raise ValueError("native MTP draft needs pairing for a multi-feature tail")
            context_tokens = [[anchor] for anchor in anchors]
        rows = self._rows(cache, len(anchors))
        start = getattr(self.target, "mtp_start_cycle", None)
        end = getattr(self.target, "mtp_end_cycle", None)
        branches = len(pattern)
        pending = []
        self.stats["draft_calls"] += 1
        for row in range(len(anchors)):
            pairing = list(context_tokens[row])
            if len(pairing) != hidden.shape[1] or pairing[-1] != anchors[row]:
                raise ValueError("native MTP pending pairing must end with the anchor")
            if len(pairing) > 1:
                self._step(hidden[row:row + 1, :-1], mx.array([pairing[:-1]], mx.uint32),
                           rows[row], full_vocab=False)
            logits, seed = self._step(hidden[row:row + 1, -1:], mx.array([pairing[-1:]], mx.uint32),
                                      rows[row], full_vocab=False)
            self.stats["committed_positions"] += len(pairing)
            values = logits[0, -1].astype(mx.float32)
            roots = mx.argpartition(-values, branches - 1)[:branches]
            # Rank the roots on the device: best root first (the primary).
            roots = roots[mx.argsort(-values[roots])]
            seed = seed[:, -1:]
            batched = self._branch_batch_cache(rows[row], branches) if not callable(start) else None
            if batched is not None:
                # All branches advance together: one head step per depth.
                tokens = roots.reshape(branches, 1).astype(mx.uint32)
                seeds = mx.repeat(seed, branches, axis=0)
                steps = [tokens]
                for _ in range(max(pattern) - 1):
                    top, seeds = self._step(seeds, steps[-1], batched, full_vocab=False)
                    steps.append(mx.argmax(top[:, -1], axis=-1).reshape(branches, 1).astype(mx.uint32))
                    self.stats["chain_steps"] += 1
                grid = mx.concatenate(steps, axis=1)
                pending.append([grid[branch, :depth] for branch, depth in enumerate(pattern)])
                self.stats["batched_tree_calls"] = self.stats.get("batched_tree_calls", 0) + 1
                continue
            branch_tokens = []
            for branch, depth in enumerate(pattern):
                private = copy.deepcopy(rows[row])
                if callable(start):
                    start(private, share_index_state=False)
                try:
                    token = roots[branch:branch + 1].reshape(1, 1).astype(mx.uint32)
                    path, branch_seed = [token], seed
                    for _ in range(depth - 1):
                        top, branch_seed = self._step(branch_seed, path[-1], private, full_vocab=False)
                        path.append(mx.argmax(top[:, -1], axis=-1).reshape(1, 1).astype(mx.uint32))
                        self.stats["chain_steps"] += 1
                finally:
                    if callable(end):
                        end(private)
                branch_tokens.append(mx.concatenate(path, axis=1))
            pending.append(branch_tokens)
        mx.eval([token for branch_tokens in pending for token in branch_tokens])
        return [
            [tuple(int(token) for token in path.reshape(-1).tolist()) for path in branch_tokens]
            for branch_tokens in pending
        ]

    # -- complete paths for the continuation pool ------------------------------
    def _remember(self, history, anchor, drafted, ranked):
        key = (len(history), tuple(history[-64:]), int(anchor))
        self._stash[key] = (tuple(drafted), tuple(ranked))
        while len(self._stash) > _STASH_LIMIT:
            self._stash.pop(next(iter(self._stash)))

    def propose_complete_paths(self, context, limit):
        """Primary chain plus one-token-deep alternatives at each position.

        The backbone ``draft_distributions`` call of the same round already
        ran the head over this exact context; its ranked logits supply every
        alternative, so no extra head step runs here.  A context the stash
        does not hold yields no path (the pool then keeps its other sources).
        """
        from ...adapters.proposal_path_sources import SourceContinuation

        self.stats["complete_path_calls"] += 1
        history = tuple(context.history)
        key = (len(history), history[-64:], int(context.anchor))
        stored = self._stash.get(key)
        if stored is None or len(stored[0]) < 1:
            self.stats["complete_path_stash_misses"] += 1
            return []
        drafted, ranked = stored
        depth = min(int(context.depth), len(drafted))
        drafted, ranked = drafted[:depth], ranked[:depth]
        primary_lp = [entry[0][1] for entry in ranked]
        paths = [
            SourceContinuation(
                tuple(drafted),
                tuple(_confidence(value) for value in primary_lp),
                float(sum(primary_lp)),
                "native_mtp_greedy_log_softmax_proxy",
            )
        ]
        alternatives = []
        for position, entry in enumerate(ranked):
            prefix_lp = float(sum(primary_lp[:position]))
            for token, value in entry[1:]:
                alternatives.append(
                    (
                        prefix_lp + value,
                        (*drafted[:position], token),
                        (*(_confidence(v) for v in primary_lp[:position]),
                         _confidence(value)),
                    )
                )
        alternatives.sort(key=lambda item: (-item[0], item[1]))
        for score, path, confidence in alternatives[: max(0, int(limit) - 1)]:
            paths.append(
                SourceContinuation(
                    path, confidence, float(score),
                    "native_mtp_greedy_log_softmax_proxy",
                )
            )
        return paths


def inspect_native_mtp_policy(policy, target_path, target_fingerprint):
    """An external policy names the native head by pointing at the target.

    ``draft_model`` equal to the target artifact path selects the head that
    ships inside that artifact.  The record mirrors an external drafter's
    inspection record so the shared adapter plumbing can bind it.
    """
    from pathlib import Path

    draft = policy.get("draft_model")
    if not isinstance(draft, str) or not draft:
        raise ValueError("native MTP draft_model must name the target artifact")
    if Path(draft).expanduser().resolve() != Path(target_path).expanduser().resolve():
        raise ValueError(
            "this target has no external drafter family; draft_model must be "
            "the target artifact itself (its native MTP head)"
        )
    num_draft = policy.get("num_draft", 3)
    if type(num_draft) is not int or not 1 <= num_draft <= 15:
        raise ValueError("native MTP num_draft must be an integer from 1 to 15")
    import hashlib

    # The head ships inside the target artifact: its revision is derived from
    # the target fingerprint and this backend's algorithm, as a hex pin.
    revision = hashlib.sha256(
        f"{NATIVE_MTP_DRAFT_VERSION}:{target_fingerprint}".encode()
    ).hexdigest()
    return {
        "fingerprint": revision,
        "config": {"architectures": ["NativeMTPHead"]},
        "args": SimpleNamespace(
            block_size=num_draft + 1,
            minimum_proposal_length=1,
            native_mtp_head=True,
        ),
        "draft_revision": revision,
        "target_revision": target_fingerprint,
    }


def load_native_mtp_drafter(record, target):
    return NativeMTPDraftModel(
        target,
        num_draft=int(record["args"].block_size) - 1,
        revision=str(record["target_revision"]),
    )


__all__ = [
    "FEATURE_SOURCE",
    "NATIVE_MTP_DRAFT_VERSION",
    "NativeMTPDraftConfig",
    "NativeMTPDraftModel",
    "inspect_native_mtp_policy",
    "load_native_mtp_drafter",
    "require_native_mtp_target",
]
