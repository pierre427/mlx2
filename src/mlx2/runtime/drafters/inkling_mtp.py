"""A depth-cascaded native MTP head as an external-draft proposal backend.

Original mlx2 implementation over the same external-draft protocol as
``NativeMTPDraftModel`` (``append_context``, ``draft_distributions``,
``draft_packed_paths``, ``propose_tree``, complete-path stash), for heads
that ship ``D`` *distinct* trained depths (Inkling-Small: eight) instead of
one head reused per draft step.  Nothing here verifies anything: every
proposal is a deterministic point mass the target's exact verifier accepts or
rejects.  Mechanism source and validation: ``provenance/inkling-mtp-20261104.json``.

Target contract (generic; no model-name branch): ``make_mtp_cache()`` returns
one cache per loaded depth, ``mtp_depth_step(k, hidden, tokens, cache,
last_logits_only=...)`` runs depth ``k`` over consecutive rows and returns
``(logits, raw)``, ``mtp_logits(raw)`` projects raw rows, ``mtp_depth_count``
names ``D``, and ``external_feature_source == "mtp_seed"`` makes the engine's
committed features the target's post-final-norm hidden (depth 0's input).

Alignment (SGLang's chained multi-layer MTP at absolute positions): with
target features ``h[0..H-1]`` and verified tokens ``x[0..H]`` (``x[H]`` is
the anchor), depth ``k`` at position ``p`` consumes depth ``k-1``'s raw output
at ``p`` (depth 0: ``h[p]``) and token ``x[p+k+1]``, and predicts
``x[p+k+2]``.  So depth ``k`` can only commit rows ``p <= H-1-k``; its last
committed row's raw output is the one pending input of depth ``k+1``.

- Commit (cascade): each new verified feature/token pair advances every
  depth by one row; depth ``k`` consumes depth ``k-1``'s pending row plus its
  new outputs.  Depth caches only ever hold verified context; there is no
  MTP rollback.
- Draft: token 1 is depth 0's last committed row; draft step ``k`` runs
  depth ``k`` over its ``min(k, H)`` uncommitted rows on a private cache copy
  (pending row + step ``k-1`` outputs, tokens = drafts ``1..k``).  At most
  ``D`` tokens are drafted; a depth is never reused modulo ``D``.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .native_mtp import (
    ALTERNATIVES_PER_POSITION,
    FEATURE_SOURCE,
    NativeMTPDraftConfig,
    NativeMTPDraftModel,
    _confidence,
)

CASCADED_MTP_DRAFT_VERSION = "cascaded-depth-mtp-external-draft-v1"
CASCADED_MTP_CACHE_VERSION = "inkling-cascaded-mtp-cache-v1"


def require_depth_mtp_target(target):
    """Validate the depth-cascaded head protocol; return ``(width, depths)``."""
    for name in (
        "make_mtp_cache",
        "mtp_depth_step",
        "mtp_logits",
        "forward_with_taps",
        "prefill_body",
    ):
        if not callable(getattr(target, name, None)):
            raise ValueError(f"cascaded MTP draft requires target.{name}")  # noqa: TRY004
    if getattr(target, "mtp", None) is None:
        raise ValueError("cascaded MTP draft requires an attached MTP head")
    if getattr(target, "external_feature_source", None) != FEATURE_SOURCE:
        raise ValueError(
            "cascaded MTP draft requires the target to expose its MTP seed as "
            "the external feature (adapter-selected)"
        )
    width = getattr(target, "external_feature_width", None)
    if type(width) is not int or width < 1:
        raise ValueError("cascaded MTP draft requires a declared feature width")
    depths = getattr(target, "mtp_depth_count", None)
    if type(depths) is not int or depths < 1:
        raise ValueError("cascaded MTP draft requires at least one loaded depth")
    return width, depths


def _contract():
    from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration

    return CacheOperationsContract(
        family="mtp_draft",
        topology="cascaded_depths",
        state_revision="mlx2.cascaded-mtp-draft.v1",
        cache_layout="flat-depth-caches-plus-pending-rows",
        numerics="stored_dtype",
        planes=(
            CachePlaneDeclaration("draft_mtp", ("caches", "pending", "offset")),
        ),
        exact_prefix=True,
        restores_at_checkpoints=False,
        rollback=None,
        offset_field="offset",
    )


class CascadedMTPDraftCache:
    """One lane's committed cascaded-head state.

    ``offset`` counts committed target features (depth-0 rows); each depth
    ``k`` the drafter commits holds ``max(0, offset - k)`` rows (depths at or
    beyond the drafter's ``num_draft`` stay empty).  ``pending[k]`` is depth ``k``'s last
    committed raw output row, the next input of depth ``k + 1`` (and, for
    depth 0, the source of draft token 1).  The deepest depth's row has no
    drafting consumer; it is kept (one row) for teacher-forced diagnostics.
    """

    def __init__(self, depth_caches):
        depth_caches = [list(pair) for pair in depth_caches]
        if not depth_caches or any(len(pair) != 2 for pair in depth_caches):
            raise ValueError("cascaded MTP cache needs one [kv, conv] pair per depth")
        self.depths = len(depth_caches)
        self.caches = [cache for pair in depth_caches for cache in pair]
        self.pending = [None] * self.depths
        self.offset = 0

    def cache_operations_contract(self):
        return _contract()

    def depth_cache(self, depth):
        return self.caches[2 * depth : 2 * depth + 2]

    def committed_rows(self, depth):
        return max(0, self.offset - depth)

    # APCv2 spills and restores this sidecar through ``save_prompt_cache``:
    # ``state`` nests each depth plane's own state (``None`` leaves are
    # recorded as placeholders) and ``meta_state`` names each plane's class.
    @property
    def state(self):
        return [[cache.state for cache in self.caches], list(self.pending)]

    @state.setter
    def state(self, value):
        planes, pending = value
        if len(planes) != len(self.caches) or len(pending) != self.depths:
            raise ValueError("cascaded MTP cache state has a different depth layout")
        for cache, plane in zip(self.caches, planes):
            cache.state = plane
        self.pending = list(pending)

    @property
    def meta_state(self):
        from ..models.cache import _cache_class_token

        return (
            CASCADED_MTP_CACHE_VERSION,
            str(self.depths),
            str(self.offset),
            [_cache_class_token(type(cache)) for cache in self.caches],
            [cache.meta_state for cache in self.caches],
        )

    @classmethod
    def from_state(cls, state, meta_state):
        from ..models.cache import _resolve_cache_class

        if len(meta_state) != 5 or meta_state[0] != CASCADED_MTP_CACHE_VERSION:
            raise ValueError("cascaded MTP cache metadata revision differs")
        depths, offset = int(meta_state[1]), int(meta_state[2])
        tokens, metas = meta_state[3], meta_state[4]
        planes, pending = state
        if not (len(tokens) == len(metas) == len(planes) == 2 * depths == 2 * len(pending)):
            raise ValueError("cascaded MTP cache state has a different depth layout")
        # Depths past the drafter's ``num_draft`` are never committed, so only
        # a row the offset cannot have produced is an inconsistency.
        if offset < 0 or any(
            row is not None and offset <= depth for depth, row in enumerate(pending)
        ):
            raise ValueError("cascaded MTP pending rows disagree with the offset")
        caches = [
            _resolve_cache_class(token).from_state(plane, meta)
            for token, plane, meta in zip(tokens, planes, metas)
        ]
        obj = cls(zip(caches[0::2], caches[1::2]))
        obj.pending = list(pending)
        obj.offset = offset
        return obj

    @property
    def nbytes(self):
        from mlx.utils import tree_flatten

        return sum(
            int(value.nbytes) for _, value in tree_flatten(self.state) if value is not None
        )


class CascadedMTPDraftModel(NativeMTPDraftModel):
    """External-draft backend over a target's ``D`` distinct MTP depths."""

    continuation_protocol = "native_mtp_head"
    receipt_kind = "external_native_mtp"

    def __init__(self, target, *, num_draft, revision, sidecar=None, tree_pipeline=False):
        width, depths = require_depth_mtp_target(target)
        if type(tree_pipeline) is not bool:
            raise ValueError("cascaded MTP tree_pipeline must be a boolean")
        # Engine opt-in for early tree prelaunch (see ``start_tree``).
        self.start_tree_pairs_context = tree_pipeline
        if type(num_draft) is not int or not 1 <= num_draft <= depths:
            raise ValueError(
                f"cascaded MTP num_draft must be an integer from 1 to the {depths} "
                "loaded depths (a depth is never reused)"
            )
        if not isinstance(revision, str) or not revision:
            raise ValueError("cascaded MTP draft requires the target revision")
        args = getattr(target, "speculative_args", None) or target.args
        self.target = target
        self.revision = revision
        self.depths = depths
        # Depth k drafts token k + 1, so only the first ``num_draft`` depths
        # ever feed a proposal; deeper loaded depths are not committed (each
        # commit is a dense block forward per depth over the accepted rows).
        self.active_depths = num_draft
        self.sidecar = dict(sidecar or {})
        self.config = NativeMTPDraftConfig(
            block_size=num_draft + 1,
            vocab_size=int(args.vocab_size),
            target_layer_ids=(len(target.layers) - 1,),
            feature_width=width,
            num_hidden_layers=depths,
            hidden_size=int(args.hidden_size),
            num_key_value_heads=int(args.num_key_value_heads),
            head_dim=int(args.head_dim),
            intermediate_size=int(
                getattr(args, "dense_intermediate_size", 0)
                or args.intermediate_size
            ),
        )
        self.stats = {
            "committed_positions": 0,
            "depth_commit_rows": [0] * depths,
            "draft_calls": 0,
            "chain_steps": 0,
            "chain_rows": 0,
            "chain_descriptor_forks": 0,
            "chain_kv_lends": 0,
            "chain_deepcopy_forks": 0,
            "complete_path_calls": 0,
            "complete_path_stash_misses": 0,
        }
        self._stash = {}

    @property
    def receipt_settings(self):
        return {
            "kind": "native_mtp_head_cascaded",
            "algorithm": CASCADED_MTP_DRAFT_VERSION,
            "target_revision": self.revision,
            "block_size": self.config.block_size,
            "depths": self.depths,
            "active_depths": self.active_depths,
            "feature_source": FEATURE_SOURCE,
            "selection": "greedy_argmax",
            "tree_pipeline": self.start_tree_pairs_context,
            "alternatives_per_position": ALTERNATIVES_PER_POSITION,
            "sidecar": self.sidecar,
            "qualified": False,
        }

    # -- cache contract ------------------------------------------------------
    def make_cache(self):
        return [CascadedMTPDraftCache(self.target.make_mtp_cache())]

    @staticmethod
    def _state(caches):
        if len(caches) != 1 or not isinstance(caches[0], CascadedMTPDraftCache):
            raise ValueError("cascaded MTP draft requires its own lane cache")
        return caches[0]

    # -- commit ----------------------------------------------------------------
    def _commit(self, state, hidden, tokens):
        """Advance every depth over the verified pairs ``(hidden[i], tokens[i])``.

        ``tokens[i]`` is the token following feature ``i``: ``x[H_old+1+i]``.
        """
        import mlx.core as mx

        length = int(hidden.shape[1])
        if length == 0:
            return
        if len(tokens) != length:
            raise ValueError("cascaded MTP context tokens must pair with features")
        old_offset, new_offset = state.offset, state.offset + length
        old_pending = list(state.pending)
        outputs = None
        for depth in range(self.active_depths):
            old_rows = max(0, old_offset - depth)
            rows = max(0, new_offset - depth) - old_rows
            if rows == 0:
                break
            if depth == 0:
                inputs = hidden
            else:
                parts = []
                if old_offset >= depth:
                    parts.append(old_pending[depth - 1])
                if outputs is not None:
                    parts.append(outputs)
                inputs = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
                if inputs.shape[1] < rows:
                    raise RuntimeError("cascaded MTP commit lost a pending depth row")
                inputs = inputs[:, :rows]
            start = old_rows + depth - old_offset
            ids = mx.array([list(tokens[start : start + rows])], mx.uint32)
            # Only the depth cache write and raw rows are used; the lazy
            # last-row logits are never evaluated on a commit.
            _, outputs = self.target.mtp_depth_step(
                depth, inputs, ids, state.depth_cache(depth), last_logits_only=True
            )
            state.pending[depth] = outputs[:, -1:]
            self.stats["depth_commit_rows"][depth] += rows
        state.offset = new_offset
        self.stats["committed_positions"] += length

    def append_context(self, hidden, cache, *, context_tokens=None):
        """Commit verified target seeds, each paired with its next token."""
        if hidden.shape[1] == 0:
            return
        if context_tokens is None:
            raise ValueError("cascaded MTP draft needs the token following each feature")
        rows = self._rows(cache, int(hidden.shape[0]))
        if len(context_tokens) != len(rows):
            raise ValueError("cascaded MTP context tokens must pair with feature rows")
        for row, (caches, tokens) in enumerate(zip(rows, context_tokens)):
            self._commit(self._state(caches), hidden[row : row + 1], list(tokens))

    def _commit_rows(self, anchors, hidden, cache, context_tokens):
        """Commit each row's pending tail; return the per-row lane states."""
        if context_tokens is None:
            if hidden.shape[1] != 1:
                raise ValueError("cascaded MTP draft needs pairing for a multi-feature tail")
            context_tokens = [[anchor] for anchor in anchors]
        rows = self._rows(cache, len(anchors))
        states = []
        for row, anchor in enumerate(anchors):
            pairing = [int(token) for token in context_tokens[row]]
            if len(pairing) != hidden.shape[1] or pairing[-1] != int(anchor):
                raise ValueError("cascaded MTP pending pairing must end with the anchor")
            state = self._state(rows[row])
            self._commit(state, hidden[row : row + 1], pairing)
            states.append(state)
        return states

    # -- chains ----------------------------------------------------------------
    def _chain(self, state, first, length, select):
        """Draft tokens ``2..length`` after ``first`` on private depth copies.

        ``first`` and every ``select(top, drafts)`` result are ``[1, 1]``
        uint32 arrays; ``select`` may return ``None`` to stop the chain.
        """
        import mlx.core as mx

        offset = state.offset
        drafts = [first]
        previous = None
        for depth in range(1, min(int(length), self.active_depths)):
            rows = offset - max(0, offset - depth)
            parts = []
            if offset >= depth:
                parts.append(state.pending[depth - 1])
            if previous is not None:
                parts.append(previous)
            hidden = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
            if hidden.shape[1] != rows:
                raise RuntimeError("cascaded MTP chain row alignment differs")
            tokens = mx.concatenate(drafts[depth - rows : depth], axis=1)
            private = self._private_depth_cache(state, depth)
            logits, previous = self.target.mtp_depth_step(
                depth, hidden, tokens, private, last_logits_only=True
            )
            self.stats["chain_steps"] += 1
            self.stats["chain_rows"] += rows
            chosen = select(logits[:, -1], drafts)
            if chosen is None:
                break
            drafts.append(chosen)
        return drafts

    def _private_depth_cache(self, state, depth):
        """One chain step's private fork of depth ``depth``'s ``[kv, conv]``.

        A deep copy duplicated every buffer on each chain step, and a global
        depth's KV is the whole context (~67 MB per round at 16K).  Committed
        planes fork as descriptors instead, and an append-only KV lends its
        live buffer as the target tree does: the step writes only past the
        live fill level, which the next commit overwrites, so the committed
        prefix the chain reads is the live one.  A plane whose descriptor
        clone declines (live transaction state) keeps the deep copy.

        The commit that precedes a chain is still lazy, and its depth step
        holds ``keys[..., :offset]`` views of the lent buffer until it is
        evaluated; a slice assignment whose input is viewed cannot be
        donated, so the lent append would copy the buffer after all.
        Submitting the pending rows first (no host wait) detaches those
        views before the chain's write is encoded.
        """
        import mlx.core as mx

        from ..cow_cache import lend_append_only_kv

        live = state.depth_cache(depth)
        private = []
        for plane in live:
            clone = getattr(plane, "committed_descriptor_clone", lambda: None)()
            if clone is None:
                self.stats["chain_deepcopy_forks"] += 1
                clone = copy.deepcopy(plane)
            else:
                self.stats["chain_descriptor_forks"] += 1
            private.append(clone)
        if lend_append_only_kv(live[0], private[0]):
            self.stats["chain_kv_lends"] += 1
            mx.async_eval(*[row for row in state.pending if row is not None])
        return private

    def _first_logits(self, state):
        return self.target.mtp_logits(state.pending[0])[:, -1]

    @staticmethod
    def _rank(top, count, processors, history):
        """Host-ranked ``(token, log-probability)`` pairs, best first."""
        import mlx.core as mx

        from ..processor_probe import probe_logits_processors

        values = top[0].astype(mx.float32)
        if processors:
            values = probe_logits_processors(
                processors, mx.array(history, dtype=mx.int32), values[None]
            )[0].astype(mx.float32)
        count = min(int(count), int(values.shape[-1]))
        candidates = mx.argpartition(-values, count - 1)[:count]
        normalizer = mx.logsumexp(values)
        chosen = values[candidates] - normalizer
        mx.eval(candidates, chosen, normalizer)
        if not math.isfinite(float(normalizer.item())):
            return []  # grammar dead end: a shorter chain, never a guess
        return sorted(
            (
                (int(token), float(value))
                for token, value in zip(
                    np.asarray(candidates).tolist(), np.asarray(chosen).tolist()
                )
                if math.isfinite(value)
            ),
            key=lambda item: (-item[1], item[0]),
        )

    @staticmethod
    def _token(value):
        import mlx.core as mx

        return mx.array([[int(value)]], mx.uint32)

    def _ranked_chain(self, state, first_pairs, first, length, processors, prefix):
        """A synced greedy chain from ``first``; returns ``(tokens, ranked)``."""
        drafted, ranked = [int(first)], [tuple(first_pairs)]

        def select(top, _drafts):
            pairs = self._rank(
                top, ALTERNATIVES_PER_POSITION + 1, processors, [*prefix, *drafted]
            )
            if not pairs:
                return None
            drafted.append(pairs[0][0])
            ranked.append(tuple(pairs))
            return self._token(pairs[0][0])

        self._chain(state, self._token(first), length, select)
        return drafted, ranked

    def _synced_chain(self, state, anchor, history, processors, length):
        """One row's host-ranked greedy chain; returns ``(tokens, ranked)``."""
        if length <= 0:
            return [], []
        prefix = [*list(history or []), anchor]
        pairs = self._rank(
            self._first_logits(state), ALTERNATIVES_PER_POSITION + 1, processors, prefix
        )
        if not pairs:
            return [], []
        return self._ranked_chain(state, pairs, pairs[0][0], length, processors, prefix)

    # -- protocol --------------------------------------------------------------
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
        **_unused,
    ):
        """Greedy-draft up to ``min(proposal_length, D)`` tokens per row."""
        from ..proposal_providers import PointMassLaw

        anchors = [int(anchor) for anchor in anchors]
        if hidden.shape[1] == 0:
            # No target seed yet (a one-token prompt): an ordinary round.
            return [[] for _ in anchors], [[] for _ in anchors]
        states = self._commit_rows(anchors, hidden, cache, context_tokens)
        logits_processors = logits_processors or [[] for _ in anchors]
        histories = processor_histories or [None for _ in anchors]
        length = min(int(proposal_length), self.active_depths)
        self.stats["draft_calls"] += 1
        if length > 0 and not any(logits_processors):
            return self._lazy_distributions(states, anchors, histories, length)
        tokens_out, laws_out = [], []
        for row, state in enumerate(states):
            drafted, ranked = self._synced_chain(
                state, anchors[row], histories[row], logits_processors[row], length
            )
            tokens_out.append(drafted)
            laws_out.append([PointMassLaw(token) for token in drafted])
            if histories[row] is not None:
                self._remember(
                    histories[row], anchors[row], drafted, ranked,
                    processors=logits_processors[row],
                )
        return tokens_out, laws_out

    def _lazy_distributions(self, states, anchors, histories, length):
        """Greedy chains for processor-free rows with one host evaluation.

        The synced chain ranks every depth on the host (an eval, a scalar read
        and NumPy copies per depth), so the GPU idled between depth steps.
        Argmax keeps the first maximum, the same token the best-first ranking
        (ties by lowest id) picks; ranked alternatives use the same ops and
        are read after one eval.  A non-finite normalizer ends the chain at
        that depth, where the synced chain stopped.
        """
        import mlx.core as mx

        from ..proposal_providers import PointMassLaw

        count = ALTERNATIVES_PER_POSITION + 1

        def lazy_rank(top):
            values = top[0].astype(mx.float32)
            k = min(count, int(values.shape[-1]))
            candidates = mx.argpartition(-values, k - 1)[:k]
            normalizer = mx.logsumexp(values)
            chosen = mx.argmax(values).reshape(1, 1).astype(mx.uint32)
            return (candidates, values[candidates] - normalizer, normalizer, chosen), chosen

        pending = []
        for state in states:
            ranks = []
            first_rank, first = lazy_rank(self._first_logits(state))
            ranks.append(first_rank)

            def select(top, _drafts, ranks=ranks):
                rank, chosen = lazy_rank(top)
                ranks.append(rank)
                return chosen

            self._chain(state, first, length, select)
            pending.append(ranks)
        mx.eval([part for ranks in pending for rank in ranks for part in rank])
        tokens_out, laws_out = [], []
        for row, ranks in enumerate(pending):
            drafted, ranked = [], []
            for candidates, chosen, normalizer, consumed in ranks:
                if not math.isfinite(float(normalizer.item())):
                    break
                pairs = sorted(
                    (
                        (int(token), float(value))
                        for token, value in zip(
                            np.asarray(candidates).tolist(), np.asarray(chosen).tolist()
                        )
                        if math.isfinite(value)
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                if not pairs:
                    break
                if pairs[0][0] != int(consumed.item()):
                    # Rounded log-probabilities tied above the raw argmax the
                    # chain consumed: redo the row on the host-ranked chain.
                    drafted = None
                    break
                drafted.append(pairs[0][0])
                ranked.append(tuple(pairs))
            if drafted is None:
                self.stats["lazy_chain_fallbacks"] = self.stats.get("lazy_chain_fallbacks", 0) + 1
                drafted, ranked = self._synced_chain(
                    states[row], anchors[row], histories[row], [], length
                )
            tokens_out.append(drafted)
            laws_out.append([PointMassLaw(token) for token in drafted])
            if histories[row] is not None:
                self._remember(histories[row], anchors[row], drafted, ranked, processors=[])
        self.stats["lazy_chain_calls"] = self.stats.get("lazy_chain_calls", 0) + 1
        return tokens_out, laws_out

    def draft_packed_paths(
        self, anchors, hidden, cache, pattern, *, context_tokens=None,
        logits_processors=None, processor_histories=None, strict=True,
    ):
        """Full-depth, distinct-root paths for one bounded AST tile.

        Verified head context advances once; each branch unrolls the deeper
        depths on independent private copies.  ``strict=False`` keeps every
        path it can make (fewer legal roots or a grammar dead end).
        """
        from ...adapters.proposal_path_sources import SourceContinuation

        pattern = tuple(pattern)
        if (not pattern or len(pattern) > 15
            or any(type(depth) is not int or not 1 <= depth <= 15 for depth in pattern)
            or sum(pattern) > 15):
            raise ValueError("packed MTP requires 1..15 proposal rows plus one anchor")
        if max(pattern) >= self.config.block_size:
            raise ValueError("packed MTP depth exceeds the loaded proposal capability")
        anchors = [int(anchor) for anchor in anchors]
        if not anchors or int(hidden.shape[0]) != len(anchors) or hidden.shape[1] == 0:
            raise ValueError("packed MTP requires one verified seed tail per anchor")
        processors = logits_processors or [[] for _ in anchors]
        histories = processor_histories or [None for _ in anchors]
        if len(processors) != len(anchors) or len(histories) != len(anchors):
            raise ValueError("packed MTP context topology mismatch")
        states = self._commit_rows(anchors, hidden, cache, context_tokens)
        self.stats["draft_calls"] += 1
        lazy = (
            self._lazy_packed_paths(states, anchors, histories, pattern, strict)
            if not any(processors)
            else [None] * len(states)
        )
        output = []
        for row, state in enumerate(states):
            if lazy[row] is not None:
                output.append(lazy[row])
                continue
            prefix = [*list(histories[row] or []), anchors[row]]
            first = self._rank(
                self._first_logits(state),
                max(len(pattern), ALTERNATIVES_PER_POSITION + 1),
                processors[row],
                prefix,
            )
            roots = first[: len(pattern)]
            if len(roots) != len(pattern) and strict:
                raise ValueError("packed MTP has fewer legal distinct roots than requested")
            paths = []
            for branch, ((token, log_probability), depth) in enumerate(zip(roots, pattern)):
                drafted, ranked = self._ranked_chain(
                    state, first[: ALTERNATIVES_PER_POSITION + 1], token, depth,
                    processors[row], prefix,
                )
                if len(drafted) < depth and strict:
                    raise ValueError("packed MTP branch reached a grammar dead end")
                if branch == 0 and histories[row] is not None:
                    self._remember(
                        histories[row], anchors[row], drafted, ranked,
                        processors=processors[row],
                    )
                logps = [log_probability, *(entry[0][1] for entry in ranked[1:])]
                paths.append(SourceContinuation(
                    tuple(drafted), tuple(_confidence(value) for value in logps),
                    float(sum(logps)), "native_mtp_conditioned_full_path_log_proxy",
                ))
            output.append(paths)
        return output

    def _lazy_packed_paths(self, states, anchors, histories, pattern, strict):
        """``draft_packed_paths`` for processor-free rows with one evaluation.

        Roots come from a lazy top-k of the first logits; each branch chains
        lazily (argmax per depth) with its ranked alternatives.  After the one
        eval, a row whose lazy root order differs from the host best-first
        order (a logit tie) or that hit a non-finite normalizer returns None
        and takes the synced path, so outputs equal ``draft_packed_paths``.
        """
        import mlx.core as mx

        from ...adapters.proposal_path_sources import SourceContinuation

        count = ALTERNATIVES_PER_POSITION + 1
        roots_wanted = len(pattern)

        def lazy_rank(top, k):
            values = top[0].astype(mx.float32)
            k = min(k, int(values.shape[-1]))
            candidates = mx.argpartition(-values, k - 1)[:k]
            normalizer = mx.logsumexp(values)
            return values, (candidates, values[candidates] - normalizer, normalizer)

        def host_pairs(rank):
            candidates, chosen, normalizer = rank[:3]
            if not math.isfinite(float(normalizer.item())):
                return None
            return sorted(
                ((int(t), float(v)) for t, v in zip(
                    np.asarray(candidates).tolist(), np.asarray(chosen).tolist())
                 if math.isfinite(v)),
                key=lambda item: (-item[1], item[0]),
            )

        pending = []
        for state in states:
            values, first_rank = lazy_rank(self._first_logits(state), max(roots_wanted, count))
            roots = mx.argpartition(-values, roots_wanted - 1)[:roots_wanted]
            roots = roots[mx.argsort(-values[roots])]
            branches = []
            for branch, depth in enumerate(pattern):
                ranks = []

                def select(top, _drafts, ranks=ranks):
                    branch_values, rank = lazy_rank(top, count)
                    consumed = mx.argmax(branch_values).reshape(1, 1).astype(mx.uint32)
                    ranks.append((*rank, consumed))
                    return consumed

                first = roots[branch : branch + 1].reshape(1, 1).astype(mx.uint32)
                self._chain(state, first, depth, select)
                branches.append(ranks)
            pending.append((first_rank, roots, branches))
        mx.eval([
            part for first_rank, roots, branches in pending
            for part in (*first_rank, roots, *(p for ranks in branches for rank in ranks for p in rank))
        ])
        output = []
        for row, (first_rank, roots, branches) in enumerate(pending):
            first = host_pairs(first_rank)
            lazy_roots = [int(t) for t in np.asarray(roots).tolist()]
            if not first or [token for token, _ in first[:roots_wanted]] != lazy_roots:
                output.append(None)
                continue
            paths, ok = [], True
            for branch, ((token, log_probability), depth, ranks) in enumerate(
                    zip(first[:roots_wanted], pattern, branches)):
                drafted, ranked = [token], [tuple(first[:count])]
                for rank in ranks:
                    pairs = host_pairs(rank)
                    if not pairs:
                        break
                    if pairs[0][0] != int(rank[3].item()):
                        ok = False  # rounded tie above the consumed argmax
                        break
                    drafted.append(pairs[0][0])
                    ranked.append(tuple(pairs))
                if not ok or (len(drafted) < depth and strict):
                    ok = False
                    break
                if branch == 0 and histories[row] is not None:
                    self._remember(histories[row], anchors[row], drafted, ranked, processors=[])
                logps = [log_probability, *(entry[0][1] for entry in ranked[1:])]
                paths.append(SourceContinuation(
                    tuple(drafted), tuple(_confidence(value) for value in logps),
                    float(sum(logps)), "native_mtp_conditioned_full_path_log_proxy",
                ))
            output.append(paths if ok else None)
        self.stats["lazy_chain_calls"] = self.stats.get("lazy_chain_calls", 0) + 1
        return output

    def _queue_tree_paths(self, anchors, hidden, cache, pattern, context_tokens):
        """Build the greedy distinct-root paths as lazy arrays; nothing is evaluated."""
        import mlx.core as mx

        anchors = [int(anchor) for anchor in anchors]
        states = self._commit_rows(anchors, hidden, cache, context_tokens)
        self.stats["draft_calls"] += 1
        branches = len(pattern)
        pending = []
        for state in states:
            values = self._first_logits(state)[0].astype(mx.float32)
            roots = mx.argpartition(-values, branches - 1)[:branches]
            roots = roots[mx.argsort(-values[roots])]

            def select(top, _drafts):
                return mx.argmax(top, axis=-1).reshape(1, 1).astype(mx.uint32)

            paths = []
            for branch, depth in enumerate(pattern):
                first = roots[branch : branch + 1].reshape(1, 1).astype(mx.uint32)
                paths.append(mx.concatenate(self._chain(state, first, depth, select), axis=1))
            pending.append(paths)
        return pending

    @staticmethod
    def _read_tree_paths(pending):
        import mlx.core as mx

        mx.eval([path for paths in pending for path in paths])
        return [
            [tuple(int(token) for token in path.reshape(-1).tolist()) for path in paths]
            for paths in pending
        ]

    def _lazy_tree_paths(self, anchors, hidden, cache, pattern, context_tokens):
        """Greedy distinct-root paths with one host evaluation at the end."""
        return self._read_tree_paths(
            self._queue_tree_paths(anchors, hidden, cache, pattern, context_tokens)
        )

    # -- pipelined tree tiles ------------------------------------------------
    # The engine may queue the next round's paths on a descriptor copy of the
    # draft plane while the host commits this round (``start_tree``), then
    # adopt them at the exact boundary they were built for (``finish_tree``).
    # The paths are the ones ``propose_tree`` returns for the same inputs: the
    # same lazy graph, read after one evaluation.  The engine queues them only
    # when ``start_tree_pairs_context`` is set (policy ``mtp_tree_pipeline``).

    def start_tree(
        self, anchors, hidden, cache, max_nodes, *, context_tokens=None,
        forbidden_token_ids=None, topology=None, logits_processors=None, **_unused,
    ):
        from ..ast_tree_proposals import pattern_for_budget

        if any(logits_processors or ()):
            raise ValueError("pipelined MTP trees take no logits processors")
        anchors = [int(anchor) for anchor in anchors]
        pattern = tuple(
            min(depth, self.config.block_size - 1)
            for depth in pattern_for_budget(topology, int(max_nodes))
        )
        if not pattern or hidden.shape[1] == 0:
            raise ValueError("pipelined MTP tree needs a verified seed and a budget")
        paths = self._queue_tree_paths(anchors, hidden, cache, pattern, context_tokens)
        self.stats["pipelined_tree_starts"] = self.stats.get("pipelined_tree_starts", 0) + 1
        return {
            "pending": [path for row in paths for path in row],
            "paths": paths,
            "max_nodes": int(max_nodes),
            "forbidden": list(forbidden_token_ids or [() for _ in anchors]),
        }

    def finish_tree(self, state, *, mark=None):
        from ..ast_tree_proposals import paths_to_tree

        rows = self._read_tree_paths(state["paths"])
        if mark is not None:
            mark("tree_draft_wait")
        return [
            paths_to_tree(paths, state["max_nodes"], forbidden_root=state["forbidden"][row])
            for row, paths in enumerate(rows)
        ]


def inspect_cascaded_mtp_record(
    *, num_draft, depths, sidecar_sha256, target_fingerprint, draft_head="bf16"
):
    """Inspection record binding the sidecar identity into the draft revision.

    ``num_draft`` decides which depth caches a boundary populates and
    ``draft_head`` the draft-logit arithmetic, so both enter the revision:
    a boundary committed under one never binds under another.
    """
    import hashlib
    from types import SimpleNamespace

    if type(depths) is not int or depths < 1:
        raise ValueError("cascaded MTP depths must be a positive integer")
    if type(num_draft) is not int or not 1 <= num_draft <= depths:
        raise ValueError(
            f"native MTP num_draft must be an integer from 1 to mtp_depths ({depths})"
        )
    revision = hashlib.sha256(
        f"{CASCADED_MTP_DRAFT_VERSION}:{target_fingerprint}:{sidecar_sha256}:{depths}"
        f":num_draft={num_draft}:draft_head={draft_head}".encode()
    ).hexdigest()
    return {
        "fingerprint": revision,
        "config": {"architectures": ["CascadedNativeMTPHead"]},
        "args": SimpleNamespace(
            block_size=num_draft + 1,
            minimum_proposal_length=1,
            native_mtp_head=True,
        ),
        "draft_revision": revision,
        "target_revision": target_fingerprint,
    }


def load_cascaded_mtp_drafter(record, target, *, sidecar=None, tree_pipeline=False):
    return CascadedMTPDraftModel(
        target,
        num_draft=int(record["args"].block_size) - 1,
        revision=str(record["target_revision"]),
        sidecar=sidecar,
        tree_pipeline=tree_pipeline,
    )


__all__ = [
    "CASCADED_MTP_CACHE_VERSION",
    "CASCADED_MTP_DRAFT_VERSION",
    "CascadedMTPDraftCache",
    "CascadedMTPDraftModel",
    "inspect_cascaded_mtp_record",
    "load_cascaded_mtp_drafter",
    "require_depth_mtp_target",
]
