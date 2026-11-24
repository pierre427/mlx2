"""Inkling 2's bundled DSpark as an mlx2 external-draft proposal backend.

Original mlx2 code around the ported drafter (``inkling2_dspark``).  Each
lane's draft state is the drafter's per-layer KV (committed target context
only) plus the last committed target feature row, which the RNN Markov head
reads.  ``append_context`` commits feature rows; drafting commits any new
tail first, then runs the full trained block against the cached KV
(``Inkling2DSparkDraftModel.draft_cached``; CPU-equal to the fused
``draft_block``).  Proposals are greedy draft tokens, i.e. deterministic
point masses; the target verifies every proposal, so output is exact.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..models.cache import _BaseCache

VERSION = "inkling2-dspark-external-draft-v1"


class LastFeatureCache(_BaseCache):
    """One lane's last committed target feature row (``[1, 1, D]``) or None."""

    cache_operations_contract = CacheOperationsContract(
        family="inkling2-dspark-feature",
        topology="append_only",
        state_revision="inkling2-dspark-last-feature-v1",
        cache_layout="inkling2-dspark-feature-row-v1",
        numerics="target-tap-concatenation",
        planes=(CachePlaneDeclaration("draft_mtp", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback="kv_trim",
    )

    def __init__(self):
        self.feature = None
        self.offset = 0

    def append(self, hidden):
        self.feature = hidden[:, -1:]
        self.offset += int(hidden.shape[1])

    @property
    def state(self):
        return [] if self.feature is None else [self.feature]

    @state.setter
    def state(self, value):
        self.feature = value[0] if value else None

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, value):
        self.offset = int(value[0]) if value else 0

    def is_trimmable(self):
        return True

    def trim(self, n):
        # A trimmed feature cannot be recovered: the lane falls back to
        # ordinary decode until new context is committed.
        n = min(int(n), self.offset)
        if n > 0:
            self.offset -= n
            self.feature = None
        return n

    def empty(self):
        return self.offset == 0

    def size(self):
        return self.offset

    @property
    def nbytes(self):
        return 0 if self.feature is None else self.feature.nbytes


def _unavailable(reason):
    from ..external_speculative import DraftUnavailable

    return DraftUnavailable(reason)


class Inkling2DSparkExternal:
    receipt_kind = "external_inkling2_dspark"
    supports_logits_processors = False
    proposal_distribution = "deterministic_point_mass"
    supports_empty_target_context = True

    def __init__(self, drafter, *, num_draft: int, engine_depth: int | None = None):
        block = drafter.config.block_size
        if type(num_draft) is not int or not 1 <= num_draft < block:
            raise ValueError(f"Inkling 2 DSpark num_draft must be 1..{block - 1}")
        self.drafter = drafter
        self.num_draft = num_draft
        # Opt-in (prompt_copy_round above the trained block): the engine's
        # depth is the copy span, so a drafter round asked for up to
        # ``engine_depth`` drafts ``num_draft`` instead of failing.
        self.clamp_requests = engine_depth is not None
        if self.clamp_requests:
            if type(engine_depth) is not int or engine_depth < num_draft:
                raise ValueError("Inkling 2 DSpark engine_depth must be an integer >= num_draft")
            self.maximum_proposal_length = engine_depth
        c = drafter.config
        self.config = SimpleNamespace(
            block_size=num_draft + 1,
            # The target's logits and the drafter's are cut to the unpadded
            # vocabulary, so proposal laws (incl. composed prompt-lookup
            # copies, sized from this field) must use that width too.
            vocab_size=int(c.unpadded_vocab_size or c.vocab_size),
            hidden_size=int(c.hidden_size),
            target_layer_ids=tuple(c.target_layer_ids),
            feature_width=int(c.feature_width),
            num_hidden_layers=int(c.num_hidden_layers),
            num_key_value_heads=int(c.num_key_value_heads),
            head_dim=int(c.head_dim),
            intermediate_size=int(c.intermediate_size),
        )
        self.stats = {"block_drafts": 0, "context_appends": 0, "context_rows": 0,
                      "unavailable": 0}

    @property
    def receipt_settings(self):
        settings = {**self.drafter.receipt_settings, "kind": self.receipt_kind,
                    "engine_backend": VERSION, "num_draft": self.num_draft,
                    "proposal_distribution": self.proposal_distribution, "qualified": False}
        if self.clamp_requests:
            settings["engine_depth"] = self.maximum_proposal_length
        return settings

    # -- cache contract -------------------------------------------------------
    def make_cache(self):
        return [*self.drafter.make_cache(), LastFeatureCache()]

    def batch_caches(self, rows):
        rows = [list(row) for row in rows]
        width = len(self.drafter.layers) + 1
        if not rows or any(len(row) != width for row in rows):
            raise ValueError("Inkling 2 DSpark cache row topology mismatch")
        return [SimpleNamespace(lanes=rows)]

    @staticmethod
    def _lanes(cache):
        entry = cache[0]
        return entry.lanes if isinstance(entry, SimpleNamespace) else [list(cache)]

    def _commit(self, lane, hidden):
        if hidden.shape[1] == 0:
            return
        self.drafter.commit_context(hidden, lane[:-1])
        lane[-1].append(hidden)
        self.stats["context_rows"] += int(hidden.shape[1])

    def append_context(self, hidden, cache, *, context_tokens=None):
        if hidden.shape[1] == 0:
            return
        lanes = self._lanes(cache)
        if len(lanes) != hidden.shape[0]:
            raise ValueError("Inkling 2 DSpark context rows do not match the lanes")
        for row, lane in enumerate(lanes):
            self._commit(lane, hidden[row : row + 1])
        self.stats["context_appends"] += 1

    # -- proposals ------------------------------------------------------------
    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures,
                            **_unused):
        from ..proposal_providers import PointMassLaw

        count = int(proposal_length)
        if self.clamp_requests:
            count = min(count, self.num_draft)
        if not 1 <= count <= self.num_draft:
            raise ValueError("Inkling 2 DSpark proposal length exceeds num_draft")
        lanes = self._lanes(cache)
        if len(lanes) != len(anchors) or hidden.shape[0] != len(anchors):
            raise ValueError("Inkling 2 DSpark draft rows do not match the lanes")
        tokens = []
        for row, (anchor, lane) in enumerate(zip(anchors, lanes)):
            self._commit(lane, hidden[row : row + 1])
            feature = lane[-1].feature
            if feature is None:
                self.stats["unavailable"] += 1
                raise _unavailable("Inkling 2 DSpark has no committed target feature for this lane")
            drafted = self.drafter.draft_cached(
                int(anchor), feature, lane[:-1], count + 1, lambda logits: mx.argmax(logits, -1)
            )
            tokens.append([int(t) for t in drafted[0].tolist()])
        self.stats["block_drafts"] += len(anchors)
        return tokens, [[PointMassLaw(t) for t in row] for row in tokens]

    # -- lazy drafts (speculative_device_round ``lazy_draft``) -----------------
    def draft_lazy(self, anchor, hidden, cache, proposal_length, sampler=None):
        """One lane's drafts as a lazy int32 ``[n]`` array, no host read.

        The ops are ``draft_distributions``' own for that lane (commit the
        pending ``hidden`` rows, then ``draft_cached``), so with the default
        argmax ``sampler`` the tokens are bitwise those it returns.  A
        ``sampler(logits [1, V]) -> [1]`` draws the drafts instead (the
        device round's sampled drafts record q there).
        """
        count = int(proposal_length)
        if not 1 <= count <= self.num_draft:
            raise ValueError("Inkling 2 DSpark proposal length exceeds num_draft")
        lanes = self._lanes(cache)
        if len(lanes) != 1 or hidden.shape[0] != 1:
            raise ValueError("Inkling 2 DSpark lazy drafts run one lane at a time")
        lane = lanes[0]
        self._commit(lane, hidden)
        feature = lane[-1].feature
        if feature is None:
            self.stats["unavailable"] += 1
            raise _unavailable("Inkling 2 DSpark has no committed target feature for this lane")
        drafted = self.drafter.draft_cached(
            int(anchor), feature, lane[:-1], count + 1,
            sampler if sampler is not None else (lambda logits: mx.argmax(logits, -1)),
        )
        self.stats["block_drafts"] += 1
        return drafted[0]

    # -- pre-drafting (ExternalDraftBatchGenerator ``predraft``) ---------------
    def predraft(self, anchor, hidden, cache, proposal_length):
        """Queue one lane's next block before the host reads acceptance.

        ``cache`` is a private copy of the lane's draft plane; ``hidden``
        ([1, L, D]) holds the verify rows the pre-draft assumes commit and
        ``anchor`` is the lazy (1,) target token after them.  The ops are
        ``draft_distributions``' own (commit, then ``draft_cached``), so the
        Markov head starts from ``hidden``'s last row, the newest row that
        commits under the assumption (omlx#5508).  Returns lazy tokens
        ``[1, n]`` or ``None``; no stats move until ``adopt_predraft``.
        """
        count = int(proposal_length)
        if not 1 <= count <= self.num_draft:
            raise ValueError("Inkling 2 DSpark proposal length exceeds num_draft")
        lanes = self._lanes(cache)
        if len(lanes) != 1 or hidden.shape[0] != 1:
            raise ValueError("Inkling 2 DSpark pre-drafts one lane at a time")
        lane = lanes[0]
        if hidden.shape[1]:
            self.drafter.commit_context(hidden, lane[:-1])
            lane[-1].append(hidden)
        feature = lane[-1].feature
        if feature is None:
            return None
        return self.drafter.draft_cached(
            anchor, feature, lane[:-1], count + 1, lambda logits: mx.argmax(logits, -1)
        )

    def adopt_predraft(self, tokens, context_rows):
        """Host-read an adopted pre-draft: the rows and laws ``draft_distributions``
        would have returned for that lane, with the same stats."""
        from ..proposal_providers import PointMassLaw

        row = [int(t) for t in tokens[0].tolist()]
        self.stats["context_rows"] += int(context_rows)
        self.stats["block_drafts"] += 1
        return row, [PointMassLaw(t) for t in row]


def load_external_drafter(record, target_model, *, num_draft, options=None,
                          cap_to_block=False, clamp_requests=False):
    """``cap_to_block`` (default off): an engine depth above the trained block
    is allowed only for a composition with ``prompt_lookup_max_span``; the
    backend then drafts at most ``block_size - 1`` and the composition wrapper
    declares the longer engine depth.  Without it, a depth above the block
    still fails closed."""
    from ...adapters.inkling2_dspark import load_drafter

    drafter = load_drafter(record, target_model).configure_options(**(options or {}))
    engine_depth = int(num_draft) if clamp_requests else None
    if cap_to_block:
        num_draft = min(int(num_draft), drafter.config.block_size - 1)
    return Inkling2DSparkExternal(drafter, num_draft=num_draft, engine_depth=engine_depth)


__all__ = ["Inkling2DSparkExternal", "LastFeatureCache", "load_external_drafter"]
