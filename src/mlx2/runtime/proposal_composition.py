"""Exact host arbitration of external, copy and MTP proposals.

The external backbone still advances its committed feature cache for every
row. Arbitration changes only selected rows: copy/MTP tokens carry their
actual point-mass q, while untouched stochastic rows retain the backend's
exact proposal laws.  The external verifier remains the sole owner of
accepted state.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .prompt_lookup import IndexedPromptLookup


@dataclass(frozen=True)
class ProposalCompositionPolicy:
    prompt_lookup: bool = True
    ngram_min: int = 3
    ngram_max: int = 6
    lookback: int = 4096
    native_mtp: bool = False
    mtp_max_history: int = 4096

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict) or not value:
            raise ValueError("proposal_composition requires a nonempty JSON object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown proposal_composition keys: {sorted(unknown)}")
        policy = cls(**value)
        for name in ("prompt_lookup", "native_mtp"):
            if type(getattr(policy, name)) is not bool:
                raise ValueError(f"proposal_composition {name} must be boolean")
        for name in ("ngram_min", "ngram_max", "lookback", "mtp_max_history"):
            if type(getattr(policy, name)) is not int or getattr(policy, name) < 1:
                raise ValueError(
                    f"proposal_composition {name} must be positive integer"
                )
        if policy.ngram_min > policy.ngram_max:
            raise ValueError("proposal_composition ngram_min exceeds ngram_max")
        if not (policy.prompt_lookup or policy.native_mtp):
            raise ValueError("proposal_composition must select a proposal source")
        return policy

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ComposedDraftModel:
    """PLD → bounded private native-MTP → exact-law external arbitration.

    Histories are supplied by the request executor at each committed boundary.
    No copy index, private head cache or rejected token survives a call. Backend
    caches retain their original APCv2 sidecar contract. Composition policy must
    therefore be included in the adapter's revision binding.
    """

    requires_processor_histories = True
    supports_logits_processors = True
    _supported_proposal_distributions = frozenset(
        {"deterministic_point_mass", "stochastic_exact_law"}
    )

    def __init__(self, backend, policy, *, native_mtp_source=None):
        self.backend = backend
        self.policy = ProposalCompositionPolicy.from_value(policy)
        proposal_distribution = getattr(backend, "proposal_distribution", None)
        if proposal_distribution not in self._supported_proposal_distributions:
            raise ValueError(
                "proposal composition requires a backend with exact proposal-law support"
            )
        # Preserve the backend's distribution class.  In particular, a mixed
        # PLD/DFlash block is not globally deterministic merely because PLD
        # rows are point masses.  This keeps deterministic-only adaptive
        # admission from inspecting stochastic rows as one-hot laws.
        self.proposal_distribution = proposal_distribution
        if bool(getattr(backend, "requires_context_tokens", False)):
            raise ValueError(
                "proposal composition refuses paired-context-token backends"
            )
        if self.policy.native_mtp and not callable(native_mtp_source):
            raise ValueError(
                "proposal composition requires an adapter-owned native MTP source"
            )
        self.native_mtp_source = native_mtp_source
        self.adaptive_confidence_features = None
        self.last_proposal_sources = ()
        self.composition_stats = {
            name: 0 for name in ("prompt_lookup", "native_mtp", "external")
        }

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def receipt_settings(self):
        return {
            **copy.deepcopy(self.backend.receipt_settings),
            "proposal_composition": {
                **self.policy.as_dict(),
                "priority": ["prompt_lookup", "native_mtp", "external"],
                "q": (
                    "selected_copy_or_mtp_rows_point_mass;"
                    "external_rows_retain_backend_exact_law"
                ),
                "external_proposal_distribution": self.proposal_distribution,
                "context": "external_backbone_advanced_before_arbitration",
                "native_mtp_state": "fresh_private_recompute_and_discard",
                "confidence": "external_proxy_only_for_unsubstituted_rows",
                "qualified": False,
            },
        }

    @property
    def proposal_composition_receipt(self):
        return {
            "selected": True,
            "qualified": False,
            "authority": "diagnostic_proposal_attempts_only",
            "proposal_attempt_rows": dict(self.composition_stats),
        }

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
        **kwargs,
    ):
        if processor_histories is None or len(processor_histories) != len(anchors):
            raise ValueError(
                "proposal composition requires committed histories per row"
            )
        histories = [list(history) for history in processor_histories]
        # Determine host copies before opening the backend call; indexes see
        # committed tokens and the current anchor exactly once.  ``propose``
        # only reads sources that start inside the last ``lookback`` tokens,
        # whose keys reach back at most ``ngram_max`` tokens, so indexing
        # that window proposes exactly what the full history would at
        # O(lookback) per row-round instead of O(context).  The index is
        # rebuilt every round, so it carries no rejected-source feedback
        # (``reject_ttl=0`` says so) and no acceptance gate.
        window = self.policy.lookback + self.policy.ngram_max
        replacements = []
        for history, anchor in zip(histories, anchors):
            copied = []
            if self.policy.prompt_lookup and proposal_length:
                lookup = IndexedPromptLookup(
                    [*history[-(window - 1):], int(anchor)],
                    ngram_min=self.policy.ngram_min,
                    ngram_max=self.policy.ngram_max,
                    reject_ttl=0,
                )
                copied = lookup.propose(proposal_length, lookback=self.policy.lookback)
            replacements.append(list(copied))
        tokens, laws = self.backend.draft_distributions(
            anchors,
            hidden,
            cache,
            proposal_length,
            rngs,
            temperatures,
            logits_processors=logits_processors,
            processor_histories=histories,
            **kwargs,
        )
        confidence = getattr(self.backend, "adaptive_confidence_features", None)
        confidence = (
            list(confidence) if confidence is not None else [None] * len(anchors)
        )
        sources = []
        for row, (history, anchor, copied) in enumerate(
            zip(histories, anchors, replacements)
        ):
            source = "prompt_lookup" if copied else "external"
            if (
                not copied
                and proposal_length
                and self.policy.native_mtp
                and 0 < len(history) <= self.policy.mtp_max_history
            ):
                copied = list(
                    self.native_mtp_source(history, int(anchor), proposal_length)
                )
                if not copied:
                    raise ValueError("native MTP source returned an empty proposal")
                source = "native_mtp"
            if copied:
                if not 0 < len(copied) <= proposal_length:
                    raise ValueError(
                        "composed source returned an invalid proposal length"
                    )
                vocab = int(self.backend.config.vocab_size)
                if any(
                    type(token) is not int or not 0 <= token < vocab for token in copied
                ):
                    raise ValueError("composed source returned an invalid token")
                tokens[row] = copied
                laws[row] = []
                for token in copied:
                    q = np.zeros(vocab, dtype=np.float64)
                    q[token] = 1.0
                    laws[row].append(q)
                confidence[row] = None
            sources.append(source)
        self.adaptive_confidence_features = confidence
        self.last_proposal_sources = tuple(sources)
        if proposal_length:
            for source in sources:
                self.composition_stats[source] += 1
        return tokens, laws
