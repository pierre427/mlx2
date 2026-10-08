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
    lookback: int = 256
    min_context_match: int = 4
    max_sources: int = 8
    native_mtp: bool = False
    mtp_max_history: int = 4096
    trusted_pld: bool = False
    trusted_pld_min_score: float = 0.9
    trusted_pld_min_verified_tokens: int = 32
    trusted_pld_min_match: int = 4
    trusted_pld_max_span: int = 8
    trusted_pld_audit_interval: int = 0

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict) or not value:
            raise ValueError("proposal_composition requires a nonempty JSON object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown proposal_composition keys: {sorted(unknown)}")
        policy = cls(**value)
        for name in ("prompt_lookup", "native_mtp", "trusted_pld"):
            if type(getattr(policy, name)) is not bool:
                raise ValueError(f"proposal_composition {name} must be boolean")
        for name in ("ngram_min", "ngram_max", "lookback", "mtp_max_history"):
            if type(getattr(policy, name)) is not int or getattr(policy, name) < 1:
                raise ValueError(
                    f"proposal_composition {name} must be positive integer"
                )
        if policy.ngram_min > policy.ngram_max:
            raise ValueError("proposal_composition ngram_min exceeds ngram_max")
        for name in ("min_context_match", "max_sources"):
            if type(getattr(policy, name)) is not int or getattr(policy, name) < 0:
                raise ValueError(
                    f"proposal_composition {name} must be nonnegative integer"
                )
        if policy.min_context_match > 64:
            raise ValueError(
                "proposal_composition min_context_match must be at most 64"
            )
        if not (policy.prompt_lookup or policy.native_mtp):
            raise ValueError("proposal_composition must select a proposal source")
        if policy.trusted_pld and not policy.prompt_lookup:
            raise ValueError("trusted PLD requires prompt_lookup")
        if (
            isinstance(policy.trusted_pld_min_score, bool)
            or not isinstance(policy.trusted_pld_min_score, (int, float))
            or not 0 <= float(policy.trusted_pld_min_score) <= 1
        ):
            raise ValueError("trusted PLD minimum score must be in 0..1")
        for name in (
            "trusted_pld_min_match",
            "trusted_pld_max_span",
            "trusted_pld_min_verified_tokens",
            "trusted_pld_audit_interval",
        ):
            value = getattr(policy, name)
            minimum = (
                0
                if name in (
                    "trusted_pld_audit_interval",
                    "trusted_pld_min_verified_tokens",
                )
                else 1
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"proposal_composition {name} is invalid")
        return policy

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ComposedDraftModel:
    """Score-ranked PLD/native-MTP/external exact-law arbitration.

    Histories are supplied by the request executor at each committed boundary.
    No copy index, private head cache or rejected token survives a call. Backend
    caches retain their original APCv2 sidecar contract. Composition policy must
    therefore be included in the adapter's revision binding. Source scores use
    only earlier committed verification labels. In particular, a stochastic
    external chain is never inspected before source selection, which would
    selection-bias its proposal law.
    """

    requires_processor_histories = True
    supports_logits_processors = True
    _supported_proposal_distributions = frozenset(
        {"deterministic_point_mass", "stochastic_exact_law"}
    )

    @classmethod
    def supports_backend(cls, backend):
        """Whether ``backend`` publishes a proposal law composition can verify
        exactly.  Callers deciding a *default* consult this before wrapping;
        an explicit request is left to fail closed in ``__init__``."""
        return (
            getattr(backend, "proposal_distribution", None)
            in cls._supported_proposal_distributions
        )

    def __init__(self, backend, policy, *, native_mtp_source=None):
        self.backend = backend
        self.policy = ProposalCompositionPolicy.from_value(policy)
        proposal_distribution = getattr(backend, "proposal_distribution", None)
        if not self.supports_backend(backend):
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
        self.last_proposal_score_keys = ()
        self.last_proposal_scores = ()
        self.last_proposal_trusted = ()
        self.last_proposal_audits = ()
        self._score_counts = {}
        self._trusted_pld_selections = 0
        self._trusted_stats = {
            "blind_rounds": 0,
            "blind_tokens": 0,
            "audit_rounds": 0,
            "audit_proposed_tokens": 0,
            "audit_accepted_tokens": 0,
        }
        self.composition_stats = {
            name: 0 for name in ("prompt_lookup", "native_mtp", "external")
        }

    _SCORE_PRIOR_ACCEPTANCE = 0.5
    _SCORE_PRIOR_STRENGTH = 8.0
    _SOURCE_ORDER = {"prompt_lookup": 0, "native_mtp": 1, "external": 2}

    def _proposal_score(self, key):
        accepted, proposed = self._score_counts.get(key, (0, 0))
        strength = self._SCORE_PRIOR_STRENGTH
        return (
            accepted + self._SCORE_PRIOR_ACCEPTANCE * strength
        ) / (proposed + strength)

    def _select_candidate(self, candidates, external_span):
        """Rank by committed score, usable span, then stable source order."""
        return min(
            candidates,
            key=lambda item: (
                -self._proposal_score(item[2]),
                -(len(item[1]) if item[1] is not None else external_span),
                self._SOURCE_ORDER[item[0]],
            ),
        )

    def observe_proposal_feedback(self, key, proposed, accepted):
        """Publish one committed verifier label set to the lagged scorer."""
        if (
            not isinstance(key, str)
            or not key
            or type(proposed) is not int
            or type(accepted) is not int
            or not 0 <= accepted <= proposed
            or proposed < 1
        ):
            raise ValueError("invalid committed proposal-composition feedback")
        old_accepted, old_proposed = self._score_counts.get(key, (0, 0))
        self._score_counts[key] = (
            old_accepted + accepted,
            old_proposed + proposed,
        )

    def observe_trusted_pld(self, *, proposed, audited, accepted=0):
        if (
            type(proposed) is not int
            or proposed < 1
            or type(audited) is not bool
            or type(accepted) is not int
            or not 0 <= accepted <= proposed
        ):
            raise ValueError("invalid trusted PLD observation")
        if audited:
            self._trusted_stats["audit_rounds"] += 1
            self._trusted_stats["audit_proposed_tokens"] += proposed
            self._trusted_stats["audit_accepted_tokens"] += accepted
        else:
            self._trusted_stats["blind_rounds"] += 1
            self._trusted_stats["blind_tokens"] += proposed

    @staticmethod
    def _prompt_lookup_score_key(lookup):
        source = lookup.last_source
        if source is None:
            raise ValueError("prompt lookup proposal has no bound source")
        if source[0] == "target":
            match = source[1]
        elif source[0] == "hot":
            match = source[2]
        else:
            raise ValueError("prompt lookup proposal has an unknown source")
        return f"prompt_lookup:match_{int(match)}"

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
                "selection": "highest_lagged_committed_token_acceptance_then_longest_span",
                "score_prior": {
                    "acceptance": self._SCORE_PRIOR_ACCEPTANCE,
                    "strength_tokens": self._SCORE_PRIOR_STRENGTH,
                },
                "stochastic_safety": "source_selected_before_current_external_draw",
                "trusted_pld": {
                    "enabled": self.policy.trusted_pld,
                    "approximate": True,
                    "min_score": self.policy.trusted_pld_min_score,
                    "min_verified_tokens": self.policy.trusted_pld_min_verified_tokens,
                    "min_match": self.policy.trusted_pld_min_match,
                    "max_span": self.policy.trusted_pld_max_span,
                    "audit_interval": self.policy.trusted_pld_audit_interval,
                    "execution": "teacher_force_target_body_project_bonus_row_only",
                    "rng": "copied_rows_consume_no_target_draws",
                    "scope": "B1_unprocessed_rows_only",
                },
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
            "score_counts": {
                key: {"accepted_tokens": value[0], "proposed_tokens": value[1]}
                for key, value in sorted(self._score_counts.items())
            },
            "trusted_pld": dict(self._trusted_stats),
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
        # only reads sources that start inside the last ``lookback`` tokens.
        # Their keys reach back at most ``ngram_max`` tokens and the context
        # check at most ``min_context_match`` tokens, so indexing that window
        # proposes exactly what the full history would, at O(lookback) per
        # row-round instead of O(context).  The index is rebuilt every round,
        # so it carries no rejected-source feedback (``reject_ttl=0`` says
        # so); source acceptance is learned only by the lagged committed
        # scorer (``observe_proposal_feedback``).
        window = self.policy.lookback + max(
            self.policy.ngram_max, self.policy.min_context_match
        )
        candidates = []
        for history, anchor in zip(histories, anchors):
            copied = []
            row = []
            if self.policy.prompt_lookup and proposal_length:
                lookup = IndexedPromptLookup(
                    [*history[-(window - 1):], int(anchor)],
                    ngram_min=self.policy.ngram_min,
                    ngram_max=self.policy.ngram_max,
                    reject_ttl=0,
                )
                copied = lookup.propose(
                    proposal_length,
                    lookback=self.policy.lookback,
                    min_context_match=self.policy.min_context_match,
                    max_sources=self.policy.max_sources,
                )
                if copied:
                    row.append(
                        (
                            "prompt_lookup",
                            list(copied),
                            self._prompt_lookup_score_key(lookup),
                        )
                    )
            if (
                proposal_length
                and self.policy.native_mtp
                and 0 < len(history) <= self.policy.mtp_max_history
            ):
                mtp = list(
                    self.native_mtp_source(history, int(anchor), proposal_length)
                )
                if not mtp:
                    raise ValueError("native MTP source returned an empty proposal")
                row.append(("native_mtp", mtp, "native_mtp"))
            if proposal_length:
                # Its actual chain is intentionally unavailable until after
                # this choice.  Declared depth is the only stochastic-safe
                # current-round span feature.
                row.append(("external", None, "external"))
            calibration = next(
                (
                    candidate
                    for candidate in row
                    if candidate[0] == "prompt_lookup"
                    and self.policy.trusted_pld
                    and self._score_counts.get(candidate[2], (0, 0))[1]
                    < self.policy.trusted_pld_min_verified_tokens
                ),
                None,
            )
            selected = (
                calibration
                if calibration is not None
                else self._select_candidate(row, proposal_length)
                if row
                else ("external", None, "external")
            )
            candidates.append(selected)
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
        score_keys, scores, trusted_rows, audit_rows = [], [], [], []
        for row, (source, copied, score_key) in enumerate(candidates):
            score = self._proposal_score(score_key)
            trusted = False
            audited = False
            if source == "prompt_lookup" and copied and self.policy.trusted_pld:
                match = int(score_key.rsplit("_", 1)[1])
                _accepted, verified = self._score_counts.get(score_key, (0, 0))
                calibrating = verified < self.policy.trusted_pld_min_verified_tokens
                eligible = (
                    len(anchors) == 1
                    and not (logits_processors or [[]])[row]
                    and score >= self.policy.trusted_pld_min_score
                    and verified >= self.policy.trusted_pld_min_verified_tokens
                    and match >= self.policy.trusted_pld_min_match
                    and len(copied) <= self.policy.trusted_pld_max_span
                )
                if calibrating:
                    audited = True
                elif eligible:
                    self._trusted_pld_selections += 1
                    interval = self.policy.trusted_pld_audit_interval
                    audited = bool(
                        interval and self._trusted_pld_selections % interval == 0
                    )
                    trusted = not audited
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
            score_keys.append(score_key)
            scores.append(score)
            trusted_rows.append(trusted)
            audit_rows.append(audited)
        self.adaptive_confidence_features = confidence
        self.last_proposal_sources = tuple(sources)
        self.last_proposal_score_keys = tuple(score_keys)
        self.last_proposal_scores = tuple(scores)
        self.last_proposal_trusted = tuple(trusted_rows)
        self.last_proposal_audits = tuple(audit_rows)
        if proposal_length:
            for source in sources:
                self.composition_stats[source] += 1
        return tokens, laws
