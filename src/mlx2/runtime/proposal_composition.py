"""Exact host arbitration of external, copy and MTP proposals.

The external backbone still advances its committed feature cache for every
row. Arbitration changes only selected rows: copy/MTP tokens carry their
actual point-mass q, while untouched stochastic rows retain the backend's
exact proposal laws.  The external verifier remains the sole owner of
accepted state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .prompt_lookup import IndexedPromptLookup


@dataclass(frozen=True)
class PromptCopySourceRange:
    """Executor-owned role/origin evidence for a committed token range.

    Offsets refer to the full processor history plus the current anchor, not
    the bounded lookup window. This is an internal contract, never inferred
    from token contents or a client assertion. Only assistant-origin copies
    can currently bypass exact target verification.
    """

    origin: str
    role: str
    start: int
    end: int

    def __post_init__(self):
        if (not isinstance(self.origin, str) or not self.origin
                or not isinstance(self.role, str)
                or self.role not in {"assistant", "user", "tool", "system"}
                or type(self.start) is not int or type(self.end) is not int
                or not 0 <= self.start < self.end):
            raise ValueError("invalid prompt-copy source range evidence")


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
    score_decay_per_token: float = 0.9
    trusted_pld: bool = False
    trusted_pld_min_score: float = 0.9
    trusted_pld_min_verified_tokens: int = 32
    trusted_pld_min_match: int = 4
    trusted_pld_max_span: int = 8
    # Trusted copies remain default-off.  When explicitly enabled, every
    # eighth otherwise-trusted selection is target-audited.  Zero may be kept
    # in an inactive policy for compatibility, but cannot enable trusted PLD.
    trusted_pld_audit_interval: int = 8

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
        if (
            isinstance(policy.score_decay_per_token, bool)
            or not isinstance(policy.score_decay_per_token, (int, float))
            or not math.isfinite(float(policy.score_decay_per_token))
            or not 0.0 <= float(policy.score_decay_per_token) < 1.0
        ):
            raise ValueError(
                "proposal_composition score_decay_per_token must be finite in [0, 1)"
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
        if policy.trusted_pld and policy.trusted_pld_audit_interval < 1:
            raise ValueError("trusted PLD requires a positive audit interval")
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
    # Trusted PLD is B1-only.  The executor may split one cohort into several
    # draft groups (by pending-tail length), so ``len(anchors)`` is not the
    # verify width; the executor passes the physical cohort width.
    requires_cohort_width = True
    # The executor must pass one mutable, rollback-snapshotted score table per
    # request lane. Serving never uses adapter-global score state.
    requires_proposal_score_states = True
    # Audit cadence, revocation and recovery are request state, never resident
    # adapter state.  The executor snapshots this mapping with its lane.
    requires_proposal_trust_states = True
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
        self.last_proposal_trust_keys = ()
        self.last_proposal_trusted = ()
        self.last_proposal_audits = ()
        self.last_proposal_periodic_audits = ()
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

    def _proposal_score(self, key, score_state):
        accepted, proposed = score_state.get(key, (0, 0))
        self._validate_score_counts(accepted, proposed)
        strength = self._SCORE_PRIOR_STRENGTH
        return (
            accepted + self._SCORE_PRIOR_ACCEPTANCE * strength
        ) / (proposed + strength)

    @staticmethod
    def _validate_score_counts(accepted, proposed):
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (accepted, proposed)) or not 0 <= accepted <= proposed:
            raise ValueError("invalid committed proposal score evidence")

    def _select_candidate(self, candidates, external_span, score_state=None):
        """Rank by committed score, usable span, then stable source order."""
        score_state = {} if score_state is None else score_state
        return min(
            candidates,
            key=lambda item: (
                -self._proposal_score(item[2], score_state),
                -(len(item[1]) if item[1] is not None else external_span),
                self._SOURCE_ORDER[item[0]],
            ),
        )

    @staticmethod
    def observe_proposal_feedback(
        score_state, key, proposed, accepted, decay_per_token=0.9
    ):
        """Publish committed verifier labels into the owning request lane."""
        if (
            not isinstance(key, str)
            or not key
            or type(proposed) is not int
            or type(accepted) is not int
            or not 0 <= accepted <= proposed
            or proposed < 1
        ):
            raise ValueError("invalid committed proposal-composition feedback")
        if not isinstance(score_state, dict):
            raise ValueError("proposal feedback requires a lane-owned score state")
        if (isinstance(decay_per_token, bool)
                or not isinstance(decay_per_token, (int, float))
                or not math.isfinite(decay_per_token)
                or not 0 <= decay_per_token < 1):
            raise ValueError("invalid proposal score decay")
        for counts in score_state.values():
            if not isinstance(counts, (tuple, list)) or len(counts) != 2:
                raise ValueError("invalid committed proposal score evidence")
            ComposedDraftModel._validate_score_counts(*counts)
        # Decay by proposed-token mass so different K values have comparable
        # memory. Values are weighted effective token counts.
        decay = float(decay_per_token) ** proposed
        # Time advances for every source in this lane, including a source the
        # current ranking did not choose. Otherwise stale high-confidence
        # buckets could permanently prevent a changed source from competing.
        for source_key, (old_accepted, old_proposed) in list(score_state.items()):
            score_state[source_key] = (old_accepted * decay, old_proposed * decay)
        old_accepted, old_proposed = score_state.get(key, (0, 0))
        score_state[key] = (
            min(1000.0, old_accepted + accepted),
            min(1000.0, old_proposed + proposed),
        )

    def observe_trusted_pld(
        self,
        trust_state,
        key,
        *,
        proposed,
        audited,
        periodic_audit=False,
        accepted=0,
    ):
        if (
            not isinstance(trust_state, dict)
            or not isinstance(key, str)
            or not key
            or type(proposed) is not int
            or proposed < 1
            or type(audited) is not bool
            or type(periodic_audit) is not bool
            or periodic_audit and not audited
            or type(accepted) is not int
            or not 0 <= accepted <= proposed
        ):
            raise ValueError("invalid trusted PLD observation")
        defaults = {
            "blind_rounds": 0,
            "blind_tokens": 0,
            "blind_since_audit": 0,
            "audit_rounds": 0,
            "audit_proposed_tokens": 0,
            "audit_accepted_tokens": 0,
            "audit_failures": 0,
            "periodic_audit_rounds": 0,
            "revoked": False,
            # Active trust evidence contains exact target-audited labels only.
            # It is reset on every mismatch; ranking EWMA never enters here.
            "exact_accepted_tokens": 0,
            "exact_proposed_tokens": 0,
        }
        record = trust_state.setdefault(key, dict(defaults))
        if not isinstance(record, dict) or any(
            name != "revoked" and (type(record.get(name, value)) is not int
                                  or record.get(name, value) < 0)
            for name, value in defaults.items()
        ) or type(record.get("revoked", False)) is not bool:
            # Corrupted feedback cannot bootstrap blind eligibility. A fresh
            # exact audit can rebuild evidence without affecting other lanes.
            record = dict(defaults)
            trust_state[key] = record
        for name, value in defaults.items():
            record.setdefault(name, value)
        if audited:
            record["blind_since_audit"] = 0
            record["audit_rounds"] += 1
            record["audit_proposed_tokens"] += proposed
            record["audit_accepted_tokens"] += accepted
            record["periodic_audit_rounds"] += int(periodic_audit)
            self._trusted_stats["audit_rounds"] += 1
            self._trusted_stats["audit_proposed_tokens"] += proposed
            self._trusted_stats["audit_accepted_tokens"] += accepted
            if accepted != proposed:
                # Any target disagreement immediately revokes blind trust for
                # this request and source bucket.  Exact audits continue until
                # a fresh, bounded body of fully accepted target labels has
                # rebuilt the minimum evidence.
                record["audit_failures"] += 1
                record["revoked"] = True
                record["exact_accepted_tokens"] = 0
                record["exact_proposed_tokens"] = 0
            else:
                record["exact_accepted_tokens"] += accepted
                record["exact_proposed_tokens"] += proposed
                if self._trusted_evidence_sufficient(record):
                    record["revoked"] = False
        else:
            record["blind_since_audit"] += 1
            record["blind_rounds"] += 1
            record["blind_tokens"] += proposed
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

    @staticmethod
    def _trusted_pld_key(score_key, copied_span):
        """Bind trust to both source-match strength and copied span length."""
        return f"{score_key}:span_{int(copied_span)}"

    @classmethod
    def _copy_trust_source(cls, lookup, copied, history_length, ranges):
        """Return a range/content-bound key, or None for unproven copies."""
        source = lookup.last_source
        if not source or source[0] != "target" or ranges is None:
            return None
        _, match, start, _continuation = source
        begin = history_length + 1 - len(lookup.tokens) + start + match
        end = begin + len(copied)
        # A containing assistant declaration cannot override an overlapping
        # user/tool/system declaration, nor a second partially covering owner.
        # Trust requires exactly one declared owner of every copied token.
        owners = [item for item in ranges if item.start < end and begin < item.end]
        if (len(owners) != 1 or owners[0].role != "assistant"
                or not (owners[0].start <= begin and end <= owners[0].end)):
            return None
        owner = owners[0]
        digest = hashlib.sha256(json.dumps(
            [owner.origin, owner.role, owner.start, owner.end, begin, end, list(copied)],
            separators=(",", ":"),
        ).encode()).hexdigest()
        return f"{cls._trusted_pld_key(cls._prompt_lookup_score_key(lookup), len(copied))}:source_{digest}"

    def _trusted_evidence_sufficient(self, record):
        """Whether exact lane-owned evidence meets the configured thresholds."""
        if not isinstance(record, dict):
            return False
        proposed = record.get("exact_proposed_tokens", 0)
        accepted = record.get("exact_accepted_tokens", 0)
        # Even an explicit zero-token threshold cannot make a cold bucket
        # trusted: at least one exact target label is always required.
        required = max(1, self.policy.trusted_pld_min_verified_tokens)
        return (
            type(proposed) is int
            and type(accepted) is int
            and proposed >= required
            and 0 <= accepted <= proposed
            and accepted / proposed >= self.policy.trusted_pld_min_score
        )

    def _trusted_evidence_ready(self, record):
        """Whether exact evidence currently permits blind copying."""
        return (
            type(record.get("revoked", False)) is bool
            and not record.get("revoked", False)
            and type(record.get("blind_since_audit", 0)) is int
            and record.get("blind_since_audit", 0) >= 0
            and self._trusted_evidence_sufficient(record)
        ) if isinstance(record, dict) else False

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
                    "trust_state": "request_lane_origin_role_range_content",
                    "source_eligibility": "executor_declared_assistant_range_only;unknown_user_tool_system_exact_audit",
                    "source_contract": "PromptCopySourceRange;unplumbed_routes_exact_audit",
                    "qualified": False,
                    "audit_failure": "revoke_until_fresh_exact_recovery",
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
            # These counters belong to the resident adapter, so keep them in
            # an explicitly process-global diagnostic namespace.  Request
            # receipts use the lane-owned committed counters instead.
            "adapter_diagnostics": {
                "scope": "process_global",
                "proposal_attempt_rows": dict(self.composition_stats),
                "trusted_pld": dict(self._trusted_stats),
            },
            "score_counts": {
                # Request-local values are attached to response receipts by
                # ExternalDraftBatchGenerator; never aggregate them here.
                "scope": "request_lane"
            },
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
        proposal_score_states=None,
        proposal_trust_states=None,
        proposal_source_ranges=None,
        cohort_width=None,
        **kwargs,
    ):
        if processor_histories is None or len(processor_histories) != len(anchors):
            raise ValueError(
                "proposal composition requires committed histories per row"
            )
        if proposal_score_states is None:
            # Direct CPU harness compatibility only. Serving declares the
            # requires_proposal_score_states contract and always supplies its
            # lane-owned tables; these transient maps do not survive this call.
            proposal_score_states = [{} for _ in anchors]
        if len(proposal_score_states) != len(anchors) or any(
            not isinstance(state, dict) for state in proposal_score_states
        ):
            raise ValueError("proposal composition requires one lane-owned score state per row")
        if proposal_trust_states is None:
            proposal_trust_states = [{} for _ in anchors]
        if len(proposal_trust_states) != len(anchors) or any(
            not isinstance(state, dict) for state in proposal_trust_states
        ):
            raise ValueError(
                "proposal composition requires one lane-owned trust state per row"
            )
        if cohort_width is None:
            cohort_width = len(anchors)
        if proposal_source_ranges is None:
            proposal_source_ranges = [None for _ in anchors]
        if len(proposal_source_ranges) != len(anchors) or any(
            ranges is not None and (not isinstance(ranges, (list, tuple)) or any(
                not isinstance(item, PromptCopySourceRange) for item in ranges
            )) for ranges in proposal_source_ranges
        ):
            raise ValueError("proposal composition requires executor-owned source ranges per row")
        histories = [list(history) for history in processor_histories]
        if any(ranges is not None and any(item.end > len(history) + 1 for item in ranges)
               for history, ranges in zip(histories, proposal_source_ranges)):
            raise ValueError("prompt-copy source range exceeds committed history")
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
        copy_trust_keys = []
        for row_index, (history, anchor, score_state, trust_state) in enumerate(
            zip(
                histories,
                anchors,
                proposal_score_states,
                proposal_trust_states,
            )
        ):
            copied = []
            copy_trust_key = None
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
                    copy_trust_key = self._copy_trust_source(
                        lookup, copied, len(history), proposal_source_ranges[row_index]
                    )
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
                    and copy_trust_key is not None
                    and cohort_width == 1
                    and not (logits_processors or [[]])[row_index]
                    and int(candidate[2].rsplit("_", 1)[1])
                    >= self.policy.trusted_pld_min_match
                    and len(candidate[1]) <= self.policy.trusted_pld_max_span
                    and not self._trusted_evidence_ready(
                        trust_state.get(
                            copy_trust_key,
                            {},
                        )
                    )
                ),
                None,
            )
            selected = (
                calibration
                if calibration is not None
                else self._select_candidate(row, proposal_length, score_state)
                if row
                else ("external", None, "external")
            )
            candidates.append(selected)
            copy_trust_keys.append(copy_trust_key)
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
        score_keys, scores, trust_keys, trusted_rows, audit_rows = [], [], [], [], []
        periodic_audit_rows = []
        for row, (source, copied, score_key) in enumerate(candidates):
            score_state = proposal_score_states[row]
            trust_state = proposal_trust_states[row]
            score = self._proposal_score(score_key, score_state)
            trusted = False
            audited = False
            periodic_audit = False
            trust_key = None
            if source == "prompt_lookup" and copied and self.policy.trusted_pld:
                match = int(score_key.rsplit("_", 1)[1])
                source_eligible = copy_trust_keys[row] is not None
                trust_key = copy_trust_keys[row] or (
                    self._trusted_pld_key(score_key, len(copied)) + ":source_unknown"
                )
                trust_record = trust_state.get(trust_key, {})
                static_eligible = (
                    source_eligible
                    and
                    cohort_width == 1
                    and not (logits_processors or [[]])[row]
                    and match >= self.policy.trusted_pld_min_match
                    and len(copied) <= self.policy.trusted_pld_max_span
                )
                ready = self._trusted_evidence_ready(trust_record)
                if not source_eligible or static_eligible and not ready:
                    # Cold and revoked buckets always take the exact target
                    # verifier. Ranking EWMA is intentionally irrelevant.
                    audited = True
                elif static_eligible:
                    interval = self.policy.trusted_pld_audit_interval
                    periodic_audit = (
                        int(trust_record.get("blind_since_audit", 0))
                        >= interval - 1
                    )
                    audited = periodic_audit
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
            trust_keys.append(trust_key)
            trusted_rows.append(trusted)
            audit_rows.append(audited)
            periodic_audit_rows.append(periodic_audit)
        self.adaptive_confidence_features = confidence
        self.last_proposal_sources = tuple(sources)
        self.last_proposal_score_keys = tuple(score_keys)
        self.last_proposal_scores = tuple(scores)
        self.last_proposal_trust_keys = tuple(trust_keys)
        self.last_proposal_trusted = tuple(trusted_rows)
        self.last_proposal_audits = tuple(audit_rows)
        self.last_proposal_periodic_audits = tuple(periodic_audit_rows)
        if proposal_length:
            for source in sources:
                self.composition_stats[source] += 1
        return tokens, laws
