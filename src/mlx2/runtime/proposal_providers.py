"""Loaded-session complete continuation providers and external route seam."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace

CONTINUATION_VERIFICATION_ALGORITHM = "stable-truncated-prefix-dedup-v2"
LONGEST_FIRST_EXACT_PREFIX = "longest-first-exact-prefix-v1"
CONTINUATION_VERIFICATION_ALGORITHMS = frozenset(
    {CONTINUATION_VERIFICATION_ALGORITHM, LONGEST_FIRST_EXACT_PREFIX}
)


@dataclass(frozen=True)
class PointMassLaw:
    """Sparse exact proposal law used by continuation-selected chains."""

    token: int


class InsufficientContinuationAlternatives(ValueError):
    """The selected complete-path slate cannot form the requested AST."""


def continuation_context_revision(binding, history, anchor):
    return hashlib.sha256(
        json.dumps(
            [binding, list(history), int(anchor)], separators=(",", ":")
        ).encode()
    ).hexdigest()


def _continuation_confidence_features(probabilities, depth):
    if (
        not isinstance(probabilities, tuple)
        or len(probabilities) != depth
        or any(
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            for probability in probabilities
        )
    ):
        raise ValueError("selected continuation path lacks valid per-token confidence")
    floor = math.exp(-40.0)
    return tuple(
        max(
            -40.0,
            min(
                40.0,
                math.log(max(probability, floor))
                - math.log(max(1.0 - probability, floor)),
            ),
        )
        for probability in probabilities
    )


@dataclass(frozen=True)
class ContinuationPoolPolicy:
    sources: tuple = ("external", "prompt_lookup")
    limit: int = 15
    ngram_min: int = 3
    ngram_max: int = 6
    lookback: int = 4096
    mtp_max_history: int = 4096
    verification_algorithm: str = CONTINUATION_VERIFICATION_ALGORITHM
    admission: dict | None = None
    ast_planner: bool = False
    packed_pattern: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.packed_pattern is not None and (
            not isinstance(self.packed_pattern, tuple)
            or not 2 <= len(self.packed_pattern) <= 5
            or any(type(n) is not int or not 1 <= n <= 15 for n in self.packed_pattern)
            or sum(self.packed_pattern) > 15
            or self.limit != len(self.packed_pattern)
            or self.sources != ("external",) or not self.ast_planner
        ):
            raise ValueError("packed_pattern requires an external AST pool with at most15 proposal rows")
        if type(self.ast_planner) is not bool:
            raise ValueError("continuation_pool ast_planner must be boolean")
        if (
            type(self.verification_algorithm) is not str
            or self.verification_algorithm not in CONTINUATION_VERIFICATION_ALGORITHMS
        ):
            raise ValueError("unsupported continuation_pool verification_algorithm")
        from .drafter_portfolio import PortfolioAdmissionPolicy

        PortfolioAdmissionPolicy.from_value(self.admission)

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict):
            raise ValueError("continuation_pool must be a JSON object")  # noqa: TRY004
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unknown continuation_pool policy keys")
        fields = dict(value)
        if "packed_pattern" in fields and fields["packed_pattern"] is not None:
            if not isinstance(fields["packed_pattern"], (list, tuple)):
                raise ValueError("packed_pattern must be a depth list")
            fields["packed_pattern"] = tuple(fields["packed_pattern"])
        if "sources" in fields:
            if not isinstance(fields["sources"], (list, tuple)):
                raise ValueError("continuation_pool sources must be a list")
            fields["sources"] = tuple(fields["sources"])
        policy = cls(**fields)
        if (
            not policy.sources
            or any(type(source) is not str for source in policy.sources)
            or len(set(policy.sources)) != len(policy.sources)
            or any(
                re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", source) is None
                for source in policy.sources
            )
        ):
            raise ValueError(
                "continuation_pool requested an unavailable or unknown source"
            )
        if type(policy.limit) is not int or not 1 <= policy.limit <= 15:
            raise ValueError("continuation_pool limit must be in 1..15 complete paths")
        for name in ("ngram_min", "ngram_max", "lookback", "mtp_max_history"):
            if type(getattr(policy, name)) is not int or getattr(policy, name) < 1:
                raise ValueError(f"continuation_pool {name} must be positive integer")
        if policy.ngram_min > policy.ngram_max:
            raise ValueError("continuation_pool ngram range is invalid")
        return policy

    def as_dict(self):
        result = {
            name: copy.deepcopy(getattr(self, name))
            for name in self.__dataclass_fields__
        }
        # Preserve the established v2 session identity for policies that do
        # not opt into portfolio admission.  A configured admission policy is
        # revision-bound because it changes which providers may run.
        if result["admission"] is None:
            del result["admission"]
        # The new planner is explicitly selected until its real-artifact
        # performance/correctness gates justify promoting a default.
        if not result["ast_planner"]:
            del result["ast_planner"]
        if result["packed_pattern"] is None:
            del result["packed_pattern"]
        return result


@dataclass(frozen=True)
class ContinuationContext:
    history: tuple[int, ...]
    anchor: int
    depth: int
    draft_cache: object
    pending_features: object
    processors: tuple = ()
    node_budget: int | None = None
    draft_lattice: object = None
    tile_policy: str | None = None


class PromptLookupContinuationSource:
    mechanism = "prompt_lookup"

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, context, limit):
        from ..adapters.proposal_path_sources import SourceContinuation

        tokens = (*context.history, context.anchor)
        candidates = {}
        for size in range(
            min(self.policy.ngram_max, len(tokens)), self.policy.ngram_min - 1, -1
        ):
            suffix = tokens[-size:]
            for start in range(
                max(0, len(tokens) - self.policy.lookback), len(tokens) - size
            ):
                if tokens[start : start + size] != suffix:
                    continue
                path = tuple(tokens[start + size : start + size + context.depth])
                if path:
                    candidates[path] = max(candidates.get(path, 0), size)
        ordered = sorted(
            candidates, key=lambda path: (-candidates[path], -len(path), path)
        )[:limit]
        return [
            SourceContinuation(
                path,
                tuple(None for _ in path),
                float(candidates[path]),
                "exact_copy_ngram_match_length_proxy",
            )
            for path in ordered
        ]


class ContinuationDraftModel:
    """Normal draft context advance plus a ranked pool of complete paths.

    Each model provider evaluates private state copied before the normal call.
    The public cache advances once. The verifier may expand selected complete
    paths into one target batch; scores remain proposal ranking features.
    """

    requires_processor_histories = True
    requires_proposal_contexts = True
    supports_logits_processors = True
    proposal_distribution = "deterministic_point_mass"

    def __init__(self, backend, policy, *, session, source_records, providers):
        from .drafter_portfolio import DrafterAdmissionPlanner
        from .proposal_pool import ProposalPool, shared_ranking_registry

        self.backend = backend
        self.policy = ContinuationPoolPolicy.from_value(policy)
        if self.policy.packed_pattern is not None and not callable(getattr(backend, "draft_packed_paths", None)):
            raise ValueError("packed_pattern requires a full-depth packed proposal source")
        if (
            getattr(backend, "proposal_distribution", None)
            != self.proposal_distribution
            and not callable(getattr(backend, "propose_tree", None))
        ):
            raise ValueError(
                "continuation pool requires deterministic or bound tree proposal backbone"
            )
        if set(providers) != set(self.policy.sources) or set(source_records) != set(
            providers
        ):
            raise ValueError(
                "continuation source inventory does not match requested sources"
            )
        if any(not callable(provider) for provider in providers.values()):
            raise ValueError(
                "continuation source inventory contains an unavailable provider"
            )
        self.session, self.providers, self.source_records = (
            session,
            dict(providers),
            dict(source_records),
        )
        self.proposal_pool = ProposalPool(
            session,
            tuple(source_records.values()),
            ranking_registry=shared_ranking_registry(),
        )
        self.last_continuation_selections = ()
        self.last_continuation_admissions = ()
        self.last_continuation_slates = ()
        self.last_proposal_sources = ()
        self.last_ordered_alternative_provenance = ()
        self.last_ordered_alternative_prefix_tokens = ()
        self.last_ordered_alternative_search_diagnostics = ()
        self.last_continuation_confidence_provenance = ()
        self.last_continuation_source_diagnostics = ()
        self.adaptive_confidence_features = None
        self.admission_planner = DrafterAdmissionPlanner(self.policy.admission)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    @property
    def receipt_settings(self):
        return {
            **copy.deepcopy(self.backend.receipt_settings),
            "continuation_pool": {
                **self.policy.as_dict(),
                "unit": "complete_continuation_sequences",
                "session_revision": self.session.session_revision,
                "sources": {
                    name: {
                        "mechanism": source.mechanism,
                        "revision": source.revision,
                        "loaded": source.loaded,
                        "maximum_proposal_length": source.max_depth,
                        "qualified": False,
                    }
                    for name, source in self.source_records.items()
                },
                "scores": "source_private_ranking_proxies_not_verification_q",
                "adaptive_confidence": (
                    "selected_path_lagged_conditional_acceptance_"
                    "log_odds_clamped_40"
                ),
                "cache": ("packed_full_depth_private_branches_verified_head_advanced_once"
                          if self.policy.packed_pattern else
                          "private_provider_recompute_public_backbone_advanced_once"),
                "admission": self.admission_planner.policy.as_dict(),
                "qualified": False,
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
        processor_histories=None,
        proposal_contexts=None,
        logits_processors=None,
        ast_source_budgets=None,
        ast_tree_node_budgets=None,
        ast_b1_tile_policy=None,
        **kwargs,
    ):
        from .proposal_pool import PoolRound, ProposalPath
        if ast_b1_tile_policy is not None and (
            ast_b1_tile_policy != "four_path15" or len(anchors) != 1
            or proposal_length != 15 or ast_source_budgets != [4]
            or not getattr(self.backend, "supports_continuation_tree_reuse", False)
            or any(not getattr(processor, "history_pure", False)
                   or not callable(getattr(processor, "forbidden_token_ids_at_length", None))
                   for row in (logits_processors or ()) for processor in row)
        ):
            raise ValueError("four_path15 source requires one unconstrained 15-position lattice")

        if (
            processor_histories is None
            or proposal_contexts is None
            or len(processor_histories) != len(anchors)
            or len(proposal_contexts) != len(anchors)
        ):
            raise ValueError(
                "continuation pool requires request-bound proposal contexts"
            )
        if ast_source_budgets is not None and (
            len(ast_source_budgets) != len(anchors)
            or any(type(n) is not int or not 1 <= n <= self.policy.limit
                   for n in ast_source_budgets)
        ):
            raise ValueError("AST source budgets must cover every request within the pool limit")
        if ast_tree_node_budgets is not None and (
            ast_source_budgets is None
            or len(ast_tree_node_budgets) != len(anchors)
            or any(type(n) is not int or not 1 <= n <= 60 for n in ast_tree_node_budgets)
        ):
            raise ValueError("AST tree node budgets must be bounded per request")
        self.last_continuation_selections = ()
        self.last_continuation_admissions = ()
        self.last_continuation_slates = ()
        self.last_proposal_sources = ()
        self.last_ordered_alternative_provenance = ()
        self.last_continuation_confidence_provenance = ()
        self.last_requested_proposal_length = int(proposal_length)
        # A batched cache exposes authoritative rows, while plain B1 cache does
        # not. Snapshot before the external context append changes its offset.
        reusable = bool(getattr(self.backend, "supports_continuation_tree_reuse", False))
        packed = self.policy.packed_pattern if proposal_length else None
        if packed is not None and (
            len(anchors) != 1 or proposal_length != max(packed)
            or ast_source_budgets != [len(packed)]
        ):
            raise ValueError("packed source must retain its exact B1 proposal geometry")
        private_needed = [packed is None and not (ast_source_budgets is not None and
                              "external" in self.source_records and
                              (ast_source_budgets[row] == 1 or
                               reusable and (
                                   ast_tree_node_budgets[row]
                                   if ast_tree_node_budgets is not None
                                   else ast_source_budgets[row]
                               ) >= proposal_length))
                          for row in range(len(anchors))]
        if all(hasattr(layer, "rows") for layer in cache):
            private_rows = [
                copy.deepcopy([layer.rows[row] for layer in cache]) if private_needed[row] else None
                for row in range(len(anchors))
            ]
        elif len(anchors) == 1:
            private_rows = [copy.deepcopy(cache) if private_needed[0] else None]
        else:
            raise ValueError(
                "continuation pool requires individually owned draft cache rows"
            )
        contexts = []
        for row, identity in enumerate(proposal_contexts):
            if (
                not isinstance(identity, dict)
                or not {"request_id", "round_id", "context_revision"} <= set(identity)
                or set(identity)
                - {"request_id", "round_id", "context_revision", "session_scope_hash"}
                or type(identity["request_id"]) is not str
                or not identity["request_id"]
                or type(identity["round_id"]) is not int
                or identity["round_id"] < 0
                or type(identity["context_revision"]) is not str
                or not identity["context_revision"]
            ):
                raise ValueError("invalid continuation proposal identity")
            expected = continuation_context_revision(
                self.session.session_revision, processor_histories[row], anchors[row]
            )
            if identity["context_revision"] != expected:
                raise ValueError("continuation proposal context revision mismatch")
            contexts.append(
                ContinuationContext(
                    tuple(processor_histories[row]),
                    int(anchors[row]),
                    proposal_length,
                    private_rows[row],
                    hidden[row : row + 1],
                    tuple((logits_processors or [[] for _ in anchors])[row]),
                )
            )
        from .speculative_sampling import RequestRNG

        private_rngs = []
        for identity in proposal_contexts:
            seed = int.from_bytes(
                hashlib.sha256(
                    json.dumps(
                        [
                            self.session.session_revision,
                            identity["request_id"],
                            identity["round_id"],
                            identity["context_revision"],
                            "continuation-backbone-private-rng-v1",
                        ],
                        separators=(",", ":"),
                    ).encode()
                ).digest()[:8],
                "big",
            )
            private_rngs.append(RequestRNG(seed))
        capture_tree = bool(ast_source_budgets is not None and
                            any(n > 1 for n in ast_source_budgets) and
                            getattr(self.backend, "supports_continuation_tree_reuse", False))
        try:
            if packed is not None:
                packed_candidates = self.backend.draft_packed_paths(
                    anchors, hidden, cache, packed,
                    processor_histories=processor_histories,
                    logits_processors=logits_processors, **kwargs,
                )
                backbone_tokens = [list(row[0].tokens) for row in packed_candidates]
                _backbone_laws = [[PointMassLaw(token) for token in row] for row in backbone_tokens]
                lattice = None
            else:
                backbone_tokens, _backbone_laws = self.backend.draft_distributions(
                anchors, hidden, cache, proposal_length, private_rngs, temperatures,
                processor_histories=processor_histories,
                logits_processors=None if ast_b1_tile_policy == "four_path15" else logits_processors,
                **({"capture_continuation_tree": True} if capture_tree else {}),
                **({"capture_lattice_only": True} if ast_b1_tile_policy == "four_path15" else {}),
                **kwargs,
            )
                lattice = getattr(self.backend, "last_continuation_lattice", None) if capture_tree else None
        finally:
            if capture_tree:
                object.__setattr__(self.backend, "last_continuation_lattice", None)
        tokens = [None] * len(anchors)
        laws = [None] * len(anchors)
        self.draft_feedback_payloads = copy.deepcopy(
            getattr(self.backend, "draft_feedback_payloads", None)
        )
        selections, sources, admissions, slates = [], [], [], []
        confidence_rows, confidence_provenance = [], []
        source_diagnostic_rows = []
        for row, (context, identity) in enumerate(zip(contexts, proposal_contexts)):
            if not proposal_length:
                tokens[row] = backbone_tokens[row]
                laws[row] = _backbone_laws[row]
                selections.append(None)
                sources.append("external")
                admissions.append(None)
                slates.append(())
                confidence_rows.append([])
                confidence_provenance.append(None)
                source_diagnostic_rows.append({})
                continue
            source_budget = self.policy.limit if ast_source_budgets is None else ast_source_budgets[row]
            node_budget = (
                ast_tree_node_budgets[row] if ast_tree_node_budgets is not None
                else None if ast_source_budgets is None else source_budget
            )
            context = replace(context, node_budget=node_budget, tile_policy=ast_b1_tile_policy)
            if lattice is not None and (node_budget or source_budget) >= proposal_length:
                if lattice["anchors"] != tuple(int(a) for a in anchors) or lattice["depth"] != proposal_length:
                    raise ValueError("DFlash continuation lattice does not belong to this proposal call")
                context = replace(context, draft_lattice={
                    "anchors": [int(anchors[row])], "depth": proposal_length,
                    "pending": tuple(value[row:row + 1] for value in lattice["pending"]),
                })
            round = PoolRound(
                self.session.session_revision,
                identity["context_revision"],
                identity["request_id"],
                identity["round_id"],
                **(
                    {"session_scope_hash": identity["session_scope_hash"]}
                    if "session_scope_hash" in identity
                    else {}
                ),
            )
            paths = []
            scope = (
                identity.get("session_scope_hash")
                or hashlib.sha256(
                    json.dumps(
                        [self.session.session_revision, identity["request_id"]],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
            )
            predictions = [
                self.proposal_pool.source_prediction(
                    source_id,
                    scope,
                    min(proposal_length, record.max_depth),
                )
                for source_id, record in self.source_records.items()
            ]
            admission = self.admission_planner.choose(
                identity["request_id"], predictions
            )
            predictions_by_source = {item["source_id"]: item for item in predictions}
            admissions.append(admission)
            row_source_diagnostics = {}

            def invoke(source_id):
                provider = self.providers[source_id]
                candidates = (tuple(packed_candidates[row]) if packed is not None
                              else tuple(provider(context, source_budget)))
                admission.setdefault("observed_invoked_sources", []).append(source_id)
                diagnostics = getattr(provider, "last_diagnostics", None)
                if diagnostics is not None:
                    row_source_diagnostics[source_id] = copy.deepcopy(diagnostics)
                return candidates

            use_backbone = (ast_source_budgets is not None and source_budget == 1
                            and "external" in self.source_records)
            if use_backbone:
                admission["ast_source_budget"] = source_budget
                admission["observed_invoked_sources"] = []
                admission["observed_backbone_source"] = "external"
                record = self.source_records["external"]
                paths.append(ProposalPath(
                    candidate_id=f"external:{row}:planned-primary",
                    source_id=record.source_id, source_revision=record.revision,
                    session_revision=self.session.session_revision,
                    context_revision=round.context_revision,
                    tokens=tuple(int(t) for t in backbone_tokens[row]),
                    confidence_features=tuple(None for _ in backbone_tokens[row]),
                    ranking_score=0.0, ranking_convention="ast_planned_backbone_primary",
                ))
                row_source_diagnostics["external"] = {
                    "mechanism": "planned_backbone_primary", "node_budget": 1,
                    "private_head_calls": 0, "backbone_reused": True,
                }
            for source_id in () if use_backbone else admission["admitted"]:
                record = self.source_records[source_id]
                for ordinal, candidate in enumerate(invoke(source_id)):
                    paths.append(
                        ProposalPath(
                            candidate_id=f"{source_id}:{row}:{ordinal}",
                            source_id=record.source_id,
                            source_revision=record.revision,
                            session_revision=self.session.session_revision,
                            context_revision=round.context_revision,
                            tokens=candidate.tokens,
                            confidence_features=candidate.confidence_features,
                            ranking_score=candidate.ranking_score,
                            ranking_convention=candidate.ranking_convention,
                        )
                    )
            # PLD is the bounded last-resort provider when selected model
            # drafters return no complete path.  This is decided after empty
            # provider output, not by comparing current private scores.
            fallback_id = next(
                (
                    source_id
                    for source_id, record in self.source_records.items()
                    if record.fallback and source_id not in admission["admitted"]
                ),
                None,
            )
            if (
                not paths
                and fallback_id is not None
                and self.admission_planner.policy.pld_fallback
            ):
                record = self.source_records[fallback_id]
                reserved = self.admission_planner.reserve_runtime_provider(
                    admission, predictions_by_source[fallback_id],
                    reason="runtime_fallback",
                )
                if reserved is not None:
                    admission = reserved
                    admissions[-1] = admission
                    for ordinal, candidate in enumerate(invoke(fallback_id)):
                        paths.append(
                            ProposalPath(
                                candidate_id=f"{fallback_id}:{row}:{ordinal}",
                                source_id=record.source_id,
                                source_revision=record.revision,
                                session_revision=self.session.session_revision,
                                context_revision=round.context_revision,
                                tokens=candidate.tokens,
                                confidence_features=candidate.confidence_features,
                                ranking_score=candidate.ranking_score,
                                ranking_convention=candidate.ranking_convention,
                            )
                        )
            # A content latch must not make the route unavailable when PLD has
            # no repeated span. Re-probe parked/non-admitted resident heads in
            # the planner's stable inventory order until one produces a
            # complete path. This is still bounded and the receipt names the
            # provider whose work was actually used.
            if not paths:
                for source_id, record in self.source_records.items():
                    if source_id in admission["admitted"] or record.fallback:
                        continue
                    reserved = self.admission_planner.reserve_runtime_provider(
                        admission, predictions_by_source[source_id],
                        reason="runtime_reprobe",
                    )
                    if reserved is None:
                        continue
                    admission = reserved
                    admissions[-1] = admission
                    candidates = invoke(source_id)
                    for ordinal, candidate in enumerate(candidates):
                        paths.append(
                            ProposalPath(
                                candidate_id=f"{source_id}:{row}:{ordinal}",
                                source_id=record.source_id,
                                source_revision=record.revision,
                                session_revision=self.session.session_revision,
                                context_revision=round.context_revision,
                                tokens=candidate.tokens,
                                confidence_features=candidate.confidence_features,
                                ranking_score=candidate.ranking_score,
                                ranking_convention=candidate.ranking_convention,
                            )
                        )
                    if paths:
                        break
            if (
                not any(len(path.tokens) == proposal_length for path in paths)
                and getattr(self.backend, "proposal_distribution", None)
                != self.proposal_distribution
                and len(backbone_tokens[row]) == proposal_length
            ):
                record = self.source_records.get("external")
                if record is None:
                    raise ValueError(
                        "continuation backbone fallback requires the external source"
                    )
                external_paths = tuple(
                    path for path in paths if path.source_id == record.source_id
                )
                if external_paths:
                    conventions = {
                        path.ranking_convention for path in external_paths
                    }
                    if len(conventions) != 1:
                        raise ValueError(
                            "external continuation source mixed ranking conventions"
                        )
                    spine_convention = next(iter(conventions))
                    if spine_convention is None:
                        spine_score = None
                    else:
                        spine_score = math.nextafter(
                            min(path.ranking_score for path in external_paths),
                            -math.inf,
                        )
                        if not math.isfinite(spine_score):
                            raise ValueError(
                                "external continuation ranking has no finite lower spine score"
                            )
                else:
                    spine_convention = "private_rng_complete_spine_fallback"
                    spine_score = 0.0
                paths.append(
                    ProposalPath(
                        candidate_id=f"external:{row}:private-spine",
                        source_id=record.source_id,
                        source_revision=record.revision,
                        session_revision=self.session.session_revision,
                        context_revision=round.context_revision,
                        tokens=tuple(int(token) for token in backbone_tokens[row]),
                        # The spine is a completeness fallback, not ranked
                        # alternative evidence. Neutral confidence keeps the
                        # primary chain usable while the AST slate correctly
                        # falls back when it cannot form distinct branches.
                        confidence_features=tuple(
                            0.5 for _ in backbone_tokens[row]
                        ),
                        ranking_score=spine_score,
                        ranking_convention=spine_convention,
                    )
                )
            if not any(len(path.tokens) == proposal_length for path in paths):
                from .external_speculative import DraftUnavailable

                error = DraftUnavailable(
                    "requested continuation sources produced no valid paths"
                )
                error.failed_rows = (row,)
                raise error
            selection = self.proposal_pool.select(
                paths,
                round,
                limit=self.policy.limit,
                required_primary_depth=proposal_length,
            )
            if not selection.paths:
                from .external_speculative import DraftUnavailable

                self.proposal_pool.discard(selection)
                error = DraftUnavailable(
                    "requested continuation sources produced no valid paths"
                )
                error.failed_rows = (row,)
                raise error
            selections.append(selection)
            self.last_continuation_selections = tuple(selections)
            primary = selection.paths[0]
            confidence_rows.append(
                list(
                    _continuation_confidence_features(
                        primary.conditional_acceptance, len(primary.tokens)
                    )
                )
            )
            confidence_provenance.append(
                {
                    "path_id": primary.path_id,
                    "sources": tuple(
                        sorted(
                            {
                                contributor.source_id
                                for contributor in primary.contributors
                            }
                        )
                    ),
                    "score_authority": primary.score_authority,
                    "feedback_revision": selection.feedback_revision,
                    "ranking_revision": selection.ranking_revision,
                    "feature_transform": "conditional_acceptance_log_odds_clamped_40",
                    "verification_contract": selection.verification_contract,
                }
            )
            source_diagnostic_rows.append(row_source_diagnostics)
            tokens[row] = list(primary.tokens)
            laws[row] = [PointMassLaw(int(token)) for token in primary.tokens]
            # Pool ranking differs from the old greedy row's confidence law.
            source = primary.representative.source_id
            sources.append(
                source if source in {"prompt_lookup", "native_mtp"} else "external"
            )
            slates.append(
                tuple(
                    {
                        "rank": rank,
                        "path_id": ranked.path_id,
                        "depth": len(ranked.tokens),
                        "sources": tuple(
                            sorted(
                                {
                                    contributor.source_id
                                    for contributor in ranked.contributors
                                }
                            )
                        ),
                        "mechanisms": tuple(
                            sorted(
                                {
                                    self.source_records[contributor.source_id].mechanism
                                    for contributor in ranked.contributors
                                }
                            )
                        ),
                        "contributors": len(
                            {
                                contributor.source_id
                                for contributor in ranked.contributors
                            }
                        ),
                        "conditional_acceptance": ranked.conditional_acceptance,
                        "expected_accepted_tokens": ranked.expected_accepted_tokens,
                        "estimated_cost": ranked.estimated_cost,
                        "score": ranked.score,
                        "score_authority": ranked.score_authority,
                        "provider_ordinal": ranked.source_ordinal,
                    }
                    for rank, ranked in enumerate(selection.paths)
                )
            )
        self.last_continuation_selections = tuple(selections)
        self.last_continuation_admissions = tuple(admissions)
        self.last_continuation_slates = tuple(slates)
        self.last_proposal_sources = tuple(sources)
        self.last_continuation_confidence_provenance = tuple(
            confidence_provenance
        )
        self.last_continuation_source_diagnostics = tuple(source_diagnostic_rows)
        self.adaptive_confidence_features = confidence_rows
        return tokens, laws

    def ordered_alternative_proposals(
        self,
        primary_tokens,
        *,
        branch_count,
        strategy,
        shared_prefix_tokens,
        allow_adaptive_prefix_search=False,
        row_indices=None,
    ):
        """Project the ranked complete-path slate into an AST branch layout.

        The proposal pool has already performed cross-provider ranking and
        full-path deduplication.  This bridge only chooses an ordered subset;
        it never re-scores paths or exports a proposal law.  Target draws and
        prefix matching therefore remain the sole verification authority.
        """

        from .proposal_pool import VERIFICATION_CONTRACT
        from .speculative_topology import validate_ordered_alternative_strategy

        self.last_ordered_alternative_provenance = ()
        self.last_ordered_alternative_prefix_tokens = ()
        self.last_ordered_alternative_search_diagnostics = ()
        requested_depth = self.__dict__.get(
            "last_requested_proposal_length",
            max(len(row) for row in primary_tokens),
        )
        if type(requested_depth) is not int or requested_depth < 1:
            raise ValueError("continuation alternative request depth is unavailable")
        if type(branch_count) is not int or not 1 <= branch_count <= 4:
            raise ValueError(
                "continuation alternatives require one through four branches"
            )
        validate_ordered_alternative_strategy(strategy, branch_count)
        selections = self.last_continuation_selections
        if row_indices is not None:
            row_indices = tuple(row_indices)
            if (
                len(row_indices) != len(primary_tokens)
                or len(set(row_indices)) != len(row_indices)
                or any(type(index) is not int or not 0 <= index < len(selections)
                       for index in row_indices)
            ):
                raise ValueError("continuation alternative row selection is invalid")
            selections = tuple(selections[index] for index in row_indices)
        if (
            not isinstance(primary_tokens, (list, tuple))
            or len(primary_tokens) != len(selections)
        ):
            raise ValueError("continuation alternative rows mismatch")
        if isinstance(shared_prefix_tokens, (list, tuple)) and len(
            shared_prefix_tokens
        ) != len(primary_tokens):
            raise ValueError("continuation alternative prefix rows mismatch")

        def row_prefix(row):
            value = (
                shared_prefix_tokens[row]
                if isinstance(shared_prefix_tokens, (list, tuple))
                else shared_prefix_tokens
            )
            return value

        def provenance(path, rank):
            contributors = tuple(
                (
                    contributor.source_id,
                    contributor.source_revision,
                    contributor.candidate_id,
                )
                for contributor in path.contributors
            )
            return {
                "rank": rank,
                "path_id": path.path_id,
                "representative_source": path.representative.source_id,
                "sources": tuple(sorted({item[0] for item in contributors})),
                "contributors": contributors,
            }

        rows = []
        provenance_rows = []
        actual_prefixes = []
        search_diagnostics = []
        adaptive_search_failed = False
        for row, (primary, selection) in enumerate(
            zip(primary_tokens, selections, strict=True)
        ):
            primary = tuple(primary)
            if not primary:
                raise ValueError("continuation alternatives require a nonempty path")
            if selection is None or not selection.paths:
                raise ValueError("continuation selection is unavailable")
            selection.require_verification_contract(VERIFICATION_CONTRACT)

            ranked_all = []
            seen = set()
            for rank, path in enumerate(selection.paths):
                tokens = tuple(path.tokens)
                if not tokens:
                    continue
                if len(tokens) > requested_depth:
                    raise ValueError(
                        "continuation alternative exceeds the requested depth"
                    )
                if tokens in seen:
                    continue
                seen.add(tokens)
                ranked_all.append((tokens, path, rank))
            if not ranked_all or ranked_all[0][0] != primary:
                raise RuntimeError(
                    "continuation ordered B1 path differs from primary proposal"
                )
            # Positive-prefix rows share B1 target work and therefore require
            # one common depth. A zero-trunk AST launch starts every ranked
            # path from the private request base and may be exactly ragged.
            ranked = [
                candidate
                for candidate in ranked_all
                if len(candidate[0]) == len(primary)
            ]

            prefix = row_prefix(row)
            if strategy != "distinct_root" and (
                type(prefix) is not int
                or not (
                    0 <= prefix < len(primary)
                    if strategy == "confidence_tree"
                    else 1 <= prefix < len(primary)
                )
            ):
                raise ValueError(
                    "continuation alternatives require a valid proper prefix"
                )

            def choose_distinct_children(
                candidate_prefix, ranked_paths=ranked, primary_path=primary
            ):
                selected = [ranked_paths[0]]
                children = {primary_path[candidate_prefix]}
                for candidate in ranked_paths[1:]:
                    tokens = candidate[0]
                    compatible = (
                        tokens[:candidate_prefix]
                        == primary_path[:candidate_prefix]
                        and tokens[candidate_prefix] not in children
                    )
                    if not compatible:
                        continue
                    selected.append(candidate)
                    children.add(tokens[candidate_prefix])
                    if len(selected) == 4:
                        break
                return selected

            def choose_distinct_roots(ranked_paths=ranked_all):
                # A zero-trunk (rootless sibling) launch verifies every row
                # from the request base as its own full row.  Rows sharing a
                # first token would re-verify their common prefix (native-MTP
                # alternatives are truncations of the primary plus one other
                # token), so keep the best-ranked row per root token.  Ragged
                # depths stay legal here.
                selected, roots = [], set()
                for candidate in ranked_paths:
                    if candidate[0][0] in roots:
                        continue
                    selected.append(candidate)
                    roots.add(candidate[0][0])
                    if len(selected) == 4:
                        break
                return selected

            chosen = [ranked[0]]
            actual_prefix = 0 if strategy == "distinct_root" else prefix
            if strategy == "distinct_root":
                roots = {primary[0]}
                for candidate in ranked[1:]:
                    if candidate[0][0] in roots:
                        continue
                    chosen.append(candidate)
                    roots.add(candidate[0][0])
                    if len(chosen) == branch_count:
                        break
            elif strategy in {"shared_prefix", "confidence_tree"}:
                prefix_candidates = [prefix]
                if allow_adaptive_prefix_search and strategy == "confidence_tree":
                    # Complete DFlash paths may fan out below the confidence
                    # clamp (a deeper shared prefix), or directly at the
                    # request state.  Search only proper prefixes of the same
                    # ranked complete paths: no descendants are fabricated.
                    prefix_candidates.extend(range(prefix + 1, len(primary)))
                    prefix_candidates.extend(range(prefix - 1, -1, -1))
                feasible_widths = []
                chosen = []
                for candidate_prefix in prefix_candidates:
                    candidate_rows = (
                        choose_distinct_roots()
                        if allow_adaptive_prefix_search and candidate_prefix == 0
                        else choose_distinct_children(candidate_prefix)
                    )
                    feasible_widths.append(
                        (int(candidate_prefix), min(4, len(candidate_rows)))
                    )
                    if not chosen and len(candidate_rows) >= branch_count:
                        chosen = candidate_rows[:branch_count]
                        actual_prefix = candidate_prefix
                search_diagnostics.append(
                    {
                        "attempted_prefixes": tuple(
                            int(candidate) for candidate in prefix_candidates
                        ),
                        "feasible_width_by_prefix": tuple(feasible_widths),
                        "chosen_prefix": (
                            int(actual_prefix) if len(chosen) == branch_count else None
                        ),
                        "chosen_width": (
                            int(branch_count) if len(chosen) == branch_count else 0
                        ),
                    }
                )
            elif strategy == "paired_continuations":
                if prefix + 1 >= len(primary):
                    raise ValueError(
                        "paired continuation alternatives require two suffix positions"
                    )
                parent_count = min(2, branch_count)
                parent_children = {primary[prefix]}
                for candidate in ranked[1:]:
                    tokens = candidate[0]
                    if (
                        tokens[:prefix] != primary[:prefix]
                        or tokens[prefix] in parent_children
                    ):
                        continue
                    chosen.append(candidate)
                    parent_children.add(tokens[prefix])
                    if len(chosen) == parent_count:
                        break
                parents = tuple(chosen)
                if len(parents) == parent_count:
                    for parent in parents[: branch_count - parent_count]:
                        parent_tokens = parent[0]
                        for candidate in ranked:
                            tokens = candidate[0]
                            if (
                                tokens[: prefix + 1]
                                == parent_tokens[: prefix + 1]
                                and tokens[prefix + 1]
                                != parent_tokens[prefix + 1]
                            ):
                                chosen.append(candidate)
                                break

            if len(chosen) != branch_count:
                if allow_adaptive_prefix_search and strategy == "confidence_tree":
                    # Finish every row so a failed cohort still exposes a
                    # token-redacted feasibility map for each session.
                    adaptive_search_failed = True
                    continue
                self.last_ordered_alternative_search_diagnostics = tuple(
                    search_diagnostics
                )
                raise InsufficientContinuationAlternatives(
                    "ranked continuation slate lacks the requested distinct branches"
                )
            rows.append([list(candidate[0]) for candidate in chosen])
            provenance_rows.append(
                tuple(provenance(path, rank) for _, path, rank in chosen)
            )
            actual_prefixes.append(actual_prefix)

        if adaptive_search_failed:
            self.last_ordered_alternative_search_diagnostics = tuple(
                search_diagnostics
            )
            raise InsufficientContinuationAlternatives(
                "ranked continuation slate lacks the requested distinct branches"
            )
        self.last_ordered_alternative_provenance = tuple(provenance_rows)
        self.last_ordered_alternative_prefix_tokens = tuple(
            int(value) for value in actual_prefixes
        )
        self.last_ordered_alternative_search_diagnostics = tuple(search_diagnostics)
        return rows
