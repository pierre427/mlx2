"""Loaded-session complete continuation providers and external route seam."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass

import numpy as np

CONTINUATION_VERIFICATION_ALGORITHM = "stable-truncated-prefix-dedup-v2"


def continuation_context_revision(binding, history, anchor):
    return hashlib.sha256(
        json.dumps(
            [binding, list(history), int(anchor)], separators=(",", ":")
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ContinuationPoolPolicy:
    sources: tuple = ("external", "prompt_lookup")
    limit: int = 15
    ngram_min: int = 3
    ngram_max: int = 6
    lookback: int = 4096
    mtp_max_history: int = 4096
    verification_algorithm: str = CONTINUATION_VERIFICATION_ALGORITHM

    def __post_init__(self):
        if (
            type(self.verification_algorithm) is not str
            or self.verification_algorithm != CONTINUATION_VERIFICATION_ALGORITHM
        ):
            raise ValueError("unsupported continuation_pool verification_algorithm")

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, dict):
            raise ValueError("continuation_pool must be a JSON object")  # noqa: TRY004
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError("unknown continuation_pool policy keys")
        fields = dict(value)
        if "sources" in fields:
            if not isinstance(fields["sources"], (list, tuple)):
                raise ValueError("continuation_pool sources must be a list")
            fields["sources"] = tuple(fields["sources"])
        policy = cls(**fields)
        if (
            not policy.sources
            or any(type(source) is not str for source in policy.sources)
            or len(set(policy.sources)) != len(policy.sources)
            or set(policy.sources) - {"external", "prompt_lookup", "native_mtp"}
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
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ContinuationContext:
    history: tuple[int, ...]
    anchor: int
    depth: int
    draft_cache: object
    pending_features: object
    processors: tuple = ()


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
        from .proposal_pool import ProposalPool, shared_ranking_registry

        self.backend = backend
        self.policy = ContinuationPoolPolicy.from_value(policy)
        if (
            getattr(backend, "proposal_distribution", None)
            != self.proposal_distribution
        ):
            raise ValueError(
                "continuation pool requires deterministic external context backbone"
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
        self.last_proposal_sources = ()
        self.adaptive_confidence_features = None

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
                        "qualified": False,
                    }
                    for name, source in self.source_records.items()
                },
                "scores": "source_private_ranking_proxies_not_verification_q",
                "cache": "private_provider_recompute_public_backbone_advanced_once",
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
        **kwargs,
    ):
        from .proposal_pool import PoolRound, ProposalPath

        if (
            processor_histories is None
            or proposal_contexts is None
            or len(processor_histories) != len(anchors)
            or len(proposal_contexts) != len(anchors)
        ):
            raise ValueError(
                "continuation pool requires request-bound proposal contexts"
            )
        self.last_continuation_selections = ()
        self.last_proposal_sources = ()
        # A batched cache exposes authoritative rows, while plain B1 cache does
        # not. Snapshot before the external context append changes its offset.
        if all(hasattr(layer, "rows") for layer in cache):
            private_rows = [
                copy.deepcopy([layer.rows[row] for layer in cache])
                for row in range(len(anchors))
            ]
        elif len(anchors) == 1:
            private_rows = [copy.deepcopy(cache)]
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
        tokens, laws = self.backend.draft_distributions(
            anchors,
            hidden,
            cache,
            proposal_length,
            rngs,
            temperatures,
            processor_histories=processor_histories,
            logits_processors=logits_processors,
            **kwargs,
        )
        self.draft_feedback_payloads = copy.deepcopy(
            getattr(self.backend, "draft_feedback_payloads", None)
        )
        selections, sources = [], []
        for row, (context, identity) in enumerate(zip(contexts, proposal_contexts)):
            if not proposal_length:
                selections.append(None)
                sources.append("external")
                continue
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
            for source_id, provider in self.providers.items():
                record = self.source_records[source_id]
                for ordinal, candidate in enumerate(
                    provider(context, self.policy.limit)
                ):
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
            selection = self.proposal_pool.select(paths, round, limit=self.policy.limit)
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
            tokens[row] = list(primary.tokens)
            laws[row] = []
            for token in primary.tokens:
                q = np.zeros(self.session.vocab_size)
                q[token] = 1.0
                laws[row].append(q)
            # Pool ranking differs from the old greedy row's confidence law.
            source = primary.representative.source_id
            sources.append(
                source if source in {"prompt_lookup", "native_mtp"} else "external"
            )
        self.last_continuation_selections = tuple(selections)
        self.last_proposal_sources = tuple(sources)
        self.adaptive_confidence_features = [None for _ in anchors]
        return tokens, laws
