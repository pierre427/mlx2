"""Session-bound ranking of up to 15 complete continuation sequences.

Original host-only mlx2 code; see provenance/proposal-pool.json. Scores predict
accepted tokens from prior committed observations, never a refined head's joint
proposal law. Exact greedy/sampled execution must draw from the processed target
law first, then follow matching sequence prefixes. Original source q is neither
combined nor exported. Selection and feedback are separate transactions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import islice
from threading import RLock

VERIFICATION_CONTRACT = "target_draw_then_prefix_match"
MECHANISMS = frozenset(
    {"prompt_lookup", "native_mtp", "xpress", "lilicorr", "dflash2", "dpara"}
)


def _pin(value, name):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None
    ):
        raise ValueError(
            f"{name} must be a pinned 40/64-character lowercase hexadecimal revision"
        )


def _name(value, name):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty bounded string")


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ProposalSession:
    session_revision: str
    target_revision: str
    tokenizer_revision: str
    vocab_size: int

    def __post_init__(self):
        for field in ("session_revision", "target_revision", "tokenizer_revision"):
            _pin(getattr(self, field), field)
        _positive(self.vocab_size, "vocab_size")


@dataclass(frozen=True)
class ProposalSource:
    source_id: str
    mechanism: str
    revision: str
    session_revision: str
    target_revision: str
    tokenizer_revision: str
    vocab_size: int
    loaded: bool = True
    token_path_support: bool = True
    max_depth: int = 15

    def __post_init__(self):
        _name(self.source_id, "source_id")
        if self.mechanism not in MECHANISMS:
            raise ValueError("unsupported proposal mechanism")
        for field in (
            "revision",
            "session_revision",
            "target_revision",
            "tokenizer_revision",
        ):
            _pin(getattr(self, field), field)
        _positive(self.vocab_size, "vocab_size")
        _positive(self.max_depth, "max_depth")
        if type(self.loaded) is not bool or type(self.token_path_support) is not bool:
            raise ValueError("source admission flags must be boolean")


@dataclass(frozen=True)
class PoolRound:
    session_revision: str
    context_revision: str
    request_id: str
    round_id: int
    session_scope_hash: str | None = None

    def __post_init__(self):
        _pin(self.session_revision, "session_revision")
        _pin(self.context_revision, "context_revision")
        _name(self.request_id, "request_id")
        if self.session_scope_hash is not None:
            _pin(self.session_scope_hash, "session_scope_hash")
        if type(self.round_id) is not int or self.round_id < 0:
            raise ValueError("round_id must be a nonnegative integer")


@dataclass(frozen=True)
class ProposalPath:
    candidate_id: str
    source_id: str
    source_revision: str
    session_revision: str
    context_revision: str
    tokens: tuple
    confidence_features: tuple = ()
    cost: float | None = None
    cost_units: str | None = None
    ranking_score: float | None = None
    ranking_convention: str | None = None

    def __post_init__(self):
        _name(self.candidate_id, "candidate_id")
        _name(self.source_id, "source_id")
        for field in ("source_revision", "session_revision", "context_revision"):
            _pin(getattr(self, field), field)
        if (
            not isinstance(self.tokens, tuple)
            or not self.tokens
            or any(type(x) is not int or x < 0 for x in self.tokens)
        ):
            raise ValueError(
                "candidate tokens must be a nonempty immutable tuple of token IDs"
            )
        if not isinstance(self.confidence_features, tuple) or len(
            self.confidence_features
        ) not in (0, len(self.tokens)):
            raise ValueError(
                "confidence_features must be an immutable per-position tuple or empty"
            )
        for feature in self.confidence_features:
            if feature is not None:
                _finite(feature, "confidence feature")
        if self.cost is not None:
            _finite(self.cost, "cost")
            if self.cost <= 0:
                raise ValueError("cost must be positive")
            _name(self.cost_units, "cost_units")
        elif self.cost_units is not None:
            raise ValueError("cost_units requires a cost")
        if self.ranking_score is not None:
            _finite(self.ranking_score, "ranking_score")
            _name(self.ranking_convention, "ranking_convention")
        elif self.ranking_convention is not None:
            raise ValueError("ranking_convention requires a score")


@dataclass(frozen=True)
class RankedProposal:
    path_id: str
    tokens: tuple
    contributors: tuple
    representative: ProposalPath
    conditional_acceptance: tuple
    expected_accepted_tokens: float
    estimated_cost: float | None
    score: float
    score_authority: str
    source_ordinal: int


@dataclass(frozen=True)
class ProposalSelection:
    ticket_id: str
    round: PoolRound
    paths: tuple
    feedback_revision: int
    score_mode: str
    ranking_revision: int = 0
    verification_contract: str = VERIFICATION_CONTRACT

    def as_dict(self):
        return asdict(self)

    def require_verification_contract(self, value):
        if value != self.verification_contract:
            raise ValueError(
                "ranked proposal pool requires target draws before prefix matching; no joint q is exported"
            )


@dataclass(frozen=True)
class SelectedPrefixTree:
    tokens: tuple
    parents: tuple
    path_rows: tuple
    contributors: tuple


def prefix_tree(selection):
    """Optional exact sharing; preserve all selected COMPLETE sequences.

    Fifteen sequences can occupy up to 225 nodes. No 15-node truncation or
    local acceptance-boundary interpretation is applied here.
    """
    tokens = []
    parents = []
    rows = []
    contributors = []
    prefix_rows = {}
    for path in selection.paths:
        route = []
        for position, token in enumerate(path.tokens):
            prefix = path.tokens[: position + 1]
            if prefix not in prefix_rows:
                prefix_rows[prefix] = len(tokens)
                tokens.append(token)
                parents.append(-1 if position == 0 else prefix_rows[prefix[:-1]])
                contributors.append(set())
            row = prefix_rows[prefix]
            route.append(row)
            contributors[row].update(
                (c.source_id, c.source_revision, c.candidate_id)
                for c in path.contributors
            )
        rows.append(tuple(route))
    return SelectedPrefixTree(
        tuple(tokens),
        tuple(parents),
        tuple(rows),
        tuple(tuple(sorted(items)) for items in contributors),
    )


class ProposalRankingRegistry:
    """Bounded empirical source critic with global/model/session shrinkage.

    This original source-selection extension is separate from the published
    LiLiCoRR token lattice. It learns candidate-prefix reliability, not a joint
    proposal law. Session keys are opaque hashes supplied by the serving boundary;
    this registry receives no raw tenant/session identifiers. Shared global priors
    contain only aggregate mechanism-position counts; model and session residuals
    are revision-bound. Cold feature values carry no pretrained calibration.
    """

    def __init__(
        self,
        *,
        prior_acceptance=0.5,
        prior_strength=2.0,
        model_shrinkage=8.0,
        session_shrinkage=8.0,
        confidence_bin_width=4.0,
        confidence_shrinkage=2.0,
        max_models=32,
        max_sessions=128,
        max_sources=32,
        max_depth=15,
    ):
        for value, name in (
            (prior_acceptance, "prior_acceptance"),
            (prior_strength, "prior_strength"),
            (model_shrinkage, "model_shrinkage"),
            (session_shrinkage, "session_shrinkage"),
            (confidence_bin_width, "confidence_bin_width"),
            (confidence_shrinkage, "confidence_shrinkage"),
        ):
            _finite(value, name)
        if (
            not 0 < prior_acceptance < 1
            or min(
                prior_strength,
                model_shrinkage,
                session_shrinkage,
                confidence_bin_width,
                confidence_shrinkage,
            )
            <= 0
        ):
            raise ValueError("invalid hierarchical critic policy")
        if not 0.5 <= confidence_bin_width <= 80.0:
            raise ValueError(
                "confidence_bin_width must bound confidence storage between 0.5 and 80"
            )
        for value, name in (
            (max_models, "max_models"),
            (max_sessions, "max_sessions"),
            (max_sources, "max_sources"),
            (max_depth, "max_depth"),
        ):
            _positive(value, name)
        self.prior_acceptance = float(prior_acceptance)
        self.prior_strength = float(prior_strength)
        self.model_shrinkage = float(model_shrinkage)
        self.session_shrinkage = float(session_shrinkage)
        self.confidence_bin_width = float(confidence_bin_width)
        self.confidence_shrinkage = float(confidence_shrinkage)
        self.max_models = max_models
        self.max_sessions = max_sessions
        self.max_sources = max_sources
        self.max_depth = max_depth
        self._global = {}
        self._models = OrderedDict()
        self._sessions = OrderedDict()
        self.revision = 0
        self.evictions = {"models": 0, "sessions": 0, "sources": 0}
        self.lock = RLock()

    @staticmethod
    def model_key(session):
        return _digest(
            (session.target_revision, session.tokenizer_revision, session.vocab_size)
        )

    @staticmethod
    def source_key(source):
        return (source.mechanism, source.revision, source.session_revision)

    def bucket(self, feature):
        return math.floor(
            max(-40.0, min(40.0, float(feature))) / self.confidence_bin_width
        )

    def _entry(self, table, key, limit, kind, *, create=False):
        if key not in table:
            if not create:
                return None
            table[key] = OrderedDict()
        table.move_to_end(key)
        while len(table) > limit:
            table.popitem(last=False)
            self.evictions[kind] += 1
        return table[key]

    def _source_entry(self, table, source, *, create=False):
        if table is None:
            return None
        key = self.source_key(source)
        if key not in table:
            if not create:
                return None
            table[key] = {"counts": {}, "bins": {}}
        table.move_to_end(key)
        while len(table) > self.max_sources:
            table.popitem(last=False)
            self.evictions["sources"] += 1
        return table[key]

    @staticmethod
    def _shrink(counts, key, prior, strength):
        yes, no = counts.get(key, (0, 0))
        return (yes + prior * strength) / (yes + no + strength), yes + no

    def predict(self, session, source, scope, position, feature):
        self._validate_identity(session, source, scope)
        if type(position) is not int or not 0 <= position < self.max_depth:
            raise ValueError("critic depth bound exceeded")
        if feature is not None:
            _finite(feature, "critic confidence feature")
        with self.lock:
            p, observed = self._shrink(
                self._global,
                (source.mechanism, position),
                self.prior_acceptance,
                self.prior_strength,
            )
            model_key = self.model_key(session)
            model = self._source_entry(
                self._entry(self._models, model_key, self.max_models, "models"), source
            )
            local = self._source_entry(
                self._entry(
                    self._sessions, (scope, model_key), self.max_sessions, "sessions"
                ),
                source,
            )
            for entry, strength in (
                (model, self.model_shrinkage),
                (local, self.session_shrinkage),
            ):
                if entry is None:
                    continue
                p, count = self._shrink(entry["counts"], position, p, strength)
                observed += count
                if feature is not None:
                    p, count = self._shrink(
                        entry["bins"],
                        (position, self.bucket(feature)),
                        p,
                        self.confidence_shrinkage,
                    )
                    observed += count
            return p, observed

    @staticmethod
    def _validate_identity(session, source, scope):
        if not isinstance(session, ProposalSession) or not isinstance(
            source, ProposalSource
        ):
            raise TypeError("critic requires pinned session and source records")
        _pin(scope, "session_scope_hash")
        for field in (
            "session_revision",
            "target_revision",
            "tokenizer_revision",
            "vocab_size",
        ):
            if getattr(session, field) != getattr(source, field):
                raise ValueError("critic source/session identity mismatch")
        if not source.loaded or not source.token_path_support:
            raise ValueError("critic source is not admitted")

    @staticmethod
    def _increment(counts, key, label):
        yes, no = counts.get(key, (0, 0))
        counts[key] = (yes + int(label), no + int(not label))

    def commit(self, session, scope, events):
        """Validated source/position/label/bins events, once per committed round."""
        if not isinstance(session, ProposalSession):
            raise TypeError("ProposalSession required")
        _pin(scope, "session_scope_hash")
        limit = self.max_sources * self.max_depth * 1024
        events = tuple(islice(iter(events), limit + 1))
        if len(events) > limit:
            raise ValueError("critic feedback event budget exceeded")
        for source, position, label, buckets in events:
            self._validate_identity(session, source, scope)
            if type(position) is not int or not 0 <= position < self.max_depth:
                raise ValueError("critic feedback depth bound exceeded")
            if type(label) is not bool:
                raise ValueError("critic labels must be boolean")
            low = math.floor(-40.0 / self.confidence_bin_width)
            high = math.floor(40.0 / self.confidence_bin_width)
            if any(
                type(bucket) is not int or not low <= bucket <= high
                for bucket in buckets
            ):
                raise ValueError("critic confidence bucket bound exceeded")
        with self.lock:
            model_key = self.model_key(session)
            model_table = self._entry(
                self._models, model_key, self.max_models, "models", create=True
            )
            session_table = self._entry(
                self._sessions,
                (scope, model_key),
                self.max_sessions,
                "sessions",
                create=True,
            )
            for source, position, label, buckets in events:
                self._increment(self._global, (source.mechanism, position), label)
                for table in (model_table, session_table):
                    entry = self._source_entry(table, source, create=True)
                    self._increment(entry["counts"], position, label)
                    for bucket in buckets:
                        self._increment(entry["bins"], (position, bucket), label)
            self.revision += 1

    def receipt(self, session, scope=None):
        with self.lock:
            key = self.model_key(session)

            def summary(table):
                if table is None:
                    return []
                return [
                    {
                        "mechanism": mechanism,
                        "source_revision": revision,
                        "route_settings_revision": route_revision,
                        "accepted": sum(y for y, n in value["counts"].values()),
                        "rejected": sum(n for y, n in value["counts"].values()),
                        "feature_bins": len(value["bins"]),
                    }
                    for (mechanism, revision, route_revision), value in table.items()
                ]

            return {
                "schema": "mlx2_lilicorr_source_critic_v1",
                "revision": self.revision,
                "training_authority": "caller_committed_target_prefix; suffixes_censored",
                "model_key": key,
                "target_revision": session.target_revision,
                "tokenizer_revision": session.tokenizer_revision,
                "session_scope_hash": scope,
                "global_counts": [
                    {"mechanism": m, "position": p, "accepted": y, "rejected": n}
                    for (m, p), (y, n) in sorted(self._global.items())
                ],
                "model_counts": summary(self._models.get(key)),
                "session_counts": summary(self._sessions.get((scope, key)))
                if scope is not None
                else [],
                "retained_models": len(self._models),
                "retained_sessions": len(self._sessions),
                "bounds": {
                    "models": self.max_models,
                    "sessions": self.max_sessions,
                    "sources": self.max_sources,
                    "depth": self.max_depth,
                    "feature_bins_per_position": math.floor(
                        80.0 / self.confidence_bin_width
                    )
                    + 2,
                },
                "evictions": dict(self.evictions),
                "prior_acceptance": self.prior_acceptance,
                "prior_strength": self.prior_strength,
                "model_shrinkage": self.model_shrinkage,
                "session_shrinkage": self.session_shrinkage,
                "confidence_shrinkage": self.confidence_shrinkage,
                "confidence_bin_width": self.confidence_bin_width,
                "paper_lattice": False,
                "pretrained_source_critic": False,
                "qualified": False,
            }


_SHARED_RANKING_REGISTRY = None
_SHARED_RANKING_LOCK = RLock()


def shared_ranking_registry():
    """Explicit opt-in process-global registry for loaded provider managers."""
    global _SHARED_RANKING_REGISTRY
    with _SHARED_RANKING_LOCK:
        if _SHARED_RANKING_REGISTRY is None:
            _SHARED_RANKING_REGISTRY = ProposalRankingRegistry()
        return _SHARED_RANKING_REGISTRY


class ProposalPool:
    """Neutral-prior source calibration, full-path dedup and committed feedback.

    Registration asserts that an adapter has checked an actually loaded source;
    this host core cannot independently prove the existence of its model tensors.
    Learned artifacts, tokenizer/target geometry and causal context must already
    be bound by the provider. Missing sources never fabricate candidate paths.

    Cold confidence/logit scales are not compared between sources. Within-source
    beam-score ordinals break prediction ties; equal predictions interleave those
    ordinals across sources. Confidence bins acquire meaning only from labels of
    that same source in prior committed rounds. Caller costs are declared or
    measured inputs, never a performance claim inferred by this pool.
    """

    def __init__(
        self,
        session,
        sources,
        *,
        score_mode="expected_accepted_tokens",
        cost_units=None,
        prior_acceptance=0.5,
        prior_strength=2.0,
        confidence_bin_width=4.0,
        confidence_shrinkage=2.0,
        max_depth=15,
        max_raw_paths=1024,
        max_pending=128,
        max_requests=1024,
        ranking_registry=None,
        session_scope_hash=None,
    ):
        if not isinstance(session, ProposalSession):
            raise TypeError("ProposalSession required")
        for value, name in (
            (max_depth, "max_depth"),
            (max_raw_paths, "max_raw_paths"),
            (max_pending, "max_pending"),
            (max_requests, "max_requests"),
        ):
            _positive(value, name)
        for value, name in (
            (prior_acceptance, "prior_acceptance"),
            (prior_strength, "prior_strength"),
            (confidence_bin_width, "confidence_bin_width"),
            (confidence_shrinkage, "confidence_shrinkage"),
        ):
            _finite(value, name)
        if (
            not 0 < prior_acceptance < 1
            or min(prior_strength, confidence_bin_width, confidence_shrinkage) <= 0
        ):
            raise ValueError("invalid declared acceptance prior/calibration policy")
        if score_mode not in ("expected_accepted_tokens", "expected_accepted_per_cost"):
            raise ValueError("unsupported proposal score mode")
        if score_mode == "expected_accepted_per_cost" or cost_units is not None:
            _name(cost_units, "cost_units")
        self.session = session
        self.score_mode = score_mode
        self.cost_units = cost_units
        self.prior_acceptance = float(prior_acceptance)
        self.prior_strength = float(prior_strength)
        self.confidence_bin_width = float(confidence_bin_width)
        self.confidence_shrinkage = float(confidence_shrinkage)
        self.max_depth = max_depth
        self.max_raw_paths = max_raw_paths
        self.max_pending = max_pending
        self.max_requests = max_requests
        if session_scope_hash is not None:
            _pin(session_scope_hash, "session_scope_hash")
        self.session_scope_hash = session_scope_hash
        if ranking_registry is not None and not isinstance(
            ranking_registry, ProposalRankingRegistry
        ):
            raise TypeError("ProposalRankingRegistry required")
        self.ranking_registry = ranking_registry or ProposalRankingRegistry(
            prior_acceptance=prior_acceptance,
            prior_strength=prior_strength,
            confidence_bin_width=confidence_bin_width,
            confidence_shrinkage=confidence_shrinkage,
            max_depth=max_depth,
        )
        if max_depth > self.ranking_registry.max_depth:
            raise ValueError("pool exceeds shared critic depth bound")
        self.sources = {}
        self._counts = {}
        self._bins = {}
        self._costs = {}
        self._pending = {}
        self._request_tickets = {}
        self._last_committed = OrderedDict()
        self._feedback_revision = 0
        for source in sources:
            if not isinstance(source, ProposalSource):
                raise TypeError("ProposalSource required")
            if source.source_id in self.sources:
                raise ValueError("duplicate proposal source identity")
            if not source.loaded or not source.token_path_support:
                raise ValueError(
                    "source is not loaded or cannot emit complete causal token paths"
                )
            for field in (
                "session_revision",
                "target_revision",
                "tokenizer_revision",
                "vocab_size",
            ):
                if getattr(source, field) != getattr(session, field):
                    raise ValueError(f"proposal source {field} mismatch")
            if source.max_depth > max_depth:
                raise ValueError("source depth exceeds pool bound")
            self.sources[source.source_id] = source
            if len(self.sources) > 32:
                raise ValueError("too many loaded proposal sources")
        if len(self.sources) > 32:
            raise ValueError("too many loaded proposal sources")

    @property
    def feedback_revision(self):
        return self._feedback_revision

    def _bucket(self, feature):
        return self.ranking_registry.bucket(feature)

    def _scope(self, round):
        return (
            round.session_scope_hash
            or self.session_scope_hash
            or _digest((round.session_revision, round.request_id))
        )

    def _prediction(self, path, scope):
        probabilities = []
        observed = 0
        for position in range(len(path.tokens)):
            feature = (
                path.confidence_features[position] if path.confidence_features else None
            )
            p, count = self.ranking_registry.predict(
                self.session, self.sources[path.source_id], scope, position, feature
            )
            probabilities.append(p)
            observed += count
        survival = 1.0
        expected = 0.0
        for p in probabilities:
            survival *= p
            expected += survival
        cost = path.cost
        units = path.cost_units
        if cost is None and path.source_id in self._costs:
            total, count, units = self._costs[path.source_id]
            cost = total / count
        if self.score_mode == "expected_accepted_per_cost":
            if cost is None or units != self.cost_units:
                raise ValueError(
                    "expected-per-cost ranking requires explicit compatible source cost evidence"
                )
            score = expected / cost
        else:
            score = expected
        return (
            tuple(probabilities),
            expected,
            cost,
            score,
            "hierarchical_committed_prefix_critic"
            if observed
            else "declared_neutral_prior",
        )

    def select(self, paths, round, *, limit=15):
        with self.ranking_registry.lock:
            return self._select(paths, round, limit=limit)

    def _select(self, paths, round, *, limit=15):
        if not isinstance(round, PoolRound):
            raise TypeError("PoolRound required")
        if round.session_revision != self.session.session_revision:
            raise ValueError("selection session revision mismatch")
        if type(limit) is not int or not 1 <= limit <= 15:
            raise ValueError(
                "pool limit must select between 1 and 15 complete sequences"
            )
        paths = tuple(islice(iter(paths), self.max_raw_paths + 1))
        round = replace(round, session_scope_hash=self._scope(round))
        if len(paths) > self.max_raw_paths:
            raise ValueError("raw proposal pool exceeds bounded candidate budget")
        if round.round_id <= self._last_committed.get(round.request_id, -1):
            raise ValueError("stale or already committed proposal round")
        by_tokens = defaultdict(list)
        by_source = defaultdict(list)
        ids = set()
        for path in paths:
            if not isinstance(path, ProposalPath):
                raise TypeError("ProposalPath required")
            source = self.sources.get(path.source_id)
            if source is None:
                raise ValueError(
                    "candidate source is not admitted in the loaded session"
                )
            if path.source_revision != source.revision:
                raise ValueError("candidate source revision mismatch")
            if (
                path.session_revision != round.session_revision
                or path.context_revision != round.context_revision
            ):
                raise ValueError("candidate session/context revision mismatch")
            if len(path.tokens) > source.max_depth or any(
                token >= self.session.vocab_size for token in path.tokens
            ):
                raise ValueError("candidate depth/vocabulary mismatch")
            key = (path.source_id, path.candidate_id)
            if key in ids:
                raise ValueError("duplicate candidate identifier within a source")
            ids.add(key)
            by_tokens[path.tokens].append(path)
            by_source[path.source_id].append(path)
        ordinals = {}
        for source_paths in by_source.values():
            conventions = {path.ranking_convention for path in source_paths}
            if len(conventions) > 1:
                raise ValueError(
                    "one source cannot mix ranking-score conventions within a round"
                )
            ordered = sorted(
                source_paths,
                key=lambda p: (
                    -(p.ranking_score if p.ranking_score is not None else 0.0),
                    p.tokens,
                    p.candidate_id,
                ),
            )
            for index, path in enumerate(ordered):
                ordinals[(path.source_id, path.candidate_id)] = index
        ranked = []
        for tokens, contributors in by_tokens.items():
            alternatives = []
            for path in contributors:
                probabilities, expected, cost, score, authority = self._prediction(
                    path, round.session_scope_hash
                )
                ordinal = ordinals[(path.source_id, path.candidate_id)]
                alternatives.append(
                    (
                        (-score, ordinal, path.source_id, path.candidate_id),
                        path,
                        probabilities,
                        expected,
                        cost,
                        score,
                        authority,
                        ordinal,
                    )
                )
            (
                _,
                representative,
                probabilities,
                expected,
                cost,
                score,
                authority,
                ordinal,
            ) = min(alternatives, key=lambda row: row[0])
            ranked.append(
                RankedProposal(
                    _digest((round.context_revision, tokens)),
                    tokens,
                    tuple(
                        sorted(
                            contributors, key=lambda p: (p.source_id, p.candidate_id)
                        )
                    ),
                    representative,
                    probabilities,
                    expected,
                    cost,
                    score,
                    authority,
                    ordinal,
                )
            )
        ranked.sort(
            key=lambda p: (
                -p.score,
                p.source_ordinal,
                p.representative.source_id,
                p.tokens,
                p.path_id,
            )
        )
        chosen = tuple(ranked[:limit])
        content = {
            "round": asdict(round),
            "paths": [asdict(path) for path in chosen],
            "feedback_revision": self.feedback_revision,
            "ranking_revision": self.ranking_registry.revision,
            "score_mode": self.score_mode,
            "verification_contract": VERIFICATION_CONTRACT,
        }
        ticket = _digest(content)
        if round.request_id in self._request_tickets:
            previous = self._request_tickets[round.request_id]
            if previous == ticket:
                return self._pending[ticket]
            raise ValueError(
                "request already has a different pending proposal selection"
            )
        if len(self._pending) >= self.max_pending:
            raise ValueError("too many pending proposal selections")
        selection = ProposalSelection(
            ticket,
            round,
            chosen,
            self.feedback_revision,
            self.score_mode,
            ranking_revision=self.ranking_registry.revision,
        )
        self._pending[ticket] = selection
        self._request_tickets[round.request_id] = ticket
        return selection

    def discard(self, selection):
        with self.ranking_registry.lock:
            self._validate_ticket(selection)
            self._pending.pop(selection.ticket_id)
            self._request_tickets.pop(selection.round.request_id)

    def _validate_ticket(self, selection):
        if (
            not isinstance(selection, ProposalSelection)
            or self._pending.get(selection.ticket_id) is not selection
        ):
            raise ValueError("unknown, stale or already consumed proposal selection")
        content = selection.as_dict()
        content.pop("ticket_id")
        if _digest(content) != selection.ticket_id:
            raise ValueError("proposal selection content changed")

    def commit_feedback(
        self, selection, teacher_tokens, *, costs=None, cost_units=None
    ):
        with self.ranking_registry.lock:
            return self._commit_feedback(
                selection, teacher_tokens, costs=costs, cost_units=cost_units
            )

    def _commit_feedback(
        self, selection, teacher_tokens, *, costs=None, cost_units=None
    ):
        """Call ONLY after the executor's entire outer round commits successfully.

        The tokens are the actually emitted processed-target draws, including a
        first mismatch/correction if delivered. Every matching prefix is positive;
        the first mismatch is negative and the remaining path is censored. Stop
        or budget exhaustion censors positions beyond delivered teacher tokens.
        Common candidate-prefix labels from the same source are counted once.
        Unvisited target branches never become teacher labels. Replaying a ticket
        or feeding a failed/cancelled round is refused by lifecycle ownership.
        """
        self._validate_ticket(selection)
        teacher_tokens = tuple(islice(iter(teacher_tokens), self.max_depth + 2))
        if (
            not teacher_tokens
            or len(teacher_tokens) > self.max_depth + 1
            or any(
                type(t) is not int or not 0 <= t < self.session.vocab_size
                for t in teacher_tokens
            )
        ):
            raise ValueError(
                "feedback requires a bounded nonempty delivered target-token prefix"
            )
        costs = {} if costs is None else dict(costs)
        if costs:
            _name(cost_units, "cost_units")
            if self.cost_units is not None and cost_units != self.cost_units:
                raise ValueError("feedback cost units mismatch")
        elif cost_units is not None:
            raise ValueError("feedback cost units require observations")
        for source_id, cost in costs.items():
            if source_id not in self.sources:
                raise ValueError("cost observation refers to an unadmitted source")
            _finite(cost, "observed source cost")
            if cost <= 0:
                raise ValueError("observed source cost must be positive")
            if source_id in self._costs and self._costs[source_id][2] != cost_units:
                raise ValueError("source cost units changed")
        labels = {}
        for ranked in selection.paths:
            for path in ranked.contributors:
                for position, (token, teacher) in enumerate(
                    zip(path.tokens, teacher_tokens)
                ):
                    label = token == teacher
                    feature = (
                        path.confidence_features[position]
                        if path.confidence_features
                        else None
                    )
                    event = (path.source_id, position, path.tokens[:position], token)
                    if event not in labels:
                        labels[event] = (label, set())
                    if feature is not None:
                        labels[event][1].add(self._bucket(feature))
                    if not label:
                        break
        # All validation precedes mutation. There is no observation during select.
        for (source_id, position, _prefix, _token), (label, buckets) in labels.items():
            key = (source_id, position)
            yes, no = self._counts.get(key, (0, 0))
            self._counts[key] = (yes + int(label), no + int(not label))
            for bucket in buckets:
                key = (source_id, position, bucket)
                yes, no = self._bins.get(key, (0, 0))
                self._bins[key] = (yes + int(label), no + int(not label))
        for source_id, cost in costs.items():
            total, count, _units = self._costs.get(source_id, (0.0, 0, cost_units))
            self._costs[source_id] = (total + float(cost), count + 1, cost_units)
        registry_events = {}
        for (source_id, position, prefix, token), (label, buckets) in labels.items():
            source = self.sources[source_id]
            key = (self.ranking_registry.source_key(source), position, prefix, token)
            if key not in registry_events:
                registry_events[key] = (source, position, label, set())
            registry_events[key][3].update(buckets)
        self.ranking_registry.commit(
            self.session, selection.round.session_scope_hash, registry_events.values()
        )
        self._feedback_revision += 1
        self._last_committed[selection.round.request_id] = selection.round.round_id
        self._last_committed.move_to_end(selection.round.request_id)
        while len(self._last_committed) > self.max_requests:
            self._last_committed.popitem(last=False)
        self._pending.pop(selection.ticket_id)
        self._request_tickets.pop(selection.round.request_id)
        return {
            "feedback_revision": self.feedback_revision,
            "ranking_revision": self.ranking_registry.revision,
            "unique_observed_positions": len(labels),
            "accepted_labels": sum(int(label) for label, _bins in labels.values()),
            "rejected_labels": sum(int(not label) for label, _bins in labels.values()),
            "authority": "caller_delivered_target_prefix_after_outer_commit",
        }

    def observation_counts(self, source_id):
        if source_id not in self.sources:
            raise ValueError("unknown proposal source")
        return tuple(
            self._counts.get((source_id, position), (0, 0))
            for position in range(self.max_depth)
        )

    def receipt(self, session_scope_hash=None):
        return {
            "session": asdict(self.session),
            "sources": [asdict(source) for source in self.sources.values()],
            "score_mode": self.score_mode,
            "cost_units": self.cost_units,
            "feedback_revision": self.feedback_revision,
            "ranking_revision": self.ranking_registry.revision,
            "verification_contract": VERIFICATION_CONTRACT,
            "selection_budget": "15_complete_sequences",
            "declared_prior_acceptance": self.ranking_registry.prior_acceptance,
            "declared_prior_strength": self.ranking_registry.prior_strength,
            "ranking_score_interpretation": "empirical conditional-reliability product proxy; not exact joint q or calibrated acceptance forecast",
            "request_replay_guard_bound": self.max_requests,
            "confidence": "same_source_lagged_label_calibration; never joint_q",
            "provider_scores": "same_source_ordinals_only; cold_ties_interleave_sources",
            "source_admission_authority": "adapter_asserted_loaded_session_identity",
            "ranking_registry": self.ranking_registry.receipt(
                self.session, session_scope_hash or self.session_scope_hash
            ),
            "qualified": False,
            "performance_claim": False,
        }


__all__ = [
    "VERIFICATION_CONTRACT",
    "PoolRound",
    "ProposalPath",
    "ProposalPool",
    "ProposalRankingRegistry",
    "ProposalSelection",
    "ProposalSession",
    "ProposalSource",
    "RankedProposal",
    "SelectedPrefixTree",
    "prefix_tree",
    "shared_ranking_registry",
]
