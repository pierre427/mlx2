# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
import threading
import time
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass
class HybridStats:
    """Per-source accounting for one prompt-lookup generation run.

    Two per-round distributions are kept, both as plain ``{value: rounds}``
    host dicts so a receipt consumer can post-process them by truncation:

    ``verify_span_hist``
        Verification-forward width (positions fed to the target) per round.
        ``sum(width * rounds)`` is the total verified positions.
    ``verify_accept_hist``
        Speculative tokens *accepted* per round, i.e. the same quantity the
        route's ``accepted`` aggregate sums.  Because a draft accepted to
        depth k would also have been accepted under any shallower cap, the
        commit rate at every depth below the one actually run follows by
        truncation from this one histogram::

            tau(k) = sum(min(a, k) + 1 for each round) / rounds

        which keeps numerator and denominator inside a single run instead of
        comparing two runs.
    """

    cycles: int = 0
    retrieval_cycles: int = 0
    plain_cycles: int = 0
    retrieval_proposed: int = 0
    retrieval_accepted: int = 0
    bonus_tokens: int = 0
    plain_tokens: int = 0
    span_snap_cycles: int = 0
    span_snap_tokens: int = 0
    span_extend_cycles: int = 0
    span_extend_tokens: int = 0
    verify_span_hist: dict[int, int] = field(default_factory=dict)
    verify_accept_hist: dict[int, int] = field(default_factory=dict)
    latched: bool = False
    lookback_current: int = 0
    lookback_peak: int = 0
    lookback_widen_events: int = 0
    lookback_narrow_events: int = 0
    admission_windows: int = 0
    admission_probe_tokens: int = 0
    admission_matches: int = 0
    admission_activations: int = 0
    admission_delatches: int = 0
    admission_reentries: int = 0

    @property
    def total_emitted(self) -> int:
        return self.retrieval_accepted + self.bonus_tokens + self.plain_tokens

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        acc = self.retrieval_accepted / max(self.retrieval_proposed, 1)
        return f"cycles {self.cycles} (retrieval {self.retrieval_cycles}, plain {self.plain_cycles}) | tokens {self.total_emitted}: retrieval {self.retrieval_accepted} ({self.retrieval_accepted / tot:.0%}) + bonus {self.bonus_tokens} + plain {self.plain_tokens} | retrieval acceptance {acc:.0%} | latched={self.latched}"


class CostAwarePLDLatch:
    """B1-only, committed-boundary admission using shadow copies and own costs.

    A shadow is compared with ordinary target tokens across later rounds; it
    never becomes a draft. Once enough multi-token shadows match, a few real
    verify rounds measure whether this executor commits tokens more cheaply
    than its own ordinary rounds. All windows and counters are request-local.
    """

    def __init__(
        self, *, shadow_span=4, probe_stride=4, shadow_window=2,
        shadow_gate=0.75, plain_rounds=8, explore_rounds=2,
        cost_margin=0.05, reprobe_interval=32, park_rounds=4,
    ):
        self.shadow_span = shadow_span
        self.probe_stride = probe_stride
        self.shadow_gate = shadow_gate
        self.plain_rounds = plain_rounds
        self.explore_rounds = explore_rounds
        self.cost_margin = cost_margin
        self.reprobe_interval = reprobe_interval
        self.park_rounds = park_rounds
        self.state = "parked"
        self.reprobe_at = 0
        self.shadow = ()
        self.shadow_index = 0
        self.shadow_hits = deque(maxlen=shadow_window)
        self.plain_samples = deque(maxlen=64)
        self.verify_samples = deque(maxlen=park_rounds)
        self.explore_samples = []
        self.empty_rounds = 0
        self.shadow_lookups = self.shadow_trials = self.shadow_matches = 0
        self.activations = self.parks = self.reentries = self.width_parks = 0
        self.memory_parks = self.memory_blocked_rounds = 0
        self.plain_ns = self.verify_ns = 0
        self.plain_tokens = self.verify_tokens = 0

    def should_probe(self, generated):
        return (
            self.state == "parked" and generated >= self.reprobe_at
            and not self.shadow and generated % self.probe_stride == 0
        )

    def start_shadow(self, candidate):
        self.shadow_lookups += 1
        if len(candidate) >= self.shadow_span:
            self.shadow = tuple(candidate[: self.shadow_span])
            self.shadow_index = 0

    def observe_plain_token(self, token):
        if not self.shadow:
            return
        matched = token == self.shadow[self.shadow_index]
        if matched:
            self.shadow_index += 1
        if not matched or self.shadow_index == len(self.shadow):
            self.shadow_trials += 1
            self.shadow_matches += self.shadow_index
            self.shadow_hits.append(self.shadow_index / len(self.shadow))
            self.shadow = ()
            self.shadow_index = 0

    @staticmethod
    def _cost(samples):
        tokens = sum(count for _ns, count in samples)
        return sum(ns for ns, _count in samples) / tokens if tokens else None

    def _park(self, generated, *, width=False):
        if self.state != "parked":
            self.parks += 1
            self.width_parks += int(width)
        self.state = "parked"
        self.reprobe_at = generated + self.reprobe_interval
        self.shadow = ()
        self.shadow_index = 0
        self.shadow_hits.clear()
        self.verify_samples.clear()
        self.explore_samples.clear()
        self.empty_rounds = 0

    def suspend_for_width(self, generated):
        self._park(generated, width=True)

    def suspend_for_memory(self, generated):
        if self.state != "parked":
            self.memory_parks += 1
        self.memory_blocked_rounds += 1
        self._park(generated)

    def observe_round(self, duration_ns, delivered, proposed, generated):
        if proposed:
            self.verify_ns += duration_ns
            self.verify_tokens += delivered
            sample = (duration_ns, delivered)
            self.verify_samples.append(sample)
            self.empty_rounds = 0
            if self.state == "explore":
                self.explore_samples.append(sample)
                if len(self.explore_samples) >= self.explore_rounds:
                    plain = self._cost(self.plain_samples)
                    explore = self._cost(self.explore_samples)
                    if plain is None or explore >= plain * (1 - self.cost_margin):
                        self._park(generated)
                    else:
                        self.state = "active"
            elif self.state == "active" and len(self.verify_samples) >= self.park_rounds:
                plain = self._cost(self.plain_samples)
                verify = self._cost(self.verify_samples)
                if plain is not None and verify >= plain * (1 - self.cost_margin):
                    self._park(generated)
        else:
            self.plain_ns += duration_ns
            self.plain_tokens += delivered
            self.plain_samples.append((duration_ns, delivered))
            if self.state == "parked":
                if (
                    len(self.plain_samples) >= self.plain_rounds
                    and len(self.shadow_hits) == self.shadow_hits.maxlen
                    and sum(self.shadow_hits) / len(self.shadow_hits) >= self.shadow_gate
                ):
                    self.state = "explore"
                    self.activations += 1
                    self.reentries += int(self.parks > 0)
                    self.shadow = ()
                    self.shadow_hits.clear()
            else:
                self.empty_rounds += 1
                if self.empty_rounds >= self.park_rounds:
                    self._park(generated)

    def receipt(self):
        return {
            "state": self.state,
            "shadow_lookups": self.shadow_lookups,
            "shadow_trials": self.shadow_trials,
            "shadow_matches": self.shadow_matches,
            "activations": self.activations,
            "parks": self.parks,
            "reentries": self.reentries,
            "width_parks": self.width_parks,
            "memory_parks": self.memory_parks,
            "memory_blocked_rounds": self.memory_blocked_rounds,
            "plain_ns": self.plain_ns,
            "plain_tokens": self.plain_tokens,
            "verify_ns": self.verify_ns,
            "verify_tokens": self.verify_tokens,
        }


def normalize_ladder(values: Sequence[int]) -> tuple[int, ...]:
    ladder = tuple(int(value) for value in values)
    if not ladder or any(value <= 0 for value in ladder):
        raise ValueError("prompt-lookup ladder must contain positive integers")
    if any(right <= left for left, right in zip(ladder, ladder[1:])):
        raise ValueError("prompt-lookup ladder must be strictly increasing")
    return ladder


def plan_proposal_around_verify_cliff(
    nominal_span: int,
    available_span: int,
    pending_rows: int = 1,
    *,
    cliff_start: int = 9,
    cliff_end: int = 15,
) -> int:
    """Avoid a qualified verify-width plateau without inventing tokens.

    If the nominal width lands inside the plateau, prefer its far side when
    enough continuation exists; otherwise stay immediately below it.  Hardware
    and model dependence is why callers must opt into this policy.
    """
    if min(nominal_span, available_span) < 0 or pending_rows < 1:
        raise ValueError("proposal spans must be nonnegative and pending_rows positive")
    if not 1 < cliff_start <= cliff_end:
        raise ValueError("invalid verify cliff")
    span = min(int(nominal_span), int(available_span))
    rows = pending_rows + span
    if cliff_start <= rows <= cliff_end:
        long_span = cliff_end + 1 - pending_rows
        if available_span >= long_span:
            return long_span
        return min(span, max(cliff_start - 1 - pending_rows, 0))
    return span


def max_proposal_span(policy) -> int:
    """Widest proposal one prompt-lookup verify round can carry.

    ``cliff_aware_span`` moves a nominal span whose verify width lands inside
    the cliff to its far side (``verify_cliff_end`` proposals), which can
    exceed ``num_draft``.  Admission and verify-row bounds charge this width.
    """
    policy = policy or {}
    num_draft = int(policy.get("num_draft", 8))
    if not policy.get("cliff_aware_span", False):
        return num_draft
    cliff_end = int(policy.get("verify_cliff_end", 15))
    return plan_proposal_around_verify_cliff(
        num_draft,
        max(num_draft, cliff_end),
        1,
        cliff_start=int(policy.get("verify_cliff_start", 9)),
        cliff_end=cliff_end,
    )


class AdaptiveLookback:
    """Miss-driven lookback with rejection backoff inside a scheduler cap."""

    def __init__(self, ladder=(256, 1024, 4096, 16384), *, misses=4, rejects=2):
        self.ladder = normalize_ladder(ladder)
        self.index = 0
        self.cap = self.ladder[-1]
        self.misses_to_widen = int(misses)
        self.rejects_to_narrow = int(rejects)
        if min(self.misses_to_widen, self.rejects_to_narrow) < 1:
            raise ValueError("adaptive thresholds must be positive")
        self._misses = self._rejects = 0
        self.widen_events = self.narrow_events = 0

    @property
    def current(self):
        return min(self.ladder[self.index], self.cap)

    def clamp(self, cap):
        self.cap = max(self.ladder[0], int(cap))
        while self.index and self.ladder[self.index] > self.cap:
            self.index -= 1

    def observe(self, proposed, accepted):
        if proposed < 0 or not 0 <= accepted <= proposed:
            raise ValueError("invalid prompt-lookup outcome")
        if not proposed:
            self._rejects = 0
            self._misses += 1
            if self._misses >= self.misses_to_widen:
                maximum = max(i for i, value in enumerate(self.ladder) if value <= self.cap)
                if self.index < maximum:
                    self.index += 1
                    self.widen_events += 1
                self._misses = 0
        elif not accepted:
            self._misses = 0
            self._rejects += 1
            if self._rejects >= self.rejects_to_narrow:
                if self.index:
                    self.index -= 1
                    self.narrow_events += 1
                self._rejects = 0
        else:
            self._misses = self._rejects = 0


@dataclass
class PromptLookupIndexStats:
    """Timer-free counters describing bounded host lookup work."""

    calls: int = 0
    hits: int = 0
    misses: int = 0
    target_key_lookups: int = 0
    hot_key_lookups: int = 0
    target_occurrences_visited: int = 0
    target_occurrences_skipped: int = 0
    hot_occurrences_visited: int = 0
    hot_starts_avoided: int = 0
    candidate_checks: int = 0
    rejected_candidates: int = 0
    sources_rejected: int = 0
    target_occurrences_indexed: int = 0
    hot_segments_indexed: int = 0
    hot_tokens_indexed: int = 0
    hot_occurrences_indexed: int = 0

    def receipt(self, *, index_window: int) -> dict:
        return {
            "schema": "mlx2.prompt-lookup-index.v1",
            "index_window": index_window,
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "target_key_lookups": self.target_key_lookups,
            "hot_key_lookups": self.hot_key_lookups,
            "target_occurrences_visited": self.target_occurrences_visited,
            "target_occurrences_skipped": self.target_occurrences_skipped,
            "hot_occurrences_visited": self.hot_occurrences_visited,
            "hot_starts_avoided": self.hot_starts_avoided,
            "candidate_checks": self.candidate_checks,
            "rejected_candidates": self.rejected_candidates,
            "sources_rejected": self.sources_rejected,
            "target_occurrences_indexed": self.target_occurrences_indexed,
            "hot_segments_indexed": self.hot_segments_indexed,
            "hot_tokens_indexed": self.hot_tokens_indexed,
            "hot_occurrences_indexed": self.hot_occurrences_indexed,
        }


@dataclass(frozen=True)
class _IndexedSegment:
    tokens: tuple[int, ...]
    index: Mapping[int, Mapping[tuple[int, ...], tuple[int, ...]]]
    source_kind: str = "retrieval"
    ngram_min: int = 3
    ngram_max: int = 6
    index_window: int | None = None


def _immutable_segment_index(
    tokens, *, ngram_min, ngram_max, index_window=None
):
    """Build read-only postings that can be shared safely across PLD lanes."""
    buckets_by_size = {}
    length = len(tokens)
    for size in range(ngram_min, ngram_max + 1):
        buckets = defaultdict(list)
        first_start = 0
        if index_window:
            first_start = max(0, length - index_window - size)
        for start in range(first_start, length - size + 1):
            buckets[tuple(tokens[start : start + size])].append(start)
        buckets_by_size[size] = MappingProxyType(
            {key: tuple(positions) for key, positions in buckets.items()}
        )
    return MappingProxyType(buckets_by_size)


@dataclass(frozen=True)
class RecentCommittedSegment:
    """One completed-response token span available to later PLD lanes."""

    scope: object
    segment_id: str
    cache_id: int
    tokens: tuple[int, ...]
    completed_at: float


class RecentCommittedSegmentStore:
    """Bounded, engine-owned source segments for cross-request PLD.

    The store owns token IDs only, never target or APCv2 state.  Callers bind
    ``scope`` to the same exact identity used for APCv2, so model, tokenizer,
    adapter, numerical and (when selected) tenant boundaries cannot mix.
    Snapshots are immutable and completed responses are published atomically.
    """

    def __init__(
        self,
        *,
        max_responses=64,
        max_tokens=131072,
        max_response_tokens=2048,
        max_age_seconds=1800.0,
        clock=time.monotonic,
    ):
        for name, value in (
            ("max_responses", max_responses),
            ("max_tokens", max_tokens),
            ("max_response_tokens", max_response_tokens),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"PLD recent segment {name} must be a positive integer"
                )
        if isinstance(max_age_seconds, bool) or not isinstance(
            max_age_seconds, (int, float)
        ) or not math.isfinite(float(max_age_seconds)) or max_age_seconds <= 0:
            raise ValueError(
                "PLD recent segment max_age_seconds must be finite and positive"
            )
        self.max_responses = max_responses
        self.max_tokens = max_tokens
        self.max_response_tokens = max_response_tokens
        self.max_age_seconds = float(max_age_seconds)
        self._clock = clock
        self._entries = deque()
        self._tokens = 0
        self._serial = 0
        self._stats = Counter()
        self._lock = threading.Lock()
        # One immutable index can serve every lane admitted against the same
        # exact APCv2 scope and source. Keep the cache no larger than the raw
        # response store; transcript entries share this same LRU budget.
        self._index_cache = OrderedDict()
        self._index_cache_limit = max_responses
        self._index_cache_tokens = 0
        self._index_scope_stats = defaultdict(Counter)

    @staticmethod
    def _index_key(
        scope,
        source_kind,
        source_id,
        tokens,
        ngram_min,
        ngram_max,
        index_window,
    ):
        try:
            hash(scope)
        except TypeError as error:
            raise ValueError(
                "PLD shared index scope must be an exact hashable APCv2 key"
            ) from error
        return (
            scope,
            str(source_kind),
            str(source_id),
            hash(tokens),
            len(tokens),
            int(ngram_min),
            int(ngram_max),
            index_window,
        )

    def _drop_indexes_locked(self, scope, source_kind, source_id):
        doomed = [
            key
            for key in self._index_cache
            if key[0] == scope and key[1] == source_kind and key[2] == str(source_id)
        ]
        for key in doomed:
            indexed = self._index_cache.pop(key, None)
            if indexed is not None:
                self._index_cache_tokens -= len(indexed.tokens)
            self._stats["shared_index_lifecycle_evictions"] += 1
            self._index_scope_stats[key[0]][
                "shared_index_lifecycle_evictions"
            ] += 1

    def _expire_locked(self, now):
        boundary = now - self.max_age_seconds
        while self._entries and self._entries[0].completed_at < boundary:
            entry = self._entries.popleft()
            self._tokens -= len(entry.tokens)
            self._drop_indexes_locked(
                entry.scope, "recent_committed", entry.cache_id
            )
            self._stats["age_evictions"] += 1

    def publish(self, scope, tokens, *, segment_id=None):
        """Publish one successfully completed response tail."""
        if scope is None:
            raise ValueError("PLD recent segment scope is required")
        values = tuple(tokens)
        if not values:
            with self._lock:
                self._stats["empty_skips"] += 1
            return None
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError("PLD recent segment tokens must be nonnegative")
        # A configured per-response limit can be wider than the store's total
        # token budget.  Clamp to both limits so a single valid publication
        # cannot immediately evict itself from an otherwise empty store.
        values = values[-min(self.max_response_tokens, self.max_tokens) :]
        now = float(self._clock())
        with self._lock:
            self._expire_locked(now)
            # Repeated deterministic responses are common in serving.  Keep
            # one newest copy per exact scope instead of letting duplicates
            # consume the global response and token budgets.
            duplicate = next(
                (
                    entry
                    for entry in reversed(self._entries)
                    if entry.scope == scope and entry.tokens == values
                ),
                None,
            )
            if duplicate is not None:
                self._entries.remove(duplicate)
                self._tokens -= len(duplicate.tokens)
                self._stats["duplicate_refreshes"] += 1
            self._serial += 1
            entry = RecentCommittedSegment(
                scope=scope,
                segment_id=(
                    str(segment_id)
                    if segment_id is not None
                    else f"completion:{self._serial}"
                ),
                # Exact same-scope tokens keep their immutable posting cache;
                # only freshness and the user-visible segment id change.
                cache_id=(duplicate.cache_id if duplicate is not None else self._serial),
                tokens=values,
                completed_at=now,
            )
            self._entries.append(entry)
            self._tokens += len(values)
            self._stats["published"] += 1
            while (
                len(self._entries) > self.max_responses
                or self._tokens > self.max_tokens
            ):
                evicted = self._entries.popleft()
                self._tokens -= len(evicted.tokens)
                self._drop_indexes_locked(
                    evicted.scope, "recent_committed", evicted.cache_id
                )
                self._stats["capacity_evictions"] += 1
            return entry

    def indexed_segment(
        self,
        scope,
        tokens,
        *,
        source_kind,
        source_id,
        ngram_min,
        ngram_max,
        index_window=None,
    ):
        """Return one bounded engine-owned immutable hot-source index."""
        values = tuple(tokens)
        if not values:
            raise ValueError("PLD shared index segment must be nonempty")
        if any(type(token) is not int or token < 0 for token in values):
            raise ValueError("PLD shared index tokens must be nonnegative integers")
        if source_kind not in {"apcv2_transcript", "recent_committed"}:
            raise ValueError("PLD shared index source kind is unsupported")
        if ngram_min < 1 or ngram_max < ngram_min:
            raise ValueError("invalid shared PLD ngram bounds")
        if index_window is not None and (
            type(index_window) is not int or index_window < 1
        ):
            raise ValueError("shared PLD index window must be a positive integer")
        key = self._index_key(
            scope,
            source_kind,
            source_id,
            values,
            ngram_min,
            ngram_max,
            index_window,
        )
        with self._lock:
            indexed = self._index_cache.get(key)
            if indexed is not None and indexed.tokens == values:
                self._index_cache.move_to_end(key)
                self._stats["shared_index_hits"] += 1
                self._index_scope_stats[scope]["shared_index_hits"] += 1
                return indexed
            self._stats["shared_index_misses"] += 1
            self._index_scope_stats[scope]["shared_index_misses"] += 1
        built = _IndexedSegment(
            values,
            _immutable_segment_index(
                values,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
                index_window=index_window,
            ),
            source_kind=source_kind,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            index_window=index_window,
        )
        with self._lock:
            indexed = self._index_cache.get(key)
            if indexed is not None and indexed.tokens == values:
                self._index_cache.move_to_end(key)
                self._stats["shared_index_race_reuses"] += 1
                self._index_scope_stats[scope]["shared_index_race_reuses"] += 1
                return indexed
            replaced = self._index_cache.pop(key, None)
            if replaced is not None:
                self._index_cache_tokens -= len(replaced.tokens)
            self._index_cache[key] = built
            self._index_cache_tokens += len(built.tokens)
            self._index_cache.move_to_end(key)
            self._stats["shared_index_builds"] += 1
            self._index_scope_stats[scope]["shared_index_builds"] += 1
            while (
                len(self._index_cache) > self._index_cache_limit
                or self._index_cache_tokens > self.max_tokens
            ):
                evicted_key, evicted = self._index_cache.popitem(last=False)
                self._index_cache_tokens -= len(evicted.tokens)
                self._stats["shared_index_capacity_evictions"] += 1
                self._index_scope_stats[evicted_key[0]][
                    "shared_index_capacity_evictions"
                ] += 1
            return built

    def snapshot(self, scope):
        """Return newest-first immutable source spans for one APCv2 scope."""
        now = float(self._clock())
        with self._lock:
            self._expire_locked(now)
            entries = tuple(
                entry for entry in reversed(self._entries) if entry.scope == scope
            )
            self._stats["snapshots"] += 1
            self._stats["snapshot_segments"] += len(entries)
            return entries

    def status(self, scope=None):
        now = float(self._clock())
        with self._lock:
            self._expire_locked(now)
            if scope is None:
                entries = tuple(self._entries)
                stats = dict(self._stats)
            else:
                entries = tuple(
                    entry for entry in self._entries if entry.scope == scope
                )
                # Per-request receipts must not disclose activity in another
                # exact APCv2 scope through global cumulative counters.
                stats = dict(self._index_scope_stats.get(scope, ()))
            shared_indexes = tuple(
                indexed
                for key, indexed in self._index_cache.items()
                if scope is None or key[0] == scope
            )
            return {
                "schema": "mlx2.pld-recent-committed-segments.v1",
                "responses": len(entries),
                "tokens": sum(len(entry.tokens) for entry in entries),
                "max_responses": self.max_responses,
                "max_tokens": self.max_tokens,
                "max_response_tokens": self.max_response_tokens,
                "max_age_seconds": self.max_age_seconds,
                "shared_index_entries": len(shared_indexes),
                "shared_index_tokens": sum(
                    len(indexed.tokens) for indexed in shared_indexes
                ),
                "shared_index_limit": self._index_cache_limit,
                **stats,
            }


class IndexedPromptLookup:
    """Exact indexed oracle with bounded hot segments and rejected-source TTL."""

    def __init__(
        self, tokens=(), *, ngram_min=3, ngram_max=6, hot_segments=4,
        reject_ttl=8, recent_prompt_segments=0, prompt_segment_tokens=1024,
        index_window=None,
    ):
        if ngram_min < 1 or ngram_max < ngram_min:
            raise ValueError("invalid ngram bounds")
        if recent_prompt_segments < 0 or prompt_segment_tokens < 1:
            raise ValueError("invalid recent prompt segment bounds")
        if index_window is not None and (
            type(index_window) is not int or index_window < 1
        ):
            raise ValueError("index_window must be a positive integer")
        self.ngram_min, self.ngram_max = ngram_min, ngram_max
        self.tokens = [int(token) for token in tokens]
        self.recent_prompt_segments = int(recent_prompt_segments)
        self.prompt_segment_tokens = int(prompt_segment_tokens)
        configured_window = self.recent_prompt_segments * self.prompt_segment_tokens
        self.index_window = index_window if index_window is not None else configured_window
        bucket = deque if self.index_window else list
        self.index = {size: defaultdict(bucket) for size in range(ngram_min, ngram_max + 1)}
        self.index_entries = 0
        self.indexed_start = max(0, len(self.tokens) - self.index_window) if self.index_window else 0
        self.hot = deque(maxlen=int(hot_segments))
        self.reject_ttl = int(reject_ttl)
        self.clock = 0
        self.rejected_until = {}
        self.last_source = None
        self.lookup_calls = 0
        self.source_sites_scanned = 0
        self.context_mismatch_sites = 0
        self.index_stats = PromptLookupIndexStats()
        for size, buckets in self.index.items():
            for start in range(self.indexed_start, len(self.tokens) - size + 1):
                buckets[tuple(self.tokens[start : start + size])].append(start)
                self.index_entries += 1
        self.index_stats.target_occurrences_indexed = self.index_entries

    def _build_hot_index(self, tokens):
        return _immutable_segment_index(
            tokens,
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            index_window=self.index_window,
        )

    def observe(self, token):
        self.tokens.append(int(token))
        end = len(self.tokens)
        for size, buckets in self.index.items():
            start = end - size
            if start >= 0 and (not self.index_window or start >= end - self.index_window):
                buckets[tuple(self.tokens[start:end])].append(start)
                self.index_entries += 1
        if self.index_window:
            expired = end - self.index_window - 1
            if expired >= self.indexed_start:
                for size, buckets in self.index.items():
                    if expired + size > end:
                        continue
                    key = tuple(self.tokens[expired : expired + size])
                    positions = buckets.get(key)
                    if positions and positions[0] == expired:
                        positions.popleft()
                        self.index_entries -= 1
                        if not positions:
                            del buckets[key]
                self.indexed_start = expired + 1
        self.index_stats.target_occurrences_indexed = self.index_entries

    def add_hot_segment(self, tokens, *, source_kind="retrieval"):
        segment = tuple(int(token) for token in tokens)
        if not segment:
            raise ValueError("hot segment must be nonempty")
        if source_kind not in {"retrieval", "apcv2_transcript", "recent_committed"}:
            raise ValueError("unknown prompt-lookup hot segment source")
        if self.hot.maxlen == 0:
            return
        indexed = _IndexedSegment(
            segment,
            self._build_hot_index(segment),
            source_kind=source_kind,
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            index_window=self.index_window,
        )
        self.add_indexed_hot_segment(indexed)

    def add_indexed_hot_segment(self, indexed):
        """Attach a compatible immutable segment without rebuilding postings."""
        if self.hot.maxlen == 0:
            return
        if not isinstance(indexed, _IndexedSegment):
            raise TypeError("indexed hot segment must be an _IndexedSegment")
        if (
            indexed.ngram_min != self.ngram_min
            or indexed.ngram_max != self.ngram_max
            or indexed.index_window != self.index_window
        ):
            raise ValueError("indexed hot segment geometry does not match proposer")
        if indexed.source_kind not in {
            "retrieval",
            "apcv2_transcript",
            "recent_committed",
        }:
            raise ValueError("unknown prompt-lookup hot segment source")
        segment = indexed.tokens
        if not segment:
            raise ValueError("hot segment must be nonempty")
        if self.hot.maxlen is not None and len(self.hot) == self.hot.maxlen:
            evicted = self.hot[0]
            self.index_stats.hot_tokens_indexed -= len(evicted.tokens)
            self.index_stats.hot_occurrences_indexed -= sum(
                len(positions)
                for buckets in evicted.index.values()
                for positions in buckets.values()
            )
        self.hot.append(indexed)
        self.index_stats.hot_segments_indexed = len(self.hot)
        self.index_stats.hot_tokens_indexed += len(segment)
        self.index_stats.hot_occurrences_indexed += sum(
            len(positions)
            for buckets in indexed.index.values()
            for positions in buckets.values()
        )

    def _best_source_candidate(
        self,
        *,
        kind,
        segment_id,
        tokens,
        positions,
        size,
        max_span,
        first_start,
        final_start,
        min_context_match,
        max_sources,
        old_hot_starts=0,
    ):
        visited = 0
        chosen = None

        def consider(start):
            continuation = tuple(tokens[start + size : start + size + max_span])
            if not continuation:
                return None
            source = (
                ("target", size, start, continuation)
                if kind == "target"
                else ("hot", segment_id, size, start, continuation)
            )
            self.index_stats.candidate_checks += 1
            if self.rejected_until.get(source, -1) >= self.clock:
                self.index_stats.rejected_candidates += 1
                return None
            return continuation, source, (len(continuation), start)

        if not min_context_match and not max_sources:
            # Skip the at-most-max_span recent partial sites without slicing
            # them. The first accepted full site is optimal; partial sites are
            # revisited oldest-first only when every full source is rejected.
            partial_starts = []
            full_boundary = len(tokens) - size - max_span
            for start in reversed(positions):
                if start >= final_start:
                    continue
                if start < first_start:
                    break
                visited += 1
                self.source_sites_scanned += 1
                if start > full_boundary:
                    partial_starts.append(start)
                    continue
                chosen = consider(start)
                if chosen is not None:
                    break
            if chosen is None:
                for start in reversed(partial_starts):
                    chosen = consider(start)
                    if chosen is not None:
                        break
        else:
            for start in reversed(positions):
                if start >= final_start:
                    continue
                if start < first_start:
                    break
                if max_sources and visited >= max_sources:
                    break
                visited += 1
                self.source_sites_scanned += 1
                if min_context_match and not self._context_matches(
                    tokens, start, size, min_context_match
                ):
                    self.context_mismatch_sites += 1
                    continue
                candidate = consider(start)
                if candidate is not None and (
                    chosen is None or candidate[2] > chosen[2]
                ):
                    chosen = candidate
                if chosen is not None and len(chosen[0]) == max_span:
                    break

        if kind == "target":
            self.index_stats.target_occurrences_visited += visited
            self.index_stats.target_occurrences_skipped += max(
                0, len(positions) - visited
            )
        else:
            self.index_stats.hot_occurrences_visited += visited
            self.index_stats.hot_starts_avoided += max(0, old_hot_starts - visited)
        return chosen

    def propose(self, max_span, *, lookback=4096, min_context_match=0, max_sources=0):
        """Find the exact best copy source without materializing candidates."""
        self.clock += 1
        self.lookup_calls += 1
        self.index_stats.calls += 1
        self.last_source = None
        if max_span <= 0 or lookback <= 0:
            self.index_stats.misses += 1
            return []
        size_limit = min(self.ngram_max, len(self.tokens))
        for size in range(size_limit, self.ngram_min - 1, -1):
            key = tuple(self.tokens[-size:])
            self.index_stats.target_key_lookups += 1
            chosen = self._best_source_candidate(
                kind="target",
                segment_id=None,
                tokens=self.tokens,
                positions=self.index[size].get(key, ()),
                size=size,
                max_span=max_span,
                first_start=len(self.tokens) - lookback,
                final_start=len(self.tokens) - size,
                min_context_match=min_context_match,
                max_sources=max_sources,
            )
            for segment_id, segment in enumerate(self.hot):
                self.index_stats.hot_key_lookups += 1
                first_start = max(0, len(segment.tokens) - lookback - size)
                final_start = len(segment.tokens) - size
                candidate = self._best_source_candidate(
                    kind="hot",
                    segment_id=segment_id,
                    tokens=segment.tokens,
                    positions=segment.index[size].get(key, ()),
                    size=size,
                    max_span=max_span,
                    first_start=first_start,
                    final_start=final_start,
                    min_context_match=min_context_match,
                    max_sources=max_sources,
                    old_hot_starts=max(0, final_start - first_start),
                )
                if candidate is not None and (
                    chosen is None or candidate[2] > chosen[2]
                ):
                    chosen = candidate
            if chosen is not None:
                self.last_source = chosen[1]
                self.index_stats.hits += 1
                return list(chosen[0])
        self.index_stats.misses += 1
        return []

    def _context_matches(self, source, start, size, minimum):
        if minimum <= size:
            return True
        extra = minimum - size
        if start < extra or len(self.tokens) - size < extra:
            return False
        # Hot segments are tuples and local history a list; a tuple never
        # equals a list, so compare like types.
        return tuple(source[start - extra : start]) == tuple(
            self.tokens[-size - extra : -size]
        )

    def feedback(self, proposed, accepted):
        if self.last_source is not None and proposed and not accepted and self.reject_ttl:
            self.rejected_until[self.last_source] = self.clock + self.reject_ttl
            self.index_stats.sources_rejected += 1

    @property
    def last_source_kind(self):
        if self.last_source is None:
            return None
        if self.last_source[0] == "target":
            return "local"
        segment_id = self.last_source[1]
        if 0 <= segment_id < len(self.hot):
            return self.hot[segment_id].source_kind
        return "unknown"

    def index_receipt(self):
        self.index_stats.target_occurrences_indexed = self.index_entries
        return self.index_stats.receipt(index_window=self.index_window)


@dataclass(frozen=True)
class PromptLookupVerification:
    accepted: tuple[int, ...]
    next_token: int
    proposed: int
    receipt: dict


def verify_prompt_lookup(proposal, target_tokens):
    """Accept the exact matching prefix and return the target boundary token."""
    proposal = tuple(int(token) for token in proposal)
    target = tuple(int(token) for token in target_tokens)
    if len(target) < len(proposal) + 1:
        raise ValueError("target verification must include one bonus token")
    accepted = 0
    while accepted < len(proposal) and proposal[accepted] == target[accepted]:
        accepted += 1
    return PromptLookupVerification(
        accepted=proposal[:accepted],
        next_token=target[accepted],
        proposed=len(proposal),
        receipt={
            "schema": "mlx2.prompt-lookup.v1",
            "verified": True,
            "proposed": len(proposal),
            "accepted": accepted,
            "bonus": accepted == len(proposal),
        },
    )
