# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field


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
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    retrieval_corpus_mode: str = "target"
    retrieval_corpus_tokens: int = 0
    lookback_current: int = 0
    lookback_peak: int = 0
    lookback_widen_events: int = 0
    lookback_narrow_events: int = 0
    source_rejections: int = 0
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


class IndexedPromptLookup:
    """Exact indexed oracle with bounded hot segments and rejected-source TTL."""

    def __init__(self, tokens=(), *, ngram_min=3, ngram_max=6, hot_segments=4, reject_ttl=8):
        if ngram_min < 1 or ngram_max < ngram_min:
            raise ValueError("invalid ngram bounds")
        self.ngram_min, self.ngram_max = ngram_min, ngram_max
        self.tokens = []
        self.index = {size: defaultdict(list) for size in range(ngram_min, ngram_max + 1)}
        self.hot = deque(maxlen=int(hot_segments))
        self.reject_ttl = int(reject_ttl)
        self.clock = 0
        self.rejected_until = {}
        self.last_source = None
        for token in tokens:
            self.observe(token)

    def observe(self, token):
        self.tokens.append(int(token))
        end = len(self.tokens)
        for size in self.index:
            start = end - size
            if start >= 0:
                self.index[size][tuple(self.tokens[start:end])].append(start)

    def add_hot_segment(self, tokens):
        segment = tuple(int(token) for token in tokens)
        if not segment:
            raise ValueError("hot segment must be nonempty")
        self.hot.append(segment)

    def propose(self, max_span, *, lookback=4096):
        self.clock += 1
        self.last_source = None
        size_limit = min(self.ngram_max, len(self.tokens))
        for size in range(size_limit, self.ngram_min - 1, -1):
            key = tuple(self.tokens[-size:])
            candidates = []
            for start in self.index[size].get(key, ()):
                if start >= len(self.tokens) - size or len(self.tokens) - start > lookback:
                    continue
                continuation = tuple(self.tokens[start + size : start + size + max_span])
                source = ("target", size, start, continuation)
                if continuation and self.rejected_until.get(source, -1) < self.clock:
                    candidates.append((start, continuation, source))
            for segment_id, segment in enumerate(self.hot):
                for start in range(max(0, len(segment) - lookback - size), len(segment) - size):
                    if segment[start : start + size] != key:
                        continue
                    continuation = segment[start + size : start + size + max_span]
                    source = ("hot", segment_id, size, start, continuation)
                    if continuation and self.rejected_until.get(source, -1) < self.clock:
                        candidates.append((start, continuation, source))
            if candidates:
                chosen = max(candidates, key=lambda item: (len(item[1]), item[0]))
                self.last_source = chosen[2]
                return list(chosen[1])
        return []

    def feedback(self, proposed, accepted):
        if self.last_source is not None and proposed and not accepted and self.reject_ttl:
            self.rejected_until[self.last_source] = self.clock + self.reject_ttl


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
