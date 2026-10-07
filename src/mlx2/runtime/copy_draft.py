# SPDX-License-Identifier: Apache-2.0
"""Copy-drafts inside the self-MTP verify transaction.

Original mlx2 code.  Design ideas (congestion-window span sizing, a windowed
ratio-of-sums gate, point-mass verification under sampling, indexing the full
APC-restored prompt) come from Rapid-MLX #3398/#3417 and SwitchSD
(arXiv 2609.20186); see docs/PROVENANCE.md.

A lane that owns a :class:`CopyDraftState` may, on any self-MTP round,
propose a verbatim continuation copied from its own context instead of the
MTP head's drafts.  The target verifies both proposal kinds in the same
batched forward under the exact acceptance law, so this module only decides
*what* to propose; it never decides what is emitted.

Everything here is host-side integer bookkeeping: no device work, no syncs,
no wall clocks.

Snapshot contract.  Segmented self-MTP deep-copies non-array lane fields at
every committed recovery boundary.  The token history and n-gram index are an
append-only store shared between a state and its copies; each state carries
its own ``length`` watermark.  A restored (older) state truncates the shared
store back to its watermark before its next read or write, so a deep copy
costs O(window) rather than O(context).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

COPY_DRAFT_RECEIPT_SCHEMA = "mlx2.self-mtp-copy-draft.v1"
# Longest agreement a lookup measures, and how many earlier occurrences of a
# suffix it examines for one that clears ``min_match``.
MATCH_SCAN_CAP = 64
MATCH_SCAN_SOURCES = 8


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"copy-draft {name} must be an integer >= {minimum}")
    return value


def _nonnegative_float(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"copy-draft {name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class CopyDraftPolicy:
    """Validated execution-policy block ``self_mtp_copy_draft``."""

    enabled: bool = False
    ngram_min: int = 3
    ngram_max: int = 6
    max_span: int = 8
    # Copy width inside a cohort of more than one lane.  0 (the default)
    # refuses to copy at all while batched: measured on GPU 2026-09-19
    # (Qwen3.8-27B, qualification/runs/copy-mtp-20260919/ab-27b-t0.json),
    # cohort copies capped at the head depth cost 5% on code and 3% on prose
    # at B=4, while B=1 gained 20%.  ``None`` restores the head-depth cap and
    # a positive value sets it explicitly.
    batched_max_span: Optional[int] = 0
    probe_span: int = 2
    lookback: int = 8192
    min_samples: int = 3
    gate_window: int = 32
    reprobe_interval: int = 16
    min_yield_ratio: float = 1.0
    verify_row_cost: float = 0.1
    draft_step_cost: float = 0.15
    # Match-strength admission (design input: ddalcu/mlx-serve #523, see
    # docs/PROVENANCE.md).  ``min_match`` > 0 copies only when the matched
    # site agrees with the live context for at least that many tokens going
    # back (the n-gram included, counted up to MATCH_SCAN_CAP); 0 keeps the
    # historical lookup exactly.  A match agreeing for ``strong_match`` (> 0)
    # tokens may copy up to ``strong_max_span`` instead of ``max_span``.
    # ``initial_span`` starts the sizer there instead of at ``probe_span``.
    min_match: int = 0
    strong_match: int = 0
    strong_max_span: int = 0
    initial_span: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("copy-draft enabled must be boolean")  # noqa: TRY004
        _positive_int("ngram_min", self.ngram_min)
        _positive_int("ngram_max", self.ngram_max)
        if self.ngram_max < self.ngram_min:
            raise ValueError("copy-draft ngram_max must be >= ngram_min")
        _positive_int("max_span", self.max_span)
        if self.batched_max_span is not None:
            _positive_int("batched_max_span", self.batched_max_span, minimum=0)
        _positive_int("probe_span", self.probe_span)
        if self.probe_span > self.max_span:
            raise ValueError("copy-draft probe_span must be <= max_span")
        _positive_int("lookback", self.lookback)
        _positive_int("min_samples", self.min_samples, minimum=0)
        _positive_int("gate_window", self.gate_window)
        _positive_int("reprobe_interval", self.reprobe_interval)
        _nonnegative_float("min_yield_ratio", self.min_yield_ratio)
        _nonnegative_float("verify_row_cost", self.verify_row_cost)
        _nonnegative_float("draft_step_cost", self.draft_step_cost)
        _positive_int("min_match", self.min_match, minimum=0)
        _positive_int("strong_match", self.strong_match, minimum=0)
        _positive_int("strong_max_span", self.strong_max_span, minimum=0)
        if (self.strong_match == 0) != (self.strong_max_span == 0):
            raise ValueError(
                "copy-draft strong_match and strong_max_span must be set together"
            )
        if self.strong_match and self.strong_match < self.min_match:
            raise ValueError("copy-draft strong_match must be >= min_match")
        if max(self.min_match, self.strong_match) > MATCH_SCAN_CAP:
            raise ValueError(f"copy-draft match lengths must be <= {MATCH_SCAN_CAP}")
        if self.initial_span is not None:
            _positive_int("initial_span", self.initial_span)
            if self.initial_span > self.span_ceiling:
                raise ValueError(
                    "copy-draft initial_span must be <= max(max_span, strong_max_span)"
                )

    @classmethod
    def from_value(cls, value: Any) -> "CopyDraftPolicy":
        if value is None or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise ValueError("self_mtp_copy_draft must be an object, boolean or null")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown self_mtp_copy_draft keys: {unknown}")
        return cls(**dict(value))

    def as_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        # Match-strength keys enter receipts only when set, so receipts and
        # qualification identities of existing copy-draft profiles are unchanged.
        for name in _MATCH_FIELDS:
            if values[name] == _MATCH_DEFAULTS[name]:
                del values[name]
        return values

    @property
    def span_ceiling(self) -> int:
        """Widest span any single-lane round may copy."""
        return max(self.max_span, self.strong_max_span)

    def clamped_to_self_mtp_proposer_depth(self, maximum: int) -> "CopyDraftPolicy":
        """Return this policy bounded by an adapter's native self-MTP contract.

        ``maximum`` is proposal depth, not verify rows.  Only unsafe widths are
        reduced; a policy already inside the adapter contract is returned
        unchanged so existing defaults and qualification identities stay put.
        """
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("copy-draft proposer-depth cap must be an integer >= 1")
        batched = self.batched_max_span
        batched_safe = batched is None or batched <= maximum
        if self.span_ceiling <= maximum and batched_safe:
            return self
        max_span = min(self.max_span, maximum)
        strong_max_span = (
            min(self.strong_max_span, maximum) if self.strong_max_span else 0
        )
        ceiling = max(max_span, strong_max_span)
        return replace(
            self,
            max_span=max_span,
            strong_max_span=strong_max_span,
            probe_span=min(self.probe_span, max_span),
            initial_span=(
                None if self.initial_span is None else min(self.initial_span, ceiling)
            ),
            batched_max_span=(
                None if batched is None else min(batched, maximum)
            ),
        )

    def head_cost(self, depth: int) -> float:
        return 1.0 + (self.draft_step_cost + self.verify_row_cost) * max(int(depth), 0)

    def copy_cost(self, span: int) -> float:
        return 1.0 + self.verify_row_cost * max(int(span), 0)


_MATCH_DEFAULTS = {
    "min_match": 0, "strong_match": 0, "strong_max_span": 0, "initial_span": None,
}
_MATCH_FIELDS = tuple(_MATCH_DEFAULTS)


class _CopyIndexStore:
    """Append-only token history with an exact n-gram start-position index."""

    __slots__ = ("tokens", "index", "ngram_min", "ngram_max")

    def __init__(self, ngram_min: int, ngram_max: int):
        self.tokens: List[int] = []
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.index: Dict[int, Dict[tuple, List[int]]] = {
            size: {} for size in range(ngram_min, ngram_max + 1)
        }

    def append(self, token: int) -> None:
        self.tokens.append(int(token))
        end = len(self.tokens)
        for size, table in self.index.items():
            start = end - size
            if start >= 0:
                key = tuple(self.tokens[start:end])
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [start]
                else:
                    bucket.append(start)

    def truncate(self, length: int) -> None:
        """Drop every token at or beyond ``length`` and its index entries."""
        while len(self.tokens) > length:
            end = len(self.tokens)
            for size, table in self.index.items():
                start = end - size
                if start < 0:
                    continue
                key = tuple(self.tokens[start:end])
                bucket = table.get(key)
                if bucket and bucket[-1] == start:
                    bucket.pop()
                    if not bucket:
                        del table[key]
            self.tokens.pop()


class CopyDraftState:
    """Per-lane copy index, span sizer and copy-vs-head gate."""

    def __init__(self, policy: CopyDraftPolicy, context: Sequence[int] = ()):
        if not isinstance(policy, CopyDraftPolicy) or not policy.enabled:
            raise ValueError("CopyDraftState requires an enabled CopyDraftPolicy")
        self.policy = policy
        self._store = _CopyIndexStore(policy.ngram_min, policy.ngram_max)
        self.length = 0
        # Sizer: next copy width (congestion window).
        self.width = (
            policy.initial_span if policy.initial_span is not None else policy.probe_span
        )
        # Gate windows: (emitted tokens, cost) per round of each source.
        self._copy_window: deque = deque(maxlen=policy.gate_window)
        self._head_window: deque = deque(maxlen=policy.gate_window)
        self._declines_since_probe = 0
        # Cumulative per-source accounting (receipt; bounded host ints).
        self.copy_rounds = 0
        self.copy_proposed = 0
        self.copy_accepted = 0
        self.head_rounds = 0
        self.head_proposed = 0
        self.head_accepted = 0
        self.gate_declines = 0
        self.probe_rounds = 0
        self.lookup_misses = 0
        self.cohort_refusals = 0
        self.strong_matches = 0
        self.observe(context)

    # -- snapshot contract ------------------------------------------------
    def __deepcopy__(self, memo):
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        for name, value in vars(self).items():
            if name == "_store" or name == "policy":
                setattr(clone, name, value)
            elif isinstance(value, deque):
                setattr(clone, name, deque(value, maxlen=value.maxlen))
            else:
                setattr(clone, name, value)
        return clone

    def _sync(self) -> None:
        if len(self._store.tokens) != self.length:
            if len(self._store.tokens) < self.length:
                raise RuntimeError("copy-draft store lost committed tokens")
            self._store.truncate(self.length)

    # -- index ------------------------------------------------------------
    def observe(self, tokens: Sequence[int]) -> None:
        self._sync()
        for token in tokens:
            self._store.append(int(token))
        self.length = len(self._store.tokens)

    @property
    def index_tokens(self) -> int:
        return self.length

    def _agreement(self, end: int) -> int:
        """Tokens ending at ``end`` that equal the live suffix, going back."""
        tokens = self._store.tokens
        length = self.length
        count = 0
        while (
            count < MATCH_SCAN_CAP
            and end - 1 - count >= 0
            and tokens[end - 1 - count] == tokens[length - 1 - count]
        ):
            count += 1
        return count

    def _find(self) -> Optional[tuple]:
        """``(begin, agreement)`` of the copy source, or None.

        With ``min_match == 0`` this is the historical choice: the most recent
        continuation of the longest indexed suffix (agreement reported as that
        suffix length).  Otherwise the most recent of at most
        MATCH_SCAN_SOURCES earlier sources per suffix length whose agreement
        with the live context reaches ``min_match``.
        """
        self._sync()
        tokens = self._store.tokens
        length = self.length
        policy = self.policy
        for size in range(min(policy.ngram_max, length), policy.ngram_min - 1, -1):
            bucket = self._store.index[size].get(tuple(tokens[length - size : length]))
            if not bucket:
                continue
            examined = 0
            for start in reversed(bucket):
                begin = start + size
                if begin >= length:
                    continue  # the live suffix itself
                if length - start > policy.lookback:
                    break
                if policy.min_match <= 0:
                    return (begin, size)
                agreement = self._agreement(begin)
                if agreement >= policy.min_match:
                    return (begin, agreement)
                examined += 1
                if examined >= MATCH_SCAN_SOURCES:
                    break
        return None

    def _span_cap(self, agreement: int) -> int:
        policy = self.policy
        if policy.strong_match and agreement >= policy.strong_match:
            return policy.strong_max_span
        return policy.max_span

    def lookup(self, max_span: int) -> List[int]:
        """Verbatim continuation of the chosen source (see :meth:`_find`)."""
        if max_span <= 0:
            return []
        found = self._find()
        if found is None:
            return []
        begin, agreement = found
        width = min(max_span, self._span_cap(agreement))
        return list(self._store.tokens[begin : min(begin + width, self.length)])

    def has_candidate(self) -> bool:
        return bool(self.lookup(1))

    # -- gate + sizer -----------------------------------------------------
    @staticmethod
    def _rate(window: deque) -> Optional[float]:
        if not window:
            return None
        tokens = sum(item[0] for item in window)
        cost = sum(item[1] for item in window)
        return tokens / cost if cost > 0 else None

    def refuse_cohort(self) -> tuple:
        """Record a round whose cohort cap refused copies (``batched_max_span``).

        Distinct from ``"miss"``: a refused round never looked the source up,
        so its receipt must not read as "this text has no matches".
        """
        self.cohort_refusals += 1
        return ([], "cohort_refused")

    def plan(self, *, head_depth: int, cap: int) -> tuple:
        """Return ``(span_tokens, decision)`` for this round.

        ``decision`` is one of ``"copy"``, ``"probe"``, ``"declined"`` or
        ``"miss"`` (a cohort refusal is :meth:`refuse_cohort`).  Only the decline/probe bookkeeping mutates here; the index
        and sizer move only at commit (:meth:`record`).
        """
        cap = int(cap)
        if cap <= 0:
            return ([], "miss")
        policy = self.policy
        found = self._find()
        if found is None:
            self.lookup_misses += 1
            return ([], "miss")
        begin, agreement = found
        span_cap = self._span_cap(agreement)
        if span_cap != policy.max_span:
            self.strong_matches += 1
        width = min(self.width, cap, span_cap)
        span = list(self._store.tokens[begin : min(begin + width, self.length)])
        if not span:
            self.lookup_misses += 1
            return ([], "miss")
        if len(self._copy_window) < policy.min_samples:
            return (span, "copy")
        copy_rate = self._rate(self._copy_window)
        head_rate = self._rate(self._head_window)
        baseline = head_rate if head_rate is not None else 1.0
        if copy_rate is not None and copy_rate >= policy.min_yield_ratio * baseline:
            return (span, "copy")
        self._declines_since_probe += 1
        if self._declines_since_probe >= policy.reprobe_interval:
            self._declines_since_probe = 0
            self.probe_rounds += 1
            return (span[: min(policy.probe_span, len(span))], "probe")
        self.gate_declines += 1
        return ([], "declined")

    def record(
        self,
        *,
        copy_span: int,
        head_depth: int,
        accepted: int,
        emitted: int,
        committed: Sequence[int],
    ) -> None:
        """Account one committed round and index its committed tokens."""
        copy_span = int(copy_span)
        accepted = int(accepted)
        if copy_span < 0 or accepted < 0 or emitted < 0:
            raise ValueError("copy-draft outcome counts must be nonnegative")
        policy = self.policy
        if copy_span:
            if accepted > copy_span:
                raise ValueError("copy-draft accepted more than it proposed")
            self.copy_rounds += 1
            self.copy_proposed += copy_span
            self.copy_accepted += accepted
            self._copy_window.append((int(emitted), policy.copy_cost(copy_span)))
            if accepted >= copy_span:
                self.width = min(max(self.width, copy_span) * 2, policy.span_ceiling)
            else:
                self.width = max(
                    policy.probe_span, min(math.ceil(1.5 * accepted), copy_span)
                )
        else:
            self.head_rounds += 1
            self.head_proposed += int(head_depth)
            self.head_accepted += min(accepted, int(head_depth))
            self._head_window.append((int(emitted), policy.head_cost(head_depth)))
        self.observe(committed)

    def receipt(self) -> Dict[str, Any]:
        return {
            "schema": COPY_DRAFT_RECEIPT_SCHEMA,
            "enabled": True,
            "verification": "exact",
            "policy": self.policy.as_dict(),
            "copy_rounds": self.copy_rounds,
            "copy_proposed": self.copy_proposed,
            "copy_accepted": self.copy_accepted,
            "copy_acceptance": self.copy_accepted / max(self.copy_proposed, 1),
            "head_rounds": self.head_rounds,
            "head_proposed": self.head_proposed,
            "head_accepted": self.head_accepted,
            "head_acceptance": self.head_accepted / max(self.head_proposed, 1),
            "gate_declines": self.gate_declines,
            "probe_rounds": self.probe_rounds,
            "lookup_misses": self.lookup_misses,
            "cohort_refusals": self.cohort_refusals,
            **({"strong_matches": self.strong_matches} if self.policy.strong_match else {}),
            "index_tokens": self.index_tokens,
            "sizer_width": self.width,
        }


def cohort_copy_cap(
    policy: CopyDraftPolicy, *, lanes: int, head_depths: Sequence[int]
) -> int:
    """Widest copy row a cohort of ``lanes`` may verify this round."""
    if lanes <= 1:
        return policy.span_ceiling
    if policy.batched_max_span is not None:
        # 0 refuses cohort copies outright (the default).
        return min(policy.batched_max_span, policy.max_span)
    return min(max(max(head_depths, default=0), 1), policy.max_span)


def verify_point_mass_by_sampling(
    proposal: Sequence[int], sampled: Sequence[int]
) -> tuple:
    """Exact law for a point-mass (copied) proposal.

    ``sampled[i]`` must be an independent draw from the fully transformed
    target law at verify row ``i``.  Accepting while the draw equals the
    proposal, and emitting the first differing draw, accepts ``d`` with
    probability ``p(d)`` and otherwise emits from ``p`` restricted to ``!= d``
    and renormalised -- the speculative-sampling law for ``q = delta_d``.
    Returns ``(n_accept, next_token)``.
    """
    if len(sampled) < len(proposal) + 1:
        raise ValueError("point-mass verification needs one bonus draw")
    n = 0
    while n < len(proposal) and int(sampled[n]) == int(proposal[n]):
        n += 1
    return (n, int(sampled[n]))
