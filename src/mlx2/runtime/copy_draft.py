# SPDX-License-Identifier: Apache-2.0
"""Copy-drafts inside the self-MTP verify transaction (default off).

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
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Sequence

COPY_DRAFT_RECEIPT_SCHEMA = "mlx2.self-mtp-copy-draft.v1"


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
        return asdict(self)

    def head_cost(self, depth: int) -> float:
        return 1.0 + (self.draft_step_cost + self.verify_row_cost) * max(int(depth), 0)

    def copy_cost(self, span: int) -> float:
        return 1.0 + self.verify_row_cost * max(int(span), 0)


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
        self.width = policy.probe_span
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

    def lookup(self, max_span: int) -> List[int]:
        """Most recent verbatim continuation of the longest suffix match."""
        if max_span <= 0:
            return []
        self._sync()
        tokens = self._store.tokens
        length = self.length
        policy = self.policy
        for size in range(min(policy.ngram_max, length), policy.ngram_min - 1, -1):
            bucket = self._store.index[size].get(tuple(tokens[length - size : length]))
            if not bucket:
                continue
            for start in reversed(bucket):
                begin = start + size
                if begin >= length:
                    continue  # the live suffix itself
                if length - start > policy.lookback:
                    break
                return list(tokens[begin : min(begin + max_span, length)])
        return []

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

    def plan(self, *, head_depth: int, cap: int) -> tuple:
        """Return ``(span_tokens, decision)`` for this round.

        ``decision`` is one of ``"copy"``, ``"probe"``, ``"declined"`` or
        ``"miss"``.  Only the decline/probe bookkeeping mutates here; the index
        and sizer move only at commit (:meth:`record`).
        """
        cap = int(cap)
        if cap <= 0:
            return ([], "miss")
        policy = self.policy
        width = min(self.width, cap, policy.max_span)
        span = self.lookup(width)
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
                self.width = min(max(self.width, copy_span) * 2, policy.max_span)
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
            "index_tokens": self.index_tokens,
            "sizer_width": self.width,
        }


def cohort_copy_cap(
    policy: CopyDraftPolicy, *, lanes: int, head_depths: Sequence[int]
) -> int:
    """Widest copy row a cohort of ``lanes`` may verify this round."""
    if lanes <= 1:
        return policy.max_span
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
