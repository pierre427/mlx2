# SPDX-License-Identifier: Apache-2.0
"""Host plan for exact longest-first continuation verification.

The planner never evaluates target logits and never mutates a cache.  It keeps
the authoritative target-token frontier for one request, orders complete
proposal paths longest first, and removes siblings that can no longer match
that frontier.  An adapter/executor may resume from ``proposal_start`` only
after attesting that its cache contains exact state for ``shared_prefix_tokens``.

After a rejection at proposal position ``n``, the target correction token is
emitted but has not yet been consumed as model input.  A surviving sibling is
therefore resumed at ``n``: proposal tokens before ``n`` stay in the exact
cache and the verification suffix begins with the correction token.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

ALGORITHM = "longest_first_exact_prefix_v1"


class PrefixCascadeError(ValueError):
    """The proposal set or a target observation violates the exact plan."""


def _paths(values: Iterable[Iterable[int]], *, maximum: int, depth: int):
    paths = tuple(tuple(path) for path in values)
    if not paths or len(paths) > maximum:
        raise PrefixCascadeError(f"proposal cascade requires 1..{maximum} paths")
    for path in paths:
        if (
            not path
            or len(path) > depth
            or any(
                isinstance(token, bool) or not isinstance(token, Integral) or token < 0
                for token in path
            )
        ):
            raise PrefixCascadeError(
                f"proposal paths require 1..{depth} nonnegative integer tokens"
            )
    return tuple(tuple(int(token) for token in path) for path in paths)


@dataclass(frozen=True)
class PrefixReuseGeometry:
    """Adapter-derived exact state geometry for one live cache boundary."""

    state_revision: str
    cache_layout: str
    position: int
    layer_count: int
    state_components: tuple[str, ...]
    component_widths: tuple[int, ...]
    exact: bool = True
    branchable: bool = True

    def __post_init__(self):
        if len(self.state_revision) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in self.state_revision
        ):
            raise PrefixCascadeError(
                "state_revision must be a pinned lowercase hexadecimal revision"
            )
        if not self.cache_layout:
            raise PrefixCascadeError("cache_layout is required")
        if (
            isinstance(self.position, bool)
            or not isinstance(self.position, Integral)
            or self.position < 0
            or isinstance(self.layer_count, bool)
            or not isinstance(self.layer_count, Integral)
            or self.layer_count < 1
        ):
            raise PrefixCascadeError("cache position/layer count is invalid")
        if (
            not self.state_components
            or len(self.state_components) != len(self.component_widths)
            or any(not name for name in self.state_components)
            or any(
                isinstance(width, bool) or not isinstance(width, Integral) or width < 1
                for width in self.component_widths
            )
        ):
            raise PrefixCascadeError("cache component geometry is incomplete")
        if type(self.exact) is not bool or type(self.branchable) is not bool:
            raise PrefixCascadeError("cache exactness flags must be boolean")
        if not self.exact or not self.branchable:
            raise PrefixCascadeError(
                "shared-prefix reuse requires exact branchable cache state"
            )

    def receipt(self):
        return {
            "state_revision": self.state_revision,
            "cache_layout": self.cache_layout,
            "position": int(self.position),
            "layer_count": int(self.layer_count),
            "state_components": list(self.state_components),
            "component_widths": [int(width) for width in self.component_widths],
            "exact": self.exact,
            "branchable": self.branchable,
        }


@dataclass(frozen=True)
class CascadeAttempt:
    candidate_index: int
    path: tuple[int, ...]
    proposal_start: int
    verification_suffix: tuple[int, ...]
    shared_prefix_tokens: int
    frontier: tuple[int, ...]

    def receipt(self):
        return {
            "candidate_index": self.candidate_index,
            "path_length": len(self.path),
            "proposal_start": self.proposal_start,
            "verification_suffix_tokens": len(self.verification_suffix),
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "frontier_tokens": len(self.frontier),
        }


@dataclass(frozen=True)
class CascadeObservation:
    full_match: bool
    matched_prefix_tokens: int
    correction_token: int | None
    frontier: tuple[int, ...]
    pruned_candidates: tuple[int, ...]
    surviving_candidates: tuple[int, ...]
    terminal: bool


class LongestFirstPrefixCascade:
    """Exact-prefix proposal ordering and sibling pruning for one round.

    ``matched_prefix_tokens`` passed to :meth:`observe` is absolute within the
    complete proposal path.  A partial match requires the target correction at
    that position.  Full acceptance terminates the round because the verifier
    also owns the target bonus/correction draw after the proposal.
    """

    def __init__(
        self,
        paths: Iterable[Iterable[int]],
        *,
        geometry: PrefixReuseGeometry | None = None,
        max_paths: int = 15,
        max_depth: int = 15,
    ):
        if (
            isinstance(max_paths, bool)
            or not isinstance(max_paths, Integral)
            or max_paths < 1
            or isinstance(max_depth, bool)
            or not isinstance(max_depth, Integral)
            or max_depth < 1
        ):
            raise PrefixCascadeError("cascade bounds must be positive integers")
        self.paths = _paths(paths, maximum=int(max_paths), depth=int(max_depth))
        self.geometry = geometry
        self._order = tuple(
            sorted(
                range(len(self.paths)),
                key=lambda index: (-len(self.paths[index]), index),
            )
        )
        self._remaining = list(self._order)
        self._attempted = set()
        self.frontier: tuple[int, ...] = ()
        self._open_attempt: CascadeAttempt | None = None
        self.terminal = False
        self.attempts = 0
        self.pruned = 0
        self.reused_prefix_tokens = 0

    def _survivors(self):
        width = len(self.frontier)
        return [
            index
            for index in self._remaining
            if index not in self._attempted
            and len(self.paths[index]) >= width
            and self.paths[index][:width] == self.frontier
        ]

    def next_attempt(self) -> CascadeAttempt | None:
        if self._open_attempt is not None:
            raise PrefixCascadeError("the current cascade attempt is still open")
        if self.terminal:
            return None
        viable = self._survivors()
        invalid = [
            index
            for index in self._remaining
            if index not in self._attempted and index not in viable
        ]
        if invalid:
            self.pruned += len(invalid)
            self._remaining = [
                index for index in self._remaining if index not in invalid
            ]
        if not viable:
            self.terminal = True
            return None
        index = viable[0]
        path = self.paths[index]
        # The latest correction is emitted but is the first not-yet-consumed
        # model input. Earlier frontier tokens may stay in an exact cache.
        proposal_start = max(0, len(self.frontier) - 1)
        shared = proposal_start if self.geometry is not None else 0
        start = proposal_start if self.geometry is not None else 0
        attempt = CascadeAttempt(
            candidate_index=index,
            path=path,
            proposal_start=start,
            verification_suffix=path[start:],
            shared_prefix_tokens=shared,
            frontier=self.frontier,
        )
        self._open_attempt = attempt
        self._attempted.add(index)
        self.attempts += 1
        self.reused_prefix_tokens += shared
        return attempt

    def observe(
        self,
        attempt: CascadeAttempt,
        *,
        matched_prefix_tokens: int,
        correction_token: int | None = None,
    ) -> CascadeObservation:
        if attempt is not self._open_attempt:
            raise PrefixCascadeError(
                "cascade observation does not match the open attempt"
            )
        if isinstance(matched_prefix_tokens, bool) or not isinstance(
            matched_prefix_tokens, Integral
        ):
            raise PrefixCascadeError("matched_prefix_tokens must be an integer")
        matched = int(matched_prefix_tokens)
        minimum = len(self.frontier)
        if not minimum <= matched <= len(attempt.path):
            raise PrefixCascadeError(
                "matched prefix must preserve the authoritative target frontier"
            )
        self._open_attempt = None
        if matched == len(attempt.path):
            if correction_token is not None:
                raise PrefixCascadeError("a fully matched path has no correction token")
            self.frontier = attempt.path
            self.terminal = True
            return CascadeObservation(True, matched, None, self.frontier, (), (), True)
        if (
            isinstance(correction_token, bool)
            or not isinstance(correction_token, Integral)
            or correction_token < 0
        ):
            raise PrefixCascadeError(
                "a partial path requires its nonnegative target correction token"
            )
        correction = int(correction_token)
        if correction == attempt.path[matched]:
            raise PrefixCascadeError(
                "correction token must reject the attempted proposal"
            )
        self.frontier = (*attempt.path[:matched], correction)
        before = [index for index in self._remaining if index not in self._attempted]
        survivors = self._survivors()
        pruned = tuple(index for index in before if index not in survivors)
        self.pruned += len(pruned)
        self._remaining = [
            index
            for index in self._remaining
            if index in self._attempted or index in survivors
        ]
        if not survivors:
            self.terminal = True
        return CascadeObservation(
            False,
            matched,
            correction,
            self.frontier,
            pruned,
            tuple(survivors),
            self.terminal,
        )

    def receipt(self):
        return {
            "schema": "mlx2.longest-first-prefix-cascade.v1",
            "algorithm": ALGORITHM,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "candidate_paths": len(self.paths),
            "attempts": self.attempts,
            "pruned_candidates": self.pruned,
            "shared_prefix_reuse": self.geometry is not None,
            "reused_prefix_tokens": self.reused_prefix_tokens,
            "frontier_tokens": len(self.frontier),
            "terminal": self.terminal,
            "geometry": None if self.geometry is None else self.geometry.receipt(),
        }


__all__ = [
    "ALGORITHM",
    "CascadeAttempt",
    "CascadeObservation",
    "LongestFirstPrefixCascade",
    "PrefixCascadeError",
    "PrefixReuseGeometry",
]
