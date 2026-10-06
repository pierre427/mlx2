"""Import-free planning for exact-prefix continuation cascades.

The planner ranks complete proposal paths and returns only the suffix that has
not already been committed.  It never declares model state reusable: adapters
own the cache/state geometry gate that makes suffix-only execution exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CascadeStage:
    candidate_index: int
    path: tuple[int, ...]
    accepted_prefix: tuple[int, ...]
    suffix: tuple[int, ...]
    viable_indices: tuple[int, ...]
    pruned_indices: tuple[int, ...]

    @property
    def reused_prefix_tokens(self) -> int:
        return len(self.accepted_prefix)


def longest_first_paths(paths: Iterable[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return stable longest-first, duplicate-free complete token paths."""

    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in paths:
        path = tuple(raw)
        if not path or any(type(token) is not int or token < 0 for token in path):
            raise ValueError("cascade paths must contain positive-length token tuples")
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise ValueError("cascade requires at least one complete path")
    ordered = sorted(enumerate(unique), key=lambda item: (-len(item[1]), item[0]))
    return tuple(path for _, path in ordered)


def next_cascade_stage(
    paths: Iterable[Sequence[int]],
    accepted_prefix: Sequence[int] = (),
    *,
    attempted: Iterable[int] = (),
) -> CascadeStage | None:
    """Choose the longest viable suffix and prune impossible siblings.

    Indices are positions in :func:`longest_first_paths`.  A sibling is viable
    only when it contains the entire accepted prefix and still has an
    unverified token.  Returning ``suffix`` does not authorize state reuse;
    the owning adapter must first prove an exact checkpoint at that boundary.
    Accepted tokens prune semantic paths before another model launch; rows
    already submitted to a device are not retroactively reclaimed.
    """

    ranked = longest_first_paths(paths)
    prefix = tuple(accepted_prefix)
    if any(type(token) is not int or token < 0 for token in prefix):
        raise ValueError("accepted prefix must contain token ids")
    attempted_set = set(attempted)
    if any(
        type(index) is not int or not 0 <= index < len(ranked)
        for index in attempted_set
    ):
        raise ValueError("attempted cascade index is out of range")
    viable = tuple(
        index
        for index, path in enumerate(ranked)
        if index not in attempted_set
        and len(path) > len(prefix)
        and path[: len(prefix)] == prefix
    )
    pruned = tuple(
        index
        for index, _path in enumerate(ranked)
        if index not in attempted_set and index not in viable
    )
    if not viable:
        return None
    selected = viable[0]
    path = ranked[selected]
    return CascadeStage(
        candidate_index=selected,
        path=path,
        accepted_prefix=prefix,
        suffix=path[len(prefix) :],
        viable_indices=viable,
        pruned_indices=pruned,
    )


__all__ = ["CascadeStage", "longest_first_paths", "next_cascade_stage"]
