"""One revision-bound recurrent checkpoint at a native KV token boundary.

Cache containers belong to one generation. Their tensor leaves may be shared
with an older immutable generation; the adapter must clone the containers
before updating any leaf. This module does not import a model or MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GDNBoundaryCheckpoint:
    revision: str
    lane_id: int
    offset: int
    generation: int
    caches: tuple[Any, ...]

    def __post_init__(self) -> None:
        if type(self.revision) is not str or not self.revision:
            raise ValueError("GDN checkpoint requires a revision")
        if type(self.lane_id) is not int or self.lane_id < 0:
            raise ValueError("GDN checkpoint requires a lane ID")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("GDN checkpoint requires a nonnegative offset")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("GDN checkpoint requires a nonnegative generation")
        if (type(self.caches) is not tuple or not self.caches or
                len({id(cache) for cache in self.caches}) != len(self.caches)):
            raise ValueError("GDN checkpoint requires distinct ordered caches")

    def private_successors(self, clone: object) -> tuple[Any, ...]:
        """Clone mutable containers while allowing immutable tensor leaves to share."""
        if not callable(clone):
            raise TypeError("GDN checkpoint needs an adapter clone callback")
        caches = clone(self.caches)
        if (type(caches) is not tuple or len(caches) != len(self.caches) or
                len({id(cache) for cache in caches}) != len(caches) or
                any(private is public for private, public in zip(caches, self.caches)) or
                {id(cache) for cache in caches} & {id(cache) for cache in self.caches}):
            raise ValueError("GDN successor containers must be private and ordered")
        return caches

    def successor(self, caches: tuple[Any, ...], *, offset: int) -> GDNBoundaryCheckpoint:
        if (type(offset) is not int or offset <= self.offset or
                type(caches) is not tuple or len(caches) != len(self.caches) or
                len({id(cache) for cache in caches}) != len(caches) or
                {id(cache) for cache in caches} & {id(cache) for cache in self.caches}):
            raise ValueError("GDN successor must replace a positive private token prefix")
        return GDNBoundaryCheckpoint(self.revision, self.lane_id, offset,
                                     self.generation + 1, caches)


__all__ = ["GDNBoundaryCheckpoint"]
