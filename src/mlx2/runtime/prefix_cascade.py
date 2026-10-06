"""Host planning for exact-prefix continuation cascades.

The planner owns no model tensors and publishes no cache state.  It orders
complete proposal paths longest-first, advances only from target tokens that
were actually drawn, and returns only the not-yet-executed suffix of a viable
path.  A model adapter still has to attest that its target forward and cache
commit preserve ordinary-decode state before a serving route may execute the
plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixCascadeStage:
    """One conditional target launch from an authoritative token frontier."""

    path_index: int
    path: tuple[int, ...]
    suffix: tuple[int, ...]
    reused_prefix_tokens: int
    pruned_before_launch: int


@dataclass(frozen=True)
class PrefixCascadeUpdate:
    """Planner state after one target result has extended the frontier."""

    frontier: tuple[int, ...]
    pruned_siblings: int
    remaining_paths: int
    finished: bool


def _paths(value, *, maximum_paths: int, maximum_depth: int):
    paths = tuple(tuple(path) for path in value)
    if (
        not paths
        or not 1 <= len(paths) <= maximum_paths <= 15
        or not 1 <= maximum_depth <= 15
        or any(
            not path
            or len(path) > maximum_depth
            or any(type(token) is not int or token < 0 for token in path)
            for path in paths
        )
    ):
        raise ValueError("invalid exact-prefix cascade paths")
    return paths


class ExactPrefixCascade:
    """Plan longest-first launches without speculating past target authority.

    ``observe`` accepts only a target result whose accepted prefix agrees with
    the staged proposal suffix.  The first mismatching target draw becomes part
    of the frontier, immediately making siblings with another prefix
    impossible.  ``next_stage`` strips the full frontier from the next viable
    path, so common proposal tokens are never requested a second time.
    """

    def __init__(self, paths, *, maximum_paths: int = 15, maximum_depth: int = 15):
        self.paths = _paths(
            paths, maximum_paths=maximum_paths, maximum_depth=maximum_depth
        )
        self.order = tuple(
            index
            for index, _path in sorted(
                enumerate(self.paths), key=lambda item: (-len(item[1]), item[0])
            )
        )
        self.frontier: tuple[int, ...] = ()
        self.attempted: set[int] = set()
        self.pruned: set[int] = set()
        self._stage: PrefixCascadeStage | None = None
        self._finished = False
        self.pruned_siblings = 0

    def _viable(self):
        size = len(self.frontier)
        return tuple(
            index
            for index in self.order
            if index not in self.attempted
            and index not in self.pruned
            and len(self.paths[index]) > size
            and self.paths[index][:size] == self.frontier
        )

    def _prune(self) -> int:
        size = len(self.frontier)
        impossible = {
            index
            for index in self.order
            if index not in self.attempted
            and index not in self.pruned
            and (
                len(self.paths[index]) <= size
                or self.paths[index][:size] != self.frontier
            )
        }
        self.pruned.update(impossible)
        self.pruned_siblings += len(impossible)
        return len(impossible)

    def next_stage(self) -> PrefixCascadeStage | None:
        """Return the next exact suffix, or ``None`` when the round is done."""
        if self._stage is not None:
            raise RuntimeError("the current prefix-cascade stage is unresolved")
        if self._finished:
            return None
        impossible = self._prune()
        viable = self._viable()
        if not viable:
            self._finished = True
            return None
        index = viable[0]
        self.attempted.add(index)
        path = self.paths[index]
        self._stage = PrefixCascadeStage(
            path_index=index,
            path=path,
            suffix=path[len(self.frontier) :],
            reused_prefix_tokens=len(self.frontier),
            pruned_before_launch=impossible,
        )
        return self._stage

    def observe(
        self,
        stage: PrefixCascadeStage,
        emitted,
        *,
        accepted: int,
        terminal: bool = False,
    ) -> PrefixCascadeUpdate:
        """Extend the authoritative frontier after one exact target launch."""
        if self._stage is None or stage is not self._stage:
            raise ValueError("prefix-cascade observation does not own the open stage")
        emitted = tuple(emitted)
        if (
            not emitted
            or any(type(token) is not int or token < 0 for token in emitted)
            or type(accepted) is not int
            or not 0 <= accepted <= len(stage.suffix)
            or accepted > len(emitted)
            or emitted[:accepted] != stage.suffix[:accepted]
            or (accepted < len(stage.suffix) and len(emitted) != accepted + 1)
        ):
            raise ValueError("invalid exact-prefix cascade target observation")
        previous = self.frontier
        self.frontier = (*previous, *emitted)
        impossible = self._prune()
        self._stage = None
        full = accepted == len(stage.suffix)
        self._finished = bool(terminal or full or not self._viable())
        return PrefixCascadeUpdate(
            frontier=self.frontier,
            pruned_siblings=impossible,
            remaining_paths=len(self._viable()),
            finished=self._finished,
        )

    @property
    def finished(self) -> bool:
        return self._finished


def exact_prefix_geometry(
    cache,
    *,
    expected_layers: int,
    expected_rotating_layers: int,
    sliding_window: int,
):
    """Describe conservative North-style cache geometry without attesting math.

    This proves only that layers share one logical offset and that rotating
    layers use the declared window.  It deliberately does not claim that a
    batched/multirow target forward matches ordinary B=1 attention arithmetic.
    """
    cache = tuple(cache)
    if len(cache) != expected_layers:
        return {"compatible": False, "reason": "layer_count"}
    offsets = tuple(getattr(layer, "offset", None) for layer in cache)
    if any(type(offset) is not int or offset < 0 for offset in offsets):
        return {"compatible": False, "reason": "host_offsets"}
    if len(set(offsets)) != 1:
        return {"compatible": False, "reason": "layer_offset_mismatch"}
    rotating = tuple(
        layer for layer in cache if "rotating" in type(layer).__name__.lower()
    )
    if len(rotating) != expected_rotating_layers:
        return {"compatible": False, "reason": "rotating_layer_count"}
    if any(
        int(getattr(layer, "max_size", 0) or 0) != sliding_window for layer in rotating
    ):
        return {"compatible": False, "reason": "sliding_window"}
    return {
        "compatible": True,
        "offset": offsets[0],
        "layers": len(cache),
        "rotating_layers": len(rotating),
        "sliding_window": sliding_window,
        "attention_math_attested": False,
    }


__all__ = [
    "ExactPrefixCascade",
    "PrefixCascadeStage",
    "PrefixCascadeUpdate",
    "exact_prefix_geometry",
]
