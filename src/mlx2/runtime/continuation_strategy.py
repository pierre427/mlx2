"""Tensor-runtime-independent complete-continuation strategy contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContinuationStrategy:
    """Adapter-declared execution law for complete continuation paths."""

    algorithm: str = "parallel_complete_paths_v1"
    prune_incompatible_siblings: bool = False
    shared_prefix_reuse: bool = False
    state_scope: str = "independent_request_private_branches"
    qualified: bool = False
    cache_layout: str | None = None
    proposal_state: str = "bound_source_only"
    target_state: str = "authoritative_exact"
    routed_experts_per_token: int | None = None
    routed_moe_layers: int | None = None

    def __post_init__(self):
        if self.algorithm not in {
            "parallel_complete_paths_v1",
            "longest_first_exact_prefix_v1",
        }:
            raise ValueError("unknown continuation verification strategy")
        for name in (
            "prune_incompatible_siblings",
            "shared_prefix_reuse",
            "qualified",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.state_scope, str) or not self.state_scope:
            raise ValueError("continuation strategy state_scope must be nonempty")
        if self.cache_layout is not None and (
            not isinstance(self.cache_layout, str) or not self.cache_layout
        ):
            raise ValueError("continuation strategy cache_layout must be nonempty")
        if self.proposal_state != "bound_source_only":
            raise ValueError("proposal state must remain a bound source only")
        if self.target_state != "authoritative_exact":
            raise ValueError("target state must remain authoritative and exact")
        routed = (self.routed_experts_per_token, self.routed_moe_layers)
        if (routed[0] is None) != (routed[1] is None):
            raise ValueError("routed-expert accounting geometry must be complete")
        if any(
            value is not None and (type(value) is not int or value < 1)
            for value in routed
        ):
            raise ValueError("routed-expert accounting geometry must be positive")
        if self.algorithm == "longest_first_exact_prefix_v1" and not (
            self.prune_incompatible_siblings and self.shared_prefix_reuse
        ):
            raise ValueError(
                "longest-first strategy requires pruning and exact shared-prefix reuse"
            )
    @classmethod
    def from_value(cls, value):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("continuation strategy must be a known policy object")
        return cls(**value)

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LongestFirstAttempt:
    path_index: int
    start: int
    suffix: tuple[int, ...]
    viable_indices: tuple[int, ...]

    @property
    def input_rows(self):
        return len(self.suffix) + 1


@dataclass(frozen=True)
class LongestFirstObservation:
    path_index: int
    accepted: int
    emitted: tuple[int, ...]
    pruned_indices: tuple[int, ...]
    surviving_indices: tuple[int, ...]
    terminal: bool


class LongestFirstExactPrefix:
    """Host controller for exact serial verification of ranked paths."""

    def __init__(self, paths, *, maximum, stop_tokens=()):
        self.paths = tuple(tuple(path) for path in paths)
        if (
            not self.paths
            or type(maximum) is not int
            or maximum < 1
            or any(
                not path or any(type(token) is not int or token < 0 for token in path)
                for path in self.paths
            )
        ):
            raise ValueError("invalid longest-first continuation paths")
        self.maximum = maximum
        self.stop_tokens = frozenset(int(token) for token in stop_tokens)
        self.order = tuple(
            index
            for index, _path in sorted(
                enumerate(self.paths), key=lambda item: (-len(item[1]), item[0])
            )
        )
        self.emitted = ()
        self.attempted = set()
        self.viable = tuple(self.order)
        self.observations = []
        self.terminal = False

    def next_attempt(self):
        if self.terminal:
            return None
        start = len(self.emitted)
        viable = tuple(
            index
            for index in self.order
            if index not in self.attempted
            and len(self.paths[index]) > start
            and self.paths[index][:start] == self.emitted
        )
        self.viable = viable
        if not viable:
            self.terminal = True
            return None
        selected = viable[0]
        self.attempted.add(selected)
        return LongestFirstAttempt(
            selected,
            start,
            self.paths[selected][start:],
            viable,
        )

    def observe(self, attempt, emitted):
        if self.terminal or not isinstance(attempt, LongestFirstAttempt):
            raise RuntimeError("no active longest-first attempt")
        if attempt.path_index not in self.attempted or attempt.start != len(self.emitted):
            raise ValueError("stale longest-first attempt")
        emitted = tuple(int(token) for token in emitted)
        if not emitted or len(emitted) > min(
            attempt.input_rows, self.maximum - len(self.emitted)
        ):
            raise ValueError("invalid target observation length")
        accepted = 0
        while (
            accepted < len(attempt.suffix)
            and accepted < len(emitted)
            and attempt.suffix[accepted] == emitted[accepted]
        ):
            accepted += 1
        stopped = emitted[-1] in self.stop_tokens
        budget = len(self.emitted) + len(emitted) >= self.maximum
        full = accepted == len(attempt.suffix)
        if not (stopped or budget or len(emitted) == accepted + 1):
            raise ValueError(
                "target observation must end at the first mismatch, stop, budget, or bonus"
            )
        self.emitted += emitted
        surviving = tuple(
            index
            for index in attempt.viable_indices
            if len(self.paths[index]) > len(self.emitted)
            and self.paths[index][: len(self.emitted)] == self.emitted
        )
        pruned = tuple(
            index
            for index in attempt.viable_indices
            if index != attempt.path_index and index not in surviving
        )
        terminal = stopped or budget or full or not surviving
        self.viable = surviving
        self.terminal = terminal
        observation = LongestFirstObservation(
            attempt.path_index,
            accepted,
            emitted,
            pruned,
            surviving,
            terminal,
        )
        self.observations.append(observation)
        return observation


__all__ = [
    "ContinuationStrategy",
    "LongestFirstAttempt",
    "LongestFirstExactPrefix",
    "LongestFirstObservation",
]
