"""Turn any drafter's proposals into one bounded AST tree tile.

Original mlx2 implementation.  The tree tile route verifies an anchor row
plus up to ``tile_rows - 1`` proposal nodes in one target forward (the
TensorFold tile).  DFlash2 builds its own best-first lattice; every other
drafter proposes ranked complete paths (a native MTP head's distinct-root
branches, prompt lookup's n-gram continuations, a chain drafter's single
greedy path).  This module owns the shared part: the topology knob
(``"3x5"``, ``"4/4/4/3"``, ``"chain"``), scaling it to a lane's node budget,
and folding ranked paths into a parent-first tree with shared prefixes
verified once.  Nothing here touches target state; verification stays exact
in the scheduler.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

# Pierre's default (2026-10-21): one full tile, one anchor row, three
# branches five deep.
DEFAULT_TOPOLOGY = "3x5"
MAX_TILE_ROWS = 16


def parse_topology(value) -> tuple[int, ...] | str:
    """Return per-branch depths, or ``"chain"`` / ``"drafter"``.

    ``"3x5"`` is three branches of depth five; ``"4/4/4/3"`` lists depths;
    ``"chain"`` is one path using the whole budget; ``"drafter"`` keeps a
    drafter's own tree search (DFlash2's lattice).
    """

    if value is None:
        value = DEFAULT_TOPOLOGY
    if isinstance(value, (list, tuple)):
        depths = tuple(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in ("chain", "drafter"):
            return text
        if "x" in text:
            branches, _, depth = text.partition("x")
            try:
                depths = (int(depth),) * int(branches)
            except ValueError:
                raise ValueError(f"invalid AST topology {value!r}") from None
        else:
            try:
                depths = tuple(int(part) for part in text.split("/"))
            except ValueError:
                raise ValueError(f"invalid AST topology {value!r}") from None
    else:
        raise ValueError(f"invalid AST topology {value!r}")
    if not depths or any(type(d) is not int or d < 1 for d in depths):
        raise ValueError(f"invalid AST topology {value!r}")
    if sum(depths) > MAX_TILE_ROWS - 1:
        raise ValueError(
            f"AST topology {value!r} needs {sum(depths)} proposal rows; "
            f"a tile holds {MAX_TILE_ROWS - 1} plus the anchor"
        )
    return depths


def pattern_for_budget(topology, max_nodes: int) -> tuple[int, ...]:
    """Fit a topology into ``max_nodes`` proposal rows.

    A smaller budget (a lane sharing its tile with peers) keeps the branch
    count while it can afford depth two, then trades branches for depth;
    a chain uses the whole budget.  The result never exceeds the budget.
    """

    max_nodes = int(max_nodes)
    if max_nodes < 1:
        return ()
    parsed = parse_topology(topology)
    if parsed in ("chain", "drafter"):
        return (max_nodes,)
    depths = list(parsed)
    if sum(depths) <= max_nodes:
        return tuple(depths)
    branches = len(depths)
    while branches > 1 and max_nodes // branches < 2:
        branches -= 1
    depth, extra = divmod(max_nodes, branches)
    return tuple(depth + (1 if index < extra else 0) for index in range(branches))


def paths_to_tree(
    paths: Iterable[Sequence[int]],
    max_nodes: int,
    *,
    forbidden_root: Iterable[int] = (),
) -> tuple[list[int], list[int]]:
    """Fold ranked paths into ``(tokens, parents)`` with shared prefixes once.

    Parents index ``tokens`` with ``-1`` for children of the anchor, and
    every parent precedes its children.  Paths are taken in rank order and
    truncated when the node budget runs out, so the best path is always
    whole.  A root token the next position forbids drops its whole path.
    """

    forbidden = {int(token) for token in forbidden_root}
    tokens: list[int] = []
    parents: list[int] = []
    children: dict[tuple[int, int], int] = {}
    for path in paths:
        parent = -1
        for depth, token in enumerate(int(token) for token in path):
            if depth == 0 and token in forbidden:
                break
            key = (parent, token)
            node = children.get(key)
            if node is None:
                if len(tokens) >= max_nodes:
                    break
                node = len(tokens)
                tokens.append(token)
                parents.append(parent)
                children[key] = node
            parent = node
        if len(tokens) >= max_nodes:
            break
    return tokens, parents


def tree_receipt(tokens: Sequence[int], parents: Sequence[int]) -> dict:
    """Shape summary for receipts: node count, root branches, depth."""

    depth = [0] * len(parents)
    for index, parent in enumerate(parents):
        depth[index] = 1 if parent < 0 else depth[parent] + 1
    return {
        "nodes": len(tokens),
        "root_branches": sum(1 for parent in parents if parent < 0),
        "max_depth": max(depth, default=0),
    }


__all__ = [
    "DEFAULT_TOPOLOGY",
    "MAX_TILE_ROWS",
    "parse_topology",
    "paths_to_tree",
    "pattern_for_budget",
    "tree_receipt",
]
