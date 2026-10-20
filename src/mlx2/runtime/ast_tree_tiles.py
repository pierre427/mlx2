"""Request-private AST tiles over exact shared proposal prefixes.

This module schedules host metadata only. Target execution, target laws and
state publication remain with their existing owners. Source acceptance priors
rank useful rows; they never replace a target decision or alter proposal tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math


def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


@dataclass(frozen=True)
class ASTProposalNode:
    token: int
    parent: int
    depth: int
    path_indices: tuple[int, ...]
    reach: float | None


@dataclass(frozen=True)
class ASTProposalTrie:
    nodes: tuple[ASTProposalNode, ...]
    children: tuple[tuple[int, ...], ...]
    paths: tuple[tuple[int, ...], ...]
    score_authority: str
    priority_mode: str

    @classmethod
    def from_paths(
        cls,
        anchor,
        paths,
        *,
        conditional_acceptance=None,
        lagged_acceptance=None,
        score_authority="unknown",
    ):
        """Deduplicate equal token prefixes, never equal suffixes.

        Conditional acceptance is supplied by the matching proposal slate.
        Unavailable entries use an explicitly supplied request-owned lagged
        acceptance; with neither available, scheduling is balanced and makes
        no probabilistic claim. Source ordinals are not probabilities.
        """
        paths = tuple(tuple(path) for path in paths)
        if (
            type(anchor) is not int
            or anchor < 0
            or not 1 <= len(paths) <= 4
            or any(
                len(path) > 15
                or any(type(token) is not int or token < 0 for token in path)
                for path in paths
            )
            or not isinstance(score_authority, str)
            or not score_authority
        ):
            raise ValueError("AST trie requires one to four bounded token paths")
        if lagged_acceptance is not None and not _probability(lagged_acceptance):
            raise ValueError("AST trie lagged acceptance must be a probability")
        confidence = (
            (None,) * len(paths)
            if conditional_acceptance is None
            else tuple(conditional_acceptance)
        )
        if len(confidence) != len(paths):
            raise ValueError("AST trie confidence must match the proposal slate")
        probabilities = []
        unknown = False
        for path, values in zip(paths, confidence):
            values = (None,) * len(path) if values is None else tuple(values)
            if len(values) != len(path) or any(
                value is not None and not _probability(value) for value in values
            ):
                raise ValueError("AST trie confidence must match exact path positions")
            row = tuple(
                lagged_acceptance if value is None else float(value) for value in values
            )
            unknown |= any(value is None for value in row)
            probabilities.append(row)

        tokens, parents, depths = [anchor], [-1], [0]
        members, reaches, children = [set()], [1.0], [[]]
        edges, node_paths = {}, []
        for rank, (path, probabilities_row) in enumerate(zip(paths, probabilities)):
            parent, reach = 0, 1.0
            indices = [0]
            members[0].add(rank)
            for depth, (token, probability) in enumerate(
                zip(path, probabilities_row), 1
            ):
                if not unknown:
                    reach *= probability
                key = (parent, token)
                node = edges.get(key)
                if node is None:
                    node = edges[key] = len(tokens)
                    tokens.append(token)
                    parents.append(parent)
                    depths.append(depth)
                    members.append(set())
                    reaches.append(None if unknown else reach)
                    children.append([])
                    children[parent].append(node)
                elif not unknown:
                    # A shared node is one opportunity, not one per duplicate
                    # path. Max source support preserves ancestor monotonicity.
                    reaches[node] = max(reaches[node], reach)
                members[node].add(rank)
                indices.append(node)
                parent = node
            node_paths.append(tuple(indices))
        return cls(
            tuple(
                ASTProposalNode(token, parent, depth, tuple(sorted(member)), reach)
                for token, parent, depth, member, reach in zip(
                    tokens, parents, depths, members, reaches
                )
            ),
            tuple(tuple(row) for row in children),
            tuple(node_paths),
            score_authority,
            "balanced_unknown_acceptance" if unknown else "conditional_reach_priority",
        )

    def matching_child(self, global_node, token):
        """Return only a child still reachable from this exact target prefix."""
        if type(global_node) is not int or not 0 <= global_node < len(self.nodes):
            raise ValueError("AST trie boundary is invalid")
        return next(
            (
                node
                for node in self.children[global_node]
                if self.nodes[node].token == token
            ),
            None,
        )

    @property
    def path_nodes(self):
        return self.paths

    def complete_path_indices(self, global_node):
        """Source paths ending here, distinct from paths passing through here."""
        if type(global_node) is not int or not 0 <= global_node < len(self.nodes):
            raise ValueError("AST trie boundary is invalid")
        return tuple(
            index for index, path in enumerate(self.paths) if path[-1] == global_node
        )


@dataclass(frozen=True)
class ASTTreeTile:
    root: int
    global_indices: tuple[int, ...]
    tokens: tuple[int, ...]
    parents: tuple[int, ...]
    path_depths: tuple[int, ...]
    shared_rows_saved: int
    reason: str
    score_authority: str

    def receipt(self):
        return {
            "schema": "mlx2.ast-tree-tile.v1",
            "physical_rows": len(self.tokens),
            "root_rows": 1,
            "proposal_rows": len(self.tokens) - 1,
            "path_depths": list(self.path_depths),
            "shared_rows_saved": self.shared_rows_saved,
            "reason": self.reason,
            "score_authority": self.score_authority,
        }


def plan_tree_tile(trie, *, root=0, max_rows=16):
    """Fill one parent-closed tile, including its single executable root.

    Monotone cumulative reach gives a best-first useful-row priority. Unknown
    confidence uses breadth then source order: four disjoint paths fill
    4/4/4/3 proposal positions, not an impossible root plus sixteen rows.
    This is fixed-budget packing, not a claim of measured end-to-end optimality.
    """
    if (
        not isinstance(trie, ASTProposalTrie)
        or type(root) is not int
        or not 0 <= root < len(trie.nodes)
        or type(max_rows) is not int
        or not 1 <= max_rows <= 16
    ):
        raise ValueError("AST tree tile requires a valid root and 1..16 total rows")
    selected = [root]
    frontier = []

    def add_children(parent):
        for child in trie.children[parent]:
            node = trie.nodes[child]
            priority = (
                0.0
                if trie.priority_mode == "balanced_unknown_acceptance"
                else -node.reach
            )
            heapq.heappush(
                frontier, (priority, node.depth, node.path_indices[0], child)
            )

    add_children(root)
    while frontier and len(selected) < max_rows:
        node = heapq.heappop(frontier)[-1]
        selected.append(node)
        add_children(node)
    local = {node: index for index, node in enumerate(selected)}
    path_depths = tuple(
        sum(node in local for node in path if node != root) if root in path else 0
        for path in trie.paths
    )
    appearances = sum(len(trie.nodes[node].path_indices) for node in selected)
    return ASTTreeTile(
        root,
        tuple(selected),
        tuple(trie.nodes[node].token for node in selected),
        tuple(
            -1 if node == root else local[trie.nodes[node].parent] for node in selected
        ),
        path_depths,
        appearances - len(selected),
        trie.priority_mode,
        trie.score_authority,
    )


@dataclass(frozen=True)
class ASTTreeAdvance:
    commit_local_indices: tuple[int, ...]
    commit_global_indices: tuple[int, ...]
    emitted: tuple[int, ...]
    next_root: int | None
    reason: str

    @property
    def finished(self):
        return self.next_root is None


def advance_tree_tile(trie, tile, next_tokens, *, stop_tokens=(), max_emitted=None):
    """Walk actual processed target choices, never the draft confidence.

    Commit exactly the visited input rows. A matching unexecuted child becomes
    the next tile's root; its already-decided token is not emitted twice.
    No root is reverified. A correction, stop, limit or leaf bonus closes the
    slate, so unrelated or lower-ranked pending alternatives cannot run later.
    """
    next_tokens = tuple(next_tokens)
    if (
        not isinstance(trie, ASTProposalTrie)
        or not isinstance(tile, ASTTreeTile)
        or len(next_tokens) != len(tile.tokens)
        or any(type(token) is not int or token < 0 for token in next_tokens)
        or (
            max_emitted is not None
            and (type(max_emitted) is not int or max_emitted < 1)
        )
    ):
        raise ValueError("AST tree advance requires one target choice per tile row")
    local = {node: index for index, node in enumerate(tile.global_indices)}
    if (
        not tile.global_indices
        or tile.global_indices[0] != tile.root
        or len(local) != len(tile.global_indices)
        or len(tile.parents) != len(tile.tokens)
        or any(
            type(node) is not int or not 0 <= node < len(trie.nodes)
            for node in tile.global_indices
        )
        or tile.tokens != tuple(trie.nodes[node].token for node in tile.global_indices)
        or tile.parents
        != tuple(
            -1 if index == 0 else local.get(trie.nodes[node].parent, -2)
            for index, node in enumerate(tile.global_indices)
        )
        or any(
            parent != -1 if index == 0 else not 0 <= parent < index
            for index, parent in enumerate(tile.parents)
        )
    ):
        raise ValueError("AST tile does not belong to its exact proposal trie")
    stops = frozenset(stop_tokens)
    current, commits, emitted = tile.root, [], []
    while True:
        index = local[current]
        commits.append(index)
        token = next_tokens[index]
        emitted.append(token)
        child = trie.matching_child(current, token)
        if token in stops:
            reason, pending = "stop", None
        elif max_emitted is not None and len(emitted) >= max_emitted:
            reason, pending = "limit", None
        elif child is None:
            reason = "leaf_bonus" if not trie.children[current] else "correction"
            pending = None
        elif child not in local:
            reason, pending = "reachable_continuation", child
        else:
            current = child
            continue
        return ASTTreeAdvance(
            tuple(commits),
            tuple(tile.global_indices[index] for index in commits),
            tuple(emitted),
            pending,
            reason,
        )
