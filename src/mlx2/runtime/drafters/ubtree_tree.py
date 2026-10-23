"""Host-side UBTree construction from a precomputed candidate lattice.

The checkpoint only supplies neural scores.  Target verification and state
commit remain owned by mlx2's AST machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Node:
    token: int
    depth: int
    score: float
    parent: _Node | None
    order: int


def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    peak = np.max(values, axis=-1, keepdims=True)
    shifted = values - peak
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def build_ubtree(
    candidates,
    unary,
    projected,
    anchor,
    predecessor_codebook,
    successor_codebook,
    unary_log_scale,
    correction_log_scale,
    *,
    codebook_ids=None,
    frontier_width=12,
    max_nodes=64,
    depth_bonus=0.0,
):
    """Return parent-first ``(tokens, parents)`` under the UBTree paper law.

    Expansion retains a single width-``W`` frontier at each depth, then
    globally keeps ``B`` non-root nodes.  Ties prefer shallower ancestors and
    deterministic materialization order, keeping the result prefix-closed.
    """

    candidates = np.asarray(candidates)
    unary = np.asarray(unary, dtype=np.float64)
    projected = np.asarray(projected, dtype=np.float64)
    # Vocabulary tables stay in their cached dtype. Widen only selected
    # rows, rather than copying hundreds of MiB on every proposal.
    pred = np.asarray(predecessor_codebook)
    succ = np.asarray(successor_codebook)
    if codebook_ids is None:
        codebook_row = None
    else:
        codebook_ids = np.asarray(codebook_ids).reshape(-1)
        if pred.shape[0] != len(codebook_ids) or succ.shape[0] != len(codebook_ids):
            raise ValueError("UBTree compact codebooks do not match their token ids")
        codebook_row = {int(token): row for row, token in enumerate(codebook_ids)}

    def lookup(table, token):
        return table[int(token) if codebook_row is None else codebook_row[int(token)]]

    unary_scale = np.exp(np.asarray(unary_log_scale, dtype=np.float64))
    correction_scale = np.exp(np.asarray(correction_log_scale, dtype=np.float64))
    depth_count, candidate_count = candidates.shape
    if unary.shape != candidates.shape or projected.shape[0] != depth_count:
        raise ValueError("UBTree lattice shapes do not agree")
    if len(unary_scale) < depth_count or len(correction_scale) < depth_count:
        raise ValueError("UBTree depth calibration is too short")
    if not 1 <= int(frontier_width) or not 1 <= int(max_nodes):
        raise ValueError("UBTree frontier and node budgets must be positive")
    if float(depth_bonus) > 0:
        raise ValueError("UBTree depth bonus must be nonpositive")

    root = _Node(int(anchor), 0, 0.0, None, 0)
    frontier = [root]
    supertree = []
    order = 1
    # A B-node prefix-closed result cannot contain a path deeper than B or
    # a depth's node ranked below B. This prunes host work without changing
    # the bidirectional neural lattice or the selected global top-B tree.
    for depth in range(min(depth_count, int(max_nodes))):
        extensions = []
        successor_ids = candidates[depth].astype(np.int64)
        successor_rows = np.stack([lookup(succ, token) for token in successor_ids]).astype(np.float64)
        parent_rows = np.stack([lookup(pred, parent.token) for parent in frontier]).astype(np.float64)
        correction = (parent_rows * projected[depth]) @ successor_rows.T
        combined = unary_scale[depth] * unary[depth] + correction_scale[depth] * correction
        conditionals = _log_softmax(combined)
        for parent, conditional in zip(frontier, conditionals, strict=True):
            for column in range(candidate_count):
                extensions.append(
                    _Node(
                        int(successor_ids[column]),
                        depth + 1,
                        parent.score + float(conditional[column]) + float(depth_bonus),
                        parent,
                        order,
                    )
                )
                order += 1
        extensions.sort(key=lambda node: (-node.score, node.depth, node.order))
        frontier = extensions[: min(int(frontier_width), int(max_nodes))]
        supertree.extend(frontier)

    # Path probabilities cannot increase with depth.  Ancestor-first tie
    # ordering therefore preserves prefix closure under the paper's zeta<=0
    # contract.  Keep a defensive closure check before publishing parents.
    selected = sorted(
        supertree, key=lambda node: (-node.score, node.depth, node.order)
    )[: int(max_nodes)]
    chosen = set(selected)
    for node in selected:
        parent = node.parent
        while parent is not None and parent is not root:
            if parent not in chosen:
                raise RuntimeError(
                    "UBTree global pruning produced a non-prefix-closed tree"
                )
            parent = parent.parent

    # Monotone cumulative scores and ancestor-first ties already put parents
    # first. Preserve this ranking: AST's adaptive budget treats each prefix
    # as the corresponding smaller-budget tree.
    row = {node: index for index, node in enumerate(selected)}
    tokens = [node.token for node in selected]
    parents = [-1 if node.parent is root else row[node.parent] for node in selected]
    return tokens, parents


__all__ = ["build_ubtree"]
