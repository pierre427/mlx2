"""Experimental DFlash best-first tree search mined from TensorFold.

Source and license are recorded in
``provenance/tensorfold-qwen38-lever-probe.json``. The search is host-side;
target verification and state commit remain separate mlx2 mechanisms.
"""

from __future__ import annotations

import heapq

import numpy as np


def best_first_tree(
    candidates,
    unary,
    projected,
    anchor,
    predecessor_codebook,
    successor_codebook,
    *,
    edge=0.6,
    tau=1.5,
    children=4,
    max_nodes=15,
):
    """Return token and parent rows ordered by cumulative sibling log-law."""

    depth_count = int(candidates.shape[0])
    tables = [None] * depth_count

    def expand(token, depth):
        table = tables[depth]
        if table is None:
            table = tables[depth] = successor_codebook[
                candidates[depth]
            ].astype(np.float64)
        scores = unary[depth] + edge * (
            table @ (predecessor_codebook[int(token)] * projected[depth])
        )
        scores = scores / tau
        scores = scores - scores.max()
        return scores - np.log(np.exp(scores).sum())

    root = expand(int(anchor), 0)
    heap = []
    for index in np.argsort(-root)[: int(children)]:
        heapq.heappush(
            heap,
            (-float(root[index]), -1, int(candidates[0][index]), 0),
        )
    tokens, parents = [], []
    while heap and len(tokens) < int(max_nodes):
        negative, parent, token, depth = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        me = len(tokens) - 1
        if depth + 1 >= depth_count:
            continue
        scores = expand(token, depth + 1)
        for index in np.argsort(-scores)[: int(children)]:
            heapq.heappush(
                heap,
                (
                    negative - float(scores[index]),
                    me,
                    int(candidates[depth + 1][index]),
                    depth + 1,
                ),
            )
    return tokens, parents


def tree_paths(parents):
    """Return each parent-first node's root-to-node row path."""

    paths = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            path = [row]
        elif parent >= row:
            raise ValueError("tree parents must precede their children")
        else:
            path = paths[parent] + [row]
        paths.append(path)
    return paths


__all__ = ["best_first_tree", "tree_paths"]
