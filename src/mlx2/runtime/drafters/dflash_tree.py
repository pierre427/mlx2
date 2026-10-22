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


def calibrated_best_first_tree(
    candidates,
    unary,
    projected,
    anchor,
    predecessor_codebook,
    successor_codebook,
    *,
    calibration,
    edge=0.6,
    children=None,
    max_nodes=15,
):
    """Best-first tree ordered by calibrated path acceptance probability.

    The drafter's own law at each node, ``q = softmax(unary + edge * pair)``
    over the depth's candidates (no temperature), is mapped through the
    per-depth calibration ``p = sigmoid(alpha_d + beta_d * logit(q))``
    (``runtime/ast_tree_calibration.py``; Kapitsa ``kf_lcal``).  A node's
    priority is the sum of ``log p`` along its root path, i.e. the
    calibrated probability that the target accepts the whole path, so the
    first ``n`` nodes are the ``n``-node tree with the most expected tokens
    and ``1 + sum(exp(path_log_p[:n]))`` is that expectation.  ``children``
    (default: every candidate) bounds the fan-out per node.

    Returns ``tokens, parents, info`` with ``info`` holding per node its
    depth (1 = a child of the anchor), ``log_q`` and ``path_log_p``.
    """

    from ..ast_tree_calibration import calibrated_log_p

    alpha, beta = calibration
    last = len(alpha) - 1
    depth_count = int(candidates.shape[0])
    width = int(candidates.shape[1])
    fan = width if children is None else max(1, min(int(children), width))
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
        scores = scores - scores.max()
        return scores - np.log(np.exp(scores).sum())

    def push(heap, log_q_row, base, parent, depth):
        level = min(depth + 1, last)
        a, b = alpha[level], beta[level]
        for index in np.argsort(-log_q_row, kind="stable")[:fan]:
            log_q = float(log_q_row[index])
            heapq.heappush(
                heap,
                (
                    -(base + calibrated_log_p(log_q, a, b)),
                    parent,
                    int(candidates[depth][index]),
                    depth,
                    log_q,
                ),
            )

    heap = []
    push(heap, expand(int(anchor), 0), 0.0, -1, 0)
    tokens, parents, depths, log_qs, path = [], [], [], [], []
    while heap and len(tokens) < int(max_nodes):
        negative, parent, token, depth, log_q = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        depths.append(depth + 1)
        log_qs.append(log_q)
        path.append(-negative)
        me = len(tokens) - 1
        if depth + 1 >= depth_count:
            continue
        push(heap, expand(token, depth + 1), -negative, me, depth + 1)
    return tokens, parents, {"depths": depths, "log_q": log_qs, "path_log_p": path}


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


__all__ = ["best_first_tree", "calibrated_best_first_tree", "tree_paths"]
