"""Bounded score-selection oracle; not imported by a serving route.

Scores arrive from an independently validated scorer in consecutive key tiles.
This isolates exact selection/merge from the unresolved Metal scorer law.
No 24-bit score truncation, approximate cutoff, or cache state is involved.
"""
from __future__ import annotations

import numpy as np


def stream_topk(tiles, *, rows, keys, topk, tile_keys, workspace_bytes):
    """Exact finite-score top-k, larger ID wins ties; output IDs chronological.

    Each callback tile is [rows, <=tile_keys] float32, in global key order.
    NaN/+inf are refused; -inf denotes an invisible key. Caller owns scoring
    arithmetic and masks. The bound includes candidate copies, sort indices,
    retained scores/IDs and selected output, rather than only the score tile.
    """
    if any(type(v) is not int or v < 1 for v in (rows, keys, topk, tile_keys, workspace_bytes)):
        raise ValueError("positive integer selection geometry required")
    if topk > keys:
        raise ValueError("topk exceeds keys")
    if tile_keys + topk >= keys:
        raise ValueError("streaming geometry must avoid full score width")
    # Conservative bound including lexsort scratch; no full [rows,keys] slab.
    bound = rows * (tile_keys + topk) * 64 + rows * topk * 16
    if bound > workspace_bytes:
        raise ValueError("selection workspace budget exceeded")
    best_scores = np.empty((rows, 0), np.float32)
    best_ids = np.empty((rows, 0), np.int64)
    offset = 0
    for tile in tiles:
        if (not isinstance(tile, np.ndarray) or tile.dtype != np.float32
                or tile.ndim != 2 or tile.shape[0] != rows
                or not 1 <= tile.shape[1] <= tile_keys
                or offset + tile.shape[1] > keys):
            raise ValueError("malformed score tile")
        if np.any(np.isnan(tile)) or np.any(np.isposinf(tile)):
            raise ValueError("nonfinite reachable score")
        ids = np.broadcast_to(np.arange(offset, offset + tile.shape[1]), tile.shape)
        scores = np.concatenate((best_scores, tile), axis=1)
        ids = np.concatenate((best_ids, ids), axis=1)
        rank = np.lexsort((ids, scores), axis=1)[:, -min(topk, scores.shape[1]):]
        best_scores = np.take_along_axis(scores, rank, axis=1)
        best_ids = np.take_along_axis(ids, rank, axis=1)
        offset += tile.shape[1]
    if offset != keys:
        raise ValueError("incomplete score stream")
    chronological = np.argsort(best_ids, axis=1)
    chosen = np.take_along_axis(best_ids, chronological, axis=1)
    valid = np.take_along_axis(best_scores, chronological, axis=1) != -np.inf
    return chosen, valid, {"workspace_bound_bytes": bound, "score_tiles_keys": tile_keys,
                           "full_score_matrix_materialized": False,
                           "scorer_parity_established": False}
