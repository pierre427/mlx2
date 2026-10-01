"""Sizing a mixed prefill-and-decode forward.

A mixed forward packs a prompt slice and the decode lanes' next tokens into
one token stream, so every projection reads its weights once for both.  The
quantized matmuls tile rows in groups of 64: on the served Qwen3.6-27B-8bit
(M5 Max, 2026-09-30) adding one decode row to a 128-row slice cost a third
tile (1.51x the slice), while a 127-row slice plus one decode row cost what
the 128-row slice alone costs.  The prompt's share of a mixed forward is
therefore sized so prompt rows plus decode rows fill whole tiles.
"""

from __future__ import annotations

MATMUL_ROW_TILE = 64


def aligned_prompt_rows(budget_rows: int, decode_rows: int, *, tile: int = MATMUL_ROW_TILE) -> int:
    """Prompt rows for a mixed forward of about ``budget_rows`` packed rows.

    The largest ``r`` with ``r + decode_rows`` a multiple of ``tile`` and at
    most ``budget_rows``.  A budget below what the decode rows need rounds up
    to the tiles they occupy, since a partial tile costs a whole one: the
    prompt then gets the rows left in those tiles (0 when the decode rows fill
    them exactly, making the round decode-only).
    """
    if isinstance(budget_rows, bool) or not isinstance(budget_rows, int) or budget_rows < 1:
        raise ValueError("budget_rows must be a positive integer")
    if isinstance(decode_rows, bool) or not isinstance(decode_rows, int) or decode_rows < 0:
        raise ValueError("decode_rows must be a nonnegative integer")
    if isinstance(tile, bool) or not isinstance(tile, int) or tile < 1:
        raise ValueError("tile must be a positive integer")
    packed = (budget_rows // tile) * tile
    if packed <= decode_rows:
        packed = -(-max(decode_rows, 1) // tile) * tile
    return max(0, packed - decode_rows)


__all__ = ["MATMUL_ROW_TILE", "aligned_prompt_rows"]
