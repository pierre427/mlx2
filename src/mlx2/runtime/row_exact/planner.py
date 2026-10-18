# SPDX-License-Identifier: Apache-2.0
"""Row grouping for the row-exact projection kernel, from measured costs.

``row_exact_qmv`` gives every row its own accumulator and K order, so how
many rows share a threadgroup is purely a speed choice.  The best grouping
depends on the chip, the MLX build and the projection shape, so it is a table
measured by ``scripts/calibrate_row_exact_tiles.py`` and stored under
``tables/`` as ``<device>__<mlx build>.json``.  Without a table entry the
heuristic below applies (M5 Max, mlx 0.32.4, 2026-10-17 measurements):

* narrow projections (fewer than ``narrow_tiles`` output tiles) at 1-2 rows
  run one row per group: grouping only adds a weight-decode pass there;
* everything else uses the fewest groups under the register budget, with
  balanced group sizes (5 rows as 3+2, 7 as 4+3, 9 as 3+3+3).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

TABLE_DIR = Path(__file__).with_name("tables")


def table_name(device: str, build: str) -> str:
    safe = lambda text: "".join(c if c.isalnum() or c in "-." else "_" for c in text)
    return f"{safe(device)}__{safe(build)}.json"


def device_name() -> str:
    try:
        return str(mx.device_info().get("device_name", "unknown"))
    except Exception:  # noqa: BLE001 - off the GPU there is no table
        return "unknown"


@lru_cache(maxsize=4)
def _load(device: str, build: str) -> dict:
    path = TABLE_DIR / table_name(device, build)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if payload.get("device") != device or payload.get("mlx_build") != build:
        return {}
    return dict(payload.get("rows_per_group", {}))


def _table() -> dict:
    from ..mlx_build import installed_mlx_build

    return _load(device_name(), installed_mlx_build())


def table_key(bits: int, n: int, k: int, rows: int) -> str:
    return f"q{int(bits)}/n{int(n)}/k{int(k)}/r{int(rows)}"


def heuristic_rows_per_group(
    *, rows: int, limit: int, tiles: int, narrow_tiles: int
) -> int:
    if rows <= 1 or (tiles < narrow_tiles and rows <= 2):
        return 1
    limit = max(1, min(rows, limit))
    groups = -(-rows // limit)
    return -(-rows // groups)


def rows_per_group(
    *, bits: int, n: int, k: int, rows: int, limit: int, tiles: int, narrow_tiles: int
) -> tuple[int, str]:
    """``(rows per threadgroup, source)``; source is ``"table"`` or ``"heuristic"``."""
    measured = _table().get(table_key(bits, n, k, rows))
    if isinstance(measured, int) and 1 <= measured <= max(1, min(rows, limit)):
        return measured, "table"
    return (
        heuristic_rows_per_group(
            rows=rows, limit=limit, tiles=tiles, narrow_tiles=narrow_tiles
        ),
        "heuristic",
    )


def reset() -> None:
    _load.cache_clear()
