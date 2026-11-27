"""Which post-0.32.5 upstream MLX changes the installed build carries.

Detected from the installed Metal kernel headers (``mlx/include/...``), not
from the version string, so a source build of MLX main is recognised by what
it contains.  Host-only: reading two small text files, no device work.

- ml-explore/mlx #4941 (merged 2026-11-27): ``sdpa_vector_2pass_1`` uses
  ``TK = 2`` when ``D + V > 512``.  That is the change mlx2's
  ``models/sdpa_vector_d512.py`` copies, so on such a build the copy is
  redundant (stock is the same arithmetic) and ``sdpa_vector_d512`` declines
  to launch it.  Delete the module at the upgrade.
- ml-explore/mlx #4802 (merged 2026-11-27): affine ``quantized_matmul`` on
  NAX GPUs quantizes ``x`` to int8 per ``group_size`` group when the product
  has at least 32 rows and enough output tiles, gated by the config key
  ``MLX_ENABLE_INT8_QMM`` whose default is **1**.  It changes the arithmetic
  of every affine-quantized prefill and of any batched forward with 32 or
  more rows in total (``M`` is ``x.size / K``), so B1 and B32 of one route
  would no longer run the same arithmetic.  ``UPGRADE_PROCESS_NUMERICS`` is
  the pin mlx2 needs at the upgrade (unused until then; see
  docs/research/2026-11-27-mlx-main-upgrade-prep.md).
- ml-explore/mlx #4953 (merged 2026-11-26) changes only which elementwise ops
  ``mx.compile`` fuses; no mlx2 code branches on it, so it is not detected.
- ml-explore/mlx #4966 (merged 2026-11-27) fixes exclusive scans; mlx2 never
  runs one.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

INT8_QMM_ENV = "MLX_ENABLE_INT8_QMM"
# The value mlx2 must pin before the first dispatch once MLX carries #4802:
# stock (input-precision) quantized_matmul until a per-artifact feature record
# qualifies MLX's int8 route.  Not applied anywhere yet: MLX 0.32.5 has no
# such key, and the merge into process_env.PROCESS_NUMERICS belongs to the
# upgrade commit (it enters every adapter profile and receipt).
UPGRADE_PROCESS_NUMERICS = {INT8_QMM_ENV: "0"}

_TK2_RULE = re.compile(r"int\s+TK\s*=\s*\(\s*D\s*\+\s*V\s*>\s*512\s*\?\s*2\s*:\s*4\s*\)")


def _kernel_dir(root: Path | None = None) -> Path | None:
    if root is None:
        try:
            import mlx.core as mx
        except ImportError:
            return None
        root = Path(mx.__file__).resolve().parent
    path = Path(root) / "include" / "mlx" / "backend" / "metal" / "kernels"
    return path if path.is_dir() else None


def _read(name: str, root: Path | None) -> str | None:
    directory = _kernel_dir(root)
    if directory is None:
        return None
    path = directory / name
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def sdpa_vector_d512_tk2(root: Path | None = None) -> bool | None:
    """True when the installed ``sdpa_vector.h`` carries #4941's TK rule.

    None when the header cannot be read (no MLX, or a wheel without headers).
    """
    text = _read("sdpa_vector.h", root)
    if text is None:
        return None
    return bool(_TK2_RULE.search(text))


def int8_qmm_available(root: Path | None = None) -> bool | None:
    """True when the installed MLX has #4802's int8 NAX quantized matmul."""
    text = _read("quantized_nax.h", root)
    if text is None:
        return None
    return "qmm_t_int_nax" in text


@lru_cache(maxsize=1)
def installed_sdpa_vector_d512_tk2() -> bool:
    """Cached form for the decode hot path (unknown counts as absent)."""
    return bool(sdpa_vector_d512_tk2())


def status(root: Path | None = None) -> dict:
    return {
        "sdpa_vector_d512_tk2_4941": sdpa_vector_d512_tk2(root),
        "int8_qmm_4802": int8_qmm_available(root),
        "int8_qmm_env": INT8_QMM_ENV,
        "upgrade_process_numerics": dict(UPGRADE_PROCESS_NUMERICS),
    }
