"""The MLX builds mlx2's exact kernels were verified against.

Several kernels reproduce stock MLX arithmetic row for row, so their exactness
holds for a specific MLX build, not for "mlx>=0.32".  Indexed QSA
(``models/qwen4_qsa_indexed.py``) enforces this list at runtime and fails
closed above ``indexed_min_context`` on any other build.  These files cite
build 39400a0d4 as the stock arithmetic or dispatch they copy, and are only
re-verified by a requalification campaign:

  models/qwen4_hc_decode.py, models/qwen4_routed_decode.py,
  models/moe_nax_gather.py, models/sp_qmm.py, models/qwen4_attn_rows.py,
  models/qwen4_attn_window.py, models/qwen4_qsa_scores.py,
  models/invariant_prefill.py, models/qwen4_moe_weighted_sum.py,
  models/switch_layers.py, models/base.py, weight_residency.py

Upstream changes known to move these references: mlx #4596 (unrolled
``sdpa_vector_2pass_1``), #4516 (``qmv_fast_rows``, which changes the kernel
stock MLX selects for some shapes) and #4633 (residency refresh).  Before
admitting a new build here, rerun the header-digest test, the
``*_is_bit_exact_on_metal`` fixtures and the row-exact references, and check
the custom-kernel ABI (oMLX #4311: nanobind 2.15 builds reject ``mx.array``).

"Verified" here is a kernel-exactness statement about the installed MLX; it is
not route qualification (docs/QUALIFICATION.md).
"""

from __future__ import annotations

import importlib.metadata

VERIFIED_MLX_BUILDS = frozenset(
    {
        "0.32.2.dev20260829+334084ce9",
        "0.32.2.dev20260911+a0d69e543",
        "0.32.2.dev20260915+2a817ad94",
        "0.32.2.dev20260919+39400a0d4",
    }
)


def installed_mlx_build() -> str:
    try:
        return importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def status(build: str | None = None) -> dict:
    """Status-report form: the installed build and whether it is verified."""
    build = installed_mlx_build() if build is None else str(build)
    verified = build in VERIFIED_MLX_BUILDS
    return {
        "build": build,
        "kernels_verified": verified,
        **(
            {}
            if verified
            else {
                "warning": (
                    "MLX build not in VERIFIED_MLX_BUILDS: indexed QSA fails "
                    "closed above indexed_min_context and the copied-arithmetic "
                    "kernels are unverified; requalify before serving"
                )
            }
        ),
    }
