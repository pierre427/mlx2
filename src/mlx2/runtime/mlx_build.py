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
stock MLX selects for some shapes) and #4633 (residency refresh).  #4780
(``expm1f`` argument reduction) moves fp32 ``erf``/``expm1``/exact-GELU
outputs in the last bits (not bf16/fp16), so rerun the F32 Nemotron3
diarization receipt; #4568 (custom-kernel source cache) renames templated
kernels and must stay bit-identical (it was on Flash-Next decode,
docs/experiments/MLX-PR4568-PR4780-INTAKE-2026-10-31.md).  Before
admitting a new build here, rerun the header-digest test, the
``*_is_bit_exact_on_metal`` fixtures and the row-exact references, and check
the custom-kernel ABI (oMLX #4311: nanobind 2.15 builds reject ``mx.array``).

Upstream 0.32.4 was admitted on 2026-10-15 after its changed
``sdpa_vector.h`` was reviewed, indexed QSA passed its focused numeric
fixtures, and the copied-arithmetic GDN, routed-MoE and hyperconnection kernels
were made source-spelling aware and passed their Metal exactness gates.  The
release adds thin-M/thin-N GEMM and wider qmv routes, changes split-K partials
to fp32, and fixes a batched qvm leading-dimension corruption.

``models/qmm_row_pad.py`` models 0.32.4's quantized-matmul dispatch and is
admitted on 0.32.4 only; a new build must re-measure it (or drop it if
mlx #4793 / #4720 has landed) before its admission list grows.

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
        "0.32.4",
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
