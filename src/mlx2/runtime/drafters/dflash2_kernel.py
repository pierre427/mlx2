# SPDX-License-Identifier: MIT
"""DFlash2 drafter projection-kernel law (original mlx2).

Kept apart from ``dflash2`` so adapters can bind the law into a drafter's
identity while inspecting a policy, before the target adapter pins its import
environment: this module imports only the lane package, never a model module.
"""

DRAFT_KERNELS = ("stock", "lane")
# One launch serves up to this many rows; longer calls (a long first context
# append) take the stock kernel.  Part of the drafter's law.
DRAFT_LANE_MAX_ROWS = 128
# Lattice-side changes that ride with the lane kernel: context and proposal
# rows fused per projection, and the block-maxima vocabulary top-k.
DRAFT_LANE_SUFFIX = "+fused-context-v1+blocktopk-v1"


def draft_kernel_law(kernel):
    """The drafter projection law ``kernel`` installs on this device, or None.

    Static (no weights): adapters bind it into the drafter's identity before
    loading, and ``configure_draft_kernel`` refuses a model whose installed
    law differs.  The lane law is per backend (M5 tensor units vs M1-M4).
    """
    if kernel is None or kernel == "stock":
        return None
    if kernel not in DRAFT_KERNELS:
        raise ValueError(f"draft_kernel must be one of {DRAFT_KERNELS}")
    from .. import lane

    name = lane.backend()
    if name is None:
        raise ValueError("draft_kernel 'lane' needs a lane matmul backend on this device")
    return f"{lane.LAW_IDS[name]}+rows-le-{DRAFT_LANE_MAX_ROWS}+grouped{DRAFT_LANE_SUFFIX}"


__all__ = ["DRAFT_KERNELS", "DRAFT_LANE_MAX_ROWS", "DRAFT_LANE_SUFFIX", "draft_kernel_law"]
