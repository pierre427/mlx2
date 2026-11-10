# SPDX-License-Identifier: MIT
"""DFlash2 drafter projection-kernel law (original mlx2).

Kept apart from ``dflash2`` so adapters can bind the law into a drafter's
identity while inspecting a policy, before the target adapter pins its import
environment: this module imports only the lane package, never a model module.
"""

DRAFT_KERNELS = ("stock", "lane", "fused")
# One launch serves up to this many rows; longer calls (a long first context
# append) take the stock kernel.  Part of the drafter's law.
DRAFT_LANE_MAX_ROWS = 128
# Lattice-side changes that ride with the lane kernel: context and proposal
# rows fused per projection, and the block-maxima vocabulary top-k.
DRAFT_LANE_SUFFIX = "+fused-context-v1+blocktopk-v1"
DRAFT_FUSED_SUFFIX = "+fused-layer-v2"
# DSpine (original mlx2): the backbone takes the same lane law and fused
# context projections; its code bridge stays on stock kernels (its weights
# are read directly, not through these modules), and it has no block top-k.
DSPINE_LANE_SKIPPED = (
    "selected_prefix_bridge.gate",
    "selected_prefix_bridge.out",
    "selected_prefix_bridge.predecessor_gate",
    "selected_prefix_bridge.prev",
)
DSPINE_LANE_SUFFIX = "+fused-context-v1"
DRAFTERS = ("dflash2", "dspine")


def _skip_digest(names):
    """The lane installer's ``+skip[...]`` law part for these module names."""
    import hashlib

    joined = "\n".join(sorted(names))
    return f"+skip[{hashlib.sha256(joined.encode()).hexdigest()[:12]}]"


def draft_kernel_law(kernel, *, drafter="dflash2"):
    """The drafter projection law ``kernel`` installs on this device, or None.

    Static (no weights): adapters bind it into the drafter's identity before
    loading, and ``configure_draft_kernel`` refuses a model whose installed
    law differs.  The lane law is per backend (M5 tensor units vs M1-M4).
    """
    if kernel is None or kernel == "stock":
        return None
    if kernel not in DRAFT_KERNELS:
        raise ValueError(f"draft_kernel must be one of {DRAFT_KERNELS}")
    if drafter not in DRAFTERS:
        raise ValueError(f"draft kernel drafter must be one of {DRAFTERS}")
    if drafter == "dspine" and kernel != "lane":
        raise ValueError('DSpine supports draft_kernel "stock" or "lane"')
    from .. import lane

    name = lane.backend()
    if name is None:
        raise ValueError(f"draft_kernel {kernel!r} needs a lane matmul backend on this device")
    if drafter == "dspine":
        return (f"{lane.LAW_IDS[name]}+rows-le-{DRAFT_LANE_MAX_ROWS}"
                f"{_skip_digest(DSPINE_LANE_SKIPPED)}+grouped{DSPINE_LANE_SUFFIX}")
    law = f"{lane.LAW_IDS[name]}+rows-le-{DRAFT_LANE_MAX_ROWS}+grouped{DRAFT_LANE_SUFFIX}"
    return law + (DRAFT_FUSED_SUFFIX if kernel == "fused" else "")


__all__ = [
    "DRAFT_KERNELS", "DRAFT_LANE_MAX_ROWS", "DRAFT_LANE_SUFFIX", "DRAFT_FUSED_SUFFIX",
    "DSPINE_LANE_SKIPPED", "DSPINE_LANE_SUFFIX", "draft_kernel_law",
]
