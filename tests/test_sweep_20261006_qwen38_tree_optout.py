"""An explicit proposal_composition opt-out is compatible with the tree route.

The default tree15 route refused any ``proposal_composition`` key, so the
documented ``false`` opt-out could not start the pinned Qwen3.8 pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx2.adapters import qwen38_27b

MODELS = Path.home() / "mlx-models"
TARGET = MODELS / "Qwen3.8-27B-oQ4e-mtp"
DRAFT = MODELS / "Qwen3.8-27B-DFlash2"
POLICY = {
    "draft_model": str(DRAFT),
    "draft_revision": "34ec93d71399f3dd6db9646194f1ad4db345715c61258d72318325e0d8095c90",
    "target_revision": "e59471c5c6fa8c6819b81cb5957bcab10736020db8bacb47fbb9089813fa93f8",
    "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
}


@pytest.mark.skipif(not (TARGET.is_dir() and DRAFT.is_dir()), reason="pinned pair absent")
def test_tree_route_accepts_false_composition_opt_out():
    qwen38_27b.inspect_external_policy({**POLICY, "proposal_composition": False}, TARGET)
    with pytest.raises(ValueError, match="chain-only"):
        qwen38_27b.inspect_external_policy(
            {**POLICY, "proposal_composition": {"prompt_lookup": True}}, TARGET
        )
