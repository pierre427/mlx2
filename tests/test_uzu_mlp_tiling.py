"""Focused CPU contracts for the Uzu-inspired producer and QMV geometry."""

import mlx.core as mx
import pytest

from mlx2.runtime.models.row_exact_qmv import TileGeometry
from scripts.research.uzu import fused_mlp_stage_a as fused


@pytest.mark.parametrize("kwargs", [
    {"rows_per_group": True}, {"rows_per_group": 0}, {"rows_per_group": 9},
    {"rows_per_group": 2, "results_per_simdgroup": 3},
    {"rows_per_group": 2, "results_per_simdgroup": True},
])
def test_bad_tile_geometry(kwargs):
    with pytest.raises(ValueError):
        TileGeometry(**kwargs)


def test_tile_register_budget():
    TileGeometry(2, 8).validate(bits=4, fast=True, rows=3)
    with pytest.raises(ValueError):
        TileGeometry(3, 8).validate(bits=4, fast=True, rows=3)
    with pytest.raises(ValueError):
        TileGeometry(2).validate(bits=8, fast=True, rows=1)


def test_fused_producer_rejects_unsupported_inputs_without_gpu_work():
    gate = mx.zeros((2, 64), dtype=mx.bfloat16)
    for up, layout in ((gate, "unknown"), (gate.astype(mx.float32), "q8"),
                       (mx.zeros((2, 128), dtype=mx.bfloat16), "q45")):
        with pytest.raises(ValueError):
            fused.prepare(gate, up, layout=layout)
    with pytest.raises(ValueError, match="requires Metal"):
        fused.prepare(gate, gate, layout="q8")



def test_production_policy_refuses_archived_fusion():
    from mlx2.runtime.int8_prefill import Int8PrefillPolicy
    with pytest.raises(ValueError, match="unknown"):
        Int8PrefillPolicy.from_value({"enabled": True, "fused_mlp_stage_a": True})
