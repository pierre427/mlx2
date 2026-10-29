"""Synthetic selected-ID oracle, checkpoint not applicable."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("sparse_adoption",
    Path(__file__).parents[1] / "scripts/research/pr_adoption_sparse.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)


@pytest.mark.parametrize("tile", [1, 7, 32, 65])
@pytest.mark.parametrize("topk", [1, 8, 31])
def test_all_tiles_match_global_selection_with_ties_and_invisible_rows(tile, topk):
    values = np.random.default_rng(4).integers(0, 5, (3, 137)).astype(np.float32)
    values[0, 3:] = -np.inf
    values[1] = 0
    tiles = (values[:, i:i + tile] for i in range(0, 137, tile))
    got, valid, receipt = oracle.stream_topk(tiles, rows=3, keys=137, topk=topk,
        tile_keys=tile, workspace_bytes=1 << 20)
    ids = np.broadcast_to(np.arange(137), values.shape)
    chosen = np.lexsort((ids, values), axis=1)[:, -topk:]
    expected = np.sort(chosen, axis=1)
    assert np.array_equal(got, expected)
    assert np.array_equal(valid, np.take_along_axis(values, expected, axis=1) != -np.inf)
    assert receipt["full_score_matrix_materialized"] is False


def test_budget_incomplete_and_nonfinite_streams_refuse():
    args = {"rows": 1, "keys": 8, "topk": 2, "tile_keys": 4, "workspace_bytes": 4096}
    with pytest.raises(ValueError, match="workspace"):
        oracle.stream_topk([], **{**args, "workspace_bytes": 1})
    with pytest.raises(ValueError, match="incomplete"):
        oracle.stream_topk([np.zeros((1, 4), np.float32)], **args)
    with pytest.raises(ValueError, match="nonfinite"):
        oracle.stream_topk([np.full((1, 4), np.nan, np.float32)], **args)
    with pytest.raises(ValueError, match="full score width"):
        oracle.stream_topk([], **{**args, "tile_keys": 8})
