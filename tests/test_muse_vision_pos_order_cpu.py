"""Muse vision position-table order on non-square patch grids.

The trained reference resamples the learned position table with
``F.grid_sample(table, meshgrid(pos_h, pos_w, indexing="xy"),
align_corners=False, padding_mode="zeros")``. grid_sample reads the first
grid coordinate as width, so an (h, w) patch grid samples the table as a
(w, h) grid (mlx-vlm #2746, transformers #50847). These tests pin that order
against an independent grid_sample oracle and against the fixed Transformers
helper call, and check that square grids keep the old values.
"""
import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.muse_vision_config import VisionConfig
from mlx2.runtime.models.muse_vision import PatchEmbedder

torch = pytest.importorskip("torch")
F = torch.nn.functional

SIDE = 6
HIDDEN = 8
GRIDS = [[1, 4, 6], [2, 3, 8], [1, 9, 5], [1, 5, 5]]


@pytest.fixture(autouse=True)
def cpu():
    mx.set_default_device(mx.cpu)


def _embedder(seed=11):
    config = VisionConfig(hidden_size=HIDDEN, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, patch_size=2, patch_temporal=2, pos_emb_height=SIDE,
        pos_emb_width=SIDE, merge_size=1, layer_types=['full_attention'])
    embedder = PatchEmbedder(config)
    table = np.random.default_rng(seed).normal(size=(SIDE * SIDE, HIDDEN)).astype(np.float32)
    embedder.position_embedding_table.weight = mx.array(table)
    return embedder, table


def _grid_sample_reference(table, grid):
    """Independent oracle: the reference's grid_sample over an xy meshgrid."""
    image = torch.from_numpy(table).T.reshape(1, HIDDEN, SIDE, SIDE)
    parts = []
    for temporal, height, width in grid:
        pos_h = (torch.arange(height, dtype=torch.float32) + 0.5) / height * 2 - 1
        pos_w = (torch.arange(width, dtype=torch.float32) + 0.5) / width * 2 - 1
        x, y = torch.meshgrid(pos_h, pos_w, indexing="xy")
        sampled = F.grid_sample(image, torch.stack((x, y), dim=-1)[None], mode="bilinear",
            padding_mode="zeros", align_corners=False)
        parts.append(sampled.flatten(2)[0].T.repeat(temporal, 1))
    return torch.cat(parts).numpy()


def test_non_square_grids_match_reference_grid_sample_order():
    embedder, table = _embedder()
    actual = np.asarray(embedder._position_embeddings(GRIDS))
    np.testing.assert_allclose(actual, _grid_sample_reference(table, GRIDS), atol=2e-6, rtol=2e-6)


def test_matches_fixed_transformers_helper_call():
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    embedder, table = _embedder(seed=12)
    grid = torch.tensor(GRIDS)
    indices, weights = get_vision_interpolation_indices_and_weights(
        grid[:, [0, 2, 1]], num_grid_per_side=SIDE, mode="bilinear",
        align_corners=False, spatial_merge_size=1, padding="zeros")
    expected = (torch.from_numpy(table)[indices] * weights[:, :, None]).sum(1).numpy()
    actual = np.asarray(embedder._position_embeddings(GRIDS))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_old_hw_order_differs_on_non_square_and_agrees_on_square():
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    embedder, table = _embedder(seed=13)

    def old_order(grid):
        indices, weights = get_vision_interpolation_indices_and_weights(
            torch.tensor(grid), num_grid_per_side=SIDE, mode="bilinear",
            align_corners=False, spatial_merge_size=1, padding="zeros")
        return (torch.from_numpy(table)[indices] * weights[:, :, None]).sum(1).numpy()

    square = [[1, 5, 5], [2, 7, 7]]
    np.testing.assert_allclose(np.asarray(embedder._position_embeddings(square)),
        old_order(square), atol=2e-6, rtol=2e-6)
    non_square = [[1, 4, 6]]
    assert np.abs(np.asarray(embedder._position_embeddings(non_square)) - old_order(non_square)).max() > 0.1
