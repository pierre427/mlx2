"""Pipeline layer ownership is pure metadata and needs no MLX runtime."""

from types import SimpleNamespace

import pytest

from mlx2.runtime.models.pipeline import PipelineMixin


def partition(layer_count, ranks):
    shards = []
    for rank in range(ranks):
        model = PipelineMixin()
        model.layers = list(range(layer_count))
        model.pipeline(SimpleNamespace(rank=lambda rank=rank: rank, size=lambda: ranks))
        shards.append(model.pipeline_layers)
    return shards


@pytest.mark.parametrize("layer_count,ranks", [(10, 3), (64, 3), (5, 4), (2, 4), (8, 2)])
def test_balanced_pipeline_owns_every_layer_once(monkeypatch, layer_count, ranks):
    monkeypatch.delenv("MLX_PIPELINE_LAYERS", raising=False)
    shards = partition(layer_count, ranks)
    # Pipeline ranks run in reverse order, from embedding to output.
    assert [layer for shard in reversed(shards) for layer in shard] == list(range(layer_count))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_custom_pipeline_matches_balanced_layer_ownership(monkeypatch):
    monkeypatch.delenv("MLX_PIPELINE_LAYERS", raising=False)
    balanced = partition(10, 3)
    monkeypatch.setenv("MLX_PIPELINE_LAYERS", "4,3,3")
    assert partition(10, 3) == balanced


def test_custom_pipeline_rejects_negative_counts_before_mutating_layers(monkeypatch):
    monkeypatch.setenv("MLX_PIPELINE_LAYERS", "-1,5")
    model = PipelineMixin()
    model.layers = list(range(4))
    with pytest.raises(ValueError, match="nonnegative"):
        model.pipeline(SimpleNamespace(rank=lambda: 0, size=lambda: 2))
    assert model.layers == list(range(4))
