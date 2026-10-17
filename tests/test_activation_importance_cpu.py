
import gc
import weakref

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters.nemotron35_super_config import (
    SOURCE_CONFIG_SHA256,
    SOURCE_INDEX_SHA256,
    SOURCE_REVISION,
)
from mlx2.conversion.capture import RoutedCapture, reduced_routed_moments
from mlx2.conversion.importance import ActivationMoments
from mlx2.runtime.models.switch_layers import SwitchMLP


def binding():
    return {"source_revision": SOURCE_REVISION, "source_config_sha256": SOURCE_CONFIG_SHA256,
            "source_index_sha256": SOURCE_INDEX_SHA256, "proxy_artifact": "a" * 64,
            "corpus_sha256": "b" * 64, "collector_sha256": "c" * 64,
            "processor_revision": "owned-v1"}


def test_sums_counts_coverage_and_round_trip_bind_exact_samples(tmp_path):
    stats = ActivationMoments({"bank": (2, 3)}, binding())
    stats.add_reduced("bank", np.array([[8., 2., 18.], [1., 4., 9.]]), np.array([2, 1]))
    stats.record_sample("image-1", modality="image_text", input_sha256="d" * 64)
    stats.require_coverage(minimum_count=1, minimum_image_samples=1)
    np.testing.assert_array_equal(stats.importance_for("bank", 0), [4., 1., 9.])
    assert stats.coverage()["complete"] is False
    stats.save(tmp_path / "stats")
    loaded = ActivationMoments.load(tmp_path / "stats", specs={"bank": (2, 3)})
    np.testing.assert_array_equal(loaded.sums["bank"], stats.sums["bank"])
    assert loaded.samples == stats.samples
    with pytest.raises(FileExistsError):
        stats.save(tmp_path / "stats")
    with (tmp_path / "stats/moments.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="tensor bytes"):
        ActivationMoments.load(tmp_path / "stats", specs={"bank": (2, 3)})


def test_missing_experts_are_not_fabricated_and_merges_reject_overlap():
    first = ActivationMoments({"bank": (2, 3)}, binding())
    second = ActivationMoments({"bank": (2, 3)}, binding())
    first.add_reduced("bank", np.array([[1., 1., 1.], [0., 0., 0.]]), np.array([1, 0]))
    second.add_reduced("bank", np.array([[0., 0., 0.], [2., 2., 2.]]), np.array([0, 1]))
    with pytest.raises(ValueError, match="unseen"):
        first.importance_for("bank", 1)
    with pytest.raises(ValueError, match="coverage"):
        first.require_coverage(minimum_count=1, minimum_image_samples=0)
    first.record_sample("one", modality="image_text", input_sha256="d" * 64)
    second.record_sample("two", modality="text", input_sha256="e" * 64)
    first.merge(second).require_coverage(minimum_count=1, minimum_image_samples=1)
    with pytest.raises(ValueError, match="double-count"):
        first.merge(second)
    other = binding(); other["proxy_artifact"] = "f" * 64
    with pytest.raises(ValueError, match="another source"):
        first.merge(ActivationMoments({"bank": (2, 3)}, other))


@pytest.mark.parametrize("sums,counts", [(np.ones((2, 3)), [0, 1]),
    (np.full((2, 3), np.nan), [1, 1]), (-np.ones((2, 3)), [1, 1]), (np.ones((2, 3)), [-1, 1])])
def test_invalid_observations_fail_closed(sums, counts):
    stats = ActivationMoments({"bank": (2, 3)}, binding())
    with pytest.raises(ValueError, match="Invalid reduced"):
        stats.add_reduced("bank", sums, np.array(counts))


def test_sorted_alignment_padding_is_excluded_from_counts_and_sums():
    values = mx.array([[[1., 2.]], [[3., 4.]], [[5., 6.]], [[5., 6.]], [[5., 6.]]])
    ids = mx.array([0, 1, 1, 1, 1])
    sums, counts = reduced_routed_moments(values, ids, experts=2, real_assignments=3, sorted_indices=True)
    np.testing.assert_array_equal(np.asarray(sums), [[1., 4.], [34., 52.]])
    np.testing.assert_array_equal(np.asarray(counts), [1, 2])


@pytest.mark.parametrize("tokens", [5, 20])
def test_capture_preserves_switch_math_and_matches_independent_routed_oracle(tokens):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = SwitchMLP(64, 64, 3, activation=nn.ReLU2())

    mx.random.seed(7)
    model = Tiny()
    original = model.switch_mlp
    x = np.random.default_rng(7).normal(size=(1, tokens, 64)).astype(np.float32)
    ids = (np.arange(tokens * 2) % 3).reshape(1, tokens, 2).astype(np.int32)
    expected = np.asarray(model.switch_mlp(mx.array(x), mx.array(ids)))
    stats = ActivationMoments({"language_model.switch_mlp.fc1": (3, 64),
                               "language_model.switch_mlp.fc2": (3, 64)}, binding())
    with RoutedCapture(model, stats):
        actual = np.asarray(model.switch_mlp(mx.array(x), mx.array(ids)))
        np.testing.assert_array_equal(actual, expected)
    assert model.switch_mlp is original
    sums_up, sums_down, counts = np.zeros((3, 64)), np.zeros((3, 64)), np.zeros(3, np.uint64)
    weights = np.asarray(original.fc1.weight)
    for token in range(tokens):
        for expert in ids[0, token]:
            activation = np.maximum(x[0, token] @ weights[expert].T, 0) ** 2
            sums_up[expert] += x[0, token] ** 2
            sums_down[expert] += activation ** 2
            counts[expert] += 1
    np.testing.assert_allclose(stats.sums["language_model.switch_mlp.fc1"], sums_up, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(stats.sums["language_model.switch_mlp.fc2"], sums_down, rtol=2e-5, atol=1e-6)
    np.testing.assert_array_equal(stats.counts["language_model.switch_mlp.fc2"], counts)


def test_capture_does_not_retain_sample_statistics_until_cyclic_gc():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = SwitchMLP(64, 64, 3, activation=nn.ReLU2())

    model = Tiny()
    original = model.switch_mlp
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            stats = ActivationMoments({"language_model.switch_mlp.fc1": (3, 64),
                                       "language_model.switch_mlp.fc2": (3, 64)}, binding())
            reference = weakref.ref(stats)
            # No external reference to an installed wrapper remains after exit.
            with RoutedCapture(model, stats):
                mx.eval(model.switch_mlp(mx.ones((1, 5, 64)), mx.zeros((1, 5, 2), mx.int32)))
            assert model.switch_mlp is original
            del stats
            assert reference() is None, "A completed sample retained its moments through a callback cycle"
    finally:
        if enabled:
            gc.enable()
