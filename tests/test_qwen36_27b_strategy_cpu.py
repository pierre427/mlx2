from pathlib import Path

import pytest

ARTIFACTS = (
    Path("~/mlx-models/Qwen3.6-27B-Abliterated-Heretic-Uncensored-MLX-4bit"),
    Path("~/mlx-models/Qwen3.6-27B-MLX-8bit"),
)


def test_longest_first_prunes_impossible_siblings_and_reuses_exact_prefix():
    from mlx2.runtime.continuation_strategy import LongestFirstExactPrefix

    paths = (
        (1, 2, 3, 4, 5, 6, 7),
        (0, 2, 3, 4, 9, 6),
        (1, 2, 0, 4, 9, 6),
        (1, 2, 3, 4, 9, 6),
    )
    controller = LongestFirstExactPrefix(paths, maximum=8)

    first = controller.next_attempt()
    assert first.path_index == 0
    assert first.start == 0 and first.input_rows == 8
    observed = controller.observe(first, (1, 2, 3, 4, 9))
    assert observed.accepted == 4
    assert observed.surviving_indices == (3,)
    assert set(observed.pruned_indices) == {1, 2}

    second = controller.next_attempt()
    assert second.path_index == 3
    assert second.start == 5
    assert second.suffix == (6,)
    assert second.input_rows == 2
    observed = controller.observe(second, (6, 7))
    assert observed.terminal
    assert controller.emitted == (1, 2, 3, 4, 9, 6, 7)
    assert sum(attempt.input_rows for attempt in (first, second)) == 10


def test_longest_first_never_launches_a_wrong_prefix_sibling():
    from mlx2.runtime.continuation_strategy import LongestFirstExactPrefix

    controller = LongestFirstExactPrefix(
        ((1, 2, 3), (0, 2, 3), (1, 0, 3)), maximum=4
    )
    first = controller.next_attempt()
    observation = controller.observe(first, (1, 2, 9))
    assert observation.surviving_indices == ()
    assert controller.next_attempt() is None


def test_strategy_requires_exact_pruning_and_reuse():
    from mlx2.runtime.continuation_strategy import ContinuationStrategy

    with pytest.raises(ValueError, match="requires pruning"):
        ContinuationStrategy(algorithm="longest_first_exact_prefix_v1")


def test_dense_qwen36_adapter_owns_prefill_and_strategy_without_loading_model():
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter

    adapter = object.__new__(Qwen3627BAdapter)
    strategy = adapter.continuation_verification_strategy()
    assert adapter.default_route == "ordinary"
    assert adapter.prefill_step_default() == 2048
    assert strategy.algorithm == "longest_first_exact_prefix_v1"
    assert strategy.prune_incompatible_siblings
    assert strategy.shared_prefix_reuse
    assert not strategy.qualified
    assert adapter.sampling_defaults.model == "Qwen/Qwen3.6-27B"
    assert (
        adapter.sampling_defaults.profiles["thinking"].presence_penalty == 0.0
    )
    assert (
        adapter.sampling_defaults.profiles["instruct"].presence_penalty == 1.5
    )


def test_dense_qwen36_selects_strategy_only_for_a_continuation_pool(monkeypatch):
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    monkeypatch.setattr(
        Qwen3827BAdapter,
        "create_external_batch",
        lambda _self, **kwargs: kwargs,
    )
    adapter = object.__new__(Qwen3627BAdapter)
    adapter.draft_model = object()
    assert "continuation_verification_strategy" not in adapter.create_external_batch()

    adapter.draft_model = type(
        "ContinuationDraft",
        (),
        {"last_continuation_selections": ()},
    )()
    strategy = adapter.create_external_batch()["continuation_verification_strategy"]
    assert strategy.algorithm == "longest_first_exact_prefix_v1"


@pytest.mark.parametrize("path", ARTIFACTS)
def test_named_dense_qwen36_artifacts_resolve_to_revision_bound_adapter(path):
    if not path.is_dir():
        pytest.skip("local Qwen3.6 artifact is not staged")
    from mlx2.adapters.qwen36_27b import inspect_artifact
    from mlx2.adapters.registry import inspect_model

    artifact = inspect_artifact(path)
    resolution = inspect_model(path)
    assert resolution.adapter_type.__name__ == "Qwen3627BAdapter"
    assert resolution.descriptor.family == "qwen3.6-27b"
    assert not artifact["has_mtp"] and artifact["mtp_tensor_count"] == 0
    assert artifact["family_revision"]
