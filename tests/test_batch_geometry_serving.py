import pytest


def test_serving_parses_batch_geometry_at_construction():
    from mlx2 import serving

    engine = serving.ServingEngine(
        "fixture",
        adapter_factory=lambda *_args, **_kwargs: None,
        execution_policy={"batch_geometry": {"token_budget": 4096}},
    )
    try:
        assert engine.batch_geometry_policy["token_budget"] == 4096
    finally:
        engine.close()


def test_serving_refuses_capability_claim_in_batch_geometry():
    from mlx2 import serving

    with pytest.raises(ValueError, match="unknown batch_geometry"):
        serving.ServingEngine(
            "fixture",
            adapter_factory=lambda *_args, **_kwargs: None,
            execution_policy={"batch_geometry": {"packed_prefill": True}},
        )


def test_batch_generator_applies_head_preserving_geometry_budget():
    from mlx2.runtime.generate import BatchGenerator
    from test_srpt_prefill import _prompt, tiny_model

    generator = BatchGenerator(
        tiny_model(),
        completion_batch_size=4,
        prefill_batch_size=2,
        prefill_batch_window=1,
        prefill_step_size=32,
        batch_geometry={"token_budget": 32},
    )
    try:
        uids = [
            generator.insert([_prompt(length, seed)], max_tokens=[1])[0]
            for length, seed in ((24, 1), (4, 2), (24, 3))
        ]
        prompt_responses, _ = generator.next()
        assert generator.prefill_batch_window == 4
        assert {response.uid for response in prompt_responses} == {uids[0]}
        assert generator.scheduler_stats["batch_geometry_budget_deferred_rows"] == 1
        assert generator.scheduler_stats["batch_geometry_charged_rows"] == 23
    finally:
        generator.close()


def test_batch_generator_selects_length_compatible_companion():
    from mlx2.runtime.generate import BatchGenerator
    from test_srpt_prefill import _prompt, tiny_model

    generator = BatchGenerator(
        tiny_model(),
        completion_batch_size=4,
        prefill_batch_size=2,
        prefill_step_size=32,
        batch_geometry=True,
    )
    try:
        uids = [
            generator.insert([_prompt(length, seed)], max_tokens=[1])[0]
            for length, seed in ((24, 1), (4, 2), (24, 3))
        ]
        prompt_responses, _ = generator.next()
        assert {response.uid for response in prompt_responses} == {uids[0], uids[2]}
        assert generator.scheduler_stats["batch_geometry_bucketed_rounds"] == 1
        assert generator.scheduler_stats["batch_geometry_padding_rows"] == 0
    finally:
        generator.close()


def test_batch_geometry_counters_export_under_their_mechanism():
    from mlx2.prometheus import PrometheusBuilder, _add_scheduler

    builder = PrometheusBuilder()
    _add_scheduler(
        builder,
        {
            "batch_geometry_rounds": 2,
            "batch_geometry_padding_rows": 17,
        },
    )
    rendered = builder.render()
    assert 'mechanism="batch_geometry"' in rendered
    assert 'event="batch_geometry_padding_rows"' in rendered
